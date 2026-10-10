"""Regression tests for the replay, event-stream, versioning and tap/yield fixes."""
import os

import numpy as np
import pytest

from scrapchef.act.plans import PlanStore
from scrapchef.api.main import resume_seq
from scrapchef.data.features import heel_chem
from scrapchef.data.store import GROUPS, cfg, heat_table
from scrapchef.sim.world import World


@pytest.fixture(scope="module")
def shadow():
    return World("SHADOW_REPLAY")


def test_unmeasured_element_does_not_poison_the_kalman_filter(shadow):
    x = shadow.replay.iloc[0][GROUPS].to_numpy(float)
    meas = {"cu": 0.2, "sn": float("nan"), "ni": 0.12, "mo": 0.02}
    rec = shadow._reveal(x, meas, float(shadow.B_all[0]), "TEST", actual_tap=125.0)
    assert np.isfinite(shadow.kf["sn"].e).all()
    assert rec["elements"]["sn"]["measured"] is None and np.isfinite(shadow.heel["sn"])


def test_replay_heel_chemistry_is_aligned_by_heatid(shadow):
    ht = heat_table()
    for el in ("cu", "sn", "ni", "mo"):
        want = dict(zip(ht.heatid, heel_chem(ht, el)[0]))
        got = shadow.replay_c_prev[el]
        assert len(got) == len(shadow.replay) and np.isfinite(got).all()
        assert np.allclose(got, [want[h] for h in shadow.replay.heatid])


def test_event_ids_resume_within_an_epoch_and_restart_after_reset():
    s = PlanStore()
    for _ in range(3):
        s.emit("t", "k", {}, [])
    assert resume_seq(s, f"{s.epoch}:2", None) == 2
    assert resume_seq(s, None, None) == 3          # fresh connection: no history replay
    s2 = PlanStore()                               # reset: new epoch, seq restarts at 1
    assert resume_seq(s2, f"{s.epoch}:3", None) == 0


def test_reject_continues_version_numbering_and_keeps_history():
    w = World("SIMULATED")
    w.make_plan()
    w.edit(1, w.current_plan().heats[0].group_t)   # v2
    w.reject("test")                               # new plan must be v3, v1/v2 kept
    vs = [(p.version, p.status) for p in w.store.versions()]
    assert vs == [(1, "SUPERSEDED"), (2, "REJECTED"), (3, "EMAILED")]
    assert w.current_plan().parent_version == 2
    with pytest.raises(Exception):                 # a stored version can never be overwritten
        w.store.save(w.current_plan(), "x", "t")


def test_plan_editor_and_tap_are_consistent():
    """Tap tolerance fix (Tap_tol_error.md): plans make the target tap, an unedited mix re-evaluates with no cost change and no
    violation, chemistry and cost per t use the tap the charge makes, and extra iron units dilute Cu."""
    w = World("SHADOW_REPLAY")
    p = w.make_plan()
    h = p.heats[0]
    assert abs(h.tap_t - h.target_tap_t) <= w.ctx.tap_tol + 1e-3
    ev = w.evaluate(1, {g: round(v, 3) for g, v in h.group_t.items()})
    assert abs(ev["delta_cost_per_t"]) < 0.01 and ev["violations"] == [] and ev["process_violations"] == []
    more = dict(h.group_t, IRON_UNITS=h.group_t.get("IRON_UNITS", 0.0) + 40)
    ev = w.evaluate(1, more)
    assert ev["predicted"]["cu"]["mean"] < h.predicted["cu"].mean
    # a hand edit that misses the tap or overfills the furnace is flagged (it looked cheaper per t and passed before)
    assert any("tap" in v for v in ev["process_violations"]) and any("capacity" in v for v in ev["process_violations"])
    with pytest.raises(ValueError, match="capacity"):
        w.edit(1, more)
    assert w.solve_candidate({})[1]["delta_per_t"] == 0.0   # re-solving unchanged inputs is not a "saving"


def test_estimator_and_optimizer_share_one_yield_model():
    from scrapchef.data.store import group_yields
    from scrapchef.intelligence.estimator.massbalance import build
    d = build("cu")
    assert np.allclose(d.B - d.X @ group_yields(), cfg("mill")["heel_t"])
    assert abs(np.median(d.X @ group_yields()) - cfg("mill")["target_tap_t"]) < 1.0   # planned tap = what the mill's charges make


def test_plan_is_done_when_every_heat_is_tapped_and_play_stops():
    from fastapi.testclient import TestClient
    import scrapchef.api.main as api
    w = World("SHADOW_REPLAY")
    w.make_plan(); w.approve(); w.release()
    api.world = w
    c = TestClient(api.app)
    w.agent_pending = 1   # an agent run is in progress: the clock waits instead of tapping its heats away
    assert c.post("/api/sim/step_heat").json().get("waiting") and not w.progress
    w.agent_pending = 0
    for _ in range(len(w.current_plan().heats)):
        assert not c.post("/api/sim/step_heat").json().get("done")
    assert w.current_plan().status == "DONE"
    cursor = w.cursor
    assert c.post("/api/sim/step_heat").json()["done"] and w.cursor == cursor   # no replay beyond the plan


def test_two_worlds_in_one_process_keep_their_own_plans():
    a, b = World("SHADOW_REPLAY"), World("SHADOW_REPLAY")   # no store_path: each gets a private in-memory store
    plan = a.make_plan()
    assert plan is not None and a.current_plan() is not None
    assert b.store.versions() == [] and b.current_plan() is None
    b.reset()
    assert a.current_plan().version == plan.version


def test_a_db_owned_by_another_process_is_refused_and_not_wiped(tmp_path):
    import sqlite3
    import subprocess
    import sys
    db = tmp_path / "app.sqlite"
    owner = subprocess.Popen([sys.executable, "-c", "import sys; from scrapchef.act.plans import PlanStore; "
                              f"s = PlanStore({str(db)!r}); s.emit('t', 'k', {{}}, []); print('owned', flush=True); sys.stdin.read()"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert owner.stdout.readline().strip() == "owned"
        with pytest.raises(RuntimeError, match="owned by another running ScrapChef process"):
            PlanStore(db)
        assert sqlite3.connect(db).execute("SELECT count(*) FROM events").fetchone()[0] == 1   # nothing was dropped
    finally:
        owner.communicate("")
    s = PlanStore(db)                      # the owner has exited: this process may take the file
    assert PlanStore(db).epoch != s.epoch  # and re-open it on a reset (the lock is held once per process)
