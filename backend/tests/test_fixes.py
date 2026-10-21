"""Regression tests for the replay, event-stream, versioning and tap/yield fixes."""
import os

import numpy as np
import pytest

from scrapchef.act.plans import PlanStore
from scrapchef.api.main import resume_seq
from scrapchef.data.features import heel_chem
from scrapchef.data.store import GROUPS, cfg, heat_table
from scrapchef.sim.world import World


# Module fixture: a SHADOW_REPLAY world without a plan.
# Shared by the replay and Kalman filter regression tests.
# Created once per module to save time.
@pytest.fixture(scope="module")
def shadow():
    return World("SHADOW_REPLAY")


# Checks a missing (NaN) lab value leaves the Kalman filter finite.
# Reveals a heat with no Sn measurement.
# Asserts Sn is reported as unmeasured and the heel stays finite.
def test_unmeasured_element_does_not_poison_the_kalman_filter(shadow):
    x = shadow.replay.iloc[0][GROUPS].to_numpy(float)
    meas = {"cu": 0.2, "sn": float("nan"), "ni": 0.12, "mo": 0.02}
    rec = shadow._reveal(x, meas, float(shadow.B_all[0]), "TEST", actual_tap=125.0)
    assert np.isfinite(shadow.kf["sn"].e).all()
    assert rec["elements"]["sn"]["measured"] is None and np.isfinite(shadow.heel["sn"])


# Checks replay heel chemistry is aligned by heat id, not row position.
# Compares the world's replay heel arrays with heel_chem for each element.
# Guards against misalignment when element designs drop different heats.
def test_replay_heel_chemistry_is_aligned_by_heatid(shadow):
    ht = heat_table()
    for el in ("cu", "sn", "ni", "mo"):
        want = dict(zip(ht.heatid, heel_chem(ht, el)[0]))
        got = shadow.replay_c_prev[el]
        assert len(got) == len(shadow.replay) and np.isfinite(got).all()
        assert np.allclose(got, [want[h] for h in shadow.replay.heatid])


# Checks SSE resume positions across reconnects and resets.
# An id from the current epoch resumes; a fresh connection starts at the latest event.
# An id from an older epoch restarts at 0.
def test_event_ids_resume_within_an_epoch_and_restart_after_reset():
    s = PlanStore()
    for _ in range(3):
        s.emit("t", "k", {}, [])
    assert resume_seq(s, f"{s.epoch}:2", None) == 2
    assert resume_seq(s, None, None) == 3          # fresh connection: no history replay
    s2 = PlanStore()                               # reset: new epoch, seq restarts at 1
    assert resume_seq(s2, f"{s.epoch}:3", None) == 0


# Checks a rejection continues version numbering and keeps history.
# Edit then reject should give v1 superseded, v2 rejected and v3 emailed.
# Also checks a stored version can never be overwritten.
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


# Checks plan taps, the editor and cost per tonne are consistent.
# An unedited mix re-evaluates with no cost change; extra iron units dilute Cu.
# Off-target taps and over-capacity charges are flagged and refused.
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


# Checks the estimator and optimiser use the same yield model.
# Bath mass minus yielded tap must equal the heel.
# The median yielded tap must match the mill's target tap.
def test_estimator_and_optimizer_share_one_yield_model():
    from scrapchef.data.store import group_yields
    from scrapchef.intelligence.estimator.massbalance import build
    d = build("cu")
    assert np.allclose(d.B - d.X @ group_yields(), cfg("mill")["heel_t"])
    assert abs(np.median(d.X @ group_yields()) - cfg("mill")["target_tap_t"]) < 1.0   # planned tap = what the mill's charges make


# Checks the API's step_heat flow over a whole plan.
# Stepping waits while the agent runs, then taps every heat and marks the plan DONE.
# Further steps report done without replaying beyond the plan.
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


# Checks two in-memory worlds do not share plan stores.
# A plan made in one world must not appear in the other.
# Resetting one world leaves the other's plan intact.
def test_two_worlds_in_one_process_keep_their_own_plans():
    a, b = World("SHADOW_REPLAY"), World("SHADOW_REPLAY")   # no store_path: each gets a private in-memory store
    plan = a.make_plan()
    assert plan is not None and a.current_plan() is not None
    assert b.store.versions() == [] and b.current_plan() is None
    b.reset()
    assert a.current_plan().version == plan.version


# Checks a DB file locked by another process is refused and not wiped.
# Starts a subprocess that owns the store and tries to open it here.
# After the owner exits, this process may take and re-open the file.
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


# Helper: builds a world with a released v1 and a pending agent re-plan.
# Reports a 3A shortage so the fallback agent proposes a new version.
# Returns the world with that version awaiting the supervisor.
def _released_with_pending_change(mode="SIMULATED"):
    """A released plan v1 and an agent re-plan v2 for a 3A shortage, awaiting the supervisor (fallback agent)."""
    w = World(mode)
    w.make_plan(); w.approve(); w.release()
    rec = w.report_issue("shift_leader", "Only 420 t of 3A left today")
    assert rec["decision"] == "replan" and w.current_plan().status == "EMAILED"
    return w


# Checks rejecting an agent change reinstates the released plan.
# The reinstated version must have v1's mix, settings and released status.
# The crane and furnace may continue and the decision is 'reinstated'.
def test_rejecting_a_replan_puts_the_released_plan_back():
    """E2E #32: rejecting the agent's change must not drop the day's plan when an approved, released plan existed before it."""
    w = _released_with_pending_change()
    v1 = w.store.get(w.plan_id, 1)
    rej = w.current_plan().version
    w.reject("Supervisor requested a new plan")
    live = w.current_plan()
    assert live is not None and live.status == "RELEASED" and live.version > rej and w.released_version == live.version
    assert [h.group_t for h in live.heats] == [h.group_t for h in v1.heats]
    assert w.store.get(w.plan_id, rej).status == "REJECTED"
    assert w.caps == {} and w.crane_hold() is None and w.sim_hold() is None   # v1's settings, crane and furnace may go on
    assert "same mix as approved v1" in w.phone("supervisor")["plan"]["explanation"]
    assert w.decisions[-1]["decision"] == "reinstated"


# Checks rejecting a never-approved plan triggers a normal re-plan.
# Expects v1 rejected and a new v2 emailed.
# The last decision must be 'replan'.
def test_rejecting_an_unapproved_plan_does_not_reinstate_anything():
    w = World("SIMULATED")
    w.make_plan()
    w.reject("Supervisor requested a new plan")   # nothing approved before it: the agent re-plans as before
    assert [(p.version, p.status) for p in w.store.versions()] == [(1, "REJECTED"), (2, "EMAILED")]
    assert w.decisions[-1]["decision"] == "replan"


# Checks the sim clock waits for approval and release of a change.
# Stepping is blocked while the change awaits the supervisor and, if needed, release.
# After release the next heat taps.
def test_sim_clock_waits_for_approval_and_release_of_a_change():
    """E2E #45: heats must not tap on the old list while the agent's change waits for the supervisor and the shift leader."""
    from fastapi.testclient import TestClient
    import scrapchef.api.main as api
    w = _released_with_pending_change()
    api.world = w
    c = TestClient(api.app)
    r = c.post("/api/sim/step_heat").json()
    assert r.get("waiting") and "approval" in r["note"] and w.sim_step == 0
    w.approve()
    if 1 in w.pending_change()[1]:   # approved but not released: heat 1's new list is not the crane's list yet
        assert "not released" in c.post("/api/sim/step_heat").json()["note"] and w.sim_step == 0
    w.release()
    assert not c.post("/api/sim/step_heat").json().get("waiting") and w.sim_step == 1


# Checks a new daily plan clears the previous agent explanation.
# The supervisor sees an explanation after the re-plan.
# After make_plan the explanation is gone.
def test_a_new_plan_does_not_show_the_previous_agent_explanation():
    """E2E #40: world.explanation belongs to the re-plan that wrote it, not to the next daily plan."""
    w = _released_with_pending_change("SHADOW_REPLAY")
    assert w.phone("supervisor")["plan"]["explanation"]
    w.make_plan()
    assert not w.phone("supervisor")["plan"].get("explanation") and w.explanation == {}
