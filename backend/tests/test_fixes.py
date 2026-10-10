"""Regression tests for the replay, event-stream and versioning fixes."""
import numpy as np
import pytest

from scrapchef.act.plans import PlanStore
from scrapchef.api.main import resume_seq
from scrapchef.data.features import heel_chem
from scrapchef.data.store import GROUPS, heat_table
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
