import json

import numpy as np
import pytest

from scrapchef.act.plans import PlanStore
from scrapchef.act.roles import project
from scrapchef.intelligence.agent import narrator
from scrapchef.intelligence.optimizer.model import make_context, solve
from scrapchef.sim.world import World


@pytest.fixture(scope="module")
def sim():
    w = World("SIMULATED")
    w.make_plan()
    return w


def _flat(o):
    return json.dumps(o).lower()


def test_crane_view_has_no_prices_or_costs(sim):
    v = _flat(project(sim.current_plan(), "crane"))
    for bad in ("cost", "price", "dual", "p95", "headroom", "binding"):
        assert bad not in v


def test_shift_leader_sees_status_only_headroom(sim):
    v = project(sim.current_plan(), "shift_leader")
    assert all(isinstance(x, str) for x in v["heats"][0]["headroom"].values())
    assert "cost" not in _flat(v["heats"][0]) and "binding" not in v
    assert all(x == "OK" for h in v["heats"] for x in h["headroom"].values())   # an optimal plan binds at alpha: not "AT RISK"


def test_only_supervisor_sees_shadow_prices(sim):
    p = sim.current_plan()
    assert "binding" in project(p, "supervisor")
    assert not any("dual" in _flat(project(p, r)) for r in ("email", "shift_leader", "crane"))


def test_state_machine_rejects_illegal_transition(sim):
    with pytest.raises(ValueError):
        sim.store.transition(sim.current_plan(), "RELEASED", "x", "t")


def test_narrator_faithfulness():
    facts = dict(decision="replan", version=2, trigger_text="Cu 2.4σ.", n_heats=3, moves=[dict(group="SCRAP_25A_3ACH", delta_t=-5.0)],
                 cost_delta_per_t=0.2, new_per_t=480.0, reason="")
    assert narrator.faithful("Moved 5.0 t, cost +0.2 over 3 heats.", facts)
    assert not narrator.faithful("Saved 17 dollars.", facts)
    est = dict(cu=0.1856, p=0.052)   # a tool value may be quoted exactly as given, not only rounded to 2 decimals
    assert narrator.faithful("The group carries 0.1856% Cu (0.19%), exceedance 0.052 or 5.2%.", est)
    assert not narrator.faithful("The group carries 0.1857% Cu.", est)


def test_scenario_drift_triggers_replan_and_shifts_to_iron_units():
    """SIMULATED: a copper-heavy 3A (shredded) lot. Lab results go above the limit, the agent re-plans, and the open heats
    move from 3A to clean iron units (the brief's "shredded looks more copper-heavy; shifted to pig iron")."""
    w = World("SIMULATED")
    w.make_plan(); w.approve(); w.release()
    base = w.current_plan()
    mean = lambda p, g: np.mean([h.group_t.get(g, 0) for h in p.heats[-4:]])
    for _ in range(8):
        w.step_heat()
    reps = [d for d in w.decisions if d["decision"] == "replan"]
    assert reps and w.current_plan().version >= 2
    assert mean(w.current_plan(), "IRON_UNITS") > mean(base, "IRON_UNITS") and mean(w.current_plan(), "SCRAP_3A") < mean(base, "SCRAP_3A")
    assert w.kf["cu"].e[0] > w.kf["cu"].batch[0]          # the filter attributes the extra copper to 3A


def test_sanity_identity_frozen_mix_reproduces_cost():
    ctx = make_context()
    x = np.array([75.0, 15.5, 44.0, 0.0])  # bulk scrap, 16A+6A, 25A+3ACh, iron
    r = solve(ctx, [float(ctx.yields @ x)], "chance", z_override=0.0, frozen={0: x})
    assert r["cost"] == pytest.approx(float((ctx.prices + ctx.energy_k) @ x), rel=1e-6)


def test_events_roundtrip():
    s = PlanStore()
    e = s.emit("t", "k", {"a": 1}, ["process"])
    assert s.events_since(0)[0]["seq"] == e["seq"]


def test_risk_matched_baseline_is_not_dearer_than_historical_and_is_in_email():
    w = World("SHADOW_REPLAY")
    plan = w.make_plan()
    rm = w.risk_match(plan)
    assert rm and rm["matched_per_t"] <= rm["hist_per_t"] + 1e-6   # the historical mix is feasible at its own risk
    assert 0 <= rm["worst_p"] <= 1 and rm["worst_el"] in ("cu", "sn", "ni", "mo")
    assert "same risk" in w.emails[-1]["body"]
    assert w.phone("supervisor")["plan"]["cost"]["risk_match"]["matched_per_t"] == rm["matched_per_t"]
