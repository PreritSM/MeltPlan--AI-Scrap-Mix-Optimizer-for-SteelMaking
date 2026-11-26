import numpy as np
import pytest

from meltplan.data.store import GROUPS
from meltplan.intelligence.optimizer.evaluate import evaluate_mix
from meltplan.intelligence.optimizer.model import ELS, make_context, predict_heat, solve, z_of
from meltplan.intelligence.optimizer.plan import build_plan


# Module fixture providing a default optimiser context.
# Built once per module with make_context().
# Shared by tests that only read or tweak limits.
@pytest.fixture(scope="module")
def ctx():
    return make_context()


# Checks the chance-constrained solve is optimal for two heats.
# Verifies each heat's tap lands within tolerance of the 125 t target.
# Guards the basic feasibility of the default plan.
def test_chance_feasible_and_tap_met(ctx):
    r = solve(ctx, [125.0, 125.0], "chance")
    assert r["status"].startswith("optimal")
    for h in range(2):
        assert abs(ctx.yields @ r["x"][h] - 125.0) <= ctx.tap_tol + 1e-3


# Checks that tightening the Cu limit never lowers the cost.
# Solves with limits 0.25, 0.23 and 0.21 and compares costs.
# Guards the monotonicity of the optimiser.
def test_tighter_limit_never_cheaper(ctx):
    costs = []
    for cu in [0.25, 0.23, 0.21]:
        ctx.limits = {**ctx.limits, "cu": cu}
        costs.append(solve(ctx, [125.0], "chance")["cost"])
    assert costs[0] <= costs[1] + 1e-6 <= costs[2] + 2e-6


# Verifies the chance constraint by Monte Carlo simulation.
# Samples group contents and noise and computes final chemistry for the optimal mix.
# Asserts the exceedance rate for every element stays at or below alpha.
def test_monte_carlo_exceedance_below_alpha():
    ctx = make_context()
    r = solve(ctx, [125.0], "chance", alpha=0.05)
    x, rng = r["x"][0], np.random.default_rng(0)
    B = 125.0 + ctx.heel_t
    for el in ELS:
        e = rng.multivariate_normal(ctx.est.mu[el], ctx.est.cov[el], 20000)
        eps = rng.normal(0, ctx.est.sigma_eps[el], 20000)
        c = (ctx.heel_t * r["heel"][0][el] + e @ x + eps) / B
        assert np.mean(c > ctx.limits[el]) <= 0.05 + 0.005


# Checks evaluate_mix reproduces a built plan's numbers.
# Compares predicted Cu mean and total cost for the same mix and heel.
# Guards consistency between the plan builder and the Mix Editor evaluator.
def test_evaluate_matches_plan():
    ctx = make_context()
    plan = build_plan(ctx, [125.0], "t")
    h = plan.heats[0]
    ev = evaluate_mix(ctx, h.group_t, 125.0, {e: ctx.heel_c0[e] for e in ELS})
    assert ev["predicted"]["cu"]["mean"] == pytest.approx(h.predicted["cu"].mean, rel=1e-3)
    assert ev["cost"]["total"] == pytest.approx(h.cost.total, rel=1e-3)


# Checks a plan uses real mill codes and two baskets.
# Looks for the known main code 205051 among the basket items.
# Guards the group-to-code split and basket logic.
def test_plan_has_real_codes_and_baskets():
    plan = build_plan(make_context(), [125.0], "t")
    items = [i for b in plan.heats[0].baskets for i in b.items]
    assert len(plan.heats[0].baskets) == 2 and any(i.mat_code == 205051 for i in items)


# Checks z_of(0.05) is about 1.645.
# Verifies the normal quantile conversion used by the chance constraint.
# Tolerance is 1e-3.
def test_z():
    assert z_of(0.05) == pytest.approx(1.645, abs=1e-3)
