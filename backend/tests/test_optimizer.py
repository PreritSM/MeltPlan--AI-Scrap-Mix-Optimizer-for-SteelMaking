import numpy as np
import pytest

from scrapchef.data.store import GROUPS
from scrapchef.intelligence.optimizer.evaluate import evaluate_mix
from scrapchef.intelligence.optimizer.model import ELS, make_context, predict_heat, solve, z_of
from scrapchef.intelligence.optimizer.plan import build_plan


@pytest.fixture(scope="module")
def ctx():
    return make_context()


def test_chance_feasible_and_tap_met(ctx):
    r = solve(ctx, [125.0, 125.0], "chance")
    assert r["status"].startswith("optimal")
    for h in range(2):
        assert abs(ctx.yields @ r["x"][h] - 125.0) <= ctx.tap_tol + 1e-3


def test_tighter_limit_never_cheaper(ctx):
    costs = []
    for cu in [0.25, 0.23, 0.21]:
        ctx.limits = {**ctx.limits, "cu": cu}
        costs.append(solve(ctx, [125.0], "chance")["cost"])
    assert costs[0] <= costs[1] + 1e-6 <= costs[2] + 2e-6


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


def test_evaluate_matches_plan():
    ctx = make_context()
    plan = build_plan(ctx, [125.0], "t")
    h = plan.heats[0]
    ev = evaluate_mix(ctx, h.group_t, 125.0, {e: ctx.heel_c0[e] for e in ELS})
    assert ev["predicted"]["cu"]["mean"] == pytest.approx(h.predicted["cu"].mean, rel=1e-3)
    assert ev["cost"]["total"] == pytest.approx(h.cost.total, rel=1e-3)


def test_plan_has_real_codes_and_baskets():
    plan = build_plan(make_context(), [125.0], "t")
    items = [i for b in plan.heats[0].baskets for i in b.items]
    assert len(plan.heats[0].baskets) == 2 and any(i.mat_code == 205051 for i in items)


def test_z():
    assert z_of(0.05) == pytest.approx(1.645, abs=1e-3)
