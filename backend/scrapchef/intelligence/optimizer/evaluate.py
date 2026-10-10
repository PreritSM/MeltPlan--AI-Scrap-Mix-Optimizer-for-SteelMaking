"""Control Suite tool: re-evaluate a hand-edited mix instantly (no solve)."""
from __future__ import annotations

import copy

import numpy as np

from scrapchef.data.store import GROUPS
from .model import ELS, Context, chol, predict_heat, solve, tap_of

P_TOL = 1e-3   # the chance constraint binds by design (p = alpha); rounding must not read as a violation

Z_FLOOR = 0.5


def z_hist(ctx: Context, x: np.ndarray, T: float, heel: dict[str, float]) -> dict[str, float]:
    """Per element, the z at which mix x just meets the limit (floored): the risk level the historical mix implicitly runs at."""
    B = T + ctx.heel_t
    z = {}
    for el in ELS:
        base = ctx.heel_t * heel[el] + ctx.est.mu[el] @ x
        nrm = np.linalg.norm(np.append(chol(ctx.est.cov[el]).T @ x, ctx.est.sigma_eps[el]))
        z[el] = max((ctx.limits[el] * B - base) / nrm, Z_FLOOR)
    return z


def risk_matched_baseline(ctx: Context, x_mean: np.ndarray, taps: list[float], heel: dict[str, float]) -> dict | None:
    """Historical mean mix vs the optimiser at that mix's own predicted risk, per tonne of steel (same prices, taps, heel).

    The mix is scaled to each planned heat's tap. Its risk is z_hist per element; each heat is then re-solved with exactly those z
    (recipe bounds widened to include the historical shares, no horizon inventory). Heats the solver cannot solve are skipped on both sides."""
    unit = x_mean / float(ctx.yields @ x_mean)
    share = x_mean / x_mean.sum()
    cost_vec = ctx.prices + ctx.energy_k
    hist = matched = steel = 0.0
    pe: dict[str, list[float]] = {el: [] for el in ELS}
    for T in taps:
        x = unit * T
        c = copy.copy(ctx)
        c.inventory, c.heel_c0 = None, dict(heel)
        c.bounds = {g: (min(lo, share[j]), max(hi, share[j])) for j, (g, (lo, hi)) in enumerate((g, ctx.bounds[g]) for g in GROUPS)}
        r = solve(c, [T], "chance", z_override=z_hist(ctx, x, T, heel))
        if r.get("x") is None:
            continue
        hist += float(cost_vec @ x)
        matched += r["cost"]
        steel += T
        for el, p in predict_heat(ctx, x, T, heel).items():
            pe[el].append(p["p_exceed"])
    if steel == 0:
        return None
    p_exceed = {el: float(np.mean(v)) for el, v in pe.items()}
    worst = max(p_exceed, key=p_exceed.get)
    return dict(hist_per_t=hist / steel, matched_per_t=matched / steel, p_exceed=p_exceed, worst_el=worst, worst_p=p_exceed[worst])


def evaluate_mix(ctx: Context, group_t: dict[str, float], tap_t: float, heel_c: dict[str, float], alpha: float = 0.05) -> dict:
    """Chemistry and cost per tonne use the tap this mix actually makes; tap_t (the target) is only reported for comparison."""
    x = np.array([group_t.get(g, 0.0) for g in GROUPS], float)
    tap_actual = tap_of(ctx, x)
    pred = predict_heat(ctx, x, None, heel_c)
    material = float(ctx.prices @ x)
    energy = float(ctx.energy_k @ x)
    return dict(predicted=pred, cost=dict(material=material, energy=energy, total=material + energy,
                                          per_t=(material + energy) / tap_actual if tap_actual > 0 else 0.0),
                tap_actual_t=tap_actual, target_tap_t=tap_t,
                violations=[e for e, p in pred.items() if p["p_exceed"] > alpha + P_TOL])
