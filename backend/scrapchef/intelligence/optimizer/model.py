"""Uncertainty-aware blend optimiser (cvxpy). Modes: aim (LP), margin (LP), chance (SOCP)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
from scipy.stats import norm

from scrapchef.data.store import GROUPS, ROOT, cfg, last_chem, recipe_bounds

ELS = ["cu", "sn", "ni", "mo"]


@dataclass
class Estimates:
    mu: dict[str, np.ndarray]
    sd: dict[str, np.ndarray]
    cov: dict[str, np.ndarray]
    sigma_eps: dict[str, float]

    @classmethod
    def load(cls, variant: str = "final") -> "Estimates":
        """variant 'final' = fit on all cohort heats; 'boot_train' = train-split only (back-test, no leakage)."""
        raw = json.loads((ROOT / "data" / "estimates.json").read_text())
        g = lambda k: {e: np.array(raw[e][variant][k]) for e in ELS}
        return cls(g("mean"), g("sd"), g("cov"), {e: raw[e]["sigma_eps"] for e in ELS})


@dataclass
class Context:
    est: Estimates
    prices: np.ndarray                 # $/t per group (ASSUMED)
    yields: np.ndarray                 # metallic yield per group (ASSUMED, Table H2)
    limits: dict[str, float]           # spec class limits (ASSUMED)
    inventory: np.ndarray | None = None  # t per group over the horizon (MOCK)
    energy_k: np.ndarray = field(default_factory=lambda: np.zeros(len(GROUPS)))  # $/t input
    bounds: dict[str, tuple[float, float]] = field(default_factory=recipe_bounds)
    heel_t: float = 17.5
    tap_tol: float = 1.0
    heel_c0: dict[str, float] = field(default_factory=last_chem)


def make_context(spec_class: str | None = None, inventory=None, variant: str = "final") -> Context:
    gm, pr, sp = cfg("grade_map"), cfg("prices"), cfg("specs")
    en = ROOT / "data" / "energy.json"
    energy_k = np.array(json.loads(en.read_text())["k_usd_per_t"]) if en.exists() else np.zeros(len(GROUPS))
    cats = gm["categories"]
    return Context(
        est=Estimates.load(variant), energy_k=energy_k,
        prices=np.array([pr["groups"][g] for g in GROUPS], float),
        yields=np.array([cats[gm["groups"][g]["category"]]["yield"] for g in GROUPS], float),
        limits=sp["classes"][spec_class or sp["default"]],
        inventory=inventory, heel_t=cfg("mill")["heel_t"],
    )


def z_of(alpha: float) -> float:
    return float(norm.ppf(1 - alpha))


def chol(cov: np.ndarray) -> np.ndarray:
    return np.linalg.cholesky(cov + 1e-12 * np.eye(len(cov)))


def predict_heat(ctx: Context, x: np.ndarray, tap_t: float, heel_c: dict[str, float]) -> dict[str, dict[str, float]]:
    """Mean, sd, p95, exceedance probability of final chemistry (wt %) for charge x (t per group)."""
    B = tap_t + ctx.heel_t
    out = {}
    for el in ELS:
        mean = (ctx.heel_t * heel_c[el] + ctx.est.mu[el] @ x) / B
        var = x @ ctx.est.cov[el] @ x + ctx.est.sigma_eps[el] ** 2
        sd = np.sqrt(var) / B
        lim = ctx.limits[el]
        out[el] = dict(mean=float(mean), sd=float(sd), p95=float(mean + 1.645 * sd), limit=lim,
                       p_exceed=float(1 - norm.cdf((lim - mean) / sd)))
    return out


def solve(ctx: Context, taps: list[float], mode: str = "chance", alpha: float = 0.05,
          z_override: float | dict | None = None, aim_margin: float = 0.9, frozen: dict[int, np.ndarray] | None = None):
    """Plan a horizon of heats. Returns dict(x [H,G], status, cost, duals, heel, constraints)."""
    H, G = len(taps), len(GROUPS)
    zd = z_override if isinstance(z_override, dict) else {e: (z_override if z_override is not None else z_of(alpha)) for e in ELS}
    heel = [dict(ctx.heel_c0) for _ in range(H)]
    res = None
    for _ in range(2 if (mode == "chance" and H > 1) else 1):  # A14: 2-pass fixed point on heel chemistry
        x = cp.Variable((H, G), nonneg=True)
        cons, tagged = [], []
        cost_vec = ctx.prices + ctx.energy_k
        for h in range(H):
            T, B = taps[h], taps[h] + ctx.heel_t
            if frozen and h in frozen:
                cons.append(x[h] == frozen[h]); continue
            c = ctx.yields @ x[h]
            cons += [c >= T - ctx.tap_tol, c <= T + ctx.tap_tol]
            tagged.append(("tap", h, None, cons[-2]))
            tot = cp.sum(x[h])
            for gi, g in enumerate(GROUPS):
                lo, hi = ctx.bounds[g]
                cons += [x[h, gi] >= lo * tot, x[h, gi] <= hi * tot]
            for el in ELS:
                mu, lim, z = ctx.est.mu[el], ctx.limits[el], zd[el]
                base = ctx.heel_t * heel[h][el] + mu @ x[h]
                if mode == "chance":
                    L = chol(ctx.est.cov[el])
                    lhs = base + z * cp.norm(cp.hstack([L.T @ x[h], ctx.est.sigma_eps[el]]), 2)
                    k = lhs <= lim * B
                elif mode == "margin":
                    k = base + z * (ctx.est.sd[el] @ x[h]) <= lim * B
                else:
                    k = base <= aim_margin * lim * B
                cons.append(k); tagged.append(("chance", h, el, k))
        if ctx.inventory is not None:
            for gi in range(G):
                k = cp.sum(x[:, gi]) <= ctx.inventory[gi]
                cons.append(k); tagged.append(("inventory", None, GROUPS[gi], k))
        obj = cp.Minimize(cp.sum(x @ cost_vec))
        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:  # numerically hard heat: retry once with SCS, else report as unsolved
            try:
                prob.solve(solver=cp.SCS, eps=1e-6)
            except cp.error.SolverError:
                return dict(status="solver_error", x=None)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return dict(status=prob.status, x=None)
        X = x.value
        for h in range(1, H):  # heel for next heat = this heat's planned P50
            B = taps[h - 1] + ctx.heel_t
            for el in ELS:
                heel[h][el] = float((ctx.heel_t * heel[h - 1][el] + ctx.est.mu[el] @ X[h - 1]) / B)
        res = dict(status=prob.status, x=X, cost=float(prob.value), heel=heel, tagged=tagged, taps=taps)
    return res


def duals(ctx: Context, res, slot0: int = 1) -> list[dict]:
    """Constraint duals converted to plain units (shadow prices). slot0 = plan slot of the first solved heat."""
    out = []
    for kind, h, key, con in res["tagged"]:
        d = con.dual_value
        if d is None or abs(float(np.sum(d))) < 1e-4:
            continue
        d = float(np.sum(d))
        if kind == "chance":
            T = res["taps"][h]; B = T + ctx.heel_t
            per_001 = d * 0.01 * B / T      # $ per t steel per 0.01 wt% of extra headroom
            out.append(dict(kind=kind, heat=h, key=key, dual=d, usd_per_t_per_0p01pct=per_001,
                            plain=f"Heat {h + slot0}: each extra 0.01% {key.upper()} headroom is worth ${per_001:.2f}/t"))
        elif kind == "inventory":
            out.append(dict(kind=kind, heat=None, key=key, dual=d, usd_per_t_per_0p01pct=None,
                            plain=f"Each extra tonne of {key} would save ${abs(d):.2f}"))  # cvxpy dual of a <= row is >= 0
    return out
