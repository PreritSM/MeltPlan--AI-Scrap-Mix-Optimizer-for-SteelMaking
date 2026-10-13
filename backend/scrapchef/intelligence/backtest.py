"""Historical back-test at the same predicted residual risk (D9). `python -m scrapchef.intelligence.backtest`.

For each held-out heat (last 15%, estimator fitted on the train split only): take the historical mix, compute per element
the z at which that mix just meets the limit, then solve the optimiser with exactly that z (floored). Both mixes are then at
equal predicted risk, so the cost difference is a fair comparison. Prices, limits and availability proxy are ASSUMED."""
from __future__ import annotations

import json
import sys

import numpy as np

from scrapchef.data.features import heel_chem, split_heats
from scrapchef.data.store import GROUPS, ROOT, heat_table, recipe_bounds
from .estimator.massbalance import build
from .optimizer.evaluate import Z_FLOOR, z_hist
from .optimizer.model import ELS, make_context, predict_heat, solve, z_of

def run(price_noise: float = 0.0, seed: int = 0, max_heats: int | None = None):
    ctx = make_context(variant="boot_train")
    rng = np.random.default_rng(seed)
    if price_noise:
        ctx.prices = ctx.prices * rng.uniform(1 - price_noise, 1 + price_noise, len(GROUPS))
    d = build("cu")
    # heel chemistry by heatid: element designs drop different heats (e.g. a missing Sn), so row positions do not line up
    ht = heat_table()
    heels = {el: dict(zip(ht.heatid, heel_chem(ht, el)[0])) for el in ELS}
    n = len(d.y)
    test = range(split_heats(d.heats)[1], n)
    bounds0 = recipe_bounds()
    rows = []
    for k, i in enumerate(test):
        if max_heats and k >= max_heats:
            break
        x = d.X[i]
        T = float(ctx.yields @ x)
        if T < 80:
            continue
        heel = {el: float(heels[el][d.heats.heatid[i]]) for el in ELS}
        # availability proxy (A10): per-group stock = max usage within +/-1 day
        ts = d.heats.start_ts
        near = (ts >= ts[i] - np.timedelta64(1, "D")) & (ts <= ts[i] + np.timedelta64(1, "D"))
        ctx.inventory = d.X[near.to_numpy()].max(0) + 1e-6
        share = x / x.sum()
        ctx.bounds = {g: (min(bounds0[g][0], share[j]), max(bounds0[g][1], share[j])) for j, g in enumerate(GROUPS)}
        z = z_hist(ctx, x, T, heel)
        ctx.heel_c0 = heel
        r = solve(ctx, [T], "chance", z_override=z)
        if r["x"] is None:
            continue
        cost_h = float((ctx.prices + ctx.energy_k) @ x)
        rows.append(dict(i=i, T=T, cost_hist=cost_h, cost_lp=r["cost"], x_hist=x.tolist(), x_lp=r["x"][0].tolist()))
    return rows


def headline(rows, seed=0):
    s = np.array([(r["cost_hist"] - r["cost_lp"]) / r["cost_hist"] for r in rows])
    rng = np.random.default_rng(seed)
    boots = [rng.choice(s, len(s)).mean() for _ in range(1000)]
    return float(s.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> None:
    rows = run()
    m, lo, hi = headline(rows)
    per_t = np.mean([(r["cost_hist"] - r["cost_lp"]) / r["T"] for r in rows])
    sens = []
    for noise in (0.2,):
        for sd in range(3):
            sens.append(headline(run(noise, sd, max_heats=150))[0])
    X_h = np.mean([r["x_hist"] for r in rows], 0); X_l = np.mean([r["x_lp"] for r in rows], 0)
    out = dict(n_heats=len(rows), saving_pct=m * 100, ci=[lo * 100, hi * 100], usd_per_t=float(per_t), price_sensitivity_pct=[s * 100 for s in sens],
               mix_hist=dict(zip(GROUPS, X_h.tolist())), mix_lp=dict(zip(GROUPS, X_l.tolist())))
    (ROOT / "data" / "backtest.json").write_text(json.dumps(out, indent=1))
    L = ["# Back-test report (auto-generated)\n",
         f"Held-out heats solved: **{len(rows)}** (estimator fit on train split only). Same-residual-risk comparison (z matched per element, floor {Z_FLOOR}).\n",
         f"**Headline: {m*100:.2f}% cheaper at the same predicted residual risk** (95% bootstrap CI [{lo*100:.2f}%, {hi*100:.2f}%]; ${per_t:.2f}/t steel).\n",
         f"Sensitivity: random ±20% group-price perturbations (3 seeds, 150 heats each): {', '.join(f'{s*100:.2f}%' for s in sens)}.\n",
         "**Caveat:** the absolute figure depends on ASSUMED prices (config/prices.yaml) and an availability proxy (stock = max usage within ±1 day). "
         "The mill's mix is ~85% one code, so there is little room to move; the relative result matters more than the level.\n",
         "| group | mean tonnes historical | mean tonnes LP |\n|---|---|---|"]
    for g in GROUPS:
        L.append(f"| {g} | {out['mix_hist'][g]:.1f} | {out['mix_lp'][g]:.1f} |")
    report = ROOT / "reports" / "backtest.md"
    report.parent.mkdir(exist_ok=True)
    report.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
