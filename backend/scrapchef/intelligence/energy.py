"""Energy & cost model: MWh_h = b0 + sum_g b_g x_{h,g} (Huber), time-split CV vs total-charge-only (A15).
`python -m scrapchef.intelligence.energy` -> Reference_Docs/reports/energy.md, data/energy.json."""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from scrapchef.data.store import GROUPS, ROOT, cfg, connect, heat_table


def frame():
    ht = heat_table()
    e = connect().execute("SELECT heatid, mwh_sum_mw AS mwh, o2_nm3, inj_c_t FROM heat_energy").df()
    d = ht[ht.in_cohort].merge(e, on="heatid").dropna(subset=["mwh"])
    return d[(d.mwh > 10) & (d.mwh < 120)].reset_index(drop=True)


def main() -> None:
    d = frame()
    n = len(d); i = int(0.8 * n)
    X = d[GROUPS].to_numpy(); tot = d[["total_charge_t"]].to_numpy(); y = d.mwh.to_numpy()
    out = {}
    for name, A in [("total-charge only", tot), ("per-group", X)]:
        m = HuberRegressor(alpha=1e-3, max_iter=1000).fit(A[:i], y[:i])
        p = m.predict(A[i:])
        out[name] = dict(r2=float(r2_score(y[i:], p)), mae=float(mean_absolute_error(y[i:], p)), coef=m.coef_.tolist(), intercept=float(m.intercept_))
    lift = out["per-group"]["r2"] - out["total-charge only"]["r2"]
    use_groups = lift > 0.05
    tariff = cfg("prices")["electricity_usd_per_mwh"]
    if use_groups:
        coef = np.array(out["per-group"]["coef"]) 
    else:  # A15 fallback: constant MWh/t
        coef = np.full(len(GROUPS), out["total-charge only"]["coef"][0])
    k = (tariff * coef).tolist()
    res = dict(use_group_effects=bool(use_groups), delta_r2=float(lift), mwh_per_t=coef.tolist(), k_usd_per_t=k,
               tariff=tariff, models=out, n=n, provenance="ESTIMATED (energy), ASSUMED (tariff)")
    (ROOT / "data" / "energy.json").write_text(json.dumps(res, indent=1))
    L = ["# Energy model report (auto-generated)\n",
         "Target: MWh per heat = Σ MW over transformer rows (**provisional** definition, decision D13). Time split 80/20.\n",
         "| model | test R² | test MAE (MWh) |\n|---|---|---|"]
    for k_, v in out.items():
        L.append(f"| {k_} | {v['r2']:.3f} | {v['mae']:.2f} |")
    L.append(f"\nΔR² (per-group − total) = **{lift:.3f}** → {'group effects used' if use_groups else 'collapsed to constant MWh/t (A15)'}.\n")
    L.append("| group | MWh/t input | $/t (tariff $%s/MWh, ASSUMED) |\n|---|---|---|" % tariff)
    for g, c, kk in zip(GROUPS, coef, k):
        L.append(f"| {g} | {c:.3f} | {kk:.2f} |")
    (ROOT / "Reference_Docs" / "reports" / "energy.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
