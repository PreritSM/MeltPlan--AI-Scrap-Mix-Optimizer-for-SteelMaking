"""Fit + validate the composition estimator. `python -m scrapchef.intelligence.estimator.validate`
writes Reference_Docs/reports/estimator.md and data/estimates.json (consumed by the optimiser)."""
from __future__ import annotations

import json
import sys

import numpy as np

from scrapchef.data.features import split_heats
from scrapchef.data.store import GROUPS, ROOT, priors
from .bayes import fit_bayes, fit_nnls
from .bootstrap import block_bootstrap
from .massbalance import ELEMENTS, build, predict, sym


RECOVERY_SEEDS = 20
RECOVERY_MIN_COV = 0.75  # nominal 0.90; 20 seeds is noisy (sd ~0.07)


def _mae(a, b):
    return float(np.mean(np.abs(a - b)))


def run_element(el: str) -> dict:
    d = build(el)
    n = len(d.y)
    i1, i2 = split_heats(d.heats)
    tr, va, te = slice(0, i1), slice(i1, i2), slice(i2, n)
    mu0, sd0 = priors(el)
    mu0, sd0 = np.array(mu0), np.array(sd0)
    actual = d.heats[el].to_numpy()

    e_nnls = fit_nnls(d.X[tr], d.y[tr])
    e_bayes, s = fit_bayes(d.X[tr], d.y[tr], mu0, sd0)
    preds = {
        "constant (train mean)": np.full(n, actual[tr].mean()),
        "prior-only": predict(d, mu0),
        "NNLS": predict(d, e_nnls),
        "Bayes-NNLS": predict(d, e_bayes),
    }
    mae = {k: _mae(v[te], actual[te]) for k, v in preds.items()}
    rmse = {k: float(np.sqrt(np.mean((v[te] - actual[te]) ** 2))) for k, v in preds.items()}

    # 90% PI from train residuals of the Bayes-NNLS predictive; coverage on test
    resid = actual[tr] - preds["Bayes-NNLS"][tr]
    q = np.quantile(resid, [0.05, 0.95])
    cov = float(np.mean((actual[te] >= preds["Bayes-NNLS"][te] + q[0]) & (actual[te] <= preds["Bayes-NNLS"][te] + q[1])))

    days = d.heats.start_ts.dt.floor("D").astype("int64").to_numpy()
    bs = block_bootstrap(d.X[tr], d.y[tr], days[tr], mu0, sd0)

    # synthetic recovery test (labelled): real X, e_true drawn from the prior (so every group is tested at a
    # different truth each seed), noise at the observed level. Reports 90% CI coverage per group over RECOVERY_SEEDS.
    cov_hits = np.zeros(len(mu0))
    for seed in range(RECOVERY_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        e_true = np.maximum(rng.normal(mu0, sd0), 0.1 * mu0)
        y_syn = d.X @ e_true + rng.normal(0, s, n)
        e_rec = block_bootstrap(d.X, y_syn, days, mu0, sd0, n=60, seed=seed)
        cov_hits += (e_true >= e_rec["lo"]) & (e_true <= e_rec["hi"])
    recovery_cov = (cov_hits / RECOVERY_SEEDS).tolist()
    recovered = [c >= RECOVERY_MIN_COV for c in recovery_cov]

    # final fit on all cohort heats for the production estimates
    e_all, _ = fit_bayes(d.X, d.y, mu0, sd0)
    bs_all = block_bootstrap(d.X, d.y, days, mu0, sd0, n=300)
    return dict(
        element=el, n_heats=n, yield_=d.yield_, mae=mae, rmse=rmse, pi90_coverage=cov, resid_q=q.tolist(),
        prior=dict(mu=mu0.tolist(), sd=sd0.tolist()),
        nnls=e_nnls.tolist(), bayes_train=e_bayes.tolist(), sigma_eps=float(s),
        boot_train=dict(mean=bs["mean"].tolist(), sd=bs["sd"].tolist(), lo=bs["lo"].tolist(), hi=bs["hi"].tolist(), cov=bs["cov"].tolist(), point=e_bayes.tolist()),
        final=dict(mean=bs_all["mean"].tolist(), point=e_all.tolist(), sd=bs_all["sd"].tolist(),
                   lo=bs_all["lo"].tolist(), hi=bs_all["hi"].tolist(), cov=bs_all["cov"].tolist()),
        recovery=recovered, recovery_cov=recovery_cov, cond=float(np.linalg.cond(d.X)),
        share=(d.X.sum(0) / d.X.sum()).tolist(),
        prev_measured_frac=float(d.prev_measured.mean()),
    )


def main() -> None:
    res = [run_element(el) for el in ELEMENTS]
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "estimates.json").write_text(json.dumps({r["element"]: r for r in res}, indent=1))
    L = ["# Composition estimator report (auto-generated)\n",
         "Time split 70/15/15 by heat order; metrics on the last 15% (held-out). Priors are ASSUMED (Table H2 via grade_map).\n",
         f"Metallic yield (A4, Table H2 group yields, shared with the optimiser): **{res[0]['yield_']:.3f}** effective over the cohort. Heats with a measured previous-heat chemistry: "
         f"{res[0]['prev_measured_frac']:.0%} (others use the trailing median as heel chemistry).\n"]
    for r in res:
        el = sym(r["element"])
        L.append(f"## {el} (n={r['n_heats']} heats)\n")
        L.append("| method | test MAE (wt %) | test RMSE |\n|---|---|---|")
        for k in r["mae"]:
            L.append(f"| {k} | {r['mae'][k]:.4f} | {r['rmse'][k]:.4f} |")
        L.append(f"\nBayes-NNLS 90% PI coverage on test: **{r['pi90_coverage']:.0%}**. Condition number of X: {r['cond']:.1f}\n")
        L.append("| group | mass share | prior | NNLS | Bayes-NNLS (all heats) ± sd | 90% CI | synthetic CI coverage (nominal 90%) |\n|---|---|---|---|---|---|---|")
        f = r["final"]
        for i, g in enumerate(GROUPS):
            L.append(f"| {g} | {r['share'][i]:.1%} | {r['prior']['mu'][i]:.3f}±{r['prior']['sd'][i]:.3f} | {r['nnls'][i]:.3f} | "
                     f"{f['mean'][i]:.3f} ± {f['sd'][i]:.3f} | [{f['lo'][i]:.3f}, {f['hi'][i]:.3f}] | {r['recovery_cov'][i]:.0%} {'ok' if r['recovery'][i] else 'FAIL'} |")
        L.append("")
    (ROOT / "Reference_Docs" / "reports" / "estimator.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
