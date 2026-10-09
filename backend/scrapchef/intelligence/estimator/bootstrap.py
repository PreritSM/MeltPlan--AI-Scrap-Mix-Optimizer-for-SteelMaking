"""Block bootstrap (by day) for mean, SD, 90% CI and covariance of the group estimates."""
from __future__ import annotations

import numpy as np

from .bayes import fit_bayes


def block_bootstrap(X, y, days, mu0, sd0, n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(days)
    idx_by_day = {d: np.flatnonzero(days == d) for d in uniq}
    draws = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_day[d] for d in pick])
        draws.append(fit_bayes(X[idx], y[idx], mu0, sd0)[0])
    D = np.array(draws)
    return dict(mean=D.mean(0), sd=D.std(0), lo=np.quantile(D, 0.05, 0), hi=np.quantile(D, 0.95, 0), cov=np.cov(D.T), draws=D)
