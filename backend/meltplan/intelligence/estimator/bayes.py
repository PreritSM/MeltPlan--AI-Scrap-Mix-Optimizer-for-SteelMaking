"""NNLS and Bayes-NNLS (MAP with Gaussian prior and e >= 0) via stacked non-negative least squares."""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


# Fits plain non-negative least squares of y on X.
# Thin wrapper around scipy's nnls solver.
# Returns only the coefficient vector.
def fit_nnls(X, y):
    return nnls(X, y)[0]


# Fits a MAP estimate with a Gaussian prior and a non-negativity constraint.
# Stacks the prior as extra rows and iterates NNLS, re-estimating the noise SD each pass.
# Returns the coefficient vector and the residual noise SD.
def fit_bayes(X, y, mu0, sd0, iters: int = 5):
    """Minimise sum((y-Xe)^2)/s^2 + sum((e-mu0)^2/sd0^2), e>=0. s estimated from residuals."""
    mu0, sd0 = np.asarray(mu0, float), np.asarray(sd0, float)
    s = max(np.std(y - X @ fit_nnls(X, y)), 1e-6)
    e = mu0.copy()
    for _ in range(iters):
        A = np.vstack([X / s, np.diag(1.0 / sd0)])
        b = np.concatenate([y / s, mu0 / sd0])
        e = nnls(A, b)[0]
        r = y - X @ e
        s = max(np.sqrt(np.mean(r ** 2)), 1e-6)
    return e, s
