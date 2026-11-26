"""Per-element Kalman filter, random-walk state e_t (group contents). Observation y = x.e + v (Zhou, Naert & Nuyens 2025)."""
from __future__ import annotations

import numpy as np


class ElementKF:
    # Initialises the per-element Kalman filter from the batch estimates.
    # Floors the starting variance and random-walk noise so the filter does not freeze.
    # Stores the state, covariance P, observation noise R and process noise Q.
    def __init__(self, mean, cov, sigma_eps, q_frac: float = 2e-3):
        self.batch = np.array(mean, float)
        self.sd_batch = np.sqrt(np.diag(cov))
        self.e = self.batch.copy()
        # bootstrap SDs are optimistic (e.g. 25A Cu +-0.005), which freezes the filter; floor the starting variance and the
        # random walk at the same effective SD the drift flag uses (15% of the estimate, at least 0.005)
        sd_eff = np.maximum(self.sd_batch, np.maximum(0.15 * np.abs(self.batch), 0.005))
        self.P = np.array(cov, float) + np.diag(np.maximum(sd_eff ** 2 - np.diag(cov), 0.0)) + 1e-8 * np.eye(len(mean))
        self.R = sigma_eps ** 2
        self.Q = q_frac * np.diag(sd_eff ** 2)

    # Runs one predict/update step of the filter for a newly measured heat.
    # Updates the group contents (clipped at zero) and the covariance from the innovation.
    # Returns the normalised prediction error (z-score) for surprise tracking.
    def update(self, x: np.ndarray, y: float) -> float:
        """x: tonnes per group, y: c*B - H*c_prev (wt% t). Returns the normalised prediction error (z-score)."""
        self.P = self.P + self.Q
        S = float(x @ self.P @ x + self.R)
        K = self.P @ x / S
        prediction_error = y - float(x @ self.e)
        self.e = np.maximum(self.e + K * prediction_error, 0.0)
        self.P = self.P - np.outer(K, x @ self.P)
        return prediction_error / np.sqrt(S)

    # Checks which groups have drifted from the batch estimate.
    # A group drifts when its online estimate is more than k effective SDs away.
    # Returns the indices of the drifting groups.
    def drift(self, k: float = 2.0) -> list[int]:
        """Groups whose online estimate left the batch estimate by > k sd_eff (sd floored: bootstrap sd is optimistic)."""
        sd_eff = np.maximum(self.sd_batch, np.maximum(0.15 * np.abs(self.batch), 0.005))
        return [i for i in range(len(self.e)) if abs(self.e[i] - self.batch[i]) > k * sd_eff[i]]
