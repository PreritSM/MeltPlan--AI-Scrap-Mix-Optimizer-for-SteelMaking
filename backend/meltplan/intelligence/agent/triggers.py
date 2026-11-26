"""Deterministic re-plan trigger rules (A13)."""
from __future__ import annotations

import numpy as np

from meltplan.intelligence.estimator.massbalance import sym


class SurpriseTracker:
    """Lab surprise: |measured - predicted| / sigma_pred for the actual charge. Single |z|>3, or mean of last 3 z > 2.0."""
    # Creates an empty tracker of lab surprise z-scores per element.
    # hist sets how many recent heats the running-mean rule looks at.
    # No triggers fire until z-scores are added.
    def __init__(self, hist: int = 3):
        self.z: dict[str, list[float]] = {}
        self.hist = hist

    # Records a new lab-vs-prediction z-score for an element.
    # Fires when a single |z| exceeds 3 or the mean of the last `hist` z-scores exceeds 2.
    # Returns a human-readable trigger message, or None if nothing fired.
    def add(self, element: str, z: float) -> str | None:
        zs = self.z.setdefault(element, [])
        zs.append(z)
        last = zs[-self.hist:]
        if abs(z) > 3.0:
            return f"{sym(element)} lab result {z:+.1f}σ from prediction"
        if len(last) == self.hist and abs(np.mean(last)) > 2.0:
            return f"{sym(element)} running {np.mean(last):+.1f}σ from prediction over last {self.hist} heats"
        return None
