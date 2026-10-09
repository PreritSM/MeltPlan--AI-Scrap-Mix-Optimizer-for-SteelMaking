"""Deterministic re-plan trigger rules (A13)."""
from __future__ import annotations

import numpy as np


class SurpriseTracker:
    """Lab surprise: |measured - predicted| / sigma_pred for the actual charge. Single |z|>3, or mean of last 3 z > 2.0."""
    def __init__(self, hist: int = 3):
        self.z: dict[str, list[float]] = {}
        self.hist = hist

    def add(self, element: str, z: float) -> str | None:
        zs = self.z.setdefault(element, [])
        zs.append(z)
        last = zs[-self.hist:]
        if abs(z) > 3.0:
            return f"{element.upper()} lab result {z:+.1f}σ from prediction"
        if len(last) == self.hist and abs(np.mean(last)) > 2.0:
            return f"{element.upper()} running {np.mean(last):+.1f}σ from prediction over last {self.hist} heats"
        return None
