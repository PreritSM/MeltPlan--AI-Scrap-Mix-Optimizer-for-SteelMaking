"""Mass-balance regression  y = c_h*B_h - H*c_{h-1} = X_h . e  (elements that do not oxidise).

B_h = W_h + H is the bath mass: tapped steel plus the heel kept for the next heat (decision D7).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scrapchef.data.features import heel_chem
from scrapchef.data.store import GROUPS, cfg, heat_table

ELEMENTS = ["cu", "sn", "ni", "mo"]


@dataclass
class Design:
    element: str
    heats: pd.DataFrame      # cohort rows aligned with X, y
    X: np.ndarray            # tonnes per group
    y: np.ndarray            # c*B - H*c_prev   (wt% * t)
    B: np.ndarray            # bath mass
    c_prev: np.ndarray       # heel chemistry used
    prev_measured: np.ndarray  # True if c_prev is a real measurement of the previous heat
    yield_: float


def build(element: str, yield_: float | None = None) -> Design:
    mill = cfg("mill")
    H = mill["heel_t"]
    ht = heat_table()
    if yield_ is None:  # A4: calibrate so the median tap equals the rated tap
        yield_ = mill["rated_tap_t"] / ht.loc[ht.in_cohort, "total_charge_t"].median()
    ht["W"] = yield_ * ht.total_charge_t
    ht["B"] = ht.W + H
    ht["c_prev"], ht["prev_measured"] = heel_chem(ht, element)
    d = ht[ht.in_cohort & ht[element].notna()].reset_index(drop=True)
    X = d[GROUPS].to_numpy(float)
    y = d[element].to_numpy(float) * d.B.to_numpy() - H * d.c_prev.to_numpy()
    return Design(element, d, X, y, d.B.to_numpy(), d.c_prev.to_numpy(), d.prev_measured.to_numpy(), yield_)


def predict(d: Design, e: np.ndarray, idx=slice(None)) -> np.ndarray:
    """Predicted final wt % for heats idx given group contents e."""
    H = cfg("mill")["heel_t"]
    return (H * d.c_prev[idx] + d.X[idx] @ e) / d.B[idx]
