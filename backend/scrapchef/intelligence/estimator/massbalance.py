"""Mass-balance regression  y = c_h*B_h - H*c_{h-1} = X_h . e  (elements that do not oxidise).

B_h = W_h + H is the bath mass: tapped steel plus the heel kept for the next heat (decision D7).
W_h = yields . X_h with the same per-group yields the optimiser uses (D19), so estimates and plans share one tap model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scrapchef.data.features import heel_chem
from scrapchef.data.store import GROUPS, cfg, group_yields, heat_table

ELEMENTS = ["cu", "sn", "ni", "mo"]


# Converts a lowercase element key into its chemical symbol form.
# For example 'cu' becomes 'Cu'.
# Used only for display; keys stay lowercase elsewhere.
def sym(el: str) -> str:
    """Display form of an element key: "cu" -> "Cu". Keys stay lowercase everywhere else."""
    return el[:1].upper() + el[1:]


@dataclass
class Design:
    element: str
    heats: pd.DataFrame      # cohort rows aligned with X, y
    X: np.ndarray            # tonnes per group
    y: np.ndarray            # c*B - H*c_prev   (wt% * t)
    B: np.ndarray            # bath mass
    c_prev: np.ndarray       # heel chemistry used
    prev_measured: np.ndarray  # True if c_prev is a real measurement of the previous heat
    yield_: float            # effective metallic yield over the cohort (tapped / charged), for the report


# Builds the mass-balance regression design for one element.
# Computes tap and bath mass from group yields plus the heel, and the heel chemistry.
# Returns a Design with X (tonnes per group), y and supporting arrays for the cohort.
def build(element: str) -> Design:
    H = cfg("mill")["heel_t"]
    ht = heat_table()
    ht["W"] = ht[GROUPS].to_numpy(float) @ group_yields()   # A4 (revised): tap from Table H2 group yields
    ht["B"] = ht.W + H
    ht["c_prev"], ht["prev_measured"] = heel_chem(ht, element)
    d = ht[ht.in_cohort & ht[element].notna()].reset_index(drop=True)
    X = d[GROUPS].to_numpy(float)
    y = d[element].to_numpy(float) * d.B.to_numpy() - H * d.c_prev.to_numpy()
    return Design(element, d, X, y, d.B.to_numpy(), d.c_prev.to_numpy(), d.prev_measured.to_numpy(), float(d.W.sum() / d.total_charge_t.sum()))


# Predicts the final wt% of an element for selected heats.
# Combines heel contribution and charged group contents, divided by bath mass.
# Returns an array of predicted concentrations.
def predict(d: Design, e: np.ndarray, idx=slice(None)) -> np.ndarray:
    """Predicted final wt % for heats idx given group contents e."""
    H = cfg("mill")["heel_t"]
    return (H * d.c_prev[idx] + d.X[idx] @ e) / d.B[idx]
