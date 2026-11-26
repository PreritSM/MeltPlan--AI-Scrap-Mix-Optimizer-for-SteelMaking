"""Heat-level features and the chronological train / held-out split (the single owner of the boundary)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .store import heat_table

FIT_FRAC = 0.70      # heats before this share of the cohort fit the estimator (boot_train); 70-85% is validation
HOLDOUT_FRAC = 0.85  # heats from this share on are held out: validation test set, back-test and the SHADOW_REPLAY set


# Computes the heel (previous-heat) chemistry for each row of the heat table.
# Uses the previous heat's measurement within a chain, else a trailing expanding median.
# Returns the heel values and a mask of which were real measurements.
def heel_chem(ht: pd.DataFrame, element: str) -> tuple[np.ndarray, np.ndarray]:
    """(c_prev, prev_measured) per row of the heat table: the previous heat's measured value as heel chemistry,
    or the trailing expanding median at a chain start / when the previous heat has no measurement."""
    prev = ht[element].shift(1)
    same_chain = ~ht.chain_start
    trailing = ht[element].expanding().median().shift(1).bfill()
    return np.where(same_chain & prev.notna(), prev, trailing), (same_chain & prev.notna()).to_numpy()


# Finds the chronological fit / held-out boundaries for a heats frame.
# Cuts the cohort at FIT_FRAC and HOLDOUT_FRAC and maps those seq_nos into `heats`.
# Returns (fit_end, test_start) row positions shared by every element.
def split_heats(heats: pd.DataFrame, ht: pd.DataFrame | None = None) -> tuple[int, int]:
    """(fit_end, test_start): row positions in `heats` (time ordered, with seq_no), cut by cohort share so every
    element's design and the replay set share one boundary even when an element is missing for a few heats."""
    cohort = (heat_table() if ht is None else ht).query("in_cohort")
    n = len(cohort)
    cuts = cohort.seq_no.to_numpy()[[int(FIT_FRAC * n), int(HOLDOUT_FRAC * n)]]
    return tuple(int(i) for i in heats.seq_no.searchsorted(cuts))
