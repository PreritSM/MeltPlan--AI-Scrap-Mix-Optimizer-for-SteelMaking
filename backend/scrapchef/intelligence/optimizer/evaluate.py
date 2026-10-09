"""Control Suite tool: re-evaluate a hand-edited mix instantly (no solve)."""
from __future__ import annotations

import numpy as np

from scrapchef.data.store import GROUPS
from .model import Context, predict_heat


def evaluate_mix(ctx: Context, group_t: dict[str, float], tap_t: float, heel_c: dict[str, float]) -> dict:
    x = np.array([group_t.get(g, 0.0) for g in GROUPS], float)
    pred = predict_heat(ctx, x, tap_t, heel_c)
    material = float(ctx.prices @ x)
    energy = float(ctx.energy_k @ x)
    tap_actual = float(ctx.yields @ x)
    return dict(predicted=pred, cost=dict(material=material, energy=energy, total=material + energy,
                                          per_t=(material + energy) / tap_actual),
                tap_actual_t=tap_actual, violations=[e for e, p in pred.items() if p["p_exceed"] > 0.05])
