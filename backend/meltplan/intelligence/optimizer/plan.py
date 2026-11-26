"""Turn a solver result into a PlanVersion: baskets, real mill codes, predictions, costs, binding constraints."""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from meltplan.contracts.models import (Basket, Constraint, CostBreakdown, ElementPrediction, HeatPlan, LoadItem,
                                        PlanVersion)
from meltplan.data.store import GROUPS, basket_profile, cfg, code_split
from .model import Context, ELS, duals, predict_heat, solve, tap_of


# Splits a heat's group tonnages into a basket-by-basket loading list.
# Divides each group between two baskets by typical basket mass, then into real mill codes.
# Returns Basket objects with grab order, tonnes and number of grabs per item.
def _baskets(group_t: dict[str, float], grab_t: float, splits, profile) -> list[Basket]:
    """Split each group across baskets in proportion to basket target mass; groups to real codes by recent usage."""
    total = sum(group_t.values())
    prof = {int(r.basket_no): r.p50 for r in profile.itertuples()}
    p1, p2 = prof.get(1, 80.0), prof.get(2, 56.0)
    n_b = 2
    frac = [p1 / (p1 + p2), p2 / (p1 + p2)]
    out = []
    for b in range(n_b):
        items, order = [], 1
        for g, t in sorted(group_t.items(), key=lambda kv: -kv[1]):
            tb = t * frac[b]
            if tb < 0.05:
                continue
            for code, name, share in sorted(splits.get(g, [(0, g, 1.0)]), key=lambda s: -s[2]):
                tc = tb * share
                if tc < 0.05:
                    continue
                items.append(LoadItem(grab_order=order, mat_code=code, mat_name=name, group=g,
                                      tonnes=round(tc, 2), grabs=max(1, math.ceil(tc / grab_t))))
                order += 1
        out.append(Basket(basket_no=b + 1, target_t=round(total * frac[b], 1), items=items))
    return out


# Solves the optimiser and converts the result into a PlanVersion.
# Builds each heat's baskets, chemistry predictions and costs, plus the binding constraints.
# Returns None if the solve is infeasible.
def build_plan(ctx: Context, taps: list[float], plan_id: str, version: int = 1, mode: str = "chance",
               alpha: float = 0.05, spec_class: str = "plant_strict", parent: int | None = None,
               created_by: str = "agent", frozen: dict | None = None, ts: str | None = None, slot0: int = 1) -> PlanVersion | None:
    """slot0: plan slot of taps[0] (re-plans solve only the heats that are not yet tapped)."""
    res = solve(ctx, taps, mode, alpha, frozen=frozen)
    if res["x"] is None:
        return None
    mill, splits, profile = cfg("mill"), code_split(), basket_profile()
    heats = []
    for h, T in enumerate(taps):
        x = np.maximum(res["x"][h], 0)
        gt = {g: round(float(v), 3) for g, v in zip(GROUPS, x) if v > 1e-3}
        tap = tap_of(ctx, x)
        pred = predict_heat(ctx, x, None, res["heel"][h])
        mat, en = float(ctx.prices @ x), float(ctx.energy_k @ x)
        heats.append(HeatPlan(
            slot=h + slot0, spec_class=spec_class, target_tap_t=T, tap_t=round(tap, 3), frozen=bool(frozen and h in frozen), group_t=gt,
            baskets=_baskets(gt, mill["grab_size_t"], splits, profile),
            predicted={e: ElementPrediction(mean=p["mean"], p95=p["p95"], limit=p["limit"], p_exceed=p["p_exceed"])
                       for e, p in pred.items()},
            cost=CostBreakdown(material=mat, energy=en, total=mat + en, per_t=(mat + en) / tap)))
    binding = [Constraint(name=f"{d['key']}@{d['heat'] + slot0}" if d["heat"] is not None else f"{d['key']}@horizon", kind=d["kind"], dual=d["dual"],
                          dual_units="$ per unit RHS", plain=d["plain"]) for d in duals(ctx, res, slot0)]
    return PlanVersion(plan_id=plan_id, version=version, parent_version=parent,
                       created_ts=ts or datetime.now().isoformat(), created_by=created_by, mode=mode,
                       alpha=alpha, heats=heats, binding=binding)


# Builds a HeatPlan from an explicitly given group mix (Mix Editor edits).
# Evaluates the mix's chemistry, tap and cost without running a solve.
# Returns the HeatPlan with baskets and predictions filled in.
def heat_from_groups(ctx: Context, slot: int, tap_t: float, group_t: dict[str, float], heel: dict, spec_class: str, frozen: bool = False) -> HeatPlan:
    """Build a HeatPlan from an explicit mix (Mix Editor edits)."""
    from .evaluate import evaluate_mix
    ev = evaluate_mix(ctx, group_t, tap_t, heel)
    gt = {g: round(float(v), 3) for g, v in group_t.items() if v > 1e-3}
    return HeatPlan(slot=slot, spec_class=spec_class, target_tap_t=tap_t, tap_t=round(ev["tap_actual_t"], 3), frozen=frozen, group_t=gt,
                    baskets=_baskets(gt, cfg("mill")["grab_size_t"], code_split(), basket_profile()),
                    predicted={e: ElementPrediction(mean=p["mean"], p95=p["p95"], limit=p["limit"], p_exceed=p["p_exceed"]) for e, p in ev["predicted"].items()},
                    cost=CostBreakdown(material=ev["cost"]["material"], energy=ev["cost"]["energy"], total=ev["cost"]["total"], per_t=ev["cost"]["per_t"]))
