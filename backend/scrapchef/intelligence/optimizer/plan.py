"""Turn a solver result into a PlanVersion: baskets, real mill codes, predictions, costs, binding constraints."""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from scrapchef.contracts.models import (Basket, Constraint, CostBreakdown, ElementPrediction, HeatPlan, LoadItem,
                                        PlanVersion)
from scrapchef.data.store import GROUPS, basket_profile, cfg, code_split
from .model import Context, ELS, duals, predict_heat, solve


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


def build_plan(ctx: Context, taps: list[float], plan_id: str, version: int = 1, mode: str = "chance",
               alpha: float = 0.05, spec_class: str = "plant_strict", parent: int | None = None,
               created_by: str = "agent", frozen: dict | None = None, ts: str | None = None) -> PlanVersion | None:
    res = solve(ctx, taps, mode, alpha, frozen=frozen)
    if res["x"] is None:
        return None
    mill, splits, profile = cfg("mill"), code_split(), basket_profile()
    energy_k = float(ctx.energy_k.mean()) if len(ctx.energy_k) else 0.0
    heats = []
    for h, T in enumerate(taps):
        x = np.maximum(res["x"][h], 0)
        gt = {g: round(float(v), 3) for g, v in zip(GROUPS, x) if v > 1e-3}
        pred = predict_heat(ctx, x, T, res["heel"][h])
        mat, en = float(ctx.prices @ x), float(ctx.energy_k @ x)
        heats.append(HeatPlan(
            slot=h + 1, spec_class=spec_class, target_tap_t=T, frozen=bool(frozen and h in frozen), group_t=gt,
            baskets=_baskets(gt, mill["grab_size_t"], splits, profile),
            predicted={e: ElementPrediction(mean=p["mean"], p95=p["p95"], limit=p["limit"], p_exceed=p["p_exceed"])
                       for e, p in pred.items()},
            cost=CostBreakdown(material=mat, energy=en, total=mat + en, per_t=(mat + en) / T)))
    binding = [Constraint(name=f"{d['key']}@{d['heat']}", kind=d["kind"], dual=d["dual"],
                          dual_units="$ per unit RHS", plain=d["plain"]) for d in duals(ctx, res)]
    return PlanVersion(plan_id=plan_id, version=version, parent_version=parent,
                       created_ts=ts or datetime.now().isoformat(), created_by=created_by, mode=mode,
                       alpha=alpha, heats=heats, binding=binding)


def heat_from_groups(ctx: Context, slot: int, tap_t: float, group_t: dict[str, float], heel: dict, spec_class: str, frozen: bool = False) -> HeatPlan:
    """Build a HeatPlan from an explicit mix (Control Suite edits)."""
    from .evaluate import evaluate_mix
    ev = evaluate_mix(ctx, group_t, tap_t, heel)
    gt = {g: round(float(v), 3) for g, v in group_t.items() if v > 1e-3}
    return HeatPlan(slot=slot, spec_class=spec_class, target_tap_t=tap_t, frozen=frozen, group_t=gt,
                    baskets=_baskets(gt, cfg("mill")["grab_size_t"], code_split(), basket_profile()),
                    predicted={e: ElementPrediction(mean=p["mean"], p95=p["p95"], limit=p["limit"], p_exceed=p["p_exceed"]) for e, p in ev["predicted"].items()},
                    cost=CostBreakdown(material=ev["cost"]["material"], energy=ev["cost"]["energy"], total=ev["cost"]["total"], per_t=ev["cost"]["total"] / tap_t))
