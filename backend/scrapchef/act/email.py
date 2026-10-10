"""Render the daily plan email (a MOCK channel addressed to the day-shift supervisor)."""
from __future__ import annotations

import math

from scrapchef.contracts.models import PlanVersion
from scrapchef.data.store import GROUPS
from scrapchef.intelligence.estimator.massbalance import sym


def _usd(v: float) -> str:
    return f"{'+' if v >= 0 else '-'}${abs(v):.2f}"


def _cost_lines(per_t: float, baseline_per_t: float | None, rm: dict | None) -> list[str]:
    if rm:
        h, m = rm["hist_per_t"], rm["matched_per_t"]
        return [f"Historical mix (last 30 days, same prices): ${h:.2f}/t, which breaches the limits with up to {rm['worst_p']:.0%} probability ({sym(rm['worst_el'])}).",
                f"At that same risk the optimiser would cost ${m:.2f}/t ({_usd(m - h)} vs historical).",
                f"This plan is {_usd(per_t - m)}/t against that same-risk optimum, because it is held to the 95% confidence limits.",
                _bottom_line(per_t, rm), ""]
    return [f"({per_t - baseline_per_t:+.2f} vs historical mix ${baseline_per_t:.2f}/t, not risk-matched)", ""] if baseline_per_t else [""]


def _bottom_line(per_t: float, rm: dict) -> str:
    """One plain-language line: net cost against the historical mix and what it buys in breach risk."""
    net, el = per_t - rm["hist_per_t"], sym(rm["worst_el"])
    a, b = (math.floor(p * 100 + 0.5) for p in (rm["worst_p"], rm["plan_worst_p"]))   # rounds like Math.round on the phone
    pay = "You pay a little more per tonne" if net > 0 else "You save money per tonne"
    risk = f"cut the {el} breach risk from {a}% to {b}%" if b < a else f"the {el} breach risk goes from {a}% to {b}%"
    return f"Net effect against the historical mix: about {_usd(net)}/t. {pay} and {risk}."


def render(plan: PlanVersion, date: str, baseline_per_t: float | None, rationale: str, risk_match: dict | None = None) -> dict:
    tot_t = sum(h.steel_t for h in plan.heats)
    per_t = sum(h.cost.total for h in plan.heats) / tot_t
    groups: dict[str, float] = {}
    for h in plan.heats:
        for g, t in h.group_t.items():
            groups[g] = groups.get(g, 0) + t
    worst = {el: max(h.predicted[el].p95 / h.predicted[el].limit for h in plan.heats) for el in plan.heats[0].predicted}
    lines = [f"Plan v{plan.version} for {date}: {len(plan.heats)} heats, {tot_t:.0f} t tap.", "",
             "Charge by grade group (t): " + ", ".join(f"{g} {groups[g]:.0f}" for g in GROUPS if g in groups),
             f"Cost: ${per_t:.2f}/t"] + _cost_lines(per_t, baseline_per_t, risk_match) + [
             "Worst-heat P95 as % of limit: " + ", ".join(f"{sym(el)} {v:.0%}" for el, v in worst.items()), ""]
    if plan.binding:
        shown = [c.plain for c in plan.binding[:2] if c.plain not in rationale]   # the rationale may already quote one
        if shown:
            lines += ["Binding constraints:"] + [f" - {p}" for p in shown] + [""]
    lines += [rationale]
    return dict(to="supervisor@mill.example", channel="MOCK", subject=f"ScrapChef plan v{plan.version} for {date}",
                body="\n".join(lines), plan_id=plan.plan_id, version=plan.version)
