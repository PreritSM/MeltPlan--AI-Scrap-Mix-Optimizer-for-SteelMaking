"""Render the daily plan email (a MOCK channel addressed to the day-shift supervisor)."""
from __future__ import annotations

from scrapchef.contracts.models import PlanVersion
from scrapchef.data.store import GROUPS


def render(plan: PlanVersion, date: str, baseline_per_t: float | None, rationale: str) -> dict:
    tot_t = sum(h.target_tap_t for h in plan.heats)
    per_t = sum(h.cost.total for h in plan.heats) / tot_t
    groups: dict[str, float] = {}
    for h in plan.heats:
        for g, t in h.group_t.items():
            groups[g] = groups.get(g, 0) + t
    worst = {el: max(h.predicted[el].p95 / h.predicted[el].limit for h in plan.heats) for el in plan.heats[0].predicted}
    lines = [f"Plan v{plan.version} for {date}: {len(plan.heats)} heats, {tot_t:.0f} t tap.", "",
             "Charge by grade group (t): " + ", ".join(f"{g} {groups[g]:.0f}" for g in GROUPS if g in groups),
             f"Cost: ${per_t:.2f}/t" + (f" ({per_t - baseline_per_t:+.2f} vs baseline ${baseline_per_t:.2f}/t)" if baseline_per_t else ""), "",
             "Worst-heat P95 as % of limit: " + ", ".join(f"{el.upper()} {v:.0%}" for el, v in worst.items()), ""]
    if plan.binding:
        lines += ["Binding constraints:"] + [f" - {c.plain}" for c in plan.binding[:2]] + [""]
    lines += [rationale, "", "[Open in ScrapChef]  [Reply]"]
    return dict(to="supervisor@mill.example", channel="MOCK", subject=f"ScrapChef plan v{plan.version} for {date}",
                body="\n".join(lines), plan_id=plan.plan_id, version=plan.version)
