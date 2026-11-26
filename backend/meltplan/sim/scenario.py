"""Scripted demo inputs (MOCK). The supplier email caps 3A at about 85% of what the open heats plan to use: the plan sits on
the 3A recipe floor (demonstrated practice), so the plain re-plan is infeasible and the agent has to explore."""
from __future__ import annotations


# Builds the scripted supplier email for the demo disruption.
# Sums the 3A tonnes planned for still-open heats and caps supply at ~85% of that.
# Returns the email text, which forces the agent to explore beyond a plain re-plan.
def supplier_email(world) -> str:
    plan = world.base_plan()
    open_slots = (world.open_eval(plan) or {}).get("open_slots") or []
    planned = sum(h.group_t.get("SCRAP_3A", 0.0) for h in plan.heats if h.slot in open_slots) if plan else 700.0
    cap = max(10.0, round(0.85 * planned, -1))  # below the 3A floor, so the plain re-plan is infeasible; options stay distinct
    return (f"Logistics: the 3A delivery is delayed until Thursday. The yard can give the melt shop only {cap:.0f} t of 3A "
            f"for the rest of today's heats.")
