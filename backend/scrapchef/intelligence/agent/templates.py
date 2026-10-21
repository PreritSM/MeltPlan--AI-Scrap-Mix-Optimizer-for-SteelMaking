"""Deterministic explanation templates (fallback when there is no API key / network). Facts in, prose out."""
from __future__ import annotations


# Converts an internal grade-group key into a readable label.
# For example SCRAP_3A becomes 'scrap 3A' and IRON_UNITS becomes 'iron units'.
# Returns the display string.
def pretty(g: str) -> str:
    return g.replace("SCRAP_", "scrap ").replace("IRON_UNITS", "iron units").replace("OTHER_SCRAP", "other scrap")


# Picks the most significant tonnage moves from a decision's facts.
# Sorts moves by absolute change in tonnes, largest first.
# Returns at most the top two moves.
def _moves(f):
    return sorted(f["moves"], key=lambda m: -abs(m["delta_t"]))[:2]


# Explains how the cost change splits between availability and recipe-floor relaxation.
# Uses the decision's delta_breakdown and lists any recipe floors that were dropped.
# Returns an extra sentence, or an empty string if there is no breakdown.
def _split(f) -> str:
    """Cost change split into recipe-floor relaxation vs availability, so a forced change does not read as a plain saving."""
    b = f.get("delta_breakdown")
    if not b:
        return ""
    floors = ", ".join(pretty(g) for g in f.get("recipe_floors_dropped") or {})
    return (f" Of that, {b['availability_limits_per_t']:+.2f} $/t is the availability limit and {b['dropping_recipe_floors_per_t']:+.2f} $/t "
            f"comes from dropping the minimum share of {floors or 'the recipe floors'}, a historical recipe rule worth reviewing.")


# Renders a deterministic explanation of a plan decision for one audience.
# Supervisor, shift leader, crane and email each get different levels of detail.
# Returns the prose text; used as the fallback when no LLM is available.
def render(f: dict, audience: str) -> str:
    if f["decision"] == "no_change":
        return f"No plan change: {f['reason']}"
    m = _moves(f)
    mv = "; ".join(f"{pretty(x['group'])} {x['delta_t']:+.1f} t" for x in m)
    trig = f["trigger_text"]
    if audience == "supervisor":
        cause = f" Binding: {f['binding']}." if f.get("binding") else ""
        return (f"{trig} Re-planned {f['n_heats']} open heat(s): {mv}. Cost change {f['cost_delta_per_t']:+.2f} $/t "
                f"(now {f['new_per_t']:.2f} $/t).{_split(f)}{cause}")
    if audience == "shift_leader":
        return f"Plan changed for {f['n_heats']} heat(s) not yet loaded: {mv}. Reason: {trig}"
    if audience == "crane":
        return f"PLAN CHANGED: {mv}. Check the new basket list before the next grab."
    return (f"Plan v{f['version']}: {trig} Net moves: {mv}. Cost {f['cost_delta_per_t']:+.2f} $/t vs previous version.{_split(f)}")
