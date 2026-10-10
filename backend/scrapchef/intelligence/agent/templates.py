"""Deterministic explanation templates (fallback when there is no API key / network). Facts in, prose out."""
from __future__ import annotations


def pretty(g: str) -> str:
    return g.replace("SCRAP_", "scrap ").replace("IRON_UNITS", "iron units").replace("OTHER_SCRAP", "other scrap")


def _moves(f):
    return sorted(f["moves"], key=lambda m: -abs(m["delta_t"]))[:2]


def _split(f) -> str:
    """Cost change split into recipe-floor relaxation vs availability, so a forced change does not read as a plain saving."""
    b = f.get("delta_breakdown")
    if not b:
        return ""
    floors = ", ".join(pretty(g) for g in f.get("recipe_floors_dropped") or {})
    return (f" Of that, {b['availability_limits_per_t']:+.2f} $/t is the availability limit and {b['dropping_recipe_floors_per_t']:+.2f} $/t "
            f"comes from dropping the minimum share of {floors or 'the recipe floors'}, a historical recipe rule worth reviewing.")


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
