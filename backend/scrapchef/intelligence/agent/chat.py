"""Chat answers from structured state only (template). The LLM may rephrase but never computes."""
from __future__ import annotations

from .templates import pretty


def answer(world, role: str, text: str) -> str:
    p = world.current_plan()
    if p is None:
        return "No plan yet. Step to the 06:00 cycle first."
    t = text.lower()
    tot = sum(h.target_tap_t for h in p.heats)
    per_t = sum(h.cost.total for h in p.heats) / tot
    if any(k in t for k in ("why", "change", "reason")) and world.decisions:
        d = world.decisions[-1]
        return d["explanation"].get(role if role in d["explanation"] else "supervisor", "No change recorded.") if d.get("explanation") else "No change recorded."
    if any(k in t for k in ("cost", "price", "$")) and role in ("supervisor", "email"):
        return f"Plan v{p.version} costs ${per_t:.2f}/t over {len(p.heats)} heats."
    if any(k in t for k in ("risk", "copper", "cu", "limit")):
        worst = max(p.heats, key=lambda h: h.predicted["cu"].p95)
        pr = worst.predicted["cu"]
        return f"Worst Cu P95 is {pr.p95:.3f}% against a {pr.limit:.3f}% limit (heat {worst.slot})."
    g = {}
    for h in p.heats:
        for k, v in h.group_t.items():
            g[k] = g.get(k, 0) + v
    return f"Plan v{p.version} ({p.status}): " + ", ".join(f"{pretty(k)} {v:.0f} t" for k, v in g.items())
