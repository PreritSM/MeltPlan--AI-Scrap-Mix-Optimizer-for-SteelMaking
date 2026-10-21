"""Chat answers from structured state only (template). The LLM may rephrase but never computes."""
from __future__ import annotations

from .templates import pretty


# Finds the newest decision explanation that a role is allowed to see.
# Walks decisions newest first; non-crane roles may fall back to other roles' text.
# Returns the explanation string, or None if there is none.
def _last_explanation(world, role: str) -> str | None:
    """The newest agent or plan explanation this role may see (the crane only gets crane text)."""
    for d in reversed(world.decisions):
        ex = d.get("explanation") or {}
        if role in ex:
            return ex[role]
        if role != "crane":
            for k in ("supervisor", "email", "shift_leader"):
                if k in ex:
                    return ex[k]
    return None


# Answers a chat question from structured plan state only (no computation by an LLM).
# Matches keywords for why / cost / risk and collects up to three relevant facts.
# Falls back to a tonnage summary of the current plan when nothing matches.
def answer(world, role: str, text: str) -> str:
    """Up to three facts that match the question: the latest explanation (why), cost vs history, worst-case Cu."""
    p = world.current_plan()
    if p is None:
        return "No plan yet. Step to the 06:00 cycle first."
    t = text.lower()
    tot = sum(h.steel_t for h in p.heats)
    per_t = sum(h.cost.total for h in p.heats) / tot
    parts = []
    if any(k in t for k in ("why", "change", "reason")):
        ex = _last_explanation(world, role)
        if ex:
            parts.append(ex)
    if any(k in t for k in ("cost", "price", "$", "history", "cheaper", "saving")) and role in ("supervisor", "email"):
        rm = world.risk_match(p)
        hist = (f" The last 30 days' mix costs ${rm['hist_per_t']:.2f}/t at the same prices; the optimiser at that mix's risk would cost "
                f"${rm['matched_per_t']:.2f}/t.") if rm else ""
        parts.append(f"Plan v{p.version} costs ${per_t:.2f}/t over {len(p.heats)} heats.{hist}")
    if any(k in t for k in ("risk", "copper", "cu", "limit", "bind")):
        worst = max(p.heats, key=lambda h: h.predicted["cu"].p95)
        pr = worst.predicted["cu"]
        parts.append(f"Worst Cu P95 is {pr.p95:.3f}% against a {pr.limit:.3f}% limit (heat {worst.slot}).")
    if parts:
        return " ".join(parts[:3])
    g = {}
    for h in p.heats:
        for k, v in h.group_t.items():
            g[k] = g.get(k, 0) + v
    return f"Plan v{p.version} ({p.status}): " + ", ".join(f"{pretty(k)} {v:.0f} t" for k, v in g.items())
