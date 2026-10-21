"""Deterministic fallback for the agent: same tools, same guards, fixed rules. Runs when there is no LLM, or the LLM run fails
(timeout, API error, malformed reply, no terminal action). It always ends with a visible outcome, never silence."""
from __future__ import annotations

import re

from . import chat
from .narrator import fact_numbers, narrate
from .templates import pretty

# free-text group aliases (full group names first: '_' is a word character, so '\b3a\b' does not match inside 'scrap_3a')
ALIASES = [
    (r"scrap_25a_3ach|\b3ach\b|\b25a\d?\b", "SCRAP_25A_3ACH"),
    (r"scrap_16a_6a|\b16a\b|\b6a\b", "SCRAP_16A_6A"),
    (r"scrap_3a\b|\b3a\b", "SCRAP_3A"),
    (r"iron_units|\biron\b|\bhbi\b|\bpig\b|\bdri\b", "IRON_UNITS"),
]
NEG = r"\b(no|none|empty|unavailable|out of|zero|not available|run out|ran out|without)\b"
QTY = r"(\d+(?:\.\d+)?)\s*(?:t|tonnes?|tons?)\b"


# Interprets a free-text issue into optimiser settings.
# Detects grade groups by alias, then a tonnage cap or an exclusion keyword.
# Returns (settings, plain description), or ({}, '') if nothing was understood.
def parse_issue(text: str) -> tuple[dict, str]:
    """Free text -> optimiser settings, with a plain description of the interpretation ('' if nothing understood)."""
    t = text.lower()
    groups = []
    for pat, g in ALIASES:
        if re.search(pat, t) and g not in groups:
            groups.append(g)
    if not groups:
        return {}, ""
    qty = re.search(QTY, t)
    if qty:
        return {"group_caps_t": {groups[0]: float(qty.group(1))}}, f"{pretty(groups[0])} capped at {qty.group(1)} t"
    if re.search(NEG, t):
        return {"exclude_groups": groups}, "excluded: " + ", ".join(pretty(g) for g in groups)
    return {}, ""


# Searches for feasible re-plans from least to most relaxed settings.
# Tries as asked, then relaxed recipe bounds, a looser spec and finally 10% risk.
# Returns up to three feasible options with distinct costs.
def explore(tools, base: dict, include_base: bool = True) -> list[dict]:
    """Least-relaxed first: as asked, then relax recipe bounds, then a looser spec, then a higher risk level. Up to 3 feasible."""
    seq = [base, {**base, "relax_recipe_bounds": True}, {**base, "relax_recipe_bounds": True, "spec_class": "plant_tight"},
           {**base, "relax_recipe_bounds": True, "alpha": 0.10}][0 if include_base else 1:]
    feasible = []
    for p in seq:
        out = tools.call("run_optimizer", p)
        if out.get("status") == "optimal" and all(abs(out["cost_per_t"] - f["cost_per_t"]) > 0.05 for f in feasible):
            feasible.append(out)   # identical-cost options are not real alternatives
            if len(feasible) == 3:
                break
    return feasible


# Proposes a candidate plan with template-narrated explanations.
# Builds the decision facts, narrates them and calls the propose_plan tool.
# Returns the decision record, or an infeasible outcome if proposing failed.
def _publish(w, tools, cand: dict, trigger_text: str, alternatives: list[str]) -> dict:
    facts = dict(decision="replan", version=w.store.next_version(w.plan_id), trigger_text=trigger_text,
                 n_heats=len(cand.get("open_slots") or []), moves=cand.get("moves", []),
                 cost_delta_per_t=cand.get("delta_per_t") or 0.0, new_per_t=cand.get("cost_per_t"),
                 binding=(cand.get("binding") or [""])[0], reason="",
                 delta_breakdown=cand.get("delta_breakdown"), recipe_floors_dropped=cand.get("recipe_floors_dropped"))
    nar = narrate(facts)  # templates only: the fallback stays deterministic
    tools.seen |= fact_numbers(facts)  # derived deterministically from tool results (version, heat count), so allowed
    out = tools.call("propose_plan", dict(candidate_id=cand["candidate_id"], rationale=nar["text"], alternatives=alternatives))
    if out.get("error"):
        return dict(decision="infeasible", reason=out["error"])
    return dict(decision="replan", version=out["version"], explanation=nar["text"], narrator=nar["narrator"], facts=facts)


# Handles the case where no feasible option exists.
# Withdraws the plan via drop_plan, or just messages the supervisor if none exists.
# Returns the resulting decision record.
def _drop(tools, msg: str) -> dict:
    """Nothing feasible: inform the supervisor and withdraw the plan. If there is no plan to drop, the message alone goes out."""
    out = tools.call("drop_plan", dict(reason=msg))
    if tools.outcome.get("decision") == "reinstated":
        return dict(tools.outcome, reason="no different feasible option: rejected change withdrawn")
    if out.get("error"):
        tools.call("send_message", dict(role="supervisor", text=msg))
        return dict(decision="infeasible", reason=f"no feasible option ({out['error']})")
    return dict(decision="dropped", reason="no feasible option", explanation={"supervisor": msg})


# Deterministic agent policy used when the LLM is unavailable or fails.
# Handles questions, lab surprises / drift, issues and rejections with fixed rules.
# Always returns a visible decision record, never silence.
def fallback(w, ev: dict, tools, reason: str = "") -> dict:
    kind, text, role = ev["kind"], ev["text"], ev.get("role")
    if kind == "question":
        ans = chat.answer(w, role or "supervisor", text)
        w.post_chat(role or "supervisor", ans)
        return dict(decision="answer", explanation={role or "supervisor": ans})

    if kind in ("lab_surprise", "drift"):
        cand = tools.call("run_optimizer", {})
        if cand.get("status") in ("no_plan", "no_open_heats"):
            return dict(decision="no_change", reason=f"nothing to re-plan ({cand['status']})")
        if cand.get("status") != "optimal":   # e.g. the bath itself is contaminated: explore before giving up
            options = explore(tools, {}, include_base=False)
            if not options:
                return _drop(tools, f"{text} No feasible re-plan under the updated estimates, even with relaxed bounds, a looser spec or 10% risk.")
            return _publish(w, tools, options[0], f"{text} The default re-plan is infeasible; relaxed settings were needed.",
                            [o["candidate_id"] for o in options[1:]])
        block = w.churn_blocks(cand)
        if block:
            return dict(decision="no_change", reason=block, facts=dict(moves=cand.get("moves", [])),
                        explanation={"supervisor": f"No plan change: {block}"})
        return _publish(w, tools, cand, text, [])

    if kind in ("issue", "inbox"):
        params, meaning = parse_issue(text)
        if not params:
            msg = f"Issue from {role or 'the yard'}: \"{text}\". The automatic fallback could not interpret it; please review."
            tools.call("send_message", dict(role="supervisor", text=msg))
            tools.call("send_message", dict(role="shift_leader", text=msg))
            return dict(decision="message", reason="issue not understood by the fallback")
        options = explore(tools, params)
        if not options:
            return _drop(tools, f"{text} ({meaning}). No feasible mix even with relaxed bounds, a looser spec or 10% risk.")
        rec = _publish(w, tools, options[0], f"{text} ({meaning}).", [o["candidate_id"] for o in options[1:]])
        if rec["decision"] == "replan":
            tools.call("send_message", dict(role="crane", text="Plan change pending approval: hold heats not yet loaded until the new list arrives."))
        return rec

    if kind == "rejection":
        params, meaning = parse_issue(text)
        options = explore(tools, params) if params else [c for c in [tools.call("run_optimizer", {})] if c.get("status") == "optimal"]
        if not options:
            return _drop(tools, f"Rejected: \"{text}\". No feasible re-plan found for the rejected version.")
        note = f"Rejected: \"{text}\"." + (f" Applied: {meaning}." if meaning else " The fallback cannot interpret the reason, so the same optimum is re-proposed; use Edit for manual changes.")
        rec = _publish(w, tools, options[0], note, [o["candidate_id"] for o in options[1:]])
        if rec["decision"] != "replan":   # e.g. the optimum is the rejected plan itself (anti-churn refuses it)
            return _drop(tools, f"Rejected: \"{text}\". No different feasible plan was found.")
        return rec

    return dict(decision="message", reason=f"unhandled event kind {kind}")


# Checks an LLM chat answer against the outgoing-text guards.
# Applies the no-cost and crane rules based on the asking role.
# Returns the reason it fails, or an empty string if it passes.
def answer_problem(text: str, tools) -> str:
    """Why an LLM answer to a question fails the guards that apply to any outgoing text ('' if it passes)."""
    if not text:
        return "the answer is empty"
    try:
        role = tools.event.get("role")
        tools._check_text(text, no_cost_text=text if role == "shift_leader" else "", crane_text=text if role == "crane" else "")
        return ""
    except Exception as e:
        return str(e).replace(", then call the tool again", "")


# Checks whether an LLM chat answer passes all guards.
# Thin wrapper around answer_problem.
# Returns True when there is no problem.
def faithful_answer(text: str, tools) -> bool:
    return not answer_problem(text, tools)
