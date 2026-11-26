"""Narrator: turns structured facts into per-audience text, plus the number check every outgoing agent text must pass
(every number in the text must appear in the facts / tool results). The LLM never computes."""
from __future__ import annotations

import json
import re

from .templates import render

AUDIENCES = ("email", "supervisor", "shift_leader", "crane")


# Extracts all numbers from a string in a normalised form.
# Strips signs and trailing zeros so '1.50' and '1.5' compare equal.
# Returns the set of number strings.
def _numbers(s: str) -> set[str]:
    return {n.lstrip("+-").rstrip("0").rstrip(".") if "." in n else n.lstrip("+-") for n in re.findall(r"[-+]?\d+(?:\.\d+)?", s)}


# Collects every number that appears in a nested fact structure.
# Adds rounded forms (0-4 decimals) and percentage forms of fractions.
# Returns the set of allowed number strings for the faithfulness check.
def fact_numbers(f) -> set[str]:
    """Every number in a nested structure, as given and rounded to 0-4 decimals; fractions also as percentages (0.05 -> 5)."""
    out: set[str] = set()

    # Recursively visits dicts, lists and scalars in the facts.
    # Adds numbers from ints, floats and strings to the output set.
    # Booleans and None are ignored.
    def walk(v):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(k); walk(x)
        elif isinstance(v, (list, tuple, set)):
            for x in v:
                walk(x)
        elif isinstance(v, bool) or v is None:
            return
        elif isinstance(v, (int, float)):
            out.update(_numbers(repr(abs(v))) if "e" not in repr(v) else set())
            for d in range(5):
                out.update(_numbers(f"{abs(v):.{d}f}"))
                if 0 < abs(v) <= 1:
                    out.update(_numbers(f"{abs(v) * 100:.{d}f}"))
        elif isinstance(v, str):
            out.update(_numbers(v))
    walk(f)
    return out


# Finds numbers in a text that are not backed by the facts.
# facts can be a nested structure or a precomputed number set.
# Returns the sorted list of unsupported numbers.
def unsupported(text: str, facts) -> list[str]:
    """Numbers in `text` that do not appear in `facts` (a nested structure, or a precomputed set from fact_numbers)."""
    allowed = facts if isinstance(facts, set) else fact_numbers(facts)
    return sorted(_numbers(text) - allowed)


# Checks that every number in a text appears in the facts.
# Used to reject LLM output that invents or computes numbers.
# Returns True when there are no unsupported numbers.
def faithful(text: str, facts) -> bool:
    """True if every number in `text` appears in `facts`."""
    return not unsupported(text, facts)


SYSTEM = ("You explain steel scrap charge-plan changes to a plant audience. Use ONLY facts in the JSON; do not do arithmetic "
          "or introduce any number not in the JSON. Never mention prices or costs to the crane audience. Max 2 sentences.")


# Produces per-audience explanation text for a decision's facts.
# Tries the LLM first and keeps its text only if faithful (and price-free for the crane).
# Falls back to templates and records which narrator was used per audience.
def narrate(facts: dict, llm=None) -> dict:
    """Returns {text: {audience: str}, narrator: {audience: 'llm'|'template'}}. Uses the LLM only if given and available."""
    out, used = {}, {}
    for a in AUDIENCES:
        t = None
        if llm is not None and llm.available:
            try:
                t = llm.chat([{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": f"Audience: {a}\nFacts: {json.dumps(facts)}"}], timeout=20, max_tokens=300).content.strip()
            except Exception:
                t = None
        if t and faithful(t, facts) and not (a == "crane" and re.search(r"\$|cost|price", t, re.I)):
            out[a], used[a] = t, "llm"
        else:
            out[a], used[a] = render(facts, a), "template"
    return dict(text=out, narrator=used)
