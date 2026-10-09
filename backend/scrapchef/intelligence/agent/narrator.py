"""Narrator: turns structured ExplanationFacts into per-audience text. The LLM never computes; its output is rejected
unless every number in it appears in the facts (then the template is used)."""
from __future__ import annotations

import json
import os
import re

from .templates import render

AUDIENCES = ("email", "supervisor", "shift_leader", "crane")
MODEL = os.environ.get("SCRAPCHEF_MODEL", "claude-sonnet-5-5")


def _numbers(s: str) -> set[str]:
    return {n.lstrip("+-").rstrip("0").rstrip(".") if "." in n else n.lstrip("+-") for n in re.findall(r"[-+]?\d+(?:\.\d+)?", s)}


def _fact_numbers(f: dict) -> set[str]:
    out = set()
    def walk(v):
        if isinstance(v, dict):
            [walk(x) for x in v.values()]
        elif isinstance(v, list):
            [walk(x) for x in v]
        elif isinstance(v, (int, float)):
            for d in (0, 1, 2):
                out.update(_numbers(f"{abs(v):.{d}f}"))
        elif isinstance(v, str):
            out.update(_numbers(v))
    walk(f)
    return out


def faithful(text: str, facts: dict) -> bool:
    return _numbers(text) <= _fact_numbers(facts)


def _llm(facts: dict, audience: str) -> str | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model=MODEL, max_tokens=250,
            system=("You explain steel scrap charge-plan changes to a plant audience. Use ONLY facts in the JSON; do not do arithmetic "
                    "or introduce any number not in the JSON. Max 2 sentences."),
            messages=[{"role": "user", "content": f"Audience: {audience}\nFacts: {json.dumps(facts)}"}])
        return msg.content[0].text.strip()
    except Exception:
        return None


def narrate(facts: dict) -> dict:
    """Returns {audience: text} and which narrator produced it."""
    out, used = {}, {}
    for a in AUDIENCES:
        t = _llm(facts, a)
        if t and faithful(t, facts):
            out[a], used[a] = t, "llm"
        else:
            out[a], used[a] = render(facts, a), "template"
    return dict(text=out, narrator=used)
