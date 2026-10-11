"""Server-side role projections. The UI never receives fields its role may not see."""
from __future__ import annotations

from scrapchef.contracts.models import PlanVersion
from scrapchef.intelligence.optimizer.evaluate import P_TOL

ROLES = ("email", "supervisor", "shift_leader", "crane")


def _headroom(h, status_only: bool):
    out = {}
    for el, p in h.predicted.items():
        if status_only:
            out[el] = "OK" if p.p_exceed <= 0.05 + P_TOL else "AT RISK"   # the chance constraint binds at exactly alpha by design
        else:
            out[el] = p.model_dump()
    return out


def project(plan: PlanVersion, role: str, baseline_per_t: float | None = None, explanation: dict | None = None,
            progress: dict | None = None, risk_match: dict | None = None) -> dict:
    """Return the role's view of a plan version."""
    assert role in ROLES, role
    progress = progress or {}
    base = dict(plan_id=plan.plan_id, version=plan.version, status=plan.status, mode=plan.mode,
                explanation=(explanation or {}).get(role))
    tot_t = sum(h.steel_t for h in plan.heats)
    tot_cost = sum(h.cost.total for h in plan.heats)
    per_t = tot_cost / tot_t if tot_t else 0
    groups: dict[str, float] = {}
    for h in plan.heats:
        for g, t in h.group_t.items():
            groups[g] = groups.get(g, 0) + t
    if role in ("email", "supervisor", "shift_leader"):
        base["summary"] = dict(n_heats=len(plan.heats), tonnes_by_group={g: round(t, 1) for g, t in groups.items()})
        base["heats"] = [dict(slot=h.slot, target_tap_t=h.target_tap_t, tap_t=h.steel_t, frozen=h.frozen, group_t=h.group_t,
                              headroom=_headroom(h, role == "shift_leader"), progress=progress.get(h.slot, "planned"))
                         for h in plan.heats]
    if role in ("email", "supervisor"):
        base["cost"] = dict(per_t=per_t, baseline_per_t=baseline_per_t,
                            delta_vs_baseline=(per_t - baseline_per_t) if baseline_per_t is not None else None,
                            risk_match=risk_match)
        if role == "supervisor":
            for hh, h in zip(base["heats"], plan.heats):
                hh["cost"] = h.cost.model_dump()
    if role == "supervisor":
        base["binding"] = [c.model_dump() for c in plan.binding]
        base["hand_edited"] = plan.hand_edited
    if role == "email":
        base["binding"] = [c.plain for c in plan.binding[:2]]
    if role in ("supervisor", "shift_leader", "crane"):
        # basket lists: no prices/costs ever
        base["loading"] = [dict(slot=h.slot, frozen=h.frozen, baskets=[b.model_dump() for b in h.baskets], progress=progress.get(h.slot, "planned"))
                           for h in plan.heats]
    return base
