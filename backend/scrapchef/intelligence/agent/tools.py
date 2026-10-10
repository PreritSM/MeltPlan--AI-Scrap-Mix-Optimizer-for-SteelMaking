"""The agent's tools: typed schemas (sent to the LLM, and used to validate its arguments) and handlers over the world's
planning services. Guards live here, not in the prompt: numbers in outgoing text must come from tool results, the crane never
receives costs, anti-churn is enforced, and nothing here can approve or release a plan."""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scrapchef.data.store import GROUPS, cfg
from .narrator import fact_numbers, unsupported

Group = Literal["SCRAP_3A", "SCRAP_16A_6A", "SCRAP_25A_3ACH", "IRON_UNITS"]
Role = Literal["email", "supervisor", "shift_leader", "crane"]
SpecClass = Literal["rebar", "sbq", "plant_tight", "plant_strict"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetState(_Strict):
    pass


class RunOptimizer(_Strict):
    exclude_groups: list[Group] = Field(default_factory=list, description="Grade groups that must not be used in the open heats.")
    group_caps_t: dict[Group, float] = Field(default_factory=dict, description="Maximum tonnes of a group available for the open heats.")
    alpha: Literal[0.01, 0.05, 0.10] | None = Field(None, description="Allowed probability of exceeding a limit per heat (default: the live setting, normally 0.05).")
    spec_class: SpecClass | None = Field(None, description="Element-limit set to plan against (default: the live one, normally plant_strict).")
    relax_recipe_bounds: bool = Field(False, description="Drop the minimum group shares (e.g. the 3A floor) and allow more clean iron "
                                                          "units. Maximum shares of scrap grades stay at the mill's demonstrated practice.")


class EvaluateMix(_Strict):
    slot: int = Field(description="Plan heat slot (1-based).")
    group_t: dict[Group, float] = Field(description="Tonnes per group for that heat.")


class ComparePlans(_Strict):
    candidate_id: str


class Rationale(_Strict):
    email: str = Field(description="Two sentences for the plan email.")
    supervisor: str = Field(description="Why this plan, with the cost change and binding constraint.")
    shift_leader: str = Field(description="Operational impact on heats not yet loaded. No prices.")
    crane: str = Field(description="What changes in the loading lists. No prices or costs.")


class ProposePlan(_Strict):
    candidate_id: str = Field(description="The candidate to send to the supervisor for approval.")
    rationale: Rationale
    alternatives: list[str] = Field(default_factory=list, description="Up to 2 other feasible candidate ids to offer as options.")


class SendMessage(_Strict):
    role: Role
    text: str = Field(description="Information-only message. Numbers must come from tool results; no prices to the crane.")


SPECS: dict[str, tuple[type[BaseModel], str]] = {
    "get_state": (GetState, "Current plan, open heats, risk under current estimates, estimate drift, availability, limits, recent heats."),
    "run_optimizer": (RunOptimizer, "Solve the chance-constrained LP for the open heats with the given settings. Returns a candidate "
                                    "(cost, moves, risk, binding constraints) or 'infeasible'. Publishes nothing."),
    "evaluate_mix": (EvaluateMix, "Predicted chemistry, exceedance probability and cost of a hand-specified mix for one heat."),
    "compare_plans": (ComparePlans, "Per-heat tonnage differences between a candidate and the live plan."),
    "propose_plan": (ProposePlan, "Send a candidate to the supervisor as a new plan version (needs human approval). "
                                  "Refused if it fails the anti-churn rule or the text contains numbers not in tool results."),
    "send_message": (SendMessage, "Send an information-only message to one role's phone."),
}


def openai_tools() -> list[dict]:
    out = []
    for name, (model, desc) in SPECS.items():
        schema = model.model_json_schema()
        schema.pop("title", None)
        out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": schema}})
    return out


class ToolError(Exception):
    """Returned to the model as an error result so it can correct itself."""


class AgentTools:
    def __init__(self, world, event: dict):
        self.w, self.event = world, event
        self.seen: set[str] = fact_numbers(event)   # numbers the agent may repeat: the event text + every tool result
        self.outcome: dict = {}                     # what the run did (proposal / messages)
        self.messages_sent: list[dict] = []
        self.candidates: list[str] = []
        self.trace: list[dict] = []                 # every call, for the Process View decision trace

    # ------------------------------------------------------------ dispatch
    def call(self, name: str, raw_args: str | dict) -> dict:
        """Validate and run one tool call. Always returns a JSON-able dict; {'error': ...} for anything the model should fix."""
        out = self._call(name, raw_args)
        self.trace.append(dict(type="tool", name=name, args=raw_args if isinstance(raw_args, dict) else _safe_json(raw_args),
                               result=_brief(out), error=out.get("error")))
        return out

    def _call(self, name: str, raw_args: str | dict) -> dict:
        if name not in SPECS:
            return {"error": f"unknown tool {name}; available: {', '.join(SPECS)}"}
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
            parsed = SPECS[name][0].model_validate(args)
        except (json.JSONDecodeError, ValidationError) as e:
            return {"error": f"invalid arguments for {name}: {str(e)[:400]}"}
        try:
            out = getattr(self, name)(parsed)
        except ToolError as e:
            return {"error": str(e)}
        except Exception as e:  # a tool crash is reported, not raised
            return {"error": f"{name} failed: {type(e).__name__}: {e}"}
        self.seen |= fact_numbers(out)
        return out

    # ------------------------------------------------------------ tools
    def get_state(self, _: GetState) -> dict:
        w = self.w
        plan = w.base_plan()
        if plan is None:
            return {"plan": None, "note": "No plan yet; the 06:00 cycle has not run."}
        cur = w.open_eval(plan) or {}
        est = {el: {g: round(float(v), 4) for g, v in zip(GROUPS, w.kf[el].e)} for el in w.kf}
        return dict(
            sim_time=w._ts(), mode=w.mode,
            plan=dict(version=plan.version, status=plan.status, n_heats=len(plan.heats), spec_class=w.spec_class, alpha=w.alpha,
                      heat_status={h.slot: w.progress.get(h.slot, "planned") for h in plan.heats},
                      open_slots=cur.get("open_slots"), open_cost_per_t=cur.get("cost_per_t"),
                      open_worst_p_exceed_under_current_estimates=cur.get("worst_p_exceed"),
                      open_tonnes_by_group={g: round(sum(h.group_t.get(g, 0.0) for h in plan.heats if h.slot in (cur.get("open_slots") or [])), 1)
                                            for g in GROUPS}),
            limits={k: v for k, v in cfg("specs")["classes"].items()},
            group_estimates_wt_pct=est, drift_alert=(w.drift_alert or {}).get("text"),
            availability_caps_t=w.caps, mock_yard_stock_t={g: round(float(v), 1) for g, v in zip(GROUPS, w.base_inventory)},
            recipe_bounds_share={g: [round(lo, 3), round(hi, 3)] for g, (lo, hi) in w.ctx.bounds.items()},
            recent_heats=[dict(heatid=h["heatid"], cu_pred=round(h["elements"]["cu"]["predicted"], 4),
                               cu_meas=None if h["elements"]["cu"]["measured"] is None else round(h["elements"]["cu"]["measured"], 4),
                               cu_z=None if h["elements"]["cu"]["z"] is None else round(h["elements"]["cu"]["z"], 2)) for h in w.heat_log[-5:]],
            anti_churn_usd_per_t=w.mill["anti_churn_usd_per_t"],
        )

    def run_optimizer(self, a: RunOptimizer) -> dict:
        _, summary = self.w.solve_candidate(a.model_dump())
        if summary.get("candidate_id"):
            self.candidates.append(summary["candidate_id"])
        return summary

    def evaluate_mix(self, a: EvaluateMix) -> dict:
        ev = self.w.evaluate(a.slot, dict(a.group_t))
        return dict(slot=a.slot, tap_t=round(ev["tap_actual_t"], 1), cost_per_t=round(ev["cost"]["per_t"], 2),
                    delta_vs_plan_per_t=round(ev["delta_cost_per_t"], 2), violations=ev["violations"],
                    predicted={el: dict(mean=round(p["mean"], 4), p95=round(p["p95"], 4), limit=p["limit"], p_exceed=round(p["p_exceed"], 3))
                               for el, p in ev["predicted"].items()})

    def compare_plans(self, a: ComparePlans) -> dict:
        cand = self.w.candidates.get(a.candidate_id)
        if not cand or cand["plan"] is None:
            raise ToolError(f"unknown or infeasible candidate {a.candidate_id}")
        live = self.w.base_plan()
        rows = []
        for hn, ho in zip(cand["plan"].heats, live.heats):
            d = {g: round(hn.group_t.get(g, 0.0) - ho.group_t.get(g, 0.0), 1) for g in GROUPS}
            if any(abs(v) >= 0.1 for v in d.values()):
                rows.append(dict(slot=ho.slot, delta_t={g: v for g, v in d.items() if abs(v) >= 0.1}))
        return dict(candidate_id=a.candidate_id, live_version=live.version, changed_heats=rows, summary=cand["summary"])

    def propose_plan(self, a: ProposePlan) -> dict:
        if self.outcome.get("version"):
            raise ToolError("a plan was already proposed in this run")
        cand = self.w.candidates.get(a.candidate_id)
        if not cand or cand["plan"] is None:
            raise ToolError(f"unknown or infeasible candidate {a.candidate_id}")
        if self.event["kind"] in ("lab_surprise", "drift"):
            block = self.w.churn_blocks(cand["summary"])
            if block:
                raise ToolError(block)
        rat = a.rationale.model_dump()
        self._check_text(" ".join(rat.values()), crane_text=rat["crane"])
        new = self.w.publish_candidate(a.candidate_id, rat, self.event["text"], a.alternatives[:2])
        self.outcome = dict(decision="replan", version=new.version, candidate_id=a.candidate_id, explanation=rat,
                            summary=cand["summary"])
        return dict(ok=True, version=new.version, status="EMAILED to supervisor for approval",
                    alternatives_offered=[x["candidate_id"] for x in self.w.alternatives.get((new.plan_id, new.version), [])])

    def send_message(self, a: SendMessage) -> dict:
        self._check_text(a.text, crane_text=a.text if a.role == "crane" else "")
        self.w.notify(a.role, a.text)
        self.messages_sent.append(dict(role=a.role, text=a.text))
        return dict(ok=True, delivered_to=a.role)

    # ------------------------------------------------------------ guards
    def _check_text(self, text: str, crane_text: str = "") -> None:
        bad = unsupported(text, self.seen)
        if bad:
            raise ToolError(f"text contains numbers that are not in any tool result of this run: {', '.join(bad[:8])}. "
                            "Remove them or use only numbers returned by tools, then call the tool again")
        if crane_text and re.search(r"\$|cost|price|usd|dollar", crane_text, re.I):
            raise ToolError("the crane role may not see prices or costs; remove them from the crane text")


def _safe_json(s: str):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


def _brief(out: dict) -> dict:
    """Compact view of a tool result for the trace (the model still receives the full result)."""
    keep = ("error", "status", "candidate_id", "cost_per_t", "delta_per_t", "moves", "worst_p_exceed", "binding", "ok", "version",
            "delivered_to", "alternatives_offered", "changed_heats", "violations", "hint")
    b = {k: out[k] for k in keep if k in out}
    if not b and "plan" in out:
        p = out.get("plan") or {}
        b = dict(plan_version=p.get("version"), status=p.get("status"), open_slots=p.get("open_slots"),
                 open_cost_per_t=p.get("open_cost_per_t"), risk=p.get("open_worst_p_exceed_under_current_estimates"))
    return b
