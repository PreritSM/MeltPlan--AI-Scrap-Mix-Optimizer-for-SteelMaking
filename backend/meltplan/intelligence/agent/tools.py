"""The agent's tools: typed schemas (sent to the LLM, and used to validate its arguments) and handlers over the world's
planning services. Guards live here, not in the prompt: numbers in outgoing text must come from tool results, the crane never
receives costs, anti-churn is enforced, and nothing here can approve or release a plan."""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from meltplan.data.store import GROUPS, cfg
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


class DropPlan(_Strict):
    reason: str = Field(description="For the supervisor: why no mix is feasible and what you tried. Numbers from tool results only.")


SPECS: dict[str, tuple[type[BaseModel], str]] = {
    "get_state": (GetState, "Current plan, open heats, risk under current estimates, estimate drift, availability, limits, recent heats."),
    "run_optimizer": (RunOptimizer, "Solve the chance-constrained LP for the open heats with the given settings. Returns a candidate "
                                    "(cost, moves, risk, binding constraints) or 'infeasible'. Publishes nothing."),
    "evaluate_mix": (EvaluateMix, "Predicted chemistry, exceedance probability and cost of a hand-specified mix for one heat."),
    "compare_plans": (ComparePlans, "Per-heat tonnage differences between a candidate and the live plan."),
    "propose_plan": (ProposePlan, "Send a candidate to the supervisor as a new plan version (needs human approval). "
                                  "Refused if it fails the anti-churn rule or the text contains numbers not in tool results."),
    "send_message": (SendMessage, "Send an information-only message to one role's phone."),
    "drop_plan": (DropPlan, "No feasible mix for the open heats: inform the supervisor with the reason, then withdraw the plan "
                            "completely (no live plan; the crane stops loading). Only after run_optimizer returned infeasible. "
                            "After a rejection it instead withdraws the rejected change and puts the last approved plan back live."),
}


# Builds the OpenAI-style tool definitions sent to the LLM.
# Converts each pydantic argument model in SPECS into a JSON schema.
# Returns a list of function-tool dicts with names and descriptions.
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
    # Creates the tool handler for one agent run on one event.
    # Seeds the allowed-number set from the event and sets up outcome, trace and candidate tracking.
    # Also tracks cost figures that must never reach no-cost roles.
    def __init__(self, world, event: dict):
        self.w, self.event = world, event
        self.seen: set[str] = fact_numbers(event)   # numbers the agent may repeat: the event text + every tool result
        self.outcome: dict = {}                     # what the run did (proposal / messages)
        self.messages_sent: list[dict] = []
        self.candidates: list[str] = []
        self.trace: list[dict] = []                 # every call, for the Process View decision trace
        self.note = ""                              # why a final answer was not accepted (shown as the path reason)
        self.cost_numbers: set[str] = set()         # numbers from cost/price fields: never in text for roles without cost access

    # ------------------------------------------------------------ dispatch
    # Validates and runs one tool call, then records it in the trace.
    # Errors are returned as {'error': ...} so the model can correct itself.
    # Always returns a JSON-serialisable dict.
    def call(self, name: str, raw_args: str | dict) -> dict:
        """Validate and run one tool call. Always returns a JSON-able dict; {'error': ...} for anything the model should fix."""
        out = self._call(name, raw_args)
        self.trace.append(dict(type="tool", name=name, args=raw_args if isinstance(raw_args, dict) else _safe_json(raw_args),
                               result=_brief(out), error=out.get("error")))
        return out

    # Parses and validates tool arguments, then dispatches to the handler.
    # Turns bad arguments, ToolErrors and crashes into error results.
    # Adds numbers (and cost numbers) from the result to the allowed sets.
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
        self.cost_numbers |= _cost_numbers(out)
        return out

    # ------------------------------------------------------------ tools
    # Tool: returns a snapshot of the planning situation for the LLM.
    # Includes the live plan, open-heat risk and cost, estimates, limits, availability and recent heats.
    # Returns a short note instead when there is no live plan.
    def get_state(self, _: GetState) -> dict:
        w = self.w
        plan = w.base_plan()
        if plan is None:
            return {"plan": None, "note": "No live plan (the 06:00 cycle has not run, or the plan was dropped)."}
        cur = w.open_eval(plan) or {}
        est = {el: {g: round(float(v), 4) for g, v in zip(GROUPS, w.kf[el].e)} for el in w.kf}
        return dict(
            sim_time=w._ts(), mode=w.mode,
            plan=dict(version=plan.version, status=plan.status, n_heats=len(plan.heats), spec_class=w.spec_class, alpha=w.alpha,
                      heat_status={h.slot: w.progress.get(h.slot, "planned") for h in plan.heats},
                      open_slots=cur.get("open_slots"), open_cost_per_t=cur.get("cost_per_t"),
                      open_worst_p_exceed_under_current_estimates=cur.get("worst_p_exceed"),
                      open_tonnes_by_group={g: round(sum(h.group_t.get(g, 0.0) for h in plan.heats if h.slot in (cur.get("open_slots") or [])), 1)
                                            for g in GROUPS},
                      open_charge_t=round(sum(sum(h.group_t.values()) for h in plan.heats if h.slot in (cur.get("open_slots") or [])), 1)),
            limits={k: v for k, v in cfg("specs")["classes"].items()},
            group_estimates_wt_pct=est, drift_alert=(w.drift_alert or {}).get("text"),
            availability_caps_t=w.caps, mock_yard_stock_t={g: round(float(v), 1) for g, v in zip(GROUPS, w.base_inventory)},
            recipe_bounds_share={g: [round(lo, 3), round(hi, 3)] for g, (lo, hi) in w.ctx.bounds.items()},
            recent_heats=[dict(heatid=h["heatid"], cu_pred=round(h["elements"]["cu"]["predicted"], 4),
                               cu_meas=None if h["elements"]["cu"]["measured"] is None else round(h["elements"]["cu"]["measured"], 4),
                               cu_z=None if h["elements"]["cu"]["z"] is None else round(h["elements"]["cu"]["z"], 2)) for h in w.heat_log[-5:]],
            anti_churn_usd_per_t=w.mill["anti_churn_usd_per_t"],
            vs_history=_vs_history(w, plan),
        )

    # Tool: solves a candidate plan for the open heats with given settings.
    # Remembers the candidate id so later tools can reference it.
    # Publishes nothing; returns the candidate summary or infeasibility.
    def run_optimizer(self, a: RunOptimizer) -> dict:
        _, summary = self.w.solve_candidate(a.model_dump())
        if summary.get("candidate_id"):
            self.candidates.append(summary["candidate_id"])
        return summary

    # Tool: evaluates a hand-specified mix for one heat.
    # Reports tap, cost per tonne, delta vs plan and any violations.
    # Also returns rounded chemistry predictions per element.
    def evaluate_mix(self, a: EvaluateMix) -> dict:
        ev = self.w.evaluate(a.slot, dict(a.group_t))
        return dict(slot=a.slot, tap_t=round(ev["tap_actual_t"], 1), cost_per_t=round(ev["cost"]["per_t"], 2),
                    delta_vs_plan_per_t=round(ev["delta_cost_per_t"], 2), violations=ev["violations"] + ev["process_violations"],
                    predicted={el: dict(mean=round(p["mean"], 4), p95=round(p["p95"], 4), limit=p["limit"], p_exceed=round(p["p_exceed"], 3))
                               for el, p in ev["predicted"].items()})

    # Tool: compares a candidate with the live plan heat by heat.
    # Lists group tonnage changes of at least 0.1 t per heat.
    # Raises a ToolError for unknown or infeasible candidates.
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

    # Tool: sends a candidate to the supervisor as a new plan version.
    # Enforces anti-churn, prefers live spec/risk settings and checks all rationale text.
    # Records the proposal outcome and returns the new version and alternatives.
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
        keeps = self._keeps_live_settings
        if not keeps(cand):   # a looser spec or a higher risk is the supervisor's quality call, not a cost lever for the agent
            same = [c for c in self.candidates if c != a.candidate_id and (self.w.candidates.get(c) or {}).get("plan") is not None
                    and keeps(self.w.candidates[c])]
            if same:
                raise ToolError(f"{a.candidate_id} changes the live spec class or risk level, but {same[0]} is feasible under the live "
                                f"settings. Propose {same[0]} and offer {a.candidate_id} as an alternative")
        rat = a.rationale.model_dump()
        self._check_text(" ".join(rat.values()), no_cost_text=rat["shift_leader"], crane_text=rat["crane"])
        new = self.w.publish_candidate(a.candidate_id, rat, self.event["text"], a.alternatives[:2])
        self.outcome = dict(decision="replan", version=new.version, candidate_id=a.candidate_id, explanation=rat,
                            summary=cand["summary"])
        return dict(ok=True, version=new.version, status="EMAILED to supervisor for approval",
                    alternatives_offered=[x["candidate_id"] for x in self.w.alternatives.get((new.plan_id, new.version), [])])

    # Tool: sends an information-only message to one role's phone.
    # Checks the text for unsupported numbers and cost leaks for restricted roles.
    # Records the message and confirms delivery.
    def send_message(self, a: SendMessage) -> dict:
        self._check_text(a.text, no_cost_text=a.text if a.role == "shift_leader" else "", crane_text=a.text if a.role == "crane" else "")
        self.w.notify(a.role, a.text)
        self.messages_sent.append(dict(role=a.role, text=a.text))
        return dict(ok=True, delivered_to=a.role)

    # Tool: withdraws the plan when no feasible mix exists.
    # Requires an infeasible optimiser run (unless withdrawing a rejected change).
    # After a rejection, reinstates the last approved plan instead of dropping everything.
    def drop_plan(self, a: DropPlan) -> dict:
        if self.outcome.get("version"):
            raise ToolError("a plan was already proposed in this run; it cannot also be dropped")
        base = self.w.base_plan()
        rejected = base is not None and base.status == "REJECTED"   # withdrawing a rejected change is always allowed
        if not rejected and not any((self.w.candidates.get(c) or {}).get("plan") is None for c in self.candidates):
            raise ToolError("drop_plan needs an infeasible run_optimizer result in this run")
        self._check_text(a.reason)
        p = self.w.drop_plan(a.reason)
        if p is None:
            raise ToolError("there is no live plan to drop")
        live = self.w.current_plan()
        if live is not None:   # a rejected change was withdrawn and the last approved version is live again
            self.outcome = dict(decision="reinstated", version=live.version, dropped=p.version,
                                explanation=dict(self.w.explanation))
            return dict(ok=True, rejected_version=p.version, live_version=live.version,
                        status="rejected change withdrawn; the last approved plan is live again")
        self.outcome = dict(decision="dropped", dropped=p.version, explanation={"supervisor": a.reason})
        return dict(ok=True, dropped_version=p.version, status="DROPPED; supervisor informed, crane told not to load")

    # ------------------------------------------------------------ guards
    # Checks whether a candidate keeps the live spec class and risk level.
    # Its alpha must not be higher than the live alpha.
    # Used to stop the agent loosening quality settings to save cost.
    def _keeps_live_settings(self, cand: dict) -> bool:
        return cand["spec_class"] == self.w.spec_class and cand["alpha"] <= self.w.alpha + 1e-9

    # Guards outgoing text before it is sent.
    # Rejects numbers not found in tool results, and any cost talk or figures for crane / shift leader.
    # Raises a ToolError describing what to fix.
    def _check_text(self, text: str, no_cost_text: str = "", crane_text: str = "") -> None:
        """no_cost_text (shift leader): no currency and no cost figures, though it may say that costs go to the supervisor.
        crane_text: the same, and no talk of cost or price at all (crane texts are loading instructions)."""
        bad = unsupported(text, self.seen)
        if bad:
            raise ToolError(f"text contains numbers that are not in any tool result of this run: {', '.join(bad[:8])}. "
                            "Remove them or use only numbers returned by tools, then call the tool again")
        if crane_text and re.search(r"cost|price", crane_text, re.I):
            raise ToolError("the crane role may not see prices or costs; remove them from the crane text")
        for t in (x for x in (no_cost_text, crane_text) if x):
            leaked = sorted(n for n in _numbers_in(t) & self.cost_numbers if "." in n or int(n) >= 100)
            if re.search(r"\$|€|usd|dollar", t, re.I) or leaked:
                raise ToolError("the shift_leader and crane roles may not see prices or costs"
                                + (f" (cost figures in the text: {', '.join(leaked[:5])})" if leaked else "")
                                + "; remove them from the text for those roles")


# Compares the plan with the last 30 days' actual mix, as the email shows it.
# Reports plan, historical and same-risk optimum costs and breach probabilities.
# Returns None if no risk-matched baseline is available.
def _vs_history(w, plan) -> dict | None:
    """The plan against the last 30 days' actual mix, as the email shows it (all heats of the plan, same assumed prices)."""
    rm = w.risk_match(plan)
    if not rm:
        return None
    per_t = sum(h.cost.total for h in plan.heats) / sum(h.steel_t for h in plan.heats)
    h, m = rm["hist_per_t"], rm["matched_per_t"]
    return dict(plan_cost_per_t=round(per_t, 2), historical_mix_cost_per_t=round(h, 2),
                historical_mix_worst_breach_probability=round(rm["worst_p"], 2), historical_mix_worst_element=rm["worst_el"], plan_worst_breach_probability=round(rm["plan_worst_p"], 2),
                optimiser_at_historical_risk_cost_per_t=round(m, 2), saving_at_same_risk_per_t=round(h - m, 2),
                plan_minus_same_risk_optimum_per_t=round(per_t - m, 2),
                note="the plan costs more than the same-risk optimum because it is held to the 95% limits")


COST_KEY = re.compile(r"cost|price|per_t|usd|saving", re.I)   # cost_per_t, delta_per_t, delta_breakdown.*_per_t, vs_history ...


# Collects numbers stored under cost-like keys anywhere in a tool result.
# Recurses through dicts and lists, also picking up '$' amounts inside strings.
# Returns the set of cost figure forms that must not leak to restricted roles.
def _cost_numbers(v, under_cost: bool = False) -> set[str]:
    """Numbers stored under cost-like keys anywhere in a tool result."""
    out: set[str] = set()
    if isinstance(v, dict):
        for k, x in v.items():
            out |= _cost_numbers(x, under_cost or bool(COST_KEY.search(str(k))))
    elif isinstance(v, (list, tuple)):
        for x in v:
            out |= _cost_numbers(x, under_cost)
    elif under_cost and isinstance(v, (int, float)) and not isinstance(v, bool) and v:
        out |= _cost_forms(float(v))
    elif isinstance(v, str):   # "$15.65/t" inside binding-constraint text
        out |= {n for m in re.findall(r"\$\s?(\d+(?:\.\d+)?)", v) for n in _cost_forms(float(m))}
    return out


# Lists the ways a cost figure might be quoted in text.
# Includes the raw value and 2 decimals; coarser roundings only for values >= 100.
# Returns the set of normalised number strings.
def _cost_forms(v: float) -> set[str]:
    """How a cost figure would be quoted: as given or to 2 decimals; shorter roundings only for prices (>= 100), so a
    tonnage such as 7.8 t is not mistaken for a rounded cost delta of 7.78."""
    forms = [repr(abs(v)), f"{abs(v):.2f}"] + ([f"{abs(v):.0f}", f"{abs(v):.1f}"] if abs(v) >= 100 else [])
    return set().union(*(fact_numbers(f) for f in forms))


# Extracts all numbers from a text in normalised form.
# Delegates to fact_numbers.
# Returns a set of number strings.
def _numbers_in(text: str) -> set[str]:
    return fact_numbers(text)


# Parses a JSON string if possible.
# Falls back to the original value when parsing fails.
# Used to show tool arguments in the trace.
def _safe_json(s: str):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


# Builds a compact view of a tool result for the decision trace.
# Keeps only key fields, or a plan summary for get_state results.
# The model itself still receives the full result.
def _brief(out: dict) -> dict:
    """Compact view of a tool result for the trace (the model still receives the full result)."""
    keep = ("error", "status", "candidate_id", "cost_per_t", "delta_per_t", "moves", "worst_p_exceed", "binding", "ok", "version",
            "delivered_to", "alternatives_offered", "dropped_version", "changed_heats", "violations", "hint")
    b = {k: out[k] for k in keep if k in out}
    if not b and "plan" in out:
        p = out.get("plan") or {}
        b = dict(plan_version=p.get("version"), status=p.get("status"), open_slots=p.get("open_slots"),
                 open_cost_per_t=p.get("open_cost_per_t"), risk=p.get("open_worst_p_exceed_under_current_estimates"))
    return b
