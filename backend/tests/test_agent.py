"""Agent: tools and guards, the deterministic fallback, and the LLM loop driven by a scripted fake LLM."""
import json

import numpy as np
import pytest

from meltplan.intelligence.agent import loop
from meltplan.intelligence.agent.llm import Reply, ToolCall
from meltplan.intelligence.agent.policy import parse_issue
from meltplan.intelligence.agent.tools import AgentTools, openai_tools
from meltplan.sim.scenario import supplier_email
from meltplan.sim.world import World


class FakeLLM:
    """Replays scripted replies; each item is a Reply or a callable(messages) -> Reply (to read earlier tool results)."""
    available = True

    # Creates a fake LLM that replays a script of replies.
    # Each item is a Reply or a callable that builds one from the messages.
    # Records every call's messages for assertions.
    def __init__(self, script):
        self.script, self.calls = list(script), []

    # Returns a fixed description for the fake LLM.
    # Matches the real LLM's interface.
    # Used in decision records.
    def describe(self):
        return "fake"

    # Returns the next scripted reply for a chat call.
    # Callables receive the messages so they can read earlier tool results.
    # Returns a plain 'done' reply once the script is used up.
    def chat(self, messages, tools=None, timeout=30, **kw):
        self.calls.append(messages)
        item = self.script.pop(0) if self.script else Reply(content="done")
        return item(messages) if callable(item) else item


# Builds a scripted Reply containing one tool call.
# Arguments are JSON-encoded as the real model would send them.
# i makes the tool call id unique.
def call(name, args, i=0):
    return Reply(tool_calls=[ToolCall(f"c{i}", name, json.dumps(args))])


# Returns the most recent tool result in a message list.
# Parses the content of the last 'tool' role message as JSON.
# Used by scripted replies that depend on earlier results.
def last_result(messages):
    return json.loads([m for m in messages if m["role"] == "tool"][-1]["content"])


# Fixture: a fresh SHADOW_REPLAY world with a 06:00 plan.
# Created anew for each test.
# Used by most agent and Mix Editor tests.
@pytest.fixture()
def world():
    w = World("SHADOW_REPLAY")
    w.make_plan()
    return w


# Checks the tool schemas sent to the LLM are complete and valid.
# Verifies the exact set of tool names.
# Every tool's parameters must be a JSON object schema.
def test_tool_schemas_are_valid_function_specs():
    names = {t["function"]["name"] for t in openai_tools()}
    assert names == {"get_state", "run_optimizer", "evaluate_mix", "compare_plans", "propose_plan", "send_message", "drop_plan"}
    assert all(t["function"]["parameters"]["type"] == "object" for t in openai_tools())


# Checks free-text issue parsing in the fallback policy.
# Covers a tonnage cap, an unavailable group and an empty pile.
# Unrelated text must yield no settings.
def test_parse_issue():
    assert parse_issue("Only 590 t of 3A left today")[0] == {"group_caps_t": {"SCRAP_3A": 590.0}}
    assert parse_issue("No 3ACh available")[0] == {"exclude_groups": ["SCRAP_25A_3ACH"]}
    assert parse_issue("3ACh pile is empty")[0] == {"exclude_groups": ["SCRAP_25A_3ACH"]}
    assert parse_issue("crane 2 is slow")[0] == {}


# Checks the tool guards on arguments and outgoing text.
# Rejects invalid groups, invented numbers and cost talk to the crane or shift leader.
# Allows costs to the supervisor and cost-free text to the shift leader.
def test_guards_reject_bad_arguments_invented_numbers_and_crane_costs(world):
    t = AgentTools(world, dict(kind="issue", text="x", role="shift_leader"))
    assert "invalid arguments" in t.call("run_optimizer", '{"exclude_groups": ["GOLD"]}')["error"]
    cand = t.call("run_optimizer", {})
    rat = dict(email="ok", supervisor="Saves 999.99 $/t", shift_leader="ok", crane="ok")
    err = t.call("propose_plan", dict(candidate_id=cand["candidate_id"], rationale=rat))["error"]
    assert "numbers" in err and "999.99" in err   # the offending number is named so the model can fix it
    assert "crane" in t.call("send_message", dict(role="crane", text="this costs more"))["error"]
    # the shift leader may not see costs either, also without a cost word (live run: "runs at 496.54 per tonne")
    per_t = str(cand["cost_per_t"])
    assert "cost figures" in t.call("send_message", dict(role="shift_leader", text=f"The plan runs at {per_t} overall."))["error"]
    assert t.call("send_message", dict(role="supervisor", text=f"The plan runs at {per_t} $/t."))["ok"]
    assert t.call("send_message", dict(role="shift_leader", text="Cost figures go to the supervisor, not to this role."))["ok"]


# Checks anti-churn blocks a re-plan with no real saving.
# A lab-surprise candidate with unchanged estimates costs the same.
# propose_plan must refuse it.
def test_anti_churn_blocks_a_pointless_replan(world):
    t = AgentTools(world, dict(kind="lab_surprise", text="Cu +3.1σ.", role=None))
    cand = t.call("run_optimizer", {})  # same estimates -> same plan -> ~0 $/t change
    rat = dict(email="ok", supervisor="ok", shift_leader="ok", crane="ok")
    assert "Anti-churn" in t.call("propose_plan", dict(candidate_id=cand["candidate_id"], rationale=rat))["error"]


# Checks the fallback handles the supplier email end to end.
# Default run is infeasible, the relaxed run is proposed and the crane is told to hold.
# Also checks the cost change is split into availability and dropped recipe floors.
def test_fallback_handles_the_supplier_email_with_costed_options(world):
    text = supplier_email(world)
    rec = loop.handle(world, dict(kind="inbox", text=text, role="logistics", ts=world._ts()), world.llm)
    assert rec["path"] == "fallback" and rec["decision"] == "replan"
    runs = [s for s in rec["steps"] if s["name"] == "run_optimizer"]
    assert runs[0]["result"]["status"] == "infeasible" and runs[1]["result"]["status"] == "optimal"
    assert world.current_plan().version == rec["version"] and world.caps["SCRAP_3A"] > 0
    assert any(m["role"] == "crane" for m in rec["messages"])
    # the forced change is split: the shortage costs money, the apparent saving is the dropped 3A floor
    cand = world.candidates[[s for s in runs if s["result"]["status"] == "optimal"][0]["result"]["candidate_id"]]["summary"]
    b = cand["delta_breakdown"]
    assert b["availability_limits_per_t"] >= 0 and "SCRAP_3A" in cand["recipe_floors_dropped"]
    assert abs(b["availability_limits_per_t"] + b["dropping_recipe_floors_per_t"] - cand["delta_per_t"]) <= 0.02
    assert "availability limit" in rec["explanation"]["supervisor"]


# Checks the crane's hold message disappears once the plan is released.
# The message stays in the log, marked as expired by the release.
# Guards message expiry.
def test_crane_hold_message_expires_on_release(world):
    loop.handle(world, dict(kind="inbox", text=supplier_email(world), role="logistics", ts=world._ts()), world.llm)
    assert any("hold" in m["text"].lower() for m in world.phone("crane")["messages"])
    world.approve()
    world.release()
    assert world.phone("crane")["messages"] == []
    assert world.messages["crane"][0]["expired"].endswith("released")   # kept in the log


# Checks a scripted LLM run that explores then proposes.
# Runs the capped optimiser, then the relaxed one, then proposes the relaxed candidate.
# Verifies the path, decision, tools used and number of LLM calls.
def test_llm_loop_explores_proposes_and_offers_an_alternative(world):
    cap = {"group_caps_t": {"SCRAP_3A": 590.0}}

    # Scripted reply that proposes the last optimiser candidate.
    # Builds a rationale using the candidate's cost figure.
    # Returns a Reply with a propose_plan tool call.
    def propose(messages):
        relaxed = last_result(messages)
        rat = dict(email=f"3A capped; new plan costs {relaxed['cost_per_t']} $/t.", supervisor=f"Relaxed recipe bounds: {relaxed['cost_per_t']} $/t.",
                   shift_leader="Open heats use less 3A.", crane="Basket lists change: less 3A.")
        return Reply(tool_calls=[ToolCall("p", "propose_plan", json.dumps(dict(candidate_id=relaxed["candidate_id"], rationale=rat)))])

    llm = FakeLLM([call("run_optimizer", cap), call("run_optimizer", {**cap, "relax_recipe_bounds": True}, 1), propose,
                   Reply(content="Proposed the relaxed plan.")])
    rec = loop.handle(world, dict(kind="issue", text="Only 590 t of 3A left today.", role="shift_leader", ts=world._ts()), llm)
    assert rec["path"] == "llm" and rec["decision"] == "replan" and world.current_plan().version == rec["version"]
    assert rec["tools"] == ["get_state", "run_optimizer", "run_optimizer", "propose_plan"]
    assert len(llm.calls) == 4


# Checks a run that only messages the crane gets one reminder.
# After the nudge, the scripted LLM proposes the relaxed candidate.
# Verifies the reminder text was sent to the model.
def test_message_without_a_proposal_gets_one_reminder(world):
    """A run that finds a feasible candidate, messages the crane and stops is reminded once, then proposes."""
    cap = {"group_caps_t": {"SCRAP_3A": 590.0}}

    # Scripted reply that proposes the relaxed optimiser candidate.
    # Reads the candidate id from the tool result before send_message.
    # Returns a Reply with a propose_plan tool call.
    def propose(messages):
        cid = [m for m in messages if m["role"] == "tool"][-2]   # the relaxed run_optimizer result (before send_message)
        cid = json.loads(cid["content"])["candidate_id"]
        rat = dict(email="3A capped.", supervisor="Relaxed recipe bounds.", shift_leader="Open heats use less 3A.", crane="Less 3A.")
        return Reply(tool_calls=[ToolCall("p", "propose_plan", json.dumps(dict(candidate_id=cid, rationale=rat)))])

    llm = FakeLLM([call("run_optimizer", cap), call("run_optimizer", {**cap, "relax_recipe_bounds": True}, 1),
                   call("send_message", dict(role="crane", text="Less 3A in the open heats."), 2), Reply(content="Told the crane."),
                   propose, Reply(content="Proposed.")])
    rec = loop.handle(world, dict(kind="issue", text="Only 590 t of 3A left today.", role="shift_leader", ts=world._ts()), llm)
    assert rec["path"] == "llm" and rec["decision"] == "replan" and world.current_plan().version == rec["version"]
    assert any(m["role"] == "user" and "did not propose" in m["content"] for m in llm.calls[4])


# Checks an ignored reminder leads to the fallback publishing a plan.
# The LLM only messages the crane and never proposes.
# The path must be 'fallback after LLM' with a replan decision.
def test_message_without_a_proposal_falls_back_to_a_plan(world):
    """If the reminder is ignored, the fallback still publishes a plan: a crane note alone must not end the run."""
    cap = {"group_caps_t": {"SCRAP_3A": 590.0}}
    llm = FakeLLM([call("run_optimizer", cap), call("run_optimizer", {**cap, "relax_recipe_bounds": True}, 1),
                   call("send_message", dict(role="crane", text="Less 3A in the open heats."), 2), Reply(content="Told the crane."),
                   Reply(content="Still just the message.")])
    rec = loop.handle(world, dict(kind="issue", text="Only 590 t of 3A left today.", role="shift_leader", ts=world._ts()), llm)
    assert rec["path"] == "fallback after LLM" and rec["decision"] == "replan" and world.current_plan().version == rec["version"]


# Helper: asserts a plan version was properly dropped.
# Checks status, audit, supervisor message before the drop event, and crane view.
# Used by the drop tests.
def _assert_dropped(world, version):
    assert world.current_plan() is None and world.base_plan() is None
    assert world.store.get(world.plan_id, version).status == "DROPPED"
    sup = [m["text"] for m in world.messages["supervisor"]]
    assert any(f"Plan v{version} is dropped" in t for t in sup)
    acts = [a["action"] for a in world.store.audit(world.plan_id)]
    assert acts[-1] == "DROPPED"
    kinds = [e["kind"] for e in world.store.events_since(0)]   # the supervisor is informed before the plan is withdrawn
    assert kinds.index("message") < kinds.index("plan_dropped")
    assert world.phone("crane")["plan"]["status"] == "NO RELEASED PLAN" and "dropped" in world.pending_banner["text"]


# Checks the fallback drops a plan when nothing is feasible.
# A later email must not resurrect the dropped plan.
# The next 06:00 plan starts cleanly.
def test_fallback_drops_an_infeasible_plan_after_informing_the_supervisor(world):
    world.approve(); world.release()
    v = world.current_plan().version
    rec = loop.handle(world, dict(kind="issue", text="only 200 t of 3A left", role="shift_leader", ts=world._ts()), world.llm)
    assert rec["path"] == "fallback" and rec["decision"] == "dropped"
    assert all(s["result"]["status"] == "infeasible" for s in rec["steps"] if s["name"] == "run_optimizer")
    _assert_dropped(world, v)
    # nothing left to re-plan: a later email does not resurrect the dropped plan
    rec = loop.handle(world, dict(kind="inbox", text="Only 480 t of 3A today.", role="logistics", ts=world._ts()), world.llm)
    assert rec["decision"] == "infeasible" and world.current_plan() is None
    assert world.make_plan() is not None and world.current_plan().status == "EMAILED" and world.pending_banner is None


# Checks a scripted LLM can drop an infeasible plan.
# Both optimiser runs are infeasible, then drop_plan is called.
# Verifies the drop via _assert_dropped.
def test_llm_drops_an_infeasible_plan(world):
    cap = {"group_caps_t": {"SCRAP_3A": 200.0}}
    v = world.current_plan().version
    llm = FakeLLM([call("run_optimizer", cap), call("run_optimizer", {**cap, "relax_recipe_bounds": True}, 1),
                   call("drop_plan", dict(reason="Only 200 t of 3A: no feasible mix, even with relaxed recipe bounds."), 2),
                   Reply(content="Dropped.")])
    rec = loop.handle(world, dict(kind="issue", text="only 200 t of 3A left", role="shift_leader", ts=world._ts()), llm)
    assert rec["path"] == "llm" and rec["decision"] == "dropped"
    _assert_dropped(world, v)


# Checks an infeasible run that never drops gets one reminder.
# If the LLM still does not drop, the fallback drops the plan.
# Verifies the reminder text and the final dropped state.
def test_infeasible_without_a_drop_gets_one_reminder_then_falls_back(world):
    cap = {"group_caps_t": {"SCRAP_3A": 200.0}}
    llm = FakeLLM([call("run_optimizer", cap), call("send_message", dict(role="supervisor", text="Nothing works with 200 t of 3A."), 1),
                   Reply(content="Told the supervisor."), Reply(content="Still not dropping.")])
    rec = loop.handle(world, dict(kind="issue", text="only 200 t of 3A left", role="shift_leader", ts=world._ts()), llm)
    assert any(m["role"] == "user" and "neither proposed nor dropped" in m["content"] for m in llm.calls[-1])
    assert rec["path"] == "fallback after LLM" and rec["decision"] == "dropped" and world.current_plan() is None


# Checks drop_plan is refused without an infeasible optimiser run.
# Tries before any run and after a feasible run.
# The live plan must remain.
def test_drop_plan_needs_an_infeasible_run(world):
    t = AgentTools(world, dict(kind="issue", text="x", role="shift_leader"))
    assert "infeasible run_optimizer" in t.call("drop_plan", dict(reason="No mix."))["error"]
    t.call("run_optimizer", {})   # feasible
    assert "infeasible run_optimizer" in t.call("drop_plan", dict(reason="No mix."))["error"]
    assert world.current_plan() is not None


# Checks an LLM connection error falls back to the policy.
# The fake LLM raises on every chat call.
# The path reason must mention the error.
def test_llm_error_falls_back(world):
    class Broken(FakeLLM):
        # Fake chat that always fails.
        # Simulates an unreachable LLM endpoint.
        # Raises ConnectionError.
        def chat(self, *a, **k):
            raise ConnectionError("endpoint down")
    rec = loop.handle(world, dict(kind="issue", text="No 3ACh available.", role="shift_leader", ts=world._ts()), Broken([]))
    assert rec["path"] == "fallback after LLM" and "endpoint down" in rec["path_reason"]


# Checks an LLM that ignores an issue triggers the fallback.
# The fake LLM just replies 'Looks fine.'.
# The fallback must produce a visible decision.
def test_llm_that_never_acts_on_an_issue_falls_back(world):
    rec = loop.handle(world, dict(kind="issue", text="No 3ACh available.", role="shift_leader", ts=world._ts()),
                      FakeLLM([Reply(content="Looks fine.")]))
    assert rec["path"] == "fallback after LLM" and rec["decision"] in ("replan", "message", "infeasible", "dropped")


# Checks the tool-call budget limits a runaway LLM.
# The fake LLM keeps calling get_state.
# At most MAX_TOOL_CALLS + 1 tool steps run and the decision is no_change.
def test_tool_budget_is_enforced(world):
    llm = FakeLLM([call("get_state", {}, i) for i in range(20)])
    rec = loop.handle(world, dict(kind="drift", text="drift", role=None, ts=world._ts()), llm)
    assert len([s for s in rec["steps"] if s.get("type") == "tool"]) <= loop.MAX_TOOL_CALLS + 1
    assert rec["decision"] == "no_change"


# Checks a chat answer with invented numbers is not sent.
# Both LLM attempts quote an unsupported cost.
# The fallback answers instead and the reason names the bad number.
def test_question_answer_must_use_tool_numbers(world):
    rec = loop.handle(world, dict(kind="question", text="What is the plan cost?", role="supervisor", ts=world._ts()),
                      FakeLLM([Reply(content="It costs 12345.67 $/t."), Reply(content="Still 12345.67 $/t.")]))
    assert rec["path"] == "fallback after LLM" and world.chat[-1]["to"] == "supervisor"
    assert "12345.67" in rec["path_reason"]   # the reason names the rejected number


# Checks a rejected chat answer gets one rewrite.
# The rewrite prompt names the bad number; the second answer uses a real tool value.
# The LLM path must succeed with an answer.
def test_question_answer_gets_one_rewrite_naming_the_bad_number(world):
    # Scripted rewrite that quotes the plan cost from get_state.
    # Reads vs_history.plan_cost_per_t from the initial state message.
    # Returns a Reply with the corrected answer.
    def fixed(messages):
        cost = json.loads(messages[1]["content"].split("(get_state):\n", 1)[1])["vs_history"]["plan_cost_per_t"]
        return Reply(content=f"The plan costs {cost} $/t.")
    llm = FakeLLM([Reply(content="It costs 12345.67 $/t."), fixed])
    rec = loop.handle(world, dict(kind="question", text="What is the plan cost?", role="supervisor", ts=world._ts()), llm)
    assert rec["path"] == "llm" and rec["decision"] == "answer"
    assert "12345.67" in llm.calls[1][-1]["content"] and world.chat[-1]["text"].startswith("The plan costs")


# Checks the template chat answer combines cost and risk facts.
# The supervisor gets $ figures and the Cu P95.
# The crane never gets a $ figure.
def test_fallback_answer_combines_why_cost_and_risk(world):
    from meltplan.intelligence.agent import chat
    a = chat.answer(world, "supervisor", "Why is copper binding, and what does the plan cost compared with history?")
    assert "No change recorded" not in a and "$" in a and "Cu P95" in a
    assert "$" not in chat.answer(world, "crane", "What does it cost?")


# Checks the supervisor can approve an agent alternative.
# Approving publishes it as a new version and approves that.
# Skips if the scenario produced no distinct alternative.
def test_approve_an_alternative(world):
    world.dispatch("issue", "No 3ACh available.", "shift_leader")
    p = world.current_plan()
    alts = world.alternatives.get((p.plan_id, p.version), [])
    if not alts:
        pytest.skip("no distinct alternative for this scenario")
    world.approve(option=alts[0]["candidate_id"])
    assert world.current_plan().status == "SUPERVISOR_APPROVED" and world.current_plan().version == p.version + 1


# Checks identical-cost alternatives are not offered.
# Publishes one candidate with an identical one as its alternative.
# The stored alternatives list must be empty.
def test_same_cost_alternatives_are_dropped(world):
    """The LLM offered identical candidates (same mix, other settings) as 'options'; only distinct costs are kept."""
    cap = {"group_caps_t": {"SCRAP_3A": 590.0}, "relax_recipe_bounds": True}
    a, b = world.solve_candidate(cap)[1]["candidate_id"], world.solve_candidate(cap)[1]["candidate_id"]
    rat = dict(email="x", supervisor="x", shift_leader="x", crane="x")
    v = world.publish_candidate(a, rat, "3A capped.", [b])
    assert world.alternatives[(v.plan_id, v.version)] == []


# Checks a looser-spec candidate cannot be the main proposal.
# Refused while a candidate under the live spec is feasible.
# It can still be offered as an alternative.
def test_looser_spec_cannot_be_the_proposal_while_the_live_spec_is_feasible(world):
    """Live run: the LLM proposed the cheaper plant_tight candidate although a plant_strict one (relaxed floors) was feasible."""
    t = AgentTools(world, dict(kind="issue", text="3ACh pile is empty", role="crane"))
    strict = t.call("run_optimizer", {"exclude_groups": ["SCRAP_25A_3ACH"], "relax_recipe_bounds": True})
    loose = t.call("run_optimizer", {"exclude_groups": ["SCRAP_25A_3ACH"], "spec_class": "plant_tight"})
    assert strict["status"] == loose["status"] == "optimal"
    rat = dict(email="ok", supervisor="ok", shift_leader="ok", crane="ok")
    err = t.call("propose_plan", dict(candidate_id=loose["candidate_id"], rationale=rat))["error"]
    assert strict["candidate_id"] in err
    ok = t.call("propose_plan", dict(candidate_id=strict["candidate_id"], rationale=rat, alternatives=[loose["candidate_id"]]))
    assert ok["version"] and ok["alternatives_offered"] == [loose["candidate_id"]]


# Checks approving a looser-spec alternative adopts its limits.
# The world's spec class and Cu limit must follow the chosen option.
# Evaluation uses those limits and slack constraints are not listed.
def test_approving_a_looser_spec_alternative_moves_the_limits_with_it(world):
    t = AgentTools(world, dict(kind="issue", text="3ACh pile is empty", role="crane"))
    strict = t.call("run_optimizer", {"exclude_groups": ["SCRAP_25A_3ACH"], "relax_recipe_bounds": True})
    loose = t.call("run_optimizer", {"exclude_groups": ["SCRAP_25A_3ACH"], "spec_class": "plant_tight"})
    rat = dict(email="ok", supervisor="ok", shift_leader="ok", crane="ok")
    t.call("propose_plan", dict(candidate_id=strict["candidate_id"], rationale=rat, alternatives=[loose["candidate_id"]]))
    world.approve(option=loose["candidate_id"])
    assert world.spec_class == "plant_tight" and world.ctx.limits["cu"] == 0.25
    h = world.current_plan().heats[-1]
    assert world.evaluate(h.slot, h.group_t)["violations"] == []   # judged against the spec it was planned for
    assert all("$0.00" not in b.plain for b in world.current_plan().binding)   # slack constraints are not listed


# Checks an over-risk live plan after a lab result is not left alone.
# The LLM gets one 'above alpha' reminder, then the fallback explores.
# The final decision must be a re-plan.
def test_unsafe_live_plan_after_a_lab_result_is_not_left_alone(world):
    """Live run: Cu risk 23% after a lab result, default re-run infeasible, the LLM stopped with 'no change'.
    It gets one reminder to explore; if it still does nothing, the fallback explores and proposes."""
    world.kf["cu"].e = world.kf["cu"].e * 1.12   # copper-heavier estimates: the live plan now breaches the risk level
    assert max(world.open_eval()["worst_p_exceed"].values()) > world.alpha + 0.01
    llm = FakeLLM([call("run_optimizer", {}), Reply(content="Infeasible, so no proposal."), Reply(content="Still no proposal.")])
    rec = loop.handle(world, dict(kind="lab_surprise", text="Cu lab result above the limit.", role=None, ts=world._ts()), llm)
    assert any(m["role"] == "user" and "above alpha" in m["content"] for m in llm.calls[-1])
    assert rec["path"] == "fallback after LLM" and rec["decision"] == "replan"


# Checks the LLM client is rebuilt once after a stale connection.
# The first call fails, the rebuilt client succeeds.
# Two failures in a row are raised to the caller.
def test_llm_rebuilds_a_stale_client_once_on_connection_error(monkeypatch):
    import httpx
    import openai
    from types import SimpleNamespace
    from meltplan.intelligence.agent.llm import LLM

    built, calls = [], []

    class FakeClient:
        # Fake OpenAI client constructor.
        # Records each construction so the test can count rebuilds.
        # Wires chat.completions.create to this object's create.
        def __init__(self, **kw):
            built.append(kw)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        # Fake completion call.
        # Raises a connection error on the first call only.
        # Afterwards returns a minimal 'ok' response.
        def create(self, **kw):
            calls.append(kw)
            if len(calls) == 1:     # the first pooled connection is stale
                raise openai.APIConnectionError(request=httpx.Request("POST", "http://llm.test"))
            msg = SimpleNamespace(content="ok", tool_calls=None, model_extra={})
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    llm = LLM()
    assert llm.chat([{"role": "user", "content": "hi"}]).content == "ok"
    assert len(built) == 2 and len(calls) == 2

    calls.clear()
    monkeypatch.setattr(FakeClient, "create", lambda self, **kw: (_ for _ in ()).throw(
        openai.APIConnectionError(request=httpx.Request("POST", "http://llm.test"))))
    with pytest.raises(openai.APIConnectionError):   # a second failure in a row is raised for the caller's fallback
        llm.chat([{"role": "user", "content": "hi"}])


# Helper: approves and releases the world's current plan.
# Gets the plan onto the crane's list.
# Returns the first heat of the plan.
def _released(w):
    w.approve(); w.release()
    return w.current_plan().heats[0]


# Checks repeated grabs are idempotent and basket_complete fills the rest.
# Unconfirmed lines are recorded at planned tonnes; a repeat completion is a no-op.
# The crane view shows the completed basket and grabbed lines.
def test_grabs_are_idempotent_and_basket_complete_fills_the_rest(world):
    h = _released(world)
    b1 = h.baskets[0]
    first = b1.items[0]
    e = world.grab_done(1, 1, first.mat_code, first.tonnes)
    assert world.grab_done(1, 1, first.mat_code, 99.0) == e and len(world.handling.log) == 1   # same line twice: one event
    r = world.basket_complete(1, 1)
    assert r["auto_grabbed"] == len(b1.items) - 1 and len(world.handling.log) == len(b1.items)
    assert {(x.mat_code, x.est_tonnes) for x in world.handling.log[1:]} == {(it.mat_code, it.tonnes) for it in b1.items[1:]}
    seq = world.store.last_seq()
    assert world.basket_complete(1, 1)["auto_grabbed"] == 0 and len(world.handling.log) == len(b1.items)
    assert world.store.last_seq() == seq                       # a repeat is a no-op: no events
    crane = world.phone("crane")["plan"]["loading"][0]
    assert crane["baskets_complete"] == [1] and set(crane["grabbed"]) == {f"1-{it.mat_code}" for it in b1.items}


# Checks completing every basket marks the heat loaded and frozen.
# Later grabs never downgrade it and a re-plan keeps its mix.
# After tapping, basket completion is refused.
def test_all_baskets_complete_loads_the_heat_and_freezes_it(world):
    h = _released(world)
    for b in h.baskets:
        assert world.basket_complete(1, b.basket_no)["auto_grabbed"] == len(b.items)
    assert world.progress[1] == "loaded"
    assert [e["kind"] for e in world.store.events_since(0)].count("heat_loaded") == 1
    it = h.baskets[0].items[0]
    world.grab_done(1, 2 if len(h.baskets) > 1 else 1, it.mat_code, 1.0)   # a later grab never downgrades
    assert world.progress[1] == "loaded"
    mix = dict(h.group_t)
    rec = loop.handle(world, dict(kind="inbox", text=supplier_email(world), role="logistics", ts=world._ts()), world.llm)
    assert rec["decision"] == "replan" and world.current_plan().heats[0].group_t == mix   # loaded = frozen like loading
    world.step_heat()
    assert world.progress[1] == "tapped"
    with pytest.raises(ValueError, match="already tapped"):
        world.basket_complete(1, 1)


# Checks basket_complete requires a released plan and a valid basket.
# Fails before release, and for a basket not on the released list.
# Guards crane input validation.
def test_basket_complete_needs_a_released_plan(world):
    with pytest.raises(ValueError, match="no released plan"):
        world.basket_complete(1, 1)
    _released(world)
    with pytest.raises(ValueError, match="not on the released list"):
        world.basket_complete(1, 99)


# ---- Mix Editor hand edits
# Helper: moves tonnes from one group to another in a heat's mix.
# Copies the heat's group tonnages first.
# Returns the new mix dict.
def _shift(h, src, dst, t):
    m = dict(h.group_t)
    m[src] = m.get(src, 0.0) - t
    m[dst] = m.get(dst, 0.0) + t
    return m


# Helper: finds the cleanest and dirtiest groups by Cu content.
# Uses the world's current estimates.
# Returns (lowest-Cu group, highest-Cu group).
def _groups_by_cu(world):
    from meltplan.data.store import GROUPS
    world._set_estimates()
    mu = world.ctx.est.mu["cu"]
    return GROUPS[int(mu.argmin())], GROUPS[int(mu.argmax())]


# Checks a hand edit re-predicts later heats without changing their mix.
# Shifting to dirtier scrap must raise later heats' Cu risk.
# Shadow prices from the edited heat on are dropped.
def test_edit_re_predicts_later_heats_and_drops_stale_shadow_prices(world):
    p = world.current_plan()
    clean, dirty = _groups_by_cu(world)
    t = min(15.0, p.heats[1].group_t.get(clean, 0.0))
    assert t > 1
    ev = world.evaluate(2, _shift(p.heats[1], clean, dirty, t))
    assert [d["slot"] for d in ev["downstream"]] == list(range(3, len(p.heats) + 1))
    new = world.edit(2, _shift(p.heats[1], clean, dirty, t))
    a, b = p.heats, new.heats
    assert b[0] == a[0]                                                   # earlier heat untouched
    assert b[2].group_t == a[2].group_t and b[2].baskets == a[2].baskets   # later heat: same mix, re-predicted only
    assert b[2].predicted["cu"].mean > a[2].predicted["cu"].mean and b[2].predicted["cu"].p_exceed > a[2].predicted["cu"].p_exceed
    assert ev["downstream"][0]["p_exceed"] == pytest.approx(max(x.p_exceed for x in b[2].predicted.values()))
    assert all(int(c.name.rpartition("@")[2]) < 2 for c in new.binding) and new.hand_edited == [2]
    assert world.phone("supervisor")["plan"]["hand_edited"] == [2]


# Checks tapped, loading, loaded and frozen heats cannot be edited.
# Each is refused for edit but can still be evaluated read-only.
# An open heat is not locked.
def test_tapped_loading_loaded_and_frozen_heats_are_locked(world):
    world.approve(); world.release()
    world.step_heat()                                          # heat 1 tapped
    p = world.current_plan()
    world.grab_done(2, 1, p.heats[1].baskets[0].items[0].mat_code, 1.0)   # heat 2 loading
    for b in p.heats[2].baskets:                               # heat 3 loaded
        world.basket_complete(3, b.basket_no)
    p.heats[3].frozen = True                                   # heat 4 frozen by the user
    world.store.db.execute("UPDATE plans SET body=? WHERE plan_id=? AND version=?", (p.model_dump_json(), p.plan_id, p.version))
    for slot, st in [(1, "tapped"), (2, "loading"), (3, "loaded"), (4, "frozen")]:
        with pytest.raises(ValueError, match=f"heat {slot} is {st}"):
            world.edit(slot, dict(p.heats[slot - 1].group_t))
        ev = world.evaluate(slot, dict(p.heats[slot - 1].group_t))   # still evaluated, read-only
        assert ev["locked"] and ev["predicted"]["cu"]["mean"] > 0
    assert world.evaluate(5, dict(p.heats[4].group_t))["locked"] is None


# Checks recipe range breaches only warn.
# Removing all 3A is flagged as a rule violation, not an availability one.
# The edit is still saved.
def test_recipe_bounds_warn_but_the_edit_saves(world):
    p = world.current_plan()
    h = p.heats[1]
    m = _shift(h, "SCRAP_3A", "SCRAP_25A_3ACH", h.group_t["SCRAP_3A"])      # no bulk scrap at all: far below its floor
    ev = world.evaluate(2, m)
    assert any("SCRAP_3A" in r and "recipe range" in r for r in ev["rule_violations"]) and not ev["availability_violations"]
    assert world.edit(2, m).heats[1].group_t.get("SCRAP_3A", 0.0) == 0.0


# Checks an edit exceeding a reported shortage cap is refused.
# After the supplier email caps 3A, adding 3A to a heat is flagged.
# Saving that edit raises.
def test_an_edit_over_a_reported_shortage_cap_is_refused(world):
    loop.handle(world, dict(kind="inbox", text=supplier_email(world), role="logistics", ts=world._ts()), world.llm)
    assert world.caps.get("SCRAP_3A", 0) > 0
    p = world.current_plan()
    h = p.heats[-1]
    src = max((g for g in h.group_t if g != "SCRAP_3A"), key=h.group_t.get)
    m = _shift(h, src, "SCRAP_3A", min(20.0, h.group_t[src]))
    ev = world.evaluate(h.slot, m)
    assert any(v.startswith("SCRAP_3A") and "reported shortage" in v for v in ev["availability_violations"])
    with pytest.raises(ValueError, match="reported shortage"):
        world.edit(h.slot, m)


# Checks the recipe check uses the relaxed bounds of a published re-plan.
# Untouched heats must show no recipe warnings afterwards.
# Regression for a live-run bug.
def test_recipe_check_follows_a_relaxed_re_plan(world):
    """Live run: after a re-plan that dropped the 3A floor, every untouched heat showed a recipe-range warning."""
    cid = world.solve_candidate({"group_caps_t": {"SCRAP_3A": 590.0}, "relax_recipe_bounds": True})[1]["candidate_id"]
    world.publish_candidate(cid, dict(email="x", supervisor="x", shift_leader="x", crane="x"), "3A capped.")
    p = world.current_plan()
    assert all(world.evaluate(h.slot, dict(h.group_t))["rule_violations"] == [] for h in p.heats)


# Checks the heat after a tapped one uses the measured heel.
# Compares evaluation against predictions from measured vs planned heel.
# They must differ, and evaluation must match the measured one.
def test_heel_for_the_heat_after_a_tapped_one_is_the_measured_result(world):
    from meltplan.data.store import GROUPS
    from meltplan.intelligence.optimizer.model import predict_heat
    world.step_heat()
    assert world.progress[1] == "tapped"
    p = world.current_plan()
    m = dict(p.heats[1].group_t)
    ev = world.evaluate(2, m)
    x = np.array([m.get(g, 0.0) for g in GROUPS])
    want = predict_heat(world.ctx, x, None, world.heel)["cu"]["mean"]
    planned = predict_heat(world.ctx, x, None, {el: p.heats[0].predicted[el].mean for el in world.heel})["cu"]["mean"]
    assert ev["predicted"]["cu"]["mean"] == pytest.approx(want) and abs(want - planned) > 1e-6


# Checks the crane holds after a newer version is rejected.
# crane_hold must mention the rejection.
# Basket completion is refused while held.
def test_crane_holds_after_a_rejection_until_a_plan_is_approved(world):
    world.approve(); world.release()
    world.edit(world.current_plan().heats[-1].slot, dict(world.current_plan().heats[-1].group_t))   # new version awaiting approval
    v = world.current_plan()
    world.store.transition(v, "REJECTED", "supervisor", world._ts(), "no")
    assert "rejected" in world.crane_hold()
    with pytest.raises(ValueError, match="rejected"):
        world.basket_complete(1, 1)


# Checks the crane sees pending lists for changed heats but cannot load them.
# Loading is blocked until approval, and changed heats until release.
# After release the new list is the crane's list.
def test_crane_sees_the_pending_list_for_changed_heats_but_cannot_load_it(world):
    world.approve(); world.release()
    rec = loop.handle(world, dict(kind="issue", text="3ACh pile is empty", role="crane", ts=world._ts()), world.llm)
    assert rec["decision"] == "replan"
    pend, changed = world.pending_change()
    assert changed and pend.version == rec["version"]
    heats = {h["slot"]: h for h in world.phone("crane")["plan"]["loading"]}
    for s in changed:
        names = [i["mat_name"] for b in heats[s]["baskets"] for i in b["items"]]
        assert "3ACh" not in names and heats[s]["pending"]["version"] == pend.version
    s = min(changed)
    code = heats[s]["baskets"][0]["items"][0]["mat_code"]
    # until the supervisor approves, the crane does not move at all
    assert "waiting for supervisor approval" in world.phone("crane")["plan"]["hold"]
    with pytest.raises(ValueError, match="waiting for supervisor approval"):
        world.grab_done(s, 1, code, 1.0)
    with pytest.raises(ValueError, match="waiting for supervisor approval"):
        world.basket_complete(s, 1)
    world.approve()   # approved, not yet released: the hold lifts, but a changed heat still waits for its new list
    assert world.crane_hold() is None
    with pytest.raises(ValueError, match="not released yet"):
        world.grab_done(s, 1, code, 1.0)
    world.release()   # once released, the new list is the crane's list
    assert world.pending_change() == (None, set())
    assert "pending" not in world.phone("crane")["plan"]["loading"][s - 1]
    world.grab_done(s, 1, code, 1.0)


# Checks views show only untapped heats once some are tapped.
# Covers supervisor, shift leader, email and crane views.
# Tapped slots are listed separately.
def test_views_show_only_the_heats_left_once_some_are_tapped(world):
    world.approve(); world.release()
    world.step_heat(); world.step_heat()                       # heats 1 and 2 tapped
    n = len(world.current_plan().heats)
    for role in ("supervisor", "shift_leader", "email"):
        v = world.phone(role)["plan"]
        assert [h["slot"] for h in v["heats"]] == list(range(3, n + 1)) and v["tapped"] == [1, 2]
    sup = world.phone("supervisor")["plan"]
    assert sup["summary"]["n_heats"] == n - 2
    assert [h["slot"] for h in world.phone("crane")["plan"]["loading"]] == list(range(3, n + 1))


# Checks act-layer logs show each role only its own actions and concerns.
# Approvals, releases, grabs and emails land in the right role logs.
# A supervisor-only note appears only in the supervisor log.
def test_act_layer_logs_are_split_by_role(world):
    from meltplan.sim.view import act_log
    world.approve(); world.release()
    p = world.current_plan()
    world.grab_done(1, 1, p.heats[0].baskets[0].items[0].mat_code, 1.0)
    world.notify("supervisor", "note for the supervisor only")
    logs = {r: act_log(world, r) for r in ("email", "supervisor", "shift_leader", "crane")}
    acts = {r: [x["action"] for x in rows] for r, rows in logs.items()}
    assert "approved" in acts["supervisor"] and "released" not in acts["supervisor"]
    assert "released" in acts["shift_leader"] and "grab" not in acts["shift_leader"]
    assert "grab" in acts["crane"] and "approved" not in acts["crane"]
    assert set(acts["email"]) == {"emailed", "email sent"}
    assert any(x["detail"] == "note for the supervisor only" for x in logs["supervisor"])
    assert not any(x["detail"] == "note for the supervisor only" for r in ("email", "shift_leader", "crane") for x in logs[r])
