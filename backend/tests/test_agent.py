"""Agent: tools and guards, the deterministic fallback, and the LLM loop driven by a scripted fake LLM."""
import json

import pytest

from scrapchef.intelligence.agent import loop
from scrapchef.intelligence.agent.llm import Reply, ToolCall
from scrapchef.intelligence.agent.policy import parse_issue
from scrapchef.intelligence.agent.tools import AgentTools, openai_tools
from scrapchef.sim.scenario import supplier_email
from scrapchef.sim.world import World


class FakeLLM:
    """Replays scripted replies; each item is a Reply or a callable(messages) -> Reply (to read earlier tool results)."""
    available = True

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def describe(self):
        return "fake"

    def chat(self, messages, tools=None, timeout=30, **kw):
        self.calls.append(messages)
        item = self.script.pop(0) if self.script else Reply(content="done")
        return item(messages) if callable(item) else item


def call(name, args, i=0):
    return Reply(tool_calls=[ToolCall(f"c{i}", name, json.dumps(args))])


def last_result(messages):
    return json.loads([m for m in messages if m["role"] == "tool"][-1]["content"])


@pytest.fixture()
def world():
    w = World("SHADOW_REPLAY")
    w.make_plan()
    return w


def test_tool_schemas_are_valid_function_specs():
    names = {t["function"]["name"] for t in openai_tools()}
    assert names == {"get_state", "run_optimizer", "evaluate_mix", "compare_plans", "propose_plan", "send_message"}
    assert all(t["function"]["parameters"]["type"] == "object" for t in openai_tools())


def test_parse_issue():
    assert parse_issue("Only 590 t of 3A left today")[0] == {"group_caps_t": {"SCRAP_3A": 590.0}}
    assert parse_issue("No 3ACh available")[0] == {"exclude_groups": ["SCRAP_25A_3ACH"]}
    assert parse_issue("crane 2 is slow")[0] == {}


def test_guards_reject_bad_arguments_invented_numbers_and_crane_costs(world):
    t = AgentTools(world, dict(kind="issue", text="x", role="shift_leader"))
    assert "invalid arguments" in t.call("run_optimizer", '{"exclude_groups": ["GOLD"]}')["error"]
    cand = t.call("run_optimizer", {})
    rat = dict(email="ok", supervisor="Saves 999.99 $/t", shift_leader="ok", crane="ok")
    assert "numbers" in t.call("propose_plan", dict(candidate_id=cand["candidate_id"], rationale=rat))["error"]
    assert "crane" in t.call("send_message", dict(role="crane", text="this costs more"))["error"]


def test_anti_churn_blocks_a_pointless_replan(world):
    t = AgentTools(world, dict(kind="lab_surprise", text="CU +3.1σ.", role=None))
    cand = t.call("run_optimizer", {})  # same estimates -> same plan -> ~0 $/t change
    rat = dict(email="ok", supervisor="ok", shift_leader="ok", crane="ok")
    assert "Anti-churn" in t.call("propose_plan", dict(candidate_id=cand["candidate_id"], rationale=rat))["error"]


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


def test_crane_hold_message_expires_on_release(world):
    loop.handle(world, dict(kind="inbox", text=supplier_email(world), role="logistics", ts=world._ts()), world.llm)
    assert any("hold" in m["text"].lower() for m in world.phone("crane")["messages"])
    world.approve()
    world.release()
    assert world.phone("crane")["messages"] == []
    assert world.messages["crane"][0]["expired"].endswith("released")   # kept in the log


def test_llm_loop_explores_proposes_and_offers_an_alternative(world):
    cap = {"group_caps_t": {"SCRAP_3A": 590.0}}

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


def test_llm_error_falls_back(world):
    class Broken(FakeLLM):
        def chat(self, *a, **k):
            raise ConnectionError("endpoint down")
    rec = loop.handle(world, dict(kind="issue", text="No 3ACh available.", role="shift_leader", ts=world._ts()), Broken([]))
    assert rec["path"] == "fallback after LLM" and "endpoint down" in rec["path_reason"]


def test_llm_that_never_acts_on_an_issue_falls_back(world):
    rec = loop.handle(world, dict(kind="issue", text="No 3ACh available.", role="shift_leader", ts=world._ts()),
                      FakeLLM([Reply(content="Looks fine.")]))
    assert rec["path"] == "fallback after LLM" and rec["decision"] in ("replan", "message", "infeasible")


def test_tool_budget_is_enforced(world):
    llm = FakeLLM([call("get_state", {}, i) for i in range(20)])
    rec = loop.handle(world, dict(kind="drift", text="drift", role=None, ts=world._ts()), llm)
    assert len([s for s in rec["steps"] if s.get("type") == "tool"]) <= loop.MAX_TOOL_CALLS + 1
    assert rec["decision"] == "no_change"


def test_question_answer_must_use_tool_numbers(world):
    rec = loop.handle(world, dict(kind="question", text="What is the plan cost?", role="supervisor", ts=world._ts()),
                      FakeLLM([Reply(content="It costs 12345.67 $/t.")]))
    assert rec["path"] == "fallback after LLM" and world.chat[-1]["to"] == "supervisor"


def test_approve_an_alternative(world):
    world.dispatch("issue", "No 3ACh available.", "shift_leader")
    p = world.current_plan()
    alts = world.alternatives.get((p.plan_id, p.version), [])
    if not alts:
        pytest.skip("no distinct alternative for this scenario")
    world.approve(option=alts[0]["candidate_id"])
    assert world.current_plan().status == "SUPERVISOR_APPROVED" and world.current_plan().version == p.version + 1
