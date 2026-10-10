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
    err = t.call("propose_plan", dict(candidate_id=cand["candidate_id"], rationale=rat))["error"]
    assert "numbers" in err and "999.99" in err   # the offending number is named so the model can fix it
    assert "crane" in t.call("send_message", dict(role="crane", text="this costs more"))["error"]
    # the shift leader may not see costs either, also without a cost word (live run: "runs at 496.54 per tonne")
    per_t = str(cand["cost_per_t"])
    assert "cost figures" in t.call("send_message", dict(role="shift_leader", text=f"The plan runs at {per_t} overall."))["error"]
    assert t.call("send_message", dict(role="supervisor", text=f"The plan runs at {per_t} $/t."))["ok"]
    assert t.call("send_message", dict(role="shift_leader", text="Cost figures go to the supervisor, not to this role."))["ok"]


def test_anti_churn_blocks_a_pointless_replan(world):
    t = AgentTools(world, dict(kind="lab_surprise", text="Cu +3.1σ.", role=None))
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


def test_message_without_a_proposal_gets_one_reminder(world):
    """A run that finds a feasible candidate, messages the crane and stops is reminded once, then proposes."""
    cap = {"group_caps_t": {"SCRAP_3A": 590.0}}

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


def test_message_without_a_proposal_falls_back_to_a_plan(world):
    """If the reminder is ignored, the fallback still publishes a plan: a crane note alone must not end the run."""
    cap = {"group_caps_t": {"SCRAP_3A": 590.0}}
    llm = FakeLLM([call("run_optimizer", cap), call("run_optimizer", {**cap, "relax_recipe_bounds": True}, 1),
                   call("send_message", dict(role="crane", text="Less 3A in the open heats."), 2), Reply(content="Told the crane."),
                   Reply(content="Still just the message.")])
    rec = loop.handle(world, dict(kind="issue", text="Only 590 t of 3A left today.", role="shift_leader", ts=world._ts()), llm)
    assert rec["path"] == "fallback after LLM" and rec["decision"] == "replan" and world.current_plan().version == rec["version"]


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
                      FakeLLM([Reply(content="It costs 12345.67 $/t."), Reply(content="Still 12345.67 $/t.")]))
    assert rec["path"] == "fallback after LLM" and world.chat[-1]["to"] == "supervisor"
    assert "12345.67" in rec["path_reason"]   # the reason names the rejected number


def test_question_answer_gets_one_rewrite_naming_the_bad_number(world):
    def fixed(messages):
        cost = json.loads(messages[1]["content"].split("(get_state):\n", 1)[1])["vs_history"]["plan_cost_per_t"]
        return Reply(content=f"The plan costs {cost} $/t.")
    llm = FakeLLM([Reply(content="It costs 12345.67 $/t."), fixed])
    rec = loop.handle(world, dict(kind="question", text="What is the plan cost?", role="supervisor", ts=world._ts()), llm)
    assert rec["path"] == "llm" and rec["decision"] == "answer"
    assert "12345.67" in llm.calls[1][-1]["content"] and world.chat[-1]["text"].startswith("The plan costs")


def test_fallback_answer_combines_why_cost_and_risk(world):
    from scrapchef.intelligence.agent import chat
    a = chat.answer(world, "supervisor", "Why is copper binding, and what does the plan cost compared with history?")
    assert "No change recorded" not in a and "$" in a and "Cu P95" in a
    assert "$" not in chat.answer(world, "crane", "What does it cost?")


def test_approve_an_alternative(world):
    world.dispatch("issue", "No 3ACh available.", "shift_leader")
    p = world.current_plan()
    alts = world.alternatives.get((p.plan_id, p.version), [])
    if not alts:
        pytest.skip("no distinct alternative for this scenario")
    world.approve(option=alts[0]["candidate_id"])
    assert world.current_plan().status == "SUPERVISOR_APPROVED" and world.current_plan().version == p.version + 1


def test_same_cost_alternatives_are_dropped(world):
    """The LLM offered identical candidates (same mix, other settings) as 'options'; only distinct costs are kept."""
    cap = {"group_caps_t": {"SCRAP_3A": 590.0}, "relax_recipe_bounds": True}
    a, b = world.solve_candidate(cap)[1]["candidate_id"], world.solve_candidate(cap)[1]["candidate_id"]
    rat = dict(email="x", supervisor="x", shift_leader="x", crane="x")
    v = world.publish_candidate(a, rat, "3A capped.", [b])
    assert world.alternatives[(v.plan_id, v.version)] == []


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


def test_unsafe_live_plan_after_a_lab_result_is_not_left_alone(world):
    """Live run: Cu risk 23% after a lab result, default re-run infeasible, the LLM stopped with 'no change'.
    It gets one reminder to explore; if it still does nothing, the fallback explores and proposes."""
    world.kf["cu"].e = world.kf["cu"].e * 1.12   # copper-heavier estimates: the live plan now breaches the risk level
    assert max(world.open_eval()["worst_p_exceed"].values()) > world.alpha + 0.01
    llm = FakeLLM([call("run_optimizer", {}), Reply(content="Infeasible, so no proposal."), Reply(content="Still no proposal.")])
    rec = loop.handle(world, dict(kind="lab_surprise", text="Cu lab result above the limit.", role=None, ts=world._ts()), llm)
    assert any(m["role"] == "user" and "above alpha" in m["content"] for m in llm.calls[-1])
    assert rec["path"] == "fallback after LLM" and rec["decision"] == "replan"


def test_llm_rebuilds_a_stale_client_once_on_connection_error(monkeypatch):
    import httpx
    import openai
    from types import SimpleNamespace
    from scrapchef.intelligence.agent.llm import LLM

    built, calls = [], []

    class FakeClient:
        def __init__(self, **kw):
            built.append(kw)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

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


def _released(w):
    w.approve(); w.release()
    return w.current_plan().heats[0]


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


def test_basket_complete_needs_a_released_plan(world):
    with pytest.raises(ValueError, match="no released plan"):
        world.basket_complete(1, 1)
    _released(world)
    with pytest.raises(ValueError, match="not on the released list"):
        world.basket_complete(1, 99)
