"""FastAPI app: REST for actions, SSE for state-change events (both UI panes subscribe to one stream)."""
from __future__ import annotations

import asyncio
import json
import os
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from scrapchef.act.roles import ROLES
from scrapchef.data.store import GROUPS, ROOT
from scrapchef.sim.view import process_state
from scrapchef.sim.world import World

app = FastAPI(title="Mini-ScrapChef")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# agent runs in a worker so the UI stays responsive; the API is the single owner of data/app.sqlite (D36)
# SCRAPCHEF_APP_DB overrides it (tests import this module and set ":memory:")
world = World("SHADOW_REPLAY", agent_async=True, store_path=os.environ.get("SCRAPCHEF_APP_DB") or ROOT / "data" / "app.sqlite")


# Runs an endpoint body and converts unexpected errors into HTTP 400s.
# HTTPExceptions are re-raised unchanged; others are logged with a traceback.
# Keeps the server alive when an action fails.
def guard(fn):
    try:
        return fn()
    except HTTPException:
        raise
    except Exception as e:  # surface as 4xx with the message, keep server alive
        traceback.print_exc()
        raise HTTPException(400, f"{type(e).__name__}: {e}")


# GET /api/state: returns the full Process View state.
# Built from the shared world by process_state.
# Polled by the UI's process pane.
@app.get("/api/state")
def state():
    return process_state(world)


# GET /api/phone/{role}: returns the phone view for one role.
# Responds 404 for unknown roles.
# Only fields the role may see are included.
@app.get("/api/phone/{role}")
def phone(role: str):
    if role not in ROLES:
        raise HTTPException(404, "unknown role")
    return world.phone(role)


# Works out which event seq an SSE connection should start after.
# Resumes from a Last-Event-ID of the current epoch, else restarts at 0.
# A fresh connection starts at the latest event so history is not replayed.
def resume_seq(store, last_event_id: str | None, since: int | None) -> int:
    """Where an SSE connection starts. Ids are '<epoch>:<seq>'; an id from an older store (reset or server restart) restarts at 0.
    A fresh connection without an id or `since` starts at the latest event, so history is not replayed as unread."""
    if last_event_id:
        epoch, _, seq = last_event_id.partition(":")
        return int(seq) if epoch == store.epoch and seq.isdigit() else 0
    return since if since is not None else store.last_seq()


# GET /api/events: streams state-change events over SSE.
# Polls the store for new events and sends them with epoch-tagged ids.
# Handles store resets and sends keepalives between polls.
@app.get("/api/events")
async def events(request: Request, since: int | None = None):
    # Async generator producing the SSE stream text.
    # Restarts from seq 0 when a reset replaces the store.
    # Yields new events and a keepalive every 0.4 s.
    async def gen():
        store = world.store
        seq = resume_seq(store, request.headers.get("last-event-id"), since)
        while True:
            if world.store is not store:  # reset replaced the store and its seq restarted at 1
                store, seq = world.store, 0
            for ev in store.events_since(seq):
                seq = ev["seq"]
                yield f"id: {store.epoch}:{seq}\ndata: {json.dumps(ev)}\n\n"
            yield ": keepalive\n\n"
            await asyncio.sleep(0.4)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


class ResetBody(BaseModel):
    mode: str = "SHADOW_REPLAY"
    drift_factor: float = 1.5           # SIMULATED: a copper-heavy 3A (shredded) lot
    drift_group: str = "SCRAP_3A"


# POST /api/sim/reset: resets the simulation world.
# Applies the requested mode and drift scenario settings.
# Returns {ok: true} on success.
@app.post("/api/sim/reset")
def reset(b: ResetBody):
    return guard(lambda: (world.reset(b.mode, b.drift_factor, drift_group=b.drift_group), dict(ok=True))[1])


# POST /api/sim/step_heat: advances the simulation by one heat.
# Waits while the agent is working or a proposed change is pending approval.
# Reports done when all heats of the plan are tapped.
@app.post("/api/sim/step_heat")
def step_heat():
    if world.agent_pending:     # the sim clock waits for the agent: an LLM run (~30 s) would otherwise see its heats tapped
        return dict(waiting=True, note="Agent working: the next heat waits.")
    hold = world.sim_hold()     # ... and for the people: a proposed change is approved and released before its heats tap
    if hold:
        return dict(waiting=True, note=hold)
    if world.plan_finished():   # play stops at the end of the plan; "Plan 06:00" replays on to the next day's plan
        return dict(done=True, note="All heats of this plan are tapped. Press Plan 06:00 for the next plan.")
    return guard(lambda: world.step_heat() or dict(done=True))


# POST /api/sim/plan: advances to the next 06:00 and plans the next 24 h.
# Replays heats on the way when in shadow-replay mode.
# Returns whether a plan was created.
@app.post("/api/sim/plan")
def make_plan():
    """Advance the clock to the next 06:00 (replaying heats on the way) and plan the next 24 h."""
    # Moves the clock to the next 06:00 (or now, if no plan exists yet).
    # Steps through replay heats that start before that time.
    # Creates the new plan and reports success.
    def run():
        from datetime import timedelta
        nxt = world.clock.normalize() + timedelta(hours=6)
        if world.clock >= nxt:
            nxt += timedelta(days=1)
        if world.plan_id is None:
            nxt = world.clock
        elif world.mode == "SHADOW_REPLAY":
            while world.cursor < len(world.replay) and world.replay.start_ts.iloc[world.cursor] < nxt:
                world.step_heat()
        world.clock = nxt
        p = world.make_plan()
        return dict(ok=p is not None)
    return guard(run)


class EvalBody(BaseModel):
    slot: int
    group_t: dict[str, float]


# POST /api/plan/evaluate: evaluates a hand-edited mix for one heat.
# No solve is run; chemistry, cost and violations are recomputed.
# Returns the evaluation result.
@app.post("/api/plan/evaluate")
def evaluate(b: EvalBody):
    return guard(lambda: world.evaluate(b.slot, b.group_t))


# POST /api/plan/edit: saves a hand-edited mix as a new plan version.
# Delegates to world.edit for the given slot and group tonnes.
# Returns the new version number.
@app.post("/api/plan/edit")
def edit(b: EvalBody):
    return guard(lambda: dict(version=world.edit(b.slot, b.group_t).version))


class ApproveBody(BaseModel):
    option: str | None = None   # an agent alternative's candidate_id; None approves the live version


# POST /api/plan/approve: supervisor approves the live plan or an alternative.
# option selects an agent alternative's candidate id; None approves the live version.
# Returns {ok: true} on success.
@app.post("/api/plan/approve")
def approve(b: ApproveBody | None = None):
    return guard(lambda: (world.approve(option=b.option if b else None), dict(ok=True))[1])


class ReasonBody(BaseModel):
    reason: str = "supervisor rejection"


# POST /api/plan/reject: supervisor rejects the current plan.
# The reason is passed to the agent to build a different plan.
# Returns {ok: true} on success.
@app.post("/api/plan/reject")
def reject(b: ReasonBody):
    return guard(lambda: (world.reject(b.reason), dict(ok=True))[1])


# POST /api/plan/release: releases the approved plan to the shop floor.
# Delegates to world.release.
# Returns {ok: true} on success.
@app.post("/api/plan/release")
def release():
    return guard(lambda: (world.release(), dict(ok=True))[1])


class GrabBody(BaseModel):
    slot: int
    basket: int
    mat_code: int
    tonnes: float


# POST /api/crane/grab_done: records a crane grab confirmation.
# Logs the material and tonnes for the given slot and basket.
# Returns the recorded handling event.
@app.post("/api/crane/grab_done")
def grab(b: GrabBody):
    return guard(lambda: world.grab_done(b.slot, b.basket, b.mat_code, b.tonnes).model_dump())


class BasketBody(BaseModel):
    slot: int
    basket: int


# POST /api/crane/basket_complete: marks a basket as fully loaded.
# Delegates to world.basket_complete for the slot and basket.
# Returns the world's result.
@app.post("/api/crane/basket_complete")
def basket_complete(b: BasketBody):
    return guard(lambda: world.basket_complete(b.slot, b.basket))


class ShortBody(BaseModel):
    group: str


# POST /api/report/shortage: reports a shortage of a grade group.
# Triggers the agent to re-plan around the missing group.
# Returns the agent's decision, if any.
@app.post("/api/report/shortage")
def shortage(b: ShortBody):
    return guard(lambda: dict(decision=(world.report_shortage(b.group) or {}).get("decision")))


class IssueBody(BaseModel):
    role: str = "shift_leader"
    text: str


# POST /api/report/issue: reports a free-text operational issue.
# The agent interprets it asynchronously; results arrive as events.
# Returns the immediate result or {queued: true}.
@app.post("/api/report/issue")
def issue(b: IssueBody):
    """Free-text operational issue; the agent interprets it (async: the result arrives as events)."""
    return guard(lambda: world.report_issue(b.role, b.text) or dict(queued=True))


class InboxBody(BaseModel):
    text: str = ""


# POST /api/sim/inbox: delivers a mock supplier email to the agent.
# Uses the scripted demo email when no text is given.
# Notifies the supervisor and dispatches an inbox event.
@app.post("/api/sim/inbox")
def inbox(b: InboxBody):
    """Scenario: a MOCK supplier / logistics email arrives in the agent's inbox."""
    # Picks the email text and notifies the supervisor of it.
    # Dispatches the email to the agent as an inbox event.
    # Returns the dispatch result or {queued: true}.
    def run():
        text = b.text or default_supplier_email()
        world.notify("supervisor", f"Inbound email (logistics): {text}", sender="logistics")
        return world.dispatch("inbox", text, "logistics") or dict(queued=True)
    return guard(run)


# Builds the scripted demo supplier email.
# Caps 3A to part of what the open heats need.
# Delegates to sim.scenario.supplier_email.
def default_supplier_email() -> str:
    """Scripted demo email: the yard can supply only part of the 3A the open heats need (MOCK)."""
    from scrapchef.sim.scenario import supplier_email
    return supplier_email(world)


class ChatBody(BaseModel):
    role: str
    text: str


# POST /api/chat: sends a role's question to the agent.
# The answer arrives asynchronously in that role's chat thread.
# Returns the immediate result or {queued: true}.
@app.post("/api/chat")
def chat(b: ChatBody):
    """Questions go to the agent; the answer arrives in the role's chat thread (async)."""
    return guard(lambda: world.ask(b.role, b.text) or dict(queued=True))


# POST /api/email/read: marks every email as read.
# Updates the world's in-memory email list.
# Returns {ok: true}.
@app.post("/api/email/read")
def email_read():
    for m in world.emails:
        m["read"] = True
    return dict(ok=True)


# GET /api/groups: returns the list of grade groups.
# Values come from the data store's GROUPS constant.
# Used by the UI to label and order groups.
@app.get("/api/groups")
def groups():
    return GROUPS
