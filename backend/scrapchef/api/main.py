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


def guard(fn):
    try:
        return fn()
    except HTTPException:
        raise
    except Exception as e:  # surface as 4xx with the message, keep server alive
        traceback.print_exc()
        raise HTTPException(400, f"{type(e).__name__}: {e}")


@app.get("/api/state")
def state():
    return process_state(world)


@app.get("/api/phone/{role}")
def phone(role: str):
    if role not in ROLES:
        raise HTTPException(404, "unknown role")
    return world.phone(role)


def resume_seq(store, last_event_id: str | None, since: int | None) -> int:
    """Where an SSE connection starts. Ids are '<epoch>:<seq>'; an id from an older store (reset or server restart) restarts at 0.
    A fresh connection without an id or `since` starts at the latest event, so history is not replayed as unread."""
    if last_event_id:
        epoch, _, seq = last_event_id.partition(":")
        return int(seq) if epoch == store.epoch and seq.isdigit() else 0
    return since if since is not None else store.last_seq()


@app.get("/api/events")
async def events(request: Request, since: int | None = None):
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


@app.post("/api/sim/reset")
def reset(b: ResetBody):
    return guard(lambda: (world.reset(b.mode, b.drift_factor, drift_group=b.drift_group), dict(ok=True))[1])


@app.post("/api/sim/step_heat")
def step_heat():
    if world.agent_pending:     # the sim clock waits for the agent: an LLM run (~30 s) would otherwise see its heats tapped
        return dict(waiting=True, note="Agent working: the next heat waits.")
    if world.plan_finished():   # play stops at the end of the plan; "Plan 06:00" replays on to the next day's plan
        return dict(done=True, note="All heats of this plan are tapped. Press Plan 06:00 for the next plan.")
    return guard(lambda: world.step_heat() or dict(done=True))


@app.post("/api/sim/plan")
def make_plan():
    """Advance the clock to the next 06:00 (replaying heats on the way) and plan the next 24 h."""
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


@app.post("/api/plan/evaluate")
def evaluate(b: EvalBody):
    return guard(lambda: world.evaluate(b.slot, b.group_t))


@app.post("/api/plan/edit")
def edit(b: EvalBody):
    return guard(lambda: dict(version=world.edit(b.slot, b.group_t).version))


class ApproveBody(BaseModel):
    option: str | None = None   # an agent alternative's candidate_id; None approves the live version


@app.post("/api/plan/approve")
def approve(b: ApproveBody | None = None):
    return guard(lambda: (world.approve(option=b.option if b else None), dict(ok=True))[1])


class ReasonBody(BaseModel):
    reason: str = "supervisor rejection"


@app.post("/api/plan/reject")
def reject(b: ReasonBody):
    return guard(lambda: (world.reject(b.reason), dict(ok=True))[1])


@app.post("/api/plan/release")
def release():
    return guard(lambda: (world.release(), dict(ok=True))[1])


class GrabBody(BaseModel):
    slot: int
    basket: int
    mat_code: int
    tonnes: float


@app.post("/api/crane/grab_done")
def grab(b: GrabBody):
    return guard(lambda: world.grab_done(b.slot, b.basket, b.mat_code, b.tonnes).model_dump())


class BasketBody(BaseModel):
    slot: int
    basket: int


@app.post("/api/crane/basket_complete")
def basket_complete(b: BasketBody):
    return guard(lambda: world.basket_complete(b.slot, b.basket))


class ShortBody(BaseModel):
    group: str


@app.post("/api/report/shortage")
def shortage(b: ShortBody):
    return guard(lambda: dict(decision=(world.report_shortage(b.group) or {}).get("decision")))


class IssueBody(BaseModel):
    role: str = "shift_leader"
    text: str


@app.post("/api/report/issue")
def issue(b: IssueBody):
    """Free-text operational issue; the agent interprets it (async: the result arrives as events)."""
    return guard(lambda: world.report_issue(b.role, b.text) or dict(queued=True))


class InboxBody(BaseModel):
    text: str = ""


@app.post("/api/sim/inbox")
def inbox(b: InboxBody):
    """Scenario: a MOCK supplier / logistics email arrives in the agent's inbox."""
    def run():
        text = b.text or default_supplier_email()
        world.notify("supervisor", f"Inbound email (logistics): {text}", sender="logistics")
        return world.dispatch("inbox", text, "logistics") or dict(queued=True)
    return guard(run)


def default_supplier_email() -> str:
    """Scripted demo email: the yard can supply only part of the 3A the open heats need (MOCK)."""
    from scrapchef.sim.scenario import supplier_email
    return supplier_email(world)


class ChatBody(BaseModel):
    role: str
    text: str


@app.post("/api/chat")
def chat(b: ChatBody):
    """Questions go to the agent; the answer arrives in the role's chat thread (async)."""
    return guard(lambda: world.ask(b.role, b.text) or dict(queued=True))


@app.post("/api/email/read")
def email_read():
    for m in world.emails:
        m["read"] = True
    return dict(ok=True)


@app.get("/api/groups")
def groups():
    return GROUPS
