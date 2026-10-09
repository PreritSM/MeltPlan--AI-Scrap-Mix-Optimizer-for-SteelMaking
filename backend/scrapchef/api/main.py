"""FastAPI app: REST for actions, SSE for state-change events (both UI panes subscribe to one stream)."""
from __future__ import annotations

import asyncio
import json
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from scrapchef.act.roles import ROLES
from scrapchef.data.store import GROUPS, ROOT
from scrapchef.sim.view import process_state
from scrapchef.sim.world import World

app = FastAPI(title="Mini-ScrapChef")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
world = World("SHADOW_REPLAY")


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


@app.get("/api/events")
async def events(since: int = 0):
    async def gen():
        seq = since
        while True:
            for ev in world.store.events_since(seq):
                seq = ev["seq"]
                yield f"id: {seq}\ndata: {json.dumps(ev)}\n\n"
            yield ": keepalive\n\n"
            await asyncio.sleep(0.4)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


class ResetBody(BaseModel):
    mode: str = "SHADOW_REPLAY"
    drift_factor: float = 12.0
    drift_group: str = "SCRAP_16A_6A"


@app.post("/api/sim/reset")
def reset(b: ResetBody):
    return guard(lambda: (world.reset(b.mode, b.drift_factor, drift_group=b.drift_group), dict(ok=True))[1])


@app.post("/api/sim/step_heat")
def step_heat():
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


@app.post("/api/plan/approve")
def approve():
    return guard(lambda: (world.approve(), dict(ok=True))[1])


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


class ShortBody(BaseModel):
    group: str


@app.post("/api/report/shortage")
def shortage(b: ShortBody):
    return guard(lambda: dict(decision=(world.report_shortage(b.group) or {}).get("decision")))


class ChatBody(BaseModel):
    role: str
    text: str


@app.post("/api/chat")
def chat(b: ChatBody):
    def run():
        from scrapchef.intelligence.agent.chat import answer
        a = answer(world, b.role, b.text)
        world.chat += [dict(who=b.role, text=b.text, ts=world._ts()), dict(who="agent", text=a, ts=world._ts())]
        world._emit("chat", {"role": b.role}, ["process", b.role])
        return dict(answer=a)
    return guard(run)


@app.post("/api/email/read")
def email_read():
    for m in world.emails:
        m["read"] = True
    return dict(ok=True)


@app.get("/api/groups")
def groups():
    return GROUPS
