"""ACT layer: plan versions, approval state machine, audit log, events (SQLite)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scrapchef.contracts.models import PlanVersion
from scrapchef.data.store import ROOT

ALLOWED = {
    "PROPOSED": {"EMAILED", "SUPERSEDED", "REJECTED"},
    "EMAILED": {"SUPERVISOR_APPROVED", "REJECTED", "SUPERSEDED"},
    "SUPERVISOR_APPROVED": {"RELEASED", "SUPERSEDED"},
    "RELEASED": {"LOADING", "DONE", "SUPERSEDED"},
    "LOADING": {"DONE", "SUPERSEDED"},
    "DONE": set(), "SUPERSEDED": set(), "REJECTED": set(),
}


class PlanStore:
    def __init__(self, path: Path | str | None = None):
        self.db = sqlite3.connect(str(path or ROOT / "data" / "app.sqlite"), check_same_thread=False)
        self.db.executescript("""
          DROP TABLE IF EXISTS plans; DROP TABLE IF EXISTS audit; DROP TABLE IF EXISTS events;
          CREATE TABLE plans(plan_id TEXT, version INT, body TEXT, PRIMARY KEY(plan_id, version));
          CREATE TABLE audit(seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, plan_id TEXT, version INT, actor TEXT, action TEXT, detail TEXT);
          CREATE TABLE events(seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, payload TEXT, audience TEXT);""")

    # --- plans
    def save(self, p: PlanVersion, actor: str, ts: str, detail: str = ""):
        self.db.execute("INSERT OR REPLACE INTO plans VALUES (?,?,?)", (p.plan_id, p.version, p.model_dump_json()))
        self._audit(ts, p, actor, "create", detail)
        self.db.commit()

    def get(self, plan_id: str, version: int | None = None) -> PlanVersion | None:
        q = "SELECT body FROM plans WHERE plan_id=? " + ("AND version=?" if version else "ORDER BY version DESC LIMIT 1")
        row = self.db.execute(q, (plan_id, version) if version else (plan_id,)).fetchone()
        return PlanVersion.model_validate_json(row[0]) if row else None

    def versions(self) -> list[PlanVersion]:
        return [PlanVersion.model_validate_json(r[0]) for r in self.db.execute("SELECT body FROM plans ORDER BY rowid")]

    def transition(self, p: PlanVersion, new: str, actor: str, ts: str, detail: str = "") -> PlanVersion:
        if new not in ALLOWED[p.status]:
            raise ValueError(f"illegal transition {p.status} -> {new}")
        p.status = new
        self.db.execute("UPDATE plans SET body=? WHERE plan_id=? AND version=?", (p.model_dump_json(), p.plan_id, p.version))
        self._audit(ts, p, actor, new, detail)
        self.db.commit()
        return p

    def _audit(self, ts, p, actor, action, detail):
        self.db.execute("INSERT INTO audit(ts,plan_id,version,actor,action,detail) VALUES (?,?,?,?,?,?)", (ts, p.plan_id, p.version, actor, action, detail))

    def audit(self, plan_id: str | None = None) -> list[dict]:
        cur = self.db.execute("SELECT ts,plan_id,version,actor,action,detail FROM audit " + ("WHERE plan_id=? " if plan_id else "") + "ORDER BY seq", (plan_id,) if plan_id else ())
        return [dict(zip(["ts", "plan_id", "version", "actor", "action", "detail"], r)) for r in cur]

    # --- events (SSE feed)
    def emit(self, ts: str, kind: str, payload: dict, audience: list[str]) -> dict:
        cur = self.db.execute("INSERT INTO events(ts,kind,payload,audience) VALUES (?,?,?,?)", (ts, kind, json.dumps(payload), json.dumps(audience)))
        self.db.commit()
        return dict(seq=cur.lastrowid, ts=ts, kind=kind, payload=payload, audience=audience)

    def events_since(self, seq: int) -> list[dict]:
        return [dict(seq=r[0], ts=r[1], kind=r[2], payload=json.loads(r[3]), audience=json.loads(r[4]))
                for r in self.db.execute("SELECT seq,ts,kind,payload,audience FROM events WHERE seq>? ORDER BY seq", (seq,))]
