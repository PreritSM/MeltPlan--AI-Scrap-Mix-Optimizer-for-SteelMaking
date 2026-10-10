"""ACT layer: plan versions, approval state machine, audit log, events (SQLite)."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path

from scrapchef.contracts.models import PlanVersion
from scrapchef.data.store import ROOT

ALLOWED = {
    "PROPOSED": {"EMAILED", "SUPERSEDED", "REJECTED"},
    "EMAILED": {"SUPERVISOR_APPROVED", "REJECTED", "SUPERSEDED"},
    "SUPERVISOR_APPROVED": {"RELEASED", "REJECTED", "SUPERSEDED"},
    "RELEASED": {"LOADING", "DONE", "SUPERSEDED"},
    "LOADING": {"DONE", "SUPERSEDED"},
    "DONE": set(), "SUPERSEDED": set(), "REJECTED": set(),
}


class PlanStore:
    def __init__(self, path: Path | str | None = None):
        self.db = sqlite3.connect(str(path or ROOT / "data" / "app.sqlite"), check_same_thread=False)
        self.lock = threading.RLock()     # one connection is shared by request threads and the SSE reader
        self.epoch = uuid.uuid4().hex[:8]  # changes on every reset; event seq restarts at 1, so SSE ids carry the epoch
        self.db.executescript("""
          DROP TABLE IF EXISTS plans; DROP TABLE IF EXISTS audit; DROP TABLE IF EXISTS events;
          CREATE TABLE plans(plan_id TEXT, version INT, body TEXT, PRIMARY KEY(plan_id, version));
          CREATE TABLE audit(seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, plan_id TEXT, version INT, actor TEXT, action TEXT, detail TEXT);
          CREATE TABLE events(seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, payload TEXT, audience TEXT);""")

    # --- plans
    def save(self, p: PlanVersion, actor: str, ts: str, detail: str = ""):
        with self.lock:  # plain INSERT: a stored version is never overwritten (raises on a duplicate version)
            try:
                self.db.execute("INSERT INTO plans VALUES (?,?,?)", (p.plan_id, p.version, p.model_dump_json()))
                self._audit(ts, p, actor, "create", detail)
                self.db.commit()
            except sqlite3.Error:
                self.db.rollback()  # do not leave a transaction open: it would lock the file for the next store
                raise

    def next_version(self, plan_id: str) -> int:
        with self.lock:
            return (self.db.execute("SELECT max(version) FROM plans WHERE plan_id=?", (plan_id,)).fetchone()[0] or 0) + 1

    def get(self, plan_id: str, version: int | None = None) -> PlanVersion | None:
        q = "SELECT body FROM plans WHERE plan_id=? " + ("AND version=?" if version else "ORDER BY version DESC LIMIT 1")
        with self.lock:
            row = self.db.execute(q, (plan_id, version) if version else (plan_id,)).fetchone()
        return PlanVersion.model_validate_json(row[0]) if row else None

    def versions(self) -> list[PlanVersion]:
        with self.lock:
            rows = self.db.execute("SELECT body FROM plans ORDER BY rowid").fetchall()
        return [PlanVersion.model_validate_json(r[0]) for r in rows]

    def transition(self, p: PlanVersion, new: str, actor: str, ts: str, detail: str = "") -> PlanVersion:
        if new not in ALLOWED[p.status]:
            raise ValueError(f"illegal transition {p.status} -> {new}")
        p.status = new
        with self.lock:
            self.db.execute("UPDATE plans SET body=? WHERE plan_id=? AND version=?", (p.model_dump_json(), p.plan_id, p.version))
            self._audit(ts, p, actor, new, detail)
            self.db.commit()
        return p

    def _audit(self, ts, p, actor, action, detail):
        self.db.execute("INSERT INTO audit(ts,plan_id,version,actor,action,detail) VALUES (?,?,?,?,?,?)", (ts, p.plan_id, p.version, actor, action, detail))

    def audit(self, plan_id: str | None = None) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT ts,plan_id,version,actor,action,detail FROM audit " + ("WHERE plan_id=? " if plan_id else "") + "ORDER BY seq",
                                   (plan_id,) if plan_id else ()).fetchall()
        return [dict(zip(["ts", "plan_id", "version", "actor", "action", "detail"], r)) for r in rows]

    # --- events (SSE feed)
    def emit(self, ts: str, kind: str, payload: dict, audience: list[str]) -> dict:
        with self.lock:
            cur = self.db.execute("INSERT INTO events(ts,kind,payload,audience) VALUES (?,?,?,?)", (ts, kind, json.dumps(payload), json.dumps(audience)))
            self.db.commit()
        return dict(seq=cur.lastrowid, ts=ts, kind=kind, payload=payload, audience=audience)

    def events_since(self, seq: int) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT seq,ts,kind,payload,audience FROM events WHERE seq>? ORDER BY seq", (seq,)).fetchall()
        return [dict(seq=r[0], ts=r[1], kind=r[2], payload=json.loads(r[3]), audience=json.loads(r[4])) for r in rows]

    def last_seq(self) -> int:
        with self.lock:
            return self.db.execute("SELECT coalesce(max(seq), 0) FROM events").fetchone()[0]
