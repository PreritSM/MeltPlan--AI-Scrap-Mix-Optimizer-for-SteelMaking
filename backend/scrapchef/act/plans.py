"""ACT layer: plan versions, approval state machine, audit log, events (SQLite)."""
from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import threading
import uuid
from pathlib import Path

from scrapchef.contracts.models import PlanVersion
from scrapchef.data.store import ROOT  # noqa: F401  (tests repoint it; the store itself no longer defaults to a file)

ALLOWED = {
    "PROPOSED": {"EMAILED", "SUPERSEDED", "REJECTED", "DROPPED"},
    "EMAILED": {"SUPERVISOR_APPROVED", "REJECTED", "SUPERSEDED", "DROPPED"},
    "SUPERVISOR_APPROVED": {"RELEASED", "REJECTED", "SUPERSEDED", "DROPPED"},
    "RELEASED": {"LOADING", "DONE", "SUPERSEDED", "DROPPED"},
    "LOADING": {"DONE", "SUPERSEDED", "DROPPED"},
    "REJECTED": {"DROPPED"},   # rejected and no feasible re-plan: the day's plan is withdrawn
    "DONE": set(), "SUPERSEDED": set(), "DROPPED": set(),
}


_OWNED: dict[str, int] = {}   # resolved db path -> lock fd held for the life of this process (D36)


def _own(db: Path) -> None:
    """Exclusive, non-blocking advisory lock on <db>.lock, taken once per path per process (flock is per open file,
    and a reset re-creates the store). Another process holding it means another server owns the file: refuse."""
    key = str(db.resolve())
    if key in _OWNED:
        return
    fd = os.open(key + ".lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError(f"{db} is owned by another running ScrapChef process (the API). "
                           "Use World() without store_path for scripts.") from None
    _OWNED[key] = fd


class PlanStore:
    def __init__(self, path: Path | str | None = None):
        """path=None: a private in-memory store (scripts, REPLs, tests). A file path is the operational DB; only one
        process may own it, so a second process raises here before anything is dropped."""
        if path is not None and str(path) != ":memory:":
            _own(Path(path))
        self.db = sqlite3.connect(str(path) if path is not None else ":memory:", check_same_thread=False)
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
