"""ACT layer: plan versions, approval state machine, audit log, events (SQLite)."""
from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import threading
import uuid
from pathlib import Path

from meltplan.contracts.models import PlanVersion
from meltplan.data.store import ROOT  # noqa: F401  (tests repoint it; the store itself no longer defaults to a file)

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


# Takes an exclusive advisory file lock on <db>.lock for this process.
# The lock is taken once per path and held for the process lifetime.
# Raises if another running MeltPlan process already owns the DB file.
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
        raise RuntimeError(f"{db} is owned by another running MeltPlan process (the API). "
                           "Use World() without store_path for scripts.") from None
    _OWNED[key] = fd


class PlanStore:
    # Opens the plan store (in memory by default, or a locked SQLite file).
    # Sets up a shared re-entrant lock and a fresh epoch id for SSE event ids.
    # Recreates the plans, audit and events tables from scratch.
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
          CREATE TABLE audit(seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, plan_id TEXT, version INT, actor TEXT, action TEXT, detail TEXT,
                            after_event INT);
          CREATE TABLE events(seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, payload TEXT, audience TEXT);""")

    # --- plans
    # Inserts a new plan version and writes a 'create' audit row.
    # Stored versions are never overwritten; a duplicate version raises.
    # Rolls back on any SQLite error so the file is not left locked.
    def save(self, p: PlanVersion, actor: str, ts: str, detail: str = ""):
        with self.lock:  # plain INSERT: a stored version is never overwritten (raises on a duplicate version)
            try:
                self.db.execute("INSERT INTO plans VALUES (?,?,?)", (p.plan_id, p.version, p.model_dump_json()))
                self._audit(ts, p, actor, "create", detail)
                self.db.commit()
            except sqlite3.Error:
                self.db.rollback()  # do not leave a transaction open: it would lock the file for the next store
                raise

    # Computes the next version number for a plan id.
    # It is the highest stored version plus one (1 if none exist).
    # Runs under the store lock.
    def next_version(self, plan_id: str) -> int:
        with self.lock:
            return (self.db.execute("SELECT max(version) FROM plans WHERE plan_id=?", (plan_id,)).fetchone()[0] or 0) + 1

    # Loads a plan version from the store.
    # Returns the requested version, or the latest one when version is None.
    # Returns None if the plan does not exist.
    def get(self, plan_id: str, version: int | None = None) -> PlanVersion | None:
        q = "SELECT body FROM plans WHERE plan_id=? " + ("AND version=?" if version else "ORDER BY version DESC LIMIT 1")
        with self.lock:
            row = self.db.execute(q, (plan_id, version) if version else (plan_id,)).fetchone()
        return PlanVersion.model_validate_json(row[0]) if row else None

    # Loads every stored plan version.
    # Versions are returned in insertion order.
    # Each row is parsed back into a PlanVersion.
    def versions(self) -> list[PlanVersion]:
        with self.lock:
            rows = self.db.execute("SELECT body FROM plans ORDER BY rowid").fetchall()
        return [PlanVersion.model_validate_json(r[0]) for r in rows]

    # Moves a plan to a new status through the approval state machine.
    # Raises ValueError if the transition is not allowed from the current status.
    # Persists the updated plan, writes an audit row and returns the plan.
    def transition(self, p: PlanVersion, new: str, actor: str, ts: str, detail: str = "") -> PlanVersion:
        if new not in ALLOWED[p.status]:
            raise ValueError(f"illegal transition {p.status} -> {new}")
        p.status = new
        with self.lock:
            self.db.execute("UPDATE plans SET body=? WHERE plan_id=? AND version=?", (p.model_dump_json(), p.plan_id, p.version))
            self._audit(ts, p, actor, new, detail)
            self.db.commit()
        return p

    # Writes one audit row for a plan action.
    # Records the latest event seq so audit rows and events can be merged in order.
    # Callers are responsible for holding the lock and committing.
    def _audit(self, ts, p, actor, action, detail):
        # after_event: the last event seq at this point, so the act-layer logs can merge audit rows and events in true order
        self.db.execute("INSERT INTO audit(ts,plan_id,version,actor,action,detail,after_event) VALUES (?,?,?,?,?,?,(SELECT coalesce(max(seq),0) FROM events))",
                        (ts, p.plan_id, p.version, actor, action, detail))

    # Returns the audit log, optionally filtered to one plan id.
    # Rows are ordered by their sequence number.
    # Each row is returned as a dict of named fields.
    def audit(self, plan_id: str | None = None) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT ts,plan_id,version,actor,action,detail,after_event FROM audit " + ("WHERE plan_id=? " if plan_id else "") + "ORDER BY seq",
                                   (plan_id,) if plan_id else ()).fetchall()
        return [dict(zip(["ts", "plan_id", "version", "actor", "action", "detail", "after_event"], r)) for r in rows]

    # --- events (SSE feed)
    # Appends an event to the SSE event feed.
    # Payload and audience are stored as JSON.
    # Returns the event as a dict including its new seq number.
    def emit(self, ts: str, kind: str, payload: dict, audience: list[str]) -> dict:
        with self.lock:
            cur = self.db.execute("INSERT INTO events(ts,kind,payload,audience) VALUES (?,?,?,?)", (ts, kind, json.dumps(payload), json.dumps(audience)))
            self.db.commit()
        return dict(seq=cur.lastrowid, ts=ts, kind=kind, payload=payload, audience=audience)

    # Returns all events with a seq greater than the given one.
    # Used by the SSE stream to send only new events.
    # Payload and audience are decoded from JSON.
    def events_since(self, seq: int) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT seq,ts,kind,payload,audience FROM events WHERE seq>? ORDER BY seq", (seq,)).fetchall()
        return [dict(seq=r[0], ts=r[1], kind=r[2], payload=json.loads(r[3]), audience=json.loads(r[4])) for r in rows]

    # Returns events of the given kinds at or after a timestamp.
    # Events are ordered by seq number.
    # Payload and audience are decoded from JSON.
    def events(self, kinds: list[str], since_ts: str = "") -> list[dict]:
        with self.lock:
            rows = self.db.execute(f"SELECT seq,ts,kind,payload,audience FROM events WHERE kind IN ({','.join('?' * len(kinds))}) AND ts>=? ORDER BY seq",
                                   (*kinds, since_ts)).fetchall()
        return [dict(seq=r[0], ts=r[1], kind=r[2], payload=json.loads(r[3]), audience=json.loads(r[4])) for r in rows]

    # Returns the highest event seq number so far.
    # Returns 0 when no events have been emitted.
    # Runs under the store lock.
    def last_seq(self) -> int:
        with self.lock:
            return self.db.execute("SELECT coalesce(max(seq), 0) FROM events").fetchone()[0]
