"""Simulation world: one clock, one plan/heat state, shared by both UI panes via events.

Modes: SHADOW_REPLAY (real held-out heats; lab results belong to the mill's real charge) and
SIMULATED (scripted drift; lab results generated from OUR plan; clearly labelled)."""
from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta

import numpy as np

from scrapchef.act import email as email_mod
from scrapchef.act.plans import PlanStore
from scrapchef.act.roles import project
from scrapchef.contracts.models import PlanVersion
from scrapchef.data.store import GROUPS, cfg, heat_table
from scrapchef.intelligence.agent import narrator
from scrapchef.intelligence.agent.triggers import SurpriseTracker
from scrapchef.intelligence.estimator.kalman import ElementKF
from scrapchef.intelligence.estimator.massbalance import build
from scrapchef.intelligence.optimizer.evaluate import evaluate_mix
from scrapchef.intelligence.optimizer.model import ELS, make_context
from scrapchef.intelligence.optimizer.plan import build_plan, heat_from_groups
from scrapchef.read.mock.adapters import AssumedPrices, MockHandling, MockInventory, MockSchedule

ISO = "%Y-%m-%dT%H:%M:%S"


class World:
    def __init__(self, mode: str = "SHADOW_REPLAY"):
        self.lock = threading.RLock()
        self.mill = cfg("mill")
        self.reset(mode)

    # ------------------------------------------------------------ setup
    def reset(self, mode: str | None = None, drift_factor: float = 12.0, seed: int = 7, drift_group: str = "SCRAP_25A_3ACH"):
        with self.lock:
            self.mode = mode or getattr(self, "mode", "SHADOW_REPLAY")
            self.store = PlanStore()
            self.handling = MockHandling()
            ds = {el: build(el) for el in ELS}
            d = ds["cu"]
            n = len(d.y)
            self.replay = d.heats.iloc[int(0.85 * n):].reset_index(drop=True)      # held-out heats (REAL)
            self.replay_c_prev = {el: ds[el].c_prev[int(0.85 * n):] for el in ELS}
            self.B_all = d.B[int(0.85 * n):]
            # start on the first day whose next-24h window holds enough heats (removed heats leave gaps in the replay)
            ts, min_heats = self.replay.start_ts, 8
            self.clock = next((t0 for t0 in (d + timedelta(hours=6) for d in ts.dt.normalize().unique())
                               if ((ts >= t0) & (ts < t0 + timedelta(hours=self.mill["plan_horizon_h"]))).sum() >= min_heats),
                              ts.iloc[0].normalize() + timedelta(hours=6))
            self.cursor = int((ts < self.clock).sum())
            self.ctx = make_context(variant="boot_train")
            raw = self.ctx.est
            sig = raw.sigma_eps
            self.kf = {el: ElementKF(raw.mu[el], raw.cov[el], sig[el]) for el in ELS}
            self.tracker = SurpriseTracker()
            self.heel = {el: float(self.replay_c_prev[el][self.cursor]) for el in ELS}
            self.drift_factor, self.rng = drift_factor, np.random.default_rng(seed)
            self.e_true = {el: raw.mu[el].copy() for el in ELS}
            if self.mode == "SIMULATED":
                self.e_true["cu"][GROUPS.index(drift_group)] *= drift_factor
            self.drift_group = drift_group
            self.plan_id: str | None = None
            self.progress: dict[int, str] = {}
            self.plan_heat0 = 0           # replay index of plan slot 1
            self.decisions: list[dict] = []
            self.emails: list[dict] = []
            self.chat: list[dict] = []
            self.heat_log: list[dict] = []
            self.explanation: dict = {}
            self.drift_alert: dict | None = None
            self.released_version: int | None = None
            self.pending_banner: dict | None = None
            self.sim_step = 0
            self._emit("reset", {"mode": self.mode}, ["process", "email", "supervisor", "shift_leader", "crane"])

    # ------------------------------------------------------------ helpers
    def _ts(self) -> str:
        return self.clock.strftime(ISO)

    def _emit(self, kind: str, payload: dict, audience: list[str]):
        return self.store.emit(self._ts(), kind, payload, audience)

    def current_plan(self) -> PlanVersion | None:
        if not self.plan_id:
            return None
        vs = [p for p in self.store.versions() if p.plan_id == self.plan_id and p.status not in ("SUPERSEDED", "REJECTED")]
        return vs[-1] if vs else None

    def baseline_per_t(self) -> float:
        ht = heat_table()
        ht = ht[(ht.start_ts < self.clock) & (ht.start_ts >= self.clock - timedelta(days=30))]
        if ht.empty:
            ht = heat_table().tail(500)
        X = ht[GROUPS].to_numpy()
        T = X @ self.ctx.yields
        ok = T > 80
        return float((((self.ctx.prices + self.ctx.energy_k) @ X[ok].T).sum()) / T[ok].sum())

    def _set_estimates(self):
        for el in ELS:
            self.ctx.est.mu[el] = self.kf[el].e.copy()

    def _inventory(self, idx: list[int]) -> np.ndarray:
        """MOCK yard stock per group: 1.3 x (recipe upper-bound share x horizon charge), at least 1.15 x observed usage.
        Stock only binds when a shortage is reported (the real feed will replace this)."""
        horizon_t = len(idx) * self.mill["rated_tap_t"] / 0.92
        ub = np.array([self.ctx.bounds[g][1] for g in GROUPS])
        usage = self.replay.iloc[idx][GROUPS].sum().to_numpy()
        stock = np.maximum(1.3 * ub * horizon_t, 1.15 * usage)
        inv = MockInventory(dict(zip(GROUPS, stock)), headroom=1.0).snapshot(self._ts())
        return np.array([i.tonnes for i in inv])

    # ------------------------------------------------------------ planning (06:00 cycle)
    def make_plan(self, trigger: str = "daily 06:00"):
        with self.lock:
            t0, t1 = self.clock, self.clock + timedelta(hours=self.mill["plan_horizon_h"])
            idx = [i for i in range(self.cursor, len(self.replay)) if t0 <= self.replay.start_ts.iloc[i] < t1]
            if not idx:
                return None
            sched = MockSchedule(len(idx)).horizon(self._ts())
            taps = [s.target_tap_t for s in sched]
            self._set_estimates()
            self.ctx.inventory = self._inventory(idx)
            self.ctx.heel_c0 = dict(self.heel)
            self.plan_heel0 = dict(self.heel)
            date = self.clock.strftime("%Y-%m-%d")
            for p in self.store.versions():
                if p.plan_id == date and p.status not in ("SUPERSEDED", "REJECTED", "DONE"):
                    self.store.transition(p, "SUPERSEDED", "agent", self._ts(), "new daily plan")
            plan = build_plan(self.ctx, taps, date, version=1, ts=self._ts())
            if plan is None:
                return None
            self.plan_id, self.plan_heat0 = date, idx[0]
            self.progress = {}
            self.released_version = None
            self.store.save(plan, "agent", self._ts(), trigger)
            facts = dict(decision="new_plan", version=1, trigger_text="Daily 06:00 plan.", n_heats=len(plan.heats), moves=[],
                         cost_delta_per_t=0.0, new_per_t=sum(h.cost.total for h in plan.heats) / sum(taps), reason="", binding=(plan.binding[0].plain if plan.binding else ""))
            rationale = ("Lowest-cost mix meeting every element limit with 95% confidence, within the mill's demonstrated recipe range."
                         + (f" {plan.binding[0].plain}." if plan.binding else ""))
            self._send_email(plan, rationale)
            self.decisions.append(dict(ts=self._ts(), trigger=trigger, tools=["get_state", "run_optimizer", "publish_plan"], decision="plan",
                                       version=1, facts=facts, explanation={"email": rationale}, narrator={"email": "template"}))
            self._emit("plan_created", {"plan_id": plan.plan_id, "version": 1}, ["process", "email", "supervisor"])
            return plan

    def _send_email(self, plan: PlanVersion, rationale: str):
        mail = email_mod.render(plan, self.clock.strftime("%Y-%m-%d"), self.baseline_per_t(), rationale)
        mail.update(ts=self._ts(), read=False)
        self.emails.append(mail)
        self.store.transition(plan, "EMAILED", "agent", self._ts(), "daily email plan")
        self._emit("email", {"subject": mail["subject"], "version": plan.version}, ["process", "email"])

    # ------------------------------------------------------------ stepping the replay / scenario
    def step_heat(self) -> dict | None:
        with self.lock:
            plan = self.current_plan()
            if self.mode == "SIMULATED":
                return self._step_sim(plan)
            if self.cursor >= len(self.replay):
                return None
            row = self.replay.iloc[self.cursor]
            self.clock = max(self.clock, row.start_ts)
            x = row[GROUPS].to_numpy(float)
            meas = {el: float(row[el]) for el in ELS}
            res = self._reveal(x, meas, float(self.B_all[self.cursor]), str(row.heatid), actual_tap=float(self.ctx.yields @ x))
            self.cursor += 1
            self._after_heat(res)
            return res

    def _step_sim(self, plan: PlanVersion | None):
        if plan is None:
            return None
        slot = self.sim_step + 1
        if slot > len(plan.heats):
            return None
        h = plan.heats[slot - 1]
        x = np.array([h.group_t.get(g, 0.0) for g in GROUPS])
        B = h.target_tap_t + self.mill["heel_t"]
        meas = {}
        for el in ELS:
            noise = self.rng.normal(0, self.ctx.est.sigma_eps[el])
            meas[el] = max(0.0, (self.mill["heel_t"] * self.heel[el] + self.e_true[el] @ x + noise) / B)
        self.clock += timedelta(minutes=60)
        res = self._reveal(x, meas, B, f"SIM-{slot}", actual_tap=h.target_tap_t)
        self.sim_step += 1
        self._after_heat(res)
        return res

    def _reveal(self, x, meas, B, heatid, actual_tap) -> dict:
        slot = self.sim_step + 1 if self.mode == "SIMULATED" else (self.cursor - self.plan_heat0 + 1 if self.plan_id else 0)
        H = self.mill["heel_t"]
        out, trig = {}, []
        for el in ELS:
            kf = self.kf[el]
            mean = (H * self.heel[el] + kf.e @ x) / B
            sd = np.sqrt(x @ self.ctx.est.cov[el] @ x + self.ctx.est.sigma_eps[el] ** 2) / B
            z = (meas[el] - mean) / sd
            y = meas[el] * B - H * self.heel[el]
            kf.update(x, y)
            t = self.tracker.add(el, float(z))
            if t:
                trig.append(t)
            out[el] = dict(predicted=float(mean), measured=meas[el], z=float(z), sd=float(sd))
        self.heel = dict(meas)
        drift = {el: [GROUPS[i] for i in self.kf[el].drift(self.mill["drift_k_sigma"])] for el in ELS}
        drift = {el: g for el, g in drift.items() if g}
        rec = dict(heatid=heatid, ts=self._ts(), slot=slot, elements=out, triggers=trig, drift=drift)
        self.heat_log.append(rec)
        if self.plan_id and 1 <= slot:
            self.progress[slot] = "tapped"
        return rec

    def _after_heat(self, rec: dict):
        self._emit("heat_tapped", {"heatid": rec["heatid"], "slot": rec["slot"], "elements": rec["elements"]}, ["process", "supervisor", "shift_leader"])
        if rec["drift"]:
            self.drift_alert = dict(ts=rec["ts"], groups=rec["drift"],
                                    text="Composition drift: " + "; ".join(f"{e.upper()} in {', '.join(g)}" for e, g in rec["drift"].items()))
            self._emit("drift", self.drift_alert, ["process", "supervisor", "shift_leader"])
        if rec["triggers"] or rec["drift"]:
            why = (rec["triggers"] or [self.drift_alert["text"]])[0]
            self.replan(trigger="lab_surprise" if rec["triggers"] else "drift", trigger_text=why + ".")

    # ------------------------------------------------------------ agent: replan with anti-churn
    def replan(self, trigger: str, trigger_text: str, extra: dict | None = None):
        with self.lock:
            plan = self.current_plan()
            if plan is None:
                return None
            open_idx = [i for i, h in enumerate(plan.heats) if self.progress.get(h.slot) not in ("tapped", "loading") and not h.frozen]
            if not open_idx:
                return None
            self._set_estimates()
            self.ctx.heel_c0 = dict(self.plan_heel0)
            frozen = {i: np.array([plan.heats[i].group_t.get(g, 0.0) for g in GROUPS]) for i in range(len(plan.heats)) if i not in open_idx}
            taps = [h.target_tap_t for h in plan.heats]
            tools = ["get_state", "evaluate_mix", "run_optimizer", "compare_plans"]
            # old plan re-evaluated under updated estimates
            heel = plan.heats[0].predicted and self.plan_heel0
            old_risk = 0.0
            old_cost = 0.0
            for i in open_idx:
                ev = evaluate_mix(self.ctx, plan.heats[i].group_t, taps[i], dict(heel))
                old_risk = max(old_risk, max(p["p_exceed"] for p in ev["predicted"].values()))
                old_cost += ev["cost"]["total"]
            new = build_plan(self.ctx, taps, plan.plan_id, version=plan.version + 1, parent=plan.version, frozen=frozen, ts=self._ts())
            if new is None:
                decision = dict(ts=self._ts(), trigger=trigger, tools=tools, decision="infeasible", facts={}, explanation={}, narrator="-", reason="optimiser infeasible")
                self.decisions.append(decision)
                return decision
            new_cost = sum(new.heats[i].cost.total for i in open_idx)
            t_open = sum(taps[i] for i in open_idx)
            d_per_t = (new_cost - old_cost) / t_open
            moves = {}
            for i in open_idx:
                for g in set(plan.heats[i].group_t) | set(new.heats[i].group_t):
                    moves[g] = moves.get(g, 0.0) + new.heats[i].group_t.get(g, 0.0) - plan.heats[i].group_t.get(g, 0.0)
            moves = [dict(group=g, delta_t=round(v, 1)) for g, v in moves.items() if abs(v) >= 0.05]
            churn = abs(d_per_t) < self.mill["anti_churn_usd_per_t"] and old_risk <= self.mill["alpha"] + 1e-9
            facts = dict(decision="no_change" if churn else "replan", version=new.version, trigger_text=trigger_text, n_heats=len(open_idx), moves=moves,
                         cost_delta_per_t=round(d_per_t, 2), new_per_t=round(new_cost / t_open, 2), old_plan_risk=round(old_risk, 3),
                         binding=(new.binding[0].plain if new.binding else ""),
                         reason=f"cost change {d_per_t:+.2f} $/t is below the {self.mill['anti_churn_usd_per_t']} $/t threshold and risk stays within limits.")
            nar = narrator.narrate(facts)
            decision = dict(ts=self._ts(), trigger=trigger, tools=tools + ([] if churn else ["publish_plan"]), decision=facts["decision"],
                            version=None if churn else new.version, facts=facts, explanation=nar["text"], narrator=nar["narrator"])
            self.decisions.append(decision)
            if churn:
                self._emit("agent_decision", {"decision": "no_change", "trigger": trigger_text}, ["process", "supervisor"])
                return decision
            old = plan
            if old.status not in ("SUPERSEDED", "REJECTED"):
                self.store.transition(old, "SUPERSEDED", "agent", self._ts(), f"replan: {trigger_text}")
            self.store.save(new, "agent", self._ts(), trigger_text)
            self.explanation = nar["text"]
            self._send_email(new, nar["text"]["email"])
            self.pending_banner = dict(version=new.version, text=nar["text"]["crane"], ts=self._ts())
            self.tracker = SurpriseTracker()  # cooldown: the estimates just changed, so start a fresh surprise history
            self._emit("replan", {"version": new.version, "trigger": trigger_text, "explanation": nar["text"]},
                       ["process", "email", "supervisor", "shift_leader", "crane"])
            return decision

    # ------------------------------------------------------------ user actions
    def approve(self, actor="supervisor"):
        with self.lock:
            p = self.current_plan()
            self.store.transition(p, "SUPERVISOR_APPROVED", actor, self._ts(), "approved")
            self._emit("approved", {"version": p.version}, ["process", "supervisor", "shift_leader"])

    def reject(self, reason: str):
        with self.lock:
            p = self.current_plan()
            self.store.transition(p, "REJECTED", "supervisor", self._ts(), reason)
            self._emit("rejected", {"version": p.version, "reason": reason}, ["process", "supervisor"])
            self.plan_id_backup = self.plan_id
            # re-plan from scratch with the supervisor's reason on record
            plan = self.make_plan(trigger=f"rejection: {reason}")
            return plan

    def release(self, actor="shift_leader"):
        with self.lock:
            p = self.current_plan()
            self.store.transition(p, "RELEASED", actor, self._ts(), "released to cranes")
            self.released_version = p.version
            self.pending_banner = None
            self._emit("released", {"version": p.version}, ["process", "shift_leader", "crane"])

    def edit(self, slot: int, group_t: dict[str, float], actor="supervisor") -> PlanVersion:
        with self.lock:
            p = self.current_plan()
            self.ctx.heel_c0 = dict(self.plan_heel0)
            self._set_estimates()
            h = p.heats[slot - 1]
            heel = dict(self.plan_heel0)
            for i in range(slot - 1):  # propagate heel via planned P50 of previous heats
                B = p.heats[i].target_tap_t + self.mill["heel_t"]
                heel = {el: p.heats[i].predicted[el].mean for el in ELS}
            new = p.model_copy(deep=True)
            new.version = max(v.version for v in self.store.versions() if v.plan_id == p.plan_id) + 1
            new.parent_version, new.created_by, new.status, new.created_ts = p.version, "user", "PROPOSED", self._ts()
            new.heats[slot - 1] = heat_from_groups(self.ctx, slot, h.target_tap_t, group_t, heel, h.spec_class)
            self.store.transition(p, "SUPERSEDED", actor, self._ts(), f"manual edit of heat {slot}")
            self.store.save(new, actor, self._ts(), f"manual edit heat {slot}")
            self.store.transition(new, "EMAILED", actor, self._ts(), "edit saved")
            self._emit("edited", {"version": new.version, "slot": slot}, ["process", "supervisor", "shift_leader"])
            return new

    def evaluate(self, slot: int, group_t: dict[str, float]) -> dict:
        with self.lock:
            p = self.current_plan()
            self._set_estimates()
            h = p.heats[slot - 1]
            heel = dict(self.plan_heel0) if slot == 1 else {el: p.heats[slot - 2].predicted[el].mean for el in ELS}
            ev = evaluate_mix(self.ctx, group_t, h.target_tap_t, heel)
            ev["delta_cost_per_t"] = ev["cost"]["per_t"] - h.cost.per_t
            return ev

    def grab_done(self, slot: int, basket: int, mat_code: int, t: float):
        with self.lock:
            e = self.handling.record(self._ts(), "crane-1", mat_code, t, slot, basket)
            p = self.current_plan()
            if p and p.status == "RELEASED":
                self.store.transition(p, "LOADING", "crane", self._ts(), "first grab")
            self.progress[slot] = self.progress.get(slot) if self.progress.get(slot) == "tapped" else "loading"
            self._emit("grab_done", e.model_dump(), ["process", "shift_leader", "crane"])
            return e

    def report_shortage(self, group: str):
        with self.lock:
            i = GROUPS.index(group)
            self.ctx.inventory = None
            inv = self.ctx.inventory
            text = f"Shortage reported: {group} unavailable."
            self._emit("shortage", {"group": group}, ["process", "supervisor", "shift_leader"])
            ctx_inv = np.full(len(GROUPS), 1e6); ctx_inv[i] = 0.0
            self.ctx.inventory = ctx_inv
            return self.replan("shortage", text)

    # ------------------------------------------------------------ views
    def phone(self, role: str) -> dict:
        with self.lock:
            p = self.current_plan() or (self.store.versions()[-1] if self.store.versions() else None)
            out = dict(role=role, sim_time=self._ts(), mode=self.mode)
            if role == "email":
                out["emails"] = list(reversed(self.emails))
            if p is not None:
                view = project(p, role, self.baseline_per_t(), self.explanation, self.progress)
                if role == "crane":
                    rv = self.released_version
                    rel = next((v for v in self.store.versions() if v.plan_id == p.plan_id and v.version == rv), None) if rv else None
                    view = project(rel, "crane", None, self.explanation, self.progress) if rel else dict(status="NO RELEASED PLAN", loading=[])
                    view["banner"] = self.pending_banner
                out["plan"] = view
            if role == "supervisor":
                out["drift_alert"] = self.drift_alert
                out["audit"] = self.store.audit(self.plan_id)
                out["backtest"] = _load("backtest.json")
            if role == "shift_leader":
                out["drift_alert"] = self.drift_alert and dict(text=self.drift_alert["text"])
                out["heat_log"] = [dict(heatid=h["heatid"], slot=h["slot"], cu=h["elements"]["cu"]) for h in self.heat_log[-12:]]
                out["changes"] = [dict(version=v.version, by=v.created_by, ts=v.created_ts, status=v.status) for v in self.store.versions() if v.plan_id == self.plan_id]
                out["chat"] = self.chat
            return out


def _load(name: str):
    import json
    from scrapchef.data.store import ROOT
    f = ROOT / "data" / name
    return json.loads(f.read_text()) if f.exists() else None
