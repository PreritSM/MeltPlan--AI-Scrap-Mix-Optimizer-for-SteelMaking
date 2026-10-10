"""Simulation world: one clock, one plan/heat state, shared by both UI panes via events.

Modes: SHADOW_REPLAY (real held-out heats; lab results belong to the mill's real charge) and
SIMULATED (scripted drift; lab results generated from OUR plan; clearly labelled)."""
from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import numpy as np
import pandas as pd

from scrapchef.act import email as email_mod
from scrapchef.act.plans import PlanStore
from scrapchef.act.roles import project
from scrapchef.contracts.models import PlanVersion
from scrapchef.data.features import heel_chem, split_heats
from scrapchef.data.store import GROUPS, cfg, heat_table
from scrapchef.intelligence.agent import loop as agent_loop
from scrapchef.intelligence.agent.llm import LLM
from scrapchef.intelligence.agent.triggers import SurpriseTracker
from scrapchef.intelligence.estimator.kalman import ElementKF
from scrapchef.intelligence.estimator.massbalance import build
from scrapchef.intelligence.optimizer.evaluate import evaluate_mix, risk_matched_baseline
from scrapchef.intelligence.optimizer.model import ELS, make_context
from scrapchef.intelligence.optimizer.plan import build_plan, heat_from_groups
from scrapchef.read.mock.adapters import MockHandling, MockInventory, MockSchedule

ISO = "%Y-%m-%dT%H:%M:%S"


class World:
    def __init__(self, mode: str = "SHADOW_REPLAY", agent_async: bool = False, llm: LLM | None = None):
        self.lock = threading.RLock()
        self.mill = cfg("mill")
        self.llm = llm or LLM()
        self.agent_async = agent_async                      # the API runs the agent in a worker; tests run it inline
        self.executor = ThreadPoolExecutor(max_workers=1)   # one agent run at a time; later events queue
        self.agent_pending = 0
        self.reset(mode)

    # ------------------------------------------------------------ setup
    def reset(self, mode: str | None = None, drift_factor: float = 1.5, seed: int = 7, drift_group: str = "SCRAP_3A"):
        with self.lock:
            self.mode = mode or getattr(self, "mode", "SHADOW_REPLAY")
            self.store = PlanStore()
            self.rm_cache: dict = {}
            self.handling = MockHandling()
            ds = {el: build(el) for el in ELS}
            d = ds["cu"]
            i0 = split_heats(d.heats)[1]
            self.replay = d.heats.iloc[i0:].reset_index(drop=True)      # held-out heats (REAL)
            # heel chemistry by heatid: element designs drop different heats (e.g. a missing Sn), so row positions do not line up
            ht = heat_table()
            self.replay_c_prev = {el: pd.Series(heel_chem(ht, el)[0], index=ht.heatid).reindex(self.replay.heatid).to_numpy()
                                  for el in ELS}
            self.B_all = d.B[i0:]
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
            # agent planning state: availability caps / spec / risk level chosen by published re-plans, candidates, outputs
            self.spec_class, self.alpha = cfg("specs")["default"], self.mill["alpha"]
            self.caps: dict[str, float] = {}
            self.base_inventory = np.full(len(GROUPS), 1e6)
            self.candidates: dict[str, dict] = {}
            self.n_candidates = 0
            self.alternatives: dict[tuple, list[dict]] = {}
            self.messages: dict[str, list[dict]] = {}
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

    def _baseline_heats(self):
        ht = heat_table()
        ht = ht[(ht.start_ts < self.clock) & (ht.start_ts >= self.clock - timedelta(days=30))]
        if ht.empty:
            ht = heat_table().tail(500)
        X = ht[GROUPS].to_numpy()
        T = X @ self.ctx.yields
        return X[T > 80], T[T > 80]

    def baseline_per_t(self) -> float:
        """Cost per tonne of steel of the last 30 days' actual mix at the assumed prices. Not risk-matched, see risk_match()."""
        X, T = self._baseline_heats()
        return float((((self.ctx.prices + self.ctx.energy_k) @ X.T).sum()) / T.sum())

    def risk_match(self, plan: PlanVersion) -> dict | None:
        """Historical mix vs optimiser at the historical mix's own predicted risk (cached per plan version)."""
        key = (plan.plan_id, plan.version)
        if key not in self.rm_cache:
            X, _ = self._baseline_heats()
            self.rm_cache[key] = risk_matched_baseline(self.ctx, X.mean(0), [h.target_tap_t for h in plan.heats], dict(self.ctx.heel_c0))
        return self.rm_cache[key]

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
            self.spec_class, self.alpha, self.caps = cfg("specs")["default"], self.mill["alpha"], {}  # a new day starts from defaults
            self.base_inventory = self._inventory(idx)
            self.ctx.inventory = self.base_inventory.copy()
            self.ctx.limits = cfg("specs")["classes"][self.spec_class]
            self.ctx.heel_c0 = dict(self.heel)
            self.plan_heel0 = dict(self.heel)
            date = self.clock.strftime("%Y-%m-%d")
            for p in self.store.versions():
                if p.plan_id == date and p.status not in ("SUPERSEDED", "REJECTED", "DONE"):
                    self.store.transition(p, "SUPERSEDED", "agent", self._ts(), "new daily plan")
            prior = [p.version for p in self.store.versions() if p.plan_id == date]
            version = self.store.next_version(date)  # a re-plan of the same day (e.g. after a reject) continues the numbering
            plan = build_plan(self.ctx, taps, date, version=version, parent=max(prior) if prior else None, ts=self._ts())
            if plan is None:
                return None
            self.plan_id, self.plan_heat0 = date, idx[0]
            self.progress = {}
            self.released_version = None
            self.store.save(plan, "agent", self._ts(), trigger)
            facts = dict(decision="new_plan", version=version, trigger_text="Daily 06:00 plan.", n_heats=len(plan.heats), moves=[],
                         cost_delta_per_t=0.0, new_per_t=sum(h.cost.total for h in plan.heats) / sum(taps), reason="", binding=(plan.binding[0].plain if plan.binding else ""))
            rationale = ("Lowest-cost mix meeting every element limit with 95% confidence, within the mill's demonstrated recipe range."
                         + (f" {plan.binding[0].plain}." if plan.binding else ""))
            self._send_email(plan, rationale)
            self.decisions.append(dict(ts=self._ts(), trigger=trigger, tools=["get_state", "run_optimizer", "publish_plan"], decision="plan",
                                       version=version, facts=facts, explanation={"email": rationale}, narrator={"email": "template"}))
            self._emit("plan_created", {"plan_id": plan.plan_id, "version": version}, ["process", "email", "supervisor"])
            return plan

    def _send_email(self, plan: PlanVersion, rationale: str):
        mail = email_mod.render(plan, self.clock.strftime("%Y-%m-%d"), self.baseline_per_t(), rationale, self.risk_match(plan))
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
        heel_next = {}
        for el in ELS:
            kf = self.kf[el]
            mean = (H * self.heel[el] + kf.e @ x) / B
            sd = np.sqrt(x @ self.ctx.est.cov[el] @ x + self.ctx.est.sigma_eps[el] ** 2) / B
            if not np.isfinite(meas[el]):  # element not measured (e.g. heat 8F4439 has no Sn): no update, heel = prediction
                out[el] = dict(predicted=float(mean), measured=None, z=None, sd=float(sd))
                heel_next[el] = float(mean)
                continue
            z = (meas[el] - mean) / sd
            y = meas[el] * B - H * self.heel[el]
            kf.update(x, y)
            t = self.tracker.add(el, float(z))
            if t:
                trig.append(t)
            lim = cfg("specs")["classes"][self.spec_class].get(el)
            if lim is not None and meas[el] > lim:  # blueprint trigger: lab result above the limit
                trig.append(f"{el.upper()} lab result {meas[el]:.3f}% is above the {lim}% limit")
            out[el] = dict(predicted=float(mean), measured=meas[el], z=float(z), sd=float(sd))
            heel_next[el] = meas[el]
        self.heel = heel_next
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
        if (rec["triggers"] or rec["drift"]) and (self.open_eval() or {}).get("open_slots"):  # nothing to re-plan once all heats are tapped
            why = (rec["triggers"] or [self.drift_alert["text"]])[0]
            self.dispatch("lab_surprise" if rec["triggers"] else "drift", why + ".")

    # ------------------------------------------------------------ planning services (called by the agent's tools and its fallback)
    def base_plan(self) -> PlanVersion | None:
        """The plan a re-plan starts from: the live version, or the latest one if it was just rejected."""
        p = self.current_plan()
        if p is None and self.plan_id:
            vs = [v for v in self.store.versions() if v.plan_id == self.plan_id]
            p = vs[-1] if vs else None
        return p

    def _split(self, plan: PlanVersion) -> tuple[int, dict[int, np.ndarray]]:
        """(start, frozen): solve from the first heat after the last tapped one; loading / user-frozen heats keep their mix."""
        tapped = [i for i, h in enumerate(plan.heats) if self.progress.get(h.slot) == "tapped"]
        start = max(tapped) + 1 if tapped else 0
        frozen = {j: np.array([h.group_t.get(g, 0.0) for g in GROUPS]) for j, h in enumerate(plan.heats[start:])
                  if self.progress.get(h.slot) == "loading" or h.frozen}
        return start, frozen

    def _heel_start(self, plan: PlanVersion) -> dict[str, float]:
        """Heel for the first heat to solve: the last measured chemistry once a heat of this plan is tapped, else the plan's start."""
        return dict(self.heel) if any(self.progress.get(h.slot) == "tapped" for h in plan.heats) else dict(self.plan_heel0)

    def open_eval(self, plan: PlanVersion | None = None) -> dict | None:
        """The live plan's not-yet-tapped heats re-evaluated under current estimates: cost per t and worst P(exceed) per element."""
        plan = plan or self.base_plan()
        if plan is None:
            return None
        self._set_estimates()
        start, frozen = self._split(plan)
        sub = plan.heats[start:]
        if not sub:
            return None
        heel, cost, steel, risk = self._heel_start(plan), 0.0, 0.0, {el: 0.0 for el in ELS}
        for j, h in enumerate(sub):
            ev = evaluate_mix(self.ctx, h.group_t, h.target_tap_t, heel)
            heel = {el: ev["predicted"][el]["mean"] for el in ELS}
            if j in frozen:
                continue
            cost += ev["cost"]["total"]; steel += h.target_tap_t
            for el in ELS:
                risk[el] = max(risk[el], ev["predicted"][el]["p_exceed"])
        return dict(open_slots=[h.slot for j, h in enumerate(sub) if j not in frozen],
                    cost_per_t=round(cost / steel, 2) if steel else None, worst_p_exceed={el: round(v, 3) for el, v in risk.items()})

    def solve_candidate(self, params: dict | None = None) -> tuple[PlanVersion | None, dict]:
        """Run the optimiser for the open heats with agent-chosen settings. Stores the candidate; publishes nothing.
        params: exclude_groups, group_caps_t, alpha, spec_class, relax_recipe_bounds."""
        params = dict(params or {})
        with self.lock:
            plan = self.base_plan()
            if plan is None:
                return None, dict(status="no_plan")
            start, frozen = self._split(plan)
            sub = plan.heats[start:]
            if len(sub) == len(frozen):
                return None, dict(status="no_open_heats")
            self._set_estimates()
            c = copy.copy(self.ctx)
            c.heel_c0 = self._heel_start(plan)
            spec = params.get("spec_class") or self.spec_class
            alpha = float(params.get("alpha") or self.alpha)
            c.limits = cfg("specs")["classes"][spec]
            if params.get("relax_recipe_bounds"):
                # drop the minimum shares (e.g. the 3A floor) and allow more clean iron units for dilution; the maximum share of
                # every scrap grade stays at demonstrated practice, so relaxing can never load more of a suspect scrap
                c.bounds = {g: (0.0, 1.0 if g == "IRON_UNITS" else hi) for g, (lo, hi) in self.ctx.bounds.items()}
            used = np.zeros(len(GROUPS))
            for h in plan.heats[:start]:
                used += np.array([h.group_t.get(g, 0.0) for g in GROUPS])
            inv = np.maximum(self.base_inventory - used, 0.0)
            caps = dict(self.caps)
            caps.update({g: float(t) for g, t in (params.get("group_caps_t") or {}).items()})
            caps.update({g: 0.0 for g in params.get("exclude_groups") or []})
            frozen_use = sum(frozen.values()) if frozen else np.zeros(len(GROUPS))
            for g, cap in caps.items():
                i = GROUPS.index(g)
                inv[i] = min(inv[i], cap + frozen_use[i])
            c.inventory = inv
            new = build_plan(c, [h.target_tap_t for h in sub], plan.plan_id, version=0, alpha=alpha, spec_class=spec,
                             parent=plan.version, frozen=frozen, ts=self._ts(), slot0=sub[0].slot)
            self.n_candidates += 1
            cid = f"C{self.n_candidates}"
            echo = dict(exclude_groups=params.get("exclude_groups") or [], group_caps_t=params.get("group_caps_t") or {},
                        alpha=alpha, spec_class=spec, relax_recipe_bounds=bool(params.get("relax_recipe_bounds")))
            if new is None:
                summary = dict(candidate_id=cid, status="infeasible", params=echo,
                               hint="No mix meets the limits, recipe bounds and availability. Try relax_recipe_bounds, a looser spec_class or a higher alpha.")
                self.candidates[cid] = dict(plan=None, summary=summary, caps=caps)
                return None, summary
            full = new.model_copy(update=dict(heats=[h.model_copy(deep=True) for h in plan.heats[:start]] + new.heats))
            cur = self.open_eval(plan) or {}
            open_j = [j for j in range(len(sub)) if j not in frozen]
            steel = sum(sub[j].target_tap_t for j in open_j)
            cost = sum(new.heats[j].cost.total for j in open_j) / steel
            moves = {}
            for j in open_j:
                for g in GROUPS:
                    moves[g] = moves.get(g, 0.0) + new.heats[j].group_t.get(g, 0.0) - sub[j].group_t.get(g, 0.0)
            summary = dict(candidate_id=cid, status="optimal", params=echo, open_slots=[sub[j].slot for j in open_j],
                           cost_per_t=round(cost, 2), current_cost_per_t=cur.get("cost_per_t"),
                           delta_per_t=round(cost - cur["cost_per_t"], 2) if cur.get("cost_per_t") is not None else None,
                           moves=[dict(group=g, delta_t=round(v, 1)) for g, v in moves.items() if abs(v) >= 0.05],
                           tonnes_by_group={g: round(sum(new.heats[j].group_t.get(g, 0.0) for j in open_j), 1) for g in GROUPS},
                           worst_p_exceed={el: round(max(new.heats[j].predicted[el].p_exceed for j in open_j), 3) for el in ELS},
                           limits=c.limits, binding=[b.plain for b in new.binding[:3]])
            self.candidates[cid] = dict(plan=full, summary=summary, caps={g: v for g, v in caps.items()},
                                        base_version=plan.version, spec_class=spec, alpha=alpha)
            return full, summary

    def churn_blocks(self, summary: dict) -> str | None:
        """Anti-churn (A13): refuse a re-plan that saves less than the threshold while the live plan is still within risk."""
        cur = self.open_eval() or {}
        # 0.005 tolerance: the plan's 2-pass heel fixed point leaves the stored mix a hair over alpha (e.g. 0.052)
        risk_ok = all(v <= self.alpha + 0.005 for v in (cur.get("worst_p_exceed") or {}).values())
        d = summary.get("delta_per_t")
        if d is not None and abs(d) < self.mill["anti_churn_usd_per_t"] and risk_ok:
            return (f"Anti-churn: cost change {d:+.2f} $/t is below the {self.mill['anti_churn_usd_per_t']} $/t threshold "
                    "and the live plan is within its risk limit. Do not re-plan.")
        return None

    def publish_candidate(self, cid: str, rationale: dict, trigger_text: str, alternatives: list[str] | None = None,
                          actor: str = "agent", sibling: bool = False) -> PlanVersion:
        """Save a candidate as the next plan version and email it to the supervisor (approval stays human).
        sibling=True: the candidate is an alternative computed alongside the live proposal (same base version)."""
        with self.lock:
            cand = self.candidates.get(cid)
            if not cand or cand["plan"] is None:
                raise ValueError(f"unknown or infeasible candidate {cid}")
            plan = self.base_plan()
            ok_bases = {plan.version, plan.parent_version} if (plan and sibling) else {plan.version if plan else None}
            if plan is None or cand["base_version"] not in ok_bases:
                raise ValueError("stale candidate: the plan changed since it was computed; run the optimiser again")
            for h_new, h_old in zip(cand["plan"].heats, plan.heats):
                if self.progress.get(h_old.slot) in ("tapped", "loading") and h_new.group_t != h_old.group_t:
                    raise ValueError(f"stale candidate: heat {h_old.slot} has started since; run the optimiser again")
            new = cand["plan"].model_copy(deep=True)
            new.version, new.parent_version = self.store.next_version(plan.plan_id), plan.version
            new.created_by, new.created_ts, new.status = ("user" if actor != "agent" else "agent"), self._ts(), "PROPOSED"
            if plan.status not in ("SUPERSEDED", "REJECTED", "DONE"):
                self.store.transition(plan, "SUPERSEDED", actor, self._ts(), f"re-plan: {trigger_text}")
            self.store.save(new, actor, self._ts(), trigger_text)
            self.caps, self.spec_class, self.alpha = dict(cand["caps"]), cand["spec_class"], cand["alpha"]
            self.explanation = dict(rationale)
            self.alternatives[(new.plan_id, new.version)] = [
                dict(self.candidates[a]["summary"], label=chr(66 + k)) for k, a in enumerate(a for a in (alternatives or [])
                                                                              if a != cid and self.candidates.get(a, {}).get("plan") is not None)][:2]
            self._send_email(new, rationale.get("email") or rationale.get("supervisor", ""))
            self.pending_banner = dict(version=new.version, text=rationale.get("crane", "Plan changed: check the new basket list."), ts=self._ts())
            self.tracker = SurpriseTracker()  # cooldown: the estimates just changed, so start a fresh surprise history
            self._emit("replan", {"version": new.version, "trigger": trigger_text, "explanation": rationale},
                       ["process", "email", "supervisor", "shift_leader", "crane"])
            return new

    def notify(self, role: str, text: str, sender: str = "agent"):
        with self.lock:
            self.messages.setdefault(role, []).append(dict(ts=self._ts(), sender=sender, text=text))
            self._emit("message", {"role": role, "text": text}, ["process", role])

    def post_chat(self, role: str, text: str, who: str = "agent"):
        with self.lock:
            self.chat.append(dict(who=who, to=role, text=text, ts=self._ts()))
            self._emit("chat", {"role": role}, ["process", role])

    # ------------------------------------------------------------ agent dispatch (LLM loop with deterministic fallback)
    def dispatch(self, kind: str, text: str, role: str | None = None):
        """Wake the agent. Async in the API (UI stays responsive); sync in tests. Lab/drift events coalesce while a run is pending."""
        ev = dict(kind=kind, text=text, role=role, ts=self._ts())
        if kind in ("lab_surprise", "drift") and self.agent_pending:
            return None
        if self.agent_async:
            self.agent_pending += 1
            self._emit("agent_busy", {"kind": kind}, ["process"])
            self.executor.submit(self._run_agent, ev)
            return dict(queued=True)
        self.agent_pending += 1
        return self._run_agent(ev)

    def _run_agent(self, ev: dict):
        try:
            rec = agent_loop.handle(self, ev, self.llm)
        except Exception as e:  # never let an agent failure take the world down
            import traceback; traceback.print_exc()
            rec = dict(ts=self._ts(), trigger=ev["kind"], trigger_text=ev["text"], path="error", decision="error", steps=[],
                       tools=[], version=None, facts={}, explanation={}, narrator={}, reason=f"{type(e).__name__}: {e}")
            self.decisions.append(rec)
        finally:
            self.agent_pending -= 1
            self._emit("agent_done", {"kind": ev["kind"]}, ["process"])
        return rec

    # ------------------------------------------------------------ user actions
    def approve(self, actor="supervisor", option: str | None = None):
        """Approve the live version, or one of the agent's costed alternatives (published as a new version, then approved)."""
        with self.lock:
            if option:
                alt = next((a for a in self.alternatives.get((self.plan_id, self.current_plan().version), []) if a["candidate_id"] == option), None)
                if alt is None:
                    raise ValueError(f"unknown option {option}")
                text = f"Supervisor chose option {alt['label']} ({alt['cost_per_t']:.2f} $/t)."
                self.publish_candidate(option, dict(email=text, supervisor=text, shift_leader=text, crane="Plan changed: check the new basket list."),
                                       text, actor=actor, sibling=True)
            p = self.current_plan()
            self.store.transition(p, "SUPERVISOR_APPROVED", actor, self._ts(), "approved" + (f" option {option}" if option else ""))
            self._emit("approved", {"version": p.version}, ["process", "supervisor", "shift_leader"])

    def reject(self, reason: str):
        """The agent re-plans with the supervisor's reason (the rejected version stays on record)."""
        with self.lock:
            p = self.current_plan()
            self.store.transition(p, "REJECTED", "supervisor", self._ts(), reason)
            self._emit("rejected", {"version": p.version, "reason": reason}, ["process", "supervisor"])
        return self.dispatch("rejection", reason, "supervisor")

    def report_issue(self, role: str, text: str):
        """Free-text operational issue ("only 40 t of 3A left", "basket 2 out of service"): the agent interprets it."""
        with self.lock:
            self._emit("issue", {"role": role, "text": text}, ["process", "supervisor", "shift_leader"])
        return self.dispatch("issue", text, role)

    def ask(self, role: str, text: str):
        """A question from a role (chat or email reply): the agent answers from tool results only."""
        self.post_chat(role, text, who=role)
        return self.dispatch("question", text, role)

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
            new.version = self.store.next_version(p.plan_id)
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
        return self.report_issue("shift_leader", f"{group} unavailable.")

    # ------------------------------------------------------------ views
    def phone(self, role: str) -> dict:
        with self.lock:
            p = self.current_plan() or (self.store.versions()[-1] if self.store.versions() else None)
            out = dict(role=role, sim_time=self._ts(), mode=self.mode)
            if role == "email":
                out["emails"] = list(reversed(self.emails))
            if p is not None:
                view = project(p, role, self.baseline_per_t() if role in ("email", "supervisor") else None, self.explanation, self.progress,
                               self.risk_match(p) if role in ("email", "supervisor") else None)
                if role == "supervisor":
                    view["alternatives"] = self.alternatives.get((p.plan_id, p.version), [])
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
            out["messages"] = self.messages.get(role, [])[-6:]
            out["chat"] = [c for c in self.chat if c["to"] == role][-8:]
            out["agent_busy"] = self.agent_pending > 0
            return out


def _load(name: str):
    import json
    from scrapchef.data.store import ROOT
    f = ROOT / "data" / name
    return json.loads(f.read_text()) if f.exists() else None
