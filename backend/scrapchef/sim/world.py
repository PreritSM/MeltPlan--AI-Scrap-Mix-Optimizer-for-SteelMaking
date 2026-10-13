"""Simulation world: one clock, one plan/heat state, shared by both UI panes via events.

Modes: SHADOW_REPLAY (real held-out heats; lab results belong to the mill's real charge) and
SIMULATED (scripted drift; lab results generated from OUR plan; clearly labelled)."""
from __future__ import annotations

import copy
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

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
from scrapchef.intelligence.estimator.massbalance import build, sym
from scrapchef.intelligence.optimizer.evaluate import P_TOL, evaluate_mix, risk_matched_baseline
from scrapchef.intelligence.optimizer.model import ELS, make_context, predict_heat
from scrapchef.intelligence.optimizer.plan import build_plan, heat_from_groups
from scrapchef.read.mock.adapters import MockHandling, MockInventory, MockSchedule

log = logging.getLogger(__name__)

ISO = "%Y-%m-%dT%H:%M:%S"


def _with(ctx, **kw):
    c = copy.copy(ctx)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


class World:
    def __init__(self, mode: str = "SHADOW_REPLAY", agent_async: bool = False, llm: LLM | None = None,
                 store_path: str | Path | None = None):
        self.lock = threading.RLock()
        self.store_path = store_path                        # None: in-memory store; only the API passes data/app.sqlite (D36)
        self._lost_plan_logged: str | None = None
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
            self.store = PlanStore(self.store_path)
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
            self.progress: dict[int, str] = {}   # planned -> loading (first grab) -> loaded (all baskets complete) -> tapped
            self.baskets_done: dict[int, set[int]] = {}   # slot -> completed basket numbers, for self.plan_id
            self.plan_heat0 = 0           # replay index of plan slot 1
            self.decisions: list[dict] = []
            self.emails: list[dict] = []
            self.chat: list[dict] = []
            self.inbox: list[dict] = []   # inbound free text the agent reads (READ layer): emails, issues, questions, rejections
            self.heat_log: list[dict] = []
            self.explanation: dict = {}
            self.version_settings: dict[int, dict] = {}   # the planning settings each version of the day was made under
            self.drift_alert: dict | None = None
            self.released_version: int | None = None
            self.pending_banner: dict | None = None
            self.sim_step = 0
            # agent planning state: availability caps / spec / risk level chosen by published re-plans, candidates, outputs
            self.spec_class, self.alpha = cfg("specs")["default"], self.mill["alpha"]
            self.caps: dict[str, float] = {}
            self.live_bounds: dict | None = None   # recipe bounds of the published re-plan (relaxed floors); None = ctx.bounds
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
        mine = [p for p in self.store.versions() if p.plan_id == self.plan_id]
        if not mine and self._lost_plan_logged != self.plan_id:   # once per plan: this is polled on every state read
            self._lost_plan_logged = self.plan_id
            log.error("plan store lost plan %s: was the DB wiped by another process?", self.plan_id)
        vs = [p for p in mine if p.status not in ("SUPERSEDED", "REJECTED", "DROPPED")]
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
        """Historical mix vs optimiser at the historical mix's own predicted risk (cached per plan version).
        plan_worst_p: the plan's own risk on the same element, averaged over heats the same way as worst_p."""
        key = (plan.plan_id, plan.version, tuple(h.slot for h in plan.heats))
        if key not in self.rm_cache:
            X, _ = self._baseline_heats()
            rm = risk_matched_baseline(self.ctx, X.mean(0), [h.target_tap_t for h in plan.heats], dict(self.ctx.heel_c0))
            if rm:
                rm["plan_worst_p"] = float(np.mean([h.predicted[rm["worst_el"]].p_exceed for h in plan.heats]))
            self.rm_cache[key] = rm
        return self.rm_cache[key]

    def _set_estimates(self):
        for el in ELS:
            self.ctx.est.mu[el] = self.kf[el].e.copy()

    def _inventory(self, idx: list[int]) -> np.ndarray:
        """MOCK yard stock per group: 1.3 x (recipe upper-bound share x horizon charge), at least 1.15 x observed usage.
        Stock only binds when a shortage is reported (the real feed will replace this)."""
        horizon_t = len(idx) * self.mill["target_tap_t"] / float(np.mean(self.ctx.yields))
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
            self.live_bounds = None
            self.base_inventory = self._inventory(idx)
            self.ctx.inventory = self.base_inventory.copy()
            self.ctx.limits = cfg("specs")["classes"][self.spec_class]
            self.ctx.heel_c0 = dict(self.heel)
            self.plan_heel0 = dict(self.heel)
            date = self.clock.strftime("%Y-%m-%d")
            for p in self.store.versions():
                if p.plan_id == date and p.status not in ("SUPERSEDED", "REJECTED", "DONE", "DROPPED"):
                    self.store.transition(p, "SUPERSEDED", "agent", self._ts(), "new daily plan")
            prior = [p.version for p in self.store.versions() if p.plan_id == date]
            version = self.store.next_version(date)  # a re-plan of the same day (e.g. after a reject) continues the numbering
            plan = build_plan(self.ctx, taps, date, version=version, parent=max(prior) if prior else None, ts=self._ts())
            if plan is None:
                return None
            self.plan_id, self.plan_heat0 = date, idx[0]
            self._expire_messages(list(self.messages), "new daily plan")
            self.progress, self.baskets_done = {}, {}
            self.released_version, self.pending_banner = None, None
            self.explanation = {}   # agent text belongs to the day's re-plans, never to the next day's plan
            self.store.save(plan, "agent", self._ts(), trigger)
            self.version_settings = {version: self._settings()}
            facts = dict(decision="new_plan", version=version, trigger_text="Daily 06:00 plan.", n_heats=len(plan.heats), moves=[],
                         cost_delta_per_t=0.0, new_per_t=sum(h.cost.total for h in plan.heats) / sum(h.steel_t for h in plan.heats), reason="", binding=(plan.binding[0].plain if plan.binding else ""))
            rationale = "Lowest-cost mix meeting every element limit with 95% confidence, within the mill's demonstrated recipe range."
            self._send_email(plan, rationale)
            self.decisions.append(dict(ts=self._ts(), trigger=trigger, tools=["get_state", "run_optimizer", "publish_plan"], decision="plan",
                                       version=version, facts=facts, explanation={"email": rationale}, narrator={"email": "template"}))
            self._emit("plan_created", {"plan_id": plan.plan_id, "version": version}, ["process", "email", "supervisor"])
            return plan

    def _remaining(self, plan: PlanVersion) -> PlanVersion:
        """The plan as it still matters: tapped heats are history, so views, emails and costs cover only the heats still to make
        (the whole plan once every heat is tapped)."""
        left = [h for h in plan.heats if self.progress.get(h.slot) != "tapped"]
        return plan.model_copy(update=dict(heats=left)) if left and len(left) < len(plan.heats) else plan

    def _tapped(self, plan: PlanVersion) -> list[int]:
        return [h.slot for h in plan.heats if self.progress.get(h.slot) == "tapped"]

    def _send_email(self, plan: PlanVersion, rationale: str):
        left = self._remaining(plan)
        mail = email_mod.render(left, self.clock.strftime("%Y-%m-%d"), self.baseline_per_t(), rationale, self.risk_match(left),
                                self._tapped(plan) if left is not plan else None)
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
        rel = next((v for v in self.store.versions() if v.plan_id == plan.plan_id and v.version == self.released_version), None)
        h = (rel or plan).heats[slot - 1]   # the furnace is charged from the crane's (released) list, not a version awaiting approval
        x = np.array([h.group_t.get(g, 0.0) for g in GROUPS])
        B = h.steel_t + self.mill["heel_t"]
        meas = {}
        for el in ELS:
            noise = self.rng.normal(0, self.ctx.est.sigma_eps[el])
            meas[el] = max(0.0, (self.mill["heel_t"] * self.heel[el] + self.e_true[el] @ x + noise) / B)
        self.clock += timedelta(minutes=60)
        res = self._reveal(x, meas, B, f"SIM-{slot}", actual_tap=h.steel_t)
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
                trig.append(f"{sym(el)} lab result {meas[el]:.3f}% is above the {lim}% limit")
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

    def plan_finished(self) -> bool:
        """Every heat of the live plan is tapped: the next step belongs to the next 06:00 plan."""
        p = self.current_plan()
        return bool(p) and all(self.progress.get(h.slot) == "tapped" for h in p.heats)

    def _after_heat(self, rec: dict):
        self._emit("heat_tapped", {"heatid": rec["heatid"], "slot": rec["slot"], "elements": rec["elements"]}, ["process", "supervisor", "shift_leader"])
        p = self.current_plan()
        if p and p.status in ("RELEASED", "LOADING") and self.plan_finished():
            self.store.transition(p, "DONE", "system", self._ts(), "all heats tapped")
            self._emit("plan_done", {"version": p.version}, ["process", "supervisor", "shift_leader", "crane"])
        if rec["drift"]:
            self.drift_alert = dict(ts=rec["ts"], groups=rec["drift"],
                                    text="Composition drift: " + "; ".join(f"{sym(e)} in {', '.join(g)}" for e, g in rec["drift"].items()))
            self._emit("drift", self.drift_alert, ["process", "supervisor", "shift_leader"])
        if (rec["triggers"] or rec["drift"]) and (self.open_eval() or {}).get("open_slots"):  # nothing to re-plan once all heats are tapped
            why = (rec["triggers"] or [self.drift_alert["text"]])[0]
            self.dispatch("lab_surprise" if rec["triggers"] else "drift", why + ".")

    # ------------------------------------------------------------ planning services (called by the agent's tools and its fallback)
    def base_plan(self) -> PlanVersion | None:
        """The plan a re-plan starts from: the live version, or the latest one if it was just rejected (None once dropped)."""
        p = self.current_plan()
        if p is None and self.plan_id:
            vs = [v for v in self.store.versions() if v.plan_id == self.plan_id]
            p = vs[-1] if vs and vs[-1].status != "DROPPED" else None
        return p

    def _split(self, plan: PlanVersion) -> tuple[int, dict[int, np.ndarray]]:
        """(start, frozen): solve from the first heat after the last tapped one; loading / loaded / user-frozen heats keep their mix."""
        tapped = [i for i, h in enumerate(plan.heats) if self.progress.get(h.slot) == "tapped"]
        start = max(tapped) + 1 if tapped else 0
        frozen = {j: np.array([h.group_t.get(g, 0.0) for g in GROUPS]) for j, h in enumerate(plan.heats[start:])
                  if self.progress.get(h.slot) in ("loading", "loaded") or h.frozen}
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
            cost += ev["cost"]["total"]; steel += ev["tap_actual_t"]
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
            relax_floor = {g: round(lo, 3) for g, (lo, _) in self.ctx.bounds.items() if lo > 0} if params.get("relax_recipe_bounds") else {}
            solve = lambda inventory: build_plan(_with(c, inventory=inventory), [h.target_tap_t for h in sub], plan.plan_id, version=0, alpha=alpha,
                                                 spec_class=spec, parent=plan.version, frozen=frozen, ts=self._ts(), slot0=sub[0].slot)
            new = solve(inv)
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
            steel = sum(new.heats[j].steel_t for j in open_j)
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
            if relax_floor and caps and summary["delta_per_t"] is not None:
                # split the cost change: how much comes from dropping the recipe floors alone (same settings, no new caps),
                # and how much from the availability limits; otherwise a shortage can look like a saving
                ref = solve(np.maximum(self.base_inventory - used, 0.0))
                if ref is not None:
                    ref_cost = sum(ref.heats[j].cost.total for j in open_j) / sum(ref.heats[j].steel_t for j in open_j)
                    summary["delta_breakdown"] = dict(dropping_recipe_floors_per_t=round(ref_cost - cur["cost_per_t"], 2),
                                                      availability_limits_per_t=round(cost - ref_cost, 2))
            if relax_floor:
                summary["recipe_floors_dropped"] = relax_floor
            self.candidates[cid] = dict(plan=full, summary=summary, caps={g: v for g, v in caps.items()},
                                        base_version=plan.version, spec_class=spec, alpha=alpha, bounds=dict(c.bounds))
            return full, summary

    def churn_blocks(self, summary: dict) -> str | None:
        """Anti-churn (A13): refuse a re-plan that saves less than the threshold while the live plan is still within risk."""
        cur = self.open_eval() or {}
        # 0.005 tolerance: solver and heel fixed-point rounding can leave the stored mix a hair over alpha
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
                if self.progress.get(h_old.slot) in ("tapped", "loading", "loaded") and h_new.group_t != h_old.group_t:
                    raise ValueError(f"stale candidate: heat {h_old.slot} has started since; run the optimiser again")
            new = cand["plan"].model_copy(deep=True)
            new.version, new.parent_version = self.store.next_version(plan.plan_id), plan.version
            new.created_by, new.created_ts, new.status = ("user" if actor != "agent" else "agent"), self._ts(), "PROPOSED"
            if plan.status not in ("SUPERSEDED", "REJECTED", "DONE"):
                self.store.transition(plan, "SUPERSEDED", actor, self._ts(), f"re-plan: {trigger_text}")
            self.store.save(new, actor, self._ts(), trigger_text)
            self.caps, self.spec_class, self.alpha = dict(cand["caps"]), cand["spec_class"], cand["alpha"]
            self.live_bounds = cand.get("bounds")   # the Control Suite's recipe check follows a relaxed re-plan
            self.ctx.limits = cfg("specs")["classes"][self.spec_class]   # Control Suite, open-heat risk and the history line follow the live spec
            self.version_settings[new.version] = self._settings()
            self.explanation = dict(rationale)
            kept, costs = [], [cand["summary"]["cost_per_t"]]
            for a in alternatives or []:   # an option at the same cost is not a real alternative (same rule as the fallback)
                c = self.candidates.get(a, {})
                if a != cid and c.get("plan") is not None and all(abs(c["summary"]["cost_per_t"] - x) > 0.05 for x in costs):
                    kept.append(a); costs.append(c["summary"]["cost_per_t"])
            self.alternatives[(new.plan_id, new.version)] = [dict(self.candidates[a]["summary"], label=chr(66 + k)) for k, a in enumerate(kept[:2])]
            self._send_email(new, rationale.get("email") or rationale.get("supervisor", ""))
            self.pending_banner = dict(version=new.version, text=rationale.get("crane", "Plan changed: check the new basket list."), ts=self._ts())
            self.tracker = SurpriseTracker()  # cooldown: the estimates just changed, so start a fresh surprise history
            self._emit("replan", {"version": new.version, "trigger": trigger_text, "explanation": rationale},
                       ["process", "email", "supervisor", "shift_leader", "crane"])
            return new

    def drop_plan(self, reason: str, actor: str = "agent") -> PlanVersion | None:
        """No feasible plan: tell the supervisor why, then withdraw the plan completely. There is no live version afterwards,
        the crane has no list to load and the agent stops re-planning; the next plan comes from the 06:00 cycle.
        A rejected change with an approved version before it is not dropped: the change is withdrawn and that version is live again."""
        with self.lock:
            p = self.base_plan()
            if p is None or p.status in ("DONE", "SUPERSEDED", "DROPPED"):
                return None
            prev = self._last_approved(p.version) if p.status == "REJECTED" else None
            if prev is not None:
                self._reinstate(p, prev, reason)
                return p
            self.notify("supervisor", f"{reason} Plan v{p.version} is dropped: no plan is live until a new one is made.")
            self.store.transition(p, "DROPPED", actor, self._ts(), reason)
            stop = f"Plan v{p.version} is dropped: no feasible charge mix. Do not load heats that are not loaded yet; wait for the supervisor."
            self._expire_messages(["shift_leader", "crane"], f"v{p.version} dropped")
            self.released_version, self.pending_banner = None, dict(version=p.version, text=stop, ts=self._ts())
            self.notify("shift_leader", stop)
            self._emit("plan_dropped", {"version": p.version, "reason": reason}, ["process", "email", "supervisor", "shift_leader", "crane"])
            return p

    def _settings(self) -> dict:
        return dict(caps=dict(self.caps), spec_class=self.spec_class, alpha=self.alpha, live_bounds=self.live_bounds)

    def _last_approved(self, before: int) -> int | None:
        """The newest version of the day before `before` that the supervisor approved (it may have been released since)."""
        vs = [a["version"] for a in self.store.audit(self.plan_id)
              if a["action"] in ("SUPERVISOR_APPROVED", "RELEASED", "LOADING") and a["version"] < before]
        return max(vs) if vs else None

    def _reinstate(self, rejected: PlanVersion, prev: int, reason: str):
        """Rejecting a change keeps the plan the supervisor had approved: approved version `prev` comes back as a new version
        (stored versions are never rewritten), with the settings it was made under, released again if it was the crane's list."""
        old = self.store.get(self.plan_id, prev)
        new = old.model_copy(deep=True)
        released = self.released_version == prev
        new.version, new.parent_version = self.store.next_version(old.plan_id), prev
        new.created_by, new.created_ts, new.status = "user", self._ts(), "RELEASED" if released else "SUPERVISOR_APPROVED"
        self.store.save(new, "supervisor", self._ts(), f"v{rejected.version} rejected: approved v{prev} reinstated")
        if released:
            self.released_version = new.version
        s = self.version_settings.get(prev) or self._settings()
        self.caps, self.spec_class, self.alpha, self.live_bounds = dict(s["caps"]), s["spec_class"], s["alpha"], s["live_bounds"]
        self.ctx.limits = cfg("specs")["classes"][self.spec_class]
        self.version_settings[new.version] = self._settings()
        sup = (f"{reason} The rejected change is withdrawn: plan v{new.version}, the same mix as approved v{prev}, is live again"
               + (" and released to the cranes." if released else "; it still needs release."))
        ops = (f"The change in v{rejected.version} was rejected. " + (f"Keep loading from v{new.version}: the same list as v{prev}."
               if released else f"v{new.version}, the same plan as approved v{prev}, waits for release."))
        self.explanation = dict(email=sup, supervisor=sup, shift_leader=ops)
        self.pending_banner = None
        self._expire_messages(["shift_leader", "crane"], f"v{rejected.version} rejected")
        self.notify("crane", ops)
        self._emit("plan_reinstated", {"version": new.version, "from": prev, "rejected": rejected.version},
                   ["process", "email", "supervisor", "shift_leader", "crane"])

    def notify(self, role: str, text: str, sender: str = "agent"):
        with self.lock:
            self.messages.setdefault(role, []).append(dict(ts=self._ts(), sender=sender, text=text))
            self._emit("message", {"role": role, "text": text, "sender": sender}, ["process", role])

    def _expire_messages(self, roles, why: str):
        """Agent notes are about a pending situation ('hold loading until the new list arrives'); once a plan is released or a new
        day starts they are out of date. They stay in the log (Process View) but leave the phone screens."""
        for r in roles:
            for m in self.messages.get(r, []):
                if not m.get("expired"):
                    m["expired"] = why

    def post_chat(self, role: str, text: str, who: str = "agent"):
        with self.lock:
            self.chat.append(dict(who=who, to=role, text=text, ts=self._ts()))
            self._emit("chat", {"role": role, "who": who, "text": text}, ["process", role])

    # ------------------------------------------------------------ agent dispatch (LLM loop with deterministic fallback)
    def dispatch(self, kind: str, text: str, role: str | None = None):
        """Wake the agent. Async in the API (UI stays responsive); sync in tests. Lab/drift events coalesce while a run is pending."""
        ev = dict(kind=kind, text=text, role=role, ts=self._ts())
        if kind in ("inbox", "issue", "question", "rejection"):
            self.inbox.append(ev)
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
            self._expire_messages(["crane"], f"v{p.version} released")
            self._emit("released", {"version": p.version}, ["process", "shift_leader", "crane"])

    # ------------------------------------------------------------ Control Suite (hand edits)
    def _heel_for(self, plan: PlanVersion, slot: int) -> dict[str, float]:
        """Heel chemistry for heat `slot`: the measured result if heat slot-1 is the last tapped one, the plan's start heel for
        heat 1, else heat slot-1's predicted mean in `plan`."""
        tapped = [h.slot for h in plan.heats if self.progress.get(h.slot) == "tapped"]
        if tapped and slot - 1 == max(tapped):
            return dict(self.heel)
        if slot == 1:
            return dict(self.plan_heel0)
        return {el: plan.heats[slot - 2].predicted[el].mean for el in ELS}

    def _edit_lock(self, h) -> str | None:
        st = "frozen" if h.frozen else self.progress.get(h.slot)
        return f"heat {h.slot} is {st}: it can no longer be edited" if st in ("tapped", "loading", "loaded", "frozen") else None

    def _edit_checks(self, plan: PlanVersion, slot: int, group_t: dict[str, float]) -> tuple[list[str], list[str]]:
        """(rule_violations, availability_violations). Recipe share bounds only warn (deliberate departures are allowed);
        shortage caps and yard stock block. Caps cover the open, non-frozen heats and stock the whole day, as in solve_candidate."""
        x = np.array([group_t.get(g, 0.0) for g in GROUPS])
        tot = x.sum()
        rules = []
        for i, g in enumerate(GROUPS):
            lo, hi = (self.live_bounds or self.ctx.bounds)[g]
            s = x[i] / tot if tot > 0 else 0.0
            if s < lo - 1e-3 or s > hi + 1e-3:
                rules.append(f"{g} is {s:.0%} of the charge, outside the usual {lo:.0%}–{hi:.0%} recipe range")
        mix = lambda h: x if h.slot == slot else np.array([h.group_t.get(g, 0.0) for g in GROUPS])
        start, frozen = self._split(plan)
        open_t = sum((mix(h) for j, h in enumerate(plan.heats[start:]) if j not in frozen), np.zeros(len(GROUPS)))
        day_t = sum((mix(h) for h in plan.heats), np.zeros(len(GROUPS)))
        avail = [f"{g}: the open heats would use {open_t[GROUPS.index(g)]:.1f} t, above the {cap:.1f} t available (reported shortage)"
                 for g, cap in self.caps.items() if open_t[GROUPS.index(g)] > cap + 0.05]
        avail += [f"{g}: the day's heats would use {day_t[i]:.1f} t, above the {self.base_inventory[i]:.1f} t in the yard (MOCK stock)"
                  for i, g in enumerate(GROUPS) if day_t[i] > self.base_inventory[i] + 0.05]
        return rules, avail

    def _downstream(self, plan: PlanVersion, slot: int, pred: dict) -> list[dict]:
        """Later untapped heats re-predicted from their unchanged mixes with the chained heel (no solve)."""
        out, heel = [], {el: pred[el]["mean"] for el in ELS}
        for h in plan.heats[slot:]:
            if self.progress.get(h.slot) == "tapped":
                continue
            pr = predict_heat(self.ctx, np.array([h.group_t.get(g, 0.0) for g in GROUPS]), None, heel)
            el = max(pr, key=lambda e: pr[e]["p_exceed"])
            out.append(dict(slot=h.slot, worst_el=el, p_exceed=pr[el]["p_exceed"], was_p_exceed=h.predicted[el].p_exceed,
                            breach=pr[el]["p_exceed"] > plan.alpha + P_TOL))
            heel = {e: pr[e]["mean"] for e in ELS}
        return out

    def edit(self, slot: int, group_t: dict[str, float], actor="supervisor") -> PlanVersion:
        with self.lock:
            p = self.current_plan()
            h = p.heats[slot - 1]
            lock = self._edit_lock(h)
            if lock:
                raise ValueError(lock)
            if sum(group_t.values()) > self.mill["capacity_t"]:
                raise ValueError(f"charge {sum(group_t.values()):.1f} t is above the {self.mill['capacity_t']:.0f} t furnace capacity")
            _, avail = self._edit_checks(p, slot, group_t)
            if avail:
                raise ValueError("; ".join(avail))
            self._set_estimates()
            new = p.model_copy(deep=True)
            new.version = self.store.next_version(p.plan_id)
            new.parent_version, new.created_by, new.status, new.created_ts = p.version, "user", "PROPOSED", self._ts()
            new.heats[slot - 1] = heat_from_groups(self.ctx, slot, h.target_tap_t, group_t, self._heel_for(p, slot), h.spec_class)
            for j, o in enumerate(new.heats[slot:], start=slot):   # later heats: same mix and baskets, re-predicted from the new heel
                if self.progress.get(o.slot) != "tapped":
                    rp = heat_from_groups(self.ctx, o.slot, o.target_tap_t, o.group_t, self._heel_for(new, o.slot), o.spec_class, o.frozen)
                    new.heats[j] = o.model_copy(update=dict(predicted=rp.predicted))
            # solver duals do not hold for a hand-edited mix or for heats whose heel it changed; horizon (stock) duals span them too
            new.binding = [c for c in p.binding if c.name.rpartition("@")[2].isdigit() and int(c.name.rpartition("@")[2]) < slot]
            new.hand_edited = sorted(set(p.hand_edited) | {slot})
            self.store.transition(p, "SUPERSEDED", actor, self._ts(), f"manual edit of heat {slot}")
            self.store.save(new, actor, self._ts(), f"manual edit heat {slot}")
            self.version_settings[new.version] = self.version_settings.get(p.version) or self._settings()
            self.store.transition(new, "EMAILED", actor, self._ts(), "edit saved")
            self._emit("edited", {"version": new.version, "slot": slot}, ["process", "supervisor", "shift_leader"])
            return new

    def evaluate(self, slot: int, group_t: dict[str, float]) -> dict:
        with self.lock:
            p = self.current_plan()
            self._set_estimates()
            h = p.heats[slot - 1]
            ev = evaluate_mix(self.ctx, group_t, h.target_tap_t, self._heel_for(p, slot), alpha=p.alpha,
                              tap_tol_t=self.mill["edit_tap_tol_t"], capacity_t=self.mill["capacity_t"])
            ev["delta_cost_per_t"] = ev["cost"]["per_t"] - h.cost.per_t
            ev["rule_violations"], ev["availability_violations"] = self._edit_checks(p, slot, group_t)
            ev["locked"] = self._edit_lock(h)   # still evaluated, read-only
            ev["downstream"] = self._downstream(p, slot, ev["predicted"])
            return ev

    def pending_change(self) -> tuple[PlanVersion | None, set[int]]:
        """A newer version than the released one (awaiting approval or release) and the heats whose mix it changes. The crane
        sees those heats' new lists but may not load them until the new version is released."""
        p, rv = self.current_plan(), self.released_version
        if p is None or rv is None or p.version == rv:
            return None, set()
        rel = next((v for v in self.store.versions() if v.plan_id == p.plan_id and v.version == rv), None)
        if rel is None:
            return None, set()
        old = {h.slot: h for h in rel.heats}
        return p, {h.slot for h in p.heats if h.slot in old and h.group_t != old[h.slot].group_t
                   and self.progress.get(h.slot) not in ("tapped", "loading", "loaded")}

    def crane_hold(self) -> str | None:
        """Why the crane may not move at all: a newer version than the released one is not approved yet (awaiting the supervisor,
        or rejected and being re-planned). The crane resumes once the supervisor approves."""
        rv = self.released_version
        vs = [v for v in self.store.versions() if v.plan_id == self.plan_id] if rv is not None else []
        latest = vs[-1] if vs else None
        if latest is None or latest.version == rv:
            return None
        if latest.status in ("PROPOSED", "EMAILED"):
            return f"Plan v{latest.version} is waiting for supervisor approval: stop loading until it is approved."
        if latest.status == "REJECTED":
            return f"Plan v{latest.version} was rejected and a new plan is being made: stop loading until one is approved."
        return None

    def sim_hold(self) -> str | None:
        """Why the furnace may not tap the next heat yet: the crane may not load it (a newer version awaits approval), or the
        heat's mix changes in a version that is not released. The sim clock waits for the people, as it waits for the agent."""
        with self.lock:
            hold = self.crane_hold()
            if hold:
                return hold
            p, changed = self.pending_change()
            nxt = self.sim_step + 1 if self.mode == "SIMULATED" else (self.cursor - self.plan_heat0 + 1 if self.plan_id else 0)
            if nxt in changed:
                return f"Heat {nxt} changes in v{p.version}, which is not released yet: release it to tap the heat."
            return None

    def _check_not_pending(self, slot: int):
        hold = self.crane_hold()
        if hold:
            raise ValueError(hold)
        p, changed = self.pending_change()
        if slot in changed:
            raise ValueError(f"heat {slot} changes in v{p.version}, which is not released yet: wait for the new list")

    def grab_done(self, slot: int, basket: int, mat_code: int, t: float):
        """Idempotent per (plan, slot, basket, code): a repeat returns the recorded event and emits nothing."""
        with self.lock:
            e = self.handling.find(self.plan_id, slot, basket, mat_code)
            if e:
                return e
            self._check_not_pending(slot)
            e = self.handling.record(self._ts(), "crane-1", mat_code, t, slot, basket, self.plan_id)
            p = self.current_plan()
            if p and p.status == "RELEASED":
                self.store.transition(p, "LOADING", "crane", self._ts(), "first grab")
            self.progress.setdefault(slot, "planned")
            if self.progress[slot] == "planned":
                self.progress[slot] = "loading"
            self._emit("grab_done", e.model_dump(), ["process", "shift_leader", "crane"])
            return e

    def basket_complete(self, slot: int, basket: int) -> dict:
        """Every line of the basket is in: lines not yet confirmed are recorded at planned tonnes (the log stays append-only).
        All baskets complete -> the heat is "loaded" (frozen, waiting for the furnace; Step heat taps it)."""
        with self.lock:
            rv = self.released_version
            rel = next((v for v in self.store.versions() if v.plan_id == self.plan_id and v.version == rv), None) if rv else None
            if rel is None:
                raise ValueError("no released plan: the crane has no list to load from")
            h = next((x for x in rel.heats if x.slot == slot), None)
            b = h and next((x for x in h.baskets if x.basket_no == basket), None)
            if b is None:
                raise ValueError(f"heat {slot} basket {basket} is not on the released list")
            if self.progress.get(slot) == "tapped":
                raise ValueError(f"heat {slot} is already tapped")
            self._check_not_pending(slot)
            done = self.baskets_done.setdefault(slot, set())
            state = lambda n: dict(slot=slot, basket=basket, auto_grabbed=n, baskets_complete=sorted(done), progress=self.progress.get(slot, "planned"))
            if basket in done:
                return state(0)
            n = 0
            for it in b.items:
                if not self.handling.find(self.plan_id, slot, basket, it.mat_code):
                    self.grab_done(slot, basket, it.mat_code, it.tonnes)
                    n += 1
            done.add(basket)
            self._emit("basket_complete", dict(slot=slot, basket=basket, auto_grabbed=n), ["process", "shift_leader", "crane"])
            if done >= {x.basket_no for x in h.baskets}:
                self.progress[slot] = "loaded"
                self._emit("heat_loaded", dict(slot=slot), ["process", "supervisor", "shift_leader", "crane"])
            return state(n)

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
                left = self._remaining(p)
                view = project(left, role, self.baseline_per_t() if role in ("email", "supervisor") else None, self.explanation, self.progress,
                               self.risk_match(left) if role in ("email", "supervisor") else None)
                view["tapped"] = self._tapped(p) if left is not p else []
                if role == "supervisor":
                    view["alternatives"] = self.alternatives.get((p.plan_id, p.version), [])
                    view["next_version"], view["released_version"] = self.store.next_version(p.plan_id), self.released_version
                if role == "crane":
                    rv = self.released_version
                    rel = next((v for v in self.store.versions() if v.plan_id == p.plan_id and v.version == rv), None) if rv else None
                    view = project(self._remaining(rel), "crane", None, self.explanation, self.progress) if rel else dict(status="NO RELEASED PLAN", loading=[])
                    pend, changed = self.pending_change()
                    if changed:   # show the new lists for changed heats, locked until that version is released
                        new = {hh["slot"]: hh for hh in project(self._remaining(pend), "crane", None, self.explanation, self.progress)["loading"]}
                        view["loading"] = [dict(new[hh["slot"]], pending=dict(version=pend.version, status=pend.status))
                                           if hh["slot"] in changed else hh for hh in view["loading"]]
                    for hh in view["loading"]:
                        hh["grabbed"] = [f"{e.basket_no}-{e.mat_code}" for e in self.handling.log if e.plan_id == rel.plan_id and e.heat_slot == hh["slot"]]
                        hh["baskets_complete"] = sorted(self.baskets_done.get(hh["slot"], ()))
                    view["banner"], view["hold"] = self.pending_banner, self.crane_hold()
                out["plan"] = view
            if role == "supervisor":
                out["drift_alert"] = self.drift_alert
                out["audit"] = self.store.audit(self.plan_id)
                out["backtest"] = _load("backtest.json")
            if role == "shift_leader":
                out["drift_alert"] = self.drift_alert and dict(text=self.drift_alert["text"])
                out["heat_log"] = [dict(heatid=h["heatid"], slot=h["slot"], cu=h["elements"]["cu"]) for h in self.heat_log[-12:]]
                out["changes"] = [dict(version=v.version, by=v.created_by, ts=v.created_ts, status=v.status) for v in self.store.versions() if v.plan_id == self.plan_id]
            out["messages"] = [m for m in self.messages.get(role, []) if not m.get("expired")][-6:]
            out["chat"] = [c for c in self.chat if c["to"] == role][-8:]
            out["agent_busy"] = self.agent_pending > 0
            return out


def _load(name: str):
    import json
    from scrapchef.data.store import ROOT
    f = ROOT / "data" / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
