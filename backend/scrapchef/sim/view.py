"""Process View state: four lanes of stage cards with state, inputs, outputs, provenance and inspector content."""
from __future__ import annotations

import json

from scrapchef.data.store import GROUPS, ROOT, connect, cfg


def _j(name):
    f = ROOT / "data" / name
    return json.loads(f.read_text()) if f.exists() else None


def card(id, lane, title, prov, state, inputs, outputs, inspector=None):
    return dict(id=id, lane=lane, title=title, provenance=prov, state=state, inputs=inputs, outputs=outputs, inspector=inspector or {})


def process_state(w) -> dict:
    con = connect()
    cnt = lambda t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
    est, energy, bt = _j("estimates.json"), _j("energy.json"), _j("backtest.json")
    plan = w.current_plan()
    last = w.heat_log[-1] if w.heat_log else None
    sel = last["heatid"] if last else None
    # NaN is not valid JSON; null DB values must reach the response as None
    rows = lambda sql, *a: (lambda d: d.astype(object).where(d.notna(), None))(con.execute(sql, list(a)).df()).to_dict("records")
    real_rows = {}
    if sel and not str(sel).startswith("SIM"):
        real_rows = dict(charges=rows("SELECT basket_no, ts, mat_code, mat_name, t FROM heat_charge_basket WHERE heatid=? ORDER BY ts", sel),
                         lab=rows("SELECT * FROM heat_chem WHERE heatid=?", sel), energy=rows("SELECT * FROM heat_energy WHERE heatid=?", sel))
    mock = lambda t, what: card(t, "read", what, "MOCK", "idle", {}, {"note": "Placeholder: awaiting real feed"}, {"contract": t, "adapter": "mock"})
    dq = rows("SELECT rule, count(*) AS n FROM dq_issues GROUP BY 1 ORDER BY 2 DESC")
    read = [
        card("mes", "read", "MES / Level-2 heat records", "REAL", "fresh", dict(rows=cnt("stg_charge"), heats=cnt("heat_fact")),
             dict(last_heat=sel), dict(adapter="KaggleMES (REAL)", selected_heat=sel, **real_rows)),
        card("lab", "read", "Lab chemistry", "REAL", "fresh", dict(rows=cnt("heat_chem")), dict(last=last and last["elements"]["cu"]["measured"]),
             dict(rows=real_rows.get("lab"))),
        card("energy_in", "read", "Energy / transformer", "REAL", "fresh", dict(rows=cnt("heat_energy")), {}, dict(rows=real_rows.get("energy"))),
        mock("inventory", "Inventory"), mock("prices", "Prices & deliveries"), mock("schedule", "Production schedule"),
        card("handling", "read", "Camera handling events", "MOCK", "fresh" if w.handling.log else "idle", dict(events=len(w.handling.log)), {},
             dict(events=[e.model_dump() for e in w.handling.log[-20:]])),
        card("inbox", "read", "Email / chat inbox", "MOCK", "fresh" if w.chat else "idle", dict(messages=len(w.chat)), {}, dict(chat=w.chat[-10:])),
    ]
    gm = [dict(code=r[0], group=r[1]) for r in con.execute("SELECT mat_code, grade_group FROM grade_map ORDER BY 2, 1").fetchall()]
    data = [
        card("ingest", "data", "Ingestion & cleaning", "DERIVED", "fresh", dict(raw_rows=cnt("stg_charge")), dict(dq_issues=sum(r["n"] for r in dq)), dict(dq=dq)),
        card("material", "data", "Material master + grade groups", "ASSUMED", "fresh", dict(codes=len(gm)), dict(groups=len(GROUPS)), dict(mapping=gm, note="Auto-proposed from names/mass share; pending review (Q3)")),
        card("features", "data", "Heat feature table", "DERIVED", "fresh", dict(heats=cnt("heat_fact")), dict(cohort=cnt("cohort")),
             dict(row=(rows("SELECT * FROM heat_fact WHERE heatid=?", sel) if sel and not str(sel).startswith("SIM") else None))),
        card("dq", "data", "Data quality", "DERIVED", "fresh", dict(), dict(chain_breaks=sum(r["n"] for r in dq if r["rule"] == "chain_start")), dict(dq=dq)),
    ]
    cu = est["cu"] if est else None
    est_card = card("estimator", "intelligence", "Composition estimator", "ESTIMATED", "running" if w.drift_alert else "fresh",
                    dict(cohort_heats=cu and cu["n_heats"]), dict(cu_mae_bayes=cu and cu["mae"]["Bayes-NNLS"], cu_mae_prior=cu and cu["mae"]["prior-only"]),
                    dict(estimates={el: dict(groups=GROUPS, mean=est[el]["final"]["mean"], lo=est[el]["final"]["lo"], hi=est[el]["final"]["hi"],
                                             prior=est[el]["prior"]["mu"]) for el in est} if est else None,
                         metrics={el: dict(mae=est[el]["mae"], coverage=est[el]["pi90_coverage"]) for el in est} if est else None,
                         online_kalman={el: [round(float(x), 4) for x in w.kf[el].e] for el in w.kf}, drift=w.drift_alert, last_heat=last,
                         caveat="Bootstrap SDs understate error; one mill code is ~85% of mass, so minor groups are weakly identified."))
    opt_card = card("optimizer", "intelligence", "Chance-constrained optimiser", "ESTIMATED", "fresh" if plan else "idle",
                    dict(mode="chance", alpha=0.05), dict(version=plan and plan.version, per_t=plan and sum(h.cost.total for h in plan.heats) / sum(h.steel_t for h in plan.heats)),
                    dict(binding=plan and [c.model_dump() for c in plan.binding], limits=cfg("specs")["classes"][cfg("specs")["default"]],
                         spec_class=cfg("specs")["default"], note="Limits are ASSUMED (A3)"))
    en_card = card("energy", "intelligence", "Energy & cost model", "ESTIMATED", "fresh", dict(n=energy and energy["n"]),
                   dict(mwh_per_t=energy and energy["mwh_per_t"][0], group_effects=energy and energy["use_group_effects"]), dict(models=energy and energy["models"], note="Tariff ASSUMED; MWh = Σ MW (D13)"))
    bt_card = card("backtest", "intelligence", "Back-test", "ESTIMATED", "fresh" if bt else "idle", dict(heats=bt and bt["n_heats"]),
                   dict(saving_pct=bt and bt["saving_pct"], ci=bt and bt["ci"]), dict(result=bt, note="Depends on ASSUMED prices"))
    runs = [d for d in w.decisions if d.get("path")]   # agent runs (the 06:00 plan is deterministic and has no path)
    ag_card = card("agent", "intelligence", "Re-planning agent", "ESTIMATED", "running" if w.agent_pending else ("fresh" if runs else "idle"),
                   dict(runs=len(runs), llm=w.llm.model if w.llm.available else "off"),
                   dict(replans=sum(1 for d in runs if d["decision"] == "replan"), no_change=sum(1 for d in runs if d["decision"] == "no_change"),
                        last_path=runs[-1]["path"] if runs else None),
                   dict(trace=w.decisions[-15:], llm=w.llm.describe()))
    intel = [est_card, en_card, opt_card, bt_card, ag_card]
    st = plan.status if plan else None
    chain = [("email", "Daily email plan", st in ("EMAILED", "SUPERVISOR_APPROVED", "RELEASED", "LOADING", "DONE")),
             ("supervisor", "Supervisor check", st in ("SUPERVISOR_APPROVED", "RELEASED", "LOADING", "DONE")),
             ("shift_leader", "Shift leaders", st in ("RELEASED", "LOADING", "DONE")),
             ("crane", "Crane tablet", st in ("LOADING", "DONE") or bool(w.handling.log))]
    act = [card("act_" + k, "act", t, "DERIVED", "fresh" if ok else "idle", dict(plan_version=plan and plan.version),
                dict(status=plan and plan.status), dict(audit=w.store.audit(w.plan_id), plan_status=plan and plan.status)) for k, t, ok in chain]
    return dict(clock=w._ts(), mode=w.mode, plan_status=st, plan_version=plan and plan.version, cursor=w.cursor, n_replay=len(w.replay),
                lanes=dict(read=read, data=data, intelligence=intel, act=act), selected_heat=sel,
                heat_strip=[dict(heatid=h["heatid"], z=h["elements"]["cu"]["z"], slot=h["slot"]) for h in w.heat_log[-60:]],
                last_event=w.store.last_seq(), drift_alert=w.drift_alert,
                agent=dict(busy=w.agent_pending > 0, llm=w.llm.describe(), available=w.llm.available),
                legend=["REAL", "DERIVED", "ESTIMATED", "ASSUMED", "MOCK"], groups=GROUPS)
