import { useEffect, useMemo, useRef, useState } from "react";
import { useSim } from "../store/sim";
import { post } from "../api/client";
import { Headroom } from "../components/Headroom";
import { AgentMessages, ChatThread } from "../components/Agent";
import { sym } from "../format";

const money = (v: number) => `$${v.toFixed(2)}`;

/** Same wording as the email's bottom line (act/email.py _bottom_line). */
function bottomLine(perT: number, rm: any) {
  const net = perT - rm.hist_per_t, el = sym(rm.worst_el);
  const a = Math.round(rm.worst_p * 100), b = Math.round(rm.plan_worst_p * 100);
  const pay = net > 0 ? "You pay a little more per tonne" : "You save money per tonne";
  const risk = b < a ? `cut the ${el} breach risk from ${a}% to ${b}%` : `the ${el} breach risk goes from ${a}% to ${b}%`;
  return `Net effect against the historical mix: about ${net >= 0 ? "+" : "-"}${money(Math.abs(net))}/t. ${pay} and ${risk}.`;
}

function ControlSuite({ plan, groups, onClose }: { plan: any; groups: string[]; onClose: () => void }) {
  const { act } = useSim();
  const [slot, setSlot] = useState(1);
  const heat = plan.heats.find((h: any) => h.slot === slot);
  const [mix, setMix] = useState<Record<string, number>>({});
  const [ev, setEv] = useState<any>(null);
  const [ms, setMs] = useState<number | null>(null);
  useEffect(() => { setMix({ ...heat.group_t }); }, [slot, plan.version]);
  const t = useRef<any>(null);
  useEffect(() => {
    clearTimeout(t.current);
    t.current = setTimeout(async () => { const s = performance.now(); try { setEv(await post("/api/plan/evaluate", { slot, group_t: mix })); setMs(performance.now() - s); } catch { /* ignore */ } }, 60);
  }, [mix, slot]);
  const total = Object.values(mix).reduce((a, b) => a + b, 0);
  return (<div>
    <div className="row"><b>Control Suite</b><span className="sp" /><button onClick={onClose}>Close</button></div>
    <select value={slot} onChange={(e) => setSlot(+e.target.value)} aria-label="heat">{plan.heats.map((h: any) => <option key={h.slot} value={h.slot}>Heat {h.slot}{h.frozen ? " (frozen)" : ""}</option>)}</select>
    {groups.map((g) => (<div key={g}><div className="small">{g.replace("SCRAP_", "")} · {(mix[g] ?? 0).toFixed(1)} t</div>
      <input type="range" min={0} max={140} step={0.5} value={mix[g] ?? 0} onChange={(e) => setMix({ ...mix, [g]: +e.target.value })} aria-label={g} /></div>))}
    <div className="small">Charge {total.toFixed(1)} t · tap {ev?.tap_actual_t?.toFixed(1) ?? "–"} t (target {heat.target_tap_t}) {ms !== null && `· ${ms.toFixed(0)} ms`}</div>
    {ev && <>
      <div><b>{money(ev.cost.per_t)}/t</b> <span style={{ color: ev.delta_cost_per_t > 0 ? "var(--bad)" : "var(--ok)" }}>({ev.delta_cost_per_t >= 0 ? "+" : ""}{ev.delta_cost_per_t.toFixed(2)} vs plan)</span></div>
      {Object.entries(ev.predicted).map(([el, p]: any) => <Headroom key={el} el={el} p={p} />)}
      {ev.violations.length > 0 && <div className="banner">Over limit: {ev.violations.map(sym).join(", ")}</div>}
      {ev.process_violations?.map((v: string) => <div key={v} className="banner">⚠ {v}</div>)}
    </>}
    <div className="row"><button onClick={() => setMix({ ...heat.group_t })}>Revert</button>
      <button className="primary" disabled={total > 140} title={total > 140 ? "The charge does not fit the furnace" : ""} onClick={async () => { await act("/api/plan/edit", { slot, group_t: mix }); onClose(); }}>Save as new version</button></div>
  </div>);
}

export default function SupervisorView() {
  const { phone, state, act } = useSim();
  const v = phone.supervisor; const plan = v?.plan;
  const [edit, setEdit] = useState(false); const [slot, setSlot] = useState(1);
  const groups = useMemo(() => state?.groups ?? [], [state]);
  if (!plan?.heats) return <div className="small">No plan yet. Use “Plan 06:00” in the Process View.</div>;
  if (edit) return <ControlSuite plan={plan} groups={groups} onClose={() => setEdit(false)} />;
  const heat = plan.heats.find((h: any) => h.slot === slot) ?? plan.heats[0];
  const canApprove = plan.status === "EMAILED";
  return (<div>
    {v.drift_alert && <div className="banner">⚠ {v.drift_alert.text}</div>}
    <div className="row"><b>Plan v{plan.version}</b><span className="chip shadow">{plan.status}</span></div>
    <div><b>{money(plan.cost.per_t)}/t</b> <span className="small">{plan.cost.risk_match ? "plan, held to the 95% limits" : `vs historical mix ${money(plan.cost.baseline_per_t)}/t (not risk-matched)`}</span></div>
    {plan.cost.risk_match && <div className="small">
      Historical mix {money(plan.cost.risk_match.hist_per_t)}/t, up to {(plan.cost.risk_match.worst_p * 100).toFixed(0)}% chance of breaching {sym(plan.cost.risk_match.worst_el)}.
      Optimiser at that same risk: {money(plan.cost.risk_match.matched_per_t)}/t ({plan.cost.risk_match.matched_per_t - plan.cost.risk_match.hist_per_t >= 0 ? "+" : ""}{(plan.cost.risk_match.matched_per_t - plan.cost.risk_match.hist_per_t).toFixed(2)}).
      Plan vs that: {plan.cost.per_t - plan.cost.risk_match.matched_per_t >= 0 ? "+" : ""}{(plan.cost.per_t - plan.cost.risk_match.matched_per_t).toFixed(2)}/t for the 95% limits.
      <div>{bottomLine(plan.cost.per_t, plan.cost.risk_match)}</div></div>}
    <div className="small">{plan.summary.n_heats} heats · {Object.entries(plan.summary.tonnes_by_group).map(([g, t]: any) => `${g.replace("SCRAP_", "")} ${t}t`).join(" · ")}</div>
    {plan.explanation && <div className="mail msg"><div className="small">Agent</div>{plan.explanation}</div>}
    <AgentMessages items={v.messages} />
    {plan.alternatives?.length > 0 && <div><div className="small"><b>Agent's alternatives</b> (approve one instead of v{plan.version})</div>
      {plan.alternatives.map((a: any) => (<div key={a.candidate_id} className="opt">
        <div className="row" style={{ margin: 0 }}><b>Option {a.label}</b><span>{money(a.cost_per_t)}/t</span><span className="sp" />
          <button disabled={!canApprove} onClick={() => act("/api/plan/approve", { option: a.candidate_id })}>Approve {a.label}</button></div>
        <div className="small">{[a.params.relax_recipe_bounds && "relaxed recipe bounds", a.params.spec_class !== "plant_strict" && `spec ${a.params.spec_class}`,
          a.params.alpha !== 0.05 && `risk ${(a.params.alpha * 100).toFixed(0)}%`].filter(Boolean).join(" · ") || "default settings"}
          {" · worst Cu P(exceed) "}{(a.worst_p_exceed.cu * 100).toFixed(1)}%</div>
      </div>))}</div>}
    <div className="row"><b>Headroom, heat</b>
      <select value={slot} onChange={(e) => setSlot(+e.target.value)} aria-label="heat">{plan.heats.map((h: any) => <option key={h.slot} value={h.slot}>{h.slot}</option>)}</select></div>
    {Object.entries(heat.headroom).map(([el, p]: any) => <Headroom key={el} el={el} p={p} />)}
    <div className="small"><b>Binding constraints</b></div>
    {plan.binding.length ? plan.binding.slice(0, 3).map((b: any, i: number) => <div key={i} className="small">• {b.plain}</div>) : <div className="small">None binding.</div>}
    {v.backtest && <div className="small" style={{ marginTop: 6 }}>Back-test: {v.backtest.saving_pct.toFixed(1)}% cheaper at same risk (ASSUMED prices)</div>}
    <div className="row">
      <button className="primary" disabled={!canApprove} onClick={() => act("/api/plan/approve", {})}>Approve</button>
      <button onClick={() => setEdit(true)}>Edit</button>
      <button className="danger" disabled={["RELEASED", "LOADING", "DONE"].includes(plan.status)} onClick={() => act("/api/plan/reject", { reason: "Supervisor requested a new plan" })}>Reject</button>
    </div>
    <ChatThread role="supervisor" items={v.chat} placeholder="Ask the agent (why / cost / what if)…" />
  </div>);
}
