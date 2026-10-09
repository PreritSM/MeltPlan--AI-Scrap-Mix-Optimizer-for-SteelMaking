import { useState } from "react";
import { useSim } from "../store/sim";

export default function ShiftLeaderView() {
  const { phone, state, act } = useSim();
  const v = phone.shift_leader; const plan = v?.plan;
  const [q, setQ] = useState(""); const [grp, setGrp] = useState("SCRAP_16A_6A");
  if (!plan?.heats) return <div className="small">No plan yet.</div>;
  const meas: Record<number, any> = Object.fromEntries((v.heat_log ?? []).map((h: any) => [h.slot, h.cu]));
  return (<div>
    {v.drift_alert && <div className="banner">⚠ {v.drift_alert.text}</div>}
    {plan.explanation && <div className="mail"><div className="small">What changed</div>{plan.explanation}</div>}
    <div className="row"><b>Plan v{plan.version}</b><span className="chip shadow">{plan.status}</span><span className="sp" />
      <button className="primary" disabled={plan.status !== "SUPERVISOR_APPROVED"} onClick={() => act("/api/plan/release")}>Release to cranes</button></div>
    <div className="small">Timeline</div>
    {plan.heats.map((h: any) => (<div key={h.slot} className="li">
      <b>Heat {h.slot}</b><span className="small">{h.target_tap_t} t</span><span className="sp" />
      <span className="small">{h.progress}</span>
      {meas[h.slot] && <span className="small">Cu pred {meas[h.slot].predicted.toFixed(3)} / meas {meas[h.slot].measured.toFixed(3)}</span>}
      <span title="headroom status" style={{ color: Object.values(h.headroom).includes("AT RISK") ? "var(--bad)" : "var(--ok)" }}>●</span>
    </div>))}
    <div className="small" style={{ marginTop: 8 }}><b>Change log</b></div>
    {(v.changes ?? []).map((c: any) => <div key={c.version} className="small">v{c.version} by {c.by} · {c.status}</div>)}
    <div className="row"><select value={grp} onChange={(e) => setGrp(e.target.value)} aria-label="material">{(state?.groups ?? []).map((g: string) => <option key={g}>{g}</option>)}</select>
      <button className="danger" onClick={() => act("/api/report/shortage", { group: grp })}>Report unavailable</button></div>
    <input type="text" placeholder="Chat with agent…" value={q} onChange={(e) => setQ(e.target.value)} />
    <div className="row"><button disabled={!q} onClick={async () => { await act("/api/chat", { role: "shift_leader", text: q }); setQ(""); }}>Send</button></div>
    {(v.chat ?? []).slice(-4).map((m: any, i: number) => <div key={i} className="mail"><span className="small">{m.who}</span> {m.text}</div>)}
  </div>);
}
