import { useSim } from "../store/sim";
import { AgentMessages, ChatThread, IssueBox } from "../components/Agent";

export default function ShiftLeaderView() {
  const { phone, act } = useSim();
  const v = phone.shift_leader; const plan = v?.plan;
  if (!plan?.heats) return <div className="small">No plan yet.</div>;
  const meas: Record<number, any> = Object.fromEntries((v.heat_log ?? []).map((h: any) => [h.slot, h.cu]));
  return (<div>
    {v.drift_alert && <div className="banner">⚠ {v.drift_alert.text}</div>}
    {plan.explanation && <div className="mail msg"><div className="small">What changed</div>{plan.explanation}</div>}
    <AgentMessages items={v.messages} />
    <div className="row"><b>Plan v{plan.version}</b><span className="chip shadow">{plan.status}</span><span className="sp" />
      <button className="primary" disabled={plan.status !== "SUPERVISOR_APPROVED"} onClick={() => act("/api/plan/release")}>Release to cranes</button></div>
    <div className="small">Timeline</div>
    {plan.heats.map((h: any) => (<div key={h.slot} className="li">
      <b>Heat {h.slot}</b><span className="small">{h.target_tap_t} t</span><span className="sp" />
      <span className="small" title={h.progress === "loaded" ? "Charged in baskets, waiting for the furnace" : ""}>{h.progress}</span>
      {meas[h.slot] && <span className="small">Cu pred {meas[h.slot].predicted.toFixed(3)} / meas {meas[h.slot].measured.toFixed(3)}</span>}
      <span title="headroom status" style={{ color: Object.values(h.headroom).includes("AT RISK") ? "var(--bad)" : "var(--ok)" }}>●</span>
    </div>))}
    <div className="small" style={{ marginTop: 8 }}><b>Change log</b></div>
    {(v.changes ?? []).map((c: any) => <div key={c.version} className="small">v{c.version} by {c.by} · {c.status}</div>)}
    <IssueBox role="shift_leader" placeholder="Report an issue, e.g. “only 590 t of 3A left today”" />
    <ChatThread role="shift_leader" items={v.chat} placeholder="Ask the agent…" />
  </div>);
}
