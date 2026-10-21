import { useState } from "react";
import { useSim } from "../store/sim";

// Shows the information-only messages the agent sent to this role.
// Displays the three newest, newest first, with sender and time.
// Renders nothing when there are no messages.
export function AgentMessages({ items }: { items?: any[] }) {
  if (!items?.length) return null;
  return (<>{items.slice().reverse().slice(0, 3).map((m: any, i: number) => (
    <div key={i} className="mail msg"><div className="small">{m.sender === "agent" ? "Agent" : m.sender} · {m.ts?.slice(11, 16)}</div>{m.text}</div>))}</>);
}

// Question box and chat thread for talking to the agent.
// Sends questions via /api/chat; replies arrive through the event stream.
// Shows the last four messages and an 'Agent working' hint while busy.
export function ChatThread({ role, items, placeholder }: { role: string; items?: any[]; placeholder: string }) {
  const { act } = useSim();
  const busy = useSim((s) => s.phone?.[role as "supervisor"]?.agent_busy);
  const [q, setQ] = useState("");
  return (<div>
    <input type="text" placeholder={placeholder} value={q} onChange={(e) => setQ(e.target.value)}
      onKeyDown={(e) => { if (e.key === "Enter" && q) { act("/api/chat", { role, text: q }); setQ(""); } }} />
    <div className="row"><button disabled={!q} onClick={() => { act("/api/chat", { role, text: q }); setQ(""); }}>Ask</button>
      {busy && <span className="small">Agent working…</span>}</div>
    {(items ?? []).slice(-4).map((m: any, i: number) => (
      <div key={i} className={`mail ${m.who === "agent" ? "msg" : ""}`}><span className="small">{m.who === "agent" ? "Agent" : "You"}</span> {m.text}</div>))}
  </div>);
}

// Free-text box for reporting an operational issue.
// Posts the text to /api/report/issue for the agent to interpret.
// Clears the input after sending.
export function IssueBox({ role, placeholder }: { role: string; placeholder: string }) {
  const { act } = useSim();
  const [t, setT] = useState("");
  return (<div>
    <input type="text" placeholder={placeholder} value={t} onChange={(e) => setT(e.target.value)} aria-label="report issue" />
    <div className="row"><button className="danger" disabled={!t} onClick={() => { act("/api/report/issue", { role, text: t }); setT(""); }}>Report issue</button></div>
  </div>);
}
