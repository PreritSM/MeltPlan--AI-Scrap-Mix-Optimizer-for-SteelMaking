import { useState } from "react";
import { useSim } from "../store/sim";
import { ChatThread } from "../components/Agent";

// Phone screen for the email role: inbox and message view.
// Opening a mail marks emails read and shows its full body.
// Offers a jump to the supervisor app and a reply thread to the agent.
export default function EmailView() {
  const { phone, act, setOpenRole } = useSim();
  const [open, setOpen] = useState<number | null>(null);
  const mails = phone.email?.emails ?? [];
  if (!mails.length) return <div className="small">Inbox empty. The 06:00 plan email arrives here (MOCK mail channel).</div>;
  if (open !== null && mails[open]) {
    const m = mails[open];
    return (<div>
      <button onClick={() => setOpen(null)}>← Inbox</button>
      <h4 style={{ margin: "8px 0 2px" }}>{m.subject}</h4><div className="small">To: {m.to} · {m.ts?.replace("T", " ")} · channel {m.channel}</div>
      <pre style={{ whiteSpace: "pre-wrap", font: "inherit", marginTop: 10 }}>{m.body}</pre>
      <div className="row"><button className="primary" onClick={() => setOpenRole("supervisor")}>Open in MeltPlan</button></div>
      <ChatThread role="email" items={phone.email?.chat} placeholder="Reply to the agent…" />
    </div>);
  }
  return (<div>{mails.map((m: any, i: number) => (
    <div key={i} className={`mail ${m.read ? "" : "unread"}`} onClick={() => { setOpen(i); act("/api/email/read"); }} role="button">
      <b>{m.subject}</b><div className="small">{m.ts?.replace("T", " ")}</div></div>))}</div>);
}
