import { useState } from "react";
import { useSim } from "../store/sim";

export default function EmailView() {
  const { phone, act, setOpenRole } = useSim();
  const [open, setOpen] = useState<number | null>(null);
  const [reply, setReply] = useState(""); const [answer, setAnswer] = useState<string | null>(null);
  const mails = phone.email?.emails ?? [];
  if (!mails.length) return <div className="small">Inbox empty. The 06:00 plan email arrives here (MOCK mail channel).</div>;
  if (open !== null && mails[open]) {
    const m = mails[open];
    return (<div>
      <button onClick={() => setOpen(null)}>← Inbox</button>
      <h4 style={{ margin: "8px 0 2px" }}>{m.subject}</h4><div className="small">To: {m.to} · {m.ts?.replace("T", " ")} · channel {m.channel}</div>
      <pre style={{ whiteSpace: "pre-wrap", font: "inherit", marginTop: 10 }}>{m.body}</pre>
      <div className="row"><button className="primary" onClick={() => setOpenRole("supervisor")}>Open in ScrapChef</button></div>
      <input type="text" placeholder="Reply to the agent…" value={reply} onChange={(e) => setReply(e.target.value)} />
      <div className="row"><button onClick={async () => { const r = await act("/api/chat", { role: "email", text: reply }); setAnswer(r.answer); setReply(""); }} disabled={!reply}>Reply</button></div>
      {answer && <div className="mail"><div className="small">Agent reply</div>{answer}</div>}
    </div>);
  }
  return (<div>{mails.map((m: any, i: number) => (
    <div key={i} className={`mail ${m.read ? "" : "unread"}`} onClick={() => { setOpen(i); act("/api/email/read"); }} role="button">
      <b>{m.subject}</b><div className="small">{m.ts?.replace("T", " ")}</div></div>))}</div>);
}
