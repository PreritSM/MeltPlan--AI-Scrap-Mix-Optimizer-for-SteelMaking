import { useState } from "react";
import { useSim } from "../store/sim";
import { AgentMessages, IssueBox } from "../components/Agent";

export default function CraneView() {
  const { phone, act } = useSim();
  const v = phone.crane; const plan = v?.plan;
  const [slot, setSlot] = useState<number | null>(null);
  const [tab, setTab] = useState(1);
  const [pending, setPending] = useState<Record<string, boolean>>({});   // optimistic ticks until the next refresh
  const banner = plan?.banner;
  if (!plan?.loading?.length) return <div>{banner && <div className="banner">{banner.text}</div>}<AgentMessages items={v?.messages} /><div className="small">No released plan. Waiting for the shift leader.</div></div>;
  if (plan.loading.every((h: any) => h.progress === "tapped"))
    return <div>{banner && <div className="banner">{banner.text}</div>}<AgentMessages items={v?.messages} /><div className="small">All heats of this plan are loaded and tapped. Waiting for the next plan.</div></div>;
  const def = plan.loading.find((x: any) => !["loaded", "tapped"].includes(x.progress)) ?? plan.loading.find((x: any) => x.progress !== "tapped");
  const open = plan.loading.find((h: any) => h.slot === (slot ?? def?.slot)) ?? plan.loading[0];
  const bas = open.baskets.find((b: any) => b.basket_no === tab) ?? open.baskets[0];
  const last = bas.basket_no === Math.max(...open.baskets.map((b: any) => b.basket_no));
  const complete = (open.baskets_complete ?? []).includes(bas.basket_no);
  const held = !!plan.hold || !!open.pending;   // a version awaits approval (crane stops), or this heat's new list is not released yet
  const waiting = plan.loading.filter((h: any) => h.progress === "loaded");
  const finish = async () => {
    await act("/api/crane/basket_complete", { slot: open.slot, basket: bas.basket_no });
    if (last) { setSlot(null); setTab(1); } else setTab(bas.basket_no + 1);
  };
  return (<div>
    {banner && <div className="banner">{banner.text}</div>}
    {waiting.map((h: any) => <div key={h.slot} className="small"><b>Heat {h.slot} loaded, waiting for the furnace</b></div>)}
    <AgentMessages items={v.messages} />
    <div className="row"><b style={{ fontSize: 22 }}>Heat {open.slot}</b><span className="sp" />
      <select value={open.slot} onChange={(e) => { setSlot(+e.target.value); setTab(1); }} aria-label="heat">{plan.loading.map((h: any) => <option key={h.slot} value={h.slot}>{h.slot}</option>)}</select></div>
    <div className="row">{open.baskets.map((b: any) => <button key={b.basket_no} className={`tab ${bas.basket_no === b.basket_no ? "on" : ""}`} onClick={() => setTab(b.basket_no)}>Basket {b.basket_no}</button>)}</div>
    {plan.hold && <div className="banner">⛔ {plan.hold}</div>}
    {open.pending && !plan.hold && <div className="banner">Updated list (v{open.pending.version}, {open.pending.status === "EMAILED" ? "awaiting supervisor approval" : "awaiting release"}). Do not load heat {open.slot} yet.</div>}
    <div className="small">Basket {bas.basket_no} target {bas.target_t} t (typical band from history)</div>
    {bas.items.map((it: any) => {
      const k = `${bas.basket_no}-${it.mat_code}`;
      const done = complete || (open.grabbed ?? []).includes(k) || pending[`${open.slot}-${k}`];
      return (
        <div key={k} className={`li ${done ? "done" : ""}`}>
          <b>{it.grab_order}.</b><div style={{ flex: 1 }}><b>{it.mat_code}</b> {it.mat_name}<div className="small">{it.tonnes} t · {it.grabs} grabs</div></div>
          <button className="primary" disabled={done || held} onClick={async () => { setPending({ ...pending, [`${open.slot}-${k}`]: true }); await act("/api/crane/grab_done", { slot: open.slot, basket: bas.basket_no, mat_code: it.mat_code, tonnes: it.tonnes }); }}>GRAB DONE</button>
        </div>); })}
    <div className="row"><button className="primary" disabled={complete || held || open.progress === "tapped"} onClick={finish}
      title="Confirms every line of this basket; lines not ticked are recorded at planned tonnes">
      {complete ? "Basket complete ✓" : last ? `Heat ${open.slot} loaded` : "Basket complete"}</button></div>
    <IssueBox role="crane" placeholder="Report a problem, e.g. “3ACh pile is empty”" />
  </div>);
}
