import { useState } from "react";
import { useSim } from "../store/sim";

export default function CraneView() {
  const { phone, state, act } = useSim();
  const v = phone.crane; const plan = v?.plan;
  const [slot, setSlot] = useState<number | null>(null);
  const [tab, setTab] = useState(1);
  const [done, setDone] = useState<Record<string, boolean>>({});
  const banner = plan?.banner;
  if (!plan?.loading?.length) return <div>{banner && <div className="banner">{banner.text}</div>}<div className="small">No released plan. Waiting for the shift leader.</div></div>;
  const open = plan.loading.find((h: any) => h.slot === (slot ?? plan.loading.find((x: any) => x.progress !== "tapped")?.slot ?? 1)) ?? plan.loading[0];
  const bas = open.baskets.find((b: any) => b.basket_no === tab) ?? open.baskets[0];
  return (<div>
    {banner && <div className="banner">{banner.text}</div>}
    <div className="row"><b style={{ fontSize: 22 }}>Heat {open.slot}</b><span className="sp" />
      <select value={open.slot} onChange={(e) => setSlot(+e.target.value)} aria-label="heat">{plan.loading.map((h: any) => <option key={h.slot} value={h.slot}>{h.slot}</option>)}</select></div>
    <div className="row">{open.baskets.map((b: any) => <button key={b.basket_no} className={`tab ${bas.basket_no === b.basket_no ? "on" : ""}`} onClick={() => setTab(b.basket_no)}>Basket {b.basket_no}</button>)}</div>
    <div className="small">Basket {bas.basket_no} target {bas.target_t} t (typical band from history)</div>
    {bas.items.map((it: any) => { const k = `${open.slot}-${bas.basket_no}-${it.mat_code}`; return (
      <div key={k} className={`li ${done[k] ? "done" : ""}`}>
        <b>{it.grab_order}.</b><div style={{ flex: 1 }}><b>{it.mat_code}</b> {it.mat_name}<div className="small">{it.tonnes} t · {it.grabs} grabs</div></div>
        <button className="primary" disabled={done[k]} onClick={async () => { setDone({ ...done, [k]: true }); await act("/api/crane/grab_done", { slot: open.slot, basket: bas.basket_no, mat_code: it.mat_code, tonnes: it.tonnes }); }}>GRAB DONE</button>
      </div>); })}
    <div className="row"><button onClick={() => { const next = bas.basket_no + 1; if (open.baskets.find((b: any) => b.basket_no === next)) setTab(next); }}>Basket complete</button>
      <button className="danger" onClick={() => act("/api/report/shortage", { group: bas.items[0]?.group ?? "SCRAP_3A" })}>Report shortage</button></div>
  </div>);
}
