import { Badge, f } from "../components/Badge";

const Table = ({ rows, cols }: { rows: any[]; cols?: string[] }) => {
  if (!rows?.length) return <div className="small">No rows.</div>;
  const c = cols ?? Object.keys(rows[0]);
  return (
    <table className="tbl"><thead><tr>{c.map((k) => <th key={k}>{k}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i}>{c.map((k) => <td key={k}>{typeof r[k] === "object" ? JSON.stringify(r[k]) : f(r[k], 4)}</td>)}</tr>)}</tbody></table>
  );
};
const Json = ({ v }: { v: any }) => <div className="mono">{JSON.stringify(v, null, 1)}</div>;

/** One agent run: event -> model reasoning/text and tool calls in order -> outcome. */
function AgentRun({ d }: { d: any }) {
  const path = d.path ?? "deterministic";
  const kind = path === "llm" ? "ESTIMATED" : path.startsWith("fallback") ? "ASSUMED" : "DERIVED";
  return (
    <div className="run">
      <div className="row" style={{ margin: 0 }}>
        <b>{d.ts?.replace("T", " ")}</b><code>{d.trigger}</code>
        <span className={`badge ${kind}`} title={d.path_reason ?? ""}>{path.toUpperCase()}</span>
        <span className="sp" /><b>{d.decision}</b>{d.version ? ` → v${d.version}` : ""}
      </div>
      {d.trigger_text && <div className="small">“{d.trigger_text}”</div>}
      {(d.steps ?? []).map((s: any, k: number) => s.type === "tool" ? (
        <div key={k} className={`step tool ${s.error ? "err" : ""}`}>
          <code>{s.name}</code>({Object.keys(s.args ?? {}).length ? JSON.stringify(s.args) : ""})
          <div className="small">{s.error ? `✗ ${s.error}` : `→ ${JSON.stringify(s.result)}`}</div>
        </div>
      ) : (
        <div key={k} className={`step ${s.type}`}>{s.type === "thinking" ? "💭 " : "💬 "}{s.text}</div>
      ))}
      {!d.steps && d.tools?.length > 0 && <div className="small">tools: {d.tools.join(" → ")}</div>}
      {d.reason && <div className="small">Reason: {d.reason}</div>}
      {d.path_reason && <div className="small">Path: {d.path_reason}</div>}
      {d.explanation?.supervisor && <div>{d.explanation.supervisor}</div>}
    </div>
  );
}

export default function Inspector({ card, onClose }: { card: any; onClose: () => void }) {
  const i = card.inspector ?? {};
  let body: any;
  if (card.id === "estimator" && i.estimates) {
    body = (<>
      {Object.entries(i.estimates).map(([el, e]: any) => (
        <div key={el}><h4>{el.toUpperCase()} (wt %): prior vs Bayes-NNLS, 90% CI</h4>
          <Table rows={e.groups.map((g: string, k: number) => ({ group: g, prior: e.prior[k], estimate: e.mean[k], ci_lo: e.lo[k], ci_hi: e.hi[k], online_kalman: i.online_kalman?.[el]?.[k] }))} /></div>))}
      <h4>Held-out test MAE (wt %)</h4>
      <Table rows={Object.entries(i.metrics).map(([el, m]: any) => ({ element: el, ...m.mae, pi90_coverage: m.coverage }))} />
      <p className="small">{i.caveat}</p>
      {i.drift && <p><b>Drift:</b> {i.drift.text}</p>}
      {i.last_heat && <><h4>Last heat {i.last_heat.heatid}: predicted vs measured</h4>
        <Table rows={Object.entries(i.last_heat.elements).map(([el, v]: any) => ({ element: el, predicted: v.predicted, measured: v.measured, z: v.z }))} /></>}
    </>);
  } else if (card.id === "optimizer") {
    body = (<><p>Spec class <b>{i.spec_class}</b> (limits ASSUMED, wt %): {Object.entries(i.limits).map(([k, v]) => `${k.toUpperCase()} ${v}`).join(" · ")}</p>
      <h4>Binding constraints / shadow prices</h4>
      {i.binding?.length ? <Table rows={i.binding.slice(0, 12).map((b: any) => ({ constraint: b.name, kind: b.kind, plain: b.plain }))} /> : <div className="small">No binding constraint.</div>}</>);
  } else if (card.id === "agent") {
    body = (<><p className="small">LLM: {i.llm}</p>
      {i.trace?.length ? i.trace.slice().reverse().map((d: any, k: number) => <AgentRun key={k} d={d} />) : <div className="small">No decisions yet.</div>}</>);
  } else if (card.id === "backtest" && i.result) {
    const r = i.result;
    body = (<><p><b>{f(r.saving_pct)}%</b> cheaper at the same predicted residual risk (95% CI {f(r.ci[0])}–{f(r.ci[1])}%) over {r.n_heats} held-out heats; ${f(r.usd_per_t)}/t.</p>
      <p className="small">Price sensitivity (±20%): {r.price_sensitivity_pct.map((x: number) => f(x) + "%").join(", ")}. {i.note}.</p>
      <Table rows={Object.keys(r.mix_hist).map((g) => ({ group: g, historical_t: r.mix_hist[g], lp_t: r.mix_lp[g] }))} /></>);
  } else if (card.id === "ingest" || card.id === "dq") {
    body = <Table rows={i.dq} />;
  } else if (card.id === "material") {
    body = (<><p className="small">{i.note}</p><Table rows={i.mapping} /></>);
  } else if (card.id.startsWith("act_")) {
    body = (<><p>Plan status: <b>{i.plan_status ?? "–"}</b></p><h4>Audit log</h4><Table rows={(i.audit ?? []).slice().reverse().slice(0, 25)} cols={["ts", "version", "actor", "action", "detail"]} /></>);
  } else if (card.id === "mes") {
    body = (<><p>Adapter: {i.adapter}. Selected heat: <b>{i.selected_heat ?? "–"}</b></p>
      {i.charges && <><h4>basket_charged rows</h4><Table rows={i.charges} /></>}</>);
  } else if (card.provenance === "MOCK" && !card.id.startsWith("handling") && card.id !== "inbox") {
    body = <p>Placeholder: awaiting real feed. Contract: <code>{i.contract}</code>. The adapter returns labelled MOCK records.</p>;
  } else {
    body = <Json v={i} />;
  }
  return (
    <div className="drawer" role="dialog" aria-label={card.title}>
      <div className="row"><h4 style={{ margin: 0 }}>{card.title}</h4><Badge p={card.provenance} /><span className="sp" /><button onClick={onClose}>Close</button></div>
      {body}
    </div>
  );
}
