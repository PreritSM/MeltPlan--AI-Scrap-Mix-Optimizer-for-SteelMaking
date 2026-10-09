import { useEffect, useState } from "react";
import { useSim } from "../store/sim";
import { Badge, f } from "../components/Badge";
import Inspector from "./Inspector";

const LANES: [string, string][] = [["read", "Read layer"], ["data", "Data layer"], ["intelligence", "Intelligence layer"], ["act", "Act layer"]];

function summarize(o: any) {
  return Object.entries(o ?? {}).filter(([, v]) => v !== null && v !== undefined && typeof v !== "object")
    .map(([k, v]) => `${k.replace(/_/g, " ")}: ${typeof v === "number" ? (Number.isInteger(v) ? v : f(v as number, 3)) : v}`).join(" · ");
}

function TopBar() {
  const { state, act, playing, setPlaying } = useSim();
  const [speed] = useState(1);
  if (!state) return null;
  const sim = state.mode === "SIMULATED";
  return (
    <div className="topbar">
      <span className="clock">{state.clock.replace("T", " ")}</span>
      <span className={`chip ${sim ? "sim" : "shadow"}`}>{sim ? "SIMULATED SCENARIO" : "SHADOW REPLAY"}</span>
      <button className="primary" onClick={() => act("/api/sim/plan")} title="Advance to the next 06:00 and plan the next 24 h">▶| Plan 06:00</button>
      <button onClick={() => setPlaying(!playing)} aria-label={playing ? "pause" : "play"}>{playing ? "⏸" : "▶"}</button>
      <button onClick={() => act("/api/sim/step_heat")} disabled={!state.plan_status}>Step heat</button>
      <select aria-label="mode" value={state.mode} onChange={(e) => act("/api/sim/reset", { mode: e.target.value })}>
        <option value="SHADOW_REPLAY">Shadow replay (real heats)</option><option value="SIMULATED">Simulated drift scenario</option>
      </select>
      <button onClick={() => act("/api/sim/reset", { mode: state.mode })}>Reset</button>
      <div className="strip" title="Heats: predicted vs measured Cu error (σ)">
        {state.heat_strip.map((h: any) => {
          const a = Math.min(Math.abs(h.z), 3) / 3;
          return <i key={h.heatid} style={{ height: 4 + a * 22, background: a > 0.66 ? "var(--bad)" : a > 0.33 ? "var(--warn)" : "var(--ok)" }} title={`${h.heatid}: ${f(h.z, 1)}σ`} />;
        })}
      </div>
      <div className="legend">{state.legend.map((p: string) => <Badge key={p} p={p} />)}</div>
      <span className="small">{speed}× · heat {state.cursor}/{state.n_replay}</span>
    </div>
  );
}

export default function ProcessView() {
  const { state, inspect, setInspect, pulse } = useSim();
  const [pl, setPl] = useState(false);
  useEffect(() => { if (!pulse) return; setPl(true); const t = setTimeout(() => setPl(false), 900); return () => clearTimeout(t); }, [pulse]);
  if (!state) return <div className="pv">Connecting to backend…</div>;
  const all = Object.values(state.lanes).flat() as any[];
  const sel = all.find((c) => c.id === inspect);
  return (
    <div className="pv">
      <TopBar />
      <div className="lanes">
        {LANES.map(([k, title]) => (
          <section key={k} className={`lane ${pl ? "pulse" : ""}`} aria-label={title}>
            <h3>{title}</h3>
            {state.lanes[k].map((c: any) => (
              <button key={c.id} className={`card ${inspect === c.id ? "sel" : ""}`} onClick={() => setInspect(inspect === c.id ? null : c.id)}>
                <div className="t"><span><i className={`dot ${c.state}`} />{c.title}</span><Badge p={c.provenance} /></div>
                <div className="kv">{c.state}{summarize(c.inputs) ? " · in " + summarize(c.inputs) : ""}</div>
                {summarize(c.outputs) && <div className="kv">out {summarize(c.outputs)}</div>}
              </button>
            ))}
          </section>
        ))}
      </div>
      {sel && <Inspector card={sel} onClose={() => setInspect(null)} />}
    </div>
  );
}
