import { useEffect, useState } from "react";
import { useSim } from "../store/sim";
import { Badge, f } from "../components/Badge";
import Inspector from "./Inspector";

const LANES: [string, string][] = [["read", "Read layer"], ["data", "Data layer"], ["intelligence", "Intelligence layer"], ["act", "Act layer"]];

// Turns a card's inputs or outputs into a one-line summary.
// Includes only scalar values, formatting numbers compactly.
// Returns 'key: value' pairs joined by dots.
function summarize(o: any) {
  return Object.entries(o ?? {}).filter(([, v]) => v !== null && v !== undefined && typeof v !== "object")
    .map(([k, v]) => `${k.replace(/_/g, " ")}: ${typeof v === "number" ? (Number.isInteger(v) ? v : f(v as number, 3)) : v}`).join(" · ");
}

// Top control bar of the Process View.
// Shows the clock, mode and agent status with plan, play, step, scenario and reset controls.
// Also shows the heat surprise strip and the provenance legend.
function TopBar() {
  const { state, act, playing, setPlaying } = useSim();
  if (!state) return null;
  const sim = state.mode === "SIMULATED";
  return (
    <div className="topbar">
      <span className="clock">{state.clock.replace("T", " ")}</span>
      <span className={`chip ${sim ? "sim" : "shadow"}`}>{sim ? "SIMULATED SCENARIO" : "SHADOW REPLAY"}</span>
      <button className="primary" onClick={() => act("/api/sim/plan")} title="Advance to the next 06:00 and plan the next 24 h">▶| Plan 06:00</button>
      <button onClick={() => setPlaying(!playing)} aria-label={playing ? "pause" : "play"}>{playing ? "⏸" : "▶"}</button>
      <button onClick={() => act("/api/sim/step_heat")} disabled={!state.plan_status || state.agent?.busy || !!state.sim_hold}
        title={state.agent?.busy ? "The next heat waits for the agent" : state.sim_hold ?? ""}>Step heat</button>
      {state.sim_hold && !state.agent?.busy && <span className="chip busy" title={state.sim_hold}>{state.sim_hold.includes("not released") ? "Waiting for release" : "Waiting for approval"}</span>}
      {state.ready_slot && <span className="chip shadow" title="All baskets charged; Step heat taps it">Heat {state.ready_slot} ready</span>}
      <button onClick={() => act("/api/sim/inbox", {})} disabled={!state.plan_status} title="Scenario (MOCK): logistics emails that 3A is short">📧 Supplier email</button>
      <span className={`chip ${state.agent?.busy ? "busy" : state.agent?.available ? "shadow" : "sim"}`} title={state.agent?.llm}>
        {state.agent?.busy ? "Agent working…" : state.agent?.available ? "Agent: LLM" : "Agent: fallback"}</span>
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
      <span className="small">heat {state.cursor}/{state.n_replay}</span>
    </div>
  );
}

// Main Process View: four lanes of stage cards (read, data, intelligence, act).
// Pulses the lanes briefly on each new event.
// Clicking a card opens it in the inspector drawer.
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
