import { sym } from "../format";

export function Headroom({ el, p }: { el: string; p: { mean: number; p95: number; limit: number; p_exceed: number } }) {
  const max = Math.max(p.limit * 1.25, p.p95 * 1.05);
  const pct = (v: number) => `${Math.min(100, (v / max) * 100)}%`;
  const over = p.p95 > p.limit * 1.001;   // a binding chance constraint puts P95 exactly on the limit; that is not over
  return (
    <div>
      <div className="small"><b>{sym(el)}</b> mean {p.mean.toFixed(3)} · P95 {p.p95.toFixed(3)} · limit {p.limit.toFixed(3)} · P(exceed) {(p.p_exceed * 100).toFixed(1)}%</div>
      <div className={`bar ${over ? "over" : ""}`} role="img" aria-label={`${el} headroom ${over ? "over limit" : "ok"}`}>
        <div className="m" style={{ width: pct(p.mean) }} /><div className="p" style={{ left: pct(p.p95) }} /><div className="l" style={{ left: pct(p.limit) }} />
      </div>
    </div>
  );
}
