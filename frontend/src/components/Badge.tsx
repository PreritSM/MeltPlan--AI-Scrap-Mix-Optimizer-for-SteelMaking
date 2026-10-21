// Renders a small provenance badge (REAL, MOCK, ESTIMATED, ...).
// The provenance also selects the badge's CSS colour class.
// Shows a tooltip naming the provenance.
export const Badge = ({ p }: { p: string }) => <span className={`badge ${p}`} title={`Provenance: ${p}`}>{p}</span>;
// Formats a value for display.
// Numbers are fixed to d decimals; other values pass through.
// null / undefined become an en dash.
export const f = (v: any, d = 2) => (typeof v === "number" ? v.toFixed(d) : v ?? "–");
