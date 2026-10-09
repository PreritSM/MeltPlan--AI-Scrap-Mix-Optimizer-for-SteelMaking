export const Badge = ({ p }: { p: string }) => <span className={`badge ${p}`} title={`Provenance: ${p}`}>{p}</span>;
export const f = (v: any, d = 2) => (typeof v === "number" ? v.toFixed(d) : v ?? "–");
