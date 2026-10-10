export type Role = "email" | "supervisor" | "shift_leader" | "crane";
export const ROLES: Role[] = ["email", "supervisor", "shift_leader", "crane"];

async function j(r: Response) {
  if (!r.ok) throw new Error((await r.json().catch(() => ({ detail: r.statusText }))).detail);
  return r.json();
}
export const get = (p: string) => fetch(p).then(j);
export const post = (p: string, body: unknown = {}) =>
  fetch(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(j);

export function subscribe(onEvent: (e: any) => void) {
  const es = new EventSource("/api/events");
  es.onmessage = (m) => onEvent(JSON.parse(m.data));
  return () => es.close();
}
