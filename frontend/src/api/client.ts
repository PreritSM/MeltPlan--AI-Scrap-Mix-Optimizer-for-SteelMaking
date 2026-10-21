export type Role = "email" | "supervisor" | "shift_leader" | "crane";
export const ROLES: Role[] = ["email", "supervisor", "shift_leader", "crane"];

// Unwraps a fetch Response as JSON.
// Throws an Error with the server's detail message on non-2xx responses.
// Used by both get and post.
async function j(r: Response) {
  if (!r.ok) throw new Error((await r.json().catch(() => ({ detail: r.statusText }))).detail);
  return r.json();
}
// Sends a GET request to an API path.
// Returns the parsed JSON body.
// Throws on HTTP errors.
export const get = (p: string) => fetch(p).then(j);
// Sends a JSON POST request to an API path.
// body defaults to an empty object.
// Returns the parsed JSON body and throws on HTTP errors.
export const post = (p: string, body: unknown = {}) =>
  fetch(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(j);

// Subscribes to the backend's server-sent event stream.
// Calls onEvent with each parsed event.
// Returns a function that closes the connection.
export function subscribe(onEvent: (e: any) => void) {
  const es = new EventSource("/api/events");
  es.onmessage = (m) => onEvent(JSON.parse(m.data));
  return () => es.close();
}
