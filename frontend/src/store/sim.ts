import { create } from "zustand";
import { get, post, subscribe, Role, ROLES } from "../api/client";

interface Toast { id: number; text: string; role?: Role }
interface S {
  state: any;
  phone: Partial<Record<Role, any>>;
  unread: Record<Role, number>;
  openRole: Role | null;
  inspect: string | null;
  pulse: number;
  toasts: Toast[];
  playing: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  act: (path: string, body?: unknown) => Promise<any>;
  setOpenRole: (r: Role | null) => void;
  setInspect: (id: string | null) => void;
  setPlaying: (p: boolean) => void;
  init: () => () => void;
}

/** Progress telemetry: refreshes the phones but is not something a person has to read, so it does not raise the unread badge. */
const ROUTINE = new Set(["reset", "grab_done", "basket_complete", "heat_loaded", "heat_tapped", "agent_busy", "agent_done"]);
let tid = 0;
let timer: any = null;
const WHO: Record<string, string> = { email: "Email", supervisor: "Supervisor", shift_leader: "Shift leader", crane: "Crane" };

// Global zustand store for backend state, phone views and UI state.
// Provides refresh, action, playback and event-subscription helpers.
// Shared by the Process View and the phone pane.
export const useSim = create<S>((set, getS) => ({
  state: null, phone: {}, unread: { email: 0, supervisor: 0, shift_leader: 0, crane: 0 }, openRole: null, inspect: null,
  pulse: 0, toasts: [], playing: false, error: null,
  // Reloads the process state and every role's phone view.
  // Fetches all endpoints in parallel.
  // Stores any error message for the error bar.
  refresh: async () => {
    try {
      const [state, ...ph] = await Promise.all([get("/api/state"), ...ROLES.map((r) => get(`/api/phone/${r}`))]);
      set({ state, phone: Object.fromEntries(ROLES.map((r, i) => [r, ph[i]])), error: null });
    } catch (e: any) { set({ error: String(e.message ?? e) }); }
  },
  // Posts a user action to the API, then refreshes all state.
  // Records the error message and rethrows on failure.
  // Returns the API's response.
  act: async (path, body) => {
    try { const r = await post(path, body); await getS().refresh(); return r; }
    catch (e: any) { set({ error: String(e.message ?? e) }); throw e; }
  },
  // Opens a role's phone app, or returns to the home screen with null.
  // Clears that role's unread counter when opening it.
  // Other roles' counters are untouched.
  setOpenRole: (r) => set((s) => ({ openRole: r, unread: r ? { ...s.unread, [r]: 0 } : s.unread })),
  // Selects which Process View card the inspector shows.
  // null closes the inspector.
  // Stores only the card id.
  setInspect: (id) => set({ inspect: id }),
  // Starts or stops automatic heat stepping.
  // While playing, steps a heat about once a second.
  // Stops automatically when the backend reports the plan is done.
  setPlaying: (p) => {
    set({ playing: p });
    if (timer) clearInterval(timer);
    if (p) timer = setInterval(async () => {
      const r = await getS().act("/api/sim/step_heat").catch(() => ({ done: true }));
      if (r?.done) getS().setPlaying(false);
    }, 1100);
  },
  // Loads initial state and subscribes to the backend event stream.
  // Updates unread counters and toasts for each non-routine event.
  // Debounces a refresh after events; returns the unsubscribe function.
  init: () => {
    getS().refresh();
    let t: any = null;
    const un = subscribe((ev) => {
      set((s) => {
        const unread = { ...s.unread };
        if (!ROUTINE.has(ev.kind))
          (ev.audience as string[]).forEach((a) => { if (a in unread && a !== s.openRole) unread[a as Role]++; });
        const msg: Record<string, string> = { email: "New plan email", replan: "Plan changed by agent", plan_reinstated: "Rejected change withdrawn", released: "Plan released", approved: "Plan approved", drift: "Composition drift alert", shortage: "Shortage reported", issue: "Issue reported", message: "Message from the agent", chat: "Agent replied", plan_done: "All heats of the plan tapped", basket_complete: "Basket complete" };
        const text = ev.kind === "heat_loaded" ? `Heat ${ev.payload?.slot} loaded` : msg[ev.kind];
        const toasts = text ? [...s.toasts.slice(-2), { id: ++tid, text }] : s.toasts;
        return { unread: ev.kind === "reset" ? { email: 0, supervisor: 0, shift_leader: 0, crane: 0 } : unread, pulse: s.pulse + 1, toasts };
      });
      clearTimeout(t); t = setTimeout(() => getS().refresh(), 150);
    });
    return un;
  },
}));
export { WHO };
