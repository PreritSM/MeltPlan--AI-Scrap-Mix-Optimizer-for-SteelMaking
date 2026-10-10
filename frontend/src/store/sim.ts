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

let tid = 0;
let timer: any = null;
const WHO: Record<string, string> = { email: "Email", supervisor: "Supervisor", shift_leader: "Shift leader", crane: "Crane" };

export const useSim = create<S>((set, getS) => ({
  state: null, phone: {}, unread: { email: 0, supervisor: 0, shift_leader: 0, crane: 0 }, openRole: null, inspect: null,
  pulse: 0, toasts: [], playing: false, error: null,
  refresh: async () => {
    try {
      const [state, ...ph] = await Promise.all([get("/api/state"), ...ROLES.map((r) => get(`/api/phone/${r}`))]);
      set({ state, phone: Object.fromEntries(ROLES.map((r, i) => [r, ph[i]])), error: null });
    } catch (e: any) { set({ error: String(e.message ?? e) }); }
  },
  act: async (path, body) => {
    try { const r = await post(path, body); await getS().refresh(); return r; }
    catch (e: any) { set({ error: String(e.message ?? e) }); throw e; }
  },
  setOpenRole: (r) => set((s) => ({ openRole: r, unread: r ? { ...s.unread, [r]: 0 } : s.unread })),
  setInspect: (id) => set({ inspect: id }),
  setPlaying: (p) => {
    set({ playing: p });
    if (timer) clearInterval(timer);
    if (p) timer = setInterval(async () => {
      const r = await getS().act("/api/sim/step_heat").catch(() => ({ done: true }));
      if (r?.done) getS().setPlaying(false);
    }, 1100);
  },
  init: () => {
    getS().refresh();
    let t: any = null;
    const un = subscribe((ev) => {
      set((s) => {
        const unread = { ...s.unread };
        (ev.audience as string[]).forEach((a) => { if (a in unread && a !== s.openRole && ev.kind !== "reset") unread[a as Role]++; });
        const msg: Record<string, string> = { email: "New plan email", replan: "Plan changed by agent", released: "Plan released", approved: "Plan approved", drift: "Composition drift alert", shortage: "Shortage reported", issue: "Issue reported", message: "Message from the agent", chat: "Agent replied", plan_done: "All heats of the plan tapped" };
        const toasts = msg[ev.kind] ? [...s.toasts.slice(-2), { id: ++tid, text: msg[ev.kind] }] : s.toasts;
        return { unread: ev.kind === "reset" ? { email: 0, supervisor: 0, shift_leader: 0, crane: 0 } : unread, pulse: s.pulse + 1, toasts };
      });
      clearTimeout(t); t = setTimeout(() => getS().refresh(), 150);
    });
    return un;
  },
}));
export { WHO };
