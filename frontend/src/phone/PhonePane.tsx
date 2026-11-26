import { useEffect, useRef, useState } from "react";
import { useSim, WHO } from "../store/sim";
import { Role, ROLES } from "../api/client";
import EmailView from "./EmailView";
import SupervisorView from "./SupervisorView";
import ShiftLeaderView from "./ShiftLeaderView";
import CraneView from "./CraneView";

const W = 390, H = 844;
const GLYPH: Record<Role, string> = { email: "✉️", supervisor: "🧑‍💼", shift_leader: "👷", crane: "🏗️" };

// Renders the simulated phone with a home screen of role apps.
// Scales the phone to fit the pane and shows toasts for new events.
// Opens the selected role's view, showing only what that role may see.
export default function PhonePane() {
  const ref = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(0.8);
  const { openRole, setOpenRole, unread, phone, toasts, state } = useSim();
  const [toast, setToast] = useState<string | null>(null);

  useEffect(() => {
    const el = ref.current!;
    const ro = new ResizeObserver(() => setScale(Math.max(0.3, Math.min((el.clientWidth - 12) / W, (el.clientHeight - 12) / H))));
    ro.observe(el); return () => ro.disconnect();
  }, []);
  useEffect(() => {
    const t = toasts[toasts.length - 1]; if (!t) return;
    setToast(t.text); const id = setTimeout(() => setToast(null), 3000); return () => clearTimeout(id);
  }, [toasts]);

  const unreadMail = (phone.email?.emails ?? []).filter((m: any) => !m.read).length;
  const time = state?.clock?.slice(11, 16) ?? "--:--";
  // Returns the unread count to show on a role's icon.
  // Email counts unread mails; other roles use the store's unread counters.
  // Zero hides the badge.
  const badge = (r: Role) => (r === "email" ? unreadMail : unread[r]);

  return (
    <div className="pp" ref={ref}>
      <div style={{ width: W * scale, height: H * scale }}>
        <div className="pframe" style={{ transform: `scale(${scale})` }} aria-label="Phone view">
          <div className="pscreen">
            <div className="pstatus"><span>{time}</span><span>{state?.mode === "SIMULATED" ? "SIMULATED" : "SHADOW"} · 5G</span></div>
            {toast && <div className="toast" role="status">🔔 {toast}</div>}
            {!openRole ? (
              <>
                <div className="pbar">MeltPlan</div>
                <div className="home">
                  {ROLES.map((r) => (
                    <button key={r} className="icon" onClick={() => setOpenRole(r)} aria-label={WHO[r]}>
                      <span className="g">{GLYPH[r]}</span>{WHO[r]}
                      {badge(r) > 0 && <span className="n">{badge(r)}</span>}
                    </button>
                  ))}
                </div>
                <div className="small" style={{ padding: "0 22px" }}>Each icon is a different person's phone: you only see what that role may see.</div>
              </>
            ) : (
              <>
                <div className="pbar"><button onClick={() => setOpenRole(null)} aria-label="Back">←</button>{WHO[openRole]}<span className="viewing">Viewing as: {WHO[openRole]}</span></div>
                <div className="pbody">
                  {openRole === "email" && <EmailView />}
                  {openRole === "supervisor" && <SupervisorView />}
                  {openRole === "shift_leader" && <ShiftLeaderView />}
                  {openRole === "crane" && <CraneView />}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
