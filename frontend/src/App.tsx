import { useEffect, useRef, useState } from "react";
import { Panel, PanelGroup, PanelResizeHandle, ImperativePanelGroupHandle, ImperativePanelHandle } from "react-resizable-panels";
import { useSim } from "./store/sim";
import ProcessView from "./process/ProcessView";
import PhonePane from "./phone/PhonePane";

const MIN_PHONE_PX = 300;

// Root component: the resizable split between Process View and phone pane.
// Starts the store's data loading and event subscription on mount.
// Keeps the phone pane at least 300 px wide as the window resizes.
export default function App() {
  const init = useSim((s) => s.init);
  const err = useSim((s) => s.error);
  const group = useRef<ImperativePanelGroupHandle>(null);
  const phone = useRef<ImperativePanelHandle>(null);
  useEffect(() => init(), [init]);

  // phone pane: min 15% and never narrower than 300 px, so derive the percentage limits from the window width
  const [w, setW] = useState(window.innerWidth);
  useEffect(() => { const f = () => setW(window.innerWidth); window.addEventListener("resize", f); return () => window.removeEventListener("resize", f); }, []);
  const phoneMin = Math.min(50, Math.max(15, (MIN_PHONE_PX / w) * 100));
  // Enforces the phone pane's minimum size.
  // Resizes the pane up if it is narrower than the current minimum.
  // Runs whenever the minimum changes.
  const guard = () => { if ((phone.current?.getSize() ?? 100) < phoneMin) phone.current?.resize(phoneMin); };
  useEffect(guard, [phoneMin]);

  return (
    <div id="split-root" className="root">
      {err && <div className="errbar" role="alert">{err}</div>}
      <PanelGroup ref={group} direction="horizontal" autoSaveId="meltplan-split">
        <Panel defaultSize={80} minSize={50} maxSize={100 - phoneMin} order={1}><ProcessView /></Panel>
        <PanelResizeHandle className="handle" onDoubleClick={() => group.current?.setLayout([80, 20])} title="Drag to resize · double-click to reset 80/20">
          <div className="grip" />
        </PanelResizeHandle>
        <Panel ref={phone} defaultSize={20} minSize={phoneMin} maxSize={50} order={2}><PhonePane /></Panel>
      </PanelGroup>
    </div>
  );
}
