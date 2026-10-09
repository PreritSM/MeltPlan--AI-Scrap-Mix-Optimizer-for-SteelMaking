# Mini-ScrapChef

Demo replica of a scrap-mix optimiser for EAF steelmaking: Process View (READ → DATA → INTELLIGENCE → ACT) next to a role-based Phone View.
Plan: `Reference_Docs/Plan.md`; layer design and decision log: `Reference_Docs/DATA_AND_INTELLIGENCE_LAYER_PLAN.md`; generated reports in `Reference_Docs/reports/`.

## Run
`make setup && make fit && make backtest` (once), then `make api` and `make web` → http://localhost:5173.
Demo: **Plan 06:00** → Phone: Email → Open in ScrapChef → Approve → Shift leader: Release → Crane: GRAB DONE.
Switch the mode selector to *Simulated drift scenario*, then ▶ to watch the agent re-plan. Set `ANTHROPIC_API_KEY` for LLM narration (template fallback otherwise; numbers are checked against the facts).

## What is real, estimated, assumed, mocked (cold-water limits)
- **REAL:** the Kaggle heats: charges by mill code, final chemistry, transformer/O₂/carbon data. Shadow replay uses real held-out heats.
- **ESTIMATED:** group Cu/Sn/Ni/Mo contents (Bayes-NNLS + block bootstrap), energy model, back-test saving. Bootstrap SDs understate model error.
- **ASSUMED:** prices, tariff, spec limits (`plant_strict` is invented to make Cu bind), tap yield, heel mass, code→group mapping, priors (research Table H2).
- **MOCK:** inventory, prices/deliveries feed, schedule, camera events, email channel.
- One mill code (`3A`) is ~85% of charged mass, so minor groups are weakly identified, and the optimiser has little room to move.
  The back-test saving (~1.2%, price sensitivity 1–6%) is only as good as the assumed prices.
- Energy: `Σ MW` is used as MWh (provisional, D13); a mm:ss integration gave implausibly small values. Confirm with the mill.
- The SIMULATED scenario (a contaminated 16A lot) is scripted; lab results there come from our own plan. Closed-loop attribution is limited because the mix barely varies.
- Human approval is required at every step; the LP decides and the LLM only narrates.
