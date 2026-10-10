# Mini-ScrapChef

Demo replica of a scrap-mix optimiser for EAF steelmaking: Process View (READ → DATA → INTELLIGENCE → ACT) next to a role-based Phone View.
Plan: `Reference_Docs/Plan.md`; layer design and decision log: `Reference_Docs/DATA_AND_INTELLIGENCE_LAYER_PLAN.md`; generated reports in `Reference_Docs/reports/`.

## Run
`make setup && make fit && make backtest` (once), then `make api` and `make web` → http://localhost:5173.
Demo: **Plan 06:00** → Phone: Email → Open in ScrapChef → Approve → Shift leader: Release → Crane: GRAB DONE.
Switch the mode selector to *Simulated drift scenario*, then ▶ to watch the agent re-plan. Set `ANTHROPIC_API_KEY` for LLM narration (template fallback otherwise; numbers are checked against the facts).

## What is real, estimated, assumed, mocked (cold-water limits)
- **REAL (filtered):** the Kaggle heats: charges by mill code, final chemistry, transformer/O₂/carbon data. Shadow replay uses real held-out heats.
  **Demo simplification:** the 1,097 heats (5%) that charged any of 14 minor scrap codes (B11 and "other") are removed in `data/ingest.py`
  (list: `dropped` in `config/grade_map.yaml`), leaving 19,708 heats and a 2,514-heat analysis cohort. The first heat after each removed one is
  treated as a chain start (heel chemistry = trailing median). The raw CSVs are untouched, and every removal is logged in `dq_issues`.
- **ESTIMATED:** group Cu/Sn/Ni/Mo contents (Bayes-NNLS + block bootstrap), energy model, back-test saving. Bootstrap SDs understate model error.
- **ASSUMED:** prices, tariff, spec limits (`plant_strict` is invented to make Cu bind), tap yield, heel mass, code→group mapping, priors (research Table H2).
  The mapping uses **4 grade groups**: `SCRAP_3A` (~87% of mass), `SCRAP_16A_6A` (high Ni), `SCRAP_25A_3ACH` (low Cu) and `IRON_UNITS` (the clean dilution lever).
- **MOCK:** inventory, prices/deliveries feed, schedule, camera events, email channel.
- One mill code (`3A`) is ~87% of charged mass, so minor groups are weakly identified, and the optimiser has little room to move.
  The back-test saving (~1.1% at matched risk over 292 held-out heats; with ±20% price noise the three seeds gave 5.4%, 3.5% and -1.2%) is only as good as the assumed prices.
- Estimator checks: the synthetic recovery test runs 20 seeds with truth drawn from the prior and reports 90% CI coverage per group (pass at ≥75%).
  `SCRAP_3A`, `SCRAP_16A_6A` and `SCRAP_25A_3ACH` pass; `IRON_UNITS` fails Cu/Ni/Mo coverage because its estimate is essentially its prior (it is in only ~16% of heats), so treat its contents as assumed.
- The email and Supervisor view compare the plan against the last 30 days' actual mix (same assumed prices) in a **risk-matched** way: the historical mix, the optimiser re-solved at that mix's own
  predicted risk (the same per-element z the back-test uses), and the plan held to the 95% limits. The gap between the first two is the saving; the gap to the plan is the price of the 95% limits.
  The historical mix's breach probability (e.g. ~40% for Cu) depends on the invented `plant_strict` limits, so it illustrates the method and is not a plant finding.
- Energy: `Σ MW` is used as MWh (provisional, D13); a mm:ss integration gave implausibly small values. Confirm with the mill.
- The SIMULATED scenario (a contaminated `SCRAP_25A_3ACH` lot, Cu ×12) is scripted; lab results there come from our own plan. Closed-loop attribution is limited because the mix barely varies,
  and the Kalman filter trusts tight bootstrap SDs, so it adapts slowly. The drift targets `SCRAP_25A_3ACH` because the plan no longer uses `SCRAP_16A_6A` (its Ni is above the limit).
- The replay starts on the first day with enough heats (2018-06-02 06:00, an 8-heat plan), because the removed heats leave gaps in June 2018.
- Human approval is required at every step; the LP decides and the LLM only narrates.
