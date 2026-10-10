# Mini-ScrapChef

Demo replica of a scrap-mix optimiser for EAF steelmaking: Process View (READ → DATA → INTELLIGENCE → ACT) next to a role-based Phone View.
Plan: `Reference_Docs/Plan.md`; layer design and decision log: `Reference_Docs/DATA_AND_INTELLIGENCE_LAYER_PLAN.md`; generated reports in `Reference_Docs/reports/`.

## Run
`make setup && make fit && make backtest` (once), then `make api` and `make web` → http://localhost:5173.
Demo: **Plan 06:00** → Phone: Email → Open in ScrapChef → Approve → Shift leader: Release → Crane: GRAB DONE.
Switch the mode selector to *Simulated drift scenario*, then ▶ to watch the agent re-plan, or press **📧 Supplier email** for the 3A-shortage scenario.

## Re-planning agent (LLM)
The agent (`backend/scrapchef/intelligence/agent/`) is a tool-use loop: the LP decides, the LLM chooses optimiser settings, explores options and explains.
It wakes on lab results (surprise or above the limit), drift, free-text issues, rejections, questions and inbound emails; the 06:00 plan stays deterministic.
Tools: `get_state`, `run_optimizer`, `evaluate_mix`, `compare_plans`, `propose_plan` (needs supervisor approval), `send_message`.
Guards: numbers in any text must come from tool results, no prices to the crane, anti-churn, 8 tool calls / 60 s per run.
Without a key, or on any LLM failure, a deterministic fallback (`policy.py`) runs the same tools. The Process View agent card shows which path ran.

LLM: GLM 5.3 through NVIDIA's OpenAI-compatible endpoint. Put the key in `backend/.env` (gitignored):
```
NVIDIA_API_KEY=nvapi-...
# optional: SCRAPCHEF_LLM_MODEL=z-ai/glm-5.3  SCRAPCHEF_LLM_BASE_URL=https://integrate.api.nvidia.com/v1  SCRAPCHEF_LLM=off
```
Tests never call the LLM (`SCRAPCHEF_LLM=off` in `tests/conftest.py`).

## What is real, estimated, assumed, mocked (cold-water limits)
- **REAL (filtered):** the Kaggle heats: charges by mill code, final chemistry, transformer/O₂/carbon data. Shadow replay uses real held-out heats.
  **Demo simplification:** the 1,097 heats (5%) that charged any of 14 minor scrap codes (B11 and "other") are removed in `data/ingest.py`
  (list: `dropped` in `config/grade_map.yaml`), leaving 19,708 heats and a 2,514-heat analysis cohort. The first heat after each removed one is
  treated as a chain start (heel chemistry = trailing median). The raw CSVs are untouched, and every removal is logged in `dq_issues`.
- **ESTIMATED:** group Cu/Sn/Ni/Mo contents (Bayes-NNLS + block bootstrap), energy model, back-test saving. Bootstrap SDs understate model error.
- **ASSUMED:** prices, tariff, spec limits (`plant_strict` is invented to make Cu bind), group yields, heel mass, code→group mapping, priors (research Table H2).
  One tap model is used everywhere: tap = Table H2 group yields · charge (effective yield ≈ 0.92) in the estimator, the optimiser and the replay.
  The planned tap is 117 t per heat, the median of what the cohort's charges make, not the 125 t nameplate. The optimiser's tap window is ±0.05 t, and
  chemistry and cost per tonne use the tap the charge actually makes (see `Reference_Docs/Tap_tol_error.md` for the ±1 t bias this removed).
  The mapping uses **4 grade groups**: `SCRAP_3A` (~87% of mass), `SCRAP_16A_6A` (high Ni), `SCRAP_25A_3ACH` (low Cu) and `IRON_UNITS` (the clean dilution lever).
- **MOCK:** inventory, prices/deliveries feed, schedule, camera events, email channel.
- One mill code (`3A`) is ~87% of charged mass, so minor groups are weakly identified, and the optimiser has little room to move.
  The back-test saving (0.25% at matched risk, 95% CI 0.23–0.27%, about $1.25/t, over 370 held-out heats; with ±20% price noise the three seeds gave 3.9%, 2.4% and -0.2%)
  is only as good as the assumed prices. An earlier ±1 t tap window inflated it to ~1.1%.
- Estimator checks: the synthetic recovery test runs 20 seeds with truth drawn from the prior and reports 90% CI coverage per group (pass at ≥75%).
  `SCRAP_3A`, `SCRAP_16A_6A` and `SCRAP_25A_3ACH` pass; `IRON_UNITS` fails Cu/Ni/Mo coverage because its estimate is essentially its prior (it is in only ~16% of heats), so treat its contents as assumed.
- The email and Supervisor view compare the plan against the last 30 days' actual mix (same assumed prices) in a **risk-matched** way: the historical mix, the optimiser re-solved at that mix's own
  predicted risk (the same per-element z the back-test uses), and the plan held to the 95% limits. The gap between the first two is the saving; the gap to the plan is the price of the 95% limits.
  The historical mix's breach probability (e.g. ~24% for Cu on the first replay day) depends on the invented `plant_strict` limits, so it illustrates the method and is not a plant finding.
- Energy: `Σ MW` is used as MWh (provisional, D13); a mm:ss integration gave implausibly small values. Confirm with the mill.
- The SIMULATED scenario (a copper-heavy 3A lot, Cu ×1.5) is scripted; lab results there come from our own plan. The Kalman filter's starting variance is floored
  (bootstrap SDs are optimistic), but closed-loop attribution to minor groups stays weak because every planned heat uses nearly the same mix; 3A, the dominant group, is identifiable.
- The supplier-email scenario caps 3A below the recipe floor, so the plain re-plan is infeasible and the agent explores. "Relax recipe bounds" only drops minimum shares and allows
  more iron units; minor-scrap maximums stay at demonstrated practice. Dropping the 3A floor is itself cheaper under the model (16A_6A displaces 3A and less iron is needed),
  so a relaxed candidate reports `delta_breakdown`: the availability limit (+$0.22/t in the scripted email) vs dropping the recipe floors (-$7.63/t). The explanation states both,
  so a forced change is never shown as a plain saving. The recipe floor is taken from history; whether it is a real plant rule is a question for the mill.
- The replay starts on the first day with enough heats (2018-06-02 06:00, an 8-heat plan), because the removed heats leave gaps in June 2018.
- Human approval is required at every step; the LP decides and the LLM only narrates.
