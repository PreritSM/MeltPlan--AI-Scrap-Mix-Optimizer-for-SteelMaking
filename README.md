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
Guards (in code, not the prompt):
- Numbers in any text must come from tool results (as given or rounded); a question answer that fails a check gets one rewrite naming the problem.
- No prices or cost figures to the crane or the shift leader (currency symbols and every cost/price/saving number from the run are blocked); the shift leader may be told that costs go to the supervisor.
- The proposal is the least-relaxed feasible candidate: one that changes the live spec class or raises α can only be an alternative while a live-settings candidate is feasible. Same-cost alternatives (within $0.05/t) are dropped.
- Anti-churn; 10 tool calls / 60 s per run; a feasible candidate that was never proposed gets one reminder, then the fallback publishes.
- After a lab result or drift, a live plan above α with no feasible candidate gets one reminder ("an unsafe plan is not no change"), then the fallback runs.
- The sim clock waits for the agent: Step heat / ▶ pause while a run is pending, so a proposal never arrives for heats that have already tapped.
The LLM client rebuilds its connection once on a connection error (a pooled connection goes stale while the demo sits idle).
Without a key, or on any LLM failure, a deterministic fallback (`policy.py`) runs the same tools. The Process View agent card shows which path ran.

LLM: any OpenAI-compatible endpoint, set in `backend/.env` (gitignored). Demo default, DeepSeek flash (supplier email: 3/3 runs on the LLM path, 25–39 s):
```
SCRAPCHEF_LLM_BASE_URL=https://api.deepseek.com
SCRAPCHEF_LLM_API_KEY=sk-...
SCRAPCHEF_LLM_MODEL=deepseek-flash
```
Alternative, GLM 5.3 via NVIDIA: `NVIDIA_API_KEY=nvapi-...` alone, plus `SCRAPCHEF_LLM_EXTRA_BODY={"reasoning_effort":"low"}` (without it GLM spends
its reply on reasoning and the proposal step times out). `SCRAPCHEF_LLM=off` forces the fallback.
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
  so a relaxed candidate reports `delta_breakdown`: the availability limit vs dropping the recipe floors. In the scripted email this is $0.00/t vs −$7.78/t (v2 at $488.76/t):
  once the 3A floor is dropped the 560 t cap no longer binds, so the whole change comes from the floor. The explanation states both, so a forced change is never shown as a plain saving. The recipe floor is taken from history; whether it is a real plant rule is a question for the mill.
- Heel chemistry for later heats is the previous heat's planned P50, iterated until it settles (≤ 6 passes, 1e-4 wt%). Two fixed passes left heats 3–8 at ~5.2% Cu risk, above α.
  The first replay plan (2018-06-02) costs $496.54/t ($492.42/t for the historical mix; the same-risk optimum is $489.88/t, so the 95% limits cost $6.65/t).
- Control Suite edits are checked for process limits as well as chemistry: a charge above the 140 t capacity cannot be saved, and a tap more than 2 t (`edit_tap_tol_t`, ASSUMED) off target is flagged.
  A hand edit over a chemistry limit is only warned about; it can still be saved and approved.
- The replay starts on the first day with enough heats (2018-06-02 06:00, an 8-heat plan), because the removed heats leave gaps in June 2018.
- Human approval is required at every step; the LP decides and the LLM only narrates.
- A plan is DONE when all its heats have tapped; ▶ then stops, and **Plan 06:00** moves on to the next day (the 06:00 plan is not automatic during ▶).

### Known gaps (end-to-end test, 2026-10-10)
- LLM wording is checked for numbers, not meaning. Seen live: 25A_3ACH called "higher-copper" (it is the low-Cu group); "148.5 t of 3A into 16A/6A" (16A/6A rose 80.2 t);
  the crane told "no pause needed" while the shift leader was told to await approval.
- DeepSeek sometimes leaves its reasoning in the reply (one rejected answer contained "Wait — need to check…"), and once sent truncated tool-call JSON (repaired on retry).
- In lab/drift runs a crane message can still go out without a plan following (the missing-proposal rule covers issue, inbox and rejection only); the sim-clock pause makes this unlikely.
- `prices.yaml` prices SCRAP_3A and SCRAP_25A_3ACH the same (430 $/t), so low-Cu scrap has no premium; that is why the plan uses ~32% 25A_3ACH.
- The binding constraints shown for a hand-edited version are its parent version's shadow prices.
- UI stubs: Reject sends a fixed reason (no reason box).
- The API is the single owner of `data/app.sqlite` (D36). Scripts and REPLs that build a `World()` get a private in-memory store; opening the file from a second process while the API runs raises an error.
- cvxpy reports "Solution may be inaccurate" on later heel passes of the exclude-3ACh option at α = 0.10; `optimal_inaccurate` is accepted (matches neighbouring passes to ~0.003%).
