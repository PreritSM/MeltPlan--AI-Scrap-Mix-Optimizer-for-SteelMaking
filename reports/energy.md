# Energy model report (auto-generated)

Target: MWh per heat = Σ MW over transformer rows (**provisional** definition, to be confirmed with the plant). Time split 80/20.

| model | test R² | test MAE (MWh) |
|---|---|---|
| total-charge only | 0.338 | 1.53 |
| per-group | 0.350 | 1.52 |

ΔR² (per-group − total) = **0.012** → collapsed to constant MWh/t.

| group | MWh/t input | $/t (tariff $90/MWh, ASSUMED) |
|---|---|---|
| SCRAP_3A | 0.275 | 24.73 |
| SCRAP_16A_6A | 0.275 | 24.73 |
| SCRAP_25A_3ACH | 0.275 | 24.73 |
| IRON_UNITS | 0.275 | 24.73 |