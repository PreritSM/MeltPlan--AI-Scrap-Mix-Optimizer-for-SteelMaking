# Back-test report (auto-generated)

Held-out heats solved: **370** (estimator fit on train split only). Same-residual-risk comparison (z matched per element, floor 0.5).

**Headline: 0.02% cheaper at the same predicted residual risk** (95% bootstrap CI [0.01%, 0.03%]; $0.10/t steel).

Sensitivity: random ±20% group-price perturbations (3 seeds, 150 heats each): 3.36%, 1.74%, -0.31%.

**Caveat:** the absolute figure depends on ASSUMED prices (config/prices.yaml) and an availability proxy (stock = max usage within ±1 day). The mill's mix is ~85% one code, so there is little room to move; the relative result matters more than the level.

| group | mean tonnes historical | mean tonnes LP |
|---|---|---|
| SCRAP_3A | 112.7 | 111.0 |
| SCRAP_16A_6A | 2.1 | 2.3 |
| SCRAP_25A_3ACH | 11.8 | 13.1 |
| IRON_UNITS | 0.0 | 0.0 |