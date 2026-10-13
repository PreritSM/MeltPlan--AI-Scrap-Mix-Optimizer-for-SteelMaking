# Composition estimator report (auto-generated)

Time split 70/15/15 by heat order; metrics on the last 15% (held-out). Priors are ASSUMED (published typical values, config/grade_map.yaml).

Metallic yield (assumed group yields, shared with the optimiser): **0.920** effective over the cohort. Heats with a measured previous-heat chemistry: 65% (others use the trailing median as heel chemistry).

## Cu (n=2514 heats)

| method | test MAE (wt %) | test RMSE |
|---|---|---|
| constant (train mean) | 0.0144 | 0.0202 |
| prior-only | 0.0420 | 0.0451 |
| NNLS | 0.0134 | 0.0186 |
| Bayes-NNLS | 0.0134 | 0.0186 |

Bayes-NNLS 90% PI coverage on test: **96%**. Condition number of X: 21.0

| group | mass share | prior | NNLS | Bayes-NNLS (all heats) ± sd | 90% CI | synthetic CI coverage (nominal 90%) |
|---|---|---|---|---|---|---|
| SCRAP_3A | 87.1% | 0.220±0.060 | 0.185 | 0.185 ± 0.002 | [0.183, 0.188] | 100% ok |
| SCRAP_16A_6A | 3.4% | 0.200±0.060 | 0.159 | 0.164 ± 0.017 | [0.136, 0.193] | 90% ok |
| SCRAP_25A_3ACH | 6.9% | 0.220±0.060 | 0.095 | 0.101 ± 0.005 | [0.093, 0.109] | 85% ok |
| IRON_UNITS | 2.6% | 0.005±0.002 | 0.018 | 0.006 ± 0.001 | [0.005, 0.007] | 25% FAIL |

## Sn (n=2513 heats)

| method | test MAE (wt %) | test RMSE |
|---|---|---|
| constant (train mean) | 0.0010 | 0.0013 |
| prior-only | 0.0075 | 0.0076 |
| NNLS | 0.0009 | 0.0012 |
| Bayes-NNLS | 0.0009 | 0.0012 |

Bayes-NNLS 90% PI coverage on test: **97%**. Condition number of X: 21.0

| group | mass share | prior | NNLS | Bayes-NNLS (all heats) ± sd | 90% CI | synthetic CI coverage (nominal 90%) |
|---|---|---|---|---|---|---|
| SCRAP_3A | 87.1% | 0.015±0.007 | 0.007 | 0.007 ± 0.000 | [0.007, 0.007] | 100% ok |
| SCRAP_16A_6A | 3.4% | 0.010±0.005 | 0.008 | 0.007 ± 0.001 | [0.006, 0.009] | 90% ok |
| SCRAP_25A_3ACH | 6.9% | 0.015±0.007 | 0.003 | 0.003 ± 0.000 | [0.002, 0.004] | 80% ok |
| IRON_UNITS | 2.6% | 0.002±0.001 | 0.000 | 0.000 ± 0.000 | [0.000, 0.000] | 75% ok |

## Ni (n=2514 heats)

| method | test MAE (wt %) | test RMSE |
|---|---|---|
| constant (train mean) | 0.0149 | 0.0171 |
| prior-only | 0.0256 | 0.0283 |
| NNLS | 0.0124 | 0.0151 |
| Bayes-NNLS | 0.0125 | 0.0151 |

Bayes-NNLS 90% PI coverage on test: **95%**. Condition number of X: 21.0

| group | mass share | prior | NNLS | Bayes-NNLS (all heats) ± sd | 90% CI | synthetic CI coverage (nominal 90%) |
|---|---|---|---|---|---|---|
| SCRAP_3A | 87.1% | 0.080±0.040 | 0.117 | 0.114 ± 0.001 | [0.112, 0.116] | 100% ok |
| SCRAP_16A_6A | 3.4% | 0.100±0.050 | 0.337 | 0.328 ± 0.020 | [0.297, 0.359] | 90% ok |
| SCRAP_25A_3ACH | 6.9% | 0.080±0.040 | 0.068 | 0.074 ± 0.005 | [0.066, 0.083] | 85% ok |
| IRON_UNITS | 2.6% | 0.010±0.005 | 0.005 | 0.014 ± 0.002 | [0.010, 0.018] | 50% FAIL |

## Mo (n=2514 heats)

| method | test MAE (wt %) | test RMSE |
|---|---|---|
| constant (train mean) | 0.0057 | 0.0076 |
| prior-only | 0.0059 | 0.0078 |
| NNLS | 0.0053 | 0.0073 |
| Bayes-NNLS | 0.0055 | 0.0075 |

Bayes-NNLS 90% PI coverage on test: **96%**. Condition number of X: 21.0

| group | mass share | prior | NNLS | Bayes-NNLS (all heats) ± sd | 90% CI | synthetic CI coverage (nominal 90%) |
|---|---|---|---|---|---|---|
| SCRAP_3A | 87.1% | 0.020±0.010 | 0.020 | 0.019 ± 0.001 | [0.018, 0.021] | 85% ok |
| SCRAP_16A_6A | 3.4% | 0.020±0.010 | 0.045 | 0.037 ± 0.006 | [0.028, 0.048] | 80% ok |
| SCRAP_25A_3ACH | 6.9% | 0.020±0.010 | 0.009 | 0.018 ± 0.007 | [0.009, 0.031] | 75% ok |
| IRON_UNITS | 2.6% | 0.005±0.003 | 0.000 | 0.005 ± 0.002 | [0.002, 0.007] | 40% FAIL |
