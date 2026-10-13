# Data profile (auto-generated)

Source: `EAF Kaggle Dataset/`

## 1. Files

| file | rows | distinct HEATID | columns |
|---|---|---|---|
| basket_charged | 137685 | 20806 | MAT_CODE, MAT_DEC, HEATID, DATETIME, CHARGED_AMOUNT |
| eaf_added_materials | 79944 | 20507 | MAT_CODE, MAT_DEC, HEATID, DATETIME, CHARGE_AMOUNT |
| eaf_final_chemical_measurements | 3709 | 3709 | HEATID, POSITIONROW, DATETIME, VALC, VALSI, VALMN, VALP, VALS, VALCU, VALCR, VALMO, VALNI, VALAS, VALSN, VALN |
| eaf_gaslance_mat | 5748194 | 20827 | REVTIME, O2_AMOUNT, GAS_AMOUNT, O2_FLOW, GAS_FLOW, HEATID |
| eaf_temp | 85104 | 20827 | HEATID, DATETIME, TEMP, VALO2_PPM |
| eaf_transformer | 271712 | 20813 | TAP, HEATID, STARTTIME, DURATION, MW |
| ferro | 52 |  | column00, Description, Slagging, Nb, Ta, Mo, V, Ca, Al, Mg, P, S, Ash, Moisture, C, Si, Mn, Cr, Ti, Cu, Zn, Ni, Pb, Sn, Al2O3, CaO+MgO, CaO, SiO2, MgO, FeO, Fe2O3, Cr2O3, TiO2, Na+K, TiO, As, SiC, Zr, MAT_CODE |
| inj_mat | 4011646 | 20827 | REVTIME, INJ_AMOUNT_CARBON, INJ_FLOW_CARBON, HEATID |
| ladle_tapping | 115873 | 20827 | HEATID, MAT_CODE, MAT_DEC, CHARGE_AMOUNT, DATETIME |
| lf_added_materials | 302445 | 20374 | FILTER_KEY_DATE, HEATID, MAT_CODE, DESCR, MAT_CHARGED |
| lf_initial_chemical_measurements | 20827 | 20827 | HEATID, POSITIONROW, DATETIME, VALC, VALSI, VALMN, VALP, VALS, VALCU, VALCR, VALMO, VALNI |

## 2. Join coverage (distinct HEATIDs)

| file \ ∩ | basket_charged | eaf_added_materials | eaf_final_chemical_measurements | eaf_gaslance_mat | eaf_temp | eaf_transformer | inj_mat | ladle_tapping | lf_added_materials | lf_initial_chemical_measurements |
|---|---|---|---|---|---|---|---|---|---|---|
| basket_charged | 20806 | 20486 | 3698 | 20806 | 20806 | 20796 | 20806 | 20806 | 20354 | 20806 |
| eaf_added_materials | 20486 | 20507 | 3688 | 20506 | 20506 | 20494 | 20506 | 20506 | 20069 | 20506 |
| eaf_final_chemical_measurements | 3698 | 3688 | 3709 | 3709 | 3709 | 3705 | 3709 | 3709 | 3636 | 3709 |
| eaf_gaslance_mat | 20806 | 20506 | 3709 | 20827 | 20827 | 20813 | 20827 | 20827 | 20374 | 20827 |
| eaf_temp | 20806 | 20506 | 3709 | 20827 | 20827 | 20813 | 20827 | 20827 | 20374 | 20827 |
| eaf_transformer | 20796 | 20494 | 3705 | 20813 | 20813 | 20813 | 20813 | 20813 | 20360 | 20813 |
| inj_mat | 20806 | 20506 | 3709 | 20827 | 20827 | 20813 | 20827 | 20827 | 20374 | 20827 |
| ladle_tapping | 20806 | 20506 | 3709 | 20827 | 20827 | 20813 | 20827 | 20827 | 20374 | 20827 |
| lf_added_materials | 20354 | 20069 | 3636 | 20374 | 20374 | 20360 | 20374 | 20374 | 20374 | 20374 |
| lf_initial_chemical_measurements | 20806 | 20506 | 3709 | 20827 | 20827 | 20813 | 20827 | 20827 | 20374 | 20827 |

**Analysis cohort** (basket ∩ final chem ∩ transformer): **3697** heats

## 3. Charged amounts (units)

| heats | p0_5 | p50 | p99_5 | median_baskets |
|---|---|---|---|---|
| 20806 | 80306.225 | 136021.40000000002 | 145973.34999999995 | 2.0 |

(If p50 ≈ 130000–150000 the unit is kg; plan expects 130–150 t per heat.)

## 4. Material codes in basket_charged (Pareto)

| MAT_CODE | mat_name | tonnes | heats | pct | cum_pct |
|---|---|---|---|---|---|
| 205051 | 3A | 2423727.0 | 20795 | 86.64 | 86.6 |
| 205180 | 16A | 105640.0 | 12928 | 3.78 | 90.4 |
| 360258 | In-house produced lime | 71993.0 | 15314 | 2.57 | 93.0 |
| 601025 | 25A1 | 41056.0 | 4192 | 1.47 | 94.5 |
| 205237 | 3ACh | 34378.0 | 1615 | 1.23 | 95.7 |
| 205032 | Ferrous scrap 6A | 21659.0 | 3158 | 0.77 | 96.5 |
| 140107 | Lime | 17993.0 | 4096 | 0.64 | 97.1 |
| 201010 | Steelmaking iron | 16588.0 | 1059 | 0.59 | 97.7 |
| 601038 | Scrap G/B 25A | 11771.0 | 1806 | 0.42 | 98.1 |
| 205179 | B11 | 10692.0 | 315 | 0.38 | 98.5 |
| 201003 | Hot briquet iron | 8480.0 | 981 | 0.3 | 98.8 |
| 205185 | 3ACh | 8289.0 | 366 | 0.3 | 99.1 |
| 203068 | Coke 10-30 mm | 5501.0 | 6390 | 0.2 | 99.3 |
| 601008 | Steel scrap G/B 25A | 5261.0 | 622 | 0.19 | 99.5 |
| 602022 | 25A | 4096.0 | 834 | 0.15 | 99.6 |
| 602031 | Scrap G/B 25AS | 2558.0 | 519 | 0.09 | 99.7 |
| 205236 | Iron scrap | 1568.0 | 476 | 0.06 | 99.8 |
| 602017 | Separation waste 3A | 1296.0 | 136 | 0.05 | 99.8 |
| 601043 | Briquets 6A | 1100.0 | 315 | 0.04 | 99.9 |
| 608017 | Separation scrap | 784.0 | 127 | 0.03 | 99.9 |
| 364301 | Lime, grade 2 | 712.0 | 144 | 0.03 | 99.9 |
| 205240 | Scrap B23 | 652.0 | 54 | 0.02 | 99.9 |
| 205044 | Liquid scrap | 542.0 | 21 | 0.02 | 100.0 |
| 705046 | FOMI B2 | 187.0 | 151 | 0.01 | 100.0 |
| 203090 | Artificial graphite 0.1-2.5 mm | 144.0 | 183 | 0.01 | 100.0 |
| 205191 | 26A | 135.0 | 15 | 0.0 | 100.0 |
| 705043 | Magma | 117.0 | 105 | 0.0 | 100.0 |
| 205181 | 3B23 | 88.0 | 10 | 0.0 | 100.0 |
| 602014 | Ladle scrap | 86.0 | 13 | 0.0 | 100.0 |
| 202059 | Mo pellets | 80.0 | 66 | 0.0 | 100.0 |
| 602023 | 26A | 69.0 | 14 | 0.0 | 100.0 |
| 603168 | B68 | 65.0 | 21 | 0.0 | 100.0 |
| 205045 | Shram 75 | 54.0 | 4 | 0.0 | 100.0 |
| 203120 | Flux FIM | 49.0 | 12 | 0.0 | 100.0 |
| 601024 | Scrap 3B22 | 42.0 | 4 | 0.0 | 100.0 |
| 202007 | MnSi17A | 39.0 | 39 | 0.0 | 100.0 |
| 202056 | Vanadium alloy | 18.0 | 3 | 0.0 | 100.0 |
| 202008 | FeSi65-4 | 17.0 | 16 | 0.0 | 100.0 |
| 205069 | Secondary aluminum AV 87 | 12.0 | 155 | 0.0 | 100.0 |
| 603111 | B11 | 12.0 | 4 | 0.0 | 100.0 |


Distinct codes: 50. Names per code >1: 0

## 5. Baskets

Distinct DATETIME per heat treated as a basket (to be verified):

| pos | n | p10 | p50 | p90 |
|---|---|---|---|---|
| 1 | 20806 | 72359.5 | 80424.5 | 84063.0 |
| 2 | 20638 | 47286.1 | 56410.0 | 62015.0 |
| 3 | 1283 | -732.3999999999969 | 4000.0 | 56153.0 |
| 4 | 258 | -57853.8 | -60.0 | 23444.300000000017 |
| 5 | 59 | -41938.8 | 1000.0 | 55207.20000000001 |
| 6 | 17 | -70003.79999999999 | 130.0 | 10632.000000000005 |

## 6. Final chemistry (EAF)

Samples per row index:

| POSITIONROW | n |
|---|---|
| 1 | 3709 |


| element | n | p1 | p50 | p99 |
|---|---|---|---|---|
| CU | 3341 | 0.1209 | 0.1883 | 0.271 |
| SN | 3340 | 0.0036 | 0.007 | 0.0131 |
| NI | 3341 | 0.0872 | 0.1204 | 0.2175 |
| MO | 3341 | 0.0095 | 0.0162 | 0.1706 |
| CR | 3341 | 0.0391 | 0.0763 | 0.2521 |
| C | 3341 | 0.0331 | 0.0662 | 0.1656 |
| SI | 3341 | 0.0142 | 0.0181 | 0.0389 |
| MN | 3341 | 0.0499 | 0.0975 | 0.2348 |
| P | 3340 | 0.0033 | 0.0063 | 0.0201 |
| S | 3341 | 0.0147 | 0.0319 | 0.0653 |

## 7. Transformer / energy

| DURATION | MW | TAP |
|---|---|---|
|  00: 29 | 0,5 | 11 |
|  10: 40 | 16 | 17 |
|  03: 00 | 11 | 17 |
|  13: 20 | 16,75 | 17 |
|  00: 40 | 0,75 | 17 |
|  00: 10 | 0,375 | 16 |
|  00: 10 | 0 | 13 |
|  00: 20 | 0,625 | 15 |

DURATION is text like ' 10: 40' (mm:ss or hh:mm — verify).

## 8. Time ordering and heat gaps

| first_heat | last_heat | heats | gap_p50_min | gap_p99_min | gaps_over_24h | heats_per_day |
|---|---|---|---|---|---|---|
| 2015-01-01 00:51:00 | 2018-07-29 08:34:00 | 20806 | 52.0 | 448.0 | 67 | 15.93 |

## 9. Tap-weight search

| file | candidate column |
|---|---|
| eaf_transformer | TAP |

## 10. ferro.csv vs scrap codes

| MAT_CODE | Description | Cu | Sn | Ni | Mo |
|---|---|---|---|---|---|
| 140107 | Lime | 0.0 | 0.0 | 0.0 | 0.0 |
| 202001 | Ferroniobium FeNb 60 | 0.0 | 0.0 | 0.0 | 0.0 |
| 202007 | MnSi17A | 0.0 | 0.0 | 0.0 | 0.0 |
| 202008 | FeSi65-4 | 0.0 | 0.0 | 0.0 | 0.0 |
| 202017 | FeSi75 | 0.0 | 0.0 | 0.0 | 0.0 |
| 202031 | FeV-50 | 0.2 | 0.0 | 0.0 | 0.0 |
| 202033 | Ferromolybdenum FeMo 60 | 0.5 | 0.0 | 0.0 | 60.0 |
| 202034 | FeCr025 | 0.0 | 0.0 | 0.0 | 0.0 |
| 202035 | Ferrotitanium FeTi 70 | 0.0 | 0.0 | 0.0 | 0.0 |
| 202039 | FeCr 850A | 0.0 | 0.0 | 0.0 | 0.0 |
| 202044 | Flux core wire FeV 50 | 0.2 | 0.0 | 0.0 | 0.0 |
| 202049 | PP-SK30 wire | 0.0 | 0.0 | 0.0 | 0.0 |
| 202050 | Ferrotitanium wire | 3.0 | 0.08 | 0.0 | 1.0 |
| 202052 | Ferrovanadium FeV 80 | 0.1 | 0.0 | 0.15 | 0.0 |
| 202056 | Vanadium alloy | 0.1 | 0.0 | 0.5 | 0.0 |
| 202058 | Niobium allow FeNbSB | 0.0 | 0.0 | 0.0 | 0.0 |
| 202063 | KRS-65 briquets | 0.0 | 0.0 | 0.0 | 0.0 |
| 202064 | Ferrotitanium FeTi 35 | 3.0 | 0.08 | 0.0 | 1.0 |
| 202065 | Ferromolybdenum FeMo 50 | 2.0 | 0.1 | 0.0 | 50.0 |
| 202067 | High-carbon ferrochromium | 0.0 | 0.0 | 0.0 | 0.0 |
| 202068 | FeSiMn17 | 0.0 | 0.0 | 0.0 | 0.0 |
| 202069 | Aluminum briquet | 3.7773359840954273 | 0.1988071570576541 | 0.0 | 0.0 |
| 202070 | Ferromolybdenum wire | 0.0 | 0.0 | 0.0 | 58.0 |
| 202071 | Ferroniobium wire | 0.0 | 0.0 | 0.0 | 0.0 |
| 202074 | FeSiV 5 | 0.0 | 0.0 | 0.0 | 0.0 |
| 202075 | FeSiCr 40 | 0.0 | 0.0 | 0.0 | 0.0 |
| 202093 | FeSi 45 | 0.0 | 0.0 | 0.0 | 0.0 |
| 203068 | Coke 10-30 mm | 0.0 | 0.0 | 0.0 | 0.0 |
| 203074 | C 0.5-2.5 mm | 0.0 | 0.0 | 0.0 | 0.0 |
| 203090 | Artificial graphite 0.1-2.5 mm | 0.0 | 0.0 | 0.0 | 0.0 |
| 203109 | Flux FM-2-1 | 0.0 | 0.0 | 0.0 | 0.0 |
| 203116 | CCM 03-10mm | 0.0 | 0.0 | 0.0 | 0.0 |
| 203124 | CCM 3-10mm | 0.0 | 0.0 | 0.0 | 0.0 |
| 205065 | ATF-75 | 0.0 | 0.0 | 0.0 | 0.0 |
| 205069 | Secondary aluminum AV 87 | 3.7773359840954273 | 0.1988071570576541 | 0.0 | 0.0 |
| 205075 | Aluminum pyramids AV-87 | 3.7773359840954273 | 0.1988071570576541 | 0.0 | 0.0 |
| 205153 | AV-87 briquets | 3.7773359840954273 | 0.1988071570576541 | 0.0 | 0.0 |
| 205159 | Pellets AV-87 | 3.7773359840954273 | 0.1988071570576541 | 0.0 | 0.0 |
| 205194 | KShU 75 | 0.0 | 0.0 | 0.0 | 0.0 |
| 205202 | Aluminous flux | 0.0 | 0.0 | 0.0 | 0.0 |
| 205219 | Marsh-75 | 0.0 | 0.0 | 0.0 | 0.0 |
| 205220 | ATF-7B | 0.0 | 0.0 | 0.0 | 0.0 |
| 205232 | Secondary aluminum AVK-65F | 0.0 | 0.0 | 0.0 | 0.0 |
| 205234 | ATF-A | 0.0 | 0.0 | 0.0 | 0.0 |
| 205235 | Pellets AGV | 0.0 | 0.0 | 0.0 | 0.0 |
| 205270 | FGM material | 0.0 | 0.0 | 0.0 | 0.0 |
| 205271 | Flux MVP | 0.0 | 0.0 | 0.0 | 0.0 |
| 323087 | Al wire (APM) | 0.2991026919242273 | 0.0 | 0.1994017946161515 | 0.0 |
| 323141 | Aluminum rolled wire AKLP | 0.0 | 0.0 | 0.0 | 0.0 |
| 360258 | In-house produced lime | 0.0 | 0.0 | 0.0 | 0.0 |
| 500314 | Catalyst KV-1, 10-20% | 0.0 | 0.0 | 0.0 | 2.0 |
| 705057 | SShU-1G | 0.0 | 0.0 | 0.0 | 0.0 |


ferro codes appearing in basket_charged: **11**

## 11. eaf_added_materials top codes

| MAT_CODE | mat_name | n | tonnes |
|---|---|---|---|
| 203068 | Coke 10-30 mm | 29625 | 22529.4 |
| 705043 | Magma | 15710 | 8994.3 |
| 705046 | FOMI B2 | 12949 | 8254.0 |
| 360258 | In-house produced lime | 11173 | 7665.4 |
| 140107 | Lime | 5283 | 6606.1 |
| 205044 | Liquid scrap | 550 | 4246.0 |
| 203109 | Flux FM-2-1 | 1952 | 1095.8 |
| 705059 | Magma-P | 1157 | 686.8 |
| 205051 | 3A | 21 | 338.4 |
| 705047 | Magnesite brick scrap | 213 | 139.8 |
| 601008 | Steel scrap G/B 25A | 15 | 136.1 |
| 202007 | MnSi17A | 175 | 122.5 |
| 203101 | CCM 30-60 mm | 60 | 89.3 |
| 205069 | Secondary aluminum AV 87 | 411 | 69.4 |
| 203120 | Flux FIM | 58 | 43.4 |
