# regfast benchmark results

Generated 2026-10-11T12:50:09 by `run.py` from `results/results.csv`.
Times are medians of the timed runs in seconds (`ms` = milliseconds); estimation call only, data in memory.
`ratio` = regfast / fastest finished competitor. Gate: PASS iff regfast is faster than every competitor that
finished (a competitor timeout counts as slower); `n/a` = regfast not run. `error`/`timeout`/`unsupported`
cells are explained in results.csv. Coefficient gap: max relative difference over shared non-constant
coefficients versus regfast (versus fixest when regfast has no result for the row).

**Gate summary:** 0 PASS, 0 FAIL.

## G1 OLS (fixest DGP)

| config | T | reghdfe | xhdfe | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `fx_difficult_10000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 18.0ms | 9.0ms | 5.0ms | 6.6ms | 7.2ms |  | n/a |
| `fx_difficult_10000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 19.0ms | 6.0ms | 5.0ms | 6.4ms | 7.2ms |  | n/a |
| `fx_difficult_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 53.0ms | 11.0ms | 31.0ms | 9.8ms | 11.9ms |  | n/a |
| `fx_difficult_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 40.0ms | 8.0ms | 6.0ms | 10.7ms | 10.6ms |  | n/a |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 20.0ms | 11.0ms | 7.0ms | 8.0ms | 8.5ms |  | n/a |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 24.0ms | 7.0ms | 6.0ms | 8.8ms | 13.6ms |  | n/a |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 43.0ms | 13.0ms | 8.0ms | 10.4ms | 11.1ms |  | n/a |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 58.0ms | 15.0ms | 8.0ms | 18.6ms | 13.9ms |  | n/a |
| `fx_difficult_100000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 0.131 | 98.0ms | 9.0ms | 23.9ms | 33.3ms |  | n/a |
| `fx_difficult_100000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 0.145 | 23.0ms | 11.0ms | 23.6ms | 30.1ms |  | n/a |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.852 | 0.139 | 67.0ms | 0.533 | 57.2ms |  | n/a |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.985 | 0.220 | 0.108 | 0.710 | 62.7ms |  | n/a |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 0.145 | 74.0ms | 14.0ms | 36.0ms | 56.5ms |  | n/a |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 0.275 | 38.0ms | 20.0ms | 29.5ms | 47.5ms |  | n/a |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 1.177 | 0.275 | 75.0ms | 0.590 | 71.0ms |  | n/a |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 2.184 | 0.573 | 0.332 | 2.126 | 0.109 |  | n/a |
| `fx_simple_10000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 18.0ms | 7.0ms | 5.0ms | 6.4ms | 7.0ms |  | n/a |
| `fx_simple_10000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 18.0ms | 6.0ms | 5.0ms | 6.9ms | 6.9ms |  | n/a |
| `fx_simple_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.306 | 0.250 | 23.0ms | 9.3ms | 12.6ms |  | n/a |
| `fx_simple_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 37.0ms | 8.0ms | 6.0ms | 10.0ms | 10.0ms |  | n/a |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 20.0ms | 8.0ms | 5.0ms | 8.3ms | 8.0ms |  | n/a |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 47.0ms | 10.0ms | 12.0ms | 11.4ms | 23.1ms |  | n/a |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 55.0ms | 10.0ms | 7.0ms | 8.8ms | 11.1ms |  | n/a |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 46.0ms | 9.0ms | 7.0ms | 10.1ms | 13.3ms |  | n/a |
| `fx_simple_100000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 0.128 | 0.213 | 9.0ms | 24.3ms | 31.7ms |  | n/a |
| `fx_simple_100000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 0.138 | 22.0ms | 9.0ms | 22.3ms | 26.3ms |  | n/a |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.286 | 0.220 | 43.0ms | 33.9ms | 64.0ms |  | n/a |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.298 | 26.0ms | 11.0ms | 37.5ms | 64.5ms |  | n/a |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 0.205 | 0.165 | 13.0ms | 32.6ms | 41.7ms |  | n/a |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 0.181 | 35.0ms | 17.0ms | 28.8ms | 38.4ms |  | n/a |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.311 | 38.0ms | 14.0ms | 32.1ms | 70.7ms |  | n/a |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.405 | 43.0ms | 25.0ms | 38.2ms | 98.2ms |  | n/a |

Coefficient gap:

| config | T | vs | reghdfe | xhdfe | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|---:|
| `fx_difficult_10000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 8.1e-16 | 1.2e-15 |  | 8.1e-16 | 8.1e-16 |
| `fx_difficult_10000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 1.5e-15 | 1.2e-15 |  | 8.1e-16 | 8.1e-16 |
| `fx_difficult_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 2.5e-13 | 2.5e-13 |  | 3.9e-12 | 2.5e-13 |
| `fx_difficult_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 2.5e-13 | 2.5e-13 |  | 4.6e-13 | 2.5e-13 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 5.4e-15 | 3.6e-15 |  | 4.6e-15 | 4.6e-15 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 5.4e-15 | 3.6e-15 |  | 4.6e-15 | 4.2e-15 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.8e-12 | 1.8e-12 |  | 1.2e-11 | 1.8e-12 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 1.8e-12 | 1.8e-12 |  | 1.2e-11 | 1.8e-12 |
| `fx_difficult_100000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 1.4e-14 | 7.0e-15 |  | 7.0e-15 | 6.8e-15 |
| `fx_difficult_100000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 1.4e-14 | 7.0e-15 |  | 6.6e-15 | 7.3e-15 |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 3.3e-10 | 3.3e-10 |  | 6.8e-08 | 3.3e-10 |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 3.3e-10 | 3.3e-10 |  | 6.8e-08 | 3.3e-10 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 3.4e-14 | 2.1e-14 |  | 2.1e-14 | 2.2e-14 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 3.4e-14 | 2.1e-14 |  | 2.1e-14 | 2.2e-14 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.7e-09 | 1.7e-09 |  | 7.9e-07 | 1.7e-09 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 1.7e-09 | 1.7e-09 |  | 7.9e-07 | 1.7e-09 |
| `fx_simple_10000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 3.5e-15 | 2.8e-15 |  | 2.6e-15 | 2.4e-15 |
| `fx_simple_10000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 3.3e-15 | 2.8e-15 |  | 2.6e-15 | 2.4e-15 |
| `fx_simple_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 5.5e-15 | 5.0e-15 |  | 5.2e-15 | 4.4e-15 |
| `fx_simple_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 5.7e-15 | 5.0e-15 |  | 5.0e-15 | 4.4e-15 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 6.8e-15 | 7.0e-15 |  | 6.8e-15 | 6.7e-15 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 6.7e-15 | 7.0e-15 |  | 6.8e-15 | 6.7e-15 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 5.0e-15 | 5.2e-15 |  | 5.0e-15 | 5.8e-15 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 4.7e-15 | 5.2e-15 |  | 6.1e-15 | 5.8e-15 |
| `fx_simple_100000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.5e-14 | 8.1e-15 |  | 7.7e-15 | 7.7e-15 |
| `fx_simple_100000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 2.4e-14 | 8.1e-15 |  | 7.7e-15 | 8.1e-15 |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 2.1e-14 | 8.2e-15 |  | 9.0e-15 | 9.3e-15 |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 2.0e-14 | 8.2e-15 |  | 8.6e-15 | 9.9e-15 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 3.3e-14 | 2.5e-14 |  | 2.5e-14 | 2.5e-14 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 3.3e-14 | 2.5e-14 |  | 2.5e-14 | 2.4e-14 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.1e-14 | 1.9e-14 |  | 1.9e-14 | 1.9e-14 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 9.8e-15 | 1.9e-14 |  | 1.9e-14 | 1.9e-14 |

## G2 AKM

| config | T | reghdfe | xhdfe | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `akm_scale_1_10000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.141 | 31.0ms | 13.0ms | 70.4ms | 12.4ms |  | n/a |
| `akm_scale_1_10000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.100 | 20.0ms | 20.0ms | 0.119 | 13.2ms |  | n/a |

Coefficient gap:

| config | T | vs | reghdfe | xhdfe | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|---:|
| `akm_scale_1_10000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 5.6e-12 | 5.6e-12 |  | 1.2e-10 | 5.6e-12 |
| `akm_scale_1_10000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 5.7e-12 | 5.6e-12 |  | 1.2e-10 | 5.6e-12 |

## G4 IV/2SLS

| config | T | ivreghdfe | xhdfe | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 0.209 | 38.0ms | 26.0ms | 54.6ms | 0.106 |  | n/a |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | 0.283 | 35.0ms | 26.0ms | 56.2ms | 67.2ms |  | n/a |

Coefficient gap:

| config | T | vs | ivreghdfe | xhdfe | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|---:|
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fixest | 2.2e-14 | 2.5e-14 |  | 2.2e-14 | 2.2e-14 |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | fixest | 2.2e-14 | 2.5e-14 |  | 2.2e-14 | 2.2e-14 |

## G5 PPML

| config | T | ppmlhdfe | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 0.571 | 49.0ms | 0.130 | 0.320 |  | n/a |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | 0.841 | 62.0ms | 0.134 | 0.487 |  | n/a |

Coefficient gap:

| config | T | vs | ppmlhdfe | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fixest | 1.1e-16 |  | 7.5e-15 | 1.0e-12 |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | fixest | 5.6e-16 |  | 7.1e-15 | 1.0e-12 |

## G6 Logit

| config | T | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---|
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 43.0ms | 0.161 | 0.319 |  | n/a |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | 50.0ms | 0.143 | 0.362 |  | n/a |

Coefficient gap:

| config | T | vs | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest |  | 1.1e-09 | 1.1e-09 |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest |  | 1.1e-09 | 1.1e-09 |

## G7 Heterogeneous slopes

| config | T | reghdfe | xhdfe | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | 0.330 | 41.0ms | 16.0ms | unsupported | unsupported |  | n/a |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | 0.334 | 40.0ms | 18.0ms | unsupported | unsupported |  | n/a |

Coefficient gap:

| config | T | vs | reghdfe | xhdfe | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|---:|
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | fixest | 1.3e-14 | 9.2e-15 |  |  |  |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | fixest | 1.3e-14 | 9.2e-15 |  |  |  |
