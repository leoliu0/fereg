# fereg benchmark results

Generated 2026-10-12T22:14:10 by `run.py` from `results/results.csv`.
Times are medians of the timed runs in seconds (`ms` = milliseconds); estimation call only, data in memory.
`ratio` = fereg / fastest finished competitor. Gate: PASS iff fereg is faster than every competitor that
finished (a competitor timeout counts as slower); `n/a` = fereg not run. `error`/`timeout`/`unsupported`
cells are explained in results.csv. Coefficient gap: max relative difference over shared non-constant
coefficients versus fereg (versus fixest when fereg has no result for the row).

**Gate summary:** 273 PASS, 0 FAIL.

**fereg_py gate summary:** 273 PASS, 0 FAIL (section below).

**fereg_r gate summary:** 273 PASS, 0 FAIL (section below).

## G1 OLS (fixest DGP)

| config | T | fereg | reghdfe | xhdfe | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `fx_difficult_10000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 2.0ms | 18.0ms | 9.0ms | 22.0ms | 6.6ms | 7.2ms | 0.30 | PASS |
| `fx_difficult_10000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 2.0ms | 19.0ms | 6.0ms | 5.0ms | 6.4ms | 7.2ms | 0.40 | PASS |
| `fx_difficult_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 3.0ms | 53.0ms | 11.0ms | 28.0ms | 9.8ms | 11.9ms | 0.31 | PASS |
| `fx_difficult_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 2.0ms | 40.0ms | 8.0ms | 6.0ms | 10.7ms | 10.6ms | 0.33 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 2.0ms | 20.0ms | 11.0ms | 10.0ms | 8.0ms | 8.5ms | 0.25 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 3.0ms | 24.0ms | 7.0ms | 6.0ms | 8.8ms | 13.6ms | 0.50 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 3.0ms | 43.0ms | 13.0ms | 8.0ms | 10.4ms | 11.1ms | 0.38 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 3.0ms | 58.0ms | 15.0ms | 8.0ms | 18.6ms | 13.9ms | 0.38 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 3.0ms | 25.0ms | 10.0ms | 11.0ms | 8.9ms | 9.5ms | 0.34 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | 3.0ms | 29.0ms | 8.0ms | 7.0ms | 9.4ms | 9.7ms | 0.43 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 4.0ms | 72.0ms | 19.0ms | 8.0ms | 10.6ms | 11.0ms | 0.50 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | 4.0ms | 80.0ms | 25.0ms | 11.0ms | 20.3ms | 17.5ms | 0.36 | PASS |
| `fx_difficult_100000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 6.0ms | 0.131 | 98.0ms | 9.0ms | 23.9ms | 33.3ms | 0.67 | PASS |
| `fx_difficult_100000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 6.0ms | 0.145 | 23.0ms | 11.0ms | 23.6ms | 30.1ms | 0.55 | PASS |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 9.0ms | 0.852 | 0.139 | 0.138 | 0.533 | 57.2ms | 0.16 | PASS |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 12.0ms | 0.985 | 0.220 | 0.108 | 0.710 | 62.7ms | 0.19 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 6.0ms | 0.145 | 74.0ms | 0.105 | 36.0ms | 56.5ms | 0.17 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 8.0ms | 0.275 | 38.0ms | 20.0ms | 29.5ms | 47.5ms | 0.40 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 10.0ms | 1.177 | 0.275 | 0.114 | 0.590 | 71.0ms | 0.14 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 13.0ms | 2.184 | 0.573 | 0.332 | 2.126 | 0.109 | 0.12 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 8.0ms | 0.169 | 42.0ms | 0.180 | 36.3ms | 40.9ms | 0.22 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | 13.0ms | 0.241 | 45.0ms | 24.0ms | 32.2ms | 48.0ms | 0.54 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 13.0ms | 1.765 | 0.437 | 0.371 | 0.478 | 82.8ms | 0.16 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | 19.0ms | 2.872 | 0.815 | 0.536 | 3.324 | 0.168 | 0.11 | PASS |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 21.0ms | 1.129 | 0.110 | 49.0ms | 0.159 | 0.183 | 0.43 | PASS |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 41.0ms | 1.232 | 0.132 | 50.0ms | 0.160 | 0.219 | 0.82 | PASS |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 46.0ms | 21.651 | 1.829 | 1.412 | 17.486 | 0.383 | 0.12 | PASS |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.121 | 33.603 | 6.402 | 2.431 | 21.943 | 0.637 | 0.19 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | 32.0ms | 0.625 | 0.204 | 82.0ms | 0.284 | 0.326 | 0.39 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 1 | 78.0ms | 1.070 | 0.262 | 0.137 | 0.290 | 0.434 | 0.57 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 27.0ms | 1.364 | 0.219 | 79.0ms | 0.224 | 0.270 | 0.34 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 59.0ms | 1.754 | 0.251 | 0.116 | 0.230 | 0.371 | 0.51 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | 59.0ms | 47.375 | 1.278 | 4.711 | 48.805 | 0.835 | 0.07 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 1 | 0.171 | 95.095 | 2.156 | 8.403 | 58.179 | 1.469 | 0.12 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 56.0ms | 47.955 | 1.278 | 4.586 | 48.271 | 0.777 | 0.07 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.160 | 95.627 | 2.116 | 8.483 | 58.841 | 1.417 | 0.11 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 36.0ms | 1.798 | 0.290 | 0.113 | 0.301 | 0.362 | 0.32 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | 0.113 | 2.495 | 0.368 | 0.213 | 0.317 | 0.553 | 0.53 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 82.0ms | 86.994 | 2.284 | 9.628 | 88.884 | 1.230 | 0.07 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.314 | 172.037 | 3.853 | 17.365 | 119.780 | 2.594 | 0.12 | PASS |
| `fx_difficult_10000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 0.151 | 12.136 | 0.942 | 0.595 | 2.069 | 2.420 | 0.25 | PASS |
| `fx_difficult_10000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.376 | timeout | 73.541 | 35.764 | timeout | 5.925 | 0.06 | PASS |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | 0.288 | 6.650 | 1.739 | 1.074 | 3.122 | 3.645 | 0.27 | PASS |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 0.247 | 14.615 | 1.886 | 0.857 | 2.916 | 3.408 | 0.29 | PASS |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | 0.589 | timeout | 20.828 | 77.090 | timeout | 13.569 | 0.04 | PASS |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.559 | timeout | 20.756 | 77.911 | timeout | 13.113 | 0.04 | PASS |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 0.352 | 22.694 | 2.882 | 1.355 | 3.797 | 4.709 | 0.26 | PASS |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.879 | timeout | 35.780 | 158.831 | timeout | 21.494 | 0.04 | PASS |
| `fx_simple_10000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 2.0ms | 18.0ms | 7.0ms | 8.0ms | 6.4ms | 7.0ms | 0.31 | PASS |
| `fx_simple_10000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 2.0ms | 18.0ms | 6.0ms | 5.0ms | 6.9ms | 6.9ms | 0.40 | PASS |
| `fx_simple_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 2.0ms | 0.306 | 0.250 | 5.0ms | 9.3ms | 12.6ms | 0.40 | PASS |
| `fx_simple_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 2.0ms | 37.0ms | 8.0ms | 6.0ms | 10.0ms | 10.0ms | 0.33 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 2.0ms | 20.0ms | 8.0ms | 6.0ms | 8.3ms | 8.0ms | 0.33 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 2.0ms | 47.0ms | 10.0ms | 12.0ms | 11.4ms | 23.1ms | 0.20 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 3.0ms | 55.0ms | 10.0ms | 6.0ms | 8.8ms | 11.1ms | 0.50 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 3.0ms | 46.0ms | 9.0ms | 7.0ms | 10.1ms | 13.3ms | 0.43 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 3.0ms | 26.0ms | 10.0ms | 6.0ms | 9.7ms | 9.5ms | 0.50 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | 3.0ms | 31.0ms | 8.0ms | 6.0ms | 9.2ms | 10.7ms | 0.50 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 3.0ms | 61.0ms | 14.0ms | 7.0ms | 9.9ms | 12.5ms | 0.43 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | 3.0ms | 65.0ms | 14.0ms | 9.0ms | 11.6ms | 16.2ms | 0.33 | PASS |
| `fx_simple_100000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 7.0ms | 0.128 | 0.213 | 10.0ms | 24.3ms | 31.7ms | 0.70 | PASS |
| `fx_simple_100000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 5.0ms | 0.138 | 22.0ms | 9.0ms | 22.3ms | 26.3ms | 0.56 | PASS |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 8.0ms | 0.286 | 0.220 | 12.0ms | 33.9ms | 64.0ms | 0.67 | PASS |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 10.0ms | 0.298 | 26.0ms | 11.0ms | 37.5ms | 64.5ms | 0.91 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 6.0ms | 0.205 | 0.165 | 13.0ms | 32.6ms | 41.7ms | 0.46 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 8.0ms | 0.181 | 35.0ms | 17.0ms | 28.8ms | 38.4ms | 0.47 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 9.0ms | 0.311 | 38.0ms | 16.0ms | 32.1ms | 70.7ms | 0.56 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 13.0ms | 0.405 | 43.0ms | 25.0ms | 38.2ms | 98.2ms | 0.52 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 9.0ms | 0.177 | 74.0ms | 15.0ms | 34.4ms | 44.8ms | 0.60 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | 15.0ms | 0.247 | 46.0ms | 28.0ms | 35.9ms | 50.6ms | 0.54 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 13.0ms | 0.432 | 0.238 | 17.0ms | 43.0ms | 91.0ms | 0.76 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | 22.0ms | 0.571 | 88.0ms | 45.0ms | 52.2ms | 0.147 | 0.49 | PASS |
| `fx_simple_1000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 19.0ms | 1.306 | 0.156 | 43.0ms | 0.174 | 0.201 | 0.44 | PASS |
| `fx_simple_1000000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 40.0ms | 1.260 | 0.137 | 53.0ms | 0.169 | 0.236 | 0.75 | PASS |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 36.0ms | 2.763 | 0.158 | 0.126 | 0.216 | 0.750 | 0.29 | PASS |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 76.0ms | 2.972 | 0.216 | 87.0ms | 0.225 | 0.974 | 0.87 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | 36.0ms | 0.666 | 0.247 | 87.0ms | 0.301 | 0.343 | 0.41 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 1 | 76.0ms | 1.087 | 0.268 | 0.137 | 0.307 | 0.440 | 0.55 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 32.0ms | 1.369 | 0.247 | 69.0ms | 0.244 | 0.290 | 0.46 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 60.0ms | 1.800 | 0.262 | 0.122 | 0.252 | 0.399 | 0.49 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | 50.0ms | 2.032 | 0.396 | 0.219 | 0.421 | 1.105 | 0.23 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 1 | 0.134 | 3.000 | 0.649 | 0.243 | 0.438 | 1.754 | 0.55 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 51.0ms | 3.488 | 0.341 | 0.167 | 0.381 | 1.057 | 0.31 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.119 | 4.503 | 0.651 | 0.224 | 0.399 | 1.692 | 0.53 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 37.0ms | 1.878 | 0.299 | 0.114 | 0.306 | 0.368 | 0.32 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | 0.114 | 2.518 | 0.377 | 0.215 | 0.318 | 0.551 | 0.53 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 75.0ms | 4.650 | 0.656 | 0.442 | 0.547 | 1.431 | 0.17 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.295 | 6.060 | 0.865 | 0.416 | 0.597 | 2.560 | 0.71 | PASS |
| `fx_simple_10000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 0.148 | 11.714 | 0.874 | 0.507 | 1.823 | 2.210 | 0.29 | PASS |
| `fx_simple_10000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.287 | 31.417 | 1.930 | 1.268 | 2.613 | 13.120 | 0.23 | PASS |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | 0.287 | 7.041 | 1.775 | 1.988 | 3.292 | 4.040 | 0.16 | PASS |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 0.241 | 14.029 | 1.790 | 0.841 | 2.594 | 3.148 | 0.29 | PASS |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | 0.532 | 21.049 | 3.904 | 3.537 | 4.521 | 17.356 | 0.15 | PASS |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.482 | 36.610 | 3.875 | 4.008 | 3.628 | 16.320 | 0.13 | PASS |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 0.340 | 21.139 | 2.630 | 1.423 | 3.636 | 4.366 | 0.24 | PASS |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.903 | 60.546 | 6.685 | 5.448 | 6.519 | 22.431 | 0.17 | PASS |

Coefficient gap:

| config | T | vs | fereg | reghdfe | xhdfe | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `fx_difficult_10000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 3.8e-15 | 3.5e-15 | 4.6e-15 | 3.8e-15 | 3.8e-15 |
| `fx_difficult_10000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 3.1e-15 | 3.5e-15 | 4.6e-15 | 3.8e-15 | 3.8e-15 |
| `fx_difficult_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 2.7e-15 | 4.5e-15 | 2.6e-13 | 4.2e-12 | 5.7e-15 |
| `fx_difficult_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 1.8e-15 | 4.5e-15 | 2.6e-13 | 2.1e-13 | 5.7e-15 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 5.6e-15 | 3.7e-15 | 8.9e-16 | 4.7e-15 | 4.7e-15 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 5.3e-15 | 3.7e-15 | 8.9e-16 | 4.7e-15 | 4.4e-15 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 6.1e-15 | 5.7e-15 | 1.8e-12 | 1.3e-11 | 6.9e-15 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 5.5e-15 | 5.7e-15 | 1.8e-12 | 1.3e-11 | 6.7e-15 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 1.1e-14 | 8.6e-15 | 2.0e-15 | 8.5e-15 | 7.8e-15 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 1.0e-14 | 8.5e-15 | 2.0e-15 | 8.5e-15 | 7.8e-15 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 1.0e-14 | 6.8e-15 | 5.9e-12 | 6.7e-11 | 7.0e-15 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 1.1e-14 | 6.8e-15 | 5.9e-12 | 1.1e-11 | 7.0e-15 |
| `fx_difficult_100000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 6.2e-15 | 8.8e-16 | 7.9e-15 | 8.8e-16 | 1.1e-15 |
| `fx_difficult_100000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 6.2e-15 | 8.8e-16 | 7.9e-15 | 1.3e-15 | 6.6e-16 |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 2.2e-13 | 1.3e-15 | 3.3e-10 | 6.7e-08 | 2.9e-15 |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 2.2e-13 | 1.3e-15 | 3.3e-10 | 6.7e-08 | 3.1e-15 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 1.3e-14 | 2.4e-15 | 2.1e-14 | 3.2e-15 | 3.5e-15 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 1.3e-14 | 2.4e-15 | 2.1e-14 | 2.6e-15 | 3.9e-15 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 2.3e-12 | 3.6e-15 | 1.7e-09 | 7.9e-07 | 7.3e-15 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 2.3e-12 | 3.6e-15 | 1.7e-09 | 7.9e-07 | 7.3e-15 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 1.4e-14 | 2.9e-15 | 2.9e-14 | 3.2e-15 | 3.7e-15 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 1.5e-14 | 3.2e-15 | 2.9e-14 | 3.2e-15 | 3.9e-15 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 2.8e-12 | 3.7e-15 | 3.3e-09 | 1.6e-06 | 7.2e-15 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 2.8e-12 | 3.7e-15 | 3.3e-09 | 1.5e-06 | 7.5e-15 |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 1.0e-14 | 6.6e-16 | 3.0e-14 | 6.6e-16 | 2.2e-16 |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 1.1e-14 | 6.6e-16 | 3.0e-14 | 2.4e-15 | 1.8e-15 |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 4.8e-14 | 1.1e-15 | 1.4e-06 | 4.0e-06 | 2.2e-16 |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 7.9e-14 | 1.1e-15 | 1.4e-06 | 3.7e-06 | 2.2e-16 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | fereg |  | 1.9e-14 | 1.7e-15 | 7.3e-14 | 1.3e-15 | 3.3e-15 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 1 | fereg |  | 1.8e-14 | 1.7e-15 | 7.3e-14 | 1.3e-15 | 3.3e-15 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 1.9e-14 | 1.7e-15 | 7.3e-14 | 1.3e-15 | 3.3e-15 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 1.8e-14 | 1.7e-15 | 7.3e-14 | 1.3e-15 | 3.3e-15 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | fereg |  | 2.2e-13 | 1.3e-15 | 1.4e-06 | 4.5e-06 | 3.8e-15 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 1 | fereg |  | 1.2e-13 | 1.3e-15 | 1.4e-06 | 5.6e-06 | 3.7e-15 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 2.2e-13 | 1.3e-15 | 1.4e-06 | 5.6e-06 | 3.7e-15 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 1.2e-13 | 1.3e-15 | 1.4e-06 | 5.6e-06 | 3.7e-15 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 6.0e-14 | 2.5e-15 | 7.3e-14 | 1.5e-15 | 4.0e-15 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 5.8e-14 | 2.3e-15 | 7.3e-14 | 1.5e-15 | 3.5e-15 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 2.7e-13 | 2.0e-15 | 4.3e-06 | 1.0e-05 | 3.3e-15 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 4.4e-13 | 2.0e-15 | 4.3e-06 | 1.1e-05 | 3.7e-15 |
| `fx_difficult_10000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 5.7e-14 | 2.0e-15 | 6.5e-14 | 1.6e-15 | 1.3e-15 |
| `fx_difficult_10000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  |  | 2.2e-15 | 1.6e-08 | 8.6e-07 | 1.6e-14 |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | fereg |  | 1.1e-13 | 4.3e-15 | 1.7e-13 | 8.0e-15 | 8.4e-15 |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 1.1e-13 | 4.3e-15 | 1.7e-13 | 8.0e-15 | 8.4e-15 |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | fereg |  |  | 4.7e-15 | 1.5e-06 |  | 4.4e-14 |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  |  | 4.7e-15 | 1.5e-06 |  | 7.4e-14 |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 1.5e-13 | 5.2e-15 | 2.2e-13 | 8.2e-15 | 8.9e-15 |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  |  | 4.8e-15 | 4.6e-06 |  | 1.1e-13 |
| `fx_simple_10000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 4.4e-16 | 2.2e-16 | 3.1e-15 | 4.4e-16 | 6.6e-16 |
| `fx_simple_10000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 2.2e-16 | 2.2e-16 | 3.1e-15 | 4.4e-16 | 6.6e-16 |
| `fx_simple_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 1.3e-15 | 1.7e-15 | 6.8e-15 | 1.5e-15 | 2.4e-15 |
| `fx_simple_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 1.1e-15 | 1.7e-15 | 6.8e-15 | 1.7e-15 | 2.4e-15 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 6.1e-15 | 6.4e-15 | 1.3e-15 | 6.2e-15 | 6.1e-15 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 6.1e-15 | 6.4e-15 | 1.3e-15 | 6.2e-15 | 6.1e-15 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 9.2e-15 | 9.3e-15 | 7.5e-15 | 9.2e-15 | 1.0e-14 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 8.9e-15 | 9.3e-15 | 7.5e-15 | 1.0e-14 | 1.0e-14 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 8.9e-15 | 8.1e-15 | 1.5e-15 | 7.7e-15 | 7.7e-15 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 9.7e-15 | 8.1e-15 | 1.5e-15 | 8.1e-15 | 7.7e-15 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 9.2e-15 | 1.1e-14 | 9.9e-15 | 9.9e-15 | 1.2e-14 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 1.1e-14 | 1.1e-14 | 9.9e-15 | 1.1e-14 | 1.2e-14 |
| `fx_simple_100000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 1.7e-14 | 4.4e-16 | 7.7e-15 | 0.0e+00 | 0.0e+00 |
| `fx_simple_100000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 1.6e-14 | 4.4e-16 | 7.7e-15 | 0.0e+00 | 4.4e-16 |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 1.1e-14 | 1.5e-15 | 9.7e-15 | 6.6e-16 | 4.4e-16 |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 1.1e-14 | 1.5e-15 | 9.7e-15 | 1.1e-15 | 2.2e-16 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 1.7e-14 | 2.8e-15 | 2.5e-14 | 3.3e-15 | 3.4e-15 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 1.6e-14 | 2.8e-15 | 2.5e-14 | 3.3e-15 | 3.4e-15 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 1.3e-14 | 2.9e-15 | 2.2e-14 | 2.6e-15 | 2.9e-15 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 1.2e-14 | 2.9e-15 | 2.2e-14 | 2.7e-15 | 2.9e-15 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 1.7e-14 | 3.9e-15 | 2.5e-14 | 3.9e-15 | 3.2e-15 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 2.0e-14 | 3.7e-15 | 2.5e-14 | 4.2e-15 | 3.7e-15 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 2.6e-14 | 2.3e-15 | 2.2e-14 | 2.3e-15 | 2.9e-15 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 2.5e-14 | 2.8e-15 | 2.2e-14 | 2.5e-15 | 3.1e-15 |
| `fx_simple_1000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 5.7e-15 | 1.0e-15 | 2.3e-15 | 1.1e-16 | 1.1e-16 |
| `fx_simple_1000000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 6.4e-15 | 1.0e-15 | 2.3e-15 | 6.7e-16 | 4.4e-16 |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 4.8e-14 | 6.7e-16 | 7.0e-15 | 4.4e-16 | 4.4e-16 |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 4.8e-14 | 6.7e-16 | 7.0e-15 | 0.0e+00 | 1.9e-15 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | fereg |  | 2.7e-14 | 2.0e-15 | 8.7e-14 | 2.4e-15 | 3.1e-15 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 1 | fereg |  | 2.7e-14 | 2.0e-15 | 8.7e-14 | 2.4e-15 | 2.9e-15 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 2.7e-14 | 2.0e-15 | 8.7e-14 | 2.4e-15 | 3.1e-15 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 2.7e-14 | 2.0e-15 | 8.7e-14 | 2.4e-15 | 2.9e-15 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | fereg |  | 5.4e-14 | 8.9e-16 | 5.6e-14 | 2.0e-15 | 1.8e-15 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 1 | fereg |  | 5.3e-14 | 8.9e-16 | 5.6e-14 | 2.2e-15 | 1.5e-15 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 5.4e-14 | 8.9e-16 | 5.6e-14 | 2.4e-15 | 1.7e-15 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 5.3e-14 | 8.9e-16 | 5.6e-14 | 2.0e-15 | 1.3e-15 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 5.7e-14 | 2.4e-15 | 8.7e-14 | 2.2e-15 | 2.9e-15 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 5.3e-14 | 2.0e-15 | 8.7e-14 | 2.2e-15 | 2.7e-15 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 3.9e-14 | 1.7e-15 | 5.6e-14 | 2.9e-15 | 3.7e-15 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 3.7e-14 | 1.6e-15 | 5.6e-14 | 2.3e-15 | 3.4e-15 |
| `fx_simple_10000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 1.1e-13 | 3.2e-15 | 4.7e-14 | 1.1e-15 | 8.9e-16 |
| `fx_simple_10000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 1.6e-13 | 4.9e-15 | 2.0e-14 | 1.2e-15 | 1.2e-15 |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | fereg |  | 2.3e-13 | 3.8e-15 | 2.7e-13 | 9.5e-15 | 1.1e-14 |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 2.3e-13 | 3.8e-15 | 2.7e-13 | 9.5e-15 | 1.1e-14 |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | fereg |  | 2.3e-13 | 5.1e-15 | 1.7e-13 | 1.0e-14 | 1.1e-14 |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 2.3e-13 | 5.1e-15 | 1.7e-13 | 5.5e-15 | 1.1e-14 |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 2.3e-13 | 5.6e-15 | 2.7e-13 | 9.2e-15 | 1.1e-14 |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 1.1e-13 | 5.5e-15 | 1.7e-13 | 5.4e-15 | 1.1e-14 |

## G2 AKM

| config | T | fereg | reghdfe | xhdfe | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `akm_baseline_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.156 | 6.436 | 0.745 | 0.425 | 6.777 | 0.471 | 0.37 | PASS |
| `akm_baseline_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.235 | 10.744 | 2.224 | 0.829 | 10.356 | 0.949 | 0.28 | PASS |
| `akm_freeze_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.121 | 9.957 | 1.136 | 0.552 | 17.859 | 0.575 | 0.22 | PASS |
| `akm_freeze_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.207 | 13.301 | 2.003 | 0.964 | 15.060 | 0.950 | 0.22 | PASS |
| `akm_freeze_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.122 | 14.928 | 1.554 | 1.232 | 19.287 | 0.775 | 0.16 | PASS |
| `akm_freeze_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.200 | 12.276 | 2.017 | 1.670 | 22.908 | 0.840 | 0.24 | PASS |
| `akm_freeze_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 91.0ms | 7.438 | 0.865 | 0.476 | 8.295 | 0.491 | 0.19 | PASS |
| `akm_freeze_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.160 | 12.095 | 1.828 | 0.930 | 13.685 | 0.963 | 0.17 | PASS |
| `akm_freeze_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 74.0ms | 10.628 | 1.017 | 0.753 | 15.710 | 0.518 | 0.14 | PASS |
| `akm_freeze_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.154 | 13.897 | 2.062 | 1.228 | 19.560 | 0.908 | 0.17 | PASS |
| `akm_freeze_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 53.0ms | 12.054 | 1.073 | 1.068 | 33.232 | 0.545 | 0.10 | PASS |
| `akm_freeze_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.126 | 15.786 | 2.265 | 1.912 | 24.496 | 0.755 | 0.17 | PASS |
| `akm_freeze_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 46.0ms | 15.932 | 1.536 | 1.842 | error | 0.573 | 0.08 | PASS |
| `akm_freeze_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.115 | 32.008 | 3.399 | 3.035 | error | 0.683 | 0.17 | PASS |
| `akm_interaction_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 74.0ms | 6.005 | 0.537 | 0.278 | 1.356 | 0.728 | 0.27 | PASS |
| `akm_interaction_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.199 | 5.590 | 0.679 | 0.413 | 1.769 | 0.987 | 0.48 | PASS |
| `akm_interaction_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 77.0ms | 4.903 | 2.142 | 0.220 | 1.282 | 0.807 | 0.35 | PASS |
| `akm_interaction_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.215 | 6.386 | 2.579 | 0.347 | 1.347 | 1.107 | 0.62 | PASS |
| `akm_interaction_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 55.0ms | 8.208 | 0.878 | 0.840 | 16.093 | 0.405 | 0.14 | PASS |
| `akm_interaction_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.127 | 12.432 | 1.817 | 1.430 | 23.808 | 0.758 | 0.17 | PASS |
| `akm_interaction_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 64.0ms | 28.902 | 1.806 | 2.608 | error | 0.568 | 0.11 | PASS |
| `akm_interaction_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.134 | 31.375 | 4.243 | 4.008 | error | 0.816 | 0.16 | PASS |
| `akm_mobility_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 57.0ms | 3.116 | 0.245 | 0.106 | 0.267 | 0.811 | 0.54 | PASS |
| `akm_mobility_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.125 | 3.581 | 0.339 | 0.146 | 0.299 | 1.210 | 0.86 | PASS |
| `akm_mobility_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 79.0ms | 3.817 | 1.208 | 0.207 | 0.824 | 0.551 | 0.38 | PASS |
| `akm_mobility_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.211 | 5.061 | 1.591 | 0.330 | 1.047 | 0.896 | 0.64 | PASS |
| `akm_mobility_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 73.0ms | 7.434 | 0.774 | 0.686 | 10.315 | 0.427 | 0.17 | PASS |
| `akm_mobility_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.153 | 11.335 | 1.806 | 1.172 | 15.429 | 0.782 | 0.20 | PASS |
| `akm_mobility_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 42.0ms | 11.061 | 1.372 | 1.145 | 21.857 | 0.359 | 0.12 | PASS |
| `akm_mobility_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.115 | 13.709 | 2.888 | 2.340 | 29.990 | 0.598 | 0.19 | PASS |
| `akm_mobility_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 47.0ms | 13.308 | 0.959 | 1.437 | error | 0.374 | 0.13 | PASS |
| `akm_mobility_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.122 | 20.450 | 2.648 | 2.591 | error | 0.697 | 0.18 | PASS |
| `akm_mobility_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 47.0ms | 19.795 | 1.750 | 1.467 | error | 0.360 | 0.13 | PASS |
| `akm_mobility_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.115 | 32.350 | 4.263 | 2.618 | error | 0.653 | 0.18 | PASS |
| `akm_scale_1_10000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 9.0ms | 0.135 | 32.0ms | 16.0ms | 83.0ms | 25.1ms | 0.56 | PASS |
| `akm_scale_1_10000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 9.0ms | 0.112 | 22.0ms | 21.0ms | 0.118 | 12.4ms | 0.73 | PASS |
| `akm_scale_2_100000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 17.0ms | 0.682 | 97.0ms | 64.0ms | 0.401 | 51.2ms | 0.33 | PASS |
| `akm_scale_2_100000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 22.0ms | 0.884 | 0.114 | 0.109 | 0.770 | 69.3ms | 0.32 | PASS |
| `akm_scale_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.132 | 5.533 | 0.711 | 0.344 | 3.794 | 0.453 | 0.38 | PASS |
| `akm_scale_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.211 | 7.431 | 1.292 | 0.589 | 4.990 | 0.828 | 0.36 | PASS |
| `akm_scale_4_10000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.910 | 61.884 | 7.384 | 4.616 | 47.856 | 4.379 | 0.21 | PASS |
| `akm_sorting_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 79.0ms | 5.205 | 0.698 | 0.313 | 3.251 | 0.409 | 0.25 | PASS |
| `akm_sorting_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.159 | 6.749 | 1.066 | 0.523 | 3.808 | 0.750 | 0.30 | PASS |
| `akm_sorting_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.178 | 6.250 | 0.773 | 0.476 | 6.387 | 0.450 | 0.40 | PASS |
| `akm_sorting_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.442 | 9.270 | 1.572 | 0.829 | 9.480 | 0.819 | 0.54 | PASS |
| `akm_sorting_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.269 | 8.590 | 0.986 | 0.508 | 5.353 | 0.661 | 0.53 | PASS |
| `akm_sorting_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.493 | 9.072 | 1.731 | 0.855 | 7.077 | 0.830 | 0.59 | PASS |
| `akm_sorting_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.280 | 7.260 | 0.914 | 0.463 | 8.786 | 0.678 | 0.60 | PASS |
| `akm_sorting_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.520 | 9.449 | 2.000 | 0.829 | 10.257 | 0.919 | 0.63 | PASS |
| `akm_sorting_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.266 | 7.681 | 0.959 | 0.712 | 14.752 | 0.482 | 0.55 | PASS |
| `akm_sorting_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.409 | 11.748 | 2.372 | 1.270 | 17.849 | 0.806 | 0.51 | PASS |

Coefficient gap:

| config | T | vs | fereg | reghdfe | xhdfe | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `akm_baseline_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 3.7e-14 | 2.2e-15 | 4.6e-14 | 7.9e-14 | 1.3e-15 |
| `akm_baseline_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 3.5e-14 | 2.2e-15 | 4.6e-14 | 7.7e-14 | 1.6e-15 |
| `akm_freeze_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 5.1e-14 | 8.9e-16 | 2.6e-14 | 4.1e-14 | 1.6e-15 |
| `akm_freeze_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 4.5e-14 | 8.9e-16 | 2.6e-14 | 2.1e-14 | 1.3e-15 |
| `akm_freeze_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 8.3e-14 | 6.7e-16 | 2.7e-15 | 7.1e-13 | 2.2e-16 |
| `akm_freeze_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 7.4e-14 | 6.7e-16 | 2.7e-15 | 7.1e-13 | 1.1e-15 |
| `akm_freeze_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 1.5e-14 | 2.3e-15 | 6.9e-14 | 6.9e-14 | 4.0e-15 |
| `akm_freeze_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 7.2e-15 | 2.3e-15 | 6.9e-14 | 9.8e-14 | 3.4e-15 |
| `akm_freeze_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 2.5e-14 | 0.0e+00 | 7.4e-14 | 2.0e-13 | 2.4e-15 |
| `akm_freeze_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 3.1e-14 | 0.0e+00 | 7.4e-14 | 2.0e-13 | 2.1e-15 |
| `akm_freeze_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 5.7e-14 | 1.6e-15 | 1.4e-14 | 9.6e-13 | 3.8e-15 |
| `akm_freeze_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 7.3e-14 | 1.6e-15 | 1.4e-14 | 9.8e-13 | 4.9e-15 |
| `akm_freeze_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 9.7e-13 | 4.4e-16 | 3.2e-10 |  | 8.9e-16 |
| `akm_freeze_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 9.0e-13 | 4.4e-16 | 3.2e-10 |  | 4.4e-16 |
| `akm_interaction_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 3.3e-14 | 2.2e-16 | 4.4e-16 | 1.6e-15 | 4.4e-16 |
| `akm_interaction_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 3.3e-14 | 2.2e-16 | 4.4e-16 | 2.4e-15 | 1.6e-15 |
| `akm_interaction_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 2.8e-14 | 1.4e-15 | 1.1e-13 | 4.3e-12 | 2.7e-15 |
| `akm_interaction_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 2.9e-14 | 1.4e-15 | 1.1e-13 | 4.3e-12 | 3.3e-15 |
| `akm_interaction_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 3.9e-14 | 5.6e-16 | 3.5e-13 | 1.2e-12 | 5.6e-16 |
| `akm_interaction_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 4.7e-14 | 5.6e-16 | 3.5e-13 | 1.2e-12 | 2.2e-15 |
| `akm_interaction_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 1.1e-12 | 1.1e-16 | 2.4e-08 |  | 2.0e-15 |
| `akm_interaction_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 1.1e-12 | 1.1e-16 | 2.4e-08 |  | 2.3e-15 |
| `akm_mobility_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 2.3e-14 | 2.2e-16 | 6.9e-14 | 1.3e-13 | 1.3e-15 |
| `akm_mobility_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 2.5e-14 | 2.2e-16 | 6.9e-14 | 1.3e-14 | 3.1e-15 |
| `akm_mobility_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 3.4e-14 | 1.6e-15 | 2.0e-14 | 2.4e-15 | 2.2e-15 |
| `akm_mobility_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 3.4e-14 | 1.6e-15 | 2.0e-14 | 8.9e-16 | 1.6e-15 |
| `akm_mobility_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 2.8e-14 | 0.0e+00 | 7.0e-14 | 1.9e-13 | 2.2e-16 |
| `akm_mobility_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 3.0e-14 | 0.0e+00 | 7.0e-14 | 2.1e-13 | 1.3e-15 |
| `akm_mobility_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 4.7e-14 | 1.4e-15 | 9.1e-13 | 8.4e-13 | 1.2e-15 |
| `akm_mobility_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 2.1e-14 | 1.4e-15 | 9.1e-13 | 9.4e-13 | 2.2e-15 |
| `akm_mobility_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 1.3e-13 | 1.1e-16 | 4.7e-10 |  | 1.4e-15 |
| `akm_mobility_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 8.5e-14 | 1.1e-16 | 4.7e-10 |  | 5.6e-16 |
| `akm_mobility_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 4.6e-12 | 1.3e-15 | 5.4e-07 |  | 2.2e-15 |
| `akm_mobility_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 4.8e-12 | 1.3e-15 | 5.4e-07 |  | 1.7e-15 |
| `akm_scale_1_10000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 9.3e-15 | 3.4e-15 | 5.6e-12 | 1.2e-10 | 3.6e-15 |
| `akm_scale_1_10000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 2.4e-14 | 3.4e-15 | 5.6e-12 | 1.2e-10 | 3.6e-15 |
| `akm_scale_2_100000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 1.0e-15 | 1.1e-15 | 3.1e-13 | 1.1e-11 | 1.1e-15 |
| `akm_scale_2_100000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 5.6e-16 | 1.1e-15 | 3.1e-13 | 1.1e-11 | 5.6e-16 |
| `akm_scale_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 2.9e-14 | 6.7e-16 | 2.8e-14 | 8.4e-15 | 2.2e-16 |
| `akm_scale_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 2.0e-14 | 6.7e-16 | 2.8e-14 | 9.8e-15 | 8.9e-16 |
| `akm_scale_4_10000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 7.8e-14 | 1.1e-14 | 2.1e-13 | 6.7e-16 | 2.1e-15 |
| `akm_sorting_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 5.0e-14 | 8.8e-16 | 8.8e-16 | 1.0e-14 | 8.8e-16 |
| `akm_sorting_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 5.6e-14 | 8.8e-16 | 8.8e-16 | 1.0e-14 | 0.0e+00 |
| `akm_sorting_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 5.0e-14 | 2.2e-16 | 2.0e-13 | 2.2e-13 | 1.1e-15 |
| `akm_sorting_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 4.9e-14 | 2.2e-16 | 2.0e-13 | 2.2e-13 | 2.4e-15 |
| `akm_sorting_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 1.3e-14 | 0.0e+00 | 3.0e-14 | 1.1e-13 | 1.3e-15 |
| `akm_sorting_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 1.5e-14 | 0.0e+00 | 3.0e-14 | 1.1e-13 | 0.0e+00 |
| `akm_sorting_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 2.1e-14 | 1.3e-15 | 1.0e-11 | 8.2e-11 | 2.0e-15 |
| `akm_sorting_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 1.2e-14 | 1.3e-15 | 1.0e-11 | 8.2e-11 | 2.7e-15 |
| `akm_sorting_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fereg |  | 8.2e-15 | 8.8e-16 | 7.1e-12 | 7.7e-11 | 1.1e-15 |
| `akm_sorting_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fereg |  | 5.8e-15 | 8.8e-16 | 7.1e-12 | 6.8e-11 | 6.6e-16 |

## G3 AKM with occupation FE

| config | T | fereg | reghdfe | xhdfe | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `akmocc_baseline_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.135 | 7.966 | 1.310 | 0.696 | 4.653 | 0.695 | 0.19 | PASS |
| `akmocc_baseline_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.251 | 9.852 | 2.337 | 1.243 | 6.924 | 1.276 | 0.20 | PASS |
| `akmocc_occlambda_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.159 | 12.417 | 1.741 | 0.816 | 6.457 | 0.908 | 0.19 | PASS |
| `akmocc_occlambda_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.272 | 11.124 | 2.203 | 1.246 | 6.810 | 1.374 | 0.22 | PASS |
| `akmocc_occlambda_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.145 | 9.118 | 1.365 | 0.883 | 6.945 | 0.715 | 0.20 | PASS |
| `akmocc_occlambda_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.257 | 11.028 | 2.438 | 1.590 | 9.488 | 1.323 | 0.19 | PASS |
| `akmocc_occlambda_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.174 | 8.824 | 1.394 | 0.978 | 5.719 | 0.773 | 0.23 | PASS |
| `akmocc_occlambda_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.295 | 10.560 | 2.523 | 1.736 | 8.012 | 1.407 | 0.21 | PASS |
| `akmocc_occlambda_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.130 | 14.886 | 1.685 | 2.303 | 24.836 | 0.830 | 0.16 | PASS |
| `akmocc_occlambda_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.254 | 21.731 | 3.479 | 4.265 | 35.423 | 1.601 | 0.16 | PASS |
| `akmocc_occsize_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.121 | 7.876 | 1.113 | 0.633 | 5.668 | 0.580 | 0.21 | PASS |
| `akmocc_occsize_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.227 | 9.857 | 1.988 | 1.278 | 9.818 | 1.116 | 0.20 | PASS |
| `akmocc_occsize_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.141 | 9.711 | 1.262 | 0.832 | 6.142 | 0.641 | 0.22 | PASS |
| `akmocc_occsize_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.255 | 10.663 | 2.277 | 1.118 | 7.149 | 1.111 | 0.23 | PASS |
| `akmocc_occsize_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.236 | 9.211 | 1.121 | 1.269 | 8.152 | 0.825 | 0.29 | PASS |
| `akmocc_occsize_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.334 | 11.201 | 2.298 | 2.120 | 10.323 | 1.517 | 0.22 | PASS |
| `akmocc_occsize_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.790 | 14.388 | 2.190 | 2.310 | 32.982 | 1.398 | 0.57 | PASS |
| `akmocc_occsize_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 1.530 | 18.292 | 4.032 | 3.668 | 46.309 | 2.806 | 0.55 | PASS |

Coefficient gap:

| config | T | vs | fereg | reghdfe | xhdfe | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `akmocc_baseline_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fereg |  | 1.8e-15 | 1.1e-15 | 2.2e-14 | 1.3e-14 | 1.7e-14 |
| `akmocc_baseline_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fereg |  | 6.4e-15 | 1.1e-15 | 2.2e-14 | 2.4e-14 | 1.6e-14 |
| `akmocc_occlambda_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fereg |  | 4.5e-14 | 1.3e-15 | 4.6e-14 | 9.4e-14 | 2.8e-14 |
| `akmocc_occlambda_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fereg |  | 4.3e-14 | 1.3e-15 | 4.6e-14 | 9.4e-14 | 2.6e-14 |
| `akmocc_occlambda_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fereg |  | 5.8e-15 | 3.3e-16 | 3.0e-14 | 2.7e-13 | 2.2e-14 |
| `akmocc_occlambda_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fereg |  | 7.2e-15 | 3.3e-16 | 3.0e-14 | 2.7e-13 | 2.1e-14 |
| `akmocc_occlambda_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fereg |  | 3.4e-14 | 3.3e-16 | 3.0e-13 | 1.4e-13 | 2.1e-14 |
| `akmocc_occlambda_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fereg |  | 2.8e-14 | 3.3e-16 | 3.0e-13 | 1.2e-13 | 2.2e-14 |
| `akmocc_occlambda_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fereg |  | 1.5e-14 | 1.8e-15 | 1.7e-09 | 6.1e-12 | 5.4e-14 |
| `akmocc_occlambda_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fereg |  | 2.2e-14 | 1.8e-15 | 1.7e-09 | 6.1e-12 | 5.4e-14 |
| `akmocc_occsize_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fereg |  | 5.9e-14 | 1.6e-15 | 6.4e-15 | 1.2e-13 | 3.1e-15 |
| `akmocc_occsize_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fereg |  | 5.5e-14 | 1.6e-15 | 6.4e-15 | 1.2e-13 | 3.3e-15 |
| `akmocc_occsize_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fereg |  | 1.2e-14 | 1.1e-15 | 1.6e-14 | 6.9e-14 | 7.3e-15 |
| `akmocc_occsize_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fereg |  | 1.1e-14 | 1.1e-15 | 1.6e-14 | 7.1e-14 | 7.5e-15 |
| `akmocc_occsize_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fereg |  | 3.2e-15 | 6.7e-16 | 1.7e-12 | 1.2e-11 | 1.3e-14 |
| `akmocc_occsize_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fereg |  | 5.3e-15 | 6.7e-16 | 1.7e-12 | 1.2e-11 | 6.8e-15 |
| `akmocc_occsize_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fereg |  | 7.3e-14 | 1.1e-15 | 2.2e-08 | 1.6e-10 | 1.4e-13 |
| `akmocc_occsize_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fereg |  | 8.1e-14 | 1.1e-15 | 2.2e-08 | 4.7e-10 | 1.6e-13 |

## G4 IV/2SLS

| config | T | fereg | ivreghdfe | xhdfe | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 9.0ms | 0.205 | 33.0ms | 20.0ms | 54.4ms | 59.0ms | 0.45 | PASS |
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | 12.0ms | 0.286 | 36.0ms | 26.0ms | 57.4ms | 65.8ms | 0.46 | PASS |
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | 14.0ms | 0.840 | 84.0ms | 0.116 | 1.473 | 97.1ms | 0.17 | PASS |
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | 20.0ms | 1.603 | 0.195 | 0.261 | 3.071 | 0.171 | 0.12 | PASS |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 37.0ms | 1.847 | 0.223 | 0.113 | 0.423 | 0.483 | 0.33 | PASS |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | 94.0ms | 2.678 | 0.272 | 0.169 | 0.432 | 0.641 | 0.56 | PASS |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | 68.0ms | 43.386 | 1.195 | 3.022 | 71.852 | 1.180 | 0.06 | PASS |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | 0.191 | 81.101 | 1.912 | 6.066 | 90.695 | 2.203 | 0.10 | PASS |
| `fx_difficult_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 0.266 | 19.098 | 1.952 | 1.321 | 4.717 | 5.406 | 0.20 | PASS |
| `fx_difficult_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | 0.553 | timeout | 18.169 | 68.012 | timeout | 20.033 | 0.03 | PASS |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 9.0ms | 0.209 | 38.0ms | 26.0ms | 54.6ms | 0.106 | 0.35 | PASS |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | 12.0ms | 0.283 | 35.0ms | 26.0ms | 56.2ms | 67.2ms | 0.46 | PASS |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | 11.0ms | 0.358 | 54.0ms | 26.0ms | 60.6ms | 96.0ms | 0.42 | PASS |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | 18.0ms | 0.474 | 43.0ms | 33.0ms | 71.9ms | 0.147 | 0.55 | PASS |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 35.0ms | 1.846 | 0.237 | 0.113 | 0.423 | 0.483 | 0.31 | PASS |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | 95.0ms | 2.669 | 0.268 | 0.168 | 0.432 | 0.646 | 0.57 | PASS |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | 53.0ms | 3.769 | 0.335 | 0.166 | 0.599 | 1.397 | 0.32 | PASS |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | 0.151 | 5.074 | 0.570 | 0.251 | 0.651 | 2.330 | 0.60 | PASS |
| `fx_simple_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 0.268 | 19.053 | 1.962 | 1.314 | 4.687 | 5.439 | 0.20 | PASS |
| `fx_simple_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | 0.481 | 41.533 | 4.098 | 2.672 | 6.996 | 22.721 | 0.18 | PASS |

Coefficient gap:

| config | T | vs | fereg | ivreghdfe | xhdfe | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fereg |  | 4.4e-16 | 3.7e-15 | 6.4e-15 | 1.3e-15 | 4.4e-16 |
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | fereg |  | 6.6e-16 | 3.7e-15 | 6.4e-15 | 1.3e-15 | 4.4e-16 |
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | fereg |  | 6.2e-14 | 5.5e-12 | 2.2e-10 | 1.2e-07 | 2.4e-15 |
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | fereg |  | 6.7e-14 | 5.5e-12 | 2.2e-10 | 1.2e-07 | 2.4e-15 |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fereg |  | 5.6e-16 | 8.2e-15 | 7.0e-14 | 1.6e-15 | 1.9e-15 |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | fereg |  | 6.7e-16 | 8.2e-15 | 7.0e-14 | 1.6e-15 | 1.9e-15 |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | fereg |  | 2.6e-13 | 4.4e-16 | 9.0e-06 | 1.4e-05 | 8.9e-16 |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | fereg |  | 3.3e-13 | 4.4e-16 | 9.0e-06 | 1.4e-05 | 6.7e-16 |
| `fx_difficult_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fereg |  | 1.4e-15 | 4.4e-15 | 3.5e-13 | 6.0e-15 | 6.9e-15 |
| `fx_difficult_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | fereg |  |  | 4.5e-14 | 2.4e-06 |  | 2.1e-14 |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fereg |  | 3.3e-15 | 3.6e-15 | 2.4e-14 | 2.8e-15 | 2.3e-15 |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | fereg |  | 3.0e-15 | 3.6e-15 | 2.4e-14 | 2.8e-15 | 2.2e-15 |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | fereg |  | 3.4e-16 | 6.7e-16 | 5.5e-15 | 1.6e-15 | 1.5e-15 |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | fereg |  | 2.2e-16 | 6.7e-16 | 5.5e-15 | 1.1e-15 | 1.8e-15 |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fereg |  | 2.2e-16 | 3.7e-15 | 4.0e-15 | 4.0e-15 | 3.3e-15 |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | fereg |  | 3.3e-16 | 3.7e-15 | 4.0e-15 | 4.0e-15 | 3.4e-15 |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | fereg |  | 8.9e-16 | 8.9e-16 | 5.8e-14 | 1.4e-15 | 1.0e-15 |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | fereg |  | 6.7e-16 | 8.9e-16 | 5.8e-14 | 1.6e-15 | 8.9e-16 |
| `fx_simple_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fereg |  | 1.4e-15 | 5.7e-15 | 1.7e-13 | 2.9e-15 | 5.6e-16 |
| `fx_simple_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | fereg |  | 2.4e-15 | 1.4e-14 | 2.4e-13 | 6.7e-16 | 2.7e-15 |

## G5 PPML

| config | T | fereg | ppmlhdfe | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 26.0ms | 0.604 | 52.0ms | 0.138 | 0.322 | 0.50 | PASS |
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | 47.0ms | 0.852 | 64.0ms | 0.140 | 0.501 | 0.73 | PASS |
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 52.0ms | 3.903 | 0.895 | error | 0.440 | 0.12 | PASS |
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 85.0ms | 5.907 | 1.572 | 26.324 | 0.789 | 0.11 | PASS |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | 33.0ms | 1.119 | 71.0ms | 0.162 | 0.478 | 0.46 | PASS |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | 71.0ms | 1.819 | 0.146 | 0.218 | 1.261 | 0.49 | PASS |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 59.0ms | 7.486 | 0.998 | error | 0.673 | 0.09 | PASS |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 0.113 | 14.327 | 4.247 | error | 2.031 | 0.06 | PASS |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 0.170 | 5.270 | 0.361 | 1.376 | 3.691 | 0.47 | PASS |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | 0.412 | 7.990 | 0.528 | 1.462 | 8.212 | 0.78 | PASS |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 0.355 | 167.613 | 40.142 | error | 6.825 | 0.05 | PASS |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 1.256 | 286.972 | 68.736 | error | 17.954 | 0.07 | PASS |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | 0.295 | 12.953 | 0.627 | 2.284 | 9.349 | 0.47 | PASS |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | 0.642 | 19.093 | 1.367 | 2.437 | 21.547 | 0.47 | PASS |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 0.493 | timeout | 137.512 | error | 18.248 | 0.03 | PASS |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 1.488 | timeout | 216.931 | error | 39.497 | 0.04 | PASS |
| `fx_difficult_10000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 1.757 | 61.869 | 4.078 | 15.348 | 49.447 | 0.43 | PASS |
| `fx_difficult_10000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 3.262 | timeout | timeout | error | 109.599 | 0.03 | PASS |
| `fx_difficult_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | 2.853 | 122.691 | 6.981 | 24.084 | 126.197 | 0.41 | PASS |
| `fx_difficult_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 5.410 | timeout | timeout | timeout | 247.045 | 0.02 | PASS |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 29.0ms | 0.693 | 74.0ms | 0.170 | 0.374 | 0.39 | PASS |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | 51.0ms | 0.944 | 79.0ms | 0.134 | 0.487 | 0.65 | PASS |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 42.0ms | 1.053 | 88.0ms | 0.227 | 0.477 | 0.48 | PASS |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 77.0ms | 1.358 | 0.132 | 0.259 | 0.907 | 0.58 | PASS |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | 37.0ms | 1.099 | 68.0ms | 0.162 | 0.454 | 0.54 | PASS |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | 65.0ms | 1.753 | 0.142 | 0.210 | 1.175 | 0.46 | PASS |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 55.0ms | 1.771 | 0.115 | 0.252 | 0.646 | 0.48 | PASS |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 0.108 | 2.734 | 0.349 | 0.584 | 2.242 | 0.31 | PASS |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 0.173 | 5.114 | 0.355 | 1.336 | 3.577 | 0.49 | PASS |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | 0.408 | 7.665 | 0.504 | 1.374 | 7.871 | 0.81 | PASS |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 0.257 | 11.669 | 1.110 | 3.677 | 8.550 | 0.23 | PASS |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 0.770 | 16.069 | 1.715 | 4.410 | 18.365 | 0.45 | PASS |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | 0.275 | 11.613 | 0.635 | 2.179 | 8.792 | 0.43 | PASS |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | 0.601 | 17.210 | 1.316 | 2.307 | 20.359 | 0.46 | PASS |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 0.439 | 25.895 | 4.216 | 11.347 | 18.015 | 0.10 | PASS |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 1.257 | 37.528 | 5.239 | 13.349 | 47.009 | 0.24 | PASS |
| `fx_simple_10000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 1.906 | 67.272 | 4.224 | 16.104 | 53.108 | 0.45 | PASS |
| `fx_simple_10000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 3.153 | 235.610 | 43.144 | 394.828 | 202.763 | 0.07 | PASS |
| `fx_simple_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | 3.097 | 137.429 | 7.291 | 26.552 | 140.230 | 0.42 | PASS |
| `fx_simple_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 5.986 | 443.293 | 94.012 | timeout | 402.220 | 0.06 | PASS |

Coefficient gap:

| config | T | vs | fereg | ppmlhdfe | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|---:|
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fereg |  | 4.1e-15 | 6.7e-16 | 1.7e-15 | 6.2e-13 |
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | fereg |  | 3.2e-15 | 6.7e-16 | 4.5e-15 | 4.2e-13 |
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fereg |  | 1.8e-13 | 5.4e-12 |  | 5.4e-13 |
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fereg |  | 8.2e-14 | 5.4e-12 | 7.9e-08 | 5.4e-13 |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | fereg |  | 1.4e-14 | 1.4e-14 | 2.7e-13 | 1.5e-12 |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | fereg |  | 1.4e-14 | 1.4e-14 | 2.6e-13 | 1.5e-12 |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fereg |  | 1.6e-13 | 1.2e-09 |  | 1.3e-12 |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fereg |  | 7.7e-14 | 1.2e-09 |  | 1.3e-12 |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fereg |  | 3.9e-14 | 7.8e-16 | 1.1e-16 | 3.7e-13 |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | fereg |  | 3.8e-14 | 7.8e-16 | 0.0e+00 | 3.7e-13 |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fereg |  | 1.7e-09 | 1.5e-06 |  | 1.5e-12 |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fereg |  | 1.7e-09 | 1.5e-06 |  | 1.5e-12 |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | fereg |  | 5.9e-14 | 4.4e-14 | 2.0e-13 | 2.5e-12 |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | fereg |  | 5.8e-14 | 4.4e-14 | 1.1e-13 | 2.6e-12 |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fereg |  |  | 5.6e-05 |  | 1.9e-12 |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fereg |  |  | 5.6e-05 |  | 1.9e-12 |
| `fx_difficult_10000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fereg |  | 1.7e-13 | 1.2e-14 | 1.3e-15 | 3.7e-13 |
| `fx_difficult_10000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fereg |  |  | 6.6e-06 |  | 3.6e-12 |
| `fx_difficult_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | fereg |  | 1.4e-13 | 2.6e-13 | 4.6e-14 | 2.1e-12 |
| `fx_difficult_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fereg |  |  |  |  | 1.9e-11 |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fereg |  | 3.9e-15 | 4.1e-15 | 6.2e-15 | 1.0e-12 |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | fereg |  | 3.5e-15 | 4.1e-15 | 3.0e-15 | 1.0e-12 |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fereg |  | 1.8e-15 | 7.1e-15 | 8.6e-14 | 1.3e-12 |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fereg |  | 4.4e-16 | 7.1e-15 | 1.3e-13 | 1.3e-12 |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | fereg |  | 1.4e-14 | 1.8e-14 | 9.0e-14 | 3.5e-12 |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | fereg |  | 1.3e-14 | 1.8e-14 | 9.2e-14 | 3.8e-12 |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fereg |  | 7.5e-15 | 1.4e-13 | 5.0e-13 | 2.1e-12 |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fereg |  | 8.1e-15 | 1.4e-13 | 5.0e-13 | 2.0e-12 |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fereg |  | 4.4e-16 | 3.3e-15 | 2.2e-16 | 4.3e-13 |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | fereg |  | 4.4e-16 | 3.3e-15 | 2.2e-16 | 4.2e-13 |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fereg |  | 3.0e-14 | 2.2e-14 | 1.5e-13 | 7.9e-13 |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fereg |  | 2.9e-14 | 2.2e-14 | 1.2e-13 | 8.0e-13 |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | fereg |  | 4.6e-14 | 8.1e-14 | 2.5e-14 | 2.8e-12 |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | fereg |  | 4.6e-14 | 8.1e-14 | 3.0e-14 | 2.8e-12 |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fereg |  | 4.7e-14 | 2.1e-13 | 1.7e-13 | 1.8e-12 |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fereg |  | 4.7e-14 | 2.1e-13 | 1.8e-13 | 1.8e-12 |
| `fx_simple_10000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fereg |  | 3.6e-14 | 6.1e-14 | 6.2e-15 | 4.6e-13 |
| `fx_simple_10000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fereg |  | 3.9e-14 | 1.1e-11 | 3.2e-15 | 3.4e-12 |
| `fx_simple_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | fereg |  | 5.1e-14 | 1.5e-13 | 5.5e-13 | 9.1e-13 |
| `fx_simple_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fereg |  | 1.5e-13 | 2.0e-12 |  | 9.6e-12 |

## G6 Logit

| config | T | fereg | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 20.0ms | 38.0ms | 0.135 | 0.283 | 0.53 | PASS |
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | 42.0ms | 48.0ms | 0.144 | 0.386 | 0.88 | PASS |
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 47.0ms | 0.460 | 4.478 | 0.563 | 0.10 | PASS |
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 82.0ms | 0.774 | 7.566 | 0.899 | 0.11 | PASS |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 31.0ms | 58.0ms | 0.220 | 0.438 | 0.53 | PASS |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 55.0ms | 0.102 | 0.218 | 0.888 | 0.54 | PASS |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 55.0ms | 0.596 | 6.191 | 0.858 | 0.09 | PASS |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.100 | 2.087 | 23.444 | 2.029 | 0.05 | PASS |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 0.122 | 0.271 | 1.486 | 3.090 | 0.45 | PASS |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | 0.363 | 0.403 | 1.531 | 6.238 | 0.90 | PASS |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.256 | 17.635 | error | 6.903 | 0.04 | PASS |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.952 | 30.193 | error | 14.916 | 0.06 | PASS |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 0.204 | 0.512 | 2.447 | 6.777 | 0.40 | PASS |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 0.482 | 0.980 | 2.657 | 14.933 | 0.49 | PASS |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.398 | 72.274 | error | 22.672 | 0.02 | PASS |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 1.276 | 103.166 | error | 50.058 | 0.03 | PASS |
| `fx_difficult_10000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 1.241 | 2.959 | 17.712 | 42.518 | 0.42 | PASS |
| `fx_difficult_10000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 2.351 | timeout | timeout | 114.788 | 0.02 | PASS |
| `fx_difficult_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 2.046 | 5.356 | 28.051 | 94.733 | 0.38 | PASS |
| `fx_difficult_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 4.004 | timeout | timeout | 418.058 | 0.01 | PASS |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 22.0ms | 43.0ms | 0.161 | 0.319 | 0.51 | PASS |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | 40.0ms | 50.0ms | 0.143 | 0.362 | 0.80 | PASS |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 32.0ms | 59.0ms | 0.179 | 0.603 | 0.54 | PASS |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 66.0ms | 85.0ms | 0.188 | 0.894 | 0.78 | PASS |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 27.0ms | 57.0ms | 0.219 | 0.426 | 0.47 | PASS |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 55.0ms | 0.102 | 0.214 | 0.838 | 0.54 | PASS |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 50.0ms | 87.0ms | 0.255 | 0.971 | 0.57 | PASS |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 94.0ms | 0.214 | 0.346 | 2.639 | 0.44 | PASS |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 0.121 | 0.268 | 1.472 | 3.073 | 0.45 | PASS |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | 0.359 | 0.396 | 1.505 | 6.203 | 0.91 | PASS |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.189 | 0.524 | 1.996 | 9.874 | 0.36 | PASS |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.615 | 0.814 | 2.145 | 19.250 | 0.76 | PASS |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 0.203 | 0.522 | 2.479 | 6.707 | 0.39 | PASS |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 0.479 | 0.969 | 2.596 | 14.642 | 0.49 | PASS |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.342 | 1.700 | 3.847 | 27.925 | 0.20 | PASS |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 1.007 | 2.230 | 4.341 | 67.534 | 0.45 | PASS |
| `fx_simple_10000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 1.242 | 3.423 | 18.477 | 45.549 | 0.36 | PASS |
| `fx_simple_10000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 2.267 | 8.835 | 26.674 | 224.324 | 0.26 | PASS |
| `fx_simple_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 2.280 | 5.432 | 29.434 | 117.308 | 0.42 | PASS |
| `fx_simple_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 4.421 | 24.112 | 46.245 | timeout | 0.18 | PASS |

Coefficient gap:

| config | T | vs | fereg | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 1.0e-08 | 1.4e-14 | 3.5e-14 |
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 1.0e-08 | 1.4e-14 | 3.5e-14 |
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 8.9e-10 | 4.0e-14 | 8.4e-14 |
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 8.9e-10 | 4.0e-14 | 8.4e-14 |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 1.6e-11 | 2.4e-14 | 3.8e-13 |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 1.6e-11 | 2.5e-14 | 3.8e-13 |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 3.0e-09 | 7.2e-13 | 2.2e-13 |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 3.0e-09 | 7.2e-13 | 2.2e-13 |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 2.8e-09 | 8.1e-15 | 6.9e-14 |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 2.8e-09 | 8.5e-15 | 6.8e-14 |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 5.9e-08 |  | 1.1e-13 |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 5.9e-08 |  | 1.2e-13 |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 7.7e-12 | 1.2e-14 | 5.0e-14 |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 7.7e-12 | 1.7e-14 | 5.1e-14 |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 4.1e-07 |  | 6.8e-13 |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 4.1e-07 |  | 6.9e-13 |
| `fx_difficult_10000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 2.1e-09 | 7.1e-15 | 7.5e-14 |
| `fx_difficult_10000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 2.2e-06 |  | 4.2e-13 |
| `fx_difficult_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 1.9e-11 | 1.3e-14 | 6.6e-14 |
| `fx_difficult_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  |  |  | 1.5e-09 |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 1.1e-09 | 1.2e-14 | 1.8e-14 |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 1.1e-09 | 1.2e-14 | 1.9e-14 |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 3.0e-11 | 1.7e-14 | 4.1e-14 |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 3.0e-11 | 1.6e-14 | 4.1e-14 |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 1.5e-11 | 1.6e-14 | 3.7e-14 |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 1.5e-11 | 1.5e-14 | 3.7e-14 |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 1.3e-09 | 3.8e-15 | 3.0e-13 |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 1.3e-09 | 5.3e-15 | 3.1e-13 |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 1.4e-09 | 6.6e-15 | 5.7e-14 |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 1.4e-09 | 6.4e-15 | 5.6e-14 |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 3.5e-10 | 9.7e-15 | 7.9e-14 |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 3.5e-10 | 9.3e-15 | 8.2e-14 |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 2.9e-12 | 9.1e-15 | 6.7e-14 |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fereg |  | 2.9e-12 | 2.9e-14 | 6.6e-14 |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 3.0e-08 | 1.9e-11 | 1.8e-11 |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fereg |  | 3.0e-08 | 1.9e-11 | 1.8e-11 |
| `fx_simple_10000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 5.3e-13 | 6.1e-15 | 5.1e-14 |
| `fx_simple_10000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 1.8e-08 | 8.5e-15 | 1.9e-13 |
| `fx_simple_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fereg |  | 2.4e-08 | 1.7e-14 | 3.0e-14 |
| `fx_simple_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fereg |  | 2.1e-13 | 9.4e-09 | 9.2e-09 |

## G7 Heterogeneous slopes

| config | T | fereg | reghdfe | xhdfe | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | 22.0ms | 1.040 | 3.445 | 0.148 | unsupported | unsupported | 0.15 | PASS |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | 29.0ms | 1.126 | 10.112 | 0.269 | unsupported | unsupported | 0.11 | PASS |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | 89.0ms | 54.779 | 20.650 | 4.148 | unsupported | unsupported | 0.02 | PASS |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | 0.249 | 51.460 | 68.619 | 6.503 | unsupported | unsupported | 0.04 | PASS |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | 11.0ms | 0.350 | 0.100 | 18.0ms | unsupported | unsupported | 0.61 | PASS |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | 16.0ms | 0.354 | 45.0ms | 21.0ms | unsupported | unsupported | 0.76 | PASS |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | 42.0ms | 3.582 | 0.361 | 0.120 | unsupported | unsupported | 0.35 | PASS |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | 0.125 | 3.850 | 0.620 | 0.166 | unsupported | unsupported | 0.75 | PASS |

Coefficient gap:

| config | T | vs | fereg | reghdfe | xhdfe | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | fereg |  | 3.3e-13 | 4.0e-15 | 4.3e-11 |  |  |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | fereg |  | 3.3e-13 | 4.0e-15 | 4.3e-11 |  |  |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | fereg |  | 2.1e-14 | 3.8e-12 | 4.9e-07 |  |  |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | fereg |  | 1.7e-14 | 3.8e-12 | 4.9e-07 |  |  |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | fereg |  | 4.6e-15 | 4.4e-16 | 8.8e-15 |  |  |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | fereg |  | 3.7e-15 | 4.4e-16 | 8.8e-15 |  |  |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | fereg |  | 4.4e-14 | 3.3e-16 | 2.5e-14 |  |  |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | fereg |  | 4.4e-14 | 3.3e-16 | 2.5e-14 |  |  |

# Python package `fereg` gate

`ratio` = fereg_py / fastest finished competitor among pyfixest_map, pyfixest_lsmr, fixest. Gate: PASS iff fereg_py is
faster than every one of them that finished (a timeout counts as slower; errors and `unsupported`
are excluded). Coefficient gap: max relative difference versus fixest (versus the Stata fereg
when fixest has no coefficients for the row).

**fereg_py gate summary:** 273 PASS, 0 FAIL.

## G1 OLS (fixest DGP)

| config | T | fereg_py | pyfixest_map | pyfixest_lsmr | fixest | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `fx_difficult_10000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 2.6ms | 6.6ms | 7.2ms | 22.0ms | 0.40 | PASS |
| `fx_difficult_10000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 2.6ms | 6.4ms | 7.2ms | 5.0ms | 0.52 | PASS |
| `fx_difficult_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 2.9ms | 9.8ms | 11.9ms | 28.0ms | 0.30 | PASS |
| `fx_difficult_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 2.9ms | 10.7ms | 10.6ms | 6.0ms | 0.48 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 3.0ms | 8.0ms | 8.5ms | 10.0ms | 0.37 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 2.9ms | 8.8ms | 13.6ms | 6.0ms | 0.48 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 3.2ms | 10.4ms | 11.1ms | 8.0ms | 0.40 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 3.4ms | 18.6ms | 13.9ms | 8.0ms | 0.43 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 3.2ms | 8.9ms | 9.5ms | 11.0ms | 0.36 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | 3.2ms | 9.4ms | 9.7ms | 7.0ms | 0.46 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 3.7ms | 10.6ms | 11.0ms | 8.0ms | 0.46 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | 3.6ms | 20.3ms | 17.5ms | 11.0ms | 0.33 | PASS |
| `fx_difficult_100000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 1.9ms | 23.9ms | 33.3ms | 9.0ms | 0.21 | PASS |
| `fx_difficult_100000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 3.3ms | 23.6ms | 30.1ms | 11.0ms | 0.30 | PASS |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 4.6ms | 0.533 | 57.2ms | 0.138 | 0.08 | PASS |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 9.1ms | 0.710 | 62.7ms | 0.108 | 0.14 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 6.3ms | 36.0ms | 56.5ms | 0.105 | 0.17 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 8.4ms | 29.5ms | 47.5ms | 20.0ms | 0.42 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 7.0ms | 0.590 | 71.0ms | 0.114 | 0.10 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 12.1ms | 2.126 | 0.109 | 0.332 | 0.11 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 7.4ms | 36.3ms | 40.9ms | 0.180 | 0.20 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | 12.4ms | 32.2ms | 48.0ms | 24.0ms | 0.52 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 10.7ms | 0.478 | 82.8ms | 0.371 | 0.13 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | 17.2ms | 3.324 | 0.168 | 0.536 | 0.10 | PASS |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 10.8ms | 0.159 | 0.183 | 49.0ms | 0.22 | PASS |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 30.7ms | 0.160 | 0.219 | 50.0ms | 0.61 | PASS |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 28.9ms | 17.486 | 0.383 | 1.412 | 0.08 | PASS |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.101 | 21.943 | 0.637 | 2.431 | 0.16 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | 26.0ms | 0.284 | 0.326 | 82.0ms | 0.32 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 1 | 56.9ms | 0.290 | 0.434 | 0.137 | 0.42 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 24.6ms | 0.224 | 0.270 | 79.0ms | 0.31 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 46.3ms | 0.230 | 0.371 | 0.116 | 0.40 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | 47.4ms | 48.805 | 0.835 | 4.711 | 0.06 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 1 | 0.148 | 58.179 | 1.469 | 8.403 | 0.10 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 46.0ms | 48.271 | 0.777 | 4.586 | 0.06 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.138 | 58.841 | 1.417 | 8.483 | 0.10 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 40.2ms | 0.301 | 0.362 | 0.113 | 0.36 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | 84.4ms | 0.317 | 0.553 | 0.213 | 0.40 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 69.7ms | 88.884 | 1.230 | 9.628 | 0.06 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.278 | 119.780 | 2.594 | 17.365 | 0.11 | PASS |
| `fx_difficult_10000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 0.141 | 2.069 | 2.420 | 0.595 | 0.24 | PASS |
| `fx_difficult_10000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.323 | timeout | 5.925 | 35.764 | 0.05 | PASS |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | 0.261 | 3.122 | 3.645 | 1.074 | 0.24 | PASS |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 0.228 | 2.916 | 3.408 | 0.857 | 0.27 | PASS |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | 0.529 | timeout | 13.569 | 77.090 | 0.04 | PASS |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.494 | timeout | 13.113 | 77.911 | 0.04 | PASS |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 0.332 | 3.797 | 4.709 | 1.355 | 0.25 | PASS |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.776 | timeout | 21.494 | 158.831 | 0.04 | PASS |
| `fx_simple_10000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 2.7ms | 6.4ms | 7.0ms | 8.0ms | 0.42 | PASS |
| `fx_simple_10000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 2.6ms | 6.9ms | 6.9ms | 5.0ms | 0.52 | PASS |
| `fx_simple_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 2.9ms | 9.3ms | 12.6ms | 5.0ms | 0.57 | PASS |
| `fx_simple_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 3.1ms | 10.0ms | 10.0ms | 6.0ms | 0.51 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 2.8ms | 8.3ms | 8.0ms | 6.0ms | 0.47 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 2.9ms | 11.4ms | 23.1ms | 12.0ms | 0.25 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 3.1ms | 8.8ms | 11.1ms | 6.0ms | 0.52 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 3.2ms | 10.1ms | 13.3ms | 7.0ms | 0.46 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 3.4ms | 9.7ms | 9.5ms | 6.0ms | 0.57 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | 3.2ms | 9.2ms | 10.7ms | 6.0ms | 0.54 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 3.7ms | 9.9ms | 12.5ms | 7.0ms | 0.53 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | 3.5ms | 11.6ms | 16.2ms | 9.0ms | 0.39 | PASS |
| `fx_simple_100000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 1.7ms | 24.3ms | 31.7ms | 10.0ms | 0.17 | PASS |
| `fx_simple_100000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 3.3ms | 22.3ms | 26.3ms | 9.0ms | 0.37 | PASS |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 3.0ms | 33.9ms | 64.0ms | 12.0ms | 0.25 | PASS |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 6.4ms | 37.5ms | 64.5ms | 11.0ms | 0.58 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 6.4ms | 32.6ms | 41.7ms | 13.0ms | 0.49 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 8.1ms | 28.8ms | 38.4ms | 17.0ms | 0.48 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 7.6ms | 32.1ms | 70.7ms | 16.0ms | 0.47 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 12.1ms | 38.2ms | 98.2ms | 25.0ms | 0.48 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 7.0ms | 34.4ms | 44.8ms | 15.0ms | 0.47 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | 12.5ms | 35.9ms | 50.6ms | 28.0ms | 0.45 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 11.1ms | 43.0ms | 91.0ms | 17.0ms | 0.66 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | 18.4ms | 52.2ms | 0.147 | 45.0ms | 0.41 | PASS |
| `fx_simple_1000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 11.0ms | 0.174 | 0.201 | 43.0ms | 0.26 | PASS |
| `fx_simple_1000000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 28.4ms | 0.169 | 0.236 | 53.0ms | 0.54 | PASS |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 19.0ms | 0.216 | 0.750 | 0.126 | 0.15 | PASS |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 60.1ms | 0.225 | 0.974 | 87.0ms | 0.69 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | 25.3ms | 0.301 | 0.343 | 87.0ms | 0.29 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 1 | 57.3ms | 0.307 | 0.440 | 0.137 | 0.42 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 23.8ms | 0.244 | 0.290 | 69.0ms | 0.34 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 46.8ms | 0.252 | 0.399 | 0.122 | 0.38 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | 36.5ms | 0.421 | 1.105 | 0.219 | 0.17 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 1 | 0.109 | 0.438 | 1.754 | 0.243 | 0.45 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 34.8ms | 0.381 | 1.057 | 0.167 | 0.21 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 98.3ms | 0.399 | 1.692 | 0.224 | 0.44 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 41.3ms | 0.306 | 0.368 | 0.114 | 0.36 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | 86.0ms | 0.318 | 0.551 | 0.215 | 0.40 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 65.0ms | 0.547 | 1.431 | 0.442 | 0.15 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.269 | 0.597 | 2.560 | 0.416 | 0.65 | PASS |
| `fx_simple_10000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 0.142 | 1.823 | 2.210 | 0.507 | 0.28 | PASS |
| `fx_simple_10000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.239 | 2.613 | 13.120 | 1.268 | 0.19 | PASS |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | 0.259 | 3.292 | 4.040 | 1.988 | 0.13 | PASS |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 0.227 | 2.594 | 3.148 | 0.841 | 0.27 | PASS |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | 0.475 | 4.521 | 17.356 | 3.537 | 0.13 | PASS |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.446 | 3.628 | 16.320 | 4.008 | 0.12 | PASS |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 0.334 | 3.636 | 4.366 | 1.423 | 0.23 | PASS |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.853 | 6.519 | 22.431 | 5.448 | 0.16 | PASS |

Coefficient gap:

| config | T | vs | fereg_py | pyfixest_map | pyfixest_lsmr | fixest |
|---|---:|---|---:|---:|---:|---:|
| `fx_difficult_10000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 4.5e-15 | 8.1e-16 | 8.1e-16 |  |
| `fx_difficult_10000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 4.5e-15 | 8.1e-16 | 8.1e-16 |  |
| `fx_difficult_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 2.6e-13 | 3.9e-12 | 2.5e-13 |  |
| `fx_difficult_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 2.6e-13 | 4.6e-13 | 2.5e-13 |  |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 6.7e-16 | 4.6e-15 | 4.6e-15 |  |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 6.7e-16 | 4.6e-15 | 4.2e-15 |  |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.8e-12 | 1.2e-11 | 1.8e-12 |  |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 1.8e-12 | 1.2e-11 | 1.8e-12 |  |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fixest | 1.7e-15 | 9.4e-15 | 8.7e-15 |  |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | fixest | 1.7e-15 | 9.4e-15 | 8.7e-15 |  |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 5.9e-12 | 6.1e-11 | 5.9e-12 |  |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 5.9e-12 | 1.1e-11 | 5.9e-12 |  |
| `fx_difficult_100000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 7.9e-15 | 7.0e-15 | 6.8e-15 |  |
| `fx_difficult_100000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 7.9e-15 | 6.6e-15 | 7.3e-15 |  |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 3.3e-10 | 6.8e-08 | 3.3e-10 |  |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 3.3e-10 | 6.8e-08 | 3.3e-10 |  |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.1e-14 | 2.1e-14 | 2.2e-14 |  |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 2.1e-14 | 2.1e-14 | 2.2e-14 |  |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.7e-09 | 7.9e-07 | 1.7e-09 |  |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 1.7e-09 | 7.9e-07 | 1.7e-09 |  |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.8e-14 | 2.6e-14 | 2.6e-14 |  |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | fixest | 2.8e-14 | 2.6e-14 | 2.6e-14 |  |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 3.3e-09 | 1.6e-06 | 3.3e-09 |  |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 3.3e-09 | 1.5e-06 | 3.3e-09 |  |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 3.0e-14 | 3.0e-14 | 3.0e-14 |  |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 3.0e-14 | 2.8e-14 | 2.8e-14 |  |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.4e-06 | 2.6e-06 | 1.4e-06 |  |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 1.4e-06 | 2.3e-06 | 1.4e-06 |  |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | fixest | 7.3e-14 | 7.3e-14 | 7.3e-14 |  |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 1 | fixest | 7.3e-14 | 7.3e-14 | 7.2e-14 |  |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 7.3e-14 | 7.3e-14 | 7.3e-14 |  |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 7.3e-14 | 7.3e-14 | 7.2e-14 |  |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | fixest | 1.4e-06 | 4.5e-06 | 1.4e-06 |  |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 1 | fixest | 1.4e-06 | 5.4e-06 | 1.4e-06 |  |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.4e-06 | 5.4e-06 | 1.4e-06 |  |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 1.4e-06 | 5.4e-06 | 1.4e-06 |  |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fixest | 7.3e-14 | 7.4e-14 | 7.3e-14 |  |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | fixest | 7.3e-14 | 7.3e-14 | 7.3e-14 |  |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 4.3e-06 | 1.0e-05 | 4.3e-06 |  |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 4.3e-06 | 1.1e-05 | 4.3e-06 |  |
| `fx_difficult_10000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 6.5e-14 | 6.7e-14 | 6.6e-14 |  |
| `fx_difficult_10000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.6e-08 | 8.8e-07 | 1.6e-08 |  |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | fixest | 1.7e-13 | 1.6e-13 | 1.6e-13 |  |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 1.7e-13 | 1.6e-13 | 1.6e-13 |  |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | fixest | 1.5e-06 |  | 1.5e-06 |  |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.5e-06 |  | 1.5e-06 |  |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.2e-13 | 2.1e-13 | 2.2e-13 |  |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 4.6e-06 |  | 4.6e-06 |  |
| `fx_simple_10000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 3.3e-15 | 2.6e-15 | 2.4e-15 |  |
| `fx_simple_10000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 3.3e-15 | 2.6e-15 | 2.4e-15 |  |
| `fx_simple_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 6.8e-15 | 5.2e-15 | 4.4e-15 |  |
| `fx_simple_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 6.8e-15 | 5.0e-15 | 4.4e-15 |  |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 1.1e-15 | 6.8e-15 | 6.7e-15 |  |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 1.1e-15 | 6.8e-15 | 6.7e-15 |  |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 7.5e-15 | 5.0e-15 | 5.8e-15 |  |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 7.5e-15 | 6.1e-15 | 5.8e-15 |  |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fixest | 1.8e-15 | 7.9e-15 | 7.9e-15 |  |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | fixest | 1.8e-15 | 8.3e-15 | 7.9e-15 |  |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.0e-14 | 6.0e-15 | 5.9e-15 |  |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 1.0e-14 | 4.9e-15 | 6.3e-15 |  |
| `fx_simple_100000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 7.7e-15 | 7.7e-15 | 7.7e-15 |  |
| `fx_simple_100000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 7.7e-15 | 7.7e-15 | 8.1e-15 |  |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 9.7e-15 | 9.0e-15 | 9.3e-15 |  |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 9.7e-15 | 8.6e-15 | 9.9e-15 |  |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.6e-14 | 2.5e-14 | 2.5e-14 |  |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 2.6e-14 | 2.5e-14 | 2.4e-14 |  |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 2.2e-14 | 1.9e-14 | 1.9e-14 |  |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 2.2e-14 | 1.9e-14 | 1.9e-14 |  |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.6e-14 | 2.4e-14 | 2.5e-14 |  |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | fixest | 2.6e-14 | 2.4e-14 | 2.4e-14 |  |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 2.2e-14 | 2.0e-14 | 2.0e-14 |  |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 2.2e-14 | 2.0e-14 | 2.0e-14 |  |
| `fx_simple_1000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.1e-15 | 2.4e-15 | 2.2e-15 |  |
| `fx_simple_1000000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 2.1e-15 | 3.0e-15 | 1.9e-15 |  |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 6.9e-15 | 7.4e-15 | 7.4e-15 |  |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 6.9e-15 | 7.0e-15 | 8.9e-15 |  |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | fixest | 8.7e-14 | 8.8e-14 | 8.9e-14 |  |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 1 | fixest | 8.7e-14 | 8.8e-14 | 8.9e-14 |  |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 8.7e-14 | 8.8e-14 | 8.9e-14 |  |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 8.7e-14 | 8.8e-14 | 8.9e-14 |  |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | fixest | 5.6e-14 | 5.6e-14 | 5.5e-14 |  |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 1 | fixest | 5.6e-14 | 5.8e-14 | 5.5e-14 |  |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 5.6e-14 | 5.7e-14 | 5.5e-14 |  |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 5.6e-14 | 5.6e-14 | 5.5e-14 |  |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fixest | 8.7e-14 | 8.8e-14 | 8.9e-14 |  |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | fixest | 8.7e-14 | 8.8e-14 | 8.9e-14 |  |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 5.6e-14 | 5.8e-14 | 5.5e-14 |  |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 5.6e-14 | 5.5e-14 | 5.5e-14 |  |
| `fx_simple_10000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 4.7e-14 | 4.8e-14 | 4.8e-14 |  |
| `fx_simple_10000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.9e-14 | 2.1e-14 | 2.1e-14 |  |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | fixest | 2.7e-13 | 2.7e-13 | 2.7e-13 |  |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.7e-13 | 2.7e-13 | 2.7e-13 |  |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | fixest | 1.7e-13 | 1.7e-13 | 1.7e-13 |  |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.7e-13 | 1.7e-13 | 1.7e-13 |  |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.7e-13 | 2.7e-13 | 2.7e-13 |  |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.7e-13 | 1.7e-13 | 1.6e-13 |  |

## G2 AKM

| config | T | fereg_py | pyfixest_map | pyfixest_lsmr | fixest | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `akm_baseline_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.134 | 6.777 | 0.471 | 0.425 | 0.32 | PASS |
| `akm_baseline_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.223 | 10.356 | 0.949 | 0.829 | 0.27 | PASS |
| `akm_freeze_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.101 | 17.859 | 0.575 | 0.552 | 0.18 | PASS |
| `akm_freeze_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.183 | 15.060 | 0.950 | 0.964 | 0.19 | PASS |
| `akm_freeze_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 94.9ms | 19.287 | 0.775 | 1.232 | 0.12 | PASS |
| `akm_freeze_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.176 | 22.908 | 0.840 | 1.670 | 0.21 | PASS |
| `akm_freeze_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 64.8ms | 8.295 | 0.491 | 0.476 | 0.14 | PASS |
| `akm_freeze_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.145 | 13.685 | 0.963 | 0.930 | 0.16 | PASS |
| `akm_freeze_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 52.9ms | 15.710 | 0.518 | 0.753 | 0.10 | PASS |
| `akm_freeze_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.133 | 19.560 | 0.908 | 1.228 | 0.15 | PASS |
| `akm_freeze_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 38.0ms | 33.232 | 0.545 | 1.068 | 0.07 | PASS |
| `akm_freeze_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.108 | 24.496 | 0.755 | 1.912 | 0.14 | PASS |
| `akm_freeze_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 30.5ms | error | 0.573 | 1.842 | 0.05 | PASS |
| `akm_freeze_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 99.0ms | error | 0.683 | 3.035 | 0.15 | PASS |
| `akm_interaction_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 52.0ms | 1.356 | 0.728 | 0.278 | 0.19 | PASS |
| `akm_interaction_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.185 | 1.769 | 0.987 | 0.413 | 0.45 | PASS |
| `akm_interaction_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 57.2ms | 1.282 | 0.807 | 0.220 | 0.26 | PASS |
| `akm_interaction_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.194 | 1.347 | 1.107 | 0.347 | 0.56 | PASS |
| `akm_interaction_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 34.8ms | 16.093 | 0.405 | 0.840 | 0.09 | PASS |
| `akm_interaction_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.108 | 23.808 | 0.758 | 1.430 | 0.14 | PASS |
| `akm_interaction_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 41.3ms | error | 0.568 | 2.608 | 0.07 | PASS |
| `akm_interaction_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.115 | error | 0.816 | 4.008 | 0.14 | PASS |
| `akm_mobility_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 38.1ms | 0.267 | 0.811 | 0.106 | 0.36 | PASS |
| `akm_mobility_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.108 | 0.299 | 1.210 | 0.146 | 0.74 | PASS |
| `akm_mobility_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 53.8ms | 0.824 | 0.551 | 0.207 | 0.26 | PASS |
| `akm_mobility_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.192 | 1.047 | 0.896 | 0.330 | 0.58 | PASS |
| `akm_mobility_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 54.9ms | 10.315 | 0.427 | 0.686 | 0.13 | PASS |
| `akm_mobility_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.133 | 15.429 | 0.782 | 1.172 | 0.17 | PASS |
| `akm_mobility_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 32.2ms | 21.857 | 0.359 | 1.145 | 0.09 | PASS |
| `akm_mobility_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 94.4ms | 29.990 | 0.598 | 2.340 | 0.16 | PASS |
| `akm_mobility_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 30.8ms | error | 0.374 | 1.437 | 0.08 | PASS |
| `akm_mobility_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.101 | error | 0.697 | 2.591 | 0.14 | PASS |
| `akm_mobility_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 27.5ms | error | 0.360 | 1.467 | 0.08 | PASS |
| `akm_mobility_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 95.1ms | error | 0.653 | 2.618 | 0.15 | PASS |
| `akm_scale_1_10000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 10.1ms | 83.0ms | 25.1ms | 16.0ms | 0.63 | PASS |
| `akm_scale_1_10000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 10.9ms | 0.118 | 12.4ms | 21.0ms | 0.88 | PASS |
| `akm_scale_2_100000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 10.9ms | 0.401 | 51.2ms | 64.0ms | 0.21 | PASS |
| `akm_scale_2_100000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 16.4ms | 0.770 | 69.3ms | 0.109 | 0.24 | PASS |
| `akm_scale_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.108 | 3.794 | 0.453 | 0.344 | 0.31 | PASS |
| `akm_scale_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.193 | 4.990 | 0.828 | 0.589 | 0.33 | PASS |
| `akm_scale_4_10000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.860 | 47.856 | 4.379 | 4.616 | 0.20 | PASS |
| `akm_sorting_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 62.9ms | 3.251 | 0.409 | 0.313 | 0.20 | PASS |
| `akm_sorting_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.140 | 3.808 | 0.750 | 0.523 | 0.27 | PASS |
| `akm_sorting_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.141 | 6.387 | 0.450 | 0.476 | 0.31 | PASS |
| `akm_sorting_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.415 | 9.480 | 0.819 | 0.829 | 0.51 | PASS |
| `akm_sorting_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.231 | 5.353 | 0.661 | 0.508 | 0.45 | PASS |
| `akm_sorting_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.469 | 7.077 | 0.830 | 0.855 | 0.57 | PASS |
| `akm_sorting_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.238 | 8.786 | 0.678 | 0.463 | 0.51 | PASS |
| `akm_sorting_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.507 | 10.257 | 0.919 | 0.829 | 0.61 | PASS |
| `akm_sorting_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.195 | 14.752 | 0.482 | 0.712 | 0.40 | PASS |
| `akm_sorting_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.395 | 17.849 | 0.806 | 1.270 | 0.49 | PASS |

Coefficient gap:

| config | T | vs | fereg_py | pyfixest_map | pyfixest_lsmr | fixest |
|---|---:|---|---:|---:|---:|---:|
| `akm_baseline_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 4.6e-14 | 1.3e-13 | 4.5e-14 |  |
| `akm_baseline_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 4.6e-14 | 1.2e-13 | 4.4e-14 |  |
| `akm_freeze_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 2.6e-14 | 6.7e-14 | 2.4e-14 |  |
| `akm_freeze_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 2.6e-14 | 4.7e-14 | 2.5e-14 |  |
| `akm_freeze_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 2.4e-15 | 7.2e-13 | 2.4e-15 |  |
| `akm_freeze_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 2.4e-15 | 7.1e-13 | 1.6e-15 |  |
| `akm_freeze_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 6.9e-14 | 1.1e-16 | 7.3e-14 |  |
| `akm_freeze_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 6.9e-14 | 1.7e-13 | 7.2e-14 |  |
| `akm_freeze_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 7.4e-14 | 2.7e-13 | 7.1e-14 |  |
| `akm_freeze_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 7.4e-14 | 2.7e-13 | 7.1e-14 |  |
| `akm_freeze_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 1.4e-14 | 9.4e-13 | 1.8e-14 |  |
| `akm_freeze_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 1.4e-14 | 9.6e-13 | 1.9e-14 |  |
| `akm_freeze_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 3.2e-10 |  | 3.2e-10 |  |
| `akm_freeze_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 3.2e-10 |  | 3.2e-10 |  |
| `akm_interaction_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 4.4e-16 | 1.1e-15 | 8.9e-16 |  |
| `akm_interaction_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 4.4e-16 | 2.0e-15 | 2.0e-15 |  |
| `akm_interaction_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 1.1e-13 | 4.4e-12 | 1.1e-13 |  |
| `akm_interaction_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 1.1e-13 | 4.4e-12 | 1.1e-13 |  |
| `akm_interaction_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 3.5e-13 | 8.7e-13 | 3.5e-13 |  |
| `akm_interaction_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 3.5e-13 | 8.7e-13 | 3.5e-13 |  |
| `akm_interaction_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 2.4e-08 |  | 2.4e-08 |  |
| `akm_interaction_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 2.4e-08 |  | 2.4e-08 |  |
| `akm_mobility_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 6.9e-14 | 6.3e-14 | 6.7e-14 |  |
| `akm_mobility_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 6.9e-14 | 5.6e-14 | 6.6e-14 |  |
| `akm_mobility_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 2.0e-14 | 2.3e-14 | 2.3e-14 |  |
| `akm_mobility_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 2.0e-14 | 2.1e-14 | 2.2e-14 |  |
| `akm_mobility_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 7.0e-14 | 2.6e-13 | 7.0e-14 |  |
| `akm_mobility_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 7.0e-14 | 2.8e-13 | 6.9e-14 |  |
| `akm_mobility_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 9.1e-13 | 1.7e-12 | 9.1e-13 |  |
| `akm_mobility_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 9.1e-13 | 1.8e-12 | 9.0e-13 |  |
| `akm_mobility_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 4.7e-10 |  | 4.7e-10 |  |
| `akm_mobility_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 4.7e-10 |  | 4.7e-10 |  |
| `akm_mobility_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 5.4e-07 |  | 5.4e-07 |  |
| `akm_mobility_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 5.4e-07 |  | 5.4e-07 |  |
| `akm_scale_1_10000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 5.6e-12 | 1.2e-10 | 5.6e-12 |  |
| `akm_scale_1_10000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 5.6e-12 | 1.2e-10 | 5.6e-12 |  |
| `akm_scale_2_100000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 3.0e-13 | 1.1e-11 | 3.0e-13 |  |
| `akm_scale_2_100000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 3.0e-13 | 1.1e-11 | 3.0e-13 |  |
| `akm_scale_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 2.8e-14 | 2.0e-14 | 2.8e-14 |  |
| `akm_scale_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 2.8e-14 | 1.8e-14 | 2.9e-14 |  |
| `akm_scale_4_10000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 2.1e-13 | 2.1e-13 | 2.1e-13 |  |
| `akm_sorting_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 8.8e-16 | 1.1e-14 | 0.0e+00 |  |
| `akm_sorting_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 8.8e-16 | 1.1e-14 | 8.8e-16 |  |
| `akm_sorting_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 2.0e-13 | 1.3e-14 | 2.1e-13 |  |
| `akm_sorting_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 2.0e-13 | 1.3e-14 | 2.1e-13 |  |
| `akm_sorting_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 3.0e-14 | 8.2e-14 | 2.9e-14 |  |
| `akm_sorting_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 3.0e-14 | 8.3e-14 | 3.0e-14 |  |
| `akm_sorting_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 1.0e-11 | 7.2e-11 | 1.0e-11 |  |
| `akm_sorting_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 1.0e-11 | 7.2e-11 | 1.0e-11 |  |
| `akm_sorting_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 7.1e-12 | 8.4e-11 | 7.1e-12 |  |
| `akm_sorting_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 7.1e-12 | 6.1e-11 | 7.1e-12 |  |

## G3 AKM with occupation FE

| config | T | fereg_py | pyfixest_map | pyfixest_lsmr | fixest | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `akmocc_baseline_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.110 | 4.653 | 0.695 | 0.696 | 0.16 | PASS |
| `akmocc_baseline_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.228 | 6.924 | 1.276 | 1.243 | 0.18 | PASS |
| `akmocc_occlambda_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.124 | 6.457 | 0.908 | 0.816 | 0.15 | PASS |
| `akmocc_occlambda_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.243 | 6.810 | 1.374 | 1.246 | 0.20 | PASS |
| `akmocc_occlambda_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.113 | 6.945 | 0.715 | 0.883 | 0.16 | PASS |
| `akmocc_occlambda_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.241 | 9.488 | 1.323 | 1.590 | 0.18 | PASS |
| `akmocc_occlambda_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.148 | 5.719 | 0.773 | 0.978 | 0.19 | PASS |
| `akmocc_occlambda_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.268 | 8.012 | 1.407 | 1.736 | 0.19 | PASS |
| `akmocc_occlambda_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.108 | 24.836 | 0.830 | 2.303 | 0.13 | PASS |
| `akmocc_occlambda_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.229 | 35.423 | 1.601 | 4.265 | 0.14 | PASS |
| `akmocc_occsize_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 90.8ms | 5.668 | 0.580 | 0.633 | 0.16 | PASS |
| `akmocc_occsize_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.199 | 9.818 | 1.116 | 1.278 | 0.18 | PASS |
| `akmocc_occsize_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.119 | 6.142 | 0.641 | 0.832 | 0.18 | PASS |
| `akmocc_occsize_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.236 | 7.149 | 1.111 | 1.118 | 0.21 | PASS |
| `akmocc_occsize_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.193 | 8.152 | 0.825 | 1.269 | 0.23 | PASS |
| `akmocc_occsize_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.314 | 10.323 | 1.517 | 2.120 | 0.21 | PASS |
| `akmocc_occsize_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.775 | 32.982 | 1.398 | 2.310 | 0.55 | PASS |
| `akmocc_occsize_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 1.527 | 46.309 | 2.806 | 3.668 | 0.54 | PASS |

Coefficient gap:

| config | T | vs | fereg_py | pyfixest_map | pyfixest_lsmr | fixest |
|---|---:|---|---:|---:|---:|---:|
| `akmocc_baseline_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fixest | 2.2e-14 | 8.7e-15 | 4.2e-15 |  |
| `akmocc_baseline_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fixest | 2.2e-14 | 4.6e-14 | 5.3e-15 |  |
| `akmocc_occlambda_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fixest | 4.6e-14 | 4.9e-14 | 1.8e-14 |  |
| `akmocc_occlambda_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fixest | 4.6e-14 | 4.9e-14 | 1.9e-14 |  |
| `akmocc_occlambda_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fixest | 3.0e-14 | 3.0e-13 | 5.2e-14 |  |
| `akmocc_occlambda_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fixest | 3.0e-14 | 3.0e-13 | 5.1e-14 |  |
| `akmocc_occlambda_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fixest | 3.0e-13 | 1.6e-13 | 2.8e-13 |  |
| `akmocc_occlambda_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fixest | 3.0e-13 | 1.8e-13 | 2.8e-13 |  |
| `akmocc_occlambda_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fixest | 1.7e-09 | 1.7e-09 | 1.7e-09 |  |
| `akmocc_occlambda_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fixest | 1.7e-09 | 1.7e-09 | 1.7e-09 |  |
| `akmocc_occsize_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fixest | 6.2e-15 | 1.2e-13 | 3.3e-15 |  |
| `akmocc_occsize_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fixest | 6.2e-15 | 1.2e-13 | 3.1e-15 |  |
| `akmocc_occsize_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fixest | 1.6e-14 | 5.3e-14 | 9.1e-15 |  |
| `akmocc_occsize_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fixest | 1.6e-14 | 5.5e-14 | 8.9e-15 |  |
| `akmocc_occsize_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fixest | 1.7e-12 | 1.0e-11 | 1.7e-12 |  |
| `akmocc_occsize_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fixest | 1.7e-12 | 1.0e-11 | 1.7e-12 |  |
| `akmocc_occsize_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fixest | 2.2e-08 | 2.2e-08 | 2.2e-08 |  |
| `akmocc_occsize_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fixest | 2.2e-08 | 2.3e-08 | 2.2e-08 |  |

## G4 IV/2SLS

| config | T | fereg_py | pyfixest_map | pyfixest_lsmr | fixest | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 5.7ms | 54.4ms | 59.0ms | 20.0ms | 0.28 | PASS |
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | 9.6ms | 57.4ms | 65.8ms | 26.0ms | 0.37 | PASS |
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | 8.2ms | 1.473 | 97.1ms | 0.116 | 0.08 | PASS |
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | 15.7ms | 3.071 | 0.171 | 0.261 | 0.09 | PASS |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 21.9ms | 0.423 | 0.483 | 0.113 | 0.19 | PASS |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | 63.9ms | 0.432 | 0.641 | 0.169 | 0.38 | PASS |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | 45.3ms | 71.852 | 1.180 | 3.022 | 0.04 | PASS |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | 0.156 | 90.695 | 2.203 | 6.066 | 0.07 | PASS |
| `fx_difficult_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 0.234 | 4.717 | 5.406 | 1.321 | 0.18 | PASS |
| `fx_difficult_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | 0.478 | timeout | 20.033 | 68.012 | 0.02 | PASS |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 5.7ms | 54.6ms | 0.106 | 26.0ms | 0.22 | PASS |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | 9.4ms | 56.2ms | 67.2ms | 26.0ms | 0.36 | PASS |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | 5.9ms | 60.6ms | 96.0ms | 26.0ms | 0.23 | PASS |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | 13.7ms | 71.9ms | 0.147 | 33.0ms | 0.42 | PASS |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 22.2ms | 0.423 | 0.483 | 0.113 | 0.20 | PASS |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | 61.5ms | 0.432 | 0.646 | 0.168 | 0.37 | PASS |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | 35.5ms | 0.599 | 1.397 | 0.166 | 0.21 | PASS |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | 0.113 | 0.651 | 2.330 | 0.251 | 0.45 | PASS |
| `fx_simple_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 0.236 | 4.687 | 5.439 | 1.314 | 0.18 | PASS |
| `fx_simple_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | 0.420 | 6.996 | 22.721 | 2.672 | 0.16 | PASS |

Coefficient gap:

| config | T | vs | fereg_py | pyfixest_map | pyfixest_lsmr | fixest |
|---|---:|---|---:|---:|---:|---:|
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fixest | 6.6e-15 | 7.7e-15 | 6.8e-15 |  |
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | fixest | 6.6e-15 | 7.7e-15 | 6.8e-15 |  |
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 2.2e-10 | 1.2e-07 | 2.2e-10 |  |
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 2.2e-10 | 1.2e-07 | 2.2e-10 |  |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fixest | 7.0e-14 | 6.9e-14 | 6.9e-14 |  |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | fixest | 7.0e-14 | 6.9e-14 | 6.9e-14 |  |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 9.0e-06 | 5.7e-06 | 9.0e-06 |  |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 9.0e-06 | 5.9e-06 | 9.0e-06 |  |
| `fx_difficult_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fixest | 3.5e-13 | 3.5e-13 | 3.5e-13 |  |
| `fx_difficult_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 2.4e-06 |  | 2.4e-06 |  |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fixest | 2.3e-14 | 2.2e-14 | 2.2e-14 |  |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | fixest | 2.3e-14 | 2.2e-14 | 2.2e-14 |  |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 5.0e-15 | 7.1e-15 | 6.2e-15 |  |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 5.0e-15 | 6.6e-15 | 6.4e-15 |  |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fixest | 4.0e-15 | 3.1e-15 | 2.1e-15 |  |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | fixest | 4.0e-15 | 3.1e-15 | 2.3e-15 |  |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 5.7e-14 | 5.7e-14 | 5.7e-14 |  |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 5.7e-14 | 5.7e-14 | 5.8e-14 |  |
| `fx_simple_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fixest | 1.7e-13 | 1.8e-13 | 1.7e-13 |  |
| `fx_simple_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 2.4e-13 | 2.4e-13 | 2.4e-13 |  |

## G5 PPML

| config | T | fereg_py | pyfixest_map | pyfixest_lsmr | fixest | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 16.3ms | 0.138 | 0.322 | 52.0ms | 0.31 | PASS |
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | 38.0ms | 0.140 | 0.501 | 64.0ms | 0.59 | PASS |
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 30.2ms | error | 0.440 | 0.895 | 0.07 | PASS |
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 74.9ms | 26.324 | 0.789 | 1.572 | 0.09 | PASS |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | 20.6ms | 0.162 | 0.478 | 71.0ms | 0.29 | PASS |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | 55.5ms | 0.218 | 1.261 | 0.146 | 0.38 | PASS |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 40.8ms | error | 0.673 | 0.998 | 0.06 | PASS |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 0.102 | error | 2.031 | 4.247 | 0.05 | PASS |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 0.126 | 1.376 | 3.691 | 0.361 | 0.35 | PASS |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | 0.356 | 1.462 | 8.212 | 0.528 | 0.67 | PASS |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 0.246 | error | 6.825 | 40.142 | 0.04 | PASS |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 0.924 | error | 17.954 | 68.736 | 0.05 | PASS |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | 0.226 | 2.284 | 9.349 | 0.627 | 0.36 | PASS |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | 0.516 | 2.437 | 21.547 | 1.367 | 0.38 | PASS |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 0.378 | error | 18.248 | 137.512 | 0.02 | PASS |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 1.216 | error | 39.497 | 216.931 | 0.03 | PASS |
| `fx_difficult_10000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 1.557 | 15.348 | 49.447 | 4.078 | 0.38 | PASS |
| `fx_difficult_10000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 3.023 | error | 109.599 | timeout | 0.03 | PASS |
| `fx_difficult_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | 2.539 | 24.084 | 126.197 | 6.981 | 0.36 | PASS |
| `fx_difficult_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 4.875 | timeout | 247.045 | timeout | 0.02 | PASS |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 16.5ms | 0.170 | 0.374 | 74.0ms | 0.22 | PASS |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | 34.8ms | 0.134 | 0.487 | 79.0ms | 0.44 | PASS |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 22.8ms | 0.227 | 0.477 | 88.0ms | 0.26 | PASS |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 65.0ms | 0.259 | 0.907 | 0.132 | 0.49 | PASS |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | 19.2ms | 0.162 | 0.454 | 68.0ms | 0.28 | PASS |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | 50.2ms | 0.210 | 1.175 | 0.142 | 0.35 | PASS |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 34.7ms | 0.252 | 0.646 | 0.115 | 0.30 | PASS |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 79.9ms | 0.584 | 2.242 | 0.349 | 0.23 | PASS |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 0.125 | 1.336 | 3.577 | 0.355 | 0.35 | PASS |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | 0.349 | 1.374 | 7.871 | 0.504 | 0.69 | PASS |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 0.189 | 3.677 | 8.550 | 1.110 | 0.17 | PASS |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 0.634 | 4.410 | 18.365 | 1.715 | 0.37 | PASS |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | 0.225 | 2.179 | 8.792 | 0.635 | 0.35 | PASS |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | 0.510 | 2.307 | 20.359 | 1.316 | 0.39 | PASS |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 0.343 | 11.347 | 18.015 | 4.216 | 0.08 | PASS |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 1.051 | 13.349 | 47.009 | 5.239 | 0.20 | PASS |
| `fx_simple_10000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 1.565 | 16.104 | 53.108 | 4.224 | 0.37 | PASS |
| `fx_simple_10000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 2.676 | 394.828 | 202.763 | 43.144 | 0.06 | PASS |
| `fx_simple_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | 2.541 | 26.552 | 140.230 | 7.291 | 0.35 | PASS |
| `fx_simple_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 5.542 | timeout | 402.220 | 94.012 | 0.06 | PASS |

Coefficient gap:

| config | T | vs | fereg_py | pyfixest_map | pyfixest_lsmr | fixest |
|---|---:|---|---:|---:|---:|---:|
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fixest | 8.9e-16 | 1.0e-15 | 6.2e-13 |  |
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | fixest | 8.9e-16 | 5.1e-15 | 4.2e-13 |  |
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 5.4e-12 |  | 6.0e-12 |  |
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 5.4e-12 | 7.9e-08 | 6.0e-12 |  |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | fixest | 1.5e-14 | 2.8e-13 | 1.5e-12 |  |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | fixest | 1.5e-14 | 2.8e-13 | 1.5e-12 |  |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 1.2e-09 |  | 1.2e-09 |  |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 1.2e-09 |  | 1.2e-09 |  |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fixest | 5.6e-16 | 6.7e-16 | 3.7e-13 |  |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | fixest | 5.6e-16 | 7.8e-16 | 3.7e-13 |  |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 1.5e-06 |  | 1.5e-06 |  |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 1.5e-06 |  | 1.5e-06 |  |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | fixest | 4.3e-14 | 1.9e-13 | 2.5e-12 |  |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | fixest | 4.3e-14 | 1.4e-13 | 2.5e-12 |  |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 5.6e-05 |  | 5.6e-05 |  |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 5.6e-05 |  | 5.6e-05 |  |
| `fx_difficult_10000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fixest | 1.2e-14 | 1.1e-14 | 3.8e-13 |  |
| `fx_difficult_10000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 6.6e-06 |  | 6.6e-06 |  |
| `fx_difficult_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | fixest | 2.6e-13 | 2.1e-13 | 2.1e-12 |  |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fixest | 4.4e-15 | 1.0e-14 | 1.0e-12 |  |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | fixest | 4.4e-15 | 7.1e-15 | 1.0e-12 |  |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 9.9e-15 | 7.9e-14 | 1.3e-12 |  |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 9.9e-15 | 1.2e-13 | 1.3e-12 |  |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | fixest | 1.8e-14 | 1.0e-13 | 3.5e-12 |  |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | fixest | 1.8e-14 | 1.0e-13 | 3.8e-12 |  |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 1.4e-13 | 4.7e-13 | 2.1e-12 |  |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 1.4e-13 | 4.7e-13 | 2.0e-12 |  |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fixest | 4.4e-15 | 3.1e-15 | 4.2e-13 |  |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | fixest | 4.4e-15 | 3.5e-15 | 4.2e-13 |  |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 2.2e-14 | 1.7e-13 | 8.1e-13 |  |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 2.2e-14 | 1.5e-13 | 8.3e-13 |  |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | fixest | 8.0e-14 | 9.1e-14 | 2.8e-12 |  |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | fixest | 8.0e-14 | 8.2e-14 | 2.8e-12 |  |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 2.1e-13 | 9.9e-14 | 1.7e-12 |  |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 2.1e-13 | 6.7e-14 | 1.7e-12 |  |
| `fx_simple_10000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fixest | 5.7e-14 | 5.5e-14 | 5.2e-13 |  |
| `fx_simple_10000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 1.1e-13 | 1.1e-11 | 7.8e-12 |  |
| `fx_simple_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | fixest | 1.6e-13 | 3.9e-13 | 8.8e-13 |  |
| `fx_simple_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 2.0e-12 |  | 7.6e-12 |  |

## G6 Logit

| config | T | fereg_py | pyfixest_map | pyfixest_lsmr | fixest | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 14.5ms | 0.135 | 0.283 | 38.0ms | 0.38 | PASS |
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | 38.7ms | 0.144 | 0.386 | 48.0ms | 0.81 | PASS |
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 29.1ms | 4.478 | 0.563 | 0.460 | 0.06 | PASS |
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 72.0ms | 7.566 | 0.899 | 0.774 | 0.09 | PASS |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 20.0ms | 0.220 | 0.438 | 58.0ms | 0.34 | PASS |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 53.6ms | 0.218 | 0.888 | 0.102 | 0.53 | PASS |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 39.4ms | 6.191 | 0.858 | 0.596 | 0.07 | PASS |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 99.3ms | 23.444 | 2.029 | 2.087 | 0.05 | PASS |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 0.118 | 1.486 | 3.090 | 0.271 | 0.43 | PASS |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | 0.362 | 1.531 | 6.238 | 0.403 | 0.90 | PASS |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.215 | error | 6.903 | 17.635 | 0.03 | PASS |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.820 | error | 14.916 | 30.193 | 0.06 | PASS |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 0.209 | 2.447 | 6.777 | 0.512 | 0.41 | PASS |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 0.510 | 2.657 | 14.933 | 0.980 | 0.52 | PASS |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.358 | error | 22.672 | 72.274 | 0.02 | PASS |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 1.114 | error | 50.058 | 103.166 | 0.02 | PASS |
| `fx_difficult_10000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 1.291 | 17.712 | 42.518 | 2.959 | 0.44 | PASS |
| `fx_difficult_10000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 2.416 | timeout | 114.788 | timeout | 0.02 | PASS |
| `fx_difficult_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 2.238 | 28.051 | 94.733 | 5.356 | 0.42 | PASS |
| `fx_difficult_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 4.039 | timeout | 418.058 | timeout | 0.01 | PASS |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 14.4ms | 0.161 | 0.319 | 43.0ms | 0.34 | PASS |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | 38.6ms | 0.143 | 0.362 | 50.0ms | 0.77 | PASS |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 21.0ms | 0.179 | 0.603 | 59.0ms | 0.36 | PASS |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 63.1ms | 0.188 | 0.894 | 85.0ms | 0.74 | PASS |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 20.6ms | 0.219 | 0.426 | 57.0ms | 0.36 | PASS |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 52.1ms | 0.214 | 0.838 | 0.102 | 0.51 | PASS |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 38.8ms | 0.255 | 0.971 | 87.0ms | 0.45 | PASS |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 95.1ms | 0.346 | 2.639 | 0.214 | 0.44 | PASS |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 0.117 | 1.472 | 3.073 | 0.268 | 0.44 | PASS |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | 0.360 | 1.505 | 6.203 | 0.396 | 0.91 | PASS |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.183 | 1.996 | 9.874 | 0.524 | 0.35 | PASS |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.633 | 2.145 | 19.250 | 0.814 | 0.78 | PASS |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 0.203 | 2.479 | 6.707 | 0.522 | 0.39 | PASS |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 0.500 | 2.596 | 14.642 | 0.969 | 0.52 | PASS |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.330 | 3.847 | 27.925 | 1.700 | 0.19 | PASS |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 1.019 | 4.341 | 67.534 | 2.230 | 0.46 | PASS |
| `fx_simple_10000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 1.440 | 18.477 | 45.549 | 3.423 | 0.42 | PASS |
| `fx_simple_10000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 2.248 | 26.674 | 224.324 | 8.835 | 0.25 | PASS |
| `fx_simple_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 2.267 | 29.434 | 117.308 | 5.432 | 0.42 | PASS |
| `fx_simple_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 4.973 | 46.245 | timeout | 24.112 | 0.21 | PASS |

Coefficient gap:

| config | T | vs | fereg_py | pyfixest_map | pyfixest_lsmr | fixest |
|---|---:|---|---:|---:|---:|---:|
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 1.9e-15 | 1.0e-08 | 1.0e-08 |  |
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 1.9e-15 | 1.0e-08 | 1.0e-08 |  |
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 7.6e-10 | 8.9e-10 | 8.9e-10 |  |
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 7.6e-10 | 8.9e-10 | 8.9e-10 |  |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 3.9e-14 | 1.6e-11 | 1.6e-11 |  |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 3.9e-14 | 1.6e-11 | 1.6e-11 |  |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 3.0e-09 | 3.0e-09 | 3.0e-09 |  |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 3.0e-09 | 3.0e-09 | 3.0e-09 |  |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 1.8e-14 | 2.8e-09 | 2.8e-09 |  |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 1.8e-14 | 2.8e-09 | 2.8e-09 |  |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 5.7e-08 |  | 5.9e-08 |  |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 5.7e-08 |  | 5.9e-08 |  |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 4.2e-14 | 7.6e-12 | 7.6e-12 |  |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 4.2e-14 | 7.7e-12 | 7.6e-12 |  |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 3.9e-07 |  | 4.1e-07 |  |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 3.9e-07 |  | 4.1e-07 |  |
| `fx_difficult_10000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 4.1e-14 | 2.1e-09 | 2.1e-09 |  |
| `fx_difficult_10000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 2.2e-06 |  | 2.2e-06 |  |
| `fx_difficult_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 1.2e-13 | 1.9e-11 | 1.9e-11 |  |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 3.6e-16 | 1.1e-09 | 1.1e-09 |  |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 3.6e-16 | 1.1e-09 | 1.1e-09 |  |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 2.4e-15 | 3.0e-11 | 3.0e-11 |  |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 2.4e-15 | 3.0e-11 | 3.0e-11 |  |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.1e-14 | 1.5e-11 | 1.5e-11 |  |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 2.1e-14 | 1.5e-11 | 1.5e-11 |  |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 3.1e-14 | 1.3e-09 | 1.3e-09 |  |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 3.1e-14 | 1.3e-09 | 1.3e-09 |  |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.0e-14 | 1.4e-09 | 1.4e-09 |  |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 2.0e-14 | 1.4e-09 | 1.4e-09 |  |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.4e-14 | 3.5e-10 | 3.5e-10 |  |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 1.4e-14 | 3.5e-10 | 3.5e-10 |  |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 9.2e-14 | 2.8e-12 | 2.9e-12 |  |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 9.2e-14 | 2.8e-12 | 2.9e-12 |  |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 4.6e-14 | 3.0e-08 | 3.0e-08 |  |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 4.6e-14 | 3.0e-08 | 3.0e-08 |  |
| `fx_simple_10000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 3.6e-14 | 5.3e-13 | 5.8e-13 |  |
| `fx_simple_10000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 4.5e-14 | 1.8e-08 | 1.8e-08 |  |
| `fx_simple_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 1.8e-13 | 2.4e-08 | 2.4e-08 |  |
| `fx_simple_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 2.1e-13 | 9.4e-09 | 9.2e-09 |  |

## G7 Heterogeneous slopes

| config | T | fereg_py | pyfixest_map | pyfixest_lsmr | fixest | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | 14.3ms | unsupported | unsupported | 0.148 | 0.10 | PASS |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | 23.1ms | unsupported | unsupported | 0.269 | 0.09 | PASS |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | 69.6ms | unsupported | unsupported | 4.148 | 0.02 | PASS |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | 0.229 | unsupported | unsupported | 6.503 | 0.04 | PASS |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | 4.6ms | unsupported | unsupported | 18.0ms | 0.26 | PASS |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | 10.2ms | unsupported | unsupported | 21.0ms | 0.49 | PASS |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | 30.6ms | unsupported | unsupported | 0.120 | 0.25 | PASS |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | 0.101 | unsupported | unsupported | 0.166 | 0.61 | PASS |

Coefficient gap:

| config | T | vs | fereg_py | pyfixest_map | pyfixest_lsmr | fixest |
|---|---:|---|---:|---:|---:|---:|
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | fixest | 4.3e-11 |  |  |  |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | fixest | 4.3e-11 |  |  |  |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | fixest | 4.9e-07 |  |  |  |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | fixest | 4.9e-07 |  |  |  |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | fixest | 8.8e-15 |  |  |  |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | fixest | 8.8e-15 |  |  |  |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | fixest | 2.5e-14 |  |  |  |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | fixest | 2.5e-14 |  |  |  |

# R package `fereg` gate

`ratio` = fereg_r / fastest finished competitor among fixest, pyfixest_map, pyfixest_lsmr. Gate: PASS iff fereg_r is
faster than every one of them that finished (a timeout counts as slower; errors and `unsupported`
are excluded). Coefficient gap: max relative difference versus fixest (versus the Stata fereg
when fixest has no coefficients for the row).

**fereg_r gate summary:** 273 PASS, 0 FAIL.

## G1 OLS (fixest DGP)

| config | T | fereg_r | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `fx_difficult_10000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 1.0ms | 22.0ms | 6.6ms | 7.2ms | 0.15 | PASS |
| `fx_difficult_10000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 1.0ms | 5.0ms | 6.4ms | 7.2ms | 0.20 | PASS |
| `fx_difficult_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 2.0ms | 28.0ms | 9.8ms | 11.9ms | 0.21 | PASS |
| `fx_difficult_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 1.0ms | 6.0ms | 10.7ms | 10.6ms | 0.17 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 2.0ms | 10.0ms | 8.0ms | 8.5ms | 0.25 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 2.0ms | 6.0ms | 8.8ms | 13.6ms | 0.33 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 2.0ms | 8.0ms | 10.4ms | 11.1ms | 0.25 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 2.0ms | 8.0ms | 18.6ms | 13.9ms | 0.25 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 2.0ms | 11.0ms | 8.9ms | 9.5ms | 0.22 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | 2.0ms | 7.0ms | 9.4ms | 9.7ms | 0.29 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 2.0ms | 8.0ms | 10.6ms | 11.0ms | 0.25 | PASS |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | 3.0ms | 11.0ms | 20.3ms | 17.5ms | 0.27 | PASS |
| `fx_difficult_100000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 3.0ms | 9.0ms | 23.9ms | 33.3ms | 0.33 | PASS |
| `fx_difficult_100000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 5.0ms | 11.0ms | 23.6ms | 30.1ms | 0.45 | PASS |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 6.0ms | 0.138 | 0.533 | 57.2ms | 0.10 | PASS |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 10.0ms | 0.108 | 0.710 | 62.7ms | 0.16 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 4.0ms | 0.105 | 36.0ms | 56.5ms | 0.11 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 6.0ms | 20.0ms | 29.5ms | 47.5ms | 0.30 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 7.0ms | 0.114 | 0.590 | 71.0ms | 0.10 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 11.0ms | 0.332 | 2.126 | 0.109 | 0.10 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 5.0ms | 0.180 | 36.3ms | 40.9ms | 0.14 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | 10.0ms | 24.0ms | 32.2ms | 48.0ms | 0.42 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 7.0ms | 0.371 | 0.478 | 82.8ms | 0.08 | PASS |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | 14.0ms | 0.536 | 3.324 | 0.168 | 0.08 | PASS |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 19.0ms | 49.0ms | 0.159 | 0.183 | 0.39 | PASS |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 35.0ms | 50.0ms | 0.160 | 0.219 | 0.70 | PASS |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 36.0ms | 1.412 | 17.486 | 0.383 | 0.09 | PASS |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.111 | 2.431 | 21.943 | 0.637 | 0.17 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | 29.0ms | 82.0ms | 0.284 | 0.326 | 0.35 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 1 | 56.0ms | 0.137 | 0.290 | 0.434 | 0.41 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 29.0ms | 79.0ms | 0.224 | 0.270 | 0.37 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 44.0ms | 0.116 | 0.230 | 0.371 | 0.38 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | 52.0ms | 4.711 | 48.805 | 0.835 | 0.06 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 1 | 0.147 | 8.403 | 58.179 | 1.469 | 0.10 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 50.0ms | 4.586 | 48.271 | 0.777 | 0.06 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.141 | 8.483 | 58.841 | 1.417 | 0.10 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 39.0ms | 0.113 | 0.301 | 0.362 | 0.35 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | 88.0ms | 0.213 | 0.317 | 0.553 | 0.41 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 72.0ms | 9.628 | 88.884 | 1.230 | 0.06 | PASS |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.288 | 17.365 | 119.780 | 2.594 | 0.11 | PASS |
| `fx_difficult_10000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 0.158 | 0.595 | 2.069 | 2.420 | 0.27 | PASS |
| `fx_difficult_10000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.345 | 35.764 | timeout | 5.925 | 0.06 | PASS |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | 0.266 | 1.074 | 3.122 | 3.645 | 0.25 | PASS |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 0.237 | 0.857 | 2.916 | 3.408 | 0.28 | PASS |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | 0.539 | 77.090 | timeout | 13.569 | 0.04 | PASS |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.521 | 77.911 | timeout | 13.113 | 0.04 | PASS |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 0.348 | 1.355 | 3.797 | 4.709 | 0.26 | PASS |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.807 | 158.831 | timeout | 21.494 | 0.04 | PASS |
| `fx_simple_10000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 1.0ms | 8.0ms | 6.4ms | 7.0ms | 0.16 | PASS |
| `fx_simple_10000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 1.0ms | 5.0ms | 6.9ms | 6.9ms | 0.20 | PASS |
| `fx_simple_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 2.0ms | 5.0ms | 9.3ms | 12.6ms | 0.40 | PASS |
| `fx_simple_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 1.0ms | 6.0ms | 10.0ms | 10.0ms | 0.17 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 2.0ms | 6.0ms | 8.3ms | 8.0ms | 0.33 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 2.0ms | 12.0ms | 11.4ms | 23.1ms | 0.18 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 2.0ms | 6.0ms | 8.8ms | 11.1ms | 0.33 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 2.0ms | 7.0ms | 10.1ms | 13.3ms | 0.29 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 3.0ms | 6.0ms | 9.7ms | 9.5ms | 0.50 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | 3.0ms | 6.0ms | 9.2ms | 10.7ms | 0.50 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 3.0ms | 7.0ms | 9.9ms | 12.5ms | 0.43 | PASS |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | 3.0ms | 9.0ms | 11.6ms | 16.2ms | 0.33 | PASS |
| `fx_simple_100000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 3.0ms | 10.0ms | 24.3ms | 31.7ms | 0.30 | PASS |
| `fx_simple_100000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 4.0ms | 9.0ms | 22.3ms | 26.3ms | 0.44 | PASS |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 5.0ms | 12.0ms | 33.9ms | 64.0ms | 0.42 | PASS |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 8.0ms | 11.0ms | 37.5ms | 64.5ms | 0.73 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 4.0ms | 13.0ms | 32.6ms | 41.7ms | 0.31 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 5.0ms | 17.0ms | 28.8ms | 38.4ms | 0.29 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 6.0ms | 16.0ms | 32.1ms | 70.7ms | 0.38 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 10.0ms | 25.0ms | 38.2ms | 98.2ms | 0.40 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 5.0ms | 15.0ms | 34.4ms | 44.8ms | 0.33 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | 10.0ms | 28.0ms | 35.9ms | 50.6ms | 0.36 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 7.0ms | 17.0ms | 43.0ms | 91.0ms | 0.41 | PASS |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | 15.0ms | 45.0ms | 52.2ms | 0.147 | 0.33 | PASS |
| `fx_simple_1000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 19.0ms | 43.0ms | 0.174 | 0.201 | 0.44 | PASS |
| `fx_simple_1000000 y~x1 fe=indiv_id+year vcov=iid` | 1 | 36.0ms | 53.0ms | 0.169 | 0.236 | 0.68 | PASS |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 26.0ms | 0.126 | 0.216 | 0.750 | 0.21 | PASS |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 66.0ms | 87.0ms | 0.225 | 0.974 | 0.76 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | 31.0ms | 87.0ms | 0.301 | 0.343 | 0.36 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 1 | 58.0ms | 0.137 | 0.307 | 0.440 | 0.42 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 26.0ms | 69.0ms | 0.244 | 0.290 | 0.38 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 44.0ms | 0.122 | 0.252 | 0.399 | 0.36 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | 42.0ms | 0.219 | 0.421 | 1.105 | 0.19 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 1 | 0.108 | 0.243 | 0.438 | 1.754 | 0.44 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 42.0ms | 0.167 | 0.381 | 1.057 | 0.25 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 95.0ms | 0.224 | 0.399 | 1.692 | 0.42 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 41.0ms | 0.114 | 0.306 | 0.368 | 0.36 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | 82.0ms | 0.215 | 0.318 | 0.551 | 0.38 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 68.0ms | 0.442 | 0.547 | 1.431 | 0.15 | PASS |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.270 | 0.416 | 0.597 | 2.560 | 0.65 | PASS |
| `fx_simple_10000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | 0.154 | 0.507 | 1.823 | 2.210 | 0.30 | PASS |
| `fx_simple_10000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.247 | 1.268 | 2.613 | 13.120 | 0.19 | PASS |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | 0.267 | 1.988 | 3.292 | 4.040 | 0.13 | PASS |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 0.240 | 0.841 | 2.594 | 3.148 | 0.29 | PASS |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | 0.473 | 3.537 | 4.521 | 17.356 | 0.13 | PASS |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.445 | 4.008 | 3.628 | 16.320 | 0.12 | PASS |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | 0.344 | 1.423 | 3.636 | 4.366 | 0.24 | PASS |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.856 | 5.448 | 6.519 | 22.431 | 0.16 | PASS |

Coefficient gap:

| config | T | vs | fereg_r | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|
| `fx_difficult_10000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 4.5e-15 |  | 8.1e-16 | 8.1e-16 |
| `fx_difficult_10000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 4.5e-15 |  | 8.1e-16 | 8.1e-16 |
| `fx_difficult_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 2.6e-13 |  | 3.9e-12 | 2.5e-13 |
| `fx_difficult_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 2.6e-13 |  | 4.6e-13 | 2.5e-13 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 6.7e-16 |  | 4.6e-15 | 4.6e-15 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 6.7e-16 |  | 4.6e-15 | 4.2e-15 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.8e-12 |  | 1.2e-11 | 1.8e-12 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 1.8e-12 |  | 1.2e-11 | 1.8e-12 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fixest | 1.7e-15 |  | 9.4e-15 | 8.7e-15 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | fixest | 1.7e-15 |  | 9.4e-15 | 8.7e-15 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 5.9e-12 |  | 6.1e-11 | 5.9e-12 |
| `fx_difficult_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 5.9e-12 |  | 1.1e-11 | 5.9e-12 |
| `fx_difficult_100000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 7.9e-15 |  | 7.0e-15 | 6.8e-15 |
| `fx_difficult_100000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 7.9e-15 |  | 6.6e-15 | 7.3e-15 |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 3.3e-10 |  | 6.8e-08 | 3.3e-10 |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 3.3e-10 |  | 6.8e-08 | 3.3e-10 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.1e-14 |  | 2.1e-14 | 2.2e-14 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 2.1e-14 |  | 2.1e-14 | 2.2e-14 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.7e-09 |  | 7.9e-07 | 1.7e-09 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 1.7e-09 |  | 7.9e-07 | 1.7e-09 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.8e-14 |  | 2.6e-14 | 2.6e-14 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | fixest | 2.8e-14 |  | 2.6e-14 | 2.6e-14 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 3.3e-09 |  | 1.6e-06 | 3.3e-09 |
| `fx_difficult_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 3.3e-09 |  | 1.5e-06 | 3.3e-09 |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 3.0e-14 |  | 3.0e-14 | 3.0e-14 |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 3.0e-14 |  | 2.8e-14 | 2.8e-14 |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.4e-06 |  | 2.6e-06 | 1.4e-06 |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 1.4e-06 |  | 2.3e-06 | 1.4e-06 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | fixest | 7.3e-14 |  | 7.3e-14 | 7.3e-14 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 1 | fixest | 7.3e-14 |  | 7.3e-14 | 7.2e-14 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 7.3e-14 |  | 7.3e-14 | 7.3e-14 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 7.3e-14 |  | 7.3e-14 | 7.2e-14 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | fixest | 1.4e-06 |  | 4.5e-06 | 1.4e-06 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 1 | fixest | 1.4e-06 |  | 5.4e-06 | 1.4e-06 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.4e-06 |  | 5.4e-06 | 1.4e-06 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 1.4e-06 |  | 5.4e-06 | 1.4e-06 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fixest | 7.3e-14 |  | 7.4e-14 | 7.3e-14 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | fixest | 7.3e-14 |  | 7.3e-14 | 7.3e-14 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 4.3e-06 |  | 1.0e-05 | 4.3e-06 |
| `fx_difficult_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 4.3e-06 |  | 1.1e-05 | 4.3e-06 |
| `fx_difficult_10000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 6.5e-14 |  | 6.7e-14 | 6.6e-14 |
| `fx_difficult_10000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.6e-08 |  | 8.8e-07 | 1.6e-08 |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | fixest | 1.7e-13 |  | 1.6e-13 | 1.6e-13 |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 1.7e-13 |  | 1.6e-13 | 1.6e-13 |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | fixest | 1.5e-06 |  |  | 1.5e-06 |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.5e-06 |  |  | 1.5e-06 |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.2e-13 |  | 2.1e-13 | 2.2e-13 |
| `fx_difficult_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 4.6e-06 |  |  | 4.6e-06 |
| `fx_simple_10000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 3.3e-15 |  | 2.6e-15 | 2.4e-15 |
| `fx_simple_10000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 3.3e-15 |  | 2.6e-15 | 2.4e-15 |
| `fx_simple_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 6.8e-15 |  | 5.2e-15 | 4.4e-15 |
| `fx_simple_10000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 6.8e-15 |  | 5.0e-15 | 4.4e-15 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 1.1e-15 |  | 6.8e-15 | 6.7e-15 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 1.1e-15 |  | 6.8e-15 | 6.7e-15 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 7.5e-15 |  | 5.0e-15 | 5.8e-15 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 7.5e-15 |  | 6.1e-15 | 5.8e-15 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fixest | 1.8e-15 |  | 7.9e-15 | 7.9e-15 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | fixest | 1.8e-15 |  | 8.3e-15 | 7.9e-15 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.0e-14 |  | 6.0e-15 | 5.9e-15 |
| `fx_simple_10000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 1.0e-14 |  | 4.9e-15 | 6.3e-15 |
| `fx_simple_100000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 7.7e-15 |  | 7.7e-15 | 7.7e-15 |
| `fx_simple_100000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 7.7e-15 |  | 7.7e-15 | 8.1e-15 |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 9.7e-15 |  | 9.0e-15 | 9.3e-15 |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 9.7e-15 |  | 8.6e-15 | 9.9e-15 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.6e-14 |  | 2.5e-14 | 2.5e-14 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 2.6e-14 |  | 2.5e-14 | 2.4e-14 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 2.2e-14 |  | 1.9e-14 | 1.9e-14 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 2.2e-14 |  | 1.9e-14 | 1.9e-14 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.6e-14 |  | 2.4e-14 | 2.5e-14 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | fixest | 2.6e-14 |  | 2.4e-14 | 2.4e-14 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 2.2e-14 |  | 2.0e-14 | 2.0e-14 |
| `fx_simple_100000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 2.2e-14 |  | 2.0e-14 | 2.0e-14 |
| `fx_simple_1000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.1e-15 |  | 2.4e-15 | 2.2e-15 |
| `fx_simple_1000000 y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 2.1e-15 |  | 3.0e-15 | 1.9e-15 |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 6.9e-15 |  | 7.4e-15 | 7.4e-15 |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 6.9e-15 |  | 7.0e-15 | 8.9e-15 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | fixest | 8.7e-14 |  | 8.8e-14 | 8.9e-14 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 1 | fixest | 8.7e-14 |  | 8.8e-14 | 8.9e-14 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 8.7e-14 |  | 8.8e-14 | 8.9e-14 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 8.7e-14 |  | 8.8e-14 | 8.9e-14 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | fixest | 5.6e-14 |  | 5.6e-14 | 5.5e-14 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 1 | fixest | 5.6e-14 |  | 5.8e-14 | 5.5e-14 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 5.6e-14 |  | 5.7e-14 | 5.5e-14 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 5.6e-14 |  | 5.6e-14 | 5.5e-14 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fixest | 8.7e-14 |  | 8.8e-14 | 8.9e-14 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 1 | fixest | 8.7e-14 |  | 8.8e-14 | 8.9e-14 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 5.6e-14 |  | 5.8e-14 | 5.5e-14 |
| `fx_simple_1000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 5.6e-14 |  | 5.5e-14 | 5.5e-14 |
| `fx_simple_10000000 y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 4.7e-14 |  | 4.8e-14 | 4.8e-14 |
| `fx_simple_10000000 y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.9e-14 |  | 2.1e-14 | 2.1e-14 |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=cluster:indiv_id` | 16 | fixest | 2.7e-13 |  | 2.7e-13 | 2.7e-13 |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.7e-13 |  | 2.7e-13 | 2.7e-13 |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=cluster:indiv_id` | 16 | fixest | 1.7e-13 |  | 1.7e-13 | 1.7e-13 |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.7e-13 |  | 1.7e-13 | 1.7e-13 |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.7e-13 |  | 2.7e-13 | 2.7e-13 |
| `fx_simple_10000000 y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.7e-13 |  | 1.7e-13 | 1.6e-13 |

## G2 AKM

| config | T | fereg_r | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `akm_baseline_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.143 | 0.425 | 6.777 | 0.471 | 0.34 | PASS |
| `akm_baseline_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.220 | 0.829 | 10.356 | 0.949 | 0.27 | PASS |
| `akm_freeze_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.107 | 0.552 | 17.859 | 0.575 | 0.19 | PASS |
| `akm_freeze_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.192 | 0.964 | 15.060 | 0.950 | 0.20 | PASS |
| `akm_freeze_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.107 | 1.232 | 19.287 | 0.775 | 0.14 | PASS |
| `akm_freeze_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.188 | 1.670 | 22.908 | 0.840 | 0.22 | PASS |
| `akm_freeze_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 76.0ms | 0.476 | 8.295 | 0.491 | 0.16 | PASS |
| `akm_freeze_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.151 | 0.930 | 13.685 | 0.963 | 0.16 | PASS |
| `akm_freeze_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 62.0ms | 0.753 | 15.710 | 0.518 | 0.12 | PASS |
| `akm_freeze_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.133 | 1.228 | 19.560 | 0.908 | 0.15 | PASS |
| `akm_freeze_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 45.0ms | 1.068 | 33.232 | 0.545 | 0.08 | PASS |
| `akm_freeze_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.116 | 1.912 | 24.496 | 0.755 | 0.15 | PASS |
| `akm_freeze_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 38.0ms | 1.842 | error | 0.573 | 0.07 | PASS |
| `akm_freeze_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.104 | 3.035 | error | 0.683 | 0.15 | PASS |
| `akm_interaction_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 60.0ms | 0.278 | 1.356 | 0.728 | 0.22 | PASS |
| `akm_interaction_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.186 | 0.413 | 1.769 | 0.987 | 0.45 | PASS |
| `akm_interaction_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 63.0ms | 0.220 | 1.282 | 0.807 | 0.29 | PASS |
| `akm_interaction_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.199 | 0.347 | 1.347 | 1.107 | 0.57 | PASS |
| `akm_interaction_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 44.0ms | 0.840 | 16.093 | 0.405 | 0.11 | PASS |
| `akm_interaction_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.109 | 1.430 | 23.808 | 0.758 | 0.14 | PASS |
| `akm_interaction_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 50.0ms | 2.608 | error | 0.568 | 0.09 | PASS |
| `akm_interaction_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.125 | 4.008 | error | 0.816 | 0.15 | PASS |
| `akm_mobility_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 45.0ms | 0.106 | 0.267 | 0.811 | 0.42 | PASS |
| `akm_mobility_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.110 | 0.146 | 0.299 | 1.210 | 0.75 | PASS |
| `akm_mobility_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 65.0ms | 0.207 | 0.824 | 0.551 | 0.31 | PASS |
| `akm_mobility_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.192 | 0.330 | 1.047 | 0.896 | 0.58 | PASS |
| `akm_mobility_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 62.0ms | 0.686 | 10.315 | 0.427 | 0.15 | PASS |
| `akm_mobility_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.134 | 1.172 | 15.429 | 0.782 | 0.17 | PASS |
| `akm_mobility_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 38.0ms | 1.145 | 21.857 | 0.359 | 0.11 | PASS |
| `akm_mobility_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.100 | 2.340 | 29.990 | 0.598 | 0.17 | PASS |
| `akm_mobility_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 37.0ms | 1.437 | error | 0.374 | 0.10 | PASS |
| `akm_mobility_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.107 | 2.591 | error | 0.697 | 0.15 | PASS |
| `akm_mobility_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 35.0ms | 1.467 | error | 0.360 | 0.10 | PASS |
| `akm_mobility_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.100 | 2.618 | error | 0.653 | 0.15 | PASS |
| `akm_scale_1_10000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 9.0ms | 16.0ms | 83.0ms | 25.1ms | 0.56 | PASS |
| `akm_scale_1_10000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 10.0ms | 21.0ms | 0.118 | 12.4ms | 0.81 | PASS |
| `akm_scale_2_100000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 12.0ms | 64.0ms | 0.401 | 51.2ms | 0.23 | PASS |
| `akm_scale_2_100000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 18.0ms | 0.109 | 0.770 | 69.3ms | 0.26 | PASS |
| `akm_scale_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.111 | 0.344 | 3.794 | 0.453 | 0.32 | PASS |
| `akm_scale_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.200 | 0.589 | 4.990 | 0.828 | 0.34 | PASS |
| `akm_scale_4_10000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.852 | 4.616 | 47.856 | 4.379 | 0.19 | PASS |
| `akm_sorting_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 71.0ms | 0.313 | 3.251 | 0.409 | 0.23 | PASS |
| `akm_sorting_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.144 | 0.523 | 3.808 | 0.750 | 0.28 | PASS |
| `akm_sorting_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.153 | 0.476 | 6.387 | 0.450 | 0.34 | PASS |
| `akm_sorting_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.416 | 0.829 | 9.480 | 0.819 | 0.51 | PASS |
| `akm_sorting_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.230 | 0.508 | 5.353 | 0.661 | 0.45 | PASS |
| `akm_sorting_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.477 | 0.855 | 7.077 | 0.830 | 0.57 | PASS |
| `akm_sorting_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.242 | 0.463 | 8.786 | 0.678 | 0.52 | PASS |
| `akm_sorting_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.507 | 0.829 | 10.257 | 0.919 | 0.61 | PASS |
| `akm_sorting_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | 0.204 | 0.712 | 14.752 | 0.482 | 0.42 | PASS |
| `akm_sorting_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | 0.408 | 1.270 | 17.849 | 0.806 | 0.51 | PASS |

Coefficient gap:

| config | T | vs | fereg_r | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|
| `akm_baseline_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 4.6e-14 |  | 1.3e-13 | 4.5e-14 |
| `akm_baseline_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 4.6e-14 |  | 1.2e-13 | 4.4e-14 |
| `akm_freeze_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 2.6e-14 |  | 6.7e-14 | 2.4e-14 |
| `akm_freeze_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 2.6e-14 |  | 4.7e-14 | 2.5e-14 |
| `akm_freeze_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 2.2e-15 |  | 7.2e-13 | 2.4e-15 |
| `akm_freeze_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 2.2e-15 |  | 7.1e-13 | 1.6e-15 |
| `akm_freeze_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 6.9e-14 |  | 1.1e-16 | 7.3e-14 |
| `akm_freeze_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 6.9e-14 |  | 1.7e-13 | 7.2e-14 |
| `akm_freeze_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 7.4e-14 |  | 2.7e-13 | 7.1e-14 |
| `akm_freeze_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 7.4e-14 |  | 2.7e-13 | 7.1e-14 |
| `akm_freeze_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 1.4e-14 |  | 9.4e-13 | 1.8e-14 |
| `akm_freeze_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 1.4e-14 |  | 9.6e-13 | 1.9e-14 |
| `akm_freeze_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 3.2e-10 |  |  | 3.2e-10 |
| `akm_freeze_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 3.2e-10 |  |  | 3.2e-10 |
| `akm_interaction_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 2.2e-16 |  | 1.1e-15 | 8.9e-16 |
| `akm_interaction_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 2.2e-16 |  | 2.0e-15 | 2.0e-15 |
| `akm_interaction_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 1.1e-13 |  | 4.4e-12 | 1.1e-13 |
| `akm_interaction_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 1.1e-13 |  | 4.4e-12 | 1.1e-13 |
| `akm_interaction_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 3.5e-13 |  | 8.7e-13 | 3.5e-13 |
| `akm_interaction_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 3.5e-13 |  | 8.7e-13 | 3.5e-13 |
| `akm_interaction_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 2.4e-08 |  |  | 2.4e-08 |
| `akm_interaction_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 2.4e-08 |  |  | 2.4e-08 |
| `akm_mobility_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 6.9e-14 |  | 6.3e-14 | 6.7e-14 |
| `akm_mobility_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 6.9e-14 |  | 5.6e-14 | 6.6e-14 |
| `akm_mobility_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 2.0e-14 |  | 2.3e-14 | 2.3e-14 |
| `akm_mobility_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 2.0e-14 |  | 2.1e-14 | 2.2e-14 |
| `akm_mobility_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 7.0e-14 |  | 2.6e-13 | 7.0e-14 |
| `akm_mobility_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 7.0e-14 |  | 2.8e-13 | 6.9e-14 |
| `akm_mobility_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 9.1e-13 |  | 1.7e-12 | 9.1e-13 |
| `akm_mobility_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 9.1e-13 |  | 1.8e-12 | 9.0e-13 |
| `akm_mobility_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 4.7e-10 |  |  | 4.7e-10 |
| `akm_mobility_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 4.7e-10 |  |  | 4.7e-10 |
| `akm_mobility_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 5.4e-07 |  |  | 5.4e-07 |
| `akm_mobility_6_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 5.4e-07 |  |  | 5.4e-07 |
| `akm_scale_1_10000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 5.6e-12 |  | 1.2e-10 | 5.6e-12 |
| `akm_scale_1_10000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 5.6e-12 |  | 1.2e-10 | 5.6e-12 |
| `akm_scale_2_100000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 3.0e-13 |  | 1.1e-11 | 3.0e-13 |
| `akm_scale_2_100000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 3.0e-13 |  | 1.1e-11 | 3.0e-13 |
| `akm_scale_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 2.8e-14 |  | 2.0e-14 | 2.8e-14 |
| `akm_scale_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 2.8e-14 |  | 1.8e-14 | 2.9e-14 |
| `akm_scale_4_10000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 2.1e-13 |  | 2.1e-13 | 2.1e-13 |
| `akm_sorting_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 1.1e-15 |  | 1.1e-14 | 0.0e+00 |
| `akm_sorting_1_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 1.1e-15 |  | 1.1e-14 | 8.8e-16 |
| `akm_sorting_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 2.0e-13 |  | 1.3e-14 | 2.1e-13 |
| `akm_sorting_2_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 2.0e-13 |  | 1.3e-14 | 2.1e-13 |
| `akm_sorting_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 3.0e-14 |  | 8.2e-14 | 2.9e-14 |
| `akm_sorting_3_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 3.0e-14 |  | 8.3e-14 | 3.0e-14 |
| `akm_sorting_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 1.0e-11 |  | 7.2e-11 | 1.0e-11 |
| `akm_sorting_4_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 1.0e-11 |  | 7.2e-11 | 1.0e-11 |
| `akm_sorting_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 16 | fixest | 7.1e-12 |  | 8.4e-11 | 7.1e-12 |
| `akm_sorting_5_1000000 y~x1 fe=indiv_id+firm_id+year vcov=iid` | 1 | fixest | 7.1e-12 |  | 6.1e-11 | 7.1e-12 |

## G3 AKM with occupation FE

| config | T | fereg_r | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `akmocc_baseline_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.116 | 0.696 | 4.653 | 0.695 | 0.17 | PASS |
| `akmocc_baseline_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.232 | 1.243 | 6.924 | 1.276 | 0.19 | PASS |
| `akmocc_occlambda_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.133 | 0.816 | 6.457 | 0.908 | 0.16 | PASS |
| `akmocc_occlambda_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.256 | 1.246 | 6.810 | 1.374 | 0.21 | PASS |
| `akmocc_occlambda_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.124 | 0.883 | 6.945 | 0.715 | 0.17 | PASS |
| `akmocc_occlambda_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.238 | 1.590 | 9.488 | 1.323 | 0.18 | PASS |
| `akmocc_occlambda_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.154 | 0.978 | 5.719 | 0.773 | 0.20 | PASS |
| `akmocc_occlambda_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.270 | 1.736 | 8.012 | 1.407 | 0.19 | PASS |
| `akmocc_occlambda_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.112 | 2.303 | 24.836 | 0.830 | 0.13 | PASS |
| `akmocc_occlambda_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.234 | 4.265 | 35.423 | 1.601 | 0.15 | PASS |
| `akmocc_occsize_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.102 | 0.633 | 5.668 | 0.580 | 0.18 | PASS |
| `akmocc_occsize_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.205 | 1.278 | 9.818 | 1.116 | 0.18 | PASS |
| `akmocc_occsize_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.125 | 0.832 | 6.142 | 0.641 | 0.19 | PASS |
| `akmocc_occsize_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.244 | 1.118 | 7.149 | 1.111 | 0.22 | PASS |
| `akmocc_occsize_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.203 | 1.269 | 8.152 | 0.825 | 0.25 | PASS |
| `akmocc_occsize_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 0.314 | 2.120 | 10.323 | 1.517 | 0.21 | PASS |
| `akmocc_occsize_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | 0.771 | 2.310 | 32.982 | 1.398 | 0.55 | PASS |
| `akmocc_occsize_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | 1.507 | 3.668 | 46.309 | 2.806 | 0.54 | PASS |

Coefficient gap:

| config | T | vs | fereg_r | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|
| `akmocc_baseline_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fixest | 2.2e-14 |  | 8.7e-15 | 4.2e-15 |
| `akmocc_baseline_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fixest | 2.2e-14 |  | 4.6e-14 | 5.3e-15 |
| `akmocc_occlambda_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fixest | 4.6e-14 |  | 4.9e-14 | 1.8e-14 |
| `akmocc_occlambda_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fixest | 4.6e-14 |  | 4.9e-14 | 1.9e-14 |
| `akmocc_occlambda_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fixest | 3.0e-14 |  | 3.0e-13 | 5.2e-14 |
| `akmocc_occlambda_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fixest | 3.0e-14 |  | 3.0e-13 | 5.1e-14 |
| `akmocc_occlambda_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fixest | 3.0e-13 |  | 1.6e-13 | 2.8e-13 |
| `akmocc_occlambda_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fixest | 3.0e-13 |  | 1.8e-13 | 2.8e-13 |
| `akmocc_occlambda_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fixest | 1.7e-09 |  | 1.7e-09 | 1.7e-09 |
| `akmocc_occlambda_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fixest | 1.7e-09 |  | 1.7e-09 | 1.7e-09 |
| `akmocc_occsize_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fixest | 6.2e-15 |  | 1.2e-13 | 3.3e-15 |
| `akmocc_occsize_1_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fixest | 6.2e-15 |  | 1.2e-13 | 3.1e-15 |
| `akmocc_occsize_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fixest | 1.7e-14 |  | 5.3e-14 | 9.1e-15 |
| `akmocc_occsize_2_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fixest | 1.7e-14 |  | 5.5e-14 | 8.9e-15 |
| `akmocc_occsize_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fixest | 1.7e-12 |  | 1.0e-11 | 1.7e-12 |
| `akmocc_occsize_3_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fixest | 1.7e-12 |  | 1.0e-11 | 1.7e-12 |
| `akmocc_occsize_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 16 | fixest | 2.2e-08 |  | 2.2e-08 | 2.2e-08 |
| `akmocc_occsize_4_1000000 y~x1 fe=indiv_id+firm_id+year+occ_id vcov=iid` | 1 | fixest | 2.2e-08 |  | 2.3e-08 | 2.2e-08 |

## G4 IV/2SLS

| config | T | fereg_r | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 6.0ms | 20.0ms | 54.4ms | 59.0ms | 0.30 | PASS |
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | 10.0ms | 26.0ms | 57.4ms | 65.8ms | 0.38 | PASS |
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | 9.0ms | 0.116 | 1.473 | 97.1ms | 0.09 | PASS |
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | 17.0ms | 0.261 | 3.071 | 0.171 | 0.10 | PASS |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 29.0ms | 0.113 | 0.423 | 0.483 | 0.26 | PASS |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | 71.0ms | 0.169 | 0.432 | 0.641 | 0.42 | PASS |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | 51.0ms | 3.022 | 71.852 | 1.180 | 0.04 | PASS |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | 0.162 | 6.066 | 90.695 | 2.203 | 0.07 | PASS |
| `fx_difficult_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 0.253 | 1.321 | 4.717 | 5.406 | 0.19 | PASS |
| `fx_difficult_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | 0.523 | 68.012 | timeout | 20.033 | 0.03 | PASS |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 6.0ms | 26.0ms | 54.6ms | 0.106 | 0.23 | PASS |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | 10.0ms | 26.0ms | 56.2ms | 67.2ms | 0.38 | PASS |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | 7.0ms | 26.0ms | 60.6ms | 96.0ms | 0.27 | PASS |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | 15.0ms | 33.0ms | 71.9ms | 0.147 | 0.45 | PASS |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 29.0ms | 0.113 | 0.423 | 0.483 | 0.26 | PASS |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | 71.0ms | 0.168 | 0.432 | 0.646 | 0.42 | PASS |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | 42.0ms | 0.166 | 0.599 | 1.397 | 0.25 | PASS |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | 0.123 | 0.251 | 0.651 | 2.330 | 0.49 | PASS |
| `fx_simple_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | 0.247 | 1.314 | 4.687 | 5.439 | 0.19 | PASS |
| `fx_simple_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | 0.425 | 2.672 | 6.996 | 22.721 | 0.16 | PASS |

Coefficient gap:

| config | T | vs | fereg_r | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fixest | 6.6e-15 |  | 7.7e-15 | 6.8e-15 |
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | fixest | 6.6e-15 |  | 7.7e-15 | 6.8e-15 |
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 2.2e-10 |  | 1.2e-07 | 2.2e-10 |
| `fx_difficult_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 2.2e-10 |  | 1.2e-07 | 2.2e-10 |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fixest | 7.0e-14 |  | 6.9e-14 | 6.9e-14 |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | fixest | 7.0e-14 |  | 6.9e-14 | 6.9e-14 |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 9.0e-06 |  | 5.7e-06 | 9.0e-06 |
| `fx_difficult_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 9.0e-06 |  | 5.9e-06 | 9.0e-06 |
| `fx_difficult_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fixest | 3.5e-13 |  | 3.5e-13 | 3.5e-13 |
| `fx_difficult_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 2.4e-06 |  |  | 2.4e-06 |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fixest | 2.3e-14 |  | 2.2e-14 | 2.2e-14 |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | fixest | 2.3e-14 |  | 2.2e-14 | 2.2e-14 |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 5.0e-15 |  | 7.1e-15 | 6.2e-15 |
| `fx_simple_100000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 5.0e-15 |  | 6.6e-15 | 6.4e-15 |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fixest | 4.0e-15 |  | 3.1e-15 | 2.1e-15 |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 1 | fixest | 4.0e-15 |  | 3.1e-15 | 2.3e-15 |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 5.7e-14 |  | 5.7e-14 | 5.7e-14 |
| `fx_simple_1000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 5.7e-14 |  | 5.7e-14 | 5.8e-14 |
| `fx_simple_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year vcov=hetero` | 16 | fixest | 1.7e-13 |  | 1.8e-13 | 1.7e-13 |
| `fx_simple_10000000 y_iv~x1+(xe=z1+z2) fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 2.4e-13 |  | 2.4e-13 | 2.4e-13 |

## G5 PPML

| config | T | fereg_r | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 18.0ms | 52.0ms | 0.138 | 0.322 | 0.35 | PASS |
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | 43.0ms | 64.0ms | 0.140 | 0.501 | 0.67 | PASS |
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 36.0ms | 0.895 | error | 0.440 | 0.08 | PASS |
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 83.0ms | 1.572 | 26.324 | 0.789 | 0.11 | PASS |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | 25.0ms | 71.0ms | 0.162 | 0.478 | 0.35 | PASS |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | 53.0ms | 0.146 | 0.218 | 1.261 | 0.36 | PASS |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 41.0ms | 0.998 | error | 0.673 | 0.06 | PASS |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 0.106 | 4.247 | error | 2.031 | 0.05 | PASS |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 0.136 | 0.361 | 1.376 | 3.691 | 0.38 | PASS |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | 0.358 | 0.528 | 1.462 | 8.212 | 0.68 | PASS |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 0.258 | 40.142 | error | 6.825 | 0.04 | PASS |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 0.934 | 68.736 | error | 17.954 | 0.05 | PASS |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | 0.240 | 0.627 | 2.284 | 9.349 | 0.38 | PASS |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | 0.516 | 1.367 | 2.437 | 21.547 | 0.38 | PASS |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 0.390 | 137.512 | error | 18.248 | 0.02 | PASS |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 1.244 | 216.931 | error | 39.497 | 0.03 | PASS |
| `fx_difficult_10000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 1.538 | 4.078 | 15.348 | 49.447 | 0.38 | PASS |
| `fx_difficult_10000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 2.998 | timeout | error | 109.599 | 0.03 | PASS |
| `fx_difficult_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | 2.489 | 6.981 | 24.084 | 126.197 | 0.36 | PASS |
| `fx_difficult_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 4.857 | timeout | timeout | 247.045 | 0.02 | PASS |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 21.0ms | 74.0ms | 0.170 | 0.374 | 0.28 | PASS |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | 41.0ms | 79.0ms | 0.134 | 0.487 | 0.52 | PASS |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 28.0ms | 88.0ms | 0.227 | 0.477 | 0.32 | PASS |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 68.0ms | 0.132 | 0.259 | 0.907 | 0.52 | PASS |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | 27.0ms | 68.0ms | 0.162 | 0.454 | 0.40 | PASS |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | 58.0ms | 0.142 | 0.210 | 1.175 | 0.41 | PASS |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 39.0ms | 0.115 | 0.252 | 0.646 | 0.34 | PASS |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 90.0ms | 0.349 | 0.584 | 2.242 | 0.26 | PASS |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 0.137 | 0.355 | 1.336 | 3.577 | 0.39 | PASS |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | 0.352 | 0.504 | 1.374 | 7.871 | 0.70 | PASS |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 0.199 | 1.110 | 3.677 | 8.550 | 0.18 | PASS |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 0.632 | 1.715 | 4.410 | 18.365 | 0.37 | PASS |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | 0.233 | 0.635 | 2.179 | 8.792 | 0.37 | PASS |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | 0.516 | 1.316 | 2.307 | 20.359 | 0.39 | PASS |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 0.347 | 4.216 | 11.347 | 18.015 | 0.08 | PASS |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | 1.038 | 5.239 | 13.349 | 47.009 | 0.20 | PASS |
| `fx_simple_10000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | 1.541 | 4.224 | 16.104 | 53.108 | 0.36 | PASS |
| `fx_simple_10000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 2.600 | 43.144 | 394.828 | 202.763 | 0.06 | PASS |
| `fx_simple_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | 2.519 | 7.291 | 26.552 | 140.230 | 0.35 | PASS |
| `fx_simple_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | 5.509 | 94.012 | timeout | 402.220 | 0.06 | PASS |

Coefficient gap:

| config | T | vs | fereg_r | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fixest | 8.9e-16 |  | 1.0e-15 | 6.2e-13 |
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | fixest | 8.9e-16 |  | 5.1e-15 | 4.2e-13 |
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 5.4e-12 |  |  | 6.0e-12 |
| `fx_difficult_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 5.4e-12 |  | 7.9e-08 | 6.0e-12 |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | fixest | 1.5e-14 |  | 2.8e-13 | 1.5e-12 |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | fixest | 1.5e-14 |  | 2.8e-13 | 1.5e-12 |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 1.2e-09 |  |  | 1.2e-09 |
| `fx_difficult_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 1.2e-09 |  |  | 1.2e-09 |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fixest | 4.5e-16 |  | 6.7e-16 | 3.7e-13 |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | fixest | 4.5e-16 |  | 7.8e-16 | 3.7e-13 |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 1.5e-06 |  |  | 1.5e-06 |
| `fx_difficult_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 1.5e-06 |  |  | 1.5e-06 |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | fixest | 4.3e-14 |  | 1.9e-13 | 2.5e-12 |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | fixest | 4.3e-14 |  | 1.4e-13 | 2.5e-12 |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 5.6e-05 |  |  | 5.6e-05 |
| `fx_difficult_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 5.6e-05 |  |  | 5.6e-05 |
| `fx_difficult_10000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fixest | 1.2e-14 |  | 1.1e-14 | 3.8e-13 |
| `fx_difficult_10000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 6.6e-06 |  |  | 6.6e-06 |
| `fx_difficult_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | fixest | 2.6e-13 |  | 2.1e-13 | 2.1e-12 |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fixest | 4.4e-15 |  | 1.0e-14 | 1.0e-12 |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | fixest | 4.4e-15 |  | 7.1e-15 | 1.0e-12 |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 9.7e-15 |  | 7.9e-14 | 1.3e-12 |
| `fx_simple_100000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 9.7e-15 |  | 1.2e-13 | 1.3e-12 |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | fixest | 1.8e-14 |  | 1.0e-13 | 3.5e-12 |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | fixest | 1.8e-14 |  | 1.0e-13 | 3.8e-12 |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 1.4e-13 |  | 4.7e-13 | 2.1e-12 |
| `fx_simple_100000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 1.4e-13 |  | 4.7e-13 | 2.0e-12 |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fixest | 4.4e-15 |  | 3.1e-15 | 4.2e-13 |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 1 | fixest | 4.4e-15 |  | 3.5e-15 | 4.2e-13 |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 2.2e-14 |  | 1.7e-13 | 8.1e-13 |
| `fx_simple_1000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 2.2e-14 |  | 1.5e-13 | 8.3e-13 |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | fixest | 8.0e-14 |  | 9.1e-14 | 2.8e-12 |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 1 | fixest | 8.0e-14 |  | 8.2e-14 | 2.8e-12 |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 2.1e-13 |  | 9.9e-14 | 1.7e-12 |
| `fx_simple_1000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 1 | fixest | 2.1e-13 |  | 6.7e-14 | 1.7e-12 |
| `fx_simple_10000000 negbin_y~x1 fe=indiv_id+year vcov=hetero` | 16 | fixest | 5.7e-14 |  | 5.5e-14 | 5.2e-13 |
| `fx_simple_10000000 negbin_y~x1 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 1.1e-13 |  | 1.1e-11 | 7.8e-12 |
| `fx_simple_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=hetero` | 16 | fixest | 1.6e-13 |  | 3.9e-13 | 8.8e-13 |
| `fx_simple_10000000 negbin_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=hetero` | 16 | fixest | 2.0e-12 |  |  | 7.6e-12 |

## G6 Logit

| config | T | fereg_r | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 16.0ms | 38.0ms | 0.135 | 0.283 | 0.42 | PASS |
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | 39.0ms | 48.0ms | 0.144 | 0.386 | 0.81 | PASS |
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 30.0ms | 0.460 | 4.478 | 0.563 | 0.07 | PASS |
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 72.0ms | 0.774 | 7.566 | 0.899 | 0.09 | PASS |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 23.0ms | 58.0ms | 0.220 | 0.438 | 0.40 | PASS |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 55.0ms | 0.102 | 0.218 | 0.888 | 0.54 | PASS |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 41.0ms | 0.596 | 6.191 | 0.858 | 0.07 | PASS |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 95.0ms | 2.087 | 23.444 | 2.029 | 0.05 | PASS |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 0.105 | 0.271 | 1.486 | 3.090 | 0.39 | PASS |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | 0.349 | 0.403 | 1.531 | 6.238 | 0.87 | PASS |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.204 | 17.635 | error | 6.903 | 0.03 | PASS |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.809 | 30.193 | error | 14.916 | 0.05 | PASS |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 0.195 | 0.512 | 2.447 | 6.777 | 0.38 | PASS |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 0.489 | 0.980 | 2.657 | 14.933 | 0.50 | PASS |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.347 | 72.274 | error | 22.672 | 0.02 | PASS |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 1.100 | 103.166 | error | 50.058 | 0.02 | PASS |
| `fx_difficult_10000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 1.140 | 2.959 | 17.712 | 42.518 | 0.39 | PASS |
| `fx_difficult_10000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 2.249 | timeout | timeout | 114.788 | 0.02 | PASS |
| `fx_difficult_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 2.040 | 5.356 | 28.051 | 94.733 | 0.38 | PASS |
| `fx_difficult_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 3.824 | timeout | timeout | 418.058 | 0.01 | PASS |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 16.0ms | 43.0ms | 0.161 | 0.319 | 0.37 | PASS |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | 39.0ms | 50.0ms | 0.143 | 0.362 | 0.78 | PASS |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 24.0ms | 59.0ms | 0.179 | 0.603 | 0.41 | PASS |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 66.0ms | 85.0ms | 0.188 | 0.894 | 0.78 | PASS |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 21.0ms | 57.0ms | 0.219 | 0.426 | 0.37 | PASS |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 54.0ms | 0.102 | 0.214 | 0.838 | 0.53 | PASS |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 38.0ms | 87.0ms | 0.255 | 0.971 | 0.44 | PASS |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 98.0ms | 0.214 | 0.346 | 2.639 | 0.46 | PASS |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 0.103 | 0.268 | 1.472 | 3.073 | 0.38 | PASS |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | 0.350 | 0.396 | 1.505 | 6.203 | 0.88 | PASS |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.170 | 0.524 | 1.996 | 9.874 | 0.32 | PASS |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.623 | 0.814 | 2.145 | 19.250 | 0.77 | PASS |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 0.195 | 0.522 | 2.479 | 6.707 | 0.37 | PASS |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | 0.487 | 0.969 | 2.596 | 14.642 | 0.50 | PASS |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 0.324 | 1.700 | 3.847 | 27.925 | 0.19 | PASS |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | 0.979 | 2.230 | 4.341 | 67.534 | 0.44 | PASS |
| `fx_simple_10000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | 1.268 | 3.423 | 18.477 | 45.549 | 0.37 | PASS |
| `fx_simple_10000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | 2.044 | 8.835 | 26.674 | 224.324 | 0.23 | PASS |
| `fx_simple_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | 2.062 | 5.432 | 29.434 | 117.308 | 0.38 | PASS |
| `fx_simple_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | 4.700 | 24.112 | 46.245 | timeout | 0.19 | PASS |

Coefficient gap:

| config | T | vs | fereg_r | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 1.7e-15 |  | 1.0e-08 | 1.0e-08 |
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 1.7e-15 |  | 1.0e-08 | 1.0e-08 |
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 7.6e-10 |  | 8.9e-10 | 8.9e-10 |
| `fx_difficult_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 7.6e-10 |  | 8.9e-10 | 8.9e-10 |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 3.9e-14 |  | 1.6e-11 | 1.6e-11 |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 3.9e-14 |  | 1.6e-11 | 1.6e-11 |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 3.0e-09 |  | 3.0e-09 | 3.0e-09 |
| `fx_difficult_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 3.0e-09 |  | 3.0e-09 | 3.0e-09 |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 1.8e-14 |  | 2.8e-09 | 2.8e-09 |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 1.8e-14 |  | 2.8e-09 | 2.8e-09 |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 5.7e-08 |  |  | 5.9e-08 |
| `fx_difficult_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 5.7e-08 |  |  | 5.9e-08 |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 4.2e-14 |  | 7.6e-12 | 7.6e-12 |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 4.2e-14 |  | 7.7e-12 | 7.6e-12 |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 3.9e-07 |  |  | 4.1e-07 |
| `fx_difficult_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 3.9e-07 |  |  | 4.1e-07 |
| `fx_difficult_10000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 4.1e-14 |  | 2.1e-09 | 2.1e-09 |
| `fx_difficult_10000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 2.2e-06 |  |  | 2.2e-06 |
| `fx_difficult_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 1.2e-13 |  | 1.9e-11 | 1.9e-11 |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 3.6e-16 |  | 1.1e-09 | 1.1e-09 |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 3.6e-16 |  | 1.1e-09 | 1.1e-09 |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 2.6e-15 |  | 3.0e-11 | 3.0e-11 |
| `fx_simple_100000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 2.6e-15 |  | 3.0e-11 | 3.0e-11 |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.1e-14 |  | 1.5e-11 | 1.5e-11 |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 2.1e-14 |  | 1.5e-11 | 1.5e-11 |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 3.1e-14 |  | 1.3e-09 | 1.3e-09 |
| `fx_simple_100000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 3.1e-14 |  | 1.3e-09 | 1.3e-09 |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 2.0e-14 |  | 1.4e-09 | 1.4e-09 |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 1 | fixest | 2.0e-14 |  | 1.4e-09 | 1.4e-09 |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 1.4e-14 |  | 3.5e-10 | 3.5e-10 |
| `fx_simple_1000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 1.4e-14 |  | 3.5e-10 | 3.5e-10 |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 9.2e-14 |  | 2.8e-12 | 2.9e-12 |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 1 | fixest | 9.2e-14 |  | 2.8e-12 | 2.9e-12 |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 4.6e-14 |  | 3.0e-08 | 3.0e-08 |
| `fx_simple_1000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 1 | fixest | 4.6e-14 |  | 3.0e-08 | 3.0e-08 |
| `fx_simple_10000000 binary_y~x1 fe=indiv_id+year vcov=iid` | 16 | fixest | 3.6e-14 |  | 5.3e-13 | 5.8e-13 |
| `fx_simple_10000000 binary_y~x1 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 4.4e-14 |  | 1.8e-08 | 1.8e-08 |
| `fx_simple_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year vcov=iid` | 16 | fixest | 1.9e-13 |  | 2.4e-08 | 2.4e-08 |
| `fx_simple_10000000 binary_y~x1+x2+x3+x4+x5 fe=indiv_id+year+firm_id vcov=iid` | 16 | fixest | 2.1e-13 |  | 9.4e-09 | 9.2e-09 |

## G7 Heterogeneous slopes

| config | T | fereg_r | fixest | pyfixest_map | pyfixest_lsmr | ratio | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | 16.0ms | 0.148 | unsupported | unsupported | 0.11 | PASS |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | 25.0ms | 0.269 | unsupported | unsupported | 0.09 | PASS |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | 76.0ms | 4.148 | unsupported | unsupported | 0.02 | PASS |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | 0.234 | 6.503 | unsupported | unsupported | 0.04 | PASS |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | 6.0ms | 18.0ms | unsupported | unsupported | 0.33 | PASS |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | 11.0ms | 21.0ms | unsupported | unsupported | 0.52 | PASS |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | 37.0ms | 0.120 | unsupported | unsupported | 0.31 | PASS |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | 0.105 | 0.166 | unsupported | unsupported | 0.63 | PASS |

Coefficient gap:

| config | T | vs | fereg_r | fixest | pyfixest_map | pyfixest_lsmr |
|---|---:|---|---:|---:|---:|---:|
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | fixest | 4.3e-11 |  |  |  |
| `fx_difficult_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | fixest | 4.3e-11 |  |  |  |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | fixest | 4.9e-07 |  |  |  |
| `fx_difficult_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | fixest | 4.9e-07 |  |  |  |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | fixest | 8.8e-15 |  |  |  |
| `fx_simple_100000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | fixest | 8.8e-15 |  |  |  |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 16 | fixest | 2.5e-14 |  |  |  |
| `fx_simple_1000000 y~x1 fe=indiv_id+year+firm_id[x2] vcov=iid` | 1 | fixest | 2.5e-14 |  |  |  |
