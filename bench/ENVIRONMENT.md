# Benchmark environment

Recorded 2026-10-11 (plan Phase 0.3). Every version below was read from the installed software.

## Machine

- AMD Ryzen 9 9950X: 16 cores / 32 threads, AVX-512.
- 186 GB RAM.
- Arch Linux, kernel 7.2.9-arch1-1.
- The machine is shared with compile jobs. Timings made while other agents were compiling are marked as such in `BASELINE.md`.

## Incumbents

| Tool | Version | Where / how installed |
|---|---|---|
| Stata | 19.5 MP (`c(processors_max)` = 32) | `~/.local/bin/stata-mp` |
| reghdfe | 6.13.1 (10Jan2026) | `~/ado/plus/r` |
| ftools | 2.50.0 (09jan2026) | `~/ado/plus/f` |
| ivreghdfe | 1.1.4 (29nov2025) | `~/ado/plus/i`; uses ivreg2 4.1.12 and ranktest 2.0.04 |
| ppmlhdfe | 2.3.3 (02nov2025) | `~/ado/plus/p` |
| xhdfe | 2.28.1 (29sep2026), CPU-only net-install plugin | `net install xhdfe, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace` |
| R | 4.6.1 | system |
| fixest | 0.14.2 | CRAN, installed to user library `~/R/library` (`R_LIBS_USER` set in `~/.Renviron`) |
| data.table | 1.18.6.1 | CRAN, same library |
| Python | 3.14.3 (uv-managed) | `bench/.venv` via `uv sync` |
| pyfixest | 0.60.0 | `bench/pyproject.toml` / `uv.lock` |
| numpy / pandas / pyarrow | 2.5.4 / 3.0.6 / 26.0.0 | same |

## How each tool is run

| Tool | Thread control | Notes |
|---|---|---|
| Stata (all commands) | `set processors T` | `stata-mp -b do job.do`. `timer` wraps `quietly <cmd>`, so display time is included. Data come from `use <vars> using <name>.dta`. |
| reghdfe, ppmlhdfe, ivreghdfe | Stata MP only | Defaults: tol 1e-8; reghdfe `iterate()` default; ppmlhdfe `vce(robust)`; ivreghdfe `robust`. |
| xhdfe | `numthreads(T)` plus `OMP_NUM_THREADS=T` | Checked: `e(threads_used)` = 16 for `numthreads(16)`. |
| fereg / feppml / felogit | `threads(T)` | Commands are built by `run.py`. They are installed later by `scripts/install_local.sh`; until then they fail with r(199). |
| fixest | `setFixest_nthreads(T)`, `nthreads = T`, `setDTthreads(T)` | Data loaded with `data.table::fread`. Timed with `system.time()[["elapsed"]]`, which has 1 ms resolution. |
| pyfixest MAP | `RAYON_NUM_THREADS=T`, set before import | `demeaner=pf.MapDemeaner()`: fixef_tol 1e-6, fixef_maxiter 10000, Rust backend. |
| pyfixest LSMR | same | `demeaner=pf.LsmrDemeaner()`: `within` backend (CPU, float64), additive-Schwarz preconditioner, atol = btol = 1e-8, fixef_maxiter 1000. |

- `run.py` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `RAYON_NUM_THREADS` and `NUMBA_NUM_THREADS` to T for every subprocess.
- In the AKM groups G2/G3, fixest (`fixef.iter`) and pyfixest (`fixef_maxiter`) get 10000. This matches upstream `benchmark_akm_sweep.py`, which uses `build_standard_feols_benchmarkers(fixef_maxiter=10000)`.
- Every other solver setting is the tool's default.

## xhdfe options (from `help xhdfe`, 2.28.1)

**Threads**
- `numthreads(#)` sets the OpenMP threads used by the absorber. 0 = auto.
- A positive request is authoritative. It bypasses the auto heuristics and is clamped only to the logical CPUs and the OpenMP limit.
- Related options: `defaultthreads()`, `maxthreads()`, `minparallelrows()`, `targetrowsperthread()`.
- Diagnostics are stored in `e(threads_used)`, `e(threads_requested)` and `e(thread_capacity)`.

**IV/2SLS**
- Syntax: `xhdfe depvar [exogvars] endogvars, absorb() endogenous(endogvars) instruments(instvars)`.
- The endogenous variables must also appear in the varlist.
- Used in G4 with `vce(robust)`.

**Slopes**
- `absorb(fe#c.x)` and `absorb(fe##c.x)` are supported, so xhdfe is used in G7.

**Not available**
- No Poisson or logit, so xhdfe is not in G5/G6.

**Other**
- Defaults: `tolerance(1e-8)`, `tolerancemode(reghdfe-comparable)`.
- The GPU backends are not part of the net-install, so every run is CPU.

## Capability gaps (shown as `unsupported` in RESULTS.md)

- pyfixest 0.60.0 has no varying slopes. `y ~ x1 | indiv_id + year + firm_id[x2]` raises `FactorEvaluationError`, so pyfixest is not run in G7.
- Logit (G6) has no Stata competitor. The competitors are fixest `feglm(family = "logit")` and pyfixest `feglm(family="logit")`.

## Data

**Generators**
- `dgp.py` and the vendored pyfixest v0.60.0 generators under `vendor/pyfixest_bench/` (MIT).
- `uv run python verify_dgp.py` checks the following and printed `VERIFY OK` on 2026-10-11:
  - the vendored files are SHA-256-identical to upstream;
  - `fixest_dgp` reproduces upstream `base_dgp(k=10)` bit-for-bit at n = 1e3, 1e4 and 1e5 for both types;
  - all 35 AKM configs equal upstream `dgps.py`;
  - AKM panels are bit-identical to upstream `simulate_akm_panel`.

**Formats**
- `gen_data.py` writes parquet (read by pyfixest), CSV (pyarrow writer, round-trip exact doubles, read by fixest) and Stata `.dta` version 118.
- In the `.dta`, ids are `long` or `byte` and values are `double`.
- Datasets are cached in `bench/data/` (gitignored) and generated lazily per grid row.
