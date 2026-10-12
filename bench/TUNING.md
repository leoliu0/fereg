# Tuning log (plan Phase 8.6)

Each entry: the change, the rows it targets, and the measured effect. Timings are wall-clock
seconds of the estimation command inside `stata-mp` (`timer` around `quietly fereg ...`,
data in memory, `set processors T` and `threads(T)`), on the machine of `ENVIRONMENT.md`
(Ryzen 9 9950X, 16 cores). Ranges are the spread over 5–9 repetitions after one warm-up run.
Data: `bench/data/fx_simple_<n>.dta` (fixest "simple" DGP; `y x1` unless noted).

## 2026-10-11

### 1. Plugin-side sample (`861b13f`)

`prepare` reads the model variables (depvar, regressors, endogenous, instruments) together
with the factors in its one-pass scan and drops observations with a missing value itself; the
ados no longer run `marksample` over the varlist or `markout`. The estimation command takes the
variables from prepare's cache instead of reading them from Stata again.

| Row (T=1) | Before (gate median) | After |
|---|---|---|
| 1e6, absorb(indiv_id year) | 0.056 | 0.042–0.047 |
| 1e6, absorb(indiv_id year firm_id) | 0.092 | 0.079–0.089 |
| 1e5, absorb(indiv_id year firm_id) | 0.011 | 0.011–0.013 |

With one column per cached variable, the estimation-time gather (11 random reads per row)
made 1e7 T=16 with 10 regressors slower than before (0.52–0.55 vs 0.50–0.52), which led to
entries 2 and 3.

### 2. Row-major model cache (`861b13f`)

The cache holds one contiguous row per observation (the C scan writes block variables with a
stride), so the gather reads one row per solver row.

| Phase, 1e7 T=16, 10 regressors | Column cache | Row-major |
|---|---|---|
| prepare read | 0.182 | 0.135 |
| estimation-time gather | 0.089 | 0.078 |
| whole command | 0.52–0.55 | 0.455–0.471 |

### 3. Cache used as is (`861b13f`)

When every estimation variable is cached in prepare's order, nothing was dropped after the
read and the solver rows are unpermuted (panels sorted by the largest FE, e.g. after `xtset`),
prepare's block is the solver block: no gather (0.078 → 0.002 s at 1e7 × 11).

| Row, 1e7 | Old ado path | Entries 1–3 |
|---|---|---|
| T=16, 2 FEs, k=1 | 0.198–0.219 | 0.159–0.169 |
| T=16, 3 FEs, k=1 | 0.350–0.359 | 0.294–0.306 |
| T=16, 2 FEs, k=10 | 0.496–0.522 | 0.385–0.402 |
| T=1, 2 FEs, k=1 | 0.468–0.477 | 0.350–0.358 |
| T=1, 3 FEs, k=1 | 0.938–0.951 | 0.805–0.810 |
| T=1, 2 FEs, k=10 | 1.338–1.360 | 1.049–1.071 |

### 4. `sample=all` (`b59f1f6`)

Without `if`, `in`, weights or ado markouts, touse is 1 everywhere; the ado says so and the scan
skips the touse read (one of the 5–6 SPI reads per observation).

| Row (T=1) | Before | After |
|---|---|---|
| 1e5, 3 FEs (mean of 20) | 0.0115 | 0.0106 |
| 1e6, 2 FEs | 0.040–0.043 | 0.037–0.039 |
| 1e6, 3 FEs | 0.077–0.080 | 0.074–0.077 |
| 1e7, 2 FEs, k=1 | 0.350–0.358 | 0.328–0.339 |
| 1e7, 3 FEs, k=1 | 0.805–0.810 | 0.774–0.789 |
| 1e7, 2 FEs, k=10 | 1.049–1.071 | 1.002–1.020 |

### Rejected

Target: 1e5, 3 FEs, T=1 (gate: fereg 0.011 s vs fixest 0.011 s). Profile of the 10.6 ms:
SPI read 2.4, session 1.9 (factorize 0.5, structure 0.7, weights 0.5), estimate 5.5, of which
the dense small-block preconditioner 4.3 (assembly 3.6 + LDLᵀ of the 444×444 Schur complement
0.65); the rest is ado code. Times below are `bench_solve --n 1e5 --fe 3` residualize phases.

- **Row-wise small assembly** (each row of S accumulated in L1 from its levels' cells and the
  per-segment level sums): assembly 3.6 → 5.5 ms at T=1 and 1.5 → 2.5 ms at T=16. The rows
  read the 2.4 MB of segment sums at random (L3), which costs more than the current
  L2-resident packed-triangle scatter.
- **Firm as a large block** (Jacobi on the 434 firm levels instead of the dense factor):
  simple R=2: 5.7 → 2.9 ms (10 PCG iterations), but R=11: 7.0 → 20.5 ms and difficult R=2:
  3.5 → 7.2 ms (escalation). Any such rule has to depend on the number of columns, which is
  only known at solve time.

## 2026-10-12

### 5. Small assembly: bitmap read-back instead of a per-segment sort

The touched small coordinates of a segment are marked in a bitmap and read back in ascending
order (clearing it), which replaces the stamp array and the sort of the touched list. Same
summation order, so the results are bit-identical.

| `bench_solve --n 1e5 --fe 3`, assembly (warm runs) | Before | After |
|---|---|---|
| simple, T=1 | 3.5–3.8 ms | 2.9–3.3 ms |
| simple, T=16 | 1.2–1.7 ms | 1.1–1.5 ms |
| difficult, T=1 | 2.3–2.9 ms | 1.9–2.8 ms |

In Stata (1e5, 3 FEs, T=1, `verbose(2)`): preconditioner 4.3 → 3.3 ms, estimate 5.5 → 4.6 ms.

### Rejected

- **Runs of consecutive coordinates as contiguous updates** (the year prefix of every row):
  all runs: simple T=1 3.1 → 3.5 ms, difficult 2.6 → 2.3 ms; runs of ≥ 4 only: simple 3.6 ms,
  difficult 3.0 ms. With random firm codes most runs have length 1, and the run bookkeeping
  per row costs more than it saves.
- **Row blocks sized for L2** (the packed triangle split into 4 row blocks of ≤ 32K entries):
  simple T=1 assembly 3.1 → about 10 ms; every row block rescans all segments.
- **Dense panel for the first small block's columns** (the 10 year columns of every row in a
  35 KB array, the rest packed): simple T=1 3.1–3.3 ms (no gain); with contiguous panel updates
  and unchecked indexing of the rest, also no gain; with the diagonal taken from the level
  weights instead of per-cell writes, 2.9–3.1 ms (≤ 0.2 ms). Gating parts of the loop off
  showed where the 3.3 ms goes: row updates 0.95, per-cell cross counts 1.1, scan, bitmap,
  combine and mirror 1.25.

### The 1e5, 3 FEs, T=1 row

The gate times each run with 1 ms resolution (Stata `timer`, R `system.time`) and compares
medians of 3 runs: both read 0.011 s, which is a FAIL (the rule needs strictly below). At finer
resolution:

| Tool | Measurement | Time |
|---|---|---|
| fereg | mean of 50 consecutive runs (one timer around the loop) | 10.5–10.7 ms |
| fixest | 50 runs, `Sys.time()` per run | median 11.6 ms, mean 12.9 ms, min 10.3 ms |

With a timer around the `plugin call` in a throwaway copy of `_rf_load` (50 commands), the two
plugin calls took 8.2 of the 10.5 ms, so 2.3 ms is ado code, spread over the sections (post
results 1.0, of which `ereturn post` 0.4, scalars 0.2, r(table) 0.3; sample 0.4; absorb parsing
0.3). Timed directly (500 runs), the ~30 scalar posts through `PostScalar` cost 64 µs per
command (22 µs from Mata), and the per-command plugin version check 4 µs.

### 6. Deferred dense small block (small R, every block small)

When every block is small and assembling + factoring the dense small block is estimated to
cost more than 8 Jacobi PCG iterations, the solve starts with Jacobi on every coordinate. At
the PCG probes (k = 6, 12) the dense factor is built, and PCG restarts from the current
iterate, when the projected remaining iterations cost more than it. Once a probe has had to
build it, later solves of the same solver (the next IRLS weights) build it up front. The
decisions use only the structure, R and PCG values, so results stay identical at any thread
count. Cost model (`solve.rs`): a PCG iteration costs 6e-10 s per cell and (block + column)
when the cells fit in 32 MiB (else `C_CELL` = 2.8e-9) plus `C_COORD` per coordinate and
column; the assembly 8e-10 s per unit of Σ_s d_s², the LDLT `C_FLOP`·m³/3.

| `bench_solve --n 1e5 --fe 3`, residualize (best of 5, 2 interleaved rounds) | Before | After | Path after |
|---|---|---|---|
| simple, R=2, T=1 | 4.0–5.0 ms | 2.7–2.8 ms | Jacobi, 10 iterations |
| simple, R=2, T=16 | 3.1–3.2 ms | 1.4–1.5 ms | Jacobi, 10 iterations |
| difficult, R=2, T=1 | 3.8 ms | 5.3 ms | factor built at k=6, 7 iterations |
| difficult, R=2, T=16 | 1.7 ms | 2.4 ms | factor built at k=6, 7 iterations |
| R=11 | | | not deferred |

The difficult rows pay the bounded cost of the policy (6 Jacobi iterations and a restart).
Gate rows that changed (1 ms resolution, medians of 3; full fereg grid re-run):

| 1e5, 3 FEs | T | Before | After |
|---|---|---|---|
| `fereg y x1`, simple | 1 | 11 | 10 (fixest 11: the former FAIL) |
| `fereg y x1`, simple | 16 | 9 | 8 |
| `fereg y x1`, difficult | 1 / 16 | 11 / 8 | 12 / 9 |
| `feppml negbin_y x1`, simple | 1 / 16 | 100 / 52 | 77 / 42 |
| `felogit binary_y x1`, simple | 1 / 16 | 83 / 41 | 66 / 32 |

Without the up-front rule every IRLS solve deferred again: difficult `x1` logit 87 → 134 ms,
Poisson 95 → 142 ms (T=1). With it (Stata, T=1, difficult 1e5, interleaved means of 20 runs):
`feppml x1` 82.4–83.2 → 81.7–83.0 ms; with `x1`–`x5`, where only the R=5 collinearity solve
is deferred, `feppml` 102.2–104.0 → 104.4–106.1 ms and `felogit` 89.8–92.5 → 91.8–95.1 ms.

In Stata, 1e5, 3 FEs, T=1, `y x1` (mean of 200 runs, three interleaved sessions each):
10.3–10.8 ms before, 9.6–9.7 ms after. Single-run `timer` readings (100 runs): ≤ 9 ms 45%,
10 ms 45%, 11 ms 9%, 12 ms 1% (before: ≤ 9 ms 4%, 10 ms 54%, 11 ms 32%, ≥ 12 ms 10%). Checked
against `reghdfe ... tol(1e-12)` (`rf_compare_e`, b 1e-7, V 1e-6) on the simple and difficult
1e5 data with 1 and 2 regressors, with bitwise-identical b at 1 and 16 threads. Gate: 273 PASS,
0 FAIL.

## Earlier tuning (commit log)

- `0a25c0f`, `738d70f`: one-pass (parallel) sample scan, uninitialized read buffers.
- `31045b3`: prepare phases at T=1 (factorize, singletons, compact, structure) 2.5–3× faster.
- `ab52a25`, `469e6d0`: const-width register-resident kernels; small dense block fused into the
  weight pass; sparse small assembly through a stamped scratch and packed lower triangle.
- `12609d4`: sparse triangular solves run concurrently per (block, column group).
- `fc249b1`: IV per-row combinations as register dot products (passes 4–10× faster).
- `6e76ba5`: GLM IRLS fused mu/deviance pass, buffers reused across iterations.
- `2ff3823`, `d8e55c1`: SlopeSolver row kernels, slot tables, sort-free small-segment grouping.
- `8ca0e4d`, `1116a51`: small-n task and pool sizing; no display work under `quietly`.
