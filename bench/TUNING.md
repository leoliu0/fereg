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

## 2026-10-12: Python and R packages (`fereg_py`, `fereg_r`)

Gate run: `uv run python run.py --grid gate --tools fereg_py,fereg_r --missing` (504 jobs, all
`ok`). It ran while other agents were compiling (load average 7–12). The 41 FAIL or < 1.2× rows
were then re-run on a quiet machine with `--thin 1.2`. G5/G6 were re-run in full after the
packages were rebuilt at `98abc75` (core-glm merge; the OLS/IV/slopes paths are unchanged since
`5cf856d`).

Under load the first pass showed spurious T=16 FAILs. The quiet re-run cleared them:

| row | T | first pass | quiet re-run |
|---|---:|---:|---:|
| R `akm_sorting_1` | 16 | 0.656 | 0.134 |
| R `akm_scale_1` | 16 | 0.049 | 0.010 |
| R 1e7 3 FEs | 16 | 1.578 | 1.101 |
| R G5 1e5 k=5 | 16 | 0.101 | 0.022 |
| Python `akm_sorting_3` | 1 | 1.117 | 0.482 |

Result: `fereg_py` 265 PASS / 8 FAIL; `fereg_r` 268 PASS / 5 FAIL. Each remaining FAIL has one of
the two causes below. Neither is a solver problem. Coefficient gaps:
- versus the Stata `fereg`, the largest relative gap is 9.4e-9 (Python) and 3.0e-8 (R);
- versus fixest the median gap is about 1e-13 and the largest is 5.6e-5. All 20 rows above 1e-6
  are difficult 3-FE or Poisson rows, where fixest itself sits 5.6e-5 from the Stata `fereg`
  (fixest's looser `fixef.tol`).

Profiling scripts timed the package call end to end and the `_core.fit` / `fereg_fit` Rust call
separately, as medians of 3–5 runs after a warm-up. Overrides of the `_core.fit` arguments
isolated each cause.

### P1. R: fixest FE normalization at every estimation (all 5 R FAILs)

The R binding always estimates with `keep_fixef = TRUE` and `FixefNorm::Fixest`. In the core,
`fit/fixef.rs::get_fe_gnl` is fully serial:
- a CSR row list per term;
- a per-row `BinaryHeap` with lazy deletion for the "most determined row" buckets, which is
  O(N log N);
- a serial per-row FE-sum pass.

fixest's `feols` computes no FE values; `fixef()` recovers them on demand. Python passes the same
core with `fixef_norm` overridden. Rust call in seconds:

| row | T | Python (minnorm) | Python, `keep_fixef=False` | Python, `fixef_norm=fixest` | R | fixest (gate) |
|---|---:|---:|---:|---:|---:|---:|
| 1e5 3 FEs `y~x1` | 1 | 0.0068 | 0.0057 | 0.0098 | 0.011 | 0.011 |
| 1e6 3 FEs `y~x1` | 1 | 0.0744 | 0.0605 | 0.1207 | 0.124 | 0.087 |
| 1e7 3 FEs `y~x1` | 16 | 0.362 | 0.230 | 1.074 | 1.105 | 1.268 |
| `akm_mobility_1` 1e6 | 1 | 0.1215 | 0.1059 | 0.1710 | 0.177 | 0.146 |
| `akm_mobility_1` 1e6 | 16 | 0.0552 | 0.0422 | 0.1070 | 0.108 | 0.106 |
| G7 1e6 `firm_id[x2]` | 1 | 0.1118 | 0.0988 | 0.1602 | 0.162 | 0.166 |

With `fixef_norm=fixest` Python reproduces R's time on every row. Without FE values, every one of
these rows passes.

Proposals (R package and core owners):
1. **Lazy FE values.** Estimate with `keep_fixef = FALSE`. Compute the values in
   `fixef()`, in `predict(newdata=)` and in `ssc(K.exact = TRUE)` (the references) from the
   retained `FitState`, as fixest does. Expected effect:
   - R time drops to the `keep_fixef=False` column: 1e7 3 FEs 1.10 → about 0.23 s, and
     1e6 3 FEs T=1 0.124 → about 0.061 s (fixest 0.087);
   - this clears all 5 R FAILs.

   pyfixest's `fixef()` is lazy too. Doing the same in Python saves its minnorm cost: 10–25%,
   e.g. 0.362 → 0.230 s at 1e7 3 FEs and 0.074 → 0.061 s at 1e6.
2. **A faster `get_fe_gnl`**, which is still needed when `fixef()` is called:
   - replace each `BinaryHeap` bucket with a two-level bitset (one u64 summary word per 64
     words), which gives O(1) insert, delete and find-min for "lowest row with r determined
     terms";
   - build the per-term CSR lists with the parallel counting sort already used by `structure.rs`;
   - compute the per-row FE sum in parallel.

### P2. R: copies of N-length outputs (about 15–20 ms at 1e7)

`output()` copies `resid` and `fitted` into new R vectors (plus `eta` and `irls_weights` for GLMs).
Each copy is serial and page-faults on first write. Python moves each `Vec` into numpy with no
copy. Evidence at 1e7 `y~x1|indiv_id+year`, T=16:
- R: 0.218 s full vs 0.199 s `lean = TRUE`;
- Python: full and lean are equal (0.230 vs 0.230 at 3 FEs).

Proposals (pick one):
- allocate the R vectors on the main thread (`Doubles::new(n)`) and fill them inside the pool
  with a parallel chunked copy (worker threads only write raw memory);
- let the core write residuals into a caller-provided buffer;
- return only `resid` and derive `fitted = y − resid` lazily in R.

### P3. Python logit (all 8 Python FAILs): pyfixest's sample, not speed

All 8 rows are G6 at T=1, where `fereg_py` is 1.0–1.8× fixest. At T=16 every G6 row passes.
`fereg_py` reproduces pyfixest's `fe` separation check, which drops all-zero FE groups but keeps
all-one groups. On those groups η diverges, and IRLS iterates until their deviance change falls
under `glm_tol`. fixest, `fereg_r` and the Stata `fereg` drop the groups.

Evidence on `fx_simple_1000000 binary_y~x1|indiv_id+year`, T=1 (Rust call):

| run | N | IRLS iterations | time (s) | per iteration (s) |
|---|---:|---:|---:|---:|
| pyfixest flavor (`fereg_py`, matches pyfixest) | 972,820 | 14 | 0.719 | 0.051 |
| fixest flavor (`fereg_r`, matches fixest) | 964,110 | 5 | 0.343 | 0.069 |

At 1e5 the counts are 15 vs 5 iterations, 0.078 vs 0.036 s. The two flavors' β differ by 1.4e-9
relative. Against pyfixest itself (MAP 1.5 s, LSMR 6.2 s on this row) `fereg_py` is 2–9× faster.

Proposal (Python package and core owners): under the pyfixest flavor, exclude all-one (and
all-zero) FE groups from the IRLS iterations, as fixest does. Their linear predictor diverges, so
in the limit they contribute nothing to the score of β. pyfixest's reported N, df and statistics
stay, and β moves by about 1e-9 relative to pyfixest. Expected effect: G6 T=1 back to the fixest
flavor's time (about 0.34 s here vs fixest 0.396), which clears all 8 rows.

This changes numbers at the 1e-9 level. If parity with pyfixest must stay bitwise-close, keep the
current behaviour and record these rows as convention-bound.

### The R-vs-Python Rust call: 0.222 vs 0.161 s (1e7, `y~x1|indiv_id+year`, T=16)

Reproduced: R 0.218 s, Python 0.162 s. Breakdown:

| step | Rust call (s) |
|---|---:|
| Python default (pyfixest flavor, minnorm FE values, zero-copy outputs) | 0.162 |
| Python with `fixef_norm=fixest` (R's FE normalization, P1) | 0.2065 |
| R `lean = TRUE` (no resid/fitted copies; FE values still computed) | 0.199 |
| R full (adds the resid/fitted copies, P2) | 0.218 |

So the gap is about 45 ms of fixest FE normalization plus about 15–20 ms of output copies.

Excluded:
- **Build profile and flags.**
  - Both crates build with opt-level 3, fat LTO, 1 codegen unit, and the same dependency versions
    (faer 0.24.4, rayon 1.12.0, pulp 0.22.3, approx-chol 0.6.0).
  - The in-place `R CMD INSTALL r/fereg` build uses the repository `.cargo/config.toml`
    (x86-64-v2): 182 SSE4.x/POPCNT instructions in `fereg.so`, 177 in the Python `_core`.
  - A tarball install (`R CMD build` + `R CMD INSTALL`, i.e. CRAN or r-universe) builds outside the
    repository for baseline x86-64: 0 such instructions. It times the same: 0.218 vs 0.218 s
    (2 FEs, 1e7), 1.064 vs 1.079 s (3 FEs, 1e7), 0.040 vs 0.039 s (2 FEs, 1e6, T=1).
- **Factor column type.** int32, double and factor columns give 1.116 / 1.113 / 1.120 s at 1e7 with
  3 FEs.
- **Threads.** Both runners pass T explicitly.
- **Data conversion on input.** Inputs are zero-copy in both bindings: R `as_real_slice` /
  `as_integer_slice`, numpy borrowed slices.

### After stage D (`888d8ec`): lazy FE values, parallel output fill, pyfixest-flavor logit

Stage D (`c583aa9`, `c55608f`, `de7a893`, `888d8ec`) implements P1–P3:
- FE values are computed on demand from the fit state (R `fixef()`/`predict(newdata)`, Python
  `FitState.fixef`);
- reference propagation is faster (parallel CSR, bitset min-sets);
- R fills `resid`/`fitted`/`eta`/weights in parallel;
- the pyfixest-flavor logit runs IRLS without its diverging all-0/all-1 levels.

Both packages were rebuilt at `888d8ec` (0.2.1). All 546 `fereg_py`/`fereg_r` gate jobs were
re-run (all `ok`), then the five rows with ratio > 1/1.2 were re-run with `--thin 1.2` on a quiet
machine. No competitor or Stata row changed.

Result: `fereg_py` 273 PASS / 0 FAIL, `fereg_r` 273 PASS / 0 FAIL. No new FAILs.

The 13 former FAIL rows (T = threads; ratio = candidate / fastest finished competitor):

| tool | row | T | before s | before ratio | after s | after ratio |
|---|---|---:|---:|---:|---:|---:|
| py | G6 difficult 1e5 `binary_y~x1` FE2 | 1 | 0.083 | 1.72 | 0.039 | 0.81 |
| py | G6 difficult 1e5 `binary_y~x1..x5` FE2 | 1 | 0.106 | 1.04 | 0.054 | 0.53 |
| py | G6 difficult 1e6 `binary_y~x1` FE2 | 1 | 0.729 | 1.81 | 0.362 | 0.90 |
| py | G6 difficult 1e6 `binary_y~x1..x5` FE2 | 1 | 0.996 | 1.02 | 0.510 | 0.52 |
| py | G6 simple 1e5 `binary_y~x1` FE2 | 1 | 0.081 | 1.61 | 0.039 | 0.77 |
| py | G6 simple 1e5 `binary_y~x1` FE3 | 1 | 0.113 | 1.33 | 0.063 | 0.74 |
| py | G6 simple 1e6 `binary_y~x1` FE2 | 1 | 0.715 | 1.81 | 0.360 | 0.91 |
| py | G6 simple 1e6 `binary_y~x1` FE3 | 1 | 1.111 | 1.37 | 0.633 | 0.78 |
| R | G2 `akm_mobility_1` 1e6 | 1 | 0.176 | 1.21 | 0.110 | 0.75 |
| R | G2 `akm_mobility_1` 1e6 | 16 | 0.111 | 1.05 | 0.045 | 0.42 |
| R | G1 simple 1e5 `y~x1` FE3 | 1 | 0.013 | 1.18 | 0.008 | 0.73 |
| R | G1 simple 1e6 `y~x1` FE3 | 1 | 0.125 | 1.44 | 0.066 | 0.76 |
| R | G7 simple 1e6 `firm_id[x2]` | 1 | 0.170 | 1.02 | 0.105 | 0.63 |

Python logit now keeps pyfixest's sample (N = 972,820 on simple 1e6 FE2) and reaches the fixest
flavor's β (1.2158325747) in the same time as R. Its gap to pyfixest is 1.4e-9 relative.

The 1e7 rows (`y~x1`, T=16, Rust call medians) moved as follows:

| model | Python before (s) | Python after (s) | R before (s) | R after (s) |
|---|---:|---:|---:|---:|
| 3 FEs | 0.362 | 0.241 | 1.105 | 0.248 |

The R-vs-Python gap is gone.

Rows still within 1.2×, all T=1 and all PASS:
- G6 1e6 `binary_y~x1` FE2 vs fixest: Python 0.90–0.91, R 0.87–0.88. Both run 5 IRLS iterations,
  on par with the Stata `fereg` on the same row (0.359 s).
- Python `akm_scale_1` 1e4: 0.88 (10.9 ms vs pyfixest LSMR 12.4 ms; mostly fixed per-call overhead).

Coefficient gaps are unchanged:
- versus fixest: median 5.7e-14, max 5.6e-5 (fixest's tolerance on difficult 3-FE/Poisson rows);
- versus the Stata `fereg`: max 3.0e-8.

## 2026-10-13: GLM loop, scatter accumulators, cluster variances, elimination choice

Measured with `crates/fereg-core/examples/bench_csv.rs` (one model on a `bench/data` CSV:
median of `--reps` fits, the first coefficient and its SE, and an FNV hash of the bits of the
coefficients and the variance matrix; `--verbose` adds one fit with phase timings). Ryzen 9
9950X, otherwise idle. Each row: minimum over 2 alternating rounds of the median.

| Row | Before (`6a51f1d`) | After | Ratio |
|---|---|---|---|
| OLS simple 1e7, 3 FEs, T=16 | 0.236 s | 0.220 s | 0.93 |
| OLS difficult 1e7, T=16 | 0.328 s | 0.311 s | 0.95 |
| OLS 5 regressors, cluster, 1e7, T=16 | 0.457 s | 0.371 s | 0.81 |
| AKM scale 1e7, T=16 | 0.838 s | 0.784 s | 0.94 |
| IV 1e7, hetero, T=16 | 0.402 s | 0.337 s | 0.84 |
| Poisson 1e7, hetero, T=16 | 2.530 s | 2.066 s | 0.82 |
| logit 1e7, T=16 | 1.980 s | 1.644 s | 0.83 |
| AKM δ = 1 / 0.1 / 0.001, 1e6, T=16 | 0.037 / 0.131 / 0.027 s | 0.034 / 0.133 / 0.027 s | 0.93 / 1.02 / 0.98 |
| OLS simple / difficult 1e6, T=1 | 0.062 / 0.106 s | 0.059 / 0.105 s | 0.95 / 0.99 |
| AKM δ = 0.1, 1e6, T=1 | 0.223 s | 0.231 s | 1.03 |
| IV / Poisson / logit 1e6, T=1 | 0.113 / 0.628 / 0.610 s | 0.109 / 0.584 / 0.588 s | 0.96 / 0.93 / 0.96 |
| Slopes 1e6, T=1 | 0.102 s | 0.102 s | 1.00 |
| Taxi OLS, route + how + day, two-way cluster, T=16 | 0.404 s | 0.343 s | 0.85 |
| Taxi OLS, 4 FEs (`m2` of the article) | 1.211 s | 0.843 s | 0.70 |
| Taxi Poisson, 4 FEs | 8.802 s | 4.185 s | 0.48 |
| Taxi logit, 4 FEs | 10.602 s | 6.259 s | 0.59 |

The AKM δ = 0.1 rows spend most of their time in the joint sparse factorization, which did not
change; the cell-code check of item 10 adds about 1 ms at T=1. Results stay bit-identical across
thread counts (checked at T = 1, 4, 7, 16 on seven of the rows); against the old version they
differ in the last bits where the scatter task count or the PCG path changed (items 8 and 9) or
the variance sums are added in another order (item 11).

### 7. IRLS loop without per-iteration copies

`Absorber::set_weights` borrows the weights and copies them into the previous buffer when the
length matches (`par::copy_into`), with a parallel positivity check (`par::all_positive`); the
solver's weight gather is parallel; cell weights are not stored when every cell holds one row.
Without the mu separation check, eta, the previous eta, mu and the deviance terms are computed
in one pass, with the per-row arithmetic and chunk sums of `update_mu_deviance`. Bit-identical:
Poisson 1e7 T=16 2.53 → 2.37 s, logit 1e7 1.97 → 1.83 s; taxi GLMs within 0.5%.

### 8. Scatter accumulators within a byte budget

`Structure::tasks` grouped the parts so that tasks × accumulator length ≤ 2^22 values: on the
taxi panel (2.6M cells, m·r = 195k) 53 tasks, 83 MB of accumulators beyond the 64 MiB of L3, 7.1 ms
per matvec. `acc_tasks` now takes 16 to 64 tasks within 8 MiB in total and 1/8 of the cell data:
13 tasks, 3.7 ms with the same iterations. The ordered combine zeroes the entries it reads
instead of a separate reset pass: 3.4 ms. Over two suites 8 MiB was 1–3% faster than 16 MiB on
six of the seven 1e7 T=16 rows (1.5% slower with 5 clustered regressors); 4 MiB was no faster.
Rejected: no accumulators (every block through the M × r buffer): 27.6 ms per matvec.

### 9. Refreshing stale block factors by cost

Between IRLS iterations the exact sparse factors of escalated large blocks were kept as
preconditioners for the new weights. They are now refactored before a solve when the modelled
rebuild cost (assembly work and nonzeros, factor flops and entries) is at most 30% of the last
solve's modelled PCG cost (`stale_refresh_pays`); approximate factors keep the iteration-count
rule. The inputs are structural, so the path does not depend on the thread count. Taxi Poisson:
654 → 361 PCG iterations, 6.15 → 4.95 s.

### 10. Unchecked gather and scatter kernels

`gather_cells`, `scatter_cell` and `scatter_cells` index the reduced vectors with
`get_unchecked`. They are `unsafe fn` with the precondition `Structure::codes_within`, asserted
once per task. It reads only private data, the cell codes (now behind `cell_codes()`) and their
level bounds, and `build` checks every cell code against its bound, so no change to the public
fields can break it. Unchecked vs checked, bit-identical: OLS difficult 1e6 T=1 0.108 → 0.105 s,
Poisson 1e6 T=1 0.611 → 0.575 s, Poisson 1e7 T=16 2.38 → 2.35 s; taxi within noise. The
asserts and the check cost 1–2%, within run-to-run noise.

### 11. Cluster variances

`group_rows` sorted rows by cluster with atomic cursors (contended at small G: 0.08–0.14 s per
dimension on the taxi panel) and then sorted each cluster. It now uses the stable counting sort
of the structure (`scatter_by_code`), which gives the same permutation. The score sums of every
subset whose sums fit (16 tasks of at most 2 MB) come from one pass over the rows in row order
(`stream_sums`: fixed row ranges, added in task order); larger subsets, such as clusters by
worker, keep the grouped path. Taxi, two-way clustering (129, 249 and 11.3k clusters): variance
phase OLS 0.204 → 0.050 s, Poisson 0.586 → 0.115 s (the GLM block is in kept order, so the
grouped pass read its rows at random). The `log_dist` SE of `m2` is 0.16102386185708500 against
0.16102386185708464 from the old code (relative 2e-15).

### 12. Elimination choice in linear time

With several candidates, E was computed from a comparison sort of the 6.7M level pairs of every
pair of terms (0.124 s on the taxi panel). Now the rows are counting-sorted by each candidate and
one pass counts the distinct levels of the other terms per level with stamp arrays; the counts
are exact, so the choice is the same, and the sort of the chosen term is reused as the FE1 sort
of the structure. Taxi `m2`: structure 0.185 → 0.137 s, fit 0.872 → 0.833 s, bit-identical.

### 13. fixest / pyfixest collinearity in the first IRLS iteration

The R and Python GLMs ran ppmlhdfe's `remove_collinears` before IRLS under the fixest
convention: one solve of X at the fixed-effects tolerance (taxi Poisson: 63 PCG iterations,
about 0.45 s), with a rule that can drop other near-collinear columns than fixest's. They now
apply fixest's `cpp_cholesky` rule where fixest does (`GlmCollinear`): to X residualized in
the first IRLS iteration, weighted by its IRLS weights (pyfixest: unweighted). The pass stays
for Stata and before pyfixest's ReLU separation check. The first FE solve keeps its loose
tolerance, which bounds the error of each residualized column by tol_in times its W-norm (the
CG stopping rule). `chol_collin_certain` accepts the outcome when 100 times that bound,
propagated through each column's coefficients on the earlier columns, plus the rounding of the
factorization, cannot move a pivot across `tol`. Otherwise the block is residualized again at
the final tolerance; the loose error lies in the FE space, so the second solve completes the
first. Solving the first iteration at the final tolerance every time was up to 9% slower
(Poisson 1e7) than the old pass.

| Row | Before | Tight first solve | Certified |
|---|---|---|---|
| Poisson 1e7, hetero, T=16 | 2.077 s | 2.270 s | 2.024 s |
| logit 1e7, T=16 | 1.647 s | 1.606 s | 1.590 s |
| Poisson / logit 1e6, T=1 | 0.584 / 0.591 s | 0.633 / 0.558 s | 0.559 / 0.554 s |
| Taxi Poisson, 4 FEs | 4.173 s | 4.164 s | 3.790 s |
| Taxi logit, 4 FEs | 6.280 s | 6.127 s | 6.065 s |

Minimum over 3 rounds. The first coefficient and its SE keep all 16 printed digits except on
taxi Poisson (relative 4e-15 and 4e-13). Against fixest 0.14.2 and pyfixest 0.60.0, 12 Poisson
and logit models with duplicated, FE-absorbed, near-collinear and combined regressors drop the
same columns (R: coefficients within 1e-14).

### Not done

- Gathering the row block (taxi: 0.09 s) makes two passes over 427 MB; its page faults already
  use 2 MB pages (THP `always`).
- The candidate passes of item 12 gather three codes per row at random (about 0.08 s on the
  taxi panel); an interleaved code matrix would cut the cache misses.

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
