# Baselines (plan Phase 0.4)

## fixest, n = 1e7, simple DGP

- **Dataset:** `fx_simple_10000000`: 10,000,000 obs; 1,000,000 indiv_id; 10 years; 43,478 firms; seed 1,000,000,042.
- **Command:** `feols(y ~ x1 | indiv_id + year, data, vcov = "iid", nthreads = 16)`. fixest 0.14.2, `setFixest_nthreads(16)`. Data were loaded beforehand with `fread`.
- **Measurement:** 1 burn-in plus 3 timed runs, using `system.time()[["elapsed"]]`.
- **Invocation:** `Rscript run_fixest.R data=data/fx_simple_10000000.csv fml='y~x1|indiv_id+year' method=feols vcov=iid threads=16 reps=3`
- **Conditions (2026-10-11):** the machine was shared with compile jobs; 1-minute load average 7.5–8.6 during the runs.

| Session | Burn-in (s) | Timed runs (s) | Median (s) |
|---|---|---|---|
| 1 | 0.582 | 0.485, 0.482, 0.476 | **0.482** |
| 2 | 0.578 | 0.473, 0.476, 0.478 | 0.476 |
| 3 | 0.563 | 0.489, 0.480, 0.481 | 0.481 |

**Baseline:** fixest takes **0.48 s** at T = 16 (median of 3, session 1). It gives x1 = 0.99951783696810914 with N = 10,000,000.

## Streaming bandwidth

Measured 2026-10-12 with `cargo run --release -p regfast-core --example stream_probe <threads> 4`
(best-of parallel sum of a 4 GiB f64 array; machine otherwise idle):

| Threads | Read bandwidth |
|---|---|
| 1 | 51.2 GB/s |
| 16 | 43.9 GB/s |
| 32 | 43.2 GB/s |

The roofline budgets use **43.9 GB/s** (16 threads).

## Roofline check (plan V-8), 2026-10-12

`regfast y x1, absorb(indiv_id year firm_id) threads(16) verbose(2)` on `fx_simple_10000000`
(after one warm-up run; commit `a22557b`). D2 budget of the solver call, as in
`examples/bench_solve.rs`: n = M = 10⁷ cells, Q − 1 = 2, R = 2, 9 PCG iterations;
1.5 × (n·R·24 + 2·M·(Q−1)·4) / bw + 9 × 1.5 × M·((Q−1)·4 + R·16) / bw = 0.022 + 9 × 0.0137 = **0.145 s**.

| Phase | Time (s) |
|---|---|
| Stata → plugin read (SPI, 5 variables) | 0.052 |
| prepare session (factorize 0.025, singletons 0.012, compact 0.003, structure 0.037, weights 0.011) | 0.090 |
| prepare store | 0.008 |
| ols: cached read | 0.002 |
| ols: weights + dof | 0.033 |
| ols: passes A + B | 0.009 |
| **solver: rhs 0.011 + pcg 0.056 + residualize 0.011** | **0.078** |
| whole command (`timer`, with display) | 0.340 |

The solver runs at 0.53× its budget. The whole command is 2.3× the solver budget: the SPI
transfer (one call per value; SPI 3.0 has no bulk read) and the prepare phases (factorization
and the cell sort) are not streaming passes the D2 model covers.
