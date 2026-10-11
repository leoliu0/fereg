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

Not measured yet. The bandwidth number comes from `crates/regfast-core/examples/stream_probe.rs` (16-thread streaming read over 4 GiB). That file belongs to the CoreSolver slice and is not on `main` yet. Run it when it lands, with `cargo run --release -p regfast-core --example stream_probe`, and record the GB/s here.
