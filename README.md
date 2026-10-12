# fereg

Fast high-dimensional fixed-effects estimation for Stata, powered by a Rust core.

| Command | Estimator | Mirrors |
|---|---|---|
| `fereg` | OLS and IV/2SLS with any number of absorbed fixed effects | `reghdfe` 6.13.1 / `ivreghdfe` 1.1.4 |
| `feppml` | Poisson pseudo-maximum likelihood | `ppmlhdfe` 2.3.3 |
| `felogit` | Logit maximum likelihood | `logit y x i.fe1 i.fe2 ...` |

All three support weights, robust and multi-way clustered standard errors, residuals and
`predict`, saved fixed effects, and heterogeneous slopes (`absorb(firm##c.x)`).

The solver eliminates the largest fixed effect exactly and solves the remaining system with a
preconditioned conjugate-gradient method on cell-compressed sufficient statistics.

## Performance

Seconds per estimation call at 10 million observations and 16 threads, data in memory: the
median of 3 timed runs after a warm-up (1 run when the warm-up takes over 60 s); `>900` = the
job did not finish within its 900 s limit. The data are the fixest/pyfixest benchmark DGPs:
panels of individuals × 10 years with firms assigned at random ("simple") or cyclically by row
("difficult", a fixed-effect graph that is hard for iterative solvers).

| Model | fereg | Stata incumbent | fixest (R) | fastest other |
|---|---:|---:|---:|---:|
| OLS `y x1`, 2 FEs | 0.15 | 11.7 (reghdfe) | 0.51 | 0.87 (xhdfe) |
| OLS `y x1`, 3 FEs | 0.29 | 31.4 (reghdfe) | 1.27 | 1.93 (xhdfe) |
| OLS 10 regressors, 3 FEs | 0.90 | 60.5 (reghdfe) | 5.45 | 6.52 (pyfixest) |
| OLS `y x1`, 3 FEs, difficult | 0.38 | >900 (reghdfe) | 35.8 | 5.93 (pyfixest LSMR) |
| IV/2SLS, 3 FEs, robust | 0.48 | 41.5 (ivreghdfe) | 2.67 | 4.10 (xhdfe) |
| Poisson PML, 3 FEs, robust | 3.15 | 235.6 (ppmlhdfe) | 43.1 | 202.8 (pyfixest LSMR) |
| Logit, 3 FEs | 2.27 | — | 8.84 | 26.7 (pyfixest) |

Across the whole benchmark grid, fereg is faster than every competitor that finished in all
273 rows (configuration × thread count). The grid covers OLS (iid and clustered), IV, Poisson,
logit, heterogeneous slopes and AKM worker–firm designs, n = 10⁴–10⁷, at 16 threads and, up to
10⁶, 1 thread. Machine: Ryzen 9 9950X (16 cores),
Stata 19.5 MP, reghdfe 6.13.1, ivreghdfe 1.1.4, ppmlhdfe 2.3.3, xhdfe 2.28.1, fixest 0.14.2,
pyfixest 0.60.0. Full tables: [`bench/RESULTS.md`](bench/RESULTS.md); method:
[`bench/ENVIRONMENT.md`](bench/ENVIRONMENT.md); rerun with
`cd bench && uv run python run.py --grid gate`.

## Install (Stata 16+)

```stata
net install fereg, from("https://raw.githubusercontent.com/leoliu0/fereg/gh-pages/stata") replace
```

Binaries are provided for Linux (x86-64), macOS (Apple Silicon and Intel) and Windows (x86-64).

## Build from source

Requires Rust (stable) and a C compiler.

```sh
scripts/build_plugin.sh      # builds target/release plugin and copies it to stata/
scripts/install_local.sh     # installs the ado files and plugin into your PERSONAL ado directory
```

The Rust library alone: `cargo test --workspace --release`.

## License

MIT
