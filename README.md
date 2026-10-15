# fereg

Fast high-dimensional fixed-effects regressions for Stata, Python and R, on one Rust core.

| Front end | Commands | Reproduces |
|---|---|---|
| Stata 16+ | `fereg`, `feppml`, `felogit` | `reghdfe` 6.13.1 / `ivreghdfe` 1.1.4, `ppmlhdfe` 2.3.3, `logit` with FE dummies |
| Python 3.10+ | `fereg.feols`, `fepois`, `feglm` | `pyfixest` 0.60.0 |
| R 4.2+ | `fereg::feols`, `fepois`, `feglm` | `fixest` 0.14.2 |

Each front end follows the syntax, defaults, small-sample corrections and output of the package
it mirrors, so results match that package's to numerical precision. All three cover OLS, IV/2SLS,
Poisson and logit with any number of fixed effects, interactions of fixed effects, varying slopes,
weights, and iid, heteroskedasticity-robust (HC1; HC2/HC3 except for IV), multiway clustered
(CRV1, CRV3), Newey–West, Driscoll–Kraay and Conley standard errors, plus residuals, `predict`
and fixed-effect values.

The solver eliminates the largest fixed effect exactly and solves the remaining system with a
preconditioned conjugate-gradient method on cell-compressed sufficient statistics.

## Performance

Seconds per estimation call at 10 million observations and 16 threads, data in memory: the
median of 3 timed runs after a warm-up (1 run when the warm-up takes over 60 s); `>900` = the
job did not finish within its 900 s limit; `—` = the tool has no such estimator.
The data are the fixest/pyfixest benchmark DGPs: panels of individuals × 10 years with firms
assigned at random ("simple") or cyclically by row ("difficult", a fixed-effect graph that is
hard for iterative solvers).

| Model | fereg Stata | fereg Python | fereg R | Stata incumbent | xhdfe | pyfixest | fixest |
|---|---:|---:|---:|---:|---:|---:|---:|
| OLS `y ~ x1`, 2 FEs | 0.15 | 0.14 | 0.15 | 11.7 | 0.87 | 1.82 | 0.51 |
| OLS `y ~ x1`, 3 FEs | 0.29 | 0.24 | 0.25 | 31.4 | 1.93 | 2.61 | 1.27 |
| OLS 10 regressors, 3 FEs | 0.90 | 0.85 | 0.86 | 60.5 | 6.68 | 6.52 | 5.45 |
| OLS `y ~ x1`, 3 FEs, difficult | 0.38 | 0.32 | 0.34 | >900 | 73.5 | 5.93 | 35.8 |
| IV/2SLS, 3 FEs, robust | 0.48 | 0.42 | 0.42 | 41.5 | 4.10 | 7.00 | 2.67 |
| Poisson PML, 3 FEs, robust | 3.15 | 2.68 | 2.60 | 235.6 | — | 202.8 | 43.1 |
| Logit, 3 FEs | 2.27 | 2.25 | 2.04 | — | — | 26.7 | 8.84 |

Stata incumbents: `reghdfe` (OLS), `ivreghdfe` (IV), `ppmlhdfe` (Poisson). pyfixest is the
faster of its MAP and LSMR solvers. The Stata times include Stata's value-by-value transfer of
the data to the plugin.

The full grid covers OLS (iid and clustered), IV, Poisson, logit, varying slopes and AKM
worker–firm designs, n = 10⁴–10⁷, at 16 threads and, up to 10⁶, 1 thread: 273 rows. In every
row the Stata commands are faster than every competitor that finished (reghdfe family, xhdfe,
fixest, pyfixest), and the Python and R packages are faster than pyfixest and fixest.
Machine: Ryzen 9 9950X (16 cores), Stata 19.5 MP, reghdfe 6.13.1, ivreghdfe 1.1.4,
ppmlhdfe 2.3.3, xhdfe 2.28.1, fixest 0.14.2, pyfixest 0.60.0. Full tables:
[`bench/RESULTS.md`](bench/RESULTS.md); method: [`bench/ENVIRONMENT.md`](bench/ENVIRONMENT.md);
rerun with `cd bench && uv run python run.py --grid gate`.

## Stata

```stata
net install fereg, from("https://leoliu0.github.io/fereg/stata") replace

fereg y x1 x2, absorb(firm year) cluster(firm)
fereg y x1 (x2 = z), absorb(firm year) vce(robust)
feppml trade dist, absorb(exporter#year importer#year) cluster(pair)
felogit default x1, absorb(firm year)
```

Binaries are provided for Linux (x86-64), macOS (Apple Silicon and Intel) and Windows (x86-64).
Variances follow reghdfe and ivreghdfe: `vce(unadjusted)`, `vce(robust)`, `vce(cluster ...)`
(multiway), reghdfe's Driscoll–Kraay `vce(dkraay #)`, and ivreghdfe's HAC options `bw(#|auto)`,
`kernel()`, `kiefer`, `dkraay()`, `psd0`/`psda` (Newey–West and the other kernels, within panels on
`xtset` data). fereg extensions: `vce(hc2)`/`vce(hc3)` (Stata `regress` formulas, exact leverage
with absorbed fixed effects), `vce(crv3 clustvars)` and `vce(conley latvar lonvar, cutoff(#))`.
See `help fereg`, `help feppml` and `help felogit`.

## Python

```sh
pip install fereg
```

Wheels are provided for CPython 3.10–3.14 on Linux (x86-64, aarch64), macOS (Apple Silicon,
Intel) and Windows (x86-64). Other platforms build from the source distribution, which needs
Rust.

```python
import fereg

m = fereg.feols("y ~ x1 + x2 | firm + year", data=df, vcov={"CRV1": "firm"})
iv = fereg.feols("y ~ x1 | firm + year | x2 ~ z", data=df)
p = fereg.fepois("count ~ x1 | firm + year", data=df, vcov="hetero")
m.summary()
fereg.etable([m, iv, p], type="md")
m.fixef()
```

Data can be pandas or polars. The result classes (`Feols`, `Feiv`, `Fepois`, `Felogit`,
`FixestMulti`) have pyfixest's methods: `summary`, `tidy`, `coef`, `se`, `tstat`, `pvalue`,
`confint`, `vcov` (post hoc), `predict`, `resid` and `fixef`; `fereg.etable` and
`fereg.summary` take lists of models. Variances as in pyfixest: `"iid"`, `"hetero"`/`"HC1"`,
`"HC2"`, `"HC3"`, `{"CRV1": "a+b"}`, `{"CRV3": "a+b"}` (up to four ways), `"NW"` and `"DK"`
with `vcov_kwargs={"time_id": ..., "panel_id": ..., "lag": ...}`, at estimation or post hoc.
fereg extensions: HC2/HC3 with fixed effects, CRV3 for IV and logit, Newey–West /
Driscoll–Kraay for IV, and fixest's Conley estimator (`vcov="conley"`,
`vcov_kwargs={"lat": ..., "lon": ..., "cutoff": "100km"}`). Not implemented: probit; `nid`
variances (`NotImplementedError`); pyfixest's wild bootstrap, `ccv`, `decompose`, `ritest`,
plotting and DiD helpers.

## R

```r
# Linux, or any R >= 4.2 with Rust (cargo, rustc >= 1.85) installed:
install.packages("https://github.com/leoliu0/fereg/releases/download/v0.4.0/fereg_0.4.0.tar.gz",
                 repos = NULL, type = "source")
```

Binaries for R 4.6 on macOS (`fereg_0.4.0_R4.6_macos-arm64.tgz`,
`fereg_0.4.0_R4.6_macos-x86_64.tgz`) and Windows (`fereg_0.4.0.zip`) are on the
[release page](https://github.com/leoliu0/fereg/releases/tag/v0.4.0); download one and run
`install.packages("<path to file>", repos = NULL, type = "binary")`.

```r
library(fereg)

m  <- feols(y ~ x1 + x2 | firm + year, df, cluster = ~firm)
iv <- feols(y ~ x1 | firm + year | x2 ~ z, df)
p  <- fepois(count ~ x1 | firm + year, df, vcov = "hetero")
summary(m)
etable(m, iv, p)
fixef(m)
```

The formula grammar is fixest's (`fe1^fe2`, `fe[x]`, `fe[[x]]`, `i()`, `factor()`, `c(y1, y2)`,
`sw()`/`csw()`), and so are the methods: `summary`, `coef`, `vcov`, `se`, `tstat`, `pvalue`,
`coeftable`, `confint`, `predict`, `resid`, `fitted`, `fixef`, `fitstat`, `r2`, `etable`,
`tidy`/`glance`. Variances follow fixest's syntax, labels and `ssc()`, at estimation and post
hoc: `"iid"`, `"hetero"`, `"hc2"`/`"hc3"` (exact leverage with fixed effects; not for IV),
clustering (`~firm`, `~firm + year`, `"twoway"`), Newey–West (`NW(2) ~ firm + year`,
`NW ~ year` for a time series, or `"NW"` with `panel.id = ~firm + year`), Driscoll–Kraay
(`DK(2) ~ year`), Conley (`conley(100) ~ lat + lon`, `conley("60mi")`) and fixest's request
objects (`NW()`, `vcov_NW()`, `vcov_conley()`, `vcov_hetero(type = "hc3")`, ...). CRV3
(`vcov = crv3 ~ firm`, `vcov_crv3()`) is a fereg extension with fixest's cluster correction.
Not implemented: `fixef.rm = "infinite_coef"`, panel operators `l()`/`f()`/`d()`, `mvsw()`, the
`.[]`/`..ctrl` formula macros, and variances given as functions or matrices. `fixef.tol` and
`fixef.iter` are the settings of fereg's solver (defaults 1e-8 and 16000), which stops on a
different criterion than fixest's (defaults 1e-6 and 10000).

## Build from source

Requires Rust (stable) and a C compiler.

```sh
cargo test --workspace --exclude fereg-py --release   # the Rust core and the Stata plugin crate
scripts/build_plugin.sh        # builds the Stata plugin and copies it to stata/
scripts/install_local.sh       # installs the ado files and plugin into your PERSONAL ado directory
cd python && maturin develop --release                # the Python package (needs maturin)
sh scripts/sync-core.sh && R CMD INSTALL r/fereg      # the R package
```

Release steps: [`PUBLISHING.md`](PUBLISHING.md).

## License

MIT
