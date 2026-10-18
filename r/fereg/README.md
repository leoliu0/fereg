# fereg (R)

Fast high-dimensional fixed-effects regressions with the fixest interface.

`feols()`, `fepois()` and `feglm()` take fixest 0.14.2's formulas, arguments and defaults and
return objects with fixest's methods, so estimates, standard errors and statistics match
fixest's. The estimation runs in a parallel Rust library that eliminates the largest fixed
effect exactly and solves the others by preconditioned conjugate gradient: at 10 million
observations and 16 threads it is 2.6–200× faster than fixest on the fixest benchmark designs.

```r
library(fereg)

m  <- feols(y ~ x1 + x2 | firm + year, df, cluster = ~firm)
iv <- feols(y ~ x1 | firm + year | x2 ~ z, df)
p  <- fepois(count ~ x1 | firm + year, df, vcov = "hetero")
lg <- feglm(default ~ x1 | firm, df, family = "logit")
summary(m)
etable(m, iv, p)
fixef(m)
```

Supported: OLS, IV/2SLS (several endogenous regressors), Poisson, logit; any number of fixed
effects, `fe1^fe2` interactions and varying slopes (`fe[x]`, `fe[[x]]`); weights and offsets;
`i()`, `factor()`, multiple estimation (`c(y1, y2)`, `sw()`, `csw()`, `split`, `fsplit`);
`predict` (with `newdata`), `resid`, `fitted`, `fixef`, `fitstat`, `etable`, `tidy`/`glance`.

Variances, at estimation and post hoc (`summary()`, `vcov()`, `confint()`, `etable()`), with
fixest's syntax, labels and `ssc()`: iid, heteroskedasticity-robust (HC1) and HC2/HC3 (exact
leverage, fixed effects included; not for IV), one- to four-way clustering, Newey–West
(`NW ~ id + period`, `NW(2) ~ id + period`, `NW ~ period` for a time series, or `"NW"` with
`panel.id = ~id + period`), Driscoll–Kraay (`DK(2) ~ period`), Conley (`conley(100) ~ lat + lon`,
`conley("60mi")`, pixel and spherical distance, fixest's deduced cutoff), and fixest's request
objects (`NW()`, `DK()`, `conley()`, `vcov_NW()`, `vcov_DK()`, `vcov_conley()`, `vcov_hetero()`,
`vcov_cluster()`). CRV3 (`vcov = crv3 ~ g`, `vcov_crv3()`) is a fereg extension (fixest has no
CRV3): the leave-one-cluster-out jackknife with fixest's cluster small-sample correction;
after estimation it re-estimates the model.

Fixed-effects logit: incidental-parameter bias correction and average partial effects with
alpaca 0.3.5's interface. `biasCorr(fit, L = 0, panel.structure = "classic")` applies the
analytical correction (Fernández-Val–Weidner; network version of Hinz–Stammann–Wanner with
`panel.structure = "network"`; bandwidth `L` for weakly exogenous regressors; `bias =
"individual"`/`"time"` corrects one bias only) or, with `method = "jackknife"`, logitfe's
split-panel and delete-one jackknives (`jackknife = "ss2"`, `"ss1"`, `"js"`, `"sj"`, `"jj"`,
`"double"`; `multiple = R` random partitions). The result is a fereg fit at the corrected
coefficients, with the fixed effects re-estimated there, so `summary()` (any variance but
CRV3), `etable()`, `fixef()` and `predict()` work as usual. `getAPEs(fit, n.pop = NULL)` gives
the average partial effects (uncorrected, or corrected for a `biasCorr()` result) with
delta-method standard errors and alpaca's finite-population term. Differences from alpaca
(exact maximum likelihood and fixed-effect solves, the finite-population term over all
observations when groups with a constant outcome are removed) are listed in `?biasCorr` and
`?getAPEs`.

Not implemented: `fixef.rm = "infinite_coef"`, the panel operators `l()`/`f()`/`d()`,
`mvsw()`, the `.[]`/`..ctrl` formula macros, and variances given as functions or matrices.

Installing from source needs Rust (cargo, rustc >= 1.85); the source tarball on the GitHub
release page bundles every Rust dependency, so the build needs no network.

The same core powers the Stata commands `fereg`/`feppml`/`felogit` and the Python package
`fereg`: see <https://github.com/leoliu0/fereg>.
