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
iid, hetero and up to four-way clustered variances (`ssc()` as in fixest), also post hoc in
`summary()`; `i()`, `factor()`, multiple estimation (`c(y1, y2)`, `sw()`, `csw()`, `split`,
`fsplit`); `predict` (with `newdata`), `resid`, `fitted`, `fixef`, `fitstat`, `etable`,
`tidy`/`glance`. Not implemented: HC2/HC3, Newey–West, Driscoll–Kraay and Conley variances,
`fixef.rm = "infinite_coef"`, the panel operators `l()`/`f()`/`d()`, `mvsw()` and the
`.[]`/`..ctrl` formula macros.

Installing from source needs Rust (cargo, rustc >= 1.85); the source tarball on the GitHub
release page bundles every Rust dependency, so the build needs no network.

The same core powers the Stata commands `fereg`/`feppml`/`felogit` and the Python package
`fereg`: see <https://github.com/leoliu0/fereg>.
