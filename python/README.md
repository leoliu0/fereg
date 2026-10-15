# fereg (Python)

Fast high-dimensional fixed-effects regressions with pyfixest-compatible syntax and results.

`fereg.feols`, `fereg.fepois` and `fereg.feglm` take pyfixest 0.60.0's formulas, arguments and
defaults and return results with pyfixest's classes, names and methods, so estimates, standard
errors and statistics match pyfixest's. The estimation runs in a parallel Rust core that
eliminates the largest fixed effect exactly and solves the others by preconditioned conjugate
gradient: at 10 million observations and 16 threads it is 5–100× faster than pyfixest and
2–200× faster than R's fixest on the fixest benchmark designs.

```python
import fereg

m = fereg.feols("y ~ x1 + x2 | firm + year", data=df, vcov={"CRV1": "firm"})
iv = fereg.feols("y ~ x1 | firm + year | x2 ~ z", data=df)
p = fereg.fepois("count ~ x1 | firm + year", data=df, vcov="hetero")
lg = fereg.feglm("default ~ x1 | firm", data=df, family="logit")
m.summary()
fereg.etable([m, iv, p], type="md")
m.fixef()
```

Supported: OLS, IV/2SLS (several endogenous regressors), Poisson, logit; any number of fixed
effects, `fe1^fe2` interactions and varying slopes `fe[x]`; weights; `i()`, `C()`, multiple
estimation (`sw`, `csw`, several left-hand sides, `split`/`fsplit`); `predict`, `resid`,
`fixef`, `tidy`, `etable`. Data can be pandas or polars.

Variances (`vcov=` at estimation, or post hoc with `m.vcov(...)`), as in pyfixest with `ssc()`:
`"iid"`, `"hetero"`/`"HC1"`, `"HC2"`, `"HC3"`; `{"CRV1": "a+b"}` and `{"CRV3": "a+b"}` (up to
four-way); Newey–West `"NW"` and Driscoll–Kraay `"DK"` with
`vcov_kwargs={"time_id": "year", "panel_id": "firm", "lag": 2}`. fereg extensions (pyfixest
raises): HC2/HC3 with fixed effects, CRV3 for IV and logit (logit refits), Newey–West /
Driscoll–Kraay for IV and Driscoll–Kraay without `panel_id`, and fixest's Conley spatial HAC:
`vcov="conley", vcov_kwargs={"lat": "lat", "lon": "lon", "cutoff": "100km"}` (`cutoff` in km
or `"60mi"`, default deduced as fixest; `pixel`, `distance="triangular"|"spherical"`).

Not implemented: probit; `nid` variances (raise `NotImplementedError`); wild bootstrap,
`ccv`, `decompose`, `ritest`, plotting and DiD helpers.

Formulas can use functions and objects from the caller through pyfixest's `context=`
(`context=0`: the calling frame; an int `k`: `k` frames further up; a mapping: the names it
gives; default `None`: none), e.g. `fereg.feols("y ~ f(x) | firm", df, context={"f": f})`.
Separation checks as in pyfixest (`separation_check=["fe"]` by default, `"ir"`; every GLM
family, models with fixed effects only), with its warning "N observations removed because of
separation.". Deliberate differences from pyfixest 0.60.0: gaussian `feglm` iid variances are
scaled by the residual variance (pyfixest's `Feglm._vcov_iid` fixes the dispersion at 1), and
the `"ir"` check also works after rows were dropped before it (pyfixest's misaligns its
predictions then and drops nothing).

`nthreads` (default: all cores) sets the number of threads. Results do not depend on it.

The same core powers the Stata commands `fereg`/`feppml`/`felogit` and the R package `fereg`:
see <https://github.com/leoliu0/fereg>.
