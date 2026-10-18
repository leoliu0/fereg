//! `fereg._core`: the compiled part of the Python package (bindings of `fereg_core::fit`).
//!
//! Input columns are borrowed zero-copy from contiguous numpy arrays; the GIL is released for
//! the whole estimation. Ctrl-C is polled on the orchestrating thread through
//! `Python::check_signals`, which briefly re-attaches to the interpreter.

use fereg_core::api::{Ctx, RfError, Separation};
use fereg_core::fit as cf;
use fereg_core::fit::{
    CoefSource, FactorCol, FeTerm, FitInput, FitSpec, FixefNorm, FixefRm, Flavor, GDf, KFixef, Model,
    Ssc, TDf, TestStat, VcovKind, VcovOutput, WeightsType,
};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyKeyboardInterrupt, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

fn to_pyerr(e: RfError) -> PyErr {
    match e {
        RfError::InvalidInput(_) | RfError::TooFewObs => PyValueError::new_err(e.to_string()),
        RfError::Interrupted => PyKeyboardInterrupt::new_err(()),
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

fn slice<'a, T: numpy::Element>(a: &'a PyReadonlyArray1<'_, T>, what: &str) -> PyResult<&'a [T]> {
    a.as_slice().map_err(|_| PyValueError::new_err(format!("{what}: array must be C-contiguous")))
}

fn slices<'a>(arrs: &'a [PyReadonlyArray1<'_, f64>], what: &str) -> PyResult<Vec<&'a [f64]>> {
    arrs.iter().map(|a| slice(a, what)).collect()
}

/// A borrowed factor column as sent by the host: `(kind, values, nlevels, base)`.
enum FactorArr<'py> {
    F64(PyReadonlyArray1<'py, f64>),
    I32(PyReadonlyArray1<'py, i32>),
    I64(PyReadonlyArray1<'py, i64>),
    Codes(PyReadonlyArray1<'py, i32>, u32, i32),
}

impl<'py> FactorArr<'py> {
    fn extract(t: &Bound<'py, PyAny>) -> PyResult<Self> {
        let (kind, values, nlevels, base): (String, Bound<'py, PyAny>, u32, i32) = t.extract()?;
        Ok(match kind.as_str() {
            "f64" => FactorArr::F64(values.extract()?),
            "i32" => FactorArr::I32(values.extract()?),
            "i64" => FactorArr::I64(values.extract()?),
            "codes" => FactorArr::Codes(values.extract()?, nlevels, base),
            other => return Err(PyValueError::new_err(format!("unknown factor kind '{other}'"))),
        })
    }

    fn col(&self) -> PyResult<FactorCol<'_>> {
        Ok(match self {
            FactorArr::F64(a) => FactorCol::F64(slice(a, "factor")?),
            FactorArr::I32(a) => FactorCol::I32(slice(a, "factor")?),
            FactorArr::I64(a) => FactorCol::I64(slice(a, "factor")?),
            FactorArr::Codes(a, nlevels, base) => {
                FactorCol::Codes { codes: slice(a, "factor")?, nlevels: *nlevels, base: *base }
            }
        })
    }
}

fn extract_factors<'py>(factors: &Bound<'py, PyList>) -> PyResult<Vec<FactorArr<'py>>> {
    factors.iter().map(|t| FactorArr::extract(&t)).collect()
}

fn factor_cols<'a>(arrs: &'a [FactorArr<'_>]) -> PyResult<Vec<FactorCol<'a>>> {
    arrs.iter().map(FactorArr::col).collect()
}

/// A variance request: `kind` (`"iid"`, `"hetero"` = HC1, `"hc2"`, `"hc3"`, `"cluster"` = CRV1,
/// `"crv3"`, `"nw"`, `"dk"`, `"conley"`) with its options: `lag` (Newey-West / Driscoll-Kraay;
/// `None` = the flavor's default rule), `cutoff` (Conley, km; `None` = fixest's deduced cutoff),
/// `pixel` (Conley, km; 0 = no pixel aggregation) and `spherical` (Conley great-circle distance;
/// `false` = fixest's "triangular" distance). Options of other kinds are ignored.
fn parse_vcov_kind(s: &str, lag: Option<usize>, cutoff: Option<f64>, pixel: f64, spherical: bool) -> PyResult<VcovKind> {
    match s {
        "iid" => Ok(VcovKind::Iid),
        "hetero" => Ok(VcovKind::Hetero),
        "hc2" => Ok(VcovKind::Hc2),
        "hc3" => Ok(VcovKind::Hc3),
        "cluster" => Ok(VcovKind::Cluster),
        "crv3" => Ok(VcovKind::Crv3),
        "nw" => Ok(VcovKind::NeweyWest { lag }),
        "dk" => Ok(VcovKind::DriscollKraay { lag }),
        "conley" => Ok(VcovKind::Conley { cutoff, pixel, spherical }),
        _ => Err(PyValueError::new_err(format!("unknown vcov kind '{s}'"))),
    }
}

/// Reads an `ssc` dict (keys as pyfixest/fixest `ssc()`: `k_adj`, `k_fixef`, `k_exact`, `G_adj`,
/// `G_df`, `t_df`); absent keys keep fixest's defaults.
fn parse_ssc(d: Option<&Bound<'_, PyDict>>) -> PyResult<Ssc> {
    let mut s = Ssc::default();
    let Some(d) = d else { return Ok(s) };
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        match key.as_str() {
            "k_adj" => s.k_adj = v.extract()?,
            "k_exact" => s.k_exact = v.extract()?,
            "G_adj" => s.g_adj = v.extract()?,
            "k_fixef" => {
                s.k_fixef = match v.extract::<String>()?.as_str() {
                    "none" => KFixef::None,
                    "full" => KFixef::Full,
                    "nonnested" => KFixef::Nonnested,
                    o => return Err(PyValueError::new_err(format!("ssc: unknown k_fixef '{o}'"))),
                }
            }
            "G_df" => {
                s.g_df = match v.extract::<String>()?.as_str() {
                    "min" => GDf::Min,
                    "conventional" => GDf::Conventional,
                    o => return Err(PyValueError::new_err(format!("ssc: unknown G_df '{o}'"))),
                }
            }
            "t_df" => {
                s.t_df = if let Ok(x) = v.extract::<f64>() {
                    TDf::Fixed(x)
                } else {
                    match v.extract::<String>()?.as_str() {
                        "min" => TDf::Min,
                        "conventional" => TDf::Conventional,
                        o => return Err(PyValueError::new_err(format!("ssc: unknown t_df '{o}'"))),
                    }
                }
            }
            o => return Err(PyValueError::new_err(format!("ssc: unknown key '{o}'"))),
        }
    }
    Ok(s)
}

fn kind_name(k: VcovKind) -> &'static str {
    match k {
        VcovKind::Iid => "iid",
        VcovKind::Hetero => "hetero",
        VcovKind::Cluster => "cluster",
        VcovKind::Hc2 => "hc2",
        VcovKind::Hc3 => "hc3",
        VcovKind::Crv3 => "crv3",
        VcovKind::NeweyWest { .. } => "nw",
        VcovKind::DriscollKraay { .. } => "dk",
        VcovKind::Conley { .. } => "conley",
    }
}

fn source_tuple<'py>(py: Python<'py>, s: &CoefSource) -> PyResult<Bound<'py, PyTuple>> {
    let (kind, j): (&str, i64) = match *s {
        CoefSource::Intercept => ("intercept", -1),
        CoefSource::X(j) => ("x", j as i64),
        CoefSource::Endog(j) => ("endog", j as i64),
        CoefSource::Inst(j) => ("inst", j as i64),
    };
    PyTuple::new(py, [kind.into_pyobject(py)?.into_any(), j.into_pyobject(py)?.into_any()])
}

fn sources<'py>(py: Python<'py>, s: &[CoefSource]) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, s.iter().map(|c| source_tuple(py, c)).collect::<PyResult<Vec<_>>>()?)
}

fn test_dict<'py>(py: Python<'py>, t: &TestStat) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("stat", t.stat)?;
    d.set_item("p", t.p)?;
    d.set_item("df1", t.df1)?;
    d.set_item("df2", t.df2)?;
    Ok(d)
}

/// `{kind, vcov, df_k, df_t, n_clusters, n_periods, lag, cutoff, ssc_factor, psd_fixed}`: `lag`
/// (Newey-West / Driscoll-Kraay) and `cutoff` (Conley, km) are the values used, defaults resolved.
fn vcov_dict<'py>(py: Python<'py>, v: VcovOutput) -> PyResult<Bound<'py, PyDict>> {
    let k = (v.vcov.len() as f64).sqrt().round() as usize;
    let (lag, cutoff) = match v.kind {
        VcovKind::NeweyWest { lag } | VcovKind::DriscollKraay { lag } => (lag, None),
        VcovKind::Conley { cutoff, .. } => (None, cutoff),
        _ => (None, None),
    };
    let d = PyDict::new(py);
    d.set_item("kind", kind_name(v.kind))?;
    d.set_item("vcov", v.vcov.into_pyarray(py).reshape([k, k])?)?;
    d.set_item("df_k", v.df_k)?;
    d.set_item("df_t", v.df_t)?;
    d.set_item("n_clusters", v.n_clusters)?;
    d.set_item("n_periods", v.n_periods)?;
    d.set_item("lag", lag)?;
    d.set_item("cutoff", cutoff)?;
    d.set_item("ssc_factor", v.ssc_factor)?;
    d.set_item("psd_fixed", v.psd_fixed)?;
    Ok(d)
}

fn opt_array(py: Python<'_>, v: Option<Vec<f64>>) -> Py<PyAny> {
    match v {
        Some(v) => v.into_pyarray(py).into_any().unbind(),
        None => py.None(),
    }
}

fn parse_fixef_norm(s: &str) -> PyResult<FixefNorm> {
    match s {
        "fixest" => Ok(FixefNorm::Fixest),
        "minnorm" => Ok(FixefNorm::MinNorm),
        _ => Err(PyValueError::new_err(format!("unknown fixef_norm '{s}'"))),
    }
}

/// `FixefValues` as dicts `{term, slope, values, first_obs, references}` (arrays moved, not copied).
fn fixef_list(py: Python<'_>, values: Vec<cf::FixefValues>) -> PyResult<Bound<'_, PyList>> {
    let out = PyList::empty(py);
    for f in values {
        let d = PyDict::new(py);
        d.set_item("term", f.term)?;
        d.set_item("slope", f.slope)?;
        d.set_item("values", f.values.into_pyarray(py))?;
        d.set_item("first_obs", f.first_obs.into_pyarray(py))?;
        d.set_item("references", f.references)?;
        out.append(d)?;
    }
    Ok(out)
}

/// Builds the core context: `threads` workers (0 = all CPUs), Ctrl-C polling and, when
/// `verbose > 0`, log lines written to `sys.stdout`. Must be called inside `py.detach`.
fn with_ctx<T>(threads: usize, verbose: u8, f: impl FnOnce(&Ctx) -> T) -> T {
    // faer calls made outside the core's scoped pools must not start rayon's global pool (all
    // cores, threads outliving the call): every parallel faer call in the core passes its pool.
    faer::set_global_parallelism(faer::Par::Seq);
    let interrupt = || Python::attach(|py| py.check_signals().is_err());
    let log = |line: &str| {
        Python::attach(|py| {
            if let Ok(out) = py.import("sys").and_then(|s| s.getattr("stdout")) {
                let _ = out.call_method1("write", (format!("{line}\n"),));
            }
        })
    };
    let ctx = Ctx { threads, verbose, interrupt: &interrupt, log: &log };
    f(&ctx)
}

/// Retained data for post-hoc variance matrices (`Feols.vcov(...)`) and lazily computed
/// fixed-effect values (`Feols.fixef()`, `predict(newdata)`).
#[pyclass(frozen, module = "fereg._core")]
struct FitState {
    inner: cf::FitState,
}

#[pymethods]
impl FitState {
    /// Variance matrix of the same fit under another estimator / small-sample correction.
    /// `kind`, `lag`, `cutoff`, `pixel`, `spherical`: see `fit`. `factors` are
    /// `(kind, values, nlevels, base)` tuples over the original input rows, referenced by
    /// `cluster` (dimensions; more than one factor = interaction), `time` and `panel` (factor
    /// indices; time levels ascending = time order); `lat` / `lon` are degrees over the original
    /// input rows. A missing value on an estimation row is an error. `"crv3"` is an error: it
    /// re-estimates the model, so hosts fit again with `vcov="crv3"`.
    #[pyo3(signature = (
        kind, ssc=None, factors=None, factor_names=None, cluster=None, *, time=None, panel=None,
        lat=None, lon=None, lag=None, cutoff=None, pixel=0.0, spherical=false, threads=0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn vcov<'py>(
        &self,
        py: Python<'py>,
        kind: &str,
        ssc: Option<&Bound<'py, PyDict>>,
        factors: Option<&Bound<'py, PyList>>,
        factor_names: Option<Vec<String>>,
        cluster: Option<Vec<Vec<usize>>>,
        time: Option<usize>,
        panel: Option<usize>,
        lat: Option<PyReadonlyArray1<'py, f64>>,
        lon: Option<PyReadonlyArray1<'py, f64>>,
        lag: Option<usize>,
        cutoff: Option<f64>,
        pixel: f64,
        spherical: bool,
        threads: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let kind = parse_vcov_kind(kind, lag, cutoff, pixel, spherical)?;
        let ssc = parse_ssc(ssc)?;
        let arrs = match factors {
            Some(f) => extract_factors(f)?,
            None => Vec::new(),
        };
        let cols = factor_cols(&arrs)?;
        let names = factor_names.unwrap_or_default();
        let cluster = cluster.unwrap_or_default();
        let data = cf::VcovData {
            factors: &cols,
            factor_names: &names,
            cluster: &cluster,
            time,
            panel,
            lat: lat.as_ref().map(|c| slice(c, "lat")).transpose()?,
            lon: lon.as_ref().map(|c| slice(c, "lon")).transpose()?,
        };
        let out = py.detach(|| with_ctx(threads, 0, |ctx| self.inner.vcov(kind, &ssc, &data, ctx))).map_err(to_pyerr)?;
        vcov_dict(py, out)
    }

    /// Fixed-effect values under `norm` ("minnorm" | "fixest"), as `FitResult.fixef`.
    #[pyo3(signature = (norm="minnorm", threads=0))]
    fn fixef<'py>(&self, py: Python<'py>, norm: &str, threads: usize) -> PyResult<Bound<'py, PyList>> {
        let norm = parse_fixef_norm(norm)?;
        let values = py.detach(|| with_ctx(threads, 0, |ctx| self.inner.fixef(norm, ctx))).map_err(to_pyerr)?;
        fixef_list(py, values)
    }

    /// Estimation rows.
    #[getter]
    fn nobs(&self) -> usize {
        self.inner.nobs()
    }
}

/// Every field of `fereg_core::fit::FitOutput`: vectors as numpy arrays, nested structs as dicts.
#[pyclass(frozen, get_all, module = "fereg._core")]
struct FitResult {
    nobs: usize,
    nobs_origin: usize,
    sumw: f64,
    obs: Py<PyAny>,
    dropped: Py<PyDict>,
    coef_source: Py<PyList>,
    coef: Py<PyAny>,
    collinear: Py<PyList>,
    vcov: Py<PyDict>,
    nparams: usize,
    fe_sizes: Vec<usize>,
    stats: Py<PyDict>,
    iv: Py<PyAny>,
    resid: Py<PyAny>,
    fitted: Py<PyAny>,
    eta: Py<PyAny>,
    irls_weights: Py<PyAny>,
    fixef: Py<PyList>,
    notes: Vec<String>,
    state: Py<PyAny>,
}

fn convert(py: Python<'_>, o: cf::FitOutput) -> PyResult<FitResult> {
    let dr = &o.dropped;
    let dropped = PyDict::new(py);
    dropped.set_item("na", dr.na)?;
    dropped.set_item("zero_weight", dr.zero_weight)?;
    dropped.set_item("singletons", dr.singletons)?;
    dropped.set_item("perfect_fit", dr.perfect_fit)?;
    dropped.set_item("separated", dr.separated)?;
    let na = PyDict::new(py);
    let nb = &dr.na_by;
    for (k, v) in [
        ("y", nb.y),
        ("x", nb.x),
        ("endog", nb.endog),
        ("inst", nb.inst),
        ("fe", nb.fe),
        ("cluster", nb.cluster),
        ("weights", nb.weights),
        ("offset", nb.offset),
        ("extra", nb.extra),
    ] {
        na.set_item(k, v)?;
    }
    dropped.set_item("na_by", na)?;

    let s = &o.stats;
    let stats = PyDict::new(py);
    for (k, v) in [
        ("ssr", s.ssr),
        ("ssr_null", s.ssr_null),
        ("ssr_fe_only", s.ssr_fe_only),
        ("sigma2", s.sigma2),
        ("rmse", s.rmse),
        ("r2", s.r2),
        ("ar2", s.ar2),
        ("wr2", s.wr2),
        ("war2", s.war2),
        ("loglik", s.loglik),
        ("ll_null", s.ll_null),
        ("deviance", s.deviance),
        ("null_deviance", s.null_deviance),
        ("pseudo_r2", s.pseudo_r2),
        ("apr2", s.apr2),
        ("sq_cor", s.sq_cor),
    ] {
        stats.set_item(k, v)?;
    }
    stats.set_item("f_test", test_dict(py, &s.f_test)?)?;
    stats.set_item("fe_iterations", s.fe_iterations)?;
    stats.set_item("irls_iterations", s.irls_iterations)?;
    stats.set_item("converged", s.converged)?;

    let iv = match o.iv {
        None => py.None(),
        Some(iv) => {
            let d = PyDict::new(py);
            let stages = PyList::empty(py);
            for st in iv.first_stage {
                let sd = PyDict::new(py);
                sd.set_item("endog", st.endog)?;
                sd.set_item("coef_source", sources(py, &st.coef_source)?)?;
                sd.set_item("coef", st.coef.into_pyarray(py))?;
                sd.set_item("vcov", vcov_dict(py, st.vcov)?)?;
                sd.set_item("collinear", sources(py, &st.collinear)?)?;
                sd.set_item("f_test", test_dict(py, &st.f_test)?)?;
                sd.set_item("wald", test_dict(py, &st.wald)?)?;
                stages.append(sd)?;
            }
            d.set_item("first_stage", stages)?;
            d.set_item("wu_hausman", test_dict(py, &iv.wu_hausman)?)?;
            d.set_item("sargan", test_dict(py, &iv.sargan)?)?;
            d.set_item("ssr_iv", iv.ssr_iv)?;
            d.into_any().unbind()
        }
    };

    let fixef = fixef_list(py, o.fixef)?;

    let state = match o.state {
        Some(st) => Py::new(py, FitState { inner: st })?.into_any(),
        None => py.None(),
    };

    Ok(FitResult {
        nobs: o.nobs,
        nobs_origin: o.nobs_origin,
        sumw: o.sumw,
        obs: o.obs.into_pyarray(py).into_any().unbind(),
        dropped: dropped.unbind(),
        coef_source: sources(py, &o.coef_source)?.unbind(),
        coef: o.coef.into_pyarray(py).into_any().unbind(),
        collinear: sources(py, &o.collinear)?.unbind(),
        vcov: vcov_dict(py, o.vcov)?.unbind(),
        nparams: o.nparams,
        fe_sizes: o.fe_sizes,
        stats: stats.unbind(),
        iv,
        resid: opt_array(py, o.resid),
        fitted: opt_array(py, o.fitted),
        eta: opt_array(py, o.eta),
        irls_weights: opt_array(py, o.irls_weights),
        fixef: fixef.unbind(),
        notes: o.notes,
        state,
    })
}

fn parse_spec(
    model: &str,
    flavor: &str,
    vcov: VcovKind,
    weights_type: &str,
    fixef_rm: &str,
    fixef_norm: &str,
) -> PyResult<FitSpec> {
    let bad = |what: &str, v: &str| PyValueError::new_err(format!("unknown {what} '{v}'"));
    Ok(FitSpec {
        model: match model {
            "ols" => Model::Ols,
            "poisson" => Model::Poisson,
            "logit" => Model::Logit,
            _ => return Err(bad("model", model)),
        },
        flavor: match flavor {
            "fixest" => Flavor::Fixest,
            "pyfixest" => Flavor::Pyfixest,
            _ => return Err(bad("flavor", flavor)),
        },
        vcov,
        weights_type: match weights_type {
            "aweights" => WeightsType::Aweights,
            "fweights" => WeightsType::Fweights,
            _ => return Err(bad("weights_type", weights_type)),
        },
        fixef_rm: match fixef_rm {
            "none" => FixefRm::None,
            "singleton" => FixefRm::Singleton,
            "perfect" => FixefRm::Perfect,
            "perfect_only" => FixefRm::PerfectOnly,
            _ => return Err(bad("fixef_rm", fixef_rm)),
        },
        fixef_norm: parse_fixef_norm(fixef_norm)?,
        ..FitSpec::default()
    })
}

/// Defines a `#[pyfunction]` taking the model arguments of `fit` (same names, order and defaults)
/// plus `extra` keyword-only arguments; `body` runs with the validated `FitInput` / `FitSpec`, the
/// requested `threads` and `verbose`. The bias-correction entry points take the very model `fit`
/// estimates, so their arguments cannot drift apart.
macro_rules! model_fn {
    (
        $(#[$meta:meta])*
        fn $name:ident($($extra:ident: $ty:ty $(= $default:expr)?),*) -> $ret:ty;
        |$py:ident, $input:ident, $spec:ident, $threads:ident, $verbose:ident| $body:block
    ) => {
        $(#[$meta])*
        #[pyfunction]
        #[pyo3(signature = (
            y, x, endog, inst, factors, factor_names, fe, slopes, weights=None, offset=None,
            cluster=Vec::new(), drop=None, *, time=None, panel=None, lat=None, lon=None, model="ols",
            flavor="pyfixest", intercept=true, vcov="iid", lag=None, cutoff=None, pixel=0.0,
            spherical=false, ssc=None, weights_type="aweights", fixef_rm="singleton", relu=false, mu=false,
            gaussian_glm=false,
            collin_tol=1e-9, fixef_tol=1e-8, fixef_maxiter=16000, glm_tol=1e-8, glm_maxiter=25,
            keep_resid=true, keep_fixef=true, fixef_norm="minnorm", keep_state=true, threads=0, verbose=0
            $(, $extra $(= $default)?)*
        ))]
        #[allow(clippy::too_many_arguments)]
        fn $name<'py>(
            $py: Python<'py>,
            y: PyReadonlyArray1<'py, f64>,
            x: Vec<PyReadonlyArray1<'py, f64>>,
            endog: Vec<PyReadonlyArray1<'py, f64>>,
            inst: Vec<PyReadonlyArray1<'py, f64>>,
            factors: &Bound<'py, PyList>,
            factor_names: Vec<String>,
            fe: Vec<(Vec<usize>, bool, Vec<usize>)>,
            slopes: Vec<PyReadonlyArray1<'py, f64>>,
            weights: Option<PyReadonlyArray1<'py, f64>>,
            offset: Option<PyReadonlyArray1<'py, f64>>,
            cluster: Vec<Vec<usize>>,
            drop: Option<PyReadonlyArray1<'py, bool>>,
            time: Option<usize>,
            panel: Option<usize>,
            lat: Option<PyReadonlyArray1<'py, f64>>,
            lon: Option<PyReadonlyArray1<'py, f64>>,
            model: &str,
            flavor: &str,
            intercept: bool,
            vcov: &str,
            lag: Option<usize>,
            cutoff: Option<f64>,
            pixel: f64,
            spherical: bool,
            ssc: Option<&Bound<'py, PyDict>>,
            weights_type: &str,
            fixef_rm: &str,
            relu: bool,
            mu: bool,
            gaussian_glm: bool,
            collin_tol: f64,
            fixef_tol: f64,
            fixef_maxiter: usize,
            glm_tol: f64,
            glm_maxiter: usize,
            keep_resid: bool,
            keep_fixef: bool,
            fixef_norm: &str,
            keep_state: bool,
            threads: usize,
            verbose: u8,
            $($extra: $ty),*
        ) -> PyResult<$ret> {
            let spec = FitSpec {
                intercept,
                ssc: parse_ssc(ssc)?,
                separation: Separation { relu, mu },
                gaussian_glm,
                collin_tol,
                fixef_tol,
                fixef_maxiter,
                glm_tol,
                glm_maxiter,
                keep_resid,
                keep_fixef,
                keep_state,
                ..parse_spec(model, flavor, parse_vcov_kind(vcov, lag, cutoff, pixel, spherical)?, weights_type, fixef_rm, fixef_norm)?
            };
            let farrs = extract_factors(factors)?;
            let y = slice(&y, "y")?;
            let input = FitInput {
                n: y.len(),
                y,
                x: slices(&x, "x")?,
                endog: slices(&endog, "endog")?,
                inst: slices(&inst, "inst")?,
                factors: factor_cols(&farrs)?,
                factor_names,
                fe: fe.into_iter().map(|(factors, intercept, slopes)| FeTerm { factors, intercept, slopes }).collect(),
                slopes: slices(&slopes, "slopes")?,
                weights: weights.as_ref().map(|w| slice(w, "weights")).transpose()?,
                offset: offset.as_ref().map(|o| slice(o, "offset")).transpose()?,
                cluster,
                drop: drop.as_ref().map(|d| slice(d, "drop")).transpose()?,
                time,
                panel,
                lat: lat.as_ref().map(|c| slice(c, "lat")).transpose()?,
                lon: lon.as_ref().map(|c| slice(c, "lon")).transpose()?,
            };
            check_lengths(&input)?;
            let ($input, $spec, $threads, $verbose) = (&input, &spec, threads, verbose);
            $body
        }
    };
}

fn check_lengths(input: &FitInput<'_>) -> PyResult<()> {
    let n = input.n;
    let lens = input.x.iter().chain(&input.endog).chain(&input.inst).chain(&input.slopes).map(|s| s.len());
    let lens = lens
        .chain(input.factors.iter().map(FactorCol::len))
        .chain(input.weights.map(<[f64]>::len))
        .chain(input.offset.map(<[f64]>::len))
        .chain(input.drop.map(<[bool]>::len))
        .chain(input.lat.map(<[f64]>::len))
        .chain(input.lon.map(<[f64]>::len));
    if lens.into_iter().any(|l| l != n) {
        return Err(PyValueError::new_err("every input column must have the length of y"));
    }
    Ok(())
}

model_fn! {
    /// Estimates one model (`fereg_core::fit::fit`). Array arguments must be C-contiguous; they are
    /// borrowed, never copied.
    ///
    /// `vcov` is the variance estimator: `"iid"`, `"hetero"` (HC1), `"hc2"`, `"hc3"`, `"cluster"`
    /// (CRV1 over `cluster`), `"crv3"` (leave-one-cluster-out jackknife over `cluster`), `"nw"`
    /// (Newey-West: panel when `panel` is given, time series otherwise), `"dk"` (Driscoll-Kraay) or
    /// `"conley"` (spatial HAC over `lat` / `lon`, degrees). `time` and `panel` are indices into
    /// `factors` (time levels ascending = time order). `lag` is the Newey-West / Driscoll-Kraay lag
    /// (`None` = the flavor's default; pyfixest: floor(T^0.25), and a time series needs it);
    /// `cutoff` the Conley cutoff in km (`None` = fixest's deduced cutoff), `pixel` its pixel size in
    /// km (0 = none) and `spherical` its great-circle distance (else fixest's "triangular").
    /// Extensions of the pyfixest flavor (pyfixest raises): HC2/HC3 with fixed effects (exact
    /// leverage of the fixed effects), CRV3 for IV and logit (logit refits are logit), Newey-West /
    /// Driscoll-Kraay for IV at estimation, Driscoll-Kraay without a panel variable, and Conley
    /// (fixest's estimator with the HC1 correction N / (N - df_k)). HC2/HC3 with IV is an error.
    /// `gaussian_glm` marks a pyfixest-flavor `model="ols"` as pyfixest's `feglm(family="gaussian")`
    /// (its separation checks run: `fixef_rm="perfect"` and `relu`; `FitSpec::gaussian_glm`).
    fn fit() -> FitResult;
    |py, input, spec, threads, verbose| {
        let out = py.detach(|| with_ctx(threads, verbose, |ctx| cf::fit(input, spec, ctx))).map_err(to_pyerr)?;
        convert(py, out)
    }
}

/// Value of a required key of an options dict.
fn item<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?.ok_or_else(|| PyValueError::new_err(format!("missing option '{key}'")))
}

fn parse_structure(s: &str) -> PyResult<cf::Structure> {
    match s {
        "classic" => Ok(cf::Structure::Classic),
        "network" => Ok(cf::Structure::Network),
        _ => Err(PyValueError::new_err(format!("unknown panel structure '{s}'"))),
    }
}

fn structure_name(s: cf::Structure) -> &'static str {
    match s {
        cf::Structure::Classic => "classic",
        cf::Structure::Network => "network",
    }
}

fn which_name(w: cf::WhichBias) -> &'static str {
    match w {
        cf::WhichBias::Both => "both",
        cf::WhichBias::Individual => "individual",
        cf::WhichBias::Time => "time",
    }
}

const VARIANTS: [(&str, cf::Variant); 6] = [
    ("ss1", cf::Variant::Ss1),
    ("ss2", cf::Variant::Ss2),
    ("js", cf::Variant::Js),
    ("sj", cf::Variant::Sj),
    ("jj", cf::Variant::Jj),
    ("double", cf::Variant::Double),
];

fn permutation(p: Option<Vec<i64>>) -> PyResult<Option<Vec<u32>>> {
    p.map(|v| {
        v.into_iter()
            .map(|i| u32::try_from(i).map_err(|_| PyValueError::new_err(format!("permutation entry {i} out of range"))))
            .collect()
    })
    .transpose()
}

/// Bias options dict (see `fereg._binding.bias_corr_design`): `{method, structure, which,
/// bandwidth, time, variant, partitions}`.
fn parse_bias(d: &Bound<'_, PyDict>) -> PyResult<cf::BiasOptions> {
    let which = match item(d, "which")?.extract::<String>()?.as_str() {
        "both" => cf::WhichBias::Both,
        "individual" => cf::WhichBias::Individual,
        "time" => cf::WhichBias::Time,
        s => return Err(PyValueError::new_err(format!("unknown bias '{s}'"))),
    };
    let method = match item(d, "method")?.extract::<String>()?.as_str() {
        "analytical" => cf::BiasMethod::Analytical,
        "jackknife" => {
            let name: String = item(d, "variant")?.extract()?;
            let variant = VARIANTS
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| *v)
                .ok_or_else(|| PyValueError::new_err(format!("unknown jackknife variant '{name}'")))?;
            let parts: Vec<(Option<Vec<i64>>, Option<Vec<i64>>)> = item(d, "partitions")?.extract()?;
            let partitions = parts
                .into_iter()
                .map(|(i, t)| Ok(cf::Permutation { individual: permutation(i)?, period: permutation(t)? }))
                .collect::<PyResult<_>>()?;
            cf::BiasMethod::Jackknife(cf::JackknifeOptions { variant, partitions })
        }
        s => return Err(PyValueError::new_err(format!("unknown bias correction method '{s}'"))),
    };
    Ok(cf::BiasOptions {
        method,
        normalization: cf::Normalization::Alpaca,
        structure: parse_structure(&item(d, "structure")?.extract::<String>()?)?,
        which,
        bandwidth: item(d, "bandwidth")?.extract()?,
        panel: None,
        time: item(d, "time")?.extract()?,
    })
}

model_fn! {
    /// `fereg_core::fit::bias_dims` of the model given as to `fit` (a logit model) under the bias
    /// options `bias` (as `bias_corr`): `{n_individuals, n_periods, n_full}`, the jackknife index
    /// spaces N_pre / T_pre (for drawing `multiple` permutations) and N*.
    fn bias_dims(bias: &Bound<'py, PyDict>) -> Py<PyDict>;
    |py, input, spec, threads, verbose| {
        let opts = parse_bias(bias)?;
        let dims = py.detach(|| with_ctx(threads, verbose, |ctx| cf::bias_dims(input, spec, &opts, ctx))).map_err(to_pyerr)?;
        let d = PyDict::new(py);
        d.set_item("n_individuals", dims.n_individuals)?;
        d.set_item("n_periods", dims.n_periods)?;
        d.set_item("n_full", dims.n_full)?;
        Ok(d.unbind())
    }
}

model_fn! {
    /// Incidental-parameter bias correction of a logit model (`fereg_core::fit::bias_corr`; alpaca
    /// `biasCorr` and logitfe's jackknife). The model arguments are those of `fit` (the core
    /// re-estimates the uncorrected model); `bias` = `{method: "analytical" | "jackknife",
    /// structure: "classic" | "network", which: "both" | "individual" | "time", bandwidth: L,
    /// time: factor index | None, variant: "ss1" | "ss2" | "js" | "sj" | "jj" | "double",
    /// partitions: [(individual permutation | None, period permutation | None), ...]}`
    /// (0-based new indices; `variant` / `partitions` are read for the jackknife only).
    ///
    /// Returns `(FitResult at the corrected point, report)` with report `{coef_uncorr, bias_term,
    /// structure, which, bandwidth, method, variant, n_partitions, ape_jackknife, ape_uncorr,
    /// n_full, n_fits, deviance_uncorr, resolve_iterations, notes}` (`bias_term` = uncorrected
    /// minus corrected coefficients; the notes are also in the fit's notes).
    fn bias_corr(bias: &Bound<'py, PyDict>) -> (FitResult, Py<PyDict>);
    |py, input, spec, threads, verbose| {
        let opts = parse_bias(bias)?;
        let bf = py.detach(|| with_ctx(threads, verbose, |ctx| cf::bias_corr(input, spec, &opts, ctx))).map_err(to_pyerr)?;
        let r = bf.report;
        let d = PyDict::new(py);
        d.set_item("coef_uncorr", r.coef_uncorr.into_pyarray(py))?;
        d.set_item("bias_term", r.bias_term.into_pyarray(py))?;
        d.set_item("structure", structure_name(r.structure))?;
        d.set_item("which", which_name(r.which))?;
        d.set_item("bandwidth", r.bandwidth)?;
        let (method, variant, n_partitions) = match &r.method {
            cf::BiasMethod::Analytical => ("analytical", None, 0),
            cf::BiasMethod::Jackknife(j) => {
                let name = VARIANTS.iter().find(|(_, v)| *v == j.variant).map(|(n, _)| *n);
                ("jackknife", name, j.partitions.len())
            }
        };
        d.set_item("method", method)?;
        d.set_item("variant", variant)?;
        d.set_item("n_partitions", n_partitions)?;
        d.set_item("ape_jackknife", opt_array(py, r.ape_jackknife))?;
        d.set_item("ape_uncorr", opt_array(py, r.ape_uncorr))?;
        d.set_item("n_full", r.n_full)?;
        d.set_item("n_fits", r.n_fits)?;
        d.set_item("deviance_uncorr", r.deviance_uncorr)?;
        d.set_item("resolve_iterations", r.resolve_iterations)?;
        d.set_item("notes", r.notes)?;
        Ok((convert(py, bf.fit)?, d.unbind()))
    }
}

model_fn! {
    /// Average partial effects of a logit model (`fereg_core::fit::apes`; alpaca `getAPEs`). The
    /// model arguments are those of `fit`; `ape` = `{n_pop: float | None, structure: "classic" |
    /// "network", sampling: "independence" | "unrestricted", weak_exo: bool, at: None | {bias:
    /// <bias options of bias_corr>, coef: corrected coefficients, ape_jackknife: array | None}}`
    /// (`at=None`: the uncorrected fit).
    ///
    /// Returns `{coef_source, delta, vcov (p x p), bias_term, binary, adj, weak_exo_used,
    /// structure, bandwidth, n_full, notes}`.
    fn apes(ape: &Bound<'py, PyDict>) -> Py<PyDict>;
    |py, input, spec, threads, verbose| {
        let sampling = match item(ape, "sampling")?.extract::<String>()?.as_str() {
            "independence" => cf::Sampling::Independence,
            "unrestricted" => cf::Sampling::Unrestricted,
            s => return Err(PyValueError::new_err(format!("unknown sampling_fe '{s}'"))),
        };
        let opts = cf::ApeOptions {
            n_pop: item(ape, "n_pop")?.extract()?,
            structure: parse_structure(&item(ape, "structure")?.extract::<String>()?)?,
            sampling,
            weak_exo: item(ape, "weak_exo")?.extract()?,
        };
        let at_obj = item(ape, "at")?;
        let corrected: Option<(cf::BiasOptions, PyReadonlyArray1<'py, f64>, Option<PyReadonlyArray1<'py, f64>>)> =
            if at_obj.is_none() {
                None
            } else {
                let at = at_obj.cast::<PyDict>()?;
                Some((parse_bias(&item(at, "bias")?.cast_into::<PyDict>()?)?, item(at, "coef")?.extract()?, item(at, "ape_jackknife")?.extract()?))
            };
        let at = match &corrected {
            None => cf::ApeAt::Uncorrected,
            Some((options, coef, jk)) => cf::ApeAt::Corrected {
                options,
                coef: slice(coef, "coef")?,
                ape_jackknife: jk.as_ref().map(|a| slice(a, "ape_jackknife")).transpose()?,
            },
        };
        let a = py.detach(|| with_ctx(threads, verbose, |ctx| cf::apes(input, spec, at, &opts, ctx))).map_err(to_pyerr)?;
        let p = a.delta.len();
        let d = PyDict::new(py);
        d.set_item("coef_source", sources(py, &a.coef_source)?)?;
        d.set_item("delta", a.delta.into_pyarray(py))?;
        d.set_item("vcov", a.vcov.into_pyarray(py).reshape([p, p])?)?;
        d.set_item("bias_term", opt_array(py, a.bias_term))?;
        d.set_item("binary", a.binary)?;
        d.set_item("adj", a.adj)?;
        d.set_item("weak_exo_used", a.weak_exo_used)?;
        d.set_item("structure", structure_name(a.structure))?;
        d.set_item("bandwidth", a.bandwidth)?;
        d.set_item("n_full", a.n_full)?;
        d.set_item("notes", a.notes)?;
        Ok(d.unbind())
    }
}

macro_rules! dist_fn {
    ($name:ident, $f:path, ($($p:ident),*), $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            x: PyReadonlyArrayDyn<'py, f64>,
            $($p: f64),*
        ) -> Bound<'py, numpy::PyArrayDyn<f64>> {
            x.as_array().mapv(|v| $f(v $(, $p)*)).into_pyarray(py)
        }
    };
}

dist_fn!(pnorm, cf::dist::pnorm, (), "Standard normal CDF (elementwise).");
dist_fn!(qnorm, cf::dist::qnorm, (), "Standard normal quantile (elementwise).");
dist_fn!(pt, cf::dist::pt, (df), "Student t CDF (elementwise).");
dist_fn!(qt, cf::dist::qt, (df), "Student t quantile (elementwise).");
dist_fn!(pf_upper, cf::dist::pf_upper, (df1, df2), "F upper tail P(F > f) (elementwise).");
dist_fn!(pchisq_upper, cf::dist::pchisq_upper, (df), "Chi-squared upper tail P(X > x) (elementwise).");

#[pymodule]
mod _core {
    #[pymodule_export]
    #[allow(non_upper_case_globals)]
    const __version__: &str = fereg_core::VERSION;

    #[pymodule_export]
    use super::{FitResult, FitState, apes, bias_corr, bias_dims, fit, pchisq_upper, pf_upper, pnorm, pt, qnorm, qt};
}
