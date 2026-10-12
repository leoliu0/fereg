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

fn parse_vcov_kind(s: &str) -> PyResult<VcovKind> {
    match s {
        "iid" => Ok(VcovKind::Iid),
        "hetero" => Ok(VcovKind::Hetero),
        "cluster" => Ok(VcovKind::Cluster),
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

fn vcov_dict<'py>(py: Python<'py>, v: VcovOutput) -> PyResult<Bound<'py, PyDict>> {
    let k = (v.vcov.len() as f64).sqrt().round() as usize;
    let d = PyDict::new(py);
    d.set_item("kind", kind_name(v.kind))?;
    d.set_item("vcov", v.vcov.into_pyarray(py).reshape([k, k])?)?;
    d.set_item("df_k", v.df_k)?;
    d.set_item("df_t", v.df_t)?;
    d.set_item("n_clusters", v.n_clusters)?;
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
    /// `factors` are `(kind, values, nlevels, base)` tuples over the original input rows.
    #[pyo3(signature = (kind, ssc=None, factors=None, factor_names=None, cluster=None, threads=0))]
    fn vcov<'py>(
        &self,
        py: Python<'py>,
        kind: &str,
        ssc: Option<&Bound<'py, PyDict>>,
        factors: Option<&Bound<'py, PyList>>,
        factor_names: Option<Vec<String>>,
        cluster: Option<Vec<Vec<usize>>>,
        threads: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let kind = parse_vcov_kind(kind)?;
        let ssc = parse_ssc(ssc)?;
        let arrs = match factors {
            Some(f) => extract_factors(f)?,
            None => Vec::new(),
        };
        let cols = factor_cols(&arrs)?;
        let names = factor_names.unwrap_or_default();
        let cluster = cluster.unwrap_or_default();
        let out = py
            .detach(|| with_ctx(threads, 0, |ctx| self.inner.vcov(kind, &ssc, &cols, &names, &cluster, ctx)))
            .map_err(to_pyerr)?;
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
    vcov: &str,
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
        vcov: parse_vcov_kind(vcov)?,
        weights_type: match weights_type {
            "aweights" => WeightsType::Aweights,
            "fweights" => WeightsType::Fweights,
            _ => return Err(bad("weights_type", weights_type)),
        },
        fixef_rm: match fixef_rm {
            "none" => FixefRm::None,
            "singleton" => FixefRm::Singleton,
            "perfect" => FixefRm::Perfect,
            _ => return Err(bad("fixef_rm", fixef_rm)),
        },
        fixef_norm: parse_fixef_norm(fixef_norm)?,
        ..FitSpec::default()
    })
}

/// Estimates one model (`fereg_core::fit::fit`). Array arguments must be C-contiguous; they are
/// borrowed, never copied.
#[pyfunction]
#[pyo3(signature = (
    y, x, endog, inst, factors, factor_names, fe, slopes, weights=None, offset=None,
    cluster=Vec::new(), drop=None, *, model="ols", flavor="pyfixest", intercept=true, vcov="iid",
    ssc=None, weights_type="aweights", fixef_rm="singleton", relu=false, mu=false,
    collin_tol=1e-9, fixef_tol=1e-8, fixef_maxiter=16000, glm_tol=1e-8, glm_maxiter=25,
    keep_resid=true, keep_fixef=true, fixef_norm="minnorm", keep_state=true, threads=0, verbose=0
))]
#[allow(clippy::too_many_arguments)]
fn fit<'py>(
    py: Python<'py>,
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
    model: &str,
    flavor: &str,
    intercept: bool,
    vcov: &str,
    ssc: Option<&Bound<'py, PyDict>>,
    weights_type: &str,
    fixef_rm: &str,
    relu: bool,
    mu: bool,
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
) -> PyResult<FitResult> {
    let spec = FitSpec {
        intercept,
        ssc: parse_ssc(ssc)?,
        separation: Separation { relu, mu },
        collin_tol,
        fixef_tol,
        fixef_maxiter,
        glm_tol,
        glm_maxiter,
        keep_resid,
        keep_fixef,
        keep_state,
        ..parse_spec(model, flavor, vcov, weights_type, fixef_rm, fixef_norm)?
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
    };
    let n = input.n;
    let lens = input.x.iter().chain(&input.endog).chain(&input.inst).chain(&input.slopes).map(|s| s.len());
    let lens = lens
        .chain(input.factors.iter().map(FactorCol::len))
        .chain(input.weights.map(<[f64]>::len))
        .chain(input.offset.map(<[f64]>::len))
        .chain(input.drop.map(<[bool]>::len));
    if lens.into_iter().any(|l| l != n) {
        return Err(PyValueError::new_err("every input column must have the length of y"));
    }
    let out = py.detach(|| with_ctx(threads, verbose, |ctx| cf::fit(&input, &spec, ctx))).map_err(to_pyerr)?;
    convert(py, out)
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
    use super::{FitResult, FitState, fit, pchisq_upper, pf_upper, pnorm, pt, qnorm, qt};
}
