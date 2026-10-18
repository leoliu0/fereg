//! R bindings of `fereg_core::fit` (fixest conventions).
//!
//! The R code parses formulas and builds every design column; this module only converts R
//! vectors to the core's input types (zero-copy for double and integer vectors), runs the fit and
//! converts the result back to an R list. Rust worker threads never touch R: the core calls
//! `Ctx::interrupt` on the calling (R main) thread only.

use std::os::raw::c_int;

use extendr_api::prelude::*;
use fereg_core::fit::{
    self, ApeAt, ApeOptions, BiasMethod, BiasOptions, CoefSource, FactorCol, FeTerm, FitInput, FitOutput, FixefValues,
    FitSpec, FitState, FixefNorm, FixefRm, Flavor, GDf, JackknifeOptions, KFixef, Model, Normalization, Permutation,
    Sampling, Ssc, Structure, TDf, TestStat, Variant, VcovKind, VcovOutput, WeightsType, WhichBias,
};
use fereg_core::{Ctx, RfError, par};
use rayon::prelude::*;

type Result<T, E = Error> = std::result::Result<T, E>;

unsafe extern "C" {
    /// `entrypoint.c`: runs `R_CheckUserInterrupt` under `R_ToplevelExec`, so the longjmp of an
    /// interrupt never crosses Rust frames. Nonzero when the user interrupted.
    fn fereg_interrupted() -> c_int;
}

fn interrupted() -> bool {
    // SAFETY: called on the R main thread (the core only polls `Ctx::interrupt` there).
    unsafe { fereg_interrupted() != 0 }
}

fn print_log(msg: &str) {
    rprintln!("{msg}");
}

fn core_err(e: RfError) -> Error {
    Error::Other(e.to_string())
}

fn invalid(msg: impl Into<String>) -> Error {
    Error::Other(msg.into())
}

/// Element `name` of a named list; `NULL` when absent.
fn field(list: &List, name: &str) -> Robj {
    list.iter().find(|(n, _)| *n == name).map(|(_, v)| v).unwrap_or_else(|| ().into())
}

fn real_slice<'a>(r: &'a Robj, what: &str) -> Result<&'a [f64]> {
    r.as_real_slice().ok_or_else(|| invalid(format!("fereg: `{what}` must be a double vector")))
}

fn opt_real<'a>(r: &'a Robj, what: &str) -> Result<Option<&'a [f64]>> {
    if r.is_null() { Ok(None) } else { real_slice(r, what).map(Some) }
}

fn list_items(list: &List, name: &str) -> Result<Vec<Robj>> {
    let r = field(list, name);
    if r.is_null() {
        return Ok(Vec::new());
    }
    let l: List = r.try_into().map_err(|_| invalid(format!("fereg: `{name}` must be a list")))?;
    Ok(l.values().collect())
}

fn real_slices<'a>(items: &'a [Robj], what: &str) -> Result<Vec<&'a [f64]>> {
    items.iter().map(|r| real_slice(r, what)).collect()
}

/// A number given as an R double or integer.
fn number(r: &Robj) -> Option<f64> {
    r.as_real().or_else(|| r.as_integer().map(f64::from))
}

fn get_f64(list: &List, name: &str) -> Result<f64> {
    number(&field(list, name)).ok_or_else(|| invalid(format!("fereg: `{name}` must be a number")))
}

fn get_usize(list: &List, name: &str) -> Result<usize> {
    let v = get_f64(list, name)?;
    if v.is_finite() && v >= 0.0 { Ok(v as usize) } else { Err(invalid(format!("fereg: invalid `{name}`"))) }
}

fn get_bool(list: &List, name: &str) -> Result<bool> {
    field(list, name).as_bool().ok_or_else(|| invalid(format!("fereg: `{name}` must be TRUE or FALSE")))
}

fn get_str<'a>(list: &'a List, name: &str) -> Result<&'a str> {
    field(list, name).as_str().ok_or_else(|| invalid(format!("fereg: `{name}` must be a string")))
}

/// 0-based indices from a 1-based integer (or double) vector.
fn indices(r: &Robj, what: &str) -> Result<Vec<usize>> {
    let bad = || invalid(format!("fereg: invalid indices in `{what}`"));
    if let Some(v) = r.as_integer_slice() {
        v.iter().map(|&i| if i >= 1 { Ok(i as usize - 1) } else { Err(bad()) }).collect()
    } else if let Some(v) = r.as_real_slice() {
        v.iter().map(|&i| if i >= 1.0 { Ok(i as usize - 1) } else { Err(bad()) }).collect()
    } else if r.is_null() {
        Ok(Vec::new())
    } else {
        Err(bad())
    }
}

/// A factor column: double (`F64`), plain integer (`I32`), or an integer vector with a `levels`
/// attribute (R factor or host-built codes, 1-based: `Codes`).
fn factor_col<'a>(r: &'a Robj, name: &str) -> Result<FactorCol<'a>> {
    if let Some(v) = r.as_real_slice() {
        return Ok(FactorCol::F64(v));
    }
    if let Some(v) = r.as_integer_slice() {
        return Ok(match r.get_attrib("levels") {
            Some(levels) => FactorCol::Codes { codes: v, nlevels: levels.len() as u32, base: 1 },
            None => FactorCol::I32(v),
        });
    }
    Err(invalid(format!("fereg: factor `{name}` must be a double, integer or factor vector")))
}

fn factor_cols<'a>(items: &'a [Robj], names: &[String]) -> Result<Vec<FactorCol<'a>>> {
    items.iter().zip(names).map(|(r, n)| factor_col(r, n)).collect()
}

fn strings(r: &Robj) -> Vec<String> {
    r.as_str_iter().map(|it| it.map(str::to_string).collect()).unwrap_or_default()
}

fn clusters(items: &[Robj]) -> Result<Vec<Vec<usize>>> {
    items.iter().map(|r| indices(r, "cluster")).collect()
}

/// The variance request built by `fereg:::core_kind()`: list(type, lag, cutoff, pixel, spherical)
/// with `lag` / `cutoff` NA for the reference's default (fixest's lag rules, deduced cutoff).
fn vcov_kind(l: &List) -> Result<VcovKind> {
    // NA_real_ (a NaN payload) = the reference's default; read through the slice (extendr's
    // `as_real` maps NA to `None`).
    let opt = |name: &str| -> Result<Option<f64>> {
        let v = field(l, name).as_real_slice().and_then(|s| s.first().copied()).ok_or_else(|| invalid(format!("fereg: `{name}` must be a number")))?;
        Ok(if v.is_nan() { None } else { Some(v) })
    };
    let lag = || -> Result<Option<usize>> {
        match opt("lag")? {
            None => Ok(None),
            Some(v) if v >= 0.0 && v.fract() == 0.0 => Ok(Some(v as usize)),
            Some(_) => Err(invalid("fereg: the lag must be a non-negative integer")),
        }
    };
    match get_str(l, "type")? {
        "iid" => Ok(VcovKind::Iid),
        "hetero" => Ok(VcovKind::Hetero),
        "hc2" => Ok(VcovKind::Hc2),
        "hc3" => Ok(VcovKind::Hc3),
        "cluster" => Ok(VcovKind::Cluster),
        "crv3" => Ok(VcovKind::Crv3),
        "nw" => Ok(VcovKind::NeweyWest { lag: lag()? }),
        "dk" => Ok(VcovKind::DriscollKraay { lag: lag()? }),
        "conley" => Ok(VcovKind::Conley { cutoff: opt("cutoff")?, pixel: get_f64(l, "pixel")?, spherical: get_bool(l, "spherical")? }),
        s => Err(invalid(format!("fereg: unknown vcov type `{s}`"))),
    }
}

/// A single 0-based factor index from a 1-based number, or `None` for `NULL`.
fn opt_index(r: &Robj, what: &str) -> Result<Option<usize>> {
    match indices(r, what)?.as_slice() {
        [] => Ok(None),
        [i] => Ok(Some(*i)),
        _ => Err(invalid(format!("fereg: `{what}` must be a single factor index"))),
    }
}

fn ssc_of(l: &List) -> Result<Ssc> {
    let k_fixef = match get_str(l, "K.fixef")? {
        "none" => KFixef::None,
        "full" => KFixef::Full,
        "nonnested" => KFixef::Nonnested,
        other => return Err(invalid(format!("fereg: unknown K.fixef `{other}`"))),
    };
    let g_df = match get_str(l, "G.df")? {
        "min" => GDf::Min,
        "conventional" => GDf::Conventional,
        other => return Err(invalid(format!("fereg: unknown G.df `{other}`"))),
    };
    let t = field(l, "t.df");
    let t_df = match (t.as_str(), number(&t)) {
        (Some("min"), _) => TDf::Min,
        (Some("conventional"), _) => TDf::Conventional,
        (None, Some(v)) => TDf::Fixed(v),
        _ => return Err(invalid("fereg: invalid t.df")),
    };
    Ok(Ssc {
        k_adj: get_bool(l, "K.adj")?,
        k_fixef,
        k_exact: get_bool(l, "K.exact")?,
        g_adj: get_bool(l, "G.adj")?,
        g_df,
        t_df,
    })
}

fn spec_of(l: &List) -> Result<FitSpec> {
    let model = match get_str(l, "model")? {
        "ols" => Model::Ols,
        "poisson" => Model::Poisson,
        "logit" => Model::Logit,
        other => return Err(invalid(format!("fereg: unknown model `{other}`"))),
    };
    let fixef_rm = match get_str(l, "fixef_rm")? {
        "none" => FixefRm::None,
        "singleton" => FixefRm::Singleton,
        "perfect_fit" => FixefRm::Perfect,
        other => return Err(invalid(format!("fereg: unknown fixef.rm `{other}`"))),
    };
    let ssc: List = field(l, "ssc").try_into().map_err(|_| invalid("fereg: `ssc` must be a list"))?;
    Ok(FitSpec {
        model,
        flavor: Flavor::Fixest,
        intercept: get_bool(l, "intercept")?,
        vcov: vcov_kind(&field(l, "vcov").try_into().map_err(|_| invalid("fereg: `vcov` must be a list"))?)?,
        ssc: ssc_of(&ssc)?,
        weights_type: WeightsType::Aweights,
        fixef_rm,
        collin_tol: get_f64(l, "collin_tol")?,
        fixef_tol: get_f64(l, "fixef_tol")?,
        fixef_maxiter: get_usize(l, "fixef_maxiter")?,
        glm_tol: get_f64(l, "glm_tol")?,
        glm_maxiter: get_usize(l, "glm_maxiter")?,
        keep_resid: get_bool(l, "keep_resid")?,
        keep_fixef: get_bool(l, "keep_fixef")?,
        fixef_norm: FixefNorm::Fixest,
        keep_state: get_bool(l, "keep_state")?,
        ..Default::default()
    })
}

fn ctx(threads: i32) -> Ctx<'static> {
    // faer calls made outside the core's scoped pools must not start rayon's global pool (all
    // cores, threads outliving the call): every parallel faer call in the core passes its pool.
    faer::set_global_parallelism(faer::Par::Seq);
    Ctx { threads: threads.max(0) as usize, verbose: 0, interrupt: &interrupted, log: &print_log }
}

// ---- output conversion --------------------------------------------------------------------

fn named(pairs: Vec<(&str, Robj)>) -> Robj {
    let (names, values): (Vec<&str>, Vec<Robj>) = pairs.into_iter().unzip();
    List::from_names_and_values(names, values).expect("names and values have equal length").into()
}

fn real(v: f64) -> Robj {
    v.into()
}

fn reals(v: &[f64]) -> Robj {
    v.into()
}

fn int(v: usize) -> Robj {
    (v as f64).into()
}

/// 1-based input rows.
fn rows_1based(v: &[u32]) -> Robj {
    Integers::from_values(v.iter().map(|&i| i as i32 + 1)).into()
}

/// Square row-major `k x k` matrix as an R matrix (column-major).
fn matrix(v: &[f64]) -> Robj {
    let k = (v.len() as f64).sqrt() as usize;
    let mut cm = vec![0.0; v.len()];
    for i in 0..k {
        for j in 0..k {
            cm[j * k + i] = v[i * k + j];
        }
    }
    let mut r: Robj = cm.into();
    r.set_attrib("dim", [k as i32, k as i32]).expect("dim attribute");
    r
}

fn test_stat(t: &TestStat) -> Robj {
    let mut r: Robj = vec![t.stat, t.p, t.df1, t.df2].into();
    r.set_names(["stat", "p", "df1", "df2"]).expect("names");
    r
}

/// Coefficient sources as list(type = c("intercept", "x", "endog", "inst"), index = 1-based).
fn sources(s: &[CoefSource]) -> Robj {
    let (types, idx): (Vec<&str>, Vec<i32>) = s
        .iter()
        .map(|c| match *c {
            CoefSource::Intercept => ("intercept", 0),
            CoefSource::X(j) => ("x", j as i32 + 1),
            CoefSource::Endog(j) => ("endog", j as i32 + 1),
            CoefSource::Inst(j) => ("inst", j as i32 + 1),
        })
        .unzip();
    named(vec![("type", Strings::from_values(types).into()), ("index", Integers::from_values(idx).into())])
}

fn vcov_out(v: &VcovOutput) -> Robj {
    let (kind, lag, cutoff) = match v.kind {
        VcovKind::Iid => ("iid", None, None),
        VcovKind::Hetero => ("hetero", None, None),
        VcovKind::Cluster => ("cluster", None, None),
        VcovKind::Hc2 => ("hc2", None, None),
        VcovKind::Hc3 => ("hc3", None, None),
        VcovKind::Crv3 => ("crv3", None, None),
        VcovKind::NeweyWest { lag } => ("nw", lag, None),
        VcovKind::DriscollKraay { lag } => ("dk", lag, None),
        VcovKind::Conley { cutoff, .. } => ("conley", None, cutoff),
    };
    named(vec![
        ("kind", kind.into()),
        ("vcov", matrix(&v.vcov)),
        ("df_k", real(v.df_k)),
        ("df_t", real(v.df_t.unwrap_or(f64::NAN))),
        ("t_dist", v.df_t.is_some().into()),
        ("n_clusters", reals(&v.n_clusters.iter().map(|&g| g as f64).collect::<Vec<_>>())),
        ("n_periods", real(v.n_periods.map_or(f64::NAN, |t| t as f64))),
        // Resolved defaults: Newey-West / Driscoll-Kraay lag, Conley cutoff (km).
        ("lag", real(lag.map_or(f64::NAN, |l| l as f64))),
        ("cutoff", real(cutoff.unwrap_or(f64::NAN))),
        ("ssc_factor", real(v.ssc_factor)),
        ("psd_fixed", v.psd_fixed.into()),
    ])
}

/// Time / panel factor indices and coordinates of a variance request, each `None` when unused.
struct VcovVars<'a> {
    time: Option<usize>,
    panel: Option<usize>,
    lat: Option<&'a [f64]>,
    lon: Option<&'a [f64]>,
}

fn vcov_vars<'a>(time: &Robj, panel: &Robj, lat: &'a Robj, lon: &'a Robj) -> Result<VcovVars<'a>> {
    Ok(VcovVars {
        time: opt_index(time, "time")?,
        panel: opt_index(panel, "panel")?,
        lat: opt_real(lat, "lat")?,
        lon: opt_real(lon, "lon")?,
    })
}

fn fixef_list(values: &[FixefValues]) -> Robj {
    let items: Vec<Robj> = values
        .iter()
        .map(|f| {
            named(vec![
                ("term", int(f.term + 1)),
                ("slope", real(f.slope.map_or(f64::NAN, |j| (j + 1) as f64))),
                ("values", reals(&f.values)),
                ("first_obs", rows_1based(&f.first_obs)),
                ("references", int(f.references)),
            ])
        })
        .collect();
    List::from_values(items).into()
}

/// Copies per-observation vectors into new R double vectors (`NULL` for `None`). The vectors are
/// allocated on the R thread; worker threads only write their memory, in chunks of `COPY_CHUNK`.
fn obs_vectors<const K: usize>(src: [Option<&[f64]>; K], threads: usize) -> Result<[Robj; K]> {
    const COPY_CHUNK: usize = 1 << 16;
    let mut out: [Robj; K] = std::array::from_fn(|k| match src[k] {
        Some(v) => Robj::alloc_vector(extendr_api::rtype_to_sxp(Rtype::Doubles), v.len()),
        None => ().into(),
    });
    let mut pairs: Vec<(&mut [f64], &[f64])> = Vec::with_capacity(K);
    for (r, s) in out.iter_mut().zip(src) {
        if let Some(s) = s {
            pairs.push((r.as_real_slice_mut().expect("double vector"), s));
        }
    }
    let total: usize = pairs.iter().map(|(_, s)| s.len()).sum();
    let threads = par::threads_for(threads, total.div_ceil(COPY_CHUNK) * par::CHUNK);
    if threads <= 1 {
        for (d, s) in pairs {
            d.copy_from_slice(s);
        }
    } else {
        par::with_pool(threads, |pool| {
            pool.install(|| {
                pairs
                    .into_par_iter()
                    .flat_map(|(d, s)| d.par_chunks_mut(COPY_CHUNK).zip(s.par_chunks(COPY_CHUNK)))
                    .for_each(|(d, s)| d.copy_from_slice(s))
            })
        })
        .map_err(core_err)?;
    }
    Ok(out)
}

fn output(out: FitOutput, n_input: usize, threads: usize) -> Result<Robj> {
    let s = &out.stats;
    let stats = named(vec![
        ("ssr", real(s.ssr)),
        ("ssr_null", real(s.ssr_null)),
        ("ssr_fe_only", real(s.ssr_fe_only)),
        ("sigma2", real(s.sigma2)),
        ("rmse", real(s.rmse)),
        ("r2", real(s.r2)),
        ("ar2", real(s.ar2)),
        ("wr2", real(s.wr2)),
        ("war2", real(s.war2)),
        ("loglik", real(s.loglik)),
        ("ll_null", real(s.ll_null)),
        ("deviance", real(s.deviance)),
        ("null_deviance", real(s.null_deviance)),
        ("pseudo_r2", real(s.pseudo_r2)),
        ("apr2", real(s.apr2)),
        ("sq_cor", real(s.sq_cor)),
        ("f_test", test_stat(&s.f_test)),
        ("fe_iterations", int(s.fe_iterations)),
        ("irls_iterations", int(s.irls_iterations)),
        ("converged", s.converged.into()),
    ]);
    let iv: Robj = match &out.iv {
        None => ().into(),
        Some(iv) => {
            let stages: Vec<Robj> = iv
                .first_stage
                .iter()
                .map(|fs| {
                    named(vec![
                        ("endog", int(fs.endog + 1)),
                        ("coef_source", sources(&fs.coef_source)),
                        ("coef", reals(&fs.coef)),
                        ("vcov", vcov_out(&fs.vcov)),
                        ("collinear", sources(&fs.collinear)),
                        ("f_test", test_stat(&fs.f_test)),
                        ("wald", test_stat(&fs.wald)),
                    ])
                })
                .collect();
            named(vec![
                ("first_stage", List::from_values(stages).into()),
                ("wu_hausman", test_stat(&iv.wu_hausman)),
                ("sargan", test_stat(&iv.sargan)),
                ("ssr_iv", real(iv.ssr_iv)),
            ])
        }
    };
    let fixef = fixef_list(&out.fixef);
    let d = &out.dropped;
    let na = &d.na_by;
    let mut na_by: Robj = [na.y, na.x, na.endog, na.inst, na.fe, na.cluster, na.weights, na.offset, na.extra]
        .iter()
        .map(|&v| v as f64)
        .collect::<Vec<f64>>()
        .into();
    na_by
        .set_names(["y", "x", "endog", "inst", "fe", "cluster", "weights", "offset", "extra"])
        .expect("names");
    let dropped = named(vec![
        ("na", int(d.na)),
        ("zero_weight", int(d.zero_weight)),
        ("singletons", int(d.singletons)),
        ("perfect_fit", int(d.perfect_fit)),
        ("separated", int(d.separated)),
        ("na_by", na_by),
    ]);
    // Per-observation outputs: R vectors allocated here (R thread), filled in parallel.
    let [resid, fitted, eta, irls_weights] =
        obs_vectors([out.resid.as_deref(), out.fitted.as_deref(), out.eta.as_deref(), out.irls_weights.as_deref()], threads)?;
    // All rows kept: the R side uses the input rows directly instead of an index vector.
    let obs: Robj = if out.obs.len() == n_input { ().into() } else { rows_1based(&out.obs) };
    let state: Robj = match out.state {
        Some(st) => ExternalPtr::new(st).into(),
        None => ().into(),
    };
    Ok(named(vec![
        ("nobs", int(out.nobs)),
        ("nobs_origin", int(out.nobs_origin)),
        ("sumw", real(out.sumw)),
        ("obs", obs),
        ("dropped", dropped),
        ("coef_source", sources(&out.coef_source)),
        ("coef", reals(&out.coef)),
        ("collinear", sources(&out.collinear)),
        ("vcov", vcov_out(&out.vcov)),
        ("nparams", int(out.nparams)),
        ("fe_sizes", reals(&out.fe_sizes.iter().map(|&g| g as f64).collect::<Vec<_>>())),
        ("stats", stats),
        ("iv", iv),
        ("resid", resid),
        ("fitted", fitted),
        ("eta", eta),
        ("irls_weights", irls_weights),
        ("fixef", fixef),
        ("notes", Strings::from_values(out.notes.iter().map(String::as_str)).into()),
        ("state", state),
    ]))
}

// ---- entry points ---------------------------------------------------------------------------

/// The R vectors of the input list built by `fereg:::fit_setup()`, kept alive while a `FitInput`
/// borrows them (zero-copy for double and integer vectors).
struct InputR {
    n: usize,
    y: Robj,
    x: Vec<Robj>,
    endog: Vec<Robj>,
    inst: Vec<Robj>,
    factors: Vec<Robj>,
    factor_names: Vec<String>,
    slopes: Vec<Robj>,
    weights: Robj,
    offset: Robj,
    cluster: Vec<Robj>,
    time: Robj,
    panel: Robj,
    lat: Robj,
    lon: Robj,
    fe: Vec<FeTerm>,
    drop: Option<Vec<bool>>,
}

impl InputR {
    fn read(input: &List) -> Result<InputR> {
        let drop_r = field(input, "drop");
        let drop = if drop_r.is_null() {
            None
        } else {
            let v = drop_r.as_logical_slice().ok_or_else(|| invalid("fereg: `drop` must be logical"))?;
            Some(v.iter().map(|b| b.is_true()).collect())
        };
        let fe = list_items(input, "fe")?
            .iter()
            .map(|t| {
                let t: List = t.try_into().map_err(|_| invalid("fereg: invalid FE term"))?;
                Ok(FeTerm {
                    factors: indices(&field(&t, "factors"), "fe")?,
                    intercept: get_bool(&t, "intercept")?,
                    slopes: indices(&field(&t, "slopes"), "fe")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let factors = list_items(input, "factors")?;
        let factor_names = strings(&field(input, "factor_names"));
        if factor_names.len() != factors.len() {
            return Err(invalid("fereg: one name per factor is required"));
        }
        Ok(InputR {
            n: get_usize(input, "n")?,
            y: field(input, "y"),
            x: list_items(input, "x")?,
            endog: list_items(input, "endog")?,
            inst: list_items(input, "inst")?,
            factors,
            factor_names,
            slopes: list_items(input, "slopes")?,
            weights: field(input, "weights"),
            offset: field(input, "offset"),
            cluster: list_items(input, "cluster")?,
            time: field(input, "time"),
            panel: field(input, "panel"),
            lat: field(input, "lat"),
            lon: field(input, "lon"),
            fe,
            drop,
        })
    }

    fn fit_input(&self) -> Result<FitInput<'_>> {
        let vv = vcov_vars(&self.time, &self.panel, &self.lat, &self.lon)?;
        Ok(FitInput {
            n: self.n,
            y: real_slice(&self.y, "y")?,
            x: real_slices(&self.x, "x")?,
            endog: real_slices(&self.endog, "endog")?,
            inst: real_slices(&self.inst, "inst")?,
            factors: factor_cols(&self.factors, &self.factor_names)?,
            factor_names: self.factor_names.clone(),
            fe: self.fe.clone(),
            slopes: real_slices(&self.slopes, "slopes")?,
            weights: opt_real(&self.weights, "weights")?,
            offset: opt_real(&self.offset, "offset")?,
            cluster: clusters(&self.cluster)?,
            time: vv.time,
            panel: vv.panel,
            lat: vv.lat,
            lon: vv.lon,
            drop: self.drop.as_deref(),
        })
    }
}

/// Estimates one model. `input` and `spec` are built by `fereg:::fit_setup()`.
#[extendr]
fn fereg_fit(input: List, spec: List, nthreads: i32) -> Result<Robj> {
    let inp = InputR::read(&input)?;
    let spec = spec_of(&spec)?;
    let out = fit::fit(&inp.fit_input()?, &spec, &ctx(nthreads)).map_err(core_err)?;
    output(out, inp.n, nthreads.max(0) as usize)
}

fn structure(s: &str) -> Result<Structure> {
    match s {
        "classic" => Ok(Structure::Classic),
        "network" => Ok(Structure::Network),
        _ => Err(invalid(format!("fereg: unknown panel structure `{s}`"))),
    }
}

/// One dimension of a `multiple` partition: the 1-based new index of every unit (`sample.int()`)
/// as 0-based indices, or `None` (not permuted).
fn permutation(r: &Robj, what: &str) -> Result<Option<Vec<u32>>> {
    if r.is_null() {
        return Ok(None);
    }
    Ok(Some(indices(r, what)?.into_iter().map(|i| i as u32).collect()))
}

/// The options built by `fereg::biasCorr()`: list(method, structure, which, bandwidth, time (1-based
/// factor index or NULL), variant, partitions = list(list(individual, period), ...)).
fn bias_options(l: &List) -> Result<BiasOptions> {
    let method = match get_str(l, "method")? {
        "analytical" => BiasMethod::Analytical,
        "jackknife" => {
            let variant = match get_str(l, "variant")? {
                "ss1" => Variant::Ss1,
                "ss2" => Variant::Ss2,
                "js" => Variant::Js,
                "sj" => Variant::Sj,
                "jj" => Variant::Jj,
                "double" => Variant::Double,
                v => return Err(invalid(format!("fereg: unknown jackknife variant `{v}`"))),
            };
            let partitions = list_items(l, "partitions")?
                .iter()
                .map(|p| {
                    let p: List = p.try_into().map_err(|_| invalid("fereg: invalid partition"))?;
                    Ok(Permutation {
                        individual: permutation(&field(&p, "individual"), "individual")?,
                        period: permutation(&field(&p, "period"), "period")?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            BiasMethod::Jackknife(JackknifeOptions { variant, partitions })
        }
        m => return Err(invalid(format!("fereg: unknown bias-correction method `{m}`"))),
    };
    let which = match get_str(l, "which")? {
        "both" => WhichBias::Both,
        "individual" => WhichBias::Individual,
        "time" => WhichBias::Time,
        w => return Err(invalid(format!("fereg: unknown bias `{w}`"))),
    };
    Ok(BiasOptions {
        method,
        normalization: Normalization::Alpaca,
        structure: structure(get_str(l, "structure")?)?,
        which,
        bandwidth: get_usize(l, "bandwidth")?,
        panel: None,
        time: opt_index(&field(l, "time"), "time")?,
    })
}

fn opt_reals(v: Option<&[f64]>) -> Robj {
    v.map_or_else(|| ().into(), reals)
}

fn notes(v: &[String]) -> Robj {
    Strings::from_values(v.iter().map(String::as_str)).into()
}

/// N_pre, T_pre and N* of a bias correction (`biasCorr()` draws the partitions of `multiple`).
#[extendr]
fn fereg_bias_dims(input: List, spec: List, bias: List, nthreads: i32) -> Result<Robj> {
    let inp = InputR::read(&input)?;
    let d = fit::bias_dims(&inp.fit_input()?, &spec_of(&spec)?, &bias_options(&bias)?, &ctx(nthreads)).map_err(core_err)?;
    Ok(named(vec![("n_individuals", int(d.n_individuals)), ("n_periods", int(d.n_periods)), ("n_full", int(d.n_full))]))
}

/// Bias-corrected logit estimation: list(fit = `fereg_fit()`'s output at the corrected point,
/// report).
#[extendr]
fn fereg_bias_corr(input: List, spec: List, bias: List, nthreads: i32) -> Result<Robj> {
    let inp = InputR::read(&input)?;
    let mut out = fit::bias_corr(&inp.fit_input()?, &spec_of(&spec)?, &bias_options(&bias)?, &ctx(nthreads)).map_err(core_err)?;
    let r = &out.report;
    let report = named(vec![
        ("coef_uncorr", reals(&r.coef_uncorr)),
        ("bias_term", reals(&r.bias_term)),
        (
            "which",
            match r.which {
                WhichBias::Both => "both",
                WhichBias::Individual => "individual",
                WhichBias::Time => "time",
            }
            .into(),
        ),
        ("bandwidth", int(r.bandwidth)),
        ("ape_jackknife", opt_reals(r.ape_jackknife.as_deref())),
        ("n_full", int(r.n_full)),
        ("n_fits", int(r.n_fits)),
        ("deviance_uncorr", real(r.deviance_uncorr)),
        ("notes", notes(&r.notes)),
    ]);
    // The report's notes are also in the fit's: R raises them once, from the report.
    out.fit.notes.retain(|n| !out.report.notes.contains(n));
    let fit = output(out.fit, inp.n, nthreads.max(0) as usize)?;
    Ok(named(vec![("fit", fit), ("report", report)]))
}

/// Average partial effects. `at`: NULL (uncorrected) or list(bias = `biasCorr()`'s options, coef =
/// corrected coefficients (core order), ape_jackknife = NULL or the jackknife APEs); `ape`:
/// list(n_pop = NULL or a number, structure, sampling, weak_exo).
#[extendr]
fn fereg_apes(input: List, spec: List, at: Robj, ape: List, nthreads: i32) -> Result<Robj> {
    let inp = InputR::read(&input)?;
    let sampling = match get_str(&ape, "sampling")? {
        "independence" => Sampling::Independence,
        "unrestricted" => Sampling::Unrestricted,
        s => return Err(invalid(format!("fereg: unknown sampling.fe `{s}`"))),
    };
    let opts = ApeOptions {
        n_pop: if field(&ape, "n_pop").is_null() { None } else { Some(get_f64(&ape, "n_pop")?) },
        structure: structure(get_str(&ape, "structure")?)?,
        sampling,
        weak_exo: get_bool(&ape, "weak_exo")?,
    };
    let (bias, coef_r, jk_r);
    let point = if at.is_null() {
        ApeAt::Uncorrected
    } else {
        let at: List = at.try_into().map_err(|_| invalid("fereg: invalid `at`"))?;
        let b: List = field(&at, "bias").try_into().map_err(|_| invalid("fereg: invalid `at$bias`"))?;
        bias = bias_options(&b)?;
        coef_r = field(&at, "coef");
        jk_r = field(&at, "ape_jackknife");
        ApeAt::Corrected { options: &bias, coef: real_slice(&coef_r, "coef")?, ape_jackknife: opt_real(&jk_r, "ape_jackknife")? }
    };
    let o = fit::apes(&inp.fit_input()?, &spec_of(&spec)?, point, &opts, &ctx(nthreads)).map_err(core_err)?;
    Ok(named(vec![
        ("coef_source", sources(&o.coef_source)),
        ("delta", reals(&o.delta)),
        ("vcov", matrix(&o.vcov)),
        ("bias_term", opt_reals(o.bias_term.as_deref())),
        ("binary", Logicals::from_values(o.binary.iter().map(|&b| Rbool::from(b))).into()),
        ("adj", real(o.adj)),
        ("weak_exo_used", o.weak_exo_used.into()),
        ("n_full", int(o.n_full)),
        ("notes", notes(&o.notes)),
    ]))
}

/// Variance matrix of a fitted model under another estimator / small-sample correction. `vars` is
/// list(time, panel, lat, lon) over the original input rows (as in `fereg_fit`'s input).
#[extendr]
fn fereg_vcov(state: Robj, kind: List, ssc: List, factors: List, factor_names: Strings, cluster: List, vars: List, nthreads: i32) -> Result<Robj> {
    let st = state_of(&state)?;
    let fac_r: Vec<Robj> = factors.values().collect();
    let names: Vec<String> = factor_names.iter().map(|s| s.to_string()).collect();
    if names.len() != fac_r.len() {
        return Err(invalid("fereg: one name per factor is required"));
    }
    let cols = factor_cols(&fac_r, &names)?;
    let cl = clusters(&cluster.values().collect::<Vec<Robj>>())?;
    let (time_r, panel_r, lat_r, lon_r) = (field(&vars, "time"), field(&vars, "panel"), field(&vars, "lat"), field(&vars, "lon"));
    let vv = vcov_vars(&time_r, &panel_r, &lat_r, &lon_r)?;
    let ctx = ctx(nthreads);
    let data = fit::VcovData {
        factors: &cols,
        factor_names: &names,
        cluster: &cl,
        time: vv.time,
        panel: vv.panel,
        lat: vv.lat,
        lon: vv.lon,
    };
    let v = st.vcov(vcov_kind(&kind)?, &ssc_of(&ssc)?, &data, &ctx).map_err(core_err)?;
    Ok(vcov_out(&v))
}

/// Fixed-effect values of a fitted model, computed from its state: fixest's normalization, or
/// (`min_norm`) the minimum-norm one, which always reproduces the fixed-effect sum of every
/// observation (fixef() falls back on it where fixest's references are only approximate).
#[extendr]
fn fereg_fixef(state: Robj, nthreads: i32, min_norm: bool) -> Result<Robj> {
    let st = state_of(&state)?;
    let norm = if min_norm { FixefNorm::MinNorm } else { FixefNorm::Fixest };
    let values = st.fixef(norm, &ctx(nthreads)).map_err(core_err)?;
    Ok(fixef_list(&values))
}

/// fixest's references per term (term order) with the terms taken in `order` (1-based term
/// indices).
#[extendr]
fn fereg_fixef_references(state: Robj, order: Robj, nthreads: i32) -> Result<Robj> {
    let st = state_of(&state)?;
    let refs = st.fixef_references(&indices(&order, "order")?, &ctx(nthreads)).map_err(core_err)?;
    Ok(Integers::from_values(refs.iter().map(|&r| r as i32)).into())
}

/// The estimation state behind an external pointer (NULL after `lean = TRUE`; a null pointer
/// after the object was saved and reloaded).
fn state_of(state: &Robj) -> Result<&FitState> {
    let unavailable = || {
        invalid("fereg: the estimation state is not available (lean = TRUE, or the object was saved and reloaded); re-estimate the model")
    };
    let st: &ExternalPtr<FitState> = state.try_into().map_err(|_| unavailable())?;
    st.try_addr().map_err(|_| unavailable())
}

/// Version of the bundled fereg-core.
#[extendr]
fn fereg_core_version() -> &'static str {
    fereg_core::VERSION
}

extendr_module! {
    mod fereg;
    fn fereg_fit;
    fn fereg_vcov;
    fn fereg_fixef;
    fn fereg_fixef_references;
    fn fereg_bias_dims;
    fn fereg_bias_corr;
    fn fereg_apes;
    fn fereg_core_version;
}
