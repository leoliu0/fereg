//! Bias correction and average partial effects of fixed-effects logit in the fit layer (alpaca
//! names: `biasCorr`, `getAPEs`), for the R and Python packages.
//!
//! Hosts pass the same [`FitInput`] and [`FitSpec`] as for the uncorrected [`super::fit`]: the
//! uncorrected model is estimated again (bit-identical), corrected through the kernels of
//! [`crate::bias`] with the session's absorber, and [`bias_corr`] returns a complete
//! [`FitOutput`] at the corrected point (coefficients, eta with the fixed effects re-estimated,
//! variance under `spec.vcov`, statistics, fixed effects, and a state for post-hoc variance
//! matrices and fixed effects).
//!
//! Samples: n = the rows of the session (the rows whose fixed effects all have finite estimates;
//! pyfixest-flavor rows reported at the bound are not among them), N* = the input rows without
//! missing values and zero weights (before singleton, perfect-fit and separation drops). The
//! jackknife subpanels are masks over those N* rows.

use super::glm::{self, Estimated};
use super::input;
use super::{CoefSource, FactorCol, FitInput, FitOutput, FitSpec, Flavor, Model, VcovKind, WeightsType};
use crate::absorber::FeCoefs;
use crate::api::{Ctx, Family};
use crate::bias::{self, BiasSpec, FeCodes, FullSample, JkPanel, JkSpec, LogitData, LogitfePanel, SubFit};
pub use crate::bias::{Normalization, Permutation, Sampling, Structure, Variant, WhichBias};
use crate::error::RfError;
use crate::factor::Codes;
use crate::glm::family;
use crate::session::Term;
use crate::par::{self, Pool};

/// Options of [`bias_corr`] (alpaca `biasCorr`).
#[derive(Clone, Debug, PartialEq)]
pub struct BiasOptions {
    pub method: BiasMethod,
    /// R and Python: [`Normalization::Alpaca`]. [`Normalization::Logitfe`] needs `panel` and `time`.
    pub normalization: Normalization,
    pub structure: Structure,
    pub which: WhichBias,
    /// Bandwidth L (analytical only).
    pub bandwidth: usize,
    /// Factor index of the panel (individual) variable: [`Normalization::Logitfe`] only.
    pub panel: Option<usize>,
    /// Factor index of the time variable: [`Normalization::Logitfe`] (calendar periods: integer
    /// values, lag j = value - j), and the time dimension of jackknife subpanels of one-FE models.
    pub time: Option<usize>,
}

impl Default for BiasOptions {
    fn default() -> Self {
        BiasOptions {
            method: BiasMethod::Analytical,
            normalization: Normalization::Alpaca,
            structure: Structure::Classic,
            which: WhichBias::Both,
            bandwidth: 0,
            panel: None,
            time: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum BiasMethod {
    Analytical,
    Jackknife(JackknifeOptions),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct JackknifeOptions {
    pub variant: Variant,
    /// `multiple(R)`: one permutation pair per partition (ss1 / ss2 only); empty = none.
    pub partitions: Vec<Permutation>,
}

/// Index spaces of the jackknife and N* ([`bias_dims`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BiasDims {
    /// N_pre: individuals over the N* rows (0 when the model has no individual dimension).
    pub n_individuals: usize,
    /// T_pre: periods over the N* rows (0 without a time dimension).
    pub n_periods: usize,
    /// N*.
    pub n_full: usize,
}

/// Result of [`bias_corr`].
pub struct BiasFit {
    /// The fit at the corrected point.
    pub fit: FitOutput,
    pub report: BiasReport,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BiasReport {
    /// Uncorrected coefficients ([`FitOutput::coef`] order).
    pub coef_uncorr: Vec<f64>,
    /// beta_uncorr - beta_corrected (alpaca's `bias.term`).
    pub bias_term: Vec<f64>,
    pub structure: Structure,
    /// The bias corrected after resolving it against the model (one FE term: its own bias).
    pub which: WhichBias,
    /// Bandwidth that entered the correction (0 when it does not apply).
    pub bandwidth: usize,
    pub method: BiasMethod,
    /// Jackknife: the combination of the subpanel APEs (regressor order).
    pub ape_jackknife: Option<Vec<f64>>,
    /// Jackknife: the uncorrected APEs of the full sample entering the combination.
    pub ape_uncorr: Option<Vec<f64>>,
    pub n_full: usize,
    /// Jackknife subpanel fits run.
    pub n_fits: usize,
    /// Deviance of the uncorrected fit over the estimation rows of the session.
    pub deviance_uncorr: f64,
    /// Newton iterations of the fixed-effect re-estimation at the corrected coefficients.
    pub resolve_iterations: usize,
    pub notes: Vec<String>,
}

/// Options of [`apes`] (alpaca `getAPEs`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ApeOptions {
    /// Population size: adj = (n_pop - N*)/(n_pop - 1); `None` = 0; below N*: note and 0.
    pub n_pop: Option<f64>,
    /// Ignored at a corrected point (taken from its options).
    pub structure: Structure,
    pub sampling: Sampling,
    /// Ignored at an analytically corrected point (bandwidth > 0, as alpaca).
    pub weak_exo: bool,
}

/// Where [`apes`] evaluates.
#[derive(Clone, Copy, Debug)]
pub enum ApeAt<'a> {
    Uncorrected,
    /// A [`bias_corr`] result: its options, its coefficients and, for the jackknife,
    /// [`BiasReport::ape_jackknife`].
    Corrected { options: &'a BiasOptions, coef: &'a [f64], ape_jackknife: Option<&'a [f64]> },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ApeOutput {
    /// [`CoefSource::X`] of every APE (the kept regressors).
    pub coef_source: Vec<CoefSource>,
    pub delta: Vec<f64>,
    /// p x p row-major.
    pub vcov: Vec<f64>,
    /// Analytically corrected point: the APE bias term (delta = delta(beta_c, eta_c) - bias_term).
    pub bias_term: Option<Vec<f64>>,
    pub binary: Vec<bool>,
    pub adj: f64,
    pub weak_exo_used: bool,
    pub structure: Structure,
    pub bandwidth: usize,
    pub n_full: usize,
    pub notes: Vec<String>,
}

fn invalid<T>(m: impl Into<String>) -> Result<T, RfError> {
    Err(RfError::InvalidInput(m.into()))
}

/// Request checks common to every entry point.
fn check(input: &FitInput, spec: &FitSpec, structure: Structure) -> Result<(), RfError> {
    if spec.model != Model::Logit {
        return invalid("bias correction and APEs are only available for logit models");
    }
    if input.fe.is_empty() {
        return invalid("bias correction and APEs need fixed effects");
    }
    if input.fe.iter().any(|t| !t.intercept || !t.slopes.is_empty()) {
        return invalid("bias correction and APEs need intercept-only fixed effects (no varying slopes)");
    }
    if !input.endog.is_empty() {
        return invalid("bias correction is not available for IV models");
    }
    if spec.weights_type == WeightsType::Fweights && input.weights.is_some() {
        return invalid("bias correction and APEs do not support frequency weights");
    }
    bias::check_structure(structure, input.fe.len())
}

/// Ascending input rows without missing values and zero weights (the N* rows).
fn pre_rows(input: &FitInput, spec: &FitSpec, pool: &Pool) -> Result<Vec<u32>, RfError> {
    let (alive, _) = input::scan(input, spec, pool)?;
    Ok(alive.iter().enumerate().filter(|(_, a)| **a).map(|(i, _)| i as u32).collect())
}

fn codes_on(input: &FitInput, factors: &[usize], rows: &[u32], pool: &Pool) -> Result<Codes, RfError> {
    input::interaction_kept_codes(&input.factors, factors, rows, pool)
}

/// Error when the factor `f` (the panel or time variable, `what`) is missing on one of `rows`.
fn no_missing(input: &FitInput, f: usize, rows: &[u32], what: &str) -> Result<(), RfError> {
    if f >= input.factors.len() {
        return invalid("the panel or time variable references an invalid factor");
    }
    if rows.iter().any(|&i| input::factor_missing(&input.factors[f], i as usize)) {
        return invalid(format!("the {what} variable has missing values in the estimation sample"));
    }
    Ok(())
}

/// Integer value of a factor column at row i (calendar time).
fn int_value(col: &FactorCol, i: usize) -> Result<i64, RfError> {
    Ok(match *col {
        FactorCol::F64(v) => {
            let x = v[i];
            if x.round() != x || x.abs() > 9e15 {
                return invalid("the time variable must hold integer values");
            }
            x as i64
        }
        FactorCol::I32(v) => i64::from(v[i]),
        FactorCol::I64(v) => v[i],
        FactorCol::Codes { codes, base, .. } => i64::from(codes[i]) - i64::from(base),
    })
}

/// Kept-order columns of the session's sample.
struct Kept {
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
    offset: Option<Vec<f64>>,
    /// Input index of every kept regressor.
    xj: Vec<usize>,
}

fn kept_columns(input: &FitInput, est: &Estimated) -> Result<Kept, RfError> {
    if est.ones || est.intercept {
        return Err(RfError::Internal("bias correction: unexpected intercept with fixed effects".into()));
    }
    let kept = &est.session.kept_index;
    let at = |v: &[f64]| kept.iter().map(|&i| v[i as usize]).collect::<Vec<f64>>();
    let xj: Vec<usize> = (0..est.nsent).filter(|&j| est.parts.status[j] == 0).collect();
    if xj.is_empty() {
        return invalid("bias correction needs at least one regressor");
    }
    Ok(Kept { y: at(input.y), x: xj.iter().map(|&j| at(input.x[j])).collect(), offset: input.offset.map(at), xj })
}

fn logit_data<'a>(k: &'a Kept, terms: &'a [Term], weights: Option<&'a [f64]>, n_full: usize) -> LogitData<'a> {
    LogitData {
        y: &k.y,
        weights,
        offset: k.offset.as_deref(),
        x: k.x.iter().map(Vec::as_slice).collect(),
        fe: terms.iter().map(|t| FeCodes { codes: &t.codes, g: t.g }).collect(),
        n_full,
    }
}

/// Owned identifiers of the logitfe normalization over the kept rows.
struct LfIds {
    individual: Vec<u32>,
    n_individuals: usize,
    period: Vec<i64>,
    individual_effects: bool,
    time_effects: bool,
}

fn logitfe_ids(input: &FitInput, opts: &BiasOptions, kept: &[u32], pool: &Pool) -> Result<LfIds, RfError> {
    let (Some(pf), Some(tf)) = (opts.panel, opts.time) else {
        return invalid("the logitfe normalization needs the panel and time variables");
    };
    if pf >= input.factors.len() || tf >= input.factors.len() {
        return invalid("the panel or time variable references an invalid factor");
    }
    no_missing(input, pf, kept, "panel")?;
    no_missing(input, tf, kept, "time")?;
    let individual_effects = input.fe.iter().any(|t| t.factors == [pf]);
    let time_effects = input.fe.iter().any(|t| t.factors == [tf]);
    if input.fe.iter().any(|t| t.factors != [pf] && t.factors != [tf]) {
        return invalid("the logitfe normalization needs fixed effects on the panel and/or time variable only");
    }
    let c = codes_on(input, &[pf], kept, pool)?;
    let period = kept.iter().map(|&i| int_value(&input.factors[tf], i as usize)).collect::<Result<_, _>>()?;
    Ok(LfIds { individual: c.codes, n_individuals: c.n as usize, period, individual_effects, time_effects })
}

fn lf_panel(ids: &LfIds) -> LogitfePanel<'_> {
    LogitfePanel {
        individual: &ids.individual,
        n_individuals: ids.n_individuals,
        period: &ids.period,
        individual_effects: ids.individual_effects,
        time_effects: ids.time_effects,
    }
}

/// The lag-group term of the request: the panel variable's term under the logitfe normalization
/// (the terms may come in any order), FE1 / FE3 otherwise.
fn lag_of(input: &FitInput, opts: &BiasOptions) -> Option<usize> {
    match opts.normalization {
        Normalization::Logitfe => opts.panel.and_then(|pf| input.fe.iter().position(|t| t.factors == [pf])),
        Normalization::Alpaca => None,
    }
}

/// The bias that exists in the model for the requested one.
fn resolve_which(input: &FitInput, opts: &BiasOptions) -> Result<WhichBias, RfError> {
    let (ind, time) = match opts.normalization {
        Normalization::Alpaca => (true, input.fe.len() == 2 && opts.structure == Structure::Classic),
        Normalization::Logitfe => {
            let has = |f: Option<usize>| f.is_some_and(|f| input.fe.iter().any(|t| t.factors == [f]));
            (has(opts.panel), has(opts.time))
        }
    };
    if opts.structure == Structure::Network {
        return match opts.which {
            WhichBias::Both => Ok(WhichBias::Both),
            _ => invalid("network structure: only the correction of both biases is available"),
        };
    }
    match opts.which {
        WhichBias::Both if ind && time => Ok(WhichBias::Both),
        WhichBias::Both if ind => Ok(WhichBias::Individual),
        WhichBias::Both => Ok(WhichBias::Time),
        WhichBias::Individual if !ind => invalid("the individual-effect bias needs individual fixed effects"),
        WhichBias::Time if !time => invalid("the time-effect bias needs a time fixed effect (a two-way model)"),
        w => Ok(w),
    }
}

/// N_pre, T_pre and N* of the request (hosts draw `multiple(R)` permutations over them).
pub fn bias_dims(input: &FitInput, spec: &FitSpec, opts: &BiasOptions, ctx: &Ctx) -> Result<BiasDims, RfError> {
    check(input, spec, opts.structure)?;
    input::validate(input, spec)?;
    par::with_pool(par::threads_for(ctx.threads, input.n), |pool| {
        let pre = pre_rows(input, spec, pool)?;
        let (ind, per) = jk_dims(input, opts, &pre, pool)?;
        Ok(BiasDims { n_individuals: ind.map_or(0, |c| c.n as usize), n_periods: per.map_or(0, |c| c.n as usize), n_full: pre.len() })
    })?
}

/// Individual and period codes over the pre-drop rows.
fn jk_dims(input: &FitInput, opts: &BiasOptions, pre: &[u32], pool: &Pool) -> Result<(Option<Codes>, Option<Codes>), RfError> {
    if opts.structure == Structure::Network {
        return Ok((None, None));
    }
    let check_f = |f: usize, what: &str| no_missing(input, f, pre, what).map(|_| f);
    let (ind_f, per_f): (Option<Vec<usize>>, Option<Vec<usize>>) = match opts.normalization {
        Normalization::Alpaca => (
            Some(input.fe[0].factors.clone()),
            if input.fe.len() == 2 { Some(input.fe[1].factors.clone()) } else { opts.time.map(|f| check_f(f, "time")).transpose()?.map(|f| vec![f]) },
        ),
        Normalization::Logitfe => (
            opts.panel.map(|f| check_f(f, "panel")).transpose()?.map(|f| vec![f]),
            opts.time.map(|f| check_f(f, "time")).transpose()?.map(|f| vec![f]),
        ),
    };
    let ind = ind_f.map(|f| codes_on(input, &f, pre, pool)).transpose()?;
    let per = per_f.map(|f| codes_on(input, &f, pre, pool)).transpose()?;
    Ok((ind, per))
}

/// [`crate::bias::jackknife`] double ids: FE1 / FE2 values among those occurring in both.
fn double_ids(input: &FitInput, pre: &[u32]) -> Result<(Vec<u32>, Vec<u32>, usize), RfError> {
    if input.fe.len() != 2 || input.fe.iter().any(|t| t.factors.len() != 1) {
        return invalid("the double jackknife needs two single-variable fixed effects");
    }
    let (a, b) = (&input.factors[input.fe[0].factors[0]], &input.factors[input.fe[1].factors[0]]);
    let is_codes = |c: &FactorCol| matches!(c, FactorCol::Codes { .. });
    if is_codes(a) != is_codes(b) {
        return invalid("the double jackknife needs both fixed-effect variables of the same kind (numeric or coded on common levels)");
    }
    let val = |c: &FactorCol, i: usize| -> f64 {
        match *c {
            FactorCol::F64(v) => v[i],
            FactorCol::I32(v) => f64::from(v[i]),
            FactorCol::I64(v) => v[i] as f64,
            FactorCol::Codes { codes, base, .. } => f64::from(codes[i] - base),
        }
    };
    let va: Vec<f64> = pre.iter().map(|&i| val(a, i as usize)).collect();
    let vb: Vec<f64> = pre.iter().map(|&i| val(b, i as usize)).collect();
    let uniq = |v: &[f64]| {
        let mut u = v.to_vec();
        u.sort_by(f64::total_cmp);
        u.dedup();
        u
    };
    let (ua, ub) = (uniq(&va), uniq(&vb));
    let common: Vec<f64> = ua.into_iter().filter(|x| ub.binary_search_by(|y| y.total_cmp(x)).is_ok()).collect();
    let id = |x: &f64| common.binary_search_by(|y| y.total_cmp(x)).map_or(u32::MAX, |p| p as u32);
    Ok((va.iter().map(id).collect(), vb.iter().map(id).collect(), common.len()))
}

/// Working memory of one subpanel fit (as `crv3::refit_bytes`).
fn fit_bytes(input: &FitInput, p: usize) -> usize {
    let r = 2 + p + usize::from(input.weights.is_some()) + usize::from(input.offset.is_some());
    input.n * (16 * r + 32 * input.fe.len() + 96)
}

/// Bias-corrected logit fit (design secs. 3-4, 7).
pub fn bias_corr(input: &FitInput, spec: &FitSpec, opts: &BiasOptions, ctx: &Ctx) -> Result<BiasFit, RfError> {
    check(input, spec, opts.structure)?;
    if spec.vcov == VcovKind::Crv3 {
        return invalid("CRV3 is not available after bias correction");
    }
    input::validate(input, spec)?;
    let which = resolve_which(input, opts)?;
    par::with_pool(par::threads_for(ctx.threads, input.n), |pool| {
        let pre = pre_rows(input, spec, pool)?;
        let sample = input::build(input, spec, ctx, pool)?;
        let mut est = glm::estimate(input, spec, sample, ctx, pool)?;
        if !est.parts.converged {
            return invalid("bias correction: the uncorrected fit did not converge");
        }
        let report = correct(input, spec, opts, which, &pre, &mut est, ctx, pool)?;
        est.bias_corrected = true;
        let mut fit = glm::output(input, spec, est, ctx, pool)?;
        fit.notes.extend(report.notes.iter().cloned());
        Ok(BiasFit { fit, report })
    })?
}

#[allow(clippy::too_many_arguments)]
fn correct(input: &FitInput, spec: &FitSpec, opts: &BiasOptions, which: WhichBias, pre: &[u32], est: &mut Estimated, ctx: &Ctx, pool: &Pool) -> Result<BiasReport, RfError> {
    let kc = kept_columns(input, est)?;
    let coef0: Vec<f64> = kc.xj.iter().map(|&j| est.parts.coef[j]).collect();
    let eta0 = est.parts.eta.clone();
    let deviance_uncorr = est.parts.deviance;
    let n_full = pre.len();
    let lf = match opts.normalization {
        Normalization::Logitfe => Some(logitfe_ids(input, opts, &est.session.kept_index, pool)?),
        Normalization::Alpaca => None,
    };
    let mut notes = Vec::new();
    let (coef_c, eta_c, fe_coefs, bandwidth, ape_jk, ape_uncorr, n_fits, iterations) = match &opts.method {
        BiasMethod::Analytical => {
            let d = logit_data(&kc, &est.session.terms, est.session.weights.as_deref(), n_full);
            let bspec = BiasSpec {
                normalization: opts.normalization,
                structure: opts.structure,
                which,
                bandwidth: opts.bandwidth,
                lag_term: lag_of(input, opts),
                order: None,
                logitfe: lf.as_ref().map(lf_panel),
            };
            let bw = bias::plan_bandwidth(&d, &bspec)?;
            let r = bias::analytical(&d, est.session.absorber.as_mut(), &coef0, &eta0, &bspec, ctx)?;
            notes.extend(r.notes);
            (r.coef, r.eta, r.fe_coefs, bw, None, None, 0, r.iterations)
        }
        BiasMethod::Jackknife(jo) => {
            if opts.structure == Structure::Network {
                return invalid("the jackknife is not available for the network structure");
            }
            if opts.bandwidth > 0 {
                return invalid(format!("bandwidth L = {} has no effect: it only applies to the analytical correction", opts.bandwidth));
            }
            let binary: Vec<bool> = kc.x.iter().map(|x| x.iter().all(|&v| v == 0.0 || v == 1.0)).collect();
            let xs: Vec<&[f64]> = kc.x.iter().map(Vec::as_slice).collect();
            let ape0 = bias::ape_point(&xs, &binary, &coef0, &eta0, n_full as f64);
            let (ind, per) = jk_dims(input, opts, pre, pool)?;
            let dbl = (jo.variant == Variant::Double && which == WhichBias::Both).then(|| double_ids(input, pre)).transpose()?;
            let panel = JkPanel {
                individual: ind.as_ref().map(|c| c.codes.as_slice()),
                n_individuals: ind.as_ref().map_or(0, |c| c.n as usize),
                period: per.as_ref().map(|c| c.codes.as_slice()),
                n_periods: per.as_ref().map_or(0, |c| c.n as usize),
                double: dbl.as_ref().map(|(a, b, j)| (a.as_slice(), b.as_slice(), *j)),
            };
            let jspec = JkSpec { variant: jo.variant, which, partitions: &jo.partitions, bytes_per_fit: fit_bytes(input, coef0.len()) };
            let full = SubFit { coef: coef0.clone(), ape: ape0.clone() };
            let closure = SubFitter::new(input, spec, &kc.xj, &binary, pre, &est.session.kept_index, &full);
            let out = bias::jackknife(&panel, &jspec, &full, |_, keep, fctx| closure.fit(keep, fctx), ctx)?;
            notes.extend(out.notes);
            let d = logit_data(&kc, &est.session.terms, est.session.weights.as_deref(), n_full);
            let start = bias::shifted_eta(&d, &eta0, &coef0, &out.coef);
            let r = bias::resolve_fe(&d, est.session.absorber.as_mut(), &out.coef, &start, ctx)?;
            (out.coef, r.eta, r.fe_coefs, 0, Some(out.ape), Some(ape0), out.n_fits, r.iterations)
        }
    };
    rewrite(est, &kc, &coef_c, eta_c, &fe_coefs, ctx, pool)?;
    let bias_term = coef0.iter().zip(&coef_c).map(|(a, b)| a - b).collect();
    Ok(BiasReport {
        coef_uncorr: coef0,
        bias_term,
        structure: opts.structure,
        which,
        bandwidth,
        method: opts.method.clone(),
        ape_jackknife: ape_jk,
        ape_uncorr,
        n_full,
        n_fits,
        deviance_uncorr,
        resolve_iterations: iterations,
        notes,
    })
}

/// Rewrites the GLM parts at (coef, eta): the fit's outputs are then evaluated there.
#[allow(clippy::too_many_arguments)]
fn rewrite(est: &mut Estimated, kc: &Kept, coef: &[f64], eta: Vec<f64>, fe_coefs: &FeCoefs, ctx: &Ctx, pool: &Pool) -> Result<(), RfError> {
    let n = eta.len();
    let session = &mut est.session;
    let parts = &mut est.parts;
    // Saved fixed effects first: `term_values` normalizes under the absorber's current weights
    // (those of the solve that produced `fe_coefs`).
    if session.terms.iter().any(|t| t.save) {
        parts.fe = crate::fe_recovery::saved_fe(session.absorber.as_ref(), &session.terms, fe_coefs, 0.0);
    }
    let wt = |r: usize| session.weights.as_ref().map_or(1.0, |w| w[r]);
    let mut mu = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut wr = vec![0.0; n];
    let mut fe_sum = vec![0.0; n];
    let mut dev = 0.0;
    for r in 0..n {
        let m = family::mean(Family::Logit, eta[r]);
        mu[r] = m;
        w[r] = wt(r) * m * (1.0 - m);
        wr[r] = (kc.y[r] - m) / (m * (1.0 - m));
        let xb: f64 = kc.x.iter().zip(coef).map(|(x, c)| x[r] * c).sum();
        fe_sum[r] = eta[r] - kc.offset.as_ref().map_or(0.0, |o| o[r]) - xb;
        dev -= 2.0 * wt(r) * family::logit_ll(kc.y[r], eta[r]);
    }
    let xs: Vec<&[f64]> = kc.x.iter().map(Vec::as_slice).collect();
    let mx = bias::center(session.absorber.as_mut(), &xs, &w, pool, ctx)?;
    let k = mx.len();
    let mut xt = par::alloc_vec(n * k, 0.0)?;
    for r in 0..n {
        for j in 0..k {
            xt[r * k + j] = mx[j][r];
        }
    }
    for (&j, &c) in kc.xj.iter().zip(coef) {
        parts.coef[j] = c;
    }
    parts.eta = eta;
    parts.mu = mu;
    parts.w = w;
    parts.k = k;
    parts.xt = xt;
    parts.working_resid = wr;
    parts.fe_sum = fe_sum;
    parts.deviance = dev;
    Ok(())
}

/// The fit-layer subpanel fit: `fit::fit` on the pre-drop rows of the mask with the request's
/// sample rules, coefficients picked by source, APEs over its estimation rows divided by its own N*.
struct SubFitter<'a> {
    base: FitInput<'a>,
    spec: FitSpec,
    xj: &'a [usize],
    binary: &'a [bool],
    pre: &'a [u32],
    /// Per pre-drop row: in the full fit's estimation sample.
    in_full: Vec<bool>,
    /// Fixest flavor: a subpanel keeping every estimation row has the full fit's sample (the
    /// iterated drops remove exactly the other rows), so its fit is the full fit.
    shortcut: bool,
    full: &'a SubFit,
    drop0: Vec<bool>,
}

impl<'a> SubFitter<'a> {
    fn new(input: &FitInput<'a>, spec: &FitSpec, xj: &'a [usize], binary: &'a [bool], pre: &'a [u32], kept: &[u32], full: &'a SubFit) -> SubFitter<'a> {
        let base = FitInput { cluster: Vec::new(), time: None, panel: None, lat: None, lon: None, drop: None, ..input.clone() };
        let spec = FitSpec { vcov: VcovKind::Iid, keep_resid: true, keep_fixef: false, keep_state: false, ..spec.clone() };
        let in_full = pre.iter().map(|i| kept.binary_search(i).is_ok()).collect();
        let shortcut = spec.flavor == Flavor::Fixest;
        // Every row outside the N* rows stays dropped (also rows the full fit excluded only for a
        // missing variance-matrix variable, which `base` no longer carries); subpanels drop the
        // masked-out N* rows on top.
        let mut drop0 = vec![true; input.n];
        pre.iter().for_each(|&i| drop0[i as usize] = false);
        SubFitter { base, spec, xj, binary, pre, in_full, shortcut, full, drop0 }
    }

    fn fit(&self, keep: &[bool], ctx: &Ctx) -> Result<SubFit, RfError> {
        let n_sub = keep.iter().filter(|k| **k).count();
        if n_sub == 0 {
            return Err(RfError::TooFewObs);
        }
        if self.shortcut && keep.iter().zip(&self.in_full).all(|(k, f)| *k || !*f) {
            let s = self.full.ape.len();
            let scale = self.pre.len() as f64 / n_sub as f64;
            return Ok(SubFit { coef: self.full.coef.clone(), ape: (0..s).map(|j| self.full.ape[j] * scale).collect() });
        }
        let mut drop = self.drop0.clone();
        for (&i, &k) in self.pre.iter().zip(keep) {
            if !k {
                drop[i as usize] = true;
            }
        }
        let out = super::fit(&FitInput { drop: Some(&drop), ..self.base.clone() }, &self.spec, ctx)?;
        if !out.stats.converged {
            return invalid(format!("no convergence in {} IRLS iterations", self.spec.glm_maxiter));
        }
        let mut coef = Vec::with_capacity(self.xj.len());
        for &j in self.xj {
            match out.coef_source.iter().position(|s| *s == CoefSource::X(j)) {
                Some(a) => coef.push(out.coef[a]),
                None => return invalid(format!("the coefficient of regressor {} (x[{j}]) is not estimable", j + 1)),
            }
        }
        let eta = out.eta.as_ref().ok_or_else(|| RfError::Internal("subpanel fit returned no eta".into()))?;
        let xs: Vec<Vec<f64>> = self.xj.iter().map(|&j| out.obs.iter().map(|&i| self.base.x[j][i as usize]).collect()).collect();
        let xr: Vec<&[f64]> = xs.iter().map(Vec::as_slice).collect();
        let ape = bias::ape_point(&xr, self.binary, &coef, eta, n_sub as f64);
        Ok(SubFit { coef, ape })
    }
}

/// APEs of a logit fit (design secs. 5-6) at the uncorrected point or at a [`bias_corr`] result.
pub fn apes(input: &FitInput, spec: &FitSpec, at: ApeAt, opts: &ApeOptions, ctx: &Ctx) -> Result<ApeOutput, RfError> {
    apes_impl(input, spec, at, opts, false, ctx)
}

/// [`apes`] with alpaca's row set for the variance sums (tests).
#[cfg(test)]
pub(crate) fn apes_retained(input: &FitInput, spec: &FitSpec, at: ApeAt, opts: &ApeOptions, ctx: &Ctx) -> Result<ApeOutput, RfError> {
    apes_impl(input, spec, at, opts, true, ctx)
}

fn apes_impl(input: &FitInput, spec: &FitSpec, at: ApeAt, opts: &ApeOptions, retained: bool, ctx: &Ctx) -> Result<ApeOutput, RfError> {
    let structure = match at {
        ApeAt::Uncorrected => opts.structure,
        ApeAt::Corrected { options, .. } => options.structure,
    };
    check(input, spec, structure)?;
    input::validate(input, spec)?;
    let which = match at {
        ApeAt::Corrected { options, .. } => Some(resolve_which(input, options)?),
        ApeAt::Uncorrected => None,
    };
    par::with_pool(par::threads_for(ctx.threads, input.n), |pool| {
        let pre = pre_rows(input, spec, pool)?;
        let n_full = pre.len();
        let sample = input::build(input, spec, ctx, pool)?;
        let mut est = glm::estimate(input, spec, sample, ctx, pool)?;
        let kc = kept_columns(input, &est)?;
        let coef0: Vec<f64> = kc.xj.iter().map(|&j| est.parts.coef[j]).collect();
        let eta0 = est.parts.eta.clone();
        let mut notes = Vec::new();
        let adj = match opts.n_pop {
            None => 0.0,
            Some(m) if m.is_nan() || m < 0.0 => return invalid("n_pop must be a non-negative number (or +Inf for an infinite population)"),
            Some(m) if m < n_full as f64 => {
                notes.push("Size of the entire population is lower than the full sample size. Correction factor set to zero.".to_string());
                0.0
            }
            Some(m) if m.is_infinite() => 1.0,
            Some(m) => (m - n_full as f64) / (m - 1.0),
        };
        // Pre-drop rows: kept index and FE codes (sampling term), time order (logitfe).
        let kept = &est.session.kept_index;
        let full_kept: Vec<u32> = pre.iter().map(|i| kept.binary_search(i).map_or(u32::MAX, |p| p as u32)).collect();
        let full_codes: Vec<Codes> = if adj > 0.0 && !retained { input.fe.iter().map(|t| codes_on(input, &t.factors, &pre, pool)).collect::<Result<_, _>>()? } else { Vec::new() };
        let opts_c = match at {
            ApeAt::Corrected { options, .. } => Some(options),
            ApeAt::Uncorrected => None,
        };
        let lf = match opts_c {
            Some(o) if o.normalization == Normalization::Logitfe => Some(logitfe_ids(input, o, kept, pool)?),
            _ => None,
        };
        let full_order: Option<Vec<i64>> = match opts_c {
            Some(o) if o.normalization == Normalization::Logitfe && adj > 0.0 && !retained => {
                let tf = o.time.expect("checked by logitfe_ids");
                no_missing(input, tf, &pre, "time")?;
                Some(pre.iter().map(|&i| int_value(&input.factors[tf], i as usize)).collect::<Result<_, _>>()?)
            }
            _ => None,
        };
        let d = logit_data(&kc, &est.session.terms, est.session.weights.as_deref(), n_full);
        let (coef, eta, correction, weak_exo, bandwidth, jk_delta) = match at {
            ApeAt::Uncorrected => (coef0.clone(), eta0.clone(), None, opts.weak_exo, 0, None),
            ApeAt::Corrected { options, coef, ape_jackknife } => {
                if coef.len() != coef0.len() {
                    return invalid("APEs: the corrected coefficients do not match the model's regressors");
                }
                let start = bias::shifted_eta(&d, &eta0, &coef0, coef);
                let r = bias::resolve_fe(&d, est.session.absorber.as_mut(), coef, &start, ctx)?;
                match &options.method {
                    BiasMethod::Analytical => {
                        let bspec = BiasSpec {
                            normalization: options.normalization,
                            structure: options.structure,
                            which: which.expect("corrected"),
                            bandwidth: options.bandwidth,
                            lag_term: lag_of(input, options),
                            order: None,
                            logitfe: lf.as_ref().map(lf_panel),
                        };
                        let bw = bias::plan_bandwidth(&d, &bspec)?;
                        (coef.to_vec(), r.eta, Some(bspec), options.bandwidth > 0, bw, None)
                    }
                    BiasMethod::Jackknife(_) => {
                        let Some(dj) = ape_jackknife else { return invalid("APEs after a jackknife correction need its APE combination") };
                        if dj.len() != coef0.len() {
                            return invalid("APEs: the jackknife APEs do not match the model's regressors");
                        }
                        (coef.to_vec(), r.eta, None, opts.weak_exo, 0, Some(dj.to_vec()))
                    }
                }
            }
        };
        let full = (adj > 0.0 && !retained).then(|| FullSample {
            kept: &full_kept,
            fe: full_codes.iter().map(|c| FeCodes { codes: &c.codes, g: c.n as usize }).collect(),
            order: full_order.as_deref(),
        });
        let lag_term = opts_c.and_then(|o| lag_of(input, o));
        let aspec = bias::ApeSpec { structure, sampling: opts.sampling, weak_exo, lag_term, adj, correction, full, want_vcov: true };
        #[cfg(test)]
        let a = if retained {
            bias::apes_retained(&d, est.session.absorber.as_mut(), &coef, &eta, &aspec, ctx)?
        } else {
            bias::apes(&d, est.session.absorber.as_mut(), &coef, &eta, &aspec, ctx)?
        };
        #[cfg(not(test))]
        let a = {
            let _ = retained;
            bias::apes(&d, est.session.absorber.as_mut(), &coef, &eta, &aspec, ctx)?
        };
        Ok(ApeOutput {
            coef_source: kc.xj.iter().map(|&j| CoefSource::X(j)).collect(),
            delta: jk_delta.unwrap_or(a.delta),
            vcov: a.vcov,
            bias_term: a.bias_term,
            binary: a.binary,
            adj,
            weak_exo_used: weak_exo,
            structure,
            bandwidth,
            n_full,
            notes,
        })
    })?
}

#[cfg(test)]
#[path = "bias_tests.rs"]
mod tests;
