//! Poisson and logit models in the fit layer (fixest `fepois` / `feglm(family = "logit")`,
//! pyfixest `fepois` / `feglm`). IRLS runs in the GLM core under the fixest convention
//! ([`Session::glm_parts`]); this module turns its per-observation parts into the fit layer's
//! outputs: sample accounting after GLM-internal drops, coefficients and intercept, the variance
//! matrix through the shared fixest code, fit statistics, fixed-effect values and the state.

use rayon::prelude::*;

use super::extra::{self, JkData, VcovInputs, fe_leverage, jk_model};
use super::fixef::{self, FeLevels, FeSubset};
use super::input::{Excluded, Sample};
use super::ols::{cor_part, sq_cor, y_stats};
use super::state::{State, StateFe, term_refs};
use super::vcov::{self, ClusterInfo, FeInfo, Jackknife, Model as VModel};
use super::{CoefSource, FitInput, FitOutput, FitSpec, FitStats, Flavor, Model, TestStat, VcovData, VcovKind, WeightsType, dist};
use crate::api::{AbsorbTerm, Ctx, DofOptions, Family, GlmCollinear, GlmConvention, GlmOptions, GlmParts, Guess, Layout, Separation, Spec, Vce, WeightKind};
use crate::error::RfError;
use crate::factor::{self, Codes};
use crate::glm::family::MU_MIN;
use crate::gather::Col;
use crate::par::{self, CHUNK, Pool};
use crate::leverage::FeCols;
use crate::session::{AbsorberKind, Parts, Session, Term};
use crate::stats;
use crate::vcov::chol_collin;

/// Checks the outcome on the estimation rows: Poisson y >= 0, logit y in {0, 1}.
fn check_outcome(y: &[f64], kept: &[u32], family: Family, pool: &Pool) -> Result<(), RfError> {
    let bad = pool.install(|| match family {
        Family::Logit => kept.par_iter().any(|&i| y[i as usize] != 0.0 && y[i as usize] != 1.0),
        _ => kept.par_iter().any(|&i| y[i as usize] < 0.0),
    });
    match (bad, family) {
        (false, _) => Ok(()),
        (true, Family::Logit) => Err(RfError::InvalidInput("logit: the dependent variable must be 0 or 1".into())),
        (true, _) => Err(RfError::InvalidInput("Poisson: the dependent variable must be non-negative".into())),
    }
}

/// pyfixest's `ir` separation check (fepois_.py `_check_for_separation_ir`, run by its `fepois`
/// and every `feglm` family): the ReLU method on `y` as a Poisson outcome (interior y > 0), over
/// the `alive` rows with their regressors and fixed effects, without weights or offset, and
/// without singleton re-checks. Returns the input rows found separated (ascending).
pub(crate) fn pyfixest_relu(input: &FitInput, spec: &FitSpec, term_codes: &[Codes], alive: &[bool], ctx: &Ctx, pool: &Pool) -> Result<Vec<u32>, RfError> {
    let y = input.y;
    let on_alive = |f: fn(f64) -> bool| pool.install(|| (0..input.n).into_par_iter().any(|i| alive[i] && f(y[i])));
    // pyfixest returns at once without a zero outcome (the ReLU target is y == 0).
    if !on_alive(|v| v == 0.0) {
        return Ok(Vec::new());
    }
    if on_alive(|v| v < 0.0) {
        if spec.model == Model::Ols {
            return Err(RfError::InvalidInput("the `ir` separation check needs a non-negative dependent variable".into()));
        }
        // Poisson: rejected by the outcome check of the fit.
        return Ok(Vec::new());
    }
    let absorb: Vec<AbsorbTerm> =
        input.fe.iter().map(|t| AbsorbTerm { factors: t.factors.clone(), intercept: t.intercept, slopes: t.slopes.clone(), save: false }).collect();
    let parts = Parts {
        n_input: input.n,
        alive: alive.to_vec(),
        dropped: true,
        absorb: &absorb,
        term_codes: term_codes.to_vec(),
        cluster_codes: Vec::new(),
        slope_vars: &input.slopes,
        weights: None,
        weight_kind: WeightKind::None,
        vce: Vce::Unadjusted,
        y_sep: None,
        family: Family::Poisson,
        drop_singletons: false,
        sep_fe: false,
        counts: Default::default(),
    };
    let mut session = Session::from_parts(parts, AbsorberKind::Fast, pool, ctx)?;
    let ones = spec.intercept && !input.fe.iter().any(|t| t.intercept);
    let mut cols = vec![Col::Values(y)];
    if ones {
        cols.push(Col::One);
    }
    cols.extend(input.x.iter().map(|v| Col::Values(v)));
    let nsent = cols.len() - 1;
    let rows = session.gather_input(&cols, pool)?;
    let kept = session.kept_index().to_vec();
    let s = Spec {
        tol: spec.fixef_tol,
        maxiter: spec.fixef_maxiter,
        dof: DofOptions { firstpair: false, pairwise: false, clusters: false, continuous: false },
        ..Spec::default()
    };
    let g = GlmOptions {
        family: Family::Poisson,
        tol: spec.glm_tol,
        maxiter: spec.glm_maxiter,
        itol: spec.fixef_tol,
        separation: Separation { relu: true, mu: false },
        guess: Guess::Simple,
        log: false,
        collinear: GlmCollinear::Pyfixest { tol: spec.collin_tol },
    };
    // The GLM core's drop messages are SMCL for the Stata plugin: the fit layer reports counts.
    let muted = Ctx { threads: ctx.threads, verbose: ctx.verbose, interrupt: ctx.interrupt, log: &|_: &str| {} };
    let check = session.glm_separation(rows, &Layout::ols(nsent), None, &s, &g, &muted)?;
    Ok(kept.iter().zip(&check.separated).filter(|(_, s)| **s).map(|(&i, _)| i).collect())
}

/// ln C(m, k) for integers 0 <= k <= m.
fn ln_choose(m: f64, k: f64) -> f64 {
    dist::ln_gamma(m + 1.0) - dist::ln_gamma(k + 1.0) - dist::ln_gamma(m - k + 1.0)
}

/// fixest's log-likelihood and that of its null (constant-only) model.
/// Poisson: sum w(y*eta - mu - ln Gamma(y+1)); null `sy*ln(sy/N) - sy - sum ln Gamma(y+1)` with sy = sum wy, N = sum w
/// and the factorial term unweighted (femlm.R `get_model_null`); the offset is ignored there.
/// Logit: `binomial()$aic` with the weights as trial counts, sum ln dbinom(round(w*y), round(w), mu);
/// null `sy*ln p + (N - sy)*ln(1 - p)` with p = sy/N.
fn loglik(y: &[f64], kept: &[u32], eta: &[f64], mu: &[f64], w: Option<&[f64]>, family: Family, pool: &Pool) -> (f64, f64) {
    let a = par::sum_chunks(pool, kept.len(), CHUNK, 4, |range, acc| {
        for i in range {
            let yi = y[kept[i] as usize];
            let wi = w.map_or(1.0, |w| w[i]);
            match family {
                Family::Poisson => {
                    let lg = dist::ln_gamma(yi + 1.0);
                    acc[0] += wi * (yi * eta[i] - mu[i] - lg);
                    acc[1] += lg;
                }
                _ => {
                    let (m, k) = (wi.round(), (wi * yi).round());
                    if m > 0.0 {
                        let mut l = ln_choose(m, k);
                        if k > 0.0 {
                            l += k * mu[i].ln();
                        }
                        if m > k {
                            l += (m - k) * (1.0 - mu[i]).ln();
                        }
                        acc[0] += l;
                    }
                }
            }
            acc[2] += wi * yi;
            acc[3] += wi;
        }
    });
    let (sy, n) = (a[2], a[3]);
    let ll_null = match family {
        Family::Poisson => sy * (sy / n).ln() - sy - a[1],
        _ => sy * (sy / n).ln() + (n - sy) * (1.0 - sy / n).ln(),
    };
    (a[0], ll_null)
}

/// Deviance of the constant-only model without offset (Poisson: mu_0 = ybar_w; logit: p = ybar_w).
fn null_deviance(y: &[f64], kept: &[u32], w: Option<&[f64]>, ybar: f64, family: Family, pool: &Pool) -> f64 {
    par::sum_chunks(pool, kept.len(), CHUNK, 1, |range, acc| {
        for i in range {
            let yi = y[kept[i] as usize];
            let wi = w.map_or(1.0, |w| w[i]);
            acc[0] += wi
                * match family {
                    Family::Poisson => 2.0 * (if yi > 0.0 { yi * (yi / ybar).ln() } else { 0.0 } - (yi - ybar)),
                    _ => -2.0 * if yi > 0.0 { ybar.ln() } else { (1.0 - ybar).ln() },
                };
        }
    })[0]
}

/// Values per level of every saved slot (per observation, kept order) of the session's terms.
fn slot_levels(terms: &[crate::session::Term], fe: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut out = Vec::with_capacity(fe.len());
    let mut slots = fe.iter();
    for t in terms {
        for _ in 0..usize::from(t.intercept) + t.slopes.len() {
            let per_obs = slots.next().expect("one saved slot per term slot");
            let mut lv = vec![0.0; t.g];
            for (&c, &v) in t.codes.iter().zip(per_obs) {
                lv[c as usize] = v;
            }
            out.push(lv);
        }
    }
    out
}

/// The reported sample of a GLM fit and every per-row quantity over it (kept order).
struct Reported {
    /// Input rows.
    kept: Vec<u32>,
    w_obs: Option<Vec<f64>>,
    /// Per term: intercept flag and number of slopes.
    terms: Vec<(bool, usize)>,
    /// Per term: codes over the rows and slope values.
    codes: Vec<(Codes, Vec<Vec<f64>>)>,
    clusters: Vec<Codes>,
    eta: Vec<f64>,
    mu: Vec<f64>,
    w: Vec<f64>,
    working_resid: Vec<f64>,
    xt: Vec<f64>,
    /// Rows left out of IRLS: where the solved rows sit, and the values of the levels without one.
    subset: Option<FeSubset>,
    /// Deviance of the rows left out of IRLS.
    excluded_deviance: f64,
}

/// Pyfixest-flavor logit: the reported sample with the rows left out of IRLS at the bound
/// mu = 1 - `MU_MIN` (y = 1) or `MU_MIN` (y = 0), the limit pyfixest's IRLS approaches (its
/// tolerance stops near eta ~= +/-25), with zero working residual and design row: they add nothing
/// to the information matrix or the scores.
fn expand(input: &FitInput, session: &Session, parts: &GlmParts, ex: Excluded, pool: &Pool) -> Result<Reported, RfError> {
    let solved = session.kept_index();
    let excluded = &ex.rows;
    let n = solved.len() + excluded.len();
    // Merge of the two ascending row lists; `src[i]`: index of reported row i among the solved
    // rows (u32::MAX for an excluded row).
    let mut kept = Vec::with_capacity(n);
    let mut src = Vec::with_capacity(n);
    let (mut a, mut b) = (0, 0);
    while kept.len() < n {
        if b == excluded.len() || (a < solved.len() && solved[a] < excluded[b]) {
            src.push(a as u32);
            kept.push(solved[a]);
            a += 1;
        } else {
            src.push(u32::MAX);
            kept.push(excluded[b]);
            b += 1;
        }
    }
    let y = input.y;
    let w_obs: Option<Vec<f64>> = input.weights.map(|w| pool.install(|| kept.par_iter().map(|&i| w[i as usize]).collect()));
    let wi = |i: usize| w_obs.as_ref().map_or(1.0, |w| w[i]);
    let bound = |i: usize| if y[kept[i] as usize] == 1.0 { 1.0 - MU_MIN } else { MU_MIN };
    let fill = |v: &[f64], e: &(dyn Fn(usize) -> f64 + Sync)| -> Vec<f64> {
        pool.install(|| (0..n).into_par_iter().map(|i| if src[i] == u32::MAX { e(i) } else { v[src[i] as usize] }).collect())
    };
    let eta = fill(&parts.eta, &|i| (bound(i) / (1.0 - bound(i))).ln());
    let mu = fill(&parts.mu, &bound);
    let w = fill(&parts.w, &|i| wi(i) * bound(i) * (1.0 - bound(i)));
    let working_resid = fill(&parts.working_resid, &|_| 0.0);
    let kk = parts.k;
    let mut xt = par::alloc_vec(n * kk, 0.0)?;
    if kk > 0 {
        pool.install(|| {
            xt.par_chunks_mut(kk).zip(src.par_iter()).filter(|(_, s)| **s != u32::MAX).for_each(|(row, &s)| {
                row.copy_from_slice(&parts.xt[s as usize * kk..(s as usize + 1) * kk]);
            })
        });
    }
    let excluded_deviance = -2.0 * (1.0 - MU_MIN).ln() * par::sum_chunks(pool, n, CHUNK, 1, |range, acc| {
        for i in range.filter(|&i| src[i] == u32::MAX) {
            acc[0] += wi(i);
        }
    })[0];

    // Codes over the reported rows: those over the sample before GLM-internal drops, restricted
    // to the rows that remain.
    let (term_codes, clusters) = if n == ex.reported.len() {
        (ex.terms, ex.clusters)
    } else {
        let mut at = Vec::with_capacity(n);
        let mut p = 0;
        for &i in &kept {
            while ex.reported[p] != i {
                p += 1;
            }
            at.push(p as u32);
        }
        let on_kept = |c: &Codes| factor::restrict(c, &at, pool);
        (ex.terms.iter().map(on_kept).collect::<Result<_, _>>()?, ex.clusters.iter().map(on_kept).collect::<Result<_, _>>()?)
    };
    let codes: Vec<(Codes, Vec<Vec<f64>>)> = term_codes
        .into_iter()
        .zip(&input.fe)
        .map(|(c, ft)| (c, ft.slopes.iter().map(|&j| pool.install(|| kept.par_iter().map(|&i| input.slopes[j][i as usize]).collect())).collect()))
        .collect();
    // Levels without a solved row: +inf / -inf when all their outcomes are 1 / 0 (the direction
    // their fixed effect diverges), 0 otherwise and for slopes (not identified).
    let mut limits = Vec::new();
    for ((c, _), ft) in codes.iter().zip(&input.fe) {
        let g = c.n as usize;
        if ft.intercept {
            let (mut one, mut zero) = (vec![false; g], vec![false; g]);
            for (i, &l) in c.codes.iter().enumerate() {
                if y[kept[i] as usize] == 1.0 { one[l as usize] = true } else { zero[l as usize] = true }
            }
            limits.push(
                one.iter()
                    .zip(&zero)
                    .map(|(&o, &z)| match (o, z) {
                        (true, false) => f64::INFINITY,
                        (false, true) => f64::NEG_INFINITY,
                        _ => 0.0,
                    })
                    .collect(),
            );
        }
        limits.extend(ft.slopes.iter().map(|_| vec![0.0; g]));
    }
    let rows = src.iter().enumerate().filter(|(_, s)| **s != u32::MAX).map(|(i, _)| i as u32).collect();
    Ok(Reported {
        kept,
        w_obs,
        terms: input.fe.iter().map(|ft| (ft.intercept, ft.slopes.len())).collect(),
        codes,
        clusters,
        eta,
        mu,
        w,
        working_resid,
        xt,
        subset: Some(FeSubset { rows, limits }),
        excluded_deviance,
    })
}

/// A GLM fit after IRLS, before its outputs are built: the session (absorber, terms, sample),
/// the GLM parts and the sample accounting. The bias correction rewrites `parts` at the corrected
/// point between [`estimate`] and [`output`].
pub(crate) struct Estimated {
    pub session: Session,
    pub dropped: super::Dropped,
    pub levels_before_separation: Option<Vec<usize>>,
    pub excluded: Option<Excluded>,
    pub parts: GlmParts,
    pub family: Family,
    pub has_fe: bool,
    pub intercept: bool,
    /// A column of ones was sent first (slope-only FE terms with an intercept).
    pub ones: bool,
    /// Regressors sent (ones column included).
    pub nsent: usize,
    /// The parts were rewritten at bias-corrected coefficients (the state refuses CRV3).
    pub bias_corrected: bool,
}

pub(crate) fn glm(input: &FitInput, spec: &FitSpec, sample: Sample, ctx: &Ctx, pool: &Pool) -> Result<FitOutput, RfError> {
    let est = estimate(input, spec, sample, ctx, pool)?;
    output(input, spec, est, ctx, pool)
}

/// Sample checks and IRLS.
pub(crate) fn estimate(input: &FitInput, spec: &FitSpec, sample: Sample, ctx: &Ctx, pool: &Pool) -> Result<Estimated, RfError> {
    let Sample { mut session, mut dropped, levels_before_separation, excluded } = sample;
    let mut t = std::time::Instant::now();
    let family = if spec.model == Model::Poisson { Family::Poisson } else { Family::Logit };
    check_outcome(input.y, session.kept_index(), family, pool)?;
    let has_fe = !input.fe.is_empty();
    let fe_intercept = input.fe.iter().any(|t| t.intercept);
    // Intercept: the GLM core's constant without FE terms (the absorber then projects out the
    // constant); a column of ones with slope-only FE terms; none with intercept FE terms.
    let intercept = spec.intercept && !fe_intercept;
    if !has_fe && !spec.intercept {
        return Err(RfError::InvalidInput("Poisson and logit models without fixed effects need an intercept".into()));
    }
    let ones = intercept && has_fe;
    let mut cols = vec![Col::Values(input.y)];
    if ones {
        cols.push(Col::One);
    }
    cols.extend(input.x.iter().map(|v| Col::Values(v)));
    let nsent = cols.len() - 1;
    let rows = session.gather_input(&cols, pool)?;
    let offset = input.offset.map(|o| session.gather_input(&[Col::Values(o)], pool)).transpose()?;
    crate::solve::log_phase(ctx, &mut t, "fit: gather");
    let (singletons0, perfect0, kept0) = (session.num_singletons(), session.num_fe_separated(), session.n());

    let s = Spec {
        tol: spec.fixef_tol,
        maxiter: spec.fixef_maxiter,
        // The core's reghdfe statistics are not used: skip its dof adjustments.
        dof: DofOptions { firstpair: false, pairwise: false, clusters: false, continuous: false },
        ..Spec::default()
    };
    let g = GlmOptions {
        family,
        tol: spec.glm_tol,
        maxiter: spec.glm_maxiter,
        itol: spec.fixef_tol,
        // pyfixest's `ir` check ran with the sample ([`pyfixest_relu`]).
        separation: if spec.flavor == Flavor::Pyfixest { Separation { relu: false, ..spec.separation } } else { spec.separation },
        // Ignored under the fixest convention (fixest's starting values).
        guess: Guess::Simple,
        log: false,
        collinear: match spec.flavor {
            Flavor::Fixest => GlmCollinear::Fixest { tol: spec.collin_tol },
            Flavor::Pyfixest => GlmCollinear::Pyfixest { tol: spec.collin_tol },
        },
    };
    let (_, parts) = session.glm_parts(rows, &Layout::ols(nsent), offset, &s, &g, GlmConvention::Fixest, ctx)?;

    // Sample after the GLM-internal (relu / mu) drops and the singleton / perfect-fit passes
    // they trigger.
    let extra_singletons = (session.num_singletons() - singletons0) as usize;
    let extra_perfect = (session.num_fe_separated() - perfect0) as usize;
    dropped.singletons += extra_singletons;
    dropped.perfect_fit += extra_perfect;
    dropped.separated += kept0 - session.n() - extra_singletons - extra_perfect;
    Ok(Estimated { session, dropped, levels_before_separation, excluded, parts, family, has_fe, intercept, ones, nsent, bias_corrected: false })
}

/// The fit layer's outputs from an [`Estimated`] fit.
pub(crate) fn output(input: &FitInput, spec: &FitSpec, est: Estimated, ctx: &Ctx, pool: &Pool) -> Result<FitOutput, RfError> {
    let Estimated { mut session, dropped, levels_before_separation, excluded, mut parts, family, has_fe, intercept, ones, nsent, bias_corrected } = est;
    let mut t = std::time::Instant::now();
    // Coefficients: intercept first, then x (fixest order).
    let source = |j: usize| if ones && j == 0 { CoefSource::Intercept } else { CoefSource::X(j - usize::from(ones)) };
    let mut coef_source = Vec::new();
    let mut coef = Vec::new();
    if intercept && !ones {
        coef_source.push(CoefSource::Intercept);
        coef.push(parts.cons);
    }
    for j in 0..nsent {
        if parts.status[j] == 0 {
            coef_source.push(source(j));
            coef.push(parts.coef[j]);
        }
    }
    let collinear: Vec<CoefSource> = (0..nsent).filter(|&j| parts.status[j] != 0).map(source).collect();
    if nsent > 0 && coef.is_empty() {
        return Err(RfError::InvalidInput("all the regressors are collinear (with the fixed effects)".into()));
    }
    let k = coef.len();

    // The reported sample: the session's rows, plus the rows left out of IRLS reported at the
    // bound (Pyfixest-flavor logit). Fixed-effect values are taken on the session's rows.
    let solved_slots = (has_fe && (spec.keep_fixef || spec.keep_state)).then(|| slot_levels(session.terms(), &parts.fe));
    // CRV3 refits run on the session's rows (the rows left out of IRLS carry no information on beta).
    let jk_rows: Option<(Vec<u32>, Vec<Term>)> = (spec.vcov == VcovKind::Crv3).then(|| (session.kept_index().to_vec(), session.terms().to_vec()));
    let rep = if let Some(ex) = excluded {
        expand(input, &session, &parts, ex, pool)?
    } else {
        let terms = std::mem::take(&mut session.terms);
        Reported {
            kept: std::mem::take(&mut session.kept_index),
            w_obs: session.weights.take(),
            terms: terms.iter().map(|t| (t.intercept, t.slopes.len())).collect(),
            codes: terms.into_iter().map(|t| (Codes { codes: t.codes, n: t.g as u32 }, t.slopes)).collect(),
            clusters: std::mem::take(&mut session.clusters).into_iter().map(|c| Codes { codes: c.codes, n: c.g as u32 }).collect(),
            eta: std::mem::take(&mut parts.eta),
            mu: std::mem::take(&mut parts.mu),
            w: std::mem::take(&mut parts.w),
            working_resid: std::mem::take(&mut parts.working_resid),
            xt: std::mem::take(&mut parts.xt),
            subset: None,
            excluded_deviance: 0.0,
        }
    };
    drop(session);
    let n = rep.kept.len();
    let kept_index = rep.kept.as_slice();
    let w_obs = rep.w_obs.as_deref();
    let (sumw, ybar, ssr_null) = y_stats(input.y, kept_index, w_obs, pool);
    let deviance = parts.deviance + rep.excluded_deviance;

    // Variance: block [working residual, design...] in kept order under the final IRLS weights.
    // With FE terms the design is x~ (`xt`, the ones column included when sent); without, the
    // raw [1, x] (x = x~ + xbar_w), as fixest's unresidualized design.
    let kk = parts.k;
    let r = 1 + k;
    let mut block = par::alloc_vec(n * r, 0.0)?;
    {
        let raw = !has_fe;
        // Without FE terms no ones column is sent: sent regressor j is x[j].
        let xs: Vec<&[f64]> = if raw { (0..nsent).filter(|&j| parts.status[j] == 0).map(|j| input.x[j]).collect() } else { Vec::new() };
        pool.install(|| {
            block.par_chunks_mut(r).enumerate().for_each(|(i, row)| {
                row[0] = rep.working_resid[i];
                if raw {
                    row[1] = 1.0;
                    for (v, x) in row[2..].iter_mut().zip(&xs) {
                        *v = x[kept_index[i] as usize];
                    }
                } else {
                    row[1..].copy_from_slice(&rep.xt[i * kk..(i + 1) * kk]);
                }
            })
        });
    }
    let (gram, _) = stats::gram(&block, r, Some(&rep.w), pool);
    let sub: Vec<f64> = (1..r).flat_map(|a| (1..r).map(move |b| (a, b))).map(|(a, b)| gram[a * r + b]).collect();
    let (singular, bread) = chol_collin(&sub, k, f64::MIN_POSITIVE);
    if singular.contains(&true) {
        return Err(RfError::Internal("GLM information matrix is singular".into()));
    }
    let design: Vec<Vec<f64>> = (0..k).map(|j| super::design::unit(r, 1 + j)).collect();
    let resid_combo = super::design::unit(r, 0);
    let row_order: Vec<u32> = (0..n as u32).collect();
    // Levels per term in the degrees of freedom: pyfixest's `feglm` counts those present before
    // its `fe` separation check (its `fepois` recounts them).
    let mut before = levels_before_separation.filter(|_| family == Family::Logit).map(Vec::into_iter);
    let sizes: Vec<usize> =
        rep.terms.iter().zip(&rep.codes).map(|(&(icpt, _), (c, _))| before.as_mut().filter(|_| icpt).and_then(Iterator::next).unwrap_or(c.n as usize)).collect();
    let n_icpt = rep.terms.iter().filter(|t| t.0).count();
    let np = k + rep.terms.iter().zip(&sizes).map(|(&(icpt, ns), g)| g * (usize::from(icpt) + ns)).sum::<usize>() - n_icpt.saturating_sub(1);
    let fweights = spec.flavor == Flavor::Pyfixest && spec.weights_type == WeightsType::Fweights && w_obs.is_some();
    let n_eff = if fweights { sumw } else { n as f64 };
    let fe_names = |ft: &super::FeTerm| -> Vec<&str> { ft.factors.iter().map(|&f| input.factor_names[f].as_str()).collect() };
    let fe_info: Vec<FeInfo> = rep
        .codes
        .iter()
        .zip(&rep.terms)
        .zip(&input.fe)
        .zip(&sizes)
        .map(|(((c, &(icpt, _)), ft), &g)| FeInfo { codes: &c.0.codes, g, intercept: icpt, names: fe_names(ft) })
        .collect();
    let cl_info: Vec<ClusterInfo> = rep
        .clusters
        .iter()
        .zip(&input.cluster)
        .map(|(c, dim)| ClusterInfo { codes: &c.codes, g: c.n as usize, names: dim.iter().map(|&f| input.factor_names[f].as_str()).collect() })
        .collect();
    let refs = || term_refs(&fe_info, pool);
    let model = VModel {
        flavor: spec.flavor,
        n: n_eff,
        nparams: np,
        bread: &bread,
        iid_scale: 1.0,
        normal: true,
        rows: &block,
        r,
        design: &design,
        resid: &resid_combo,
        w: Some(&rep.w),
        fweights,
        row_order: &row_order,
    };
    // Inputs of the HC2/HC3, HAC, Conley and CRV3 variance matrices.
    let vdata = VcovData {
        factors: &input.factors,
        factor_names: &input.factor_names,
        cluster: &input.cluster,
        time: input.time,
        panel: input.panel,
        lat: input.lat,
        lon: input.lon,
    };
    let inputs = VcovInputs::read(spec.vcov, spec.flavor, &vdata, kept_index, pool)?;
    let fe_hat = if matches!(spec.vcov, VcovKind::Hc2 | VcovKind::Hc3) && has_fe {
        // fixest `hatvalues` for `feglm`: the projection under the IRLS weights.
        let terms: Vec<(&[u32], usize, bool, Vec<&[f64]>)> = rep
            .codes
            .iter()
            .zip(&rep.terms)
            .map(|((c, slopes), &(icpt, _))| (c.codes.as_slice(), c.n as usize, icpt, slopes.iter().map(Vec::as_slice).collect()))
            .collect();
        Some(fe_leverage(&terms, Some(&rep.w), n, ctx, pool)?)
    } else {
        None
    };
    let jk_data = jk_rows.as_ref().map(|(solved, terms)| {
        let x: Vec<&[f64]> = coef_source.iter().filter_map(|s| if let CoefSource::X(j) = s { Some(input.x[*j]) } else { None }).collect();
        let w = input.weights.map(|w| extra::kept_values(w, solved, pool));
        let fe = terms.iter().map(|t| FeCols { codes: &t.codes, g: t.g, intercept: t.intercept, slopes: t.slopes.iter().map(Vec::as_slice).collect() }).collect();
        JkData::new(input.y, &x, &[], &[], input.offset, w.as_deref(), fe, solved, pool)
    });
    let jk_problem = jk_data.as_ref().map(|d| d.problem(jk_model(spec.model), intercept, &coef, spec));
    // Cluster codes over the reported rows -> over the session's rows.
    let solved_at = rep.subset.as_ref().map(|s| s.rows.as_slice());
    let jk_fn = jk_problem.as_ref().map(|p| {
        move |codes: &[u32], g: usize| match solved_at {
            None => extra::jackknife_meat(p, codes, g, ctx, pool),
            Some(at) => {
                let c = factor::restrict(&Codes { codes: codes.to_vec(), n: g as u32 }, at, pool)?;
                extra::jackknife_meat(p, &c.codes, c.n as usize, ctx, pool)
            }
        }
    });
    let intercept_idx = coef_source.iter().position(|s| *s == CoefSource::Intercept);
    let mut extra_in = inputs.extra(has_fe, intercept_idx);
    extra_in.fe_hat = fe_hat.as_deref();
    extra_in.jackknife = jk_fn.as_ref().map(|f| f as &Jackknife);
    let (vcov, mut notes) = vcov::compute(&model, spec.vcov, &spec.ssc, &fe_info, &cl_info, &extra_in, &refs, pool)?;
    drop(fe_info);
    crate::solve::log_phase(ctx, &mut t, "fit: vcov");
    if !parts.converged {
        notes.insert(
            0,
            format!("Absence of convergence: maximum number of IRLS iterations reached ({}). Final deviance: {}.", spec.glm_maxiter, deviance),
        );
    }

    // Fit statistics.
    let y = input.y;
    let (ll, ll_null) = loglik(y, kept_index, &rep.eta, &rep.mu, w_obs, family, pool);
    let null_dev = null_deviance(y, kept_index, w_obs, ybar, family, pool);
    let ssr = par::sum_chunks(pool, n, CHUNK, 1, |range, acc| {
        for i in range {
            acc[0] += w_obs.map_or(1.0, |w| w[i]) * (y[kept_index[i] as usize] - rep.mu[i]).powi(2);
        }
    })[0];
    let cor_parts = par::map_chunks(pool, n, CHUNK, |_, range| cor_part(range.map(|i| (y[kept_index[i] as usize], rep.mu[i]))));
    let wn = if w_obs.is_some() { sumw } else { n as f64 };
    let only_intercept = np == 1 && intercept && k == 1;
    let (pseudo_r2, apr2) = if only_intercept { (f64::NAN, f64::NAN) } else { (1.0 - ll / ll_null, 1.0 - (ll - np as f64 + 1.0) / ll_null) };
    let stats = FitStats {
        ssr,
        ssr_null,
        ssr_fe_only: f64::NAN,
        sigma2: f64::NAN,
        rmse: (ssr / wn).sqrt(),
        r2: f64::NAN,
        ar2: f64::NAN,
        wr2: f64::NAN,
        war2: f64::NAN,
        loglik: ll,
        ll_null,
        deviance,
        null_deviance: null_dev,
        pseudo_r2,
        apr2,
        sq_cor: sq_cor(cor_parts),
        f_test: TestStat::NA,
        fe_iterations: 0,
        irls_iterations: parts.iterations,
        converged: parts.converged,
    };

    // Fixed effects: the saved slots sum to eta - offset - x'beta (constant included).
    let Reported { kept, terms, codes, eta, mu, w, subset, .. } = rep;
    let fe_sizes: Vec<usize> = codes.iter().map(|(c, _)| c.n as usize).collect();
    let state_fe: Vec<StateFe> = codes
        .into_iter()
        .zip(terms)
        .zip(&input.fe)
        .zip(&sizes)
        .map(|((((c, slopes), (intercept, _)), ft), &g)| StateFe {
            codes: c.codes,
            g,
            levels: c.n as usize,
            intercept,
            names: fe_names(ft).into_iter().map(String::from).collect(),
            slopes,
            slope_ids: ft.slopes.clone(),
        })
        .collect();
    let fe_levels = solved_slots.map(|slots| FeLevels { slots, subset });
    let fixef = match &fe_levels {
        Some(l) if spec.keep_fixef => fixef::fixef_values(&state_fe, &kept, l, spec.fixef_norm, pool)?,
        _ => Vec::new(),
    };
    let (resid, fitted, eta, irls_weights) = if spec.keep_resid {
        let resid: Vec<f64> = pool.install(|| mu.par_iter().zip(kept.par_iter()).map(|(m, &i)| y[i as usize] - m).collect());
        (Some(resid), Some(mu), Some(eta), Some(w.clone()))
    } else {
        (None, None, None, None)
    };
    crate::solve::log_phase(ctx, &mut t, "fit: outputs");
    let obs = kept;
    let state = spec.keep_state.then(|| super::FitState {
        inner: State {
            flavor: spec.flavor,
            n: n_eff,
            nparams: np,
            bread,
            iid_scale: 1.0,
            normal: true,
            rows: block,
            r,
            design,
            resid: resid_combo,
            weights: Some(w),
            fweights,
            row_order,
            obs: obs.clone(),
            fe: state_fe,
            fixef: fe_levels,
            iv: false,
            intercept: intercept_idx,
            fe_hat: fe_hat.map(std::sync::OnceLock::from).unwrap_or_default(),
            bias_corrected,
        },
    });
    Ok(FitOutput {
        nobs: n,
        nobs_origin: input.n,
        sumw,
        obs,
        dropped,
        coef_source,
        coef,
        collinear,
        vcov,
        nparams: np,
        fe_sizes,
        stats,
        iv: None,
        resid,
        fitted,
        eta,
        irls_weights,
        fixef,
        notes,
        state,
    })
}
