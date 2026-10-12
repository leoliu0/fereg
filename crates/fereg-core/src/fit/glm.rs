//! Poisson and logit models in the fit layer (fixest `fepois` / `feglm(family = "logit")`,
//! pyfixest `fepois` / `feglm`). IRLS runs in the GLM core under the fixest convention
//! ([`Session::glm_parts`]); this module turns its per-observation parts into the fit layer's
//! outputs: sample accounting after GLM-internal drops, coefficients and intercept, the variance
//! matrix through the shared fixest code, fit statistics, fixed-effect values and the state.

use rayon::prelude::*;

use super::design::chol_collin;
use super::input::Sample;
use super::ols::{cor_part, nparams_with, sq_cor, y_stats};
use super::state::{State, StateFe, term_refs};
use super::vcov::{self, ClusterInfo, FeInfo, Model as VModel};
use super::{CoefSource, FitInput, FitOutput, FitSpec, FitStats, Flavor, Model, TestStat, WeightsType, dist, fixef};
use crate::api::{Ctx, DofOptions, Family, GlmConvention, GlmOptions, GlmParts, Guess, Layout, Spec};
use crate::error::RfError;
use crate::gather::Col;
use crate::par::{self, CHUNK, Pool};
use crate::stats;

/// Checks the outcome on the estimation rows: Poisson y ≥ 0, logit y ∈ {0, 1}.
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

/// ln C(m, k) for integers 0 ≤ k ≤ m.
fn ln_choose(m: f64, k: f64) -> f64 {
    dist::ln_gamma(m + 1.0) - dist::ln_gamma(k + 1.0) - dist::ln_gamma(m - k + 1.0)
}

/// fixest's log-likelihood and that of its null (constant-only) model.
/// Poisson: Σw(y·η − μ − lnΓ(y+1)); null `sy·ln(sy/N) − sy − ΣlnΓ(y+1)` with sy = Σwy, N = Σw
/// and the factorial term unweighted (femlm.R `get_model_null`); the offset is ignored there.
/// Logit: `binomial()$aic` with the weights as trial counts, Σ ln dbinom(round(w·y), round(w), μ);
/// null `sy·ln p + (N − sy)·ln(1 − p)` with p = sy/N.
fn loglik(y: &[f64], kept: &[u32], p: &GlmParts, w: Option<&[f64]>, family: Family, pool: &Pool) -> (f64, f64) {
    let a = par::sum_chunks(pool, kept.len(), CHUNK, 4, |range, acc| {
        for i in range {
            let yi = y[kept[i] as usize];
            let wi = w.map_or(1.0, |w| w[i]);
            match family {
                Family::Poisson => {
                    let lg = dist::ln_gamma(yi + 1.0);
                    acc[0] += wi * (yi * p.eta[i] - p.mu[i] - lg);
                    acc[1] += lg;
                }
                _ => {
                    let (m, k) = (wi.round(), (wi * yi).round());
                    if m > 0.0 {
                        let mut l = ln_choose(m, k);
                        if k > 0.0 {
                            l += k * p.mu[i].ln();
                        }
                        if m > k {
                            l += (m - k) * (1.0 - p.mu[i]).ln();
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

/// Deviance of the constant-only model without offset (Poisson: μ₀ = ȳ_w; logit: p = ȳ_w).
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

pub(crate) fn glm(input: &FitInput, spec: &FitSpec, sample: Sample, ctx: &Ctx, pool: &Pool) -> Result<FitOutput, RfError> {
    let Sample { mut session, mut dropped, levels_before_separation } = sample;
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
        separation: spec.separation,
        // Ignored under the fixest convention (fixest's starting values).
        guess: Guess::Simple,
        log: false,
    };
    let (_, parts) = session.glm_parts(rows, &Layout::ols(nsent), offset, &s, &g, GlmConvention::Fixest, ctx)?;
    let mut t = std::time::Instant::now();

    // Sample after the GLM-internal (relu / mu) drops and the singleton / perfect-fit passes
    // they trigger.
    let n = session.n();
    let extra_singletons = (session.num_singletons() - singletons0) as usize;
    let extra_perfect = (session.num_fe_separated() - perfect0) as usize;
    dropped.singletons += extra_singletons;
    dropped.perfect_fit += extra_perfect;
    dropped.separated = kept0 - n - extra_singletons - extra_perfect;
    let kept_index = session.kept_index();
    let w_obs = session.weights();
    let (sumw, ybar, ssr_null) = y_stats(input.y, kept_index, w_obs, pool);

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

    // Variance: block [working residual, design…] in kept order under the final IRLS weights.
    // With FE terms the design is x̃ (`xt`, the ones column included when sent); without, the
    // raw [1, x] (x = x̃ + x̄_w), as fixest's unresidualized design.
    let kk = parts.k;
    let r = 1 + k;
    let mut block = par::alloc_vec(n * r, 0.0)?;
    {
        let raw = !has_fe;
        // Without FE terms no ones column is sent: sent regressor j is x[j].
        let xs: Vec<&[f64]> = if raw { (0..nsent).filter(|&j| parts.status[j] == 0).map(|j| input.x[j]).collect() } else { Vec::new() };
        pool.install(|| {
            block.par_chunks_mut(r).enumerate().for_each(|(i, row)| {
                row[0] = parts.working_resid[i];
                if raw {
                    row[1] = 1.0;
                    for (v, x) in row[2..].iter_mut().zip(&xs) {
                        *v = x[kept_index[i] as usize];
                    }
                } else {
                    row[1..].copy_from_slice(&parts.xt[i * kk..(i + 1) * kk]);
                }
            })
        });
    }
    let (gram, _) = stats::gram(&block, r, Some(&parts.w), pool);
    let sub: Vec<f64> = (1..r).flat_map(|a| (1..r).map(move |b| (a, b))).map(|(a, b)| gram[a * r + b]).collect();
    let (singular, bread) = chol_collin(&sub, k, f64::MIN_POSITIVE);
    if singular.contains(&true) {
        return Err(RfError::Internal("GLM information matrix is singular".into()));
    }
    let design: Vec<Vec<f64>> = (0..k).map(|j| super::design::unit(r, 1 + j)).collect();
    let resid_combo = super::design::unit(r, 0);
    let row_order: Vec<u32> = (0..n as u32).collect();
    let terms = session.terms();
    // Levels per term in the degrees of freedom: pyfixest's `feglm` counts those present before
    // its `fe` separation check (its `fepois` recounts them).
    let mut before = levels_before_separation.filter(|_| family == Family::Logit).map(Vec::into_iter);
    let sizes: Vec<usize> = terms.iter().map(|t| before.as_mut().filter(|_| t.intercept).and_then(Iterator::next).unwrap_or(t.g)).collect();
    let np = nparams_with(k, terms, &sizes);
    let fweights = spec.flavor == Flavor::Pyfixest && spec.weights_type == WeightsType::Fweights && w_obs.is_some();
    let n_eff = if fweights { sumw } else { n as f64 };
    let fe_names = |ft: &super::FeTerm| -> Vec<&str> { ft.factors.iter().map(|&f| input.factor_names[f].as_str()).collect() };
    let fe_info: Vec<FeInfo> = terms
        .iter()
        .zip(&input.fe)
        .zip(&sizes)
        .map(|((t, ft), &g)| FeInfo { codes: &t.codes, g, intercept: t.intercept, names: fe_names(ft) })
        .collect();
    let cl_info: Vec<ClusterInfo> = session
        .clusters()
        .iter()
        .zip(&input.cluster)
        .map(|(c, dim)| ClusterInfo { codes: &c.codes, g: c.g, names: dim.iter().map(|&f| input.factor_names[f].as_str()).collect() })
        .collect();
    let refs = || term_refs(&fe_info);
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
        w: Some(&parts.w),
        fweights,
        row_order: &row_order,
    };
    let (vcov, mut notes) = vcov::compute(&model, spec.vcov, &spec.ssc, &fe_info, &cl_info, &refs, pool)?;
    crate::solve::log_phase(ctx, &mut t, "fit: vcov");
    if !parts.converged {
        notes.insert(
            0,
            format!("Absence of convergence: maximum number of IRLS iterations reached ({}). Final deviance: {}.", spec.glm_maxiter, parts.deviance),
        );
    }

    // Fit statistics.
    let y = input.y;
    let (ll, ll_null) = loglik(y, kept_index, &parts, w_obs, family, pool);
    let null_dev = null_deviance(y, kept_index, w_obs, ybar, family, pool);
    let ssr = par::sum_chunks(pool, n, CHUNK, 1, |range, acc| {
        for i in range {
            acc[0] += w_obs.map_or(1.0, |w| w[i]) * (y[kept_index[i] as usize] - parts.mu[i]).powi(2);
        }
    })[0];
    let cor_parts = par::map_chunks(pool, n, CHUNK, |_, range| cor_part(range.map(|i| (y[kept_index[i] as usize], parts.mu[i]))));
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
        deviance: parts.deviance,
        null_deviance: null_dev,
        pseudo_r2,
        apr2,
        sq_cor: sq_cor(cor_parts),
        f_test: TestStat::NA,
        fe_iterations: 0,
        irls_iterations: parts.iterations,
        converged: parts.converged,
    };

    // Fixed effects: the saved slots sum to η − offset − x'β (constant included).
    let fixef = if spec.keep_fixef && has_fe { fixef::fixef_values(input, terms, slot_levels(terms, &parts.fe), kept_index, spec.fixef_norm) } else { Vec::new() };
    let (resid, fitted, eta, irls_weights) = if spec.keep_resid {
        let resid: Vec<f64> = pool.install(|| parts.mu.par_iter().zip(kept_index.par_iter()).map(|(m, &i)| y[i as usize] - m).collect());
        (Some(resid), Some(parts.mu), Some(parts.eta), Some(parts.w.clone()))
    } else {
        (None, None, None, None)
    };
    crate::solve::log_phase(ctx, &mut t, "fit: outputs");
    let obs = kept_index.to_vec();
    let fe_sizes = terms.iter().map(|t| t.g).collect();
    let state = if spec.keep_state {
        let fe = std::mem::take(&mut session.terms)
            .into_iter()
            .zip(&input.fe)
            .zip(&sizes)
            .map(|((t, ft), &g)| StateFe { codes: t.codes, g, intercept: t.intercept, names: fe_names(ft).into_iter().map(String::from).collect() })
            .collect();
        drop(session);
        Some(super::FitState {
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
                weights: Some(parts.w),
                fweights,
                row_order,
                obs: obs.clone(),
                fe,
            },
        })
    } else {
        None
    };
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
