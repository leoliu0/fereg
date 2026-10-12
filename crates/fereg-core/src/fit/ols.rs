//! OLS in the fit layer (fixest `feols` / pyfixest `feols`): one gather of the input columns
//! into solver row order, one residualization, one Gram pass, fixest's collinearity rule,
//! coefficients, residuals and fitted values, the variance matrix and the fit statistics.

use rayon::prelude::*;

use super::input::Sample;
use super::vcov::{self, Base, ClusterInfo, FeInfo, Scores};
use super::{CoefSource, FitInput, FitOutput, FitSpec, FitStats, Flavor, TestStat, WeightsType, dist, fixef};
use crate::absorber::SolveOpts;
use crate::api::Ctx;
use crate::error::RfError;
use crate::fe_recovery::to_kept_order;
use crate::gather::Col;
use crate::par::{self, CHUNK, Pool};
use crate::session::Session;
use crate::stats;

/// fixest `cpp_cholesky(XtX, tol)` (lm_related.cpp): Cholesky in column order where column j is
/// excluded when its pivot X_jj − Σ_{kept i<j} R_ij² is below `tol` (absolute, unscaled).
/// Returns the exclusion flags and the inverse of the kept block (kept order, row-major).
pub(crate) fn chol_collin(a: &[f64], k: usize, tol: f64) -> (Vec<bool>, Vec<f64>) {
    let mut r = vec![0.0; k * k];
    let mut excluded = vec![false; k];
    for j in 0..k {
        let mut rjj = a[j * k + j];
        for i in (0..j).filter(|&i| !excluded[i]) {
            rjj -= r[i * k + j] * r[i * k + j];
        }
        if !(rjj >= tol) {
            excluded[j] = true;
            continue;
        }
        let d = rjj.sqrt();
        r[j * k + j] = d;
        for c in j + 1..k {
            let mut v = a[c * k + j];
            for i in (0..j).filter(|&i| !excluded[i]) {
                v -= r[i * k + c] * r[i * k + j];
            }
            r[j * k + c] = v / d;
        }
    }
    let kept: Vec<usize> = (0..k).filter(|&j| !excluded[j]).collect();
    (excluded, inverse_from_chol(&r, k, &kept))
}

/// (RᵀR)⁻¹ = R⁻¹R⁻ᵀ over the `kept` rows/columns of the upper-triangular `r` (k×k).
fn inverse_from_chol(r: &[f64], k: usize, kept: &[usize]) -> Vec<f64> {
    let m = kept.len();
    // R⁻¹ (upper triangular, m×m) by back substitution, column by column.
    let mut ri = vec![0.0; m * m];
    for c in 0..m {
        ri[c * m + c] = 1.0 / r[kept[c] * k + kept[c]];
        for row in (0..c).rev() {
            let mut s = 0.0;
            for l in row + 1..=c {
                s += r[kept[row] * k + kept[l]] * ri[l * m + c];
            }
            ri[row * m + c] = -s / r[kept[row] * k + kept[row]];
        }
    }
    let mut inv = vec![0.0; m * m];
    for i in 0..m {
        for j in i..m {
            let s: f64 = (j..m).map(|l| ri[i * m + l] * ri[j * m + l]).sum();
            inv[i * m + j] = s;
            inv[j * m + i] = s;
        }
    }
    inv
}

/// fixest `nparams`: kept coefficients plus FE parameters (fixest_env.R): every intercept term
/// counts G·(1 + #slopes), a slope-only term G·#slopes, minus one per intercept term beyond the first.
pub(crate) fn nparams(k: usize, session: &Session) -> usize {
    let terms = session.terms();
    let n_icpt = terms.iter().filter(|t| t.intercept).count();
    let fe: usize = terms.iter().map(|t| t.g * (usize::from(t.intercept) + t.slopes.len())).sum();
    k + fe - n_icpt.saturating_sub(1)
}

/// Weighted mean and Σw(y − ȳ_w)² of the input y over the kept rows (kept order weights),
/// chunked with Chan's merge in chunk order. Returns (Σw, ȳ_w, ssr_null).
fn y_stats(y: &[f64], kept: &[u32], w: Option<&[f64]>, pool: &Pool) -> (f64, f64, f64) {
    let parts = par::map_chunks(pool, kept.len(), CHUNK, |_, range| {
        let wi = |i: usize| w.map_or(1.0, |w| w[i]);
        let sw: f64 = range.clone().map(wi).sum();
        let mean = range.clone().map(|i| wi(i) * y[kept[i] as usize]).sum::<f64>() / sw;
        let m2: f64 = range.map(|i| wi(i) * (y[kept[i] as usize] - mean).powi(2)).sum();
        (sw, mean, m2)
    });
    let mut it = parts.into_iter();
    let first = it.next().unwrap_or((0.0, 0.0, 0.0));
    let (sw, mean, m2) = it.fold(first, |(sa, ma, qa), (sb, mb, qb)| {
        let s = sa + sb;
        let d = mb - ma;
        (s, ma + d * sb / s, qa + qb + d * d * sa * sb / s)
    });
    (sw, mean, m2)
}

/// Squared correlation of y and the fitted values (fixest `sq.cor`; NaN when the fitted values
/// are constant), chunked with Chan's merge.
fn sq_cor(y: &[f64], kept: &[u32], fitted: &[f64], pool: &Pool) -> f64 {
    let parts = par::map_chunks(pool, kept.len(), CHUNK, |_, range| {
        let c = range.len() as f64;
        let my = range.clone().map(|i| y[kept[i] as usize]).sum::<f64>() / c;
        let mf = range.clone().map(|i| fitted[i]).sum::<f64>() / c;
        let (mut syy, mut sff, mut syf) = (0.0, 0.0, 0.0);
        for i in range {
            let (dy, df) = (y[kept[i] as usize] - my, fitted[i] - mf);
            syy += dy * dy;
            sff += df * df;
            syf += dy * df;
        }
        [c, my, mf, syy, sff, syf]
    });
    let mut it = parts.into_iter();
    let mut a = it.next().unwrap_or([0.0; 6]);
    for b in it {
        let n = a[0] + b[0];
        let (dy, df) = (b[1] - a[1], b[2] - a[2]);
        let f = a[0] * b[0] / n;
        a = [n, a[1] + dy * b[0] / n, a[2] + df * b[0] / n, a[3] + b[3] + dy * dy * f, a[4] + b[4] + df * df * f, a[5] + b[5] + dy * df * f];
    }
    if a[4] == 0.0 { f64::NAN } else { a[5] * a[5] / (a[3] * a[4]) }
}

pub(crate) fn ols(input: &FitInput, spec: &FitSpec, sample: Sample, ctx: &Ctx, pool: &Pool) -> Result<FitOutput, RfError> {
    let Sample { mut session, dropped } = sample;
    let mut t = std::time::Instant::now();
    let n = session.n();
    let has_fe = !input.fe.is_empty();
    let intercept = spec.intercept && !input.fe.iter().any(|t| t.intercept);

    // Row block [y, (1), x…] in solver row order, residualized on the fixed effects.
    let mut cols = vec![Col::Values(input.y)];
    let mut sources = Vec::new();
    if intercept {
        cols.push(Col::One);
        sources.push(CoefSource::Intercept);
    }
    cols.extend(input.x.iter().map(|x| Col::Values(x)));
    sources.extend((0..input.x.len()).map(CoefSource::X));
    let r = cols.len();
    let kx = r - 1;
    let mut rows = session.gather_input(&cols, pool)?;
    crate::solve::log_phase(ctx, &mut t, "fit: gather");
    let (sumw, _, ssr_null) = y_stats(input.y, session.kept_index(), session.weights(), pool);
    ctx.check()?;
    let fe_iterations = if has_fe {
        let opts = SolveOpts { tol: spec.fixef_tol, maxiter: spec.fixef_maxiter, want_coefs: false };
        session.absorber_mut().residualize(&mut rows, r, &opts, None, pool, ctx)?.iterations
    } else {
        0
    };
    let mut t = std::time::Instant::now();
    let w_rows = session.absorber().weights();
    let (g, _) = stats::gram(&rows, r, w_rows, pool);
    crate::solve::log_phase(ctx, &mut t, "fit: gram");

    // Collinearity: fixest on the weighted Gram, pyfixest on the unweighted one (it drops
    // columns before applying the weights); the bread is the inverse of the weighted kept block.
    let block = |g: &[f64], cols: &[usize]| -> Vec<f64> { cols.iter().flat_map(|&a| cols.iter().map(move |&b| g[(1 + a) * r + 1 + b])).collect() };
    let all: Vec<usize> = (0..kx).collect();
    let (excluded, bread) = if spec.flavor == Flavor::Pyfixest && w_rows.is_some() {
        let (gu, _) = stats::gram(&rows, r, None, pool);
        let (excluded, _) = chol_collin(&block(&gu, &all), kx, spec.collin_tol);
        let kept: Vec<usize> = all.iter().copied().filter(|&j| !excluded[j]).collect();
        let (singular, inv) = chol_collin(&block(&g, &kept), kept.len(), f64::MIN_POSITIVE);
        if singular.contains(&true) {
            return Err(RfError::InvalidInput("the weighted cross-product of the regressors is singular".into()));
        }
        (excluded, inv)
    } else {
        chol_collin(&block(&g, &all), kx, spec.collin_tol)
    };
    let kept: Vec<usize> = all.iter().copied().filter(|&j| !excluded[j]).collect();
    if kx > 0 && kept.is_empty() {
        return Err(RfError::InvalidInput("all the regressors are collinear (with the fixed effects)".into()));
    }
    let collinear: Vec<CoefSource> = all.iter().filter(|&&j| excluded[j]).map(|&j| sources[j]).collect();
    let k = kept.len();
    let coef: Vec<f64> = (0..k).map(|a| (0..k).map(|b| bread[a * k + b] * g[(1 + kept[b]) * r]).sum()).collect();
    let coef_source: Vec<CoefSource> = kept.iter().map(|&j| sources[j]).collect();

    // Residuals (row order) with Σw·e² and Σe².
    let mut e = par::alloc_vec(n, 0.0)?;
    let sums: Vec<[f64; 2]> = pool.install(|| {
        e.par_chunks_mut(CHUNK)
            .enumerate()
            .map(|(c, es)| {
                let mut s = [0.0; 2];
                for (o, ev) in es.iter_mut().enumerate() {
                    let i = c * CHUNK + o;
                    let row = &rows[i * r..(i + 1) * r];
                    let mut v = row[0];
                    for (a, &j) in kept.iter().enumerate() {
                        v -= coef[a] * row[1 + j];
                    }
                    *ev = v;
                    s[0] += w_rows.map_or(1.0, |w| w[i]) * v * v;
                    s[1] += v * v;
                }
                s
            })
            .collect()
    });
    let (ssr, sse) = sums.iter().fold((0.0, 0.0), |a, s| (a.0 + s[0], a.1 + s[1]));
    crate::solve::log_phase(ctx, &mut t, "fit: residuals");
    ctx.check()?;

    // Variance matrix.
    let fweights = spec.flavor == Flavor::Pyfixest && spec.weights_type == WeightsType::Fweights && w_rows.is_some();
    let n_eff = if fweights { sumw } else { n as f64 };
    let np = nparams(k, &session);
    let base = Base { flavor: spec.flavor, n: n_eff, k, nparams: np, bread, iid_scale: ssr / (n_eff - 1.0), normal: false };
    let terms = session.terms();
    let fe_info: Vec<FeInfo> = terms
        .iter()
        .zip(&input.fe)
        .map(|(t, ft)| FeInfo { codes: &t.codes, g: t.g, intercept: t.intercept, names: ft.factors.iter().map(|&f| input.factor_names[f].as_str()).collect() })
        .collect();
    let cl_info: Vec<ClusterInfo> = session
        .clusters()
        .iter()
        .zip(&input.cluster)
        .map(|(c, dim)| ClusterInfo { codes: &c.codes, g: c.g, names: dim.iter().map(|&f| input.factor_names[f].as_str()).collect() })
        .collect();
    let refs = || {
        let icpt: Vec<&crate::session::Term> = terms.iter().filter(|t| t.intercept).collect();
        let codes: Vec<&[u32]> = icpt.iter().map(|t| t.codes.as_slice()).collect();
        let sizes: Vec<usize> = icpt.iter().map(|t| t.g).collect();
        fixef::references(&codes, &sizes)
    };
    let scores = Scores {
        n_rows: n,
        row_order: session.row_order(),
        hetero_div: if fweights { w_rows } else { None },
        row: |i: usize, a: &mut [f64]| {
            let row = &rows[i * r..(i + 1) * r];
            for (p, &j) in kept.iter().enumerate() {
                a[p] = row[1 + j];
            }
            e[i] * w_rows.map_or(1.0, |w| w[i])
        },
    };
    let (vcov, notes) = vcov::compute(&base, spec.vcov, &spec.ssc, &fe_info, &cl_info, &refs, &scores, pool)?;
    crate::solve::log_phase(ctx, &mut t, "fit: vcov");

    // Per-observation outputs (estimation order) and fit statistics.
    let kept_index = session.kept_index();
    let resid = to_kept_order(&e, session.row_order(), pool)?;
    drop(e);
    let mut fitted = par::alloc_vec(n, 0.0)?;
    pool.install(|| {
        fitted.par_iter_mut().zip(resid.par_iter()).zip(kept_index.par_iter()).for_each(|((f, &res), &i)| *f = input.y[i as usize] - res)
    });
    let cor2 = sq_cor(input.y, kept_index, &fitted, pool);
    let nf = n as f64;
    let weighted = w_rows.is_some();
    let ssr_fe_only = if has_fe { g[0] } else { f64::NAN };
    let only_intercept = np == 1 && intercept && k == 1;
    let n_stat = if spec.flavor == Flavor::Pyfixest { n_eff } else { nf };
    let di = if has_fe || intercept { 1.0 } else { 0.0 };
    let (r2, ar2) = match spec.flavor {
        _ if only_intercept && spec.flavor == Flavor::Fixest => (f64::NAN, f64::NAN),
        Flavor::Fixest => (1.0 - ssr / ssr_null * (nf - di) / (nf - 1.0), 1.0 - ssr / ssr_null * (nf - di) / (nf - np as f64)),
        Flavor::Pyfixest => {
            let hi = if spec.intercept { 1.0 } else { 0.0 };
            (1.0 - ssr / ssr_null, 1.0 - ssr / ssr_null * (n_eff - hi) / (n_eff - np as f64))
        }
    };
    let nb_fe = (np - k) as f64;
    let (wr2, war2) = if has_fe && k > 0 {
        (1.0 - ssr / ssr_fe_only, 1.0 - ssr / ssr_fe_only * (n_stat - nb_fe) / (n_stat - np as f64))
    } else {
        (f64::NAN, f64::NAN)
    };
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    let loglik = -0.5 * nf * (1.0 + ln2pi + (sse / nf).ln());
    let wn = if weighted { sumw } else { nf };
    let ll_null = -0.5 * wn * (1.0 + ln2pi + (ssr_null / wn).ln());
    let (pseudo_r2, apr2) =
        if only_intercept { (f64::NAN, f64::NAN) } else { (1.0 - loglik / ll_null, 1.0 - (loglik - np as f64 + 1.0) / ll_null) };
    let rmse = match spec.flavor {
        Flavor::Fixest => (ssr / wn).sqrt(),
        Flavor::Pyfixest => (ssr / n_eff).sqrt(),
    };
    let f_test = match vcov.df_t {
        Some(df2) => {
            let df1 = vcov.df_k - 1.0;
            let stat = ((ssr_null - ssr) / df1) / (ssr / df2);
            TestStat { stat, p: dist::pf_upper(stat, df1, df2), df1, df2 }
        }
        None => TestStat::NA,
    };
    let stats = FitStats {
        ssr,
        ssr_null,
        ssr_fe_only,
        sigma2: ssr / (nf - np as f64),
        rmse,
        r2,
        ar2,
        wr2,
        war2,
        loglik,
        ll_null,
        deviance: ssr,
        null_deviance: ssr_null,
        pseudo_r2,
        apr2,
        sq_cor: cor2,
        f_test,
        fe_iterations,
        irls_iterations: 0,
        converged: true,
    };
    crate::solve::log_phase(ctx, &mut t, "fit: outputs");
    Ok(FitOutput {
        nobs: n,
        nobs_origin: input.n,
        sumw,
        obs: kept_index.to_vec(),
        dropped,
        coef_source,
        coef,
        collinear,
        vcov,
        nparams: np,
        fe_sizes: session.terms().iter().map(|t| t.g).collect(),
        stats,
        iv: None,
        resid: spec.keep_resid.then_some(resid),
        fitted: spec.keep_resid.then_some(fitted),
        eta: None,
        irls_weights: None,
        fixef: Vec::new(),
        notes,
        state: None,
    })
}
