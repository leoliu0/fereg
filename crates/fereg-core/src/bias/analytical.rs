//! Analytical bias correction of the coefficients (design sec. 3) and the exact re-estimation of
//! the fixed effects at given coefficients (sec. 3.3).

use super::{
    BiasSpec, FeCodes, LogitData, Normalization, Structure, WhichBias, center, check_data, check_structure, chol_solve, default_key, derivs,
    invalid, lag_term, ordered_groups, wcross, CENTER_MAXITER, CENTER_TOL,
};
use crate::absorber::{Absorber, FeCoefs, SolveOpts};
use crate::api::Ctx;
use crate::error::RfError;
use crate::glm::family::{self, logit_ll};
use crate::par::{self, Pool};

/// Result of [`analytical`].
#[derive(Clone, Debug)]
pub struct Corrected {
    /// beta_c = beta - bias_term.
    pub coef: Vec<f64>,
    /// alpaca's `bias.term` = solve(H/n, -b) = beta - beta_c.
    pub bias_term: Vec<f64>,
    /// The raw bias vector b (beta_c = beta + (H/n)^-1 b).
    pub b: Vec<f64>,
    /// Linear predictor at beta_c with the fixed effects re-estimated (kept order, offset included).
    pub eta: Vec<f64>,
    /// The absorber's FE coefficients of that solution (one column; `Absorber::term_values`).
    pub fe_coefs: FeCoefs,
    /// Newton iterations of the re-estimation.
    pub iterations: usize,
    /// Plain-text notes (e.g. a bandwidth that does not apply).
    pub notes: Vec<String>,
}

/// Result of [`resolve_fe`].
#[derive(Clone, Debug)]
pub struct Resolved {
    /// Linear predictor (kept order, offset included).
    pub eta: Vec<f64>,
    pub fe_coefs: FeCoefs,
    pub iterations: usize,
}

/// The resolved bias request: which static terms, the spectral term, and the notes.
pub(crate) struct Plan {
    /// Alpaca: FE terms of the static group-ratio sums.
    pub stat: Vec<usize>,
    /// Alpaca: the lag-group term of the spectral sum (L > 0 only).
    pub spec: Option<usize>,
    /// Logitfe: individual and time biases.
    pub ib: bool,
    pub tb: bool,
    pub bandwidth: usize,
    pub notes: Vec<String>,
}

/// Validates `spec` against the data and resolves it.
pub(crate) fn plan(d: &LogitData, spec: &BiasSpec) -> Result<Plan, RfError> {
    let k = d.fe.len();
    let n = d.y.len();
    check_structure(spec.structure, k)?;
    let l = spec.bandwidth;
    let mut notes = Vec::new();
    if spec.structure == Structure::Network && spec.which != WhichBias::Both {
        return invalid("network structure: only the correction of both biases is available");
    }
    match spec.normalization {
        Normalization::Alpaca => {
            if spec.order.is_some_and(|o| o.len() != n) {
                return invalid("bias correction: the time order needs one value per kept row");
            }
            let (stat, lag_used): (Vec<usize>, bool) = match (spec.structure, spec.which) {
                (Structure::Network, _) => ((0..k).collect(), true),
                (Structure::Classic, WhichBias::Both) => ((0..k).collect(), true),
                (Structure::Classic, WhichBias::Individual) => (vec![0], true),
                (Structure::Classic, WhichBias::Time) => {
                    if k < 2 {
                        return invalid("the time-effect bias needs a time fixed effect (a two-way model)");
                    }
                    (vec![1], false)
                }
            };
            let lt = lag_term(spec.structure, k);
            if l > 0 && !lag_used {
                return invalid(format!("bandwidth L = {l} has no effect: it only applies to the individual-effect bias"));
            }
            let spec_term = if l > 0 {
                match lt {
                    Some(q) => Some(q),
                    None => {
                        // alpaca accepts it (and skips the spectral term): a note, not an error.
                        notes.push(format!("L ignored: no pair fixed effect (two-way network model), bandwidth L = {l} has no effect"));
                        None
                    }
                }
            } else {
                None
            };
            if let Some(q) = spec_term {
                let mut t = vec![0usize; d.fe[q].g];
                d.fe[q].codes.iter().for_each(|&c| t[c as usize] += 1);
                let tmax = t.iter().copied().max().unwrap_or(0);
                if l >= tmax {
                    return invalid(format!("bandwidth L = {l} is too large: the longest group has {tmax} observations"));
                }
            }
            Ok(Plan { stat, spec: spec_term, ib: false, tb: false, bandwidth: if spec_term.is_some() { l } else { 0 }, notes })
        }
        Normalization::Logitfe => {
            if spec.structure != Structure::Classic {
                return invalid("the logitfe normalization is only defined for the classic structure");
            }
            if d.weights.is_some() {
                return invalid("the logitfe normalization does not allow weights");
            }
            let Some(p) = spec.logitfe else { return invalid("the logitfe normalization needs the panel identifiers") };
            if p.individual.len() != n || p.period.len() != n || p.individual.iter().any(|&c| c as usize >= p.n_individuals) {
                return invalid("bias correction: panel identifiers must be dense codes, one per kept row");
            }
            let ib = matches!(spec.which, WhichBias::Both | WhichBias::Individual) && p.individual_effects;
            let tb = matches!(spec.which, WhichBias::Both | WhichBias::Time) && p.time_effects;
            if spec.which == WhichBias::Individual && !p.individual_effects {
                return invalid("the individual-effect bias needs individual fixed effects");
            }
            if spec.which == WhichBias::Time && !p.time_effects {
                return invalid("the time-effect bias needs time fixed effects");
            }
            if !ib && !tb {
                return invalid("bias correction: no bias to correct");
            }
            let bandwidth = l;
            if l > 0 && !ib {
                return invalid(format!("lags({l}) has no effect: lags only apply to the individual-effect bias"));
            }
            if bandwidth > 0 {
                let mut t = vec![0usize; p.n_individuals];
                p.individual.iter().for_each(|&c| t[c as usize] += 1);
                let tmin = t.iter().copied().filter(|&c| c > 0).min().unwrap_or(0);
                if bandwidth + 1 > tmin {
                    return invalid(format!("lags({bandwidth}) is too large: maximum possible lags are {} (shortest time series minus one)", tmin.saturating_sub(1)));
                }
            }
            Ok(Plan { stat: Vec::new(), spec: None, ib, tb, bandwidth, notes })
        }
    }
}

/// The bandwidth that enters the correction (0 when it does not apply).
pub(crate) fn plan_bandwidth(d: &LogitData, spec: &BiasSpec) -> Result<usize, RfError> {
    Ok(plan(d, spec)?.bandwidth)
}

/// sum_g (sum_{r in g} A_r)/(sum_{r in g} w_r) over the groups of `f` (p-vector), in group order.
fn group_ratio(a: &[Vec<f64>], w: &[f64], f: &FeCodes) -> Vec<f64> {
    let p = a.len();
    let mut num = vec![0.0; f.g * p];
    let mut den = vec![0.0; f.g];
    for (r, &c) in f.codes.iter().enumerate() {
        let c = c as usize;
        den[c] += w[r];
        for j in 0..p {
            num[c * p + j] += a[j][r];
        }
    }
    let mut out = vec![0.0; p];
    for c in 0..f.g {
        if den[c] > 0.0 {
            for j in 0..p {
                out[j] += num[c * p + j] / den[c];
            }
        }
    }
    out
}

/// alpaca's spectral sum: sum_g [sum_{l=1..L} sum_{t=l..T_g-1} A_{g[t]} v_{g[t-l]} T_g/(T_g-l)] / sum_g w.
fn spectral(a: &[Vec<f64>], v: &[f64], w: &[f64], l: usize, off: &[usize], ids: &[u32]) -> Vec<f64> {
    let p = a.len();
    let mut out = vec![0.0; p];
    let mut num = vec![0.0; p];
    for q in 0..off.len() - 1 {
        let g = &ids[off[q]..off[q + 1]];
        let tg = g.len();
        num.iter_mut().for_each(|x| *x = 0.0);
        for lag in 1..=l {
            if tg <= lag {
                continue;
            }
            let f = tg as f64 / (tg - lag) as f64;
            for t in lag..tg {
                let (r, s) = (g[t] as usize, g[t - lag] as usize);
                for j in 0..p {
                    num[j] += a[j][r] * v[s] * f;
                }
            }
        }
        let sw: f64 = g.iter().map(|&r| w[r as usize]).sum();
        for j in 0..p {
            out[j] += num[j] / sw;
        }
    }
    out
}

/// The bias sum shared by the coefficients and the APEs:
/// - Alpaca: sum over the static terms of group_ratio(A)/(2n) + `sign` * spectral(Bl, v)/n;
/// - Logitfe: ib * mean_{i in I_L} [ (1/2) sum_t A/sum_t w/T_i + `sign` * sum_j sum_t v_{i,t-j} Bl_it / sum_t w / TN_ij ]
///   + tb * mean_t [ (1/2) sum_i A / sum_i w ] / ng.
///
/// Coefficients: A = MX z, Bl = MX w, sign +1 (b of beta_c = beta + (H/n)^-1 b). APEs: A = Delta2 + P Psi z,
/// Bl = M Psi w, sign -1 (delta_c = delta - b).
#[allow(clippy::too_many_arguments)]
pub(crate) fn bias_sum(d: &LogitData, spec: &BiasSpec, plan: &Plan, a: &[Vec<f64>], bl: &[Vec<f64>], w: &[f64], v: &[f64], sign: f64) -> Result<Vec<f64>, RfError> {
    let n = d.y.len();
    let p = a.len();
    let mut b = vec![0.0; p];
    match spec.normalization {
        Normalization::Alpaca => {
            for &q in &plan.stat {
                let s = group_ratio(a, w, &d.fe[q]);
                b.iter_mut().zip(s).for_each(|(b, s)| *b += s / (2.0 * n as f64));
            }
            if let Some(q) = plan.spec {
                let f = &d.fe[q];
                let (off, ids) = match spec.order {
                    Some(o) => ordered_groups(f.codes, f.g, Some(&|r| o[r])),
                    None => {
                        let key = default_key(spec.structure, &d.fe);
                        ordered_groups(f.codes, f.g, key.as_deref())
                    }
                };
                let s = spectral(bl, v, w, plan.bandwidth, &off, &ids);
                b.iter_mut().zip(s).for_each(|(b, s)| *b += sign * s / n as f64);
            }
        }
        Normalization::Logitfe => {
            let pn = spec.logitfe.expect("checked by plan");
            if plan.ib {
                let l = plan.bandwidth;
                let (off, ids) = ordered_groups(pn.individual, pn.n_individuals, Some(&|r| pn.period[r]));
                let mut acc = vec![0.0; p];
                let mut count = 0usize;
                let mut stat = vec![0.0; p];
                let mut lagsum = vec![0.0; p];
                let mut term = vec![0.0; p];
                for i in 0..pn.n_individuals {
                    let g = &ids[off[i]..off[i + 1]];
                    if g.is_empty() {
                        continue;
                    }
                    for t in 1..g.len() {
                        if pn.period[g[t] as usize] == pn.period[g[t - 1] as usize] {
                            return invalid("bias correction: repeated (individual, period) pair in the sample");
                        }
                    }
                    let ti = g.len() as f64;
                    let sw: f64 = g.iter().map(|&r| w[r as usize]).sum();
                    stat.iter_mut().for_each(|x| *x = 0.0);
                    for &r in g {
                        for j in 0..p {
                            stat[j] += a[j][r as usize];
                        }
                    }
                    for j in 0..p {
                        term[j] = 0.5 * stat[j] / sw / ti;
                    }
                    let mut complete = true;
                    for lag in 1..=l as i64 {
                        lagsum.iter_mut().for_each(|x| *x = 0.0);
                        let mut tn = 0usize;
                        // Rows sorted by period: the row of period t - lag, if present, is found by
                        // a merge-like scan.
                        let mut s = 0usize;
                        for &r in g {
                            let want = pn.period[r as usize] - lag;
                            while s < g.len() && pn.period[g[s] as usize] < want {
                                s += 1;
                            }
                            if s < g.len() && pn.period[g[s] as usize] == want {
                                tn += 1;
                                let vs = v[g[s] as usize];
                                for j in 0..p {
                                    lagsum[j] += vs * bl[j][r as usize];
                                }
                            }
                        }
                        if tn == 0 {
                            complete = false;
                            break;
                        }
                        for j in 0..p {
                            term[j] += sign * lagsum[j] / sw / tn as f64;
                        }
                    }
                    if complete {
                        count += 1;
                        acc.iter_mut().zip(&term).for_each(|(a, t)| *a += t);
                    }
                }
                if count == 0 {
                    return invalid("bias correction: no individual has every lag in the sample");
                }
                b.iter_mut().zip(&acc).for_each(|(b, a)| *b += a / count as f64);
            }
            if plan.tb {
                let (codes, nt) = dense_rank(pn.period);
                let mut seen = vec![false; pn.n_individuals];
                pn.individual.iter().for_each(|&c| seen[c as usize] = true);
                let ng = seen.iter().filter(|s| **s).count() as f64;
                let s = group_ratio(a, w, &FeCodes { codes: &codes, g: nt });
                b.iter_mut().zip(s).for_each(|(b, s)| *b += 0.5 * s / nt as f64 / ng);
            }
        }
    }
    Ok(b)
}

/// Dense codes of integer values in ascending order and their number.
pub(crate) fn dense_rank(v: &[i64]) -> (Vec<u32>, usize) {
    let mut u: Vec<i64> = v.to_vec();
    u.sort_unstable();
    u.dedup();
    (v.iter().map(|x| u.binary_search(x).expect("present") as u32).collect(), u.len())
}

/// Corrected coefficients (design sec. 3) at the fit (`coef`, `eta`: the uncorrected estimates,
/// eta in kept order with the offset), then the fixed effects re-estimated at them ([`resolve_fe`]).
/// The absorber is left with the weights of the last solve.
pub fn analytical(d: &LogitData, abs: &mut dyn Absorber, coef: &[f64], eta: &[f64], spec: &BiasSpec, ctx: &Ctx) -> Result<Corrected, RfError> {
    let n = check_data(d, abs, coef, eta)?;
    let plan = plan(d, spec)?;
    par::with_pool(par::threads_for(ctx.threads, n), |pool| {
        let (bias_term, b) = coef_bias(d, abs, eta, spec, &plan, pool, ctx)?;
        let coef_c: Vec<f64> = coef.iter().zip(&bias_term).map(|(c, t)| c - t).collect();
        let start = shifted_eta(d, eta, coef, &coef_c);
        let r = resolve_in(d, abs, &coef_c, &start, pool, ctx)?;
        Ok(Corrected { coef: coef_c, bias_term, b, eta: r.eta, fe_coefs: r.fe_coefs, iterations: r.iterations, notes: plan.notes })
    })?
}

/// (bias_term, b) at the point.
fn coef_bias(d: &LogitData, abs: &mut dyn Absorber, eta: &[f64], spec: &BiasSpec, plan: &Plan, pool: &Pool, ctx: &Ctx) -> Result<(Vec<f64>, Vec<f64>), RfError> {
    let n = d.y.len();
    let p = d.x.len();
    let wt = |r: usize| d.weights.map_or(1.0, |w| w[r]);
    let mut w = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut z = vec![0.0; n];
    for r in 0..n {
        let (m, d1, d2, _) = derivs(eta[r]);
        w[r] = wt(r) * d1;
        v[r] = wt(r) * (d.y[r] - m);
        z[r] = wt(r) * d2;
    }
    let mx = center(abs, &d.x, &w, pool, ctx)?;
    ctx.check()?;
    let h = wcross(&mx, &w);
    let a: Vec<Vec<f64>> = mx.iter().map(|c| c.iter().zip(&z).map(|(x, z)| x * z).collect()).collect();
    let bl: Vec<Vec<f64>> = mx.iter().map(|c| c.iter().zip(&w).map(|(x, w)| x * w).collect()).collect();
    let b = bias_sum(d, spec, plan, &a, &bl, &w, &v, 1.0)?;
    let hn: Vec<f64> = h.iter().map(|x| x / n as f64).collect();
    let neg: Vec<f64> = b.iter().map(|x| -x).collect();
    let bias_term = chol_solve(&hn, p, &neg, 1)?;
    Ok((bias_term, b))
}

/// eta + X (to - from): the uncorrected fixed effects with the linear index at `to`.
pub(crate) fn shifted_eta(d: &LogitData, eta: &[f64], from: &[f64], to: &[f64]) -> Vec<f64> {
    let delta: Vec<f64> = to.iter().zip(from).map(|(a, b)| a - b).collect();
    (0..eta.len()).map(|r| eta[r] + d.x.iter().zip(&delta).map(|(x, c)| x[r] * c).sum::<f64>()).collect()
}

/// Newton iterations of the fixed-effects-only problem.
const RESOLVE_MAXITER: usize = 100;
/// Convergence: largest change of eta in a full Newton step.
const RESOLVE_TOL: f64 = 1e-10;

/// Maximum-likelihood fixed effects with the linear index `X coef` (+ offset) held fixed: Newton
/// (IRLS on the fixed effects only) from `eta_start` (kept order, offset included; e.g. the
/// uncorrected fixed effects plus X coef), each step an exact weighted FE projection through the
/// absorber at tolerance 1e-13, with step halving on the deviance; converged when a full step moves
/// eta by less than 1e-10 everywhere. Not converging in 100 steps is an error. The absorber is left
/// with the weights of the last step.
pub fn resolve_fe(d: &LogitData, abs: &mut dyn Absorber, coef: &[f64], eta_start: &[f64], ctx: &Ctx) -> Result<Resolved, RfError> {
    let n = check_data(d, abs, coef, eta_start)?;
    par::with_pool(par::threads_for(ctx.threads, n), |pool| resolve_in(d, abs, coef, eta_start, pool, ctx))?
}

fn deviance(y: &[f64], wt: &[f64], eta: &[f64]) -> f64 {
    -2.0 * y.iter().zip(wt).zip(eta).map(|((y, w), e)| w * logit_ll(*y, *e)).sum::<f64>()
}

pub(crate) fn resolve_in(d: &LogitData, abs: &mut dyn Absorber, coef: &[f64], eta_start: &[f64], pool: &Pool, ctx: &Ctx) -> Result<Resolved, RfError> {
    let order = abs.row_order().to_vec();
    let n = order.len();
    let at = |k: usize| order[k] as usize;
    let off: Vec<f64> = (0..n)
        .map(|k| {
            let i = at(k);
            d.offset.map_or(0.0, |o| o[i]) + d.x.iter().zip(coef).map(|(x, c)| x[i] * c).sum::<f64>()
        })
        .collect();
    let y: Vec<f64> = (0..n).map(|k| d.y[at(k)]).collect();
    let wt: Vec<f64> = (0..n).map(|k| d.weights.map_or(1.0, |w| w[at(k)])).collect();
    let mut eta: Vec<f64> = (0..n).map(|k| eta_start[at(k)]).collect();
    let mut dev = deviance(&y, &wt, &eta);
    let mut w = vec![0.0; n];
    let mut u = vec![0.0; n];
    let opts = SolveOpts { tol: CENTER_TOL, maxiter: CENTER_MAXITER, want_coefs: false };
    for it in 1..=RESOLVE_MAXITER {
        ctx.check()?;
        // Newton step in increment form: eta += P_w (y - mu)/mu' (P_w = the weighted FE projection).
        for k in 0..n {
            let m = family::mean(crate::api::Family::Logit, eta[k]);
            let d1 = m * (1.0 - m);
            w[k] = wt[k] * d1;
            u[k] = (y[k] - m) / d1;
        }
        abs.set_weights(Some(&w), pool, ctx)?;
        let mut block = u.clone();
        abs.residualize(&mut block, 1, &opts, None, pool, ctx)?;
        let mut next: Vec<f64> = (0..n).map(|k| eta[k] + (u[k] - block[k])).collect();
        let mut dev_next = deviance(&y, &wt, &next);
        let mut halved = false;
        let mut tries = 0;
        while dev_next.is_nan() || dev_next > dev + 1e-12 * (0.1 + dev.abs()) {
            tries += 1;
            if tries > 50 {
                return Err(RfError::Code(430, "bias correction: the re-estimation of the fixed effects could not decrease the deviance".into()));
            }
            halved = true;
            next.iter_mut().zip(&eta).for_each(|(a, b)| *a = 0.5 * (*a + b));
            dev_next = deviance(&y, &wt, &next);
        }
        let step = next.iter().zip(&eta).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        if !step.is_finite() {
            return Err(RfError::Code(430, "bias correction: the re-estimation of the fixed effects diverged".into()));
        }
        eta = next;
        dev = dev_next;
        if !halved && step < RESOLVE_TOL {
            // FE coefficients of the solution: the FE part eta - offset - X coef lies in the FE space.
            let mut fe: Vec<f64> = (0..n).map(|k| eta[k] - off[k]).collect();
            let info = abs.residualize(&mut fe, 1, &SolveOpts { want_coefs: true, ..opts }, None, pool, ctx)?;
            let fe_coefs = info.coefs.ok_or_else(|| RfError::Internal("absorber returned no FE coefficients".into()))?;
            let mut out = vec![0.0; n];
            for k in 0..n {
                out[at(k)] = eta[k];
            }
            return Ok(Resolved { eta: out, fe_coefs, iterations: it });
        }
    }
    Err(RfError::Code(
        430,
        format!("bias correction: the fixed effects at the corrected coefficients did not converge in {RESOLVE_MAXITER} Newton iterations"),
    ))
}
