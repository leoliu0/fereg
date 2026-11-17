//! Analytical bias correction of the coefficients (design sec. 3) and the exact re-estimation of
//! the fixed effects at given coefficients (sec. 3.3).

use rayon::prelude::*;

use super::{
    BiasSpec, FeCodes, LogitData, Normalization, Structure, WhichBias, center, check_data, check_structure, chol_solve, default_key, derivs, groups, invalid,
    lag_term, ordered_groups, wcross, zeros, CENTER_MAXITER, CENTER_TOL,
};
use crate::absorber::{Absorber, FeCoefs, SolveOpts};
use crate::api::Ctx;
use crate::error::RfError;
use crate::glm::family::{self, logit_ll};
use crate::par::{self, CHUNK, Pool};

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
            let lt = lag_term(spec.structure, k, spec.lag_term)?;
            let (stat, lag_used): (Vec<usize>, bool) = match (spec.structure, spec.which) {
                (Structure::Network, _) => ((0..k).collect(), true),
                (Structure::Classic, WhichBias::Both) => ((0..k).collect(), true),
                (Structure::Classic, WhichBias::Individual) => (vec![lt.unwrap_or(0)], true),
                (Structure::Classic, WhichBias::Time) => {
                    if k < 2 {
                        return invalid("the time-effect bias needs a time fixed effect (a two-way model)");
                    }
                    (vec![1 - lt.unwrap_or(0)], false)
                }
            };
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

/// sum_g (sum_{r in g} A_r)/(sum_{r in g} w_r) over the groups of `f` (p-vector), deterministic.
fn group_ratio(a: &[Vec<f64>], w: &[f64], f: &FeCodes, pool: &Pool) -> Result<Vec<f64>, RfError> {
    let p = a.len();
    let n = f.codes.len();
    if f.g.saturating_mul(p + 1) <= 4096 {
        let stride = p + 1;
        let sums = par::sum_chunks(pool, n, CHUNK, f.g * stride, |range, acc| {
            for r in range {
                let base = f.codes[r] as usize * stride;
                acc[base] += w[r];
                for j in 0..p {
                    acc[base + 1 + j] += a[j][r];
                }
            }
        });
        let mut out = vec![0.0; p];
        for c in 0..f.g {
            let base = c * stride;
            let den = sums[base];
            if den > 0.0 {
                for j in 0..p {
                    out[j] += sums[base + 1 + j] / den;
                }
            }
        }
        return Ok(out);
    }
    let (off, ids) = groups(f.codes, f.g, pool)?;
    Ok(par::sum_chunks(pool, f.g, 256, p, |range, out| {
        let mut num = vec![0.0; p];
        for c in range {
            let rows = &ids[off[c]..off[c + 1]];
            if rows.is_empty() {
                continue;
            }
            let mut den = 0.0;
            num.fill(0.0);
            for &r in rows {
                let r = r as usize;
                den += w[r];
                for j in 0..p {
                    num[j] += a[j][r];
                }
            }
            if den > 0.0 {
                for j in 0..p {
                    out[j] += num[j] / den;
                }
            }
        }
    }))
}

/// alpaca's spectral sum: sum_g [sum_{l=1..L} sum_{t=l..T_g-1} A_{g[t]} v_{g[t-l]} T_g/(T_g-l)] / sum_g w.
fn spectral(a: &[Vec<f64>], v: &[f64], w: &[f64], l: usize, off: &[usize], ids: &[u32], pool: &Pool) -> Vec<f64> {
    let p = a.len();
    let ng = off.len().saturating_sub(1);
    par::sum_chunks(pool, ng, 256, p, |range, out| {
        let mut num = vec![0.0; p];
        for q in range {
            let g = &ids[off[q]..off[q + 1]];
            let tg = g.len();
            if tg == 0 {
                continue;
            }
            num.fill(0.0);
            for lag in 1..=l {
                if tg <= lag {
                    continue;
                }
                let f = tg as f64 / (tg - lag) as f64;
                for t in lag..tg {
                    let (r, s) = (g[t] as usize, g[t - lag] as usize);
                    let vsf = v[s] * f;
                    for j in 0..p {
                        num[j] += a[j][r] * vsf;
                    }
                }
            }
            let sw: f64 = g.iter().map(|&r| w[r as usize]).sum();
            for j in 0..p {
                out[j] += num[j] / sw;
            }
        }
    })
}

/// The bias sum shared by the coefficients and the APEs:
/// - Alpaca: sum over the static terms of group_ratio(A)/(2n) + `sign` * spectral(Bl, v)/n;
/// - Logitfe: ib * mean_{i in I_L} [ (1/2) sum_t A/sum_t w/T_i + `sign` * sum_j sum_t v_{i,t-j} Bl_it / sum_t w / TN_ij ]
///   + tb * mean_t [ (1/2) sum_i A / sum_i w ] / ng.
///
/// Coefficients: A = MX z, Bl = MX w, sign +1 (b of beta_c = beta + (H/n)^-1 b). APEs: A = Delta2 + P Psi z,
/// Bl = M Psi w, sign -1 (delta_c = delta - b).
#[allow(clippy::too_many_arguments)]
pub(crate) fn bias_sum(
    d: &LogitData,
    spec: &BiasSpec,
    plan: &Plan,
    a: &[Vec<f64>],
    bl: &[Vec<f64>],
    w: &[f64],
    v: &[f64],
    sign: f64,
    pool: &Pool,
) -> Result<Vec<f64>, RfError> {
    let n = d.y.len();
    let p = a.len();
    let mut b = vec![0.0; p];
    match spec.normalization {
        Normalization::Alpaca => {
            for &q in &plan.stat {
                let s = group_ratio(a, w, &d.fe[q], pool)?;
                b.iter_mut().zip(s).for_each(|(b, s)| *b += s / (2.0 * n as f64));
            }
            if let Some(q) = plan.spec {
                let f = &d.fe[q];
                let (off, ids) = match spec.order {
                    Some(o) => ordered_groups(f.codes, f.g, Some(&|r| o[r]), pool)?,
                    None => {
                        let key = default_key(&d.fe, q);
                        ordered_groups(f.codes, f.g, key.as_deref(), pool)?
                    }
                };
                let s = spectral(bl, v, w, plan.bandwidth, &off, &ids, pool);
                b.iter_mut().zip(s).for_each(|(b, s)| *b += sign * s / n as f64);
            }
        }
        Normalization::Logitfe => {
            let pn = spec.logitfe.expect("checked by plan");
            if plan.ib {
                let l = plan.bandwidth;
                let (off, ids) = ordered_groups(pn.individual, pn.n_individuals, Some(&|r| pn.period[r]), pool)?;
                struct IndChunk {
                    dup: bool,
                    count: usize,
                    acc: Vec<f64>,
                }
                let chunks = par::map_chunks(pool, pn.n_individuals, 256, |_, range| {
                    let mut acc = vec![0.0; p];
                    let mut count = 0usize;
                    let mut stat = vec![0.0; p];
                    let mut lagsum = vec![0.0; p];
                    let mut term = vec![0.0; p];
                    for i in range {
                        let g = &ids[off[i]..off[i + 1]];
                        if g.is_empty() {
                            continue;
                        }
                        for t in 1..g.len() {
                            if pn.period[g[t] as usize] == pn.period[g[t - 1] as usize] {
                                return IndChunk { dup: true, count: 0, acc };
                            }
                        }
                        let ti = g.len() as f64;
                        let sw: f64 = g.iter().map(|&r| w[r as usize]).sum();
                        stat.fill(0.0);
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
                            lagsum.fill(0.0);
                            let mut tn = 0usize;
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
                    IndChunk { dup: false, count, acc }
                });
                let mut acc = vec![0.0; p];
                let mut count = 0usize;
                for ch in chunks {
                    if ch.dup {
                        return invalid("bias correction: repeated (individual, period) pair in the sample");
                    }
                    count += ch.count;
                    acc.iter_mut().zip(&ch.acc).for_each(|(a, x)| *a += x);
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
                let s = group_ratio(a, w, &FeCodes { codes: &codes, g: nt }, pool)?;
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
    par::with_pool(par::threads_for(ctx.threads, n), |pool| analytical_inner(d, abs, coef, eta, spec, plan, pool, ctx))?
}

/// [`analytical`] running on an existing [`Pool`].
pub(crate) fn analytical_in(
    d: &LogitData,
    abs: &mut dyn Absorber,
    coef: &[f64],
    eta: &[f64],
    spec: &BiasSpec,
    pool: &Pool,
    ctx: &Ctx,
) -> Result<Corrected, RfError> {
    check_data(d, abs, coef, eta)?;
    let plan = plan(d, spec)?;
    analytical_inner(d, abs, coef, eta, spec, plan, pool, ctx)
}

#[allow(clippy::too_many_arguments)]
fn analytical_inner(
    d: &LogitData,
    abs: &mut dyn Absorber,
    coef: &[f64],
    eta: &[f64],
    spec: &BiasSpec,
    plan: Plan,
    pool: &Pool,
    ctx: &Ctx,
) -> Result<Corrected, RfError> {
    let (bias_term, b) = coef_bias(d, abs, eta, spec, &plan, pool, ctx)?;
    let coef_c: Vec<f64> = coef.iter().zip(&bias_term).map(|(c, t)| c - t).collect();
    let start = shifted_eta(d, eta, coef, &coef_c, pool);
    let r = resolve_in(d, abs, &coef_c, &start, pool, ctx)?;
    Ok(Corrected { coef: coef_c, bias_term, b, eta: r.eta, fe_coefs: r.fe_coefs, iterations: r.iterations, notes: plan.notes })
}

/// (bias_term, b) at the point.
fn coef_bias(d: &LogitData, abs: &mut dyn Absorber, eta: &[f64], spec: &BiasSpec, plan: &Plan, pool: &Pool, ctx: &Ctx) -> Result<(Vec<f64>, Vec<f64>), RfError> {
    let n = d.y.len();
    let p = d.x.len();
    let wt = |r: usize| d.weights.map_or(1.0, |w| w[r]);
    let mut w = par::alloc_uninit::<f64>(n)?;
    let mut v = par::alloc_uninit::<f64>(n)?;
    let mut z = par::alloc_uninit::<f64>(n)?;
    if n <= CHUNK || pool.threads() <= 1 {
        for r in 0..n {
            let (m, d1, d2, _) = derivs(eta[r]);
            let wr = wt(r);
            w[r].write(wr * d1);
            v[r].write(wr * (d.y[r] - m));
            z[r].write(wr * d2);
        }
    } else {
        pool.install(|| {
            w.par_chunks_mut(CHUNK)
                .zip(v.par_chunks_mut(CHUNK))
                .zip(z.par_chunks_mut(CHUNK))
                .enumerate()
                .for_each(|(ci, ((wc, vc), zc))| {
                    let base = ci * CHUNK;
                    for (j, ((wo, vo), zo)) in wc.iter_mut().zip(vc).zip(zc).enumerate() {
                        let r = base + j;
                        let (m, d1, d2, _) = derivs(eta[r]);
                        let wr = wt(r);
                        wo.write(wr * d1);
                        vo.write(wr * (d.y[r] - m));
                        zo.write(wr * d2);
                    }
                })
        });
    }
    // SAFETY: every slot of `0..n` in `w`, `v`, `z` was written above.
    let (w, v, z) = unsafe { (par::assume_init(w), par::assume_init(v), par::assume_init(z)) };
    let mx = center(abs, &d.x, &w, pool, ctx)?;
    ctx.check()?;
    let h = wcross(&mx, &w, pool);
    // A = MX z (new), Bl = MX w (over MX).
    let mut a_uninit: Vec<Vec<std::mem::MaybeUninit<f64>>> = (0..p).map(|_| par::alloc_uninit::<f64>(n)).collect::<Result<_, _>>()?;
    let mut bl = mx;
    for (ac, bc) in a_uninit.iter_mut().zip(bl.iter_mut()) {
        if n <= CHUNK || pool.threads() <= 1 {
            for r in 0..n {
                ac[r].write(bc[r] * z[r]);
                bc[r] *= w[r];
            }
        } else {
            pool.install(|| {
                ac.par_chunks_mut(CHUNK)
                    .zip(bc.par_chunks_mut(CHUNK))
                    .zip(z.par_chunks(CHUNK).zip(w.par_chunks(CHUNK)))
                    .for_each(|((ao, bo), (zc, wc))| {
                        for ((av, bv), (&zv, &wv)) in ao.iter_mut().zip(bo).zip(zc.iter().zip(wc)) {
                            av.write(*bv * zv);
                            *bv *= wv;
                        }
                    })
            });
        }
    }
    drop(z);
    // SAFETY: every slot of `0..n` in each column of `a_uninit` was written above.
    let a: Vec<Vec<f64>> = a_uninit.into_iter().map(|col| unsafe { par::assume_init(col) }).collect();
    let b = bias_sum(d, spec, plan, &a, &bl, &w, &v, 1.0, pool)?;
    let hn: Vec<f64> = h.iter().map(|x| x / n as f64).collect();
    let neg: Vec<f64> = b.iter().map(|x| -x).collect();
    let bias_term = chol_solve(&hn, p, &neg, 1)?;
    Ok((bias_term, b))
}

/// eta + X (to - from): the uncorrected fixed effects with the linear index at `to`.
pub(crate) fn shifted_eta(d: &LogitData, eta: &[f64], from: &[f64], to: &[f64], pool: &Pool) -> Vec<f64> {
    let delta: Vec<f64> = to.iter().zip(from).map(|(a, b)| a - b).collect();
    let n = eta.len();
    if n <= CHUNK || pool.threads() <= 1 {
        return (0..n).map(|r| eta[r] + d.x.iter().zip(&delta).map(|(x, c)| x[r] * c).sum::<f64>()).collect();
    }
    pool.install(|| {
        (0..n)
            .into_par_iter()
            .with_min_len(CHUNK)
            .map(|r| eta[r] + d.x.iter().zip(&delta).map(|(x, c)| x[r] * c).sum::<f64>())
            .collect()
    })
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

fn deviance(y: &[f64], wt: &[f64], eta: &[f64], pool: &Pool) -> f64 {
    -2.0 * par::sum_chunks(pool, y.len(), CHUNK, 1, |range, acc| {
        for i in range {
            acc[0] += wt[i] * logit_ll(y[i], eta[i]);
        }
    })[0]
}

pub(crate) fn resolve_in(d: &LogitData, abs: &mut dyn Absorber, coef: &[f64], eta_start: &[f64], pool: &Pool, ctx: &Ctx) -> Result<Resolved, RfError> {
    let order = abs.row_order().to_vec();
    let n = order.len();
    let mut off = par::alloc_uninit::<f64>(n)?;
    let mut y = par::alloc_uninit::<f64>(n)?;
    let mut wt = par::alloc_uninit::<f64>(n)?;
    let mut eta = par::alloc_uninit::<f64>(n)?;
    if n <= CHUNK || pool.threads() <= 1 {
        for (k, &idx) in order.iter().enumerate() {
            let i = idx as usize;
            off[k].write(d.offset.map_or(0.0, |o| o[i]) + d.x.iter().zip(coef).map(|(x, c)| x[i] * c).sum::<f64>());
            y[k].write(d.y[i]);
            wt[k].write(d.weights.map_or(1.0, |w| w[i]));
            eta[k].write(eta_start[i]);
        }
    } else {
        pool.install(|| {
            off.par_chunks_mut(CHUNK)
                .zip(y.par_chunks_mut(CHUNK))
                .zip(wt.par_chunks_mut(CHUNK))
                .zip(eta.par_chunks_mut(CHUNK))
                .zip(order.par_chunks(CHUNK))
                .for_each(|((((oc, yc), wtc), ec), ord)| {
                    for ((((oo, yo), wto), eo), &idx) in oc.iter_mut().zip(yc).zip(wtc).zip(ec).zip(ord) {
                        let i = idx as usize;
                        oo.write(d.offset.map_or(0.0, |o| o[i]) + d.x.iter().zip(coef).map(|(x, c)| x[i] * c).sum::<f64>());
                        yo.write(d.y[i]);
                        wto.write(d.weights.map_or(1.0, |w| w[i]));
                        eo.write(eta_start[i]);
                    }
                })
        });
    }
    // SAFETY: every slot of `0..n` in `off`, `y`, `wt`, `eta` was written above.
    let (off, y, wt, mut eta) = unsafe { (par::assume_init(off), par::assume_init(y), par::assume_init(wt), par::assume_init(eta)) };
    let mut dev = deviance(&y, &wt, &eta, pool);
    let mut w = zeros(n, pool)?;
    let mut u = zeros(n, pool)?;
    let mut block = zeros(n, pool)?;
    let mut next = zeros(n, pool)?;
    let opts = SolveOpts { tol: CENTER_TOL, maxiter: CENTER_MAXITER, want_coefs: false };
    for it in 1..=RESOLVE_MAXITER {
        ctx.check()?;
        // Newton step in increment form: eta += P_w (y - mu)/mu' (P_w = the weighted FE projection).
        if n <= CHUNK || pool.threads() <= 1 {
            for k in 0..n {
                let m = family::mean(crate::api::Family::Logit, eta[k]);
                let d1 = m * (1.0 - m);
                w[k] = wt[k] * d1;
                u[k] = (y[k] - m) / d1;
                block[k] = u[k];
            }
        } else {
            pool.install(|| {
                w.par_chunks_mut(CHUNK)
                    .zip(u.par_chunks_mut(CHUNK))
                    .zip(block.par_chunks_mut(CHUNK))
                    .zip(eta.par_chunks(CHUNK).zip(wt.par_chunks(CHUNK)).zip(y.par_chunks(CHUNK)))
                    .for_each(|(((wc, uc), bc), ((ec, wtc), yc))| {
                        for (((wo, uo), bo), ((&ek, &wtk), &yk)) in wc.iter_mut().zip(uc).zip(bc).zip(ec.iter().zip(wtc).zip(yc)) {
                            let m = family::mean(crate::api::Family::Logit, ek);
                            let d1 = m * (1.0 - m);
                            *wo = wtk * d1;
                            let uk = (yk - m) / d1;
                            *uo = uk;
                            *bo = uk;
                        }
                    })
            });
        }
        abs.set_weights(Some(&w), pool, ctx)?;
        abs.residualize(&mut block, 1, &opts, None, pool, ctx)?;
        if n <= CHUNK || pool.threads() <= 1 {
            for k in 0..n {
                next[k] = eta[k] + (u[k] - block[k]);
            }
        } else {
            pool.install(|| {
                next.par_chunks_mut(CHUNK)
                    .zip(eta.par_chunks(CHUNK))
                    .zip(u.par_chunks(CHUNK).zip(block.par_chunks(CHUNK)))
                    .for_each(|((nc, ec), (uc, bc))| {
                        for ((no, &ek), (&uk, &bk)) in nc.iter_mut().zip(ec).zip(uc.iter().zip(bc)) {
                            *no = ek + (uk - bk);
                        }
                    })
            });
        }
        let mut dev_next = deviance(&y, &wt, &next, pool);
        let mut halved = false;
        let mut tries = 0;
        while dev_next.is_nan() || dev_next > dev + 1e-12 * (0.1 + dev.abs()) {
            tries += 1;
            if tries > 50 {
                return Err(RfError::Code(430, "bias correction: the re-estimation of the fixed effects could not decrease the deviance".into()));
            }
            halved = true;
            if n <= CHUNK || pool.threads() <= 1 {
                next.iter_mut().zip(&eta).for_each(|(a, b)| *a = 0.5 * (*a + b));
            } else {
                pool.install(|| next.par_chunks_mut(CHUNK).zip(eta.par_chunks(CHUNK)).for_each(|(nc, ec)| nc.iter_mut().zip(ec).for_each(|(a, b)| *a = 0.5 * (*a + b))));
            }
            dev_next = deviance(&y, &wt, &next, pool);
        }
        let step = par::map_chunks(pool, n, CHUNK, |_, range| {
            next[range.clone()].iter().zip(&eta[range]).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
        })
        .into_iter()
        .fold(0.0, f64::max);
        if !step.is_finite() {
            return Err(RfError::Code(430, "bias correction: the re-estimation of the fixed effects diverged".into()));
        }
        std::mem::swap(&mut eta, &mut next);
        dev = dev_next;
        if !halved && step < RESOLVE_TOL {
            // FE coefficients of the solution: the FE part eta - offset - X coef lies in the FE space.
            if n <= CHUNK || pool.threads() <= 1 {
                for k in 0..n {
                    block[k] = eta[k] - off[k];
                }
            } else {
                pool.install(|| {
                    block.par_chunks_mut(CHUNK).zip(eta.par_chunks(CHUNK)).zip(off.par_chunks(CHUNK)).for_each(|((bc, ec), oc)| {
                        for ((bo, &ek), &ok) in bc.iter_mut().zip(ec).zip(oc) {
                            *bo = ek - ok;
                        }
                    })
                });
            }
            let info = abs.residualize(&mut block, 1, &SolveOpts { want_coefs: true, ..opts }, None, pool, ctx)?;
            let fe_coefs = info.coefs.ok_or_else(|| RfError::Internal("absorber returned no FE coefficients".into()))?;
            let out = crate::fe_recovery::to_kept_order(&eta, &order, pool)?;
            return Ok(Resolved { eta: out, fe_coefs, iterations: it });
        }
    }
    Err(RfError::Code(
        430,
        format!("bias correction: the fixed effects at the corrected coefficients did not converge in {RESOLVE_MAXITER} Newton iterations"),
    ))
}
