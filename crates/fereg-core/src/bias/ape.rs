//! Average partial effects (design sec. 5) and their variance (sec. 6).

use rayon::prelude::*;

use super::analytical::{bias_sum, plan};
use super::{
    BiasSpec, FeCodes, LogitData, Sampling, Structure, alloc_cols, center, check_data, check_structure, chol_solve, default_key, derivs, groups, invalid,
    lag_term, ordered_groups, plogis, wcross,
};
use crate::absorber::Absorber;
use crate::api::Ctx;
use crate::error::RfError;
use crate::par::{self, CHUNK, Pool};

/// The N* rows before the fixed-effect-based drops, for the sampling-of-effects term: every
/// variance sum runs over them, with zero partial effect (and zero score) on the dropped rows.
#[derive(Clone, Debug)]
pub struct FullSample<'a> {
    /// Per pre-drop row (input order): its kept-row index, or `u32::MAX` when it was dropped.
    /// Length N*; every kept row appears exactly once.
    pub kept: &'a [u32],
    /// Codes over the pre-drop rows of every fixed-effect term (same terms and order as
    /// [`LogitData::fe`]; dense in their own space).
    pub fe: Vec<FeCodes<'a>>,
    /// Time key over the pre-drop rows ordering the weak-exogeneity groups (as
    /// [`BiasSpec::order`]); `None`: alpaca's order applied to `fe`.
    pub order: Option<&'a [i64]>,
}

/// What to compute.
#[derive(Clone, Debug)]
pub struct ApeSpec<'a> {
    pub structure: Structure,
    pub sampling: Sampling,
    /// Weakly exogenous regressors: add adj * (C + C') (design sec. 6.2), C within the groups of
    /// the lag term.
    pub weak_exo: bool,
    /// Index into [`LogitData::fe`] (and [`FullSample::fe`]) of the lag-group term (individuals
    /// classic, pairs network), as [`BiasSpec::lag_term`]; `None` = FE1 / FE3.
    pub lag_term: Option<usize>,
    /// Finite-population factor of the sampling-of-effects term (0 = delta method only).
    pub adj: f64,
    /// Analytically corrected APEs: the bias term of this specification, evaluated at the point
    /// (which should then be the corrected coefficients and re-estimated eta).
    pub correction: Option<BiasSpec<'a>>,
    /// Required when `adj > 0` and the sampling term or the weak-exogeneity term applies.
    pub full: Option<FullSample<'a>>,
    /// Compute the variance (else `Apes::vcov` is empty).
    pub want_vcov: bool,
}

/// Result of [`apes`]; vectors in coefficient order.
#[derive(Clone, Debug, PartialEq)]
pub struct Apes {
    /// APEs: delta(beta, eta), minus `bias_term` when corrected.
    pub delta: Vec<f64>,
    /// delta(beta, eta) = sum Delta / N* (before any bias term).
    pub delta_uncorrected: Vec<f64>,
    /// p x p row-major.
    pub vcov: Vec<f64>,
    pub bias_term: Option<Vec<f64>>,
    /// Binary regressors (every value 0 or 1 on the sample): discrete-change effect.
    pub binary: Vec<bool>,
}

/// APEs, their bias term and variance at the point (`coef`, `eta`: kept order, offset included),
/// design secs. 5-6. The absorber is left with the weights at the point.
pub fn apes(d: &LogitData, abs: &mut dyn Absorber, coef: &[f64], eta: &[f64], spec: &ApeSpec, ctx: &Ctx) -> Result<Apes, RfError> {
    apes_impl(d, abs, coef, eta, spec, false, ctx)
}

pub(crate) fn apes_in_pool(
    d: &LogitData,
    abs: &mut dyn Absorber,
    coef: &[f64],
    eta: &[f64],
    spec: &ApeSpec,
    retained: bool,
    pool: &Pool,
    ctx: &Ctx,
) -> Result<Apes, RfError> {
    let n = check_data(d, abs, coef, eta)?;
    let k = d.fe.len();
    check_structure(spec.structure, k)?;
    lag_term(spec.structure, k, spec.lag_term)?;
    if !(spec.adj >= 0.0 && spec.adj.is_finite()) {
        return invalid("APEs: the finite-population factor must be finite and non-negative");
    }
    let plan = spec.correction.as_ref().map(|c| plan(d, c)).transpose()?;
    let sampling_term = spec.adj > 0.0 && spec.want_vcov && (spec.sampling == Sampling::Independence || spec.weak_exo);
    if sampling_term && !retained {
        let Some(f) = &spec.full else { return invalid("APEs: the sampling-of-effects term needs the rows before the fixed-effect drops") };
        check_full(f, d, n)?;
    }
    apes_in(d, abs, coef, eta, spec, plan.as_ref(), sampling_term, retained, pool, ctx)
}

fn apes_impl(d: &LogitData, abs: &mut dyn Absorber, coef: &[f64], eta: &[f64], spec: &ApeSpec, retained: bool, ctx: &Ctx) -> Result<Apes, RfError> {
    let n = check_data(d, abs, coef, eta)?;
    par::with_pool(par::threads_for(ctx.threads, n), |pool| apes_in_pool(d, abs, coef, eta, spec, retained, pool, ctx))?
}

fn check_full(f: &FullSample, d: &LogitData, n: usize) -> Result<(), RfError> {
    let nf = f.kept.len();
    if nf != d.n_full {
        return invalid("APEs: the pre-drop rows must number N*");
    }
    if f.fe.len() != d.fe.len() || f.fe.iter().any(|t| t.codes.len() != nf || t.codes.iter().any(|&c| c as usize >= t.g)) {
        return invalid("APEs: pre-drop fixed-effect codes must be dense, one per pre-drop row and term");
    }
    if f.order.is_some_and(|o| o.len() != nf) {
        return invalid("APEs: the pre-drop time order needs one value per pre-drop row");
    }
    let mut seen = vec![false; n];
    for &kr in f.kept {
        if kr != u32::MAX && (kr as usize >= n || std::mem::replace(&mut seen[kr as usize], true)) {
            return invalid("APEs: every kept row must appear exactly once among the pre-drop rows");
        }
    }
    if seen.iter().any(|s| !s) {
        return invalid("APEs: every kept row must appear exactly once among the pre-drop rows");
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ColPtr(*mut std::mem::MaybeUninit<f64>);
// SAFETY: chunks write disjoint row ranges `range.start..range.end`.
unsafe impl Send for ColPtr {}
unsafe impl Sync for ColPtr {}

impl ColPtr {
    #[inline(always)]
    unsafe fn write(self, idx: usize, val: f64) {
        unsafe { self.0.add(idx).write(std::mem::MaybeUninit::new(val)) };
    }
}

fn binary_in(x: &[&[f64]], pool: &Pool) -> Vec<bool> {
    x.iter()
        .map(|col| {
            if col.len() <= CHUNK || pool.threads() <= 1 {
                col.iter().all(|&v| v == 0.0 || v == 1.0)
            } else {
                pool.install(|| col.par_chunks(CHUNK).all(|c| c.iter().all(|&v| v == 0.0 || v == 1.0)))
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn apes_in(
    d: &LogitData,
    abs: &mut dyn Absorber,
    coef: &[f64],
    eta: &[f64],
    spec: &ApeSpec,
    plan: Option<&super::analytical::Plan>,
    sampling_term: bool,
    retained: bool,
    pool: &Pool,
    ctx: &Ctx,
) -> Result<Apes, RfError> {
    let n = d.y.len();
    let p = d.x.len();
    let nf = d.n_full as f64;
    let binary = binary_in(&d.x, pool);
    let wt = |r: usize| d.weights.map_or(1.0, |w| w[r]);
    let mut w_un = par::alloc_uninit::<f64>(n)?;
    let mut v_un = par::alloc_uninit::<f64>(n)?;
    let mut z_un = par::alloc_uninit::<f64>(n)?;
    let mut dm_un: Vec<Vec<std::mem::MaybeUninit<f64>>> = (0..p).map(|_| par::alloc_uninit::<f64>(n)).collect::<Result<_, _>>()?;
    let mut d1_un: Vec<Vec<std::mem::MaybeUninit<f64>>> = (0..p).map(|_| par::alloc_uninit::<f64>(n)).collect::<Result<_, _>>()?;
    let mut d2_un: Vec<Vec<std::mem::MaybeUninit<f64>>> = (0..p).map(|_| par::alloc_uninit::<f64>(n)).collect::<Result<_, _>>()?;
    let (wp, vp, zp) = (ColPtr(w_un.as_mut_ptr()), ColPtr(v_un.as_mut_ptr()), ColPtr(z_un.as_mut_ptr()));
    let dmp: Vec<ColPtr> = dm_un.iter_mut().map(|c| ColPtr(c.as_mut_ptr())).collect();
    let d1p: Vec<ColPtr> = d1_un.iter_mut().map(|c| ColPtr(c.as_mut_ptr())).collect();
    let d2p: Vec<ColPtr> = d2_un.iter_mut().map(|c| ColPtr(c.as_mut_ptr())).collect();
    // Single row pass: evaluates `derivs(eta[r])` once per row instead of `p + 1` times, and
    // accumulates `[sum_d1, sum_f1[0..p], sum_dm[0..p]]` in fixed chunk order.
    let sums = par::sum_chunks(pool, n, CHUNK, 1 + 2 * p, |range, acc| {
        for r in range {
            let er = eta[r];
            let (m, d1, d2, d3) = derivs(er);
            let wr = wt(r);
            // SAFETY: each chunk owns disjoint row indices `r in range`.
            unsafe {
                wp.write(r, wr * d1);
                vp.write(r, wr * (d.y[r] - m));
                zp.write(r, wr * d2);
            }
            acc[0] += d1;
            for j in 0..p {
                let bj = coef[j];
                let dm_val = if binary[j] {
                    let e0 = er - d.x[j][r] * bj;
                    let e1 = e0 + bj;
                    let (m0, f0, g0, _) = derivs(e0);
                    let (m1, f1, g1, _) = derivs(e1);
                    unsafe {
                        dmp[j].write(r, m1 - m0);
                        d1p[j].write(r, f1 - f0);
                        d2p[j].write(r, g1 - g0);
                    }
                    acc[1 + j] += f1;
                    m1 - m0
                } else {
                    let v0 = bj * d1;
                    unsafe {
                        dmp[j].write(r, v0);
                        d1p[j].write(r, bj * d2);
                        d2p[j].write(r, bj * d3);
                    }
                    v0
                };
                acc[1 + p + j] += dm_val;
            }
        }
    });
    // SAFETY: every row `0..n` was written above.
    let (w, v, z) = unsafe { (par::assume_init(w_un), par::assume_init(v_un), par::assume_init(z_un)) };
    let delta_m: Vec<Vec<f64>> = dm_un.into_iter().map(|c| unsafe { par::assume_init(c) }).collect();
    let delta1: Vec<Vec<f64>> = d1_un.into_iter().map(|c| unsafe { par::assume_init(c) }).collect();
    let mut delta2: Vec<Vec<f64>> = d2_un.into_iter().map(|c| unsafe { par::assume_init(c) }).collect();
    let sum_d1 = sums[0];
    let sum_f1 = &sums[1..1 + p];
    let delta: Vec<f64> = sums[1 + p..1 + 2 * p].iter().map(|&s| s / nf).collect();
    // MX, H and J only enter the variance. J[q * p + j]: derivative of APE j with respect to
    // coefficient q.
    let (mx, h, jm) = if spec.want_vcov {
        let mx = center(abs, &d.x, &w, pool, ctx)?;
        ctx.check()?;
        let h = wcross(&mx, &w, pool);
        let jsums = par::sum_chunks(pool, n, CHUNK, p * p + p, |range, acc| {
            for r in range {
                for j in 0..p {
                    let d1j = delta1[j][r];
                    for q in 0..p {
                        if !(binary[j] && q == j) {
                            acc[q * p + j] += mx[q][r] * d1j;
                        }
                    }
                    if binary[j] {
                        acc[p * p + j] += (d.x[j][r] - mx[j][r]) * d1j;
                    }
                }
            }
        });
        let mut jm: Vec<f64> = jsums[..p * p].iter().map(|&s| s / nf).collect();
        for j in 0..p {
            if binary[j] {
                jm[j * p + j] = (sum_f1[j] - jsums[p * p + j]) / nf;
            } else {
                jm[j * p + j] += sum_d1 / nf;
            }
        }
        (mx, h, jm)
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };
    let need_psi = plan.is_some() || spec.want_vcov;
    // Psi = -Delta1 / w in place over Delta1, M Psi, then P Psi = Psi - M Psi in place over Psi.
    let (mut mpsi, ppsi) = if need_psi {
        let mut psi = delta1;
        for c in &mut psi {
            if n <= CHUNK || pool.threads() <= 1 {
                c.iter_mut().zip(&w).for_each(|(a, w)| *a = -*a / w);
            } else {
                pool.install(|| c.par_chunks_mut(CHUNK).zip(w.par_chunks(CHUNK)).for_each(|(ac, wc)| ac.iter_mut().zip(wc).for_each(|(a, w)| *a = -*a / w)));
            }
        }
        let refs: Vec<&[f64]> = psi.iter().map(Vec::as_slice).collect();
        let mpsi = center(abs, &refs, &w, pool, ctx)?;
        for (a, m) in psi.iter_mut().zip(&mpsi) {
            if n <= CHUNK || pool.threads() <= 1 {
                a.iter_mut().zip(m).for_each(|(a, m)| *a -= m);
            } else {
                pool.install(|| a.par_chunks_mut(CHUNK).zip(m.par_chunks(CHUNK)).for_each(|(ac, mc)| ac.iter_mut().zip(mc).for_each(|(a, m)| *a -= m)));
            }
        }
        (mpsi, psi)
    } else {
        drop(delta1);
        (Vec::new(), Vec::new())
    };
    ctx.check()?;
    let bias_term = match (plan, &spec.correction) {
        (Some(pl), Some(c)) => {
            // A = Delta2 + P Psi z over Delta2; Bl = M Psi w over M Psi.
            for j in 0..p {
                if n <= CHUNK || pool.threads() <= 1 {
                    delta2[j].iter_mut().zip(&ppsi[j]).zip(&z).for_each(|((a, pp), z)| *a += pp * z);
                    mpsi[j].iter_mut().zip(&w).for_each(|(m, w)| *m *= w);
                } else {
                    let (d2j, ppsij, mpsij) = (&mut delta2[j], &ppsi[j], &mut mpsi[j]);
                    pool.install(|| {
                        d2j.par_chunks_mut(CHUNK)
                            .zip(ppsij.par_chunks(CHUNK))
                            .zip(z.par_chunks(CHUNK))
                            .for_each(|((ac, pc), zc)| ac.iter_mut().zip(pc).zip(zc).for_each(|((a, pp), z)| *a += pp * z));
                        mpsij.par_chunks_mut(CHUNK).zip(w.par_chunks(CHUNK)).for_each(|(mc, wc)| mc.iter_mut().zip(wc).for_each(|(m, w)| *m *= w));
                    });
                }
            }
            Some(bias_sum(d, c, pl, &delta2, &mpsi, &w, &v, -1.0, pool)?)
        }
        _ => None,
    };
    drop(delta2);
    drop(mpsi);
    let corrected: Vec<f64> = match &bias_term {
        Some(b) => delta.iter().zip(b).map(|(a, b)| a - b).collect(),
        None => delta.clone(),
    };
    let mut vcov = Vec::new();
    if spec.want_vcov {
        let hn: Vec<f64> = h.iter().map(|x| x / nf).collect();
        let winvj = chol_solve(&hn, p, &jm, p)?;
        // Gamma (n x p, per column).
        let mut gamma = alloc_cols(p, n, pool)?;
        for (j, g) in gamma.iter_mut().enumerate() {
            let ppj = &ppsi[j];
            if n <= CHUNK || pool.threads() <= 1 {
                for (r, gr) in g.iter_mut().enumerate() {
                    *gr = ((0..p).map(|q| mx[q][r] * winvj[q * p + j]).sum::<f64>() - ppj[r]) * v[r] / nf;
                }
            } else {
                pool.install(|| {
                    g.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, gc)| {
                        let base = ci * CHUNK;
                        for (k, gr) in gc.iter_mut().enumerate() {
                            let r = base + k;
                            *gr = ((0..p).map(|q| mx[q][r] * winvj[q * p + j]).sum::<f64>() - ppj[r]) * v[r] / nf;
                        }
                    })
                });
            }
        }
        vcov = cross(&gamma, &gamma, pool);
        if sampling_term {
            let add = sampling(d, spec, &delta_m, &delta, &gamma, retained, pool)?;
            vcov.iter_mut().zip(add).for_each(|(a, b)| *a += spec.adj * b);
        }
    }
    Ok(Apes { delta: corrected, delta_uncorrected: delta, vcov, bias_term, binary })
}

/// A'B for column sets (p x p row-major), reduced in fixed chunk order.
fn cross(a: &[Vec<f64>], b: &[Vec<f64>], pool: &Pool) -> Vec<f64> {
    let p = a.len();
    if p == 0 {
        return Vec::new();
    }
    let n = a[0].len();
    par::sum_chunks(pool, n, CHUNK, p * p, |range, acc| {
        for r in range {
            for i in 0..p {
                let ai = a[i][r];
                for j in 0..p {
                    acc[i * p + j] += ai * b[j][r];
                }
            }
        }
    })
}

/// The sampling-of-effects term and the weak-exogeneity covariance, without the factor adj:
/// independence: S(FE1) + sum_{q >= 2} (S(FEq) - D'D); weak exogeneity: + C + C'.
fn sampling(
    d: &LogitData,
    spec: &ApeSpec,
    delta_m: &[Vec<f64>],
    delta: &[f64],
    gamma: &[Vec<f64>],
    retained: bool,
    pool: &Pool,
) -> Result<Vec<f64>, RfError> {
    let p = delta.len();
    let nf = d.n_full as f64;
    let n = d.y.len();
    let identity: Vec<u32>;
    let (kept, fe, order): (&[u32], Vec<FeCodes>, Option<&[i64]>) = if retained {
        identity = (0..n as u32).collect();
        (&identity, d.fe.clone(), None)
    } else {
        let f = spec.full.as_ref().expect("checked");
        (f.kept, f.fe.clone(), f.order)
    };
    let m = kept.len();
    // D (m x p row-major) over the row set.
    let mut dm_un = par::alloc_uninit::<f64>(m * p)?;
    if m <= CHUNK || pool.threads() <= 1 {
        for (r, (&kr, row)) in kept.iter().zip(dm_un.chunks_exact_mut(p)).enumerate() {
            let _ = r;
            for j in 0..p {
                let dv = if kr == u32::MAX { 0.0 } else { delta_m[j][kr as usize] };
                row[j].write((dv - delta[j]) / nf);
            }
        }
    } else {
        pool.install(|| {
            dm_un.par_chunks_mut(CHUNK * p).zip(kept.par_chunks(CHUNK)).for_each(|(dc, kc)| {
                for (&kr, row) in kc.iter().zip(dc.chunks_exact_mut(p)) {
                    for j in 0..p {
                        let dv = if kr == u32::MAX { 0.0 } else { delta_m[j][kr as usize] };
                        row[j].write((dv - delta[j]) / nf);
                    }
                }
            })
        });
    }
    // SAFETY: every slot of `0..m*p` was written above.
    let dm = unsafe { par::assume_init(dm_un) };
    let mut out = vec![0.0; p * p];
    if spec.sampling == Sampling::Independence {
        let dd = par::sum_chunks(pool, m, CHUNK, p * p, |range, acc| {
            for r in range {
                let row = &dm[r * p..(r + 1) * p];
                outer_add(acc, row, row);
            }
        });
        for (q, f) in fe.iter().enumerate() {
            let add_q = if f.g.saturating_mul(p) <= 4096 {
                let s = par::sum_chunks(pool, m, CHUNK, f.g * p, |range, acc| {
                    for r in range {
                        let dst = &mut acc[f.codes[r] as usize * p..(f.codes[r] as usize + 1) * p];
                        let src = &dm[r * p..(r + 1) * p];
                        for j in 0..p {
                            dst[j] += src[j];
                        }
                    }
                });
                let mut part = vec![0.0; p * p];
                for g in 0..f.g {
                    let sg = &s[g * p..(g + 1) * p];
                    outer_add(&mut part, sg, sg);
                }
                part
            } else {
                let (off, ids) = groups(f.codes, f.g, pool)?;
                par::sum_chunks(pool, f.g, 256, p * p, |range, acc| {
                    let mut sg = vec![0.0; p];
                    for g in range {
                        sg.fill(0.0);
                        for &r in &ids[off[g]..off[g + 1]] {
                            let src = &dm[r as usize * p..(r as usize + 1) * p];
                            for j in 0..p {
                                sg[j] += src[j];
                            }
                        }
                        outer_add(acc, &sg, &sg);
                    }
                })
            };
            out.iter_mut().zip(&add_q).for_each(|(a, b)| *a += b);
            if q > 0 {
                out.iter_mut().zip(&dd).for_each(|(a, b)| *a -= b);
            }
        }
    }
    if spec.weak_exo {
        if let Some(q) = lag_term(spec.structure, fe.len(), spec.lag_term)? {
            let f = &fe[q];
            let (off, ids) = match order {
                Some(o) => ordered_groups(f.codes, f.g, Some(&|r| o[r]), pool)?,
                None => {
                    let key = default_key(&fe, q);
                    ordered_groups(f.codes, f.g, key.as_deref(), pool)?
                }
            };
            let c = par::sum_chunks(pool, f.g, 256, p * p, |range, acc| {
                let mut prefix = vec![0.0; p];
                let mut gs = vec![0.0; p];
                for g in range {
                    prefix.fill(0.0);
                    for &r in &ids[off[g]..off[g + 1]] {
                        let r = r as usize;
                        let kr = kept[r];
                        if kr != u32::MAX {
                            for j in 0..p {
                                gs[j] = gamma[j][kr as usize];
                            }
                            outer_add(acc, &prefix, &gs);
                        }
                        let src = &dm[r * p..(r + 1) * p];
                        for j in 0..p {
                            prefix[j] += src[j];
                        }
                    }
                }
            });
            for i in 0..p {
                for j in 0..p {
                    out[i * p + j] += c[i * p + j] + c[j * p + i];
                }
            }
        }
    }
    Ok(out)
}

fn outer_add(m: &mut [f64], a: &[f64], b: &[f64]) {
    let p = a.len();
    for i in 0..p {
        for j in 0..p {
            m[i * p + j] += a[i] * b[j];
        }
    }
}

/// Binary regressors (every value 0 or 1): discrete-change APE.
pub fn binary(x: &[&[f64]]) -> Vec<bool> {
    x.iter().map(|x| x.iter().all(|&v| v == 0.0 || v == 1.0)).collect()
}

/// [`ape_point`] running in parallel chunks on `pool` (evaluating `derivs(eta[r]).1` once per row).
pub(crate) fn ape_point_in(x: &[&[f64]], binary: &[bool], coef: &[f64], eta: &[f64], n_full: f64, pool: &Pool) -> Vec<f64> {
    let p = x.len();
    if p == 0 {
        return Vec::new();
    }
    let any_cont = binary.iter().any(|&b| !b);
    let sums = par::sum_chunks(pool, eta.len(), CHUNK, p, |range, acc| {
        for r in range {
            let er = eta[r];
            let d1 = if any_cont { derivs(er).1 } else { 0.0 };
            for j in 0..p {
                let bj = coef[j];
                acc[j] += if binary[j] {
                    let e0 = er - x[j][r] * bj;
                    plogis(e0 + bj) - plogis(e0)
                } else {
                    d1 * bj
                };
            }
        }
    });
    sums.into_iter().map(|s| s / n_full).collect()
}

/// APEs of one fit at its point without variance or bias term (jackknife subpanels): sum over the
/// rows of Delta, divided by `n_full` (the fit's own N*). `binary` should be the full sample's
/// classification ([`binary`]); rows of `x` / `eta` are the fit's estimation rows.
pub fn ape_point(x: &[&[f64]], binary: &[bool], coef: &[f64], eta: &[f64], n_full: f64) -> Vec<f64> {
    let p = x.len();
    if p == 0 {
        return Vec::new();
    }
    let any_cont = binary.iter().any(|&b| !b);
    let mut tot = vec![0.0; p];
    let mut acc = vec![0.0; p];
    for c in 0..par::n_chunks(eta.len(), CHUNK) {
        acc.fill(0.0);
        for r in par::chunk_range(c, eta.len(), CHUNK) {
            let er = eta[r];
            let d1 = if any_cont { derivs(er).1 } else { 0.0 };
            for j in 0..p {
                let bj = coef[j];
                acc[j] += if binary[j] {
                    let e0 = er - x[j][r] * bj;
                    plogis(e0 + bj) - plogis(e0)
                } else {
                    d1 * bj
                };
            }
        }
        for j in 0..p {
            tot[j] += acc[j];
        }
    }
    tot.into_iter().map(|s| s / n_full).collect()
}
