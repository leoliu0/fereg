//! Average partial effects (design sec. 5) and their variance (sec. 6).

use super::analytical::{bias_sum, plan};
use super::{BiasSpec, FeCodes, LogitData, Sampling, Structure, center, check_data, check_structure, chol_solve, default_key, derivs, invalid, lag_term, ordered_groups, plogis, wcross};
use crate::absorber::Absorber;
use crate::api::Ctx;
use crate::error::RfError;
use crate::par::{self, Pool};

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
    /// Weakly exogenous regressors: add adj * (C + C') (design sec. 6.2).
    pub weak_exo: bool,
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

/// [`apes`] with alpaca's row set for the variance sums (the retained rows only); tests check
/// the shared machinery against alpaca with it.
#[cfg(test)]
pub(crate) fn apes_retained(d: &LogitData, abs: &mut dyn Absorber, coef: &[f64], eta: &[f64], spec: &ApeSpec, ctx: &Ctx) -> Result<Apes, RfError> {
    apes_impl(d, abs, coef, eta, spec, true, ctx)
}

fn apes_impl(d: &LogitData, abs: &mut dyn Absorber, coef: &[f64], eta: &[f64], spec: &ApeSpec, retained: bool, ctx: &Ctx) -> Result<Apes, RfError> {
    let n = check_data(d, abs, coef, eta)?;
    let k = d.fe.len();
    check_structure(spec.structure, k)?;
    if !(spec.adj >= 0.0 && spec.adj.is_finite()) {
        return invalid("APEs: the finite-population factor must be finite and non-negative");
    }
    let plan = spec.correction.as_ref().map(|c| plan(d, c)).transpose()?;
    let sampling_term = spec.adj > 0.0 && spec.want_vcov && (spec.sampling == Sampling::Independence || spec.weak_exo);
    if sampling_term && !retained {
        let Some(f) = &spec.full else { return invalid("APEs: the sampling-of-effects term needs the rows before the fixed-effect drops") };
        check_full(f, d, n)?;
    }
    par::with_pool(par::threads_for(ctx.threads, n), |pool| apes_in(d, abs, coef, eta, spec, plan.as_ref(), sampling_term, retained, pool, ctx))?
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
    let binary: Vec<bool> = d.x.iter().map(|x| x.iter().all(|&v| v == 0.0 || v == 1.0)).collect();
    let wt = |r: usize| d.weights.map_or(1.0, |w| w[r]);
    let mut w = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut d1s = vec![0.0; n];
    let mut d2s = vec![0.0; n];
    let mut d3s = vec![0.0; n];
    for r in 0..n {
        let (m, d1, d2, d3) = derivs(eta[r]);
        w[r] = wt(r) * d1;
        v[r] = wt(r) * (d.y[r] - m);
        z[r] = wt(r) * d2;
        d1s[r] = d1;
        d2s[r] = d2;
        d3s[r] = d3;
    }
    // MX, H and J only enter the variance.
    let mx = if spec.want_vcov { center(abs, &d.x, &w, pool, ctx)? } else { vec![vec![0.0; n]; p] };
    ctx.check()?;
    let h = if spec.want_vcov { wcross(&mx, &w) } else { Vec::new() };
    let mut delta_m = vec![vec![0.0; n]; p];
    let mut delta1 = vec![vec![0.0; n]; p];
    let mut delta2 = vec![vec![0.0; n]; p];
    // J[q * p + j]: derivative of APE j with respect to coefficient q.
    let mut jm = vec![0.0; p * p];
    let sum_d1: f64 = d1s.iter().sum();
    for j in 0..p {
        let bj = coef[j];
        if binary[j] {
            let mut s1 = 0.0;
            for r in 0..n {
                let e0 = eta[r] - d.x[j][r] * bj;
                let e1 = e0 + bj;
                let (m0, f0, g0, _) = derivs(e0);
                let (m1, f1, g1, _) = derivs(e1);
                delta_m[j][r] = m1 - m0;
                delta1[j][r] = f1 - f0;
                delta2[j][r] = g1 - g0;
                s1 += f1;
            }
            for q in 0..p {
                if q != j {
                    jm[q * p + j] = mx[q].iter().zip(&delta1[j]).map(|(a, b)| a * b).sum::<f64>() / nf;
                }
            }
            let px: f64 = (0..n).map(|r| (d.x[j][r] - mx[j][r]) * delta1[j][r]).sum();
            jm[j * p + j] = (s1 - px) / nf;
        } else {
            for r in 0..n {
                delta_m[j][r] = bj * d1s[r];
                delta1[j][r] = bj * d2s[r];
                delta2[j][r] = bj * d3s[r];
            }
            for q in 0..p {
                jm[q * p + j] = mx[q].iter().zip(&delta1[j]).map(|(a, b)| a * b).sum::<f64>() / nf;
            }
            jm[j * p + j] += sum_d1 / nf;
        }
    }
    let delta: Vec<f64> = delta_m.iter().map(|c| c.iter().sum::<f64>() / nf).collect();
    let need_psi = plan.is_some() || spec.want_vcov;
    let (mpsi, ppsi) = if need_psi {
        let psi: Vec<Vec<f64>> = delta1.iter().map(|c| c.iter().zip(&w).map(|(a, w)| -a / w).collect()).collect();
        let refs: Vec<&[f64]> = psi.iter().map(Vec::as_slice).collect();
        let mpsi = center(abs, &refs, &w, pool, ctx)?;
        let ppsi: Vec<Vec<f64>> = psi.iter().zip(&mpsi).map(|(a, m)| a.iter().zip(m).map(|(a, m)| a - m).collect()).collect();
        (mpsi, ppsi)
    } else {
        (Vec::new(), Vec::new())
    };
    ctx.check()?;
    let bias_term = match (plan, &spec.correction) {
        (Some(pl), Some(c)) => {
            let a: Vec<Vec<f64>> = (0..p).map(|j| (0..n).map(|r| delta2[j][r] + ppsi[j][r] * z[r]).collect()).collect();
            let bl: Vec<Vec<f64>> = mpsi.iter().map(|m| m.iter().zip(&w).map(|(a, w)| a * w).collect()).collect();
            Some(bias_sum(d, c, pl, &a, &bl, &w, &v, -1.0)?)
        }
        _ => None,
    };
    let corrected: Vec<f64> = match &bias_term {
        Some(b) => delta.iter().zip(b).map(|(a, b)| a - b).collect(),
        None => delta.clone(),
    };
    let mut vcov = Vec::new();
    if spec.want_vcov {
        let hn: Vec<f64> = h.iter().map(|x| x / nf).collect();
        let winvj = chol_solve(&hn, p, &jm, p)?;
        // Gamma (n x p, per column).
        let gamma: Vec<Vec<f64>> = (0..p)
            .map(|j| (0..n).map(|r| ((0..p).map(|q| mx[q][r] * winvj[q * p + j]).sum::<f64>() - ppsi[j][r]) * v[r] / nf).collect())
            .collect();
        vcov = cross(&gamma, &gamma);
        if sampling_term {
            let add = sampling(d, spec, &delta_m, &delta, &gamma, retained)?;
            vcov.iter_mut().zip(add).for_each(|(a, b)| *a += spec.adj * b);
        }
    }
    Ok(Apes { delta: corrected, delta_uncorrected: delta, vcov, bias_term, binary })
}

/// A'B for column sets (p x p row-major), serial.
fn cross(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<f64> {
    let p = a.len();
    let mut out = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            out[i * p + j] = a[i].iter().zip(&b[j]).map(|(x, y)| x * y).sum();
        }
    }
    out
}

/// The sampling-of-effects term and the weak-exogeneity covariance, without the factor adj:
/// independence: S(FE1) + sum_{q >= 2} (S(FEq) - D'D); weak exogeneity: + C + C'.
fn sampling(d: &LogitData, spec: &ApeSpec, delta_m: &[Vec<f64>], delta: &[f64], gamma: &[Vec<f64>], retained: bool) -> Result<Vec<f64>, RfError> {
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
    let mut dm = vec![0.0; m * p];
    for (r, &kr) in kept.iter().enumerate() {
        for j in 0..p {
            let dv = if kr == u32::MAX { 0.0 } else { delta_m[j][kr as usize] };
            dm[r * p + j] = (dv - delta[j]) / nf;
        }
    }
    let mut out = vec![0.0; p * p];
    if spec.sampling == Sampling::Independence {
        let mut dd = vec![0.0; p * p];
        for r in 0..m {
            outer_add(&mut dd, &dm[r * p..(r + 1) * p], &dm[r * p..(r + 1) * p]);
        }
        for (q, f) in fe.iter().enumerate() {
            let mut s = vec![0.0; f.g * p];
            for (r, &c) in f.codes.iter().enumerate() {
                for j in 0..p {
                    s[c as usize * p + j] += dm[r * p + j];
                }
            }
            for g in 0..f.g {
                let sg = &s[g * p..(g + 1) * p];
                outer_add(&mut out, sg, sg);
            }
            if q > 0 {
                out.iter_mut().zip(&dd).for_each(|(a, b)| *a -= b);
            }
        }
    }
    if spec.weak_exo {
        if let Some(q) = lag_term(spec.structure, fe.len()) {
            let f = &fe[q];
            let (off, ids) = match order {
                Some(o) => ordered_groups(f.codes, f.g, Some(&|r| o[r])),
                None => {
                    let key = default_key(spec.structure, &fe);
                    ordered_groups(f.codes, f.g, key.as_deref())
                }
            };
            let mut c = vec![0.0; p * p];
            let mut prefix = vec![0.0; p];
            let mut gs = vec![0.0; p];
            for g in 0..f.g {
                prefix.iter_mut().for_each(|x| *x = 0.0);
                for &r in &ids[off[g]..off[g + 1]] {
                    let r = r as usize;
                    let kr = kept[r];
                    if kr != u32::MAX {
                        for j in 0..p {
                            gs[j] = gamma[j][kr as usize];
                        }
                        outer_add(&mut c, &prefix, &gs);
                    }
                    for j in 0..p {
                        prefix[j] += dm[r * p + j];
                    }
                }
            }
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

/// APEs of one fit at its point without variance or bias term (jackknife subpanels): sum over the
/// rows of Delta, divided by `n_full` (the fit's own N*). `binary` should be the full sample's
/// classification ([`binary`]); rows of `x` / `eta` are the fit's estimation rows.
pub fn ape_point(x: &[&[f64]], binary: &[bool], coef: &[f64], eta: &[f64], n_full: f64) -> Vec<f64> {
    (0..x.len())
        .map(|j| {
            let bj = coef[j];
            let s: f64 = if binary[j] {
                (0..eta.len())
                    .map(|r| {
                        let e0 = eta[r] - x[j][r] * bj;
                        plogis(e0 + bj) - plogis(e0)
                    })
                    .sum()
            } else {
                eta.iter().map(|&e| derivs(e).1).sum::<f64>() * bj
            };
            s / n_full
        })
        .collect()
}
