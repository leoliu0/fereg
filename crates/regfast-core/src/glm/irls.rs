//! IRLS loop shared by Poisson and logit (ppmlhdfe 2.3.3 `GLM::inner_irls`, ppmlhdfe.mata
//! l.559-853): inner-tolerance schedule, eps-history prediction, stopping rule, the `mu`
//! separation rule, warm starts through [`FeCoefs`] and (logit only) step halving.
//!
//! Deviations from the source, all confined to solver bookkeeping:
//! - ppmlhdfe accumulates the partialled-out working variable across iterations
//!   (`data[.,1] + z − z_last`); here a fresh copy of [z, X] is residualized every iteration with
//!   the previous FE coefficients as warm start. Both are the same projection.
//! - ppmlhdfe compares the realized accuracy of its MAP solver (`HDFE.accuracy`) with the target
//!   inner tolerance; the absorber does not report one, so the requested inner tolerance (an upper
//!   bound of the accuracy) is used instead. Convergence can only be declared later, never earlier.

use rayon::prelude::*;

use super::GlmData;
use super::family::{self, EPS1, MU_MIN};
use super::wls;
use crate::absorber::{Absorber, FeCoefs, SolveOpts};
use crate::api::{Ctx, Family};
use crate::error::RfError;
use crate::par::{self, CHUNK, Pool};
use crate::stats;

/// ppmlhdfe `start_inner_tol`.
pub(crate) const START_INNER_TOL: f64 = 1e-4;
/// ppmlhdfe `mu_tol`.
const MU_TOL: f64 = 1e-6;
/// Logit step halving: at most this many halvings per iteration.
const MAX_HALVINGS: usize = 25;

pub(crate) struct IrlsCfg {
    /// IRLS deviance tolerance.
    pub tol: f64,
    pub maxiter: usize,
    /// ppmlhdfe `highest_inner_tol` = max(1e-12, min(target_inner_tol, 0.1·tol)).
    pub highest: f64,
    /// FE solver iteration cap.
    pub inner_maxiter: usize,
    /// The absorber solves exactly (at most one absorbed term; reghdfe `G == 1`).
    pub exact: bool,
    pub log: bool,
    /// Scale of y (deviance is reported as deviance·sd_y).
    pub sd_y: f64,
    /// Run the `mu` separation rule.
    pub check_sep: bool,
}

pub(crate) struct IrlsOut {
    pub iter: usize,
    pub subiter: usize,
    /// Deviance on the standardized scale.
    pub deviance: f64,
    /// Residualized [z, X] block of the final iteration (row-major, R = 1 + k).
    pub block: Vec<f64>,
    /// IRLS weights of the final iteration (true weight × μ or μ(1 − μ)).
    pub w: Vec<f64>,
    pub beta: Vec<f64>,
    /// Working residuals z̃ − X̃β of the final iteration.
    pub resid: Vec<f64>,
    /// FE coefficients of the final residualization (columns [z, X]).
    pub coefs: Option<FeCoefs>,
    /// Observations flagged by the `mu` rule (row order).
    pub separated: Vec<bool>,
    pub n_sep: usize,
}

/// Mata `x :<= lb ... ub` with missing (NaN) bounds treated as +∞ (ftools `inrange`).
fn inrange(x: f64, lb: f64, ub: f64) -> bool {
    !lb.is_nan() && lb <= x && (ub.is_nan() || x <= ub)
}

/// ppmlhdfe `predict_eps`: stores ln(eps) in the history (size 3) and, once it is full,
/// extrapolates the next value with the cubic through (i, ln eps_i), i = 1..3 (no constant).
fn predict_eps(hist: &mut [f64; 3], eps: f64) -> f64 {
    let size = hist.len();
    let pos = hist.iter().filter(|v| !v.is_nan()).count();
    let abort = pos + 1 < size;
    let le = if eps > 0.0 { eps.ln() } else { f64::NAN };
    if pos < size {
        hist[pos] = le;
    } else {
        hist.rotate_left(1);
        hist[size - 1] = le;
    }
    if abort || hist.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    // Solve [i, i², i³]·c = h for i = 1..3, then forecast at i = 4.
    let mut a = [[1.0, 1.0, 1.0, hist[0]], [2.0, 4.0, 8.0, hist[1]], [3.0, 9.0, 27.0, hist[2]]];
    for p in 0..3 {
        for q in p + 1..3 {
            let f = a[q][p] / a[p][p];
            for j in p..4 {
                a[q][j] -= f * a[p][j];
            }
        }
    }
    let mut c = [0.0; 3];
    for p in (0..3).rev() {
        let mut s = a[p][3];
        for j in p + 1..3 {
            s -= a[p][j] * c[j];
        }
        c[p] = s / a[p][p];
    }
    (4.0 * c[0] + 16.0 * c[1] + 64.0 * c[2]).exp()
}

/// Mata `_edittozerotol(A, tol)`: |a| ≤ tol → 0.
pub(crate) fn edit_to_zero(v: &mut [f64], tol: f64, pool: &Pool) {
    pool.install(|| {
        v.par_chunks_mut(CHUNK).for_each(|c| {
            for x in c {
                if x.abs() <= tol {
                    *x = 0.0;
                }
            }
        })
    });
}

/// Stata-like `%.{prec}e` (two-digit exponent); missing prints as `.`.
pub(crate) fn fmt_e(x: f64, prec: usize) -> String {
    if !x.is_finite() {
        return ".".into();
    }
    let s = format!("{x:.prec$e}");
    let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
    let e: i32 = e.parse().unwrap_or(0);
    format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
}

/// ppmlhdfe `censor_mu`: μ ← max(μ, 0.05·y, 1e-3).
fn censor_mu(mu: &mut [f64], y: &[f64]) {
    for (m, &yi) in mu.iter_mut().zip(y) {
        let c = (0.05 * yi).max(1e-3);
        if *m < c {
            *m = c;
        }
    }
}

/// Deviance on the standardized scale with ppmlhdfe's clean-up of values within macheps of 0.
pub(crate) fn deviance(fam: Family, d: &GlmData, mu: &[f64], eta: &[f64], pool: &Pool) -> f64 {
    let n = d.n;
    let tw = d.tw.as_deref();
    let raw = par::sum_chunks(pool, n, CHUNK, 1, |range, acc| {
        for i in range {
            let w = tw.map_or(1.0, |t| t[i]);
            let y = d.y[i];
            acc[0] += match fam {
                Family::Logit => -w * family::logit_ll(y, eta[i]),
                _ => w * (mu[i] - y) + if y > 0.0 { w * y * (y.ln() - eta[i]) } else { 0.0 },
            };
        }
    })[0];
    let mut dev = raw;
    if 2.0 * dev / (n as f64) < EPS1 {
        dev = 0.0;
    }
    if dev.abs() <= EPS1 {
        dev = 0.0;
    }
    dev *= 2.0;
    if dev < 0.0 { 0.0 } else { dev }
}

/// μ from η with the separated observations pinned to the boundary.
fn update_mu(fam: Family, d: &GlmData, eta: &[f64], sep: &[bool], mu: &mut [f64], pool: &Pool) {
    let y = &d.y;
    pool.install(|| {
        mu.par_iter_mut().enumerate().for_each(|(i, m)| {
            *m = if sep[i] {
                if fam == Family::Logit && y[i] > 0.0 { 1.0 - MU_MIN } else { MU_MIN }
            } else {
                family::mean(fam, eta[i])
            };
        })
    });
}

/// Runs IRLS from `mu` (updated in place, as are `eta`) until convergence.
pub(crate) fn irls(
    abs: &mut dyn Absorber,
    d: &GlmData,
    mu: &mut [f64],
    eta: &mut [f64],
    c: &IrlsCfg,
    pool: &Pool,
    ctx: &Ctx,
) -> Result<IrlsOut, RfError> {
    let fam = d.family;
    let n = d.n;
    let k = d.k;
    let r = 1 + k;
    if fam == Family::Poisson {
        censor_mu(mu, &d.y);
    }
    for (e, &m) in eta.iter_mut().zip(mu.iter()) {
        *e = family::link(fam, m);
    }
    let log_septol = MU_TOL.ln();
    let xcols: Vec<usize> = (1..r).collect();
    let mut tol_in = START_INNER_TOL.max(c.tol);
    let mut alt_tol = f64::NAN;
    let mut hist = [f64::NAN; 3];
    let mut eps = f64::NAN;
    let mut dev = f64::NAN;
    let mut ok = 0usize;
    let mut sep = vec![false; n];
    let mut n_sep = 0usize;
    let mut subiter = 0usize;
    let mut warm: Option<FeCoefs> = None;
    let mut old_eta = vec![0.0; n];
    let mut z = par::alloc_vec(n, 0.0)?;
    let mut w = par::alloc_vec(n, 0.0)?;
    let mut iter = 0usize;
    loop {
        iter += 1;
        if iter > c.maxiter {
            return Err(RfError::Code(430, format!("Failed to converge in {} iterations (eps={})", c.maxiter, fmt_e(eps, 6))));
        }
        ctx.check()?;
        let predicted = predict_eps(&mut hist, eps);
        // ppmlhdfe `iter_fast_solver` (normal equations while far from convergence).
        let fast = k > 0 && tol_in > c.tol * 11.0 && !(predicted <= c.tol);

        // (a) weights, (b) working variable.
        let tw = d.tw.as_deref();
        let off = d.off.as_deref();
        pool.install(|| {
            w.par_iter_mut().zip(z.par_iter_mut()).enumerate().for_each(|(i, (wi, zi))| {
                *wi = tw.map_or(1.0, |t| t[i]) * family::irls_weight(fam, mu[i]);
                let e = eta[i] - off.map_or(0.0, |o| o[i]);
                *zi = family::working(fam, e, d.y[i], mu[i]);
            })
        });
        abs.set_weights(Some(w.clone()), pool, ctx)?;

        // (c)-(d) residualize [z, X].
        let mut block = par::alloc_vec(n * r, 0.0)?;
        pool.install(|| {
            block.par_chunks_mut(r).enumerate().for_each(|(i, row)| {
                row[0] = z[i];
                row[1..].copy_from_slice(&d.x[i * k..(i + 1) * k]);
            })
        });
        let ztol = c.tol.min(1e-12);
        edit_to_zero(&mut block, ztol, pool);
        let opts = SolveOpts { tol: tol_in, maxiter: c.inner_maxiter, want_coefs: true };
        let info = abs.residualize(&mut block, r, &opts, warm.as_ref(), pool, ctx)?;
        edit_to_zero(&mut block, ztol, pool);
        subiter += info.iterations;
        warm = info.coefs;

        let beta = if k == 0 {
            Vec::new()
        } else if fast {
            wls::normal_coefs(&stats::gram(&block, r, Some(&w), pool).0, r)
        } else {
            wls::qr_coefs(&block, r, 0, &xcols, Some(&w), pool)
        };
        let mut resid = par::alloc_vec(n, 0.0)?;
        pool.install(|| {
            resid.par_iter_mut().enumerate().for_each(|(i, e)| {
                let row = &block[i * r..(i + 1) * r];
                *e = row[0] - beta.iter().zip(&row[1..]).map(|(b, x)| b * x).sum::<f64>();
            })
        });

        // (e) η = z − resid (+ offset).
        old_eta.copy_from_slice(eta);
        pool.install(|| {
            eta.par_iter_mut().enumerate().for_each(|(i, e)| *e = z[i] - resid[i] + off.map_or(0.0, |o| o[i]))
        });
        if c.check_sep {
            // ppmlhdfe: tolerance in terms of μ, made more conservative when η is low for y > 0.
            let y = &d.y;
            let min_pos = (0..n).filter(|&i| y[i] > 0.0).map(|i| eta[i]).fold(f64::INFINITY, f64::min);
            let adj0 = log_septol + (min_pos + 5.0).min(0.0);
            let max_zero = (0..n).filter(|&i| y[i] == 0.0).map(|i| eta[i]).fold(f64::NEG_INFINITY, f64::max);
            // Logit: the mirror rule for y = 1 (μ within mu_tol of 1).
            let adj1 = log_septol + (-max_zero + 5.0).min(0.0);
            for i in 0..n {
                if y[i] == 0.0 && eta[i] <= adj0 {
                    sep[i] = true;
                } else if fam == Family::Logit && y[i] > 0.0 && -eta[i] <= adj1 {
                    sep[i] = true;
                }
            }
            n_sep = sep.iter().filter(|s| **s).count();
        }
        // (f) μ, deviance.
        update_mu(fam, d, eta, &sep, mu, pool);
        let old_dev = dev;
        dev = deviance(fam, d, mu, eta, pool);
        let mut halved = false;
        if fam == Family::Logit && iter > 1 {
            let mut h = 0;
            while h < MAX_HALVINGS && !(dev <= old_dev + 1e-9 * old_dev.abs().max(1.0)) {
                pool.install(|| eta.par_iter_mut().zip(old_eta.par_iter()).for_each(|(e, o)| *e = 0.5 * (*e + o)));
                update_mu(fam, d, eta, &sep, mu, pool);
                dev = deviance(fam, d, mu, eta, pool);
                h += 1;
                halved = true;
            }
        }
        let mut delta = old_dev - dev;
        if !delta.is_nan() && dev < 0.1 * delta {
            delta = dev;
        }
        if iter > 1 {
            let denom = dev.min(old_dev).max(0.1);
            eps = delta.abs() / denom;
            if eps < c.tol && !fast && (tol_in <= 1.1 * c.highest || c.exact) {
                ok += 1;
            }
        }

        if c.log {
            let min_eta = (0..n).filter(|&i| !sep[i]).map(|i| eta[i]).fold(f64::INFINITY, f64::min);
            let flags = format!(
                "{}{}{}{}",
                if iter > 1 { ' ' } else { 'P' },
                if fast { ' ' } else { 'S' },
                if halved { 'H' } else { ' ' },
                if ok > 0 { 'O' } else { ' ' }
            );
            let mut line = format!(
                "{{txt}}Iteration {iter}:{{col 16}}deviance = {{res}}{:<11}{{col 39}}{{txt}}eps = {{res}}{:<9}{{txt}}{{col 55}}iters = {{res}}{}{{col 68}}{{txt}}tol ={{res}}{:>8}{{col 82}}{{txt}} min(eta) = {{res}}{:>6.2}  {{txt}}{flags}",
                fmt_e(dev * c.sd_y, 4),
                fmt_e(eps, 2),
                info.iterations,
                fmt_e(tol_in, 1),
                min_eta
            );
            if n_sep > 0 {
                line.push_str(&format!("{{col 108}}{{txt}} sep.obs. = {{res}}{n_sep}"));
            }
            line.push('\n');
            ctx.log(&line);
        }

        if ok >= 1 {
            return Ok(IrlsOut {
                iter,
                subiter,
                deviance: dev,
                block,
                w,
                beta,
                resid,
                coefs: warm,
                separated: sep,
                n_sep,
            });
        }

        // Tighten the inner tolerance as IRLS converges (use_heuristic_tol = 1).
        if eps < tol_in {
            tol_in = (0.1 * tol_in).min(alt_tol).max(c.highest);
            alt_tol = 10f64.powf(-(1.0 / (0.1 * eps).max(EPS1)).log10().ceil());
        }
        if inrange(c.tol, predicted, eps) && inrange(tol_in / c.highest, 1.1, 10.1) {
            tol_in *= 0.1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_eps_follows_ppmlhdfe_history() {
        let mut h = [f64::NAN; 3];
        assert!(predict_eps(&mut h, f64::NAN).is_nan());
        assert!(predict_eps(&mut h, f64::NAN).is_nan());
        assert!(predict_eps(&mut h, 1e-2).is_nan());
        assert!(predict_eps(&mut h, 1e-4).is_nan());
        // ln eps linear in i (slope ln 1e-2): the cubic through the origin is not exact, but
        // the forecast must be finite and below the last value.
        let p = predict_eps(&mut h, 1e-6);
        assert!(p.is_finite() && p < 1e-6, "{p}");
        assert_eq!(fmt_e(2.4064e1, 4), "2.4064e+01");
        assert_eq!(fmt_e(1e-4, 1), "1.0e-04");
        assert!(inrange(1e-8, 1e-9, f64::NAN));
        assert!(!inrange(1e-8, f64::NAN, 1.0));
    }
}
