//! ReLU separation check (ppmlhdfe 2.3.3 `relu_fix_separation`,
//! ppmlhdfe_separation_relu.mata, with `solve_lse` from ppmlhdfe_functions.mata).
//!
//! Iteratively solves the least-squares problem with equality constraints (method of weighting:
//! weight M = ⌈ε^{−1/2}⌉ on y > 0, 1 on y = 0) of the ReLU working variable u on the regressors
//! and the fixed effects, until the prediction is non-negative on the boundary sample.
//! Deviation: ppmlhdfe aborts after 100 iterations and keeps every observation; here the
//! iteration continues while it makes progress (see [`RELU_MAXITER`]), and a check that still
//! stops uncertified is reported ([`ReluOut::complete`]).
//! The `simplex` method is accepted as an alias of this one; the IRLS-step `mu` rule lives in
//! [`super::irls`].

use rayon::prelude::*;

use super::GlmData;
use super::family::EPS1;
use super::wls;
use crate::absorber::{Absorber, FeCoefs, SolveOpts};
use crate::api::Ctx;
use crate::error::RfError;
use crate::par::{self, CHUNK, Pool};

/// ppmlhdfe `relu_tol`.
const RELU_TOL: f64 = 1e-4;
/// ppmlhdfe `relu_zero_tol`.
const RELU_ZERO_TOL: f64 = 1e-8;
/// ppmlhdfe `relu_maxiter`: ppmlhdfe aborts here. The iteration converges linearly (each step
/// shrinks the negative predictions on the boundary sample by a roughly constant factor, ≈0.94
/// on a 36k-row panel that needed 138 iterations), so past this point it continues while that
/// shrinkage continues ([`RELU_STALL`]), up to [`RELU_HARD_MAXITER`].
const RELU_MAXITER: usize = 100;
const RELU_HARD_MAXITER: usize = 2000;
/// Past [`RELU_MAXITER`], stop (incomplete) when the sum of the negative boundary predictions
/// did not fall below this share of its value [`RELU_STALL_SPAN`] iterations earlier.
const RELU_STALL: f64 = 0.99;
const RELU_STALL_SPAN: usize = 10;

/// The sum of the negative boundary predictions (history `hist`, one entry per iteration) has
/// not fallen below [`RELU_STALL`] times its value [`RELU_STALL_SPAN`] iterations earlier.
fn stalled(hist: &[f64]) -> bool {
    let n = hist.len();
    n > RELU_STALL_SPAN && hist[n - 1] > RELU_STALL * hist[n - 1 - RELU_STALL_SPAN]
}

/// Outcome of [`relu`].
pub(crate) struct ReluOut {
    /// Separated observations (row order), when any were found.
    pub separated: Option<Vec<bool>>,
    /// The check certified its result: false when it stopped at the iteration cap or for lack
    /// of progress (separated observations may remain in the sample).
    pub complete: bool,
    /// Certificate of separation (row order): the final prediction x'b + d on the boundary
    /// sample, positive exactly on the separated observations, 0 elsewhere (ppmlhdfe `zvarname`).
    pub certificate: Vec<f64>,
    pub iterations: usize,
}

/// Runs the ReLU check on the current sample. The absorber weights are left at the true weights.
pub(crate) fn relu(abs: &mut dyn Absorber, d: &GlmData, inner_maxiter: usize, pool: &Pool, ctx: &Ctx) -> Result<ReluOut, RfError> {
    let n = d.n;
    let k = d.k;
    let r = 1 + k;
    let boundary: Vec<bool> = d.y.iter().map(|&y| y == 0.0).collect();
    let num_boundary = boundary.iter().filter(|b| **b).count();
    if num_boundary == 0 {
        return Ok(ReluOut { separated: None, complete: true, certificate: vec![0.0; n], iterations: 0 });
    }
    let regtol = (RELU_TOL * RELU_TOL).max(1e-13);
    // Van Loan's rule of thumb (solve_lse: M = ceil(epsilon(1)^(-1/2))).
    let big_m = EPS1.powf(-0.5).ceil();
    let w: Vec<f64> = boundary.iter().map(|&b| if b { 1.0 } else { big_m }).collect();
    abs.set_weights(Some(w.clone()), pool, ctx)?;

    // Mata norm(X, 1) on each sample: the largest column sum of |x|.
    let col_abs = |want_boundary: bool| -> f64 {
        let s = par::sum_chunks(pool, n, CHUNK, k, |range, acc| {
            for i in range {
                if boundary[i] == want_boundary {
                    for (a, x) in acc.iter_mut().zip(&d.x[i * k..(i + 1) * k]) {
                        *a += x.abs();
                    }
                }
            }
        });
        s.into_iter().fold(0.0, f64::max)
    };
    let ratio = col_abs(false) / col_abs(true);
    // Mata max() ignores missing values (0/0 and x/0 are missing in Mata).
    let ratio = if ratio.is_finite() { ratio.max(1.0) } else { 1.0 };

    let xcols: Vec<usize> = (1..r).collect();
    let mut u: Vec<f64> = boundary.iter().map(|&b| f64::from(u8::from(b))).collect();
    let mut warm: Option<FeCoefs> = None;
    let mut xbd = vec![0.0; n];
    let mut total_iter = 0usize;
    let mut num_sep = 0usize;
    let mut found = false;
    let mut iter = 0usize;
    // Sum of the negative boundary predictions per iteration (progress past RELU_MAXITER).
    let mut negsum_hist: Vec<f64> = Vec::new();
    loop {
        iter += 1;
        ctx.check()?;
        let mut block = par::alloc_vec(n * r, 0.0)?;
        pool.install(|| {
            block.par_chunks_mut(r).enumerate().for_each(|(i, row)| {
                row[0] = u[i];
                row[1..].copy_from_slice(&d.x[i * k..(i + 1) * k]);
            })
        });
        let opts = SolveOpts { tol: regtol, maxiter: inner_maxiter, want_coefs: true };
        let info = abs.residualize(&mut block, r, &opts, warm.as_ref(), pool, ctx)?;
        total_iter += info.iterations;
        warm = info.coefs;
        let b = wls::qr_coefs(&block, r, 0, &xcols, Some(&w), pool);
        let mut resid: Vec<f64> = (0..n)
            .map(|i| {
                let row = &block[i * r..(i + 1) * r];
                row[0] - b.iter().zip(&row[1..]).map(|(b, x)| b * x).sum::<f64>()
            })
            .collect();

        // solve_lse's implied tolerance (maxiter = 0 refinements).
        let uu: f64 = par::sum_chunks(pool, n, CHUNK, 1, |range, acc| acc[0] += u[range].iter().map(|v| v * v).sum::<f64>())[0];
        let epsilon = (uu / big_m * ratio * ratio).max(1e-8);
        let delta = epsilon + RELU_TOL;
        for i in 0..n {
            xbd[i] = u[i] - resid[i];
            if !boundary[i] || (-0.1 * delta <= xbd[i] && xbd[i] <= delta) {
                xbd[i] = 0.0;
            }
        }
        // Separation found when the prediction is non-negative on the boundary sample.
        if (0..n).all(|i| !boundary[i] || xbd[i] >= 0.0) {
            num_sep = (0..n).filter(|&i| boundary[i] && xbd[i] > 0.0).count();
            found = true;
            break;
        }
        // ... or when no boundary residual is negative (normal equations: z'e = 0 with z ≥ 0).
        pool.install(|| {
            resid.par_iter_mut().for_each(|e| {
                if e.abs() <= RELU_ZERO_TOL {
                    *e = 0.0;
                }
            })
        });
        let min_resid = (0..n).filter(|&i| boundary[i]).map(|i| resid[i]).fold(f64::INFINITY, f64::min);
        if min_resid >= 0.0 {
            for i in 0..n {
                if boundary[i] && resid[i] > delta {
                    xbd[i] = 0.0;
                }
            }
            num_sep = (0..n).filter(|&i| boundary[i] && xbd[i] > 0.0).count();
            ctx.log(&format!(
                "{{txt}} $$ Stopping (no negative residuals); separation found in {{res}}{num_sep}{{txt}} observations ({iter} iterations and {total_iter} subiterations)"
            ));
            found = true;
            break;
        }
        let negsum: f64 = (0..n).filter(|&i| boundary[i] && xbd[i] < 0.0).map(|i| -xbd[i]).sum();
        negsum_hist.push(negsum);
        if iter >= RELU_MAXITER && (stalled(&negsum_hist) || iter >= RELU_HARD_MAXITER) {
            ctx.log(&format!("{{err}}(ReLU separation check: no convergence in {iter} iterations; separated observations may remain in the sample)"));
            break;
        }
        // ReLU update of the working variable on the boundary sample.
        for i in 0..n {
            if boundary[i] {
                u[i] = xbd[i].max(0.0);
            }
        }
    }
    abs.set_weights(d.tw.clone(), pool, ctx)?;
    if !found || num_sep == 0 {
        return Ok(ReluOut { separated: None, complete: found, certificate: vec![0.0; n], iterations: iter });
    }
    ctx.log(&format!("{{txt}}(ReLU method dropped {num_sep} separated observation{} in {iter} iterations)", if num_sep > 1 { "s" } else { "" }));
    let separated = (0..n).map(|i| boundary[i] && xbd[i] > 0.0).collect();
    let certificate = (0..n).map(|i| if boundary[i] && xbd[i] > 0.0 { xbd[i] } else { 0.0 }).collect();
    Ok(ReluOut { separated: Some(separated), complete: true, certificate, iterations: iter })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stall_rule_follows_linear_convergence() {
        // Contraction 0.94 per iteration (the field panel): progressing.
        let h: Vec<f64> = (0..40).map(|i| 0.94f64.powi(i)).collect();
        assert!(!stalled(&h));
        // Too short a history to judge.
        assert!(!stalled(&h[..RELU_STALL_SPAN]));
        // Contraction 0.9995 per iteration: 0.995 over the span, stalled.
        let h: Vec<f64> = (0..40).map(|i| 0.9995f64.powi(i)).collect();
        assert!(stalled(&h));
        // Flat: stalled.
        assert!(stalled(&[1.0; 20]));
    }
}
