//! Lockstep preconditioned conjugate gradient over R columns (plan D5).
//!
//! Each column keeps its own scalars. Column j stops when the Hestenes–Stiefel window estimate
//! Σ_{i=k−d}^{k−1} γ_iρ_i (a lower bound on ‖x* − x_{k−d}‖²_S = W-norm error of the residualized
//! column) falls below max(tol²·SS_k, (1e-13)²·SS₀). With an exact preconditioner (M = S on the
//! whole reduced space) ρ_k = ‖x* − x_k‖²_S exactly and is used directly. Independently, a
//! column stops once ρ_k ≤ (1e-13)²·SS₀: the recursive residual has reached roundoff, and further
//! steps break down (γ explodes) before a tight window could be satisfied. A single escalation
//! probe at iteration 12 may strengthen the preconditioner, after which PCG restarts from the
//! current x. Converged columns are frozen (their direction is zeroed so the shared matvec
//! leaves them untouched). Dot products use the fixed-chunk reduction (thread-count invariant).

use rayon::prelude::*;

use crate::api::Ctx;
use crate::error::RfError;
use crate::par::{CHUNK, Pool, sum_chunks};

/// Window length d of the stopping rule.
pub const WINDOW: usize = 5;
/// Iteration of the escalation probe, and the reference iteration of the contraction estimate
/// ρ̂ = (ρ_12/ρ_7)^{1/5}. (The plan's ρ_2 reference overstates the asymptotic rate: the first
/// iterations remove the high-frequency error, e.g. ρ drops 800× from k = 2 to 12 on a band
/// graph that then needs 270 Jacobi iterations.)
pub const PROBE_AT: usize = 12;
pub const PROBE_FROM: usize = 7;
/// Escalate when the per-iteration contraction ρ̂ of some active column exceeds this.
pub const PROBE_RATIO: f64 = 0.6;

/// The reduced system as seen by PCG.
pub trait CgSystem {
    /// q = S p (m × w row-major).
    fn matvec(&mut self, p: &[f64], q: &mut [f64], pool: &Pool) -> Result<(), RfError>;
    /// z = M⁻¹ r.
    fn precond(&mut self, r: &[f64], z: &mut [f64], pool: &Pool);
    /// M = S exactly (the preconditioner residual norm is the error).
    fn exact(&self) -> bool;
    /// Whether a stronger preconditioner is available.
    fn can_escalate(&self) -> bool;
    /// Builds the stronger preconditioner; `Ok(true)` when it changed.
    fn escalate(&mut self, pool: &Pool, ctx: &Ctx) -> Result<bool, RfError>;
}

pub struct CgOutcome {
    pub iterations: usize,
    pub escalated: bool,
}

/// Per-column dot products of two m × w row-major blocks.
pub fn dots(a: &[f64], b: &[f64], w: usize, pool: &Pool) -> Vec<f64> {
    let m = a.len() / w.max(1);
    sum_chunks(pool, m, CHUNK, w, |range, acc| {
        for i in range {
            let (ra, rb) = (&a[i * w..(i + 1) * w], &b[i * w..(i + 1) * w]);
            for j in 0..w {
                acc[j] += ra[j] * rb[j];
            }
        }
    })
}

/// Solves S x = rhs for the w columns of `rhs` (m × w); `x` holds the start (warm or zero).
/// `ss1`: per-column sum of squares after removing FE1 only; `ss0`: centered sum of squares
/// (0 = constant column, stops at once).
#[allow(clippy::too_many_arguments)]
pub fn pcg(
    sys: &mut dyn CgSystem,
    rhs: &[f64],
    x: &mut [f64],
    w: usize,
    ss1: &[f64],
    ss0: &[f64],
    tol: f64,
    maxiter: usize,
    pool: &Pool,
    ctx: &Ctx,
) -> Result<CgOutcome, RfError> {
    let n = rhs.len();
    let mut r = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut q = vec![0.0; n];
    let mut active: Vec<bool> = ss0.iter().map(|&s| s > 0.0).collect();
    let mut iterations = 0usize;
    let mut escalated = false;
    let floor: Vec<f64> = ss0.iter().map(|s| 1e-26 * s).collect();

    'restart: loop {
        // r = rhs − S x, z = M⁻¹ r, p = z, ρ = r'z.
        let x_nonzero = x.iter().any(|v| *v != 0.0);
        if x_nonzero {
            sys.matvec(x, &mut q, pool)?;
            par_zip3(&mut r, rhs, &q, pool, |r, a, b| *r = a - b);
        } else {
            r.copy_from_slice(rhs);
        }
        ctx.check()?;
        sys.precond(&r, &mut z, pool);
        p.copy_from_slice(&z);
        let mut rho = dots(&r, &z, w, pool);
        for j in 0..w {
            if !(rho[j] > 0.0) {
                active[j] = false;
            }
        }
        mask_inactive(&mut p, &active, w, pool);
        let exact = sys.exact();
        let mut ring = vec![[0.0f64; WINDOW]; w];
        let mut rho_from = vec![f64::NAN; w];
        let mut k = 0usize;
        loop {
            if !active.iter().any(|a| *a) {
                break 'restart;
            }
            if iterations >= maxiter {
                return Err(RfError::Convergence(maxiter));
            }
            ctx.check()?;
            sys.matvec(&p, &mut q, pool)?;
            let pq = dots(&p, &q, w, pool);
            let mut gamma = vec![0.0; w];
            for j in 0..w {
                if active[j] {
                    if pq[j] > 0.0 && pq[j].is_finite() {
                        gamma[j] = rho[j] / pq[j];
                    } else {
                        active[j] = false;
                    }
                }
            }
            // x += γp, r −= γq (active columns only; γ = 0 elsewhere).
            update_xr(x, &mut r, &p, &q, &gamma, w, pool);
            sys.precond(&r, &mut z, pool);
            let rho_new = dots(&r, &z, w, pool);
            iterations += 1;
            k += 1;
            let xb = dots(x, rhs, w, pool);
            let xr = dots(x, &r, w, pool);
            for j in 0..w {
                if !active[j] {
                    continue;
                }
                ring[j][(k - 1) % WINDOW] = gamma[j] * rho[j];
                let ssk = ss1[j] - (xb[j] + xr[j]);
                let thr = (tol * tol * ssk).max(floor[j]);
                let done = if exact {
                    rho_new[j] <= thr
                } else {
                    k >= WINDOW && ring[j].iter().sum::<f64>() <= thr
                };
                // The preconditioned residual reached roundoff: the recursion cannot improve x
                // further and more steps only amplify rounding errors (breakdown).
                let at_roundoff = rho_new[j] <= floor[j];
                if done || at_roundoff || !(rho_new[j] > 0.0) {
                    active[j] = false;
                }
            }
            if k == PROBE_FROM {
                rho_from.copy_from_slice(&rho_new);
            }
            if k == PROBE_AT && !escalated && sys.can_escalate() {
                let slow = (0..w).any(|j| {
                    active[j] && rho_from[j] > 0.0 && (rho_new[j] / rho_from[j]).powf(1.0 / (PROBE_AT - PROBE_FROM) as f64) > PROBE_RATIO
                });
                if slow && sys.escalate(pool, ctx)? {
                    escalated = true;
                    continue 'restart;
                }
            }
            // p = z + β p.
            let mut beta = vec![0.0; w];
            for j in 0..w {
                if active[j] {
                    beta[j] = rho_new[j] / rho[j];
                }
            }
            update_p(&mut p, &z, &beta, &active, w, pool);
            rho = rho_new;
        }
    }
    Ok(CgOutcome { iterations, escalated })
}

fn par_zip3(out: &mut [f64], a: &[f64], b: &[f64], pool: &Pool, f: impl Fn(&mut f64, f64, f64) + Sync) {
    pool.install(|| {
        out.par_chunks_mut(CHUNK).zip(a.par_chunks(CHUNK)).zip(b.par_chunks(CHUNK)).for_each(|((o, a), b)| {
            for ((o, a), b) in o.iter_mut().zip(a).zip(b) {
                f(o, *a, *b);
            }
        })
    });
}

fn mask_inactive(p: &mut [f64], active: &[bool], w: usize, pool: &Pool) {
    if active.iter().all(|a| *a) {
        return;
    }
    pool.install(|| {
        p.par_chunks_mut(w * 1024).for_each(|ch| {
            for row in ch.chunks_exact_mut(w) {
                for (v, a) in row.iter_mut().zip(active) {
                    if !a {
                        *v = 0.0;
                    }
                }
            }
        })
    });
}

fn update_xr(x: &mut [f64], r: &mut [f64], p: &[f64], q: &[f64], gamma: &[f64], w: usize, pool: &Pool) {
    let rows = w * 1024;
    pool.install(|| {
        x.par_chunks_mut(rows).zip(r.par_chunks_mut(rows)).zip(p.par_chunks(rows).zip(q.par_chunks(rows))).for_each(
            |((xc, rc), (pc, qc))| {
                for (((xr, rr), pr), qr) in xc.chunks_exact_mut(w).zip(rc.chunks_exact_mut(w)).zip(pc.chunks_exact(w)).zip(qc.chunks_exact(w)) {
                    for j in 0..w {
                        xr[j] += gamma[j] * pr[j];
                        rr[j] -= gamma[j] * qr[j];
                    }
                }
            },
        )
    });
}

fn update_p(p: &mut [f64], z: &[f64], beta: &[f64], active: &[bool], w: usize, pool: &Pool) {
    let rows = w * 1024;
    pool.install(|| {
        p.par_chunks_mut(rows).zip(z.par_chunks(rows)).for_each(|(pc, zc)| {
            for (pr, zr) in pc.chunks_exact_mut(w).zip(zc.chunks_exact(w)) {
                for j in 0..w {
                    pr[j] = if active[j] { zr[j] + beta[j] * pr[j] } else { 0.0 };
                }
            }
        })
    });
}
