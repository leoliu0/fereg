//! Lockstep preconditioned conjugate gradient over R columns (plan D5).
//!
//! Each column keeps its own scalars. Column j stops when the Hestenes-Stiefel window estimate
//! sum_{i=k-d}^{k-1} gamma_i rho_i (a lower bound on ||x* - x_{k-d}||^2_S = W-norm error of the residualized
//! column) falls below max(tol^2*SS_k, (1e-13)^2*SS_0). With an exact preconditioner (M = S on the
//! whole reduced space, a reflexive generalized inverse) the solution is x + M^-1 (rhs - S x): it
//! is applied directly, with no matvec or stopping check. Independently, a
//! column stops once rho_k <= (1e-13)^2*SS_0: the recursive residual has reached roundoff, and further
//! steps break down (gamma explodes) before a tight window could be satisfied. At iterations 6 and
//! 12 after the start and after every restart the system is probed ([`CgSystem::probe`]) with
//! the observed contraction and the projected remaining iterations, and may strengthen the
//! preconditioner (default rule at iteration 12: escalate once when the contraction is slow),
//! after which PCG restarts from the current x.
//! Converged columns are frozen (their direction is zeroed so the shared matvec leaves them
//! untouched). Dot products use the fixed-chunk reduction (thread-count invariant).

use rayon::prelude::*;

use crate::api::Ctx;
use crate::error::RfError;
use crate::par::{CHUNK, Pool, sum_chunks};

/// Window length d of the stopping rule.
pub const WINDOW: usize = 5;
/// Iteration of the escalation probe, and the reference iteration of the contraction estimate
/// rhohat = (rho_12/rho_7)^{1/5}. (The plan's rho_2 reference overstates the asymptotic rate: the first
/// iterations remove the high-frequency error, e.g. rho drops 800x from k = 2 to 12 on a band
/// graph that then needs 270 Jacobi iterations.)
pub const PROBE_AT: usize = 12;
pub const PROBE_FROM: usize = 7;
/// Escalate when the per-iteration contraction rhohat of some active column exceeds this.
pub const PROBE_RATIO: f64 = 0.6;
/// Earlier probe (no default rule; only systems with a cost model act on it), with
/// rhohat = (rho_6/rho_3)^{1/3}. The early contraction is faster than the asymptotic one, so the
/// remaining-iteration estimate is low (conservative): ~33 instead of 56 on the AKM panels.
pub const EARLY_PROBE_AT: usize = 6;
pub const EARLY_PROBE_FROM: usize = 3;

/// State of the iteration at a probe ([`CgSystem::probe`]).
pub struct Probe {
    /// Iterations done since the start: [`EARLY_PROBE_AT`] or [`PROBE_AT`].
    pub k: usize,
    /// Largest contraction estimate rhohat = (rho_k/rho_from)^{1/(k-from)} over the active columns
    /// (from = [`EARLY_PROBE_FROM`] or [`PROBE_FROM`]).
    pub rho_hat: f64,
    /// Estimated further iterations with the current preconditioner:
    /// max_j ln(rho_k,j / thr_j) / ln(1/rhohat_j) over the active columns (thr_j the stopping
    /// threshold; capped by the remaining `maxiter`).
    pub remaining: f64,
}

/// The reduced system as seen by PCG.
pub trait CgSystem {
    /// q = S p (m x w row-major).
    fn matvec(&mut self, p: &[f64], q: &mut [f64], pool: &Pool) -> Result<(), RfError>;
    /// z = M^-1 r.
    fn precond(&mut self, r: &[f64], z: &mut [f64], pool: &Pool);
    /// M = S exactly (the preconditioner residual norm is the error).
    fn exact(&self) -> bool;
    /// Whether a stronger preconditioner is available.
    fn can_escalate(&self) -> bool;
    /// Builds the stronger preconditioner; `Ok(true)` when it changed.
    fn escalate(&mut self, pool: &Pool, ctx: &Ctx) -> Result<bool, RfError>;
    /// Policy hook at k = [`EARLY_PROBE_AT`] and k = [`PROBE_AT`] after the start and after
    /// every restart. `Ok(Some(changed))` is the system's own decision; `Ok(None)` applies the
    /// default rule (at [`PROBE_AT`] only: [`CgSystem::escalate`] when rhohat > [`PROBE_RATIO`] and
    /// [`CgSystem::can_escalate`]). Implementations must not report a change indefinitely.
    fn probe(&mut self, _p: &Probe, _pool: &Pool, _ctx: &Ctx) -> Result<Option<bool>, RfError> {
        Ok(None)
    }
}

pub struct CgOutcome {
    pub iterations: usize,
    pub escalated: bool,
    /// The final (re)start used an exact preconditioner: x solves S x = rhs directly.
    pub exact: bool,
}

/// Per-column dot products of two m x w row-major blocks.
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

/// Solves S x = rhs for the w columns of `rhs` (m x w); `x` holds the start (warm or zero).
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
    let mut exact = false;
    let floor: Vec<f64> = ss0.iter().map(|s| 1e-26 * s).collect();
    let mut thr = vec![0.0; w];

    'restart: loop {
        // r = rhs - S x, z = M^-1 r, p = z, rho = r'z.
        let x_nonzero = x.iter().any(|v| *v != 0.0);
        if x_nonzero {
            sys.matvec(x, &mut q, pool)?;
            par_zip3(&mut r, rhs, &q, pool, |r, a, b| *r = a - b);
        } else {
            r.copy_from_slice(rhs);
        }
        ctx.check()?;
        sys.precond(&r, &mut z, pool);
        if sys.exact() {
            // x* = x + M^-1 (rhs - S x) for the active columns (the others stay frozen).
            let act = &active;
            pool.install(|| {
                x.par_chunks_mut(CHUNK * w).zip(z.par_chunks(CHUNK * w)).for_each(|(xc, zc)| {
                    for (xr, zr) in xc.chunks_exact_mut(w).zip(zc.chunks_exact(w)) {
                        for j in 0..w {
                            if act[j] {
                                xr[j] += zr[j];
                            }
                        }
                    }
                })
            });
            iterations += 1;
            exact = true;
            break 'restart;
        }
        p.copy_from_slice(&z);
        let mut rho = dots(&r, &z, w, pool);
        for j in 0..w {
            if !(rho[j] > 0.0) {
                active[j] = false;
            }
        }
        mask_inactive(&mut p, &active, w, pool);
        let mut ring = vec![[0.0f64; WINDOW]; w];
        let mut rho_from = vec![f64::NAN; w];
        let mut rho_early = vec![f64::NAN; w];
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
            // x += gamma p, r -= gamma q (active columns only; gamma = 0 elsewhere).
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
                thr[j] = (tol * tol * ssk).max(floor[j]);
                let done = k >= WINDOW && ring[j].iter().sum::<f64>() <= thr[j];
                // The preconditioned residual reached roundoff: the recursion cannot improve x
                // further and more steps only amplify rounding errors (breakdown).
                let at_roundoff = rho_new[j] <= floor[j];
                if done || at_roundoff || !(rho_new[j] > 0.0) {
                    active[j] = false;
                }
            }
            if k == EARLY_PROBE_FROM {
                rho_early.copy_from_slice(&rho_new);
            }
            if k == PROBE_FROM {
                rho_from.copy_from_slice(&rho_new);
            }
            if k == EARLY_PROBE_AT || k == PROBE_AT {
                let (from, refv) = if k == PROBE_AT { (PROBE_FROM, &rho_from) } else { (EARLY_PROBE_FROM, &rho_early) };
                let span = (k - from) as f64;
                let (mut rho_hat, mut remaining) = (0.0f64, 0.0f64);
                let left = maxiter.saturating_sub(iterations) as f64;
                for j in (0..w).filter(|&j| active[j] && refv[j] > 0.0) {
                    let rh = (rho_new[j] / refv[j]).powf(1.0 / span);
                    rho_hat = rho_hat.max(rh);
                    let rem = if rh < 1.0 { (rho_new[j] / thr[j]).ln().max(0.0) / (1.0 / rh).ln() } else { left };
                    remaining = remaining.max(rem.min(left));
                }
                let changed = match sys.probe(&Probe { k, rho_hat, remaining }, pool, ctx)? {
                    Some(changed) => changed,
                    None => k == PROBE_AT && rho_hat > PROBE_RATIO && sys.can_escalate() && sys.escalate(pool, ctx)?,
                };
                if changed {
                    escalated = true;
                    continue 'restart;
                }
            }
            // p = z + beta p.
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
    Ok(CgOutcome { iterations, escalated, exact })
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
