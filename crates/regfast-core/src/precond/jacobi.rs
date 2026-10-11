//! Jacobi (diagonal) preconditioner entries: 1/d, 0 for redundant coordinates.

use rayon::prelude::*;

use crate::par::Pool;

pub fn inverse_diagonal(diag: &[f64], redundant: &[bool]) -> Vec<f64> {
    diag.iter().zip(redundant).map(|(d, red)| if *red || !(*d > 0.0) { 0.0 } else { 1.0 / d }).collect()
}

/// z = inv_diag ∘ r row-wise (`w` columns).
pub fn apply(inv_diag: &[f64], r: &[f64], z: &mut [f64], w: usize, pool: &Pool) {
    pool.install(|| {
        z.par_chunks_mut(w * 4096).zip(r.par_chunks(w * 4096)).enumerate().for_each(|(ci, (zc, rc))| {
            let base = ci * 4096;
            for (i, (zr, rr)) in zc.chunks_exact_mut(w).zip(rc.chunks_exact(w)).enumerate() {
                let d = inv_diag[base + i];
                for (a, b) in zr.iter_mut().zip(rr) {
                    *a = d * b;
                }
            }
        })
    });
}
