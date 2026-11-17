//! Jacobi (diagonal) preconditioner entries: 1/d, 0 for redundant coordinates.

use rayon::prelude::*;

use crate::par::{Pool, dispatch_width, width};

pub fn inverse_diagonal(diag: &[f64], redundant: &[bool]) -> Vec<f64> {
    diag.iter().zip(redundant).map(|(d, red)| if *red || !(*d > 0.0) { 0.0 } else { 1.0 / d }).collect()
}

/// z = inv_diag .* r row-wise (`w` columns).
pub fn apply(inv_diag: &[f64], r: &[f64], z: &mut [f64], w: usize, pool: &Pool) {
    if z.is_empty() || w == 0 {
        return;
    }
    dispatch_width!(w, false, apply_w(inv_diag, r, z, w, pool));
}

fn apply_w<const W: usize, const _B: bool>(inv_diag: &[f64], r: &[f64], z: &mut [f64], w: usize, pool: &Pool) {
    let w = width::<W>(w);
    let step = w * 4096;
    if z.len() <= step || pool.threads() <= 1 {
        for (i, (zr, rr)) in z.chunks_exact_mut(w).zip(r.chunks_exact(w)).enumerate() {
            let d = inv_diag[i];
            for j in 0..w {
                zr[j] = d * rr[j];
            }
        }
        return;
    }
    pool.install(|| {
        z.par_chunks_mut(step).zip(r.par_chunks(step)).enumerate().for_each(|(ci, (zc, rc))| {
            let base = ci * 4096;
            for (i, (zr, rr)) in zc.chunks_exact_mut(w).zip(rc.chunks_exact(w)).enumerate() {
                let d = inv_diag[base + i];
                for j in 0..w {
                    zr[j] = d * rr[j];
                }
            }
        })
    });
}
