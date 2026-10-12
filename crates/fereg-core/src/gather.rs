//! Row-major data blocks in solver row order (plan §1.5).

use rayon::prelude::*;

use crate::error::RfError;
use crate::par::{CHUNK, Pool, alloc_vec};

/// At least half of consecutive entries of `order` differ by ≤ 8.
pub fn is_local(order: &[u32]) -> bool {
    if order.len() < 2 {
        return true;
    }
    let near = order.windows(2).filter(|w| w[1].abs_diff(w[0]) <= 8).count();
    2 * near >= order.len() - 1
}

/// `out[k·R + j] = cols[j][order[k]]`. When `order` is local the columns are read directly;
/// otherwise they are first transposed to kept-order rows (sequential reads) and the rows are
/// then permuted (one random read of R contiguous values per row).
pub fn gather_rows(order: &[u32], cols: &[&[f64]], pool: &Pool) -> Result<Vec<f64>, RfError> {
    let r = cols.len();
    if r > 1 && !is_local(order) {
        let mut kept = alloc_vec(order.len() * r, 0.0)?;
        pool.install(|| {
            kept.par_chunks_mut(CHUNK * r).enumerate().for_each(|(c, block)| {
                let base = c * CHUNK;
                for (k, row) in block.chunks_exact_mut(r).enumerate() {
                    for (j, v) in row.iter_mut().enumerate() {
                        *v = cols[j][base + k];
                    }
                }
            })
        });
        return permute_rows(order, &kept, r, pool);
    }
    let mut out = alloc_vec(order.len() * r, 0.0)?;
    if r == 0 {
        return Ok(out);
    }
    pool.install(|| {
        out.par_chunks_mut(CHUNK * r).enumerate().for_each(|(c, block)| {
            let base = c * CHUNK;
            for (k, row) in block.chunks_exact_mut(r).enumerate() {
                let src = order[base + k] as usize;
                for (j, v) in row.iter_mut().enumerate() {
                    *v = cols[j][src];
                }
            }
        })
    });
    Ok(out)
}

/// Row permutation of a row-major block: `out[k] = rows[order[k]]` (R values per row).
pub fn permute_rows(order: &[u32], rows: &[f64], r: usize, pool: &Pool) -> Result<Vec<f64>, RfError> {
    let mut out = alloc_vec(order.len() * r, 0.0)?;
    if r == 0 {
        return Ok(out);
    }
    pool.install(|| {
        out.par_chunks_mut(CHUNK * r).enumerate().for_each(|(c, block)| {
            let base = c * CHUNK;
            for (k, row) in block.chunks_exact_mut(r).enumerate() {
                let src = order[base + k] as usize * r;
                row.copy_from_slice(&rows[src..src + r]);
            }
        })
    });
    Ok(out)
}

/// Inverse of a row-order permutation: `inv[order[k]] = k`.
pub fn inverse(order: &[u32]) -> Vec<u32> {
    let mut inv = vec![0u32; order.len()];
    for (k, &o) in order.iter().enumerate() {
        inv[o as usize] = k as u32;
    }
    inv
}

/// Row-order values → kept order: `out[order[k]] = v[k]`.
pub fn to_kept_order(order: &[u32], v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for (k, &o) in order.iter().enumerate() {
        out[o as usize] = v[k];
    }
    out
}
