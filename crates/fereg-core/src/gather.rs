//! Row-major data blocks in solver row order (plan §1.5).

use std::mem::MaybeUninit;

use rayon::prelude::*;

use crate::error::RfError;
use crate::par::{CHUNK, Pool, alloc_uninit, alloc_vec, assume_init};

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

/// A column of [`gather_input`].
#[derive(Clone, Copy, Debug)]
pub enum Col<'a> {
    /// Values over the input observations.
    Values(&'a [f64]),
    /// The constant 1 (an intercept column).
    One,
}

impl Col<'_> {
    #[inline]
    fn at(&self, i: usize) -> f64 {
        match self {
            Col::Values(v) => v[i],
            Col::One => 1.0,
        }
    }
}

/// Input-order columns → row-major block in row order: `out[k·R + j] = cols[j][kept[order[k]]]`
/// (`kept = None`: nothing dropped, the identity), composing the two index maps instead of
/// compacting the columns first. Local orders read the columns directly; otherwise the kept rows
/// are transposed first (ascending reads) and then permuted, as in [`gather_rows`]. Buffers are
/// written in parallel without a zeroing pass.
pub fn gather_input(order: &[u32], kept: Option<&[u32]>, cols: &[Col], pool: &Pool) -> Result<Vec<f64>, RfError> {
    let r = cols.len();
    let src = |k: usize| kept.map_or(k, |kept| kept[k] as usize);
    if r > 1 && !is_local(order) {
        let rows = fill_rows(order.len(), r, pool, |k, row| {
            let i = src(k);
            row.iter_mut().zip(cols).for_each(|(v, c)| {
                v.write(c.at(i));
            });
        })?;
        return fill_rows(order.len(), r, pool, |k, row| {
            let s = order[k] as usize * r;
            row.iter_mut().zip(&rows[s..s + r]).for_each(|(v, x)| {
                v.write(*x);
            });
        });
    }
    fill_rows(order.len(), r, pool, |k, row| {
        let i = src(order[k] as usize);
        row.iter_mut().zip(cols).for_each(|(v, c)| {
            v.write(c.at(i));
        });
    })
}

/// An n×r row-major block whose row k is written by `fill(k, row)` (every value; parallel over
/// fixed chunks, without a zeroing pass).
fn fill_rows(n: usize, r: usize, pool: &Pool, fill: impl Fn(usize, &mut [MaybeUninit<f64>]) + Sync) -> Result<Vec<f64>, RfError> {
    if r == 0 {
        return Ok(Vec::new());
    }
    let mut out = alloc_uninit::<f64>(n * r)?;
    pool.install(|| {
        out.par_chunks_mut(CHUNK * r).enumerate().for_each(|(c, block)| {
            for (k, row) in block.chunks_exact_mut(r).enumerate() {
                fill(c * CHUNK + k, row);
            }
        })
    });
    // SAFETY: `fill` wrote every value of every row.
    Ok(unsafe { assume_init(out) })
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
