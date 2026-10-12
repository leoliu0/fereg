//! Exact factor of the small super-block S_SS (plan D5 (a)): dense LDLᵀ, or a sparse LDLᵀ
//! (AMD ordering) when S_SS is sparse enough for that to cost a fraction of the dense one.
//!
//! S_SS is singular (one null direction per connected component). It is scaled to unit
//! diagonal and factored by LDLᵀ with dynamic regularization: a pivot below [`PIVOT_TOL`] marks a
//! null direction and is replaced by a huge value, which grounds that coordinate (its component
//! of the solution becomes ≈ 0). (faer's LLᵀ leaves a stale diagonal for regularized pivots,
//! its LDLᵀ does not; the sparse factor grounds the same way.) The result is a reflexive generalized inverse, so null-space components are
//! neither amplified nor needed (every null direction of a diagonal block of S is a null
//! direction of S itself).

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::{LdltRegularization, cholesky_in_place, cholesky_in_place_scratch};
use faer::linalg::cholesky::ldlt::solve::{solve_in_place_scratch, solve_in_place_with_conj};
use faer::{Conj, Mat, Par};
use rayon::prelude::*;
use std::ops::Range;

use super::SOLVE_COLS;
use super::sparse_chol::SparseChol;
use crate::error::RfError;
use crate::operator::Csr;
use crate::par::Pool;

/// Relative pivot below which a coordinate is grounded (unit-diagonal scale).
pub const PIVOT_TOL: f64 = 1e-10;
/// Replacement pivot for grounded coordinates.
pub const GROUND: f64 = 1e30;

/// LDLᵀ regularization grounding pivots below [`PIVOT_TOL`] (`signs` all +1: S is PSD).
pub fn regularization(signs: &[i8]) -> LdltRegularization<'_, f64> {
    LdltRegularization {
        dynamic_regularization_signs: Some(signs),
        dynamic_regularization_delta: GROUND,
        dynamic_regularization_epsilon: PIVOT_TOL,
    }
}

/// Parallelism for a dense operation of dimension `k` inside `pool`: sequential below 1024
/// (faer's parallel start-up costs more than it saves on small blocks).
pub fn par_for(k: usize, pool: &Pool) -> Par {
    if k >= 1024 && pool.threads() > 1 { Par::rayon(pool.threads()) } else { Par::Seq }
}

/// Off-diagonal density (nonzeros / k²) up to which a sparse factor is tried, and the size
/// below which the dense factor is always used (its cost is negligible there).
pub const SPARSE_DENSITY: f64 = 1.0 / 16.0;
const SPARSE_MIN_K: usize = 256;
/// The sparse factor is kept when its multiply-adds are at most this share of the dense ones
/// (k³/3); a supernodal factor runs at a lower flop rate than the dense LDLᵀ.
const SPARSE_FLOP_SHARE: f64 = 0.25;

/// Representation chosen by [`Dense::factor`]: dense, or sparse with its multiply-adds and
/// factor entries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SmallWork {
    Dense,
    Sparse { flops: f64, entries: usize },
}

/// Exact factor of S_SS over the kept coordinates.
enum Factor {
    /// Unit lower L (strict lower part) and D (diagonal), of S_SS scaled by `scale`.
    Dense { scale: Vec<f64>, l: Mat<f64> },
    /// Sparse LDLᵀ (AMD ordering, same grounding) when S_SS is sparse: the small blocks of
    /// panels whose levels meet few others within FE1 levels (e.g. firms within industry-years)
    /// factor in a fraction of the dense cost.
    Sparse(Box<SparseChol>),
}

pub struct Dense {
    /// Kept coordinates (indices into the small range).
    keep: Vec<u32>,
    f: Factor,
}

impl Dense {
    /// Factors the m_s × m_s row-major matrix `a`, skipping `redundant` coordinates.
    pub fn factor(a: &[f64], ms: usize, redundant: &[bool], pool: &Pool) -> Result<Dense, RfError> {
        let keep: Vec<u32> = (0..ms).filter(|&i| !redundant[i] && a[i * ms + i] > 0.0).map(|i| i as u32).collect();
        let k = keep.len();
        if k >= SPARSE_MIN_K
            && let Some(f) = sparse_factor(a, ms, &keep, pool)?
        {
            return Ok(Dense { keep, f: Factor::Sparse(Box::new(f)) });
        }
        let scale: Vec<f64> = keep.iter().map(|&i| 1.0 / a[i as usize * (ms + 1)].sqrt()).collect();
        let mut l = Mat::<f64>::from_fn(k, k, |i, j| {
            if j > i { 0.0 } else { a[keep[i] as usize * ms + keep[j] as usize] * scale[i] * scale[j] }
        });
        let par = par_for(k, pool);
        let signs = vec![1i8; k];
        let reg = regularization(&signs);
        let res = pool.install(|| {
            let mut buf = MemBuffer::new(cholesky_in_place_scratch::<f64>(k, par, Default::default()));
            cholesky_in_place(l.as_mut(), reg, par, MemStack::new(&mut buf), Default::default())
        });
        res.map_err(|e| RfError::Internal(format!("dense block factorization failed: {e:?}")))?;
        Ok(Dense { keep, f: Factor::Dense { scale, l } })
    }

    /// Factored coordinates (indices into the small range): the non-redundant ones with a
    /// positive diagonal at factorization time.
    pub fn keep(&self) -> &[u32] {
        &self.keep
    }

    /// Shape of the factorization, for cost models (it does not change with the weights).
    pub fn work(&self) -> SmallWork {
        match &self.f {
            Factor::Dense { .. } => SmallWork::Dense,
            Factor::Sparse(f) => SmallWork::Sparse { flops: f.flops(), entries: f.len_val() },
        }
    }

    /// z[0..m_s] = S_SS⁺ r[0..m_s] (both row-major with `w` columns; redundant rows get 0).
    pub fn apply(&self, r: &[f64], z: &mut [f64], ms: usize, w: usize, pool: &Pool) {
        z[..ms * w].iter_mut().for_each(|v| *v = 0.0);
        let k = self.keep.len();
        if k == 0 {
            return;
        }
        match &self.f {
            Factor::Dense { scale, l } => {
                let mut rhs = Mat::<f64>::from_fn(k, w, |i, j| r[self.keep[i] as usize * w + j] * scale[i]);
                let par = par_for(k, pool);
                pool.install(|| {
                    let mut buf = MemBuffer::new(solve_in_place_scratch::<f64>(k, w, par));
                    solve_in_place_with_conj(l.as_ref(), l.diagonal(), Conj::No, rhs.as_mut(), par, MemStack::new(&mut buf));
                });
                for (i, &c) in self.keep.iter().enumerate() {
                    for j in 0..w {
                        z[c as usize * w + j] = rhs[(i, j)] * scale[i];
                    }
                }
            }
            Factor::Sparse(f) => {
                // Column groups of SOLVE_COLS solved sequentially in concurrent tasks (the
                // grouping, hence the bits, does not depend on the thread count).
                let groups: Vec<Range<usize>> = (0..w.div_ceil(SOLVE_COLS)).map(|c| c * SOLVE_COLS..((c + 1) * SOLVE_COLS).min(w)).collect();
                let sols: Vec<Mat<f64>> = pool.install(|| {
                    groups
                        .par_iter()
                        .map(|cols| {
                            let mut rhs = Mat::<f64>::from_fn(k, cols.len(), |i, j| r[self.keep[i] as usize * w + cols.start + j]);
                            f.solve(&mut rhs);
                            rhs
                        })
                        .collect()
                });
                for (cols, sol) in groups.iter().zip(&sols) {
                    for (i, &c) in self.keep.iter().enumerate() {
                        for (j, col) in cols.clone().enumerate() {
                            z[c as usize * w + col] = sol[(i, j)];
                        }
                    }
                }
            }
        }
    }
}

/// Sparse factor of S_SS over `keep` when its off-diagonal density is at most
/// [`SPARSE_DENSITY`] and the factorization takes at most [`SPARSE_FLOP_SHARE`] of the dense
/// multiply-adds; `None` otherwise (the dense factor is used).
fn sparse_factor(a: &[f64], ms: usize, keep: &[u32], pool: &Pool) -> Result<Option<SparseChol>, RfError> {
    let k = keep.len();
    let row_nnz: Vec<u32> = pool.install(|| {
        keep.par_iter()
            .map(|&i| {
                let row = &a[i as usize * ms..(i as usize + 1) * ms];
                keep.iter().filter(|&&j| row[j as usize] != 0.0).count() as u32
            })
            .collect()
    });
    let nnz: u64 = row_nnz.iter().map(|&c| u64::from(c)).sum();
    if (nnz as f64) > SPARSE_DENSITY * (k * k) as f64 {
        return Ok(None);
    }
    let mut ptr = Vec::with_capacity(k + 1);
    ptr.push(0u32);
    for c in &row_nnz {
        ptr.push(ptr.last().copied().unwrap_or(0) + c);
    }
    let (mut col, mut val) = (Vec::with_capacity(nnz as usize), Vec::with_capacity(nnz as usize));
    for &i in keep {
        let row = &a[i as usize * ms..(i as usize + 1) * ms];
        for (jj, &j) in keep.iter().enumerate() {
            let v = row[j as usize];
            if v != 0.0 {
                col.push(jj as u32);
                val.push(v);
            }
        }
    }
    let csr = Csr { n: k, ptr, col, val };
    let Some((mut f, vals)) = SparseChol::analyze(&csr, pool) else { return Ok(None) };
    if f.flops() > SPARSE_FLOP_SHARE * (k as f64).powi(3) / 3.0 {
        return Ok(None);
    }
    Ok(f.factor(&vals, pool)?.then_some(f))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::par::with_pool;

    #[test]
    fn grounded_factor_solves_laplacian_systems() {
        // Two components: path 0-1-2 (weights 1, 2) and edge 3-4 (weight 0.5).
        let ms = 5;
        let mut a = vec![0.0; ms * ms];
        for &(i, j, w) in &[(0usize, 1usize, 1.0f64), (1, 2, 2.0), (3, 4, 0.5)] {
            a[i * ms + j] -= w;
            a[j * ms + i] -= w;
            a[i * ms + i] += w;
            a[j * ms + j] += w;
        }
        with_pool(1, |pool| {
            let d = Dense::factor(&a, ms, &[false; 5], pool).unwrap();
            let r = [1.0, -3.0, 2.0, 0.7, -0.7];
            let mut z = [0.0; 5];
            d.apply(&r, &mut z, ms, 1, pool);
            for i in 0..ms {
                let s: f64 = (0..ms).map(|j| a[i * ms + j] * z[j]).sum();
                assert!((s - r[i]).abs() < 1e-12, "row {i}: {s} vs {} (z = {z:?})", r[i]);
            }
        })
        .unwrap();
    }

    /// A sparse S_SS (two ring components with chords, k = 600) takes the sparse factor and
    /// solves consistent systems exactly, as the dense factor does.
    #[test]
    fn sparse_small_block_solves_laplacian_systems() {
        let ms = 600;
        let half = ms / 2;
        let mut a = vec![0.0; ms * ms];
        let mut edge = |i: usize, j: usize, w: f64| {
            a[i * ms + j] -= w;
            a[j * ms + i] -= w;
            a[i * ms + i] += w;
            a[j * ms + j] += w;
        };
        for c in 0..2 {
            for t in 0..half {
                let (i, j) = (c * half + t, c * half + (t + 1) % half);
                edge(i, j, 1.0 + (t % 5) as f64);
                edge(i, c * half + (t * 7 + 3) % half, 0.5);
            }
        }
        // Right-hand side with zero sum per component (in the range of S).
        let r: Vec<f64> = (0..ms).map(|i| ((i * 13) % 17) as f64 - 8.0).collect();
        let mut r = r;
        for c in 0..2 {
            let mean = r[c * half..(c + 1) * half].iter().sum::<f64>() / half as f64;
            r[c * half..(c + 1) * half].iter_mut().for_each(|v| *v -= mean);
        }
        with_pool(3, |pool| {
            let d = Dense::factor(&a, ms, &vec![false; ms], pool).unwrap();
            assert!(matches!(d.f, Factor::Sparse(_)));
            let mut z = vec![0.0; ms];
            d.apply(&r, &mut z, ms, 1, pool);
            for i in 0..ms {
                let s: f64 = (0..ms).map(|j| a[i * ms + j] * z[j]).sum();
                assert!((s - r[i]).abs() < 1e-9, "row {i}: {s} vs {}", r[i]);
            }
        })
        .unwrap();
    }
}
