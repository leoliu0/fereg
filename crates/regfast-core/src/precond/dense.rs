//! Exact dense factor of the small super-block S_SS (plan D5 (a)).
//!
//! S_SS is singular (one null direction per connected component). It is scaled to unit
//! diagonal and factored by LDLᵀ with dynamic regularization: a pivot below [`PIVOT_TOL`] marks a
//! null direction and is replaced by a huge value, which grounds that coordinate (its component
//! of the solution becomes ≈ 0). (faer's LLᵀ leaves a stale diagonal for regularized pivots,
//! its LDLᵀ does not.) The result is a reflexive generalized inverse, so null-space components are
//! neither amplified nor needed (every null direction of a diagonal block of S is a null
//! direction of S itself).

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::{LdltRegularization, cholesky_in_place, cholesky_in_place_scratch};
use faer::linalg::cholesky::ldlt::solve::{solve_in_place_scratch, solve_in_place_with_conj};
use faer::{Conj, Mat, Par};

use crate::error::RfError;
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

/// Parallelism for a dense operation of dimension `k` inside `pool`.
pub fn par_for(k: usize, pool: &Pool) -> Par {
    if k >= 256 && pool.threads() > 1 { Par::rayon(pool.threads()) } else { Par::Seq }
}

pub struct Dense {
    /// Kept coordinates (indices into the small range).
    keep: Vec<u32>,
    /// 1/√S_ii of each kept coordinate.
    scale: Vec<f64>,
    /// Unit lower L (strict lower part) and D (diagonal).
    l: Mat<f64>,
}

impl Dense {
    /// Factors the m_s × m_s row-major matrix `a`, skipping `redundant` coordinates.
    pub fn factor(a: &[f64], ms: usize, redundant: &[bool], pool: &Pool) -> Result<Dense, RfError> {
        let keep: Vec<u32> = (0..ms).filter(|&i| !redundant[i] && a[i * ms + i] > 0.0).map(|i| i as u32).collect();
        let k = keep.len();
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
        Ok(Dense { keep, scale, l })
    }

    /// z[0..m_s] = S_SS⁺ r[0..m_s] (both row-major with `w` columns; redundant rows get 0).
    pub fn apply(&self, r: &[f64], z: &mut [f64], ms: usize, w: usize, pool: &Pool) {
        z[..ms * w].iter_mut().for_each(|v| *v = 0.0);
        let k = self.keep.len();
        if k == 0 {
            return;
        }
        let mut rhs = Mat::<f64>::from_fn(k, w, |i, j| r[self.keep[i] as usize * w + j] * self.scale[i]);
        let par = par_for(k, pool);
        pool.install(|| {
            let mut buf = MemBuffer::new(solve_in_place_scratch::<f64>(k, w, par));
            solve_in_place_with_conj(self.l.as_ref(), self.l.diagonal(), Conj::No, rhs.as_mut(), par, MemStack::new(&mut buf));
        });
        for (i, &c) in self.keep.iter().enumerate() {
            for j in 0..w {
                z[c as usize * w + j] = rhs[(i, j)] * self.scale[i];
            }
        }
    }
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
}
