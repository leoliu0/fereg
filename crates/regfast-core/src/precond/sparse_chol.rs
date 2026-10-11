//! Exact sparse Cholesky of one large block S_bb (plan D5 (b) 1): faer supernodal/simplicial
//! LDLᵀ with AMD ordering, on the unit-diagonal scaled Laplacian, with the same dynamic
//! regularization (grounding of null directions) as the dense block.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::cholesky::{CholeskySymbolicParams, LdltRef, SymbolicCholesky, SymmetricOrdering, factorize_symbolic_cholesky};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, Mat, Par, Side};

use super::dense::regularization;
use crate::error::RfError;
use crate::operator::Csr;
use crate::par::Pool;

pub struct SparseChol {
    /// Lower-triangle pattern (CSC) of the scaled matrix; reused by numeric refactorization.
    col_ptr: Vec<u32>,
    row_idx: Vec<u32>,
    scale: Vec<f64>,
    symbolic: SymbolicCholesky<u32>,
    values: Vec<f64>,
}

fn par(pool: &Pool) -> Par {
    if pool.threads() > 1 { Par::rayon(pool.threads()) } else { Par::Seq }
}

/// Lower triangle (CSC) of the symmetric full CSR `a`, scaled to unit diagonal.
fn lower_scaled(a: &Csr) -> (Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>) {
    let n = a.n;
    let mut scale = vec![0.0; n];
    for i in 0..n {
        for k in a.ptr[i] as usize..a.ptr[i + 1] as usize {
            if a.col[k] as usize == i {
                scale[i] = if a.val[k] > 0.0 { 1.0 / a.val[k].sqrt() } else { 1.0 };
            }
        }
    }
    let mut col_ptr = Vec::with_capacity(n + 1);
    col_ptr.push(0u32);
    let mut row_idx = Vec::new();
    let mut vals = Vec::new();
    for i in 0..n {
        // Row i of the symmetric matrix = column i; keep entries with row index ≥ i.
        for k in a.ptr[i] as usize..a.ptr[i + 1] as usize {
            let j = a.col[k] as usize;
            if j >= i {
                row_idx.push(j as u32);
                vals.push(a.val[k] * scale[i] * scale[j]);
            }
        }
        col_ptr.push(row_idx.len() as u32);
    }
    (col_ptr, row_idx, vals, scale)
}

impl SparseChol {
    /// Symbolic analysis and, when `len_val ≤ max(8·nnz, 2·10⁷)`, the numeric factorization.
    /// Returns `None` when the fill budget is exceeded or the factorization fails.
    pub fn try_factor(a: &Csr, pool: &Pool) -> Result<Option<SparseChol>, RfError> {
        let n = a.n;
        let nnz = a.col.len();
        let (col_ptr, row_idx, vals, scale) = lower_scaled(a);
        let symbolic = {
            let sym = SymbolicSparseColMatRef::<u32>::new_checked(n, n, &col_ptr, None, &row_idx);
            pool.install(|| factorize_symbolic_cholesky(sym, Side::Lower, SymmetricOrdering::Amd, CholeskySymbolicParams::default()))
        };
        let symbolic = match symbolic {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };
        if symbolic.len_val() > (8 * nnz).max(20_000_000) {
            return Ok(None);
        }
        let mut sc = SparseChol { col_ptr, row_idx, scale, symbolic, values: Vec::new() };
        Ok(sc.numeric(&vals, pool)?.then_some(sc))
    }

    /// Numeric refactorization for a matrix with the same pattern (GLM weight updates).
    /// Returns `false` when the pattern differs (caller rebuilds).
    pub fn refactor(&mut self, a: &Csr, pool: &Pool) -> Result<bool, RfError> {
        let (col_ptr, row_idx, vals, scale) = lower_scaled(a);
        if col_ptr != self.col_ptr || row_idx != self.row_idx {
            return Ok(false);
        }
        self.scale = scale;
        self.numeric(&vals, pool)
    }

    fn numeric(&mut self, vals: &[f64], pool: &Pool) -> Result<bool, RfError> {
        let n = self.scale.len();
        let mut values = Vec::new();
        values.try_reserve_exact(self.symbolic.len_val()).map_err(|_| RfError::OutOfMemory)?;
        values.resize(self.symbolic.len_val(), 0.0);
        let p = par(pool);
        let signs = vec![1i8; n];
        let reg = regularization(&signs);
        let ok = pool.install(|| {
            let sym = SymbolicSparseColMatRef::<u32>::new_checked(n, n, &self.col_ptr, None, &self.row_idx);
            let a = SparseColMatRef::new(sym, vals);
            let mut buf = MemBuffer::new(self.symbolic.factorize_numeric_ldlt_scratch::<f64>(p, Default::default()));
            self.symbolic.factorize_numeric_ldlt(&mut values, a, Side::Lower, reg, p, MemStack::new(&mut buf), Default::default()).is_ok()
        });
        if ok {
            self.values = values;
        }
        Ok(ok)
    }

    /// Solves in place for the k × w column-major block `rhs` (kept coordinates of the block).
    /// Sequential: faer's parallel triangular solve is several times slower than a sequential
    /// one for the few right-hand sides of a PCG iteration, so callers parallelize across
    /// blocks and column groups instead.
    pub fn solve(&self, rhs: &mut Mat<f64>) {
        let w = rhs.ncols();
        for j in 0..w {
            for i in 0..rhs.nrows() {
                rhs[(i, j)] *= self.scale[i];
            }
        }
        let mut buf = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(w, Par::Seq));
        LdltRef::new(&self.symbolic, &self.values).solve_in_place_with_conj(Conj::No, rhs.as_mut(), Par::Seq, MemStack::new(&mut buf));
        for j in 0..w {
            for i in 0..rhs.nrows() {
                rhs[(i, j)] *= self.scale[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::par::with_pool;

    #[test]
    fn grounded_sparse_factor_solves_laplacian_systems() {
        // Components: ring 0-1-2-3-0 and path 4-5-6, as a full symmetric CSR.
        let n = 7;
        let edges = [(0usize, 1usize, 1.0f64), (1, 2, 2.0), (2, 3, 0.5), (0, 3, 1.5), (4, 5, 3.0), (5, 6, 0.25)];
        let mut dense = vec![0.0; n * n];
        for &(i, j, w) in &edges {
            dense[i * n + j] -= w;
            dense[j * n + i] -= w;
            dense[i * n + i] += w;
            dense[j * n + j] += w;
        }
        let mut ptr = vec![0u32];
        let (mut col, mut val) = (Vec::new(), Vec::new());
        for i in 0..n {
            for j in 0..n {
                if dense[i * n + j] != 0.0 {
                    col.push(j as u32);
                    val.push(dense[i * n + j]);
                }
            }
            ptr.push(col.len() as u32);
        }
        let a = Csr { n, ptr, col, val };
        for threads in [1, 3] {
            with_pool(threads, |pool| {
                let f = SparseChol::try_factor(&a, pool).unwrap().expect("factor");
                let r = [1.0, -3.0, 2.0, 0.0, 0.7, -0.2, -0.5];
                let mut m = Mat::<f64>::from_fn(n, 1, |i, _| r[i]);
                f.solve(&mut m);
                for i in 0..n {
                    let s: f64 = (0..n).map(|j| dense[i * n + j] * m[(j, 0)]).sum();
                    assert!((s - r[i]).abs() < 1e-12, "row {i}: {s} vs {}", r[i]);
                }
            })
            .unwrap();
        }
    }
}
