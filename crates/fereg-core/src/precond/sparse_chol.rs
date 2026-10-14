//! Exact sparse Cholesky of one large block S_bb (plan D5 (b) 1): faer supernodal/simplicial
//! LDL^T with AMD ordering, on the unit-diagonal scaled Laplacian, with the same dynamic
//! regularization (grounding of null directions) as the dense block.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::perm::PermRef;
use faer::sparse::linalg::amd;
use faer::sparse::linalg::cholesky::{CholeskySymbolicParams, LdltRef, SymbolicCholesky, SymbolicCholeskyRaw, SymmetricOrdering, factorize_symbolic_cholesky};
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

/// Multiply-adds below which the numeric factorization runs sequentially: faer's parallel
/// supernodal factorization is slower than the sequential one for small factors (measured on
/// the AKM panels: 1e9 flops take 0.08 s on 16 threads vs 0.035 s on one). The result bits do
/// not depend on the parallelism (tests/determinism.rs).
const PAR_FLOPS: f64 = 1e10;

fn par(pool: &Pool, flops: f64) -> Par {
    if pool.threads() > 1 && flops >= PAR_FLOPS { Par::rayon(pool.threads()) } else { Par::Seq }
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
        // Row i of the symmetric matrix = column i; keep entries with row index >= i.
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
    /// Symbolic analysis and, when `len_val <= max(8*nnz, 2*10^7)`, the numeric factorization.
    /// Returns `None` when the fill budget is exceeded or the factorization fails.
    pub fn try_factor(a: &Csr, pool: &Pool) -> Result<Option<SparseChol>, RfError> {
        let nnz = a.col.len();
        let Some((mut sc, vals)) = SparseChol::analyze(a, pool) else { return Ok(None) };
        if sc.len_val() > (8 * nnz).max(20_000_000) {
            return Ok(None);
        }
        Ok(sc.factor(&vals, pool)?.then_some(sc))
    }

    /// Symbolic analysis only (AMD ordering, elimination structure): returns the unfactored
    /// solver and the scaled lower-triangle values for [`SparseChol::factor`]; `None` when faer
    /// rejects the pattern.
    pub fn analyze(a: &Csr, pool: &Pool) -> Option<(SparseChol, Vec<f64>)> {
        SparseChol::analyze_last(a, 0, pool)
    }

    /// As [`SparseChol::analyze`], with coordinates `0..last` eliminated last (in their order)
    /// and AMD applied to the submatrix of the others only. For dense-ish leading rows (the small
    /// blocks of the joint matrix, coupled to almost every coordinate) this is the ordering AMD
    /// would approach, at the AMD cost of the sparse part alone.
    pub fn analyze_last(a: &Csr, last: usize, pool: &Pool) -> Option<(SparseChol, Vec<f64>)> {
        let n = a.n;
        let (col_ptr, row_idx, vals, scale) = lower_scaled(a);
        let order = if last == 0 || last >= n {
            None
        } else {
            let nl = n - last;
            // Pattern of the trailing submatrix (rows of the symmetric CSR = columns).
            let mut cp = Vec::with_capacity(nl + 1);
            cp.push(0u32);
            let mut ri = Vec::new();
            for i in last..n {
                for k in a.ptr[i] as usize..a.ptr[i + 1] as usize {
                    let j = a.col[k] as usize;
                    if j >= last {
                        ri.push((j - last) as u32);
                    }
                }
                cp.push(ri.len() as u32);
            }
            let (mut p, mut pinv) = (vec![0u32; nl], vec![0u32; nl]);
            let sym = SymbolicSparseColMatRef::<u32>::new_checked(nl, nl, &cp, None, &ri);
            let mut mem = MemBuffer::new(amd::order_scratch::<u32>(nl, ri.len()));
            amd::order(&mut p, &mut pinv, sym, amd::Control::default(), MemStack::new(&mut mem)).ok()?;
            let mut fwd: Vec<u32> = p.iter().map(|&v| v + last as u32).collect();
            fwd.extend(0..last as u32);
            let mut inv = vec![0u32; n];
            for (i, &o) in fwd.iter().enumerate() {
                inv[o as usize] = i as u32;
            }
            Some((fwd, inv))
        };
        let symbolic = {
            let sym = SymbolicSparseColMatRef::<u32>::new_checked(n, n, &col_ptr, None, &row_idx);
            let ordering = match &order {
                Some((fwd, inv)) => SymmetricOrdering::Custom(PermRef::new_checked(fwd, inv, n)),
                None => SymmetricOrdering::Amd,
            };
            pool.install(|| factorize_symbolic_cholesky(sym, Side::Lower, ordering, CholeskySymbolicParams::default()))
        };
        let symbolic = symbolic.ok()?;
        Some((SparseChol { col_ptr, row_idx, scale, symbolic, values: Vec::new() }, vals))
    }

    /// Numeric factorization of the analyzed pattern with the values from [`SparseChol::analyze`].
    pub fn factor(&mut self, vals: &[f64], pool: &Pool) -> Result<bool, RfError> {
        self.numeric(vals, pool)
    }

    /// Entries of the factor L (the solve streams them twice per right-hand side).
    pub fn len_val(&self) -> usize {
        self.symbolic.len_val()
    }

    /// sum_j (entries of column j of L)^2: the multiply-add count of the numeric factorization.
    pub fn flops(&self) -> f64 {
        let sq = |x: f64| x * (x + 1.0) * (2.0 * x + 1.0) / 6.0;
        match self.symbolic.raw() {
            SymbolicCholeskyRaw::Simplicial(s) => s.col_ptr().windows(2).map(|w| ((w[1] - w[0]) as f64).powi(2)).sum(),
            SymbolicCholeskyRaw::Supernodal(s) => {
                let (b, e, rp) = (s.supernode_begin(), s.supernode_end(), s.col_ptr_for_row_idx());
                (0..s.n_supernodes())
                    .map(|k| {
                        let ncols = (e[k] - b[k]) as f64;
                        let below = (rp[k + 1] - rp[k]) as f64;
                        sq(ncols + below) - sq(below)
                    })
                    .sum()
            }
        }
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
        let p = par(pool, self.flops());
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

    /// Solves in place for the k x w column-major block `rhs` (kept coordinates of the block).
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
