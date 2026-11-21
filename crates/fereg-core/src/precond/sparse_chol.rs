//! Exact sparse Cholesky of one large block S_bb (plan D5 (b) 1): faer supernodal/simplicial
//! LDL^T with AMD ordering, on the unit-diagonal scaled Laplacian, with the same dynamic
//! regularization (grounding of null directions) as the dense block.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::perm::PermRef;
use faer::sparse::linalg::amd;
use faer::sparse::linalg::cholesky::{CholeskySymbolicParams, LdltRef, SymbolicCholesky, SymbolicCholeskyRaw, SymmetricOrdering, factorize_symbolic_cholesky};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, Mat, Par, Side};
use rayon::prelude::*;

use super::dense::regularization;
use crate::error::RfError;
use crate::operator::Csr;
use crate::par::{CHUNK, Pool, alloc_uninit, assume_init};
use crate::structure::split_bounds;

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

/// AMD dense-row threshold: columns of degree > AMD_DENSE * sqrt(ncol) are ordered last without
/// quadratic degree-update work on high-degree hub levels (see `leverage::AMD_DENSE`).
const AMD_DENSE: f64 = 1.0;

fn par(pool: &Pool, flops: f64) -> Par {
    if pool.threads() > 1 && flops >= PAR_FLOPS { Par::rayon(pool.threads()) } else { Par::Seq }
}

/// Lower triangle (CSC) of the symmetric full CSR `a` (sorted columns), scaled to unit diagonal.
fn lower_scaled(a: &Csr, pool: &Pool) -> Result<(Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>), RfError> {
    let n = a.n;
    let mut scale = vec![0.0; n];
    let mut lo_start = Vec::with_capacity(n);
    let mut col_ptr = Vec::with_capacity(n + 1);
    col_ptr.push(0u32);
    let mut acc = 0u32;
    for i in 0..n {
        let (a0, a1) = (a.ptr[i] as usize, a.ptr[i + 1] as usize);
        let cols = &a.col[a0..a1];
        let p = cols.partition_point(|&c| (c as usize) < i);
        if p < cols.len() && cols[p] as usize == i {
            let d = a.val[a0 + p];
            scale[i] = if d > 0.0 { 1.0 / d.sqrt() } else { 1.0 };
        }
        let s = a0 + p;
        lo_start.push(s);
        acc += (a1 - s) as u32;
        col_ptr.push(acc);
    }
    let total = acc as usize;
    let mut row_idx = alloc_uninit::<u32>(total)?;
    let mut vals = alloc_uninit::<f64>(total)?;
    if total <= CHUNK || pool.threads() <= 1 || n < 512 {
        for i in 0..n {
            let (s, a1) = (lo_start[i], a.ptr[i + 1] as usize);
            let dst = col_ptr[i] as usize;
            let si = scale[i];
            for (k, (&c, &v)) in a.col[s..a1].iter().zip(&a.val[s..a1]).enumerate() {
                row_idx[dst + k].write(c);
                vals[dst + k].write(v * si * scale[c as usize]);
            }
        }
    } else {
        let step = 512;
        let chunks: Vec<usize> = (0..=n.div_ceil(step)).map(|c| (c * step).min(n)).collect();
        let bounds: Vec<usize> = chunks.iter().map(|&i| col_ptr[i] as usize).collect();
        let rp = split_bounds(&mut row_idx, &bounds);
        let vp = split_bounds(&mut vals, &bounds);
        pool.install(|| {
            rp.into_par_iter().zip(vp).enumerate().for_each(|(ci, (rc, vc))| {
                let base = bounds[ci];
                for i in chunks[ci]..chunks[ci + 1] {
                    let (s, a1) = (lo_start[i], a.ptr[i + 1] as usize);
                    let dst = col_ptr[i] as usize - base;
                    let si = scale[i];
                    for (k, (&c, &v)) in a.col[s..a1].iter().zip(&a.val[s..a1]).enumerate() {
                        rc[dst + k].write(c);
                        vc[dst + k].write(v * si * scale[c as usize]);
                    }
                }
            })
        });
    }
    // SAFETY: every slot of `0..total` was written by its owning column `i`.
    Ok((col_ptr, unsafe { assume_init(row_idx) }, unsafe { assume_init(vals) }, scale))
}

/// Outcome of [`SparseChol::analyze_last_budget`].
pub enum AnalyzeResult {
    Ok(SparseChol, Vec<f64>),
    /// Exact simplicial column counts already exceed `max_len` or `accept_lb` returned `false`.
    OverBudget { len_lb: usize, flops_lb: f64 },
    Failed,
}

/// Exact simplicial factor entry count and multiply-add count (`sum c_j`, `sum c_j^2`) of the
/// permuted matrix `P A P^T` (`inv[old] = new_pos`), from its lower triangle (`col_ptr`, `row_idx`).
/// Supernodal amalgamation only adds relaxed zeros, so both are strict lower bounds on
/// [`SparseChol::len_val`] and [`SparseChol::flops`]. Runs in O(nnz) with O(n) scratch, avoiding
/// supernodal pattern allocation when a factor exceeds its fill or flop budget.
fn simplicial_counts(n: usize, col_ptr: &[u32], row_idx: &[u32], inv: Option<&[u32]>) -> Option<(usize, f64)> {
    if n == 0 {
        return Some((0, 0.0));
    }
    // Strictly upper triangle of P A P^T (unsorted columns).
    let mut deg = vec![0u32; n];
    for c in 0..n {
        let pc = inv.map_or(c as u32, |p| p[c]);
        for &r in &row_idx[col_ptr[c] as usize..col_ptr[c + 1] as usize] {
            if r as usize != c {
                let pr = inv.map_or(r, |p| p[r as usize]);
                deg[pc.max(pr) as usize] += 1;
            }
        }
    }
    let mut up_ptr = Vec::with_capacity(n + 1);
    up_ptr.push(0u32);
    let mut off = 0u32;
    for &d in &deg {
        off = off.checked_add(d)?;
        up_ptr.push(off);
    }
    let mut cur = up_ptr[..n].to_vec();
    let mut up_row = vec![0u32; off as usize];
    for c in 0..n {
        let pc = inv.map_or(c as u32, |p| p[c]);
        for &r in &row_idx[col_ptr[c] as usize..col_ptr[c + 1] as usize] {
            if r as usize != c {
                let pr = inv.map_or(r, |p| p[r as usize]);
                let (lo, hi) = if pc < pr { (pc, pr) } else { (pr, pc) };
                let dst = cur[hi as usize] as usize;
                up_row[dst] = lo;
                cur[hi as usize] += 1;
            }
        }
    }
    drop(deg);
    drop(cur);
    let mut etree = vec![u32::MAX; n];
    let mut visited = vec![u32::MAX; n];
    let mut col_counts = vec![1u32; n];
    for j in 0..n {
        let j32 = j as u32;
        visited[j] = j32;
        for &i0 in &up_row[up_ptr[j] as usize..up_ptr[j + 1] as usize] {
            let mut i = i0 as usize;
            while visited[i] != j32 {
                let next = if etree[i] != u32::MAX {
                    etree[i] as usize
                } else {
                    etree[i] = j32;
                    j
                };
                col_counts[i] += 1;
                visited[i] = j32;
                i = next;
            }
        }
    }
    let mut len = 0usize;
    let mut flops = 0.0f64;
    for &c in &col_counts {
        len = len.checked_add(c as usize)?;
        let cf = c as f64;
        flops += cf * cf;
    }
    Some((len, flops))
}

impl SparseChol {
    /// Symbolic analysis and, when `len_val <= max(8*nnz, 2*10^7)`, the numeric factorization.
    /// Returns `None` when the fill budget is exceeded or the factorization fails.
    pub fn try_factor(a: &Csr, pool: &Pool) -> Result<Option<SparseChol>, RfError> {
        let nnz = a.col.len();
        let max_len = (8 * nnz).max(20_000_000);
        let (mut sc, vals) = match SparseChol::analyze_last_budget(a, 0, max_len, |_, _| true, pool) {
            AnalyzeResult::Ok(sc, vals) if sc.len_val() <= max_len => (sc, vals),
            _ => return Ok(None),
        };
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
        match SparseChol::analyze_last_budget(a, last, usize::MAX, |_, _| true, pool) {
            AnalyzeResult::Ok(sc, vals) => Some((sc, vals)),
            _ => None,
        }
    }

    /// As [`SparseChol::analyze_last`], checking exact simplicial `(len_val, flops)` lower bounds
    /// against `max_len` and `accept_lb` before allocating supernodal symbolic structures.
    pub fn analyze_last_budget(
        a: &Csr,
        last: usize,
        max_len: usize,
        accept_lb: impl FnOnce(usize, f64) -> bool,
        pool: &Pool,
    ) -> AnalyzeResult {
        let n = a.n;
        let Ok((col_ptr, row_idx, vals, scale)) = lower_scaled(a, pool) else { return AnalyzeResult::Failed };
        let ctrl = amd::Control { dense: AMD_DENSE, aggressive: true };
        let order = if n == 0 {
            None
        } else if last == 0 {
            let (mut fwd, mut inv) = (vec![0u32; n], vec![0u32; n]);
            let sym = SymbolicSparseColMatRef::<u32>::new_checked(n, n, &col_ptr, None, &row_idx);
            let Ok(mut mem) = MemBuffer::try_new(amd::order_scratch::<u32>(n, row_idx.len())) else { return AnalyzeResult::Failed };
            if amd::order(&mut fwd, &mut inv, sym, ctrl, MemStack::new(&mut mem)).is_err() {
                return AnalyzeResult::Failed;
            }
            Some((fwd, inv))
        } else if last >= n {
            // Natural order, as an explicit permutation: faer's `SymmetricOrdering::Identity`
            // analyzes the upper triangle of the pattern whatever `side` says, and the pattern
            // here is the lower one (the symbolic factor would be the diagonal alone).
            let id: Vec<u32> = (0..n as u32).collect();
            Some((id.clone(), id))
        } else {
            let nl = n - last;
            // Lower triangle of the trailing submatrix: for i >= last, every entry j >= i in
            // `row_idx` already satisfies j >= last.
            let base = col_ptr[last];
            let cp: Vec<u32> = col_ptr[last..=n].iter().map(|&p| p - base).collect();
            let tail = &row_idx[base as usize..];
            let ri: Vec<u32> = if tail.len() <= CHUNK || pool.threads() <= 1 {
                tail.iter().map(|&r| r - last as u32).collect()
            } else {
                pool.install(|| tail.par_iter().with_min_len(CHUNK).map(|&r| r - last as u32).collect())
            };
            let (mut p, mut pinv) = (vec![0u32; nl], vec![0u32; nl]);
            let sym = SymbolicSparseColMatRef::<u32>::new_checked(nl, nl, &cp, None, &ri);
            let Ok(mut mem) = MemBuffer::try_new(amd::order_scratch::<u32>(nl, ri.len())) else { return AnalyzeResult::Failed };
            if amd::order(&mut p, &mut pinv, sym, ctrl, MemStack::new(&mut mem)).is_err() {
                return AnalyzeResult::Failed;
            }
            let mut fwd: Vec<u32> = Vec::with_capacity(n);
            fwd.extend(p.iter().map(|&v| v + last as u32));
            fwd.extend(0..last as u32);
            let mut inv = vec![0u32; n];
            for (i, &o) in fwd.iter().enumerate() {
                inv[o as usize] = i as u32;
            }
            Some((fwd, inv))
        };
        if max_len < usize::MAX {
            let Some((len_lb, flops_lb)) = simplicial_counts(n, &col_ptr, &row_idx, order.as_ref().map(|(_, inv)| inv.as_slice())) else {
                return AnalyzeResult::Failed;
            };
            if len_lb > max_len || !accept_lb(len_lb, flops_lb) {
                return AnalyzeResult::OverBudget { len_lb, flops_lb };
            }
        }
        let symbolic = {
            let sym = SymbolicSparseColMatRef::<u32>::new_checked(n, n, &col_ptr, None, &row_idx);
            let ordering = match &order {
                Some((fwd, inv)) => SymmetricOrdering::Custom(PermRef::new_checked(fwd, inv, n)),
                None => SymmetricOrdering::Identity,
            };
            factorize_symbolic_cholesky(sym, Side::Lower, ordering, CholeskySymbolicParams::default())
        };
        let Ok(symbolic) = symbolic else { return AnalyzeResult::Failed };
        AnalyzeResult::Ok(SparseChol { col_ptr, row_idx, scale, symbolic, values: Vec::new() }, vals)
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
        let (col_ptr, row_idx, vals, scale) = lower_scaled(a, pool)?;
        if col_ptr != self.col_ptr || row_idx != self.row_idx {
            return Ok(false);
        }
        self.scale = scale;
        self.numeric(&vals, pool)
    }

    fn numeric(&mut self, vals: &[f64], pool: &Pool) -> Result<bool, RfError> {
        let n = self.scale.len();
        let len_val = self.symbolic.len_val();
        let p = par(pool, self.flops());
        if self.values.len() != len_val {
            self.values = crate::par::alloc_par(len_val, 0.0, pool)?;
        }
        let values = &mut self.values;
        let signs = vec![1i8; n];
        let reg = regularization(&signs);
        let ok = if matches!(p, Par::Seq) {
            let sym = SymbolicSparseColMatRef::<u32>::new_checked(n, n, &self.col_ptr, None, &self.row_idx);
            let a = SparseColMatRef::new(sym, vals);
            let mut buf = MemBuffer::try_new(self.symbolic.factorize_numeric_ldlt_scratch::<f64>(p, Default::default())).map_err(|_| RfError::OutOfMemory)?;
            self.symbolic.factorize_numeric_ldlt(values, a, Side::Lower, reg, p, MemStack::new(&mut buf), Default::default()).is_ok()
        } else {
            let scratch = self.symbolic.factorize_numeric_ldlt_scratch::<f64>(p, Default::default());
            let mut buf = MemBuffer::try_new(scratch).map_err(|_| RfError::OutOfMemory)?;
            pool.install(|| {
                let sym = SymbolicSparseColMatRef::<u32>::new_checked(n, n, &self.col_ptr, None, &self.row_idx);
                let a = SparseColMatRef::new(sym, vals);
                self.symbolic.factorize_numeric_ldlt(values, a, Side::Lower, reg, p, MemStack::new(&mut buf), Default::default()).is_ok()
            })
        };
        Ok(ok)
    }

    /// Solves in place for the k x w column-major block `rhs` (kept coordinates of the block).
    /// Sequential: faer's parallel triangular solve is several times slower than a sequential
    /// one for the few right-hand sides of a PCG iteration, so callers parallelize across
    /// blocks and column groups instead.
    pub fn solve(&self, rhs: &mut Mat<f64>) {
        let w = rhs.ncols();
        for j in 0..w {
            for (v, &s) in rhs.col_as_slice_mut(j).iter_mut().zip(&self.scale) {
                *v *= s;
            }
        }
        let mut buf = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(w, Par::Seq));
        LdltRef::new(&self.symbolic, &self.values).solve_in_place_with_conj(Conj::No, rhs.as_mut(), Par::Seq, MemStack::new(&mut buf));
        for j in 0..w {
            for (v, &s) in rhs.col_as_slice_mut(j).iter_mut().zip(&self.scale) {
                *v *= s;
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

    /// Every coordinate eliminated last (`last = n`: the joint matrix when no large coordinate
    /// is kept): natural order, on a dense positive definite matrix.
    #[test]
    fn natural_order_factor_solves_dense_systems() {
        let n = 6;
        let mut dense = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                dense[i * n + j] = if i == j { 4.0 + i as f64 } else { -1.0 / (1.0 + (i as f64 - j as f64).abs()) };
            }
        }
        let ptr: Vec<u32> = (0..=n).map(|i| (i * n) as u32).collect();
        let col: Vec<u32> = (0..n * n).map(|k| (k % n) as u32).collect();
        let a = Csr { n, ptr, col, val: dense.clone() };
        with_pool(1, |pool| {
            for last in [n, n + 3] {
                let (mut f, vals) = SparseChol::analyze_last(&a, last, pool).expect("analysis");
                assert!(f.factor(&vals, pool).unwrap(), "factorization failed");
                let r = [1.0, -3.0, 2.0, 0.0, 0.7, -0.2];
                let mut m = Mat::<f64>::from_fn(n, 1, |i, _| r[i]);
                f.solve(&mut m);
                for i in 0..n {
                    let s: f64 = (0..n).map(|j| dense[i * n + j] * m[(j, 0)]).sum();
                    assert!((s - r[i]).abs() < 1e-12, "last {last}, row {i}: {s} vs {}", r[i]);
                }
            }
        })
        .unwrap();
    }
}
