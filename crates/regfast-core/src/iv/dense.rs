//! Small dense matrices (row-major) for the IV algebra. Every matrix here is at most a few
//! dozen rows: the O(N) work happens in the streaming passes of `iv.rs`.

use crate::vcov::{self, INVSYM_TOL};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct M {
    pub r: usize,
    pub c: usize,
    pub d: Vec<f64>,
}

impl M {
    pub fn zeros(r: usize, c: usize) -> M {
        M { r, c, d: vec![0.0; r * c] }
    }

    pub fn from_fn(r: usize, c: usize, f: impl Fn(usize, usize) -> f64) -> M {
        let mut d = Vec::with_capacity(r * c);
        for i in 0..r {
            for j in 0..c {
                d.push(f(i, j));
            }
        }
        M { r, c, d }
    }

    pub fn eye(n: usize) -> M {
        M::from_fn(n, n, |i, j| f64::from(u8::from(i == j)))
    }

    pub fn col(v: &[f64]) -> M {
        M { r: v.len(), c: 1, d: v.to_vec() }
    }

    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.d[i * self.c + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.d[i * self.c + j] = v;
    }

    pub fn t(&self) -> M {
        M::from_fn(self.c, self.r, |i, j| self.at(j, i))
    }

    pub fn mul(&self, b: &M) -> M {
        assert_eq!(self.c, b.r, "dense: dimension mismatch");
        let mut out = M::zeros(self.r, b.c);
        for i in 0..self.r {
            for k in 0..self.c {
                let a = self.at(i, k);
                if a == 0.0 {
                    continue;
                }
                for j in 0..b.c {
                    out.d[i * b.c + j] += a * b.d[k * b.c + j];
                }
            }
        }
        out
    }

    pub fn sub(&self, b: &M) -> M {
        M { r: self.r, c: self.c, d: self.d.iter().zip(&b.d).map(|(x, y)| x - y).collect() }
    }

    pub fn scale(&self, s: f64) -> M {
        M { r: self.r, c: self.c, d: self.d.iter().map(|x| x * s).collect() }
    }

    /// Rows `ri` and columns `ci`.
    pub fn pick(&self, ri: &[usize], ci: &[usize]) -> M {
        M::from_fn(ri.len(), ci.len(), |i, j| self.at(ri[i], ci[j]))
    }

    /// Contiguous block rows `r0..r1`, columns `c0..c1`.
    pub fn block(&self, r0: usize, r1: usize, c0: usize, c1: usize) -> M {
        M::from_fn(r1 - r0, c1 - c0, |i, j| self.at(r0 + i, c0 + j))
    }

    /// Kronecker product A ⊗ B.
    pub fn kron(&self, b: &M) -> M {
        M::from_fn(self.r * b.r, self.c * b.c, |i, j| self.at(i / b.r, j / b.c) * b.at(i % b.r, j % b.c))
    }

    /// Column-major vectorization (Mata `vec()`).
    pub fn vec(&self) -> Vec<f64> {
        (0..self.c).flat_map(|j| (0..self.r).map(move |i| (i, j))).map(|(i, j)| self.at(i, j)).collect()
    }

    pub fn diag(&self) -> Vec<f64> {
        (0..self.r.min(self.c)).map(|i| self.at(i, i)).collect()
    }

    pub fn symmetrize(&mut self) {
        for i in 0..self.r {
            for j in 0..i {
                let v = 0.5 * (self.at(i, j) + self.at(j, i));
                self.set(i, j, v);
                self.set(j, i, v);
            }
        }
    }

    /// x'·A·x for a square A.
    pub fn quad(&self, x: &[f64]) -> f64 {
        (0..self.r).map(|i| x[i] * (0..self.c).map(|j| self.at(i, j) * x[j]).sum::<f64>()).sum()
    }
}

/// Lower Cholesky factor of a symmetric positive definite matrix, `None` if not PD.
pub(crate) fn chol(a: &M) -> Option<M> {
    let n = a.r;
    let mut l = M::zeros(n, n);
    for j in 0..n {
        let mut s = a.at(j, j);
        for k in 0..j {
            s -= l.at(j, k) * l.at(j, k);
        }
        if !(s > 0.0) || !s.is_finite() {
            return None;
        }
        let d = s.sqrt();
        l.set(j, j, d);
        for i in j + 1..n {
            let mut s = a.at(i, j);
            for k in 0..j {
                s -= l.at(i, k) * l.at(j, k);
            }
            l.set(i, j, s / d);
        }
    }
    Some(l)
}

/// Inverse of a lower-triangular matrix.
pub(crate) fn tri_lower_inv(l: &M) -> M {
    let n = l.r;
    let mut out = M::zeros(n, n);
    for j in 0..n {
        out.set(j, j, 1.0 / l.at(j, j));
        for i in j + 1..n {
            let s: f64 = (j..i).map(|k| l.at(i, k) * out.at(k, j)).sum();
            out.set(i, j, -s / l.at(i, i));
        }
    }
    out
}

/// Jacobi-scaled Cholesky inverse of an SPD matrix (scaling makes the factorization insensitive
/// to the units of the variables); falls back to the generalized inverse when not PD.
pub(crate) fn spd_inv(a: &M) -> M {
    let n = a.r;
    let s: Vec<f64> = a.diag().iter().map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 1.0 }).collect();
    let scaled = M::from_fn(n, n, |i, j| a.at(i, j) * s[i] * s[j]);
    let inv = match chol(&scaled) {
        Some(l) => {
            let li = tri_lower_inv(&l);
            let mut x = li.t().mul(&li);
            x.symmetrize();
            x
        }
        None => ginv(&scaled).0,
    };
    M::from_fn(n, n, |i, j| inv.at(i, j) * s[i] * s[j])
}

/// A⁻¹·B for SPD A (Mata `cholsolve`, with the generalized-inverse fallback of `cholqrsolve`).
pub(crate) fn spd_solve(a: &M, b: &M) -> M {
    spd_inv(a).mul(b)
}

/// Mata `invsym()`: generalized inverse by sweeping in column order; returns (inverse, rank).
pub(crate) fn ginv(a: &M) -> (M, usize) {
    let (inv, dropped) = vcov::invsym(&a.d, a.r, INVSYM_TOL);
    let rank = dropped.iter().filter(|d| !**d).count();
    (M { r: a.r, c: a.c, d: inv }, rank)
}

/// Eigen-decomposition of a symmetric matrix, eigenvalues in descending order (Mata
/// `symeigensystem`); eigenvectors are the columns of the returned matrix.
pub(crate) fn sym_eig(a: &M) -> (Vec<f64>, M) {
    let n = a.r;
    let (vals, vecs) = vcov::sym_eigen(&a.d, n);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&x, &y| vals[y].total_cmp(&vals[x]));
    let v = M { r: n, c: n, d: vecs };
    (idx.iter().map(|&i| vals[i]).collect(), M::from_fn(n, n, |i, j| v.at(i, idx[j])))
}

/// Symmetric PSD square root U·diag(√max(λ,0))·U'.
pub(crate) fn sqrtm_psd(a: &M) -> M {
    let (vals, u) = sym_eig(a);
    let n = a.r;
    M::from_fn(n, n, |i, j| (0..n).map(|k| u.at(i, k) * vals[k].max(0.0).sqrt() * u.at(j, k)).sum())
}

/// Condition number in the 2-norm of a symmetric PSD matrix (Mata `cond()`).
pub(crate) fn cond_sym(a: &M) -> f64 {
    if a.r == 0 {
        return f64::NAN;
    }
    let (vals, _) = sym_eig(a);
    let max = vals[0].abs();
    let min = vals.iter().fold(f64::INFINITY, |m, v| m.min(v.abs()));
    if min == 0.0 { f64::NAN } else { max / min }
}

/// Inverse of a general square matrix (Gauss–Jordan with partial pivoting); `None` if singular.
pub(crate) fn inv_general(a: &M) -> Option<M> {
    let n = a.r;
    let mut m = a.clone();
    let mut inv = M::eye(n);
    for p in 0..n {
        let piv = (p..n).max_by(|&x, &y| m.at(x, p).abs().total_cmp(&m.at(y, p).abs()))?;
        if m.at(piv, p) == 0.0 {
            return None;
        }
        for j in 0..n {
            m.d.swap(p * n + j, piv * n + j);
            inv.d.swap(p * n + j, piv * n + j);
        }
        let d = m.at(p, p);
        for j in 0..n {
            m.d[p * n + j] /= d;
            inv.d[p * n + j] /= d;
        }
        for i in 0..n {
            if i == p {
                continue;
            }
            let f = m.at(i, p);
            if f == 0.0 {
                continue;
            }
            for j in 0..n {
                m.d[i * n + j] -= f * m.d[p * n + j];
                inv.d[i * n + j] -= f * inv.d[p * n + j];
            }
        }
    }
    Some(inv)
}

/// Full SVD A = U·diag(s)·V' (Mata `fullsvd`): U is r×r, V is c×c, singular values descending.
pub(crate) fn svd_full(a: &M) -> (M, Vec<f64>, M) {
    let fa = faer::Mat::<f64>::from_fn(a.r, a.c, |i, j| a.at(i, j));
    let svd = fa.svd().expect("svd of a small dense matrix");
    let (u, s, v) = (svd.U(), svd.S().column_vector(), svd.V());
    let k = a.r.min(a.c);
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&x, &y| s[y].total_cmp(&s[x]));
    // Reorder the leading singular pairs; trailing (null-space) columns keep their positions.
    let ucol = |j: usize| if j < k { order[j] } else { j };
    let vcol = |j: usize| if j < k { order[j] } else { j };
    let um = M::from_fn(a.r, a.r, |i, j| u[(i, ucol(j))]);
    let vm = M::from_fn(a.c, a.c, |i, j| v[(i, vcol(j))]);
    (um, order.iter().map(|&j| s[j]).collect(), vm)
}
