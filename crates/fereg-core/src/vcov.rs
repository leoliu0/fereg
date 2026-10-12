//! Variance building blocks with reghdfe 6.13.1 conventions (plan Phase 3.7), shared by OLS,
//! IV (Phase 5) and GLM (Phase 6).
//!
//! Everything here works on small dense K×K matrices (row-major) plus streaming "meat" kernels
//! over the rows. Meat kernels are deterministic for any thread count: rows (or rows sorted by
//! cluster) are split into fixed chunks of [`CHUNK`] and partial sums are combined in chunk order.

use std::sync::atomic::{AtomicU32, Ordering};

use rayon::prelude::*;

use crate::error::RfError;
use crate::factor::{self, Codes};
use crate::par::{self, CHUNK, Pool};

/// Relative pivot tolerance of [`invsym`]: a pivot is zero when it is ≤ `INVSYM_TOL`·(its
/// original diagonal). Stata does not document the tolerance of Mata `invsym()`; plan §3.2 fixes it.
pub(crate) const INVSYM_TOL: f64 = 1e-13;

/// Mata `invsym(A, 1..k)`: generalized inverse of the symmetric k×k matrix `a` (row-major) by
/// sweeping the pivots in column order. A pivot ≤ `tol`·(original diagonal) (or ≤ 0) marks the
/// column as dropped: its row and column of the result are 0. Returns (inverse, dropped).
pub(crate) fn invsym(a: &[f64], k: usize, tol: f64) -> (Vec<f64>, Vec<bool>) {
    assert_eq!(a.len(), k * k);
    let mut m = a.to_vec();
    let diag: Vec<f64> = (0..k).map(|i| a[i * k + i]).collect();
    let mut dropped = vec![false; k];
    for p in 0..k {
        let d = m[p * k + p];
        if d.is_nan() || d <= tol * diag[p] || d <= 0.0 {
            dropped[p] = true;
            for j in 0..k {
                m[p * k + j] = 0.0;
                m[j * k + p] = 0.0;
            }
            continue;
        }
        // Goodnight sweep on pivot p (yields +A⁻¹ after sweeping every pivot).
        for j in 0..k {
            m[p * k + j] /= d;
        }
        for i in 0..k {
            if i == p {
                continue;
            }
            let b = m[i * k + p];
            if b == 0.0 {
                continue;
            }
            for j in 0..k {
                m[i * k + j] -= b * m[p * k + j];
            }
            m[i * k + p] = -b / d;
        }
        m[p * k + p] = 1.0 / d;
    }
    // Exact symmetry (the sweep is symmetric up to rounding).
    for i in 0..k {
        for j in 0..i {
            let v = 0.5 * (m[i * k + j] + m[j * k + i]);
            m[i * k + j] = v;
            m[j * k + i] = v;
        }
    }
    (m, dropped)
}

/// reghdfe `reghdfe_extend_b_and_xx`: extends inv(X̃'WX̃) (k×k) with the constant via the
/// partitioned-inverse formula, using the regression-through-means constant:
/// `[[inv, −inv·x̄], [−x̄'inv, 1/n_c + x̄'inv·x̄]]` ((k+1)×(k+1), row-major).
/// `n_c` is N_obs for none/aw/pw and Σfw for fw (reghdfe `tmp_N`).
pub(crate) fn inv_ext(inv: &[f64], k: usize, means: &[f64], n_c: f64) -> Vec<f64> {
    let k1 = k + 1;
    let mut out = vec![0.0; k1 * k1];
    // side = −x̄'·inv (1×k)
    let side: Vec<f64> = (0..k).map(|j| -(0..k).map(|i| means[i] * inv[i * k + j]).sum::<f64>()).collect();
    let corner = 1.0 / n_c - side.iter().zip(means).map(|(s, m)| s * m).sum::<f64>();
    for i in 0..k {
        out[i * k1..i * k1 + k].copy_from_slice(&inv[i * k..(i + 1) * k]);
        out[i * k1 + k] = side[i];
        out[k * k1 + i] = side[i];
    }
    out[k * k1 + k] = corner;
    out
}

/// A·B for row-major square matrices of order k.
pub(crate) fn matmul(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    let mut out = vec![0.0; k * k];
    for i in 0..k {
        for l in 0..k {
            let x = a[i * k + l];
            if x == 0.0 {
                continue;
            }
            for j in 0..k {
                out[i * k + j] += x * b[l * k + j];
            }
        }
    }
    out
}

/// Sandwich D·M·D·scale (D symmetric), symmetrized: the two triangles of the product differ by
/// rounding, which near collinearity is enough for Stata to reject V as nonsymmetric.
pub(crate) fn sandwich(d: &[f64], meat: &[f64], k: usize, scale: f64) -> Vec<f64> {
    let mut v = matmul(&matmul(d, meat, k), d, k);
    for i in 0..k {
        for j in 0..i {
            let s = 0.5 * (v[i * k + j] + v[j * k + i]);
            v[i * k + j] = s;
            v[j * k + i] = s;
        }
    }
    v.iter_mut().for_each(|x| *x *= scale);
    v
}

/// Adds a·aᵀ·s into the m×m row-major accumulator (upper triangle only).
#[inline]
fn add_outer_upper(acc: &mut [f64], a: &[f64], s: f64, m: usize) {
    for i in 0..m {
        let ai = a[i] * s;
        if ai == 0.0 {
            continue;
        }
        let row = &mut acc[i * m..(i + 1) * m];
        for j in i..m {
            row[j] += ai * a[j];
        }
    }
}

fn mirror_upper(acc: &mut [f64], m: usize) {
    for i in 0..m {
        for j in 0..i {
            acc[i * m + j] = acc[j * m + i];
        }
    }
}

/// Robust meat Σ_i m_i·a_i a_iᵀ (m×m). `row(i, a)` fills the design row a_i (length m) of row i
/// and returns its multiplier m_i (reghdfe: e² unweighted, e²·w for fw, (e·w_v)² for aw/pw).
pub(crate) fn robust_meat(n: usize, m: usize, pool: &Pool, row: impl Fn(usize, &mut [f64]) -> f64 + Sync) -> Vec<f64> {
    let mut acc = par::sum_chunks(pool, n, CHUNK, m * m, |range, acc| {
        let mut a = vec![0.0; m];
        for i in range {
            let s = row(i, &mut a);
            add_outer_upper(acc, &a, s, m);
        }
    });
    mirror_upper(&mut acc, m);
    acc
}

/// Rows grouped by `codes` (row order, values < g): `None` when the codes are already
/// non-decreasing in row order (contiguous segments), otherwise the rows listed group by group
/// (ascending row within a group). Atomic counting sort followed by a per-group sort by row, so
/// the result does not depend on the thread count.
pub(crate) fn group_rows(codes: &[u32], g: usize, pool: &Pool) -> Option<Vec<u32>> {
    let n = codes.len();
    if pool.install(|| codes.par_windows(2).all(|w| w[0] <= w[1])) {
        return None;
    }
    let counts: Vec<AtomicU32> = (0..g).map(|_| AtomicU32::new(0)).collect();
    pool.install(|| {
        codes.par_chunks(CHUNK).for_each(|c| {
            for &v in c {
                counts[v as usize].fetch_add(1, Ordering::Relaxed);
            }
        })
    });
    let mut off = Vec::with_capacity(g + 1);
    off.push(0usize);
    for c in &counts {
        off.push(off.last().copied().unwrap_or(0) + c.load(Ordering::Relaxed) as usize);
    }
    let cursor: Vec<AtomicU32> = off[..g].iter().map(|&o| AtomicU32::new(o as u32)).collect();
    let slots: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
    pool.install(|| {
        codes.par_chunks(CHUNK).enumerate().for_each(|(c, chunk)| {
            let base = c * CHUNK;
            for (k, &v) in chunk.iter().enumerate() {
                let pos = cursor[v as usize].fetch_add(1, Ordering::Relaxed);
                slots[pos as usize].store((base + k) as u32, Ordering::Relaxed);
            }
        })
    });
    let mut perm: Vec<u32> = slots.into_iter().map(AtomicU32::into_inner).collect();
    let mut segs: Vec<&mut [u32]> = Vec::with_capacity(g);
    let mut rest: &mut [u32] = &mut perm;
    for q in 0..g {
        let (head, tail) = rest.split_at_mut(off[q + 1] - off[q]);
        segs.push(head);
        rest = tail;
    }
    pool.install(|| segs.par_iter_mut().for_each(|s| s.sort_unstable()));
    Some(perm)
}

/// Cluster meat Σ_g s_g s_gᵀ with s_g = Σ_{i∈g} score_i·a_i (m×m). `row(i, a)` fills a_i and
/// returns score_i (reghdfe: e_i·w_i with w = w_v, fw or 1). `codes` are row-order cluster codes < g.
///
/// Deterministic: positions in grouped order are split into fixed chunks; groups interior to a
/// chunk are accumulated locally, groups crossing chunk boundaries are stitched in chunk order.
pub(crate) fn cluster_meat(
    codes: &[u32],
    g: usize,
    m: usize,
    pool: &Pool,
    row: impl Fn(usize, &mut [f64]) -> f64 + Sync,
) -> Vec<f64> {
    let n = codes.len();
    let perm = group_rows(codes, g, pool);
    let at = |pos: usize| -> usize { perm.as_ref().map_or(pos, |p| p[pos] as usize) };
    struct Part {
        local: Vec<f64>,
        first: (u32, Vec<f64>),
        last: Option<(u32, Vec<f64>)>,
    }
    let parts: Vec<Part> = par::map_chunks(pool, n, CHUNK, |_, range| {
        let mut local = vec![0.0; m * m];
        let mut a = vec![0.0; m];
        let mut cur_g = codes[at(range.start)];
        let mut cur = vec![0.0; m];
        let mut first: Option<(u32, Vec<f64>)> = None;
        for pos in range {
            let i = at(pos);
            let gi = codes[i];
            if gi != cur_g {
                let done = std::mem::replace(&mut cur, vec![0.0; m]);
                if first.is_none() {
                    first = Some((cur_g, done));
                } else {
                    add_outer_upper(&mut local, &done, 1.0, m);
                }
                cur_g = gi;
            }
            let s = row(i, &mut a);
            for (c, v) in cur.iter_mut().zip(&a) {
                *c += s * v;
            }
        }
        match first {
            None => Part { local, first: (cur_g, cur), last: None },
            Some(f) => Part { local, first: f, last: Some((cur_g, cur)) },
        }
    });
    let mut acc_local = vec![0.0; m * m];
    let mut acc_bound = vec![0.0; m * m];
    let mut carry: Option<(u32, Vec<f64>)> = None;
    for p in parts {
        match &mut carry {
            Some((cg, cs)) if *cg == p.first.0 => {
                for (c, v) in cs.iter_mut().zip(&p.first.1) {
                    *c += v;
                }
            }
            _ => {
                if let Some((_, cs)) = carry.take() {
                    add_outer_upper(&mut acc_bound, &cs, 1.0, m);
                }
                carry = Some(p.first);
            }
        }
        if let Some(l) = p.last {
            if let Some((_, cs)) = carry.take() {
                add_outer_upper(&mut acc_bound, &cs, 1.0, m);
            }
            carry = Some(l);
        }
        for (a, v) in acc_local.iter_mut().zip(&p.local) {
            *a += v;
        }
    }
    if let Some((_, cs)) = carry {
        add_outer_upper(&mut acc_bound, &cs, 1.0, m);
    }
    for (a, v) in acc_local.iter_mut().zip(&acc_bound) {
        *a += v;
    }
    mirror_upper(&mut acc_local, m);
    acc_local
}

/// One cluster dimension: kept-order codes with `g` levels.
pub(crate) struct ClusterCodes<'a> {
    pub codes: &'a [u32],
    pub g: usize,
}

/// Multiway cluster meat (reghdfe `reghdfe_vce_cluster`): Σ over non-empty subsets S of the
/// dimensions, in order of |S| and then lexicographically, of (−1)^{|S|+1}·meat(∩S), where the
/// intersection cluster is the [`factor::combine`] of its dimensions. `row_order[k]` is the kept
/// index of row k; `row` is as in [`cluster_meat`] (indexed by row).
pub(crate) fn multiway_meat(
    dims: &[ClusterCodes],
    row_order: &[u32],
    m: usize,
    pool: &Pool,
    row: impl Fn(usize, &mut [f64]) -> f64 + Sync,
) -> Result<Vec<f64>, RfError> {
    multiway_meat_scaled(dims, row_order, m, pool, |_| 1.0, row)
}

/// [`multiway_meat`] with every subset meat multiplied by `scale(G_S)`, G_S the number of
/// clusters of the subset's intersection (fixest `ssc(G.df = "conventional")`: G/(G−1)).
pub(crate) fn multiway_meat_scaled(
    dims: &[ClusterCodes],
    row_order: &[u32],
    m: usize,
    pool: &Pool,
    scale: impl Fn(usize) -> f64,
    row: impl Fn(usize, &mut [f64]) -> f64 + Sync,
) -> Result<Vec<f64>, RfError> {
    let q = dims.len();
    let mut meat = vec![0.0; m * m];
    let owned: Vec<Codes> = dims.iter().map(|d| Codes { codes: d.codes.to_vec(), n: d.g as u32 }).collect();
    for size in 1..=q {
        let sign = if size % 2 == 1 { 1.0 } else { -1.0 };
        for subset in subsets(q, size) {
            let combined;
            let (kept_codes, g): (&[u32], usize) = if size == 1 {
                (dims[subset[0]].codes, dims[subset[0]].g)
            } else {
                let parts: Vec<&Codes> = subset.iter().map(|&d| &owned[d]).collect();
                combined = factor::combine(&parts, pool)?;
                (&combined.codes, combined.n as usize)
            };
            let row_codes = to_row_order(kept_codes, row_order, pool)?;
            let part = cluster_meat(&row_codes, g, m, pool, &row);
            let f = sign * scale(g);
            for (a, v) in meat.iter_mut().zip(&part) {
                *a += f * v;
            }
        }
    }
    Ok(meat)
}

/// Kept-order values → row order (`out[k] = v[row_order[k]]`).
pub(crate) fn to_row_order<T: Copy + Send + Sync + Default>(v: &[T], row_order: &[u32], pool: &Pool) -> Result<Vec<T>, RfError> {
    let mut out = par::alloc_vec(row_order.len(), T::default())?;
    pool.install(|| out.par_iter_mut().zip(row_order.par_iter()).for_each(|(o, &k)| *o = v[k as usize]));
    Ok(out)
}

/// k-subsets of 0..q in lexicographic order (reghdfe `reghdfe_choose_n_k` enumerates the same set
/// in the same order).
fn subsets(q: usize, k: usize) -> Vec<Vec<usize>> {
    fn rec(start: usize, q: usize, k: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for i in start..q {
            cur.push(i);
            rec(i + 1, q, k, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    rec(0, q, k, &mut Vec::new(), &mut out);
    out
}

/// Symmetric eigendecomposition by cyclic Jacobi rotations (deterministic, high relative
/// accuracy; K is small). Returns (eigenvalues, eigenvectors as columns of a row-major k×k matrix).
pub(crate) fn sym_eigen(a: &[f64], k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    let mut v = vec![0.0; k * k];
    for i in 0..k {
        v[i * k + i] = 1.0;
    }
    for _sweep in 0..100 {
        let off: f64 = (0..k).flat_map(|i| (0..k).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| m[i * k + j].powi(2)).sum();
        let total: f64 = m.iter().map(|x| x * x).sum();
        if off <= f64::EPSILON.powi(2) * total * 1e-4 || off == 0.0 {
            break;
        }
        for p in 0..k {
            for q in p + 1..k {
                let apq = m[p * k + q];
                if apq == 0.0 {
                    continue;
                }
                let app = m[p * k + p];
                let aqq = m[q * k + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for r in 0..k {
                    let mrp = m[r * k + p];
                    let mrq = m[r * k + q];
                    m[r * k + p] = c * mrp - s * mrq;
                    m[r * k + q] = s * mrp + c * mrq;
                }
                for r in 0..k {
                    let mpr = m[p * k + r];
                    let mqr = m[q * k + r];
                    m[p * k + r] = c * mpr - s * mqr;
                    m[q * k + r] = s * mpr + c * mqr;
                }
                for r in 0..k {
                    let vrp = v[r * k + p];
                    let vrq = v[r * k + q];
                    v[r * k + p] = c * vrp - s * vrq;
                    v[r * k + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    ((0..k).map(|i| m[i * k + i]).collect(), v)
}

/// Clips negative eigenvalues: returns U·diag(max(λ, 0))·Uᵀ, or `None` when min λ ≥ 0.
fn clip_psd(v: &[f64], k: usize) -> Option<Vec<f64>> {
    let (lambda, u) = sym_eigen(v, k);
    if !lambda.iter().any(|&l| l < 0.0) {
        return None;
    }
    let mut out = vec![0.0; k * k];
    for (e, &l) in lambda.iter().enumerate() {
        if l <= 0.0 {
            continue;
        }
        for i in 0..k {
            let ui = u[i * k + e] * l;
            for j in 0..k {
                out[i * k + j] += ui * u[j * k + e];
            }
        }
    }
    Some(out)
}

/// reghdfe `reghdfe_fix_psd` (Cameron–Gelbach–Miller): if V (k×k, symmetric) has a negative
/// eigenvalue, replace it by U·diag(λ⁺)·Uᵀ; with a reported constant (last row/column), the
/// regressor block is then replaced by the separately clipped original regressor block.
/// Callers pass V on reghdfe's standardized scale. Returns whether the fix was applied.
pub(crate) fn fix_psd(v: &mut [f64], k: usize, report_constant: bool) -> bool {
    for i in 0..k {
        for j in 0..i {
            let s = 0.5 * (v[i * k + j] + v[j * k + i]);
            v[i * k + j] = s;
            v[j * k + i] = s;
        }
    }
    let kb = if report_constant { k - 1 } else { k };
    let backup: Vec<f64> = (0..kb).flat_map(|i| (0..kb).map(move |j| (i, j))).map(|(i, j)| v[i * k + j]).collect();
    let Some(fixed) = clip_psd(v, k) else { return false };
    v.copy_from_slice(&fixed);
    if report_constant {
        let sub = clip_psd(&backup, kb).unwrap_or(backup);
        for i in 0..kb {
            for j in 0..kb {
                v[i * k + j] = sub[i * kb + j];
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invsym_inverts_and_drops_collinear() {
        let x = [[1.0, 2.0, 3.0], [0.5, -1.0, -0.5], [2.0, 0.3, 2.3], [1.0, 1.0, 2.0], [-1.0, 0.0, -1.0]];
        // column 2 = column 0 + column 1
        let k = 3;
        let mut a = vec![0.0; 9];
        for r in &x {
            for i in 0..k {
                for j in 0..k {
                    a[i * k + j] += r[i] * r[j];
                }
            }
        }
        let (inv, dropped) = invsym(&a, k, INVSYM_TOL);
        assert_eq!(dropped, vec![false, false, true]);
        // inverse of the leading 2×2 block
        let det = a[0] * a[4] - a[1] * a[3];
        assert!((inv[0] - a[4] / det).abs() < 1e-12);
        assert!((inv[1] + a[1] / det).abs() < 1e-12);
        assert!(inv[2] == 0.0 && inv[8] == 0.0);
    }

    #[test]
    fn jacobi_eigen_reconstructs() {
        let a = [4.0, 1.0, -2.0, 1.0, 2.0, 0.5, -2.0, 0.5, -3.0];
        let (l, u) = sym_eigen(&a, 3);
        for i in 0..3 {
            for j in 0..3 {
                let r: f64 = (0..3).map(|e| u[i * 3 + e] * l[e] * u[j * 3 + e]).sum();
                assert!((r - a[i * 3 + j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn cluster_meat_matches_bruteforce_for_any_threads() {
        let n = 40_000;
        let codes: Vec<u32> = (0..n).map(|i| ((i * 7919) % 337) as u32).collect();
        let row = |i: usize, a: &mut [f64]| {
            a[0] = (i as f64).sin();
            a[1] = 1.0;
            (i as f64 * 0.37).cos()
        };
        let mut s = vec![[0.0f64; 2]; 337];
        for i in 0..n {
            let mut a = [0.0; 2];
            let e = row(i, &mut a);
            s[codes[i] as usize][0] += e * a[0];
            s[codes[i] as usize][1] += e * a[1];
        }
        let brute: Vec<f64> = (0..4).map(|ij| s.iter().map(|v| v[ij / 2] * v[ij % 2]).sum()).collect();
        let run = |t| par::with_pool(t, |pool| cluster_meat(&codes, 337, 2, pool, row)).unwrap();
        let a = run(1);
        assert_eq!(a, run(5));
        for (x, y) in a.iter().zip(&brute) {
            assert!((x - y).abs() < 1e-9 * y.abs().max(1.0));
        }
        let mut sorted = codes.clone();
        sorted.sort();
        let b = par::with_pool(3, |pool| cluster_meat(&sorted, 337, 2, pool, |i, a| {
            a[0] = 1.0;
            a[1] = i as f64;
            1.0
        }))
        .unwrap();
        assert!(b[0] > 0.0);
    }
}
