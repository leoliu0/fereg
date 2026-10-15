//! Variance building blocks with reghdfe 6.13.1 conventions (plan Phase 3.7), shared by OLS,
//! IV (Phase 5) and GLM (Phase 6).
//!
//! Everything here works on small dense K x K matrices (row-major) plus streaming "meat" kernels
//! over the rows. Meat kernels are deterministic for any thread count: rows (or rows sorted by
//! cluster) are split into fixed chunks of [`CHUNK`], or into a data-determined number of row
//! ranges ([`stream_sums`]), and partial sums are combined in order.

use rayon::prelude::*;

use crate::error::RfError;
use crate::factor::{self, Codes};
use crate::par::{self, CHUNK, Pool};

/// Relative pivot tolerance of [`invsym`]: a pivot is zero when it is <= `INVSYM_TOL`*(its
/// original diagonal). Stata does not document the tolerance of Mata `invsym()`; plan sec. 3.2 fixes it.
pub(crate) const INVSYM_TOL: f64 = 1e-13;

/// Mata `invsym(A, 1..k)`: generalized inverse of the symmetric k x k matrix `a` (row-major) by
/// sweeping the pivots in column order. A pivot <= `tol`*(original diagonal) (or <= 0) marks the
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
        // Goodnight sweep on pivot p (yields +A^-1 after sweeping every pivot).
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

/// fixest `cpp_cholesky(XtX, tol)` (lm_related.cpp): Cholesky in column order where column j is
/// excluded when its pivot X_jj - sum_{kept i<j} R_ij^2 is below `tol` (absolute, unscaled).
/// Returns the exclusion flags and the inverse of the kept block (kept order, row-major).
pub(crate) fn chol_collin(a: &[f64], k: usize, tol: f64) -> (Vec<bool>, Vec<f64>) {
    let (r, excluded, _) = chol_pivots(a, k, tol);
    let kept: Vec<usize> = (0..k).filter(|&j| !excluded[j]).collect();
    (excluded, inverse_from_chol(&r, k, &kept))
}

/// Rounding of the Gram matrix (chunked sums) and of its Cholesky pivots, bounded as a
/// perturbation of each column by sqrt(PIVOT_ROUNDING) times its norm.
const PIVOT_ROUNDING: f64 = 1e-11;

/// The exclusions of [`chol_collin`] when perturbing the columns behind the Gram matrix `a` by
/// vectors of norm at most `err[j]` (column j, in the metric of `a`) cannot change them, `None`
/// otherwise. The square root of column j's pivot is the distance of the column from the span of
/// the kept earlier columns, which such perturbations move by at most
/// err_j + sum_i |b_i| err_i (b: the column's coefficients on those columns). The bounds include
/// the rounding of this factorization and of the one on the unperturbed columns: an outcome
/// within rounding of `tol` is never certified.
pub(crate) fn chol_collin_certain(a: &[f64], k: usize, tol: f64, err: &[f64]) -> Option<Vec<bool>> {
    let (r, excluded, pivots) = chol_pivots(a, k, tol);
    let err: Vec<f64> = (0..k).map(|j| err[j] + 2.0 * (PIVOT_ROUNDING * a[j * k + j]).sqrt()).collect();
    let mut b = vec![0.0; k];
    for j in 0..k {
        let kept: Vec<usize> = (0..j).filter(|&i| !excluded[i]).collect();
        for (m, &i) in kept.iter().enumerate().rev() {
            let s: f64 = kept[m + 1..].iter().map(|&l| r[i * k + l] * b[l]).sum();
            b[i] = (r[i * k + j] - s) / r[i * k + i];
        }
        let margin = err[j] + kept.iter().map(|&i| b[i].abs() * err[i]).sum::<f64>();
        let dist = pivots[j].max(0.0).sqrt();
        let stays = if excluded[j] { dist + margin < tol.sqrt() } else { dist - margin >= tol.sqrt() };
        if !stays {
            return None;
        }
    }
    Some(excluded)
}

/// The factorization of [`chol_collin`]: R (upper triangular, kept rows), the exclusion flags
/// and every column's pivot.
fn chol_pivots(a: &[f64], k: usize, tol: f64) -> (Vec<f64>, Vec<bool>, Vec<f64>) {
    let mut r = vec![0.0; k * k];
    let mut excluded = vec![false; k];
    let mut pivots = vec![0.0; k];
    for j in 0..k {
        let mut rjj = a[j * k + j];
        for i in (0..j).filter(|&i| !excluded[i]) {
            rjj -= r[i * k + j] * r[i * k + j];
        }
        pivots[j] = rjj;
        if !(rjj >= tol) {
            excluded[j] = true;
            continue;
        }
        let d = rjj.sqrt();
        r[j * k + j] = d;
        for c in j + 1..k {
            let mut v = a[c * k + j];
            for i in (0..j).filter(|&i| !excluded[i]) {
                v -= r[i * k + c] * r[i * k + j];
            }
            r[j * k + c] = v / d;
        }
    }
    (r, excluded, pivots)
}

/// (R^T R)^-1 = R^-1 R^-T over the `kept` rows/columns of the upper-triangular `r` (k x k).
fn inverse_from_chol(r: &[f64], k: usize, kept: &[usize]) -> Vec<f64> {
    let m = kept.len();
    // R^-1 (upper triangular, m x m) by back substitution, column by column.
    let mut ri = vec![0.0; m * m];
    for c in 0..m {
        ri[c * m + c] = 1.0 / r[kept[c] * k + kept[c]];
        for row in (0..c).rev() {
            let mut s = 0.0;
            for l in row + 1..=c {
                s += r[kept[row] * k + kept[l]] * ri[l * m + c];
            }
            ri[row * m + c] = -s / r[kept[row] * k + kept[row]];
        }
    }
    let mut inv = vec![0.0; m * m];
    for i in 0..m {
        for j in i..m {
            let s: f64 = (j..m).map(|l| ri[i * m + l] * ri[j * m + l]).sum();
            inv[i * m + j] = s;
            inv[j * m + i] = s;
        }
    }
    inv
}

/// reghdfe `reghdfe_extend_b_and_xx`: extends inv(X~'WX~) (k x k) with the constant via the
/// partitioned-inverse formula, using the regression-through-means constant:
/// `[[inv, -inv*xbar], [-xbar'inv, 1/n_c + xbar'inv*xbar]]` ((k+1)x(k+1), row-major).
/// `n_c` is N_obs for none/aw/pw and sum fw for fw (reghdfe `tmp_N`).
pub(crate) fn inv_ext(inv: &[f64], k: usize, means: &[f64], n_c: f64) -> Vec<f64> {
    let k1 = k + 1;
    let mut out = vec![0.0; k1 * k1];
    // side = -xbar'*inv (1xk)
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

/// A*B for row-major square matrices of order k.
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

/// Sandwich D*M*D*scale (D symmetric), symmetrized: the two triangles of the product differ by
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

/// Adds a*a^T*s into the m x m row-major accumulator (upper triangle only).
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

/// Robust meat sum_i m_i*a_i a_i^T (m x m). `row(i, a)` fills the design row a_i (length m) of row i
/// and returns its multiplier m_i (reghdfe: e^2 unweighted, e^2*w for fw, (e*w_v)^2 for aw/pw).
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
/// (ascending row within a group): the stable counting sort of the cell structure, whose result
/// does not depend on the thread count.
pub(crate) fn group_rows(codes: &[u32], g: usize, pool: &Pool) -> Result<Option<Vec<u32>>, RfError> {
    if pool.install(|| codes.par_windows(2).all(|w| w[0] <= w[1])) {
        return Ok(None);
    }
    Ok(Some(crate::structure::scatter_by_code(codes, g, pool)?.1))
}

/// Cluster meat sum_g s_g s_g^T with s_g = sum_{i in g} score_i*a_i (m x m). `row(i, a)` fills a_i and
/// returns score_i (reghdfe: e_i*w_i with w = w_v, fw or 1). `codes` are row-order cluster codes < g.
///
/// Deterministic: positions in grouped order are split into fixed chunks; groups interior to a
/// chunk are accumulated locally, groups crossing chunk boundaries are stitched in chunk order.
pub(crate) fn cluster_meat(
    codes: &[u32],
    g: usize,
    m: usize,
    pool: &Pool,
    row: impl Fn(usize, &mut [f64]) -> f64 + Sync,
) -> Result<Vec<f64>, RfError> {
    let n = codes.len();
    let perm = group_rows(codes, g, pool)?;
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
    Ok(acc_local)
}

/// Bytes of the per-task cluster sums of [`stream_sums`], all tasks together.
const STREAM_BYTES: usize = 32 << 20;
/// Fewest and most tasks of [`stream_sums`]: a subset is streamed only while the sums still
/// allow `STREAM_MIN_TASKS` tasks (each within L2 at 2 MB), otherwise it is grouped by cluster.
const STREAM_MIN_TASKS: usize = 16;
const STREAM_MAX_TASKS: usize = 64;

/// Cluster score sums s_c = sum_{i in c} score_i*a_i (g x m, row-major) of several cluster
/// dimensions (`codes` in row order, `g` levels each) in one pass over the rows. The rows are cut
/// into `tasks` fixed ranges, each summed in row order into its own sums, which are added in
/// task order: the result depends only on the data and `tasks`, never on the thread count.
fn stream_sums(dims: &[(&[u32], usize)], n: usize, m: usize, tasks: usize, pool: &Pool, row: &(impl Fn(usize, &mut [f64]) -> f64 + Sync)) -> Vec<Vec<f64>> {
    let mut offs = Vec::with_capacity(dims.len());
    let mut len = 0usize;
    for &(_, g) in dims {
        offs.push(len);
        len += g * m;
    }
    let parts: Vec<Vec<f64>> = pool.install(|| {
        (0..tasks)
            .into_par_iter()
            .map(|t| {
                let mut acc = vec![0.0; len];
                let mut a = vec![0.0; m];
                for i in t * n / tasks..(t + 1) * n / tasks {
                    let s = row(i, &mut a);
                    for (&(codes, _), &o) in dims.iter().zip(&offs) {
                        let at = o + codes[i] as usize * m;
                        for (c, v) in acc[at..at + m].iter_mut().zip(&a) {
                            *c += s * v;
                        }
                    }
                }
                acc
            })
            .collect()
    });
    let mut parts = parts.into_iter();
    let mut sums = parts.next().unwrap_or_else(|| vec![0.0; len]);
    for p in parts {
        for (x, v) in sums.iter_mut().zip(&p) {
            *x += v;
        }
    }
    dims.iter().zip(&offs).map(|(&(_, g), &o)| sums[o..o + g * m].to_vec()).collect()
}

/// Score sums s_c (g x m, row-major) of one code vector by [`stream_sums`], with the task count
/// of [`multiway_meat`]'s streamed subsets; `None` when the sums do not fit the streaming budget
/// with `STREAM_MIN_TASKS` tasks (callers then group the rows by code).
pub(crate) fn stream_score_sums(codes: &[u32], g: usize, m: usize, pool: &Pool, row: &(impl Fn(usize, &mut [f64]) -> f64 + Sync)) -> Option<Vec<f64>> {
    let len = g.checked_mul(m)?;
    if len.checked_mul(8 * STREAM_MIN_TASKS)? > STREAM_BYTES {
        return None;
    }
    let n = codes.len();
    let tasks = (STREAM_BYTES / (8 * len).max(1)).clamp(1, STREAM_MAX_TASKS).min(n.div_ceil(CHUNK).max(1));
    stream_sums(&[(codes, g)], n, m, tasks, pool, row).pop()
}

/// sum_c s_c s_c^T (m x m) of cluster sums `sums` (g x m, row-major), clusters in order.
fn meat_of_sums(sums: &[f64], m: usize) -> Vec<f64> {
    let mut meat = vec![0.0; m * m];
    for s in sums.chunks_exact(m.max(1)) {
        add_outer_upper(&mut meat, s, 1.0, m);
    }
    mirror_upper(&mut meat, m);
    meat
}

/// One cluster dimension: kept-order codes with `g` levels.
pub(crate) struct ClusterCodes<'a> {
    pub codes: &'a [u32],
    pub g: usize,
}

/// Multiway cluster meat (reghdfe `reghdfe_vce_cluster`): sum over non-empty subsets S of the
/// dimensions, in order of |S| and then lexicographically, of (-1)^{|S|+1}*meat(intersect S), where the
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
/// clusters of the subset's intersection (fixest `ssc(G.df = "conventional")`: G/(G-1)).
pub(crate) fn multiway_meat_scaled(
    dims: &[ClusterCodes],
    row_order: &[u32],
    m: usize,
    pool: &Pool,
    scale: impl Fn(usize) -> f64,
    row: impl Fn(usize, &mut [f64]) -> f64 + Sync,
) -> Result<Vec<f64>, RfError> {
    multiway_meat_with(dims, row_order, m, pool, scale, row, STREAM_BYTES)
}

/// [`multiway_meat_scaled`] with `stream_bytes` for the sums of the streamed subsets: subsets (in
/// order) join the one pass of [`stream_sums`] while all their sums fit `STREAM_MIN_TASKS` tasks
/// within it; the others are grouped by cluster ([`cluster_meat`]).
fn multiway_meat_with(
    dims: &[ClusterCodes],
    row_order: &[u32],
    m: usize,
    pool: &Pool,
    scale: impl Fn(usize) -> f64,
    row: impl Fn(usize, &mut [f64]) -> f64 + Sync,
    stream_bytes: usize,
) -> Result<Vec<f64>, RfError> {
    let q = dims.len();
    let n = row_order.len();
    let owned: Vec<Codes> = if q > 1 { dims.iter().map(|d| Codes { codes: d.codes.to_vec(), n: d.g as u32 }).collect() } else { Vec::new() };
    // Every subset with its sign, row-order codes and number of clusters.
    let mut subs: Vec<(f64, Vec<u32>, usize)> = Vec::new();
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
            subs.push((sign, to_row_order(kept_codes, row_order, pool)?, g));
        }
    }
    drop(owned);
    let mut len = 0usize;
    let streamed: Vec<bool> = subs
        .iter()
        .map(|&(_, _, g)| {
            let fits = (len + g * m) * 8 * STREAM_MIN_TASKS <= stream_bytes;
            if fits {
                len += g * m;
            }
            fits
        })
        .collect();
    let tasks = (stream_bytes / (8 * len).max(1)).clamp(1, STREAM_MAX_TASKS).min(n.div_ceil(CHUNK).max(1));
    let stream_dims: Vec<(&[u32], usize)> = subs.iter().zip(&streamed).filter(|&(_, &s)| s).map(|((_, c, g), _)| (c.as_slice(), *g)).collect();
    let mut sums = if stream_dims.is_empty() { Vec::new() } else { stream_sums(&stream_dims, n, m, tasks, pool, &row) }.into_iter();
    let mut meat = vec![0.0; m * m];
    for ((sign, codes, g), s) in subs.iter().zip(&streamed) {
        let part = if *s { meat_of_sums(&sums.next().expect("one sum per streamed subset"), m) } else { cluster_meat(codes, *g, m, pool, &row)? };
        let f = sign * scale(*g);
        for (a, v) in meat.iter_mut().zip(&part) {
            *a += f * v;
        }
    }
    Ok(meat)
}

/// Kept-order values -> row order (`out[k] = v[row_order[k]]`).
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
/// accuracy; K is small). Returns (eigenvalues, eigenvectors as columns of a row-major k x k matrix).
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

/// Clips negative eigenvalues: returns U*diag(max(lambda, 0))*U^T, or `None` when min lambda >= 0.
/// The result is exactly symmetric (the lower triangle is computed and mirrored, as Mata's
/// `quadcross(U', lambda, U')` in reghdfe): Stata rejects a V whose triangles differ in the last
/// bit as "nonsymmetric or highly singular" and posts zero standard errors.
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
            for j in 0..=i {
                out[i * k + j] += ui * u[j * k + e];
            }
        }
    }
    make_symmetric(&mut out, k);
    Some(out)
}

/// Mata `_makesymmetric`: copies the lower triangle of the k x k row-major matrix to the upper one.
pub(crate) fn make_symmetric(v: &mut [f64], k: usize) {
    for i in 0..k {
        for j in 0..i {
            v[j * k + i] = v[i * k + j];
        }
    }
}

/// reghdfe `reghdfe_fix_psd` (Cameron-Gelbach-Miller): if V (k x k, symmetric) has a negative
/// eigenvalue, replace it by U*diag(lambda^+)*U^T; with a reported constant (last row/column), the
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
        // inverse of the leading 2x2 block
        let det = a[0] * a[4] - a[1] * a[3];
        assert!((inv[0] - a[4] / det).abs() < 1e-12);
        assert!((inv[1] + a[1] / det).abs() < 1e-12);
        assert!(inv[2] == 0.0 && inv[8] == 0.0);
    }

    fn cross(cols: &[Vec<f64>]) -> Vec<f64> {
        let k = cols.len();
        (0..k * k).map(|ab| cols[ab / k].iter().zip(&cols[ab % k]).map(|(a, b)| a * b).sum()).collect()
    }

    #[test]
    fn certified_collinearity_is_that_of_the_unperturbed_columns() {
        // Sets of columns (independent, near-copies, exact combinations, with heavy cancellation
        // when they involve a near-copy, tiny ones just above the threshold) perturbed by random
        // vectors of random norm up to `scale` times the column's. A certificate for twice that
        // bound always equals the outcome on the exact columns, and well-separated sets (no
        // combination) are certified.
        let (n, tol, k) = (300, 1e-10, 5);
        let mut rng = crate::rng::SplitMix64::new(5);
        let mut normal = move || {
            let (u, v) = (rng.next_f64(), rng.next_f64());
            (-2.0 * (1.0 - u).ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        };
        let norm = |c: &[f64]| c.iter().map(|v| v * v).sum::<f64>().sqrt();
        for trial in 0..768 {
            let (pattern, scale) = (trial / 3, [1e-12, 1e-7, 1e-4][trial % 3]);
            let mut cols: Vec<Vec<f64>> = vec![(0..n).map(|_| normal()).collect()];
            let mut combo = false;
            for j in 0..k - 1 {
                let col: Vec<f64> = match (pattern >> (2 * j)) & 3 {
                    0 => (0..n).map(|_| normal()).collect(),
                    1 => cols[0].iter().map(|v| v + 1e-3 * normal()).collect(),
                    2 => {
                        // (A copy of column 0 when it is the only one.)
                        combo = true;
                        let last = cols.len() - 1;
                        (0..n).map(|i| 1000.0 * cols[0][i] - 999.0 * cols[last][i]).collect()
                    }
                    _ => (0..n).map(|_| 1e-6 * normal()).collect(),
                };
                cols.push(col);
            }
            let truth = chol_collin(&cross(&cols), k, tol).0;
            let pert: Vec<Vec<f64>> = cols
                .iter()
                .map(|c| {
                    let e: Vec<f64> = (0..n).map(|_| normal()).collect();
                    let f = scale * norm(c) * (0.5 * (normal().tanh() + 1.0)) / norm(&e);
                    c.iter().zip(&e).map(|(v, d)| v + f * d).collect()
                })
                .collect();
            let err: Vec<f64> = pert.iter().map(|c| 2.0 * scale * norm(c)).collect();
            match chol_collin_certain(&cross(&pert), k, tol, &err) {
                Some(e) => assert_eq!(e, truth, "trial {trial}"),
                None => assert!(combo, "trial {trial}: no combination, not certified at {scale}"),
            }
        }
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
    fn fix_psd_returns_an_exactly_symmetric_psd_matrix() {
        // Indefinite two-way clustered V's, with and without a reported constant: Stata rejects a
        // V whose two triangles differ in the last bit ("nonsymmetric or highly singular").
        let mut rng = crate::rng::SplitMix64::new(20261015);
        for trial in 0..400 {
            let k = 2 + trial % 5;
            let a: Vec<f64> = (0..k * k).map(|_| rng.next_f64() - 0.5).collect();
            let d: Vec<f64> = (0..k).map(|e| if e == 0 { -0.3 * rng.next_f64() } else { rng.next_f64() }).collect();
            let mut v: Vec<f64> = (0..k * k).map(|ij| (0..k).map(|e| a[(ij / k) * k + e] * d[e] * a[(ij % k) * k + e]).sum()).collect();
            make_symmetric(&mut v, k);
            let report_constant = trial % 2 == 1;
            assert!(fix_psd(&mut v, k, report_constant), "trial {trial}: V is indefinite");
            for i in 0..k {
                for j in 0..i {
                    assert_eq!(v[i * k + j].to_bits(), v[j * k + i].to_bits(), "trial {trial}: V[{i},{j}] != V[{j},{i}]");
                }
            }
            let scale = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let kb = if report_constant { k - 1 } else { k };
            let block: Vec<f64> = (0..kb * kb).map(|ij| v[(ij / kb) * k + ij % kb]).collect();
            let (lambda, _) = sym_eigen(&block, kb);
            assert!(lambda.iter().all(|&l| l >= -1e-12 * scale), "trial {trial}: regressor block not psd: {lambda:?}");
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
        let run = |t| par::with_pool(t, |pool| cluster_meat(&codes, 337, 2, pool, row)).unwrap().unwrap();
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
        .unwrap()
        .unwrap();
        assert!(b[0] > 0.0);
    }

    #[test]
    fn multiway_meat_streamed_grouped_and_mixed_match_bruteforce_for_any_threads() {
        // Two dimensions in kept order, rows in reversed kept order: every split between the
        // streamed pass and the grouped path gives the inclusion-exclusion meat of the cluster
        // sums, bit-identical across thread counts.
        let n = 50_000;
        let (g0, g1, m) = (211usize, 97usize, 3usize);
        let d0: Vec<u32> = (0..n).map(|i| ((i * 7919) % g0) as u32).collect();
        let d1: Vec<u32> = (0..n).map(|i| ((i * 104_729) % g1) as u32).collect();
        let row_order: Vec<u32> = (0..n as u32).rev().collect();
        let row = |i: usize, a: &mut [f64]| {
            a[0] = (i as f64).sin();
            a[1] = 1.0;
            a[2] = (i as f64 * 0.01).cos();
            (i as f64 * 0.37).cos()
        };
        let brute_meat = |code: &dyn Fn(usize) -> usize, g: usize| -> Vec<f64> {
            let mut sums = vec![0.0; g * m];
            let mut a = vec![0.0; m];
            for i in 0..n {
                let e = row(i, &mut a);
                let c = code(row_order[i] as usize);
                for j in 0..m {
                    sums[c * m + j] += e * a[j];
                }
            }
            (0..m * m).map(|ij| sums.chunks_exact(m).map(|s| s[ij / m] * s[ij % m]).sum()).collect()
        };
        let (b0, b1) = (brute_meat(&|k| d0[k] as usize, g0), brute_meat(&|k| d1[k] as usize, g1));
        let b01 = brute_meat(&|k| d0[k] as usize * g1 + d1[k] as usize, g0 * g1);
        let brute: Vec<f64> = (0..m * m).map(|j| b0[j] + b1[j] - b01[j]).collect();
        let dims = [ClusterCodes { codes: &d0, g: g0 }, ClusterCodes { codes: &d1, g: g1 }];
        // Every subset streamed, only the two dimensions (not the intersection), none.
        for budget in [STREAM_BYTES, (g0 + g1) * m * 8 * STREAM_MIN_TASKS, 0] {
            let run = |t| par::with_pool(t, |pool| multiway_meat_with(&dims, &row_order, m, pool, |_| 1.0, row, budget)).unwrap().unwrap();
            let a = run(1);
            assert_eq!(a, run(4), "budget {budget}");
            for (x, y) in a.iter().zip(&brute) {
                assert!((x - y).abs() < 1e-9 * y.abs().max(1.0), "budget {budget}: {x} vs {y}");
            }
        }
    }
}
