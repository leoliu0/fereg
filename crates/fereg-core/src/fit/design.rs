//! Regressions on one residualized row block. Every regressor, target and residual of a linear
//! model (OLS, both IV stages, the auxiliary F, Wu-Hausman and Sargan regressions) is a linear
//! combination ("combo", one coefficient per block column) of the R columns of the block, so all
//! of them come from the block's weighted Gram matrix G = sum w*r r' and one further pass over the
//! rows evaluates every residual sum of squares at once.

use rayon::prelude::*;

use crate::error::RfError;
use crate::par::{self, CHUNK, Pool};
use crate::vcov::chol_collin;

/// Column `c` of an R-column block as a combo.
pub(crate) fn unit(r: usize, c: usize) -> Vec<f64> {
    let mut v = vec![0.0; r];
    v[c] = 1.0;
    v
}

/// a'*G*b for combos a, b over the R x R Gram G.
pub(crate) fn cross(g: &[f64], r: usize, a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, &ai) in a.iter().enumerate().filter(|(_, v)| **v != 0.0) {
        let row = &g[i * r..(i + 1) * r];
        s += ai * b.iter().zip(row).filter(|(bj, _)| **bj != 0.0).map(|(bj, gij)| bj * gij).sum::<f64>();
    }
    s
}

/// Moore-Penrose inverse of a symmetric k x k matrix (numpy `pinv`: eigenvalues below 1e-15 x the
/// largest magnitude count as zero).
pub(crate) fn pinv_sym(a: &[f64], k: usize) -> Vec<f64> {
    let (lambda, u) = crate::vcov::sym_eigen(a, k);
    let cut = 1e-15 * lambda.iter().fold(0.0f64, |m, l| m.max(l.abs()));
    let mut inv = vec![0.0; k * k];
    for (e, &l) in lambda.iter().enumerate().filter(|(_, l)| l.abs() > cut) {
        for i in 0..k {
            let ui = u[i * k + e] / l;
            for j in 0..k {
                inv[i * k + j] += ui * u[j * k + e];
            }
        }
    }
    inv
}

/// A least-squares fit of a target combo on design combos.
pub(crate) struct Reg {
    /// Per design column: removed for collinearity.
    pub excluded: Vec<bool>,
    /// Indices of the kept design columns.
    pub kept: Vec<usize>,
    /// (D'WD)^-1 over the kept columns (kept order, row-major).
    pub inv: Vec<f64>,
    /// Coefficients of the kept columns.
    pub coef: Vec<f64>,
    /// Residual combo: target - sum coef*design.
    pub resid: Vec<f64>,
}

/// Regresses `target` on `design` with fixest's collinearity rule on the weighted Gram `g`.
/// `g_collin` (pyfixest with weights: the unweighted Gram) decides the exclusions instead; the
/// inverse is then that of the weighted kept block.
pub(crate) fn regress(g: &[f64], g_collin: Option<&[f64]>, r: usize, target: &[f64], design: &[Vec<f64>], tol: f64) -> Result<Reg, RfError> {
    let k = design.len();
    let block = |g: &[f64], cols: &[usize]| -> Vec<f64> { cols.iter().flat_map(|&a| cols.iter().map(move |&b| cross(g, r, &design[a], &design[b]))).collect() };
    let all: Vec<usize> = (0..k).collect();
    let (excluded, inv) = match g_collin {
        Some(gc) => {
            let (excluded, _) = chol_collin(&block(gc, &all), k, tol);
            let kept: Vec<usize> = all.iter().copied().filter(|&j| !excluded[j]).collect();
            let (singular, inv) = chol_collin(&block(g, &kept), kept.len(), f64::MIN_POSITIVE);
            if singular.contains(&true) {
                return Err(RfError::InvalidInput("the weighted cross-product of the regressors is singular".into()));
            }
            (excluded, inv)
        }
        None => chol_collin(&block(g, &all), k, tol),
    };
    let kept: Vec<usize> = all.into_iter().filter(|&j| !excluded[j]).collect();
    let m = kept.len();
    let xty: Vec<f64> = kept.iter().map(|&a| cross(g, r, &design[a], target)).collect();
    let coef: Vec<f64> = (0..m).map(|a| (0..m).map(|b| inv[a * m + b] * xty[b]).sum()).collect();
    let mut resid = target.to_vec();
    for (&j, &c) in kept.iter().zip(&coef) {
        resid.iter_mut().zip(&design[j]).for_each(|(v, d)| *v -= c * d);
    }
    Ok(Reg { excluded, kept, inv, coef, resid })
}

/// Nonzero entries of a combo (the per-row evaluation skips the zeros).
pub(crate) fn sparse(a: &[f64]) -> Vec<(usize, f64)> {
    a.iter().enumerate().filter(|(_, v)| **v != 0.0).map(|(j, &v)| (j, v)).collect()
}

/// row*a for a sparse combo.
#[inline]
pub(crate) fn eval(row: &[f64], a: &[(usize, f64)]) -> f64 {
    a.iter().map(|&(j, v)| v * row[j]).sum()
}

/// Sums of one residual pass.
pub(crate) struct PassSums {
    /// sum w*v^2 per combo.
    pub wss: Vec<f64>,
    /// sum v^2 per combo (unweighted).
    pub uss: Vec<f64>,
    /// sum (v - vbar)^2 (unweighted) of the `centered` combo.
    pub centered: f64,
}

/// Chunk partials: weighted and unweighted sums of squares per combo, plus count, mean and M2 of
/// the centered combo (merged with Chan's formula in chunk order).
struct Part {
    wss: Vec<f64>,
    uss: Vec<f64>,
    count: f64,
    mean: f64,
    m2: f64,
}

/// Output buffer written at scattered indices from parallel chunks.
#[derive(Clone, Copy)]
struct Scatter(*mut f64);
// SAFETY: every index is written by exactly one chunk (see `residual_pass`).
unsafe impl Send for Scatter {}
unsafe impl Sync for Scatter {}

/// Chunk partials of rows `row0..` (`rows` holds just the chunk); the values of each stored
/// combo `(c, buf)` go to `buf[order[row]]`.
#[allow(clippy::too_many_arguments)]
fn chunk_part(rows: &[f64], r: usize, row0: usize, w: Option<&[f64]>, combos: &[Vec<(usize, f64)>], centered: Option<usize>, out: &[(usize, Scatter)], order: &[u32]) -> Part {
    let nr = rows.len() / r;
    let mut p = Part { wss: vec![0.0; combos.len()], uss: vec![0.0; combos.len()], count: nr as f64, mean: 0.0, m2: 0.0 };
    let mut sum_c = 0.0;
    for (o, row) in rows.chunks_exact(r).enumerate() {
        let wi = w.map_or(1.0, |w| w[row0 + o]);
        for (c, a) in combos.iter().enumerate() {
            let v = eval(row, a);
            p.wss[c] += wi * v * v;
            p.uss[c] += v * v;
            if centered == Some(c) {
                sum_c += v;
            }
            for (_, buf) in out.iter().filter(|(s, _)| *s == c) {
                // SAFETY: `order` is a permutation of the rows and the buffers have n entries,
                // so this chunk is the only writer of this index.
                unsafe { *buf.0.add(order[row0 + o] as usize) = v };
            }
        }
    }
    if let Some(c) = centered {
        p.mean = sum_c / p.count;
        p.m2 = rows.chunks_exact(r).map(|row| (eval(row, &combos[c]) - p.mean).powi(2)).sum();
    }
    p
}

/// One pass over the row block (row order): sum w*v^2 and sum v^2 for every combo v, the centered
/// unweighted sum of squares of combo `centered`, and the values of the `store` combos (combo
/// index, output buffer of n entries) written in kept order (row k goes to `row_order[k]`).
/// Fixed chunks, combined in chunk order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_pass(
    rows: &[f64],
    r: usize,
    w: Option<&[f64]>,
    combos: &[Vec<f64>],
    centered: Option<usize>,
    store: Vec<(usize, &mut [f64])>,
    row_order: &[u32],
    pool: &Pool,
) -> PassSums {
    let sp: Vec<Vec<(usize, f64)>> = combos.iter().map(|c| sparse(c)).collect();
    let n = rows.len() / r;
    assert!(row_order.len() == n && store.iter().all(|(_, b)| b.len() == n), "residual_pass: output sizes");
    let out: Vec<(usize, Scatter)> = store.into_iter().map(|(c, b)| (c, Scatter(b.as_mut_ptr()))).collect();
    let parts: Vec<Part> = pool.install(|| {
        (0..par::n_chunks(n, CHUNK))
            .into_par_iter()
            .map(|c| {
                let range = par::chunk_range(c, n, CHUNK);
                chunk_part(&rows[range.start * r..range.end * r], r, range.start, w, &sp, centered, &out, row_order)
            })
            .collect()
    });
    let mut wss = vec![0.0; combos.len()];
    let mut uss = vec![0.0; combos.len()];
    let (mut count, mut mean, mut m2) = (0.0, 0.0, 0.0);
    for p in parts {
        wss.iter_mut().zip(&p.wss).for_each(|(a, b)| *a += b);
        uss.iter_mut().zip(&p.uss).for_each(|(a, b)| *a += b);
        let tot = count + p.count;
        let d = p.mean - mean;
        m2 += p.m2 + d * d * count * p.count / tot;
        mean += d * p.count / tot;
        count = tot;
    }
    PassSums { wss, uss, centered: m2 }
}
