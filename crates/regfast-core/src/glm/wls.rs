//! Weighted least squares on residualized row blocks: Givens QR (accurate; ppmlhdfe uses Mata
//! `qrsolve` for these solves) and normal equations (ppmlhdfe `fastsolve`).
//!
//! The QR kernel accumulates the upper-triangular factor of √w·[X̃, ỹ] per fixed chunk of rows and
//! merges chunk factors in chunk order, so results do not depend on the thread count.

use crate::par::{self, CHUNK, Pool};
use crate::vcov::{self, INVSYM_TOL};

/// Rotates `row` (length m) into the upper-triangular m×m factor `r` (row-major).
fn givens_add(r: &mut [f64], m: usize, row: &mut [f64]) {
    for j in 0..m {
        let a = row[j];
        if a == 0.0 {
            continue;
        }
        let d = r[j * m + j];
        let h = d.hypot(a);
        let (c, s) = (d / h, a / h);
        r[j * m + j] = h;
        for k in j + 1..m {
            let rk = r[j * m + k];
            let xk = row[k];
            r[j * m + k] = c * rk + s * xk;
            row[k] = c * xk - s * rk;
        }
    }
}

/// R factor (m×m, upper) of the QR decomposition of the n×m matrix whose row i is
/// √w_i·(rows[i·r + cols[0]], …, rows[i·r + cols[m−1]]).
pub(crate) fn qr_factor(rows: &[f64], r: usize, cols: &[usize], w: Option<&[f64]>, pool: &Pool) -> Vec<f64> {
    let n = rows.len().checked_div(r).unwrap_or(0);
    let m = cols.len();
    let parts = par::map_chunks(pool, n, CHUNK, |_, range| {
        let mut acc = vec![0.0; m * m];
        let mut buf = vec![0.0; m];
        for i in range {
            let sw = w.map_or(1.0, |w| w[i].sqrt());
            let row = &rows[i * r..(i + 1) * r];
            for (b, &c) in buf.iter_mut().zip(cols) {
                *b = sw * row[c];
            }
            givens_add(&mut acc, m, &mut buf);
        }
        acc
    });
    let mut out = vec![0.0; m * m];
    let mut buf = vec![0.0; m];
    for part in parts {
        for j in 0..m {
            buf.copy_from_slice(&part[j * m..(j + 1) * m]);
            givens_add(&mut out, m, &mut buf);
        }
    }
    out
}

/// Least-squares coefficients from the R factor of [X, y] (m = k + 1): solves R_xx b = R_xy by
/// back substitution. Diagonal entries ≤ 1e-13·max|R_ii| are treated as zero (coefficient 0),
/// as Mata `_solveupper` does for rank-deficient systems.
pub(crate) fn qr_solve(rr: &[f64], m: usize) -> Vec<f64> {
    let k = m - 1;
    let dmax = (0..k).map(|j| rr[j * m + j].abs()).fold(0.0f64, f64::max);
    let mut b = vec![0.0; k];
    for j in (0..k).rev() {
        let d = rr[j * m + j];
        if d.abs() <= 1e-13 * dmax || d == 0.0 {
            b[j] = 0.0;
            continue;
        }
        let mut s = rr[j * m + k];
        for l in j + 1..k {
            s -= rr[j * m + l] * b[l];
        }
        b[j] = s / d;
    }
    b
}

/// Coefficients of y on X by QR, for a row block whose column `ycol` is y and `xcols` are X.
pub(crate) fn qr_coefs(rows: &[f64], r: usize, ycol: usize, xcols: &[usize], w: Option<&[f64]>, pool: &Pool) -> Vec<f64> {
    if xcols.is_empty() {
        return Vec::new();
    }
    let cols: Vec<usize> = xcols.iter().copied().chain(std::iter::once(ycol)).collect();
    let rr = qr_factor(rows, r, &cols, w, pool);
    qr_solve(&rr, cols.len())
}

/// Normal-equation coefficients from a Gram matrix `g` (r×r, columns [y, x…]):
/// b = G_xx⁻ G_xy with the sweep generalized inverse.
pub(crate) fn normal_coefs(g: &[f64], r: usize) -> Vec<f64> {
    let k = r - 1;
    if k == 0 {
        return Vec::new();
    }
    let gxx: Vec<f64> = (0..k * k).map(|ab| g[(1 + ab / k) * r + 1 + ab % k]).collect();
    let (inv, _) = vcov::invsym(&gxx, k, INVSYM_TOL);
    (0..k).map(|a| (0..k).map(|b| inv[a * k + b] * g[(1 + b) * r]).sum()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qr_matches_normal_equations_for_any_threads() {
        let n = 40_000;
        let r = 3;
        let mut rows = vec![0.0; n * r];
        let mut w = vec![0.0; n];
        for i in 0..n {
            let a = (i as f64 * 0.37).sin();
            let b = (i as f64 * 0.11).cos();
            rows[i * r + 1] = a;
            rows[i * r + 2] = b;
            rows[i * r] = 2.0 * a - 0.5 * b + 0.01 * (i as f64 * 1.7).sin();
            w[i] = 1.0 + (i % 7) as f64;
        }
        let run = |t| par::with_pool(t, |pool| qr_coefs(&rows, r, 0, &[1, 2], Some(&w), pool)).unwrap();
        let b1 = run(1);
        assert_eq!(b1, run(6));
        let g = par::with_pool(2, |pool| crate::stats::gram(&rows, r, Some(&w), pool).0).unwrap();
        let bn = normal_coefs(&g, r);
        for (a, b) in b1.iter().zip(&bn) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
    }
}
