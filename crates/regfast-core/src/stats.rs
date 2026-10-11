//! Streaming column statistics and fit statistics (plan Phase 3.1 and 3.8, reghdfe 6.13.1).
//!
//! All row passes are chunked by [`CHUNK`] with chunk-ordered combination, so results do not
//! depend on the thread count. Blocks are N×R row-major.

use crate::par::{self, CHUNK, Pool};

/// Pass-A statistics of the raw (un-residualized) columns.
#[derive(Clone, Debug)]
pub(crate) struct ColStats {
    /// Unweighted raw Σx² (reghdfe `norm2`, the stage-1 collinearity denominator).
    pub sumsq: Vec<f64>,
    /// reghdfe `compute_stdevs`: unweighted sample sd (denominator n−1), floored at 1e-3.
    pub sd: Vec<f64>,
    /// Weighted means with the raw weights (reghdfe `mean(data, weights)`).
    pub wmean: Vec<f64>,
    /// Σw(x − x̄_w)² with the raw weights.
    pub wss: Vec<f64>,
    /// Σw·x² with the raw weights.
    pub wss_unc: Vec<f64>,
}

#[derive(Clone)]
struct Part {
    n: f64,
    mean: Vec<f64>,
    m2: Vec<f64>,
    sumsq: Vec<f64>,
    w: f64,
    wmean: Vec<f64>,
    wm2: Vec<f64>,
    wsumsq: Vec<f64>,
}

/// Fused Pass A over `rows` (N×r, row-major) with optional raw weights (same row order).
/// Within a chunk means and centered sums are two-pass; chunks are merged with Chan's update in
/// chunk order.
pub(crate) fn col_stats(rows: &[f64], r: usize, w: Option<&[f64]>, pool: &Pool) -> ColStats {
    let n = rows.len().checked_div(r).unwrap_or(0);
    let parts: Vec<Part> = par::map_chunks(pool, n, CHUNK, |_, range| {
        let cn = range.len() as f64;
        let mut p = Part {
            n: cn,
            mean: vec![0.0; r],
            m2: vec![0.0; r],
            sumsq: vec![0.0; r],
            w: 0.0,
            wmean: vec![0.0; r],
            wm2: vec![0.0; r],
            wsumsq: vec![0.0; r],
        };
        let mut wsx = vec![0.0; r];
        for i in range.clone() {
            let wi = w.map_or(1.0, |w| w[i]);
            p.w += wi;
            let row = &rows[i * r..(i + 1) * r];
            for j in 0..r {
                let x = row[j];
                p.mean[j] += x;
                p.sumsq[j] += x * x;
                wsx[j] += wi * x;
                p.wsumsq[j] += wi * x * x;
            }
        }
        for j in 0..r {
            p.mean[j] /= cn;
            p.wmean[j] = wsx[j] / p.w;
        }
        for i in range {
            let wi = w.map_or(1.0, |w| w[i]);
            let row = &rows[i * r..(i + 1) * r];
            for j in 0..r {
                let d = row[j] - p.mean[j];
                p.m2[j] += d * d;
                let dw = row[j] - p.wmean[j];
                p.wm2[j] += wi * dw * dw;
            }
        }
        p
    });
    let mut it = parts.into_iter();
    let mut acc = it.next().unwrap_or(Part {
        n: 0.0,
        mean: vec![0.0; r],
        m2: vec![0.0; r],
        sumsq: vec![0.0; r],
        w: 0.0,
        wmean: vec![0.0; r],
        wm2: vec![0.0; r],
        wsumsq: vec![0.0; r],
    });
    for p in it {
        let n = acc.n + p.n;
        let w = acc.w + p.w;
        for j in 0..r {
            let d = p.mean[j] - acc.mean[j];
            acc.mean[j] += d * p.n / n;
            acc.m2[j] += p.m2[j] + d * d * acc.n * p.n / n;
            acc.sumsq[j] += p.sumsq[j];
            let dw = p.wmean[j] - acc.wmean[j];
            acc.wmean[j] += dw * p.w / w;
            acc.wm2[j] += p.wm2[j] + dw * dw * acc.w * p.w / w;
            acc.wsumsq[j] += p.wsumsq[j];
        }
        acc.n = n;
        acc.w = w;
    }
    let sd = acc.m2.iter().map(|m2| (m2 / (acc.n - 1.0)).sqrt().max(1e-3)).collect();
    ColStats { sumsq: acc.sumsq, sd, wmean: acc.wmean, wss: acc.wm2, wss_unc: acc.wsumsq }
}

/// reghdfe `compute_stdevs` for one variable: unweighted sample sd floored at 1e-3.
pub(crate) fn stdev(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let m2: f64 = x.iter().map(|v| (v - mean) * (v - mean)).sum();
    (m2 / (n - 1.0)).sqrt().max(1e-3)
}

/// Pass B: weighted Gram G = Σ w_i z_i z_iᵀ (r×r) and unweighted Σz² per column.
pub(crate) fn gram(rows: &[f64], r: usize, w: Option<&[f64]>, pool: &Pool) -> (Vec<f64>, Vec<f64>) {
    let n = rows.len().checked_div(r).unwrap_or(0);
    let acc = par::sum_chunks(pool, n, CHUNK, r * r + r, |range, acc| {
        let (g, ss) = acc.split_at_mut(r * r);
        for i in range {
            let wi = w.map_or(1.0, |w| w[i]);
            let row = &rows[i * r..(i + 1) * r];
            for a in 0..r {
                let za = row[a];
                ss[a] += za * za;
                let wa = wi * za;
                let grow = &mut g[a * r..(a + 1) * r];
                for b in a..r {
                    grow[b] += wa * row[b];
                }
            }
        }
    });
    let mut g = acc[..r * r].to_vec();
    for a in 0..r {
        for b in 0..a {
            g[a * r + b] = g[b * r + a];
        }
    }
    (g, acc[r * r..].to_vec())
}

/// reghdfe fit statistics (`reghdfe_solve_ols`, "Results" block).
#[derive(Clone, Copy, Debug)]
pub(crate) struct FitStats {
    pub r2: f64,
    pub r2_a: f64,
    pub r2_within: f64,
    pub r2_a_within: f64,
    pub rmse: f64,
    pub ll: f64,
    pub ll_0: f64,
    pub mss: f64,
}

/// `n` = N (Σfw for fweights), `used_df_r` = N − df_a − df_m − df_a_nested.
pub(crate) fn fit_stats(n: f64, rss: f64, tss: f64, tss_within: f64, used_df_r: f64, rank: f64, has_intercept: bool) -> FitStats {
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    let hi = if has_intercept { 1.0 } else { 0.0 };
    FitStats {
        r2: 1.0 - rss / tss,
        r2_a: 1.0 - (rss / used_df_r) / (tss / (n - hi)),
        r2_within: 1.0 - rss / tss_within,
        r2_a_within: 1.0 - (rss / used_df_r) / (tss_within / (used_df_r + rank)),
        rmse: if used_df_r != 0.0 { (rss / used_df_r).sqrt() } else { rss.sqrt() },
        ll: -0.5 * n * (1.0 + ln2pi + (rss / n).ln()),
        ll_0: -0.5 * n * (1.0 + ln2pi + (tss_within / n).ln()),
        mss: tss - rss,
    }
}
