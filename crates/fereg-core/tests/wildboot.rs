//! Wild bootstrap integration tests covering OLS (WCR/WCU, MNW 2023 11/13/31/33,
//! Hetero HC1/HC2/HC3), GLM score (Poisson PPML & Logit), and IV/2SLS (WRE & Score).
//!
//! Includes an independent explicit-dummy dense oracle in pure `faer`, CI-inversion
//! p-value boundary checks, full enumeration, weight/p-value variants, multi-thread
//! bit-exactness, and `fereg_core::fit::wildboot` (`Fixest` / `Pyfixest` flavors).

use std::collections::HashMap;

use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use fereg_core::fit::{
    self, CoefSource, FactorCol, FeTerm, FitInput, FitSpec, Flavor, GDf, KFixef, Model, Ssc, TDf, VcovKind, WeightsType, WildBootOptions,
};
use fereg_core::rng::SplitMix64;
use fereg_core::wildboot::{BootMethod, BootType, BootWeights, Hypothesis, PvalType, StatType, WildBootResult, WildBootSpec};
use fereg_core::{
    AbsorbTerm, AbsorberKind, Ctx, Family, FactorValues, GlmCollinear, GlmOptions, Guess, IvOptions, Layout, Prep, Separation, Session, Spec, Vce,
    WeightKind,
};

// ---------------------------------------------------------------------------------------------
// Small matrix & RNG helpers

fn mat_from(r: usize, c: usize, mut f: impl FnMut(usize, usize) -> f64) -> Mat<f64> {
    Mat::from_fn(r, c, |i, j| f(i, j))
}

fn inverse(a: &Mat<f64>) -> Mat<f64> {
    let k = a.nrows();
    a.partial_piv_lu().solve(Mat::<f64>::identity(k, k))
}

fn next_normal(rng: &mut SplitMix64) -> f64 {
    let u1 = rng.next_f64().max(1e-300);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Compact 0..g codes sorted in ascending numeric value order (matching `factorize_col`).
fn compact_sorted_codes(v: &[f64]) -> (Vec<u32>, usize) {
    let mut uniq: Vec<f64> = v.iter().copied().collect();
    uniq.sort_by(f64::total_cmp);
    uniq.dedup_by(|a, b| a.to_bits() == b.to_bits());
    let mut map = HashMap::<u64, u32>::with_capacity(uniq.len());
    for (id, &val) in uniq.iter().enumerate() {
        map.insert(val.to_bits(), id as u32);
    }
    let codes = v.iter().map(|val| map[&val.to_bits()]).collect();
    (codes, uniq.len())
}

/// Full-column-rank orthonormal-like basis (eigen positive subspace) for explicit FE dummies D,
/// plus projector `M_D` under diagonal weight vector `wv`.
struct DenseAbsorber {
    n: usize,
    d_cols: Vec<Vec<f64>>,
    d_basis: Vec<Vec<f64>>,
    rank: usize,
}

impl DenseAbsorber {
    fn from_session(s: &Session) -> Self {
        let n = s.kept_index().len();
        let terms = s.terms();
        let mut d_cols: Vec<Vec<f64>> = Vec::new();
        for t in terms {
            if t.intercept {
                for g in 0..t.g {
                    d_cols.push(t.codes.iter().map(|&c| if c as usize == g { 1.0 } else { 0.0 }).collect());
                }
            }
            for x in &t.slopes {
                for g in 0..t.g {
                    d_cols.push(t.codes.iter().zip(x).map(|(&c, &v)| if c as usize == g { v } else { 0.0 }).collect());
                }
            }
        }
        let d_basis = independent_columns(&d_cols, n);
        let rank = d_basis.len();
        DenseAbsorber { n, d_cols, d_basis, rank }
    }

    /// Weighted projection operator `(I - D (D' W D)^+ D' W)` via positive eigenpairs of `D' W D`.
    fn project_vec(&self, v: &[f64], wv: &[f64]) -> Vec<f64> {
        let p = self.d_cols.len();
        if p == 0 {
            return v.to_vec();
        }
        let dwd = mat_from(p, p, |a, b| (0..self.n).map(|i| wv[i] * self.d_cols[a][i] * self.d_cols[b][i]).sum());
        let evd = dwd.self_adjoint_eigen(Side::Lower).unwrap();
        let (u, s) = (evd.U(), evd.S().column_vector());
        let smax = (0..p).map(|i| s[i].abs()).fold(0.0, f64::max);
        let dwv: Vec<f64> = (0..p).map(|a| (0..self.n).map(|i| wv[i] * self.d_cols[a][i] * v[i]).sum()).collect();
        let mut alpha = vec![0.0; p];
        for e in 0..p {
            if s[e] > 1e-11 * smax {
                let utb: f64 = (0..p).map(|a| u[(a, e)] * dwv[a]).sum();
                let coef = utb / s[e];
                for a in 0..p {
                    alpha[a] += u[(a, e)] * coef;
                }
            }
        }
        (0..self.n)
            .map(|i| v[i] - (0..p).map(|a| self.d_cols[a][i] * alpha[a]).sum::<f64>())
            .collect()
    }
}

fn independent_columns(cols: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    independent_columns_weighted(cols, &vec![1.0; n])
}

fn independent_columns_weighted(cols: &[Vec<f64>], wv: &[f64]) -> Vec<Vec<f64>> {
    if cols.is_empty() {
        return Vec::new();
    }
    let n = wv.len();
    let p = cols.len();
    let gram = mat_from(p, p, |a, b| (0..n).map(|i| wv[i] * cols[a][i] * cols[b][i]).sum());
    let evd = gram.self_adjoint_eigen(Side::Lower).unwrap();
    let (u, s) = (evd.U(), evd.S().column_vector());
    let smax = (0..p).map(|i| s[i].abs()).fold(0.0, f64::max);
    let mut basis = Vec::new();
    for e in 0..p {
        if s[e] > 1e-10 * smax {
            let norm = s[e].sqrt();
            let col: Vec<f64> = (0..n)
                .map(|i| (0..p).map(|a| cols[a][i] * u[(a, e)]).sum::<f64>() / norm)
                .collect();
            basis.push(col);
        }
    }
    basis
}

// ---------------------------------------------------------------------------------------------
// Bootstrap weight draws matching SplitMix64 streams and full enumeration in `weights.rs`

const WEBB_PTS: [f64; 6] = [
    -1.224_744_871_391_589,
    -1.0,
    -0.707_106_781_186_547_6,
    0.707_106_781_186_547_6,
    1.0,
    1.224_744_871_391_589,
];

fn fill_oracle_rng(weights: BootWeights, rng: &mut SplitMix64, out: &mut [f64]) {
    match weights {
        BootWeights::Rademacher => {
            let mut bits = 0u64;
            let mut rem = 0u32;
            for v in out.iter_mut() {
                if rem == 0 {
                    bits = rng.next_u64();
                    rem = 64;
                }
                *v = if bits & 1 == 0 { 1.0 } else { -1.0 };
                bits >>= 1;
                rem -= 1;
            }
        }
        BootWeights::Webb => {
            for v in out.iter_mut() {
                let idx = (((rng.next_u64() >> 32) * 6) >> 32) as usize;
                *v = WEBB_PTS[idx];
            }
        }
        BootWeights::Mammen => {
            let sqrt5 = 5.0_f64.sqrt();
            let v1 = -(sqrt5 - 1.0) / 2.0;
            let v2 = (sqrt5 + 1.0) / 2.0;
            let p1 = (sqrt5 + 1.0) / (2.0 * sqrt5);
            for v in out.iter_mut() {
                *v = if rng.next_f64() < p1 { v1 } else { v2 };
            }
        }
        BootWeights::Normal => {
            let mut i = 0;
            while i + 1 < out.len() {
                let u1 = (1.0 - rng.next_f64()).max(f64::MIN_POSITIVE);
                let u2 = rng.next_f64();
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = std::f64::consts::TAU * u2;
                out[i] = r * theta.cos();
                out[i + 1] = r * theta.sin();
                i += 2;
            }
            if i < out.len() {
                let u1 = (1.0 - rng.next_f64()).max(f64::MIN_POSITIVE);
                let u2 = rng.next_f64();
                out[i] = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            }
        }
    }
}

fn oracle_cluster_weights(weights: BootWeights, g_cnt: usize, reps: usize, seed: u64) -> (Vec<Vec<f64>>, bool) {
    if weights == BootWeights::Rademacher && g_cnt <= 20 && (1usize << g_cnt) <= reps {
        let total = 1usize << g_cnt;
        let out = (0..total)
            .map(|b| (0..g_cnt).map(|g| if (b >> g) & 1 == 0 { 1.0 } else { -1.0 }).collect())
            .collect();
        return (out, true);
    }
    if weights == BootWeights::Webb && g_cnt <= 8 && 6usize.pow(g_cnt as u32) <= reps {
        let total = 6usize.pow(g_cnt as u32);
        let out = (0..total)
            .map(|b| {
                let mut v = Vec::with_capacity(g_cnt);
                let mut cur = b;
                for _ in 0..g_cnt {
                    v.push(WEBB_PTS[cur % 6]);
                    cur /= 6;
                }
                v
            })
            .collect();
        return (out, true);
    }
    let out = (0..reps)
        .map(|b| {
            let mut rng = SplitMix64::stream(seed, b as u64);
            let mut row = vec![0.0; g_cnt];
            fill_oracle_rng(weights, &mut rng, &mut row);
            row
        })
        .collect();
    (out, false)
}

/// Generates per-row weights in solver row order (`row_counts[i]` copies per compressed row `i`),
/// matching `run_hetero`'s chunked `hetero_chunk_rng` consumption.
fn oracle_hetero_weights_rows(weights: BootWeights, row_counts: &[usize], reps: usize, seed: u64) -> Vec<Vec<Vec<f64>>> {
    const CHUNK: usize = 65_536;
    let n = row_counts.len();
    (0..reps)
        .map(|b| {
            let mixed = seed ^ (b as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
            let mut out = vec![Vec::new(); n];
            for (cidx, start) in (0..n).step_by(CHUNK).enumerate() {
                let mut rng = SplitMix64::stream(mixed, cidx as u64);
                let end = (start + CHUNK).min(n);
                for i in start..end {
                    let cnt = row_counts[i];
                    let mut buf = vec![0.0; cnt];
                    fill_oracle_rng(weights, &mut rng, &mut buf);
                    out[i] = buf;
                }
            }
            out
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Synthetic multi-purpose dataset generator

struct SynthData {
    n: usize,
    y: Vec<f64>,
    y_pois: Vec<f64>,
    y_logit: Vec<f64>,
    y_iv: Vec<f64>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    x3_collinear: Vec<f64>,
    x4_absorbed: Vec<f64>,
    e1: Vec<f64>,
    e2: Vec<f64>,
    z1: Vec<f64>,
    z2: Vec<f64>,
    z3: Vec<f64>,
    slope: Vec<f64>,
    offset: Vec<f64>,
    aw: Vec<f64>,
    fw: Vec<f64>,
    fe_nested: Vec<f64>,
    fe_nn: Vec<f64>,
    fe_5: Vec<f64>,
    cl_4: Vec<f64>,
    cl_8: Vec<f64>,
    cl_12: Vec<f64>,
    cl_b: Vec<f64>,
}

fn gen_synth(n: usize, seed: u64) -> SynthData {
    let mut r = SplitMix64::new(seed);
    let mut r2 = SplitMix64::new(seed ^ 0x9E37_79B9_7F4A_7C15);

    let mut fe_nested = Vec::with_capacity(n);
    let mut fe_nn = Vec::with_capacity(n);
    let mut fe_5 = Vec::with_capacity(n);
    let mut cl_4 = Vec::with_capacity(n);
    let mut cl_8 = Vec::with_capacity(n);
    let mut cl_12 = Vec::with_capacity(n);
    let mut cl_b = Vec::with_capacity(n);

    for i in 0..n {
        let f8 = ((i * 3 + i / 7) % 8) as f64;
        fe_nested.push(f8);
        cl_8.push(f8);
        cl_4.push((f8 as usize / 2) as f64);
        cl_12.push(((i + (i / 11) * 3) % 12) as f64);
        fe_nn.push(((i * 5 + (i / 5) * 2) % 7) as f64);
        fe_5.push(((i * 2 + (i / 9)) % 5) as f64);
        cl_b.push(((i * 7 + (i / 3)) % 6) as f64);
    }

    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut x3_collinear = Vec::with_capacity(n);
    let mut x4_absorbed = Vec::with_capacity(n);
    let mut z1 = Vec::with_capacity(n);
    let mut z2 = Vec::with_capacity(n);
    let mut z3 = Vec::with_capacity(n);
    let mut e1 = Vec::with_capacity(n);
    let mut e2 = Vec::with_capacity(n);
    let mut slope = Vec::with_capacity(n);
    let mut offset = Vec::with_capacity(n);
    let mut aw = Vec::with_capacity(n);
    let mut fw = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_pois = Vec::with_capacity(n);
    let mut y_logit = Vec::with_capacity(n);
    let mut y_iv = Vec::with_capacity(n);

    let fe_eff_nested: Vec<f64> = (0..8).map(|_| 0.35 * next_normal(&mut r)).collect();
    let fe_eff_nn: Vec<f64> = (0..7).map(|_| -0.25 * next_normal(&mut r)).collect();
    let cl_shock: Vec<f64> = (0..12).map(|_| 0.4 * next_normal(&mut r)).collect();

    for i in 0..n {
        let fn_i = fe_nested[i] as usize;
        let fnn_i = fe_nn[i] as usize;
        let c12_i = cl_12[i] as usize;

        let xv1 = next_normal(&mut r) + 0.3 * fe_eff_nn[fnn_i];
        let xv2 = next_normal(&mut r) + 0.25 * cl_shock[c12_i];
        x1.push(xv1);
        x2.push(xv2);
        x3_collinear.push(2.0 * xv1 - 0.5 * xv2);
        x4_absorbed.push(fe_nested[i] * 0.4 + 1.0);

        let zv1 = next_normal(&mut r);
        let zv2 = next_normal(&mut r);
        let zv3 = next_normal(&mut r);
        z1.push(zv1);
        z2.push(zv2);
        z3.push(zv3);

        let u_iv = next_normal(&mut r) + 0.3 * cl_shock[c12_i];
        let ev1 = 0.65 * zv1 - 0.4 * zv2 + 0.25 * xv1 + 0.5 * u_iv + 0.2 * fe_eff_nn[fnn_i] + 0.4 * next_normal(&mut r);
        let ev2 = -0.35 * zv2 + 0.7 * zv3 + 0.2 * xv2 + 0.4 * u_iv + 0.4 * next_normal(&mut r);
        e1.push(ev1);
        e2.push(ev2);

        let slp = 0.5 + 1.5 * r2.next_f64();
        slope.push(slp);
        let off = 0.2 * next_normal(&mut r);
        offset.push(off);
        aw.push(0.5 + 1.5 * r2.next_f64());
        fw.push((1 + (r2.next_f64() * 3.0).floor() as usize) as f64);

        let mu_fe = fe_eff_nested[fn_i] + fe_eff_nn[fnn_i];
        y.push(0.75 * xv1 - 0.5 * xv2 + 0.4 + mu_fe + 0.3 * fe_nn[i] * 0.05 + cl_shock[c12_i] + next_normal(&mut r));
        y_iv.push(0.9 * ev1 - 0.6 * ev2 + 0.5 * xv1 - 0.3 * xv2 + 0.25 + mu_fe + u_iv);

        let eta_p = (0.3 + 0.35 * xv1 - 0.25 * xv2 + 0.3 * mu_fe + 0.2 * cl_shock[c12_i]).clamp(-2.5, 2.5);
        let lam = eta_p.exp();
        let l = (-lam).exp();
        let mut k = 0u64;
        let mut p = 1.0;
        while p > l {
            k += 1;
            p *= r2.next_f64().max(1e-12);
        }
        y_pois.push((k - 1) as f64);

        let eta_l = 0.1 + 0.6 * xv1 - 0.45 * xv2 + 0.25 * mu_fe + 0.2 * cl_shock[c12_i];
        let prob = 1.0 / (1.0 + (-eta_l).exp());
        y_logit.push(if r2.next_f64() < prob { 1.0 } else { 0.0 });
    }

    for lv in 0..7 {
        let mut idx0 = None;
        let mut idx1 = None;
        let mut pois_sum = 0.0;
        let mut log_sum = 0.0;
        let mut cnt = 0usize;
        for i in 0..n {
            if fe_nn[i] as usize == lv {
                pois_sum += y_pois[i];
                log_sum += y_logit[i];
                cnt += 1;
                if y_logit[i] == 0.0 && idx0.is_none() {
                    idx0 = Some(i);
                }
                if y_logit[i] == 1.0 && idx1.is_none() {
                    idx1 = Some(i);
                }
            }
        }
        if pois_sum == 0.0
            && let Some(i) = (0..n).find(|&i| fe_nn[i] as usize == lv)
        {
            y_pois[i] = 1.0;
        }
        if log_sum == 0.0 {
            if let Some(i) = idx0 {
                y_logit[i] = 1.0;
            }
        } else if (log_sum as usize) == cnt
            && let Some(i) = idx1
        {
            y_logit[i] = 0.0;
        }
    }

    SynthData {
        n,
        y,
        y_pois,
        y_logit,
        y_iv,
        x1,
        x2,
        x3_collinear,
        x4_absorbed,
        e1,
        e2,
        z1,
        z2,
        z3,
        slope,
        offset,
        aw,
        fw,
        fe_nested,
        fe_nn,
        fe_5,
        cl_4,
        cl_8,
        cl_12,
        cl_b,
    }
}

// ---------------------------------------------------------------------------------------------
// Independent Dense Brute-Force Oracles

#[derive(Debug)]
struct OracleOut {
    stat: f64,
    se: Option<f64>,
    point: Option<f64>,
    t_boot: Vec<f64>,
}

/// Extract weights in kept order (matching Session).
fn extract_weights_kept(s: &Session, w: Option<(&[f64], WeightKind)>) -> (Vec<f64>, Vec<f64>, WeightKind, f64) {
    let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
    let n = kept.len();
    let (raw, kind) = match w {
        Some((arr, k)) => (kept.iter().map(|&i| arr[i]).collect(), k),
        None => (vec![1.0; n], WeightKind::None),
    };
    let sumw: f64 = raw.iter().sum();
    let big_n = if kind == WeightKind::Fweight { sumw } else { n as f64 };
    let wv = match kind {
        WeightKind::Aweight | WeightKind::Pweight => raw.iter().map(|v| v * (n as f64) / sumw).collect(),
        _ => raw.clone(),
    };
    (wv, raw, kind, big_n)
}

fn dense_wls(cols: &[Vec<f64>], y: &[f64], wv: &[f64]) -> (Vec<f64>, Mat<f64>, Mat<f64>, Vec<f64>) {
    let n = y.len();
    let p = cols.len();
    let xwx = mat_from(p, p, |a, b| (0..n).map(|i| wv[i] * cols[a][i] * cols[b][i]).sum());
    let xwy: Vec<f64> = (0..p).map(|a| (0..n).map(|i| wv[i] * cols[a][i] * y[i]).sum()).collect();
    let inv = inverse(&xwx);
    let beta: Vec<f64> = (0..p).map(|a| (0..p).map(|b| inv[(a, b)] * xwy[b]).sum()).collect();
    let resid: Vec<f64> = (0..n).map(|i| y[i] - (0..p).map(|a| cols[a][i] * beta[a]).sum::<f64>()).collect();
    (beta, xwx, inv, resid)
}

fn solve_restricted(beta_u: &[f64], xwx_inv: &Mat<f64>, r_mat: &[f64], r_vec: &[f64], q: usize, k_kept: usize) -> Vec<f64> {
    let p = beta_u.len();
    let mut rxr = Mat::<f64>::zeros(q, q);
    let mut rx_cols = vec![vec![0.0; p]; q];
    for a in 0..q {
        for i in 0..p {
            rx_cols[a][i] = (0..k_kept).map(|j| xwx_inv[(i, j)] * r_mat[a * k_kept + j]).sum();
        }
    }
    for a in 0..q {
        for b in 0..q {
            rxr[(a, b)] = (0..k_kept).map(|j| r_mat[a * k_kept + j] * rx_cols[b][j]).sum();
        }
    }
    let rxr_inv = inverse(&rxr);
    let disc: Vec<f64> = (0..q).map(|a| (0..k_kept).map(|j| r_mat[a * k_kept + j] * beta_u[j]).sum::<f64>() - r_vec[a]).collect();
    let lam: Vec<f64> = (0..q).map(|a| (0..q).map(|b| rxr_inv[(a, b)] * disc[b]).sum()).collect();
    (0..p).map(|i| beta_u[i] - (0..q).map(|a| rx_cols[a][i] * lam[a]).sum::<f64>()).collect()
}

fn eval_stat(beta: &[f64], v_k: &Mat<f64>, r_mat: &[f64], r_vec: &[f64], q: usize, k: usize) -> (f64, Option<f64>, Option<f64>) {
    let r_v_rt = mat_from(q, q, |a, b| {
        let mut s = 0.0;
        for c in 0..k {
            for d in 0..k {
                s += r_mat[a * k + c] * v_k[(c, d)] * r_mat[b * k + d];
            }
        }
        s
    });
    let gap: Vec<f64> = (0..q).map(|a| (0..k).map(|j| r_mat[a * k + j] * beta[j]).sum::<f64>() - r_vec[a]).collect();
    if q == 1 {
        let var = r_v_rt[(0, 0)];
        let pt = (0..k).map(|j| r_mat[j] * beta[j]).sum::<f64>();
        if var <= 0.0 || !var.is_finite() {
            (f64::NAN, None, Some(pt))
        } else {
            let se = var.sqrt();
            (gap[0] / se, Some(se), Some(pt))
        }
    } else {
        let vi = inverse(&r_v_rt);
        let wald: f64 = (0..q).map(|a| (0..q).map(|b| gap[a] * vi[(a, b)] * gap[b]).sum::<f64>()).sum();
        (wald / q as f64, None, None)
    }
}

/// Independent dense explicit-dummy OLS wild cluster bootstrap oracle (all CRV variants: 11, 13, 31, 33).
#[allow(clippy::too_many_arguments)]
fn oracle_ols_cluster(
    s: &Session,
    xs_orig: &[&[f64]],
    y_orig: &[f64],
    w_orig: Option<(&[f64], WeightKind)>,
    noconstant: bool,
    boot_codes: &[u32],
    boot_g: usize,
    err_dims: &[(&[u32], usize)],
    r_mat: &[f64],
    r_vec: &[f64],
    q: usize,
    wb: &WildBootSpec,
    nested_in_err: bool,
    df_a_override: Option<f64>,
) -> OracleOut {
    let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
    let n = kept.len();
    let pick = |v: &[f64]| -> Vec<f64> { kept.iter().map(|&i| v[i]).collect() };
    let y = pick(y_orig);
    let mut x_cols: Vec<Vec<f64>> = xs_orig.iter().map(|x| pick(x)).collect();
    if s.terms().is_empty() && !noconstant {
        x_cols.push(vec![1.0; n]);
    }
    let k = x_cols.len();
    let (wv, _raw_w, _kind, big_n) = extract_weights_kept(s, w_orig);
    let mut da = DenseAbsorber::from_session(s);
    if s.terms().is_empty() && noconstant {
        // reghdfe noabsorb+noconstant absorbs the constant implicitly in absorber (`df_a = 1`).
        let norm = (n as f64).sqrt();
        da.d_cols = vec![vec![1.0; n]];
        da.d_basis = vec![vec![1.0 / norm; n]];
        da.rank = 1;
    }
    let mut full_cols = x_cols.clone();
    full_cols.extend(da.d_basis.iter().cloned());
    let p_tot = full_cols.len();

    let (beta_u, _a_mat, a_inv, e_u) = dense_wls(&full_cols, &y, &wv);
    let beta_r = solve_restricted(&beta_u, &a_inv, r_mat, r_vec, q, k);
    let e_r: Vec<f64> = (0..n).map(|i| y[i] - (0..p_tot).map(|c| full_cols[c][i] * beta_r[c]).sum::<f64>()).collect();

    let beta_star = if wb.impose_null { &beta_r } else { &beta_u };
    let e_star = if wb.impose_null { &e_r } else { &e_u };
    let yhat_star: Vec<f64> = (0..n).map(|i| (0..p_tot).map(|c| full_cols[c][i] * beta_star[c]).sum()).collect();

    let g_min = err_dims.iter().map(|&(_, g)| g).min().unwrap() as f64;
    let df_a = df_a_override.unwrap_or(da.rank as f64);
    let nested_adj = if nested_in_err { 1.0 } else { 0.0 };
    let k_slopes = k - usize::from(s.terms().is_empty() && !noconstant);
    let ssc_crv1 = match wb.boot_type {
        BootType::T11 | BootType::T31 => ((big_n - 1.0) / (big_n - nested_adj - k_slopes as f64 - df_a)) * (g_min / (g_min - 1.0)),
        BootType::T13 | BootType::T33 => 1.0,
        _ => unreachable!(),
    };

    let crv1_cov = |resid: &[f64]| -> Mat<f64> {
        let mut meat = Mat::<f64>::zeros(p_tot, p_tot);
        let nd = err_dims.len();
        for mask in 1u32..(1u32 << nd) {
            let sign = if mask.count_ones() % 2 == 1 { 1.0 } else { -1.0 };
            let mut sums: HashMap<Vec<u32>, Vec<f64>> = HashMap::new();
            for i in 0..n {
                let key: Vec<u32> = (0..nd).filter(|d| (mask >> d) & 1 == 1).map(|d| err_dims[d].0[i]).collect();
                let ent = sums.entry(key).or_insert_with(|| vec![0.0; p_tot]);
                let sc = wv[i] * resid[i];
                for c in 0..p_tot {
                    ent[c] += sc * full_cols[c][i];
                }
            }
            for sg in sums.values() {
                for a in 0..p_tot {
                    for b in 0..p_tot {
                        meat[(a, b)] += sign * sg[a] * sg[b];
                    }
                }
            }
        }
        let bread = &a_inv * &meat * &a_inv;
        mat_from(k, k, |a, b| bread[(a, b)] * ssc_crv1)
    };

    let loo_solve = |y_in: &[f64], w_g: &[f64]| -> (Vec<f64>, Mat<f64>, Vec<Vec<f64>>) {
        let active_d = independent_columns_weighted(&da.d_cols, w_g);
        let mut cols_g = x_cols.clone();
        cols_g.extend(active_d);
        let (bg, _, inv_g, _) = dense_wls(&cols_g, y_in, w_g);
        (bg, inv_g, cols_g)
    };

    let crv3_cov = |y_in: &[f64], beta_full: &[f64]| -> Mat<f64> {
        assert_eq!(err_dims.len(), 1);
        let (ec, eg) = err_dims[0];
        let mut b_loo = Vec::with_capacity(eg);
        for g in 0..eg {
            let mut w_g = wv.clone();
            for i in 0..n {
                if ec[i] as usize == g {
                    w_g[i] = 0.0;
                }
            }
            let (bg, _, _) = loo_solve(y_in, &w_g);
            b_loo.push(bg[..k].to_vec());
        }
        let scale = (eg as f64 - 1.0) / (eg as f64);
        let mut v = Mat::<f64>::zeros(k, k);
        for g in 0..eg {
            for a in 0..k {
                for b in 0..k {
                    v[(a, b)] += scale * (b_loo[g][a] - beta_full[a]) * (b_loo[g][b] - beta_full[b]);
                }
            }
        }
        v
    };

    // Brute-force leave-one-cluster-out jackknife residuals: y_i - X_full[i] * beta^{(-g)} (or beta^{R(-g)}).
    let jack_resids: Vec<f64> = if matches!(wb.boot_type, BootType::T31 | BootType::T33) {
        let mut out = vec![0.0; n];
        for g in 0..boot_g {
            let mut w_g = wv.clone();
            for i in 0..n {
                if boot_codes[i] as usize == g {
                    w_g[i] = 0.0;
                }
            }
            let (bg, inv_g, cols_g) = loo_solve(&y, &w_g);
            let b_use = if wb.impose_null {
                solve_restricted(&bg, &inv_g, r_mat, r_vec, q, k)
            } else {
                bg
            };
            let p_g = cols_g.len();
            for i in 0..n {
                if boot_codes[i] as usize == g {
                    out[i] = y[i] - (0..p_g).map(|c| cols_g[c][i] * b_use[c]).sum::<f64>();
                }
            }
        }
        out
    } else {
        e_star.clone()
    };

    let v_orig = match wb.boot_type {
        BootType::T11 | BootType::T31 => crv1_cov(&e_u),
        BootType::T13 | BootType::T33 => crv3_cov(&y, &beta_u),
        _ => unreachable!(),
    };
    let (stat, se, point) = eval_stat(&beta_u[..k], &v_orig, r_mat, r_vec, q, k);

    let (draws, _) = oracle_cluster_weights(wb.weights, boot_g, wb.reps, wb.seed);
    let r_target: Vec<f64> = (0..q).map(|a| (0..k).map(|j| r_mat[a * k + j] * beta_star[j]).sum()).collect();
    let mut t_boot = Vec::with_capacity(draws.len());
    for v_g in draws {
        let y_b: Vec<f64> = (0..n).map(|i| yhat_star[i] + v_g[boot_codes[i] as usize] * jack_resids[i]).collect();
        let (beta_b, _, _, e_b) = dense_wls(&full_cols, &y_b, &wv);
        let v_b = match wb.boot_type {
            BootType::T11 | BootType::T31 => crv1_cov(&e_b),
            BootType::T13 | BootType::T33 => crv3_cov(&y_b, &beta_b),
            _ => unreachable!(),
        };
        let (tb, _, _) = eval_stat(&beta_b[..k], &v_b, r_mat, &r_target, q, k);
        t_boot.push(tb);
    }

    OracleOut { stat, se, point, t_boot }
}

/// Independent dense explicit-dummy OLS Heteroskedastic bootstrap oracle (`11`, `21`, `31`, `13`, `33`).
#[allow(clippy::too_many_arguments)]
fn oracle_ols_hetero(
    s: &Session,
    xs_orig: &[&[f64]],
    y_orig: &[f64],
    w_orig: Option<(&[f64], WeightKind)>,
    noconstant: bool,
    r_mat: &[f64],
    r_vec: &[f64],
    q: usize,
    wb: &WildBootSpec,
) -> OracleOut {
    let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
    let row_order = s.row_order();
    let n = kept.len();
    let pick = |v: &[f64]| -> Vec<f64> { kept.iter().map(|&i| v[i]).collect() };
    let y = pick(y_orig);
    let mut x_cols: Vec<Vec<f64>> = xs_orig.iter().map(|x| pick(x)).collect();
    let explicit_cons = s.terms().is_empty() && !noconstant;
    if explicit_cons {
        x_cols.push(vec![1.0; n]);
    }
    let k = x_cols.len();
    let (wv, _raw_w, kind, big_n) = extract_weights_kept(s, w_orig);
    let mut da = DenseAbsorber::from_session(s);
    if s.terms().is_empty() && noconstant {
        let norm = (n as f64).sqrt();
        da.d_cols = vec![vec![1.0; n]];
        da.d_basis = vec![vec![1.0 / norm; n]];
        da.rank = 1;
    }
    let mut full_cols = x_cols.clone();
    full_cols.extend(da.d_basis.iter().cloned());
    let p_tot = full_cols.len();

    let (beta_u, _a_mat, a_inv, e_u) = dense_wls(&full_cols, &y, &wv);
    let beta_r = solve_restricted(&beta_u, &a_inv, r_mat, r_vec, q, k);
    let e_r: Vec<f64> = (0..n).map(|i| y[i] - (0..p_tot).map(|c| full_cols[c][i] * beta_r[c]).sum::<f64>()).collect();
    let beta_star = if wb.impose_null { &beta_r } else { &beta_u };
    let e_star = if wb.impose_null { &e_r } else { &e_u };
    let yhat_star: Vec<f64> = (0..n).map(|i| (0..p_tot).map(|c| full_cols[c][i] * beta_star[c]).sum()).collect();

    let h: Vec<f64> = (0..n)
        .map(|i| {
            let mut quad = 0.0;
            for a in 0..p_tot {
                for b in 0..p_tot {
                    quad += full_cols[a][i] * a_inv[(a, b)] * full_cols[b][i];
                }
            }
            if kind == WeightKind::Fweight { quad } else { wv[i] * quad }
        })
        .collect();

    let k_slopes = k - usize::from(explicit_cons);
    let df_a_eff = if explicit_cons { 1.0 } else { da.rank as f64 };
    let q_hc1 = big_n / (big_n - k_slopes as f64 - df_a_eff);
    let hc_cov = |resid: &[f64], is_hc3: bool| -> Mat<f64> {
        let mut meat = Mat::<f64>::zeros(p_tot, p_tot);
        for i in 0..n {
            let denom = if is_hc3 { (1.0 - h[i]).max(1e-12).powi(2) } else { 1.0 };
            let mi = match kind {
                WeightKind::None => resid[i] * resid[i] / denom,
                WeightKind::Fweight => wv[i] * resid[i] * resid[i] / denom,
                _ => (wv[i] * resid[i]).powi(2) / denom,
            };
            for a in 0..p_tot {
                for b in 0..p_tot {
                    meat[(a, b)] += mi * full_cols[a][i] * full_cols[b][i];
                }
            }
        }
        let scale = if is_hc3 { 1.0 } else { q_hc1 };
        let bread = &a_inv * &meat * &a_inv;
        mat_from(k, k, |a, b| bread[(a, b)] * scale)
    };

    let (alpha_dgp, eval_hc3) = match wb.boot_type {
        BootType::T11 => (0.0, false),
        BootType::T21 => (0.5, false),
        BootType::T31 => (1.0, false),
        BootType::T13 => (0.0, true),
        BootType::T33 => (1.0, true),
    };
    let e_dgp: Vec<f64> = (0..n).map(|i| e_star[i] / (1.0 - h[i]).max(1e-12).powf(alpha_dgp)).collect();
    let v_orig = hc_cov(&e_u, eval_hc3);
    let (stat, se, point) = eval_stat(&beta_u[..k], &v_orig, r_mat, r_vec, q, k);

    let r_target: Vec<f64> = (0..q).map(|a| (0..k).map(|j| r_mat[a * k + j] * beta_star[j]).sum()).collect();
    // RNG chunk consumption is in solver row order (`row_order` maps solver row `r -> kept_pos`).
    let counts_row: Vec<usize> = row_order
        .iter()
        .map(|&ki| if kind == WeightKind::Fweight { wv[ki as usize].round().max(1.0) as usize } else { 1 })
        .collect();
    let draws_row = oracle_hetero_weights_rows(wb.weights, &counts_row, wb.reps, wb.seed);
    let mut t_boot = Vec::with_capacity(wb.reps);

    for draw in draws_row {
        // Permute per-row draws back from solver row order -> kept order.
        let mut s1 = vec![0.0; n];
        let mut s2 = vec![0.0; n];
        for (row_idx, &ki_u32) in row_order.iter().enumerate() {
            let ki = ki_u32 as usize;
            s1[ki] = draw[row_idx].iter().sum();
            s2[ki] = draw[row_idx].iter().map(|v| v * v).sum();
        }
        if kind == WeightKind::Fweight {
            let mut xwy_b = vec![0.0; p_tot];
            for c in 0..p_tot {
                xwy_b[c] = (0..n).map(|i| full_cols[c][i] * (wv[i] * yhat_star[i] + s1[i] * e_dgp[i])).sum();
            }
            let beta_b: Vec<f64> = (0..p_tot).map(|a| (0..p_tot).map(|b| a_inv[(a, b)] * xwy_b[b]).sum()).collect();
            let diff: Vec<f64> = (0..p_tot).map(|c| beta_b[c] - beta_star[c]).collect();
            let mut meat_b = Mat::<f64>::zeros(p_tot, p_tot);
            for i in 0..n {
                let xdiff: f64 = (0..p_tot).map(|c| full_cols[c][i] * diff[c]).sum();
                let sum_e2 = s2[i] * e_dgp[i] * e_dgp[i] - 2.0 * s1[i] * e_dgp[i] * xdiff + wv[i] * xdiff * xdiff;
                let denom = if eval_hc3 { (1.0 - h[i]).max(1e-12).powi(2) } else { 1.0 };
                let mi = sum_e2 / denom;
                for a in 0..p_tot {
                    for b in 0..p_tot {
                        meat_b[(a, b)] += mi * full_cols[a][i] * full_cols[b][i];
                    }
                }
            }
            let scale = if eval_hc3 { 1.0 } else { q_hc1 };
            let bread = &a_inv * &meat_b * &a_inv;
            let v_b = mat_from(k, k, |a, b| bread[(a, b)] * scale);
            let (tb, _, _) = eval_stat(&beta_b[..k], &v_b, r_mat, &r_target, q, k);
            t_boot.push(tb);
        } else {
            let y_b: Vec<f64> = (0..n).map(|i| yhat_star[i] + s1[i] * e_dgp[i]).collect();
            let (beta_b, _, _, e_b) = dense_wls(&full_cols, &y_b, &wv);
            let v_b = hc_cov(&e_b, eval_hc3);
            let (tb, _, _) = eval_stat(&beta_b[..k], &v_b, r_mat, &r_target, q, k);
            t_boot.push(tb);
        }
    }

    OracleOut { stat, se, point, t_boot }
}

fn dense_glm_newton(
    cols: &[Vec<f64>],
    y: &[f64],
    wt: &[f64],
    offset: &[f64],
    family: Family,
    constraint: Option<(&[f64], &[f64], usize, usize)>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Mat<f64>) {
    let n = y.len();
    let p = cols.len();
    let mut gamma = vec![0.0; p];
    if let Some((rm, rv, q, kk)) = constraint {
        let rrt = mat_from(q, q, |a, b| (0..kk).map(|j| rm[a * kk + j] * rm[b * kk + j]).sum());
        let inv = inverse(&rrt);
        let u: Vec<f64> = (0..q).map(|a| (0..q).map(|b| inv[(a, b)] * rv[b]).sum()).collect();
        for j in 0..kk {
            gamma[j] = (0..q).map(|a| rm[a * kk + j] * u[a]).sum();
        }
    }

    for _it in 0..200 {
        let mut mu = vec![0.0; n];
        let mut w_irls = vec![0.0; n];
        let mut grad = vec![0.0; p];
        for i in 0..n {
            let e = offset[i] + (0..p).map(|c| cols[c][i] * gamma[c]).sum::<f64>();
            let (m, vm) = match family {
                Family::Poisson => {
                    let m = e.clamp(-30.0, 30.0).exp();
                    (m, m)
                }
                Family::Logit => {
                    let m = if e >= 0.0 {
                        1.0 / (1.0 + (-e).exp())
                    } else {
                        let ex = e.exp();
                        ex / (1.0 + ex)
                    };
                    (m, m * (1.0 - m))
                }
                Family::Gaussian => (e, 1.0),
            };
            mu[i] = m;
            w_irls[i] = wt[i] * vm;
            let s = wt[i] * (y[i] - m);
            for c in 0..p {
                grad[c] += cols[c][i] * s;
            }
        }
        let h = mat_from(p, p, |a, b| (0..n).map(|i| w_irls[i] * cols[a][i] * cols[b][i]).sum());
        let h_inv = inverse(&h);
        let mut step_un: Vec<f64> = (0..p).map(|a| (0..p).map(|b| h_inv[(a, b)] * grad[b]).sum()).collect();
        if let Some((rm, rv, q, kk)) = constraint {
            let gap: Vec<f64> = (0..q).map(|a| (0..kk).map(|j| rm[a * kk + j] * (gamma[j] + step_un[j])).sum::<f64>() - rv[a]).collect();
            let mut hx_cols = vec![vec![0.0; p]; q];
            for a in 0..q {
                for i in 0..p {
                    hx_cols[a][i] = (0..kk).map(|j| h_inv[(i, j)] * rm[a * kk + j]).sum();
                }
            }
            let rhr = mat_from(q, q, |a, b| (0..kk).map(|j| rm[a * kk + j] * hx_cols[b][j]).sum());
            let rhr_inv = inverse(&rhr);
            let lam: Vec<f64> = (0..q).map(|a| (0..q).map(|b| rhr_inv[(a, b)] * gap[b]).sum()).collect();
            for i in 0..p {
                step_un[i] -= (0..q).map(|a| hx_cols[a][i] * lam[a]).sum::<f64>();
            }
        }
        let max_step = step_un.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let mut step_scale = 1.0;
        if family == Family::Poisson && max_step > 5.0 {
            step_scale = 5.0 / max_step;
        }
        for c in 0..p {
            gamma[c] += step_scale * step_un[c];
        }
        if max_step < 1e-13 {
            break;
        }
    }

    let mut mu = vec![0.0; n];
    let mut w_irls = vec![0.0; n];
    for i in 0..n {
        let e = offset[i] + (0..p).map(|c| cols[c][i] * gamma[c]).sum::<f64>();
        let (m, vm) = match family {
            Family::Poisson => {
                let m = e.exp();
                (m, m)
            }
            Family::Logit => {
                let m = 1.0 / (1.0 + (-e).exp());
                (m, m * (1.0 - m))
            }
            Family::Gaussian => (e, 1.0),
        };
        mu[i] = m;
        w_irls[i] = wt[i] * vm;
    }
    let h = mat_from(p, p, |a, b| (0..n).map(|i| w_irls[i] * cols[a][i] * cols[b][i]).sum());
    (gamma, mu, w_irls, inverse(&h))
}

/// Independent dense explicit-dummy GLM Score bootstrap oracle.
#[allow(clippy::too_many_arguments)]
fn oracle_glm_score(
    s: &Session,
    family: Family,
    xs_orig: &[&[f64]],
    y_orig: &[f64],
    offset_orig: Option<&[f64]>,
    w_orig: Option<(&[f64], WeightKind)>,
    noconstant: bool,
    boot_codes: Option<(&[u32], usize)>,
    err_dims: Option<&[(&[u32], usize)]>,
    r_mat: &[f64],
    r_vec: &[f64],
    q: usize,
    wb: &WildBootSpec,
) -> OracleOut {
    let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
    let row_order = s.row_order();
    let n = kept.len();
    let pick = |v: &[f64]| -> Vec<f64> { kept.iter().map(|&i| v[i]).collect() };
    let y = pick(y_orig);
    let offset = offset_orig.map(pick).unwrap_or_else(|| vec![0.0; n]);
    let mut x_cols: Vec<Vec<f64>> = xs_orig.iter().map(|x| pick(x)).collect();
    if s.terms().is_empty() && !noconstant {
        x_cols.push(vec![1.0; n]);
    }
    let k = x_cols.len();
    let (wv, _raw_w, _kind, big_n) = extract_weights_kept(s, w_orig);
    let mut da = DenseAbsorber::from_session(s);
    if s.terms().is_empty() && noconstant {
        // reghdfe noabsorb+noconstant absorbs the constant implicitly in absorber (`df_a = 1`).
        let norm = (n as f64).sqrt();
        da.d_cols = vec![vec![1.0; n]];
        da.d_basis = vec![vec![1.0 / norm; n]];
        da.rank = 1;
    }
    let mut full_cols = x_cols.clone();
    full_cols.extend(da.d_basis.iter().cloned());
    let p_tot = full_cols.len();

    let (gamma_u, mu_u, w_u, h_u_inv) = dense_glm_newton(&full_cols, &y, &wv, &offset, family, None);
    let (_gamma_r, mu_r, w_r, h_r_inv) = dense_glm_newton(&full_cols, &y, &wv, &offset, family, Some((r_mat, r_vec, q, k)));

    let (mu_s, w_s, h_s_inv) = if wb.impose_null { (&mu_r, &w_r, &h_r_inv) } else { (&mu_u, &w_u, &h_u_inv) };
    let u_s: Vec<f64> = (0..n).map(|i| wv[i] * (y[i] - mu_s[i])).collect();

    let x_tilde: Vec<Vec<f64>> = (0..k).map(|c| da.project_vec(&x_cols[c], w_s)).collect();
    let j_k = mat_from(k, k, |a, b| (0..n).map(|i| w_s[i] * x_tilde[a][i] * x_tilde[b][i]).sum());
    let j_k_inv = inverse(&j_k);
    let ssc_uncl = big_n / (big_n - 1.0);

    let cov_from_scores = |weights_v: Option<&[f64]>, boot_c: Option<(&[u32], usize)>| -> (Vec<f64>, Mat<f64>) {
        let z: Vec<f64> = (0..n)
            .map(|i| {
                let vi = match (weights_v, boot_c) {
                    (Some(vg), Some((bc, _))) => vg[bc[i] as usize],
                    (Some(vi_arr), None) => vi_arr[i],
                    _ => 1.0,
                };
                vi * u_s[i]
            })
            .collect();
        let atz: Vec<f64> = (0..p_tot).map(|c| (0..n).map(|i| full_cols[c][i] * z[i]).sum()).collect();
        let delta_full: Vec<f64> = (0..p_tot).map(|a| (0..p_tot).map(|b| h_s_inv[(a, b)] * atz[b]).sum()).collect();

        let (r_delta, resid_s) = if weights_v.is_none() && !wb.impose_null {
            (gamma_u[..k].to_vec(), u_s.clone())
        } else {
            let res: Vec<f64> = (0..n)
                .map(|i| z[i] - w_s[i] * (0..p_tot).map(|c| full_cols[c][i] * delta_full[c]).sum::<f64>())
                .collect();
            (delta_full[..k].to_vec(), res)
        };

        let mut omega = Mat::<f64>::zeros(k, k);
        if let Some(dims) = err_dims {
            let nd = dims.len();
            for mask in 1u32..(1u32 << nd) {
                let sign = if mask.count_ones() % 2 == 1 { 1.0 } else { -1.0 };
                let mut sums: HashMap<Vec<u32>, Vec<f64>> = HashMap::new();
                for i in 0..n {
                    let key: Vec<u32> = (0..nd).filter(|d| (mask >> d) & 1 == 1).map(|d| dims[d].0[i]).collect();
                    let ent = sums.entry(key).or_insert_with(|| vec![0.0; k]);
                    for c in 0..k {
                        ent[c] += x_tilde[c][i] * resid_s[i];
                    }
                }
                for sg in sums.values() {
                    for a in 0..k {
                        for b in 0..k {
                            omega[(a, b)] += sign * sg[a] * sg[b];
                        }
                    }
                }
            }
            let g_min = dims.iter().map(|&(_, g)| g).min().unwrap() as f64;
            let scale = g_min / (g_min - 1.0);
            for a in 0..k {
                for b in 0..k {
                    omega[(a, b)] *= scale;
                }
            }
        } else {
            for i in 0..n {
                let s2 = ssc_uncl * resid_s[i] * resid_s[i];
                for a in 0..k {
                    for b in 0..k {
                        omega[(a, b)] += s2 * x_tilde[a][i] * x_tilde[b][i];
                    }
                }
            }
        }
        let v_cov = &j_k_inv * &omega * &j_k_inv;
        (r_delta, v_cov)
    };

    let (b0, v0) = cov_from_scores(None, boot_codes);
    let r_v0 = if wb.impose_null { vec![0.0; q] } else { r_vec.to_vec() };
    let (stat, se, _) = eval_stat(&b0, &v0, r_mat, &r_v0, q, k);
    let point = (q == 1).then(|| (0..k).map(|j| r_mat[j] * gamma_u[j]).sum());

    let draws: Vec<Vec<f64>> = if let Some((_, bg)) = boot_codes {
        oracle_cluster_weights(wb.weights, bg, wb.reps, wb.seed).0
    } else {
        let counts_row = vec![1usize; n];
        let d_row = oracle_hetero_weights_rows(wb.weights, &counts_row, wb.reps, wb.seed);
        d_row
            .into_iter()
            .map(|draw| {
                let mut v_kept = vec![0.0; n];
                for (row_idx, &ki_u32) in row_order.iter().enumerate() {
                    v_kept[ki_u32 as usize] = draw[row_idx][0];
                }
                v_kept
            })
            .collect()
    };

    let mut t_boot = Vec::with_capacity(draws.len());
    let zero_r = vec![0.0; q];
    for v_arr in draws {
        let (bb, vb) = cov_from_scores(Some(&v_arr), boot_codes);
        let (tb, _, _) = eval_stat(&bb, &vb, r_mat, &zero_r, q, k);
        t_boot.push(tb);
    }

    OracleOut { stat, se, point, t_boot }
}

/// Independent dense explicit-dummy 2SLS WRE bootstrap oracle (Davidson & MacKinnon 2010 /
/// Roodman et al. 2019 Sec. 4.4) running explicit 2SLS in N-space per draw.
#[allow(clippy::too_many_arguments)]
fn oracle_iv_wre(
    s: &Session,
    y_orig: &[f64],
    endog_orig: &[&[f64]],
    exog_orig: &[&[f64]],
    inst_orig: &[&[f64]],
    w_orig: Option<(&[f64], WeightKind)>,
    noconstant: bool,
    iv_small: bool,
    boot_codes: Option<(&[u32], usize)>,
    err_dims: Option<&[(&[u32], usize)]>,
    r_mat: &[f64],
    r_vec: &[f64],
    q: usize,
    wb: &WildBootSpec,
    nested_in_err: bool,
) -> OracleOut {
    let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
    let row_order = s.row_order();
    let n = kept.len();
    let pick = |v: &[f64]| -> Vec<f64> { kept.iter().map(|&i| v[i]).collect() };
    let y = pick(y_orig);
    let endogs: Vec<Vec<f64>> = endog_orig.iter().map(|c| pick(c)).collect();
    let mut exogs: Vec<Vec<f64>> = exog_orig.iter().map(|c| pick(c)).collect();
    let insts: Vec<Vec<f64>> = inst_orig.iter().map(|c| pick(c)).collect();
    let explicit_cons = s.terms().is_empty() && !noconstant;
    if explicit_cons {
        exogs.push(vec![1.0; n]);
    }
    let m = endogs.len();
    let kx = exogs.len();
    let k = m + kx;
    let (wv, _raw_w, _kind, big_n) = extract_weights_kept(s, w_orig);
    let mut da = DenseAbsorber::from_session(s);
    if s.terms().is_empty() && noconstant {
        // When `noabsorb` with `!report_constant`, IV Session absorbs a 1-group constant FE.
        let norm = (n as f64).sqrt();
        da.d_cols = vec![vec![1.0; n]];
        da.d_basis = vec![vec![1.0 / norm; n]];
        da.rank = 1;
    }

    let x1_t: Vec<Vec<f64>> = endogs.iter().map(|c| da.project_vec(c, &wv)).collect();
    let x2_t: Vec<Vec<f64>> = exogs.iter().map(|c| da.project_vec(c, &wv)).collect();
    let z2_t: Vec<Vec<f64>> = insts.iter().map(|c| da.project_vec(c, &wv)).collect();
    let y_t = da.project_vec(&y, &wv);

    let mut x_struct = x1_t.clone();
    x_struct.extend(x2_t.iter().cloned());
    let mut z_full = x2_t.clone();
    z_full.extend(z2_t.iter().cloned());
    let l = z_full.len();

    let zwz = mat_from(l, l, |a, b| (0..n).map(|i| wv[i] * z_full[a][i] * z_full[b][i]).sum());
    let zwz_inv = inverse(&zwz);
    let zwx = mat_from(l, k, |a, b| (0..n).map(|i| wv[i] * z_full[a][i] * x_struct[b][i]).sum());
    let xwz = mat_from(k, l, |a, b| zwx[(b, a)]);
    let p_gram = &xwz * &zwz_inv * &zwx;
    let p_gram_inv = inverse(&p_gram);
    let zwy: Vec<f64> = (0..l).map(|a| (0..n).map(|i| wv[i] * z_full[a][i] * y_t[i]).sum()).collect();
    let xhat_wy: Vec<f64> = (0..k)
        .map(|a| {
            let mut s = 0.0;
            for c in 0..l {
                for d in 0..l {
                    s += xwz[(a, c)] * zwz_inv[(c, d)] * zwy[d];
                }
            }
            s
        })
        .collect();
    let beta_2sls: Vec<f64> = (0..k).map(|a| (0..k).map(|b| p_gram_inv[(a, b)] * xhat_wy[b]).sum()).collect();
    let e_2sls: Vec<f64> = (0..n).map(|i| y_t[i] - (0..k).map(|a| beta_2sls[a] * x_struct[a][i]).sum::<f64>()).collect();

    let beta_r = solve_restricted(&beta_2sls, &p_gram_inv, r_mat, r_vec, q, k);
    let beta_star = if wb.impose_null { &beta_r } else { &beta_2sls };
    let e_star: Vec<f64> = (0..n).map(|i| y_t[i] - (0..k).map(|a| beta_star[a] * x_struct[a][i]).sum::<f64>()).collect();

    // Restricted/Unrestricted reduced form (Davidson & MacKinnon 2010 WRE/WUE):
    // regress each endogenous X1~ on [Z~, e_star].
    // Crucially, predicted Xhat* uses ONLY `Z~ * pi_tilde[..l]` (excluding e_star * gamma_u),
    // and e_endog = X1~ - Xhat* (so v_b scales `gamma_u * e_star + xi_2` together)!
    let mut rf_cols = z_full.clone();
    rf_cols.push(e_star.clone());
    let l_rf = rf_cols.len();
    let rfw = mat_from(l_rf, l_rf, |a, b| (0..n).map(|i| wv[i] * rf_cols[a][i] * rf_cols[b][i]).sum());
    let rfw_inv = inverse(&rfw);
    let mut e_endog = vec![vec![0.0; n]; m];
    let mut xhat_star = vec![vec![0.0; n]; m];
    for j in 0..m {
        let rhs: Vec<f64> = (0..l_rf).map(|a| (0..n).map(|i| wv[i] * rf_cols[a][i] * x1_t[j][i]).sum()).collect();
        let pi: Vec<f64> = (0..l_rf).map(|a| (0..l_rf).map(|b| rfw_inv[(a, b)] * rhs[b]).sum()).collect();
        for i in 0..n {
            let fit_z: f64 = (0..l).map(|c| z_full[c][i] * pi[c]).sum();
            xhat_star[j][i] = fit_z;
            e_endog[j][i] = x1_t[j][i] - fit_z;
        }
    }

    let z_coef = &zwz_inv * &zwx;
    let xhat_cols: Vec<Vec<f64>> = (0..k)
        .map(|b| (0..n).map(|i| (0..l).map(|a| z_full[a][i] * z_coef[(a, b)]).sum()).collect())
        .collect();

    let k_slopes = k - usize::from(explicit_cons);
    let df_a_val = if nested_in_err || explicit_cons { 0.0 } else { da.rank as f64 };
    let sdof = if explicit_cons { 0.0 } else { df_a_val.max(1.0) };
    let ssc = if let Some(dims) = err_dims {
        if iv_small {
            let g_min = dims.iter().map(|&(_, g)| g).min().unwrap() as f64;
            ((big_n - 1.0) / (big_n - k as f64 - sdof)) * (g_min / (g_min - 1.0))
        } else {
            1.0
        }
    } else if iv_small {
        big_n / (big_n - k as f64 - sdof)
    } else {
        big_n / (big_n - k_slopes as f64 - df_a_val)
    };

    let iv_cov = |xh: &[Vec<f64>], inv_gram: &Mat<f64>, resid: &[f64]| -> Mat<f64> {
        let mut meat = Mat::<f64>::zeros(k, k);
        if let Some(dims) = err_dims {
            let nd = dims.len();
            for mask in 1u32..(1u32 << nd) {
                let sign = if mask.count_ones() % 2 == 1 { 1.0 } else { -1.0 };
                let mut sums: HashMap<Vec<u32>, Vec<f64>> = HashMap::new();
                for i in 0..n {
                    let key: Vec<u32> = (0..nd).filter(|d| (mask >> d) & 1 == 1).map(|d| dims[d].0[i]).collect();
                    let ent = sums.entry(key).or_insert_with(|| vec![0.0; k]);
                    let sc = wv[i] * resid[i];
                    for c in 0..k {
                        ent[c] += sc * xh[c][i];
                    }
                }
                for sg in sums.values() {
                    for a in 0..k {
                        for b in 0..k {
                            meat[(a, b)] += sign * sg[a] * sg[b];
                        }
                    }
                }
            }
        } else {
            for i in 0..n {
                let s2 = (wv[i] * resid[i]).powi(2);
                for a in 0..k {
                    for b in 0..k {
                        meat[(a, b)] += s2 * xh[a][i] * xh[b][i];
                    }
                }
            }
        }
        let v = inv_gram * &meat * inv_gram;
        mat_from(k, k, |a, b| v[(a, b)] * ssc)
    };

    let v_orig = iv_cov(&xhat_cols, &p_gram_inv, &e_2sls);
    let (stat, se, point) = eval_stat(&beta_2sls, &v_orig, r_mat, r_vec, q, k);

    let draws: Vec<Vec<f64>> = if let Some((_, bg)) = boot_codes {
        oracle_cluster_weights(wb.weights, bg, wb.reps, wb.seed).0
    } else {
        let counts_row = vec![1usize; n];
        let d_row = oracle_hetero_weights_rows(wb.weights, &counts_row, wb.reps, wb.seed);
        d_row
            .into_iter()
            .map(|draw| {
                let mut v_kept = vec![0.0; n];
                for (row_idx, &ki_u32) in row_order.iter().enumerate() {
                    v_kept[ki_u32 as usize] = draw[row_idx][0];
                }
                v_kept
            })
            .collect()
    };

    let r_target: Vec<f64> = (0..q).map(|a| (0..k).map(|j| r_mat[a * k + j] * beta_star[j]).sum()).collect();
    let mut t_boot = Vec::with_capacity(draws.len());

    let y0_rf: Vec<f64> = (0..n)
        .map(|i| {
            let mut s = (0..kx).map(|j| beta_star[m + j] * x2_t[j][i]).sum::<f64>();
            for j in 0..m {
                s += beta_star[j] * xhat_star[j][i];
            }
            s
        })
        .collect();

    for v_arr in draws {
        let vi = |i: usize| -> f64 {
            if let Some((bc, _)) = boot_codes {
                v_arr[bc[i] as usize]
            } else {
                v_arr[i]
            }
        };
        let mut x_b = Vec::with_capacity(k);
        for j in 0..m {
            x_b.push((0..n).map(|i| xhat_star[j][i] + vi(i) * e_endog[j][i]).collect::<Vec<f64>>());
        }
        x_b.extend(x2_t.iter().cloned());
        let y_b: Vec<f64> = (0..n)
            .map(|i| {
                let mut s = y0_rf[i] + vi(i) * e_star[i];
                for j in 0..m {
                    s += beta_star[j] * vi(i) * e_endog[j][i];
                }
                s
            })
            .collect();
        let x_b_proj: Vec<Vec<f64>> = x_b.iter().map(|c| da.project_vec(c, &wv)).collect();
        let y_b_proj = da.project_vec(&y_b, &wv);

        let zwx_b = mat_from(l, k, |a, b| (0..n).map(|i| wv[i] * z_full[a][i] * x_b_proj[b][i]).sum());
        let xwz_b = mat_from(k, l, |a, b| zwx_b[(b, a)]);
        let p_b = &xwz_b * &zwz_inv * &zwx_b;
        let p_b_inv = inverse(&p_b);
        let zwy_b: Vec<f64> = (0..l).map(|a| (0..n).map(|i| wv[i] * z_full[a][i] * y_b_proj[i]).sum()).collect();
        let xhat_wy_b: Vec<f64> = (0..k)
            .map(|a| {
                let mut s = 0.0;
                for c in 0..l {
                    for d in 0..l {
                        s += xwz_b[(a, c)] * zwz_inv[(c, d)] * zwy_b[d];
                    }
                }
                s
            })
            .collect();
        let beta_b: Vec<f64> = (0..k).map(|a| (0..k).map(|b| p_b_inv[(a, b)] * xhat_wy_b[b]).sum()).collect();
        let resid_b: Vec<f64> = (0..n).map(|i| y_b_proj[i] - (0..k).map(|a| beta_b[a] * x_b_proj[a][i]).sum::<f64>()).collect();
        let zc_b = &zwz_inv * &zwx_b;
        let xhat_b: Vec<Vec<f64>> = (0..k)
            .map(|b| (0..n).map(|i| (0..l).map(|a| z_full[a][i] * zc_b[(a, b)]).sum()).collect())
            .collect();
        let v_b = iv_cov(&xhat_b, &p_b_inv, &resid_b);
        let (tb, _, _) = eval_stat(&beta_b, &v_b, r_mat, &r_target, q, k);
        t_boot.push(tb);
    }

    OracleOut { stat, se, point, t_boot }
}

// ---------------------------------------------------------------------------------------------
// Helpers to run Session and assert exact parity

fn assert_close(what: &str, got: f64, want: f64, rtol: f64, atol: f64) {
    if got.is_nan() && want.is_nan() {
        return;
    }
    let err = (got - want).abs();
    let lim = atol + rtol * got.abs().max(want.abs());
    assert!(err <= lim, "{what}: got {got:.15e}, want {want:.15e}, diff {err:.3e} > lim {lim:.3e}");
}

fn assert_oracle(what: &str, got: &WildBootResult, want: &OracleOut, tol: f64) {
    assert_close(&format!("{what} stat"), got.stat, want.stat, tol, tol);
    match (got.se, want.se) {
        (Some(gs), Some(ws)) => assert_close(&format!("{what} se"), gs, ws, tol, tol),
        (None, None) => {}
        _ => panic!("{what} se mismatch: {:?} vs {:?}", got.se, want.se),
    }
    match (got.point_estimate, want.point) {
        (Some(gp), Some(wp)) => assert_close(&format!("{what} point"), gp, wp, tol, tol),
        (None, None) => {}
        _ => panic!("{what} point mismatch: {:?} vs {:?}", got.point_estimate, want.point),
    }
    assert_eq!(got.t_boot.len(), want.t_boot.len(), "{what} t_boot len");
    for (b, (&gb, &wb)) in got.t_boot.iter().zip(&want.t_boot).enumerate() {
        assert_close(&format!("{what} t_boot[{b}]"), gb, wb, tol, tol);
    }
}

fn default_wb() -> WildBootSpec {
    WildBootSpec {
        reps: 29,
        weights: BootWeights::Rademacher,
        boot_type: BootType::T11,
        method: BootMethod::Auto,
        impose_null: true,
        pval_type: PvalType::TwoTailed,
        conf_int: true,
        level: 0.95,
        seed: 20261123,
        tol: 1e-7,
        maxiter: 50,
    }
}

#[allow(clippy::too_many_arguments)]
fn prep_and_run_ols(
    d: &SynthData,
    absorb: Vec<AbsorbTerm>,
    vce: Vce,
    w_kind: Option<WeightKind>,
    xs: Vec<&[f64]>,
    noconstant: bool,
    boot_cl_idx: Option<usize>,
    hyp: &Hypothesis,
    wb: &WildBootSpec,
    kind: AbsorberKind,
    threads: usize,
) -> (Session, WildBootResult, OracleOut) {
    let ctx = Ctx::quiet(threads);
    let w_slice = w_kind.map(|k| if k == WeightKind::Fweight { d.fw.as_slice() } else { d.aw.as_slice() });
    let prep = Prep {
        n: d.n,
        factors: vec![
            FactorValues::Num(&d.fe_nested), // 0
            FactorValues::Num(&d.fe_nn),     // 1
            FactorValues::Num(&d.fe_5),      // 2
            FactorValues::Num(&d.cl_4),      // 3
            FactorValues::Num(&d.cl_8),      // 4
            FactorValues::Num(&d.cl_12),     // 5
            FactorValues::Num(&d.cl_b),      // 6
        ],
        absorb,
        slope_vars: vec![&d.slope],
        weights: w_slice,
        weight_kind: w_kind.unwrap_or(WeightKind::None),
        vce,
        drop_singletons: true,
        family: Family::Gaussian,
        y_for_sep: None,
        sep_fe: false,
    };
    let mut s = Session::prepare_with(prep, &ctx, kind).unwrap();
    if let Some(bc) = boot_cl_idx {
        let f_ref = match bc {
            0 => &d.fe_nested,
            1 => &d.fe_nn,
            2 => &d.fe_5,
            3 => &d.cl_4,
            4 => &d.cl_8,
            5 => &d.cl_12,
            6 => &d.cl_b,
            _ => unreachable!(),
        };
        s.set_wb_cluster(&[FactorValues::Num(f_ref)], &ctx).unwrap();
    }
    let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
    let pick = |v: &[f64]| -> Vec<f64> { kept.iter().map(|&i| v[i]).collect() };
    let mut cols: Vec<Vec<f64>> = vec![pick(&d.y)];
    for x in &xs {
        cols.push(pick(x));
    }
    let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
    let rows = s.gather(&refs, &ctx).unwrap();
    let lay = Layout::ols(xs.len());
    let spec = Spec { tol: 1e-12, noconstant, ..Spec::default() };
    let (res, wb_res) = s.ols_with_wildboot(rows, &lay, &spec, hyp, wb, &ctx).unwrap();
    let df_a = res.scalar("df_a");

    let err_dims: Vec<(&[u32], usize)> = s.clusters().iter().map(|c| (c.codes.as_slice(), c.g)).collect();
    let kept_boot_vec: Option<(Vec<u32>, usize)> = boot_cl_idx.map(|bc| {
        let f_ref = match bc {
            3 => &d.cl_4,
            4 => &d.cl_8,
            5 => &d.cl_12,
            _ => unreachable!(),
        };
        let f_kept: Vec<f64> = kept.iter().map(|&i| f_ref[i]).collect();
        compact_sorted_codes(&f_kept)
    });
    let (bc_codes, bg): (&[u32], usize) = match &kept_boot_vec {
        Some((v, g)) => (v.as_slice(), *g),
        None if !s.clusters().is_empty() => (s.clusters()[0].codes.as_slice(), s.clusters()[0].g),
        None => (&[], 0),
    };

    let explicit_cons = s.terms().is_empty() && !noconstant;
    let k_kept = xs.len() + usize::from(explicit_cons);
    let r_full = expand_r_to_kept(&res.status, hyp, explicit_cons, xs.len());
    let oracle = if wb.method == BootMethod::Hetero || (wb.method == BootMethod::Auto && s.clusters().is_empty()) {
        oracle_ols_hetero(&s, &xs, &d.y, w_slice.zip(w_kind), noconstant, &r_full.0, &r_full.1, r_full.2, wb)
    } else {
        let xs_kept_only: Vec<&[f64]> = xs.iter().enumerate().filter(|&(j, _)| res.status[j] == 0).map(|(_, x)| *x).collect();
        let _ = k_kept;
        oracle_ols_cluster(
            &s,
            &xs_kept_only,
            &d.y,
            w_slice.zip(w_kind),
            noconstant,
            bc_codes,
            bg,
            &err_dims,
            &r_full.0,
            &r_full.1,
            r_full.2,
            wb,
            false,
            df_a,
        )
    };
    (s, wb_res, oracle)
}

fn expand_r_to_kept(status: &[u8], hyp: &Hypothesis, explicit_cons: bool, nx: usize) -> (Vec<f64>, Vec<f64>, usize) {
    let mut keep_idx: Vec<usize> = status.iter().enumerate().filter(|&(_, &st)| st == 0).map(|(j, _)| j).collect();
    if explicit_cons {
        keep_idx.push(nx);
    }
    let kk = keep_idx.len();
    let q = hyp.q;
    let mut r_out = Vec::with_capacity(q * kk);
    for a in 0..q {
        for &j in &keep_idx {
            r_out.push(hyp.r_mat[a * hyp.k + j]);
        }
    }
    (r_out, hyp.r_vec.clone(), q)
}

// ---------------------------------------------------------------------------------------------
// Tests

#[test]
fn test_ols_cluster_11_full_suite() {
    let d = gen_synth(120, 101);
    let t_nest = AbsorbTerm { factors: vec![0], intercept: true, slopes: vec![], save: false };
    let t_slope = AbsorbTerm { factors: vec![1], intercept: true, slopes: vec![0], save: false };

    for kind in [AbsorberKind::Fast, AbsorberKind::Dense] {
        // 1. noabsorb (slope & _cons, q=1 & q=2), WCR11 & WCU11
        let h_slope = Hypothesis::new(vec![1.0, 0.0, 0.0], vec![0.2], 1, 3).unwrap();
        let h_cons = Hypothesis::new(vec![0.0, 0.0, 1.0], vec![0.1], 1, 3).unwrap();
        let h_joint = Hypothesis::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0], 2, 3).unwrap();

        for imp in [true, false] {
            for (label, hyp) in [("slope", &h_slope), ("cons", &h_cons), ("joint", &h_joint)] {
                let mut wb = default_wb();
                wb.impose_null = imp;
                wb.weights = BootWeights::Rademacher;
                wb.reps = 999; // triggers full enum for cl_8 (2^8 = 256)
                let (_, got, want) = prep_and_run_ols(
                    &d,
                    vec![],
                    Vce::Cluster(vec![vec![4]]), // cl_8
                    None,
                    vec![&d.x1, &d.x2],
                    false,
                    None,
                    hyp,
                    &wb,
                    kind,
                    2,
                );
                assert_oracle(&format!("ols_noabsorb_{label}_imp{imp}_{kind:?}"), &got, &want, 1e-10);
            }
        }

        // 1b. noabsorb noconstant (explicit fix check)
        {
            let h_nc = Hypothesis::new(vec![1.0, 0.0], vec![0.1], 1, 2).unwrap();
            let mut wb = default_wb();
            wb.reps = 25;
            wb.method = BootMethod::Hetero;
            let (_, got, want) = prep_and_run_ols(&d, vec![], Vce::Robust, None, vec![&d.x1, &d.x2], true, None, &h_nc, &wb, kind, 2);
            assert_oracle(&format!("ols_noabsorb_noconst_hetero_{kind:?}"), &got, &want, 1e-10);

            wb.method = BootMethod::WildCluster;
            let (_, got_c, want_c) =
                prep_and_run_ols(&d, vec![], Vce::Cluster(vec![vec![4]]), None, vec![&d.x1, &d.x2], true, None, &h_nc, &wb, kind, 2);
            assert_oracle(&format!("ols_noabsorb_noconst_cl_{kind:?}"), &got_c, &want_c, 1e-10);
        }

        // 2. nested FE (df_a = 0 since t_nest is nested in cl_4; Session e(b) has 3 cols [x1, x2, _cons])
        let h2 = Hypothesis::new(vec![1.0, -0.5, 0.0], vec![0.1], 1, 3).unwrap();
        let h2_k2 = [1.0, -0.5];
        let mut wb = default_wb();
        wb.reps = 16; // cl_4 full enum
        let ctx = Ctx::quiet(2);
        let prep = Prep {
            n: d.n,
            factors: vec![FactorValues::Num(&d.fe_nested), FactorValues::Num(&d.cl_4)],
            absorb: vec![t_nest.clone()],
            slope_vars: vec![],
            weights: None,
            weight_kind: WeightKind::None,
            vce: Vce::Cluster(vec![vec![1]]),
            drop_singletons: true,
            family: Family::Gaussian,
            y_for_sep: None,
            sep_fe: false,
        };
        let mut s = Session::prepare_with(prep, &ctx, kind).unwrap();
        let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
        let cols: Vec<Vec<f64>> = vec![&d.y, &d.x1, &d.x2].into_iter().map(|v| kept.iter().map(|&i| v[i]).collect()).collect();
        let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
        let rows = s.gather(&refs, &ctx).unwrap();
        let (res, wb_res) = s.ols_with_wildboot(rows, &Layout::ols(2), &Spec { tol: 1e-12, ..Spec::default() }, &h2, &wb, &ctx).unwrap();
        let err_dims = [(s.clusters()[0].codes.as_slice(), s.clusters()[0].g)];
        let oracle = oracle_ols_cluster(&s, &[&d.x1, &d.x2], &d.y, None, false, err_dims[0].0, err_dims[0].1, &err_dims, &h2_k2, &h2.r_vec, 1, &wb, true, res.scalar("df_a"));
        assert_oracle(&format!("ols_nested_fe_{kind:?}"), &wb_res, &oracle, 1e-10);

        // 3. Non-nested FE Path 1 (G=12, m_nn=7) & Path 2 (G=4, m_nn=7)
        for (tag, cl_col) in [("path1_g12", 5), ("path2_g4", 3)] {
            let mut wb = default_wb();
            wb.reps = 24;
            wb.weights = BootWeights::Webb;
            let ctx = Ctx::quiet(2);
            let f_cl = if cl_col == 5 { &d.cl_12 } else { &d.cl_4 };
            let prep = Prep {
                n: d.n,
                factors: vec![FactorValues::Num(&d.fe_nn), FactorValues::Num(f_cl)],
                absorb: vec![AbsorbTerm { factors: vec![0], intercept: true, slopes: vec![], save: false }],
                slope_vars: vec![],
                weights: None,
                weight_kind: WeightKind::None,
                vce: Vce::Cluster(vec![vec![1]]),
                drop_singletons: true,
                family: Family::Gaussian,
                y_for_sep: None,
                sep_fe: false,
            };
            let mut s = Session::prepare_with(prep, &ctx, kind).unwrap();
            let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
            let cols: Vec<Vec<f64>> = vec![&d.y, &d.x1, &d.x2].into_iter().map(|v| kept.iter().map(|&i| v[i]).collect()).collect();
            let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
            let rows = s.gather(&refs, &ctx).unwrap();
            let (res, wb_res) = s.ols_with_wildboot(rows, &Layout::ols(2), &Spec { tol: 1e-12, ..Spec::default() }, &h2, &wb, &ctx).unwrap();
            let err_dims = [(s.clusters()[0].codes.as_slice(), s.clusters()[0].g)];
            let oracle = oracle_ols_cluster(&s, &[&d.x1, &d.x2], &d.y, None, false, err_dims[0].0, err_dims[0].1, &err_dims, &h2_k2, &h2.r_vec, 1, &wb, false, res.scalar("df_a"));
            assert_oracle(&format!("ols_{tag}_{kind:?}"), &wb_res, &oracle, 1e-10);
        }

        // 4. 2 FEs (nested + non-nested) + varying slopes + weighted (aw/pw/fw) + 2-way clustering
        let d2w = gen_synth(240, 109);
        for wk in [None, Some(WeightKind::Aweight), Some(WeightKind::Pweight), Some(WeightKind::Fweight)] {
            let mut wb = default_wb();
            wb.reps = 19;
            wb.weights = BootWeights::Mammen;
            let ctx = Ctx::quiet(2);
            let w_sl = wk.map(|k| if k == WeightKind::Fweight { d2w.fw.as_slice() } else { d2w.aw.as_slice() });
            let prep = Prep {
                n: d2w.n,
                factors: vec![
                    FactorValues::Num(&d2w.fe_nested),
                    FactorValues::Num(&d2w.fe_nn),
                    FactorValues::Num(&d2w.cl_8),
                    FactorValues::Num(&d2w.cl_b),
                ],
                absorb: vec![t_nest.clone(), t_slope.clone()],
                slope_vars: vec![&d2w.slope],
                weights: w_sl,
                weight_kind: wk.unwrap_or(WeightKind::None),
                vce: Vce::Cluster(vec![vec![2], vec![3]]),
                drop_singletons: true,
                family: Family::Gaussian,
                y_for_sep: None,
                sep_fe: false,
            };
            let mut s = Session::prepare_with(prep, &ctx, kind).unwrap();
            let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
            let cols: Vec<Vec<f64>> = vec![&d2w.y, &d2w.x1, &d2w.x2].into_iter().map(|v| kept.iter().map(|&i| v[i]).collect()).collect();
            let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
            let rows = s.gather(&refs, &ctx).unwrap();
            let (res, wb_res) = s.ols_with_wildboot(rows, &Layout::ols(2), &Spec { tol: 1e-12, ..Spec::default() }, &h2, &wb, &ctx).unwrap();
            assert!(wb_res.stat.is_finite());
            let err_dims: Vec<(&[u32], usize)> = s.clusters().iter().map(|c| (c.codes.as_slice(), c.g)).collect();

            let mut pairs: Vec<(u32, u32)> = (0..kept.len()).map(|i| (err_dims[0].0[i], err_dims[1].0[i])).collect();
            pairs.sort_unstable();
            pairs.dedup();
            let mut pair_map = HashMap::<(u32, u32), u32>::with_capacity(pairs.len());
            for (id, &p) in pairs.iter().enumerate() {
                pair_map.insert(p, id as u32);
            }
            let bcodes: Vec<u32> = (0..kept.len()).map(|i| pair_map[&(err_dims[0].0[i], err_dims[1].0[i])]).collect();

            let oracle = oracle_ols_cluster(&s, &[&d2w.x1, &d2w.x2], &d2w.y, w_sl.zip(wk), false, &bcodes, pairs.len(), &err_dims, &h2_k2, &h2.r_vec, 1, &wb, true, res.scalar("df_a"));
            assert_oracle(&format!("ols_twoway_{wk:?}_{kind:?}"), &wb_res, &oracle, 1e-10);
        }

        // 5. Collinear + absorbed column dropping via filter_hypothesis_by_kept + mismatched bootcluster
        let h_col = Hypothesis::new(vec![1.0, 0.0, 0.0, 0.0, 0.0], vec![0.0], 1, 5).unwrap();
        let mut wb = default_wb();
        wb.reps = 25;
        wb.weights = BootWeights::Normal;
        let (_, got, want) = prep_and_run_ols(
            &d,
            vec![t_nest.clone()],
            Vce::Cluster(vec![vec![5]]), // err cluster = cl_12
            None,
            vec![&d.x1, &d.x2, &d.x3_collinear, &d.x4_absorbed],
            false,
            Some(3), // bootcluster = cl_4
            &h_col,
            &wb,
            kind,
            2,
        );
        assert_oracle(&format!("ols_collinear_mismatched_bc_{kind:?}"), &got, &want, 1e-10);
    }
}

#[test]
fn test_ols_mnw_2023_all_types() {
    let d = gen_synth(110, 202);
    let t_nest = AbsorbTerm { factors: vec![0], intercept: true, slopes: vec![], save: false };
    let t_nn = AbsorbTerm { factors: vec![1], intercept: true, slopes: vec![], save: false };

    let h1 = Hypothesis::new(vec![1.0, -0.3, 0.0], vec![0.15], 1, 3).unwrap();
    let h2 = Hypothesis::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0], 2, 3).unwrap();
    let h1_k2 = vec![1.0, -0.3];
    let h2_k2 = vec![1.0, 0.0, 0.0, 1.0];

    for kind in [AbsorberKind::Fast, AbsorberKind::Dense] {
        for boot_type in [BootType::T13, BootType::T31, BootType::T33] {
            for imp in [true, false] {
                for (fe_label, absorb) in [("nofe", vec![]), ("nested", vec![t_nest.clone()]), ("nonnested", vec![t_nn.clone()])] {
                    let h = if imp { h1.clone() } else { h2.clone() };
                    let r_oracle = if absorb.is_empty() {
                        &h.r_mat
                    } else if imp {
                        &h1_k2
                    } else {
                        &h2_k2
                    };
                    let mut wb = default_wb();
                    wb.boot_type = boot_type;
                    wb.impose_null = imp;
                    wb.reps = 15;
                    let ctx = Ctx::quiet(2);
                    let prep = Prep {
                        n: d.n,
                        factors: vec![FactorValues::Num(&d.fe_nested), FactorValues::Num(&d.fe_nn), FactorValues::Num(&d.cl_4)],
                        absorb: absorb.clone(),
                        slope_vars: vec![],
                        weights: None,
                        weight_kind: WeightKind::None,
                        vce: Vce::Cluster(vec![vec![2]]),
                        drop_singletons: true,
                        family: Family::Gaussian,
                        y_for_sep: None,
                        sep_fe: false,
                    };
                    let mut s = Session::prepare_with(prep, &ctx, kind).unwrap();
                    let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
                    let cols: Vec<Vec<f64>> = vec![&d.y, &d.x1, &d.x2].into_iter().map(|v| kept.iter().map(|&i| v[i]).collect()).collect();
                    let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
                    let rows = s.gather(&refs, &ctx).unwrap();
                    let (res, got) = s.ols_with_wildboot(rows, &Layout::ols(2), &Spec { tol: 1e-12, ..Spec::default() }, &h, &wb, &ctx).unwrap();
                    let err_dims = [(s.clusters()[0].codes.as_slice(), s.clusters()[0].g)];
                    let oracle = oracle_ols_cluster(
                        &s,
                        &[&d.x1, &d.x2],
                        &d.y,
                        None,
                        false,
                        err_dims[0].0,
                        err_dims[0].1,
                        &err_dims,
                        r_oracle,
                        &h.r_vec,
                        h.q,
                        &wb,
                        fe_label == "nested",
                        res.scalar("df_a"),
                    );
                    assert_oracle(&format!("mnw_{boot_type:?}_{fe_label}_imp{imp}_{kind:?}"), &got, &oracle, 1e-9);
                }
            }
        }
    }
}

#[test]
fn test_ols_hetero_all_types_and_fw() {
    let d = gen_synth(95, 303);
    let t_oneway = AbsorbTerm { factors: vec![0], intercept: true, slopes: vec![], save: false };
    let t_nn = AbsorbTerm { factors: vec![1], intercept: true, slopes: vec![], save: false };
    let h1_k2 = vec![1.0, 0.0];
    let h2_k2 = vec![1.0, 0.0, 0.0, 1.0];

    for kind in [AbsorberKind::Fast, AbsorberKind::Dense] {
        for boot_type in [BootType::T11, BootType::T21, BootType::T31, BootType::T13, BootType::T33] {
            for imp in [true, false] {
                for (fe_label, absorb, wk) in [
                    ("nofe", vec![], None),
                    ("oneway", vec![t_oneway.clone()], Some(WeightKind::Aweight)),
                    ("twofe_fw", vec![t_oneway.clone(), t_nn.clone()], Some(WeightKind::Fweight)),
                ] {
                    let h = if imp {
                        Hypothesis::new(vec![1.0, 0.0, 0.0], vec![0.3], 1, 3).unwrap()
                    } else {
                        Hypothesis::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0], 2, 3).unwrap()
                    };
                    let r_oracle = if absorb.is_empty() {
                        &h.r_mat
                    } else if imp {
                        &h1_k2
                    } else {
                        &h2_k2
                    };
                    let mut wb = default_wb();
                    wb.boot_type = boot_type;
                    wb.method = BootMethod::Hetero;
                    wb.impose_null = imp;
                    wb.reps = 23;
                    let w_sl = wk.map(|k| if k == WeightKind::Fweight { d.fw.as_slice() } else { d.aw.as_slice() });
                    let ctx = Ctx::quiet(2);
                    let prep = Prep {
                        n: d.n,
                        factors: vec![FactorValues::Num(&d.fe_nested), FactorValues::Num(&d.fe_nn)],
                        absorb: absorb.clone(),
                        slope_vars: vec![],
                        weights: w_sl,
                        weight_kind: wk.unwrap_or(WeightKind::None),
                        vce: Vce::Robust,
                        drop_singletons: true,
                        family: Family::Gaussian,
                        y_for_sep: None,
                        sep_fe: false,
                    };
                    let mut s = Session::prepare_with(prep, &ctx, kind).unwrap();
                    let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
                    let cols: Vec<Vec<f64>> = vec![&d.y, &d.x1, &d.x2].into_iter().map(|v| kept.iter().map(|&i| v[i]).collect()).collect();
                    let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
                    let rows = s.gather(&refs, &ctx).unwrap();
                    let (_res, got) = s.ols_with_wildboot(rows, &Layout::ols(2), &Spec { tol: 1e-12, ..Spec::default() }, &h, &wb, &ctx).unwrap();
                    let oracle = oracle_ols_hetero(&s, &[&d.x1, &d.x2], &d.y, w_sl.zip(wk), false, r_oracle, &h.r_vec, h.q, &wb);
                    assert_oracle(&format!("hetero_{boot_type:?}_{fe_label}_imp{imp}_{kind:?}"), &got, &oracle, 1e-10);
                }
            }
        }
    }
}

#[test]
fn test_glm_score_poisson_and_logit() {
    let d = gen_synth(140, 404);
    let t_nest = AbsorbTerm { factors: vec![0], intercept: true, slopes: vec![], save: false };
    let t_nn = AbsorbTerm { factors: vec![1], intercept: true, slopes: vec![], save: false };

    for kind in [AbsorberKind::Fast, AbsorberKind::Dense] {
        for family in [Family::Poisson, Family::Logit] {
            let y_vec = if family == Family::Poisson { &d.y_pois } else { &d.y_logit };
            for (tag, absorb, cl_vec, has_cl, wk, off_opt) in [
                ("nofe_cons", vec![], Vce::Cluster(vec![vec![2]]), true, None, Some(d.offset.as_slice())),
                ("nested_aw", vec![t_nest.clone()], Vce::Cluster(vec![vec![2]]), true, Some(WeightKind::Aweight), None),
                ("nonnested_path1", vec![t_nn.clone()], Vce::Cluster(vec![vec![3]]), true, None, None),
                ("nonnested_path2", vec![t_nn.clone()], Vce::Cluster(vec![vec![2]]), true, None, None),
                ("unclustered", vec![t_nn.clone()], Vce::Robust, false, None, None),
            ] {
                for imp in [true, false] {
                    for q in [1usize, 2usize] {
                        let hyp = if absorb.is_empty() {
                            if q == 1 {
                                Hypothesis::new(vec![0.0, 0.0, 1.0], vec![0.05], 1, 3).unwrap()
                            } else {
                                Hypothesis::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0], 2, 3).unwrap()
                            }
                        } else if q == 1 {
                            Hypothesis::new(vec![1.0, 0.0, 0.0], vec![0.1], 1, 3).unwrap()
                        } else {
                            Hypothesis::new(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0], 2, 3).unwrap()
                        };
                        let r_oracle: Vec<f64> = if absorb.is_empty() {
                            hyp.r_mat.clone()
                        } else if q == 1 {
                            vec![1.0, 0.0]
                        } else {
                            vec![1.0, 0.0, 0.0, 1.0]
                        };
                        let mut wb = default_wb();
                        wb.reps = 21;
                        wb.impose_null = imp;
                        let w_sl = wk.map(|_| d.aw.as_slice());
                        let ctx = Ctx::quiet(2);
                        let prep = Prep {
                            n: d.n,
                            factors: vec![
                                FactorValues::Num(&d.fe_nested),
                                FactorValues::Num(&d.fe_nn),
                                FactorValues::Num(&d.cl_4),
                                FactorValues::Num(&d.cl_12),
                            ],
                            absorb: absorb.clone(),
                            slope_vars: vec![],
                            weights: w_sl,
                            weight_kind: wk.unwrap_or(WeightKind::None),
                            vce: cl_vec.clone(),
                            drop_singletons: true,
                            family,
                            y_for_sep: Some(y_vec),
                            sep_fe: true,
                        };
                        let mut s = Session::prepare_with(prep, &ctx, kind).unwrap();
                        let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
                        let pick = |v: &[f64]| -> Vec<f64> { kept.iter().map(|&i| v[i]).collect() };
                        let cols: Vec<Vec<f64>> = vec![y_vec.as_slice(), &d.x1, &d.x2]
                            .into_iter()
                            .map(pick)
                            .collect();
                        let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
                        let rows = s.gather(&refs, &ctx).unwrap();
                        let off_block = off_opt.map(|o| s.gather(&[&pick(o)], &ctx).unwrap());
                        let opts = GlmOptions {
                            family,
                            tol: 1e-15,
                            maxiter: 200,
                            itol: 1e-12,
                            separation: Separation { relu: false, mu: false },
                            guess: Guess::Simple,
                            log: false,
                            collinear: GlmCollinear::Ppmlhdfe,
                        };
                        let (_res, got) = s
                            .glm_with_wildboot(rows, &Layout::ols(2), off_block, &Spec { tol: 1e-12, ..Spec::default() }, &opts, &hyp, &wb, &ctx)
                            .unwrap();
                        let boot_c: Option<(&[u32], usize)> = if has_cl {
                            Some((s.clusters()[0].codes.as_slice(), s.clusters()[0].g))
                        } else {
                            None
                        };
                        let err_d: Option<Vec<(&[u32], usize)>> = boot_c.map(|c| vec![c]);
                        let oracle = oracle_glm_score(
                            &s,
                            family,
                            &[&d.x1, &d.x2],
                            y_vec,
                            off_opt,
                            w_sl.zip(wk),
                            false,
                            boot_c,
                            err_d.as_deref(),
                            &r_oracle,
                            &hyp.r_vec,
                            hyp.q,
                            &wb,
                        );
                        assert_oracle(&format!("glm_{family:?}_{tag}_imp{imp}_q{q}_{kind:?}"), &got, &oracle, 1e-8);
                    }
                }
            }
        }
    }
}

#[test]
fn test_iv_wre_and_score() {
    let d = gen_synth(130, 505);
    let t_nest = AbsorbTerm { factors: vec![0], intercept: true, slopes: vec![], save: false };
    let t_nn = AbsorbTerm { factors: vec![1], intercept: true, slopes: vec![], save: false };

    for kind in [AbsorberKind::Fast, AbsorberKind::Dense] {
        for (tag, absorb, cl_vce, endog_cols, exog_cols, inst_cols, is_nested, small, rep_cons) in [
            ("nofe_justid_repcons", vec![], Vce::Cluster(vec![vec![2]]), vec![&d.e1], vec![&d.x1], vec![&d.z1], false, false, true),
            ("nofe_abscons", vec![], Vce::Cluster(vec![vec![2]]), vec![&d.e1], vec![&d.x1], vec![&d.z1], false, false, false),
            ("nested_overid_small", vec![t_nest.clone()], Vce::Cluster(vec![vec![2]]), vec![&d.e1, &d.e2], vec![&d.x1, &d.x2], vec![&d.z1, &d.z2, &d.z3], true, true, false),
            ("nn_path1", vec![t_nn.clone()], Vce::Cluster(vec![vec![3]]), vec![&d.e1], vec![&d.x1, &d.x2], vec![&d.z1, &d.z2, &d.z3], false, true, false),
            ("nn_path2", vec![t_nn.clone()], Vce::Cluster(vec![vec![2]]), vec![&d.e1], vec![&d.x1, &d.x2], vec![&d.z1, &d.z2], false, false, false),
            ("unclustered", vec![t_nn.clone()], Vce::Robust, vec![&d.e1], vec![&d.x1, &d.x2], vec![&d.z1, &d.z2], false, true, false),
        ] {
            let k_rep = endog_cols.len() + exog_cols.len() + usize::from(rep_cons && absorb.is_empty());
            for imp in [true, false] {
                for q in [1usize, 2usize] {
                    if q > k_rep {
                        continue;
                    }
                    let mut r_mat = vec![0.0; q * k_rep];
                    r_mat[0] = 1.0;
                    if q == 2 {
                        r_mat[k_rep + 1] = 1.0;
                    }
                    let hyp = Hypothesis::new(r_mat, vec![0.1; q], q, k_rep).unwrap();
                    let mut wb = default_wb();
                    wb.reps = 19;
                    wb.impose_null = imp;
                    let ctx = Ctx::quiet(2);
                    let prep = Prep {
                        n: d.n,
                        factors: vec![
                            FactorValues::Num(&d.fe_nested),
                            FactorValues::Num(&d.fe_nn),
                            FactorValues::Num(&d.cl_4),
                            FactorValues::Num(&d.cl_12),
                        ],
                        absorb: absorb.clone(),
                        slope_vars: vec![],
                        weights: None,
                        weight_kind: WeightKind::None,
                        vce: cl_vce.clone(),
                        drop_singletons: true,
                        family: Family::Gaussian,
                        y_for_sep: None,
                        sep_fe: false,
                    };
                    let mut s = Session::prepare_with(prep, &ctx, kind).unwrap();
                    let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
                    let pick = |v: &[f64]| -> Vec<f64> { kept.iter().map(|&i| v[i]).collect() };
                    let mut cols: Vec<Vec<f64>> = vec![pick(&d.y_iv)];
                    for c in &exog_cols {
                        cols.push(pick(c));
                    }
                    for c in &endog_cols {
                        cols.push(pick(c));
                    }
                    for c in &inst_cols {
                        cols.push(pick(c));
                    }
                    let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
                    let rows = s.gather(&refs, &ctx).unwrap();
                    let lay = Layout { nx: exog_cols.len(), nendog: endog_cols.len(), ninst: inst_cols.len() };
                    let spec = Spec {
                        tol: 1e-12,
                        iv: IvOptions { small, report_constant: rep_cons, ..IvOptions::default() },
                        ..Spec::default()
                    };
                    let (_res, got) = s.iv_with_wildboot(rows, &lay, &spec, &hyp, &wb, &ctx).unwrap();
                    assert_eq!(got.method, BootMethod::Wre);

                    let boot_c = if s.clusters().is_empty() {
                        None
                    } else {
                        Some((s.clusters()[0].codes.as_slice(), s.clusters()[0].g))
                    };
                    let err_d = boot_c.map(|x| vec![x]);
                    let endog_refs: Vec<&[f64]> = endog_cols.iter().map(|c| c.as_slice()).collect();
                    let exog_refs: Vec<&[f64]> = exog_cols.iter().map(|c| c.as_slice()).collect();
                    let inst_refs: Vec<&[f64]> = inst_cols.iter().map(|c| c.as_slice()).collect();
                    // Wait: when absorb.is_empty() && !rep_cons, IV Session absorbed constant internally:
                    // represent that implicit constant via `DenseAbsorber` or noconstant=false?
                    // Look: oracle_iv_wre checks `s.terms().is_empty() && !noconstant`!
                    // When `!rep_cons && absorb.is_empty()`, constant is demeaned via a 1-group FE dummy in oracle too!
                    let oracle = oracle_iv_wre(
                        &s,
                        &d.y_iv,
                        &endog_refs,
                        &exog_refs,
                        &inst_refs,
                        None,
                        !rep_cons && absorb.is_empty(),
                        small,
                        boot_c,
                        err_d.as_deref(),
                        &hyp.r_mat,
                        &hyp.r_vec,
                        hyp.q,
                        &wb,
                        is_nested,
                    );
                    assert_oracle(&format!("iv_wre_{tag}_imp{imp}_q{q}_{kind:?}"), &got, &oracle, 1e-9);
                }
            }
        }
    }

    for bt in [BootType::T11, BootType::T13, BootType::T31, BootType::T33] {
        let mut wb = default_wb();
        wb.method = BootMethod::Score;
        wb.boot_type = bt;
        wb.impose_null = true;
        wb.conf_int = true;
        wb.reps = 19;
        let ctx = Ctx::quiet(2);
        let prep = Prep {
            n: d.n,
            factors: vec![FactorValues::Num(&d.fe_nn), FactorValues::Num(&d.cl_8)],
            absorb: vec![t_nn.clone()],
            slope_vars: vec![],
            weights: None,
            weight_kind: WeightKind::None,
            vce: Vce::Cluster(vec![vec![1]]),
            drop_singletons: true,
            family: Family::Gaussian,
            y_for_sep: None,
            sep_fe: false,
        };
        let mut s = Session::prepare_with(prep.clone(), &ctx, AbsorberKind::Fast).unwrap();
        let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
        let pick = |v: &[f64]| -> Vec<f64> { kept.iter().map(|&i| v[i]).collect() };
        let cols: Vec<Vec<f64>> = vec![pick(&d.y_iv), pick(&d.x1), pick(&d.e1), pick(&d.z1), pick(&d.z2)];
        let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
        let rows = s.gather(&refs, &ctx).unwrap();
        let hyp = Hypothesis::new(vec![1.0, 0.0], vec![0.0], 1, 2).unwrap();
        let (res, got) = s
            .iv_with_wildboot(rows, &Layout { nx: 1, nendog: 1, ninst: 2 }, &Spec { tol: 1e-12, ..Spec::default() }, &hyp, &wb, &ctx)
            .unwrap();
        assert_eq!(got.method, BootMethod::Score);
        assert!(matches!(got.stat_type, StatType::ScoreT));
        assert!(got.stat.is_finite() && got.stat.abs() > 0.1, "iv score stat collapsed for {bt:?}: {}", got.stat);
        assert!(got.p_val >= 0.0 && got.p_val <= 1.0 && got.conf_int.is_some());
        if matches!(bt, BootType::T11 | BootType::T31) {
            let b0 = res.matrix("b").unwrap().data[0];
            let se0 = res.matrix("V").unwrap().data[0].sqrt();
            let wald_t = b0 / se0;
            assert!((got.stat - wald_t).abs() < 1e-10, "iv score stat mismatch for {bt:?}: {} vs {wald_t}", got.stat);
            assert!((got.se.unwrap() - se0).abs() < 1e-10);
        }

        // Also regression-test OLS clustered score bootstrap with impose_null + conf_int.
        let mut s_ols = Session::prepare_with(prep, &ctx, AbsorberKind::Fast).unwrap();
        let kept_o: Vec<usize> = s_ols.kept_index().iter().map(|&i| i as usize).collect();
        let pick_o = |v: &[f64]| -> Vec<f64> { kept_o.iter().map(|&i| v[i]).collect() };
        let cols_o: Vec<Vec<f64>> = vec![pick_o(&d.y), pick_o(&d.x1), pick_o(&d.x2)];
        let refs_o: Vec<&[f64]> = cols_o.iter().map(Vec::as_slice).collect();
        let rows_o = s_ols.gather(&refs_o, &ctx).unwrap();
        let hyp_o = Hypothesis::new(vec![1.0, 0.0, 0.0], vec![0.0], 1, 3).unwrap();
        let (res_o, got_o) = s_ols
            .ols_with_wildboot(rows_o, &Layout::ols(2), &Spec { tol: 1e-12, ..Spec::default() }, &hyp_o, &wb, &ctx)
            .unwrap();
        assert_eq!(got_o.method, BootMethod::Score);
        assert!(matches!(got_o.stat_type, StatType::ScoreT));
        assert!(got_o.stat.is_finite() && got_o.stat.abs() > 0.1, "ols score stat collapsed for {bt:?}: {}", got_o.stat);
        assert!(got_o.conf_int.is_some());
        if matches!(bt, BootType::T11 | BootType::T31) {
            let b0 = res_o.matrix("b").unwrap().data[0];
            let se0 = res_o.matrix("V").unwrap().data[0].sqrt();
            let wald_t = b0 / se0;
            assert!((got_o.stat - wald_t).abs() < 1e-10, "ols score stat mismatch for {bt:?}: {} vs {wald_t}", got_o.stat);
            assert!((got_o.se.unwrap() - se0).abs() < 1e-10);
        }
    }
}

#[test]
fn test_ci_inversion_and_pvalues_and_weights() {
    let d = gen_synth(120, 606);
    let t_nn = AbsorbTerm { factors: vec![1], intercept: true, slopes: vec![], save: false };

    let eval_p = |run_fn: &dyn Fn(f64) -> WildBootResult, r0: f64| -> f64 { run_fn(r0).p_val };

    for (bt, meth, cl) in [
        (BootType::T11, BootMethod::WildCluster, Some(vec![vec![1]])),
        (BootType::T31, BootMethod::WildCluster, Some(vec![vec![1]])),
        (BootType::T33, BootMethod::WildCluster, Some(vec![vec![1]])),
        (BootType::T21, BootMethod::Hetero, None),
    ] {
        for ptype in [PvalType::TwoTailed, PvalType::EqualTailed, PvalType::Greater, PvalType::Less] {
            let mut wb = default_wb();
            wb.boot_type = bt;
            wb.method = meth;
            wb.pval_type = ptype;
            wb.reps = 99;
            wb.level = 0.90;
            wb.tol = 1e-7;
            let run_r = |r0: f64| -> WildBootResult {
                let ctx = Ctx::quiet(2);
                let prep = Prep {
                    n: d.n,
                    factors: vec![FactorValues::Num(&d.fe_nn), FactorValues::Num(&d.cl_8)],
                    absorb: vec![t_nn.clone()],
                    slope_vars: vec![],
                    weights: None,
                    weight_kind: WeightKind::None,
                    vce: cl.clone().map(Vce::Cluster).unwrap_or(Vce::Robust),
                    drop_singletons: true,
                    family: Family::Gaussian,
                    y_for_sep: None,
                    sep_fe: false,
                };
                let mut s = Session::prepare_with(prep, &ctx, AbsorberKind::Fast).unwrap();
                let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
                let pick = |v: &[f64]| -> Vec<f64> { kept.iter().map(|&i| v[i]).collect() };
                let cols: Vec<Vec<f64>> = vec![pick(&d.y), pick(&d.x1), pick(&d.x2)];
                let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
                let rows = s.gather(&refs, &ctx).unwrap();
            let hyp = Hypothesis::new(vec![1.0, 0.0, 0.0], vec![r0], 1, 3).unwrap();
                s.ols_with_wildboot(rows, &Layout::ols(2), &Spec { tol: 1e-12, ..Spec::default() }, &hyp, &wb, &ctx).unwrap().1
            };
            let res0 = run_r(0.0);
            let (lo, hi) = res0.conf_int.unwrap();
            let alpha = 1.0 - wb.level;
            let bound = 2.0 / wb.reps as f64 + 1e-9;
            if lo.is_finite() {
                let p_lo = eval_p(&run_r, lo);
                assert!((p_lo - alpha).abs() <= bound, "{bt:?} {ptype:?}: p(lo={lo})={p_lo} vs alpha={alpha}");
            }
            if hi.is_finite() {
                let p_hi = eval_p(&run_r, hi);
                assert!((p_hi - alpha).abs() <= bound, "{bt:?} {ptype:?}: p(hi={hi})={p_hi} vs alpha={alpha}");
            }
        }
    }

    for fam in [Family::Poisson, Family::Logit] {
        let y_v = if fam == Family::Poisson { &d.y_pois } else { &d.y_logit };
        let mut wb = default_wb();
        wb.reps = 99;
        wb.level = 0.90;
        let run_glm = |r0: f64| -> WildBootResult {
            let ctx = Ctx::quiet(2);
            let prep = Prep {
                n: d.n,
                factors: vec![FactorValues::Num(&d.fe_nn), FactorValues::Num(&d.cl_8)],
                absorb: vec![t_nn.clone()],
                slope_vars: vec![],
                weights: None,
                weight_kind: WeightKind::None,
                vce: Vce::Cluster(vec![vec![1]]),
                drop_singletons: true,
                family: fam,
                y_for_sep: Some(y_v),
                sep_fe: true,
            };
            let mut s = Session::prepare_with(prep, &ctx, AbsorberKind::Fast).unwrap();
            let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
            let pick = |v: &[f64]| -> Vec<f64> { kept.iter().map(|&i| v[i]).collect() };
            let cols = vec![pick(y_v), pick(&d.x1), pick(&d.x2)];
            let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
            let rows = s.gather(&refs, &ctx).unwrap();
            let opts = GlmOptions {
                family: fam,
                tol: 1e-12,
                maxiter: 200,
                itol: 1e-10,
                separation: Separation { relu: false, mu: false },
                guess: Guess::Simple,
                log: false,
                collinear: GlmCollinear::Ppmlhdfe,
            };
            let hyp = Hypothesis::new(vec![1.0, 0.0, 0.0], vec![r0], 1, 3).unwrap();
            s.glm_with_wildboot(rows, &Layout::ols(2), None, &Spec { tol: 1e-12, ..Spec::default() }, &opts, &hyp, &wb, &ctx)
                .unwrap()
                .1
        };
        let (lo, hi) = run_glm(0.0).conf_int.unwrap();
        let alpha = 1.0 - wb.level;
        let bound = 2.0 / wb.reps as f64 + 1e-9;
        assert!((run_glm(lo).p_val - alpha).abs() <= bound, "glm {fam:?}: lo");
        assert!((run_glm(hi).p_val - alpha).abs() <= bound, "glm {fam:?}: hi");
    }

    {
        let mut wb = default_wb();
        wb.reps = 99;
        wb.level = 0.90;
        let run_iv = |r0: f64| -> WildBootResult {
            let ctx = Ctx::quiet(2);
            let prep = Prep {
                n: d.n,
                factors: vec![FactorValues::Num(&d.fe_nn), FactorValues::Num(&d.cl_8)],
                absorb: vec![t_nn.clone()],
                slope_vars: vec![],
                weights: None,
                weight_kind: WeightKind::None,
                vce: Vce::Cluster(vec![vec![1]]),
                drop_singletons: true,
                family: Family::Gaussian,
                y_for_sep: None,
                sep_fe: false,
            };
            let mut s = Session::prepare_with(prep, &ctx, AbsorberKind::Fast).unwrap();
            let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
            let pick = |v: &[f64]| -> Vec<f64> { kept.iter().map(|&i| v[i]).collect() };
            let cols = vec![pick(&d.y_iv), pick(&d.x1), pick(&d.e1), pick(&d.z1), pick(&d.z2)];
            let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
            let rows = s.gather(&refs, &ctx).unwrap();
            let hyp = Hypothesis::new(vec![1.0, 0.0], vec![r0], 1, 2).unwrap();
            s.iv_with_wildboot(rows, &Layout { nx: 1, nendog: 1, ninst: 2 }, &Spec { tol: 1e-12, ..Spec::default() }, &hyp, &wb, &ctx)
                .unwrap()
                .1
        };
        let (lo, hi) = run_iv(0.0).conf_int.unwrap();
        let alpha = 1.0 - wb.level;
        let bound = 2.0 / wb.reps as f64 + 1e-9;
        assert!((run_iv(lo).p_val - alpha).abs() <= bound, "iv wre lo");
        assert!((run_iv(hi).p_val - alpha).abs() <= bound, "iv wre hi");
    }

    {
        let mut wb = default_wb();
        wb.impose_null = false;
        wb.reps = 999;
        wb.weights = BootWeights::Rademacher;
        let (_, wcu_cl8, _) = prep_and_run_ols(
            &d,
            vec![t_nn.clone()],
            Vce::Cluster(vec![vec![4]]),
            None,
            vec![&d.x1, &d.x2],
            false,
            None,
            &Hypothesis::new(vec![1.0, 0.0, 0.0], vec![0.0], 1, 3).unwrap(),
            &wb,
            AbsorberKind::Fast,
            2,
        );
        assert!(wcu_cl8.full_enumeration);
        assert_eq!(wcu_cl8.reps_feasible, 256);
        assert_eq!(wcu_cl8.n_clusters, Some(8));
        let (lo, hi) = wcu_cl8.conf_int.unwrap();
        let pt = wcu_cl8.point_estimate.unwrap();
        assert!(lo < pt && pt < hi);

        let cl_3: Vec<f64> = (0..d.n).map(|i| (i % 3) as f64).collect();
        let ctx = Ctx::quiet(2);
        let prep = Prep {
            n: d.n,
            factors: vec![FactorValues::Num(&cl_3)],
            absorb: vec![],
            slope_vars: vec![],
            weights: None,
            weight_kind: WeightKind::None,
            vce: Vce::Cluster(vec![vec![0]]),
            drop_singletons: true,
            family: Family::Gaussian,
            y_for_sep: None,
            sep_fe: false,
        };
        let mut s = Session::prepare_with(prep, &ctx, AbsorberKind::Fast).unwrap();
        let cols: Vec<Vec<f64>> = vec![d.y.clone(), d.x1.clone()];
        let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
        let rows = s.gather(&refs, &ctx).unwrap();
        let mut wb_webb = default_wb();
        wb_webb.weights = BootWeights::Webb;
        wb_webb.reps = 500;
        let (_r, wb_res) = s
            .ols_with_wildboot(
                rows,
                &Layout::ols(1),
                &Spec { tol: 1e-12, ..Spec::default() },
                &Hypothesis::new(vec![1.0, 0.0], vec![0.0], 1, 2).unwrap(),
                &wb_webb,
                &ctx,
            )
            .unwrap();
        assert!(wb_res.full_enumeration);
        assert_eq!(wb_res.reps_feasible, 216);
    }
}

#[test]
fn test_thread_bit_exactness_and_fit_wildboot() {
    let d = gen_synth(120, 707);
    let t_nn = AbsorbTerm { factors: vec![1], intercept: true, slopes: vec![], save: false };

    for (label, meth, bt) in [
        ("ols_cl", BootMethod::WildCluster, BootType::T33),
        ("ols_het", BootMethod::Hetero, BootType::T21),
    ] {
        let mut wb = default_wb();
        wb.method = meth;
        wb.boot_type = bt;
        wb.reps = 25;
        let vce = if meth == BootMethod::Hetero { Vce::Robust } else { Vce::Cluster(vec![vec![4]]) };
        let (_, r1, _) = prep_and_run_ols(&d, vec![t_nn.clone()], vce.clone(), None, vec![&d.x1, &d.x2], false, None, &Hypothesis::new(vec![1.0, 0.0, 0.0], vec![0.0], 1, 3).unwrap(), &wb, AbsorberKind::Fast, 1);
        let (_, r4, _) = prep_and_run_ols(&d, vec![t_nn.clone()], vce, None, vec![&d.x1, &d.x2], false, None, &Hypothesis::new(vec![1.0, 0.0, 0.0], vec![0.0], 1, 3).unwrap(), &wb, AbsorberKind::Fast, 4);
        assert_eq!(r1.stat.to_bits(), r4.stat.to_bits(), "{label} stat bits");
        assert_eq!(r1.p_val.to_bits(), r4.p_val.to_bits(), "{label} p_val bits");
        assert_eq!(r1.t_boot.len(), r4.t_boot.len());
        for (a, b) in r1.t_boot.iter().zip(&r4.t_boot) {
            assert_eq!(a.to_bits(), b.to_bits(), "{label} t_boot bits");
        }
    }

    let input = FitInput {
        n: d.n,
        y: &d.y,
        x: vec![&d.x1, &d.x2],
        endog: vec![],
        inst: vec![],
        factors: vec![FactorCol::F64(&d.fe_nn), FactorCol::F64(&d.cl_8), FactorCol::F64(&d.cl_4)],
        factor_names: vec!["fe_nn".into(), "cl_8".into(), "cl_4".into()],
        fe: vec![FeTerm { factors: vec![0], intercept: true, slopes: vec![] }],
        slopes: vec![],
        weights: None,
        offset: None,
        cluster: vec![vec![1]],
        ..Default::default()
    };
    for flavor in [Flavor::Fixest, Flavor::Pyfixest] {
        let fspec = FitSpec {
            model: Model::Ols,
            flavor,
            intercept: true,
            vcov: VcovKind::Cluster,
            ssc: Ssc {
                k_adj: true,
                k_fixef: KFixef::Nonnested,
                k_exact: false,
                g_adj: true,
                g_df: GDf::Min,
                t_df: TDf::Min,
            },
            weights_type: WeightsType::Aweights,
            fixef_tol: 1e-12,
            glm_tol: 1e-12,
            glm_maxiter: 100,
            ..Default::default()
        };
        let wb_opts = WildBootOptions {
            hyp: Hypothesis::new(vec![1.0, 0.0], vec![0.0], 1, 2).unwrap(),
            spec: WildBootSpec { reps: 39, weights: BootWeights::Webb, ..WildBootSpec::default() },
            bootcluster: Some(vec![2]),
        };
        let out = fit::wildboot(&input, &fspec, &wb_opts, &Ctx::quiet(2)).unwrap();
        assert_eq!(out.result.n_clusters, Some(4));
        assert_eq!(out.result.method, BootMethod::WildCluster);
        assert!(out.result.conf_int.is_some());

        let iv_input = FitInput {
            y: &d.y_iv,
            x: vec![&d.x1, &d.x2],
            endog: vec![&d.e1],
            inst: vec![&d.z1, &d.z2],
            ..input.clone()
        };
        let iv_opts = WildBootOptions {
            hyp: Hypothesis::new(vec![0.0, 0.0, 1.0], vec![0.0], 1, 3).unwrap(),
            spec: WildBootSpec { reps: 29, ..WildBootSpec::default() },
            bootcluster: None,
        };
        let iv_out = fit::wildboot(&iv_input, &fspec, &iv_opts, &Ctx::quiet(2)).unwrap();
        assert_eq!(iv_out.coef_source, vec![CoefSource::X(0), CoefSource::X(1), CoefSource::Endog(0)]);
        assert_eq!(iv_out.result.method, BootMethod::Wre);

        for (model, yv) in [(Model::Poisson, &d.y_pois), (Model::Logit, &d.y_logit)] {
            let glm_input = FitInput { y: yv, ..input.clone() };
            let glm_spec = FitSpec { model, ..fspec.clone() };
            let glm_out = fit::wildboot(&glm_input, &glm_spec, &wb_opts, &Ctx::quiet(2)).unwrap();
            assert_eq!(glm_out.result.method, BootMethod::Score);
            assert!(matches!(glm_out.result.stat_type, StatType::ScoreT));
        }
    }
}
