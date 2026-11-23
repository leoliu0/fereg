//! Fast wild bootstrap engines for OLS: clustered WCR/WCU (incl. MNW 2023 `13/31/33` and exact
//! non-nested HDFE projection) and unclustered heteroskedastic `HC1`/`HC2`/`HC3` (`11/21/31/13/33`).

use rayon::prelude::*;

use super::weights::{cluster_matrix, fill_from_rng, hetero_chunk_rng, resolve_reps};
use super::{
    BootMethod, BootType, Cells, ClusterConfig, Hypothesis, NonNestedPlan, StatType, WildBootResult, WildBootSpec, accumulate_dnn_groups, classify_non_nested,
    compute_pvalue, invert_ci_search, invert_ci_wcu, path2_project_matrices, quad_inv_sym, sym_pinv, to_kept_u32,
};
use crate::absorber::Absorber;
use crate::api::Ctx;
use crate::error::RfError;
use crate::leverage::{FeCols, fe_hat};
use crate::par::{self, CHUNK, Pool};
use crate::session::Term;
use crate::vcov::to_row_order;

/// Inputs for OLS wild bootstrap inference (residuals/design/clusters in solver row order).
pub(crate) struct OlsBootInput<'a> {
    pub n: usize,
    pub k: usize,
    /// Residualized design `X_tilde` (`n * k`, row-major).
    pub x: &'a [f64],
    /// Unrestricted residuals `u_hat = y_tilde - X_tilde * beta` (length `n`).
    pub resid: &'a [f64],
    /// Unrestricted coefficient vector (`k`).
    pub beta: &'a [f64],
    /// `(X_tilde' W X_tilde)^{-1}` (`k * k`, row-major).
    pub bread: &'a [f64],
    /// Regression weights `w_v` in solver row order (for `fw`, integer multiplicities).
    pub w_rows: Option<&'a [f64]>,
    pub is_fweight: bool,
    pub terms: &'a [Term],
    pub row_order: &'a [u32],
    /// Heteroskedastic small-sample multiplier `big_n / df_r` (`1.0` when unadjusted).
    pub ssc_hetero: f64,
    pub hyp: &'a Hypothesis,
    pub spec: &'a WildBootSpec,
    pub cluster: Option<&'a ClusterConfig<'a>>,
    pub fixef_tol: f64,
    pub fixef_maxiter: usize,
}

/// Inputs for unclustered heteroskedastic wild bootstrap (`ols` and unclustered one-step `score`).
pub(crate) struct HeteroInput<'a> {
    pub n: usize,
    pub k: usize,
    pub x: &'a [f64],
    /// Bootstrap residual / working residual (`n`).
    pub resid: &'a [f64],
    /// Optional residual for observed variance `V_obs` (differs from `resid` in restricted score).
    pub obs_resid: Option<&'a [f64]>,
    pub bread: &'a [f64],
    pub w_rows: Option<&'a [f64]>,
    pub is_fweight: bool,
    pub terms: &'a [Term],
    pub row_order: &'a [u32],
    pub ssc_factor: f64,
    /// Numerator of the observed statistic (`q`): `R * beta - r` or restricted score `R * d_beta^(1)`.
    pub num_obs: Vec<f64>,
    /// Unrestricted `point = R * beta` (when `q == 1`, used for WCU CI and bracket center).
    pub point: Option<f64>,
    pub allow_impose_null_affine: bool,
    pub dgp_pow: f64,
    pub denom_pow: f64,
    pub hyp: &'a Hypothesis,
    pub spec: &'a WildBootSpec,
    pub stat_type_scalar: StatType,
    pub stat_type_joint: StatType,
    pub method: BootMethod,
    pub notes: Vec<String>,
}

/// Precomputed linear hypothesis projection matrices from `bread` and `R` (`q x k`).
pub(crate) struct ConstraintProj {
    /// `C = bread * R'` (`k * q`, row-major).
    pub c_mat: Vec<f64>,
    /// `M_R = C * (R * bread * R')^+` (`k * q`, row-major).
    pub m_r: Vec<f64>,
}

impl ConstraintProj {
    pub(crate) fn new(bread: &[f64], r_mat: &[f64], q: usize, k: usize) -> Self {
        let mut c_mat = vec![0.0; k * q];
        for i in 0..k {
            for a in 0..q {
                let mut s = 0.0;
                for j in 0..k {
                    s += bread[i * k + j] * r_mat[a * k + j];
                }
                c_mat[i * q + a] = s;
            }
        }
        let mut v0 = vec![0.0; q * q];
        for a in 0..q {
            for b in a..q {
                let mut s = 0.0;
                for j in 0..k {
                    s += r_mat[a * k + j] * c_mat[j * q + b];
                }
                v0[a * q + b] = s;
                v0[b * q + a] = s;
            }
        }
        let v0_inv = sym_pinv(&v0, q, 1e-12);
        let mut m_r = vec![0.0; k * q];
        for i in 0..k {
            for a in 0..q {
                let mut s = 0.0;
                for b in 0..q {
                    s += c_mat[i * q + b] * v0_inv[b * q + a];
                }
                m_r[i * q + a] = s;
            }
        }
        Self { c_mat, m_r }
    }
}

/// Evaluates `stat` (`t` when `q == 1`, `F = quad / q` when `q > 1`) from numerator and `V`.
#[inline]
pub(crate) fn eval_stat_and_se(num: &[f64], v: &[f64], q: usize) -> (f64, Option<f64>) {
    if q == 1 {
        let vv = v[0];
        if !vv.is_finite() || vv <= 1e-300 {
            return (f64::NAN, None);
        }
        let se = vv.sqrt();
        (num[0] / se, Some(se))
    } else {
        let qf = quad_inv_sym(num, v, q);
        (if qf.is_finite() { qf / q as f64 } else { f64::NAN }, None)
    }
}

/// Evaluates scalar bootstrap `t*(delta)` from affine components `(n0, n1, d00, d01, d11)`.
#[inline]
pub(crate) fn eval_affine_t(delta: f64, n0: f64, n1: f64, d00: f64, d01: f64, d11: f64) -> f64 {
    let v = d00 + delta * (2.0 * d01 + delta * d11);
    if !v.is_finite() || v <= 1e-300 { f64::NAN } else { (n0 + delta * n1) / v.sqrt() }
}

/// Matrix-vector product `y = A * x` where `A` is `m x n` row-major.
#[inline]
pub(crate) fn matvec(a: &[f64], x: &[f64], m: usize, n: usize, out: &mut [f64]) {
    for i in 0..m {
        let row = &a[i * n..(i + 1) * n];
        let mut s = 0.0;
        for j in 0..n {
            s += row[j] * x[j];
        }
        out[i] = s;
    }
}

/// Computes `R * beta - r` and scalar `point = R[0..k] * beta` when `q == 1`.
pub(crate) fn linear_num_obs(beta: &[f64], hyp: &Hypothesis) -> (Vec<f64>, Option<f64>) {
    let (q, k) = (hyp.q, hyp.k);
    let mut num = vec![0.0; q];
    for a in 0..q {
        let mut s = -hyp.r_vec[a];
        for j in 0..k {
            s += hyp.r_mat[a * k + j] * beta[j];
        }
        num[a] = s;
    }
    let point = (q == 1).then(|| num[0] + hyp.r_vec[0]);
    (num, point)
}

/// Dispatches OLS wild bootstrap inference.
pub(crate) fn run(input: OlsBootInput<'_>, absorber: Option<&mut dyn Absorber>, ctx: &Ctx, pool: &Pool) -> Result<WildBootResult, RfError> {
    input.spec.validate(input.hyp.q)?;
    if input.hyp.k != input.k {
        return Err(RfError::InvalidInput("wildboot: hypothesis column count does not match kept regressors".into()));
    }
    let method = match (input.spec.method, input.cluster.is_some()) {
        (BootMethod::Auto, true) => BootMethod::WildCluster,
        (BootMethod::Auto, false) => BootMethod::Hetero,
        (BootMethod::WildCluster, false) => {
            return Err(RfError::InvalidInput("wildboot: method(wildcluster) requires cluster variables".into()));
        }
        (BootMethod::Wre, _) => return Err(RfError::InvalidInput("wildboot: WRE method requires an IV/2SLS estimation".into())),
        (m, _) => m,
    };

    if method == BootMethod::Score {
        return super::score::linear_score_ols(input, absorber, ctx, pool);
    }

    if method == BootMethod::Hetero || input.cluster.is_none() {
        let (dgp_pow, denom_pow) = match input.spec.boot_type {
            BootType::T11 => (0.0, 0.0),
            BootType::T21 => (0.5, 0.0),
            BootType::T31 => (1.0, 0.0),
            BootType::T13 => (0.0, 1.0),
            BootType::T33 => (1.0, 1.0),
        };
        let (num_obs, point) = linear_num_obs(input.beta, input.hyp);
        let hi = HeteroInput {
            n: input.n,
            k: input.k,
            x: input.x,
            resid: input.resid,
            obs_resid: None,
            bread: input.bread,
            w_rows: input.w_rows,
            is_fweight: input.is_fweight,
            terms: input.terms,
            row_order: input.row_order,
            ssc_factor: input.ssc_hetero,
            num_obs,
            point,
            allow_impose_null_affine: true,
            dgp_pow,
            denom_pow,
            hyp: input.hyp,
            spec: input.spec,
            stat_type_scalar: StatType::T,
            stat_type_joint: StatType::F,
            method: BootMethod::Hetero,
            notes: Vec::new(),
        };
        return run_hetero(hi, ctx, pool);
    }

    let cl = input.cluster.expect("checked");
    if cl.g_boot < 2 {
        return Err(RfError::InvalidInput("wildboot: cluster bootstrap requires at least 2 clusters".into()));
    }
    if input.spec.boot_type == BootType::T21 {
        return Err(RfError::InvalidInput("wildboot: bootstrap type 21 is only defined for unclustered heteroskedastic OLS".into()));
    }
    if input.spec.boot_type == BootType::T11 {
        run_cluster_11(input, absorber, cl, ctx, pool)
    } else {
        run_cluster_mnw(input, cl, ctx, pool)
    }
}

/// Fast clustered OLS WCR11 / WCU11 with arbitrary `Cells` and exact non-nested FE projection.
fn run_cluster_11(
    input: OlsBootInput<'_>,
    mut absorber: Option<&mut dyn Absorber>,
    cl: &ClusterConfig<'_>,
    ctx: &Ctx,
    pool: &Pool,
) -> Result<WildBootResult, RfError> {
    let (n, k, q) = (input.n, input.k, input.hyp.q);
    let g_boot = cl.g_boot;
    let cells = Cells::build(cl)?;
    let cell_kept = if cl.err_equals_boot { None } else { Some(to_kept_u32(&cells.cell_of_row, input.row_order)) };
    let mut notes = Vec::new();
    let plan = classify_non_nested(
        input.terms,
        input.row_order,
        cl.boot_kept,
        cell_kept.as_deref(),
        g_boot,
        input.w_rows,
        false,
        input.fixef_tol,
        input.fixef_maxiter,
        pool,
        ctx,
    )?;

    let (reps, full_enum) = resolve_reps(input.spec.weights, input.spec.reps, Some(g_boot));
    let v_mat = cluster_matrix(input.spec.weights, g_boot, reps, full_enum, input.spec.seed, pool)?;

    let proj = ConstraintProj::new(input.bread, &input.hyp.r_mat, q, k);
    let (num_obs, point) = linear_num_obs(input.beta, input.hyp);
    let affine_ci = q == 1 && input.spec.impose_null && input.spec.conf_int;

    // Construct target vectors in row order:
    // - Always compute restricted residuals when impose_null && q > 1 (or q == 1 && !conf_int).
    // - For affine_ci (q == 1 && impose_null && conf_int), split u^R(delta) = u_hat + delta * (-z0).
    let mut targets_owned: Vec<Vec<f64>> = Vec::with_capacity(if affine_ci { 2 } else { 1 });
    if affine_ci {
        let mut u1 = vec![0.0; n];
        for i in 0..n {
            let xr = &input.x[i * k..(i + 1) * k];
            let mut z0 = 0.0;
            for l in 0..k {
                z0 += xr[l] * proj.m_r[l];
            }
            u1[i] = -z0;
        }
        targets_owned.push(input.resid.to_vec());
        targets_owned.push(u1);
    } else if !input.spec.impose_null {
        targets_owned.push(input.resid.to_vec());
    } else {
        // step = M_R * (r - R*beta) = -M_R * num_obs
        let mut neg_num = vec![0.0; q];
        for a in 0..q {
            neg_num[a] = -num_obs[a];
        }
        let mut step = vec![0.0; k];
        matvec(&proj.m_r, &neg_num, k, q, &mut step);
        let mut ur = vec![0.0; n];
        for i in 0..n {
            let xr = &input.x[i * k..(i + 1) * k];
            let mut adj = 0.0;
            for l in 0..k {
                adj += xr[l] * step[l];
            }
            ur[i] = input.resid[i] - adj;
        }
        targets_owned.push(ur);
    }

    let n_tgt = targets_owned.len();
    let w_rows = input.w_rows;
    let fill_z = |i: usize, z: &mut [f64]| {
        let xr = &input.x[i * k..(i + 1) * k];
        for a in 0..q {
            let mut s = 0.0;
            for l in 0..k {
                s += xr[l] * proj.c_mat[l * q + a];
            }
            z[a] = s;
        }
    };

    // Single pass over rows accumulates:
    // - obs cell score q_obs: n_cells * q (always from unrestricted resid!)
    // - A_cell: n_cells * n_tgt * q
    // - B_cell: n_cells * q * k
    // - S_X: g_boot * n_tgt * k
    let len_qobs = cells.n_cells * q;
    let len_acell = cells.n_cells * n_tgt * q;
    let len_bcell = cells.n_cells * q * k;
    let len_sx = g_boot * n_tgt * k;
    let tot_acc = len_qobs + len_acell + len_bcell + len_sx;
    let off_acell = len_qobs;
    let off_bcell = off_acell + len_acell;
    let off_sx = off_bcell + len_bcell;

    let packed = par::sum_chunks(pool, n, CHUNK, tot_acc, |range, acc| {
        let mut z = vec![0.0; q];
        for i in range {
            let wi = w_rows.map_or(1.0, |w| w[i]);
            let cell = cells.cell_of_row[i] as usize;
            let gb = cl.boot_codes[i] as usize;
            fill_z(i, &mut z);
            let xr = &input.x[i * k..(i + 1) * k];

            let q_dst = &mut acc[cell * q..(cell + 1) * q];
            let u_obs = input.resid[i];
            for a in 0..q {
                q_dst[a] += wi * z[a] * u_obs;
            }
            for t in 0..n_tgt {
                let ut = targets_owned[t][i];
                let a_dst = &mut acc[off_acell + (cell * n_tgt + t) * q..off_acell + (cell * n_tgt + t + 1) * q];
                for a in 0..q {
                    a_dst[a] += wi * z[a] * ut;
                }
                let s_dst = &mut acc[off_sx + (gb * n_tgt + t) * k..off_sx + (gb * n_tgt + t + 1) * k];
                let wut = wi * ut;
                for l in 0..k {
                    s_dst[l] += wut * xr[l];
                }
            }
            let b_dst = &mut acc[off_bcell + cell * q * k..off_bcell + (cell + 1) * q * k];
            for a in 0..q {
                let wza = wi * z[a];
                let brow = &mut b_dst[a * k..(a + 1) * k];
                for l in 0..k {
                    brow[l] += wza * xr[l];
                }
            }
        }
    });

    let q_obs = &packed[0..off_acell];
    let a_cell = &packed[off_acell..off_bcell];
    let b_cell = &packed[off_bcell..off_sx];
    let s_x = &packed[off_sx..];

    let mut v_obs = vec![0.0; q * q];
    cells.variance_from_cell_q(q_obs, q, cl.ssc_factor, cl.per_subset_g_adj, &mut v_obs);
    let (stat, se) = eval_stat_and_se(&num_obs, &v_obs, q);

    // Phi[g, t] = bread * S_X[g, t], R_Phi[g, t] = R * Phi[g, t]
    let mut phi = vec![0.0; g_boot * n_tgt * k];
    let mut r_phi = vec![0.0; g_boot * n_tgt * q];
    for g in 0..g_boot {
        for t in 0..n_tgt {
            let svec = &s_x[(g * n_tgt + t) * k..(g * n_tgt + t + 1) * k];
            let pvec = &mut phi[(g * n_tgt + t) * k..(g * n_tgt + t + 1) * k];
            matvec(input.bread, svec, k, k, pvec);
            let rvec = &mut r_phi[(g * n_tgt + t) * q..(g * n_tgt + t + 1) * q];
            matvec(&input.hyp.r_mat, pvec, q, k, rvec);
        }
    }

    enum NnData {
        None,
        Path1 { m_nn: usize, psi: Vec<f64>, f_cell_z: Vec<f64> },
        Path2 { p2_mat: Vec<f64> },
    }
    let nn_data = match plan {
        NonNestedPlan::None => NnData::None,
        NonNestedPlan::TooLarge { g_boot: gb, m_nn } => {
            notes.push(format!("warning: fixed effects non-nested in bootstrap clusters (G={gb}, FE params={m_nn} > 512); exact non-nested FE projection skipped"));
            NnData::None
        }
        NonNestedPlan::Path1 { m_nn, nn_indices, g_nn_pinv, .. } => {
            let f_boot = accumulate_dnn_groups(
                input.terms,
                &nn_indices,
                m_nn,
                input.row_order,
                cl.boot_codes,
                g_boot,
                w_rows,
                n_tgt,
                |i, out| {
                    for t in 0..n_tgt {
                        out[t] = targets_owned[t][i];
                    }
                },
                pool,
            );
            let mut psi = vec![0.0; g_boot * n_tgt * m_nn];
            for g in 0..g_boot {
                for t in 0..n_tgt {
                    let src = &f_boot[(g * n_tgt + t) * m_nn..(g * n_tgt + t + 1) * m_nn];
                    let dst = &mut psi[(g * n_tgt + t) * m_nn..(g * n_tgt + t + 1) * m_nn];
                    matvec(&g_nn_pinv, src, m_nn, m_nn, dst);
                }
            }
            let f_cell_z = accumulate_dnn_groups(
                input.terms,
                &nn_indices,
                m_nn,
                input.row_order,
                &cells.cell_of_row,
                cells.n_cells,
                w_rows,
                q,
                fill_z,
                pool,
            );
            NnData::Path1 { m_nn, psi, f_cell_z }
        }
        NonNestedPlan::Path2 => {
            let tgt_refs: Vec<&[f64]> = targets_owned.iter().map(Vec::as_slice).collect();
            let abs = absorber.as_deref_mut().ok_or_else(|| RfError::Internal("wildboot Path 2 requires absorber".into()))?;
            let p2_mat = path2_project_matrices(
                abs,
                cl.boot_codes,
                g_boot,
                &cells.cell_of_row,
                cells.n_cells,
                w_rows,
                &tgt_refs,
                q,
                fill_z,
                input.fixef_tol,
                input.fixef_maxiter,
                pool,
                ctx,
            )?;
            NnData::Path2 { p2_mat }
        }
    };

    // Draw loop (supports affine q == 1 CI precomputation when affine_ci == true).
    let n_cells = cells.n_cells;
    let eval_draw = |b: usize,
                     db_buf: &mut [f64],
                     gamma_buf: &mut [f64],
                     q_cell_buf: &mut [f64],
                     v_buf: &mut [f64],
                     num_buf: &mut [f64]|
     -> (f64, Option<[f64; 5]>) {
        let vb = &v_mat[b * g_boot..(b + 1) * g_boot];
        num_buf[..n_tgt * q].fill(0.0);
        db_buf[..n_tgt * k].fill(0.0);
        for g in 0..g_boot {
            let vg = vb[g];
            for t in 0..n_tgt {
                let r_src = &r_phi[(g * n_tgt + t) * q..(g * n_tgt + t + 1) * q];
                let r_dst = &mut num_buf[t * q..(t + 1) * q];
                for a in 0..q {
                    r_dst[a] += vg * r_src[a];
                }
                let p_src = &phi[(g * n_tgt + t) * k..(g * n_tgt + t + 1) * k];
                let p_dst = &mut db_buf[t * k..(t + 1) * k];
                for l in 0..k {
                    p_dst[l] += vg * p_src[l];
                }
            }
        }
        if let NnData::Path1 { m_nn, psi, .. } = &nn_data {
            gamma_buf[..n_tgt * *m_nn].fill(0.0);
            for g in 0..g_boot {
                let vg = vb[g];
                for t in 0..n_tgt {
                    let ps = &psi[(g * n_tgt + t) * *m_nn..(g * n_tgt + t + 1) * *m_nn];
                    let gd = &mut gamma_buf[t * *m_nn..(t + 1) * *m_nn];
                    for p in 0..*m_nn {
                        gd[p] += vg * ps[p];
                    }
                }
            }
        }

        for c in 0..n_cells {
            let gb = cells.cell_boot[c] as usize;
            let vg = vb[gb];
            let b_c = &b_cell[c * q * k..(c + 1) * q * k];
            for t in 0..n_tgt {
                let a_c = &a_cell[(c * n_tgt + t) * q..(c * n_tgt + t + 1) * q];
                let db_t = &db_buf[t * k..(t + 1) * k];
                let dst = if affine_ci {
                    &mut q_cell_buf[c * 2 + t..c * 2 + t + 1]
                } else {
                    &mut q_cell_buf[c * q..(c + 1) * q]
                };
                for a in 0..q {
                    let brow = &b_c[a * k..(a + 1) * k];
                    let mut fit_a = 0.0;
                    for l in 0..k {
                        fit_a += brow[l] * db_t[l];
                    }
                    let mut p_a = 0.0;
                    match &nn_data {
                        NnData::None => {}
                        NnData::Path1 { m_nn, f_cell_z, .. } => {
                            let frow = &f_cell_z[(c * q + a) * *m_nn..(c * q + a + 1) * *m_nn];
                            let gam_t = &gamma_buf[t * *m_nn..(t + 1) * *m_nn];
                            for p in 0..*m_nn {
                                p_a += frow[p] * gam_t[p];
                            }
                        }
                        NnData::Path2 { p2_mat } => {
                            let p2_row = &p2_mat[((c * q + a) * n_tgt + t) * g_boot..((c * q + a) * n_tgt + t + 1) * g_boot];
                            for g in 0..g_boot {
                                p_a += vb[g] * p2_row[g];
                            }
                        }
                    }
                    dst[a] = vg * a_c[a] - fit_a - p_a;
                }
            }
        }

        if affine_ci {
            cells.variance_from_cell_q(q_cell_buf, 2, cl.ssc_factor, cl.per_subset_g_adj, v_buf);
            let coef = [num_buf[0], num_buf[1], v_buf[0], v_buf[1], v_buf[3]];
            let delta = input.hyp.r_vec[0] - point.expect("q==1");
            (eval_affine_t(delta, coef[0], coef[1], coef[2], coef[3], coef[4]), Some(coef))
        } else {
            cells.variance_from_cell_q(q_cell_buf, q, cl.ssc_factor, cl.per_subset_g_adj, v_buf);
            let (tb, _) = eval_stat_and_se(num_buf, v_buf, q);
            (tb, None)
        }
    };

    let m_nn_buf = match &nn_data {
        NnData::Path1 { m_nn, .. } => *m_nn,
        _ => 0,
    };
    let q_cell_len = if affine_ci { n_cells * 2 } else { n_cells * q };
    let v_len = if affine_ci { 4 } else { q * q };

    let mut t_boot = vec![0.0; reps];
    let mut affine_coefs = if affine_ci { vec![[0.0; 5]; reps] } else { Vec::new() };
    pool.install(|| {
        if affine_ci {
            t_boot.par_chunks_mut(CHUNK).zip(affine_coefs.par_chunks_mut(CHUNK)).enumerate().for_each(|(ci, (tc, ac))| {
                let mut db_buf = vec![0.0; n_tgt * k];
                let mut gamma_buf = vec![0.0; n_tgt * m_nn_buf];
                let mut q_cell_buf = vec![0.0; q_cell_len];
                let mut v_buf = vec![0.0; v_len];
                let mut num_buf = vec![0.0; n_tgt * q];
                let base = ci * CHUNK;
                for (j, (tb, ab)) in tc.iter_mut().zip(ac.iter_mut()).enumerate() {
                    let (val, c) = eval_draw(base + j, &mut db_buf, &mut gamma_buf, &mut q_cell_buf, &mut v_buf, &mut num_buf);
                    *tb = val;
                    *ab = c.expect("affine");
                }
            });
        } else {
            t_boot.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, tc)| {
                let mut db_buf = vec![0.0; n_tgt * k];
                let mut gamma_buf = vec![0.0; n_tgt * m_nn_buf];
                let mut q_cell_buf = vec![0.0; q_cell_len];
                let mut v_buf = vec![0.0; v_len];
                let mut num_buf = vec![0.0; n_tgt * q];
                let base = ci * CHUNK;
                for (j, tb) in tc.iter_mut().enumerate() {
                    *tb = eval_draw(base + j, &mut db_buf, &mut gamma_buf, &mut q_cell_buf, &mut v_buf, &mut num_buf).0;
                }
            });
        }
    });

    let (p_val, reps_feasible) = compute_pvalue(stat, &t_boot, q, input.spec.pval_type);
    let conf_int = if q == 1 && input.spec.conf_int {
        let alpha = 1.0 - input.spec.level;
        let pt = point.expect("q==1");
        if input.spec.impose_null {
            let se_val = se.unwrap_or(f64::NAN);
            let mut trial_tb = vec![0.0; reps];
            let ci = invert_ci_search(pt, se_val, alpha, input.spec.pval_type, input.spec.tol, input.spec.maxiter, reps_feasible, |r_trial| {
                let delta = r_trial - pt;
                let st_trial = (pt - r_trial) / se_val;
                for (dst, c) in trial_tb.iter_mut().zip(&affine_coefs) {
                    *dst = eval_affine_t(delta, c[0], c[1], c[2], c[3], c[4]);
                }
                Ok(compute_pvalue(st_trial, &trial_tb, 1, input.spec.pval_type).0)
            })?;
            Some(ci)
        } else {
            se.map(|se_val| invert_ci_wcu(pt, se_val, &t_boot, input.spec.pval_type, alpha))
        }
    } else {
        None
    };

    Ok(WildBootResult {
        stat,
        p_val,
        stat_type: if q == 1 { StatType::T } else { StatType::F },
        point_estimate: point,
        se,
        conf_int,
        t_boot,
        reps,
        reps_feasible,
        nobs: n,
        n_clusters: Some(g_boot),
        q,
        full_enumeration: full_enum,
        method: BootMethod::WildCluster,
        boot_type: input.spec.boot_type,
        weights: input.spec.weights,
        impose_null: input.spec.impose_null,
        pval_type: input.spec.pval_type,
        ssc: cl.ssc_factor,
        notes,
    })
}

/// MNW (2023) clustered OLS `T13`/`T31`/`T33` (WCR/WCU with CRV3 jackknife DGP and/or denominator).
fn run_cluster_mnw(input: OlsBootInput<'_>, cl: &ClusterConfig<'_>, ctx: &Ctx, pool: &Pool) -> Result<WildBootResult, RfError> {
    if !cl.err_equals_boot {
        return Err(RfError::InvalidInput("wildboot: bootstrap types 13, 31, and 33 require matching 1-way bootcluster and error cluster".into()));
    }
    let (n, k, q) = (input.n, input.k, input.hyp.q);
    let g_boot = cl.g_boot;
    let mut notes = Vec::new();
    let plan = classify_non_nested(
        input.terms,
        input.row_order,
        cl.boot_kept,
        None,
        g_boot,
        input.w_rows,
        true,
        input.fixef_tol,
        input.fixef_maxiter,
        pool,
        ctx,
    )?;

    let (q_nn, r_nn) = match plan {
        NonNestedPlan::None => (None, 0usize),
        NonNestedPlan::Path1 { q_nn: Some((mat, r)), .. } if k + r <= 256 => (Some(mat), r),
        _ => {
            notes.push("warning: MNW 13/31/33 with large non-nested fixed effects ignores leave-out non-nested FE refitting".into());
            (None, 0usize)
        }
    };
    let p = k + r_nn;
    let mut r_aug = vec![0.0; q * p];
    for a in 0..q {
        r_aug[a * p..a * p + k].copy_from_slice(&input.hyp.r_mat[a * k..(a + 1) * k]);
    }
    let hyp_aug = Hypothesis { r_mat: r_aug, r_vec: input.hyp.r_vec.clone(), q, k: p };

    let (num_obs, point) = linear_num_obs(input.beta, input.hyp);
    let proj_k = ConstraintProj::new(input.bread, &input.hyp.r_mat, q, k);
    let affine_ci = q == 1 && input.spec.impose_null && input.spec.conf_int;
    let joint_restricted = q > 1 && input.spec.impose_null;

    // Build residual targets:
    // - Always include obs u_hat first.
    // - If affine_ci: target 1 is u_1 = -z_0. (dgp_n_tgt = 2 starting at slot 0).
    // - Else if !impose_null: dgp_n_tgt = 1 (slot 0 = u_hat).
    // - Else (impose_null && !affine_ci): include u^R as extra target (slot 1), dgp_n_tgt = 1 from slot 1!
    let mut extra_res: Vec<f64> = Vec::new();
    if affine_ci {
        extra_res.resize(n, 0.0);
        for i in 0..n {
            let xr = &input.x[i * k..(i + 1) * k];
            let mut z0 = 0.0;
            for l in 0..k {
                z0 += xr[l] * proj_k.m_r[l];
            }
            extra_res[i] = -z0;
        }
    } else if input.spec.impose_null {
        let mut neg_num = vec![0.0; q];
        for a in 0..q {
            neg_num[a] = -num_obs[a];
        }
        let mut step = vec![0.0; k];
        matvec(&proj_k.m_r, &neg_num, k, q, &mut step);
        extra_res.resize(n, 0.0);
        for i in 0..n {
            let xr = &input.x[i * k..(i + 1) * k];
            let mut adj = 0.0;
            for l in 0..k {
                adj += xr[l] * step[l];
            }
            extra_res[i] = input.resid[i] - adj;
        }
    }

    let n_acc_u = if extra_res.is_empty() { 1 } else { 2 };
    let len_a = g_boot * p * p;
    let len_s = g_boot * n_acc_u * p;
    let w_rows = input.w_rows;
    let q_nn_ref = q_nn.as_deref();

    let packed = par::sum_chunks(pool, n, CHUNK, len_a + len_s, |range, acc| {
        let mut xa = vec![0.0; p];
        for i in range {
            let wi = w_rows.map_or(1.0, |w| w[i]);
            let g = cl.boot_codes[i] as usize;
            xa[..k].copy_from_slice(&input.x[i * k..(i + 1) * k]);
            if let Some(qm) = q_nn_ref {
                xa[k..].copy_from_slice(&qm[i * r_nn..(i + 1) * r_nn]);
            }
            let a_dst = &mut acc[g * p * p..(g + 1) * p * p];
            for r0 in 0..p {
                let wx = wi * xa[r0];
                let arow = &mut a_dst[r0 * p..(r0 + 1) * p];
                for c0 in r0..p {
                    arow[c0] += wx * xa[c0];
                }
            }
            let u0 = wi * input.resid[i];
            let s0_dst = &mut acc[len_a + (g * n_acc_u) * p..len_a + (g * n_acc_u + 1) * p];
            for r0 in 0..p {
                s0_dst[r0] += u0 * xa[r0];
            }
            if n_acc_u == 2 {
                let u1 = wi * extra_res[i];
                let s1_dst = &mut acc[len_a + (g * n_acc_u + 1) * p..len_a + (g * n_acc_u + 2) * p];
                for r0 in 0..p {
                    s1_dst[r0] += u1 * xa[r0];
                }
            }
        }
    });

    let mut a_mats = packed[..len_a].to_vec();
    let s_raw = &packed[len_a..];
    let mut g_tot = vec![0.0; p * p];
    for g in 0..g_boot {
        let ag = &mut a_mats[g * p * p..(g + 1) * p * p];
        for r0 in 0..p {
            for c0 in 0..r0 {
                ag[r0 * p + c0] = ag[c0 * p + r0];
            }
        }
        for idx in 0..p * p {
            g_tot[idx] += ag[idx];
        }
    }
    let g_tot_inv = sym_pinv(&g_tot, p, 1e-12);
    let proj_aug = ConstraintProj::new(&g_tot_inv, &hyp_aug.r_mat, q, p);

    let dgp_is_3 = input.spec.boot_type.dgp_is_3();
    let denom_is_3 = input.spec.boot_type.denom_is_3();
    let crv1_mult = cl.ssc_factor * if cl.per_subset_g_adj { g_boot as f64 / (g_boot as f64 - 1.0) } else { 1.0 };
    let denom_mult = if denom_is_3 { (g_boot - 1) as f64 / g_boot as f64 } else { crv1_mult };

    // Per cluster: compute J_g = R_aug * G_(-g)^+ (when denom_is_3), JA_g = J_g * A_g,
    // D_g[0..n_dgp], k_dgp[g, t] = J_g * D_g[t], phi[g, t] = G_tot^+ * D_g[t], r_phi = R_aug * phi,
    // and j_obs[g] for V_obs.
    let n_dgp = if affine_ci { 2 } else { 1 };
    let dgp_slot0 = if input.spec.impose_null && !affine_ci { 1 } else { 0 };
    let _ = joint_restricted;

    struct ClusterPre {
        ja: Vec<f64>,     // q * p
        k_dgp: Vec<f64>,  // n_dgp * q
        phi: Vec<f64>,    // n_dgp * p
        r_phi: Vec<f64>,  // n_dgp * q
        j_obs: Vec<f64>,  // q
    }

    let pres: Vec<ClusterPre> = pool.install(|| {
        (0..g_boot)
            .into_par_iter()
            .map(|g| {
                let ag = &a_mats[g * p * p..(g + 1) * p * p];
                let mut g_neg_inv = Vec::new();
                if dgp_is_3 || denom_is_3 {
                    let mut g_neg = vec![0.0; p * p];
                    for idx in 0..p * p {
                        g_neg[idx] = g_tot[idx] - ag[idx];
                    }
                    g_neg_inv = sym_pinv(&g_neg, p, 1e-10);
                }

                // J_mat (q x p) = R_aug * G_(-g)^+ (denom_is_3) OR C_aug' = R_aug * G_tot^+ (CRV1 denom).
                let mut j_mat = vec![0.0; q * p];
                if denom_is_3 {
                    for a in 0..q {
                        for c0 in 0..p {
                            let mut s = 0.0;
                            for j in 0..p {
                                s += hyp_aug.r_mat[a * p + j] * g_neg_inv[j * p + c0];
                            }
                            j_mat[a * p + c0] = s;
                        }
                    }
                } else {
                    for a in 0..q {
                        for c0 in 0..p {
                            j_mat[a * p + c0] = proj_aug.c_mat[c0 * q + a];
                        }
                    }
                }

                let mut ja = vec![0.0; q * p];
                for a in 0..q {
                    for c0 in 0..p {
                        let mut s = 0.0;
                        for j in 0..p {
                            s += j_mat[a * p + j] * ag[j * p + c0];
                        }
                        ja[a * p + c0] = s;
                    }
                }

                let s_obs = &s_raw[(g * n_acc_u) * p..(g * n_acc_u + 1) * p];
                let mut j_obs = vec![0.0; q];
                matvec(&j_mat, s_obs, q, p, &mut j_obs);

                // Restricted inverse helper matrices if dgp_is_3 && impose_null:
                let mut g_neg_r_mat = Vec::new();
                let mut h_neg_inv = Vec::new();
                if dgp_is_3 && input.spec.impose_null {
                    g_neg_r_mat.resize(p * q, 0.0);
                    for i in 0..p {
                        for a in 0..q {
                            let mut s = 0.0;
                            for j in 0..p {
                                s += g_neg_inv[i * p + j] * hyp_aug.r_mat[a * p + j];
                            }
                            g_neg_r_mat[i * q + a] = s;
                        }
                    }
                    let mut h_neg = vec![0.0; q * q];
                    for a in 0..q {
                        for b in a..q {
                            let mut s = 0.0;
                            for j in 0..p {
                                s += hyp_aug.r_mat[a * p + j] * g_neg_r_mat[j * q + b];
                            }
                            h_neg[a * q + b] = s;
                            h_neg[b * q + a] = s;
                        }
                    }
                    h_neg_inv = sym_pinv(&h_neg, q, 1e-10);
                }

                let mut k_dgp = vec![0.0; n_dgp * q];
                let mut phi = vec![0.0; n_dgp * p];
                let mut r_phi = vec![0.0; n_dgp * q];
                let mut d_vec = vec![0.0; p];
                let mut y1 = vec![0.0; p];
                let mut y2 = vec![0.0; q];
                let mut y3 = vec![0.0; q];
                let mut y4 = vec![0.0; p];
                let mut diff = vec![0.0; p];

                for t in 0..n_dgp {
                    let s_in = &s_raw[(g * n_acc_u + dgp_slot0 + t) * p..(g * n_acc_u + dgp_slot0 + t + 1) * p];
                    if !dgp_is_3 {
                        d_vec.copy_from_slice(s_in);
                    } else if !input.spec.impose_null {
                        matvec(&g_neg_inv, s_in, p, p, &mut y1);
                        matvec(ag, &y1, p, p, &mut y4);
                        for j in 0..p {
                            d_vec[j] = s_in[j] + y4[j];
                        }
                    } else {
                        matvec(&g_neg_inv, s_in, p, p, &mut y1);
                        matvec(&hyp_aug.r_mat, &y1, q, p, &mut y2);
                        matvec(&h_neg_inv, &y2, q, q, &mut y3);
                        matvec(&g_neg_r_mat, &y3, p, q, &mut y4);
                        for j in 0..p {
                            diff[j] = y1[j] - y4[j];
                        }
                        matvec(ag, &diff, p, p, &mut y1);
                        for j in 0..p {
                            d_vec[j] = s_in[j] + y1[j];
                        }
                    }

                    matvec(&j_mat, &d_vec, q, p, &mut k_dgp[t * q..(t + 1) * q]);
                    let ph = &mut phi[t * p..(t + 1) * p];
                    matvec(&g_tot_inv, &d_vec, p, p, ph);
                    matvec(&hyp_aug.r_mat, ph, q, p, &mut r_phi[t * q..(t + 1) * q]);
                }

                ClusterPre { ja, k_dgp, phi, r_phi, j_obs }
            })
            .collect()
    });

    // Observed V_obs and stat:
    let mut v_obs = vec![0.0; q * q];
    for pr in &pres {
        for a in 0..q {
            for b in a..q {
                v_obs[a * q + b] += denom_mult * pr.j_obs[a] * pr.j_obs[b];
            }
        }
    }
    for a in 0..q {
        for b in 0..a {
            v_obs[a * q + b] = v_obs[b * q + a];
        }
    }
    let (stat, se) = eval_stat_and_se(&num_obs, &v_obs, q);

    let (reps, full_enum) = resolve_reps(input.spec.weights, input.spec.reps, Some(g_boot));
    let v_mat = cluster_matrix(input.spec.weights, g_boot, reps, full_enum, input.spec.seed, pool)?;

    let mut t_boot = vec![0.0; reps];
    let mut affine_coefs = if affine_ci { vec![[0.0; 5]; reps] } else { Vec::new() };

    pool.install(|| {
        let work = |b: usize, db_buf: &mut [f64], ja_db: &mut [f64], v_buf: &mut [f64], num_buf: &mut [f64]| -> (f64, Option<[f64; 5]>) {
            let vb = &v_mat[b * g_boot..(b + 1) * g_boot];
            num_buf[..n_dgp * q].fill(0.0);
            db_buf[..n_dgp * p].fill(0.0);
            for (g, pr) in pres.iter().enumerate() {
                let vg = vb[g];
                for t in 0..n_dgp {
                    let rs = &pr.r_phi[t * q..(t + 1) * q];
                    for a in 0..q {
                        num_buf[t * q + a] += vg * rs[a];
                    }
                    let ps = &pr.phi[t * p..(t + 1) * p];
                    for l in 0..p {
                        db_buf[t * p + l] += vg * ps[l];
                    }
                }
            }
            if affine_ci {
                let (mut d00, mut d01, mut d11) = (0.0, 0.0, 0.0);
                for (g, pr) in pres.iter().enumerate() {
                    let vg = vb[g];
                    let mut p0 = 0.0;
                    let mut p1 = 0.0;
                    for l in 0..p {
                        p0 += pr.ja[l] * db_buf[l];
                        p1 += pr.ja[l] * db_buf[p + l];
                    }
                    let j0 = vg * pr.k_dgp[0] - p0;
                    let j1 = vg * pr.k_dgp[1] - p1;
                    d00 += j0 * j0;
                    d01 += j0 * j1;
                    d11 += j1 * j1;
                }
                let coef = [num_buf[0], num_buf[1], denom_mult * d00, denom_mult * d01, denom_mult * d11];
                let delta = input.hyp.r_vec[0] - point.expect("q==1");
                (eval_affine_t(delta, coef[0], coef[1], coef[2], coef[3], coef[4]), Some(coef))
            } else {
                v_buf[..q * q].fill(0.0);
                for (g, pr) in pres.iter().enumerate() {
                    let vg = vb[g];
                    matvec(&pr.ja, db_buf, q, p, ja_db);
                    for a in 0..q {
                        ja_db[a] = vg * pr.k_dgp[a] - ja_db[a];
                    }
                    for a in 0..q {
                        let va = denom_mult * ja_db[a];
                        for b0 in a..q {
                            v_buf[a * q + b0] += va * ja_db[b0];
                        }
                    }
                }
                for a in 0..q {
                    for b0 in 0..a {
                        v_buf[a * q + b0] = v_buf[b0 * q + a];
                    }
                }
                let (tb, _) = eval_stat_and_se(num_buf, v_buf, q);
                (tb, None)
            }
        };

        if affine_ci {
            t_boot.par_chunks_mut(CHUNK).zip(affine_coefs.par_chunks_mut(CHUNK)).enumerate().for_each(|(ci, (tc, ac))| {
                let mut db_buf = vec![0.0; n_dgp * p];
                let mut ja_db = vec![0.0; q];
                let mut v_buf = vec![0.0; q * q];
                let mut num_buf = vec![0.0; n_dgp * q];
                let base = ci * CHUNK;
                for (j, (tb, ab)) in tc.iter_mut().zip(ac.iter_mut()).enumerate() {
                    let (val, c) = work(base + j, &mut db_buf, &mut ja_db, &mut v_buf, &mut num_buf);
                    *tb = val;
                    *ab = c.expect("affine");
                }
            });
        } else {
            t_boot.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, tc)| {
                let mut db_buf = vec![0.0; n_dgp * p];
                let mut ja_db = vec![0.0; q];
                let mut v_buf = vec![0.0; q * q];
                let mut num_buf = vec![0.0; n_dgp * q];
                let base = ci * CHUNK;
                for (j, tb) in tc.iter_mut().enumerate() {
                    *tb = work(base + j, &mut db_buf, &mut ja_db, &mut v_buf, &mut num_buf).0;
                }
            });
        }
    });

    let (p_val, reps_feasible) = compute_pvalue(stat, &t_boot, q, input.spec.pval_type);
    let conf_int = if q == 1 && input.spec.conf_int {
        let alpha = 1.0 - input.spec.level;
        let pt = point.expect("q==1");
        if input.spec.impose_null {
            let se_val = se.unwrap_or(f64::NAN);
            let mut trial_tb = vec![0.0; reps];
            let ci = invert_ci_search(pt, se_val, alpha, input.spec.pval_type, input.spec.tol, input.spec.maxiter, reps_feasible, |r_trial| {
                let delta = r_trial - pt;
                let st_trial = (pt - r_trial) / se_val;
                for (dst, c) in trial_tb.iter_mut().zip(&affine_coefs) {
                    *dst = eval_affine_t(delta, c[0], c[1], c[2], c[3], c[4]);
                }
                Ok(compute_pvalue(st_trial, &trial_tb, 1, input.spec.pval_type).0)
            })?;
            Some(ci)
        } else {
            se.map(|se_val| invert_ci_wcu(pt, se_val, &t_boot, input.spec.pval_type, alpha))
        }
    } else {
        None
    };

    Ok(WildBootResult {
        stat,
        p_val,
        stat_type: if q == 1 { StatType::T } else { StatType::F },
        point_estimate: point,
        se,
        conf_int,
        t_boot,
        reps,
        reps_feasible,
        nobs: n,
        n_clusters: Some(g_boot),
        q,
        full_enumeration: full_enum,
        method: BootMethod::WildCluster,
        boot_type: input.spec.boot_type,
        weights: input.spec.weights,
        impose_null: input.spec.impose_null,
        pval_type: input.spec.pval_type,
        ssc: denom_mult,
        notes,
    })
}

/// Unclustered heteroskedastic wild bootstrap (`ols` HC1/HC2/HC3 and unclustered one-step `score`).
pub(crate) fn run_hetero(hi: HeteroInput<'_>, ctx: &Ctx, pool: &Pool) -> Result<WildBootResult, RfError> {
    let (n, k, q) = (hi.n, hi.k, hi.hyp.q);
    let mut notes = hi.notes;
    let proj = ConstraintProj::new(hi.bread, &hi.hyp.r_mat, q, k);

    // FE projection strategy under unclustered Hetero.
    enum FeProj {
        None,
        OneWay { codes_row: Vec<u32>, g: usize, inv_sum_w: Vec<f64> },
        Dense { m_fe: usize, g_pinv: Vec<f64> },
        Approx,
    }

    let fe_proj = if hi.terms.is_empty() {
        FeProj::None
    } else if hi.terms.len() == 1 && hi.terms[0].intercept && hi.terms[0].slopes.is_empty() {
        let g = hi.terms[0].g;
        let mut codes_row = vec![0u32; n];
        for (i, &ki) in hi.row_order.iter().enumerate() {
            codes_row[i] = hi.terms[0].codes[ki as usize];
        }
        let mut sum_w = vec![0.0; g];
        for (i, &lvl) in codes_row.iter().enumerate() {
            sum_w[lvl as usize] += hi.w_rows.map_or(1.0, |w| w[i]);
        }
        let inv_sum_w: Vec<f64> = sum_w.into_iter().map(|s| if s > 0.0 { 1.0 / s } else { 0.0 }).collect();
        FeProj::OneWay { codes_row, g, inv_sum_w }
    } else {
        let m_fe: usize = hi.terms.iter().map(|t| t.g * (usize::from(t.intercept) + t.slopes.len())).sum();
        if m_fe <= 256 {
            let mut off = 0usize;
            let mut offsets = Vec::with_capacity(hi.terms.len());
            for t in hi.terms {
                offsets.push(off);
                off += t.g * (usize::from(t.intercept) + t.slopes.len());
            }
            let g_fe = par::sum_chunks(pool, n, CHUNK, m_fe * m_fe, |range, acc| {
                let mut cols: Vec<(usize, f64)> = Vec::with_capacity(16);
                for i in range {
                    let ki = hi.row_order[i] as usize;
                    let wi = hi.w_rows.map_or(1.0, |w| w[i]);
                    cols.clear();
                    for (qi, t) in hi.terms.iter().enumerate() {
                        let lvl = t.codes[ki] as usize;
                        let base = offsets[qi];
                        let mut slot = 0usize;
                        if t.intercept {
                            cols.push((base + lvl, 1.0));
                            slot += 1;
                        }
                        for sl in &t.slopes {
                            cols.push((base + slot * t.g + lvl, sl[ki]));
                            slot += 1;
                        }
                    }
                    for &(ca, va) in &cols {
                        let wva = wi * va;
                        for &(cb, vb) in &cols {
                            acc[ca * m_fe + cb] += wva * vb;
                        }
                    }
                }
            });
            FeProj::Dense { m_fe, g_pinv: sym_pinv(&g_fe, m_fe, 1e-10) }
        } else {
            notes.push("warning: heteroskedastic wild bootstrap with large multi-term fixed effects approximates residual FE partial-out".into());
            FeProj::Approx
        }
    };

    // Leverage h_i when HC2 or HC3 powers are nonzero.
    let need_hat = hi.dgp_pow != 0.0 || hi.denom_pow != 0.0;
    let h_fe_row = if need_hat && !hi.terms.is_empty() {
        let term_refs: Vec<(&[u32], usize, bool, Vec<&[f64]>)> = hi
            .terms
            .iter()
            .map(|t| (t.codes.as_slice(), t.g, t.intercept, t.slopes.iter().map(Vec::as_slice).collect()))
            .collect();
        let w_kept = hi.w_rows.map(|wr| crate::fe_recovery::to_kept_order(wr, hi.row_order, pool)).transpose()?;
        let cols: Vec<FeCols> = term_refs
            .iter()
            .map(|(codes, g, intercept, slopes)| FeCols { codes, g: *g, intercept: *intercept, slopes: slopes.clone() })
            .collect();
        let hk = fe_hat(&cols, w_kept.as_deref(), n, ctx, pool)?;
        Some(to_row_order(&hk, hi.row_order, pool)?)
    } else {
        None
    };

    let mut dgp_mult = vec![1.0; n];
    let mut denom_sq_mult = vec![1.0; n];
    if need_hat {
        let thr = f64::EPSILON.sqrt();
        for i in 0..n {
            let wi = hi.w_rows.map_or(1.0, |w| w[i]);
            let xr = &hi.x[i * k..(i + 1) * k];
            let mut quad_x = 0.0;
            for r0 in 0..k {
                let mut acc = 0.0;
                for c0 in 0..k {
                    acc += hi.bread[r0 * k + c0] * xr[c0];
                }
                quad_x += xr[r0] * acc;
            }
            let h_tot = h_fe_row.as_deref().map_or(0.0, |h| h[i]) + wi * quad_x;
            let h_unit = if hi.is_fweight { h_tot / wi } else { h_tot };
            let d = 1.0 - h_unit;
            if d <= thr {
                dgp_mult[i] = 0.0;
                denom_sq_mult[i] = 0.0;
            } else {
                dgp_mult[i] = if hi.dgp_pow == 0.0 { 1.0 } else { d.powf(-hi.dgp_pow) };
                denom_sq_mult[i] = if hi.denom_pow == 0.0 { 1.0 } else { d.powf(-2.0 * hi.denom_pow) };
            }
        }
    }

    // Precompute C' * x_i (`n * q`).
    let mut z_row = par::alloc_vec(n * q, 0.0)?;
    for i in 0..n {
        let xr = &hi.x[i * k..(i + 1) * k];
        let zr = &mut z_row[i * q..(i + 1) * q];
        for a in 0..q {
            let mut s = 0.0;
            for l in 0..k {
                s += xr[l] * proj.c_mat[l * q + a];
            }
            zr[a] = s;
        }
    }

    // Observed variance and statistic (HC3 uses denom_pow > 0 and has no N/(N-K) multiplier).
    let eff_ssc = if hi.denom_pow > 0.0 { 1.0 } else { hi.ssc_factor };
    let obs_res = hi.obs_resid.unwrap_or(hi.resid);
    let mut v_obs = vec![0.0; q * q];
    for i in 0..n {
        let wi = hi.w_rows.map_or(1.0, |w| w[i]);
        let r_i = obs_res[i];
        let w2 = if hi.is_fweight { wi } else { wi * wi };
        let fac = eff_ssc * denom_sq_mult[i] * w2 * r_i * r_i;
        let zr = &z_row[i * q..(i + 1) * q];
        for a in 0..q {
            for b in a..q {
                v_obs[a * q + b] += fac * zr[a] * zr[b];
            }
        }
    }
    for a in 0..q {
        for b in 0..a {
            v_obs[a * q + b] = v_obs[b * q + a];
        }
    }
    let (stat, se) = eval_stat_and_se(&hi.num_obs, &v_obs, q);

    // Prepare target vectors u_t (scaled by dgp_mult later on the fly or pre-scaled).
    let affine_ci = hi.allow_impose_null_affine && q == 1 && hi.spec.impose_null && hi.spec.conf_int;
    let mut targets_owned: Vec<Vec<f64>> = Vec::with_capacity(if affine_ci { 2 } else { 1 });
    if affine_ci {
        let mut u1 = vec![0.0; n];
        for i in 0..n {
            let xr = &hi.x[i * k..(i + 1) * k];
            let mut z0 = 0.0;
            for l in 0..k {
                z0 += xr[l] * proj.m_r[l];
            }
            u1[i] = -z0;
        }
        targets_owned.push(hi.resid.to_vec());
        targets_owned.push(u1);
    } else if !hi.spec.impose_null || !hi.allow_impose_null_affine {
        targets_owned.push(hi.resid.to_vec());
    } else {
        let mut neg_num = vec![0.0; q];
        for a in 0..q {
            neg_num[a] = -hi.num_obs[a];
        }
        let mut step = vec![0.0; k];
        matvec(&proj.m_r, &neg_num, k, q, &mut step);
        let mut ur = vec![0.0; n];
        for i in 0..n {
            let xr = &hi.x[i * k..(i + 1) * k];
            let mut adj = 0.0;
            for l in 0..k {
                adj += xr[l] * step[l];
            }
            ur[i] = hi.resid[i] - adj;
        }
        targets_owned.push(ur);
    }

    let n_tgt = targets_owned.len();
    for t in &mut targets_owned {
        if need_hat {
            for (v, &m) in t.iter_mut().zip(&dgp_mult) {
                *v *= m;
            }
        }
    }

    let reps = hi.spec.reps;
    let fe_dim = match &fe_proj {
        FeProj::None | FeProj::Approx => 0,
        FeProj::OneWay { g, .. } => *g,
        FeProj::Dense { m_fe, .. } => *m_fe,
    };
    let batch_size = if fe_dim > 4096 {
        (262_144 / (fe_dim * pool.threads().max(1))).clamp(1, 32)
    } else {
        64
    };
    let mut t_boot = vec![0.0; reps];
    let mut affine_coefs = if affine_ci { vec![[0.0; 5]; reps] } else { Vec::new() };
    let sym_sz = q * (q + 1) / 2;

    let mut b0 = 0usize;
    while b0 < reps {
        ctx.check()?;
        let b_cnt = (reps - b0).min(batch_size);
        let sx_len = b_cnt * n_tgt * k;
        let sfe_len = b_cnt * n_tgt * fe_dim;

        // Pass 1: accumulate X' W u* and D' W u* for draws b0..b0+b_cnt in deterministic chunks.
        let p1 = par::sum_chunks(pool, n, CHUNK, sx_len + sfe_len, |range, acc| {
            let ci = range.start / CHUNK;
            let mut rngs: Vec<_> = (0..b_cnt).map(|bi| hetero_chunk_rng(hi.spec.seed, b0 + bi, ci)).collect();
            let mut v_one = [0.0f64; 1];
            let mut v_rep = Vec::new();
            for i in range {
                let wi = hi.w_rows.map_or(1.0, |w| w[i]);
                let f_cnt = if hi.is_fweight { wi.round().max(1.0) as usize } else { 1 };
                let xr = &hi.x[i * k..(i + 1) * k];
                let ki = hi.row_order[i] as usize;
                for bi in 0..b_cnt {
                    let s1 = if f_cnt == 1 {
                        fill_from_rng(hi.spec.weights, &mut rngs[bi], &mut v_one);
                        v_one[0]
                    } else {
                        v_rep.resize(f_cnt, 0.0);
                        fill_from_rng(hi.spec.weights, &mut rngs[bi], &mut v_rep);
                        v_rep.iter().sum()
                    };
                    let w_s1 = if hi.is_fweight { s1 } else { wi * s1 };
                    for t in 0..n_tgt {
                        let val = w_s1 * targets_owned[t][i];
                        let dst_x = &mut acc[(bi * n_tgt + t) * k..(bi * n_tgt + t + 1) * k];
                        for l in 0..k {
                            dst_x[l] += val * xr[l];
                        }
                        match &fe_proj {
                            FeProj::None | FeProj::Approx => {}
                            FeProj::OneWay { codes_row, .. } => {
                                let lvl = codes_row[i] as usize;
                                acc[sx_len + (bi * n_tgt + t) * fe_dim + lvl] += val;
                            }
                            FeProj::Dense { .. } => {
                                let dst_fe = &mut acc[sx_len + (bi * n_tgt + t) * fe_dim..sx_len + (bi * n_tgt + t + 1) * fe_dim];
                                let mut off = 0usize;
                                for term in hi.terms {
                                    let lvl = term.codes[ki] as usize;
                                    let mut slot = 0usize;
                                    if term.intercept {
                                        dst_fe[off + lvl] += val;
                                        slot += 1;
                                    }
                                    for sl in &term.slopes {
                                        dst_fe[off + slot * term.g + lvl] += val * sl[ki];
                                        slot += 1;
                                    }
                                    off += term.g * (usize::from(term.intercept) + term.slopes.len());
                                }
                            }
                        }
                    }
                }
            }
        });

        let mut delta_beta = vec![0.0; b_cnt * n_tgt * k];
        let mut r_db = vec![0.0; b_cnt * n_tgt * q];
        let mut gamma_fe = vec![0.0; b_cnt * n_tgt * fe_dim];
        for bi in 0..b_cnt {
            for t in 0..n_tgt {
                let svec = &p1[(bi * n_tgt + t) * k..(bi * n_tgt + t + 1) * k];
                let dvec = &mut delta_beta[(bi * n_tgt + t) * k..(bi * n_tgt + t + 1) * k];
                matvec(hi.bread, svec, k, k, dvec);
                matvec(&hi.hyp.r_mat, dvec, q, k, &mut r_db[(bi * n_tgt + t) * q..(bi * n_tgt + t + 1) * q]);
                let sfe = &p1[sx_len + (bi * n_tgt + t) * fe_dim..sx_len + (bi * n_tgt + t + 1) * fe_dim];
                let gdst = &mut gamma_fe[(bi * n_tgt + t) * fe_dim..(bi * n_tgt + t + 1) * fe_dim];
                match &fe_proj {
                    FeProj::None | FeProj::Approx => {}
                    FeProj::OneWay { inv_sum_w, .. } => {
                        for l in 0..fe_dim {
                            gdst[l] = sfe[l] * inv_sum_w[l];
                        }
                    }
                    FeProj::Dense { g_pinv, m_fe, .. } => {
                        matvec(g_pinv, sfe, *m_fe, *m_fe, gdst);
                    }
                }
            }
        }

        // Pass 2: exact same RNG stream per (bi, chunk) accumulates quadratic meat in R^q (or affine 2x2).
        let meat_len = if affine_ci { b_cnt * 3 } else { b_cnt * sym_sz };
        let p2 = par::sum_chunks(pool, n, CHUNK, meat_len, |range, acc| {
            let ci = range.start / CHUNK;
            let mut rngs: Vec<_> = (0..b_cnt).map(|bi| hetero_chunk_rng(hi.spec.seed, b0 + bi, ci)).collect();
            let mut v_one = [0.0f64; 1];
            let mut v_rep = Vec::new();
            let mut proj_t = [0.0f64; 2];
            for i in range {
                let wi = hi.w_rows.map_or(1.0, |w| w[i]);
                let f_cnt = if hi.is_fweight { wi.round().max(1.0) as usize } else { 1 };
                let xr = &hi.x[i * k..(i + 1) * k];
                let zr = &z_row[i * q..(i + 1) * q];
                let ki = hi.row_order[i] as usize;
                let base_mult = eff_ssc * denom_sq_mult[i];

                for bi in 0..b_cnt {
                    let (s1, s2) = if f_cnt == 1 {
                        fill_from_rng(hi.spec.weights, &mut rngs[bi], &mut v_one);
                        (v_one[0], v_one[0] * v_one[0])
                    } else {
                        v_rep.resize(f_cnt, 0.0);
                        fill_from_rng(hi.spec.weights, &mut rngs[bi], &mut v_rep);
                        let mut sum1 = 0.0;
                        let mut sum2 = 0.0;
                        for &vv in &v_rep {
                            sum1 += vv;
                            sum2 += vv * vv;
                        }
                        (sum1, sum2)
                    };

                    for t in 0..n_tgt {
                        let dvec = &delta_beta[(bi * n_tgt + t) * k..(bi * n_tgt + t + 1) * k];
                        let mut p = 0.0;
                        for l in 0..k {
                            p += xr[l] * dvec[l];
                        }
                        match &fe_proj {
                            FeProj::None | FeProj::Approx => {}
                            FeProj::OneWay { codes_row, .. } => {
                                let lvl = codes_row[i] as usize;
                                p += gamma_fe[(bi * n_tgt + t) * fe_dim + lvl];
                            }
                            FeProj::Dense { .. } => {
                                let gd = &gamma_fe[(bi * n_tgt + t) * fe_dim..(bi * n_tgt + t + 1) * fe_dim];
                                let mut off = 0usize;
                                for term in hi.terms {
                                    let lvl = term.codes[ki] as usize;
                                    let mut slot = 0usize;
                                    if term.intercept {
                                        p += gd[off + lvl];
                                        slot += 1;
                                    }
                                    for sl in &term.slopes {
                                        p += sl[ki] * gd[off + slot * term.g + lvl];
                                        slot += 1;
                                    }
                                    off += term.g * (usize::from(term.intercept) + term.slopes.len());
                                }
                            }
                        }
                        proj_t[t] = p;
                    }

                    if affine_ci {
                        let u0 = targets_owned[0][i];
                        let u1 = targets_owned[1][i];
                        let p0 = proj_t[0];
                        let p1 = proj_t[1];
                        let (e00, e01, e11) = if f_cnt == 1 {
                            let s0 = wi * (u0 * s1 - p0);
                            let s1v = wi * (u1 * s1 - p1);
                            (s0 * s0, s0 * s1v, s1v * s1v)
                        } else {
                            let fi = wi;
                            (
                                u0 * u0 * s2 - 2.0 * u0 * p0 * s1 + fi * p0 * p0,
                                u0 * u1 * s2 - (u0 * p1 + u1 * p0) * s1 + fi * p0 * p1,
                                u1 * u1 * s2 - 2.0 * u1 * p1 * s1 + fi * p1 * p1,
                            )
                        };
                        let fac = base_mult * zr[0] * zr[0];
                        acc[bi * 3] += fac * e00;
                        acc[bi * 3 + 1] += fac * e01;
                        acc[bi * 3 + 2] += fac * e11;
                    } else {
                        let u0 = targets_owned[0][i];
                        let p0 = proj_t[0];
                        let scalar = if f_cnt == 1 {
                            let st = wi * (u0 * s1 - p0);
                            base_mult * st * st
                        } else {
                            base_mult * (u0 * u0 * s2 - 2.0 * u0 * p0 * s1 + wi * p0 * p0)
                        };
                        let mdst = &mut acc[bi * sym_sz..(bi + 1) * sym_sz];
                        let mut pos = 0usize;
                        for a in 0..q {
                            let sza = scalar * zr[a];
                            for b in a..q {
                                mdst[pos] += sza * zr[b];
                                pos += 1;
                            }
                        }
                    }
                }
            }
        });

        if affine_ci {
            let delta = hi.hyp.r_vec[0] - hi.point.expect("q==1");
            for bi in 0..b_cnt {
                let n0 = r_db[bi * 2];
                let n1 = r_db[bi * 2 + 1];
                let coef = [n0, n1, p2[bi * 3], p2[bi * 3 + 1], p2[bi * 3 + 2]];
                t_boot[b0 + bi] = eval_affine_t(delta, coef[0], coef[1], coef[2], coef[3], coef[4]);
                affine_coefs[b0 + bi] = coef;
            }
        } else {
            let mut v_buf = vec![0.0; q * q];
            for bi in 0..b_cnt {
                let mut pos = 0usize;
                for a in 0..q {
                    for b in a..q {
                        let val = p2[bi * sym_sz + pos];
                        v_buf[a * q + b] = val;
                        v_buf[b * q + a] = val;
                        pos += 1;
                    }
                }
                let nvec = &r_db[bi * q..(bi + 1) * q];
                t_boot[b0 + bi] = eval_stat_and_se(nvec, &v_buf, q).0;
            }
        }
        b0 += b_cnt;
    }

    let (p_val, reps_feasible) = compute_pvalue(stat, &t_boot, q, hi.spec.pval_type);
    let conf_int = if q == 1 && hi.spec.conf_int {
        let alpha = 1.0 - hi.spec.level;
        if let Some(pt) = hi.point {
            if hi.allow_impose_null_affine && hi.spec.impose_null {
                let se_val = se.unwrap_or(f64::NAN);
                let mut trial_tb = vec![0.0; reps];
                let ci = invert_ci_search(pt, se_val, alpha, hi.spec.pval_type, hi.spec.tol, hi.spec.maxiter, reps_feasible, |r_trial| {
                    let delta = r_trial - pt;
                    let st_trial = (pt - r_trial) / se_val;
                    for (dst, c) in trial_tb.iter_mut().zip(&affine_coefs) {
                        *dst = eval_affine_t(delta, c[0], c[1], c[2], c[3], c[4]);
                    }
                    Ok(compute_pvalue(st_trial, &trial_tb, 1, hi.spec.pval_type).0)
                })?;
                Some(ci)
            } else if !hi.spec.impose_null {
                se.map(|se_val| invert_ci_wcu(pt, se_val, &t_boot, hi.spec.pval_type, alpha))
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    Ok(WildBootResult {
        stat,
        p_val,
        stat_type: if q == 1 { hi.stat_type_scalar } else { hi.stat_type_joint },
        point_estimate: hi.point,
        se,
        conf_int,
        t_boot,
        reps,
        reps_feasible,
        nobs: n,
        n_clusters: None,
        q,
        full_enumeration: false,
        method: hi.method,
        boot_type: hi.spec.boot_type,
        weights: hi.spec.weights,
        impose_null: hi.spec.impose_null,
        pval_type: hi.spec.pval_type,
        ssc: eff_ssc,
        notes,
    })
}
