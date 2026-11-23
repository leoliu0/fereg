//! Davidson & MacKinnon (2010) / Roodman et al. (2019 Sec. 4.4) Wild Restricted Efficient
//! (WRE) bootstrap for IV / 2SLS (clustered with exact non-nested FE projection, and unclustered).

use rayon::prelude::*;

use super::ols::{ConstraintProj, eval_stat_and_se, linear_num_obs, matvec};
use super::weights::{cluster_matrix, fill_from_rng, hetero_chunk_rng, resolve_reps};
use super::{
    BootMethod, BootType, Cells, ClusterConfig, Hypothesis, NonNestedPlan, StatType, WildBootResult, WildBootSpec, accumulate_dnn_groups,
    classify_non_nested, compute_pvalue, invert_ci_search, invert_ci_wcu, path2_project_matrices, sym_pinv, to_kept_u32,
};
use crate::absorber::Absorber;
use crate::api::Ctx;
use crate::error::RfError;
use crate::par::{self, CHUNK, Pool};
use crate::session::Term;

/// Inputs for IV/2SLS WRE wild bootstrap in internal `[endog, exog]` / `[excl_inst, exog]` order.
pub(crate) struct IvBootInput<'a> {
    pub n: usize,
    /// Number of endogenous regressors (`m >= 1`).
    pub m: usize,
    /// Number of included exogenous regressors (`k1 >= 0`, `k = m + k1`).
    pub k1: usize,
    /// Number of excluded instruments (`l2 >= m`, `L = l2 + k1`).
    pub l2: usize,
    /// Residualized `[Y_2, X_1]` (`n * (m + k1)`, row-major).
    pub x: &'a [f64],
    /// Residualized `[X_2, X_1]` (`n * (l2 + k1)`, row-major).
    pub z: &'a [f64],
    /// Unrestricted 2SLS residuals `u_hat = y - X * beta` (`n`).
    pub resid: &'a [f64],
    /// Unrestricted 2SLS coefficient estimates (`m + k1`).
    pub beta: &'a [f64],
    /// Unrestricted 2SLS bread `(X' P_Z X)^{-1}` (`k * k`).
    pub bread: &'a [f64],
    /// `(Z' W Z)^{-1}` (`L * L`).
    pub zz_inv: &'a [f64],
    /// First-stage reduced form coefficients `Pi_hat = zz_inv * (Z' W Y_2)` (`L * m`, row-major).
    pub pi_hat: &'a [f64],
    /// First-stage reduced form residuals `U_2_hat = Y_2 - Z * Pi_hat` (`n * m`, row-major).
    pub u2_hat: &'a [f64],
    pub w_rows: Option<&'a [f64]>,
    pub is_fweight: bool,
    pub terms: &'a [Term],
    pub row_order: &'a [u32],
    pub ssc_hetero: f64,
    pub hyp: &'a Hypothesis,
    pub spec: &'a WildBootSpec,
    pub cluster: Option<&'a ClusterConfig<'a>>,
    pub fixef_tol: f64,
    pub fixef_maxiter: usize,
}

/// Dispatches IV/2SLS wild bootstrap (WRE or Score).
pub(crate) fn run(input: IvBootInput<'_>, absorber: Option<&mut dyn Absorber>, ctx: &Ctx, pool: &Pool) -> Result<WildBootResult, RfError> {
    input.spec.validate(input.hyp.q)?;
    let k = input.m + input.k1;
    let l = input.l2 + input.k1;
    if input.hyp.k != k || input.m == 0 || input.l2 < input.m {
        return Err(RfError::InvalidInput("wildboot: invalid IV dimensions for hypothesis or instruments".into()));
    }
    let method = match input.spec.method {
        BootMethod::Auto => {
            if input.spec.boot_type == BootType::T11 {
                BootMethod::Wre
            } else {
                BootMethod::Score
            }
        }
        BootMethod::WildCluster => {
            return Err(RfError::InvalidInput("wildboot: method(wildcluster) is only for OLS; use wre or score for IV".into()));
        }
        m => m,
    };

    if matches!(method, BootMethod::Score | BootMethod::Hetero) {
        return super::score::linear_score_iv(input, absorber, ctx, pool);
    }

    if input.spec.boot_type != BootType::T11 {
        return Err(RfError::InvalidInput("wildboot: WRE only supports bootstrap type 11; use method(score) for 13/31/33".into()));
    }

    if let Some(cl) = input.cluster {
        if cl.g_boot < 2 {
            return Err(RfError::InvalidInput("wildboot: cluster bootstrap requires at least 2 clusters".into()));
        }
        run_wre_clustered(input, absorber, cl, k, l, ctx, pool)
    } else {
        run_wre_unclustered(input, k, l, ctx, pool)
    }
}

/// Computes `gamma_u` (`m`), `lambda` (`L`), `pi_tilde` (`L * m`), and residualized `u_tilde_r` (`n`)
/// at a scalar offset `delta = r - R * beta_hat` (when `q == 1`) or joint step `step` (`k`).
struct RestrictedIvRf {
    /// `gamma_u` (`m`).
    gamma_u: Vec<f64>,
    /// `lambda = zz_inv * (Z' W u_tilde)` (`L`).
    lambda: Vec<f64>,
    /// `Pi_tilde = Pi_hat - lambda * gamma_u'` (`L * m`, row-major).
    pi_tilde: Vec<f64>,
    /// Restricted second-stage parameters `beta_tilde` (`k`).
    beta_tilde: Vec<f64>,
}

fn solve_restricted_rf(
    pi_hat: &[f64],
    zz_inv: &[f64],
    beta_step: &[f64], // step to add to beta_hat: beta_tilde = beta_hat + beta_step
    beta_u: &[f64],
    zwu: &[f64],       // Z' W u_tilde (L)
    utu: f64,          // u_tilde' W u_tilde
    utu2: &[f64],      // u_tilde' W U2_hat (m)
    m: usize,
    l: usize,
) -> RestrictedIvRf {
    let mut lambda = vec![0.0; l];
    matvec(zz_inv, zwu, l, l, &mut lambda);
    let mut pz_u = 0.0;
    for d in 0..l {
        pz_u += zwu[d] * lambda[d];
    }
    let denom = utu - pz_u;
    let mut gamma_u = vec![0.0; m];
    if denom > 1e-14 * utu.max(1e-30) {
        for r in 0..m {
            gamma_u[r] = utu2[r] / denom;
        }
    }
    let mut pi_tilde = pi_hat.to_vec();
    for d in 0..l {
        for r in 0..m {
            pi_tilde[d * m + r] -= lambda[d] * gamma_u[r];
        }
    }
    let beta_tilde: Vec<f64> = beta_u.iter().zip(beta_step).map(|(b, s)| b + s).collect();
    RestrictedIvRf { gamma_u, lambda, pi_tilde, beta_tilde }
}

/// Computes `u2_tilde` and [`RestrictedIvRf`] from a single linear residual vector `ur`.
#[allow(clippy::too_many_arguments)]
fn efficient_rf_from_resid(
    z: &[f64],
    u2_hat: &[f64],
    pi_hat: &[f64],
    zz_inv_glob: &[f64],
    w_rows: Option<&[f64]>,
    ur: &[f64],
    beta_step: &[f64],
    beta_u: &[f64],
    m: usize,
    l: usize,
    n: usize,
    pool: &Pool,
) -> (Vec<f64>, RestrictedIvRf) {
    let stats = par::sum_chunks(pool, n, CHUNK, l + 1 + m, |range, acc| {
        for i in range {
            let wi = w_rows.map_or(1.0, |w| w[i]);
            let wu = wi * ur[i];
            let zr = &z[i * l..(i + 1) * l];
            for d in 0..l {
                acc[d] += wu * zr[d];
            }
            acc[l] += wu * ur[i];
            let u2r = &u2_hat[i * m..(i + 1) * m];
            for r in 0..m {
                acc[l + 1 + r] += wu * u2r[r];
            }
        }
    });
    let rf = solve_restricted_rf(pi_hat, zz_inv_glob, beta_step, beta_u, &stats[..l], stats[l], &stats[l + 1..], m, l);
    let mut u2_tilde = u2_hat.to_vec();
    for i in 0..n {
        let zr = &z[i * l..(i + 1) * l];
        let mut pz_i = 0.0;
        for d in 0..l {
            pz_i += zr[d] * rf.lambda[d];
        }
        for r in 0..m {
            u2_tilde[i * m + r] += pz_i * rf.gamma_u[r];
        }
    }
    (u2_tilde, rf)
}

/// Evaluates `H = (J' A_zz J)^+` (`k * k`), `J = [Pi_boot (L x m), [0; I_k1]]` (`L * k`), and
/// `C = R * H * J'` (`q * L`), plus `delta = H * J' * zw_u1`.
fn wre_solve_2sls(
    pi_boot: &[f64],
    a_zz: &[f64],
    zw_u1: &[f64],
    r_mat: &[f64],
    m: usize,
    k1: usize,
    l2: usize,
    q: usize,
    j_mat: &mut [f64],
    a_zz_j: &mut [f64],
    h_in: &mut [f64],
    rhs: &mut [f64],
    delta: &mut [f64],
    hj: &mut [f64],
    c_out: &mut [f64],
) {
    let k = m + k1;
    let l = l2 + k1;
    // Construct J (`L x k` row-major): first m cols = pi_boot, last k1 cols = [0_{l2 x k1}; I_{k1}].
    j_mat[..l * k].fill(0.0);
    for d in 0..l {
        for r in 0..m {
            j_mat[d * k + r] = pi_boot[d * m + r];
        }
    }
    for j in 0..k1 {
        j_mat[(l2 + j) * k + (m + j)] = 1.0;
    }

    // a_zz_j = A_ZZ * J (`L * k`)
    for d in 0..l {
        let arow = &a_zz[d * l..(d + 1) * l];
        for c in 0..k {
            let mut s = 0.0;
            for r in 0..l {
                s += arow[r] * j_mat[r * k + c];
            }
            a_zz_j[d * k + c] = s;
        }
    }

    // h_in = J' * a_zz_j (`k * k`, symmetric)
    for a in 0..k {
        for b in a..k {
            let mut s = 0.0;
            for d in 0..l {
                s += j_mat[d * k + a] * a_zz_j[d * k + b];
            }
            h_in[a * k + b] = s;
            h_in[b * k + a] = s;
        }
    }

    let h_inv = sym_pinv(h_in, k, 1e-12);
    // rhs = J' * zw_u1 (`k`)
    for a in 0..k {
        let mut s = 0.0;
        for d in 0..l {
            s += j_mat[d * k + a] * zw_u1[d];
        }
        rhs[a] = s;
    }
    matvec(&h_inv, rhs, k, k, delta);

    // hj = H * J' (`k * L`)
    for a in 0..k {
        for d in 0..l {
            let mut s = 0.0;
            for b in 0..k {
                s += h_inv[a * k + b] * j_mat[d * k + b];
            }
            hj[a * l + d] = s;
        }
    }
    // c_out = R * hj (`q * L`)
    for a in 0..q {
        for d in 0..l {
            let mut s = 0.0;
            for b in 0..k {
                s += r_mat[a * k + b] * hj[b * l + d];
            }
            c_out[a * l + d] = s;
        }
    }
}

fn run_wre_clustered(
    input: IvBootInput<'_>,
    mut absorber: Option<&mut dyn Absorber>,
    cl: &ClusterConfig<'_>,
    k: usize,
    l: usize,
    ctx: &Ctx,
    pool: &Pool,
) -> Result<WildBootResult, RfError> {
    let (n, m, k1, l2, q) = (input.n, input.m, input.k1, input.l2, input.hyp.q);
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

    // Global A_ZZ = Z' W Z (L x L) from Z.
    let w_rows = input.w_rows;
    let a_zz_glob = par::sum_chunks(pool, n, CHUNK, l * l, |range, acc| {
        for i in range {
            let wi = w_rows.map_or(1.0, |w| w[i]);
            let zr = &input.z[i * l..(i + 1) * l];
            for d1 in 0..l {
                let wz = wi * zr[d1];
                for d2 in d1..l {
                    acc[d1 * l + d2] += wz * zr[d2];
                }
            }
        }
    });
    let mut a_zz = a_zz_glob;
    for d1 in 0..l {
        for d2 in 0..d1 {
            a_zz[d1 * l + d2] = a_zz[d2 * l + d1];
        }
    }
    let zz_inv_glob = sym_pinv(&a_zz, l, 1e-12);

    // Prepare targets:
    // If affine_ci: we build cells for [u_hat (1), -z0 (1), U2_hat (m), Z (L)] = 2 + m + L cols!
    // Otherwise: just [u_tilde (1), U2_tilde (m)] = 1 + m cols!
    let mut z0_vec = Vec::new();
    let mut single_rf: Option<(Vec<f64>, Vec<f64>, RestrictedIvRf)> = None;

    if affine_ci {
        z0_vec.resize(n, 0.0);
        for i in 0..n {
            let xr = &input.x[i * k..(i + 1) * k];
            let mut s = 0.0;
            for c in 0..k {
                s += xr[c] * proj.m_r[c];
            }
            z0_vec[i] = -s; // u1 = -z0, so u_tilde(delta) = u_hat + delta * u1
        }
    } else {
        let (beta_step, ur) = if !input.spec.impose_null {
            (vec![0.0; k], input.resid.to_vec())
        } else {
            // Compute restricted (beta_tilde, u_tilde) once at r = hyp.r_vec.
            let mut neg_num = vec![0.0; q];
            for a in 0..q {
                neg_num[a] = -num_obs[a];
            }
            let mut step = vec![0.0; k];
            matvec(&proj.m_r, &neg_num, k, q, &mut step);
            let mut resid_r = vec![0.0; n];
            for i in 0..n {
                let xr = &input.x[i * k..(i + 1) * k];
                let mut adj = 0.0;
                for c in 0..k {
                    adj += xr[c] * step[c];
                }
                resid_r[i] = input.resid[i] - adj;
            }
            (step, resid_r)
        };
        let (u2_tilde, rf) = efficient_rf_from_resid(
            input.z,
            input.u2_hat,
            input.pi_hat,
            &zz_inv_glob,
            w_rows,
            &ur,
            &beta_step,
            input.beta,
            m,
            l,
            n,
            pool,
        );
        single_rf = Some((ur, u2_tilde, rf));
    }

    // Build cell-level tensors (plus P_D projection matrices for targets!).
    let n_cells = cells.n_cells;
    let n_tgt = if affine_ci { 2 + m + l } else { 1 + m };

    // When affine_ci == true: also precompute inner products for gamma_u(delta):
    // u_tilde(delta) = u_0 + delta * u_1 (where u_0 = resid, u_1 = -z0)
    // We need zwu0[L], zwu1[L], utu_{00,01,11}, utu2_0[m], utu2_1[m].
    let mut scalar_ci_pre = [0.0; 3]; // utu00, utu01, utu11
    let mut zwu0 = vec![0.0; l];
    let mut zwu1 = vec![0.0; l];
    let mut utu2_0 = vec![0.0; m];
    let mut utu2_1 = vec![0.0; m];
    if affine_ci {
        let s = par::sum_chunks(pool, n, CHUNK, 3 + 2 * l + 2 * m, |range, acc| {
            for i in range {
                let wi = w_rows.map_or(1.0, |w| w[i]);
                let u0 = input.resid[i];
                let u1 = z0_vec[i];
                acc[0] += wi * u0 * u0;
                acc[1] += wi * u0 * u1;
                acc[2] += wi * u1 * u1;
                let zr = &input.z[i * l..(i + 1) * l];
                let wu0 = wi * u0;
                let wu1 = wi * u1;
                for d in 0..l {
                    acc[3 + d] += wu0 * zr[d];
                    acc[3 + l + d] += wu1 * zr[d];
                }
                let u2r = &input.u2_hat[i * m..(i + 1) * m];
                for r in 0..m {
                    acc[3 + 2 * l + r] += wu0 * u2r[r];
                    acc[3 + 2 * l + m + r] += wu1 * u2r[r];
                }
            }
        });
        scalar_ci_pre.copy_from_slice(&s[0..3]);
        zwu0.copy_from_slice(&s[3..3 + l]);
        zwu1.copy_from_slice(&s[3 + l..3 + 2 * l]);
        utu2_0.copy_from_slice(&s[3 + 2 * l..3 + 2 * l + m]);
        utu2_1.copy_from_slice(&s[3 + 2 * l + m..]);
    }

    // Helper closure reading target col `t` at row `i`.
    let read_tgt = |i: usize, t: usize| -> f64 {
        if affine_ci {
            if t == 0 {
                input.resid[i]
            } else if t == 1 {
                z0_vec[i]
            } else if t < 2 + m {
                input.u2_hat[i * m + (t - 2)]
            } else {
                input.z[i * l + (t - 2 - m)]
            }
        } else {
            let (u1, u2, _) = single_rf.as_ref().expect("single_rf");
            if t == 0 { u1[i] } else { u2[i * m + (t - 1)] }
        }
    };

    // Single pass accumulates:
    // - obs cell score q_obs_z (`n_cells * L`): Z_c' W_c u_hat_c
    // - a_zz_cell (`n_cells * L * L`): Z_c' W_c Z_c
    // - a_ztgt_cell (`n_cells * L * n_tgt`): Z_c' W_c target_c (note: includes g via cell_boot)
    let len_qobs = n_cells * l;
    let len_azzc = n_cells * l * l;
    let len_aztc = n_cells * l * n_tgt;
    let packed = par::sum_chunks(pool, n, CHUNK, len_qobs + len_azzc + len_aztc, |range, acc| {
        for i in range {
            let wi = w_rows.map_or(1.0, |w| w[i]);
            let cell = cells.cell_of_row[i] as usize;
            let zr = &input.z[i * l..(i + 1) * l];
            let qdst = &mut acc[cell * l..(cell + 1) * l];
            let wu_obs = wi * input.resid[i];
            for d in 0..l {
                qdst[d] += wu_obs * zr[d];
            }
            let azzc = &mut acc[len_qobs + cell * l * l..len_qobs + (cell + 1) * l * l];
            for d1 in 0..l {
                let wz = wi * zr[d1];
                for d2 in d1..l {
                    azzc[d1 * l + d2] += wz * zr[d2];
                }
            }
            let aztc = &mut acc[len_qobs + len_azzc + cell * l * n_tgt..len_qobs + len_azzc + (cell + 1) * l * n_tgt];
            for d in 0..l {
                let wz = wi * zr[d];
                for t in 0..n_tgt {
                    aztc[d * n_tgt + t] += wz * read_tgt(i, t);
                }
            }
        }
    });

    let q_obs_z = &packed[..len_qobs];
    let mut a_zz_cell = packed[len_qobs..len_qobs + len_azzc].to_vec();
    for c in 0..n_cells {
        let azzc = &mut a_zz_cell[c * l * l..(c + 1) * l * l];
        for d1 in 0..l {
            for d2 in 0..d1 {
                azzc[d1 * l + d2] = azzc[d2 * l + d1];
            }
        }
    }
    let a_ztgt_cell = &packed[len_qobs + len_azzc..];

    // Boot-cluster sums `a_ztgt_boot[g, d, t]` = sum_{c in g} a_ztgt_cell[c, d, t]
    let mut a_ztgt_boot = vec![0.0; g_boot * l * n_tgt];
    for c in 0..n_cells {
        let g = cells.cell_boot[c] as usize;
        let src = &a_ztgt_cell[c * l * n_tgt..(c + 1) * l * n_tgt];
        let dst = &mut a_ztgt_boot[g * l * n_tgt..(g + 1) * l * n_tgt];
        for idx in 0..l * n_tgt {
            dst[idx] += src[idx];
        }
    }

    // Observed variance and stat:
    // C_obs = R * H_obs * J_obs' = R * bmat (`q * L`)
    let mut c_obs = vec![0.0; q * l];
    let mut a_zz_j_obs = vec![0.0; l * k];
    let mut h_obs_buf = vec![0.0; k * k];
    let mut j_obs_buf = vec![0.0; l * k];
    let mut rhs_obs = vec![0.0; k];
    let mut delta_obs = vec![0.0; k];
    let mut hj_obs = vec![0.0; k * l];
    let zero_l = vec![0.0; l];
    wre_solve_2sls(
        input.pi_hat,
        &a_zz,
        &zero_l,
        &input.hyp.r_mat,
        m,
        k1,
        l2,
        q,
        &mut j_obs_buf,
        &mut a_zz_j_obs,
        &mut h_obs_buf,
        &mut rhs_obs,
        &mut delta_obs,
        &mut hj_obs,
        &mut c_obs,
    );

    let mut q_cell_obs = vec![0.0; n_cells * q];
    for c in 0..n_cells {
        matvec(&c_obs, &q_obs_z[c * l..(c + 1) * l], q, l, &mut q_cell_obs[c * q..(c + 1) * q]);
    }
    let mut v_obs = vec![0.0; q * q];
    cells.variance_from_cell_q(&q_cell_obs, q, cl.ssc_factor, cl.per_subset_g_adj, &mut v_obs);
    let (stat, se) = eval_stat_and_se(&num_obs, &v_obs, q);

    // Non-nested FE projection matrices for `Z_c' W_c P_D(1_g * target_t)`.
    // For Path 1: P_Z_tgt_cell[c, g] requires F_boot[g, t, p] (`g_boot * n_tgt * m_nn`),
    // psi_boot = G_nn^+ F_boot, F_cell_z = accumulate_dnn_groups(Z).
    // For Path 2: path2_project_matrices produces `((c * l + d) * n_tgt + t) * g_boot + g`.
    enum NnWre {
        None,
        Path1 { m_nn: usize, psi: Vec<f64>, f_cell_z: Vec<f64> },
        Path2 { p2_mat: Vec<f64> },
    }
    let fill_z_all = |i: usize, out: &mut [f64]| {
        out.copy_from_slice(&input.z[i * l..(i + 1) * l]);
    };
    let nn_wre = match plan {
        NonNestedPlan::None => NnWre::None,
        NonNestedPlan::TooLarge { g_boot: gb, m_nn } => {
            notes.push(format!("warning: fixed effects non-nested in bootstrap clusters (G={gb}, FE params={m_nn} > 512); exact non-nested FE projection skipped"));
            NnWre::None
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
                        out[t] = read_tgt(i, t);
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
                n_cells,
                w_rows,
                l,
                fill_z_all,
                pool,
            );
            NnWre::Path1 { m_nn, psi, f_cell_z }
        }
        NonNestedPlan::Path2 => {
            let mut tgt_vecs: Vec<Vec<f64>> = (0..n_tgt).map(|t| (0..n).map(|i| read_tgt(i, t)).collect()).collect();
            let tgt_refs: Vec<&[f64]> = tgt_vecs.iter().map(Vec::as_slice).collect();
            let abs = absorber.as_deref_mut().ok_or_else(|| RfError::Internal("wildboot WRE Path 2 requires absorber".into()))?;
            let p2_mat = path2_project_matrices(
                abs,
                cl.boot_codes,
                g_boot,
                &cells.cell_of_row,
                n_cells,
                w_rows,
                &tgt_refs,
                l,
                fill_z_all,
                input.fixef_tol,
                input.fixef_maxiter,
                pool,
                ctx,
            )?;
            // Free `tgt_vecs` immediately.
            tgt_vecs.clear();
            NnWre::Path2 { p2_mat }
        }
    };

    // Precompute per-draw `P_cell` basis tensors when affine_ci == true, or run fixed-`r` evaluation:
    // Actually look how fast precomputing per-draw cluster matrices is for WRE:
    // For fixed `(r_trial, delta_trial, rf)`: we already have ALL cell-level `a_ztgt_cell` (`n_cells * L * n_tgt`),
    // `a_ztgt_boot` (`g_boot * L * n_tgt`), and `P_D` (`psi` + `f_cell_z` or `p2_mat`)!
    // Wait: during `invert_ci_search`, ONLY the linear combination weights of targets `0..n_tgt` change!
    // Specifically, for ANY `delta = r_trial - point`:
    //   target u_1 is: w_{u1} = [1.0, delta] on cols 0, 1
    //   target U_2[:, r] is: col (2 + r) + gamma_u[r] * sum_d lambda[d] * col(2 + m + d)!
    // So inside the draw loop for a trial `delta`, zero row scans and zero FE solves occur: only O(g_boot * L * m + n_cells * L) work per draw!
    // And even faster: precompute for each draw `b`:
    //   Wait - in `NonNestedPlan::Path1`, `gamma_b` has `n_tgt * m_nn` elements, or in `Path2`,
    //   we only combine target slots by `(1, delta)` and `(gamma_u, lambda)`!
    let eval_at = |beta_tilde: &[f64], pi_tilde: &[f64], w_u1_pair: (f64, f64), rf_gam: &[f64], rf_lam: &[f64], r_vec: &[f64]| -> Result<(f64, Vec<f64>), RfError> {
        let mut t_boot = vec![0.0; reps];
        let mut num_tilde_offset = vec![0.0; q]; // R * beta_tilde - r (0 under null; 0 under beta_u if r = R beta_u)
        if !input.spec.impose_null {
            num_tilde_offset.copy_from_slice(&num_obs);
        } else {
            // Note R * beta_tilde - r = 0 up to roundoff; compute explicitly for precision.
            for a in 0..q {
                let mut s = -r_vec[a];
                for c in 0..k {
                    s += input.hyp.r_mat[a * k + c] * beta_tilde[c];
                }
                num_tilde_offset[a] = s;
            }
        }
        let m_nn_val = match &nn_wre {
            NnWre::Path1 { m_nn, .. } => *m_nn,
            _ => 0,
        };

        // Precompute transformed `g_boot * L * (1 + m)` boot sums and `n_cells * L * (1 + m)` cell sums
        // for the current restricted `(u_tilde, U2_tilde)` (takes microseconds in cell space!).
        let mut boot_u1 = vec![0.0; g_boot * l];
        let mut boot_u2 = vec![0.0; g_boot * l * m];
        let mut cell_u1 = vec![0.0; n_cells * l];
        let mut cell_u2 = vec![0.0; n_cells * l * m];

        if affine_ci {
            for g in 0..g_boot {
                for d in 0..l {
                    let s0 = a_ztgt_boot[(g * l + d) * n_tgt];
                    let s1 = a_ztgt_boot[(g * l + d) * n_tgt + 1];
                    boot_u1[g * l + d] = w_u1_pair.0 * s0 + w_u1_pair.1 * s1;
                    for r in 0..m {
                        let base = a_ztgt_boot[(g * l + d) * n_tgt + 2 + r];
                        let mut pz = 0.0;
                        for dd in 0..l {
                            pz += rf_lam[dd] * a_ztgt_boot[(g * l + d) * n_tgt + 2 + m + dd];
                        }
                        boot_u2[(g * l + d) * m + r] = base + rf_gam[r] * pz;
                    }
                }
            }
            for c in 0..n_cells {
                for d in 0..l {
                    let s0 = a_ztgt_cell[(c * l + d) * n_tgt];
                    let s1 = a_ztgt_cell[(c * l + d) * n_tgt + 1];
                    cell_u1[c * l + d] = w_u1_pair.0 * s0 + w_u1_pair.1 * s1;
                    for r in 0..m {
                        let base = a_ztgt_cell[(c * l + d) * n_tgt + 2 + r];
                        let mut pz = 0.0;
                        for dd in 0..l {
                            pz += rf_lam[dd] * a_ztgt_cell[(c * l + d) * n_tgt + 2 + m + dd];
                        }
                        cell_u2[(c * l + d) * m + r] = base + rf_gam[r] * pz;
                    }
                }
            }
        } else {
            for g in 0..g_boot {
                for d in 0..l {
                    boot_u1[g * l + d] = a_ztgt_boot[(g * l + d) * n_tgt];
                    for r in 0..m {
                        boot_u2[(g * l + d) * m + r] = a_ztgt_boot[(g * l + d) * n_tgt + 1 + r];
                    }
                }
            }
            for c in 0..n_cells {
                for d in 0..l {
                    cell_u1[c * l + d] = a_ztgt_cell[(c * l + d) * n_tgt];
                    for r in 0..m {
                        cell_u2[(c * l + d) * m + r] = a_ztgt_cell[(c * l + d) * n_tgt + 1 + r];
                    }
                }
            }
        }

        // Precompute transformed Path 1 `psi` (`g_boot * (1 + m) * m_nn`) for this `(delta, rf)` if needed.
        let psi_cur: Vec<f64> = if let NnWre::Path1 { m_nn, psi, .. } = &nn_wre {
            let mut out = vec![0.0; g_boot * (1 + m) * *m_nn];
            for g in 0..g_boot {
                if affine_ci {
                    let dst1 = &mut out[(g * (1 + m)) * *m_nn..(g * (1 + m) + 1) * *m_nn];
                    let p0 = &psi[(g * n_tgt) * *m_nn..(g * n_tgt + 1) * *m_nn];
                    let p1 = &psi[(g * n_tgt + 1) * *m_nn..(g * n_tgt + 2) * *m_nn];
                    for p in 0..*m_nn {
                        dst1[p] = w_u1_pair.0 * p0[p] + w_u1_pair.1 * p1[p];
                    }
                    for r in 0..m {
                        let dst2 = &mut out[(g * (1 + m) + 1 + r) * *m_nn..(g * (1 + m) + 2 + r) * *m_nn];
                        let pr = &psi[(g * n_tgt + 2 + r) * *m_nn..(g * n_tgt + 3 + r) * *m_nn];
                        dst2.copy_from_slice(pr);
                        let gr = rf_gam[r];
                        if gr != 0.0 {
                            for dd in 0..l {
                                let ld = rf_lam[dd] * gr;
                                let pd = &psi[(g * n_tgt + 2 + m + dd) * *m_nn..(g * n_tgt + 2 + m + dd + 1) * *m_nn];
                                for p in 0..*m_nn {
                                    dst2[p] += ld * pd[p];
                                }
                            }
                        }
                    }
                } else {
                    let src = &psi[g * n_tgt * *m_nn..(g + 1) * n_tgt * *m_nn];
                    let dst = &mut out[g * (1 + m) * *m_nn..(g + 1) * (1 + m) * *m_nn];
                    dst.copy_from_slice(src);
                }
            }
            out
        } else {
            Vec::new()
        };

        pool.install(|| {
            t_boot.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, tc)| {
                let mut zw_u1 = vec![0.0; l];
                let mut zw_u2 = vec![0.0; l * m];
                let mut pi_b = vec![0.0; l * m];
                let mut j_mat = vec![0.0; l * k];
                let mut a_zz_j = vec![0.0; l * k];
                let mut h_in = vec![0.0; k * k];
                let mut rhs = vec![0.0; k];
                let mut delta_b = vec![0.0; k];
                let mut hj = vec![0.0; k * l];
                let mut c_b = vec![0.0; q * l];
                let mut num_b = vec![0.0; q];
                let mut eta_b = vec![0.0; l];
                let mut sz_c = vec![0.0; l];
                let mut q_cell = vec![0.0; n_cells * q];
                let mut v_b = vec![0.0; q * q];
                let mut gamma_fe = vec![0.0; (1 + m) * m_nn_val];
                let mut p_u1 = vec![0.0; l];
                let mut p_u2 = vec![0.0; l * m];

                let base = ci * CHUNK;
                for (j, tb) in tc.iter_mut().enumerate() {
                    let b = base + j;
                    let vb = &v_mat[b * g_boot..(b + 1) * g_boot];
                    zw_u1.fill(0.0);
                    zw_u2.fill(0.0);
                    for g in 0..g_boot {
                        let vg = vb[g];
                        let bu1 = &boot_u1[g * l..(g + 1) * l];
                        for d in 0..l {
                            zw_u1[d] += vg * bu1[d];
                        }
                        let bu2 = &boot_u2[g * l * m..(g + 1) * l * m];
                        for idx in 0..l * m {
                            zw_u2[idx] += vg * bu2[idx];
                        }
                    }

                    // pi_b = pi_tilde + zz_inv * zw_u2 (`L * m`)
                    pi_b.copy_from_slice(pi_tilde);
                    for d in 0..l {
                        let zrow = &zz_inv_glob[d * l..(d + 1) * l];
                        for r in 0..m {
                            let mut s = 0.0;
                            for dd in 0..l {
                                s += zrow[dd] * zw_u2[dd * m + r];
                            }
                            pi_b[d * m + r] += s;
                        }
                    }

                    wre_solve_2sls(
                        &pi_b,
                        &a_zz,
                        &zw_u1,
                        &input.hyp.r_mat,
                        m,
                        k1,
                        l2,
                        q,
                        &mut j_mat,
                        &mut a_zz_j,
                        &mut h_in,
                        &mut rhs,
                        &mut delta_b,
                        &mut hj,
                        &mut c_b,
                    );

                    // Notice for UCU (impose_null == false), test H0*b: R*beta* = R*beta_hat => num*b = R * delta_b!
                    // For UCR (impose_null == true), test H0*b: R*beta* = r => num*b = R * delta_b!
                    matvec(&input.hyp.r_mat, &delta_b, q, k, &mut num_b);

                    // eta_b = Pi_tilde * delta_2 + [0_{l2}; delta_1]
                    for d in 0..l {
                        let mut s = 0.0;
                        for r in 0..m {
                            s += pi_tilde[d * m + r] * delta_b[r];
                        }
                        if d >= l2 {
                            s += delta_b[m + (d - l2)];
                        }
                        eta_b[d] = s;
                    }

                    if let NnWre::Path1 { m_nn, .. } = &nn_wre {
                        gamma_fe.fill(0.0);
                        for g in 0..g_boot {
                            let vg = vb[g];
                            let src = &psi_cur[g * (1 + m) * *m_nn..(g + 1) * (1 + m) * *m_nn];
                            for idx in 0..(1 + m) * *m_nn {
                                gamma_fe[idx] += vg * src[idx];
                            }
                        }
                    }

                    for c in 0..n_cells {
                        let vg = vb[cells.cell_boot[c] as usize];
                        let cu1 = &cell_u1[c * l..(c + 1) * l];
                        let cu2 = &cell_u2[c * l * m..(c + 1) * l * m];
                        let azzc = &a_zz_cell[c * l * l..(c + 1) * l * l];

                        p_u1.fill(0.0);
                        p_u2.fill(0.0);
                        match &nn_wre {
                            NnWre::None => {}
                            NnWre::Path1 { m_nn, f_cell_z, .. } => {
                                let fc = &f_cell_z[c * l * *m_nn..(c + 1) * l * *m_nn];
                                let g1 = &gamma_fe[..*m_nn];
                                for d in 0..l {
                                    let frow = &fc[d * *m_nn..(d + 1) * *m_nn];
                                    let mut s = 0.0;
                                    for p in 0..*m_nn {
                                        s += frow[p] * g1[p];
                                    }
                                    p_u1[d] = s;
                                    for r in 0..m {
                                        let gr = &gamma_fe[(1 + r) * *m_nn..(2 + r) * *m_nn];
                                        let mut sr = 0.0;
                                        for p in 0..*m_nn {
                                            sr += frow[p] * gr[p];
                                        }
                                        p_u2[d * m + r] = sr;
                                    }
                                }
                            }
                            NnWre::Path2 { p2_mat } => {
                                for d in 0..l {
                                    if affine_ci {
                                        let r0 = &p2_mat[((c * l + d) * n_tgt) * g_boot..((c * l + d) * n_tgt + 1) * g_boot];
                                        let r1 = &p2_mat[((c * l + d) * n_tgt + 1) * g_boot..((c * l + d) * n_tgt + 2) * g_boot];
                                        let mut s = 0.0;
                                        for g in 0..g_boot {
                                            s += vb[g] * (w_u1_pair.0 * r0[g] + w_u1_pair.1 * r1[g]);
                                        }
                                        p_u1[d] = s;
                                        for r in 0..m {
                                            let rr = &p2_mat[((c * l + d) * n_tgt + 2 + r) * g_boot..((c * l + d) * n_tgt + 3 + r) * g_boot];
                                            let mut sr = 0.0;
                                            for g in 0..g_boot {
                                                sr += vb[g] * rr[g];
                                            }
                                            let gr = rf_gam[r];
                                            if gr != 0.0 {
                                                for dd in 0..l {
                                                    let ld = rf_lam[dd] * gr;
                                                    let rd = &p2_mat[((c * l + d) * n_tgt + 2 + m + dd) * g_boot
                                                        ..((c * l + d) * n_tgt + 2 + m + dd + 1) * g_boot];
                                                    for g in 0..g_boot {
                                                        sr += ld * vb[g] * rd[g];
                                                    }
                                                }
                                            }
                                            p_u2[d * m + r] = sr;
                                        }
                                    } else {
                                        let r0 = &p2_mat[((c * l + d) * n_tgt) * g_boot..((c * l + d) * n_tgt + 1) * g_boot];
                                        let mut s = 0.0;
                                        for g in 0..g_boot {
                                            s += vb[g] * r0[g];
                                        }
                                        p_u1[d] = s;
                                        for r in 0..m {
                                            let rr = &p2_mat[((c * l + d) * n_tgt + 1 + r) * g_boot..((c * l + d) * n_tgt + 2 + r) * g_boot];
                                            let mut sr = 0.0;
                                            for g in 0..g_boot {
                                                sr += vb[g] * rr[g];
                                            }
                                            p_u2[d * m + r] = sr;
                                        }
                                    }
                                }
                            }
                        }

                        for d in 0..l {
                            let mut az_eta = 0.0;
                            let arow = &azzc[d * l..(d + 1) * l];
                            for dd in 0..l {
                                az_eta += arow[dd] * eta_b[dd];
                            }
                            let mut u2_d2 = 0.0;
                            for r in 0..m {
                                u2_d2 += (vg * cu2[d * m + r] - p_u2[d * m + r]) * delta_b[r];
                            }
                            sz_c[d] = vg * cu1[d] - p_u1[d] - az_eta - u2_d2;
                        }
                        matvec(&c_b, &sz_c, q, l, &mut q_cell[c * q..(c + 1) * q]);
                    }

                    cells.variance_from_cell_q(&q_cell, q, cl.ssc_factor, cl.per_subset_g_adj, &mut v_b);
                    *tb = eval_stat_and_se(&num_b, &v_b, q).0;
                }
            });
        });
        let _ = num_tilde_offset;
        let (p_val, _) = compute_pvalue(stat, &t_boot, q, input.spec.pval_type);
        Ok((p_val, t_boot))
    };

    // Helper building restricted RF at a scalar `r_trial` (when `affine_ci == true`).
    let rf_for_r = |r_trial: f64| -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let delta = r_trial - point.expect("q==1");
        let step: Vec<f64> = proj.m_r.iter().map(|&v| v * delta).collect();
        let mut zwu = vec![0.0; l];
        for d in 0..l {
            zwu[d] = zwu0[d] + delta * zwu1[d];
        }
        let utu = scalar_ci_pre[0] + delta * (2.0 * scalar_ci_pre[1] + delta * scalar_ci_pre[2]);
        let mut utu2 = vec![0.0; m];
        for r in 0..m {
            utu2[r] = utu2_0[r] + delta * utu2_1[r];
        }
        let rf = solve_restricted_rf(input.pi_hat, &zz_inv_glob, &step, input.beta, &zwu, utu, &utu2, m, l);
        (rf.beta_tilde, rf.pi_tilde, rf.gamma_u, rf.lambda)
    };

    let t_boot = if affine_ci {
        let (bt, pt, gam, lam) = rf_for_r(input.hyp.r_vec[0]);
        let delta = input.hyp.r_vec[0] - point.expect("q==1");
        eval_at(&bt, &pt, (1.0, delta), &gam, &lam, &input.hyp.r_vec)?.1
    } else {
        let (_, _, rf) = single_rf.as_ref().expect("single_rf");
        eval_at(&rf.beta_tilde, &rf.pi_tilde, (1.0, 0.0), &rf.gamma_u, &rf.lambda, &input.hyp.r_vec)?.1
    };

    let (p_val, reps_feasible) = compute_pvalue(stat, &t_boot, q, input.spec.pval_type);

    let conf_int = if q == 1 && input.spec.conf_int {
        let alpha = 1.0 - input.spec.level;
        let pt = point.expect("q==1");
        if input.spec.impose_null {
            let se_val = se.unwrap_or(f64::NAN);
            let ci = invert_ci_search(pt, se_val, alpha, input.spec.pval_type, input.spec.tol, input.spec.maxiter, reps_feasible, |r_trial| {
                let delta = r_trial - pt;
                let st_trial = (pt - r_trial) / se_val;
                let (bt, ptilde, gam, lam) = rf_for_r(r_trial);
                let tb = eval_at(&bt, &ptilde, (1.0, delta), &gam, &lam, &[r_trial])?.1;
                Ok(compute_pvalue(st_trial, &tb, 1, input.spec.pval_type).0)
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
        method: BootMethod::Wre,
        boot_type: input.spec.boot_type,
        weights: input.spec.weights,
        impose_null: input.spec.impose_null,
        pval_type: input.spec.pval_type,
        ssc: cl.ssc_factor,
        notes,
    })
}

/// Unclustered WRE bootstrap using two-pass streaming over `CHUNK`.
fn run_wre_unclustered(input: IvBootInput<'_>, k: usize, l: usize, ctx: &Ctx, pool: &Pool) -> Result<WildBootResult, RfError> {
    let (n, m, k1, l2, q) = (input.n, input.m, input.k1, input.l2, input.hyp.q);
    let mut notes = Vec::new();
    let proj = ConstraintProj::new(input.bread, &input.hyp.r_mat, q, k);
    let (num_obs, point) = linear_num_obs(input.beta, input.hyp);
    let w_rows = input.w_rows;

    // FE projection helper for unclustered WRE (same hierarchy as Hetero OLS).
    enum FeProjWre {
        None,
        OneWay { codes_row: Vec<u32>, g: usize, inv_sum_w: Vec<f64> },
        Dense { m_fe: usize, g_pinv: Vec<f64> },
        Approx,
    }
    let fe_proj = if input.terms.is_empty() {
        FeProjWre::None
    } else if input.terms.len() == 1 && input.terms[0].intercept && input.terms[0].slopes.is_empty() {
        let g = input.terms[0].g;
        let mut codes_row = vec![0u32; n];
        for (i, &ki) in input.row_order.iter().enumerate() {
            codes_row[i] = input.terms[0].codes[ki as usize];
        }
        let mut sum_w = vec![0.0; g];
        for (i, &lvl) in codes_row.iter().enumerate() {
            sum_w[lvl as usize] += w_rows.map_or(1.0, |w| w[i]);
        }
        let inv_sum_w: Vec<f64> = sum_w.into_iter().map(|s| if s > 0.0 { 1.0 / s } else { 0.0 }).collect();
        FeProjWre::OneWay { codes_row, g, inv_sum_w }
    } else {
        let m_fe: usize = input.terms.iter().map(|t| t.g * (usize::from(t.intercept) + t.slopes.len())).sum();
        if m_fe <= 256 {
            let mut offsets = Vec::with_capacity(input.terms.len());
            let mut off = 0usize;
            for t in input.terms {
                offsets.push(off);
                off += t.g * (usize::from(t.intercept) + t.slopes.len());
            }
            let g_fe = par::sum_chunks(pool, n, CHUNK, m_fe * m_fe, |range, acc| {
                let mut cols: Vec<(usize, f64)> = Vec::with_capacity(16);
                for i in range {
                    let ki = input.row_order[i] as usize;
                    let wi = w_rows.map_or(1.0, |w| w[i]);
                    cols.clear();
                    for (qi, t) in input.terms.iter().enumerate() {
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
            FeProjWre::Dense { m_fe, g_pinv: sym_pinv(&g_fe, m_fe, 1e-10) }
        } else {
            notes.push("warning: unclustered WRE with large multi-term fixed effects approximates residual FE partial-out".into());
            FeProjWre::Approx
        }
    };
    let fe_dim = match &fe_proj {
        FeProjWre::None | FeProjWre::Approx => 0,
        FeProjWre::OneWay { g, .. } => *g,
        FeProjWre::Dense { m_fe, .. } => *m_fe,
    };

    // Global A_ZZ = Z' W Z (`L * L`), plus scalars for unrestricted/restricted RF.
    let a_zz_half = par::sum_chunks(pool, n, CHUNK, l * l, |range, acc| {
        for i in range {
            let wi = w_rows.map_or(1.0, |w| w[i]);
            let zr = &input.z[i * l..(i + 1) * l];
            for d1 in 0..l {
                let wz = wi * zr[d1];
                for d2 in d1..l {
                    acc[d1 * l + d2] += wz * zr[d2];
                }
            }
        }
    });
    let mut a_zz = a_zz_half;
    for d1 in 0..l {
        for d2 in 0..d1 {
            a_zz[d1 * l + d2] = a_zz[d2 * l + d1];
        }
    }
    let zz_inv_glob = sym_pinv(&a_zz, l, 1e-12);

    // Observed variance and stat via C_obs = R * bread * X' Z zz_inv:
    let mut c_obs = vec![0.0; q * l];
    let mut a_zz_j_obs = vec![0.0; l * k];
    let mut h_obs_buf = vec![0.0; k * k];
    let mut j_obs_buf = vec![0.0; l * k];
    let mut rhs_obs = vec![0.0; k];
    let mut delta_obs = vec![0.0; k];
    let mut hj_obs = vec![0.0; k * l];
    let zero_l = vec![0.0; l];
    wre_solve_2sls(
        input.pi_hat,
        &a_zz,
        &zero_l,
        &input.hyp.r_mat,
        m,
        k1,
        l2,
        q,
        &mut j_obs_buf,
        &mut a_zz_j_obs,
        &mut h_obs_buf,
        &mut rhs_obs,
        &mut delta_obs,
        &mut hj_obs,
        &mut c_obs,
    );
    let mut v_obs = vec![0.0; q * q];
    let mut cz = vec![0.0; q];
    for i in 0..n {
        let wi = w_rows.map_or(1.0, |w| w[i]);
        let w2 = if input.is_fweight { wi } else { wi * wi };
        let u = input.resid[i];
        let fac = input.ssc_hetero * w2 * u * u;
        matvec(&c_obs, &input.z[i * l..(i + 1) * l], q, l, &mut cz);
        for a in 0..q {
            for b in a..q {
                v_obs[a * q + b] += fac * cz[a] * cz[b];
            }
        }
    }
    for a in 0..q {
        for b in 0..a {
            v_obs[a * q + b] = v_obs[b * q + a];
        }
    }
    let (stat, se) = eval_stat_and_se(&num_obs, &v_obs, q);

    let mut z0_vec = Vec::new();
    let mut pz_u0 = Vec::new();
    let mut pz_u1 = Vec::new();
    let mut zwu0 = vec![0.0; l];
    let mut zwu1 = vec![0.0; l];
    let mut utu2_0 = vec![0.0; m];
    let mut utu2_1 = vec![0.0; m];
    let mut scalar_pre = [0.0; 3];
    if input.spec.impose_null {
        z0_vec.resize(n, 0.0);
        if q == 1 {
            for i in 0..n {
                let xr = &input.x[i * k..(i + 1) * k];
                let mut s = 0.0;
                for c in 0..k {
                    s += xr[c] * proj.m_r[c];
                }
                z0_vec[i] = -s;
            }
        } else {
            let mut neg_num = vec![0.0; q];
            for a in 0..q {
                neg_num[a] = -num_obs[a];
            }
            let mut step = vec![0.0; k];
            matvec(&proj.m_r, &neg_num, k, q, &mut step);
            for i in 0..n {
                let xr = &input.x[i * k..(i + 1) * k];
                let mut s = 0.0;
                for c in 0..k {
                    s += xr[c] * step[c];
                }
                z0_vec[i] = input.resid[i] - s;
            }
        }
        let s = par::sum_chunks(pool, n, CHUNK, 3 + 2 * l + 2 * m, |range, acc| {
            for i in range {
                let wi = w_rows.map_or(1.0, |w| w[i]);
                let u0 = input.resid[i];
                let u1 = z0_vec[i];
                acc[0] += wi * u0 * u0;
                acc[1] += wi * u0 * u1;
                acc[2] += wi * u1 * u1;
                let zr = &input.z[i * l..(i + 1) * l];
                let wu0 = wi * u0;
                let wu1 = wi * u1;
                for d in 0..l {
                    acc[3 + d] += wu0 * zr[d];
                    acc[3 + l + d] += wu1 * zr[d];
                }
                let u2r = &input.u2_hat[i * m..(i + 1) * m];
                for r in 0..m {
                    acc[3 + 2 * l + r] += wu0 * u2r[r];
                    acc[3 + 2 * l + m + r] += wu1 * u2r[r];
                }
            }
        });
        scalar_pre.copy_from_slice(&s[0..3]);
        zwu0.copy_from_slice(&s[3..3 + l]);
        zwu1.copy_from_slice(&s[3 + l..3 + 2 * l]);
        utu2_0.copy_from_slice(&s[3 + 2 * l..3 + 2 * l + m]);
        utu2_1.copy_from_slice(&s[3 + 2 * l + m..]);
        if q == 1 && input.spec.conf_int {
            let mut l0 = vec![0.0; l];
            let mut l1 = vec![0.0; l];
            matvec(&zz_inv_glob, &zwu0, l, l, &mut l0);
            matvec(&zz_inv_glob, &zwu1, l, l, &mut l1);
            pz_u0.resize(n, 0.0);
            pz_u1.resize(n, 0.0);
            for i in 0..n {
                let zr = &input.z[i * l..(i + 1) * l];
                let mut a0 = 0.0;
                let mut a1 = 0.0;
                for d in 0..l {
                    a0 += zr[d] * l0[d];
                    a1 += zr[d] * l1[d];
                }
                pz_u0[i] = a0;
                pz_u1[i] = a1;
            }
        }
    }

    let build_u1_u2 = |delta_opt: Option<f64>| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        if !input.spec.impose_null {
            let (u2, rf) = efficient_rf_from_resid(
                input.z,
                input.u2_hat,
                input.pi_hat,
                &zz_inv_glob,
                w_rows,
                input.resid,
                &vec![0.0; k],
                input.beta,
                m,
                l,
                n,
                pool,
            );
            return (input.resid.to_vec(), u2, rf.pi_tilde);
        }
        let delta = delta_opt.unwrap_or(1.0);
        let (u1, zwu, utu, utu2): (Vec<f64>, Vec<f64>, f64, Vec<f64>) = if q == 1 {
            let mut u1 = vec![0.0; n];
            for i in 0..n {
                u1[i] = input.resid[i] + delta * z0_vec[i];
            }
            let zwu: Vec<f64> = zwu0.iter().zip(&zwu1).map(|(a, b)| a + delta * b).collect();
            let utu = scalar_pre[0] + delta * (2.0 * scalar_pre[1] + delta * scalar_pre[2]);
            let utu2: Vec<f64> = utu2_0.iter().zip(&utu2_1).map(|(a, b)| a + delta * b).collect();
            (u1, zwu, utu, utu2)
        } else {
            (z0_vec.clone(), zwu1.clone(), scalar_pre[2], utu2_1.clone())
        };
        let rf = solve_restricted_rf(input.pi_hat, &zz_inv_glob, &vec![0.0; k], input.beta, &zwu, utu, &utu2, m, l);
        let mut u2 = input.u2_hat.to_vec();
        if q == 1 && !pz_u0.is_empty() {
            for i in 0..n {
                let pzi = pz_u0[i] + delta * pz_u1[i];
                for r in 0..m {
                    u2[i * m + r] += pzi * rf.gamma_u[r];
                }
            }
        } else {
            for i in 0..n {
                let zr = &input.z[i * l..(i + 1) * l];
                let mut pzi = 0.0;
                for d in 0..l {
                    pzi += zr[d] * rf.lambda[d];
                }
                for r in 0..m {
                    u2[i * m + r] += pzi * rf.gamma_u[r];
                }
            }
        }
        (u1, u2, rf.pi_tilde)
    };

    let reps = input.spec.reps;
    let batch_size = if fe_dim > 4096 { (262_144 / (fe_dim * pool.threads().max(1))).clamp(1, 32) } else { 32 };
    let sym_sz = q * (q + 1) / 2;

    let run_streaming = |u1: &[f64], u2: &[f64], pi_tilde: &[f64]| -> Result<Vec<f64>, RfError> {
        let mut t_boot = vec![0.0; reps];
        let n_comp = 1 + m;
        let mut b0 = 0usize;
        while b0 < reps {
            ctx.check()?;
            let b_cnt = (reps - b0).min(batch_size);
            let sz_len = b_cnt * l * n_comp;
            let sfe_len = b_cnt * fe_dim * n_comp;
            let p1 = par::sum_chunks(pool, n, CHUNK, sz_len + sfe_len, |range, acc| {
                let ci = range.start / CHUNK;
                let mut rngs: Vec<_> = (0..b_cnt).map(|bi| hetero_chunk_rng(input.spec.seed, b0 + bi, ci)).collect();
                let mut v_one = [0.0; 1];
                let mut v_rep = Vec::new();
                for i in range {
                    let wi = w_rows.map_or(1.0, |w| w[i]);
                    let f_cnt = if input.is_fweight { wi.round().max(1.0) as usize } else { 1 };
                    let zr = &input.z[i * l..(i + 1) * l];
                    let ki = input.row_order[i] as usize;
                    for bi in 0..b_cnt {
                        let s1 = if f_cnt == 1 {
                            fill_from_rng(input.spec.weights, &mut rngs[bi], &mut v_one);
                            v_one[0]
                        } else {
                            v_rep.resize(f_cnt, 0.0);
                            fill_from_rng(input.spec.weights, &mut rngs[bi], &mut v_rep);
                            v_rep.iter().sum()
                        };
                        let w_s1 = if input.is_fweight { s1 } else { wi * s1 };
                        for comp in 0..n_comp {
                            let val = w_s1 * if comp == 0 { u1[i] } else { u2[i * m + comp - 1] };
                            let dst_z = &mut acc[(bi * n_comp + comp) * l..(bi * n_comp + comp + 1) * l];
                            for d in 0..l {
                                dst_z[d] += val * zr[d];
                            }
                            match &fe_proj {
                                FeProjWre::None | FeProjWre::Approx => {}
                                FeProjWre::OneWay { codes_row, .. } => {
                                    let lvl = codes_row[i] as usize;
                                    acc[sz_len + (bi * n_comp + comp) * fe_dim + lvl] += val;
                                }
                                FeProjWre::Dense { .. } => {
                                    let dst_fe = &mut acc[sz_len + (bi * n_comp + comp) * fe_dim..sz_len + (bi * n_comp + comp + 1) * fe_dim];
                                    let mut off = 0usize;
                                    for term in input.terms {
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

            // Solve 2SLS per draw in this batch.
            let mut delta_bs = vec![0.0; b_cnt * k];
            let mut eta_bs = vec![0.0; b_cnt * l];
            let mut c_bs = vec![0.0; b_cnt * q * l];
            let mut nums = vec![0.0; b_cnt * q];
            let mut gamma_fe = vec![0.0; b_cnt * n_comp * fe_dim];
            {
                let mut pi_b = vec![0.0; l * m];
                let mut j_mat = vec![0.0; l * k];
                let mut a_zz_j = vec![0.0; l * k];
                let mut h_in = vec![0.0; k * k];
                let mut rhs = vec![0.0; k];
                let mut hj = vec![0.0; k * l];
                for bi in 0..b_cnt {
                    let zw_u1 = &p1[(bi * n_comp) * l..(bi * n_comp + 1) * l];
                    pi_b.copy_from_slice(pi_tilde);
                    for d in 0..l {
                        let zrow = &zz_inv_glob[d * l..(d + 1) * l];
                        for r in 0..m {
                            let zw_r = &p1[(bi * n_comp + 1 + r) * l..(bi * n_comp + 2 + r) * l];
                            let mut s = 0.0;
                            for dd in 0..l {
                                s += zrow[dd] * zw_r[dd];
                            }
                            pi_b[d * m + r] += s;
                        }
                    }
                    wre_solve_2sls(
                        &pi_b,
                        &a_zz,
                        zw_u1,
                        &input.hyp.r_mat,
                        m,
                        k1,
                        l2,
                        q,
                        &mut j_mat,
                        &mut a_zz_j,
                        &mut h_in,
                        &mut rhs,
                        &mut delta_bs[bi * k..(bi + 1) * k],
                        &mut hj,
                        &mut c_bs[bi * q * l..(bi + 1) * q * l],
                    );
                    let db = &delta_bs[bi * k..(bi + 1) * k];
                    matvec(&input.hyp.r_mat, db, q, k, &mut nums[bi * q..(bi + 1) * q]);
                    let et = &mut eta_bs[bi * l..(bi + 1) * l];
                    for d in 0..l {
                        let mut s = 0.0;
                        for r in 0..m {
                            s += pi_tilde[d * m + r] * db[r];
                        }
                        if d >= l2 {
                            s += db[m + (d - l2)];
                        }
                        et[d] = s;
                    }
                    for comp in 0..n_comp {
                        let sfe = &p1[sz_len + (bi * n_comp + comp) * fe_dim..sz_len + (bi * n_comp + comp + 1) * fe_dim];
                        let gdst = &mut gamma_fe[(bi * n_comp + comp) * fe_dim..(bi * n_comp + comp + 1) * fe_dim];
                        match &fe_proj {
                            FeProjWre::None | FeProjWre::Approx => {}
                            FeProjWre::OneWay { inv_sum_w, .. } => {
                                for idx in 0..fe_dim {
                                    gdst[idx] = sfe[idx] * inv_sum_w[idx];
                                }
                            }
                            FeProjWre::Dense { g_pinv, m_fe } => {
                                matvec(g_pinv, sfe, *m_fe, *m_fe, gdst);
                            }
                        }
                    }
                }
            }

            // Pass 2: accumulate quadratic score meat in R^q.
            let p2 = par::sum_chunks(pool, n, CHUNK, b_cnt * sym_sz, |range, acc| {
                let ci = range.start / CHUNK;
                let mut rngs: Vec<_> = (0..b_cnt).map(|bi| hetero_chunk_rng(input.spec.seed, b0 + bi, ci)).collect();
                let mut v_one = [0.0; 1];
                let mut v_rep = Vec::new();
                let mut fe_p = vec![0.0; n_comp];
                let mut qrow = vec![0.0; q];
                for i in range {
                    let wi = w_rows.map_or(1.0, |w| w[i]);
                    let f_cnt = if input.is_fweight { wi.round().max(1.0) as usize } else { 1 };
                    let zr = &input.z[i * l..(i + 1) * l];
                    let ki = input.row_order[i] as usize;
                    for bi in 0..b_cnt {
                        let (s1, s2) = if f_cnt == 1 {
                            fill_from_rng(input.spec.weights, &mut rngs[bi], &mut v_one);
                            (v_one[0], v_one[0] * v_one[0])
                        } else {
                            v_rep.resize(f_cnt, 0.0);
                            fill_from_rng(input.spec.weights, &mut rngs[bi], &mut v_rep);
                            let mut a1 = 0.0;
                            let mut a2 = 0.0;
                            for &vv in &v_rep {
                                a1 += vv;
                                a2 += vv * vv;
                            }
                            (a1, a2)
                        };
                        fe_p.fill(0.0);
                        match &fe_proj {
                            FeProjWre::None | FeProjWre::Approx => {}
                            FeProjWre::OneWay { codes_row, .. } => {
                                let lvl = codes_row[i] as usize;
                                for comp in 0..n_comp {
                                    fe_p[comp] = gamma_fe[(bi * n_comp + comp) * fe_dim + lvl];
                                }
                            }
                            FeProjWre::Dense { .. } => {
                                for comp in 0..n_comp {
                                    let gd = &gamma_fe[(bi * n_comp + comp) * fe_dim..(bi * n_comp + comp + 1) * fe_dim];
                                    let mut p = 0.0;
                                    let mut off = 0usize;
                                    for term in input.terms {
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
                                    fe_p[comp] = p;
                                }
                            }
                        }
                        let db = &delta_bs[bi * k..(bi + 1) * k];
                        let et = &eta_bs[bi * l..(bi + 1) * l];
                        let mut z_eta = 0.0;
                        for d in 0..l {
                            z_eta += zr[d] * et[d];
                        }
                        // Unperturbed linear residual shift c_const = z_eta + fe_p[0] - sum_r db[r]*fe_p[1+r]
                        let mut p_fe_u2 = 0.0;
                        let mut u2_d2 = 0.0;
                        for r in 0..m {
                            p_fe_u2 += db[r] * fe_p[1 + r];
                            u2_d2 += db[r] * u2[i * m + r];
                        }
                        let a_rand = u1[i] - u2_d2;
                        let p_det = z_eta + fe_p[0] - p_fe_u2;
                        let e_sq_sum = if f_cnt == 1 {
                            let s = wi * (a_rand * s1 - p_det);
                            s * s
                        } else {
                            a_rand * a_rand * s2 - 2.0 * a_rand * p_det * s1 + wi * p_det * p_det
                        };
                        matvec(&c_bs[bi * q * l..(bi + 1) * q * l], zr, q, l, &mut qrow);
                        let mdst = &mut acc[bi * sym_sz..(bi + 1) * sym_sz];
                        let mut pos = 0usize;
                        for a in 0..q {
                            let va = input.ssc_hetero * e_sq_sum * qrow[a];
                            for b in a..q {
                                mdst[pos] += va * qrow[b];
                                pos += 1;
                            }
                        }
                    }
                }
            });
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
                t_boot[b0 + bi] = eval_stat_and_se(&nums[bi * q..(bi + 1) * q], &v_buf, q).0;
            }
            b0 += b_cnt;
        }
        Ok(t_boot)
    };

    let delta0 = if input.spec.impose_null && q == 1 { Some(input.hyp.r_vec[0] - point.expect("q==1")) } else { None };
    let (u1, u2, pi_t) = build_u1_u2(delta0);
    let t_boot = run_streaming(&u1, &u2, &pi_t)?;
    let (p_val, reps_feasible) = compute_pvalue(stat, &t_boot, q, input.spec.pval_type);

    let conf_int = if q == 1 && input.spec.conf_int {
        let alpha = 1.0 - input.spec.level;
        let pt = point.expect("q==1");
        if input.spec.impose_null {
            let se_val = se.unwrap_or(f64::NAN);
            let ci = invert_ci_search(pt, se_val, alpha, input.spec.pval_type, input.spec.tol, input.spec.maxiter, reps_feasible, |r_trial| {
                let delta = r_trial - pt;
                let st_trial = (pt - r_trial) / se_val;
                let (tu1, tu2, tpi) = build_u1_u2(Some(delta));
                let tb = run_streaming(&tu1, &tu2, &tpi)?;
                Ok(compute_pvalue(st_trial, &tb, 1, input.spec.pval_type).0)
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
        n_clusters: None,
        q,
        full_enumeration: false,
        method: BootMethod::Wre,
        boot_type: input.spec.boot_type,
        weights: input.spec.weights,
        impose_null: input.spec.impose_null,
        pval_type: input.spec.pval_type,
        ssc: input.ssc_hetero,
        notes,
    })
}
