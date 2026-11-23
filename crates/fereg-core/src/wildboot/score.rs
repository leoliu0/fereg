//! Wild score bootstrap (Kline and Santos 2012; Roodman et al. 2019) via profiled-Hessian /
//! one-step Newton equations for GLM (Poisson PPML, Logit), IV/2SLS, and OLS.

use rayon::prelude::*;

use super::ols::{ConstraintProj, HeteroInput, eval_affine_t, eval_stat_and_se, linear_num_obs, matvec, run_hetero};
use super::weights::{cluster_matrix, resolve_reps};
use super::{
    BootMethod, BootType, Cells, ClusterConfig, Hypothesis, NonNestedPlan, StatType, WildBootResult, WildBootSpec, accumulate_dnn_groups,
    classify_non_nested, compute_pvalue, invert_ci_search, invert_ci_wcu, nullspace_reparam, path2_project_matrices, sym_pinv, to_kept_u32,
};
use crate::absorber::{Absorber, SolveOpts};
use crate::api::{Ctx, Family, GlmCollinear};
use crate::error::RfError;
use crate::glm::family;
use crate::glm::irls::{self, IrlsCfg, START_INNER_TOL};
use crate::glm::{GlmData, map_exact};
use crate::par::{self, CHUNK, Pool};
use crate::session::Term;
use crate::stats;

/// GLM wild score bootstrap input (all observation vectors in solver row order).
pub(crate) struct GlmScoreInput<'a> {
    pub family: Family,
    pub n: usize,
    pub k: usize,
    /// Original response in solver row order (not scaled by `sd_y`).
    pub y_raw: &'a [f64],
    /// Raw kept regressors (`n * k`, row-major, original scale, solver row order).
    pub x_raw: &'a [f64],
    /// Residualized kept regressors under unrestricted IRLS weights (`n * k`, original scale).
    pub x_tilde_u: &'a [f64],
    /// Unrestricted coefficient estimates (`k`, original scale).
    pub beta_u: &'a [f64],
    /// Unrestricted IRLS working residuals `(y_raw - mu_raw_u) / irls_weight_u` (`n`).
    pub e_work_u: &'a [f64],
    /// Unrestricted IRLS weights `tw * irls_weight_u` (`n`, original scale).
    pub w_irls_u: &'a [f64],
    /// Unrestricted bread `(X_tilde_u' W_irls_u X_tilde_u)^{-1}` (`k * k`, original scale).
    pub bread_u: &'a [f64],
    /// Unrestricted `mu` in original scale (`n`), used as warm start for `impose_null` fits.
    pub mu_u: &'a [f64],
    /// Raw link offset in solver row order, if any (`n`).
    pub offset: Option<&'a [f64]>,
    /// True user weights in solver row order (`n`).
    pub tw: Option<&'a [f64]>,
    pub terms: &'a [Term],
    pub row_order: &'a [u32],
    pub cluster: Option<&'a ClusterConfig<'a>>,
    pub ssc_hetero: f64,
    pub glm_tol: f64,
    pub glm_maxiter: usize,
    pub fixef_tol: f64,
    pub fixef_maxiter: usize,
    pub hyp: &'a Hypothesis,
    pub spec: &'a WildBootSpec,
}

/// Linear wild score bootstrap for OLS.
pub(crate) fn linear_score_ols(input: super::ols::OlsBootInput<'_>, absorber: Option<&mut dyn Absorber>, ctx: &Ctx, pool: &Pool) -> Result<WildBootResult, RfError> {
    linear_score_common(
        LinearScoreCfg {
            n: input.n,
            k: input.k,
            x_tilde: input.x,
            resid_u: input.resid,
            beta: input.beta,
            bread: input.bread,
            w_rows: input.w_rows,
            is_fweight: input.is_fweight,
            terms: input.terms,
            row_order: input.row_order,
            ssc_hetero: input.ssc_hetero,
            cluster: input.cluster,
            fixef_tol: input.fixef_tol,
            fixef_maxiter: input.fixef_maxiter,
            hyp: input.hyp,
            spec: input.spec,
        },
        absorber,
        ctx,
        pool,
    )
}

/// Linear wild score bootstrap for IV/2SLS (using projected instruments `x_tilde = Z * zz_inv * Z' W X`).
pub(crate) fn linear_score_iv(input: super::wre::IvBootInput<'_>, absorber: Option<&mut dyn Absorber>, ctx: &Ctx, pool: &Pool) -> Result<WildBootResult, RfError> {
    let (n, k, l) = (input.n, input.m + input.k1, input.l2 + input.k1);
    // Construct Z_hat = Z * (zz_inv * (Z' W X)). Actually bmat' (L x k) isn't Z_hat coefs;
    // compute X' W Z (k x L) then P_hat = zz_inv * (Z' W X) (`L x k`), then Z_hat = Z * P_hat (`n x k`).
    let w_rows = input.w_rows;
    let zwx = par::sum_chunks(pool, n, CHUNK, l * k, |range, acc| {
        for i in range {
            let wi = w_rows.map_or(1.0, |w| w[i]);
            let zr = &input.z[i * l..(i + 1) * l];
            let xr = &input.x[i * k..(i + 1) * k];
            for d in 0..l {
                let wz = wi * zr[d];
                let dst = &mut acc[d * k..(d + 1) * k];
                for c in 0..k {
                    dst[c] += wz * xr[c];
                }
            }
        }
    });
    let mut p_hat = vec![0.0; l * k];
    for d in 0..l {
        for c in 0..k {
            let mut s = 0.0;
            for j in 0..l {
                s += input.zz_inv[d * l + j] * zwx[j * k + c];
            }
            p_hat[d * k + c] = s;
        }
    }
    let mut z_hat = par::alloc_vec(n * k, 0.0)?;
    pool.install(|| {
        z_hat.par_chunks_mut(CHUNK * k).zip(input.z.par_chunks(CHUNK * l)).for_each(|(xc, zc)| {
            for (xrow, zrow) in xc.chunks_exact_mut(k).zip(zc.chunks_exact(l)) {
                for c in 0..k {
                    let mut s = 0.0;
                    for d in 0..l {
                        s += zrow[d] * p_hat[d * k + c];
                    }
                    xrow[c] = s;
                }
            }
        });
    });

    linear_score_common(
        LinearScoreCfg {
            n,
            k,
            x_tilde: &z_hat,
            resid_u: input.resid,
            beta: input.beta,
            bread: input.bread,
            w_rows: input.w_rows,
            is_fweight: input.is_fweight,
            terms: input.terms,
            row_order: input.row_order,
            ssc_hetero: input.ssc_hetero,
            cluster: input.cluster,
            fixef_tol: input.fixef_tol,
            fixef_maxiter: input.fixef_maxiter,
            hyp: input.hyp,
            spec: input.spec,
        },
        absorber,
        ctx,
        pool,
    )
}

struct LinearScoreCfg<'a> {
    n: usize,
    k: usize,
    x_tilde: &'a [f64],
    resid_u: &'a [f64],
    beta: &'a [f64],
    bread: &'a [f64],
    w_rows: Option<&'a [f64]>,
    is_fweight: bool,
    terms: &'a [Term],
    row_order: &'a [u32],
    ssc_hetero: f64,
    cluster: Option<&'a ClusterConfig<'a>>,
    fixef_tol: f64,
    fixef_maxiter: usize,
    hyp: &'a Hypothesis,
    pub spec: &'a WildBootSpec,
}

fn linear_score_common(cfg: LinearScoreCfg<'_>, absorber: Option<&mut dyn Absorber>, ctx: &Ctx, pool: &Pool) -> Result<WildBootResult, RfError> {
    cfg.spec.validate(cfg.hyp.q)?;
    let (n, k, q) = (cfg.n, cfg.k, cfg.hyp.q);
    if cfg.spec.boot_type == BootType::T21 {
        return Err(RfError::InvalidInput("wildboot: bootstrap type 21 is only defined for heteroskedastic OLS".into()));
    }
    let proj = ConstraintProj::new(cfg.bread, &cfg.hyp.r_mat, q, k);
    let (num_wald, point) = linear_num_obs(cfg.beta, cfg.hyp);

    // Unclustered score: delegates to heteroskedastic kernel.
    if cfg.cluster.is_none() {
        let mut ur;
        let (resid_in, obs_res, num_obs, allow_affine) = if cfg.spec.impose_null {
            let mut neg_num = vec![0.0; q];
            for a in 0..q {
                neg_num[a] = -num_wald[a];
            }
            let mut step = vec![0.0; k];
            matvec(&proj.m_r, &neg_num, k, q, &mut step);
            ur = vec![0.0; n];
            for i in 0..n {
                let xr = &cfg.x_tilde[i * k..(i + 1) * k];
                let mut adj = 0.0;
                for l in 0..k {
                    adj += xr[l] * step[l];
                }
                ur[i] = cfg.resid_u[i] - adj;
            }
            // Note at v = 1, linear score step delta_beta^(1) = -step = M_R * (R*beta - r), so e_hat^(1) = resid_u!
            (&ur[..], Some(cfg.resid_u), num_wald.clone(), true)
        } else {
            (cfg.resid_u, None, num_wald.clone(), false)
        };
        let (dgp_pow, denom_pow) = match cfg.spec.boot_type {
            BootType::T11 => (0.0, 0.0),
            BootType::T13 => (0.0, 1.0),
            BootType::T31 => (1.0, 0.0),
            BootType::T33 => (1.0, 1.0),
            BootType::T21 => (0.5, 0.0),
        };
        let mut res = run_hetero(
            HeteroInput {
                n,
                k,
                x: cfg.x_tilde,
                resid: resid_in,
                obs_resid: obs_res,
                bread: cfg.bread,
                w_rows: cfg.w_rows,
                is_fweight: cfg.is_fweight,
                terms: cfg.terms,
                row_order: cfg.row_order,
                ssc_factor: cfg.ssc_hetero,
                num_obs,
                point,
                allow_impose_null_affine: allow_affine,
                dgp_pow,
                denom_pow,
                hyp: cfg.hyp,
                spec: cfg.spec,
                stat_type_scalar: if cfg.spec.impose_null { StatType::ScoreT } else { StatType::T },
                stat_type_joint: if cfg.spec.impose_null { StatType::ScoreF } else { StatType::F },
                method: BootMethod::Score,
                notes: Vec::new(),
            },
            ctx,
            pool,
        )?;
        res.stat_type = if q == 1 {
            if cfg.spec.impose_null { StatType::ScoreT } else { StatType::T }
        } else if cfg.spec.impose_null {
            StatType::ScoreF
        } else {
            StatType::F
        };
        return Ok(res);
    }

    let cl = cfg.cluster.expect("checked");
    if cl.g_boot < 2 {
        return Err(RfError::InvalidInput("wildboot: cluster bootstrap requires at least 2 clusters".into()));
    }
    let (reps, full_enum) = resolve_reps(cfg.spec.weights, cfg.spec.reps, Some(cl.g_boot));
    let v_mat = cluster_matrix(cfg.spec.weights, cl.g_boot, reps, full_enum, cfg.spec.seed, pool)?;

    let affine_ci = q == 1 && cfg.spec.impose_null && cfg.spec.conf_int;
    let mut e1_buf = Vec::new();
    let e0_buf: Vec<f64>;
    if affine_ci {
        let mut u1 = vec![0.0; n];
        for i in 0..n {
            let xr = &cfg.x_tilde[i * k..(i + 1) * k];
            let mut z0 = 0.0;
            for l in 0..k {
                z0 += xr[l] * proj.m_r[l];
            }
            u1[i] = -z0;
        }
        e0_buf = cfg.resid_u.to_vec();
        e1_buf = u1;
    } else if !cfg.spec.impose_null {
        e0_buf = cfg.resid_u.to_vec();
    } else {
        let mut neg_num = vec![0.0; q];
        for a in 0..q {
            neg_num[a] = -num_wald[a];
        }
        let mut step = vec![0.0; k];
        matvec(&proj.m_r, &neg_num, k, q, &mut step);
        let mut ur = vec![0.0; n];
        for i in 0..n {
            let xr = &cfg.x_tilde[i * k..(i + 1) * k];
            let mut adj = 0.0;
            for l in 0..k {
                adj += xr[l] * step[l];
            }
            ur[i] = cfg.resid_u[i] - adj;
        }
        e0_buf = ur;
    }

    let targets: Vec<&[f64]> = if affine_ci { vec![&e0_buf, &e1_buf] } else { vec![&e0_buf] };
    let mut core = clustered_score_kernel(
        ClusteredScoreKernelCfg {
            n,
            k,
            x_tilde: cfg.x_tilde,
            bread: cfg.bread,
            w_rows: cfg.w_rows,
            terms: cfg.terms,
            row_order: cfg.row_order,
            cluster: cl,
            hyp: cfg.hyp,
            spec: cfg.spec,
            targets: &targets,
            e_obs_override: cfg.spec.impose_null.then_some(cfg.resid_u),
            num_obs_override: Some(&num_wald),
            point_for_affine: point,
            reps,
            v_mat: &v_mat,
            fixef_tol: cfg.fixef_tol,
            fixef_maxiter: cfg.fixef_maxiter,
        },
        absorber,
        ctx,
        pool,
    )?;

    let conf_int = if q == 1 && cfg.spec.conf_int {
        let alpha = 1.0 - cfg.spec.level;
        let pt = point.expect("q==1");
        if cfg.spec.impose_null {
            let se_val = core.se.unwrap_or(f64::NAN);
            let mut trial_tb = vec![0.0; reps];
            let ci = invert_ci_search(pt, se_val, alpha, cfg.spec.pval_type, cfg.spec.tol, cfg.spec.maxiter, core.reps_feasible, |r_trial| {
                let delta = r_trial - pt;
                let st_trial = (pt - r_trial) / se_val;
                for (dst, c) in trial_tb.iter_mut().zip(&core.affine_coefs) {
                    *dst = eval_affine_t(delta, c[0], c[1], c[2], c[3], c[4]);
                }
                Ok(compute_pvalue(st_trial, &trial_tb, 1, cfg.spec.pval_type).0)
            })?;
            Some(ci)
        } else {
            core.se.map(|se_val| invert_ci_wcu(pt, se_val, &core.t_boot, cfg.spec.pval_type, alpha))
        }
    } else {
        None
    };

    Ok(WildBootResult {
        stat: core.stat,
        p_val: core.p_val,
        stat_type: if q == 1 { StatType::ScoreT } else { StatType::ScoreF },
        point_estimate: point,
        se: core.se,
        conf_int,
        t_boot: std::mem::take(&mut core.t_boot),
        reps,
        reps_feasible: core.reps_feasible,
        nobs: n,
        n_clusters: Some(cl.g_boot),
        q,
        full_enumeration: full_enum,
        method: BootMethod::Score,
        boot_type: cfg.spec.boot_type,
        weights: cfg.spec.weights,
        impose_null: cfg.spec.impose_null,
        pval_type: cfg.spec.pval_type,
        ssc: core.ssc,
        notes: core.notes,
    })
}

/// Runs wild score bootstrap for GLM (Poisson PPML or Logit).
pub(crate) fn glm_score(input: GlmScoreInput<'_>, mut absorber: Option<&mut dyn Absorber>, ctx: &Ctx, pool: &Pool) -> Result<WildBootResult, RfError> {
    input.spec.validate(input.hyp.q)?;
    if input.hyp.k != input.k {
        return Err(RfError::InvalidInput("wildboot: hypothesis column count does not match kept regressors".into()));
    }
    match input.spec.method {
        BootMethod::Auto | BootMethod::Score => {}
        BootMethod::Hetero if input.cluster.is_none() => {}
        _ => {
            return Err(RfError::InvalidInput(
                "wildboot: GLM models (Poisson/Logit) only support method(auto) or method(score)".into(),
            ));
        }
    }
    if input.spec.boot_type == BootType::T21 {
        return Err(RfError::InvalidInput("wildboot: bootstrap type 21 is only defined for heteroskedastic OLS".into()));
    }
    if let Some(cl) = input.cluster
        && cl.g_boot < 2
    {
        return Err(RfError::InvalidInput("wildboot: cluster bootstrap requires at least 2 clusters".into()));
    }

    let (num_wald, point) = linear_num_obs(input.beta_u, input.hyp);
    let q = input.hyp.q;
    let n = input.n;

    // Unrestricted score bootstrap (`impose_null == false`): single pass on unrestricted quantities!
    if !input.spec.impose_null {
        let mut res = if let Some(cl) = input.cluster {
            let (reps, full_enum) = resolve_reps(input.spec.weights, input.spec.reps, Some(cl.g_boot));
            let v_mat = cluster_matrix(input.spec.weights, cl.g_boot, reps, full_enum, input.spec.seed, pool)?;
            let targets = [input.e_work_u];
            let core = clustered_score_kernel(
                ClusteredScoreKernelCfg {
                    n,
                    k: input.k,
                    x_tilde: input.x_tilde_u,
                    bread: input.bread_u,
                    w_rows: Some(input.w_irls_u),
                    terms: input.terms,
                    row_order: input.row_order,
                    cluster: cl,
                    hyp: input.hyp,
                    spec: input.spec,
                    targets: &targets,
                    e_obs_override: Some(input.e_work_u),
                    num_obs_override: Some(&num_wald),
                    point_for_affine: None,
                    reps,
                    v_mat: &v_mat,
                    fixef_tol: input.fixef_tol,
                    fixef_maxiter: input.fixef_maxiter,
                },
                absorber,
                ctx,
                pool,
            )?;
            let conf_int = if q == 1 && input.spec.conf_int {
                core.se.map(|se| invert_ci_wcu(point.expect("q==1"), se, &core.t_boot, input.spec.pval_type, 1.0 - input.spec.level))
            } else {
                None
            };
            WildBootResult {
                stat: core.stat,
                p_val: core.p_val,
                stat_type: if q == 1 { StatType::ScoreT } else { StatType::ScoreF },
                point_estimate: point,
                se: core.se,
                conf_int,
                t_boot: core.t_boot,
                reps,
                reps_feasible: core.reps_feasible,
                nobs: n,
                n_clusters: Some(cl.g_boot),
                q,
                full_enumeration: full_enum,
                method: BootMethod::Score,
                boot_type: input.spec.boot_type,
                weights: input.spec.weights,
                impose_null: false,
                pval_type: input.spec.pval_type,
                ssc: core.ssc,
                notes: core.notes,
            }
        } else {
            let (dgp_pow, denom_pow) = match input.spec.boot_type {
                BootType::T11 => (0.0, 0.0),
                BootType::T13 => (0.0, 1.0),
                BootType::T31 => (1.0, 0.0),
                BootType::T33 => (1.0, 1.0),
                BootType::T21 => (0.5, 0.0),
            };
            run_hetero(
                HeteroInput {
                    n,
                    k: input.k,
                    x: input.x_tilde_u,
                    resid: input.e_work_u,
                    obs_resid: None,
                    bread: input.bread_u,
                    w_rows: Some(input.w_irls_u),
                    is_fweight: false,
                    terms: input.terms,
                    row_order: input.row_order,
                    ssc_factor: input.ssc_hetero,
                    num_obs: num_wald,
                    point,
                    allow_impose_null_affine: false,
                    dgp_pow,
                    denom_pow,
                    hyp: input.hyp,
                    spec: input.spec,
                    stat_type_scalar: StatType::ScoreT,
                    stat_type_joint: StatType::ScoreF,
                    method: BootMethod::Score,
                    notes: Vec::new(),
                },
                ctx,
                pool,
            )?
        };
        res.stat_type = if q == 1 { StatType::ScoreT } else { StatType::ScoreF };
        return Ok(res);
    }

    // Unrestricted SE estimate for CI search initial step scale.
    let se_unrestricted = if q == 1 && input.spec.conf_int {
        let proj = ConstraintProj::new(input.bread_u, &input.hyp.r_mat, 1, input.k);
        if let Some(cl) = input.cluster {
            let cells = Cells::build(cl)?;
            let mut q_cell = vec![0.0; cells.n_cells];
            for i in 0..n {
                let xr = &input.x_tilde_u[i * input.k..(i + 1) * input.k];
                let mut z = 0.0;
                for l in 0..input.k {
                    z += xr[l] * proj.c_mat[l];
                }
                q_cell[cells.cell_of_row[i] as usize] += input.w_irls_u[i] * z * input.e_work_u[i];
            }
            let mut v1 = [0.0; 1];
            cells.variance_from_cell_q(&q_cell, 1, cl.ssc_factor, cl.per_subset_g_adj, &mut v1);
            Some(v1[0].sqrt()).filter(|s| s.is_finite() && *s > 0.0)
        } else {
            let mut v = 0.0;
            for i in 0..n {
                let xr = &input.x_tilde_u[i * input.k..(i + 1) * input.k];
                let mut z = 0.0;
                for l in 0..input.k {
                    z += xr[l] * proj.c_mat[l];
                }
                let s = input.w_irls_u[i] * z * input.e_work_u[i];
                v += input.ssc_hetero * s * s;
            }
            Some(v.sqrt()).filter(|s| s.is_finite() && *s > 0.0)
        }
    } else {
        None
    };

    // Pre-generate clustered weight matrix once (reused identically for every CI search evaluation).
    let (reps, full_enum, v_mat_opt) = if let Some(cl) = input.cluster {
        let (r, fe) = resolve_reps(input.spec.weights, input.spec.reps, Some(cl.g_boot));
        let vm = cluster_matrix(input.spec.weights, cl.g_boot, r, fe, input.spec.seed, pool)?;
        (r, fe, Some(vm))
    } else {
        (input.spec.reps, false, None)
    };

    let mut warm_mu = input.mu_u.to_vec();
    let mut out = eval_glm_restricted(&input, &input.hyp.r_vec, absorber.as_deref_mut(), &mut warm_mu, v_mat_opt.as_deref(), reps, full_enum, point, ctx, pool)?;
    if q == 1 && input.spec.conf_int {
        let pt = point.expect("q==1");
        let se_scale = se_unrestricted.or(out.se).unwrap_or(1.0);
        let alpha = 1.0 - input.spec.level;
        // Start warm mu near `pt` by resetting to unrestricted mu.
        let mut ci_warm_lo = input.mu_u.to_vec();
        let mut ci_warm_hi = input.mu_u.to_vec();
        let ci = invert_ci_search(pt, se_scale, alpha, input.spec.pval_type, input.spec.tol, input.spec.maxiter, out.reps_feasible, |r_trial| {
            let wref = if r_trial < pt { &mut ci_warm_lo } else { &mut ci_warm_hi };
            match eval_glm_restricted(&input, &[r_trial], absorber.as_deref_mut(), wref, v_mat_opt.as_deref(), reps, full_enum, point, ctx, pool) {
                Ok(res) => Ok(res.p_val),
                Err(_) => Ok(f64::NAN),
            }
        })?;
        out.conf_int = Some(ci);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn eval_glm_restricted<'b, 'obj>(
    input: &GlmScoreInput<'_>,
    r_vec: &[f64],
    mut absorber: Option<&'b mut (dyn Absorber + 'obj)>,
    warm: &mut [f64],
    v_mat_opt: Option<&[f64]>,
    reps: usize,
    full_enum: bool,
    point: Option<f64>,
    ctx: &Ctx,
    pool: &Pool,
) -> Result<WildBootResult, RfError> {
    let (n, q) = (input.n, input.hyp.q);
    let fit = restricted_glm_fit(input, r_vec, absorber.as_deref_mut(), warm, ctx, pool)?;
    warm.copy_from_slice(&fit.mu);
    if let Some(cl) = input.cluster {
        let v_mat = v_mat_opt.expect("cluster weights");
        let targets = [fit.e_work.as_slice()];
        let core = clustered_score_kernel(
            ClusteredScoreKernelCfg {
                n,
                k: input.k,
                x_tilde: &fit.x_tilde,
                bread: &fit.bread,
                w_rows: Some(&fit.w_irls),
                terms: input.terms,
                row_order: input.row_order,
                cluster: cl,
                hyp: input.hyp,
                spec: input.spec,
                targets: &targets,
                e_obs_override: None,
                num_obs_override: None,
                point_for_affine: None,
                reps,
                v_mat,
                fixef_tol: input.fixef_tol,
                fixef_maxiter: input.fixef_maxiter,
            },
            absorber,
            ctx,
            pool,
        )?;
        Ok(WildBootResult {
            stat: core.stat,
            p_val: core.p_val,
            stat_type: if q == 1 { StatType::ScoreT } else { StatType::ScoreF },
            point_estimate: point,
            se: core.se,
            conf_int: None,
            t_boot: core.t_boot,
            reps,
            reps_feasible: core.reps_feasible,
            nobs: n,
            n_clusters: Some(cl.g_boot),
            q,
            full_enumeration: full_enum,
            method: BootMethod::Score,
            boot_type: input.spec.boot_type,
            weights: input.spec.weights,
            impose_null: true,
            pval_type: input.spec.pval_type,
            ssc: core.ssc,
            notes: core.notes,
        })
    } else {
        // Unclustered restricted GLM score: num_obs = R * bread * X_tilde' * W * e_work,
        // obs_resid = e_work - X_tilde * delta_beta^(1).
        let mut sx = vec![0.0; input.k];
        for i in 0..n {
            let wi = fit.w_irls[i] * fit.e_work[i];
            let xr = &fit.x_tilde[i * input.k..(i + 1) * input.k];
            for l in 0..input.k {
                sx[l] += wi * xr[l];
            }
        }
        let mut db1 = vec![0.0; input.k];
        matvec(&fit.bread, &sx, input.k, input.k, &mut db1);
        let mut num1 = vec![0.0; q];
        matvec(&input.hyp.r_mat, &db1, q, input.k, &mut num1);
        let mut e_obs1 = vec![0.0; n];
        for i in 0..n {
            let xr = &fit.x_tilde[i * input.k..(i + 1) * input.k];
            let mut f = 0.0;
            for l in 0..input.k {
                f += xr[l] * db1[l];
            }
            e_obs1[i] = fit.e_work[i] - f;
        }
        let (dgp_pow, denom_pow) = match input.spec.boot_type {
            BootType::T11 => (0.0, 0.0),
            BootType::T13 => (0.0, 1.0),
            BootType::T31 => (1.0, 0.0),
            BootType::T33 => (1.0, 1.0),
            BootType::T21 => (0.5, 0.0),
        };
        run_hetero(
            HeteroInput {
                n,
                k: input.k,
                x: &fit.x_tilde,
                resid: &fit.e_work,
                obs_resid: Some(&e_obs1),
                bread: &fit.bread,
                w_rows: Some(&fit.w_irls),
                is_fweight: false,
                terms: input.terms,
                row_order: input.row_order,
                ssc_factor: input.ssc_hetero,
                num_obs: num1,
                point,
                allow_impose_null_affine: false,
                dgp_pow,
                denom_pow,
                hyp: input.hyp,
                spec: input.spec,
                stat_type_scalar: StatType::ScoreT,
                stat_type_joint: StatType::ScoreF,
                method: BootMethod::Score,
                notes: Vec::new(),
            },
            ctx,
            pool,
        )
    }
}

struct RestrictedGlmOut {
    x_tilde: Vec<f64>,
    bread: Vec<f64>,
    w_irls: Vec<f64>,
    e_work: Vec<f64>,
    mu: Vec<f64>,
}

/// Fits GLM subject to `R * beta = r_vec` via nullspace reparameterization `beta = beta0 + N * theta`.
fn restricted_glm_fit<'b, 'obj>(
    input: &GlmScoreInput<'_>,
    r_vec: &[f64],
    absorber: Option<&'b mut (dyn Absorber + 'obj)>,
    warm_mu: &[f64],
    ctx: &Ctx,
    pool: &Pool,
) -> Result<RestrictedGlmOut, RfError> {
    let (n, k, q) = (input.n, input.k, input.hyp.q);
    let m = k - q;
    let (beta0, n_basis) = nullspace_reparam(&input.hyp.r_mat, r_vec, q, k)?;

    // New offset = off_raw + X_raw * beta0.
    let mut off_new = vec![0.0; n];
    for i in 0..n {
        let xr = &input.x_raw[i * k..(i + 1) * k];
        let mut o = input.offset.map_or(0.0, |v| v[i]);
        for j in 0..k {
            o += xr[j] * beta0[j];
        }
        off_new[i] = o;
    }

    let mut mu = warm_mu.to_vec();
    let mut eta = vec![0.0; n];
    for (e, &mi) in eta.iter_mut().zip(&mu) {
        *e = family::link(input.family, mi);
    }

    let mut w_irls = vec![0.0; n];
    if m == 0 && input.terms.is_empty() {
        // Zero free parameters and no absorbed FEs: eta = off_new is exact.
        for i in 0..n {
            let e = off_new[i];
            eta[i] = e;
            let mi = family::mean(input.family, e);
            mu[i] = mi;
            w_irls[i] = input.tw.map_or(1.0, |t| t[i]) * family::irls_weight(input.family, mi);
        }
    } else {
        let mut x_new = par::alloc_vec(n * m, 0.0)?;
        if m > 0 {
            pool.install(|| {
                x_new.par_chunks_mut(CHUNK * m).zip(input.x_raw.par_chunks(CHUNK * k)).for_each(|(xc, rawc)| {
                    for (dst, xr) in xc.chunks_exact_mut(m).zip(rawc.chunks_exact(k)) {
                        for c in 0..m {
                            let mut s = 0.0;
                            for j in 0..k {
                                s += xr[j] * n_basis[j * m + c];
                            }
                            dst[c] = s;
                        }
                    }
                });
            });
        }
        let mut sd_x = vec![1.0; m];
        if m > 0 {
            let ss = par::sum_chunks(pool, n, CHUNK, m, |range, acc| {
                for i in range {
                    let wi = input.tw.map_or(1.0, |t| t[i]);
                    let xr = &x_new[i * m..(i + 1) * m];
                    for c in 0..m {
                        acc[c] += wi * xr[c] * xr[c];
                    }
                }
            });
            let sum_w: f64 = input.tw.map_or(n as f64, |t| t.iter().sum());
            for c in 0..m {
                let s = (ss[c] / sum_w.max(1.0)).sqrt();
                sd_x[c] = if s > 0.0 { s } else { 1.0 };
            }
            for row in x_new.chunks_exact_mut(m) {
                for c in 0..m {
                    row[c] /= sd_x[c];
                }
            }
        }
        let d = GlmData {
            family: input.family,
            n,
            y: input.y_raw.to_vec(),
            k: m,
            x: x_new,
            cols: (0..m).collect(),
            sd_x,
            off: Some(off_new),
            tw: input.tw.map(<[f64]>::to_vec),
        };
        let highest = 1e-12_f64.max((0.1 * input.glm_tol).min(input.fixef_tol));
        let cfg = IrlsCfg {
            tol: input.glm_tol,
            maxiter: input.glm_maxiter.max(250),
            highest,
            start_tol: START_INNER_TOL.max(input.glm_tol),
            inner_maxiter: input.fixef_maxiter,
            exact: map_exact(input.terms),
            log: false,
            sd_y: 1.0,
            check_sep: false,
            fixest: false,
            collinear: GlmCollinear::Ppmlhdfe,
            partial_out: !input.terms.is_empty(),
            ml_ltol: None,
            ml_ll_const: 0.0,
            init_mean: false,
        };
        let abs = absorber.ok_or_else(|| RfError::Internal("wildboot GLM restricted fit requires absorber when FEs exist".into()))?;
        let out = irls::irls(abs, &d, &mut mu, &mut eta, &cfg, pool, ctx)?;
        w_irls = out.w;
        // Re-residualize raw full k columns under final restricted IRLS weights w_irls.
        let mut x_tilde = input.x_raw.to_vec();
        if !input.terms.is_empty() {
            abs.set_weights(Some(&w_irls), pool, ctx)?;
            let opts = SolveOpts { tol: input.fixef_tol.min(1e-10), maxiter: input.fixef_maxiter, want_coefs: false };
            abs.residualize(&mut x_tilde, k, &opts, None, pool, ctx)?;
        }
        let (gram, _) = stats::gram(&x_tilde, k, Some(&w_irls), pool);
        let bread = sym_pinv(&gram, k, 1e-12);
        let mut e_work = vec![0.0; n];
        for i in 0..n {
            let iw = family::irls_weight(input.family, mu[i]);
            e_work[i] = if iw > 1e-30 { (input.y_raw[i] - mu[i]) / iw } else { 0.0 };
        }
        return Ok(RestrictedGlmOut { x_tilde, bread, w_irls, e_work, mu });
    }

    // m == 0 && terms.is_empty(): no FEs to residualize.
    let x_tilde = input.x_raw.to_vec();
    let (gram, _) = stats::gram(&x_tilde, k, Some(&w_irls), pool);
    let bread = sym_pinv(&gram, k, 1e-12);
    let mut e_work = vec![0.0; n];
    for i in 0..n {
        let iw = family::irls_weight(input.family, mu[i]);
        e_work[i] = if iw > 1e-30 { (input.y_raw[i] - mu[i]) / iw } else { 0.0 };
    }
    Ok(RestrictedGlmOut { x_tilde, bread, w_irls, e_work, mu })
}

struct ClusteredScoreKernelCfg<'a> {
    n: usize,
    k: usize,
    x_tilde: &'a [f64],
    bread: &'a [f64],
    w_rows: Option<&'a [f64]>,
    terms: &'a [Term],
    row_order: &'a [u32],
    cluster: &'a ClusterConfig<'a>,
    hyp: &'a Hypothesis,
    spec: &'a WildBootSpec,
    targets: &'a [&'a [f64]],
    e_obs_override: Option<&'a [f64]>,
    num_obs_override: Option<&'a [f64]>,
    point_for_affine: Option<f64>,
    reps: usize,
    v_mat: &'a [f64],
    fixef_tol: f64,
    fixef_maxiter: usize,
}

struct ClusteredScoreOut {
    stat: f64,
    p_val: f64,
    se: Option<f64>,
    t_boot: Vec<f64>,
    reps_feasible: usize,
    ssc: f64,
    notes: Vec<String>,
    affine_coefs: Vec<[f64; 5]>,
}

/// Core clustered score bootstrap kernel (profiled Hessian + non-nested FE projection + MNW 13/31/33).
fn clustered_score_kernel<'b, 'obj>(
    cfg: ClusteredScoreKernelCfg<'_>,
    mut absorber: Option<&'b mut (dyn Absorber + 'obj)>,
    ctx: &Ctx,
    pool: &Pool,
) -> Result<ClusteredScoreOut, RfError> {
    if cfg.spec.boot_type != BootType::T11 {
        return clustered_score_mnw(cfg, ctx, pool);
    }
    let (n, k, q) = (cfg.n, cfg.k, cfg.hyp.q);
    let cl = cfg.cluster;
    let g_boot = cl.g_boot;
    let cells = Cells::build(cl)?;
    let cell_kept = if cl.err_equals_boot { None } else { Some(to_kept_u32(&cells.cell_of_row, cfg.row_order)) };
    let mut notes = Vec::new();
    let plan = classify_non_nested(
        cfg.terms,
        cfg.row_order,
        cl.boot_kept,
        cell_kept.as_deref(),
        g_boot,
        cfg.w_rows,
        false,
        cfg.fixef_tol,
        cfg.fixef_maxiter,
        pool,
        ctx,
    )?;

    let proj = ConstraintProj::new(cfg.bread, &cfg.hyp.r_mat, q, k);
    let n_tgt = cfg.targets.len();
    let affine_ci = n_tgt == 2 && q == 1 && cfg.point_for_affine.is_some();

    let fill_z = |i: usize, z: &mut [f64]| {
        let xr = &cfg.x_tilde[i * k..(i + 1) * k];
        for a in 0..q {
            let mut s = 0.0;
            for l in 0..k {
                s += xr[l] * proj.c_mat[l * q + a];
            }
            z[a] = s;
        }
    };

    let obs_src = cfg.e_obs_override.unwrap_or(cfg.targets[0]);
    let have_obs_override = cfg.e_obs_override.is_some();

    let len_obs = if have_obs_override { cells.n_cells * q } else { 0 };
    let len_acell = cells.n_cells * n_tgt * q;
    let len_bcell = cells.n_cells * q * k;
    let len_sx = g_boot * n_tgt * k;
    let off_acell = len_obs;
    let off_bcell = off_acell + len_acell;
    let off_sx = off_bcell + len_bcell;
    let tot = off_sx + len_sx;

    let packed = par::sum_chunks(pool, n, CHUNK, tot, |range, acc| {
        let mut z = vec![0.0; q];
        for i in range {
            let wi = cfg.w_rows.map_or(1.0, |w| w[i]);
            let cell = cells.cell_of_row[i] as usize;
            let gb = cl.boot_codes[i] as usize;
            fill_z(i, &mut z);
            let xr = &cfg.x_tilde[i * k..(i + 1) * k];

            if have_obs_override {
                let q_dst = &mut acc[cell * q..(cell + 1) * q];
                let e_obs = obs_src[i];
                for a in 0..q {
                    q_dst[a] += wi * z[a] * e_obs;
                }
            }
            for t in 0..n_tgt {
                let ut = cfg.targets[t][i];
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

    let a_cell = &packed[off_acell..off_bcell];
    let b_cell = &packed[off_bcell..off_sx];
    let s_x = &packed[off_sx..];

    let mut phi = vec![0.0; g_boot * n_tgt * k];
    let mut r_phi = vec![0.0; g_boot * n_tgt * q];
    for g in 0..g_boot {
        for t in 0..n_tgt {
            let svec = &s_x[(g * n_tgt + t) * k..(g * n_tgt + t + 1) * k];
            let pvec = &mut phi[(g * n_tgt + t) * k..(g * n_tgt + t + 1) * k];
            matvec(cfg.bread, svec, k, k, pvec);
            let rvec = &mut r_phi[(g * n_tgt + t) * q..(g * n_tgt + t + 1) * q];
            matvec(&cfg.hyp.r_mat, pvec, q, k, rvec);
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
            notes.push(format!(
                "warning: fixed effects non-nested in bootstrap clusters (G={gb}, FE params={m_nn} > 512); falling back to score recentering without exact FE projection"
            ));
            NnData::None
        }
        NonNestedPlan::Path1 { m_nn, nn_indices, g_nn_pinv, .. } => {
            let f_boot = accumulate_dnn_groups(
                cfg.terms,
                &nn_indices,
                m_nn,
                cfg.row_order,
                cl.boot_codes,
                g_boot,
                cfg.w_rows,
                n_tgt,
                |i, out| {
                    for t in 0..n_tgt {
                        out[t] = cfg.targets[t][i];
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
                cfg.terms,
                &nn_indices,
                m_nn,
                cfg.row_order,
                &cells.cell_of_row,
                cells.n_cells,
                cfg.w_rows,
                q,
                fill_z,
                pool,
            );
            NnData::Path1 { m_nn, psi, f_cell_z }
        }
        NonNestedPlan::Path2 => {
            let abs = absorber.as_deref_mut().ok_or_else(|| RfError::Internal("wildboot score Path 2 requires absorber".into()))?;
            if let Some(wr) = cfg.w_rows {
                abs.set_weights(Some(wr), pool, ctx)?;
            }
            let p2_mat = path2_project_matrices(
                abs,
                cl.boot_codes,
                g_boot,
                &cells.cell_of_row,
                cells.n_cells,
                cfg.w_rows,
                cfg.targets,
                q,
                fill_z,
                cfg.fixef_tol,
                cfg.fixef_maxiter,
                pool,
                ctx,
            )?;
            NnData::Path2 { p2_mat }
        }
    };

    let n_cells = cells.n_cells;
    let m_nn_buf = match &nn_data {
        NnData::Path1 { m_nn, .. } => *m_nn,
        _ => 0,
    };

    // Observed statistic and SE:
    let (stat, se) = if let Some(num_ov) = cfg.num_obs_override {
        let mut v_obs = vec![0.0; q * q];
        if have_obs_override {
            cells.variance_from_cell_q(&packed[0..off_acell], q, cl.ssc_factor, cl.per_subset_g_adj, &mut v_obs);
        } else {
            // Compute q_cell at v == 1 directly from a_cell[t=0] - b_cell * db^(1) - P^(1).
            let mut db1 = vec![0.0; k];
            let mut gam1 = vec![0.0; m_nn_buf];
            for g in 0..g_boot {
                for l in 0..k {
                    db1[l] += phi[(g * n_tgt) * k + l];
                }
                if let NnData::Path1 { m_nn, psi, .. } = &nn_data {
                    for p in 0..*m_nn {
                        gam1[p] += psi[(g * n_tgt) * *m_nn + p];
                    }
                }
            }
            let mut q_obs_buf = vec![0.0; n_cells * q];
            for c in 0..n_cells {
                let a_c = &a_cell[(c * n_tgt) * q..(c * n_tgt + 1) * q];
                let b_c = &b_cell[c * q * k..(c + 1) * q * k];
                for a in 0..q {
                    let mut fit_a = 0.0;
                    for l in 0..k {
                        fit_a += b_c[a * k + l] * db1[l];
                    }
                    let mut p_a = 0.0;
                    match &nn_data {
                        NnData::None => {}
                        NnData::Path1 { m_nn, f_cell_z, .. } => {
                            let frow = &f_cell_z[(c * q + a) * *m_nn..(c * q + a + 1) * *m_nn];
                            for p in 0..*m_nn {
                                p_a += frow[p] * gam1[p];
                            }
                        }
                        NnData::Path2 { p2_mat } => {
                            let p2_row = &p2_mat[((c * q + a) * n_tgt) * g_boot..((c * q + a) * n_tgt + 1) * g_boot];
                            p_a = p2_row.iter().sum();
                        }
                    }
                    q_obs_buf[c * q + a] = a_c[a] - fit_a - p_a;
                }
            }
            cells.variance_from_cell_q(&q_obs_buf, q, cl.ssc_factor, cl.per_subset_g_adj, &mut v_obs);
        }
        eval_stat_and_se(num_ov, &v_obs, q)
    } else {
        // Observed score statistic at v == 1 from targets[0]:
        let mut num1 = vec![0.0; q];
        let mut db1 = vec![0.0; k];
        let mut gam1 = vec![0.0; m_nn_buf];
        for g in 0..g_boot {
            for a in 0..q {
                num1[a] += r_phi[(g * n_tgt) * q + a];
            }
            for l in 0..k {
                db1[l] += phi[(g * n_tgt) * k + l];
            }
            if let NnData::Path1 { m_nn, psi, .. } = &nn_data {
                for p in 0..*m_nn {
                    gam1[p] += psi[(g * n_tgt) * *m_nn + p];
                }
            }
        }
        let mut q_obs_buf = vec![0.0; n_cells * q];
        for c in 0..n_cells {
            let a_c = &a_cell[(c * n_tgt) * q..(c * n_tgt + 1) * q];
            let b_c = &b_cell[c * q * k..(c + 1) * q * k];
            for a in 0..q {
                let mut fit_a = 0.0;
                for l in 0..k {
                    fit_a += b_c[a * k + l] * db1[l];
                }
                let mut p_a = 0.0;
                match &nn_data {
                    NnData::None => {}
                    NnData::Path1 { m_nn, f_cell_z, .. } => {
                        let frow = &f_cell_z[(c * q + a) * *m_nn..(c * q + a + 1) * *m_nn];
                        for p in 0..*m_nn {
                            p_a += frow[p] * gam1[p];
                        }
                    }
                    NnData::Path2 { p2_mat } => {
                        let p2_row = &p2_mat[((c * q + a) * n_tgt) * g_boot..((c * q + a) * n_tgt + 1) * g_boot];
                        p_a = p2_row.iter().sum();
                    }
                }
                q_obs_buf[c * q + a] = a_c[a] - fit_a - p_a;
            }
        }
        let mut v_obs = vec![0.0; q * q];
        cells.variance_from_cell_q(&q_obs_buf, q, cl.ssc_factor, cl.per_subset_g_adj, &mut v_obs);
        eval_stat_and_se(&num1, &v_obs, q)
    };

    let q_cell_len = if affine_ci { n_cells * 2 } else { n_cells * q };
    let v_len = if affine_ci { 4 } else { q * q };

    let eval_draw = |b: usize,
                     db_buf: &mut [f64],
                     gamma_buf: &mut [f64],
                     q_cell_buf: &mut [f64],
                     v_buf: &mut [f64],
                     num_buf: &mut [f64]|
     -> (f64, Option<[f64; 5]>) {
        let vb = &cfg.v_mat[b * g_boot..(b + 1) * g_boot];
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
            let vg = vb[cells.cell_boot[c] as usize];
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
            let delta = cfg.hyp.r_vec[0] - cfg.point_for_affine.expect("q==1");
            (eval_affine_t(delta, coef[0], coef[1], coef[2], coef[3], coef[4]), Some(coef))
        } else {
            cells.variance_from_cell_q(q_cell_buf, q, cl.ssc_factor, cl.per_subset_g_adj, v_buf);
            (eval_stat_and_se(num_buf, v_buf, q).0, None)
        }
    };

    let reps = cfg.reps;
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
                    let (v, c) = eval_draw(base + j, &mut db_buf, &mut gamma_buf, &mut q_cell_buf, &mut v_buf, &mut num_buf);
                    *tb = v;
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

    let (p_val, reps_feasible) = compute_pvalue(stat, &t_boot, q, cfg.spec.pval_type);
    Ok(ClusteredScoreOut { stat, p_val, se, t_boot, reps_feasible, ssc: cl.ssc_factor, notes, affine_coefs })
}

/// MNW `T13`/`T31`/`T33` support for clustered wild score bootstrap (`err_equals_boot` required).
fn clustered_score_mnw(cfg: ClusteredScoreKernelCfg<'_>, ctx: &Ctx, pool: &Pool) -> Result<ClusteredScoreOut, RfError> {
    let cl = cfg.cluster;
    if !cl.err_equals_boot {
        return Err(RfError::InvalidInput("wildboot: bootstrap types 13, 31, and 33 require matching 1-way bootcluster and error cluster".into()));
    }
    let (n, k, q) = (cfg.n, cfg.k, cfg.hyp.q);
    let g_boot = cl.g_boot;
    let mut notes = Vec::new();
    let plan = classify_non_nested(
        cfg.terms,
        cfg.row_order,
        cl.boot_kept,
        None,
        g_boot,
        cfg.w_rows,
        true,
        cfg.fixef_tol,
        cfg.fixef_maxiter,
        pool,
        ctx,
    )?;

    let (q_nn, r_nn) = match plan {
        NonNestedPlan::None => (None, 0usize),
        NonNestedPlan::Path1 { q_nn: Some((mat, r)), .. } if k + r <= 256 => (Some(mat), r),
        _ => {
            notes.push("warning: MNW score bootstrap 13/31/33 with large non-nested fixed effects ignores leave-out non-nested FE refitting".into());
            (None, 0usize)
        }
    };
    let p = k + r_nn;
    let mut r_aug = vec![0.0; q * p];
    for a in 0..q {
        r_aug[a * p..a * p + k].copy_from_slice(&cfg.hyp.r_mat[a * k..(a + 1) * k]);
    }
    let hyp_aug = Hypothesis { r_mat: r_aug, r_vec: cfg.hyp.r_vec.clone(), q, k: p };

    let n_tgt = cfg.targets.len();
    let affine_ci = n_tgt == 2 && q == 1 && cfg.point_for_affine.is_some();
    let have_obs_override = cfg.e_obs_override.is_some();
    let obs_src = cfg.e_obs_override.unwrap_or(cfg.targets[0]);
    let n_acc = n_tgt + usize::from(have_obs_override);

    let len_a = g_boot * p * p;
    let len_s = g_boot * n_acc * p;
    let q_nn_ref = q_nn.as_deref();
    let packed = par::sum_chunks(pool, n, CHUNK, len_a + len_s, |range, acc| {
        let mut xa = vec![0.0; p];
        for i in range {
            let wi = cfg.w_rows.map_or(1.0, |w| w[i]);
            let g = cl.boot_codes[i] as usize;
            xa[..k].copy_from_slice(&cfg.x_tilde[i * k..(i + 1) * k]);
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
            for t in 0..n_tgt {
                let wu = wi * cfg.targets[t][i];
                let dst = &mut acc[len_a + (g * n_acc + t) * p..len_a + (g * n_acc + t + 1) * p];
                for r0 in 0..p {
                    dst[r0] += wu * xa[r0];
                }
            }
            if have_obs_override {
                let wu = wi * obs_src[i];
                let dst = &mut acc[len_a + (g * n_acc + n_tgt) * p..len_a + (g * n_acc + n_tgt + 1) * p];
                for r0 in 0..p {
                    dst[r0] += wu * xa[r0];
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

    let dgp_is_3 = cfg.spec.boot_type.dgp_is_3();
    let denom_is_3 = cfg.spec.boot_type.denom_is_3();
    let crv1_mult = cl.ssc_factor * if cl.per_subset_g_adj { g_boot as f64 / (g_boot as f64 - 1.0) } else { 1.0 };
    let denom_mult = if denom_is_3 { (g_boot - 1) as f64 / g_boot as f64 } else { crv1_mult };

    // Total s_obs (v == 1) across clusters (can be nonzero when impose_null == true in GLM!).
    let mut s_obs_tot = vec![0.0; p];
    let obs_slot = if have_obs_override { n_tgt } else { 0 };
    for g in 0..g_boot {
        let sg = &s_raw[(g * n_acc + obs_slot) * p..(g * n_acc + obs_slot + 1) * p];
        for j in 0..p {
            s_obs_tot[j] += sg[j];
        }
    }
    let mut db_obs = vec![0.0; p];
    matvec(&g_tot_inv, &s_obs_tot, p, p, &mut db_obs);
    let mut num_obs_calc = vec![0.0; q];
    matvec(&hyp_aug.r_mat, &db_obs, q, p, &mut num_obs_calc);

    struct Pre {
        ja: Vec<f64>,
        k_dgp: Vec<f64>,
        phi: Vec<f64>,
        r_phi: Vec<f64>,
        j_obs: Vec<f64>,
    }

    let pres: Vec<Pre> = pool.install(|| {
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
                let s_obs = &s_raw[(g * n_acc + obs_slot) * p..(g * n_acc + obs_slot + 1) * p];
                let mut j_obs_part1 = vec![0.0; q];
                let mut j_obs_part2 = vec![0.0; q];
                matvec(&j_mat, s_obs, q, p, &mut j_obs_part1);
                matvec(&ja, &db_obs, q, p, &mut j_obs_part2);
                let mut j_obs = vec![0.0; q];
                for a in 0..q {
                    j_obs[a] = j_obs_part1[a] - j_obs_part2[a];
                }

                let mut g_neg_r_mat = Vec::new();
                let mut h_neg_inv = Vec::new();
                if dgp_is_3 && cfg.spec.impose_null {
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

                let mut k_dgp = vec![0.0; n_tgt * q];
                let mut phi = vec![0.0; n_tgt * p];
                let mut r_phi = vec![0.0; n_tgt * q];
                let mut d_vec = vec![0.0; p];
                let mut y1 = vec![0.0; p];
                let mut y2 = vec![0.0; q];
                let mut y3 = vec![0.0; q];
                let mut y4 = vec![0.0; p];
                let mut diff = vec![0.0; p];
                for t in 0..n_tgt {
                    let s_in = &s_raw[(g * n_acc + t) * p..(g * n_acc + t + 1) * p];
                    if !dgp_is_3 {
                        d_vec.copy_from_slice(s_in);
                    } else if !cfg.spec.impose_null {
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

                Pre { ja, k_dgp, phi, r_phi, j_obs }
            })
            .collect()
    });

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
    let num_obs = cfg.num_obs_override.unwrap_or(&num_obs_calc);
    let (stat, se) = eval_stat_and_se(num_obs, &v_obs, q);

    let reps = cfg.reps;
    let mut t_boot = vec![0.0; reps];
    let mut affine_coefs = if affine_ci { vec![[0.0; 5]; reps] } else { Vec::new() };
    pool.install(|| {
        let eval_b = |b: usize, db_buf: &mut [f64], ja_db: &mut [f64], v_buf: &mut [f64], num_buf: &mut [f64]| -> (f64, Option<[f64; 5]>) {
            let vb = &cfg.v_mat[b * g_boot..(b + 1) * g_boot];
            num_buf[..n_tgt * q].fill(0.0);
            db_buf[..n_tgt * p].fill(0.0);
            for (g, pr) in pres.iter().enumerate() {
                let vg = vb[g];
                for t in 0..n_tgt {
                    for a in 0..q {
                        num_buf[t * q + a] += vg * pr.r_phi[t * q + a];
                    }
                    for l in 0..p {
                        db_buf[t * p + l] += vg * pr.phi[t * p + l];
                    }
                }
            }
            if affine_ci {
                let (mut d00, mut d01, mut d11) = (0.0, 0.0, 0.0);
                for (g, pr) in pres.iter().enumerate() {
                    let vg = vb[g];
                    let (mut p0, mut p1) = (0.0, 0.0);
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
                let delta = cfg.hyp.r_vec[0] - cfg.point_for_affine.expect("q==1");
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
                (eval_stat_and_se(num_buf, v_buf, q).0, None)
            }
        };

        if affine_ci {
            t_boot.par_chunks_mut(CHUNK).zip(affine_coefs.par_chunks_mut(CHUNK)).enumerate().for_each(|(ci, (tc, ac))| {
                let mut db_buf = vec![0.0; n_tgt * p];
                let mut ja_db = vec![0.0; q];
                let mut v_buf = vec![0.0; q * q];
                let mut num_buf = vec![0.0; n_tgt * q];
                let base = ci * CHUNK;
                for (j, (tb, ab)) in tc.iter_mut().zip(ac.iter_mut()).enumerate() {
                    let (v, c) = eval_b(base + j, &mut db_buf, &mut ja_db, &mut v_buf, &mut num_buf);
                    *tb = v;
                    *ab = c.expect("affine");
                }
            });
        } else {
            t_boot.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, tc)| {
                let mut db_buf = vec![0.0; n_tgt * p];
                let mut ja_db = vec![0.0; q];
                let mut v_buf = vec![0.0; q * q];
                let mut num_buf = vec![0.0; n_tgt * q];
                let base = ci * CHUNK;
                for (j, tb) in tc.iter_mut().enumerate() {
                    *tb = eval_b(base + j, &mut db_buf, &mut ja_db, &mut v_buf, &mut num_buf).0;
                }
            });
        }
    });

    let (p_val, reps_feasible) = compute_pvalue(stat, &t_boot, q, cfg.spec.pval_type);
    Ok(ClusteredScoreOut { stat, p_val, se, t_boot, reps_feasible, ssc: denom_mult, notes, affine_coefs })
}
