//! Fit-layer wild bootstrap entry point (`fit::wildboot`) for R and Python front ends.

use rayon::prelude::*;

use super::design::{regress, unit};
use super::extra::kept_values;
use super::glm;
use super::input::{self, Sample};
use super::vcov::{self as fvcov, ClusterInfo, FeInfo, Model as VModel};
use super::{CoefSource, FactorCol, FitInput, FitSpec, Flavor, GDf, Model, VcovKind, WeightsType};
use crate::absorber::SolveOpts;
use crate::api::Ctx;
use crate::error::RfError;
use crate::factor::{self, Codes};
use crate::gather::Col;
use crate::par::{self, CHUNK, Pool};
use crate::session::ClusterDim;
use crate::stats;
use crate::vcov as cvcov;
use crate::wildboot::{ClusterConfig, Hypothesis, OwnedClusterSetup, WildBootResult, WildBootSpec, ols as wb_ols, score as wb_score, sym_pinv, wre as wb_wre};

/// Options for [`wildboot`].
#[derive(Clone, Debug)]
pub struct WildBootOptions {
    /// Null hypothesis `R * beta = r` over the kept coefficient vector `[intercept?, x..., endog...]`.
    pub hyp: Hypothesis,
    pub spec: WildBootSpec,
    /// Optional override for the bootstrap cluster dimension (factor indices into `input.factors`).
    /// - `None`: intersection of `input.cluster` (or unclustered when `input.cluster` is empty).
    /// - `Some(&[])`: explicitly unclustered bootstrap even when `input.cluster` is non-empty.
    pub bootcluster: Option<Vec<usize>>,
}

/// Complete wild bootstrap output returned by [`wildboot`].
#[derive(Clone, Debug)]
pub struct WildBootOutput {
    pub coef_source: Vec<CoefSource>,
    pub collinear: Vec<CoefSource>,
    pub nobs_origin: usize,
    pub result: WildBootResult,
}

fn factor_is_missing(col: &FactorCol<'_>, row: usize) -> bool {
    match col {
        FactorCol::F64(v) => !v[row].is_finite(),
        FactorCol::I32(v) => v[row] == i32::MIN,
        FactorCol::I64(v) => v[row] == i64::MIN,
        FactorCol::Codes { codes, nlevels, base } => {
            let c = codes[row];
            c < *base || (c - *base) as u32 >= *nlevels
        }
    }
}

fn factorize_col(col: &FactorCol<'_>, pool: &Pool) -> Result<Codes, RfError> {
    factor::factorize_col(col, pool)
}

/// Runs wild bootstrap inference on the model defined by `(input, spec)` and options `wb`.
pub fn wildboot(input: &FitInput, spec: &FitSpec, wb: &WildBootOptions, ctx: &Ctx) -> Result<WildBootOutput, RfError> {
    wb.spec.validate(wb.hyp.q)?;
    input::validate(input, spec)?;
    if let Some(bc) = &wb.bootcluster {
        for &f in bc {
            if f >= input.factors.len() {
                return Err(RfError::InvalidInput(format!("wildboot: bootcluster factor index {f} out of range")));
            }
        }
    }

    // Handle missing values in `wb.bootcluster` if any factor isn't covered by `input.cluster`.
    let mut drop_storage: Option<Vec<bool>> = None;
    if let Some(bc) = &wb.bootcluster {
        let mut any_extra_na = false;
        for &f in bc {
            let in_err = input.cluster.iter().any(|dim| dim.contains(&f));
            if !in_err {
                for i in 0..input.n {
                    if input.drop.is_none_or(|d| !d[i]) && factor_is_missing(&input.factors[f], i) {
                        if spec.flavor == Flavor::Pyfixest {
                            return Err(RfError::InvalidInput("missing values in the bootcluster variable".into()));
                        }
                        any_extra_na = true;
                        break;
                    }
                }
            }
        }
        if any_extra_na {
            let mut mask = input.drop.map_or_else(|| vec![false; input.n], <[bool]>::to_vec);
            for &f in bc {
                let col = &input.factors[f];
                for (i, m) in mask.iter_mut().enumerate() {
                    if !*m && factor_is_missing(col, i) {
                        *m = true;
                    }
                }
            }
            drop_storage = Some(mask);
        }
    }
    let mut input_adj = input.clone();
    if let Some(ref mask) = drop_storage {
        input_adj.drop = Some(mask.as_slice());
    }

    let mut spec_run = spec.clone();
    spec_run.keep_resid = false;
    spec_run.keep_fixef = false;
    spec_run.keep_state = false;

    par::with_pool(par::threads_for(ctx.threads, input_adj.n), |pool| {
        let sample = input::build(&input_adj, &spec_run, ctx, pool)?;
        match spec_run.model {
            Model::Ols => wildboot_linear(&input_adj, &spec_run, sample, wb, ctx, pool),
            Model::Poisson | Model::Logit => wildboot_glm(&input_adj, &spec_run, sample, wb, ctx, pool),
        }
    })?
}

/// Computes `ClusterDim` for explicit `bootcluster` factor indices over `kept_index`, and builds `OwnedClusterSetup`.
fn build_fit_clusters(
    input: &FitInput,
    err_clusters: &[ClusterDim],
    wb: &WildBootOptions,
    kept_index: &[u32],
    row_order: &[u32],
    ssc_factor: f64,
    per_subset_g_adj: bool,
    pool: &Pool,
) -> Result<Option<OwnedClusterSetup>, RfError> {
    match &wb.bootcluster {
        Some(bc) if bc.is_empty() => Ok(None),
        Some(bc) => {
            let mut codes_list = Vec::with_capacity(bc.len());
            for &f in bc {
                let c = factorize_col(&input.factors[f], pool)?;
                codes_list.push(factor::restrict(&c, kept_index, pool)?);
            }
            let comb = if codes_list.len() == 1 {
                codes_list.into_iter().next().expect("non-empty")
            } else {
                let refs: Vec<&Codes> = codes_list.iter().collect();
                factor::combine(&refs, pool)?
            };
            let wb_dim = ClusterDim { factors: Vec::new(), codes: comb.codes, g: comb.n as usize };
            OwnedClusterSetup::from_dims(err_clusters, Some(&wb_dim), row_order, ssc_factor, per_subset_g_adj, pool)
        }
        None => OwnedClusterSetup::from_dims(err_clusters, None, row_order, ssc_factor, per_subset_g_adj, pool),
    }
}

/// Computes `fit`-flavor SSC factor and `per_subset_g_adj` for wild bootstrap.
fn compute_fit_ssc(
    spec: &FitSpec,
    fe_info: &[FeInfo],
    cl_info: &[ClusterInfo],
    terms_refs_fn: &dyn Fn() -> Vec<usize>,
    design_len: usize,
    nparams: usize,
    n_eff: f64,
    is_clustered: bool,
) -> (f64, bool, f64) {
    let dummy_bread = Vec::new();
    let dummy_rows = Vec::new();
    let dummy_design: Vec<Vec<f64>> = vec![Vec::new(); design_len];
    let dummy_resid = Vec::new();
    let dummy_order = Vec::new();
    let vmodel = VModel {
        flavor: spec.flavor,
        n: n_eff,
        nparams,
        bread: &dummy_bread,
        iid_scale: 1.0,
        normal: spec.model != Model::Ols,
        rows: &dummy_rows,
        r: 1,
        design: &dummy_design,
        resid: &dummy_resid,
        w: None,
        fweights: false,
        row_order: &dummy_order,
    };
    if is_clustered {
        let df_k = fvcov::df_k(&vmodel, VcovKind::Cluster, &spec.ssc, fe_info, cl_info, terms_refs_fn);
        let mut factor = if spec.ssc.k_adj { (n_eff - 1.0) / (n_eff - df_k) } else { 1.0 };
        let g_min = cl_info.iter().map(|c| c.g).min().unwrap_or(1) as f64;
        let multi = cl_info.len() > 1;
        let per_subset = multi && spec.ssc.g_df == GDf::Conventional && spec.ssc.g_adj;
        if spec.ssc.g_adj && (!multi || spec.ssc.g_df == GDf::Min) && g_min > 1.0 {
            factor *= g_min / (g_min - 1.0);
        }
        let ssc_hetero = if spec.ssc.k_adj { n_eff / (n_eff - df_k) } else { 1.0 };
        (factor, per_subset, ssc_hetero)
    } else {
        let df_k = fvcov::df_k(&vmodel, VcovKind::Hetero, &spec.ssc, fe_info, cl_info, terms_refs_fn);
        let ssc_hetero = if spec.ssc.k_adj { n_eff / (n_eff - df_k) } else { 1.0 };
        (ssc_hetero, false, ssc_hetero)
    }
}

fn wildboot_linear(
    input: &FitInput,
    spec: &FitSpec,
    sample: Sample,
    wb: &WildBootOptions,
    ctx: &Ctx,
    pool: &Pool,
) -> Result<WildBootOutput, RfError> {
    let Sample { mut session, levels_before_separation, .. } = sample;
    let n = session.n();
    let has_fe = !input.fe.is_empty();
    let intercept = spec.intercept && !input.fe.iter().any(|t| t.intercept);
    let iv = !input.endog.is_empty();
    let m_en = input.endog.len();
    let l_in = input.inst.len();
    let nx_in = input.x.len();
    let r = 1 + m_en + usize::from(intercept) + l_in + nx_in;

    let col_endog = |j: usize| 1 + j;
    let col_one = || 1 + m_en;
    let col_inst = |j: usize| 1 + m_en + usize::from(intercept) + j;
    let col_x = |j: usize| 1 + m_en + usize::from(intercept) + l_in + j;

    let mut cols = vec![Col::Values(input.y)];
    cols.extend(input.endog.iter().map(|v| Col::Values(v)));
    if intercept {
        cols.push(Col::One);
    }
    cols.extend(input.inst.iter().chain(&input.x).map(|v| Col::Values(v)));
    let mut rows = session.gather_input(&cols, pool)?;
    ctx.check()?;

    if has_fe {
        let opts = SolveOpts { tol: spec.fixef_tol, maxiter: spec.fixef_maxiter, want_coefs: false };
        session.absorber_mut().residualize(&mut rows, r, &opts, None, pool, ctx)?;
    }

    let w_rows = session.absorber().weights();
    let (g, _) = stats::gram(&rows, r, w_rows, pool);
    let g_u = (spec.flavor == Flavor::Pyfixest && w_rows.is_some()).then(|| stats::gram(&rows, r, None, pool).0);
    let gc = g_u.as_deref();
    let tol = spec.collin_tol;
    let y_unit = unit(r, 0);

    struct RegCol {
        source: CoefSource,
        col_idx: usize,
        design: Vec<f64>,
    }

    let mut exog: Vec<RegCol> = Vec::new();
    if intercept {
        let idx = col_one();
        exog.push(RegCol { source: CoefSource::Intercept, col_idx: idx, design: unit(r, idx) });
    }
    for j in 0..nx_in {
        let idx = col_x(j);
        exog.push(RegCol { source: CoefSource::X(j), col_idx: idx, design: unit(r, idx) });
    }

    let mut second: Vec<RegCol> = Vec::new();
    let mut z_sources: Vec<CoefSource> = Vec::new();
    let mut z_cols_idx: Vec<usize> = Vec::new();
    let mut first_kept: Vec<usize> = Vec::new();

    if iv {
        let mut z_design: Vec<Vec<f64>> = Vec::new();
        if intercept {
            let idx = col_one();
            z_sources.push(CoefSource::Intercept);
            z_cols_idx.push(idx);
            z_design.push(unit(r, idx));
        }
        for j in 0..l_in {
            let idx = col_inst(j);
            z_sources.push(CoefSource::Inst(j));
            z_cols_idx.push(idx);
            z_design.push(unit(r, idx));
        }
        for j in 0..nx_in {
            let idx = col_x(j);
            z_sources.push(CoefSource::X(j));
            z_cols_idx.push(idx);
            z_design.push(unit(r, idx));
        }

        let mut firsts = Vec::with_capacity(m_en);
        for j in 0..m_en {
            let target = unit(r, col_endog(j));
            let reg = regress(&g, gc, r, &target, &z_design, tol)?;
            firsts.push(reg);
        }
        if let Some(f0) = firsts.first() {
            first_kept = f0.kept.clone();
        }
        if intercept {
            second.push(exog.remove(0));
        }
        for (j, f) in firsts.iter().enumerate() {
            let idx = col_endog(j);
            let target = unit(r, idx);
            let fitted: Vec<f64> = target.iter().zip(&f.resid).map(|(a, b)| a - b).collect();
            second.push(RegCol { source: CoefSource::Endog(j), col_idx: idx, design: fitted });
        }
        second.append(&mut exog);
    } else {
        second = exog;
    }

    let design2: Vec<Vec<f64>> = second.iter().map(|x| x.design.clone()).collect();
    let reg2 = regress(&g, gc, r, &y_unit, &design2, tol)?;
    if !design2.is_empty() && reg2.kept.is_empty() {
        return Err(RfError::InvalidInput("all the regressors are collinear (with the fixed effects)".into()));
    }

    let rank_key = |s: &CoefSource| match s {
        CoefSource::Intercept => (0, 0),
        CoefSource::X(j) => (1, *j),
        CoefSource::Endog(j) => (2, *j),
        CoefSource::Inst(j) => (3, *j),
    };
    let mut order: Vec<usize> = (0..reg2.kept.len()).collect();
    order.sort_by_key(|&a| rank_key(&second[reg2.kept[a]].source));
    let k = order.len();
    let kk = reg2.kept.len();
    let coef: Vec<f64> = order.iter().map(|&a| reg2.coef[a]).collect();
    let coef_source: Vec<CoefSource> = order.iter().map(|&a| second[reg2.kept[a]].source).collect();
    let inv2 = &reg2.inv;
    let bread: Vec<f64> = order.iter().flat_map(|&a| order.iter().map(move |&b| inv2[a * kk + b])).collect();
    let collinear: Vec<CoefSource> = second.iter().zip(&reg2.excluded).filter(|(_, x)| **x).map(|(s, _)| s.source).collect();

    if wb.hyp.k != k {
        return Err(RfError::InvalidInput(format!(
            "wildboot: hypothesis dimension k={} differs from kept regressors {k}",
            wb.hyp.k
        )));
    }

    let fweights = spec.flavor == Flavor::Pyfixest && spec.weights_type == WeightsType::Fweights && w_rows.is_some();
    let sumw: f64 = w_rows.map_or(n as f64, |w| w.iter().sum());
    let n_eff = if fweights { sumw } else { n as f64 };
    let terms = session.terms().to_vec();
    let mut before = levels_before_separation.map(Vec::into_iter);
    let df_sizes: Vec<usize> = terms.iter().map(|t| before.as_mut().filter(|_| t.intercept).and_then(Iterator::next).unwrap_or(t.g)).collect();
    let np = super::ols::nparams_with(k, &terms, &df_sizes);
    let fe_names = |ft: &super::FeTerm| -> Vec<&str> { ft.factors.iter().map(|&f| input.factor_names[f].as_str()).collect() };
    let fe_info: Vec<FeInfo> = terms
        .iter()
        .zip(&input.fe)
        .zip(&df_sizes)
        .map(|((t, ft), &g)| FeInfo { codes: &t.codes, g, intercept: t.intercept, names: fe_names(ft) })
        .collect();
    let mut cl_info: Vec<ClusterInfo> = session
        .clusters()
        .iter()
        .zip(&input.cluster)
        .map(|(c, dim)| ClusterInfo { codes: &c.codes, g: c.g, names: dim.iter().map(|&f| input.factor_names[f].as_str()).collect() })
        .collect();
    let is_clustered = match &wb.bootcluster {
        Some(bc) => !bc.is_empty(),
        None => !cl_info.is_empty(),
    };
    let extra_bc_info = if cl_info.is_empty() && wb.bootcluster.as_ref().is_some_and(|b| !b.is_empty()) {
        let bc = wb.bootcluster.as_ref().expect("some");
        let names: Vec<&str> = bc.iter().map(|&f| input.factor_names[f].as_str()).collect();
        Some((bc, names))
    } else {
        None
    };

    let row_order = session.row_order().to_vec();
    let kept_index = session.kept_index().to_vec();
    let mut dummy_err = Vec::new();
    let cl_setup_tmp_info = if extra_bc_info.is_some() {
        let bc = extra_bc_info.as_ref().expect("some").0;
        let setup = build_fit_clusters(input, session.clusters(), wb, &kept_index, &row_order, 1.0, false, pool)?;
        if let Some(ref s) = setup {
            dummy_err.push((s.boot_kept.clone(), s.g_boot));
            cl_info.push(ClusterInfo {
                codes: &dummy_err[0].0,
                g: s.g_boot,
                names: bc.iter().map(|&f| input.factor_names[f].as_str()).collect(),
            });
        }
        setup
    } else {
        None
    };
    let refs = || super::state::term_refs(&fe_info, pool);
    let (ssc_cluster, per_subset_g_adj, ssc_hetero) = compute_fit_ssc(spec, &fe_info, &cl_info, &refs, k, np, n_eff, is_clustered);

    let cl_setup = if let Some(mut s) = cl_setup_tmp_info {
        s.ssc_factor = ssc_cluster;
        s.per_subset_g_adj = per_subset_g_adj;
        Some(s)
    } else {
        build_fit_clusters(input, session.clusters(), wb, &kept_index, &row_order, ssc_cluster, per_subset_g_adj, pool)?
    };
    let cl_cfg: Option<ClusterConfig<'_>> = cl_setup.as_ref().map(OwnedClusterSetup::as_config);

    // Build residualized design columns and residuals in solver row order.
    let mut abs = session.absorber;
    let w_rows_owned = abs.weights().map(<[f64]>::to_vec);
    let w_slice = w_rows_owned.as_deref();

    if !iv {
        let x_cols: Vec<usize> = order.iter().map(|&a| second[reg2.kept[a]].col_idx).collect();
        let mut x_kept = par::alloc_vec(n * k, 0.0)?;
        let mut resid = par::alloc_vec(n, 0.0)?;
        for i in 0..n {
            let row = &rows[i * r..(i + 1) * r];
            let mut u = row[0];
            for (c, &col) in x_cols.iter().enumerate() {
                let v = row[col];
                x_kept[i * k + c] = v;
                u -= coef[c] * v;
            }
            resid[i] = u;
        }
        let result = wb_ols::run(
            wb_ols::OlsBootInput {
                n,
                k,
                x: &x_kept,
                resid: &resid,
                beta: &coef,
                bread: &bread,
                w_rows: w_slice,
                is_fweight: fweights,
                terms: &terms,
                row_order: &row_order,
                ssc_hetero,
                hyp: &wb.hyp,
                spec: &wb.spec,
                cluster: cl_cfg.as_ref(),
                fixef_tol: spec.fixef_tol,
                fixef_maxiter: spec.fixef_maxiter,
            },
            has_fe.then_some(abs.as_mut()),
            ctx,
            pool,
        )?;
        return Ok(WildBootOutput { coef_source, collinear, nobs_origin: input.n, result });
    }

    // IV/2SLS: reorder [intercept?, x..., endog...] -> [endog..., exog_and_icpt...] for WRE.
    let mut en_fit_pos = Vec::new();
    let mut ex_fit_pos = Vec::new();
    let mut en_col_idx = Vec::new();
    let mut ex_col_idx = Vec::new();
    let mut ex_sources = Vec::new();
    for (p, &src) in coef_source.iter().enumerate() {
        let a = order[p];
        let col = second[reg2.kept[a]].col_idx;
        if matches!(src, CoefSource::Endog(_)) {
            en_fit_pos.push(p);
            en_col_idx.push(col);
        } else {
            ex_fit_pos.push(p);
            ex_col_idx.push(col);
            ex_sources.push(src);
        }
    }
    let m = en_fit_pos.len();
    let k1 = ex_fit_pos.len();
    let perm: Vec<usize> = en_fit_pos.iter().chain(&ex_fit_pos).copied().collect();
    let wre_col_idx: Vec<usize> = en_col_idx.iter().chain(&ex_col_idx).copied().collect();

    // Z_wre = [non_ex_z_kept..., ex_col_idx...]
    let mut z_other_idx: Vec<usize> = Vec::new();
    for &zi in &first_kept {
        let zc = z_cols_idx[zi];
        if !ex_col_idx.contains(&zc) {
            z_other_idx.push(zc);
        }
    }
    let l2 = z_other_idx.len();
    let l = l2 + k1;
    let z_wre_idx: Vec<usize> = z_other_idx.into_iter().chain(ex_col_idx).collect();

    let mut x_wre = par::alloc_vec(n * k, 0.0)?;
    let mut z_wre = par::alloc_vec(n * l, 0.0)?;
    let mut resid = par::alloc_vec(n, 0.0)?;
    let beta_wre: Vec<f64> = perm.iter().map(|&p| coef[p]).collect();
    let bread_ref = &bread;
    let bread_wre: Vec<f64> = perm.iter().flat_map(|&pa| perm.iter().map(move |&pb| bread_ref[pa * k + pb])).collect();

    for i in 0..n {
        let row = &rows[i * r..(i + 1) * r];
        let mut u = row[0];
        for (c, &col) in wre_col_idx.iter().enumerate() {
            let v = row[col];
            x_wre[i * k + c] = v;
            u -= beta_wre[c] * v;
        }
        resid[i] = u;
        for (d, &col) in z_wre_idx.iter().enumerate() {
            z_wre[i * l + d] = row[col];
        }
    }

    // Compute A_ZZ = Z' W Z, zz_inv, Pi_hat = zz_inv * (Z' W Y_2), U2_hat = Y_2 - Z * Pi_hat
    let azz_zy = par::sum_chunks(pool, n, CHUNK, l * l + l * m, |range, acc| {
        for i in range {
            let wi = w_slice.map_or(1.0, |w| w[i]);
            let zr = &z_wre[i * l..(i + 1) * l];
            for d1 in 0..l {
                let wz = wi * zr[d1];
                for d2 in d1..l {
                    acc[d1 * l + d2] += wz * zr[d2];
                }
                for j in 0..m {
                    acc[l * l + d1 * m + j] += wz * x_wre[i * k + j];
                }
            }
        }
    });
    let mut azz = azz_zy[..l * l].to_vec();
    for d1 in 0..l {
        for d2 in 0..d1 {
            azz[d1 * l + d2] = azz[d2 * l + d1];
        }
    }
    let zz_inv = sym_pinv(&azz, l, 1e-12);
    let zwy = &azz_zy[l * l..];
    let mut pi_hat = vec![0.0; l * m];
    for d in 0..l {
        for j in 0..m {
            let mut s = 0.0;
            for dd in 0..l {
                s += zz_inv[d * l + dd] * zwy[dd * m + j];
            }
            pi_hat[d * m + j] = s;
        }
    }
    let mut u2_hat = par::alloc_vec(n * m, 0.0)?;
    for i in 0..n {
        let zr = &z_wre[i * l..(i + 1) * l];
        for j in 0..m {
            let mut fit = 0.0;
            for d in 0..l {
                fit += zr[d] * pi_hat[d * m + j];
            }
            u2_hat[i * m + j] = x_wre[i * k + j] - fit;
        }
    }
    let hyp_wre = wb.hyp.permute_cols(&perm);
    let result = wb_wre::run(
        wb_wre::IvBootInput {
            n,
            m,
            k1,
            l2,
            x: &x_wre,
            z: &z_wre,
            resid: &resid,
            beta: &beta_wre,
            bread: &bread_wre,
            zz_inv: &zz_inv,
            pi_hat: &pi_hat,
            u2_hat: &u2_hat,
            w_rows: w_slice,
            is_fweight: fweights,
            terms: &terms,
            row_order: &row_order,
            ssc_hetero,
            hyp: &hyp_wre,
            spec: &wb.spec,
            cluster: cl_cfg.as_ref(),
            fixef_tol: spec.fixef_tol,
            fixef_maxiter: spec.fixef_maxiter,
        },
        has_fe.then_some(abs.as_mut()),
        ctx,
        pool,
    )?;
    Ok(WildBootOutput { coef_source, collinear, nobs_origin: input.n, result })
}

fn wildboot_glm(
    input: &FitInput,
    spec: &FitSpec,
    sample: Sample,
    wb: &WildBootOptions,
    ctx: &Ctx,
    pool: &Pool,
) -> Result<WildBootOutput, RfError> {
    let est = glm::estimate(input, spec, sample, ctx, pool)?;
    let glm::Estimated { session, levels_before_separation, excluded, parts, family, has_fe, intercept, ones, nsent, .. } = est;
    let source = |j: usize| if ones && j == 0 { CoefSource::Intercept } else { CoefSource::X(j - usize::from(ones)) };
    let mut coef_source = Vec::new();
    let mut coef = Vec::new();
    let mut kept_sent = Vec::new();
    if intercept && !ones {
        coef_source.push(CoefSource::Intercept);
        coef.push(parts.cons);
    }
    for j in 0..nsent {
        if parts.status[j] == 0 {
            coef_source.push(source(j));
            coef.push(parts.coef[j]);
            kept_sent.push(j);
        }
    }
    let collinear: Vec<CoefSource> = (0..nsent).filter(|&j| parts.status[j] != 0).map(source).collect();
    if nsent > 0 && coef.is_empty() {
        return Err(RfError::InvalidInput("all the regressors are collinear (with the fixed effects)".into()));
    }
    let k = coef.len();
    if wb.hyp.k != k {
        return Err(RfError::InvalidInput(format!(
            "wildboot: hypothesis dimension k={} differs from kept regressors {k}",
            wb.hyp.k
        )));
    }

    let n = session.n();
    let row_order = session.row_order().to_vec();
    let kept_index = session.kept_index().to_vec();
    let terms = session.terms().to_vec();

    // Effective sample size for SSC (includes pyfixest boundary rows on logit when excluded.is_some()).
    let n_rep = n + excluded.as_ref().map_or(0, |ex| ex.rows.len());
    let w_obs_row = session.weights().map(|w| cvcov::to_row_order(w, &row_order, pool)).transpose()?;
    let sumw: f64 = match &excluded {
        Some(ex) if input.weights.is_some() => {
            let iw = input.weights.expect("some");
            kept_index.iter().chain(&ex.rows).map(|&i| iw[i as usize]).sum()
        }
        None => session.weights().map_or(n as f64, |w| w.iter().sum()),
        _ => n_rep as f64,
    };
    let fweights = spec.flavor == Flavor::Pyfixest && spec.weights_type == WeightsType::Fweights && input.weights.is_some();
    let n_eff = if fweights { sumw } else { n_rep as f64 };

    let mut before = levels_before_separation.filter(|_| family == crate::api::Family::Logit).map(Vec::into_iter);
    let df_sizes: Vec<usize> = terms.iter().map(|t| before.as_mut().filter(|_| t.intercept).and_then(Iterator::next).unwrap_or(t.g)).collect();
    let np = super::ols::nparams_with(k, &terms, &df_sizes);
    let fe_names = |ft: &super::FeTerm| -> Vec<&str> { ft.factors.iter().map(|&f| input.factor_names[f].as_str()).collect() };
    let fe_info: Vec<FeInfo> = terms
        .iter()
        .zip(&input.fe)
        .zip(&df_sizes)
        .map(|((t, ft), &g)| FeInfo { codes: &t.codes, g, intercept: t.intercept, names: fe_names(ft) })
        .collect();
    let mut cl_info: Vec<ClusterInfo> = session
        .clusters()
        .iter()
        .zip(&input.cluster)
        .map(|(c, dim)| ClusterInfo { codes: &c.codes, g: c.g, names: dim.iter().map(|&f| input.factor_names[f].as_str()).collect() })
        .collect();
    let is_clustered = match &wb.bootcluster {
        Some(bc) => !bc.is_empty(),
        None => !cl_info.is_empty(),
    };
    let mut dummy_err = Vec::new();
    let extra_setup = if cl_info.is_empty() && wb.bootcluster.as_ref().is_some_and(|b| !b.is_empty()) {
        let bc = wb.bootcluster.as_ref().expect("some");
        let s = build_fit_clusters(input, session.clusters(), wb, &kept_index, &row_order, 1.0, false, pool)?;
        if let Some(ref ss) = s {
            dummy_err.push((ss.boot_kept.clone(), ss.g_boot));
            cl_info.push(ClusterInfo {
                codes: &dummy_err[0].0,
                g: ss.g_boot,
                names: bc.iter().map(|&f| input.factor_names[f].as_str()).collect(),
            });
        }
        s
    } else {
        None
    };
    let refs = || super::state::term_refs(&fe_info, pool);
    let (ssc_cluster, per_subset_g_adj, ssc_hetero) = compute_fit_ssc(spec, &fe_info, &cl_info, &refs, k, np, n_eff, is_clustered);
    let cl_setup = if let Some(mut s) = extra_setup {
        s.ssc_factor = ssc_cluster;
        s.per_subset_g_adj = per_subset_g_adj;
        Some(s)
    } else {
        build_fit_clusters(input, session.clusters(), wb, &kept_index, &row_order, ssc_cluster, per_subset_g_adj, pool)?
    };
    let cl_cfg: Option<ClusterConfig<'_>> = cl_setup.as_ref().map(OwnedClusterSetup::as_config);

    // Gather row-ordered y_raw, mu_u, w_irls_u, e_work_u, offset, x_raw, x_tilde_u.
    let y_raw: Vec<f64> = pool.install(|| row_order.par_iter().map(|&ki| input.y[kept_index[ki as usize] as usize]).collect());
    let off_row: Option<Vec<f64>> = input.offset.map(|o| pool.install(|| row_order.par_iter().map(|&ki| o[kept_index[ki as usize] as usize]).collect()));
    let mu_u = cvcov::to_row_order(&parts.mu, &row_order, pool)?;
    let w_irls_u = cvcov::to_row_order(&parts.w, &row_order, pool)?;
    let e_work_u = cvcov::to_row_order(&parts.working_resid, &row_order, pool)?;

    let mut x_raw = par::alloc_vec(n * k, 0.0)?;
    for c in 0..k {
        match coef_source[c] {
            CoefSource::Intercept => {
                for i in 0..n {
                    x_raw[i * k + c] = 1.0;
                }
            }
            CoefSource::X(xj) => {
                let col = input.x[xj];
                for (i, &ki) in row_order.iter().enumerate() {
                    x_raw[i * k + c] = col[kept_index[ki as usize] as usize];
                }
            }
            _ => unreachable!(),
        }
    }

    let x_tilde_u = if !has_fe {
        x_raw.clone()
    } else {
        // parts.xt has `kk == k` columns in kept order.
        let mut xt_row = par::alloc_vec(n * k, 0.0)?;
        for (i, &ki) in row_order.iter().enumerate() {
            let src = &parts.xt[ki as usize * k..(ki as usize + 1) * k];
            xt_row[i * k..(i + 1) * k].copy_from_slice(src);
        }
        xt_row
    };
    let (gram, _) = stats::gram(&x_tilde_u, k, Some(&w_irls_u), pool);
    let bread_u = sym_pinv(&gram, k, 1e-12);

    let mut abs = session.absorber;
    let result = wb_score::glm_score(
        wb_score::GlmScoreInput {
            family,
            n,
            k,
            y_raw: &y_raw,
            x_raw: &x_raw,
            x_tilde_u: &x_tilde_u,
            beta_u: &coef,
            e_work_u: &e_work_u,
            w_irls_u: &w_irls_u,
            bread_u: &bread_u,
            mu_u: &mu_u,
            offset: off_row.as_deref(),
            tw: w_obs_row.as_deref(),
            terms: &terms,
            row_order: &row_order,
            cluster: cl_cfg.as_ref(),
            ssc_hetero,
            glm_tol: spec.glm_tol,
            glm_maxiter: spec.glm_maxiter,
            fixef_tol: spec.fixef_tol,
            fixef_maxiter: spec.fixef_maxiter,
            hyp: &wb.hyp,
            spec: &wb.spec,
        },
        has_fe.then_some(abs.as_mut()),
        ctx,
        pool,
    )?;
    let _ = kept_values;
    Ok(WildBootOutput { coef_source, collinear, nobs_origin: input.n, result })
}
