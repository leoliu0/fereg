//! Variance matrices with fixest's small-sample corrections (fixest `vcov.fixest`,
//! `ssc_compute_K`, `vcov_{iid,hetero,cluster}_internal`; pyfixest `get_ssc`), shared by every
//! model of the fit layer and by [`super::FitState::vcov`].

use super::design::{eval, sparse};
use super::{Flavor, GDf, KFixef, Ssc, TDf, VcovKind, VcovOutput};
use crate::dof::nested_within;
use crate::error::RfError;
use crate::par::Pool;
use crate::vcov::{self, ClusterCodes};

/// A fitted model as the variance matrix sees it: the bread, and per row (solver row order) the
/// design row a_i = (design combos)·row_i and the score multiplier s_i = (resid combo)·row_i·w_i,
/// so that the score is s_i·a_i (OLS and IV: residual × weight × residualized regressors).
pub(crate) struct Model<'a> {
    pub flavor: Flavor,
    /// fixest `n`: estimation rows (pyfixest frequency weights: Σw).
    pub n: f64,
    pub nparams: usize,
    /// k×k bread (X̃'WX̃)⁻¹ over the design columns, row-major.
    pub bread: &'a [f64],
    /// iid V = bread·`iid_scale`·adj (OLS: ssr/(n − 1); GLM: 1).
    pub iid_scale: f64,
    /// Normal (z) inference: `df_t` is `None`.
    pub normal: bool,
    /// Row block (row order, `r` columns) the combos apply to.
    pub rows: &'a [f64],
    pub r: usize,
    pub design: &'a [Vec<f64>],
    pub resid: &'a [f64],
    /// Weights in row order.
    pub w: Option<&'a [f64]>,
    /// pyfixest frequency weights: hetero meat terms are s_i²/w_i.
    pub fweights: bool,
    /// Kept index of each row (cluster codes are in kept order).
    pub row_order: &'a [u32],
}

/// One fixed-effect term over the estimation rows (kept order).
pub(crate) struct FeInfo<'a> {
    pub codes: &'a [u32],
    pub g: usize,
    pub intercept: bool,
    /// Data-column names of its factors (`a^b`: both).
    pub names: Vec<&'a str>,
}

/// One cluster dimension over the estimation rows (kept order).
pub(crate) struct ClusterInfo<'a> {
    pub codes: &'a [u32],
    pub g: usize,
    pub names: Vec<&'a str>,
}

/// fixest `df.K` for this variance matrix (`ssc_compute_K`); pyfixest's `df_k` differs only in
/// the nesting test (by value against every cluster dimension) and the absence of the clamp.
pub(crate) fn df_k(m: &Model, kind: VcovKind, ssc: &Ssc, fe: &[FeInfo], clusters: &[ClusterInfo], refs: &dyn Fn() -> Vec<usize>) -> f64 {
    let np = m.nparams as f64;
    let sizes_ok: Vec<f64> = fe.iter().map(|t| if t.intercept { t.g as f64 } else { 0.0 }).collect();
    let n_fe = fe.len();
    let n_fe_ok = sizes_ok.iter().filter(|g| **g > 0.0).count();
    let fe_params = if n_fe_ok > 0 { sizes_ok.iter().sum::<f64>() - (n_fe_ok as f64 - 1.0) } else { 0.0 };
    let cluster = kind == VcovKind::Cluster;
    // Only cluster variance matrices allow "nonnested"; the others use "full".
    let k_fixef = if !cluster && ssc.k_fixef == KFixef::Nonnested { KFixef::Full } else { ssc.k_fixef };
    match k_fixef {
        KFixef::None => np - fe_params,
        KFixef::Full => {
            if ssc.k_exact && n_fe >= 2 && n_fe_ok >= 1 {
                np + (n_fe_ok as f64 - 1.0) - refs().iter().sum::<usize>() as f64
            } else {
                np
            }
        }
        KFixef::Nonnested if n_fe_ok == 0 => np,
        KFixef::Nonnested => {
            let nested = nested_terms(m.flavor, fe, clusters);
            let n_nested = nested.iter().filter(|x| **x).count();
            let mut k = np;
            if n_nested == n_fe {
                k = k - fe_params + 1.0;
            } else if ssc.k_exact && n_fe >= 2 {
                let mut nb_ref = refs();
                if n_nested > 1 {
                    for (q, r) in nb_ref.iter_mut().enumerate() {
                        if nested[q] {
                            *r = fe[q].g;
                        }
                    }
                }
                k -= nb_ref.iter().sum::<usize>() as f64;
            } else {
                let s: f64 = sizes_ok.iter().zip(&nested).filter(|(_, n)| **n).map(|(g, _)| *g).sum();
                let c = sizes_ok.iter().zip(&nested).filter(|(g, n)| **n && **g > 0.0).count() as f64;
                k -= s - c;
            }
            match m.flavor {
                Flavor::Fixest => k.max(m.design.len() as f64 + 1.0),
                Flavor::Pyfixest => k,
            }
        }
    }
}

/// FE terms nested in a cluster dimension. fixest: by name first (the term is a cluster
/// variable, or an `a^b` term contains one), then by value (every level within one cluster) for
/// the rest, unless every cluster variable is itself a fixed-effect term. pyfixest: by value.
fn nested_terms(flavor: Flavor, fe: &[FeInfo], clusters: &[ClusterInfo]) -> Vec<bool> {
    let by_value = |t: &FeInfo| clusters.iter().any(|c| nested_within(t.codes, t.g, c.codes));
    match flavor {
        Flavor::Pyfixest => fe.iter().map(by_value).collect(),
        Flavor::Fixest => {
            let cl_names: Vec<String> = clusters.iter().map(|c| c.names.join("^")).collect();
            let fe_names: Vec<String> = fe.iter().map(|t| t.names.join("^")).collect();
            let mut nested: Vec<bool> = fe
                .iter()
                .zip(&fe_names)
                .map(|(t, name)| cl_names.contains(name) || (t.names.len() > 1 && t.names.iter().any(|f| cl_names.iter().any(|c| c == f))))
                .collect();
            if nested.iter().any(|x| !x) && !cl_names.iter().all(|c| fe_names.contains(c)) {
                for (t, x) in fe.iter().zip(nested.iter_mut()) {
                    if !*x {
                        *x = by_value(t);
                    }
                }
            }
            nested
        }
    }
}

/// The variance matrix of `kind` under `ssc` (fixest `vcov.fixest`). Returns it with the notes.
pub(crate) fn compute(
    m: &Model,
    kind: VcovKind,
    ssc: &Ssc,
    fe: &[FeInfo],
    clusters: &[ClusterInfo],
    refs: &dyn Fn() -> Vec<usize>,
    pool: &Pool,
) -> Result<(VcovOutput, Vec<String>), RfError> {
    if kind == VcovKind::Cluster && clusters.is_empty() {
        return Err(RfError::InvalidInput("a clustered variance matrix needs cluster dimensions".into()));
    }
    let k = m.design.len();
    let n = m.n;
    let df_k = df_k(m, kind, ssc, fe, clusters, refs);
    let n_clusters: Vec<usize> = if kind == VcovKind::Cluster { clusters.iter().map(|c| c.g).collect() } else { Vec::new() };
    let g_min = n_clusters.iter().copied().min().unwrap_or(0) as f64;
    let design: Vec<Vec<(usize, f64)>> = m.design.iter().map(|d| sparse(d)).collect();
    let resid = sparse(m.resid);
    let r = m.r;
    let score = |i: usize, a: &mut [f64]| {
        let row = &m.rows[i * r..(i + 1) * r];
        for (ap, d) in a.iter_mut().zip(&design) {
            *ap = eval(row, d);
        }
        eval(row, &resid) * m.w.map_or(1.0, |w| w[i])
    };
    let mut factor = 1.0;
    let mut v = if k == 0 {
        Vec::new()
    } else {
        match kind {
            VcovKind::Iid => {
                if ssc.k_adj {
                    factor = (n - 1.0) / (n - df_k);
                }
                m.bread.iter().map(|b| b * m.iid_scale * factor).collect()
            }
            VcovKind::Hetero => {
                if ssc.k_adj {
                    factor = n / (n - df_k);
                }
                let w_div = if m.fweights { m.w } else { None };
                let meat = vcov::robust_meat(m.rows.len() / r, k, pool, |i, a| {
                    let s = score(i, a);
                    w_div.map_or(s * s, |w| s * s / w[i])
                });
                vcov::sandwich(m.bread, &meat, k, factor)
            }
            VcovKind::Cluster => {
                let dims: Vec<ClusterCodes> = clusters.iter().map(|c| ClusterCodes { codes: c.codes, g: c.g }).collect();
                let conventional = ssc.g_adj && ssc.g_df == GDf::Conventional;
                let scale = |g: usize| if conventional { g as f64 / (g as f64 - 1.0) } else { 1.0 };
                let meat = vcov::multiway_meat_scaled(&dims, m.row_order, k, pool, scale, score)?;
                if ssc.g_adj && ssc.g_df == GDf::Min {
                    factor *= g_min / (g_min - 1.0);
                }
                if ssc.k_adj {
                    factor *= (n - 1.0) / (n - df_k);
                }
                vcov::sandwich(m.bread, &meat, k, factor)
            }
        }
    };
    let mut notes = Vec::new();
    let mut psd_fixed = false;
    if m.flavor == Flavor::Fixest && k > 0 && v.iter().all(|x| x.is_finite()) {
        psd_fixed = posdef_fix(&mut v, k);
        if psd_fixed {
            notes.push("The VCOV matrix is not positive definite and was 'fixed'.".to_string());
        }
    }
    let df_t = if m.normal {
        None
    } else {
        Some(match (ssc.t_df, m.flavor) {
            (TDf::Fixed(t), _) => t,
            (TDf::Min, Flavor::Fixest) if kind == VcovKind::Cluster => (g_min - 1.0).max(1.0),
            (TDf::Min, Flavor::Pyfixest) if kind == VcovKind::Cluster => g_min - 1.0,
            (_, Flavor::Fixest) => (n - df_k).max(1.0),
            (_, Flavor::Pyfixest) => n - df_k,
        })
    };
    Ok((VcovOutput { kind, vcov: v, df_k, df_t, n_clusters, ssc_factor: factor, psd_fixed }, notes))
}

/// fixest `mat_posdef_fix`: when an eigenvalue is ≤ 0, rebuilds the matrix with the eigenvalues
/// clipped at 1e-16. Returns whether it changed the matrix by more than 1e-8 (fixest's note).
pub(crate) fn posdef_fix(v: &mut [f64], k: usize) -> bool {
    let (lambda, u) = vcov::sym_eigen(v, k);
    if lambda.iter().all(|&l| l > 0.0) {
        return false;
    }
    let mut fixed = vec![0.0; k * k];
    for (e, &l) in lambda.iter().enumerate() {
        let l = l.max(1e-16);
        for i in 0..k {
            let ui = u[i * k + e] * l;
            for j in 0..k {
                fixed[i * k + j] += ui * u[j * k + e];
            }
        }
    }
    let changed = v.iter().zip(&fixed).any(|(a, b)| (a - b).abs() > 1e-8);
    v.copy_from_slice(&fixed);
    changed
}
