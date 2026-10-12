//! Variance matrices with fixest's small-sample corrections (fixest `vcov.fixest`,
//! `ssc_compute_K`, `vcov_{iid,hetero,cluster}_internal`; pyfixest `get_ssc`), shared by every
//! model of the fit layer and by [`super::FitState::vcov`].

use super::{Flavor, GDf, KFixef, Ssc, TDf, VcovKind, VcovOutput};
use crate::dof::nested_within;
use crate::error::RfError;
use crate::par::Pool;
use crate::vcov::{self, ClusterCodes};

/// The model-level inputs of a variance matrix.
#[derive(Clone, Debug)]
pub(crate) struct Base {
    pub flavor: Flavor,
    /// fixest `n`: estimation rows (pyfixest frequency weights: Σw).
    pub n: f64,
    /// Kept coefficients, intercept included.
    pub k: usize,
    pub nparams: usize,
    /// k×k bread (X̃'WX̃)⁻¹, row-major.
    pub bread: Vec<f64>,
    /// iid V = bread·`iid_scale`·adj (OLS: ssr/(n − 1); GLM: 1).
    pub iid_scale: f64,
    /// Normal (z) inference: `df_t` is `None`.
    pub normal: bool,
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

/// The per-row design and scores: `row(i, a)` fills the design row a_i of row i (row order) and
/// returns its score multiplier s_i (score = s_i·a_i; OLS s_i = e_i·w_i). Hetero meat terms are
/// s_i²·`hetero_weight(i)` (1, or 1/w_i for pyfixest frequency weights).
pub(crate) struct Scores<'a, F: Fn(usize, &mut [f64]) -> f64 + Sync> {
    pub n_rows: usize,
    pub row_order: &'a [u32],
    pub hetero_div: Option<&'a [f64]>,
    pub row: F,
}

/// fixest `df.K` for this variance matrix (`ssc_compute_K`); pyfixest's `df_k` differs only in
/// the nesting test (by value against every cluster dimension) and the absence of the clamp.
pub(crate) fn df_k(base: &Base, kind: VcovKind, ssc: &Ssc, fe: &[FeInfo], clusters: &[ClusterInfo], refs: &dyn Fn() -> Vec<usize>) -> f64 {
    let np = base.nparams as f64;
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
            let nested = nested_terms(base.flavor, fe, clusters);
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
            match base.flavor {
                Flavor::Fixest => k.max(base.k as f64 + 1.0),
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute<F: Fn(usize, &mut [f64]) -> f64 + Sync>(
    base: &Base,
    kind: VcovKind,
    ssc: &Ssc,
    fe: &[FeInfo],
    clusters: &[ClusterInfo],
    refs: &dyn Fn() -> Vec<usize>,
    scores: &Scores<F>,
    pool: &Pool,
) -> Result<(VcovOutput, Vec<String>), RfError> {
    if kind == VcovKind::Cluster && clusters.is_empty() {
        return Err(RfError::InvalidInput("a clustered variance matrix needs cluster dimensions".into()));
    }
    let k = base.k;
    let n = base.n;
    let df_k = df_k(base, kind, ssc, fe, clusters, refs);
    let n_clusters: Vec<usize> = if kind == VcovKind::Cluster { clusters.iter().map(|c| c.g).collect() } else { Vec::new() };
    let g_min = n_clusters.iter().copied().min().unwrap_or(0) as f64;
    let mut factor = 1.0;
    let mut v = if k == 0 {
        Vec::new()
    } else {
        match kind {
            VcovKind::Iid => {
                if ssc.k_adj {
                    factor = (n - 1.0) / (n - df_k);
                }
                base.bread.iter().map(|b| b * base.iid_scale * factor).collect()
            }
            VcovKind::Hetero => {
                if ssc.k_adj {
                    factor = n / (n - df_k);
                }
                let meat = vcov::robust_meat(scores.n_rows, k, pool, |i, a| {
                    let s = (scores.row)(i, a);
                    scores.hetero_div.map_or(s * s, |w| s * s / w[i])
                });
                vcov::sandwich(&base.bread, &meat, k, factor)
            }
            VcovKind::Cluster => {
                let dims: Vec<ClusterCodes> = clusters.iter().map(|c| ClusterCodes { codes: c.codes, g: c.g }).collect();
                let conventional = ssc.g_adj && ssc.g_df == GDf::Conventional;
                let scale = |g: usize| if conventional { g as f64 / (g as f64 - 1.0) } else { 1.0 };
                let meat = vcov::multiway_meat_scaled(&dims, scores.row_order, k, pool, scale, |i, a| (scores.row)(i, a))?;
                if ssc.g_adj && ssc.g_df == GDf::Min {
                    factor *= g_min / (g_min - 1.0);
                }
                if ssc.k_adj {
                    factor *= (n - 1.0) / (n - df_k);
                }
                vcov::sandwich(&base.bread, &meat, k, factor)
            }
        }
    };
    let mut notes = Vec::new();
    let mut psd_fixed = false;
    if base.flavor == Flavor::Fixest && k > 0 && v.iter().all(|x| x.is_finite()) {
        let (lambda, u) = vcov::sym_eigen(&v, k);
        if lambda.iter().any(|&l| l <= 0.0) {
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
            if v.iter().zip(&fixed).any(|(a, b)| (a - b).abs() > 1e-8) {
                notes.push("The VCOV matrix is not positive definite and was 'fixed'.".to_string());
            }
            v = fixed;
            psd_fixed = true;
        }
    }
    let df_t = if base.normal {
        None
    } else {
        Some(match (ssc.t_df, base.flavor) {
            (TDf::Fixed(t), _) => t,
            (TDf::Min, Flavor::Fixest) if kind == VcovKind::Cluster => (g_min - 1.0).max(1.0),
            (TDf::Min, Flavor::Pyfixest) if kind == VcovKind::Cluster => g_min - 1.0,
            (_, Flavor::Fixest) => (n - df_k).max(1.0),
            (_, Flavor::Pyfixest) => n - df_k,
        })
    };
    Ok((VcovOutput { kind, vcov: v, df_k, df_t, n_clusters, ssc_factor: factor, psd_fixed }, notes))
}
