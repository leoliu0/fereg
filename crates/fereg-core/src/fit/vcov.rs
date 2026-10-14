//! Variance matrices with fixest's small-sample corrections (fixest `vcov.fixest`,
//! `ssc_compute_K`, `vcov_{iid,hetero,hc2_hc3,cluster,newey_west,driscoll_kraay,conley}_internal`;
//! pyfixest `get_ssc`, `Feols.vcov`), shared by every model of the fit layer and by
//! [`super::FitState::vcov`].

use rayon::prelude::*;

use super::design::{eval, sparse};
use super::{Flavor, GDf, KFixef, Ssc, TDf, VcovKind, VcovOutput};
use crate::conley::{self, ConleyOpts};
use crate::dof::nested_within;
use crate::error::RfError;
use crate::factor::{self, Codes};
use crate::hac::{self, Panel, PanelError};
use crate::par::{self, Pool};
use crate::vcov::{self, ClusterCodes};

/// A fitted model as the variance matrix sees it: the bread, and per row (solver row order) the
/// design row a_i = (design combos)*row_i and the score multiplier s_i = (resid combo)*row_i*w_i,
/// so that the score is s_i*a_i (OLS and IV: residual x weight x residualized regressors).
pub(crate) struct Model<'a> {
    pub flavor: Flavor,
    /// fixest `n`: estimation rows (pyfixest frequency weights: sum w).
    pub n: f64,
    pub nparams: usize,
    /// k x k bread (X~'WX~)^-1 over the design columns, row-major.
    pub bread: &'a [f64],
    /// iid V = bread*`iid_scale`*adj (OLS: ssr/(n - 1); GLM: 1).
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
    /// pyfixest frequency weights: hetero meat terms are s_i^2/w_i.
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
    let cluster = kind.is_clustered();
    // Only cluster variance matrices (CRV1, CRV3) allow "nonnested"; the others use "full"
    // (pyfixest's HAC ignores nesting too).
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

/// Which inputs a variance request comes with (validation of the request).
pub(crate) struct KindVars {
    pub cluster: bool,
    pub time: bool,
    pub panel: bool,
    pub coords: bool,
}

/// Validates a request against the model and its inputs, with the reference's message where it
/// has one.
pub(crate) fn check_kind(kind: VcovKind, flavor: Flavor, iv: bool, v: &KindVars) -> Result<(), RfError> {
    let bad = |m: &str| Err(RfError::InvalidInput(m.to_string()));
    match kind {
        _ if kind.is_clustered() && !v.cluster => bad("a clustered variance matrix needs cluster dimensions"),
        VcovKind::Hc2 | VcovKind::Hc3 if iv => Err(RfError::InvalidInput(match flavor {
            // fixest `vcov_hc2_hc3_internal`; pyfixest `_deparse_vcov_input`.
            Flavor::Fixest => format!("The VCOV type `{}` is not defined in IV estimations.", if kind == VcovKind::Hc2 { "hc2" } else { "hc3" }),
            Flavor::Pyfixest => "HC2 and HC3 inference types are not supported for IV regressions.".to_string(),
        })),
        VcovKind::NeweyWest { .. } | VcovKind::DriscollKraay { .. } if !v.time => {
            bad("Newey-West and Driscoll-Kraay variance matrices need a time variable")
        }
        // pyfixest `_vcov_hac`.
        VcovKind::NeweyWest { lag: None } if flavor == Flavor::Pyfixest && !v.panel => {
            bad("We have not yet implemented the default Newey-West HAC lag. Please provide a lag value via the `vcov_kwargs`.")
        }
        VcovKind::Conley { .. } if !v.coords => bad("a Conley variance matrix needs latitude and longitude"),
        VcovKind::Conley { cutoff, pixel, .. } if cutoff.is_some_and(|c| !(c.is_finite() && c >= 0.0)) || !(pixel.is_finite() && pixel >= 0.0) => {
            bad("the Conley cutoff and pixel must be non-negative numbers")
        }
        _ => Ok(()),
    }
}

/// One leave-one-cluster-out jackknife: given kept-order cluster codes (every code below `g`
/// present), sum_g (beta_(-g) - beta)(beta_(-g) - beta)' (k x k, the coefficient order of the
/// model). Called serially.
pub(crate) type Jackknife<'a> = dyn Fn(&[u32], usize) -> Result<Vec<f64>, RfError> + 'a;

/// Inputs of the HC2/HC3, HAC, Conley and CRV3 variance matrices over the estimation rows.
#[derive(Default)]
pub(crate) struct Extra<'a> {
    /// Kept-order codes of the time and panel variables with their numbers of levels (time codes
    /// ascending in time: fixest's dense rank).
    pub time: Option<(&'a [u32], usize)>,
    pub panel: Option<(&'a [u32], usize)>,
    /// Latitude and longitude, kept order.
    pub coords: Option<(&'a [f64], &'a [f64])>,
    /// HC2/HC3: leverage of the fixed-effect columns, kept order (`None` without fixed effects).
    pub fe_hat: Option<&'a [f64]>,
    pub has_fe: bool,
    /// Design index of the intercept (fixest's time-series lag rule drops its scores).
    pub intercept: Option<usize>,
    /// CRV3 (fit time only).
    pub jackknife: Option<&'a Jackknife<'a>>,
}

/// Every subset of `size` of `0..q` in lexicographic order (the order of [`vcov::multiway_meat`]).
fn subsets(q: usize, size: usize) -> Vec<Vec<usize>> {
    fn rec(start: usize, q: usize, size: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == size {
            out.push(cur.clone());
            return;
        }
        for i in start..q {
            cur.push(i);
            rec(i + 1, q, size, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    rec(0, q, size, &mut Vec::new(), &mut out);
    out
}

/// dreamerr `enum` of row numbers as fixest prints them: all of up to seven, else the first six
/// and "and N others".
fn enum_rows(rows: &[usize]) -> String {
    let shown: Vec<String> = rows.iter().take(if rows.len() > 7 { 6 } else { rows.len() }).map(usize::to_string).collect();
    match (rows.len(), shown.split_last()) {
        (_, None) => String::new(),
        (n, Some((last, head))) if n <= 7 => {
            if head.is_empty() { last.clone() } else { format!("{} and {last}", head.join(", ")) }
        }
        (n, _) => format!("{} and {} others", shown.join(", "), n - 6),
    }
}

/// Diagonal of the hat matrix of the full model per row (row order): the fixed-effect part plus
/// w_i a_i' bread a_i (FWL). fixest's `hatvalues` leaves the weight out for `feols` without fixed
/// effects (`cpp_diag_XUtX(X, cov.iid/sigma2)` on the unweighted design).
fn leverage(m: &Model, extra: &Extra, pool: &Pool) -> Result<Vec<f64>, RfError> {
    let k = m.design.len();
    let r = m.r;
    let n = m.rows.len() / r;
    let design: Vec<Vec<(usize, f64)>> = m.design.iter().map(|d| sparse(d)).collect();
    let weighted = !(m.flavor == Flavor::Fixest && !m.normal && !extra.has_fe);
    let fe = extra.fe_hat.map(|h| vcov::to_row_order(h, m.row_order, pool)).transpose()?;
    let mut out = par::alloc_vec(n, 0.0)?;
    pool.install(|| {
        out.par_chunks_mut(par::CHUNK).enumerate().for_each(|(c, o)| {
            let mut a = vec![0.0; k];
            for (j, h) in o.iter_mut().enumerate() {
                let i = c * par::CHUNK + j;
                let row = &m.rows[i * r..(i + 1) * r];
                for (ap, d) in a.iter_mut().zip(&design) {
                    *ap = eval(row, d);
                }
                let q: f64 = (0..k).map(|p| a[p] * (0..k).map(|s| m.bread[p * k + s] * a[s]).sum::<f64>()).sum();
                let w = if weighted { m.w.map_or(1.0, |w| w[i]) } else { 1.0 };
                *h = fe.as_ref().map_or(0.0, |f| f[i]) + w * q;
            }
        })
    });
    Ok(out)
}

/// fixest's message for a missing NW / DK / Conley variable on estimation rows (post-hoc).
pub(crate) fn fixest_na_message(name: &str, label: &str) -> String {
    format!(
        "The variable '{name}' used to estimate the VCOV (employed as {label}) has NA values which would lead to a sample used to compute the VCOV different from the sample used to estimate the parameters. This would lead to wrong inference.\nPossible solutions: i) ex ante prune them, ii) impute them, or iii) use the argument 'vcov' at estimation time."
    )
}

const FIXEST_NW_PANEL_DUP: &str = "In vcov.fixest, Newey-West VCOV cannot be computed: there are (unit x time) duplicates. You have to sort that out first, eg by creating new units free of duplicates or dropping duplicates. Or you can use a Driscoll-Kraay VCOV for which duplicates does not matter.";
const FIXEST_NW_TIME_DUP: &str = "In vcov.fixest, Newey-West VCOV cannot be computed: there are time duplicates. You may provide a panel identifier (if relevant) to sort that out. Or you can use a Driscoll-Kraay VCOV for which duplicates does not matter.";
const PYFIXEST_PANEL_DUP: &str = "There are duplicate time periods for the same panel id. This is not supported for HAC SEs.";
const PYFIXEST_TIME_DUP: &str = "There are duplicate time periods in the data. This is not supported for HAC SEs.";

fn panel_or(p: Result<Panel, PanelError>, dup: &str) -> Result<Panel, RfError> {
    p.map_err(|e| match e {
        PanelError::Duplicate => RfError::InvalidInput(dup.to_string()),
        PanelError::Rf(e) => e,
    })
}

/// The variance matrix of `kind` under `ssc` (fixest `vcov.fixest`, pyfixest `Feols.vcov`).
/// Returns it with the notes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute(
    m: &Model,
    kind: VcovKind,
    ssc: &Ssc,
    fe: &[FeInfo],
    clusters: &[ClusterInfo],
    extra: &Extra,
    refs: &dyn Fn() -> Vec<usize>,
    pool: &Pool,
) -> Result<(VcovOutput, Vec<String>), RfError> {
    if kind.is_clustered() && clusters.is_empty() {
        return Err(RfError::InvalidInput("a clustered variance matrix needs cluster dimensions".into()));
    }
    let k = m.design.len();
    let n = m.n;
    let fixest = m.flavor == Flavor::Fixest;
    let df_k = df_k(m, kind, ssc, fe, clusters, refs);
    let n_clusters: Vec<usize> = if kind.is_clustered() { clusters.iter().map(|c| c.g).collect() } else { Vec::new() };
    let g_min = n_clusters.iter().copied().min().unwrap_or(0) as f64;
    let design: Vec<Vec<(usize, f64)>> = m.design.iter().map(|d| sparse(d)).collect();
    let resid = sparse(m.resid);
    let r = m.r;
    let nrows = m.rows.len() / r;
    let score = |i: usize, a: &mut [f64]| {
        let row = &m.rows[i * r..(i + 1) * r];
        for (ap, d) in a.iter_mut().zip(&design) {
            *ap = eval(row, d);
        }
        eval(row, &resid) * m.w.map_or(1.0, |w| w[i])
    };
    let hac = matches!(kind, VcovKind::NeweyWest { .. } | VcovKind::DriscollKraay { .. });
    let n_periods = if hac { extra.time.map(|t| t.1) } else { None };
    let mut kind_out = kind;
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
                let meat = vcov::robust_meat(nrows, k, pool, |i, a| {
                    let s = score(i, a);
                    w_div.map_or(s * s, |w| s * s / w[i])
                });
                vcov::sandwich(m.bread, &meat, k, factor)
            }
            VcovKind::Hc2 | VcovKind::Hc3 => {
                let mut h = leverage(m, extra, pool)?;
                if fixest {
                    // fixest `vcov_hc2_hc3_internal`: no small-sample correction ("HC2/HC3 are
                    // SSCs"); an error when a leverage is 1.
                    let thr = 1.0 - f64::EPSILON.sqrt();
                    let mut bad: Vec<usize> = h.iter().zip(m.row_order).filter(|(h, _)| **h > thr).map(|(_, &kk)| kk as usize + 1).collect();
                    if !bad.is_empty() {
                        bad.sort_unstable();
                        let (s, were, is) = if bad.len() > 1 { ("s", "were", "are") } else { ("", "was", "is") };
                        // `{n ? problem_idx}` prints the first problematic row, not their count.
                        return Err(RfError::InvalidInput(format!(
                            "When calculating the diagonals of the projection matrix, {} value{s} {were} found to equal 1. This is most likely due to dummy variables/fixed-effects with only 1 observation, so that removing that observation would make that coefficient non-estimable. \nThe problematic row{s} {is}: {}.",
                            bad[0],
                            enum_rows(&bad)
                        )));
                    }
                } else if ssc.k_adj {
                    // pyfixest: the HC1 correction N/(N - df_k).
                    factor = n / (n - df_k);
                }
                // pyfixest frequency weights: leverage per unit h/w and scores s/sqrt(w).
                let w_div = if m.fweights { m.w } else { None };
                if let Some(w) = w_div {
                    h.iter_mut().zip(w).for_each(|(h, w)| *h /= w);
                }
                let hc3 = kind == VcovKind::Hc3;
                let meat = vcov::robust_meat(nrows, k, pool, |i, a| {
                    let s = score(i, a);
                    let d = 1.0 - h[i];
                    let s2 = if hc3 { s * s / (d * d) } else { s * s / d };
                    w_div.map_or(s2, |w| s2 / w[i])
                });
                vcov::sandwich(m.bread, &meat, k, factor)
            }
            VcovKind::Cluster | VcovKind::Crv3 => {
                let conventional = ssc.g_adj && ssc.g_df == GDf::Conventional;
                let scale = |g: usize| if conventional { g as f64 / (g as f64 - 1.0) } else { 1.0 };
                if ssc.g_adj && ssc.g_df == GDf::Min {
                    factor *= g_min / (g_min - 1.0);
                }
                if ssc.k_adj {
                    factor *= (n - 1.0) / (n - df_k);
                }
                if kind == VcovKind::Cluster {
                    let dims: Vec<ClusterCodes> = clusters.iter().map(|c| ClusterCodes { codes: c.codes, g: c.g }).collect();
                    let meat = vcov::multiway_meat_scaled(&dims, m.row_order, k, pool, scale, score)?;
                    vcov::sandwich(m.bread, &meat, k, factor)
                } else {
                    // pyfixest `_vcov_crv3_*`: V = sum over the subsets of the dimensions
                    // (inclusion-exclusion, as CRV1) of sign*ssc*jackknife meat, centered at beta.
                    let jk = extra.jackknife.ok_or_else(|| {
                        RfError::InvalidInput("a CRV3 variance matrix re-estimates the model: estimate it with vcov = CRV3".into())
                    })?;
                    let q = clusters.len();
                    let owned: Vec<Codes> = if q > 1 { clusters.iter().map(|c| Codes { codes: c.codes.to_vec(), n: c.g as u32 }).collect() } else { Vec::new() };
                    let mut meat = vec![0.0; k * k];
                    for size in 1..=q {
                        let sign = if size % 2 == 1 { 1.0 } else { -1.0 };
                        for subset in subsets(q, size) {
                            let (part, g) = if size == 1 {
                                let c = &clusters[subset[0]];
                                (jk(c.codes, c.g)?, c.g)
                            } else {
                                let parts: Vec<&Codes> = subset.iter().map(|&d| &owned[d]).collect();
                                let c = factor::combine(&parts, pool)?;
                                (jk(&c.codes, c.n as usize)?, c.n as usize)
                            };
                            let f = sign * scale(g);
                            meat.iter_mut().zip(&part).for_each(|(a, p)| *a += f * p);
                        }
                    }
                    meat.iter().map(|x| x * factor).collect()
                }
            }
            VcovKind::NeweyWest { lag } | VcovKind::DriscollKraay { lag } => {
                let (tc, t) = extra.time.ok_or_else(|| RfError::InvalidInput("Newey-West and Driscoll-Kraay variance matrices need a time variable".into()))?;
                let rank: Vec<i64> = vcov::to_row_order(tc, m.row_order, pool)?.into_iter().map(i64::from).collect();
                let unit = extra.panel.map(|(u, _)| vcov::to_row_order(u, m.row_order, pool)).transpose()?;
                let dk = matches!(kind, VcovKind::DriscollKraay { .. });
                if !fixest {
                    if let Some(u) = &unit {
                        // pyfixest `_get_panel_idx`: (panel, time) duplicates are an error for both
                        // NW and DK (DK sorts by (time, panel) for it).
                        panel_or(Panel::build(Some(u), &rank, pool), PYFIXEST_PANEL_DUP)?;
                    }
                }
                // fixest caps the weights at T - 1 (`lag_fixest_clip`); pyfixest's pairs end at the
                // span by themselves. The weights are those of the requested lag.
                let max_lag = |lag: usize| if fixest { hac::lag_fixest_clip(lag, t) } else { lag };
                let meat = if dk {
                    // fixest `cpp_driscoll_kraay`, pyfixest `_dk_meat_panel`: per-period score sums,
                    // lags over consecutive observed periods.
                    let lag = lag.unwrap_or_else(|| hac::lag_quarter_root(t));
                    kind_out = VcovKind::DriscollKraay { lag: Some(lag) };
                    let codes: Vec<u32> = rank.iter().map(|&x| x as u32).collect();
                    let hsum = hac::slot_sums(&codes, t, k, pool, score)?;
                    let pos: Vec<i64> = (0..t as i64).collect();
                    let c = hac::lag_cross_dense(&hsum, &pos, k, max_lag(lag));
                    hac::hac_combine(&c, &hac::bartlett_weights(lag), k)
                } else {
                    let (panel, lag) = match (&unit, fixest) {
                        // fixest `cpp_newey_west_panel`: pairs of the same unit at dense time ranks
                        // t and t - l.
                        (Some(u), true) => (panel_or(Panel::build(Some(u), &rank, pool), FIXEST_NW_PANEL_DUP)?, lag.unwrap_or_else(|| hac::lag_quarter_root(t))),
                        // pyfixest `_nw_meat_panel`: row adjacency within the unit after sorting.
                        (Some(u), false) => (
                            panel_or(Panel::build(Some(u), &rank, pool), PYFIXEST_PANEL_DUP)?.within_unit_positions(),
                            lag.unwrap_or_else(|| hac::lag_quarter_root(t)),
                        ),
                        (None, _) => {
                            let dup = if fixest { FIXEST_NW_TIME_DUP } else { PYFIXEST_TIME_DUP };
                            let panel = panel_or(Panel::build(None, &rank, pool), dup)?;
                            let lag = match lag {
                                Some(l) => l,
                                None => {
                                    // fixest: floor(sandwich::bwNeweyWest) on the scores in time
                                    // order, intercept scores dropped unless they are the only ones.
                                    let mut order = vec![0u32; nrows];
                                    rank.iter().enumerate().for_each(|(i, &t)| order[t as usize] = i as u32);
                                    let drop_col = extra.intercept.filter(|_| k > 1);
                                    hac::lag_fixest_ts(&order, k, drop_col, pool, score)?
                                }
                            };
                            (panel, lag)
                        }
                    };
                    kind_out = VcovKind::NeweyWest { lag: Some(lag) };
                    let c = hac::lag_cross(&panel, max_lag(lag), k, pool, score)?;
                    hac::hac_combine(&c, &hac::bartlett_weights(lag), k)
                };
                if ssc.g_adj {
                    factor = t as f64 / (t as f64 - 1.0);
                }
                if ssc.k_adj {
                    factor *= (n - 1.0) / (n - df_k);
                }
                vcov::sandwich(m.bread, &meat, k, factor)
            }
            VcovKind::Conley { cutoff, pixel, spherical } => {
                let (lat, lon) = extra.coords.ok_or_else(|| RfError::InvalidInput("a Conley variance matrix needs latitude and longitude".into()))?;
                let lat = vcov::to_row_order(lat, m.row_order, pool)?;
                let lon = vcov::to_row_order(lon, m.row_order, pool)?;
                let opts = ConleyOpts { cutoff_km: cutoff, pixel_km: pixel, spherical };
                let (meat, used) = conley::conley_meat(&lat, &lon, &opts, k, pool, score)?;
                kind_out = VcovKind::Conley { cutoff: Some(used), pixel, spherical };
                // fixest `vcov_conley_internal`: n/(n - K), no G adjustment.
                if ssc.k_adj {
                    factor = n / (n - df_k);
                }
                vcov::sandwich(m.bread, &meat, k, factor)
            }
        }
    };
    let mut notes = Vec::new();
    let mut psd_fixed = false;
    if fixest && k > 0 && v.iter().all(|x| x.is_finite()) {
        psd_fixed = posdef_fix(&mut v, k);
        if psd_fixed {
            notes.push("The VCOV matrix is not positive definite and was 'fixed'.".to_string());
        }
    }
    let df_t = if m.normal {
        None
    } else {
        let clustered = kind.is_clustered();
        let periods = n_periods.map(|t| t as f64);
        Some(match (ssc.t_df, m.flavor) {
            (TDf::Fixed(t), _) => t,
            (TDf::Min, Flavor::Fixest) if clustered => (g_min - 1.0).max(1.0),
            (TDf::Min, Flavor::Pyfixest) if clustered => g_min - 1.0,
            // fixest: `min_cluster_size` = T; pyfixest `get_ssc("HAC")`: G - 1 with G = T.
            (TDf::Min, Flavor::Fixest) if periods.is_some() => (periods.unwrap_or(0.0) - 1.0).max(1.0),
            (TDf::Min, Flavor::Pyfixest) if periods.is_some() => periods.unwrap_or(0.0) - 1.0,
            (_, Flavor::Fixest) => (n - df_k).max(1.0),
            (_, Flavor::Pyfixest) => n - df_k,
        })
    };
    Ok((VcovOutput { kind: kind_out, vcov: v, df_k, df_t, n_clusters, n_periods, ssc_factor: factor, psd_fixed }, notes))
}

/// fixest `mat_posdef_fix`: when an eigenvalue is <= 0, rebuilds the matrix with the eigenvalues
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
