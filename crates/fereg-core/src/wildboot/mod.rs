//! Wild bootstrap inference: fast OLS WCR/WCU (incl. MNW 2023 WCR13/31/33 and exact
//! non-nested HDFE), wild score bootstrap (GLM Poisson/logit, IV, OLS), Davidson-MacKinnon WRE for
//! IV/2SLS, and heteroskedastic wild bootstrap (HC1/HC2/HC3).

pub mod ols;
pub mod score;
pub mod weights;
pub mod wre;

use std::collections::HashMap;

use rayon::prelude::*;

use crate::absorber::{Absorber, SolveOpts};
use crate::api::Ctx;
use crate::dof::nested_within;
use crate::error::RfError;
use crate::factor::{self, Codes};
use crate::fit::dist::qnorm;
use crate::par::{self, CHUNK, Pool};
use crate::session::{ClusterDim, Term};
use crate::{EstResult, NamedMat};

pub use weights::BootWeights;

/// Two-digit wild bootstrap type (MacKinnon, Nielsen, and Webb 2023; Roodman et al. 2019).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BootType {
    /// CRV1/HC1 DGP + CRV1/HC1 test statistic (classic WCR11 / WCU11).
    #[default]
    T11,
    /// CRV1 DGP + CRV3 jackknife test statistic (WCR13 / WCU13).
    T13,
    /// CRV3/HC3 DGP + CRV1/HC1 test statistic (WCR31 / WCU31, or HC3 for unclustered OLS).
    T31,
    /// CRV3 DGP + CRV3 jackknife test statistic (WCR33 / WCU33).
    T33,
    /// HC2 DGP for unclustered heteroskedastic OLS wild bootstrap.
    T21,
}

impl BootType {
    pub fn name(self) -> &'static str {
        match self {
            BootType::T11 => "11",
            BootType::T13 => "13",
            BootType::T31 => "31",
            BootType::T33 => "33",
            BootType::T21 => "21",
        }
    }

    pub(crate) fn dgp_is_3(self) -> bool {
        matches!(self, BootType::T31 | BootType::T33)
    }

    pub(crate) fn denom_is_3(self) -> bool {
        matches!(self, BootType::T13 | BootType::T33)
    }
}

/// Wild bootstrap algorithm variant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BootMethod {
    /// Selects automatically: `WildCluster` (OLS clustered), `Hetero` (OLS unclustered),
    /// `Wre` (IV with type `11`), or `Score` (GLMs, or IV with other types).
    #[default]
    Auto,
    WildCluster,
    Score,
    Wre,
    Hetero,
}

impl BootMethod {
    pub fn name(self) -> &'static str {
        match self {
            BootMethod::Auto => "auto",
            BootMethod::WildCluster => "wildcluster",
            BootMethod::Score => "score",
            BootMethod::Wre => "wre",
            BootMethod::Hetero => "hetero",
        }
    }
}

/// P-value definition from the empirical bootstrap distribution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PvalType {
    /// Two-tailed symmetric: `P(|t*| > |t0|)` (or `P(W* > W0)` for `q > 1`).
    #[default]
    TwoTailed,
    /// Two-tailed equal-tailed: `2 * min(P(t* > t0), P(t* < t0))` (`q == 1` only).
    EqualTailed,
    /// Upper one-tailed: `P(t* > t0)` (`q == 1` only).
    Greater,
    /// Lower one-tailed: `P(t* < t0)` (`q == 1` only).
    Less,
}

impl PvalType {
    pub fn name(self) -> &'static str {
        match self {
            PvalType::TwoTailed => "two-tailed",
            PvalType::EqualTailed => "equal-tailed",
            PvalType::Greater => ">",
            PvalType::Less => "<",
        }
    }
}

/// Observed test statistic type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatType {
    T,
    F,
    ScoreT,
    ScoreF,
}

impl StatType {
    pub fn name(self) -> &'static str {
        match self {
            StatType::T => "t",
            StatType::F => "F",
            StatType::ScoreT => "score t",
            StatType::ScoreF => "score F",
        }
    }
}

/// Linear null hypothesis `H0: R * beta = r` over `k` kept parameters (`q` rows).
#[derive(Clone, Debug, PartialEq)]
pub struct Hypothesis {
    /// Row-major `q x k` matrix `R`.
    pub r_mat: Vec<f64>,
    /// Length-`q` target vector `r`.
    pub r_vec: Vec<f64>,
    pub q: usize,
    pub k: usize,
}

impl Hypothesis {
    /// Validates dimensions, finiteness, and full row rank `q` of `R`.
    pub fn new(r_mat: Vec<f64>, r_vec: Vec<f64>, q: usize, k: usize) -> Result<Self, RfError> {
        if k == 0 {
            return Err(RfError::InvalidInput("wildboot: model has no estimable parameters".into()));
        }
        if q == 0 || q > k {
            return Err(RfError::InvalidInput(format!("wildboot: hypothesis rank q={q} must be in 1..={k}")));
        }
        if r_mat.len() != q * k || r_vec.len() != q {
            return Err(RfError::InvalidInput("wildboot: R matrix or r vector dimension mismatch".into()));
        }
        if r_mat.iter().chain(&r_vec).any(|v| !v.is_finite()) {
            return Err(RfError::InvalidInput("wildboot: R and r must be finite".into()));
        }
        // Check full row rank q via SVD of R (q x k).
        let fa = faer::Mat::<f64>::from_fn(q, k, |i, j| r_mat[i * k + j]);
        let svd = fa.svd().map_err(|_| RfError::InvalidInput("wildboot: SVD of R failed".into()))?;
        let s = svd.S().column_vector();
        let smax = (0..q).map(|i| s[i]).fold(0.0f64, f64::max);
        let smin = (0..q).map(|i| s[i]).fold(f64::INFINITY, f64::min);
        if smax == 0.0 || smin <= 1e-10 * smax {
            return Err(RfError::InvalidInput("wildboot: hypothesis matrix R does not have full row rank".into()));
        }
        Ok(Hypothesis { r_mat, r_vec, q, k })
    }

    /// Returns a copy of `Hypothesis` with columns reordered according to `perm` (`new_col -> old_col`).
    pub(crate) fn permute_cols(&self, perm: &[usize]) -> Self {
        let kn = perm.len();
        let mut r_new = vec![0.0; self.q * kn];
        for row in 0..self.q {
            for (c, &old) in perm.iter().enumerate() {
                r_new[row * kn + c] = self.r_mat[row * self.k + old];
            }
        }
        Hypothesis { r_mat: r_new, r_vec: self.r_vec.clone(), q: self.q, k: kn }
    }
}

/// Configuration for a wild bootstrap run.
#[derive(Clone, Debug, PartialEq)]
pub struct WildBootSpec {
    pub reps: usize,
    pub weights: BootWeights,
    pub boot_type: BootType,
    pub method: BootMethod,
    pub impose_null: bool,
    pub pval_type: PvalType,
    pub conf_int: bool,
    pub level: f64,
    pub seed: u64,
    pub tol: f64,
    pub maxiter: usize,
}

impl Default for WildBootSpec {
    fn default() -> Self {
        WildBootSpec {
            reps: 999,
            weights: BootWeights::Rademacher,
            boot_type: BootType::T11,
            method: BootMethod::Auto,
            impose_null: true,
            pval_type: PvalType::TwoTailed,
            conf_int: true,
            level: 0.95,
            seed: 1,
            tol: 1e-6,
            maxiter: 50,
        }
    }
}

impl WildBootSpec {
    pub fn validate(&self, q: usize) -> Result<(), RfError> {
        if self.reps == 0 {
            return Err(RfError::InvalidInput("wildboot: reps must be at least 1".into()));
        }
        if !(self.level > 0.0 && self.level < 1.0) || !self.level.is_finite() {
            return Err(RfError::InvalidInput("wildboot: confidence level must be in (0, 1)".into()));
        }
        if !(self.tol > 0.0 && self.tol.is_finite()) || self.maxiter == 0 {
            return Err(RfError::InvalidInput("wildboot: tol and maxiter must be positive".into()));
        }
        if q > 1 && self.pval_type != PvalType::TwoTailed {
            return Err(RfError::InvalidInput("wildboot: joint tests (q > 1) only support two-tailed p-values".into()));
        }
        Ok(())
    }
}

/// Complete wild bootstrap inference output.
#[derive(Clone, Debug)]
pub struct WildBootResult {
    pub stat: f64,
    pub p_val: f64,
    pub stat_type: StatType,
    pub point_estimate: Option<f64>,
    pub se: Option<f64>,
    pub conf_int: Option<(f64, f64)>,
    pub t_boot: Vec<f64>,
    pub reps: usize,
    pub reps_feasible: usize,
    pub nobs: usize,
    pub n_clusters: Option<usize>,
    pub q: usize,
    pub full_enumeration: bool,
    pub method: BootMethod,
    pub boot_type: BootType,
    pub weights: BootWeights,
    pub impose_null: bool,
    pub pval_type: PvalType,
    pub ssc: f64,
    pub notes: Vec<String>,
}

/// Cluster configuration in row order.
pub(crate) struct ClusterConfig<'a> {
    /// Bootstrap cluster codes in solver row order (`0..g_boot`).
    pub boot_codes: &'a [u32],
    /// Bootstrap cluster codes in kept order (`0..g_boot`) for FE nesting checks.
    pub boot_kept: &'a [u32],
    pub g_boot: usize,
    /// Error-cluster dimensions in solver row order (`codes`, `g`), `>= 1` dimension.
    pub err_dims: Vec<(&'a [u32], usize)>,
    /// True if `err_dims.len() == 1` and its codes equal `boot_codes` identically.
    pub err_equals_boot: bool,
    /// Small-sample multiplier applied to each subset variance, or global `ssc_factor`.
    pub ssc_factor: f64,
    /// Whether `ssc_factor` is already the per-G factor (`true`: use `ssc_factor` directly;
    /// `false`: `per_subset` applies `g_S / (g_S - 1)` per subset when multiway `GDf::Conventional`).
    pub per_subset_g_adj: bool,
}

/// Filters a hypothesis `hyp` defined over `0..k_all` down to `kept_pos` (`0..k_kept`), returning
/// Stata error 111 (`Code(111, ...)`) if `R` puts nonzero weight on any dropped/collinear column.
pub(crate) fn filter_hypothesis_by_kept(hyp: &Hypothesis, kept_pos: &[usize], k_all: usize) -> Result<Hypothesis, RfError> {
    if hyp.k != k_all {
        return Err(RfError::InvalidInput(format!(
            "wildboot: hypothesis dimension k={} differs from model regressor count {k_all}",
            hyp.k
        )));
    }
    let mut is_kept = vec![false; k_all];
    for &p in kept_pos {
        is_kept[p] = true;
    }
    for col in 0..k_all {
        if !is_kept[col] {
            for row in 0..hyp.q {
                if hyp.r_mat[row * k_all + col] != 0.0 {
                    return Err(RfError::Code(111, "wildboot: cannot test absorbed or omitted collinear variable".into()));
                }
            }
        }
    }
    let kk = kept_pos.len();
    let mut r_sub = vec![0.0; hyp.q * kk];
    for row in 0..hyp.q {
        for (c, &p) in kept_pos.iter().enumerate() {
            r_sub[row * kk + c] = hyp.r_mat[row * k_all + p];
        }
    }
    Hypothesis::new(r_sub, hyp.r_vec.clone(), hyp.q, kk)
}

/// Rejects testing `_cons` when fixed effects are absorbed (`report_cons_col == Some(idx)`).
pub(crate) fn reject_absorbed_cons(hyp: &Hypothesis, cons_col: usize) -> Result<(), RfError> {
    if cons_col >= hyp.k {
        return Err(RfError::InvalidInput(format!(
            "wildboot: hypothesis column count {} is smaller than expected constant slot {}",
            hyp.k, cons_col
        )));
    }
    for row in 0..hyp.q {
        if hyp.r_mat[row * hyp.k + cons_col] != 0.0 {
            return Err(RfError::Code(198, "wildboot: cannot test _cons when fixed effects are absorbed".into()));
        }
    }
    Ok(())
}

/// Owned row-ordered cluster buffers backing a [`ClusterConfig`].
pub(crate) struct OwnedClusterSetup {
    pub boot_row: Vec<u32>,
    pub boot_kept: Vec<u32>,
    pub g_boot: usize,
    pub err_rows: Vec<(Vec<u32>, usize)>,
    pub err_equals_boot: bool,
    pub ssc_factor: f64,
    pub per_subset_g_adj: bool,
}

impl OwnedClusterSetup {
    pub(crate) fn from_dims(
        err_clusters: &[ClusterDim],
        wb_cluster: Option<&ClusterDim>,
        row_order: &[u32],
        ssc_factor: f64,
        per_subset_g_adj: bool,
        pool: &Pool,
    ) -> Result<Option<Self>, RfError> {
        let to_row = |codes: &[u32]| -> Vec<u32> {
            let mut out = vec![0u32; codes.len()];
            for (pos, &ki) in row_order.iter().enumerate() {
                out[pos] = codes[ki as usize];
            }
            out
        };

        let (boot_kept, g_boot) = if let Some(wb) = wb_cluster {
            (wb.codes.clone(), wb.g)
        } else if err_clusters.is_empty() {
            return Ok(None);
        } else if err_clusters.len() == 1 {
            (err_clusters[0].codes.clone(), err_clusters[0].g)
        } else {
            let owned: Vec<Codes> = err_clusters.iter().map(|c| Codes { codes: c.codes.clone(), n: c.g as u32 }).collect();
            let refs: Vec<&Codes> = owned.iter().collect();
            let comb = factor::combine(&refs, pool)?;
            (comb.codes, comb.n as usize)
        };
        let boot_row = to_row(&boot_kept);

        // When bootcluster was given without vce(cluster ...), error clustering defaults to bootcluster.
        let err_rows: Vec<(Vec<u32>, usize)> = if err_clusters.is_empty() {
            vec![(boot_row.clone(), g_boot)]
        } else {
            err_clusters.iter().map(|c| (to_row(&c.codes), c.g)).collect()
        };

        let err_equals_boot = err_rows.len() == 1 && err_rows[0].1 == g_boot && err_rows[0].0 == boot_row;
        Ok(Some(Self { boot_row, boot_kept, g_boot, err_rows, err_equals_boot, ssc_factor, per_subset_g_adj }))
    }

    pub(crate) fn as_config(&self) -> ClusterConfig<'_> {
        let err_dims = self.err_rows.iter().map(|(c, g)| (c.as_slice(), *g)).collect();
        ClusterConfig {
            boot_codes: &self.boot_row,
            boot_kept: &self.boot_kept,
            g_boot: self.g_boot,
            err_dims,
            err_equals_boot: self.err_equals_boot,
            ssc_factor: self.ssc_factor,
            per_subset_g_adj: self.per_subset_g_adj,
        }
    }
}

/// Writes `wb_*` scalars, macros, confidence interval matrix, and notes onto an `EstResult`.
pub(crate) fn attach_wildboot_result(res: &mut EstResult, wb: &WildBootResult) {
    let scal = |r: &mut EstResult, name: &str, v: f64| {
        r.set_scalar(name, if v.is_finite() { v } else { f64::NAN });
    };
    scal(res, "wb_stat", wb.stat);
    scal(res, "wb_p", wb.p_val);
    scal(res, "wb_reps", wb.reps as f64);
    scal(res, "wb_reps_feasible", wb.reps_feasible as f64);
    scal(res, "wb_q", wb.q as f64);
    scal(res, "wb_full_enum", f64::from(u8::from(wb.full_enumeration)));
    scal(res, "wb_impose_null", f64::from(u8::from(wb.impose_null)));
    scal(res, "wb_ssc", wb.ssc);
    if let Some(g) = wb.n_clusters {
        scal(res, "wb_n_clust", g as f64);
    }
    if let Some(pt) = wb.point_estimate {
        scal(res, "wb_point", pt);
    }
    if let Some(se) = wb.se {
        scal(res, "wb_se", se);
    }
    if let Some((lo, hi)) = wb.conf_int {
        scal(res, "wb_ci_lo", lo);
        scal(res, "wb_ci_hi", hi);
        res.matrices.push(NamedMat { name: "wb_ci".into(), rows: 1, cols: 2, data: vec![lo, hi] });
    }
    res.macros.push(("wb_method".into(), wb.method.name().into()));
    res.macros.push(("wb_boottype".into(), wb.boot_type.name().into()));
    res.macros.push(("wb_weights".into(), wb.weights.name().into()));
    res.macros.push(("wb_pvaltype".into(), wb.pval_type.name().into()));
    res.macros.push(("wb_stattype".into(), wb.stat_type.name().into()));
    for note in &wb.notes {
        if !res.notes.contains(note) {
            res.notes.push(note.clone());
        }
    }
}

/// Symmetric pseudoinverse `A^+` via `faer` self-adjoint eigendecomposition.
/// Eigenvalues `<= tol_rel * max(lambda_abs)` are zeroed.
pub(crate) fn sym_pinv(a: &[f64], n: usize, tol_rel: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let m = faer::Mat::<f64>::from_fn(n, n, |i, j| 0.5 * (a[i * n + j] + a[j * n + i]));
    let Ok(evd) = m.self_adjoint_eigen(faer::Side::Lower) else {
        return vec![0.0; n * n];
    };
    let s = evd.S().column_vector();
    let u = evd.U();
    let lmax = (0..n).map(|i| s[i].abs()).fold(0.0f64, f64::max);
    let cut = tol_rel * lmax;
    let mut inv = vec![0.0; n * n];
    if lmax == 0.0 {
        return inv;
    }
    for e in 0..n {
        let lam = s[e];
        if lam <= cut {
            continue;
        }
        let inv_l = 1.0 / lam;
        for i in 0..n {
            let ui = u[(i, e)] * inv_l;
            for j in 0..=i {
                let val = ui * u[(j, e)];
                inv[i * n + j] += val;
                if i != j {
                    inv[j * n + i] += val;
                }
            }
        }
    }
    inv
}

/// Symmetric positive-semidefinite Cholesky / pseudoinverse quadratic form `x' V^{-1} x` (for `q x q`).
pub(crate) fn quad_inv_sym(x: &[f64], v: &[f64], q: usize) -> f64 {
    if q == 1 {
        let vv = v[0];
        if !vv.is_finite() || vv <= 1e-300 {
            return f64::NAN;
        }
        return x[0] * x[0] / vv;
    }
    if x.iter().chain(v).any(|z| !z.is_finite()) {
        return f64::NAN;
    }
    let vinv = sym_pinv(v, q, 1e-12);
    let mut s = 0.0;
    for i in 0..q {
        let mut row = 0.0;
        for j in 0..q {
            row += vinv[i * q + j] * x[j];
        }
        s += x[i] * row;
    }
    if s < 0.0 && s > -1e-12 { 0.0 } else { s }
}

/// Reparameterizes `R * beta = r` (`R` is `q x k`) as `beta = beta0 + N * theta` (`theta` in `R^{k-q}`),
/// where `beta0` is minimum-norm (`R * beta0 = r`) and `N` (`k x (k-q)`, row-major) is an
/// orthonormal basis of `ker(R)`.
pub(crate) fn nullspace_reparam(r_mat: &[f64], r_vec: &[f64], q: usize, k: usize) -> Result<(Vec<f64>, Vec<f64>), RfError> {
    let fa = faer::Mat::<f64>::from_fn(q, k, |i, j| r_mat[i * k + j]);
    let svd = fa.svd().map_err(|_| RfError::InvalidInput("wildboot: SVD of R failed".into()))?;
    let u = svd.U();
    let s = svd.S().column_vector();
    let v = svd.V();
    let mut order: Vec<usize> = (0..q).collect();
    order.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
    let mut ut_r = vec![0.0; q];
    for &idx in &order {
        let sv = s[idx];
        if sv == 0.0 {
            return Err(RfError::InvalidInput("wildboot: singular constraint matrix R".into()));
        }
        let dot: f64 = (0..q).map(|i| u[(i, idx)] * r_vec[i]).sum();
        ut_r[idx] = dot / sv;
    }
    let mut beta0 = vec![0.0; k];
    for i in 0..k {
        beta0[i] = (0..q).map(|j| v[(i, j)] * ut_r[j]).sum();
    }
    let m = k - q;
    let mut n_basis = vec![0.0; k * m];
    for i in 0..k {
        for c in 0..m {
            n_basis[i * m + c] = v[(i, q + c)];
        }
    }
    Ok((beta0, n_basis))
}

/// Empirical p-value from `t_boot` (skips non-finite draws).
pub(crate) fn compute_pvalue(stat: f64, t_boot: &[f64], q: usize, pval_type: PvalType) -> (f64, usize) {
    if !stat.is_finite() {
        return (f64::NAN, 0);
    }
    let mut feas = 0usize;
    let mut count_hi = 0usize;
    let mut count_lo = 0usize;
    let eps = 1e-12 * (1.0 + stat.abs());
    let abs_stat = stat.abs();
    for &tb in t_boot {
        if !tb.is_finite() {
            continue;
        }
        feas += 1;
        if q > 1 {
            if tb > stat + eps {
                count_hi += 1;
            }
        } else {
            match pval_type {
                PvalType::TwoTailed => {
                    if tb.abs() > abs_stat + eps {
                        count_hi += 1;
                    }
                }
                PvalType::EqualTailed => {
                    if tb > stat + eps {
                        count_hi += 1;
                    }
                    if tb < stat - eps {
                        count_lo += 1;
                    }
                }
                PvalType::Greater => {
                    if tb > stat + eps {
                        count_hi += 1;
                    }
                }
                PvalType::Less => {
                    if tb < stat - eps {
                        count_lo += 1;
                    }
                }
            }
        }
    }
    if feas == 0 {
        return (f64::NAN, 0);
    }
    let f = feas as f64;
    let p = if q > 1 {
        count_hi as f64 / f
    } else {
        match pval_type {
            PvalType::TwoTailed | PvalType::Greater => count_hi as f64 / f,
            PvalType::Less => count_lo as f64 / f,
            PvalType::EqualTailed => (2.0 * (count_hi.min(count_lo) as f64 / f)).min(1.0),
        }
    };
    (p, feas)
}

/// Closed-form WCU confidence interval from bootstrap t-quantiles: `point +/- se * crit`.
pub(crate) fn invert_ci_wcu(point: f64, se: f64, t_boot: &[f64], pval_type: PvalType, alpha: f64) -> (f64, f64) {
    let mut vals: Vec<f64> = t_boot.iter().copied().filter(|x| x.is_finite()).collect();
    if vals.is_empty() || !se.is_finite() || !point.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let quantile = |sorted: &[f64], p: f64| -> f64 {
        let m = sorted.len();
        if m == 1 {
            return sorted[0];
        }
        let pos = p.clamp(0.0, 1.0) * (m - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(m - 1);
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    };
    match pval_type {
        PvalType::TwoTailed => {
            vals.iter_mut().for_each(|x| *x = x.abs());
            vals.sort_by(f64::total_cmp);
            let crit = quantile(&vals, 1.0 - alpha);
            (point - crit * se, point + crit * se)
        }
        PvalType::EqualTailed => {
            vals.sort_by(f64::total_cmp);
            // p_lo(r): H0 r accepted iff t_lo <= (point - r)/se <= t_hi -> r in [point - t_hi*se, point - t_lo*se].
            let t_lo = quantile(&vals, alpha / 2.0);
            let t_hi = quantile(&vals, 1.0 - alpha / 2.0);
            (point - t_hi * se, point - t_lo * se)
        }
        PvalType::Greater => {
            vals.sort_by(f64::total_cmp);
            let crit = quantile(&vals, 1.0 - alpha);
            (point - crit * se, f64::INFINITY)
        }
        PvalType::Less => {
            vals.sort_by(f64::total_cmp);
            let crit = quantile(&vals, alpha);
            (f64::NEG_INFINITY, point - crit * se)
        }
    }
}

/// Chandrupatla / bisection test-inversion confidence interval matching `WildBootTests.jl/src/plot-CI.jl`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn invert_ci_search(
    point: f64,
    se: f64,
    alpha: f64,
    pval_type: PvalType,
    tol: f64,
    maxiter: usize,
    reps_feas: usize,
    mut eval_p: impl FnMut(f64) -> Result<f64, RfError>,
) -> Result<(f64, f64), RfError> {
    if !point.is_finite() || !se.is_finite() || se <= 0.0 || reps_feas == 0 {
        return Ok((f64::NAN, f64::NAN));
    }
    let z = qnorm(1.0 - alpha / 2.0).abs().max(1.5);
    let hw = (1.5 * z * se).max(1e-6 * (1.0 + point.abs()));
    let eq = if pval_type == PvalType::EqualTailed { 2.0 } else { 1.0 };
    let p_tol = (eq / reps_feas as f64) * 1.0001;
    let x_tol = tol * (1.0 + point.abs());

    let find_bound = |sign: f64, eval: &mut dyn FnMut(f64) -> Result<f64, RfError>| -> Result<f64, RfError> {
        // `x_in` has p >= alpha (start at `point`), `x_out` steps out until p < alpha.
        let mut x_in = point;
        let mut p_in = eval(x_in)?;
        if !p_in.is_finite() || p_in < alpha {
            p_in = 1.0;
        }
        let mut step = hw;
        let mut x_out = point + sign * step;
        let mut p_out = f64::NAN;
        let mut bracketed = false;
        for _ in 0..16 {
            let p = eval(x_out)?;
            if p.is_finite() {
                if p < alpha {
                    p_out = p;
                    bracketed = true;
                    break;
                }
                x_in = x_out;
                p_in = p;
            }
            step *= 2.0;
            x_out = point + sign * step;
        }
        if !bracketed {
            return Ok(if sign < 0.0 { f64::NEG_INFINITY } else { f64::INFINITY });
        }
        // Bisect/Chandrupatla between (x_in, p_in >= alpha) and (x_out, p_out < alpha).
        let (mut x1, mut f1) = (x_in, p_in - alpha);
        let (mut x2, mut f2) = (x_out, p_out - alpha);
        let mut x3;
        let mut f3;
        let mut t = 0.5_f64;
        for _ in 0..maxiter {
            if (x2 - x1).abs() <= x_tol || f1.abs().min(f2.abs()) <= p_tol {
                break;
            }
            let x = x1 + t * (x2 - x1);
            let fx = eval(x)? - alpha;
            if !fx.is_finite() {
                // Step back to midpoint on non-finite evaluation.
                t = 0.5;
                let xm = 0.5 * (x1 + x2);
                let fm = eval(xm)? - alpha;
                if !fm.is_finite() {
                    break;
                }
                if (fm < 0.0) == (f1 < 0.0) {
                    x1 = xm;
                    f1 = fm;
                } else {
                    x2 = xm;
                    f2 = fm;
                }
                continue;
            }
            if (fx < 0.0) == (f1 < 0.0) {
                x3 = x1;
                f3 = f1;
                x1 = x;
                f1 = fx;
            } else {
                x3 = x2;
                f3 = f2;
                x2 = x;
                f2 = fx;
            }
            if (x2 - x1).abs() <= x_tol || f1.abs().min(f2.abs()) <= p_tol {
                break;
            }
            let denom32 = f3 - f2;
            let denom12 = f1 - f2;
            let denom31 = f3 - f1;
            if denom32 == 0.0 || denom12 == 0.0 || denom31 == 0.0 || (x2 - x1) == 0.0 || (x3 - x2) == 0.0 {
                t = 0.5;
                continue;
            }
            let phi1 = denom12 / denom32;
            let xi1 = (x1 - x2) / (x3 - x2);
            if phi1 * phi1 > xi1 || xi1 > 2.0 * phi1 - phi1 * phi1 {
                t = 0.5;
            } else {
                let raw_t = (f3 / denom12 + (x3 - x1) / ((x2 - x1) * denom31) * f2) * (f1 / denom32);
                t = raw_t.clamp(1e-6, 1.0 - 1e-6);
            }
        }
        Ok(if f1.abs() <= f2.abs() { x1 } else { x2 })
    };

    let lo = if pval_type == PvalType::Less { f64::NEG_INFINITY } else { find_bound(-1.0, &mut eval_p)? };
    let hi = if pval_type == PvalType::Greater { f64::INFINITY } else { find_bound(1.0, &mut eval_p)? };
    Ok((lo, hi))
}

/// Intersection cells of `(boot_cluster, err_dims...)` over `n` rows in solver row order.
pub(crate) struct Cells {
    pub n_cells: usize,
    /// `cell_of_row[i] in 0..n_cells` (row order).
    pub cell_of_row: Vec<u32>,
    /// `cell_boot[c] in 0..g_boot`.
    pub cell_boot: Vec<u32>,
    /// All subsets `S` of `0..D` (inclusion-exclusion): `(sign, g_s, subset_cluster_of_cell)`.
    pub subsets: Vec<(f64, usize, Vec<u32>)>,
}

impl Cells {
    pub(crate) fn build(cl: &ClusterConfig<'_>) -> Result<Self, RfError> {
        let n = cl.boot_codes.len();
        if cl.err_equals_boot {
            let g = cl.g_boot;
            let cell_of_row = cl.boot_codes.to_vec();
            let cell_boot: Vec<u32> = (0..g as u32).collect();
            let subsets = vec![(1.0, g, cell_boot.clone())];
            return Ok(Cells { n_cells: g, cell_of_row, cell_boot, subsets });
        }
        let d = cl.err_dims.len();
        // Fast encoding when product of cardinalities fits in u64; otherwise fallback to Vec<u32> key.
        let prod_u64 = std::iter::once(cl.g_boot)
            .chain(cl.err_dims.iter().map(|(_, g)| *g))
            .try_fold(1u64, |acc, g| acc.checked_mul((g as u64).max(1)));
        let mut cell_of_row = vec![0u32; n];
        let mut cell_boot = Vec::<u32>::new();
        let mut cell_err: Vec<Vec<u32>> = vec![Vec::new(); d];
        if let Some(prod) = prod_u64
            && prod <= 10_000_000
        {
            let mut map: HashMap<u64, u32> = HashMap::new();
            for i in 0..n {
                let mut key = cl.boot_codes[i] as u64;
                for (codes, g) in &cl.err_dims {
                    key = key * (*g as u64) + codes[i] as u64;
                }
                let id = if let Some(&id) = map.get(&key) {
                    id
                } else {
                    let id = cell_boot.len() as u32;
                    map.insert(key, id);
                    cell_boot.push(cl.boot_codes[i]);
                    for (j, (codes, _)) in cl.err_dims.iter().enumerate() {
                        cell_err[j].push(codes[i]);
                    }
                    id
                };
                cell_of_row[i] = id;
            }
        } else {
            let mut map: HashMap<Vec<u32>, u32> = HashMap::new();
            let mut key = vec![0u32; 1 + d];
            for i in 0..n {
                key[0] = cl.boot_codes[i];
                for (j, (codes, _)) in cl.err_dims.iter().enumerate() {
                    key[1 + j] = codes[i];
                }
                let id = if let Some(&id) = map.get(&key) {
                    id
                } else {
                    let id = cell_boot.len() as u32;
                    map.insert(key.clone(), id);
                    cell_boot.push(key[0]);
                    for j in 0..d {
                        cell_err[j].push(key[1 + j]);
                    }
                    id
                };
                cell_of_row[i] = id;
            }
        }
        let n_cells = cell_boot.len();
        let mut subsets = Vec::new();
        for mask in 1usize..(1usize << d) {
            let bits: Vec<usize> = (0..d).filter(|&j| (mask >> j) & 1 == 1).collect();
            let sign = if bits.len() % 2 == 1 { 1.0 } else { -1.0 };
            if bits.len() == 1 {
                let dim = bits[0];
                subsets.push((sign, cl.err_dims[dim].1, cell_err[dim].clone()));
            } else {
                let mut cmap: HashMap<Vec<u32>, u32> = HashMap::new();
                let mut sub_codes = vec![0u32; n_cells];
                let mut g_s = 0usize;
                let mut key = vec![0u32; bits.len()];
                for c in 0..n_cells {
                    for (p, &b) in bits.iter().enumerate() {
                        key[p] = cell_err[b][c];
                    }
                    let id = if let Some(&id) = cmap.get(&key) {
                        id
                    } else {
                        let id = g_s as u32;
                        cmap.insert(key.clone(), id);
                        g_s += 1;
                        id
                    };
                    sub_codes[c] = id;
                }
                subsets.push((sign, g_s, sub_codes));
            }
        }
        Ok(Cells { n_cells, cell_of_row, cell_boot, subsets })
    }

    /// Computes subset-summed quadratic variance `V = sum_S sign_S * mult_S * sum_m q_{S,m} q_{S,m}'`
    /// from cell-level score vectors `cell_q` (`n_cells * q`, row-major).
    pub(crate) fn variance_from_cell_q(&self, cell_q: &[f64], q: usize, ssc_factor: f64, per_subset_g_adj: bool, out: &mut [f64]) {
        out.fill(0.0);
        if self.subsets.len() == 1 && self.subsets[0].1 == self.n_cells {
            let mult = ssc_factor * if per_subset_g_adj {
                let g = self.n_cells as f64;
                if g > 1.0 { g / (g - 1.0) } else { 0.0 }
            } else {
                1.0
            };
            for r in cell_q.chunks_exact(q) {
                for i in 0..q {
                    let ri = r[i] * mult;
                    for j in i..q {
                        out[i * q + j] += ri * r[j];
                    }
                }
            }
            for i in 0..q {
                for j in 0..i {
                    out[i * q + j] = out[j * q + i];
                }
            }
            return;
        }
        let mut buf = Vec::new();
        for (sign, g_s, sub_code) in &self.subsets {
            let g_mult = if per_subset_g_adj {
                if *g_s > 1 { *g_s as f64 / (*g_s as f64 - 1.0) } else { 0.0 }
            } else {
                1.0
            };
            let mult = sign * ssc_factor * g_mult;
            buf.clear();
            buf.resize(*g_s * q, 0.0);
            for (c, r) in cell_q.chunks_exact(q).enumerate() {
                let m = sub_code[c] as usize;
                let dst = &mut buf[m * q..(m + 1) * q];
                for j in 0..q {
                    dst[j] += r[j];
                }
            }
            for r in buf.chunks_exact(q) {
                for i in 0..q {
                    let ri = r[i] * mult;
                    for j in i..q {
                        out[i * q + j] += ri * r[j];
                    }
                }
            }
        }
        for i in 0..q {
            for j in 0..i {
                out[i * q + j] = out[j * q + i];
            }
        }
    }
}

/// Non-nested FE column descriptor for `Path 1`.
pub(crate) struct NnBasisTerm<'a> {
    pub codes: &'a [u32],
    pub g: usize,
    pub intercept: bool,
    pub slopes: &'a [Vec<f64>],
    /// Global column offset in `0..m_nn`.
    pub col_offset: usize,
}

impl NnBasisTerm<'_> {
    fn n_slots(&self) -> usize {
        usize::from(self.intercept) + self.slopes.len()
    }
}

/// Non-nested fixed-effect projection strategy for exact clustered wild bootstrap.
pub(crate) enum NonNestedPlan {
    /// All FE terms are nested in `boot_cluster` (or no FE terms): zero projection overhead.
    None,
    /// Low-rank basis (`m_nn <= min(g_boot, 512)`): exact factored `D_nn (D_nn' W D_nn)^+ D_nn' W`.
    Path1 {
        m_nn: usize,
        nn_indices: Vec<usize>,
        g_nn_pinv: Vec<f64>,
        /// Optional `Q_nn` (`n x r_nn` row-major, solver row order) when needed for MNW `T31/13/33`.
        q_nn: Option<(Vec<f64>, usize)>,
    },
    /// Small cluster count (`g_boot <= 512 && g_boot < m_nn`): solve `1_g * target` through `absorber`.
    Path2,
    /// Too large for both (`g_boot > 512 && m_nn > 512`).
    TooLarge { g_boot: usize, m_nn: usize },
}

pub(crate) fn to_kept_u32(v: &[u32], row_order: &[u32]) -> Vec<u32> {
    let mut out = vec![0u32; v.len()];
    for (pos, &ki) in row_order.iter().enumerate() {
        out[ki as usize] = v[pos];
    }
    out
}

pub(crate) fn classify_non_nested(
    terms: &[Term],
    row_order: &[u32],
    boot_kept: &[u32],
    cell_kept: Option<&[u32]>,
    g_boot: usize,
    w_rows: Option<&[f64]>,
    want_q_nn: bool,
    tol: f64,
    maxiter: usize,
    pool: &Pool,
    ctx: &Ctx,
) -> Result<NonNestedPlan, RfError> {
    if terms.is_empty() || terms.iter().all(|t| nested_within(&t.codes, t.g, boot_kept)) {
        return Ok(NonNestedPlan::None);
    }
    let nest_sup = cell_kept.unwrap_or(boot_kept);
    let mut nest_terms = Vec::new();
    let mut nn_indices = Vec::new();
    for (q, t) in terms.iter().enumerate() {
        if nested_within(&t.codes, t.g, nest_sup) {
            nest_terms.push(t.clone());
        } else {
            nn_indices.push(q);
        }
    }
    if nn_indices.is_empty() {
        return Ok(NonNestedPlan::None);
    }
    let m_nn: usize = nn_indices
        .iter()
        .map(|&q| terms[q].g * (usize::from(terms[q].intercept) + terms[q].slopes.len()))
        .sum();
    if (!want_q_nn && m_nn > g_boot.min(512)) || (want_q_nn && m_nn > 256) {
        if !want_q_nn && g_boot <= 512 {
            return Ok(NonNestedPlan::Path2);
        }
        return Ok(NonNestedPlan::TooLarge { g_boot, m_nn });
    }

    // Path 1: compute G_nn = D_nn' W M_{D_nest} D_nn (m_nn x m_nn) and its pseudoinverse.
    let n = row_order.len();
    let mut basis_terms = Vec::with_capacity(nn_indices.len());
    let mut off = 0usize;
    for &q in &nn_indices {
        let t = &terms[q];
        let bt = NnBasisTerm { codes: &t.codes, g: t.g, intercept: t.intercept, slopes: &t.slopes, col_offset: off };
        off += bt.n_slots() * t.g;
        basis_terms.push(bt);
    }

    let (g_nn, q_nn) = if nest_terms.is_empty() {
        // Zero solves: D_nn' W D_nn directly.
        let g_nn = par::sum_chunks(pool, n, CHUNK, m_nn * m_nn, |range, acc| {
            let mut idx_val: Vec<(usize, f64)> = Vec::with_capacity(nn_indices.len() * 2);
            for i in range {
                let ki = row_order[i] as usize;
                let wi = w_rows.map_or(1.0, |w| w[i]);
                idx_val.clear();
                for bt in &basis_terms {
                    let lvl = bt.codes[ki] as usize;
                    let mut slot = 0usize;
                    if bt.intercept {
                        idx_val.push((bt.col_offset + lvl, 1.0));
                        slot += 1;
                    }
                    for s in bt.slopes {
                        idx_val.push((bt.col_offset + slot * bt.g + lvl, s[ki]));
                        slot += 1;
                    }
                }
                for &(ca, va) in &idx_val {
                    for &(cb, vb) in &idx_val {
                        acc[ca * m_nn + cb] += wi * va * vb;
                    }
                }
            }
        });
        let g_nn_pinv = sym_pinv(&g_nn, m_nn, 1e-10);
        let q_nn = if want_q_nn && m_nn <= 256 {
            let m = faer::Mat::<f64>::from_fn(m_nn, m_nn, |i, j| 0.5 * (g_nn[i * m_nn + j] + g_nn[j * m_nn + i]));
            let evd = m.self_adjoint_eigen(faer::Side::Lower).map_err(|_| RfError::Internal("EVD failed".into()))?;
            let s = evd.S().column_vector();
            let u = evd.U();
            let lmax = (0..m_nn).map(|i| s[i].abs()).fold(0.0f64, f64::max);
            let keep: Vec<usize> = (0..m_nn).filter(|&e| s[e] > 1e-10 * lmax).collect();
            let r_nn = keep.len();
            let mut qmat = par::alloc_vec(n * r_nn, 0.0)?;
            if r_nn > 0 {
                pool.install(|| {
                    qmat.par_chunks_mut(CHUNK * r_nn).enumerate().for_each(|(ci, qchunk)| {
                        let base = ci * CHUNK;
                        for (j, row_q) in qchunk.chunks_exact_mut(r_nn).enumerate() {
                            let ki = row_order[base + j] as usize;
                            for bt in &basis_terms {
                                let lvl = bt.codes[ki] as usize;
                                let mut slot = 0usize;
                                if bt.intercept {
                                    let col = bt.col_offset + lvl;
                                    for (e_idx, &e) in keep.iter().enumerate() {
                                        row_q[e_idx] += u[(col, e)] / s[e].sqrt();
                                    }
                                    slot += 1;
                                }
                                for sl in bt.slopes {
                                    let col = bt.col_offset + slot * bt.g + lvl;
                                    let v = sl[ki];
                                    for (e_idx, &e) in keep.iter().enumerate() {
                                        row_q[e_idx] += v * u[(col, e)] / s[e].sqrt();
                                    }
                                    slot += 1;
                                }
                            }
                        }
                    });
                });
            }
            Some((qmat, r_nn))
        } else {
            None
        };
        return Ok(NonNestedPlan::Path1 { m_nn, nn_indices, g_nn_pinv, q_nn });
    } else {
        // Build absorber on nested terms (weights in kept order).
        let w_kept = w_rows.map(|wr| crate::fe_recovery::to_kept_order(wr, row_order, pool)).transpose()?;
        let mut nest_abs = crate::solve::build(&nest_terms, w_kept.as_deref(), n, pool, ctx)?;
        let n_ord = nest_abs.row_order().to_vec();
        let nw = nest_abs.weights().map(<[f64]>::to_vec);
        let opts = SolveOpts { tol: tol.min(1e-10), maxiter, want_coefs: false };
        let batch = if want_q_nn && m_nn <= 256 { m_nn } else { m_nn.min(64) };
        let mut g_nn = vec![0.0; m_nn * m_nn];
        let mut d_tilde_all = if want_q_nn && m_nn <= 256 { Some(par::alloc_vec(n * m_nn, 0.0)?) } else { None };
        let mut c0 = 0;
        while c0 < m_nn {
            let b = (m_nn - c0).min(batch);
            let mut blk = par::alloc_vec(n * b, 0.0)?;
            pool.install(|| {
                blk.par_chunks_mut(CHUNK * b).enumerate().for_each(|(ci, chunk)| {
                    let base = ci * CHUNK;
                    for (j, row) in chunk.chunks_exact_mut(b).enumerate() {
                        let ki = n_ord[base + j] as usize;
                        for bt in &basis_terms {
                            let lvl = bt.codes[ki] as usize;
                            let mut slot = 0usize;
                            if bt.intercept {
                                let col = bt.col_offset + lvl;
                                if col >= c0 && col < c0 + b {
                                    row[col - c0] = 1.0;
                                }
                                slot += 1;
                            }
                            for sl in bt.slopes {
                                let col = bt.col_offset + slot * bt.g + lvl;
                                if col >= c0 && col < c0 + b {
                                    row[col - c0] = sl[ki];
                                }
                                slot += 1;
                            }
                        }
                    }
                });
            });
            nest_abs.residualize(&mut blk, b, &opts, None, pool, ctx)?;
            if let Some(ref mut all) = d_tilde_all {
                for i in 0..n {
                    all[i * m_nn + c0..i * m_nn + c0 + b].copy_from_slice(&blk[i * b..(i + 1) * b]);
                }
            }
            let part = par::sum_chunks(pool, n, CHUNK, m_nn * b, |range, acc| {
                let mut idx_val: Vec<(usize, f64)> = Vec::with_capacity(nn_indices.len() * 2);
                for i in range {
                    let ki = n_ord[i] as usize;
                    let wi = nw.as_deref().map_or(1.0, |w| w[i]);
                    let brow = &blk[i * b..(i + 1) * b];
                    idx_val.clear();
                    for bt in &basis_terms {
                        let lvl = bt.codes[ki] as usize;
                        let mut slot = 0usize;
                        if bt.intercept {
                            idx_val.push((bt.col_offset + lvl, 1.0));
                            slot += 1;
                        }
                        for sl in bt.slopes {
                            idx_val.push((bt.col_offset + slot * bt.g + lvl, sl[ki]));
                            slot += 1;
                        }
                    }
                    for &(ca, va) in &idx_val {
                        let wva = wi * va;
                        let dst = &mut acc[ca * b..(ca + 1) * b];
                        for j in 0..b {
                            dst[j] += wva * brow[j];
                        }
                    }
                }
            });
            for ca in 0..m_nn {
                for j in 0..b {
                    g_nn[ca * m_nn + (c0 + j)] = part[ca * b + j];
                }
            }
            c0 += b;
        }
        // Symmetrize g_nn.
        for i in 0..m_nn {
            for j in 0..i {
                let v = 0.5 * (g_nn[i * m_nn + j] + g_nn[j * m_nn + i]);
                g_nn[i * m_nn + j] = v;
                g_nn[j * m_nn + i] = v;
            }
        }
        let q_nn = if let Some(d_tilde) = d_tilde_all {
            let m = faer::Mat::<f64>::from_fn(m_nn, m_nn, |i, j| g_nn[i * m_nn + j]);
            let evd = m.self_adjoint_eigen(faer::Side::Lower).map_err(|_| RfError::Internal("EVD failed".into()))?;
            let s = evd.S().column_vector();
            let u = evd.U();
            let lmax = (0..m_nn).map(|i| s[i].abs()).fold(0.0f64, f64::max);
            let keep: Vec<usize> = (0..m_nn).filter(|&e| s[e] > 1e-10 * lmax).collect();
            let r_nn = keep.len();
            let mut u_scaled = vec![0.0; m_nn * r_nn];
            for col in 0..m_nn {
                for (e_idx, &e) in keep.iter().enumerate() {
                    u_scaled[col * r_nn + e_idx] = u[(col, e)] / s[e].sqrt();
                }
            }
            // Compute Q_nn in nest_abs order, then permute to caller's row_order via kept index.
            let mut q_nest = par::alloc_vec(n * r_nn, 0.0)?;
            if r_nn > 0 {
                pool.install(|| {
                    q_nest.par_chunks_mut(CHUNK * r_nn).zip(d_tilde.par_chunks(CHUNK * m_nn)).for_each(|(qc, dc)| {
                        for (qrow, drow) in qc.chunks_exact_mut(r_nn).zip(dc.chunks_exact(m_nn)) {
                            for col in 0..m_nn {
                                let dv = drow[col];
                                if dv != 0.0 {
                                    let ur = &u_scaled[col * r_nn..(col + 1) * r_nn];
                                    for e in 0..r_nn {
                                        qrow[e] += dv * ur[e];
                                    }
                                }
                            }
                        }
                    });
                });
            }
            drop(d_tilde);
            // Permute q_nest (n_ord) -> kept -> row_order.
            let mut nest_pos_of_kept = vec![0u32; n];
            for (r, &k) in n_ord.iter().enumerate() {
                nest_pos_of_kept[k as usize] = r as u32;
            }
            let mut q_main = par::alloc_vec(n * r_nn, 0.0)?;
            if r_nn > 0 {
                pool.install(|| {
                    q_main.par_chunks_mut(CHUNK * r_nn).enumerate().for_each(|(ci, qc)| {
                        let base = ci * CHUNK;
                        for (j, dst) in qc.chunks_exact_mut(r_nn).enumerate() {
                            let src_row = nest_pos_of_kept[row_order[base + j] as usize] as usize;
                            dst.copy_from_slice(&q_nest[src_row * r_nn..(src_row + 1) * r_nn]);
                        }
                    });
                });
            }
            Some((q_main, r_nn))
        } else {
            None
        };
        (g_nn, q_nn)
    };
    let g_nn_pinv = sym_pinv(&g_nn, m_nn, 1e-10);
    Ok(NonNestedPlan::Path1 { m_nn, nn_indices, g_nn_pinv, q_nn })
}

/// Helper for Path 1: accumulates `D_nn' W * cols` (`n_cols` columns in `cols_fn`) grouped by `group_of_row`
/// (`n_groups` groups), returning `n_groups * n_cols * m_nn` row-major (`(g * n_cols + c) * m_nn + p`).
pub(crate) fn accumulate_dnn_groups(
    terms: &[Term],
    nn_indices: &[usize],
    m_nn: usize,
    row_order: &[u32],
    group_of_row: &[u32],
    n_groups: usize,
    w_rows: Option<&[f64]>,
    n_cols: usize,
    fill_cols: impl Fn(usize, &mut [f64]) + Sync,
    pool: &Pool,
) -> Vec<f64> {
    let mut basis_terms = Vec::with_capacity(nn_indices.len());
    let mut off = 0usize;
    for &q in nn_indices {
        let t = &terms[q];
        let bt = NnBasisTerm { codes: &t.codes, g: t.g, intercept: t.intercept, slopes: &t.slopes, col_offset: off };
        off += bt.n_slots() * t.g;
        basis_terms.push(bt);
    }
    let n = row_order.len();
    let total = n_groups * n_cols * m_nn;
    par::sum_chunks(pool, n, CHUNK, total, |range, acc| {
        let mut vals = vec![0.0; n_cols];
        for i in range {
            let ki = row_order[i] as usize;
            let g = group_of_row[i] as usize;
            let wi = w_rows.map_or(1.0, |w| w[i]);
            fill_cols(i, &mut vals);
            let gbase = g * n_cols * m_nn;
            for bt in &basis_terms {
                let lvl = bt.codes[ki] as usize;
                let mut slot = 0usize;
                if bt.intercept {
                    let col = bt.col_offset + lvl;
                    for c in 0..n_cols {
                        acc[gbase + c * m_nn + col] += wi * vals[c];
                    }
                    slot += 1;
                }
                for sl in bt.slopes {
                    let col = bt.col_offset + slot * bt.g + lvl;
                    let wv = wi * sl[ki];
                    for c in 0..n_cols {
                        acc[gbase + c * m_nn + col] += wv * vals[c];
                    }
                    slot += 1;
                }
            }
        }
    })
}

/// Path 2 helper: given `target_cols` (`n_tgt` vectors of length `n` in solver row order) and `g_boot`
/// bootstrap groups (`boot_codes`), computes `P_D(1_g * target_t)` using `absorber.residualize` in
/// batches of 32 columns, and accumulates `C[(c * n_z + d) * n_tgt * g_boot + t * g_boot + g] = sum_{i in c} w_i Z[i, d] * p_{t, g}[i]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn path2_project_matrices(
    absorber: &mut dyn Absorber,
    boot_codes: &[u32],
    g_boot: usize,
    cell_of_row: &[u32],
    n_cells: usize,
    w_rows: Option<&[f64]>,
    target_cols: &[&[f64]],
    n_z: usize,
    fill_z: impl Fn(usize, &mut [f64]) + Sync,
    tol: f64,
    maxiter: usize,
    pool: &Pool,
    ctx: &Ctx,
) -> Result<Vec<f64>, RfError> {
    let n = boot_codes.len();
    let n_tgt = target_cols.len();
    let total_cols = n_tgt * g_boot;
    let mut out = vec![0.0; n_cells * n_z * total_cols];
    let opts = SolveOpts { tol: tol.min(1e-10), maxiter, want_coefs: false };
    let batch = 32usize;
    let mut c0 = 0usize;
    while c0 < total_cols {
        let b = (total_cols - c0).min(batch);
        let mut raw = par::alloc_vec(n * b, 0.0)?;
        pool.install(|| {
            raw.par_chunks_mut(CHUNK * b).enumerate().for_each(|(ci, chunk)| {
                let base = ci * CHUNK;
                for (j, row) in chunk.chunks_exact_mut(b).enumerate() {
                    let i = base + j;
                    let gi = boot_codes[i] as usize;
                    for col in 0..b {
                        let global = c0 + col;
                        let tgt = global / g_boot;
                        let g = global % g_boot;
                        if gi == g {
                            row[col] = target_cols[tgt][i];
                        }
                    }
                }
            });
        });
        let mut res = raw.clone();
        absorber.residualize(&mut res, b, &opts, None, pool, ctx)?;
        // p = raw - res = P_D(1_g * target).
        let part = par::sum_chunks(pool, n, CHUNK, n_cells * n_z * b, |range, acc| {
            let mut z = vec![0.0; n_z];
            for i in range {
                let cell = cell_of_row[i] as usize;
                let wi = w_rows.map_or(1.0, |w| w[i]);
                fill_z(i, &mut z);
                let raw_row = &raw[i * b..(i + 1) * b];
                let res_row = &res[i * b..(i + 1) * b];
                let cell_off = cell * n_z * b;
                for d in 0..n_z {
                    let wz = wi * z[d];
                    let dst = &mut acc[cell_off + d * b..cell_off + (d + 1) * b];
                    for col in 0..b {
                        dst[col] += wz * (raw_row[col] - res_row[col]);
                    }
                }
            }
        });
        for cell in 0..n_cells {
            for d in 0..n_z {
                let src = &part[(cell * n_z + d) * b..(cell * n_z + d + 1) * b];
                let dst = &mut out[(cell * n_z + d) * total_cols + c0..(cell * n_z + d) * total_cols + c0 + b];
                dst.copy_from_slice(src);
            }
        }
        c0 += b;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::dist::pnorm;

    #[test]
    fn hypothesis_validation_and_nullspace() {
        assert!(Hypothesis::new(vec![1.0, 0.0, 2.0, 0.0], vec![0.0, 1.0], 2, 2).is_err());
        let h = Hypothesis::new(vec![1.0, -1.0, 0.0], vec![0.5], 1, 3).unwrap();
        let (beta0, n) = nullspace_reparam(&h.r_mat, &h.r_vec, 1, 3).unwrap();
        let lhs = h.r_mat[0] * beta0[0] + h.r_mat[1] * beta0[1] + h.r_mat[2] * beta0[2];
        assert!((lhs - 0.5).abs() < 1e-13);
        for c in 0..2 {
            let rn = h.r_mat[0] * n[c] + h.r_mat[1] * n[2 + c] + h.r_mat[2] * n[4 + c];
            assert!(rn.abs() < 1e-13);
        }
    }

    #[test]
    fn pnorm_qnorm_consistent() {
        let p = 0.05;
        let z = qnorm(1.0 - p / 2.0);
        assert!((2.0 * (1.0 - pnorm(z)) - p).abs() < 1e-12);
    }
}
