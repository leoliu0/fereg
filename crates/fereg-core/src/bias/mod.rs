//! Incidental-parameter bias correction for fixed-effects logit (Fernández-Val and Weidner 2016;
//! network version Hinz, Stammann and Wanner 2020), average partial effects and the split-panel /
//! delete-one jackknife.
//!
//! Host-neutral kernels over the estimation rows of a finished logit fit (kept order) and an
//! [`Absorber`] over the same rows and fixed-effect terms:
//! - [`analytical`]: the bias vector under the [`Normalization::Alpaca`] (alpaca 0.3.5
//!   `biasCorr`) or [`Normalization::Logitfe`] (Stata `logitfe` 3.0.3, its lag bug B1 fixed)
//!   normalization, the corrected coefficients beta_c = beta + (H/n)^-1 b, and the exact
//!   re-estimation of the fixed effects at beta_c ([`resolve_fe`]).
//! - [`apes`]: average partial effects at a point, optionally analytically corrected, with the
//!   delta-method variance and the sampling-of-effects term (all N* rows, design sec. 6.3).
//! - [`jackknife`]: the subpanel plan of every variant, the subpanel fits through a host closure
//!   (concurrent, deterministic), and the combination of coefficients and APEs.
//!
//! The fit layer ([`crate::fit::bias_corr`], [`crate::fit::apes`]) drives these for the R and
//! Python packages; the Stata plugin calls them after `Session::glm_parts` with the session's
//! absorber.
//!
//! Determinism: every per-group sum runs serially in a fixed row order, centering goes through the
//! absorber (bit-identical for any thread count), so every result is independent of the thread
//! count.

use crate::absorber::{Absorber, SolveOpts};
use crate::api::Ctx;
use crate::error::RfError;
use crate::gather;
use crate::par::Pool;

mod analytical;
mod ape;
mod jackknife;

pub use analytical::{Corrected, Resolved, analytical, resolve_fe};
pub use ape::{ApeSpec, Apes, FullSample, ape_point, apes, binary};
pub use jackknife::{JkOut, JkPanel, JkSpec, Permutation, SubFit, Subpanel, jackknife};

pub(crate) use analytical::{plan_bandwidth, shifted_eta};
#[cfg(test)]
pub(crate) use ape::apes_retained;

/// Which reference's normalization of the bias terms (they coincide on balanced panels).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Normalization {
    /// alpaca 0.3.5: group ratios sum_g (sum_g A)/(sum_g w) over the FE groups, divided by n;
    /// lags count rows within the group in time order (gaps ignored), weighted by T_g/(T_g - l).
    #[default]
    Alpaca,
    /// logitfe 3.0.3: per-individual means (1/T_i), calendar lags normalized by the number of
    /// rows whose lag is in the sample, the mean over individuals with every lag observed,
    /// time bias divided by the number of individuals. Classic structure, no weights.
    Logitfe,
}

/// Panel structure of the fixed effects.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Structure {
    /// FE1 = individual, FE2 = time (one or two terms).
    #[default]
    Classic,
    /// FE1 = exporter-time, FE2 = importer-time, FE3 = pair (two or three terms).
    Network,
}

/// Which incidental-parameter bias to correct.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WhichBias {
    #[default]
    Both,
    /// The individual-effect bias (FE1 terms and the spectral term; logitfe `tbias(no)`).
    Individual,
    /// The time-effect bias (FE2 term; logitfe `ibias(no)`).
    Time,
}

/// alpaca `sampling.fe`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Sampling {
    #[default]
    Independence,
    Unrestricted,
}

/// Jackknife variants (design sec. 7).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Variant {
    Ss1,
    #[default]
    Ss2,
    Js,
    Sj,
    Jj,
    Double,
}

/// Codes of one fixed-effect term over rows: dense, every code below `g`.
#[derive(Clone, Copy, Debug)]
pub struct FeCodes<'a> {
    pub codes: &'a [u32],
    pub g: usize,
}

/// The estimation sample of a logit fit whose fixed-effect terms are all intercept-only, over the
/// kept rows in kept order (the absorber's rows).
#[derive(Clone, Debug)]
pub struct LogitData<'a> {
    pub y: &'a [f64],
    /// Prior (observation) weights.
    pub weights: Option<&'a [f64]>,
    /// Offset on the link scale (excluded from x'beta, included in eta).
    pub offset: Option<&'a [f64]>,
    /// Kept regressors on their original scale, in coefficient order (no intercept).
    pub x: Vec<&'a [f64]>,
    /// The absorbed terms, in the absorber's order (FE1, FE2[, FE3]).
    pub fe: Vec<FeCodes<'a>>,
    /// N*: rows before any fixed-effect-based drop (the APE divisor).
    pub n_full: usize,
}

/// Panel identifiers of the [`Normalization::Logitfe`] bias terms, over the kept rows.
#[derive(Clone, Copy, Debug)]
pub struct LogitfePanel<'a> {
    /// Individual (xtset panel) codes, dense below `n_individuals`.
    pub individual: &'a [u32],
    pub n_individuals: usize,
    /// Calendar period ((t - t_min)/delta of the xtset time variable): the lag j of row (i, t) is
    /// row (i, t - j) when it is in the sample.
    pub period: &'a [i64],
    /// The model has individual effects (some absorbed term is the panel variable).
    pub individual_effects: bool,
    /// The model has time effects.
    pub time_effects: bool,
}

/// What to correct and how.
#[derive(Clone, Copy, Debug)]
pub struct BiasSpec<'a> {
    pub normalization: Normalization,
    pub structure: Structure,
    pub which: WhichBias,
    /// Bandwidth L of weakly exogenous regressors.
    pub bandwidth: usize,
    /// [`Normalization::Alpaca`]: time key of every kept row ordering the rows inside the lag
    /// groups (FE1 groups classic, FE3 network); ties keep row order. `None`: alpaca's order
    /// (classic two-way: FE2 code; one-way: row order; network: (FE1 code, FE2 code)).
    pub order: Option<&'a [i64]>,
    /// Required with [`Normalization::Logitfe`].
    pub logitfe: Option<LogitfePanel<'a>>,
}

/// Tolerance of every centering of the kernels (relative W-norm, absorber semantics).
pub(crate) const CENTER_TOL: f64 = 1e-13;
pub(crate) const CENTER_MAXITER: usize = 100_000;

pub(crate) fn invalid<T>(m: impl Into<String>) -> Result<T, RfError> {
    Err(RfError::InvalidInput(m.into()))
}

/// Logistic cdf without clamping.
#[inline]
pub(crate) fn plogis(e: f64) -> f64 {
    crate::glm::family::invlogit(e)
}

/// mu, mu', mu'', mu''' of the logit link at eta.
#[inline]
pub(crate) fn derivs(e: f64) -> (f64, f64, f64, f64) {
    let m = plogis(e);
    let d1 = m * (1.0 - m);
    let d2 = d1 * (1.0 - 2.0 * m);
    let d3 = d1 * ((1.0 - 2.0 * m).powi(2) - 2.0 * d1);
    (m, d1, d2, d3)
}

/// Checks the shapes of `d` against the absorber and returns n.
pub(crate) fn check_data(d: &LogitData, abs: &dyn Absorber, coef: &[f64], eta: &[f64]) -> Result<usize, RfError> {
    let n = d.y.len();
    if abs.n() != n {
        return invalid("bias correction: the absorber and the data have different rows");
    }
    if d.weights.is_some_and(|w| w.len() != n) || d.offset.is_some_and(|o| o.len() != n) || d.x.iter().any(|x| x.len() != n) || eta.len() != n {
        return invalid("bias correction: every column must have one value per kept row");
    }
    if coef.len() != d.x.len() {
        return invalid("bias correction: one coefficient per regressor is required");
    }
    if d.x.is_empty() {
        return invalid("bias correction needs at least one regressor");
    }
    if d.fe.is_empty() {
        return invalid("bias correction needs fixed effects");
    }
    for f in &d.fe {
        if f.codes.len() != n || f.codes.iter().any(|&c| c as usize >= f.g) {
            return invalid("bias correction: fixed-effect codes must be dense, one per kept row");
        }
    }
    if d.n_full < n {
        return invalid("bias correction: N* (rows before the fixed-effect drops) is smaller than the estimation sample");
    }
    Ok(n)
}

/// Checks the number of terms against the structure.
pub(crate) fn check_structure(structure: Structure, k: usize) -> Result<(), RfError> {
    match structure {
        Structure::Classic if !(1..=2).contains(&k) => invalid("panel structure \"classic\" expects a one- or two-way fixed effects model"),
        Structure::Network if !(2..=3).contains(&k) => invalid("panel structure \"network\" expects a two- or three-way fixed effects model"),
        _ => Ok(()),
    }
}

/// Kept-order columns residualized on the fixed effects under the weights `w` (kept order). The
/// absorber is left with these weights.
pub(crate) fn center(abs: &mut dyn Absorber, cols: &[&[f64]], w: &[f64], pool: &Pool, ctx: &Ctx) -> Result<Vec<Vec<f64>>, RfError> {
    let order = abs.row_order().to_vec();
    let w_rows: Vec<f64> = order.iter().map(|&k| w[k as usize]).collect();
    abs.set_weights(Some(&w_rows), pool, ctx)?;
    let r = cols.len();
    if r == 0 {
        return Ok(Vec::new());
    }
    let mut block = gather::gather_rows(&order, cols, pool)?;
    let opts = SolveOpts { tol: CENTER_TOL, maxiter: CENTER_MAXITER, want_coefs: false };
    abs.residualize(&mut block, r, &opts, None, pool, ctx)?;
    let n = order.len();
    let mut out = vec![vec![0.0; n]; r];
    for (row, &k) in order.iter().enumerate() {
        for (j, o) in out.iter_mut().enumerate() {
            o[k as usize] = block[row * r + j];
        }
    }
    Ok(out)
}

/// Fixed-effect values of a solution ([`Corrected::fe_coefs`], [`Resolved::fe_coefs`]): per
/// session term (absorb order) its intercept slot, per kept row, as `GlmParts::fe` (every
/// intercept-only term except FE1 has weighted mean 0 under the absorber's current weights, FE1
/// absorbs the rest; their sum per row is eta - offset - x'beta). Call it right after
/// [`analytical`] / [`resolve_fe`], before another kernel changes the absorber's weights (the sum
/// does not depend on them, the split across terms does).
pub fn saved_fe(abs: &dyn Absorber, terms: &[crate::session::Term], fe_coefs: &crate::absorber::FeCoefs) -> Vec<Vec<f64>> {
    let all: Vec<crate::session::Term> = terms.iter().map(|t| crate::session::Term { save: true, ..t.clone() }).collect();
    crate::fe_recovery::saved_fe(abs, &all, fe_coefs, 0.0)
}

/// A' diag(w) A for kept-order columns (p x p row-major), serial in row order.
pub(crate) fn wcross(a: &[Vec<f64>], w: &[f64]) -> Vec<f64> {
    let p = a.len();
    let mut h = vec![0.0; p * p];
    for i in 0..p {
        for j in i..p {
            let s: f64 = a[i].iter().zip(&a[j]).zip(w).map(|((x, y), w)| x * y * w).sum();
            h[i * p + j] = s;
            h[j * p + i] = s;
        }
    }
    h
}

/// Solves A X = B (A p x p symmetric positive definite, B p x m, both row-major) by Cholesky.
pub(crate) fn chol_solve(a: &[f64], p: usize, b: &[f64], m: usize) -> Result<Vec<f64>, RfError> {
    let mut l = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut s = a[i * p + j];
            for k in 0..j {
                s -= l[i * p + k] * l[j * p + k];
            }
            if i == j {
                if s.is_nan() || s <= 0.0 || !s.is_finite() {
                    return invalid("bias correction: the information matrix of the coefficients is not positive definite");
                }
                l[i * p + i] = s.sqrt();
            } else {
                l[i * p + j] = s / l[j * p + j];
            }
        }
    }
    let mut x = b.to_vec();
    for c in 0..m {
        for i in 0..p {
            let mut s = x[i * m + c];
            for k in 0..i {
                s -= l[i * p + k] * x[k * m + c];
            }
            x[i * m + c] = s / l[i * p + i];
        }
        for i in (0..p).rev() {
            let mut s = x[i * m + c];
            for k in i + 1..p {
                s -= l[k * p + i] * x[k * m + c];
            }
            x[i * m + c] = s / l[i * p + i];
        }
    }
    Ok(x)
}

/// Rows of every group of `codes` (ascending row within the group), as offsets and ids.
pub(crate) fn groups(codes: &[u32], g: usize) -> (Vec<usize>, Vec<u32>) {
    let mut off = vec![0usize; g + 1];
    for &c in codes {
        off[c as usize + 1] += 1;
    }
    for i in 0..g {
        off[i + 1] += off[i];
    }
    let mut pos = off.clone();
    let mut ids = vec![0u32; codes.len()];
    for (r, &c) in codes.iter().enumerate() {
        ids[pos[c as usize]] = r as u32;
        pos[c as usize] += 1;
    }
    (off, ids)
}

/// Groups of `codes` with the rows of each group sorted by `key` (stable: ties keep row order).
pub(crate) fn ordered_groups(codes: &[u32], g: usize, key: Option<&dyn Fn(usize) -> i64>) -> (Vec<usize>, Vec<u32>) {
    let (off, mut ids) = groups(codes, g);
    if let Some(key) = key {
        for q in 0..g {
            ids[off[q]..off[q + 1]].sort_by_key(|&r| key(r as usize));
        }
    }
    (off, ids)
}

/// Time-order key of the lag groups under alpaca's rules, over rows with the given term codes:
/// classic two-way: FE2 code; network with three terms: (FE1, FE2) codes; otherwise row order.
pub(crate) fn default_key<'a>(structure: Structure, fe: &'a [FeCodes<'a>]) -> Option<Box<dyn Fn(usize) -> i64 + 'a>> {
    match (structure, fe.len()) {
        (Structure::Classic, 2) => Some(Box::new(move |r| i64::from(fe[1].codes[r]))),
        (Structure::Network, 3) => {
            let g2 = fe[1].g as i64;
            Some(Box::new(move |r| i64::from(fe[0].codes[r]) * g2 + i64::from(fe[1].codes[r])))
        }
        _ => None,
    }
}

/// Index of the lag-group term (FE1 classic, FE3 network with three terms).
pub(crate) fn lag_term(structure: Structure, k: usize) -> Option<usize> {
    match (structure, k) {
        (Structure::Classic, _) => Some(0),
        (Structure::Network, 3) => Some(2),
        _ => None,
    }
}
