//! Host-neutral estimation layer for the Python and R packages (fixest / pyfixest conventions).
//!
//! Hosts (the Python and R `fereg` packages) parse formulas and build numeric design columns.
//! This module does everything else, identically for both hosts: missing-value, zero-weight,
//! singleton and perfect-fit removal, the fixed-effect structure, estimation (OLS, IV/2SLS,
//! Poisson, logit), collinearity, variance matrices with fixest small-sample corrections, fit
//! statistics, fixed-effect recovery, and the retained state behind post-hoc variance matrices.
//!
//! Conventions follow fixest 0.14.2 ([`Flavor::Fixest`], the R default) or pyfixest 0.60.0
//! ([`Flavor::Pyfixest`], the Python default). The Stata plugin does not use this module; its
//! reghdfe / ppmlhdfe conventions live in `ols`, `iv` and `glm`.

use crate::api::{Ctx, RfError, Separation};
use crate::par;

mod bias;
pub mod dist;
mod design;
mod extra;
mod fixef;
mod glm;
mod input;
mod ols;
mod state;
mod vcov;

pub use bias::{
    ApeAt, ApeOptions, ApeOutput, BiasDims, BiasFit, BiasMethod, BiasOptions, BiasReport, JackknifeOptions, Normalization, Permutation, Sampling, Structure,
    Variant, WhichBias, apes, bias_corr, bias_dims,
};

/// A categorical column over the input rows.
#[derive(Clone, Copy, Debug)]
pub enum FactorCol<'a> {
    /// Numeric values; NaN = missing. Levels are the distinct values in ascending order.
    F64(&'a [f64]),
    /// Integer values; `i32::MIN` (R's `NA_integer_`) = missing. Levels in ascending order.
    I32(&'a [i32]),
    /// Integer values; `i64::MIN` = missing. Levels in ascending order.
    I64(&'a [i64]),
    /// Pre-coded levels (R factor, pandas categorical / `factorize`): row `i` has level
    /// `codes[i] - base` when `base <= codes[i] < base + nlevels`; any other code = missing.
    /// Level order is the host's order; unused levels are allowed.
    Codes { codes: &'a [i32], nlevels: u32, base: i32 },
}

impl FactorCol<'_> {
    /// Number of rows.
    pub fn len(&self) -> usize {
        match self {
            FactorCol::F64(v) => v.len(),
            FactorCol::I32(v) => v.len(),
            FactorCol::I64(v) => v.len(),
            FactorCol::Codes { codes, .. } => codes.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One fixed-effect term: fixest `fe`, `fe1^fe2`, `fe[x]` (intercept + slope), `fe[[x]]` (slope only).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FeTerm {
    /// Indices into [`FitInput::factors`]; more than one = interaction (`a^b`).
    pub factors: Vec<usize>,
    /// `true` for `fe` and `fe[x]`; `false` for the slope-only `fe[[x]]`.
    pub intercept: bool,
    /// Indices into [`FitInput::slopes`] (varying slopes `fe[x1, x2]`).
    pub slopes: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Model {
    /// OLS, or IV/2SLS when [`FitInput::endog`] is not empty.
    #[default]
    Ols,
    Poisson,
    Logit,
}

/// Which package's conventions to reproduce where fixest and pyfixest differ.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Flavor {
    /// fixest 0.14.2: PSD fix of every variance matrix, `K = max(K, k + 1)` clamp, rows with a
    /// missing cluster value, or a missing time / panel / coordinate of the requested
    /// variance matrix, are dropped at estimation.
    #[default]
    Fixest,
    /// pyfixest 0.60.0: no PSD fix, no clamp, FE nesting is also checked against the two-way
    /// intersection, a missing cluster (time, panel, coordinate) value is an error, `fweights`
    /// semantics.
    Pyfixest,
}

/// Variance estimator. Parameters left `None` are resolved by the flavor's default rule and
/// echoed resolved in [`VcovOutput::kind`]. Every kind applies to every model (OLS, IV, Poisson,
/// logit; IV first stages get the same kind) except HC2/HC3 with IV.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum VcovKind {
    #[default]
    Iid,
    /// HC1.
    Hetero,
    /// HC2 / HC3 (scores / sqrt(1 - h), / (1 - h)) with the exact leverage of the full model:
    /// fixed effects and slopes included, GLMs under the IRLS weights. Fixest: no small-sample
    /// correction, an error when a leverage is 1, df_t from K with every fixed effect; `feols`
    /// without fixed effects leaves the weight out of the leverage, as fixest's `hatvalues`.
    /// Pyfixest: N/(N - df_k), frequency weights h/w and scores/sqrt(w); with fixed effects this is
    /// fereg's extension (pyfixest raises). Not defined for IV (each reference's message).
    /// With varying slopes fixest's own `hatvalues` fails (Cholesky of its dummy cross-product)
    /// or, for `feglm`, demeans the regressors by the intercept fixed effects only: fereg uses the
    /// exact leverage there.
    Hc2,
    Hc3,
    /// One- to four-way clustering (Cameron-Gelbach-Miller); dimensions come with the request.
    Cluster,
    /// Leave-one-cluster-out jackknife centered at the full-sample coefficients (pyfixest CRV3)
    /// over the same dimensions, with CRV1's small-sample factors, sign and G rules per subset of
    /// the dimensions (inclusion-exclusion); logit refits are logit. Fixest flavor (extension):
    /// fixest's cluster correction and PSD fix. Fit time only.
    Crv3,
    /// Newey-West (Bartlett): panel when a panel variable is given, time series otherwise.
    /// Fixest: lags over the dense rank of time (pairs within the unit), default floor(T^(1/4))
    /// (panel) or floor(sandwich::bwNeweyWest) (time series). Pyfixest: lags over the rows of a
    /// unit sorted by time, default floor(T^0.25), the time series needs `lag`. Both:
    /// T/(T - 1) * (n - 1)/(n - K), df_t T - 1.
    NeweyWest { lag: Option<usize> },
    /// Driscoll-Kraay over the per-period score sums; default lag floor(T^(1/4)); ssc and df_t
    /// as Newey-West. The panel variable is optional (pyfixest: duplicate check only).
    DriscollKraay { lag: Option<usize> },
    /// Conley spatial HAC (fixest): cutoff and pixel in km; `cutoff: None` = fixest's deduced
    /// cutoff; `spherical` = great-circle distance (fixest `distance = "spherical"`), else fixest's
    /// `"triangular"`. n/(n - K) (pyfixest flavor, an extension: N/(N - df_k)), df_t n - K.
    Conley { cutoff: Option<f64>, pixel: f64, spherical: bool },
}

impl VcovKind {
    /// Clustered (CRV1 or CRV3): needs cluster dimensions and allows fixest's `K.fixef = "nonnested"`.
    pub fn is_clustered(&self) -> bool {
        matches!(self, VcovKind::Cluster | VcovKind::Crv3)
    }
}

/// fixest `ssc(K.fixef=)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KFixef {
    None,
    Full,
    Nonnested,
}

/// fixest `ssc(G.df=)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GDf {
    Min,
    Conventional,
}

/// fixest `ssc(t.df=)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TDf {
    Min,
    Conventional,
    Fixed(f64),
}

/// fixest `ssc()`; [`Default`] is fixest's default `ssc()` (also pyfixest's).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ssc {
    pub k_adj: bool,
    pub k_fixef: KFixef,
    pub k_exact: bool,
    pub g_adj: bool,
    pub g_df: GDf,
    pub t_df: TDf,
}

impl Default for Ssc {
    fn default() -> Self {
        Ssc { k_adj: true, k_fixef: KFixef::Nonnested, k_exact: false, g_adj: true, g_df: GDf::Min, t_df: TDf::Min }
    }
}

/// pyfixest `weights_type` (fixest weights behave as `Aweights`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WeightsType {
    /// Observations count as rows (`nobs` = number of rows).
    #[default]
    Aweights,
    /// Frequency weights: `nobs` = sum w.
    Fweights,
}

/// fixest `fixef.rm` / pyfixest `fixef_rm` (+ `separation_check="fe"`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FixefRm {
    None,
    /// Drop FE singletons (recursively).
    Singleton,
    /// Singletons plus, for GLMs, FE levels whose outcome is constant at a bound (Poisson: all 0;
    /// logit: all 0 or all 1), recursively. Same as `Singleton` for OLS. fixest `perfect_fit`.
    /// [`Flavor::Pyfixest`]: pyfixest's `fe` separation check instead, once after the
    /// singletons, dropping levels whose outcome is all 0 (logit too; all <= 0 for
    /// [`FitSpec::gaussian_glm`]; nothing without fixed effects). For logit the rest of
    /// fixest's rule applies to IRLS only: the rows of levels whose remaining outcomes are all 0
    /// or all 1 stay in the reported sample (pyfixest's nobs and degrees of freedom) at the limit
    /// pyfixest's IRLS approaches: mu at the bound (1 - 100*eps or 100*eps), IRLS weight w*mu(1 - mu), no
    /// contribution to the scores, and fixed effects +/-inf on levels with no other row.
    #[default]
    Perfect,
    /// [`FixefRm::Perfect`] without the singleton removal (pyfixest `fixef_rm="none"` with its
    /// `fe` separation check).
    PerfectOnly,
}

/// Normalization of the recovered fixed effects.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FixefNorm {
    /// fixest `fixef()`: the first FE is free; in each connected component one level of every
    /// other FE (the one on the component's first observation) is 0.
    #[default]
    Fixest,
    /// pyfixest `fixef()`: the first intercept term free, the first level of every other
    /// intercept term pinned at 0 (pyfixest omits it), the remaining indeterminacy (shifts per
    /// connected component) resolved by minimum norm.
    MinNorm,
}

#[derive(Clone, Debug, Default)]
pub struct FitInput<'a> {
    /// Number of input rows; every slice below has this length.
    pub n: usize,
    /// Dependent variable; NaN / +/-inf = missing.
    pub y: &'a [f64],
    /// Exogenous regressors (host-built design columns, no intercept column), in coefficient order.
    pub x: Vec<&'a [f64]>,
    /// Endogenous regressors (IV).
    pub endog: Vec<&'a [f64]>,
    /// Excluded instruments (IV).
    pub inst: Vec<&'a [f64]>,
    /// Distinct categorical columns referenced by [`FitInput::fe`] and [`FitInput::cluster`].
    pub factors: Vec<FactorCol<'a>>,
    /// Data-column name of each factor (nesting-by-name checks).
    pub factor_names: Vec<String>,
    /// Fixed-effect terms in formula order.
    pub fe: Vec<FeTerm>,
    /// Slope variables referenced by [`FeTerm::slopes`]; NaN = missing.
    pub slopes: Vec<&'a [f64]>,
    /// Observation weights: NaN = missing, 0 = row dropped, negative = error.
    pub weights: Option<&'a [f64]>,
    /// GLM offset on the link scale; NaN = missing.
    pub offset: Option<&'a [f64]>,
    /// Cluster dimensions of the variance matrix (factor indices; more than one = interaction).
    pub cluster: Vec<Vec<usize>>,
    /// Time variable of Newey-West / Driscoll-Kraay (factor index; levels ascending = time order).
    pub time: Option<usize>,
    /// Panel (unit) variable of Newey-West (factor index); also the duplicate check of pyfixest's
    /// Driscoll-Kraay.
    pub panel: Option<usize>,
    /// Latitude and longitude of Conley variance matrices, degrees; NaN = missing.
    pub lat: Option<&'a [f64]>,
    pub lon: Option<&'a [f64]>,
    /// Extra rows to drop (missing values the host found while building the design); `true` = drop.
    pub drop: Option<&'a [bool]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FitSpec {
    pub model: Model,
    pub flavor: Flavor,
    /// Formula intercept. Reported (first, as [`CoefSource::Intercept`]) only when no FE term has
    /// an intercept; with FEs there is no intercept, as in fixest and pyfixest.
    pub intercept: bool,
    pub vcov: VcovKind,
    pub ssc: Ssc,
    pub weights_type: WeightsType,
    pub fixef_rm: FixefRm,
    /// Extra GLM separation checks (ppmlhdfe `relu` / `mu`); off in fixest and pyfixest.
    /// [`Flavor::Pyfixest`]: `relu` is pyfixest's `ir` check instead (`separation_check=["ir"]`):
    /// ppmlhdfe's ReLU method on the outcome as a Poisson outcome (logit and
    /// [`FitSpec::gaussian_glm`] too), run once on the sample of the `fe` check, before IRLS,
    /// without singleton re-checks, and only with fixed effects; `mu` is not available.
    pub separation: Separation,
    /// [`Flavor::Pyfixest`] linear model ([`Model::Ols`]) estimated as pyfixest's
    /// `feglm(family="gaussian")`, which runs the separation checks of its GLMs:
    /// [`FixefRm::Perfect`] / [`FixefRm::PerfectOnly`] is pyfixest's `fe` check (levels whose
    /// outcome is all <= 0), and
    /// the degrees of freedom count the levels present before the checks.
    pub gaussian_glm: bool,
    /// Collinearity tolerance (fixest/pyfixest `collin.tol`).
    pub collin_tol: f64,
    /// FE solver tolerance and iteration cap (fereg solver semantics).
    pub fixef_tol: f64,
    pub fixef_maxiter: usize,
    /// GLM deviance tolerance and IRLS iteration cap (fixest `glm.tol`, `glm.iter`).
    pub glm_tol: f64,
    pub glm_maxiter: usize,
    /// Return residuals and fitted values (and GLM `eta`, IRLS weights).
    pub keep_resid: bool,
    /// Return the fixed-effect values (normalized at fit time; [`FitState::fixef`] computes
    /// them on demand instead).
    pub keep_fixef: bool,
    pub fixef_norm: FixefNorm,
    /// Keep the [`FitState`] for post-hoc variance matrices and fixed-effect values (`false` =
    /// fixest/pyfixest `lean`).
    pub keep_state: bool,
}

impl Default for FitSpec {
    fn default() -> Self {
        FitSpec {
            model: Model::Ols,
            flavor: Flavor::Fixest,
            intercept: true,
            vcov: VcovKind::Iid,
            ssc: Ssc::default(),
            weights_type: WeightsType::Aweights,
            fixef_rm: FixefRm::Perfect,
            separation: Separation::default(),
            gaussian_glm: false,
            collin_tol: 1e-9,
            fixef_tol: 1e-8,
            fixef_maxiter: 16_000,
            glm_tol: 1e-8,
            glm_maxiter: 25,
            keep_resid: true,
            keep_fixef: true,
            fixef_norm: FixefNorm::Fixest,
            keep_state: true,
        }
    }
}

/// Where a coefficient comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoefSource {
    Intercept,
    /// [`FitInput::x`]`[j]`.
    X(usize),
    /// [`FitInput::endog`]`[j]`.
    Endog(usize),
    /// [`FitInput::inst`]`[j]` (first stages only).
    Inst(usize),
}

/// A test statistic; NaN fields when not available.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TestStat {
    pub stat: f64,
    pub p: f64,
    pub df1: f64,
    pub df2: f64,
}

impl TestStat {
    pub const NA: TestStat = TestStat { stat: f64::NAN, p: f64::NAN, df1: f64::NAN, df2: f64::NAN };
}

#[derive(Clone, Debug, PartialEq)]
pub struct VcovOutput {
    /// The request with every default resolved: `lag: Some(L)` as the flavor reports it (fixest's
    /// label `(L=...)`), `cutoff: Some(km)`.
    pub kind: VcovKind,
    /// k x k row-major over the coefficient order it was computed for.
    pub vcov: Vec<f64>,
    /// K used by the small-sample correction (fixest `df.K`).
    pub df_k: f64,
    /// Degrees of freedom of the t distribution (fixest `df.t`); `None` = normal (Poisson, logit).
    pub df_t: Option<f64>,
    /// Number of clusters per dimension (empty unless clustered).
    pub n_clusters: Vec<usize>,
    /// Newey-West / Driscoll-Kraay: number of time periods T of the estimation sample.
    pub n_periods: Option<usize>,
    /// Total small-sample multiplier applied.
    pub ssc_factor: f64,
    /// An eigenvalue was clipped (fixest PSD fix).
    pub psd_fixed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FitStats {
    /// sum w*e^2 (GLM: response residuals y - mu).
    pub ssr: f64,
    /// sum w*(y - ybar_w)^2.
    pub ssr_null: f64,
    /// sum w*y~^2 (y net of the fixed effects); NaN for GLMs.
    pub ssr_fe_only: f64,
    /// ssr / (n - nparams); NaN for GLMs.
    pub sigma2: f64,
    /// sqrt(ssr / n), n = sum w when weighted.
    pub rmse: f64,
    /// R^2 family; NaN for GLMs.
    pub r2: f64,
    pub ar2: f64,
    pub wr2: f64,
    pub war2: f64,
    /// fixest `logLik`: OLS -n/2*(1 + ln 2*pi + ln(sum e^2/n)); Poisson sum w(y*eta - mu - ln Gamma(y+1)); logit
    /// sum ln dbinom(round(w*y), round(w), mu) (fixest's binomial with the weights as trials).
    pub loglik: f64,
    /// Log-likelihood of the constant-only model. GLM (fixest `ll_null`): without the offset;
    /// Poisson `sy*ln(sy/N) - sy - sum ln Gamma(y+1)` (sy = sum wy, N = sum w, unweighted factorials), logit
    /// `sy*ln p + (N - sy)*ln(1 - p)` with p = sy/N.
    pub ll_null: f64,
    /// OLS: ssr. GLM: the model deviance.
    pub deviance: f64,
    /// OLS: ssr_null. GLM: deviance of the constant-only model without offset.
    pub null_deviance: f64,
    /// 1 - ll/ll_null.
    pub pseudo_r2: f64,
    /// 1 - (ll - nparams + 1)/ll_null.
    pub apr2: f64,
    /// cor(y, fitted)^2.
    pub sq_cor: f64,
    /// fixest fitstat `f` (OLS).
    pub f_test: TestStat,
    /// FE solver iterations (last solve).
    pub fe_iterations: usize,
    pub irls_iterations: usize,
    pub converged: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IvFirstStage {
    /// [`FitInput::endog`] index of this stage's dependent variable.
    pub endog: usize,
    /// Coefficient order: intercept (when reported), instruments, then exogenous regressors.
    pub coef_source: Vec<CoefSource>,
    pub coef: Vec<f64>,
    /// Same estimator and small-sample correction as the second stage.
    pub vcov: VcovOutput,
    pub collinear: Vec<CoefSource>,
    /// fixest `ivf1`: iid F test of the excluded instruments.
    pub f_test: TestStat,
    /// fixest `ivwald1`: Wald test of the excluded instruments with the stage's vcov, p from
    /// F(df1, df2). Pyfixest flavor: pyfixest's first-stage `wald_test` (pseudo-inverse; p from
    /// chi^2 (df1) on stat*df1; df2 NaN).
    pub wald: TestStat,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IvStats {
    pub first_stage: Vec<IvFirstStage>,
    /// fixest `wh`.
    pub wu_hausman: TestStat,
    /// fixest `sargan` (NaN when exactly identified).
    pub sargan: TestStat,
    /// sum w*(uncorrected second-stage residual)^2 (fixest `r2` for IV uses it).
    pub ssr_iv: f64,
}

/// Fixed-effect values of one coefficient slot of one term.
#[derive(Clone, Debug, PartialEq)]
pub struct FixefValues {
    /// Index into [`FitInput::fe`].
    pub term: usize,
    /// `None` = the term's intercept; `Some(j)` = coefficient on [`FitInput::slopes`]`[j]`.
    pub slope: Option<usize>,
    /// One value per level present in the estimation sample.
    pub values: Vec<f64>,
    /// Input row of the first estimation observation of each level (hosts read labels there).
    pub first_obs: Vec<u32>,
    /// Levels fixed at 0 by the normalization (fixest `attr(, "references")`; slope slots: levels
    /// whose slope is not identified; [`FixefNorm::MinNorm`]: the pinned first level).
    pub references: usize,
}

/// Rows removed before estimation. A row counts once, in the first applicable class.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Dropped {
    /// Missing values (NaN / +/-inf / missing factor codes / `FitInput::drop`).
    pub na: usize,
    pub zero_weight: usize,
    pub singletons: usize,
    /// FE levels with an outcome constant at a bound ([`FixefRm::Perfect`]).
    pub perfect_fit: usize,
    /// ppmlhdfe `relu` / `mu` separation.
    pub separated: usize,
    /// Rows with a missing value per input group (a row may count in several):
    /// y, x, endog, inst, fe (factors and slopes), cluster (every variable of the requested
    /// variance matrix: cluster, time, panel, latitude, longitude; fixest's `vcov` group),
    /// weights, offset, extra drop mask.
    pub na_by: NaCounts,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NaCounts {
    pub y: usize,
    pub x: usize,
    pub endog: usize,
    pub inst: usize,
    pub fe: usize,
    pub cluster: usize,
    pub weights: usize,
    pub offset: usize,
    pub extra: usize,
}

pub struct FitOutput {
    /// Estimation rows.
    pub nobs: usize,
    /// Input rows.
    pub nobs_origin: usize,
    /// sum weights over the estimation rows (= nobs when unweighted).
    pub sumw: f64,
    /// Ascending input-row index of every estimation row. Every per-observation output below
    /// follows this order.
    pub obs: Vec<u32>,
    pub dropped: Dropped,
    /// Coefficient order: intercept (when reported), then the kept `x`, then the kept `endog`.
    pub coef_source: Vec<CoefSource>,
    pub coef: Vec<f64>,
    /// Regressors removed for collinearity, in removal order (fixest `collin.var`).
    pub collinear: Vec<CoefSource>,
    pub vcov: VcovOutput,
    /// fixest `nparams`: kept coefficients + FE parameters (one reference per extra FE).
    pub nparams: usize,
    /// Levels per FE term in the estimation sample.
    pub fe_sizes: Vec<usize>,
    pub stats: FitStats,
    pub iv: Option<IvStats>,
    /// Residuals (IV: fixest's corrected residuals y - X beta - FE with the original endogenous
    /// variables; GLM: response residuals y - mu).
    pub resid: Option<Vec<f64>>,
    /// Fitted values including the fixed effects (IV: fixest's y - second-stage residual, i.e.
    /// with the first-stage fitted endogenous variables; GLM: mu).
    pub fitted: Option<Vec<f64>>,
    /// GLM linear predictor eta (including the offset).
    pub eta: Option<Vec<f64>>,
    /// GLM final IRLS weights (including observation weights).
    pub irls_weights: Option<Vec<f64>>,
    pub fixef: Vec<FixefValues>,
    /// Plain-text messages (convergence, PSD fix, ...); hosts format sample and collinearity notes.
    pub notes: Vec<String>,
    pub state: Option<FitState>,
}

/// Inputs of a post-hoc variance matrix ([`FitState::vcov`]), all over the rows of the original
/// [`FitInput`] (input length): the factor columns that `cluster`, `time` and `panel` index, and
/// the coordinates. A missing value on an estimation row is an error.
#[derive(Clone, Copy, Debug, Default)]
pub struct VcovData<'a> {
    pub factors: &'a [FactorCol<'a>],
    pub factor_names: &'a [String],
    pub cluster: &'a [Vec<usize>],
    pub time: Option<usize>,
    pub panel: Option<usize>,
    pub lat: Option<&'a [f64]>,
    pub lon: Option<&'a [f64]>,
}

/// Retained data for post-hoc variance matrices (fixest `summary(m, vcov=)`, pyfixest
/// `m.vcov()`) and fixed-effect values (fixest / pyfixest `fixef()`).
pub struct FitState {
    inner: state::State,
}

impl FitState {
    /// Variance matrix of the same fit under another estimator or small-sample correction, with
    /// the conventions of the fit's flavor. [`VcovKind::Crv3`] is an error: the jackknife
    /// re-estimates the model (hosts fit again with `vcov = Crv3`).
    pub fn vcov(&self, kind: VcovKind, ssc: &Ssc, data: &VcovData, ctx: &Ctx) -> Result<VcovOutput, RfError> {
        self.inner.vcov(kind, ssc, data, ctx)
    }

    /// Fixed-effect values of the fit under `norm`, as [`FitOutput::fixef`] with
    /// [`FitSpec::keep_fixef`] (empty without fixed-effect terms). Hosts that fit with
    /// `keep_fixef = false` call this on demand (fixest `fixef()`, `predict(newdata)`).
    pub fn fixef(&self, norm: FixefNorm, ctx: &Ctx) -> Result<Vec<FixefValues>, RfError> {
        self.inner.fixef(norm, ctx)
    }

    /// fixest's references (`attr(fixef(), "references")`) with the terms taken in `order` (a
    /// permutation of the term indices; fixest orders the terms of models with varying slopes by
    /// number of coefficients): per term, in term order, the number of its levels set to 0 by
    /// [`FixefNorm::Fixest`] over the intercept terms in that order, 0 for slope-only terms.
    pub fn fixef_references(&self, order: &[usize], ctx: &Ctx) -> Result<Vec<usize>, RfError> {
        self.inner.fixef_references(order, ctx)
    }

    /// Estimation rows.
    pub fn nobs(&self) -> usize {
        self.inner.obs.len()
    }
}

/// Estimates one model.
pub fn fit(input: &FitInput, spec: &FitSpec, ctx: &Ctx) -> Result<FitOutput, RfError> {
    input::validate(input, spec)?;
    par::with_pool(par::threads_for(ctx.threads, input.n), |pool| {
        let sample = input::build(input, spec, ctx, pool)?;
        match spec.model {
            Model::Ols => ols::linear(input, spec, sample, ctx, pool),
            Model::Poisson | Model::Logit => glm::glm(input, spec, sample, ctx, pool),
        }
    })?
}
