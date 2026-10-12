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

pub mod dist;
mod design;
mod fixef;
mod glm;
mod input;
mod ols;
mod state;
mod vcov;

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
    /// missing cluster value are dropped at estimation.
    #[default]
    Fixest,
    /// pyfixest 0.60.0: no PSD fix, no clamp, FE nesting is also checked against the two-way
    /// intersection, a missing cluster value is an error, `fweights` semantics.
    Pyfixest,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VcovKind {
    #[default]
    Iid,
    /// HC1.
    Hetero,
    /// One- to four-way clustering (Cameron-Gelbach-Miller); dimensions come with the request.
    Cluster,
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
    /// Frequency weights: `nobs` = Σw.
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
    /// singletons, dropping levels whose outcome is all 0 (logit too).
    #[default]
    Perfect,
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
    /// Dependent variable; NaN / ±inf = missing.
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
    pub separation: Separation,
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
    /// Return the fixed-effect values.
    pub keep_fixef: bool,
    pub fixef_norm: FixefNorm,
    /// Keep the [`FitState`] for post-hoc variance matrices (`false` = fixest/pyfixest `lean`).
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
    pub kind: VcovKind,
    /// k × k row-major over the coefficient order it was computed for.
    pub vcov: Vec<f64>,
    /// K used by the small-sample correction (fixest `df.K`).
    pub df_k: f64,
    /// Degrees of freedom of the t distribution (fixest `df.t`); `None` = normal (Poisson, logit).
    pub df_t: Option<f64>,
    /// Number of clusters per dimension (empty unless clustered).
    pub n_clusters: Vec<usize>,
    /// Total small-sample multiplier applied.
    pub ssc_factor: f64,
    /// An eigenvalue was clipped (fixest PSD fix).
    pub psd_fixed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FitStats {
    /// Σw·e² (GLM: response residuals y − μ).
    pub ssr: f64,
    /// Σw·(y − ȳ_w)².
    pub ssr_null: f64,
    /// Σw·ỹ² (y net of the fixed effects); NaN for GLMs.
    pub ssr_fe_only: f64,
    /// ssr / (n − nparams); NaN for GLMs.
    pub sigma2: f64,
    /// sqrt(ssr / n), n = Σw when weighted.
    pub rmse: f64,
    /// R² family; NaN for GLMs.
    pub r2: f64,
    pub ar2: f64,
    pub wr2: f64,
    pub war2: f64,
    /// fixest `logLik`: OLS −n/2·(1 + ln 2π + ln(Σe²/n)); Poisson Σw(y·η − μ − lnΓ(y+1)); logit
    /// Σ ln dbinom(round(w·y), round(w), μ) (fixest's binomial with the weights as trials).
    pub loglik: f64,
    /// Log-likelihood of the constant-only model. GLM (fixest `ll_null`): without the offset;
    /// Poisson `sy·ln(sy/N) − sy − Σ lnΓ(y+1)` (sy = Σwy, N = Σw, unweighted factorials), logit
    /// `sy·ln p + (N − sy)·ln(1 − p)` with p = sy/N.
    pub ll_null: f64,
    /// OLS: ssr. GLM: the model deviance.
    pub deviance: f64,
    /// OLS: ssr_null. GLM: deviance of the constant-only model without offset.
    pub null_deviance: f64,
    /// 1 − ll/ll_null.
    pub pseudo_r2: f64,
    /// 1 − (ll − nparams + 1)/ll_null.
    pub apr2: f64,
    /// cor(y, fitted)².
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
    /// χ²(df1) on stat·df1; df2 NaN).
    pub wald: TestStat,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IvStats {
    pub first_stage: Vec<IvFirstStage>,
    /// fixest `wh`.
    pub wu_hausman: TestStat,
    /// fixest `sargan` (NaN when exactly identified).
    pub sargan: TestStat,
    /// Σw·(uncorrected second-stage residual)² (fixest `r2` for IV uses it).
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
    /// Missing values (NaN / ±inf / missing factor codes / `FitInput::drop`).
    pub na: usize,
    pub zero_weight: usize,
    pub singletons: usize,
    /// FE levels with an outcome constant at a bound ([`FixefRm::Perfect`]).
    pub perfect_fit: usize,
    /// ppmlhdfe `relu` / `mu` separation.
    pub separated: usize,
    /// Rows with a missing value per input group (a row may count in several):
    /// y, x, endog, inst, fe (factors and slopes), cluster, weights, offset, extra drop mask.
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
    /// Σ weights over the estimation rows (= nobs when unweighted).
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
    /// Residuals (IV: fixest's corrected residuals y − Xβ − FE with the original endogenous
    /// variables; GLM: response residuals y − μ).
    pub resid: Option<Vec<f64>>,
    /// Fitted values including the fixed effects (IV: fixest's y − second-stage residual, i.e.
    /// with the first-stage fitted endogenous variables; GLM: μ).
    pub fitted: Option<Vec<f64>>,
    /// GLM linear predictor η (including the offset).
    pub eta: Option<Vec<f64>>,
    /// GLM final IRLS weights (including observation weights).
    pub irls_weights: Option<Vec<f64>>,
    pub fixef: Vec<FixefValues>,
    /// Plain-text messages (convergence, PSD fix, ...); hosts format sample and collinearity notes.
    pub notes: Vec<String>,
    pub state: Option<FitState>,
}

/// Retained data for post-hoc variance matrices (fixest `summary(m, vcov=)`, pyfixest `m.vcov()`).
pub struct FitState {
    inner: state::State,
}

impl FitState {
    /// Variance matrix of the same fit under another estimator or small-sample correction.
    /// `factors` are input-length columns (the rows of the original [`FitInput`]); `cluster`
    /// indexes into them. A missing cluster value on an estimation row is an error.
    pub fn vcov(
        &self,
        kind: VcovKind,
        ssc: &Ssc,
        factors: &[FactorCol],
        factor_names: &[String],
        cluster: &[Vec<usize>],
        ctx: &Ctx,
    ) -> Result<VcovOutput, RfError> {
        self.inner.vcov(kind, ssc, factors, factor_names, cluster, ctx)
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
