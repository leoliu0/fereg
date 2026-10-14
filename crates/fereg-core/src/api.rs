//! Public input/output types shared by every estimator and by the Stata plugin.

pub use crate::error::RfError;

/// Kind of observation weights (Stata `[fw]`, `[aw]`, `[pw]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightKind {
    None,
    Fweight,
    Aweight,
    Pweight,
}

/// Raw values of one factor variable over the input observations.
#[derive(Clone, Copy, Debug)]
pub enum FactorValues<'a> {
    /// Numeric factor. Missing values (NaN) are rejected.
    Num(&'a [f64]),
    /// String factor: observation `i` is `bytes[offsets[i]..offsets[i + 1]]`; `offsets` has n + 1 entries.
    Str { bytes: &'a [u8], offsets: &'a [u32] },
}

impl FactorValues<'_> {
    /// Number of observations.
    pub fn len(&self) -> usize {
        match self {
            FactorValues::Num(v) => v.len(),
            FactorValues::Str { offsets, .. } => offsets.len().saturating_sub(1),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One absorbed term: an interaction of factors, with an optional intercept and optional slopes.
///
/// `i.firm` is `{factors: [firm], intercept: true, slopes: []}`; `firm#c.x` is
/// `{factors: [firm], intercept: false, slopes: [x]}`; `firm##c.x` has both.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbsorbTerm {
    /// Indices into [`Prep::factors`]; more than one means an interaction (`a#b`).
    pub factors: Vec<usize>,
    pub intercept: bool,
    /// Indices into [`Prep::slope_vars`].
    pub slopes: Vec<usize>,
    /// Whether the estimated FE values of this term are returned (savefe / `target=`).
    pub save: bool,
}

/// reghdfe `dofadjustments()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DofOptions {
    pub firstpair: bool,
    pub pairwise: bool,
    pub clusters: bool,
    pub continuous: bool,
}

impl Default for DofOptions {
    fn default() -> Self {
        DofOptions { firstpair: false, pairwise: true, clusters: true, continuous: true }
    }
}

impl DofOptions {
    /// `dofadjustments(none)`.
    pub fn none() -> Self {
        DofOptions { firstpair: false, pairwise: false, clusters: false, continuous: false }
    }
}

/// Variance estimator of the Stata layer (reghdfe / ivreghdfe / ppmlhdfe conventions). Each
/// cluster dimension is an interaction of factor indices into [`Prep::factors`].
#[derive(Clone, Debug, PartialEq)]
pub enum Vce {
    Unadjusted,
    Robust,
    Cluster(Vec<Vec<usize>>),
    /// Stata `regress` `vce(hc2)` / `vce(hc3)` with the leverage of the full design.
    Hc2,
    Hc3,
    /// Leave-one-cluster-out jackknife over the dimensions (inclusion–exclusion when several).
    Crv3(Vec<Vec<usize>>),
    /// reghdfe `vce(dkraay [bw])`; `bw` = lags + 1 (None = reghdfe's default).
    Dkraay { time: TimeVars, bw: Option<usize> },
    /// ivreg2 kernel family (ivreghdfe `bw() kernel() kiefer dkraay() psd0 psda`).
    Kernel(KernelVce),
    /// fixest Conley: coordinates are [`Prep::slope_vars`] entries `lat`, `lon` (degrees).
    Conley(ConleyVce),
}

impl Vce {
    /// Dimensions treated as clusters by the DoF rules (reghdfe `clusters`) and the cluster
    /// counts: the cluster dims, the time variable of `dkraay`, the clusters of a kernel VCE.
    pub fn cluster_dims(&self) -> Vec<Vec<usize>> {
        match self {
            Vce::Cluster(d) | Vce::Crv3(d) => d.clone(),
            Vce::Dkraay { time, .. } => vec![vec![time.time]],
            Vce::Kernel(k) => k.cluster.clone(),
            _ => Vec::new(),
        }
    }

    /// The tsset/xtset variables the estimator needs.
    pub fn time_vars(&self) -> Option<&TimeVars> {
        match self {
            Vce::Dkraay { time, .. } => Some(time),
            Vce::Kernel(k) => Some(&k.time),
            _ => None,
        }
    }
}

/// tsset/xtset variables: factor indices of the (numeric) time and panel variables, and the
/// time step `_dta[_TSdelta]`.
#[derive(Clone, Debug, PartialEq)]
pub struct TimeVars {
    pub time: usize,
    pub panel: Option<usize>,
    pub delta: f64,
}

/// ivreg2 bandwidth: a positive real, or `auto` (Newey–West 1994, time series only).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bandwidth {
    Fixed(f64),
    Auto,
}

/// ivreg2 `psd0` / `psda`: make S positive semidefinite (negative eigenvalues set to 0 or to
/// their absolute value).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Psd {
    #[default]
    None,
    Zero,
    Abs,
}

/// ivreg2 kernel-robust variance. Without clusters: HAC (`robust`) or AC (homoskedastic
/// autocorrelation); clusters `[time]`: Driscoll–Kraay; `[panel], [time]` (either order):
/// Thompson two-way with kernel. `kiefer`: Truncated kernel, bw = T, AC.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelVce {
    pub kernel: crate::hac::Kernel,
    /// Ignored with `kiefer` (bw = T).
    pub bw: Bandwidth,
    pub robust: bool,
    pub cluster: Vec<Vec<usize>>,
    pub kiefer: bool,
    pub psd: Psd,
    pub time: TimeVars,
}

/// fixest Conley options; `lat`/`lon` index [`Prep::slope_vars`].
#[derive(Clone, Debug, PartialEq)]
pub struct ConleyVce {
    pub lat: usize,
    pub lon: usize,
    /// km; None = fixest's deduced cutoff.
    pub cutoff: Option<f64>,
    pub pixel: f64,
    pub spherical: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Gaussian,
    Poisson,
    Logit,
}

/// Everything needed to fix the estimation sample and build the FE structure.
#[derive(Clone, Debug)]
pub struct Prep<'a> {
    /// Number of input observations; every slice below has this length.
    pub n: usize,
    /// Factor variables referenced by absorb terms and cluster dimensions.
    pub factors: Vec<FactorValues<'a>>,
    /// Absorbed terms in user order. Empty = `noabsorb` (only the constant is absorbed).
    pub absorb: Vec<AbsorbTerm>,
    /// Slope variables referenced by [`AbsorbTerm::slopes`].
    pub slope_vars: Vec<&'a [f64]>,
    /// Raw weights (strictly positive, finite).
    pub weights: Option<&'a [f64]>,
    pub weight_kind: WeightKind,
    pub vce: Vce,
    pub drop_singletons: bool,
    pub family: Family,
    /// Depvar, needed by the GLM `fe` separation rule.
    pub y_for_sep: Option<&'a [f64]>,
    /// Apply the GLM `fe` separation rule (Poisson: sum y = 0; logit: sum y in {0, count}) inside the singleton loop.
    pub sep_fe: bool,
}

/// Execution context. The callbacks are deliberately not `Sync`: they may only be called on the
/// orchestration (Stata) thread, never inside a parallel kernel (plan D4).
pub struct Ctx<'a> {
    /// Worker threads; 0 = all logical CPUs.
    pub threads: usize,
    /// 0 quiet, 1 summary, 2 per-phase timings.
    pub verbose: u8,
    /// Polled between kernels and iterations; `true` aborts with [`RfError::Interrupted`].
    pub interrupt: &'a dyn Fn() -> bool,
    /// Real-time log sink (iteration logs, verbose output): one line per call, without a
    /// trailing newline (the plugin appends it).
    pub log: &'a dyn Fn(&str),
}

fn never_interrupt() -> bool {
    false
}

fn discard_log(_: &str) {}

impl Ctx<'static> {
    /// Context for tests and library users: no interrupts, no logging.
    pub fn quiet(threads: usize) -> Ctx<'static> {
        Ctx { threads, verbose: 0, interrupt: &never_interrupt, log: &discard_log }
    }
}

impl Ctx<'_> {
    /// Returns `Err(Interrupted)` when the user pressed Break.
    pub fn check(&self) -> Result<(), RfError> {
        if (self.interrupt)() { Err(RfError::Interrupted) } else { Ok(()) }
    }

    pub fn log(&self, msg: &str) {
        (self.log)(msg)
    }
}

/// Column layout of the row-major data block handed to estimators:
/// `[y, x_1..x_nx, endog_1..endog_nendog, inst_1..inst_ninst]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub nx: usize,
    pub nendog: usize,
    pub ninst: usize,
}

impl Layout {
    pub fn ols(nx: usize) -> Self {
        Layout { nx, nendog: 0, ninst: 0 }
    }

    /// Number of columns R.
    pub fn ncols(&self) -> usize {
        1 + self.nx + self.nendog + self.ninst
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IvOptions {
    pub first: bool,
    pub ffirst: bool,
}

/// Estimation options shared by OLS and IV (GLM adds [`GlmOptions`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Spec {
    /// FE solver tolerance (reghdfe `tolerance()`, default 1e-8).
    pub tol: f64,
    /// FE solver iteration cap (reghdfe `iterate()`, default 16000).
    pub maxiter: usize,
    pub dof: DofOptions,
    pub noconstant: bool,
    pub want_resid: bool,
    pub want_d: bool,
    /// Return values for terms with `save = true`.
    pub want_fe: bool,
    pub want_groupvar: bool,
    pub iv: IvOptions,
}

impl Default for Spec {
    fn default() -> Self {
        Spec {
            tol: 1e-8,
            maxiter: 16_000,
            dof: DofOptions::default(),
            noconstant: false,
            want_resid: false,
            want_d: false,
            want_fe: false,
            want_groupvar: false,
            iv: IvOptions::default(),
        }
    }
}

/// Initial values for Poisson IRLS (ppmlhdfe `guess()`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Guess {
    Simple,
    Ols,
}

/// Separation checks run during GLM estimation (the `fe` rule lives in [`Prep::sep_fe`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Separation {
    /// ReLU method (ppmlhdfe `relu`; `simplex` is an alias).
    pub relu: bool,
    /// IRLS-step detection (ppmlhdfe `mu`).
    pub mu: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlmOptions {
    pub family: Family,
    /// Deviance convergence tolerance (default 1e-8).
    pub tol: f64,
    /// IRLS iteration cap (default 10000).
    pub maxiter: usize,
    /// Final inner (FE solver) tolerance tau_final (default 1e-10).
    pub itol: f64,
    pub separation: Separation,
    pub guess: Guess,
    /// Print the iteration log through [`Ctx::log`].
    pub log: bool,
    /// How collinear regressors are found.
    pub collinear: GlmCollinear,
}

/// Collinearity rule of a GLM fit. The fixest and pyfixest rules apply fixest's
/// `cpp_cholesky` (a column is excluded when its pivot is below `tol`, absolute) to the
/// regressors residualized with the first IRLS weights, on their original scale. The first FE
/// solve runs at its usual (loose) tolerance and is repeated at the final inner tolerance only
/// when its error bound could change the outcome. Excluded columns stay out of the fit.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum GlmCollinear {
    /// ppmlhdfe `remove_collinears`: a solve of X with the true weights before IRLS.
    #[default]
    Ppmlhdfe,
    /// fixest `feglm`: the IRLS-weighted cross-product X~'WX~ (fixest re-checks in every
    /// iteration with its current weights and reports the last iteration's set).
    Fixest { tol: f64 },
    /// pyfixest `fepois`/`feglm`: the unweighted cross-product X~'X~ of the first iteration.
    Pyfixest { tol: f64 },
}

/// Conventions of a GLM fit ([`crate::Session::glm_parts`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GlmConvention {
    /// ppmlhdfe 2.3.3 / Stata `logit` (the Stata plugin): Poisson and pweighted VCE forced to
    /// robust; stop when ppmlhdfe's eps = |Delta dev| / max(min(dev, dev_old), 0.1) < `tol` with the
    /// inner tolerance at its final value; reaching `maxiter` is error 430.
    #[default]
    Ppmlhdfe,
    /// fixest / pyfixest: the VCE is the session's as requested; fixest's starting values
    /// (Poisson mu = y + 0.1, logit mu = (w*y + 1/2)/(w + 1)), so `guess` is ignored; fixest's step
    /// halving for both families (the first iteration compares with the deviance of eta = 1e-5 and
    /// halves toward it at most twice); stop when |Delta dev| / (0.1 + |dev|) < `tol` (fixest
    /// `glm.tol`, deviance on the original scale) in an iteration without step halving whose FE
    /// solve was exact or ran at the final inner tolerance max(min(`itol`, `tol`/10), 1e-12); the
    /// inner tolerance follows eps^{3/2} (eps = relative deviance change) instead of ppmlhdfe's
    /// schedule. `maxiter` is the IRLS cap (fixest `glm.iter`, 25); reaching it is a note in
    /// [`EstResult::notes`] with [`GlmParts::converged`] = false (scalar `converged` = 0), not an
    /// error. Iteration logs are not printed (SMCL text). Separation checks run only as set in
    /// [`GlmOptions::separation`].
    Fixest,
}

/// Per-observation pieces of a GLM fit, for host-side inference (fixest/pyfixest variance,
/// predict, fixef). Every vector is in **kept order** after the GLM-internal drops (separation):
/// [`crate::Session::kept_index`] is the final sample. Everything is on the original scale of y
/// and of the regressors.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GlmParts {
    /// One coefficient per regressor sent (layout order); NaN when `status` != 0.
    pub coef: Vec<f64>,
    /// Per regressor sent: 0 ok, 2 absorbed by the fixed effects, 3 collinear.
    pub status: Vec<u8>,
    /// ppmlhdfe constant: regression through the means under the final IRLS weights, so that
    /// eta ~= x'beta + cons + offset when there are no FE terms (it is then the intercept). With FE
    /// terms it is only a normalization, included in `fe_sum`.
    pub cons: f64,
    /// Linear predictor eta (offset included), from the final beta.
    pub eta: Vec<f64>,
    /// Mean mu = g^-1 (eta).
    pub mu: Vec<f64>,
    /// IRLS weights of the final iteration: observation weight x mu (Poisson) or x mu(1 - mu)
    /// (logit), mu taken at the start of that iteration. `xt` and `working_resid` were computed
    /// under these weights, so inv(xt'*diag(w)*xt) is the inverse Hessian of the slopes (fixest
    /// `cov.iid`).
    pub w: Vec<f64>,
    /// Number of columns of `xt` = regressors with status 0, in layout order.
    pub k: usize,
    /// Row-major nobs x `k`: kept regressors residualized on the FE structure under `w` (with no
    /// FE terms the structure is the constant, so the columns are `w`-demeaned).
    pub xt: Vec<f64>,
    /// Working residuals z~ - x~'beta of the final weighted least squares (fixest
    /// `working_residuals`); fixest's GLM scores are `working_resid * w * xt`.
    pub working_resid: Vec<f64>,
    /// eta - offset - x'beta per observation: the sum of the fixed effects, constant included.
    pub fe_sum: Vec<f64>,
    /// For each saved term (absorb order): the intercept slot (if any), then one slot per slope;
    /// values per observation. When every term is saved, sum intercepts + sum slope*x = `fe_sum`.
    pub fe: Vec<Vec<f64>>,
    pub deviance: f64,
    /// IRLS iterations.
    pub iterations: usize,
    pub converged: bool,
}

/// Result of [`crate::Session::glm_separation`] (ppmlhdfe `tagsep()` / `zvarname()`): the
/// separation check (the `fe` rule unless prepare applied it, and ReLU rounds) on the current
/// sample, without estimation. Vectors are in kept order over the session sample at the call
/// (the session is left on the sample without the separated observations).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SeparationCheck {
    pub separated: Vec<bool>,
    /// Certificate of separation: positive on separated observations, 0 elsewhere. Each value
    /// comes from the step that separated the observation (the ReLU prediction, or 1 for the
    /// `fe` rule: the group indicator).
    pub certificate: Vec<f64>,
    pub num_separated: usize,
    /// False when the check stopped without certifying its result.
    pub complete: bool,
    pub iterations: usize,
}

/// A named dense matrix (row-major).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NamedMat {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

/// The single result type of every estimator. Missing statistics are `f64::NAN`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EstResult {
    /// Named exactly as the Stata e() scalar; the plugin stores each as `__rf_<name>`.
    pub scalars: Vec<(String, f64)>,
    /// `b` (1xK), `V` (K x K), `dof_table`, `first`, `bfirst1`, ...
    pub matrices: Vec<NamedMat>,
    pub macros: Vec<(String, String)>,
    /// One entry per non-y column of the [`Layout`]: 0 ok, 2 absorbed by the FEs, 3 collinear.
    pub status: Vec<u8>,
    /// Messages for the user, printed in order by the plugin.
    pub notes: Vec<String>,
    /// Per kept observation (kept order).
    pub resid: Option<Vec<f64>>,
    /// Sum of the fixed effects per kept observation (kept order).
    pub d: Option<Vec<f64>>,
    /// Mobility-group id per kept observation (kept order).
    pub groupvar: Option<Vec<f64>>,
    /// For each saved term in absorb order: the intercept slot (if any), then one slot per slope;
    /// values per kept observation (kept order).
    pub fe: Vec<Vec<f64>>,
}

impl EstResult {
    pub fn scalar(&self, name: &str) -> Option<f64> {
        self.scalars.iter().find(|(k, _)| k == name).map(|(_, v)| *v)
    }

    /// Sets (or overwrites) a scalar.
    pub fn set_scalar(&mut self, name: &str, value: f64) {
        match self.scalars.iter_mut().find(|(k, _)| k == name) {
            Some(slot) => slot.1 = value,
            None => self.scalars.push((name.to_string(), value)),
        }
    }

    pub fn matrix(&self, name: &str) -> Option<&NamedMat> {
        self.matrices.iter().find(|m| m.name == name)
    }

    pub fn macro_value(&self, name: &str) -> Option<&str> {
        self.macros.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
}
