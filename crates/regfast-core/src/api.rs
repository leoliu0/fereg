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

/// Variance estimator. Each cluster dimension is an interaction of factor indices into [`Prep::factors`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Vce {
    Unadjusted,
    Robust,
    Cluster(Vec<Vec<usize>>),
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
    /// Apply the GLM `fe` separation rule (Poisson: Σy = 0; logit: Σy ∈ {0, count}) inside the singleton loop.
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
    /// Real-time log sink (iteration logs, verbose output).
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
    /// Final inner (FE solver) tolerance τ_final (default 1e-10).
    pub itol: f64,
    pub separation: Separation,
    pub guess: Guess,
    /// Print the iteration log through [`Ctx::log`].
    pub log: bool,
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
    /// `b` (1×K), `V` (K×K), `dof_table`, `first`, `bfirst1`, …
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
