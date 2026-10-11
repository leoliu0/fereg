//! The boundary between the FE solver and the estimators (plan §1.1b).
//!
//! An [`Absorber`] projects the absorbed fixed effects out of row-major column blocks:
//! ṽ = v − D(D'WD)⁺D'Wv for every column v. Rows are always in [`Absorber::row_order`]
//! (the solver's preferred observation order); estimators never see the internal layout.
//!
//! With no absorb terms (`noabsorb`) the absorber projects out the constant.
//!
//! Implementations: `solve::FeSolver` (fast, plan D5) and [`crate::reference::DenseAbsorber`]
//! (explicit dummies, the test oracle). Any change here updates both.

use crate::api::Ctx;
use crate::error::RfError;
use crate::par::Pool;

/// Options for one [`Absorber::residualize`] call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveOpts {
    /// Relative W-norm accuracy of each residualized column.
    pub tol: f64,
    pub maxiter: usize,
    /// Return the FE coefficients (needed for d and savefe).
    pub want_coefs: bool,
}

#[derive(Clone, Debug)]
pub struct SolveInfo {
    /// Solver iterations (closed-form cases report 1, as reghdfe `e(ic)`).
    pub iterations: usize,
    /// The preconditioner was escalated during this solve.
    pub escalated: bool,
    /// FE coefficients of every column, when requested (also usable as the next warm start).
    pub coefs: Option<FeCoefs>,
}

/// FE coefficients for R columns, in absorber-specific coordinates (`p` rows, row-major p×R).
#[derive(Clone, Debug, PartialEq)]
pub struct FeCoefs {
    pub p: usize,
    pub r: usize,
    pub data: Vec<f64>,
}

impl FeCoefs {
    /// Column `j` as a vector of length `p`.
    pub fn col(&self, j: usize) -> Vec<f64> {
        assert!(j < self.r);
        (0..self.p).map(|i| self.data[i * self.r + j]).collect()
    }

    /// The single column Σ_j c_j·col_j (e.g. α_y − Σ β_j α_xj with c = [1, −β…]).
    pub fn combine(&self, c: &[f64]) -> FeCoefs {
        assert_eq!(c.len(), self.r, "combine: one weight per column");
        let data = self.data.chunks_exact(self.r).map(|row| row.iter().zip(c).map(|(a, b)| a * b).sum()).collect();
        FeCoefs { p: self.p, r: 1, data }
    }
}

pub trait Absorber: Send {
    /// Number of rows (kept observations).
    fn n(&self) -> usize;

    /// Kept-observation index of each row. Every block passed to [`Absorber::residualize`] uses this order.
    fn row_order(&self) -> &[u32];

    /// Current weights in row order (`None` = all ones).
    fn weights(&self) -> Option<&[f64]>;

    /// Replaces the weights (row order, strictly positive). GLM calls this every IRLS iteration;
    /// the structure is reused and only weight-dependent quantities are refreshed.
    fn set_weights(&mut self, w: Option<Vec<f64>>, pool: &Pool, ctx: &Ctx) -> Result<(), RfError>;

    /// Residualizes the N×R row-major block `z` in place. `warm` (same R) seeds the solver.
    fn residualize(
        &mut self,
        z: &mut [f64],
        r: usize,
        o: &SolveOpts,
        warm: Option<&FeCoefs>,
        pool: &Pool,
        ctx: &Ctx,
    ) -> Result<SolveInfo, RfError>;

    /// FE fit of coefficient column `col`, per row (row order).
    fn fit_rows(&self, c: &FeCoefs, col: usize, pool: &Pool) -> Vec<f64>;

    /// Values of coefficient column `col` per session term and slot: for each term in absorb
    /// order, the intercept slot (if any) then one slot per slope, each with G_q level values.
    /// Normalized: every intercept-only term except FE1 has weighted (current weights) mean 0 over
    /// the rows, and FE1 absorbs the shift. FE1 is the intercept-only term with the most levels
    /// (first such term on ties); without one, no normalization is applied.
    fn term_values(&self, c: &FeCoefs, col: usize) -> Vec<Vec<f64>>;
}

/// Index of FE1 among `terms`: the intercept-only term (no slopes) with the most levels, first on ties.
pub fn fe1_index(terms: &[crate::session::Term]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (q, t) in terms.iter().enumerate() {
        if t.intercept && t.slopes.is_empty() && best.is_none_or(|b| t.g > terms[b].g) {
            best = Some(q);
        }
    }
    best
}
