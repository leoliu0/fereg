//! FE solver (plan D5, Phase 2): `FeSolver` implements [`crate::absorber::Absorber`].

use crate::absorber::Absorber;
use crate::api::Ctx;
use crate::error::RfError;
use crate::par::Pool;
use crate::session::Term;

/// Builds the production absorber for the kept sample (`weights` in kept order).
pub fn build(
    terms: &[Term],
    weights: Option<&[f64]>,
    n: usize,
    pool: &Pool,
    ctx: &Ctx,
) -> Result<Box<dyn Absorber>, RfError> {
    let _ = (terms, weights, n, pool, ctx);
    Err(RfError::Internal("fast absorber not built yet (use the dense absorber)".into()))
}
