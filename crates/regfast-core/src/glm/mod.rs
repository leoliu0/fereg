//! GLM estimation by IRLS: Poisson PML (ppmlhdfe 2.3.3) and logit (plan Phase 6).

pub mod family;
pub mod irls;
pub mod separation;

use crate::api::{Ctx, EstResult, GlmOptions, Layout, Spec};
use crate::error::RfError;
use crate::session::Session;

impl Session {
    /// GLM on `rows` (layout `[y, x...]`), `offset` in row order.
    pub fn glm(
        &mut self,
        rows: Vec<f64>,
        lay: &Layout,
        offset: Option<Vec<f64>>,
        s: &Spec,
        g: &GlmOptions,
        ctx: &Ctx,
    ) -> Result<EstResult, RfError> {
        let _ = (rows, lay, offset, s, g, ctx);
        Err(RfError::Internal("glm: not implemented yet".into()))
    }
}
