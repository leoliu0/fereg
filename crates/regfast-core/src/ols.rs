//! OLS with reghdfe 6.13.1 conventions (plan Phase 3).

use crate::api::{Ctx, EstResult, Layout, Spec};
use crate::error::RfError;
use crate::session::Session;

impl Session {
    /// OLS on `rows` (N×R row-major in [`Session::row_order`], layout `[y, x...]`).
    pub fn ols(&mut self, rows: Vec<f64>, lay: &Layout, s: &Spec, ctx: &Ctx) -> Result<EstResult, RfError> {
        let _ = (rows, lay, s, ctx);
        Err(RfError::Internal("ols: not implemented yet".into()))
    }
}
