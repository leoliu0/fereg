//! IV/2SLS with ivreghdfe 1.1.4 conventions (plan Phase 5).

use crate::api::{Ctx, EstResult, Layout, Spec};
use crate::error::RfError;
use crate::session::Session;

impl Session {
    /// 2SLS on `rows` (layout `[y, exog..., endog..., inst...]`).
    pub fn iv(&mut self, rows: Vec<f64>, lay: &Layout, s: &Spec, ctx: &Ctx) -> Result<EstResult, RfError> {
        let _ = (rows, lay, s, ctx);
        Err(RfError::Internal("iv: not implemented yet".into()))
    }
}
