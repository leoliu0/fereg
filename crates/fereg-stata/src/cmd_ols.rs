//! `cmd=ols`. Varlist: depvar, regressors (`nx`), touse, output variables (`nout`).

use fereg_core::{Layout, RfError};

use crate::args::Args;
use crate::{Phases, check_nvars, est, io, wb, with_ctx};

pub fn ols(a: &Args) -> Result<(), RfError> {
    let k = est::keys(a, &[("b", "b"), ("V", "V"), ("doft", "dof_table")])?;
    let nx: usize = a.num_req("nx")?;
    let spec = est::spec(a, &k.outputs)?;
    let wb_req = wb::parse(a)?;
    a.finish()?;
    check_nvars(nx + 2 + k.nout)?;
    let mut stored = est::take_session(&k)?;
    let lay = Layout::ols(nx);
    let vars: Vec<i32> = (1..=lay.ncols() as i32).collect();

    with_ctx(&k.common, |ctx| {
        let mut ph = Phases::new(ctx.verbose);
        let rows = io::read_block(&mut stored, &vars, &k.cached, k.mode, !k.common.serialio, ctx)?;
        ph.mark(&format!("ols: read {} x {}", stored.session.n(), vars.len()));
        let (res, wb_out) = match &wb_req {
            Some(req) => {
                let (r, w) = stored.session.ols_with_wildboot(rows, &lay, &spec, &req.hyp, &req.spec, ctx)?;
                (r, Some(w))
            }
            None => (stored.session.ols(rows, &lay, &spec, ctx)?, None),
        };
        ph.mark("ols: estimate");
        est::finish(&res, &k, &stored, nx + 2)?;
        if let (Some(req), Some(w)) = (&wb_req, &wb_out) {
            wb::store_extra(w, req)?;
        }
        ph.mark("ols: store");
        Ok(())
    })
}
