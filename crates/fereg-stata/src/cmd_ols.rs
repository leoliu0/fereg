//! `cmd=ols`. Varlist: depvar, regressors (`nx`), touse, output variables (`nout`).

use fereg_core::{Layout, RfError};

use crate::args::Args;
use crate::{Phases, check_nvars, est, io, with_ctx};

pub fn ols(a: &Args) -> Result<(), RfError> {
    let k = est::keys(a, &[("b", "b"), ("V", "V"), ("doft", "dof_table")])?;
    let nx: usize = a.num_req("nx")?;
    let spec = est::spec(a, &k.outputs)?;
    a.finish()?;
    check_nvars(nx + 2 + k.nout)?;
    let mut stored = est::take_session(&k)?;
    let lay = Layout::ols(nx);
    let vars: Vec<i32> = (1..=lay.ncols() as i32).collect();

    with_ctx(&k.common, |ctx| {
        let mut ph = Phases::new(ctx.verbose);
        let rows = io::read_block(&mut stored, &vars, &k.cached, k.mode, !k.common.serialio, ctx)?;
        ph.mark(&format!("ols: read {} x {}", stored.session.n(), vars.len()));
        let res = stored.session.ols(rows, &lay, &spec, ctx)?;
        ph.mark("ols: estimate");
        est::finish(&res, &k, &stored, nx + 2)?;
        ph.mark("ols: store");
        Ok(())
    })
}
