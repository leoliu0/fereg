//! `cmd=iv`. Varlist: depvar, exogenous regressors (`nx`), endogenous (`nendog`), excluded
//! instruments (`ninst`), touse, output variables (`nout`).
//!
//! Keys beyond `ols`: `nendog ninst first=0|1 ffirst=0|1`, matrix name `firstm=` (e(first)).
//! Further IV outputs (first-stage regressions, Stock–Yogo locals) are added in Wave 2.

use regfast_core::{IvOptions, Layout, RfError};

use crate::args::Args;
use crate::{Phases, check_nvars, est, io, with_ctx};

pub fn iv(a: &Args) -> Result<(), RfError> {
    let k = est::keys(a, &[("b", "b"), ("V", "V"), ("doft", "dof_table"), ("firstm", "first")])?;
    let lay = Layout { nx: a.num_req("nx")?, nendog: a.num_req("nendog")?, ninst: a.num_req("ninst")? };
    let mut spec = est::spec(a, &k.outputs)?;
    spec.iv = IvOptions { first: a.flag("first", false)?, ffirst: a.flag("ffirst", false)? };
    a.finish()?;
    let r = lay.ncols();
    check_nvars(r + 1 + k.nout)?;
    let mut stored = est::take_session(&k)?;
    let vars: Vec<i32> = (1..=r as i32).collect();

    with_ctx(&k.common, |ctx| {
        let mut ph = Phases::new(ctx.verbose);
        let rows = io::read_session_rows(&stored.session, &stored.obs, &vars, k.mode, !k.common.serialio, ctx)?;
        ph.mark(&format!("iv: read {} x {r}", stored.session.n()));
        let res = stored.session.iv(rows, &lay, &spec, ctx)?;
        ph.mark("iv: estimate");
        est::finish(&res, &k, &stored, r + 1)?;
        ph.mark("iv: store");
        Ok(())
    })
}
