//! `cmd=iv`. Varlist: depvar, exogenous regressors (`nx`), endogenous (`nendog`), excluded
//! instruments (`ninst`), touse, output variables (`nout`).
//!
//! Keys beyond `ols`: `nendog ninst first=0|1 ffirst=0|1`; always-filled matrices `S=` `W=`
//! ((ninst+nx)², instruments sent [inst, exog]); optional matrices `firstm=` (21×nendog),
//! `ccev=` `cdev=` (1×nendog), filled only when the estimator produced them (K1 > 0 after
//! collinearity drops; e(first) needs first or ffirst); first-stage prefixes `bfirst=P`
//! `Vfirst=P` `sfirst=P`: for every kept endogenous regressor # (position among those sent),
//! `P#` receives its first-stage coefficients (1×(ninst+nx)), VCE ((ninst+nx)²) and
//! (rmse, df_r, df_m, N) (1×4). Stock–Yogo critical values come back as locals `_rf_sy_<table>`.

use regfast_core::{EstResult, IvOptions, Layout, RfError};

use crate::args::Args;
use crate::{Phases, check_nvars, est, io, with_ctx};

/// Optional matrices: (key, EstResult matrix name).
const OPTIONAL: [(&str, &str); 3] = [("firstm", "first"), ("ccev", "ccev"), ("cdev", "cdev")];
/// First-stage prefixes: (key, EstResult matrix-name stem).
const FIRST_EQ: [(&str, &str); 3] = [("bfirst", "bfirst"), ("Vfirst", "Vfirst"), ("sfirst", "sfirst")];

pub fn iv(a: &Args) -> Result<(), RfError> {
    let k = est::keys(a, &[("b", "b"), ("V", "V"), ("doft", "dof_table"), ("S", "S"), ("W", "W")])?;
    let lay = Layout { nx: a.num_req("nx")?, nendog: a.num_req("nendog")?, ninst: a.num_req("ninst")? };
    let mut spec = est::spec(a, &k.outputs)?;
    spec.iv = IvOptions { first: a.flag("first", false)?, ffirst: a.flag("ffirst", false)? };
    let optional: Vec<(String, &str)> =
        OPTIONAL.iter().filter_map(|(key, name)| a.str_opt(key).map(|s| (s.to_string(), *name))).collect();
    let first_eq: Vec<(String, &str)> =
        FIRST_EQ.iter().filter_map(|(key, stem)| a.str_opt(key).map(|s| (s.to_string(), *stem))).collect();
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
        store_optional(&res, &optional, &first_eq, lay.nendog)?;
        ph.mark("iv: store");
        Ok(())
    })
}

fn store_optional(res: &EstResult, optional: &[(String, &str)], first_eq: &[(String, &str)], nendog: usize) -> Result<(), RfError> {
    for (stata, name) in optional {
        if let Some(m) = res.matrix(name) {
            io::store_matrix(stata, m.rows, m.cols, &m.data)?;
        }
    }
    for (prefix, stem) in first_eq {
        for i in 1..=nendog {
            if let Some(m) = res.matrix(&format!("{stem}{i}")) {
                io::store_matrix(&format!("{prefix}{i}"), m.rows, m.cols, &m.data)?;
            }
        }
    }
    Ok(())
}
