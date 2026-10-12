//! `cmd=glm` (feppml / felogit). Varlist: depvar, regressors (`nx`), offset (iff
//! `offset=1`), touse, output variables (`nout`).
//!
//! Keys beyond the shared estimation keys (`token nout d fe groupvar b V doft threads verbose
//! serialio rowread`): `family=poisson|logit` (required), `nx`, `tol` (IRLS deviance tolerance,
//! default 1e-8), `maxiter` (IRLS iterations, default 10000), `itol` (target inner tolerance,
//! ppmlhdfe `itolerance()`, default 1e-9), `separation=<comma list of relu,simplex,mu>|none`
//! (default `relu` for Poisson, `none` for logit; `fe` is a prepare key), `guess=simple|ols`,
//! `log=0|1`, `offset=0|1`, `dof`. There is no residual output (`resid=` is rejected).
//! See `PROTOCOL.md`.

use fereg_core::{Family, GlmOptions, Guess, Layout, RfError, Separation, Spec};

use crate::args::{self, Args};
use crate::{Phases, check_nvars, est, io, with_ctx};

fn parse_separation(s: &str) -> Result<Separation, RfError> {
    let mut sep = Separation::default();
    if s == "none" {
        return Ok(sep);
    }
    for t in s.split(',') {
        match t {
            // ppmlhdfe's simplex method is implemented as an alias of relu.
            "relu" | "simplex" => sep.relu = true,
            "mu" => sep.mu = true,
            _ => return Err(RfError::InvalidInput(format!("fereg plugin: invalid separation item `{t}`"))),
        }
    }
    Ok(sep)
}

pub fn glm(a: &Args) -> Result<(), RfError> {
    let k = est::keys(a, &[("b", "b"), ("V", "V"), ("doft", "dof_table")])?;
    if k.outputs.resid.is_some() {
        return Err(RfError::InvalidInput("fereg plugin: cmd=glm has no residual output (use d and predict)".into()));
    }
    let nx: usize = a.num_req("nx")?;
    let family = args::parse_family(a.str_req("family")?)?;
    if family == Family::Gaussian {
        return Err(RfError::InvalidInput("fereg plugin: cmd=glm needs family=poisson|logit".into()));
    }
    let default_sep = if family == Family::Poisson { "relu" } else { "none" };
    let g = GlmOptions {
        family,
        tol: a.num_or("tol", 1e-8)?,
        maxiter: a.num_or("maxiter", 10_000)?,
        itol: a.num_or("itol", 1e-9)?,
        separation: parse_separation(a.str_or("separation", default_sep))?,
        guess: match a.str_or("guess", "simple") {
            "simple" => Guess::Simple,
            "ols" => Guess::Ols,
            v => return Err(RfError::InvalidInput(format!("fereg plugin: invalid guess `{v}`"))),
        },
        log: a.flag("log", true)?,
    };
    let has_offset = a.flag("offset", false)?;
    let spec = Spec {
        dof: args::parse_dof(a.str_or("dof", "pairwise,clusters,continuous"))?,
        want_d: k.outputs.d.is_some(),
        want_fe: !k.outputs.fe.is_empty(),
        want_groupvar: k.outputs.groupvar.is_some(),
        ..Spec::default()
    };
    a.finish()?;
    let lay = Layout::ols(nx);
    let r = lay.ncols();
    let before_out = r + usize::from(has_offset) + 1;
    check_nvars(before_out + k.nout)?;
    let mut stored = est::take_session(&k)?;
    let vars: Vec<i32> = (1..=r as i32).collect();

    with_ctx(&k.common, |ctx| {
        let mut ph = Phases::new(ctx.verbose);
        let parallel = !k.common.serialio;
        let rows = io::read_block(&mut stored, &vars, &k.cached, k.mode, parallel, ctx)?;
        let offset = if has_offset {
            Some(io::read_session_rows(&stored.session, &stored.obs, &[r as i32 + 1], k.mode, parallel, ctx)?)
        } else {
            None
        };
        ph.mark(&format!("glm: read {} x {r}", stored.session.n()));
        let res = stored.session.glm(rows, &lay, offset, &spec, &g, ctx)?;
        ph.mark("glm: estimate");
        est::finish(&res, &k, &stored, before_out)?;
        ph.mark("glm: store");
        Ok(())
    })
}
