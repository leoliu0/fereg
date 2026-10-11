//! `cmd=glm`. Varlist: depvar, regressors (`nx`), offset (iff `offset=1`), touse, output
//! variables (`nout`).
//!
//! Keys beyond the shared ones: `family=poisson|logit`, `tol` / `maxiter` (IRLS),
//! `itol` (final inner tolerance), `separation=<comma list of relu,simplex,mu>|none`
//! (`fe` is applied at prepare), `guess=simple|ols`, `log=0|1`, `offset=0|1`,
//! `dof`, `noconstant`. Further GLM outputs are defined in Wave 2.

use regfast_core::{Family, GlmOptions, Guess, Layout, RfError, Separation, Spec};

use crate::args::{self, Args};
use crate::{Phases, check_nvars, est, io, with_ctx};

fn parse_separation(s: &str) -> Result<Separation, RfError> {
    let mut sep = Separation::default();
    if s == "none" {
        return Ok(sep);
    }
    for t in s.split(',') {
        match t {
            "relu" | "simplex" => sep.relu = true,
            "mu" => sep.mu = true,
            _ => return Err(RfError::InvalidInput(format!("regfast plugin: invalid separation item `{t}`"))),
        }
    }
    Ok(sep)
}

pub fn glm(a: &Args) -> Result<(), RfError> {
    let k = est::keys(a, &[("b", "b"), ("V", "V"), ("doft", "dof_table")])?;
    let nx: usize = a.num_req("nx")?;
    let family = args::parse_family(a.str_req("family")?)?;
    if family == Family::Gaussian {
        return Err(RfError::InvalidInput("regfast plugin: cmd=glm needs family=poisson|logit".into()));
    }
    let g = GlmOptions {
        family,
        tol: a.num_or("tol", 1e-8)?,
        maxiter: a.num_or("maxiter", 10_000)?,
        itol: a.num_or("itol", 1e-10)?,
        separation: parse_separation(a.str_or("separation", "relu"))?,
        guess: match a.str_or("guess", "simple") {
            "simple" => Guess::Simple,
            "ols" => Guess::Ols,
            v => return Err(RfError::InvalidInput(format!("regfast plugin: invalid guess `{v}`"))),
        },
        log: a.flag("log", true)?,
    };
    let has_offset = a.flag("offset", false)?;
    let spec = Spec {
        dof: args::parse_dof(a.str_or("dof", "pairwise,clusters,continuous"))?,
        noconstant: a.flag("noconstant", false)?,
        want_resid: k.outputs.resid.is_some(),
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
        let rows = io::read_session_rows(&stored.session, &stored.obs, &vars, k.mode, parallel, ctx)?;
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
