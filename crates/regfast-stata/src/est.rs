//! Key handling and result storage shared by `ols`, `iv` and `glm`.

use regfast_core::{EstResult, RfError, Spec};

use crate::args::{self, Args, Common, RowRead};
use crate::store::{self, Stored};
use crate::{SERIALIO_DEFAULT, io};

/// Output-variable positions (1-based among the `nout` output variables).
#[derive(Debug, Default)]
pub struct Outputs {
    pub resid: Option<usize>,
    pub d: Option<usize>,
    pub groupvar: Option<usize>,
    /// One position per saved FE slot.
    pub fe: Vec<usize>,
}

/// Parsed keys common to every estimation command.
pub struct EstKeys {
    pub token: String,
    pub common: Common,
    pub mode: RowRead,
    pub nout: usize,
    pub outputs: Outputs,
    /// `(Stata matrix name, EstResult matrix name)` to fill.
    pub matrices: Vec<(String, &'static str)>,
}

/// Parses token, threads/verbose/serialio, rowread, nout and the output positions, plus the
/// matrix-name keys in `mat_keys` (`(key, EstResult matrix name)`).
pub fn keys(a: &Args, mat_keys: &[(&str, &'static str)]) -> Result<EstKeys, RfError> {
    let token = a.str_req("token")?.to_string();
    let common = a.common(SERIALIO_DEFAULT)?;
    let mode = a.row_read()?;
    let nout: usize = a.num_or("nout", 0)?;
    let outputs = Outputs {
        resid: args::parse_position(a.str_or("resid", "0"), nout, "resid")?,
        d: args::parse_position(a.str_or("d", "0"), nout, "d")?,
        groupvar: args::parse_position(a.str_or("groupvar", "0"), nout, "groupvar")?,
        fe: args::parse_positions(a.str_or("fe", "0"), nout, "fe")?,
    };
    let matrices = mat_keys
        .iter()
        .filter_map(|(key, res)| a.str_opt(key).map(|name| (name.to_string(), *res)))
        .collect();
    Ok(EstKeys { token, common, mode, nout, outputs, matrices })
}

/// Solver/dof keys shared by OLS and IV: `tol maxiter dof noconstant`; the `want_*` flags
/// follow the requested outputs.
pub fn spec(a: &Args, out: &Outputs) -> Result<Spec, RfError> {
    let d = Spec::default();
    Ok(Spec {
        tol: a.num_or("tol", d.tol)?,
        maxiter: a.num_or("maxiter", d.maxiter)?,
        dof: args::parse_dof(a.str_or("dof", "pairwise,clusters,continuous"))?,
        noconstant: a.flag("noconstant", false)?,
        want_resid: out.resid.is_some(),
        want_d: out.d.is_some(),
        want_fe: !out.fe.is_empty(),
        want_groupvar: out.groupvar.is_some(),
        iv: Default::default(),
    })
}

/// Consumes the stored session (token must match) and checks `fe=` against the saved slots.
pub fn take_session(k: &EstKeys) -> Result<Stored, RfError> {
    let stored = store::take(&k.token)?;
    let slots: usize = stored
        .session
        .terms()
        .iter()
        .filter(|t| t.save)
        .map(|t| usize::from(t.intercept) + t.slopes.len())
        .sum();
    if !k.outputs.fe.is_empty() && k.outputs.fe.len() != slots {
        return Err(RfError::InvalidInput(format!(
            "regfast plugin: fe= lists {} slots but the model saves {slots}",
            k.outputs.fe.len()
        )));
    }
    Ok(stored)
}

/// Prints notes, then stores matrices, scalars, locals and output variables, and zeroes touse
/// (varlist position `out_base`) for observations the estimator dropped after prepare.
/// `out_base` = number of varlist variables before the first output variable (touse is last).
pub fn finish(res: &EstResult, k: &EstKeys, stored: &Stored, out_base: usize) -> Result<(), RfError> {
    io::print_notes(&res.notes);
    if stored.session.n() != stored.n_kept {
        let dropped: Vec<i32> = stored
            .session
            .keep()
            .iter()
            .zip(&stored.obs)
            .filter(|(k, _)| !**k)
            .map(|(_, o)| *o)
            .collect();
        crate::spi::store_num(out_base as i32, &dropped, &vec![0.0; dropped.len()], 1)?;
    }
    for (stata, name) in &k.matrices {
        let m = res
            .matrix(name)
            .ok_or_else(|| RfError::Internal(format!("estimator returned no matrix `{name}`")))?;
        io::store_matrix(stata, m.rows, m.cols, &m.data)?;
    }
    io::store_scalars_macros(res)?;

    let o = &k.outputs;
    if o.resid.is_none() && o.d.is_none() && o.groupvar.is_none() && o.fe.is_empty() {
        return Ok(());
    }
    let kept = io::kept_obs(&stored.session, &stored.obs);
    let var = |pos: usize| (out_base + pos) as i32;
    let missing = |what: &str| RfError::Internal(format!("estimator returned no {what}"));
    if let Some(p) = o.resid {
        io::store_kept(var(p), &kept, res.resid.as_deref().ok_or_else(|| missing("residuals"))?)?;
    }
    if let Some(p) = o.d {
        io::store_kept(var(p), &kept, res.d.as_deref().ok_or_else(|| missing("d"))?)?;
    }
    if let Some(p) = o.groupvar {
        io::store_kept(var(p), &kept, res.groupvar.as_deref().ok_or_else(|| missing("groupvar"))?)?;
    }
    if !o.fe.is_empty() {
        if o.fe.len() != res.fe.len() {
            return Err(RfError::InvalidInput(format!(
                "regfast plugin: fe= lists {} slots but the model saves {}",
                o.fe.len(),
                res.fe.len()
            )));
        }
        for (p, v) in o.fe.iter().zip(&res.fe) {
            io::store_kept(var(*p), &kept, v)?;
        }
    }
    Ok(())
}
