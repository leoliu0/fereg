//! `cmd=prepare`: fix the sample, build the session, zero touse for dropped observations.

use regfast_core::{Family, FactorValues, Prep, RfError, Session, WeightKind};

use crate::args::{self, Args};
use crate::store::{self, Stored};
use crate::{Phases, SERIALIO_DEFAULT, check_nvars, io, spi, with_ctx};

enum RawFactor {
    Num(Vec<f64>),
    Str(Vec<u8>, Vec<u32>),
}

pub fn prepare(a: &Args) -> Result<(), RfError> {
    // A new prepare discards any previous session before allocating a new one.
    store::clear();
    let common = a.common(SERIALIO_DEFAULT)?;
    let nfac: usize = a.num_req("nfac")?;
    let nslope: usize = a.num_or("nslope", 0)?;
    let factype = a.str_or("factype", "");
    let absorb = args::parse_absorb(a.str_req("absorb")?, nfac, nslope)?;
    let vce = args::parse_vce(a.str_or("vce", "unadjusted"), a.str_opt("cluster"), nfac)?;
    let wtype = args::parse_wtype(a.str_or("wtype", "none"))?;
    let drop_singletons = a.flag("singletons", true)?;
    let family = args::parse_family(a.str_or("family", "gaussian"))?;
    let sep_fe = match a.str_or("sep", "none") {
        "fe" => true,
        "none" => false,
        v => return Err(RfError::InvalidInput(format!("regfast plugin: invalid sep `{v}` (fe|none)"))),
    };
    let kind = a.absorber()?;
    a.finish()?;

    if factype.len() != nfac || factype.bytes().any(|c| c != b'n' && c != b's') {
        return Err(RfError::InvalidInput("regfast plugin: factype must have one n|s per factor".into()));
    }
    if sep_fe && family == Family::Gaussian {
        return Err(RfError::InvalidInput("regfast plugin: sep=fe requires a GLM family".into()));
    }
    let has_w = wtype != WeightKind::None;
    let has_y = family != Family::Gaussian;
    let nvars = nfac + nslope + usize::from(has_w) + usize::from(has_y) + 1;
    check_nvars(nvars)?;
    let touse_var = nvars as i32;
    if spi::is_str(touse_var) {
        return Err(RfError::InvalidInput("regfast plugin: touse must be numeric".into()));
    }

    with_ctx(&common, |ctx| {
        let mut ph = Phases::new(ctx.verbose);
        let parallel = !common.serialio;
        let candidates = io::sample_obs();
        let touse = io::read_col(touse_var, &candidates, ctx.threads, parallel)?;
        let obs: Vec<i32> = candidates.iter().zip(&touse).filter(|(_, t)| **t != 0.0).map(|(o, _)| *o).collect();
        drop(touse);
        let n = obs.len();
        ctx.check()?;

        let mut raw = Vec::with_capacity(nfac);
        for (f, t) in factype.bytes().enumerate() {
            let var = f as i32 + 1;
            let is_str = spi::is_str(var);
            if is_str != (t == b's') {
                return Err(RfError::InvalidInput(format!(
                    "regfast plugin: factor {} is {} but factype says {}",
                    f + 1,
                    if is_str { "string" } else { "numeric" },
                    t as char
                )));
            }
            raw.push(if is_str {
                let (b, o) = io::read_strings(var, &obs)?;
                RawFactor::Str(b, o)
            } else {
                RawFactor::Num(io::read_col(var, &obs, ctx.threads, parallel)?)
            });
            ctx.check()?;
        }
        let mut next = nfac as i32 + 1;
        let mut numeric = |what: &str| -> Result<Vec<f64>, RfError> {
            let var = next;
            next += 1;
            if spi::is_str(var) {
                return Err(RfError::InvalidInput(format!("regfast plugin: {what} variable must be numeric")));
            }
            io::read_col(var, &obs, ctx.threads, parallel)
        };
        let slopes: Vec<Vec<f64>> = (0..nslope).map(|_| numeric("slope")).collect::<Result<_, _>>()?;
        let weights = if has_w { Some(numeric("weight")?) } else { None };
        let y = if has_y { Some(numeric("dependent")?) } else { None };
        ph.mark(&format!("prepare: read {n} obs"));
        ctx.check()?;

        let prep = Prep {
            n,
            factors: raw
                .iter()
                .map(|r| match r {
                    RawFactor::Num(v) => FactorValues::Num(v),
                    RawFactor::Str(b, o) => FactorValues::Str { bytes: b, offsets: o },
                })
                .collect(),
            absorb,
            slope_vars: slopes.iter().map(Vec::as_slice).collect(),
            weights: weights.as_deref(),
            weight_kind: wtype,
            vce,
            drop_singletons,
            family,
            y_for_sep: y.as_deref(),
            sep_fe,
        };
        let session = Session::prepare_with(prep, ctx, kind)?;
        drop(raw);
        ph.mark("prepare: session");

        let dropped: Vec<i32> =
            session.keep().iter().zip(&obs).filter(|(k, _)| !**k).map(|(_, o)| *o).collect();
        spi::store_num(touse_var, &dropped, &vec![0.0; dropped.len()], 1)?;

        let token = store::new_token();
        spi::scal_save("__rf_N_full", n as f64)?;
        spi::scal_save("__rf_N", session.n() as f64)?;
        spi::scal_save("__rf_num_singletons", session.num_singletons() as f64)?;
        spi::scal_save("__rf_num_fe_separated", session.num_fe_separated() as f64)?;
        spi::local_save("_rf_token", &token.to_string())?;
        if ctx.verbose >= 1 {
            ctx.log(&format!(
                "{{txt}}regfast: {} of {n} observations kept ({} singletons, {} fe-separated); row order {}",
                session.n(),
                session.num_singletons(),
                session.num_fe_separated(),
                if session.row_order_is_local() { "local" } else { "scattered" }
            ));
        }
        let n_kept = session.n();
        store::put(Stored { token, obs, nobs: spi::nobs(), n_kept, session });
        ph.mark("prepare: store");
        Ok(())
    })
}
