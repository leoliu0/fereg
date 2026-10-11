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
    let nmodel: usize = a.num_or("nmodel", 0)?;
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
    // `sample=all`: touse is 1 at every observation, so the scan does not read it.
    let read_touse = match a.str_or("sample", "touse") {
        "touse" => true,
        "all" => false,
        v => return Err(RfError::InvalidInput(format!("regfast plugin: invalid sample `{v}` (touse|all)"))),
    };
    let kind = a.absorber()?;
    a.finish()?;

    if factype.len() != nfac || factype.bytes().any(|c| c != b'n' && c != b's') {
        return Err(RfError::InvalidInput("regfast plugin: factype must have one n|s per factor".into()));
    }
    if sep_fe && family == Family::Gaussian {
        return Err(RfError::InvalidInput("regfast plugin: sep=fe requires a GLM family".into()));
    }
    // GLM: model variable 1 is the dependent variable (separation needs it).
    if family != Family::Gaussian && nmodel == 0 {
        return Err(RfError::InvalidInput("regfast plugin: a GLM family needs the dependent variable as model variable 1".into()));
    }
    let has_w = wtype != WeightKind::None;
    let nvars = nfac + nslope + usize::from(has_w) + nmodel + 1;
    check_nvars(nvars)?;
    let touse_var = nvars as i32;
    if spi::is_str(touse_var) {
        return Err(RfError::InvalidInput("regfast plugin: touse must be numeric".into()));
    }

    // Every numeric variable is read in one pass over the sample: factors, slopes and the
    // weight as columns, the model variables as a row-major block; string factors afterwards.
    let mut col_vars = Vec::with_capacity(nvars);
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
        if !is_str {
            col_vars.push(var);
        }
    }
    let nlead = nslope + usize::from(has_w);
    let extras = (0..nslope).map(|_| "slope").chain(has_w.then_some("weight")).chain((0..nmodel).map(|_| "model"));
    let mut block_vars = Vec::with_capacity(nmodel);
    for (i, what) in extras.enumerate() {
        let var = (nfac + i + 1) as i32;
        if spi::is_str(var) {
            return Err(RfError::InvalidInput(format!("regfast plugin: {what} variable must be numeric")));
        }
        if i < nlead { col_vars.push(var) } else { block_vars.push(var) }
    }
    // GLM: the depvar (model variable 1) is also read as a column, for the separation rule.
    if family != Family::Gaussian {
        col_vars.push(block_vars[0]);
    }

    with_ctx(&common, |ctx| {
        let mut ph = Phases::new(ctx.verbose);
        let parallel = !common.serialio;
        let scan_touse = if read_touse { touse_var } else { 0 };
        let (mut sample, nmiss) = io::read_sample(scan_touse, &col_vars, &block_vars, ctx.threads, parallel)?;
        ctx.check()?;
        // A missing value of any factor, slope, weight or model variable excludes the
        // observation (the ado leaves these to the plugin: protocol); touse is zeroed below.
        let mut dropped = if nmiss > 0 {
            let keep = sample.complete_rows();
            sample.drop_rows(&keep)
        } else {
            Vec::new()
        };
        let mut strs = Vec::new();
        for (f, t) in factype.bytes().enumerate() {
            if t == b's' {
                strs.push(io::read_strings(f as i32 + 1, &sample.obs)?);
                ctx.check()?;
            }
        }
        if let Some(keep) = io::nonempty_rows(&strs) {
            for (b, o) in &mut strs {
                io::drop_strings(b, o, &keep);
            }
            dropped.extend(sample.drop_rows(&keep));
        }
        let io::Sample { obs, cols, block: model, width: _ } = sample;
        let n = obs.len();
        let mut cols = cols.into_iter();
        let mut strs = strs.into_iter();
        let raw: Vec<RawFactor> = factype
            .bytes()
            .map(|t| {
                if t == b's' {
                    let (b, o) = strs.next().expect("one value list per string factor");
                    RawFactor::Str(b, o)
                } else {
                    RawFactor::Num(cols.next().expect("one column per numeric factor"))
                }
            })
            .collect();
        let slopes: Vec<Vec<f64>> = cols.by_ref().take(nslope).collect();
        let weights = if has_w { cols.next() } else { None };
        let y_sep = cols.next();
        ph.mark(&format!("prepare: read {n} obs"));

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
            y_for_sep: y_sep.as_deref(),
            sep_fe,
        };
        let session = Session::prepare_with(prep, ctx, kind)?;
        drop(raw);
        ph.mark("prepare: session");

        dropped.extend(session.keep().iter().zip(&obs).filter(|(k, _)| !**k).map(|(_, o)| *o));
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
        store::put(Stored { token, obs, nobs: spi::nobs(), n_kept, session, model, nmodel });
        ph.mark("prepare: store");
        Ok(())
    })
}
