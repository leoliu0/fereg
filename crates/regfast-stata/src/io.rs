//! Stata data transfer: observation lists, numeric and string reads, result stores (plan §4.3).

use regfast_core::par::alloc_vec;
use regfast_core::{Ctx, EstResult, RfError, Session};

use crate::args::RowRead;
use crate::spi;

/// Observations in the `in` range that satisfy the call's `if` (ascending, 1-based).
pub fn sample_obs() -> Vec<i32> {
    let (in1, in2) = spi::in_range();
    (in1.max(1)..=in2).filter(|&o| spi::ifobs(o)).collect()
}

/// Reads `vars` at `obs` into a row-major block (`obs.len() × vars.len()`).
///
/// With `parallel`, the observation list is split into `threads` contiguous ranges read
/// concurrently (plan §4.3, gated by `tests/stata/stress_io.do`); results are identical.
pub fn read_rows(vars: &[i32], obs: &[i32], threads: usize, parallel: bool) -> Result<Vec<f64>, RfError> {
    let r = vars.len();
    let mut out = alloc_vec(obs.len() * r, 0.0)?;
    if r == 0 || obs.is_empty() {
        return Ok(out);
    }
    let t = regfast_core::par::resolve_threads(threads).min(obs.len().div_ceil(65_536)).max(1);
    if !parallel || t == 1 {
        spi::read_rows(vars, obs, &mut out)?;
        return Ok(out);
    }
    let per = obs.len().div_ceil(t);
    std::thread::scope(|s| {
        let handles: Vec<_> = obs
            .chunks(per)
            .zip(out.chunks_mut(per * r))
            .map(|(o, block)| s.spawn(move || spi::read_rows(vars, o, block)))
            .collect();
        handles.into_iter().try_for_each(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
    })?;
    Ok(out)
}

/// One numeric column at `obs`.
pub fn read_col(var: i32, obs: &[i32], threads: usize, parallel: bool) -> Result<Vec<f64>, RfError> {
    read_rows(&[var], obs, threads, parallel)
}

/// String values of `var` at `obs` as `(bytes, offsets)` with `offsets.len() == obs.len() + 1`.
pub fn read_strings(var: i32, obs: &[i32]) -> Result<(Vec<u8>, Vec<u32>), RfError> {
    let strl = spi::is_strl(var);
    let mut bytes = Vec::new();
    let mut offsets = Vec::new();
    offsets.try_reserve_exact(obs.len() + 1).map_err(|_| RfError::OutOfMemory)?;
    offsets.push(0u32);
    let mut buf = Vec::new();
    for &o in obs {
        spi::read_str_into(var, o, strl, &mut bytes, &mut buf)?;
        let end = u32::try_from(bytes.len())
            .map_err(|_| RfError::InvalidInput("regfast: string factor exceeds 4 GB in total".into()))?;
        offsets.push(end);
    }
    Ok((bytes, offsets))
}

/// Stata observation of every kept observation (kept order).
pub fn kept_obs(session: &Session, obs: &[i32]) -> Vec<i32> {
    session.kept_index().iter().map(|&i| obs[i as usize]).collect()
}

/// Reads `vars` for the kept sample into an N×R block in [`Session::row_order`].
///
/// `obs` is the prepare-time observation list (input index → Stata observation).
pub fn read_session_rows(
    session: &Session,
    obs: &[i32],
    vars: &[i32],
    mode: RowRead,
    parallel: bool,
    ctx: &Ctx,
) -> Result<Vec<f64>, RfError> {
    let kept = session.kept_index();
    let order = session.row_order();
    let direct = match mode {
        RowRead::Auto => session.row_order_is_local(),
        RowRead::Direct => true,
        RowRead::Kept => false,
    };
    let rows = if direct {
        let row_obs: Vec<i32> = order.iter().map(|&k| obs[kept[k as usize] as usize]).collect();
        read_rows(vars, &row_obs, ctx.threads, parallel)?
    } else {
        let kept_rows = read_rows(vars, &kept_obs(session, obs), ctx.threads, parallel)?;
        session.permute_rows(kept_rows, vars.len(), ctx)?
    };
    if rows.iter().any(|v| v.is_nan()) {
        return Err(RfError::Code(416, "regfast: missing values in estimation variables of the sample".into()));
    }
    Ok(rows)
}

/// Writes a kept-order vector into variable `var` at the kept observations.
pub fn store_kept(var: i32, kept_obs: &[i32], v: &[f64]) -> Result<(), RfError> {
    if v.len() != kept_obs.len() {
        return Err(RfError::Internal(format!("output vector has {} values for {} observations", v.len(), kept_obs.len())));
    }
    spi::store_num(var, kept_obs, v, 1)
}

/// Fills a pre-created Stata matrix, checking its dimensions (rc 503 on mismatch).
pub fn store_matrix(stata_name: &str, rows: usize, cols: usize, data: &[f64]) -> Result<(), RfError> {
    let (r, c) = spi::mat_dims(stata_name);
    if r != rows as i32 || c != cols as i32 {
        return Err(RfError::Code(503, format!("regfast: matrix {stata_name} has wrong dimensions")));
    }
    spi::mat_store(stata_name, rows, cols, data)
}

/// Prints each note followed by a newline.
pub fn print_notes(notes: &[String]) {
    for n in notes {
        spi::display(&format!("{n}\n"));
    }
}

/// Stores `__rf_<name>` scalars, `_rf_<name>` locals and the `_rf_status` local of a result.
pub fn store_scalars_macros(res: &EstResult) -> Result<(), RfError> {
    for (k, v) in &res.scalars {
        spi::scal_save(&format!("__rf_{k}"), *v)?;
    }
    for (k, v) in &res.macros {
        spi::local_save(&format!("_rf_{k}"), v)?;
    }
    let status: Vec<String> = res.status.iter().map(u8::to_string).collect();
    spi::local_save("_rf_status", &status.join(" "))
}
