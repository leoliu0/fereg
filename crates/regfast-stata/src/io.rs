//! Stata data transfer: observation lists, numeric and string reads, result stores (plan §4.3).
//!
//! Numeric reads are split into contiguous row ranges read concurrently (gated by
//! `tests/stata/stress_io.do`; results are identical to a serial read). Output buffers are
//! allocated uninitialized and first touched by the reader threads, so page faults and writes
//! are parallel too.

use std::mem::MaybeUninit;
use std::ops::Range;

use regfast_core::{Ctx, EstResult, RfError, Session};

use crate::args::RowRead;
use crate::spi;

/// Rows per C read call inside a reader thread (keeps the observation buffer in cache).
const READ_BLOCK: usize = 16_384;

/// Minimum rows per reader thread: below this, spawning costs more than it saves.
const ROWS_PER_THREAD: usize = 65_536;

/// Observations in the `in` range that satisfy the call's `if` (ascending, 1-based).
pub fn sample_obs() -> Vec<i32> {
    let (in1, in2) = spi::in_range();
    (in1.max(1)..=in2).filter(|&o| spi::ifobs(o)).collect()
}

fn reader_threads(rows: usize, threads: usize, parallel: bool) -> usize {
    if !parallel {
        return 1;
    }
    regfast_core::par::resolve_threads(threads).min(rows.div_ceil(ROWS_PER_THREAD)).max(1)
}

fn uninit_vec<T>(len: usize) -> Result<Vec<T>, RfError> {
    let mut v = Vec::new();
    v.try_reserve_exact(len).map_err(|_| RfError::OutOfMemory)?;
    Ok(v)
}

fn join<T>(h: std::thread::ScopedJoinHandle<'_, T>) -> T {
    h.join().unwrap_or_else(|p| std::panic::resume_unwind(p))
}

/// Reads `vars` for `n` rows into a row-major block (`n × vars.len()`, NaN for missing) and
/// counts the missing values. `obs_of(a..b, buf)` appends the Stata observations of rows a..b.
fn read_rows_by<F>(vars: &[i32], n: usize, threads: usize, parallel: bool, obs_of: F) -> Result<(Vec<f64>, u64), RfError>
where
    F: Fn(Range<usize>, &mut Vec<i32>) + Sync,
{
    let r = vars.len();
    let len = n.checked_mul(r).ok_or(RfError::OutOfMemory)?;
    let mut out = uninit_vec::<f64>(len)?;
    if len == 0 {
        return Ok((out, 0));
    }
    let read_range = |start: usize, block: &mut [MaybeUninit<f64>]| -> Result<u64, RfError> {
        let mut buf = Vec::with_capacity(READ_BLOCK.min(block.len() / r));
        let mut miss = 0;
        let mut a = start;
        for chunk in block.chunks_mut(READ_BLOCK * r) {
            let b = a + chunk.len() / r;
            buf.clear();
            obs_of(a..b, &mut buf);
            miss += spi::read_rows(vars, &buf, chunk)?;
            a = b;
        }
        Ok(miss)
    };
    let t = reader_threads(n, threads, parallel);
    let spare = &mut out.spare_capacity_mut()[..len];
    let miss = if t == 1 {
        read_range(0, spare)?
    } else {
        let per = n.div_ceil(t);
        std::thread::scope(|s| {
            let read_range = &read_range;
            let handles: Vec<_> =
                spare.chunks_mut(per * r).enumerate().map(|(i, block)| s.spawn(move || read_range(i * per, block))).collect();
            handles.into_iter().map(join).sum::<Result<u64, RfError>>()
        })?
    };
    // SAFETY: every read call wrote its whole chunk, and the chunks cover all `len` elements.
    unsafe { out.set_len(len) };
    Ok((out, miss))
}

/// Reads `vars` at `obs` into a row-major block (`obs.len() × vars.len()`, NaN for missing).
pub fn read_rows(vars: &[i32], obs: &[i32], threads: usize, parallel: bool) -> Result<Vec<f64>, RfError> {
    Ok(read_rows_by(vars, obs.len(), threads, parallel, |rows, buf| buf.extend_from_slice(&obs[rows]))?.0)
}

/// One numeric column at `obs`.
pub fn read_col(var: i32, obs: &[i32], threads: usize, parallel: bool) -> Result<Vec<f64>, RfError> {
    read_rows(&[var], obs, threads, parallel)
}

/// Sample of a `prepare` call: the observations of the `in` range whose `touse` is not 0
/// (ascending), and the numeric `vars` at those observations as columns (NaN for missing).
/// `if` is not evaluated: callers fold it into touse (protocol).
///
/// Two parallel passes over contiguous observation ranges: the first counts each range's
/// sample, the second writes every range straight into its slice of the output columns.
pub fn read_sample(touse: i32, vars: &[i32], threads: usize, parallel: bool) -> Result<(Vec<i32>, Vec<Vec<f64>>), RfError> {
    let (in1, in2) = spi::in_range();
    let o1 = in1.max(1);
    let total = usize::try_from(i64::from(in2) - i64::from(o1) + 1).unwrap_or(0);
    let t = reader_threads(total, threads, parallel);
    let per = total.div_ceil(t);
    // Inclusive observation range of part i (empty when first > last).
    let range = |i: usize| (o1 + (i * per).min(total) as i32, o1 + (((i + 1) * per).min(total)) as i32 - 1);

    let counts: Vec<usize> = if t == 1 {
        let (a, b) = range(0);
        vec![spi::count_sample(touse, a, b)?]
    } else {
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..t)
                .map(|i| {
                    let (a, b) = range(i);
                    s.spawn(move || spi::count_sample(touse, a, b))
                })
                .collect();
            handles.into_iter().map(join).collect::<Result<Vec<_>, RfError>>()
        })?
    };

    let n: usize = counts.iter().sum();
    let mut obs = uninit_vec::<i32>(n)?;
    let mut cols: Vec<Vec<f64>> = (0..vars.len()).map(|_| uninit_vec::<f64>(n)).collect::<Result<_, _>>()?;
    {
        let mut obs_dst = split_sizes(&mut obs.spare_capacity_mut()[..n], &counts).into_iter();
        let mut col_dst: Vec<_> =
            cols.iter_mut().map(|c| split_sizes(&mut c.spare_capacity_mut()[..n], &counts).into_iter()).collect();
        let mut parts = (0..t).map(|i| {
            let dst_obs = obs_dst.next().expect("one slice per part");
            let dst_cols: Vec<_> = col_dst.iter_mut().map(|c| c.next().expect("one slice per part")).collect();
            (range(i), dst_obs, dst_cols)
        });
        if t == 1 {
            let ((a, b), dst_obs, mut dst_cols) = parts.next().expect("one part");
            spi::read_sample(touse, vars, a, b, dst_obs, &mut dst_cols)?;
        } else {
            std::thread::scope(|s| {
                let handles: Vec<_> = parts
                    .map(|((a, b), dst_obs, mut dst_cols)| s.spawn(move || spi::read_sample(touse, vars, a, b, dst_obs, &mut dst_cols)))
                    .collect();
                handles.into_iter().map(join).collect::<Result<(), RfError>>()
            })?;
        }
    }
    // SAFETY: every part filled its whole slice, and the slices cover all `n` positions.
    unsafe {
        obs.set_len(n);
        for c in &mut cols {
            c.set_len(n);
        }
    }
    Ok((obs, cols))
}

/// Splits `s` into consecutive slices of the given sizes (which must sum to `s.len()`).
fn split_sizes<'a, T>(mut s: &'a mut [T], sizes: &[usize]) -> Vec<&'a mut [T]> {
    let mut out = Vec::with_capacity(sizes.len());
    for &k in sizes {
        let (a, b) = std::mem::take(&mut s).split_at_mut(k);
        out.push(a);
        s = b;
    }
    out
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
    let direct = match mode {
        RowRead::Auto => session.row_order_is_local(),
        RowRead::Direct => true,
        RowRead::Kept => false,
    };
    let n = session.n();
    let (rows, miss) = if direct {
        let order = session.row_order();
        read_rows_by(vars, n, ctx.threads, parallel, |rows, buf| {
            buf.extend(order[rows].iter().map(|&k| obs[kept[k as usize] as usize]))
        })?
    } else {
        let (kept_rows, miss) =
            read_rows_by(vars, n, ctx.threads, parallel, |rows, buf| buf.extend(kept[rows].iter().map(|&i| obs[i as usize])))?;
        (session.permute_rows(kept_rows, vars.len(), ctx)?, miss)
    };
    if miss > 0 {
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
