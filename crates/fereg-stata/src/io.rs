//! Stata data transfer: observation lists, numeric and string reads, result stores (plan §4.3).
//!
//! Numeric reads are split into contiguous row ranges read concurrently (gated by
//! `tests/stata/stress_io.do`; results are identical to a serial read). Output buffers are
//! allocated uninitialized and first touched by the reader threads, so page faults and writes
//! are parallel too.

use std::mem::MaybeUninit;
use std::ops::Range;

use fereg_core::{Ctx, EstResult, RfError, Session};

use crate::args::RowRead;
use crate::spi;
use crate::store::Stored;

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
    fereg_core::par::resolve_threads(threads).min(rows.div_ceil(ROWS_PER_THREAD)).max(1)
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

/// Largest `in` range read in a single pass by one thread: the outputs are sized for the whole
/// range (only the pages actually written are touched), which skips the count pass.
const SINGLE_PASS_MAX: usize = 1 << 24;

/// Sample of a `prepare` call (see [`read_sample`]).
pub struct Sample {
    /// Stata observation of each row (ascending).
    pub obs: Vec<i32>,
    /// One column per column variable.
    pub cols: Vec<Vec<f64>>,
    /// The block variables, row-major: row k holds their `width` values at `obs[k]`.
    pub block: Vec<f64>,
    pub width: usize,
}

/// Sample of a `prepare` call: the observations of the `in` range whose `touse` is not 0
/// (ascending; `touse` 0: every observation, without reading touse) with the numeric
/// `col_vars` at those observations as columns and the numeric `block_vars` as a row-major
/// block (NaN for missing), and the number of missing values read. `if` is not evaluated:
/// callers fold it into touse (protocol).
///
/// One reader thread scans the range once into outputs sized for the whole range. Several
/// make two parallel passes over contiguous observation ranges: the first counts each
/// range's sample, the second writes every range straight into its slice of the outputs.
pub fn read_sample(touse: i32, col_vars: &[i32], block_vars: &[i32], threads: usize, parallel: bool) -> Result<(Sample, u64), RfError> {
    let vars: Vec<i32> = col_vars.iter().chain(block_vars).copied().collect();
    let w = block_vars.len();
    let (in1, in2) = spi::in_range();
    let o1 = in1.max(1);
    let total = usize::try_from(i64::from(in2) - i64::from(o1) + 1).unwrap_or(0);
    let t = reader_threads(total, threads, parallel);
    if t == 1 && total <= SINGLE_PASS_MAX {
        let mut obs = uninit_vec::<i32>(total)?;
        let mut cols: Vec<Vec<f64>> = col_vars.iter().map(|_| uninit_vec::<f64>(total)).collect::<Result<_, _>>()?;
        let mut block = uninit_vec::<f64>(total * w)?;
        let (n, miss) = {
            let mut dst: Vec<_> = cols.iter_mut().map(|c| &mut c.spare_capacity_mut()[..total]).collect();
            let dst_obs = &mut obs.spare_capacity_mut()[..total];
            spi::read_sample(touse, &vars, o1, in2, dst_obs, &mut dst, &mut block.spare_capacity_mut()[..total * w])?
        };
        // SAFETY: the scan wrote the first `n` positions of every column and `n` block rows.
        unsafe {
            obs.set_len(n);
            for c in &mut cols {
                c.set_len(n);
            }
            block.set_len(n * w);
        }
        if n < total / 2 {
            obs.shrink_to_fit();
            cols.iter_mut().for_each(Vec::shrink_to_fit);
            block.shrink_to_fit();
        }
        return Ok((Sample { obs, cols, block, width: w }, miss));
    }

    let per = total.div_ceil(t);
    // Inclusive observation range of part i (empty when first > last).
    let range = |i: usize| (o1 + (i * per).min(total) as i32, o1 + (((i + 1) * per).min(total)) as i32 - 1);

    let counts: Vec<usize> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..t)
            .map(|i| {
                let (a, b) = range(i);
                s.spawn(move || spi::count_sample(touse, a, b))
            })
            .collect();
        handles.into_iter().map(join).collect::<Result<Vec<_>, RfError>>()
    })?;

    let n: usize = counts.iter().sum();
    let mut obs = uninit_vec::<i32>(n)?;
    let mut cols: Vec<Vec<f64>> = col_vars.iter().map(|_| uninit_vec::<f64>(n)).collect::<Result<_, _>>()?;
    let mut block = uninit_vec::<f64>(n.checked_mul(w).ok_or(RfError::OutOfMemory)?)?;
    let miss;
    {
        let mut obs_dst = split_sizes(&mut obs.spare_capacity_mut()[..n], &counts).into_iter();
        let mut col_dst: Vec<_> =
            cols.iter_mut().map(|c| split_sizes(&mut c.spare_capacity_mut()[..n], &counts).into_iter()).collect();
        let block_sizes: Vec<usize> = counts.iter().map(|&c| c * w).collect();
        let mut block_dst = split_sizes(&mut block.spare_capacity_mut()[..n * w], &block_sizes).into_iter();
        let parts = (0..t).map(|i| {
            let dst_obs = obs_dst.next().expect("one slice per part");
            let dst_cols: Vec<_> = col_dst.iter_mut().map(|c| c.next().expect("one slice per part")).collect();
            let dst_block = block_dst.next().expect("one slice per part");
            (range(i), dst_obs, dst_cols, dst_block)
        });
        let vars = &vars;
        // Every part must find exactly the sample it counted.
        let read_part = |(a, b): (i32, i32),
                         dst_obs: &mut [MaybeUninit<i32>],
                         mut dst_cols: Vec<&mut [MaybeUninit<f64>]>,
                         dst_block: &mut [MaybeUninit<f64>]| {
            let want = dst_obs.len();
            match spi::read_sample(touse, vars, a, b, dst_obs, &mut dst_cols, dst_block)? {
                (c, miss) if c == want => Ok(miss),
                _ => Err(RfError::Internal("the estimation sample changed while it was read".into())),
            }
        };
        miss = std::thread::scope(|s| {
            let read_part = &read_part;
            let handles: Vec<_> = parts
                .map(|(r, dst_obs, dst_cols, dst_block)| s.spawn(move || read_part(r, dst_obs, dst_cols, dst_block)))
                .collect();
            handles.into_iter().map(join).sum::<Result<u64, RfError>>()
        })?;
    }
    // SAFETY: every part filled its whole slices, and the slices cover all `n` positions.
    unsafe {
        obs.set_len(n);
        for c in &mut cols {
            c.set_len(n);
        }
        block.set_len(n * w);
    }
    Ok((Sample { obs, cols, block, width: w }, miss))
}

impl Sample {
    /// Rows without a missing value (NaN) in any column or block variable.
    pub fn complete_rows(&self) -> Vec<bool> {
        let mut keep = vec![true; self.obs.len()];
        for c in &self.cols {
            for (k, v) in keep.iter_mut().zip(c) {
                *k &= !v.is_nan();
            }
        }
        if self.width > 0 {
            for (k, row) in keep.iter_mut().zip(self.block.chunks_exact(self.width)) {
                *k &= !row.iter().any(|v| v.is_nan());
            }
        }
        keep
    }

    /// Keeps the rows where `keep` is true (order preserved) and returns the observations of
    /// the other rows.
    pub fn drop_rows(&mut self, keep: &[bool]) -> Vec<i32> {
        let dropped: Vec<i32> = self.obs.iter().zip(keep).filter(|(_, k)| !**k).map(|(o, _)| *o).collect();
        if dropped.is_empty() {
            return dropped;
        }
        retain_by(&mut self.obs, keep);
        for c in &mut self.cols {
            retain_by(c, keep);
        }
        let w = self.width;
        let mut rows = 0;
        for (i, _) in keep.iter().enumerate().filter(|(_, k)| **k) {
            self.block.copy_within(i * w..(i + 1) * w, rows * w);
            rows += 1;
        }
        self.block.truncate(rows * w);
        dropped
    }
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
            .map_err(|_| RfError::InvalidInput("fereg: string factor exceeds 4 GB in total".into()))?;
        offsets.push(end);
    }
    Ok((bytes, offsets))
}

/// Rows where every string factor is non-empty (Stata's string missing), or `None` when that
/// is every row.
pub fn nonempty_rows(strs: &[(Vec<u8>, Vec<u32>)]) -> Option<Vec<bool>> {
    let empty = |o: &[u32]| o.windows(2).any(|w| w[0] == w[1]);
    if !strs.iter().any(|(_, o)| empty(o)) {
        return None;
    }
    let n = strs[0].1.len() - 1;
    let mut keep = vec![true; n];
    for (_, o) in strs {
        for (k, w) in keep.iter_mut().zip(o.windows(2)) {
            *k &= w[0] != w[1];
        }
    }
    Some(keep)
}

fn retain_by<T: Copy>(v: &mut Vec<T>, keep: &[bool]) {
    let mut w = 0;
    for i in 0..v.len() {
        if keep[i] {
            v[w] = v[i];
            w += 1;
        }
    }
    v.truncate(w);
}

/// Keeps the values of a `(bytes, offsets)` string column at the rows where `keep` is true.
pub fn drop_strings(bytes: &mut Vec<u8>, offsets: &mut Vec<u32>, keep: &[bool]) {
    let mut b = Vec::with_capacity(bytes.len());
    let mut o = Vec::with_capacity(offsets.len());
    o.push(0u32);
    for (i, &k) in keep.iter().enumerate() {
        if k {
            b.extend_from_slice(&bytes[offsets[i] as usize..offsets[i + 1] as usize]);
            o.push(b.len() as u32);
        }
    }
    *bytes = b;
    *offsets = o;
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
        return Err(RfError::Code(416, "fereg: missing values in estimation variables of the sample".into()));
    }
    Ok(rows)
}

/// The N×R block of an estimation command in [`Session::row_order`]: variable `vars[c]` is
/// gathered from `prepare`'s model variable `cached[c]` (1-based; protocol `cached`) or, where
/// that is 0, read from Stata. An empty `cached` reads every variable. The model block is
/// consumed (one estimation command per prepare); each solver row reads one contiguous row.
pub fn read_block(
    stored: &mut Stored,
    vars: &[i32],
    cached: &[usize],
    mode: RowRead,
    parallel: bool,
    ctx: &Ctx,
) -> Result<Vec<f64>, RfError> {
    let model = std::mem::take(&mut stored.model);
    let nm = std::mem::take(&mut stored.nmodel);
    if cached.iter().all(|&k| k == 0) {
        drop(model);
        return read_session_rows(&stored.session, &stored.obs, vars, mode, parallel, ctx);
    }
    if cached.len() != vars.len() || cached.iter().any(|&k| k > nm) {
        return Err(RfError::InvalidInput("fereg plugin: cached must map every variable to 0 or a model variable".into()));
    }
    let read: Vec<i32> = vars.iter().zip(cached).filter(|&(_, &k)| k == 0).map(|(&v, _)| v).collect();
    let part = if read.is_empty() { Vec::new() } else { read_session_rows(&stored.session, &stored.obs, &read, mode, parallel, ctx)? };
    let session = &stored.session;
    let (n, r, ru) = (session.n(), vars.len(), read.len());
    let (order, kept) = (session.row_order(), session.kept_index());
    let len = n.checked_mul(r).ok_or(RfError::OutOfMemory)?;
    // Every variable cached in prepare's order, no observation dropped since and the solver
    // rows unpermuted: prepare's block already is the result.
    let whole = ru == 0 && r == nm && cached.iter().enumerate().all(|(c, &k)| k == c + 1);
    if whole && model.len() == len && order.iter().enumerate().all(|(k, &o)| o as usize == k) {
        return Ok(model);
    }
    let mut block = uninit_vec::<f64>(len)?;
    if len == 0 {
        return Ok(block);
    }
    let fill = |row0: usize, out: &mut [MaybeUninit<f64>]| {
        for (j, row) in out.chunks_exact_mut(r).enumerate() {
            let rr = row0 + j;
            let i = kept[order[rr] as usize] as usize;
            let src = &model[i * nm..(i + 1) * nm];
            let mut u = rr * ru;
            for (slot, &k) in row.iter_mut().zip(cached) {
                if k > 0 {
                    slot.write(src[k - 1]);
                } else {
                    slot.write(part[u]);
                    u += 1;
                }
            }
        }
    };
    let t = reader_threads(n, ctx.threads, parallel);
    let dst = &mut block.spare_capacity_mut()[..len];
    if t == 1 {
        fill(0, dst);
    } else {
        let per = n.div_ceil(t);
        std::thread::scope(|s| {
            for (p, out) in dst.chunks_mut(per * r).enumerate() {
                let fill = &fill;
                s.spawn(move || fill(p * per, out));
            }
        });
    }
    // SAFETY: every row of the block was written.
    unsafe { block.set_len(len) };
    Ok(block)
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
        return Err(RfError::Code(503, format!("fereg: matrix {stata_name} has wrong dimensions")));
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
