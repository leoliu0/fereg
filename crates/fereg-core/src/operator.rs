//! Kernels over the cell structure (plan D5): the reduced operator S*P, the right-hand side,
//! diag(S), back-substitution / in-place residualization, and the assembly of the blocks used by
//! the preconditioner.
//!
//! Every parallel kernel splits the segments into the fixed parts of [`Structure::tasks`]
//! (independent of the thread count); per-task accumulators are combined in task order and
//! large-block scatters are pulls along the level-sorted cell lists, so results are bit-identical
//! for any number of threads (plan D3).

use std::mem::MaybeUninit;
use std::sync::Mutex;

use rayon::prelude::*;

use crate::error::RfError;
use crate::par::{CHUNK, MAX_DISPATCH_WIDTH, Pool, alloc_par, alloc_uninit, alloc_vec, assume_init, dispatch_width, has_avx2_fma, sum_chunks, width};
use crate::rng::SplitMix64;
use crate::structure::{Structure, split_bounds};

/// Coordinates with d <= REDUNDANT*n are redundant (no information beyond FE1).
pub const REDUNDANT: f64 = 1e-12;

/// Largest reduced vector (bytes, all R columns) scattered entirely through per-task accumulators.
pub const ACC_BYTES_MAX: usize = 8 << 20;
/// Most scatter tasks when every block is accumulated (fixed: thread-count independent).
pub const MAX_ACC_TASKS: usize = 64;
/// Fewest scatter tasks when every block is accumulated (when the segments allow): parallelism
/// for up to this many threads, whatever the accumulator size.
pub const MIN_ACC_TASKS: usize = 16;
/// Accumulated values per cell value: the tasks' accumulators together hold at most 1/8 as many
/// values as the cell data they reduce.
const ACC_CELL_RATIO: usize = 8;
/// Total bytes of the per-task accumulators of one scatter. Beyond the last-level cache of the
/// reference machine (Ryzen 9 9950X, 2 x 32 MiB) their scatters and combine run from DRAM: on the
/// 6.7M-trip taxi panel (2.6M cells, m*r = 195k) 53 tasks (83 MB) took 7.1 ms per matvec, 13 tasks
/// (20 MB) 3.7 ms, with the same iterations. 8 MiB was 1-3% faster than 16 MiB on the 1e7
/// benchmark panels at 16 threads (least squares, IV, Poisson, logit) and 4 MiB no faster.
const ACC_TOTAL_BYTES: usize = 8 << 20;
/// Largest small super-block assembled by dense products (when dense within the segments).
const GEMM_SMALL_MAX: usize = 512;


/// Weight-dependent quantities (refreshed by `set_weights`).
pub struct WeightState {
    /// Observation weights in row order (`None` = all ones).
    pub w_rows: Option<Vec<f64>>,
    /// Cell weights W_c when cells aggregate rows (with one row per cell they are `w_rows`; see
    /// [`WeightState::cell_w`]).
    cell_sums: Option<Vec<f64>>,
    /// FE1 level weights.
    pub n1: Vec<f64>,
    /// Level weights of every reduced coordinate.
    pub nr: Vec<f64>,
    /// diag(S).
    pub diag: Vec<f64>,
    /// Redundant coordinates (excluded from the preconditioner).
    pub redundant: Vec<bool>,
    /// sum w.
    pub sw: f64,
    /// Dense S over the small coordinates (m_s x m_s, row-major) when the weight pass built it
    /// (see [`level_weights`]); otherwise [`Ops::assemble_small`] computes it on demand.
    pub small_s: Option<Vec<f64>>,
}

impl WeightState {
    /// Cell weights W_c (`None` = every W_c is 1).
    #[inline]
    pub fn cell_w(&self) -> Option<&[f64]> {
        self.cell_sums.as_deref().or(self.w_rows.as_deref())
    }

    #[inline]
    pub fn cw(&self, c: usize) -> f64 {
        match self.cell_w() {
            Some(w) => w[c],
            None => 1.0,
        }
    }
}

/// Symmetric matrix in CSR form (full pattern, sorted columns, diagonal included).
pub struct Csr {
    pub n: usize,
    pub ptr: Vec<u32>,
    pub col: Vec<u32>,
    pub val: Vec<f64>,
}

/// Per-segment aggregated codes of one block: for segment s, `codes[off[s]..off[s+1]]` are the
/// distinct codes (ascending) and `sums` their cell-weight totals c_{s,g}.
pub struct SegAgg {
    pub off: Vec<u32>,
    pub codes: Vec<u32>,
    pub sums: Vec<f64>,
    /// E = sum_s d_s^2 (assembly work).
    pub work: u64,
}

impl SegAgg {
    fn seg(&self, s: usize) -> (&[u32], &[f64]) {
        let r = self.off[s] as usize..self.off[s + 1] as usize;
        (&self.codes[r.clone()], &self.sums[r])
    }

    /// sum_s d_s*l_s, with d_s the codes of segment s and l_s those >= `first_large` (the walk of
    /// the large-coordinate rows of the joint assembly; codes are global coordinates).
    pub fn large_work(&self, first_large: usize) -> u64 {
        (0..self.off.len() - 1)
            .map(|s| {
                let hs = self.seg(s).0;
                let l = hs.len() - hs.partition_point(|&h| (h as usize) < first_large);
                (hs.len() * l) as u64
            })
            .sum()
    }
}

/// Structural lookups of the joint assembly (weight independent).
pub struct JointCells {
    /// Block of every reduced coordinate.
    coord_block: Vec<u32>,
    /// Segment of every cell.
    cell_seg: Vec<u32>,
}

impl JointCells {
    /// Cells (ascending) whose code in the coordinate's (large) block is coordinate `k`.
    fn cells<'a>(&'a self, st: &'a Structure, k: usize) -> &'a [u32] {
        let blk = &st.blocks[self.coord_block[k] as usize];
        let g = k - blk.off;
        &blk.perm[blk.perm_off[g] as usize..blk.perm_off[g + 1] as usize]
    }
}

/// Accumulator, marks and touched list of one row.
type RowBufs = (Vec<f64>, Vec<u32>, Vec<u32>);

/// Row-assembly scratch per pool thread, allocated on a thread's first use. Each row stamps the
/// marks with its own id, so stale contents never affect a result, and results do not depend on
/// which thread computes a row.
struct RowScratch {
    len: usize,
    slots: Vec<Mutex<Option<RowBufs>>>,
}

impl RowScratch {
    fn new(len: usize, pool: &Pool) -> RowScratch {
        RowScratch { len, slots: (0..pool.threads().max(1)).map(|_| Mutex::new(None)).collect() }
    }

    fn with<T>(&self, f: impl FnOnce(&mut [f64], &mut [u32], &mut Vec<u32>) -> T) -> T {
        let i = rayon::current_thread_index().unwrap_or(0) % self.slots.len();
        let mut slot = self.slots[i].lock().unwrap_or_else(|e| e.into_inner());
        let (acc, mark, touched) = slot.get_or_insert_with(|| (vec![0.0; self.len], vec![u32::MAX; self.len], Vec::new()));
        f(acc, mark, touched)
    }
}

/// Sums of `w` over the cells of a segment that share each cell's code in one block:
/// `out[k] = sum_{c' : code(c') = code(c_k)} W_c'`.
fn seg_level_sums(codes: &[u32], w: &[f64], out: &mut [f64], tmp: &mut Vec<(u32, u32)>) {
    let k = codes.len();
    if k <= 16 {
        for i in 0..k {
            let mut s = 0.0;
            for j in 0..k {
                if codes[j] == codes[i] {
                    s += w[j];
                }
            }
            out[i] = s;
        }
        return;
    }
    tmp.clear();
    tmp.extend(codes.iter().enumerate().map(|(i, &c)| (c, i as u32)));
    tmp.sort_unstable();
    let mut a = 0;
    while a < k {
        let mut b = a;
        let mut s = 0.0;
        while b < k && tmp[b].0 == tmp[a].0 {
            s += w[tmp[b].1 as usize];
            b += 1;
        }
        for t in &tmp[a..b] {
            out[t.1 as usize] = s;
        }
        a = b;
    }
}

/// Computes the weight state for row weights `w_rows` (row order).
pub fn weight_state(st: &Structure, w_rows: Option<Vec<f64>>, pool: &Pool) -> Result<WeightState, RfError> {
    let mut ws = WeightState {
        w_rows: None,
        cell_sums: None,
        n1: Vec::new(),
        nr: Vec::new(),
        diag: Vec::new(),
        redundant: Vec::new(),
        sw: 0.0,
        small_s: None,
    };
    refresh_weights(st, &mut ws, w_rows, pool)?;
    Ok(ws)
}

/// Refreshes `ws` in place for `w_rows`, reusing existing buffers (`cell_sums`, `n1`, `redundant`)
/// across IRLS weight updates.
pub fn refresh_weights(st: &Structure, ws: &mut WeightState, w_rows: Option<Vec<f64>>, pool: &Pool) -> Result<(), RfError> {
    let mc = st.ncells;
    ws.w_rows = w_rows;
    // With one row per cell (no `cell_obs_off`) the cell weights are the row weights.
    if st.cell_obs_off.is_none() {
        ws.cell_sums = None;
    } else {
        let w = ws.w_rows.as_deref();
        let fill_cell = |c: usize| match w {
            Some(w) => w[st.rows_of_cell(c)].iter().sum(),
            None => st.rows_of_cell(c).len() as f64,
        };
        let cw = match ws.cell_sums.take() {
            Some(mut buf) if buf.len() == mc => {
                if mc <= CHUNK || pool.threads() <= 1 {
                    for (c, v) in buf.iter_mut().enumerate() {
                        *v = fill_cell(c);
                    }
                } else {
                    pool.install(|| {
                        buf.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, ch)| {
                            let base = ci * CHUNK;
                            for (k, v) in ch.iter_mut().enumerate() {
                                *v = fill_cell(base + k);
                            }
                        })
                    });
                }
                buf
            }
            _ => {
                let mut buf = alloc_uninit::<f64>(mc)?;
                if mc <= CHUNK || pool.threads() <= 1 {
                    for (c, v) in buf.iter_mut().enumerate() {
                        v.write(fill_cell(c));
                    }
                } else {
                    pool.install(|| {
                        buf.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, ch)| {
                            let base = ci * CHUNK;
                            for (k, v) in ch.iter_mut().enumerate() {
                                v.write(fill_cell(base + k));
                            }
                        })
                    });
                }
                // SAFETY: every slot of `0..mc` was written above.
                unsafe { assume_init(buf) }
            }
        };
        ws.cell_sums = Some(cw);
    }

    let mut n1 = std::mem::take(&mut ws.n1);
    let cw_opt = ws.cell_w();
    let seg_sum = |s: usize| -> f64 {
        let cells = st.cells_of(s);
        match cw_opt {
            Some(cw) => cw[cells].iter().sum(),
            None => cells.len() as f64,
        }
    };
    if n1.len() == st.g1 {
        if st.g1 <= 1024 || pool.threads() <= 1 {
            for (s, v) in n1.iter_mut().enumerate() {
                *v = seg_sum(s);
            }
        } else {
            pool.install(|| {
                n1.par_chunks_mut(1024).enumerate().for_each(|(ci, ch)| {
                    let base = ci * 1024;
                    for (k, v) in ch.iter_mut().enumerate() {
                        *v = seg_sum(base + k);
                    }
                })
            });
        }
    } else {
        let mut buf = alloc_uninit::<f64>(st.g1)?;
        if st.g1 <= 1024 || pool.threads() <= 1 {
            for (s, v) in buf.iter_mut().enumerate() {
                v.write(seg_sum(s));
            }
        } else {
            pool.install(|| {
                buf.par_chunks_mut(1024).enumerate().for_each(|(ci, ch)| {
                    let base = ci * 1024;
                    for (k, v) in ch.iter_mut().enumerate() {
                        v.write(seg_sum(base + k));
                    }
                })
            });
        }
        // SAFETY: every slot of `0..st.g1` was written above.
        n1 = unsafe { assume_init(buf) };
    }
    ws.sw = sum_chunks(pool, n1.len(), CHUNK, 1, |r, acc| {
        acc[0] += n1[r].iter().sum::<f64>();
    })[0];
    ws.n1 = n1;

    let (nr, mut dq, small_s) = level_weights(st, ws, pool)?;
    ws.small_s = small_s;
    let m = nr.len();
    let mut red = std::mem::take(&mut ws.redundant);
    if red.len() != m {
        red = vec![false; m];
    }
    if m <= CHUNK || pool.threads() <= 1 {
        for ((d, &n), r) in dq.iter_mut().zip(&nr).zip(red.iter_mut()) {
            *d = n - *d;
            *r = *d <= REDUNDANT * n;
        }
    } else {
        pool.install(|| {
            dq.par_chunks_mut(CHUNK).zip(nr.par_chunks(CHUNK)).zip(red.par_chunks_mut(CHUNK)).for_each(|((dc, nc), rc)| {
                for ((d, &n), r) in dc.iter_mut().zip(nc).zip(rc.iter_mut()) {
                    *d = n - *d;
                    *r = *d <= REDUNDANT * n;
                }
            })
        });
    }
    ws.diag = dq;
    ws.redundant = red;
    ws.nr = nr;
    Ok(())
}

/// Level weights n_r and sum_{g1} c^2/n1 of every reduced coordinate, plus the dense small block
/// of S when it is dense within the segments (same pass: the per-segment level sums c_s are its
/// GEMM columns, see [`Ops::assemble_small_gemm`]).
#[allow(clippy::type_complexity)]
fn level_weights(st: &Structure, ws: &WeightState, pool: &Pool) -> Result<(Vec<f64>, Vec<f64>, Option<Vec<f64>>), RfError> {
    let (m, nb, ms) = (st.m, st.nb, st.m_small);
    if nb == 0 {
        return Ok((Vec::new(), Vec::new(), None));
    }
    if 2 * m * 8 > st.acc_bytes_max {
        let (nr, dq) = level_weights_pull(st, ws, pool)?;
        return Ok((nr, dq, None));
    }
    // Per-task accumulators [n_r | sum c^2/n1] over the whole reduced space; the per-segment level
    // sums c_{s,g} go through a dense scratch that is reset after each segment.
    let tb = acc_tasks(st, 4, 2 * m);
    let nt = tb.len() - 1;
    let fuse = gemm_small(st) && nt * ms * ms <= 4 * st.ncells;
    let ns = st.blocks.iter().filter(|b| b.small).count();
    let batch = (4096 / ms.max(1)).max(128);
    let offs = block_offsets(st, 1);
    let cw_opt = ws.cell_w();
    let task_fn = |t: usize| -> (Vec<f64>, Vec<f64>) {
        let (s0, s1) = (tb[t] as usize, tb[t + 1] as usize);
        let mut acc = vec![0.0; 2 * m];
        let mut cs = vec![0.0; m];
        let mut acc_s = if fuse { vec![0.0; ms * ms] } else { Vec::new() };
        let ld = if fuse { batch.min(s1 - s0) } else { 0 };
        let mut cbuf = vec![0.0; ms * ld];
        let mut ncol = 0;
        let all_codes = st.cell_codes();
        if nb == 1 && !fuse {
            let off0 = offs[0];
            for s in s0..s1 {
                let cells = st.cells_of(s);
                let codes = &all_codes[cells.clone()];
                let n1 = ws.n1[s];
                match cw_opt {
                    Some(cw) => {
                        let wc = &cw[cells];
                        for (&code, &w) in codes.iter().zip(wc) {
                            cs[off0 + code as usize] += w;
                        }
                        for (&code, &w) in codes.iter().zip(wc) {
                            let k = off0 + code as usize;
                            acc[k] += w;
                            acc[m + k] += w * cs[k] / n1;
                        }
                    }
                    None => {
                        for &code in codes {
                            cs[off0 + code as usize] += 1.0;
                        }
                        for &code in codes {
                            let k = off0 + code as usize;
                            acc[k] += 1.0;
                            acc[m + k] += cs[k] / n1;
                        }
                    }
                }
                for &code in codes {
                    cs[off0 + code as usize] = 0.0;
                }
            }
            return (acc, acc_s);
        }
        if nb == 2 && (!fuse || ns <= 1) {
            let (off0, off1) = (offs[0], offs[1]);
            for s in s0..s1 {
                let cells = st.cells_of(s);
                let codes = &all_codes[cells.start * 2..cells.end * 2];
                let n1 = ws.n1[s];
                match cw_opt {
                    Some(cw) => {
                        let wc = &cw[cells];
                        for (cc, &w) in codes.chunks_exact(2).zip(wc) {
                            cs[off0 + cc[0] as usize] += w;
                            cs[off1 + cc[1] as usize] += w;
                        }
                        for (cc, &w) in codes.chunks_exact(2).zip(wc) {
                            let k0 = off0 + cc[0] as usize;
                            let k1 = off1 + cc[1] as usize;
                            acc[k0] += w;
                            acc[m + k0] += w * cs[k0] / n1;
                            acc[k1] += w;
                            acc[m + k1] += w * cs[k1] / n1;
                        }
                    }
                    None => {
                        for cc in codes.chunks_exact(2) {
                            cs[off0 + cc[0] as usize] += 1.0;
                            cs[off1 + cc[1] as usize] += 1.0;
                        }
                        for cc in codes.chunks_exact(2) {
                            let k0 = off0 + cc[0] as usize;
                            let k1 = off1 + cc[1] as usize;
                            acc[k0] += 1.0;
                            acc[m + k0] += cs[k0] / n1;
                            acc[k1] += 1.0;
                            acc[m + k1] += cs[k1] / n1;
                        }
                    }
                }
                if fuse {
                    let f = 1.0 / n1.sqrt();
                    for (k, v) in cs[..ms].iter().enumerate() {
                        cbuf[k * ld + ncol] = v * f;
                    }
                    ncol += 1;
                    if ncol == ld {
                        sub_gram_lower(&mut acc_s, ms, &cbuf, ld, ncol);
                        ncol = 0;
                    }
                }
                for cc in codes.chunks_exact(2) {
                    cs[off0 + cc[0] as usize] = 0.0;
                    cs[off1 + cc[1] as usize] = 0.0;
                }
            }
            if fuse && ncol > 0 {
                sub_gram_lower(&mut acc_s, ms, &cbuf, ld, ncol);
            }
            return (acc, acc_s);
        }
        for s in s0..s1 {
            let cells = st.cells_of(s);
            let codes = &all_codes[cells.start * nb..cells.end * nb];
            for (cc, c) in codes.chunks_exact(nb).zip(cells.clone()) {
                let w = match cw_opt {
                    Some(cw) => cw[c],
                    None => 1.0,
                };
                for (&code, &off) in cc.iter().zip(&offs) {
                    cs[off + code as usize] += w;
                }
            }
            let n1 = ws.n1[s];
            for (cc, c) in codes.chunks_exact(nb).zip(cells.clone()) {
                let w = match cw_opt {
                    Some(cw) => cw[c],
                    None => 1.0,
                };
                for (&code, &off) in cc.iter().zip(&offs) {
                    let k = off + code as usize;
                    acc[k] += w;
                    acc[m + k] += w * cs[k] / n1;
                }
                if fuse && ns > 1 {
                    // Cross counts between distinct small blocks (the diagonal is n_r).
                    for (b, (&code, &off)) in cc[..ns].iter().zip(&offs).enumerate() {
                        let row = &mut acc_s[(off + code as usize) * ms..(off + code as usize + 1) * ms];
                        for (b2, (&code2, &off2)) in cc[..ns].iter().zip(&offs).enumerate() {
                            if b2 != b {
                                row[off2 + code2 as usize] += w;
                            }
                        }
                    }
                }
            }
            if fuse {
                let f = 1.0 / n1.sqrt();
                for (k, v) in cs[..ms].iter().enumerate() {
                    cbuf[k * ld + ncol] = v * f;
                }
                ncol += 1;
                if ncol == ld {
                    sub_gram_lower(&mut acc_s, ms, &cbuf, ld, ncol);
                    ncol = 0;
                }
            }
            for cc in codes.chunks_exact(nb) {
                for (&code, &off) in cc.iter().zip(&offs) {
                    cs[off + code as usize] = 0.0;
                }
            }
        }
        if fuse && ncol > 0 {
            sub_gram_lower(&mut acc_s, ms, &cbuf, ld, ncol);
        }
        (acc, acc_s)
    };
    let accs: Vec<(Vec<f64>, Vec<f64>)> = if nt <= 1 || pool.threads() <= 1 {
        (0..nt).map(task_fn).collect()
    } else {
        pool.install(|| (0..nt).into_par_iter().map(task_fn).collect())
    };
    let mut out = alloc_uninit::<f64>(2 * m)?;
    if 2 * m <= 4096 || pool.threads() <= 1 {
        for (i, o) in out.iter_mut().enumerate() {
            let mut s = 0.0;
            for (acc, _) in &accs {
                s += acc[i];
            }
            o.write(s);
        }
    } else {
        pool.install(|| {
            out.par_chunks_mut(4096).enumerate().for_each(|(ci, o)| {
                let base = ci * 4096;
                for (k, slot) in o.iter_mut().enumerate() {
                    let mut s = 0.0;
                    for (acc, _) in &accs {
                        s += acc[base + k];
                    }
                    slot.write(s);
                }
            })
        });
    }
    // SAFETY: every slot of `0..2*m` was written above.
    let mut out = unsafe { assume_init(out) };
    let dq = out.split_off(m);
    let small = fuse.then(|| {
        let mut full = vec![0.0; ms * ms];
        for (_, acc_s) in &accs {
            for k in 0..ms {
                for (a, v) in full[k * ms..k * ms + k + 1].iter_mut().zip(&acc_s[k * ms..k * ms + k + 1]) {
                    *a += v;
                }
            }
        }
        for k in 0..ms {
            full[k * ms + k] += out[k];
            for l in 0..k {
                full[l * ms + k] = full[k * ms + l];
            }
        }
        full
    });
    Ok((out, dq, small))
}

/// The small super-block is dense within the segments: assembled by dense products
/// ([`Ops::assemble_small_gemm`]).
fn gemm_small(st: &Structure) -> bool {
    let ms = st.m_small;
    ms > 0 && ms <= GEMM_SMALL_MAX && (ms * ms * st.g1) as u64 <= 8 * st.small_work
}

/// Lower triangle of `acc` (ms x ms, row-major) -= C*C^T for the first `ncol` columns of the
/// row-major ms x `ld` block `c` (fixed summation order: deterministic).
fn sub_gram_lower(acc: &mut [f64], ms: usize, c: &[f64], ld: usize, ncol: usize) {
    for k in 0..ms {
        let ck = &c[k * ld..k * ld + ncol];
        let row = &mut acc[k * ms..k * ms + k + 1];
        for (l, a) in row.iter_mut().enumerate() {
            *a -= dot(ck, &c[l * ld..l * ld + ncol]);
        }
    }
}

/// Dot product with four interleaved partial sums (vectorizable; fixed order).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n4 = a.len() / 4 * 4;
    let mut s = [0.0f64; 4];
    for (x, y) in a[..n4].chunks_exact(4).zip(b[..n4].chunks_exact(4)) {
        for i in 0..4 {
            s[i] += x[i] * y[i];
        }
    }
    let mut t = (s[0] + s[1]) + (s[2] + s[3]);
    for (x, y) in a[n4..].iter().zip(&b[n4..]) {
        t += x * y;
    }
    t
}

/// [`level_weights`] for reduced spaces too large for per-task accumulators: small blocks
/// through per-task accumulators, large ones through per-cell values pulled along `perm`.
fn level_weights_pull(st: &Structure, ws: &WeightState, pool: &Pool) -> Result<(Vec<f64>, Vec<f64>), RfError> {
    let mc = st.ncells;
    let ns = st.blocks.iter().filter(|b| b.small).count();
    let nl = st.nb - ns;
    let ms = st.m_small;
    let tb = st.tasks(2 * ms);
    let cb = st.cell_bounds(&tb);
    let mut zq = alloc_uninit::<f64>(mc * nl)?;
    let zb: Vec<usize> = cb.iter().map(|c| c * nl).collect();
    let pieces = split_bounds(&mut zq, &zb);
    let cw_opt = ws.cell_w();
    let all_codes = st.cell_codes();
    let nb = st.nb;
    let run_piece = |(t, zt): (usize, &mut [MaybeUninit<f64>])| -> Vec<f64> {
        let mut acc = vec![0.0; 2 * ms];
        let (mut codes_buf, mut w_unit, mut csum, mut tmp) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for s in tb[t] as usize..tb[t + 1] as usize {
            let cells = st.cells_of(s);
            let k = cells.len();
            let w: &[f64] = match cw_opt {
                Some(cw) => &cw[cells.clone()],
                None => {
                    if w_unit.len() < k {
                        w_unit.resize(k, 1.0);
                    }
                    &w_unit[..k]
                }
            };
            if csum.len() < k {
                csum.resize(k, 0.0);
            }
            let n1 = ws.n1[s];
            for (b, blk) in st.blocks.iter().enumerate() {
                let codes: &[u32] = if nb == 1 {
                    &all_codes[cells.clone()]
                } else {
                    codes_buf.clear();
                    codes_buf.extend(all_codes[cells.start * nb..cells.end * nb].chunks_exact(nb).map(|cc| cc[b]));
                    &codes_buf
                };
                seg_level_sums(codes, w, &mut csum[..k], &mut tmp);
                for i in 0..k {
                    let q = w[i] * csum[i] / n1;
                    if blk.small {
                        let at = blk.off + codes[i] as usize;
                        acc[at] += w[i];
                        acc[ms + at] += q;
                    } else {
                        zt[(cells.start + i - cb[t]) * nl + (b - ns)].write(q);
                    }
                }
            }
        }
        acc
    };
    let accs: Vec<Vec<f64>> = if pieces.len() <= 1 || pool.threads() <= 1 {
        pieces.into_iter().enumerate().map(run_piece).collect()
    } else {
        pool.install(|| pieces.into_par_iter().enumerate().map(run_piece).collect())
    };
    // SAFETY: every `(cell, large_block)` slot in `0..mc*nl` was written above.
    let zq = unsafe { assume_init(zq) };
    let mut nr = alloc_uninit::<f64>(st.m)?;
    let mut dq = alloc_uninit::<f64>(st.m)?;
    for k in 0..ms {
        let (mut sn, mut sd) = (0.0, 0.0);
        for acc in &accs {
            sn += acc[k];
            sd += acc[ms + k];
        }
        nr[k].write(sn);
        dq[k].write(sd);
    }
    for (b, blk) in st.blocks.iter().enumerate().skip(ns) {
        let col = b - ns;
        let rng = blk.off..blk.off + blk.g;
        let (nr_b, dq_b) = (&mut nr[rng.clone()], &mut dq[rng]);
        let pull_level = |(g, (n, d)): (usize, (&mut MaybeUninit<f64>, &mut MaybeUninit<f64>))| {
            let (mut sn, mut sd) = (0.0, 0.0);
            for &c in &blk.perm[blk.perm_off[g] as usize..blk.perm_off[g + 1] as usize] {
                let ci = c as usize;
                sn += match cw_opt {
                    Some(cw) => cw[ci],
                    None => 1.0,
                };
                sd += zq[ci * nl + col];
            }
            n.write(sn);
            d.write(sd);
        };
        if blk.g <= 1024 || pool.threads() <= 1 {
            nr_b.iter_mut().zip(dq_b.iter_mut()).enumerate().for_each(pull_level);
        } else {
            pool.install(|| {
                nr_b.par_iter_mut().zip(dq_b.par_iter_mut()).enumerate().with_min_len(1024).for_each(pull_level);
            });
        }
    }
    // SAFETY: `0..ms` and every large block `ms..st.m` were written above.
    Ok((unsafe { assume_init(nr) }, unsafe { assume_init(dq) }))
}

/// Segment boundaries of tasks with per-task accumulators of `acc_len` values over the whole
/// reduced space: [`MIN_ACC_TASKS`] to [`MAX_ACC_TASKS`] tasks, within [`ACC_TOTAL_BYTES`] and
/// [`ACC_CELL_RATIO`] when the floor allows. Depends only on the data, `r` and `acc_len`.
fn acc_tasks(st: &Structure, r: usize, acc_len: usize) -> Vec<u32> {
    let np = st.nparts();
    let by_cells = st.ncells * r / (ACC_CELL_RATIO * acc_len).max(1);
    let by_bytes = ACC_TOTAL_BYTES / (8 * acc_len).max(1);
    let k = by_cells.min(by_bytes).clamp(MIN_ACC_TASKS, MAX_ACC_TASKS).min(np);
    let per = np.div_ceil(k);
    let mut t: Vec<u32> = (0..np).step_by(per).map(|p| st.parts[p]).collect();
    t.push(st.parts[np]);
    t
}

/// Output of [`Ops::rhs`].
pub struct Rhs {
    /// Reduced right-hand side (m x r).
    pub rhs: Vec<f64>,
    /// FE1 level sums b1 = sum w v per segment (g1 x r).
    pub b1: Vec<f64>,
    /// sum w v^2 per column.
    pub wsq: Vec<f64>,
    /// sum w v per column.
    pub wsum: Vec<f64>,
}

/// Kernels bound to a structure and a weight state.
pub struct Ops<'a> {
    pub st: &'a Structure,
    pub ws: &'a WeightState,
}

/// Scratch of the scatter kernels, reused across calls: the per-task accumulators and the
/// M x r buffer of the large-block pulls. Between calls every accumulator entry is zero (the
/// combine resets the entries it reads), so tasks start without a zeroing pass.
#[derive(Default)]
pub struct Scratch {
    zbuf: Vec<f64>,
    accs: Vec<Vec<f64>>,
    /// A call stopped between its tasks and the combine (a panic): the accumulators are not zero.
    dirty: bool,
}

/// One scatter task: segments `segs` (cells from `c0` on), the per-task accumulator of the
/// first `nacc` blocks, the task's piece of the M x r buffer, per-segment outputs and per-task
/// sums.
struct Task<'s> {
    segs: std::ops::Range<usize>,
    c0: usize,
    acc: &'s mut [f64],
    nacc: usize,
    zt: Option<&'s mut [f64]>,
    seg_out: Option<&'s mut [f64]>,
    extra: &'s mut [f64],
}

/// Block offsets premultiplied by the width.
fn block_offsets(st: &Structure, r: usize) -> Vec<usize> {
    st.blocks.iter().map(|b| b.off * r).collect()
}

/// u_c = sum_b P[off_b + g_b(c)] for the `k` cells whose codes are `codes` (k x nb) into `u`.
///
/// # Safety
/// `codes` are rows of [`Structure::cell_codes`] of a structure `st`, and
/// `st.codes_within(nb, offs, r, nb, p.len())` holds.
#[inline(always)]
unsafe fn gather_cells<const R: usize>(codes: &[u32], nb: usize, offs: &[usize], p: &[f64], r: usize, u: &mut [f64]) {
    let r = width::<R>(r);
    if nb == 0 {
        u.fill(0.0);
        return;
    }
    if R > 0 && R <= MAX_DISPATCH_WIDTH {
        // Sums in a local array (registers): accumulating into `u` would chain every term
        // through store-to-load forwarding.
        for (cc, uc) in codes.chunks_exact(nb).zip(u.chunks_exact_mut(r)) {
            let mut acc = [0.0f64; MAX_DISPATCH_WIDTH];
            for (&code, &off) in cc.iter().zip(offs) {
                let at = off + code as usize * r;
                // SAFETY: code < code_bound(b), so at + r <= offs[b] + code_bound(b)*r <= p.len().
                let pv = unsafe { p.get_unchecked(at..at + r) };
                for j in 0..r {
                    acc[j] += pv[j];
                }
            }
            uc.copy_from_slice(&acc[..r]);
        }
        return;
    }
    for (cc, uc) in codes.chunks_exact(nb).zip(u.chunks_exact_mut(r)) {
        uc.fill(0.0);
        for (&code, &off) in cc.iter().zip(offs) {
            let at = off + code as usize * r;
            // SAFETY: as above.
            let pv = unsafe { p.get_unchecked(at..at + r) };
            for j in 0..r {
                uc[j] += pv[j];
            }
        }
    }
}

/// Per-column accumulator of a kernel instance: a local array (kept in registers) for const
/// widths, else `v` resized to `r`.
#[inline(always)]
fn regs<'a, const R: usize>(arr: &'a mut [f64; MAX_DISPATCH_WIDTH], v: &'a mut Vec<f64>, r: usize) -> &'a mut [f64] {
    if R > 0 && R <= MAX_DISPATCH_WIDTH {
        &mut arr[..R]
    } else {
        v.clear();
        v.resize(r, 0.0);
        &mut v[..]
    }
}

/// Adds z_c to the accumulator entries of the first `nacc` blocks.
///
/// # Safety
/// `cc` is a row of [`Structure::cell_codes`] of a structure `st` (`st.nb` wide), and
/// `st.codes_within(st.nb, offs, r, nacc, acc.len())` holds.
#[inline(always)]
unsafe fn scatter_cell<const R: usize>(cc: &[u32], nacc: usize, offs: &[usize], z: &[f64], r: usize, acc: &mut [f64]) {
    let r = width::<R>(r);
    let z = &z[..r];
    for (&code, &off) in cc[..nacc].iter().zip(offs) {
        let at = off + code as usize * r;
        // SAFETY: code < code_bound(b), so at + r <= offs[b] + code_bound(b)*r <= acc.len().
        let a = unsafe { acc.get_unchecked_mut(at..at + r) };
        for j in 0..r {
            a[j] += z[j];
        }
    }
}

/// Scatters z_c (in `u`, k x r) of the cells `cells` of a task: accumulated blocks into
/// `acc`, and the whole z_c into the task's piece of the M x r buffer when there is one.
///
/// # Safety
/// `codes` are rows of [`Structure::cell_codes`] of a structure `st`, and
/// `st.codes_within(nb, offs, r, t.nacc, t.acc.len())` holds.
#[inline(always)]
unsafe fn scatter_cells<const R: usize>(t: &mut Task, codes: &[u32], nb: usize, offs: &[usize], cells: std::ops::Range<usize>, u: &[f64], r: usize) {
    let r = width::<R>(r);
    for (cc, z) in codes.chunks_exact(nb.max(1)).zip(u.chunks_exact(r)) {
        // SAFETY: the precondition of this function.
        unsafe { scatter_cell::<R>(cc, t.nacc, offs, z, r, t.acc) };
    }
    if let Some(zt) = t.zt.as_deref_mut() {
        let at = (cells.start - t.c0) * r;
        zt[at..at + cells.len() * r].copy_from_slice(&u[..cells.len() * r]);
    }
}

/// Task body of [`Ops::apply`], with the block count as a constant (1..=3; 0 = runtime).
fn apply_task_nb<const R: usize, const UNIT: bool>(ops: &Ops, p: &[f64], r: usize, t: Task) {
    match ops.st.nb {
        1 => apply_task::<R, UNIT, 1>(ops, p, r, t),
        2 => apply_task::<R, UNIT, 2>(ops, p, r, t),
        3 => apply_task::<R, UNIT, 3>(ops, p, r, t),
        _ => apply_task::<R, UNIT, 0>(ops, p, r, t),
    }
}

fn apply_task<const R: usize, const UNIT: bool, const NB: usize>(ops: &Ops, p: &[f64], r: usize, t: Task) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: `has_avx2_fma()` verified AVX2 and FMA support on this CPU.
        unsafe { apply_task_avx2::<R, UNIT, NB>(ops, p, r, t) };
        return;
    }
    apply_task_inner::<R, UNIT, NB>(ops, p, r, t);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_task_avx2<const R: usize, const UNIT: bool, const NB: usize>(ops: &Ops, p: &[f64], r: usize, t: Task) {
    apply_task_inner::<R, UNIT, NB>(ops, p, r, t);
}

#[inline(always)]
fn apply_task_inner<const R: usize, const UNIT: bool, const NB: usize>(ops: &Ops, p: &[f64], r: usize, mut t: Task) {
    let r = width::<R>(r);
    let (st, ws) = (ops.st, ops.ws);
    let nb = width::<NB>(st.nb);
    let offs_v = block_offsets(st, r);
    let offs = &offs_v[..nb];
    // The unchecked gathers below read `p`, the scatters write the task's accumulator.
    assert!(st.codes_within(nb, offs, r, nb, p.len()) && st.codes_within(nb, offs, r, t.nacc, t.acc.len()));
    let cw: &[f64] = ws.cell_w().unwrap_or(&[]);
    let mut u: Vec<f64> = Vec::new();
    let (mut tarr, mut tvec) = ([0.0f64; MAX_DISPATCH_WIDTH], Vec::new());
    let tv = regs::<R>(&mut tarr, &mut tvec, r);
    for s in t.segs.clone() {
        let cells = st.cells_of(s);
        let k = cells.len();
        if u.len() < k * r {
            u.resize(k * r, 0.0);
        }
        let uk = &mut u[..k * r];
        let codes = &st.cell_codes()[cells.start * nb..cells.end * nb];
        // SAFETY: rows of `st.cell_codes()`, within `p` (asserted above).
        unsafe { gather_cells::<R>(codes, nb, offs, p, r, uk) };
        tv.fill(0.0);
        if UNIT {
            for uc in uk.chunks_exact(r) {
                for j in 0..r {
                    tv[j] += uc[j];
                }
            }
        } else {
            for (uc, &w) in uk.chunks_exact(r).zip(&cw[cells.clone()]) {
                for j in 0..r {
                    tv[j] += w * uc[j];
                }
            }
        }
        let inv = 1.0 / ws.n1[s];
        tv.iter_mut().for_each(|v| *v *= inv);
        if UNIT {
            for uc in uk.chunks_exact_mut(r) {
                for j in 0..r {
                    uc[j] -= tv[j];
                }
            }
        } else {
            for (uc, &w) in uk.chunks_exact_mut(r).zip(&cw[cells.clone()]) {
                for j in 0..r {
                    uc[j] = w * (uc[j] - tv[j]);
                }
            }
        }
        // SAFETY: rows of `st.cell_codes()`, within the accumulator (asserted above).
        unsafe { scatter_cells::<R>(&mut t, codes, nb, offs, cells, uk, r) };
    }
}

/// Task body of [`Ops::rhs`]: cell sums s_c = sum w v, b1 (per segment), sum w v^2 and sum w v
/// (`extra`), then z_c = s_c - W_c*b1/n1.
fn rhs_task<const R: usize, const UNIT: bool>(ops: &Ops, rows: &[f64], r: usize, t: Task) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: `has_avx2_fma()` verified AVX2 and FMA support on this CPU.
        unsafe { rhs_task_avx2::<R, UNIT>(ops, rows, r, t) };
        return;
    }
    rhs_task_inner::<R, UNIT>(ops, rows, r, t);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rhs_task_avx2<const R: usize, const UNIT: bool>(ops: &Ops, rows: &[f64], r: usize, t: Task) {
    rhs_task_inner::<R, UNIT>(ops, rows, r, t);
}

#[inline(always)]
fn rhs_task_inner<const R: usize, const UNIT: bool>(ops: &Ops, rows: &[f64], r: usize, mut t: Task) {
    let r = width::<R>(r);
    let (st, ws) = (ops.st, ops.ws);
    let nb = st.nb;
    let offs = block_offsets(st, r);
    // The unchecked scatters below write the task's accumulator.
    assert!(st.codes_within(nb, &offs, r, t.nacc, t.acc.len()));
    let cw: &[f64] = ws.cell_w().unwrap_or(&[]);
    let mut u: Vec<f64> = Vec::new();
    let (mut a1, mut v1, mut a2, mut v2, mut a3, mut v3) = (
        [0.0f64; MAX_DISPATCH_WIDTH],
        Vec::new(),
        [0.0f64; MAX_DISPATCH_WIDTH],
        Vec::new(),
        [0.0f64; MAX_DISPATCH_WIDTH],
        Vec::new(),
    );
    let esq = regs::<R>(&mut a1, &mut v1, r);
    let esum = regs::<R>(&mut a2, &mut v2, r);
    // b1 of the current segment, accumulated in registers and stored once.
    let b1s = regs::<R>(&mut a3, &mut v3, r);
    let s0 = t.segs.start;
    let seg_out = t.seg_out.take().expect("b1 slot");
    for s in t.segs.clone() {
        let cells = st.cells_of(s);
        let k = cells.len();
        if u.len() < k * r {
            u.resize(k * r, 0.0);
        }
        let uk = &mut u[..k * r];
        b1s.fill(0.0);
        if UNIT {
            let rv = &rows[cells.start * r..cells.end * r];
            for (sc, v) in uk.chunks_exact_mut(r).zip(rv.chunks_exact(r)) {
                for j in 0..r {
                    sc[j] = v[j];
                    esq[j] += v[j] * v[j];
                    esum[j] += v[j];
                }
                for j in 0..r {
                    b1s[j] += sc[j];
                }
            }
        } else if let (None, Some(wr)) = (&st.cell_obs_off, &ws.w_rows) {
            // One row per cell, weighted.
            let rv = &rows[cells.start * r..cells.end * r];
            for ((sc, v), &w) in uk.chunks_exact_mut(r).zip(rv.chunks_exact(r)).zip(&wr[cells.clone()]) {
                for j in 0..r {
                    let wv = w * v[j];
                    sc[j] = wv;
                    esq[j] += wv * v[j];
                    esum[j] += wv;
                }
                for j in 0..r {
                    b1s[j] += sc[j];
                }
            }
        } else {
            for (sc, c) in uk.chunks_exact_mut(r).zip(cells.clone()) {
                sc.fill(0.0);
                for row in st.rows_of_cell(c) {
                    let v = &rows[row * r..(row + 1) * r];
                    let w = ws.w_rows.as_ref().map_or(1.0, |w| w[row]);
                    for j in 0..r {
                        let wv = w * v[j];
                        sc[j] += wv;
                        esq[j] += wv * v[j];
                        esum[j] += wv;
                    }
                }
                for j in 0..r {
                    b1s[j] += sc[j];
                }
            }
        }
        seg_out[(s - s0) * r..(s - s0 + 1) * r].copy_from_slice(b1s);
        let inv = 1.0 / ws.n1[s];
        if UNIT {
            for sc in uk.chunks_exact_mut(r) {
                for j in 0..r {
                    sc[j] -= inv * b1s[j];
                }
            }
        } else {
            for (sc, &w) in uk.chunks_exact_mut(r).zip(&cw[cells.clone()]) {
                let w = w * inv;
                for j in 0..r {
                    sc[j] -= w * b1s[j];
                }
            }
        }
        let codes = &st.cell_codes()[cells.start * nb..cells.end * nb];
        // SAFETY: rows of `st.cell_codes()`, within the accumulator (asserted above).
        unsafe { scatter_cells::<R>(&mut t, codes, nb, &offs, cells, uk, r) };
    }
    t.extra[..r].copy_from_slice(esq);
    t.extra[r..].copy_from_slice(esum);
}

/// Task body of [`Ops::residualize_rows`] over the segments `segs` whose rows start at `row0`.
#[allow(clippy::too_many_arguments)]
fn residualize_task<const R: usize, const UNIT: bool>(
    ops: &Ops,
    rp: &mut [f64],
    row0: usize,
    r: usize,
    segs: std::ops::Range<usize>,
    x: &[f64],
    b1: &[f64],
    alpha: Option<&mut [f64]>,
) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: `has_avx2_fma()` verified AVX2 and FMA support on this CPU.
        unsafe { residualize_task_avx2::<R, UNIT>(ops, rp, row0, r, segs, x, b1, alpha) };
        return;
    }
    residualize_task_inner::<R, UNIT>(ops, rp, row0, r, segs, x, b1, alpha);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn residualize_task_avx2<const R: usize, const UNIT: bool>(
    ops: &Ops,
    rp: &mut [f64],
    row0: usize,
    r: usize,
    segs: std::ops::Range<usize>,
    x: &[f64],
    b1: &[f64],
    alpha: Option<&mut [f64]>,
) {
    residualize_task_inner::<R, UNIT>(ops, rp, row0, r, segs, x, b1, alpha);
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn residualize_task_inner<const R: usize, const UNIT: bool>(
    ops: &Ops,
    rp: &mut [f64],
    row0: usize,
    r: usize,
    segs: std::ops::Range<usize>,
    x: &[f64],
    b1: &[f64],
    mut alpha: Option<&mut [f64]>,
) {
    let r = width::<R>(r);
    let (st, ws) = (ops.st, ops.ws);
    let nb = st.nb;
    let offs = block_offsets(st, r);
    // The unchecked gathers below read `x`.
    assert!(st.codes_within(nb, &offs, r, nb, x.len()));
    let cw: &[f64] = ws.cell_w().unwrap_or(&[]);
    let mut u: Vec<f64> = Vec::new();
    let (mut aarr, mut avec) = ([0.0f64; MAX_DISPATCH_WIDTH], Vec::new());
    let a = regs::<R>(&mut aarr, &mut avec, r);
    let s0 = segs.start;
    for s in segs {
        let cells = st.cells_of(s);
        let k = cells.len();
        if u.len() < k * r {
            u.resize(k * r, 0.0);
        }
        let uk = &mut u[..k * r];
        let codes = &st.cell_codes()[cells.start * nb..cells.end * nb];
        // SAFETY: rows of `st.cell_codes()`, within `x` (asserted above).
        unsafe { gather_cells::<R>(codes, nb, &offs, x, r, uk) };
        a.copy_from_slice(&b1[s * r..(s + 1) * r]);
        if UNIT {
            for uc in uk.chunks_exact(r) {
                for j in 0..r {
                    a[j] -= uc[j];
                }
            }
        } else {
            for (uc, &w) in uk.chunks_exact(r).zip(&cw[cells.clone()]) {
                for j in 0..r {
                    a[j] -= w * uc[j];
                }
            }
        }
        let inv = 1.0 / ws.n1[s];
        a.iter_mut().for_each(|v| *v *= inv);
        if let Some(at) = alpha.as_deref_mut() {
            at[(s - s0) * r..(s - s0 + 1) * r].copy_from_slice(a);
        }
        if UNIT || st.cell_obs_off.is_none() {
            let rv = &mut rp[(cells.start - row0) * r..(cells.end - row0) * r];
            for (v, uc) in rv.chunks_exact_mut(r).zip(uk.chunks_exact_mut(r)) {
                for j in 0..r {
                    uc[j] += a[j];
                    v[j] -= uc[j];
                }
            }
        } else {
            for (uc, c) in uk.chunks_exact_mut(r).zip(cells) {
                for j in 0..r {
                    uc[j] += a[j];
                }
                for row in st.rows_of_cell(c) {
                    let v = &mut rp[(row - row0) * r..(row - row0 + 1) * r];
                    for j in 0..r {
                        v[j] -= uc[j];
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn pull_level_inner<const R: usize>(zbuf: &[f64], r: usize, cells: &[u32], out: &mut [f64]) {
    let r = width::<R>(r);
    if R > 0 && R <= MAX_DISPATCH_WIDTH {
        let mut acc = [0.0f64; MAX_DISPATCH_WIDTH];
        for &c in cells {
            let at = c as usize * r;
            let zv = &zbuf[at..at + r];
            for j in 0..r {
                acc[j] += zv[j];
            }
        }
        out.copy_from_slice(&acc[..r]);
    } else {
        out.fill(0.0);
        for &c in cells {
            let at = c as usize * r;
            for (o, &v) in out.iter_mut().zip(&zbuf[at..at + r]) {
                *o += v;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pull_level_avx2<const R: usize>(zbuf: &[f64], r: usize, cells: &[u32], out: &mut [f64]) {
    pull_level_inner::<R>(zbuf, r, cells, out);
}

fn pull_large_block<const R: usize, const _B: bool>(zbuf: &[f64], r: usize, yb: &mut [f64], perm: &[u32], perm_off: &[u32], pool: &Pool) {
    let r = width::<R>(r);
    let avx2 = has_avx2_fma();
    if yb.len() <= 512 * r || pool.threads() <= 1 {
        for (g, out) in yb.chunks_exact_mut(r).enumerate() {
            let cells = &perm[perm_off[g] as usize..perm_off[g + 1] as usize];
            #[cfg(target_arch = "x86_64")]
            if avx2 {
                unsafe { pull_level_avx2::<R>(zbuf, r, cells, out) };
                continue;
            }
            let _ = avx2;
            pull_level_inner::<R>(zbuf, r, cells, out);
        }
        return;
    }
    pool.install(|| {
        yb.par_chunks_mut(r).enumerate().with_min_len(512).for_each(|(g, out)| {
            let cells = &perm[perm_off[g] as usize..perm_off[g + 1] as usize];
            #[cfg(target_arch = "x86_64")]
            if avx2 {
                unsafe { pull_level_avx2::<R>(zbuf, r, cells, out) };
                return;
            }
            let _ = avx2;
            pull_level_inner::<R>(zbuf, r, cells, out);
        })
    });
}

impl Ops<'_> {
    /// Every cell weight is 1 (one row per cell, unweighted).
    fn unit(&self) -> bool {
        self.ws.cell_w().is_none()
    }

    /// Whether every block (small and large) is scattered through per-task accumulators for a
    /// block of `r` columns: true when the whole reduced vector is at most [`ACC_BYTES_MAX`], so
    /// the random accumulation stays cache resident. Otherwise large blocks go through the scratch
    /// `zbuf` (M x r) and level-sorted pulls.
    pub fn accumulates_all(&self, r: usize) -> bool {
        self.st.has_large() && self.st.m * r * 8 <= self.st.acc_bytes_max
    }

    /// Whether [`Ops::apply`] / [`Ops::rhs`] need the M x r scratch.
    pub fn needs_zbuf(&self, r: usize) -> bool {
        self.st.has_large() && !self.accumulates_all(r)
    }

    /// Segment boundaries of the scatter tasks for accumulators of `acc_len` values: with every
    /// block accumulated see [`acc_tasks`]; otherwise the fixed parts (grouped by
    /// [`Structure::tasks`]). Depends only on the data, `r` and `acc_len` (never on the thread
    /// count).
    fn scatter_tasks(&self, r: usize, acc_len: usize, all: bool) -> Vec<u32> {
        if all { acc_tasks(self.st, r, acc_len) } else { self.st.tasks(acc_len) }
    }

    /// Runs `task` over the scatter tasks and scatters their per-cell values into the reduced
    /// space `y`: accumulated blocks through per-task accumulators combined in task order, the
    /// other large blocks through the M x r buffer and level pulls (see
    /// [`Ops::accumulates_all`]). Tasks may also write per-segment outputs (`seg_out`, of the
    /// given width) and per-task sums (`extra_len`, combined in task order and returned).
    #[allow(clippy::too_many_arguments)]
    fn scatter(
        &self,
        r: usize,
        y: &mut [f64],
        sc: &mut Scratch,
        seg_out: Option<(&mut [f64], usize)>,
        extra_len: usize,
        pool: &Pool,
        task: impl Fn(Task) + Sync,
    ) -> Result<Vec<f64>, RfError> {
        let st = self.st;
        let all = self.accumulates_all(r);
        let use_z = self.needs_zbuf(r);
        let acc_len = if all { st.m * r } else { st.m_small * r };
        // Blocks scattered through the accumulators: small ones (first), or all of them.
        let nacc = if all { st.nb } else { st.blocks.iter().filter(|b| b.small).count() };
        let tb = self.scatter_tasks(r, acc_len, all);
        let cb = st.cell_bounds(&tb);
        let nt = tb.len() - 1;
        if use_z && sc.zbuf.len() < st.ncells * r {
            sc.zbuf = Vec::new();
            sc.zbuf = alloc_par(st.ncells * r, 0.0, pool)?;
        }
        if sc.accs.len() < nt {
            sc.accs.resize_with(nt, Vec::new);
        }
        if sc.dirty {
            sc.accs.iter_mut().for_each(|a| a.fill(0.0));
        }
        sc.dirty = true;
        let Scratch { zbuf, accs, dirty } = sc;
        let zp: Vec<Option<&mut [f64]>> = if use_z {
            let zb: Vec<usize> = cb.iter().map(|c| c * r).collect();
            split_bounds(&mut zbuf[..st.ncells * r], &zb).into_iter().map(Some).collect()
        } else {
            (0..nt).map(|_| None).collect()
        };
        let so: Vec<Option<&mut [f64]>> = match seg_out {
            Some((buf, width)) => {
                let sb: Vec<usize> = tb.iter().map(|&s| s as usize * width).collect();
                split_bounds(buf, &sb).into_iter().map(Some).collect()
            }
            None => (0..nt).map(|_| None).collect(),
        };
        let el = extra_len.max(1);
        let mut extras = vec![0.0; nt * el];
        pool.install(|| {
            accs[..nt].par_iter_mut().zip(zp).zip(so).zip(extras.par_chunks_mut(el)).enumerate().for_each(|(t, (((acc, zt), seg_out), ex))| {
                if acc.len() < acc_len {
                    acc.resize(acc_len, 0.0);
                }
                let segs = tb[t] as usize..tb[t + 1] as usize;
                task(Task { segs, c0: cb[t], acc: &mut acc[..acc_len], nacc, zt, seg_out, extra: &mut ex[..extra_len] });
            })
        });
        // Ordered combine, parallel over entries; it resets the accumulator entries it reads.
        let mut pieces: Vec<Vec<&mut [f64]>> = (0..acc_len.div_ceil(4096)).map(|_| Vec::with_capacity(nt)).collect();
        for acc in accs[..nt].iter_mut() {
            for (piece, ch) in pieces.iter_mut().zip(acc[..acc_len].chunks_mut(4096)) {
                piece.push(ch);
            }
        }
        pool.install(|| {
            y[..acc_len].par_chunks_mut(4096).zip(pieces).for_each(|(out, mut accs)| {
                out.iter_mut().for_each(|v| *v = 0.0);
                for acc in accs.iter_mut() {
                    for (o, v) in out.iter_mut().zip(acc.iter_mut()) {
                        *o += *v;
                        *v = 0.0;
                    }
                }
            })
        });
        *dirty = false;
        let mut extra = vec![0.0; extra_len];
        for ex in extras.chunks_exact(el) {
            for (a, v) in extra.iter_mut().zip(ex) {
                *a += v;
            }
        }
        if use_z {
            self.pull_large(&sc.zbuf, r, y, pool);
        }
        Ok(extra)
    }

    /// y[off_b + g] = sum_{c in level g of b} Z[c] for every large block.
    fn pull_large(&self, zbuf: &[f64], r: usize, y: &mut [f64], pool: &Pool) {
        let st = self.st;
        let ms = st.m_small;
        let mut rest = &mut y[ms * r..];
        for blk in st.blocks.iter().filter(|b| !b.small) {
            let (yb, tail) = rest.split_at_mut(blk.g * r);
            rest = tail;
            dispatch_width!(r, false, pull_large_block(zbuf, r, yb, &blk.perm, &blk.perm_off, pool));
        }
    }

    /// Y = S*P (both m x r row-major).
    pub fn apply(&self, p: &[f64], r: usize, y: &mut [f64], sc: &mut Scratch, pool: &Pool) -> Result<(), RfError> {
        let unit = self.unit();
        self.scatter(r, y, sc, None, 0, pool, |t| dispatch_width!(r, unit, apply_task_nb(self, p, r, t)))?;
        Ok(())
    }

    /// Right-hand side of the reduced system for the row block `rows` (N x r, row order).
    pub fn rhs(&self, rows: &[f64], r: usize, sc: &mut Scratch, pool: &Pool) -> Result<Rhs, RfError> {
        let st = self.st;
        let mut y = alloc_par(st.m * r, 0.0, pool)?;
        let mut b1 = alloc_par(st.g1 * r, 0.0, pool)?;
        let unit = self.unit();
        let extra = self.scatter(r, &mut y, sc, Some((&mut b1, r)), 2 * r, pool, |t| dispatch_width!(r, unit, rhs_task(self, rows, r, t)))?;
        let (wsq, wsum) = extra.split_at(r);
        Ok(Rhs { rhs: y, b1, wsq: wsq.to_vec(), wsum: wsum.to_vec() })
    }

    /// In-place residualization of `rows` given the reduced solution `x` (m x r) and b1:
    /// alpha1 = (b1 - sum_c W_c u_c)/n1 per segment and v_i -= alpha1 + u_c for every row (F is never
    /// materialized). Writes alpha1 (g1xr) to `alpha` when given.
    pub fn residualize_rows(&self, rows: &mut [f64], r: usize, x: &[f64], b1: &[f64], alpha: Option<&mut [f64]>, pool: &Pool) {
        let st = self.st;
        let tb = &st.parts;
        let cb = st.cell_bounds(tb);
        let rb: Vec<usize> = cb.iter().map(|&c| st.cell_row(c) * r).collect();
        let row_pieces = split_bounds(&mut rows[..st.n * r], &rb);
        let ap: Vec<Option<&mut [f64]>> = match alpha {
            Some(a) => {
                let ab: Vec<usize> = tb.iter().map(|&s| s as usize * r).collect();
                split_bounds(a, &ab).into_iter().map(Some).collect()
            }
            None => (0..tb.len() - 1).map(|_| None).collect(),
        };
        let unit = self.unit();
        if row_pieces.len() <= 1 || pool.threads() <= 1 {
            row_pieces.into_iter().zip(ap).enumerate().for_each(|(t, (rp, at))| {
                let segs = tb[t] as usize..tb[t + 1] as usize;
                dispatch_width!(r, unit, residualize_task(self, rp, rb[t] / r, r, segs, x, b1, at));
            });
            return;
        }
        pool.install(|| {
            row_pieces.into_par_iter().zip(ap).enumerate().for_each(|(t, (rp, at))| {
                let segs = tb[t] as usize..tb[t + 1] as usize;
                dispatch_width!(r, unit, residualize_task(self, rp, rb[t] / r, r, segs, x, b1, at));
            })
        });
    }

    /// FE fit per row for one coefficient column: alpha1[g1] + sum_b x_b[g_b] (`x` m-vector, `alpha` g1).
    pub fn fit_rows(&self, x: &[f64], alpha: &[f64], pool: &Pool) -> Vec<f64> {
        let st = self.st;
        let mut out = alloc_uninit::<f64>(st.n).expect("fit_rows allocation");
        let tb = &st.parts;
        let cb = st.cell_bounds(tb);
        let rb: Vec<usize> = cb.iter().map(|&c| st.cell_row(c)).collect();
        let pieces = split_bounds(&mut out, &rb);
        let fill = |(t, piece): (usize, &mut [MaybeUninit<f64>])| {
            for s in tb[t] as usize..tb[t + 1] as usize {
                for c in st.cells_of(s) {
                    let mut f = alpha[s];
                    for (b, &code) in st.codes(c).iter().enumerate() {
                        f += x[st.blocks[b].off + code as usize];
                    }
                    for row in st.rows_of_cell(c) {
                        piece[row - rb[t]].write(f);
                    }
                }
            }
        };
        if pieces.len() <= 1 || pool.threads() <= 1 {
            pieces.into_iter().enumerate().for_each(fill);
        } else {
            pool.install(|| pieces.into_par_iter().enumerate().for_each(fill));
        }
        // SAFETY: every row belongs to one cell of one segment in `0..st.g1`, written above.
        unsafe { assume_init(out) }
    }

    /// [`Ops::assemble_small`] for small blocks dense within the segments: the cell-weight
    /// cross counts per cell, and sum_s c_s c_s^T/n1_s as dense products of batches of columns
    /// c_s/sqrt n1_s ([`sub_gram_lower`] per batch, tasks combined in order: thread-count invariant).
    fn assemble_small_gemm(&self, pool: &Pool) -> Result<Vec<f64>, RfError> {
        let (st, ws) = (self.st, self.ws);
        let (ms, nb) = (st.m_small, st.nb);
        // Columns per product: >= 128, and ~= 4K entries (L1-resident) for tiny blocks.
        let batch = (4096 / ms.max(1)).max(128);
        let ns = st.blocks.iter().filter(|b| b.small).count();
        let offs = block_offsets(st, 1);
        let flops = (ms * ms * st.g1) as u64;
        let nt = ((st.ncells / CHUNK) as u64).max(flops >> 22).min(flops / (8 * (ms * ms) as u64)).clamp(1, 64);
        let tb = st.even_tasks(nt as usize);
        let accs: Vec<Vec<f64>> = pool.install(|| {
            (0..tb.len() - 1)
                .into_par_iter()
                .map(|t| {
                    let mut acc = vec![0.0; ms * ms];
                    let (s0, s1) = (tb[t] as usize, tb[t + 1] as usize);
                    let ld = batch.min(s1 - s0);
                    let mut cbuf = vec![0.0; ms * ld];
                    let mut b0 = s0;
                    while b0 < s1 {
                        let b1 = (b0 + batch).min(s1);
                        cbuf.fill(0.0);
                        for s in b0..b1 {
                            let j = s - b0;
                            for c in st.cells_of(s) {
                                let w = ws.cw(c);
                                let cc = &st.cell_codes()[c * nb..c * nb + ns];
                                for (&code, &off) in cc.iter().zip(&offs) {
                                    let k = off + code as usize;
                                    cbuf[k * ld + j] += w;
                                    let row = &mut acc[k * ms..(k + 1) * ms];
                                    for (&code2, &off2) in cc.iter().zip(&offs) {
                                        row[off2 + code2 as usize] += w;
                                    }
                                }
                            }
                            let f = 1.0 / ws.n1[s].sqrt();
                            for k in 0..ms {
                                cbuf[k * ld + j] *= f;
                            }
                        }
                        sub_gram_lower(&mut acc, ms, &cbuf, ld, b1 - b0);
                        b0 = b1;
                    }
                    acc
                })
                .collect()
        });
        let mut full = alloc_vec(ms * ms, 0.0)?;
        pool.install(|| {
            full.par_chunks_mut(ms).enumerate().for_each(|(k, row)| {
                for acc in &accs {
                    for (a, v) in row[..=k].iter_mut().zip(&acc[k * ms..k * ms + k + 1]) {
                        *a += v;
                    }
                }
            })
        });
        for k in 0..ms {
            for l in 0..k {
                full[l * ms + k] = full[k * ms + l];
            }
        }
        Ok(full)
    }

    /// Dense S restricted to the small coordinates (m_s x m_s, row-major).
    pub fn assemble_small(&self, pool: &Pool) -> Result<Vec<f64>, RfError> {
        let (st, ws) = (self.st, self.ws);
        let ms = st.m_small;
        if gemm_small(st) {
            return self.assemble_small_gemm(pool);
        }
        let (nb, ns) = (st.nb, st.blocks.iter().filter(|b| b.small).count());
        let offs = block_offsets(st, 1);
        // Packed lower triangle: row k starts at tri(k).
        let tri = |k: usize| k * (k + 1) / 2;
        let nrb = tri(ms).div_ceil(1 << 21).max(1);
        let rows_per = ms.div_ceil(nrb);
        // Segment tasks: enough to parallelize the sum s^2 outer products, few enough that zeroing
        // and combining the per-task accumulators (~= tri(m_s)/nrb each) stays <= 1/8 of that work.
        let nt_want = (st.small_work / (8 * (tri(ms) / nrb) as u64).max(1)).clamp(1, 64) as usize;
        let tb = st.even_tasks(nt_want);
        let nt = tb.len() - 1;
        let jobs: Vec<(usize, usize)> = (0..nrb).flat_map(|rbk| (0..nt).map(move |t| (rbk, t))).collect();
        let accs: Vec<Vec<f64>> = pool.install(|| {
            jobs.par_iter()
                .map(|&(rbk, t)| {
                    let k0 = rbk * rows_per;
                    let k1 = ((rbk + 1) * rows_per).min(ms);
                    let base = tri(k0);
                    let mut acc = vec![0.0; tri(k1) - base];
                    // Per-segment level sums: a bitmap of the touched coordinates, read back in
                    // ascending order (and cleared) after the segment, and their sums.
                    let mut bits = vec![0u64; ms.div_ceil(64)];
                    let mut cv = vec![0.0; ms];
                    let mut touched: Vec<u32> = Vec::new();
                    let mut tvals: Vec<f64> = Vec::new();
                    for s in tb[t] as usize..tb[t + 1] as usize {
                        let (mut wlo, mut whi) = (usize::MAX, 0usize);
                        for c in st.cells_of(s) {
                            let w = ws.cw(c);
                            let cc = &st.cell_codes()[c * nb..c * nb + ns];
                            for (&code, &off) in cc.iter().zip(&offs) {
                                let kb = off + code as usize;
                                let (wi, bit) = (kb >> 6, 1u64 << (kb & 63));
                                if bits[wi] & bit == 0 {
                                    bits[wi] |= bit;
                                    cv[kb] = 0.0;
                                    wlo = wlo.min(wi);
                                    whi = whi.max(wi);
                                }
                                cv[kb] += w;
                                if kb >= k0 && kb < k1 {
                                    let row = tri(kb) - base;
                                    for (&code2, &off2) in cc.iter().zip(&offs) {
                                        let l = off2 + code2 as usize;
                                        if l <= kb {
                                            acc[row + l] += w;
                                        }
                                    }
                                }
                            }
                        }
                        let inv = 1.0 / ws.n1[s];
                        // Ascending, the coordinates l <= k of row k are a prefix; their values
                        // are gathered once.
                        touched.clear();
                        tvals.clear();
                        for wi in wlo..=whi {
                            let mut x = std::mem::take(&mut bits[wi]);
                            while x != 0 {
                                let l = wi * 64 + x.trailing_zeros() as usize;
                                touched.push(l as u32);
                                tvals.push(cv[l]);
                                x &= x - 1;
                            }
                        }
                        for (i, &k) in touched.iter().enumerate() {
                            let k = k as usize;
                            if k < k0 || k >= k1 {
                                continue;
                            }
                            let row = &mut acc[tri(k) - base..tri(k) - base + k + 1];
                            let f = tvals[i] * inv;
                            for (&l, &cl) in touched[..=i].iter().zip(&tvals[..=i]) {
                                row[l as usize] -= f * cl;
                            }
                        }
                    }
                    acc
                })
                .collect()
        });
        // Combine in task order (parallel over rows), then mirror the lower triangle.
        let mut full = alloc_vec(ms * ms, 0.0)?;
        pool.install(|| {
            full.par_chunks_mut(ms).enumerate().for_each(|(k, row)| {
                let rbk = k / rows_per;
                let at = tri(k) - tri(rbk * rows_per);
                for t in 0..nt {
                    let acc = &accs[rbk * nt + t];
                    for (a, v) in row[..=k].iter_mut().zip(&acc[at..at + k + 1]) {
                        *a += v;
                    }
                }
            })
        });
        for k in 0..ms {
            for l in 0..k {
                full[l * ms + k] = full[k * ms + l];
            }
        }
        Ok(full)
    }

    /// Per-segment aggregated codes of block `b`.
    pub fn seg_agg(&self, b: usize, pool: &Pool) -> Result<SegAgg, RfError> {
        let (st, ws) = (self.st, self.ws);
        self.agg_segments(pool, |c, out| out.push((st.codes(c)[b], ws.cw(c))))
    }

    /// Per-segment aggregated reduced coordinates of all blocks (`codes` are global coordinates,
    /// ascending since block offsets increase with the block index).
    pub fn joint_seg_agg(&self, pool: &Pool) -> Result<SegAgg, RfError> {
        let (st, ws) = (self.st, self.ws);
        self.agg_segments(pool, |c, out| {
            let w = ws.cw(c);
            for (blk, &code) in st.blocks.iter().zip(st.codes(c)) {
                out.push(((blk.off as u32) + code, w));
            }
        })
    }

    /// Structural (weight-independent) joint assembly statistics without building the lists of
    /// [`Ops::joint_seg_agg`]: (sum_s d_s^2, sum_s d_s*l_s), with d_s the distinct reduced coordinates
    /// of segment s and l_s those of large blocks. Integer sums: any partition gives the same.
    pub fn joint_seg_counts(&self, pool: &Pool) -> (u64, u64) {
        let st = self.st;
        let (m, ms, nb) = (st.m, st.m_small, st.nb);
        let offs = block_offsets(st, 1);
        pool.install(|| {
            (0..st.nparts())
                .into_par_iter()
                .map_init(
                    || vec![u32::MAX; m],
                    |stamp, p| {
                        let (mut sq, mut dl) = (0u64, 0u64);
                        for s in st.parts[p] as usize..st.parts[p + 1] as usize {
                            let (mut d, mut l) = (0u64, 0u64);
                            let cells = st.cells_of(s);
                            for cc in st.cell_codes()[cells.start * nb..cells.end * nb].chunks_exact(nb.max(1)) {
                                for (&code, &off) in cc.iter().zip(&offs) {
                                    let k = off + code as usize;
                                    if stamp[k] != s as u32 {
                                        stamp[k] = s as u32;
                                        d += 1;
                                        l += u64::from(k >= ms);
                                    }
                                }
                            }
                            sq += d * d;
                            dl += d * l;
                        }
                        (sq, dl)
                    },
                )
                .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
        })
    }

    /// Per segment: the distinct keys pushed by `push` for its cells (ascending) with their summed
    /// weights, in segment order (fixed parts; thread-count invariant).
    fn agg_segments(&self, pool: &Pool, push: impl Fn(usize, &mut Vec<(u32, f64)>) + Sync) -> Result<SegAgg, RfError> {
        let st = self.st;
        let lists: Vec<(Vec<u32>, Vec<u32>, Vec<f64>)> = pool.install(|| {
            (0..st.nparts())
                .into_par_iter()
                .map(|p| {
                    let (mut cnt, mut codes, mut sums) = (Vec::new(), Vec::new(), Vec::new());
                    let mut tmp: Vec<(u32, f64)> = Vec::new();
                    for s in st.parts[p] as usize..st.parts[p + 1] as usize {
                        tmp.clear();
                        for c in st.cells_of(s) {
                            push(c, &mut tmp);
                        }
                        tmp.sort_unstable_by_key(|e| e.0);
                        let before = codes.len();
                        for &(g, w) in &tmp {
                            if codes.len() > before && *codes.last().expect("nonempty") == g {
                                *sums.last_mut().expect("nonempty") += w;
                            } else {
                                codes.push(g);
                                sums.push(w);
                            }
                        }
                        cnt.push((codes.len() - before) as u32);
                    }
                    (cnt, codes, sums)
                })
                .collect()
        });
        let mut off = alloc_vec(st.g1 + 1, 0u32)?;
        let mut part_bounds = Vec::with_capacity(lists.len() + 1);
        part_bounds.push(0usize);
        let mut s = 0usize;
        let mut work = 0u64;
        let mut pos = 0usize;
        for (cnt, c, _) in &lists {
            for &k in cnt {
                off[s + 1] = off[s] + k;
                work += u64::from(k) * u64::from(k);
                s += 1;
            }
            pos += c.len();
            part_bounds.push(pos);
        }
        let total = pos;
        let mut codes = alloc_uninit::<u32>(total)?;
        let mut sums = alloc_uninit::<f64>(total)?;
        if total <= CHUNK || pool.threads() <= 1 {
            for (p, (_, c, w)) in lists.iter().enumerate() {
                let b0 = part_bounds[p];
                for (i, (&cv, &wv)) in c.iter().zip(w).enumerate() {
                    codes[b0 + i].write(cv);
                    sums[b0 + i].write(wv);
                }
            }
        } else {
            let cp = split_bounds(&mut codes, &part_bounds);
            let sp = split_bounds(&mut sums, &part_bounds);
            pool.install(|| {
                cp.into_par_iter().zip(sp).zip(&lists).for_each(|((co, so), (_, c, w))| {
                    for ((dc, ds), (&cv, &wv)) in co.iter_mut().zip(so.iter_mut()).zip(c.iter().zip(w)) {
                        dc.write(cv);
                        ds.write(wv);
                    }
                })
            });
        }
        // SAFETY: `part_bounds` partitions `0..total`, and every slice was written above.
        Ok(SegAgg { off, codes: unsafe { assume_init(codes) }, sums: unsafe { assume_init(sums) }, work })
    }

    /// Block of every reduced coordinate and segment of every cell (structural; for the joint
    /// assembly, whose rows of large coordinates walk the blocks' level-sorted cell lists).
    pub fn joint_cells(&self, pool: &Pool) -> Result<JointCells, RfError> {
        let st = self.st;
        let mut coord_block = alloc_vec(st.m, 0u32)?;
        for (b, blk) in st.blocks.iter().enumerate() {
            coord_block[blk.off..blk.off + blk.g].iter_mut().for_each(|v| *v = b as u32);
        }
        Ok(JointCells { coord_block, cell_seg: st.cell_segments(pool)? })
    }

    /// Structural nonzeros of the rows of large coordinates of the joint S (diagonal included):
    /// the union of the coordinates of every segment containing the row's coordinate (cell
    /// co-occurrences are a subset); 0 for small coordinates. Also returns the total nonzeros of
    /// the full joint matrix (small rows mirror the small columns of the large rows, plus a dense
    /// bound m_s^2 for the small-small block) and of the large-coordinate submatrix (AMD input).
    /// Cost ~= sum_s d_s*(d_s - small codes of s).
    pub fn joint_row_nnz(&self, agg: &SegAgg, jc: &JointCells, pool: &Pool) -> (Vec<u32>, u64, u64) {
        let (m, ms) = (self.st.m, self.st.m_small);
        let scratch = RowScratch::new(m, pool);
        let counts: Vec<(u32, u32)> = pool.install(|| {
            (0..m)
                .into_par_iter()
                .map(|row| {
                    if row < ms {
                        return (0, 0);
                    }
                    scratch.with(|_, mark, _| {
                        let (mut cnt, mut cnt_small) = (0u32, 0u32);
                        let mut last = u32::MAX;
                        for &c in jc.cells(self.st, row) {
                            let s = jc.cell_seg[c as usize];
                            if s == last {
                                continue;
                            }
                            last = s;
                            for &h in agg.seg(s as usize).0 {
                                if mark[h as usize] != row as u32 {
                                    mark[h as usize] = row as u32;
                                    cnt += 1;
                                    cnt_small += u32::from((h as usize) < ms);
                                }
                            }
                        }
                        (cnt, cnt_small)
                    })
                })
                .collect()
        });
        let all: u64 = counts.iter().map(|c| u64::from(c.0)).sum();
        let small: u64 = counts.iter().map(|c| u64::from(c.1)).sum();
        let total = all + small + (ms * ms) as u64;
        (counts.into_iter().map(|c| c.0).collect(), total, all - small)
    }

    /// Exact joint S over the kept coordinates `keep_idx` (global, ascending) of all blocks, as a
    /// full CSR: S_kl = sum_c W_c [k, l in codes(c)] - sum_s c_k(s) c_l(s) / n1(s), i.e. the cell
    /// co-occurrence weights A_RR (diagonal n_k) minus the FE1 elimination term. The diagonal is
    /// computed directly (the Laplacian row-sum identity does not hold across blocks).
    ///
    /// Rows of large coordinates are accumulated row by row in a fixed order (cells ascending).
    /// Rows of small coordinates are not walked (they touch almost every segment): their
    /// small-small part is the dense S_SS ([`Ops::assemble_small`]) and their small-large part
    /// mirrors the small columns of the large rows (exact symmetry). Values never depend on the
    /// thread count. `row_nnz` are the capacities from [`Ops::joint_row_nnz`].
    pub fn assemble_joint(&self, agg: &SegAgg, jc: &JointCells, row_nnz: &[u32], keep_idx: &[u32], pool: &Pool) -> Result<Csr, RfError> {
        let (st, ws) = (self.st, self.ws);
        let (m, ms) = (st.m, st.m_small);
        let mut newid = vec![u32::MAX; m];
        for (k, &l) in keep_idx.iter().enumerate() {
            newid[l as usize] = k as u32;
        }
        let nk = keep_idx.len();
        let sk = keep_idx.partition_point(|&k| (k as usize) < ms);
        let large = &keep_idx[sk..];
        let nl = large.len();
        let mut lptr = alloc_vec(nl + 1, 0u32)?;
        for (i, &l) in large.iter().enumerate() {
            lptr[i + 1] = lptr[i] + row_nnz[l as usize];
        }
        let cap = lptr[nl] as usize;
        let mut lcol = alloc_uninit::<u32>(cap)?;
        let mut lval = alloc_uninit::<f64>(cap)?;
        let mut lused = alloc_vec(nl, 0u32)?;
        let scratch = RowScratch::new(m, pool);
        {
            let bounds: Vec<usize> = lptr.iter().map(|&p| p as usize).collect();
            let cp = split_bounds(&mut lcol, &bounds);
            let vp = split_bounds(&mut lval, &bounds);
            pool.install(|| {
                cp.into_par_iter().zip(vp).zip(lused.par_iter_mut()).enumerate().for_each(|(k, ((cols, vals), used))| {
                    scratch.with(|acc, mark, touched| {
                        let row = large[k] as usize;
                        touched.clear();
                        let mut add = |h: usize, v: f64| {
                            if mark[h] != row as u32 {
                                mark[h] = row as u32;
                                acc[h] = 0.0;
                                touched.push(h as u32);
                            }
                            acc[h] += v;
                        };
                        // Cells of the row's coordinate, grouped by segment (cells ascending).
                        let cells = jc.cells(st, row);
                        let mut i = 0;
                        while i < cells.len() {
                            let s = jc.cell_seg[cells[i] as usize];
                            let mut ck = 0.0;
                            while i < cells.len() && jc.cell_seg[cells[i] as usize] == s {
                                let c = cells[i] as usize;
                                let w = ws.cw(c);
                                ck += w;
                                // A_RR: every coordinate of the cell (the row's own gives n_k).
                                for (blk, &code) in st.blocks.iter().zip(st.codes(c)) {
                                    add(blk.off + code as usize, w);
                                }
                                i += 1;
                            }
                            let f = ck / ws.n1[s as usize];
                            let (hs, cs) = agg.seg(s as usize);
                            for (&h, &ch) in hs.iter().zip(cs) {
                                add(h as usize, -f * ch);
                            }
                        }
                        // Columns ascending (kept ids preserve the global order).
                        touched.sort_unstable();
                        let mut i = 0;
                        for &h in touched.iter() {
                            let hn = newid[h as usize];
                            if hn != u32::MAX {
                                cols[i].write(hn);
                                vals[i].write(acc[h as usize]);
                                i += 1;
                            }
                        }
                        *used = i as u32;
                    })
                })
            });
        }
        let lrow_col = |r: usize| -> &[u32] {
            let (a, u) = (lptr[r] as usize, lused[r] as usize);
            // SAFETY: row `r` wrote `0..u` at offset `a`.
            unsafe { std::slice::from_raw_parts(lcol.as_ptr().cast::<u32>().add(a), u) }
        };
        let lrow_val = |r: usize| -> &[f64] {
            let (a, u) = (lptr[r] as usize, lused[r] as usize);
            // SAFETY: row `r` wrote `0..u` at offset `a`.
            unsafe { std::slice::from_raw_parts(lval.as_ptr().cast::<f64>().add(a), u) }
        };
        if sk == 0 && (0..nl).all(|r| lptr[r] + lused[r] == lptr[r + 1]) {
            // SAFETY: every row filled its entire structural slice (`lused[r] == row_nnz`), so all
            // `0..cap` entries of `lcol` and `lval` are initialized.
            return Ok(Csr { n: nk, ptr: lptr, col: unsafe { assume_init(lcol) }, val: unsafe { assume_init(lval) } });
        }
        // Small rows: dense S_SS over the kept small coordinates, then the mirrored small
        // columns of the large rows (large rows ascending, so columns stay sorted).
        let dense = if sk > 0 { self.assemble_small(pool)? } else { Vec::new() };
        let small_len = |r: usize| lrow_col(r).partition_point(|&c| (c as usize) < sk);
        let mut ptr = alloc_vec(nk + 1, 0u32)?;
        let mut cnt = alloc_vec(sk, 0u32)?;
        for i in 0..sk {
            let gi = keep_idx[i] as usize;
            cnt[i] = (0..sk).filter(|&j| j == i || dense[gi * ms + keep_idx[j] as usize] != 0.0).count() as u32;
        }
        let mut slens = Vec::new();
        if sk > 0 {
            slens.reserve_exact(nl);
            for r in 0..nl {
                let sl = small_len(r);
                slens.push(sl as u32);
                for &c in &lrow_col(r)[..sl] {
                    cnt[c as usize] += 1;
                }
            }
        }
        for i in 0..sk {
            ptr[i + 1] = ptr[i] + cnt[i];
        }
        for r in 0..nl {
            ptr[sk + r + 1] = ptr[sk + r] + lused[r];
        }
        let total = ptr[nk] as usize;
        let small_total = ptr[sk] as usize;
        let mut col = alloc_uninit::<u32>(total)?;
        let mut val = alloc_uninit::<f64>(total)?;
        {
            let (scol, lcol_out) = col.split_at_mut(small_total);
            let (sval, lval_out) = val.split_at_mut(small_total);
            let fill_small = |scol: &mut [MaybeUninit<u32>], sval: &mut [MaybeUninit<f64>]| {
                if sk == 0 {
                    return;
                }
                let mut cur: Vec<usize> = ptr[..sk].iter().map(|&p| p as usize).collect();
                for i in 0..sk {
                    let gi = keep_idx[i] as usize;
                    for j in 0..sk {
                        let v = dense[gi * ms + keep_idx[j] as usize];
                        if j == i || v != 0.0 {
                            scol[cur[i]].write(j as u32);
                            sval[cur[i]].write(v);
                            cur[i] += 1;
                        }
                    }
                }
                for r in 0..nl {
                    let sl = slens[r] as usize;
                    let (rc, rv) = (lrow_col(r), lrow_val(r));
                    for e in 0..sl {
                        let i = rc[e] as usize;
                        scol[cur[i]].write((sk + r) as u32);
                        sval[cur[i]].write(rv[e]);
                        cur[i] += 1;
                    }
                }
            };
            if total - small_total <= CHUNK || pool.threads() <= 1 || nl < 512 {
                fill_small(scol, sval);
                for r in 0..nl {
                    let b = ptr[sk + r] as usize - small_total;
                    for (k, (&c, &v)) in lrow_col(r).iter().zip(lrow_val(r)).enumerate() {
                        lcol_out[b + k].write(c);
                        lval_out[b + k].write(v);
                    }
                }
            } else {
                let step = 512;
                let chunks: Vec<usize> = (0..=nl.div_ceil(step)).map(|c| (c * step).min(nl)).collect();
                let bounds: Vec<usize> = chunks.iter().map(|&r| ptr[sk + r] as usize - small_total).collect();
                let cp = split_bounds(lcol_out, &bounds);
                let vp = split_bounds(lval_out, &bounds);
                pool.install(|| {
                    rayon::join(
                        || fill_small(scol, sval),
                        || {
                            cp.into_par_iter().zip(vp).enumerate().for_each(|(ci, (co, vo))| {
                                let base = bounds[ci];
                                for r in chunks[ci]..chunks[ci + 1] {
                                    let b = ptr[sk + r] as usize - small_total - base;
                                    for (k, (&c, &v)) in lrow_col(r).iter().zip(lrow_val(r)).enumerate() {
                                        co[b + k].write(c);
                                        vo[b + k].write(v);
                                    }
                                }
                            });
                        },
                    );
                });
            }
        }
        // SAFETY: `0..small_total` was filled by `fill_small` (`cur[i]` advances by `cnt[i]`), and
        // `small_total..total` was filled by each large row `r` (`lused[r]` entries).
        Ok(Csr { n: nk, ptr, col: unsafe { assume_init(col) }, val: unsafe { assume_init(val) } })
    }

    /// Number of structural nonzeros of each row of S_bb (diagonal included). Cost ~= E_b.
    pub fn block_row_nnz(&self, b: usize, agg: &SegAgg, cell_seg: &[u32], pool: &Pool) -> Vec<u32> {
        let blk = &self.st.blocks[b];
        let g = blk.g;
        pool.install(|| {
            (0..g)
                .into_par_iter()
                .with_min_len(256)
                .map_init(
                    || vec![u32::MAX; g],
                    |mark, row| {
                        let mut cnt = 0u32;
                        let mut last = u32::MAX;
                        for &c in &blk.perm[blk.perm_off[row] as usize..blk.perm_off[row + 1] as usize] {
                            let s = cell_seg[c as usize];
                            if s == last {
                                continue;
                            }
                            last = s;
                            for &h in agg.seg(s as usize).0 {
                                if mark[h as usize] != row as u32 {
                                    mark[h as usize] = row as u32;
                                    cnt += 1;
                                }
                            }
                        }
                        cnt
                    },
                )
                .collect()
        })
    }

    /// Exact S_bb (Laplacian) over the non-redundant levels of block `b`, as a full CSR whose
    /// diagonal is minus the sum of the row's off-diagonals. `keep_idx` maps kept index -> level.
    pub fn assemble_block(&self, b: usize, agg: &SegAgg, cell_seg: &[u32], row_nnz: &[u32], keep_idx: &[u32], pool: &Pool) -> Result<Csr, RfError> {
        let (st, ws) = (self.st, self.ws);
        let blk = &st.blocks[b];
        let g = blk.g;
        let mut newid = vec![u32::MAX; g];
        for (k, &l) in keep_idx.iter().enumerate() {
            newid[l as usize] = k as u32;
        }
        let nk = keep_idx.len();
        let mut ptr = alloc_vec(nk + 1, 0u32)?;
        for (k, &l) in keep_idx.iter().enumerate() {
            ptr[k + 1] = ptr[k] + row_nnz[l as usize];
        }
        let cap = ptr[nk] as usize;
        let mut col = alloc_vec(cap, 0u32)?;
        let mut val = alloc_vec(cap, 0.0)?;
        let mut used = alloc_vec(nk, 0u32)?;
        {
            let bounds: Vec<usize> = ptr.iter().map(|&p| p as usize).collect();
            let cp = split_bounds(&mut col, &bounds);
            let vp = split_bounds(&mut val, &bounds);
            pool.install(|| {
                cp.into_par_iter().zip(vp).zip(used.par_iter_mut()).enumerate().with_min_len(64).for_each_init(
                    || (vec![0.0f64; g], vec![u32::MAX; g], Vec::<u32>::new()),
                    |(acc, mark, touched), (k, ((cols, vals), used))| {
                        let row = keep_idx[k] as usize;
                        touched.clear();
                        let mut last = u32::MAX;
                        for &c in &blk.perm[blk.perm_off[row] as usize..blk.perm_off[row + 1] as usize] {
                            let s = cell_seg[c as usize];
                            if s == last {
                                continue;
                            }
                            last = s;
                            let (hs, cs) = agg.seg(s as usize);
                            let pos = hs.binary_search(&(row as u32)).expect("row level in segment");
                            let f = cs[pos] / ws.n1[s as usize];
                            for (&h, &ch) in hs.iter().zip(cs) {
                                if h as usize == row || newid[h as usize] == u32::MAX {
                                    continue;
                                }
                                if mark[h as usize] != k as u32 {
                                    mark[h as usize] = k as u32;
                                    acc[h as usize] = 0.0;
                                    touched.push(h);
                                }
                                acc[h as usize] -= f * ch;
                            }
                        }
                        touched.sort_unstable();
                        let mut d = 0.0;
                        for &h in touched.iter() {
                            d -= acc[h as usize];
                        }
                        let me = k as u32;
                        let mut i = 0;
                        let mut placed = false;
                        for &h in touched.iter() {
                            let hn = newid[h as usize];
                            if !placed && hn > me {
                                cols[i] = me;
                                vals[i] = d;
                                i += 1;
                                placed = true;
                            }
                            cols[i] = hn;
                            vals[i] = acc[h as usize];
                            i += 1;
                        }
                        if !placed {
                            cols[i] = me;
                            vals[i] = d;
                            i += 1;
                        }
                        *used = i as u32;
                    },
                )
            });
        }
        compact_csr(nk, &ptr, col, val, &used)
    }

    /// Clique-sampled Laplacian L~_bb (Kyng-Sachdeva clique trees per FE1 segment, plan D5) over
    /// the non-redundant levels (`keep_idx`), as a full CSR.
    pub fn sample_block(&self, agg: &SegAgg, g: usize, keep_idx: &[u32], pool: &Pool) -> Result<Csr, RfError> {
        let st = self.st;
        let mut newid = vec![u32::MAX; g];
        for (k, &l) in keep_idx.iter().enumerate() {
            newid[l as usize] = k as u32;
        }
        let parts: Vec<Vec<(u64, f64)>> = pool.install(|| {
            (0..st.nparts())
                .into_par_iter()
                .map(|p| {
                    let mut edges = Vec::new();
                    let mut nb: Vec<(f64, u32)> = Vec::new();
                    let mut csum: Vec<f64> = Vec::new();
                    for s in st.parts[p] as usize..st.parts[p + 1] as usize {
                        let (hs, cs) = agg.seg(s);
                        if hs.len() < 2 {
                            continue;
                        }
                        nb.clear();
                        nb.extend(cs.iter().zip(hs).map(|(&w, &h)| (w, h)));
                        nb.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                        csum.clear();
                        let mut acc = 0.0;
                        for e in nb.iter() {
                            acc += e.0;
                            csum.push(acc);
                        }
                        let mut rng = SplitMix64::stream(0x5EED, s as u64);
                        let mut cap = acc;
                        let mut scale = 1.0;
                        let d = nb.len();
                        for j in 0..d - 1 {
                            let f = nb[j].0 * scale / cap;
                            let lo = csum[j];
                            let target = lo + rng.next_f64() * (csum[d - 1] - lo);
                            let mut k = j + 1 + csum[j + 1..].partition_point(|&v| v < target);
                            if k >= d {
                                k = d - 1;
                            }
                            let w = f * (1.0 - f) * cap;
                            let (a, b) = (newid[nb[j].1 as usize], newid[nb[k].1 as usize]);
                            if w > 0.0 && a != u32::MAX && b != u32::MAX {
                                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                                edges.push(((u64::from(lo) << 32) | u64::from(hi), w));
                            }
                            cap *= (1.0 - f) * (1.0 - f);
                            scale *= 1.0 - f;
                        }
                    }
                    edges
                })
                .collect()
        });
        let total: usize = parts.iter().map(Vec::len).sum();
        let mut edges = alloc_vec(0, (0u64, 0.0f64))?;
        edges.try_reserve_exact(total).map_err(|_| RfError::OutOfMemory)?;
        for p in parts {
            edges.extend(p);
        }
        pool.install(|| edges.par_sort_by_key(|e| e.0));
        let mut out = 0;
        for i in 0..edges.len() {
            if out > 0 && edges[out - 1].0 == edges[i].0 {
                edges[out - 1].1 += edges[i].1;
            } else {
                edges[out] = edges[i];
                out += 1;
            }
        }
        edges.truncate(out);
        laplacian_from_edges(keep_idx.len(), &edges)
    }
}

/// Removes the unused tail of each row (rows were allocated at their structural size).
fn compact_csr(n: usize, ptr: &[u32], col: Vec<u32>, val: Vec<f64>, used: &[u32]) -> Result<Csr, RfError> {
    if (0..n).all(|k| ptr[k] + used[k] == ptr[k + 1]) {
        return Ok(Csr { n, ptr: ptr.to_vec(), col, val });
    }
    let mut nptr = alloc_vec(n + 1, 0u32)?;
    for k in 0..n {
        nptr[k + 1] = nptr[k] + used[k];
    }
    let mut ncol = alloc_vec(nptr[n] as usize, 0u32)?;
    let mut nval = alloc_vec(nptr[n] as usize, 0.0)?;
    for k in 0..n {
        let (a, u) = (ptr[k] as usize, used[k] as usize);
        let b = nptr[k] as usize;
        ncol[b..b + u].copy_from_slice(&col[a..a + u]);
        nval[b..b + u].copy_from_slice(&val[a..a + u]);
    }
    Ok(Csr { n, ptr: nptr, col: ncol, val: nval })
}

/// Full Laplacian CSR from unique undirected edges (key = lo << 32 | hi, lo < hi), sorted by key.
fn laplacian_from_edges(n: usize, edges: &[(u64, f64)]) -> Result<Csr, RfError> {
    let mut deg = alloc_vec(n, 0u32)?;
    for &(k, _) in edges {
        deg[(k >> 32) as usize] += 1;
        deg[(k & 0xFFFF_FFFF) as usize] += 1;
    }
    let mut ptr = alloc_vec(n + 1, 0u32)?;
    for i in 0..n {
        ptr[i + 1] = ptr[i] + deg[i] + 1;
    }
    let nnz = ptr[n] as usize;
    let mut col = alloc_vec(nnz, 0u32)?;
    let mut val = alloc_vec(nnz, 0.0)?;
    let mut cur: Vec<u32> = ptr[..n].to_vec();
    // Lower neighbours arrive in ascending order (edges sorted by (lo, hi), visited by hi), then
    // the diagonal, then upper neighbours (ascending hi for a fixed lo).
    // Pass 1: entries with column < row.
    for &(k, w) in edges {
        let (lo, hi) = ((k >> 32) as usize, (k & 0xFFFF_FFFF) as usize);
        let at = cur[hi] as usize;
        col[at] = lo as u32;
        val[at] = -w;
        cur[hi] += 1;
    }
    // Diagonal slots.
    let mut diag_at = vec![0u32; n];
    diag_at.copy_from_slice(&cur);
    cur.iter_mut().for_each(|c| *c += 1);
    // Pass 2: entries with column > row.
    for &(k, w) in edges {
        let (lo, hi) = ((k >> 32) as usize, (k & 0xFFFF_FFFF) as usize);
        let at = cur[lo] as usize;
        col[at] = hi as u32;
        val[at] = -w;
        cur[lo] += 1;
    }
    for i in 0..n {
        let r = ptr[i] as usize..ptr[i + 1] as usize;
        let d = diag_at[i] as usize;
        let s: f64 = r.clone().filter(|&a| a != d).map(|a| -val[a]).sum();
        col[d] = i as u32;
        val[d] = s;
    }
    Ok(Csr { n, ptr, col, val })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::par::with_pool;
    use crate::session::Term;
    use crate::structure;

    fn term(codes: Vec<u32>) -> Term {
        let g = *codes.iter().max().unwrap() as usize + 1;
        Term { factors: vec![0], codes, g, intercept: true, slopes: vec![], slope_vars: vec![], save: false }
    }

    /// Explicit S = A_RR - A_R1 N1^-1 A_1R from the cells.
    fn explicit_s(st: &Structure, ws: &WeightState) -> Vec<f64> {
        let m = st.m;
        let mut s = vec![0.0; m * m];
        for seg in 0..st.g1 {
            let mut agg = vec![0.0; m];
            for c in st.cells_of(seg) {
                let w = ws.cw(c);
                let idx: Vec<usize> = (0..st.nb).map(|b| st.blocks[b].off + st.codes(c)[b] as usize).collect();
                for &i in &idx {
                    agg[i] += w;
                    for &j in &idx {
                        s[i * m + j] += w;
                    }
                }
            }
            for i in 0..m {
                for j in 0..m {
                    s[i * m + j] -= agg[i] * agg[j] / ws.n1[seg];
                }
            }
        }
        s
    }

    #[test]
    fn kernels_match_explicit_reduced_matrix() {
        let mut rng = SplitMix64::new(42);
        let n = 3000;
        let mk = |g: u64, rng: &mut SplitMix64| (0..n).map(|_| (rng.next_u64() % g) as u32).collect::<Vec<u32>>();
        let terms = vec![term(mk(200, &mut rng)), term(mk(30, &mut rng)), term(mk(90, &mut rng))];
        let w: Vec<f64> = (0..n).map(|_| 0.5 + rng.next_f64()).collect();
        for limit in [4096, 50, 0] {
            with_pool(3, |pool| {
                let st = structure::build(&terms, n, limit, pool).unwrap();
                let wr: Vec<f64> = st.pi.iter().map(|&k| w[k as usize]).collect();
                let ws = weight_state(&st, Some(wr), pool).unwrap();
                let ops = Ops { st: &st, ws: &ws };
                let m = st.m;
                let s = explicit_s(&st, &ws);
                for i in 0..m {
                    assert!((s[i * m + i] - ws.diag[i]).abs() < 1e-10 * (1.0 + ws.nr[i]), "diag {i} (limit {limit})");
                }
                let r = 2;
                let p: Vec<f64> = (0..m * r).map(|_| rng.next_f64() - 0.5).collect();
                let mut y = vec![0.0; m * r];
                let mut sc = Scratch::default();
                ops.apply(&p, r, &mut y, &mut sc, pool).unwrap();
                for i in 0..m {
                    for j in 0..r {
                        let want: f64 = (0..m).map(|k| s[i * m + k] * p[k * r + j]).sum();
                        assert!((want - y[i * r + j]).abs() < 1e-10 * 50.0, "apply row {i} col {j}: {want} vs {} (limit {limit})", y[i * r + j]);
                    }
                }
                if st.m_small > 0 {
                    let ms = st.m_small;
                    let a = ops.assemble_small(pool).unwrap();
                    for i in 0..ms {
                        for j in 0..ms {
                            assert!((a[i * ms + j] - s[i * m + j]).abs() < 1e-10 * 50.0, "small ({i},{j}) limit {limit}");
                        }
                    }
                    // M^-1 is a generalized inverse on the small block: S_SS M^-1 r = r for r in range.
                    let d = crate::precond::dense::Dense::factor(&a, ms, &ws.redundant[..ms], pool).unwrap();
                    let p: Vec<f64> = (0..ms).map(|_| rng.next_f64() - 0.5).collect();
                    let rr: Vec<f64> = (0..ms).map(|i| (0..ms).map(|k| a[i * ms + k] * p[k]).sum()).collect();
                    let mut z = vec![0.0; ms];
                    d.apply(&rr, &mut z, ms, 1, pool);
                    for i in 0..ms {
                        let back: f64 = (0..ms).map(|k| a[i * ms + k] * z[k]).sum();
                        assert!((back - rr[i]).abs() < 1e-9, "ginv row {i}: {back} vs {} (limit {limit})", rr[i]);
                    }
                }
            })
            .unwrap();
        }
    }

    /// Small blocks dense within the segments (GEMM assembly, fused into the weight pass), with
    /// and without weights, against the explicit S; apply/rhs/residualize kernels against
    /// explicit formulas for widths with and without a const instance.
    #[test]
    fn dense_small_blocks_and_kernel_widths() {
        let mut rng = SplitMix64::new(7);
        let n = 4000;
        let mk = |g: u64, rng: &mut SplitMix64| (0..n).map(|_| (rng.next_u64() % g) as u32).collect::<Vec<u32>>();
        let terms = vec![term(mk(150, &mut rng)), term(mk(6, &mut rng)), term(mk(4, &mut rng)), term(mk(70, &mut rng))];
        for weighted in [false, true] {
            let w: Vec<f64> = (0..n).map(|_| 0.5 + rng.next_f64()).collect();
            with_pool(2, |pool| {
                let st = structure::build(&terms, n, 12, pool).unwrap();
                assert_eq!(st.m_small, 10);
                let wr = weighted.then(|| st.pi.iter().map(|&k| w[k as usize]).collect::<Vec<f64>>());
                let ws = weight_state(&st, wr, pool).unwrap();
                let ops = Ops { st: &st, ws: &ws };
                let (m, ms) = (st.m, st.m_small);
                let s = explicit_s(&st, &ws);
                let fused = ws.small_s.as_ref().expect("fused small block");
                let a = ops.assemble_small(pool).unwrap();
                for i in 0..ms {
                    for j in 0..ms {
                        assert!((a[i * ms + j] - s[i * m + j]).abs() < 1e-9, "gemm ({i},{j})");
                        assert!((fused[i * ms + j] - s[i * m + j]).abs() < 1e-9, "fused ({i},{j})");
                    }
                }
                for r in [1, 3, 9] {
                    let p: Vec<f64> = (0..m * r).map(|_| rng.next_f64() - 0.5).collect();
                    let mut y = vec![0.0; m * r];
                    let mut sc = Scratch::default();
                    ops.apply(&p, r, &mut y, &mut sc, pool).unwrap();
                    for i in 0..m {
                        for j in 0..r {
                            let want: f64 = (0..m).map(|k| s[i * m + k] * p[k * r + j]).sum();
                            assert!((want - y[i * r + j]).abs() < 1e-9, "apply r={r} ({i},{j})");
                        }
                    }
                    // rhs = D_R^T W v - A_R1 N1^-1 b1, residualize = v - D(alpha1, x).
                    let rows: Vec<f64> = (0..n * r).map(|_| rng.next_f64()).collect();
                    let rh = ops.rhs(&rows, r, &mut sc, pool).unwrap();
                    let wrow = |i: usize| ws.w_rows.as_ref().map_or(1.0, |w| w[i]);
                    let mut want = vec![0.0; m * r];
                    let mut b1 = vec![0.0; st.g1 * r];
                    for seg in 0..st.g1 {
                        for c in st.cells_of(seg) {
                            for i in st.rows_of_cell(c) {
                                for j in 0..r {
                                    b1[seg * r + j] += wrow(i) * rows[i * r + j];
                                }
                            }
                        }
                    }
                    for seg in 0..st.g1 {
                        for c in st.cells_of(seg) {
                            for i in st.rows_of_cell(c) {
                                for b in 0..st.nb {
                                    let k = st.blocks[b].off + st.codes(c)[b] as usize;
                                    for j in 0..r {
                                        want[k * r + j] += wrow(i) * (rows[i * r + j] - b1[seg * r + j] / ws.n1[seg]);
                                    }
                                }
                            }
                        }
                    }
                    for k in 0..m * r {
                        assert!((want[k] - rh.rhs[k]).abs() < 1e-9, "rhs r={r} {k}");
                    }
                    for k in 0..st.g1 * r {
                        assert!((b1[k] - rh.b1[k]).abs() < 1e-9, "b1 r={r} {k}");
                    }
                    let x: Vec<f64> = (0..m * r).map(|_| rng.next_f64() - 0.5).collect();
                    let mut res = rows.clone();
                    let mut alpha = vec![0.0; st.g1 * r];
                    ops.residualize_rows(&mut res, r, &x, &rh.b1, Some(&mut alpha), pool);
                    for seg in 0..st.g1 {
                        for c in st.cells_of(seg) {
                            for i in st.rows_of_cell(c) {
                                for j in 0..r {
                                    let fit: f64 = alpha[seg * r + j] + (0..st.nb).map(|b| x[(st.blocks[b].off + st.codes(c)[b] as usize) * r + j]).sum::<f64>();
                                    assert!((rows[i * r + j] - fit - res[i * r + j]).abs() < 1e-9, "resid r={r}");
                                }
                            }
                        }
                    }
                    // alpha1 = (b1 - sum_c W_c u_c)/n1.
                    for seg in 0..st.g1 {
                        for j in 0..r {
                            let mut a = b1[seg * r + j];
                            for c in st.cells_of(seg) {
                                a -= ws.cw(c) * (0..st.nb).map(|b| x[(st.blocks[b].off + st.codes(c)[b] as usize) * r + j]).sum::<f64>();
                            }
                            assert!((a / ws.n1[seg] - alpha[seg * r + j]).abs() < 1e-9, "alpha r={r}");
                        }
                    }
                }
            })
            .unwrap();
        }
    }
}
