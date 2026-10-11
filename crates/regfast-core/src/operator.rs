//! Kernels over the cell structure (plan D5): the reduced operator S·P, the right-hand side,
//! diag(S), back-substitution / in-place residualization, and the assembly of the blocks used by
//! the preconditioner.
//!
//! Every parallel kernel splits the segments into the fixed parts of [`Structure::tasks`]
//! (independent of the thread count); per-task accumulators are combined in task order and
//! large-block scatters are pulls along the level-sorted cell lists, so results are bit-identical
//! for any number of threads (plan D3).

use rayon::prelude::*;

use crate::error::RfError;
use crate::par::{CHUNK, Pool, alloc_vec};
use crate::rng::SplitMix64;
use crate::structure::{Structure, split_bounds};

/// Coordinates with d ≤ REDUNDANT·n are redundant (no information beyond FE1).
pub const REDUNDANT: f64 = 1e-12;

/// Weight-dependent quantities (refreshed by `set_weights`).
pub struct WeightState {
    /// Observation weights in row order (`None` = all ones).
    pub w_rows: Option<Vec<f64>>,
    /// Cell weights W_c (`None` = every W_c is 1).
    pub cell_w: Option<Vec<f64>>,
    /// FE1 level weights.
    pub n1: Vec<f64>,
    /// Level weights of every reduced coordinate.
    pub nr: Vec<f64>,
    /// diag(S).
    pub diag: Vec<f64>,
    /// Redundant coordinates (excluded from the preconditioner).
    pub redundant: Vec<bool>,
    /// Σw.
    pub sw: f64,
}

impl WeightState {
    #[inline]
    pub fn cw(&self, c: usize) -> f64 {
        match &self.cell_w {
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
    /// E = Σ_s d_s² (assembly work).
    pub work: u64,
}

impl SegAgg {
    fn seg(&self, s: usize) -> (&[u32], &[f64]) {
        let r = self.off[s] as usize..self.off[s + 1] as usize;
        (&self.codes[r.clone()], &self.sums[r])
    }
}

/// Sums of `w` over the cells of a segment that share each cell's code in one block:
/// `out[k] = Σ_{c' : code(c') = code(c_k)} W_c'`.
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
    let mc = st.ncells;
    let cell_w = match (&w_rows, &st.cell_obs_off) {
        (None, None) => None,
        (Some(w), None) => Some(w.clone()),
        (w, Some(_)) => {
            let mut cw = alloc_vec(mc, 0.0)?;
            pool.install(|| {
                cw.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, ch)| {
                    for (k, v) in ch.iter_mut().enumerate() {
                        let rows = st.rows_of_cell(ci * CHUNK + k);
                        *v = match w {
                            Some(w) => w[rows].iter().sum(),
                            None => rows.len() as f64,
                        };
                    }
                })
            });
            Some(cw)
        }
    };
    let mut ws = WeightState {
        w_rows,
        cell_w,
        n1: Vec::new(),
        nr: Vec::new(),
        diag: Vec::new(),
        redundant: Vec::new(),
        sw: 0.0,
    };
    ws.n1 = pool.install(|| (0..st.g1).into_par_iter().with_min_len(1024).map(|s| st.cells_of(s).map(|c| ws.cw(c)).sum()).collect());
    let parts: Vec<f64> = pool.install(|| ws.n1.par_chunks(CHUNK).map(|ch| ch.iter().sum::<f64>()).collect());
    ws.sw = parts.iter().sum();

    // Level weights and Σ_{g1} c²/n1 per coordinate.
    let ns = st.blocks.iter().filter(|b| b.small).count();
    let nl = st.nb - ns;
    let ms = st.m_small;
    let tb = st.tasks(2 * ms);
    let cb = st.cell_bounds(&tb);
    let mut zq = alloc_vec(mc * nl, 0.0)?;
    let zb: Vec<usize> = cb.iter().map(|c| c * nl).collect();
    let pieces = split_bounds(&mut zq, &zb);
    let wsr = &ws;
    let accs: Vec<Vec<f64>> = pool.install(|| {
        pieces
            .into_par_iter()
            .enumerate()
            .map(|(t, zt)| {
                let mut acc = vec![0.0; 2 * ms];
                let (mut codes, mut w, mut csum, mut tmp) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for s in tb[t] as usize..tb[t + 1] as usize {
                    let cells = st.cells_of(s);
                    let k = cells.len();
                    w.clear();
                    w.extend(cells.clone().map(|c| wsr.cw(c)));
                    csum.resize(k, 0.0);
                    let n1 = wsr.n1[s];
                    for (b, blk) in st.blocks.iter().enumerate() {
                        codes.clear();
                        codes.extend(cells.clone().map(|c| st.codes(c)[b]));
                        seg_level_sums(&codes, &w, &mut csum, &mut tmp);
                        for i in 0..k {
                            let q = w[i] * csum[i] / n1;
                            if blk.small {
                                let at = blk.off + codes[i] as usize;
                                acc[at] += w[i];
                                acc[ms + at] += q;
                            } else {
                                zt[(cells.start + i - cb[t]) * nl + (b - ns)] = q;
                            }
                        }
                    }
                }
                acc
            })
            .collect()
    });
    let mut nr = alloc_vec(st.m, 0.0)?;
    let mut dq = alloc_vec(st.m, 0.0)?;
    for acc in &accs {
        for k in 0..ms {
            nr[k] += acc[k];
            dq[k] += acc[ms + k];
        }
    }
    for (b, blk) in st.blocks.iter().enumerate().skip(ns) {
        let col = b - ns;
        let rng = blk.off..blk.off + blk.g;
        let (nr_b, dq_b) = (&mut nr[rng.clone()], &mut dq[rng]);
        pool.install(|| {
            nr_b.par_iter_mut().zip(dq_b.par_iter_mut()).enumerate().with_min_len(1024).for_each(|(g, (n, d))| {
                for &c in &blk.perm[blk.perm_off[g] as usize..blk.perm_off[g + 1] as usize] {
                    *n += wsr.cw(c as usize);
                    *d += zq[c as usize * nl + col];
                }
            })
        });
    }
    ws.diag = nr.iter().zip(&dq).map(|(n, d)| n - d).collect();
    ws.redundant = ws.diag.iter().zip(&nr).map(|(d, n)| *d <= REDUNDANT * n).collect();
    ws.nr = nr;
    Ok(ws)
}

/// Kernels bound to a structure and a weight state.
pub struct Ops<'a> {
    pub st: &'a Structure,
    pub ws: &'a WeightState,
}

impl Ops<'_> {
    /// Σ_b P[off_b + g_b(c)] for every cell of `cells` into `u` (len × r).
    #[inline]
    fn cell_sums(&self, cells: std::ops::Range<usize>, p: &[f64], r: usize, u: &mut Vec<f64>) {
        u.clear();
        u.resize(cells.len() * r, 0.0);
        let st = self.st;
        for (k, c) in cells.enumerate() {
            let uc = &mut u[k * r..(k + 1) * r];
            for (b, &code) in st.codes(c).iter().enumerate() {
                let at = (st.blocks[b].off + code as usize) * r;
                for (a, v) in uc.iter_mut().zip(&p[at..at + r]) {
                    *a += v;
                }
            }
        }
    }

    /// Scatters per-cell values `z_c` (computed by `fill` per segment into `zc`, len × r) into the
    /// reduced space: small coordinates through per-task accumulators, large ones through the
    /// scratch `zbuf` (M × r) and level pulls. `fill(s, cells, zc, extra)` may also write
    /// per-segment outputs (`seg_out`, `seg_w` values per segment) and per-task sums (`extra_len`).
    #[allow(clippy::too_many_arguments)]
    fn scatter_cells(
        &self,
        r: usize,
        y: &mut [f64],
        zbuf: &mut [f64],
        seg_out: Option<(&mut [f64], usize)>,
        extra_len: usize,
        pool: &Pool,
        fill: impl Fn(usize, std::ops::Range<usize>, &mut Vec<f64>, Option<&mut [f64]>, &mut [f64]) + Sync,
    ) -> Vec<f64> {
        let st = self.st;
        let ms = st.m_small;
        let has_large = st.has_large();
        let acc_len = ms * r;
        let tb = st.tasks(acc_len);
        let cb = st.cell_bounds(&tb);
        let nt = tb.len() - 1;
        let zp: Vec<Option<&mut [f64]>> = if has_large {
            let zb: Vec<usize> = cb.iter().map(|c| c * r).collect();
            split_bounds(&mut zbuf[..st.ncells * r], &zb).into_iter().map(Some).collect()
        } else {
            (0..nt).map(|_| None).collect()
        };
        let sw = seg_out.as_ref().map_or(0, |s| s.1);
        let so: Vec<Option<&mut [f64]>> = match seg_out {
            Some((buf, width)) => {
                let sb: Vec<usize> = tb.iter().map(|&s| s as usize * width).collect();
                split_bounds(buf, &sb).into_iter().map(Some).collect()
            }
            None => (0..nt).map(|_| None).collect(),
        };
        let results: Vec<(Vec<f64>, Vec<f64>)> = pool.install(|| {
            zp.into_par_iter()
                .zip(so)
                .enumerate()
                .map(|(t, (mut zt, mut sot))| {
                    let mut acc = vec![0.0; acc_len];
                    let mut extra = vec![0.0; extra_len];
                    let mut zc = Vec::new();
                    let s0 = tb[t] as usize;
                    for s in s0..tb[t + 1] as usize {
                        let cells = st.cells_of(s);
                        let segslot = sot.as_deref_mut().map(|o| &mut o[(s - s0) * sw..(s - s0 + 1) * sw]);
                        fill(s, cells.clone(), &mut zc, segslot, &mut extra);
                        for (k, c) in cells.enumerate() {
                            let z = &zc[k * r..(k + 1) * r];
                            for (b, &code) in st.codes(c).iter().enumerate() {
                                let blk = &st.blocks[b];
                                if !blk.small {
                                    break;
                                }
                                let at = (blk.off + code as usize) * r;
                                for (a, v) in acc[at..at + r].iter_mut().zip(z) {
                                    *a += v;
                                }
                            }
                            if let Some(zt) = zt.as_deref_mut() {
                                let at = (c - cb[t]) * r;
                                zt[at..at + r].copy_from_slice(z);
                            }
                        }
                    }
                    (acc, extra)
                })
                .collect()
        });
        let mut extra = vec![0.0; extra_len];
        y[..acc_len].iter_mut().for_each(|v| *v = 0.0);
        for (acc, ex) in &results {
            for (a, v) in y[..acc_len].iter_mut().zip(acc) {
                *a += v;
            }
            for (a, v) in extra.iter_mut().zip(ex) {
                *a += v;
            }
        }
        if has_large {
            self.pull_large(zbuf, r, y, pool);
        }
        extra
    }

    /// y[off_b + g] = Σ_{c ∈ level g of b} Z[c] for every large block.
    fn pull_large(&self, zbuf: &[f64], r: usize, y: &mut [f64], pool: &Pool) {
        let st = self.st;
        let ms = st.m_small;
        let ylarge = &mut y[ms * r..];
        let mut rest = ylarge;
        for blk in st.blocks.iter().filter(|b| !b.small) {
            let (yb, tail) = rest.split_at_mut(blk.g * r);
            rest = tail;
            pool.install(|| {
                yb.par_chunks_mut(r).enumerate().with_min_len(512).for_each(|(g, out)| {
                    out.iter_mut().for_each(|v| *v = 0.0);
                    for &c in &blk.perm[blk.perm_off[g] as usize..blk.perm_off[g + 1] as usize] {
                        let at = c as usize * r;
                        for (o, v) in out.iter_mut().zip(&zbuf[at..at + r]) {
                            *o += v;
                        }
                    }
                })
            });
        }
    }

    /// Y = S·P (both m × r row-major). `zbuf` is scratch of M × r (unused without large blocks).
    pub fn apply(&self, p: &[f64], r: usize, y: &mut [f64], zbuf: &mut [f64], pool: &Pool) {
        let ws = self.ws;
        self.scatter_cells(r, y, zbuf, None, 0, pool, |s, cells, zc, _, _| {
            self.cell_sums(cells.clone(), p, r, zc);
            let mut t = [0.0f64; 64];
            let mut tv = Vec::new();
            let t: &mut [f64] = if r <= 64 {
                &mut t[..r]
            } else {
                tv.resize(r, 0.0);
                &mut tv
            };
            for (k, c) in cells.clone().enumerate() {
                let w = ws.cw(c);
                for (a, v) in t.iter_mut().zip(&zc[k * r..(k + 1) * r]) {
                    *a += w * v;
                }
            }
            let inv = 1.0 / ws.n1[s];
            t.iter_mut().for_each(|v| *v *= inv);
            for (k, c) in cells.enumerate() {
                let w = ws.cw(c);
                for (a, tv) in zc[k * r..(k + 1) * r].iter_mut().zip(t.iter()) {
                    *a = w * (*a - tv);
                }
            }
        });
    }

    /// Right-hand side of the reduced system for the row block `rows` (N × r, row order).
    /// Returns (rhs m×r, b1 g1×r, Σw v² per column, Σw v per column).
    pub fn rhs(&self, rows: &[f64], r: usize, zbuf: &mut [f64], pool: &Pool) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), RfError> {
        let (st, ws) = (self.st, self.ws);
        let mut y = alloc_vec(st.m * r, 0.0)?;
        let mut b1 = alloc_vec(st.g1 * r, 0.0)?;
        let extra = self.scatter_cells(r, &mut y, zbuf, Some((&mut b1, r)), 2 * r, pool, |s, cells, zc, b1s, extra| {
            zc.clear();
            zc.resize(cells.len() * r, 0.0);
            for (k, c) in cells.clone().enumerate() {
                let sc = &mut zc[k * r..(k + 1) * r];
                for row in st.rows_of_cell(c) {
                    let v = &rows[row * r..(row + 1) * r];
                    let w = ws.w_rows.as_ref().map_or(1.0, |w| w[row]);
                    for j in 0..r {
                        let wv = w * v[j];
                        sc[j] += wv;
                        extra[j] += wv * v[j];
                        extra[r + j] += wv;
                    }
                }
            }
            let b1s = b1s.expect("b1 slot");
            for k in 0..cells.len() {
                for (a, v) in b1s.iter_mut().zip(&zc[k * r..(k + 1) * r]) {
                    *a += v;
                }
            }
            let inv = 1.0 / ws.n1[s];
            for (k, c) in cells.enumerate() {
                let w = ws.cw(c) * inv;
                for (a, b) in zc[k * r..(k + 1) * r].iter_mut().zip(b1s.iter()) {
                    *a -= w * b;
                }
            }
        });
        let (wsq, wsum) = extra.split_at(r);
        Ok((y, b1, wsq.to_vec(), wsum.to_vec()))
    }

    /// In-place residualization of `rows` given the reduced solution `x` (m×r) and b1:
    /// α1 = (b1 − Σ_c W_c u_c)/n1 per segment and v_i −= α1 + u_c for every row (F is never
    /// materialized). Writes α1 (g1×r) to `alpha` when given.
    pub fn residualize_rows(&self, rows: &mut [f64], r: usize, x: &[f64], b1: &[f64], alpha: Option<&mut [f64]>, pool: &Pool) {
        let (st, ws) = (self.st, self.ws);
        let tb = st.parts.clone();
        let cb = st.cell_bounds(&tb);
        let rb: Vec<usize> = cb.iter().map(|&c| st.cell_row(c) * r).collect();
        let row_pieces = split_bounds(&mut rows[..st.n * r], &rb);
        let ap: Vec<Option<&mut [f64]>> = match alpha {
            Some(a) => {
                let ab: Vec<usize> = tb.iter().map(|&s| s as usize * r).collect();
                split_bounds(a, &ab).into_iter().map(Some).collect()
            }
            None => (0..tb.len() - 1).map(|_| None).collect(),
        };
        pool.install(|| {
            row_pieces.into_par_iter().zip(ap).enumerate().for_each(|(t, (rp, mut at))| {
                let mut u = Vec::new();
                let mut a = vec![0.0; r];
                let row0 = rb[t] / r;
                let s0 = tb[t] as usize;
                for s in s0..tb[t + 1] as usize {
                    let cells = st.cells_of(s);
                    self.cell_sums(cells.clone(), x, r, &mut u);
                    a.copy_from_slice(&b1[s * r..(s + 1) * r]);
                    for (k, c) in cells.clone().enumerate() {
                        let w = ws.cw(c);
                        for (av, v) in a.iter_mut().zip(&u[k * r..(k + 1) * r]) {
                            *av -= w * v;
                        }
                    }
                    let inv = 1.0 / ws.n1[s];
                    a.iter_mut().for_each(|v| *v *= inv);
                    if let Some(at) = at.as_deref_mut() {
                        at[(s - s0) * r..(s - s0 + 1) * r].copy_from_slice(&a);
                    }
                    for (k, c) in cells.enumerate() {
                        let uc = &mut u[k * r..(k + 1) * r];
                        for (f, av) in uc.iter_mut().zip(&a) {
                            *f += av;
                        }
                        for row in st.rows_of_cell(c) {
                            let v = &mut rp[(row - row0) * r..(row - row0 + 1) * r];
                            for (vv, f) in v.iter_mut().zip(uc.iter()) {
                                *vv -= f;
                            }
                        }
                    }
                }
            })
        });
    }

    /// FE fit per row for one coefficient column: α1[g1] + Σ_b x_b[g_b] (`x` m-vector, `alpha` g1).
    pub fn fit_rows(&self, x: &[f64], alpha: &[f64], pool: &Pool) -> Vec<f64> {
        let st = self.st;
        let mut out = vec![0.0; st.n];
        let tb = st.parts.clone();
        let cb = st.cell_bounds(&tb);
        let rb: Vec<usize> = cb.iter().map(|&c| st.cell_row(c)).collect();
        let pieces = split_bounds(&mut out, &rb);
        pool.install(|| {
            pieces.into_par_iter().enumerate().for_each(|(t, piece)| {
                for s in tb[t] as usize..tb[t + 1] as usize {
                    for c in st.cells_of(s) {
                        let mut f = alpha[s];
                        for (b, &code) in st.codes(c).iter().enumerate() {
                            f += x[st.blocks[b].off + code as usize];
                        }
                        for row in st.rows_of_cell(c) {
                            piece[row - rb[t]] = f;
                        }
                    }
                }
            })
        });
        out
    }

    /// Dense S restricted to the small coordinates (m_s × m_s, row-major).
    pub fn assemble_small(&self, pool: &Pool) -> Result<Vec<f64>, RfError> {
        let (st, ws) = (self.st, self.ws);
        let ms = st.m_small;
        let ns = st.blocks.iter().filter(|b| b.small).count();
        let nrb = (ms * ms).div_ceil(1 << 21).max(1);
        let rows_per = ms.div_ceil(nrb);
        let tb = st.tasks(rows_per * ms);
        let nt = tb.len() - 1;
        let jobs: Vec<(usize, usize)> = (0..nrb).flat_map(|rbk| (0..nt).map(move |t| (rbk, t))).collect();
        let accs: Vec<Vec<f64>> = pool.install(|| {
            jobs.par_iter()
                .map(|&(rbk, t)| {
                    let k0 = rbk * rows_per;
                    let k1 = ((rbk + 1) * rows_per).min(ms);
                    let mut acc = vec![0.0; (k1 - k0) * ms];
                    let mut agg: Vec<(u32, f64)> = Vec::new();
                    for s in tb[t] as usize..tb[t + 1] as usize {
                        agg.clear();
                        for c in st.cells_of(s) {
                            let w = ws.cw(c);
                            let codes = st.codes(c);
                            for b in 0..ns {
                                let kb = st.blocks[b].off + codes[b] as usize;
                                agg.push((kb as u32, w));
                                if kb >= k0 && kb < k1 {
                                    for b2 in 0..ns {
                                        let l = st.blocks[b2].off + codes[b2] as usize;
                                        acc[(kb - k0) * ms + l] += w;
                                    }
                                }
                            }
                        }
                        agg.sort_by_key(|e| e.0);
                        let mut out = 0;
                        for i in 0..agg.len() {
                            if out > 0 && agg[out - 1].0 == agg[i].0 {
                                agg[out - 1].1 += agg[i].1;
                            } else {
                                agg[out] = agg[i];
                                out += 1;
                            }
                        }
                        agg.truncate(out);
                        let inv = 1.0 / ws.n1[s];
                        for &(k, ck) in &agg {
                            let k = k as usize;
                            if k < k0 || k >= k1 {
                                continue;
                            }
                            let row = &mut acc[(k - k0) * ms..(k - k0 + 1) * ms];
                            for &(l, cl) in &agg {
                                row[l as usize] -= ck * cl * inv;
                            }
                        }
                    }
                    acc
                })
                .collect()
        });
        let mut full = alloc_vec(ms * ms, 0.0)?;
        for (job, acc) in jobs.iter().zip(&accs) {
            let k0 = job.0 * rows_per;
            for (a, v) in full[k0 * ms..k0 * ms + acc.len()].iter_mut().zip(acc) {
                *a += v;
            }
        }
        Ok(full)
    }

    /// Per-segment aggregated codes of block `b`.
    pub fn seg_agg(&self, b: usize, pool: &Pool) -> Result<SegAgg, RfError> {
        let (st, ws) = (self.st, self.ws);
        let lists: Vec<(Vec<u32>, Vec<u32>, Vec<f64>)> = pool.install(|| {
            (0..st.nparts())
                .into_par_iter()
                .map(|p| {
                    let (mut cnt, mut codes, mut sums) = (Vec::new(), Vec::new(), Vec::new());
                    let mut tmp: Vec<(u32, f64)> = Vec::new();
                    for s in st.parts[p] as usize..st.parts[p + 1] as usize {
                        tmp.clear();
                        tmp.extend(st.cells_of(s).map(|c| (st.codes(c)[b], ws.cw(c))));
                        tmp.sort_by_key(|e| e.0);
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
        let total: usize = lists.iter().map(|l| l.1.len()).sum();
        let mut off = alloc_vec(st.g1 + 1, 0u32)?;
        let mut codes = alloc_vec(0, 0u32)?;
        codes.try_reserve_exact(total).map_err(|_| RfError::OutOfMemory)?;
        let mut sums = alloc_vec(0, 0.0)?;
        sums.try_reserve_exact(total).map_err(|_| RfError::OutOfMemory)?;
        let mut s = 0usize;
        let mut work = 0u64;
        for (cnt, c, w) in lists {
            for k in cnt {
                off[s + 1] = off[s] + k;
                work += u64::from(k) * u64::from(k);
                s += 1;
            }
            codes.extend(c);
            sums.extend(w);
        }
        Ok(SegAgg { off, codes, sums, work })
    }

    /// Number of structural nonzeros of each row of S_bb (diagonal included). Cost ≈ E_b.
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
    /// diagonal is minus the sum of the row's off-diagonals. `keep_idx` maps kept index → level.
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

    /// Clique-sampled Laplacian L̃_bb (Kyng–Sachdeva clique trees per FE1 segment, plan D5) over
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
    for i in 0..n {
        diag_at[i] = cur[i];
        cur[i] += 1;
    }
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

    /// Explicit S = A_RR − A_R1 N1⁻¹ A_1R from the cells.
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
                let mut zbuf = vec![0.0; st.ncells * r];
                ops.apply(&p, r, &mut y, &mut zbuf, pool);
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
                    // M⁻¹ is a generalized inverse on the small block: S_SS M⁻¹ r = r for r ∈ range.
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
}
