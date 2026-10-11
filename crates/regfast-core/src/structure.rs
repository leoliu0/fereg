//! Cell structure of the intercept-only FE design (plan D5).
//!
//! Rows (observations) are ordered by π: sorted by (g1, g2, …, gQ, obs). A *cell* is a run of
//! rows with the same code tuple; a *segment* is the run of cells of one FE1 level. Blocks are
//! the non-FE1 terms; the reduced unknown x stacks their levels: small blocks first (jointly at
//! most [`SMALL_LIMIT`] levels, factored densely), then large blocks, each in term order.
//!
//! Everything here is structural (weight independent) and built once per sample.

use std::ops::Range;
use std::sync::atomic::{AtomicU32, Ordering};

use rayon::prelude::*;

use crate::absorber::fe1_index;
use crate::error::RfError;
use crate::par::{CHUNK, Pool, alloc_vec};
use crate::session::Term;

/// Default total levels of the blocks factored jointly and exactly (dense).
pub const SMALL_LIMIT: usize = 4096;

/// One non-FE1 term.
#[derive(Clone, Debug)]
pub struct Block {
    /// Session term index.
    pub term: usize,
    /// Levels.
    pub g: usize,
    /// Offset of level 0 in the reduced coordinate vector.
    pub off: usize,
    pub small: bool,
    /// Large blocks: cell ids sorted by (level, cell).
    pub perm: Vec<u32>,
    /// Large blocks: `perm` offsets per level (g + 1 entries).
    pub perm_off: Vec<u32>,
}

pub struct Structure {
    /// Rows (kept observations).
    pub n: usize,
    /// Session index of FE1 (`None` under `noabsorb`: one synthetic level).
    pub fe1: Option<usize>,
    /// FE1 levels (segments).
    pub g1: usize,
    /// Kept-observation index of each row.
    pub pi: Vec<u32>,
    /// Cell offsets per segment (g1 + 1 entries).
    pub seg_off: Vec<u32>,
    /// Row offsets per cell (M + 1 entries); `None` when every cell holds exactly one row.
    pub cell_obs_off: Option<Vec<u32>>,
    /// Number of cells M.
    pub ncells: usize,
    /// Number of blocks (Q − 1).
    pub nb: usize,
    /// Codes of blocks 0..nb per cell, row-major M × nb.
    pub cell_codes: Vec<u32>,
    pub blocks: Vec<Block>,
    /// Reduced dimension Σ_b g_b.
    pub m: usize,
    /// Small coordinates occupy `0..m_small`.
    pub m_small: usize,
    /// Fixed partition of the segments into parts of ≳ [`CHUNK`] cells (boundaries, len = parts + 1).
    pub parts: Vec<u32>,
    /// Largest reduced vector (bytes) scattered entirely through accumulators
    /// (see `operator::Ops::accumulates_all`).
    pub acc_bytes_max: usize,
    /// Σ_seg s², s = distinct small coordinates per segment (dense assembly work).
    pub small_work: u64,
}

impl Structure {
    #[inline]
    pub fn cells_of(&self, s: usize) -> Range<usize> {
        self.seg_off[s] as usize..self.seg_off[s + 1] as usize
    }

    #[inline]
    pub fn rows_of_cell(&self, c: usize) -> Range<usize> {
        match &self.cell_obs_off {
            Some(o) => o[c] as usize..o[c + 1] as usize,
            None => c..c + 1,
        }
    }

    /// First row of cell `c` (`c` may equal M).
    #[inline]
    pub fn cell_row(&self, c: usize) -> usize {
        match &self.cell_obs_off {
            Some(o) => o[c] as usize,
            None => c,
        }
    }

    #[inline]
    pub fn codes(&self, c: usize) -> &[u32] {
        &self.cell_codes[c * self.nb..(c + 1) * self.nb]
    }

    pub fn has_large(&self) -> bool {
        self.blocks.iter().any(|b| !b.small)
    }

    pub fn nparts(&self) -> usize {
        self.parts.len() - 1
    }

    /// Groups the fixed parts so that `ntasks × acc_len ≤ 2²²` (the per-task accumulators stay
    /// bounded). Returns segment boundaries of the tasks. Depends only on the data and `acc_len`.
    pub fn tasks(&self, acc_len: usize) -> Vec<u32> {
        let np = self.nparts();
        let per = if acc_len == 0 { 1 } else { (np * acc_len).div_ceil(1 << 22).max(1) };
        let mut t: Vec<u32> = (0..np).step_by(per).map(|k| self.parts[k]).collect();
        t.push(self.parts[np]);
        t
    }

    /// `k` tasks of nearly equal cell counts (segment boundaries); depends only on the data and `k`.
    pub fn even_tasks(&self, k: usize) -> Vec<u32> {
        let k = k.clamp(1, self.g1.max(1));
        let mut t = vec![0u32];
        for j in 1..k {
            let target = (self.ncells * j / k) as u32;
            let s = self.seg_off[..self.g1].partition_point(|&c| c < target) as u32;
            if s > *t.last().expect("nonempty") {
                t.push(s);
            }
        }
        t.push(self.g1 as u32);
        t
    }

    /// Cell for each segment boundary in `bounds`.
    pub fn cell_bounds(&self, bounds: &[u32]) -> Vec<usize> {
        bounds.iter().map(|&s| self.seg_off[s as usize] as usize).collect()
    }

    /// Segment of every cell.
    pub fn cell_segments(&self, pool: &Pool) -> Result<Vec<u32>, RfError> {
        let mut out = alloc_vec(self.ncells, 0u32)?;
        let tb = self.parts.clone();
        let cb = self.cell_bounds(&tb);
        let pieces = split_bounds(&mut out, &cb);
        pool.install(|| {
            pieces.into_par_iter().enumerate().for_each(|(t, piece)| {
                let base = cb[t];
                for s in tb[t] as usize..tb[t + 1] as usize {
                    for c in self.cells_of(s) {
                        piece[c - base] = s as u32;
                    }
                }
            })
        });
        Ok(out)
    }
}

/// Splits `v` into consecutive pieces at the ascending absolute `bounds` (first = 0 offset of `v`).
pub fn split_bounds<'a, T>(mut v: &'a mut [T], bounds: &[usize]) -> Vec<&'a mut [T]> {
    let mut out = Vec::with_capacity(bounds.len().saturating_sub(1));
    for w in bounds.windows(2) {
        let (a, b) = v.split_at_mut(w[1] - w[0]);
        out.push(a);
        v = b;
    }
    out
}

fn bits_for(g: usize) -> u32 {
    if g <= 1 { 0 } else { usize::BITS - (g - 1).leading_zeros() }
}

/// Builds the structure for intercept-only `terms` over `n` kept observations; small blocks
/// jointly hold at most `small_limit` levels.
pub fn build(terms: &[Term], n: usize, small_limit: usize, pool: &Pool) -> Result<Structure, RfError> {
    if terms.iter().any(|t| !t.slopes.is_empty() || !t.intercept) {
        return Err(RfError::InvalidInput("slopes not yet supported".into()));
    }
    let fe1 = fe1_index(terms);
    let (g1, code1): (usize, Option<&[u32]>) = match fe1 {
        Some(f) => (terms[f].g, Some(&terms[f].codes)),
        None => (1, None),
    };

    // Small/large classification: ascending G while the running total stays ≤ small_limit.
    let others: Vec<usize> = (0..terms.len()).filter(|&q| Some(q) != fe1).collect();
    let mut by_g = others.clone();
    by_g.sort_by_key(|&q| (terms[q].g, q));
    let mut small_set = vec![false; terms.len()];
    let mut total = 0usize;
    for &q in &by_g {
        if total + terms[q].g > small_limit {
            break;
        }
        total += terms[q].g;
        small_set[q] = true;
    }
    let mut order: Vec<usize> = others.iter().copied().filter(|&q| small_set[q]).collect();
    order.extend(others.iter().copied().filter(|&q| !small_set[q]));
    let mut blocks = Vec::with_capacity(order.len());
    let mut off = 0usize;
    for &q in &order {
        blocks.push(Block { term: q, g: terms[q].g, off, small: small_set[q], perm: Vec::new(), perm_off: Vec::new() });
        off += terms[q].g;
    }
    let m = off;
    let m_small: usize = blocks.iter().filter(|b| b.small).map(|b| b.g).sum();
    let nb = blocks.len();
    let bcodes: Vec<&[u32]> = blocks.iter().map(|b| terms[b.term].codes.as_slice()).collect();

    // 1. Histogram of g1 and scatter of observation ids by segment.
    let (seg_obs_off, mut pi) = match code1 {
        None => (vec![0u32, n as u32], (0..n as u32).collect::<Vec<u32>>()),
        Some(c1) => scatter_by_code(c1, g1, pool)?,
    };

    // 2–3. Sort every segment by (codes, obs) and count its cells.
    let groups = obs_groups(&seg_obs_off);
    let row_bounds: Vec<usize> = groups.iter().map(|&s| seg_obs_off[s as usize] as usize).collect();
    let mut cells_per_seg = alloc_vec(g1, 0u32)?;
    {
        let pi_pieces = split_bounds(&mut pi, &row_bounds);
        let seg_bounds: Vec<usize> = groups.iter().map(|&s| s as usize).collect();
        let cnt_pieces = split_bounds(&mut cells_per_seg, &seg_bounds);
        let layout = key_layout(&blocks);
        pool.install(|| {
            pi_pieces.into_par_iter().zip(cnt_pieces).enumerate().for_each(|(gi, (pis, cnts))| {
                let (s0, s1) = (groups[gi] as usize, groups[gi + 1] as usize);
                let base = seg_obs_off[s0] as usize;
                let mut k64: Vec<u64> = Vec::new();
                let mut k128: Vec<u128> = Vec::new();
                for s in s0..s1 {
                    let r = seg_obs_off[s] as usize - base..seg_obs_off[s + 1] as usize - base;
                    let seg = &mut pis[r];
                    sort_segment(seg, &bcodes, &layout, &mut k64, &mut k128);
                    let mut cells = u32::from(!seg.is_empty());
                    for w in seg.windows(2) {
                        if bcodes.iter().any(|c| c[w[0] as usize] != c[w[1] as usize]) {
                            cells += 1;
                        }
                    }
                    cnts[s - s0] = cells;
                }
            })
        });
    }
    // 4. Cell numbering.
    let mut seg_off = alloc_vec(g1 + 1, 0u32)?;
    for s in 0..g1 {
        seg_off[s + 1] = seg_off[s] + cells_per_seg[s];
    }
    drop(cells_per_seg);
    let ncells = seg_off[g1] as usize;
    let trivial = ncells == n;
    let mut cell_codes = alloc_vec(ncells * nb, 0u32)?;
    let mut cell_obs = if trivial { Vec::new() } else { alloc_vec(ncells + 1, 0u32)? };
    {
        let cell_bounds: Vec<usize> = groups.iter().map(|&s| seg_off[s as usize] as usize).collect();
        let cc_bounds: Vec<usize> = cell_bounds.iter().map(|&c| c * nb).collect();
        let cc_pieces = split_bounds(&mut cell_codes, &cc_bounds);
        let co_pieces: Vec<Option<&mut [u32]>> = if trivial {
            (0..groups.len() - 1).map(|_| None).collect()
        } else {
            split_bounds(&mut cell_obs[..ncells], &cell_bounds).into_iter().map(Some).collect()
        };
        let pi_ro = &pi;
        pool.install(|| {
            cc_pieces.into_par_iter().zip(co_pieces).enumerate().for_each(|(gi, (cc, mut co))| {
                let (s0, s1) = (groups[gi] as usize, groups[gi + 1] as usize);
                let mut c = 0usize;
                for s in s0..s1 {
                    let (r0, r1) = (seg_obs_off[s] as usize, seg_obs_off[s + 1] as usize);
                    for row in r0..r1 {
                        let obs = pi_ro[row] as usize;
                        let new_cell = row == r0 || bcodes.iter().any(|bc| bc[pi_ro[row - 1] as usize] != bc[obs]);
                        if new_cell {
                            for (b, bc) in bcodes.iter().enumerate() {
                                cc[c * nb + b] = bc[obs];
                            }
                            if let Some(co) = co.as_deref_mut() {
                                co[c] = row as u32;
                            }
                            c += 1;
                        }
                    }
                }
            })
        });
    }
    if !trivial {
        cell_obs[ncells] = n as u32;
    }

    // Fixed partition of the segments.
    let mut parts = vec![0u32];
    let mut next = CHUNK;
    for s in 1..g1 {
        if seg_off[s] as usize >= next {
            parts.push(s as u32);
            next = seg_off[s] as usize + CHUNK;
        }
    }
    parts.push(g1 as u32);

    let mut st = Structure {
        n,
        fe1,
        g1,
        pi,
        seg_off,
        cell_obs_off: (!trivial).then_some(cell_obs),
        ncells,
        nb,
        cell_codes,
        blocks,
        m,
        m_small,
        parts,
        acc_bytes_max: crate::operator::ACC_BYTES_MAX,
        small_work: 0,
    };

    // Dense small-block work Σ_seg s², s = distinct small coordinates in the segment.
    if st.m_small > 0 {
        let nsmall = st.blocks.iter().filter(|b| b.small).count();
        let work: u64 = pool.install(|| {
            (0..st.nparts())
                .into_par_iter()
                .map(|p| {
                    let mut tmp: Vec<u32> = Vec::new();
                    let mut w = 0u64;
                    for s in st.parts[p] as usize..st.parts[p + 1] as usize {
                        tmp.clear();
                        for c in st.cells_of(s) {
                            let codes = st.codes(c);
                            for b in 0..nsmall {
                                tmp.push((st.blocks[b].off as u32) + codes[b]);
                            }
                        }
                        tmp.sort_unstable();
                        tmp.dedup();
                        w += (tmp.len() as u64).pow(2);
                    }
                    w
                })
                .sum()
        });
        st.small_work = work;
        if work > 64 * st.ncells as u64 {
            st.blocks.iter_mut().for_each(|b| b.small = false);
            st.m_small = 0;
        }
    }

    // Level-sorted cell lists of the large blocks.
    for b in 0..nb {
        if !st.blocks[b].small {
            let (perm, perm_off) = level_perm(&st, b, pool)?;
            st.blocks[b].perm = perm;
            st.blocks[b].perm_off = perm_off;
        }
    }
    Ok(st)
}

/// Segment boundaries grouping consecutive segments into ≳ CHUNK rows (for parallel passes).
fn obs_groups(seg_obs_off: &[u32]) -> Vec<u32> {
    let g = seg_obs_off.len() - 1;
    let mut out = vec![0u32];
    let mut next = CHUNK;
    for s in 1..g {
        if seg_obs_off[s] as usize >= next {
            out.push(s as u32);
            next = seg_obs_off[s] as usize + CHUNK;
        }
    }
    out.push(g as u32);
    out
}

/// Counting sort of observation ids by `codes` (parallel; order within a level is arbitrary).
fn scatter_by_code(codes: &[u32], g: usize, pool: &Pool) -> Result<(Vec<u32>, Vec<u32>), RfError> {
    let n = codes.len();
    let cnt: Vec<AtomicU32> = (0..g).map(|_| AtomicU32::new(0)).collect();
    pool.install(|| {
        codes.par_chunks(CHUNK).for_each(|ch| {
            for &c in ch {
                cnt[c as usize].fetch_add(1, Ordering::Relaxed);
            }
        })
    });
    let mut off = alloc_vec(g + 1, 0u32)?;
    for k in 0..g {
        off[k + 1] = off[k] + cnt[k].load(Ordering::Relaxed);
    }
    for (k, c) in cnt.iter().enumerate() {
        c.store(off[k], Ordering::Relaxed);
    }
    let out: Vec<AtomicU32> = alloc_vec(n, 0u32)?.into_iter().map(AtomicU32::new).collect();
    pool.install(|| {
        codes.par_chunks(CHUNK).enumerate().for_each(|(ci, ch)| {
            let base = ci * CHUNK;
            for (k, &c) in ch.iter().enumerate() {
                let pos = cnt[c as usize].fetch_add(1, Ordering::Relaxed);
                out[pos as usize].store((base + k) as u32, Ordering::Relaxed);
            }
        })
    });
    let out = out.into_iter().map(AtomicU32::into_inner).collect();
    Ok((off, out))
}

/// Packed sort key layout: bit shift of each block's code (block 0 most significant) and the
/// total code width; `None` when the codes do not fit 96 bits (comparator sort).
struct KeyLayout {
    shifts: Vec<u32>,
    width: u32,
}

fn key_layout(blocks: &[Block]) -> Option<KeyLayout> {
    let bits: Vec<u32> = blocks.iter().map(|b| bits_for(b.g)).collect();
    let width: u32 = bits.iter().sum();
    if width > 96 {
        return None;
    }
    let mut shifts = vec![0u32; blocks.len()];
    let mut acc = 0u32;
    for b in (0..blocks.len()).rev() {
        shifts[b] = acc;
        acc += bits[b];
    }
    Some(KeyLayout { shifts, width })
}

/// Sorts the observation ids of one segment by (block codes, obs).
fn sort_segment(seg: &mut [u32], bcodes: &[&[u32]], layout: &Option<KeyLayout>, k64: &mut Vec<u64>, k128: &mut Vec<u128>) {
    if seg.len() <= 1 {
        return;
    }
    if bcodes.is_empty() {
        seg.sort_unstable();
        return;
    }
    match layout {
        Some(kl) if kl.width <= 32 => {
            k64.clear();
            k64.extend(seg.iter().map(|&o| {
                let key: u64 = bcodes.iter().zip(&kl.shifts).map(|(c, &s)| u64::from(c[o as usize]) << s).sum();
                (key << 32) | u64::from(o)
            }));
            k64.sort_unstable();
            for (d, k) in seg.iter_mut().zip(k64.iter()) {
                *d = *k as u32;
            }
        }
        Some(kl) => {
            k128.clear();
            k128.extend(seg.iter().map(|&o| {
                let key: u128 = bcodes.iter().zip(&kl.shifts).map(|(c, &s)| u128::from(c[o as usize]) << s).sum();
                (key << 32) | u128::from(o)
            }));
            k128.sort_unstable();
            for (d, k) in seg.iter_mut().zip(k128.iter()) {
                *d = *k as u32;
            }
        }
        None => seg.sort_unstable_by(|&a, &b| {
            for c in bcodes {
                match c[a as usize].cmp(&c[b as usize]) {
                    std::cmp::Ordering::Equal => {}
                    o => return o,
                }
            }
            a.cmp(&b)
        }),
    }
}

/// Cells sorted by (level of block `b`, cell) with per-level offsets.
fn level_perm(st: &Structure, b: usize, pool: &Pool) -> Result<(Vec<u32>, Vec<u32>), RfError> {
    let g = st.blocks[b].g;
    let nb = st.nb;
    let col: Vec<u32> = pool.install(|| st.cell_codes.par_chunks(nb.max(1)).map(|c| c[b]).collect());
    let (off, mut perm) = scatter_by_code(&col, g, pool)?;
    let bounds: Vec<usize> = off.iter().map(|&o| o as usize).collect();
    let pieces = split_bounds(&mut perm, &bounds);
    pool.install(|| pieces.into_par_iter().with_min_len(64).for_each(|p| p.sort_unstable()));
    Ok((perm, off))
}
