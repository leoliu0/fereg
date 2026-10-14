//! Cell structure of the intercept-only FE design (plan D5).
//!
//! Rows (observations) are ordered by pi: sorted by (g1, g2, ..., gQ, obs). A *cell* is a run of
//! rows with the same code tuple; a *segment* is the run of cells of one FE1 level. Blocks are
//! the non-FE1 terms; the reduced unknown x stacks their levels: small blocks first (jointly at
//! most [`SMALL_LIMIT`] levels, factored densely), then large blocks, each in term order.
//!
//! Everything here is structural (weight independent) and built once per sample.

use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::atomic::{AtomicU32, Ordering};

use rayon::prelude::*;

use crate::absorber::fe1_index;
use crate::error::RfError;
use crate::par::{CHUNK, Pool, alloc_par, alloc_uninit, alloc_vec, assume_init, chunk_range};
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
    /// Session index of the eliminated term, "FE1" of the solver ([`elimination_index`]; `None`
    /// under `noabsorb`: one synthetic level). It need not be [`fe1_index`], the term that
    /// absorbs the shifts of the FE normalization.
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
    /// Number of blocks (Q - 1).
    pub nb: usize,
    /// Codes of blocks 0..nb per cell, row-major M x nb ([`Structure::cell_codes`]). Private,
    /// as is `code_bound`: [`build`] checks that every code of block b is below `code_bound[b]`,
    /// and the operator kernels index the reduced vectors with the codes unchecked.
    cell_codes: Vec<u32>,
    /// Level count of each block at the build (the exclusive bound of its codes).
    code_bound: Vec<usize>,
    pub blocks: Vec<Block>,
    /// Reduced dimension sum_b g_b.
    pub m: usize,
    /// Small coordinates occupy `0..m_small`.
    pub m_small: usize,
    /// Fixed partition of the segments into parts of >~ [`CHUNK`] cells (boundaries, len = parts + 1).
    pub parts: Vec<u32>,
    /// Largest reduced vector (bytes) scattered entirely through accumulators
    /// (see `operator::Ops::accumulates_all`).
    pub acc_bytes_max: usize,
    /// sum_seg s^2, s = distinct small coordinates per segment (dense assembly work).
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

    /// Codes of blocks 0..nb per cell, row-major M x nb.
    #[inline]
    pub fn cell_codes(&self) -> &[u32] {
        &self.cell_codes
    }

    /// Whether reading the rows of [`Structure::cell_codes`] with width `nb` gives every cell's
    /// codes, and every code of the first `nblocks` blocks addresses `offs[b] + code*w .. + w`
    /// within `len` values: the precondition of the unchecked gathers and scatters of the
    /// operator kernels. It uses only private data (the codes and their bounds, checked in
    /// [`build`]), so no change of the public fields can invalidate it.
    pub fn codes_within(&self, nb: usize, offs: &[usize], w: usize, nblocks: usize, len: usize) -> bool {
        nb == self.code_bound.len()
            && nblocks <= nb
            && offs.len() >= nblocks
            && (0..nblocks).all(|b| self.code_bound[b].checked_mul(w).and_then(|e| e.checked_add(offs[b])).is_some_and(|e| e <= len))
    }

    pub fn has_large(&self) -> bool {
        self.blocks.iter().any(|b| !b.small)
    }

    pub fn nparts(&self) -> usize {
        self.parts.len() - 1
    }

    /// Groups the fixed parts so that `ntasks x acc_len <= 2^22` (the per-task accumulators stay
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

/// Candidates for elimination have at least this share of the largest term's levels.
const ELIM_CANDIDATE_SHARE: usize = 4;
/// Assembly work per observation below which the default elimination is kept unexamined.
const ELIM_CHEAP_PER_OBS: u64 = 64;

/// The term eliminated exactly (the solver's FE1). With two or fewer terms, or one candidate,
/// it is [`fe1_index`] (the most levels). With three or more terms, among the terms with at
/// least 1/[`ELIM_CANDIDATE_SHARE`] of the largest G it is the one minimizing
/// E = sum_{other terms o} sum_{levels s} d_{s,o}^2, d_{s,o} = distinct levels of o within level s:
/// the assembly work of the reduced system, which also measures how densely its blocks are
/// coupled. Measured on a 7.3M-row citation panel (stkcd#year 41k, lpid#year 121k,
/// stkcd#lsid 151k levels): eliminating the largest term (E = 1.7e10) took 1665 PCG iterations
/// (36 s), lpid#year (E = 1.7e9) 208 (3.5 s). On the usual panels (individual, firm, year) the
/// largest term is the only candidate or its levels are small, so only a histogram is computed.
pub fn elimination_index(terms: &[Term], pool: &Pool) -> Result<Option<usize>, RfError> {
    Ok(choose_elimination(terms, pool)?.0)
}

/// Observation ids sorted by a term: its g + 1 level offsets and the ids ([`scatter_by_code`]).
type SortedObs = (Vec<u32>, Vec<u32>);

/// [`elimination_index`], with the observations sorted by the chosen term when the choice
/// sorted them (several candidates).
fn choose_elimination(terms: &[Term], pool: &Pool) -> Result<(Option<usize>, Option<SortedObs>), RfError> {
    let default = fe1_index(terms);
    let Some(f0) = default else { return Ok((None, None)) };
    if terms.len() < 3 {
        return Ok((default, None));
    }
    let cands: Vec<usize> = (0..terms.len()).filter(|&q| terms[q].g * ELIM_CANDIDATE_SHARE >= terms[f0].g).collect();
    if cands.len() < 2 {
        return Ok((default, None));
    }
    // E of the default is at most (Q - 1)*sum_s n_s^2 (n_s: observations of level s): when that
    // is already small the reduced system is cheap whatever the choice (individual-level
    // panels), so the pair counts are skipped.
    let mut size = vec![0u64; terms[f0].g];
    for &c in &terms[f0].codes {
        size[c as usize] += 1;
    }
    let bound = (terms.len() as u64 - 1) * size.iter().map(|n| n * n).sum::<u64>();
    if bound <= ELIM_CHEAP_PER_OBS * terms[f0].codes.len() as u64 {
        return Ok((default, None));
    }
    // E of every candidate from the observations sorted by it; the sort of the best one so far
    // is kept for the structure.
    let mut best: Option<((u64, std::cmp::Reverse<usize>, usize), SortedObs)> = None;
    for &f in &cands {
        let sorted = scatter_by_code(&terms[f].codes, terms[f].g, pool)?;
        let key = (distinct_work(terms, f, &sorted, pool), std::cmp::Reverse(terms[f].g), f);
        if best.as_ref().is_none_or(|(k, _)| key < *k) {
            best = Some((key, sorted));
        }
    }
    Ok(best.map_or((default, None), |((_, _, f), sorted)| (Some(f), Some(sorted))))
}

/// E of eliminating term `f` (see [`elimination_index`]): sum over the other terms o and the
/// levels s of f of d_{s,o}^2, d_{s,o} the distinct levels of o among the observations of s.
/// `sorted`: the observations sorted by f. One pass over the levels of f in parallel groups, each
/// marking the levels of o it meets with the level s (stamp arrays); the integer sums do not
/// depend on the grouping.
fn distinct_work(terms: &[Term], f: usize, sorted: &SortedObs, pool: &Pool) -> u64 {
    let (off, ids) = (&sorted.0, &sorted.1);
    let others: Vec<&Term> = terms.iter().enumerate().filter(|&(q, _)| q != f).map(|(_, t)| t).collect();
    let groups = obs_groups(off);
    pool.install(|| {
        groups
            .par_windows(2)
            .map_init(
                || others.iter().map(|t| vec![0u32; t.g]).collect::<Vec<_>>(),
                |stamps, w| {
                    let mut e = 0u64;
                    let mut d = vec![0u64; others.len()];
                    for s in w[0] as usize..w[1] as usize {
                        // Level s marks the levels of o it holds with s + 1 (0 = unmarked).
                        let tag = s as u32 + 1;
                        d.fill(0);
                        for &i in &ids[off[s] as usize..off[s + 1] as usize] {
                            for ((t, st), dk) in others.iter().zip(stamps.iter_mut()).zip(d.iter_mut()) {
                                let c = t.codes[i as usize] as usize;
                                if st[c] != tag {
                                    st[c] = tag;
                                    *dk += 1;
                                }
                            }
                        }
                        e += d.iter().map(|x| x * x).sum::<u64>();
                    }
                    e
                },
            )
            .sum()
    })
}

/// Builds the structure for intercept-only `terms` over `n` kept observations; small blocks
/// jointly hold at most `small_limit` levels. Designs with slopes use [`crate::slopes`].
pub fn build(terms: &[Term], n: usize, small_limit: usize, pool: &Pool) -> Result<Structure, RfError> {
    if terms.iter().any(|t| !t.slopes.is_empty() || !t.intercept) {
        return Err(RfError::Internal("structure::build: slope terms belong to slopes::SlopeSolver".into()));
    }
    let (fe1, sorted1) = choose_elimination(terms, pool)?;
    let (g1, code1): (usize, Option<&[u32]>) = match fe1 {
        Some(f) => (terms[f].g, Some(&terms[f].codes)),
        None => (1, None),
    };

    // Small/large classification: ascending G while the running total stays <= small_limit.
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

    // 1. Observation ids by segment (ascending within each segment).
    let (seg_obs_off, mut pi) = match (code1, sorted1) {
        (None, _) => (vec![0u32, n as u32], identity(n, pool)?),
        (Some(_), Some(sorted)) => sorted,
        (Some(c1), None) => scatter_by_code(c1, g1, pool)?,
    };

    // 2. Per segment: sort the rows by (codes, obs), find the cells, record their codes, first
    // rows and distinct small coordinates. A group never has more cells than rows, so it writes
    // its cells from its first row on; they are moved into place once the counts are known.
    let groups = obs_groups(&seg_obs_off);
    let row_bounds: Vec<usize> = groups.iter().map(|&s| seg_obs_off[s as usize] as usize).collect();
    let mut seg_off = alloc_par(g1 + 1, 0u32, pool)?;
    let mut cell_codes = alloc_uninit::<u32>(n * nb)?;
    let pass = SegPass {
        seg_obs_off: &seg_obs_off,
        bcodes: &bcodes,
        layout: key_layout(&blocks),
        small_off: blocks.iter().filter(|b| b.small).map(|b| b.off as u32).collect(),
        m_small,
        nb,
    };
    let outs: Vec<GroupCells> = {
        let pi_pieces = split_bounds(&mut pi, &row_bounds);
        let seg_bounds: Vec<usize> = groups.iter().map(|&s| s as usize).collect();
        let cnt_pieces = split_bounds(&mut seg_off[1..], &seg_bounds);
        let cc_bounds: Vec<usize> = row_bounds.iter().map(|&r| r * nb).collect();
        let cc_pieces = split_bounds(&mut cell_codes, &cc_bounds);
        pool.install(|| {
            pi_pieces
                .into_par_iter()
                .zip(cnt_pieces)
                .zip(cc_pieces)
                .enumerate()
                .map_init(KeyBufs::default, |bufs, (gi, ((pis, cnt), cc))| {
                    let (segs, base) = (groups[gi] as usize..groups[gi + 1] as usize, row_bounds[gi]);
                    let (k64, k128) = (&mut bufs.k64, &mut bufs.k128);
                    match (&pass.layout, nb) {
                        (Some(kl), 1) if kl.width <= 32 => pass.group_c::<u64, 1>(segs, base, pis, cnt, cc, k64),
                        (Some(kl), 2) if kl.width <= 32 => pass.group_c::<u64, 2>(segs, base, pis, cnt, cc, k64),
                        (Some(kl), 3) if kl.width <= 32 => pass.group_c::<u64, 3>(segs, base, pis, cnt, cc, k64),
                        (Some(kl), _) if kl.width <= 32 => pass.group(segs, base, pis, cnt, cc, k64),
                        (Some(_), 2) => pass.group_c::<u128, 2>(segs, base, pis, cnt, cc, k128),
                        (Some(_), 3) => pass.group_c::<u128, 3>(segs, base, pis, cnt, cc, k128),
                        (Some(_), _) => pass.group(segs, base, pis, cnt, cc, k128),
                        (None, _) => pass.group_cmp(segs, base, pis, cnt, cc),
                    }
                })
                .collect()
        })
    };
    for s in 0..g1 {
        seg_off[s + 1] += seg_off[s];
    }
    let ncells = seg_off[g1] as usize;
    let trivial = ncells == n;
    let small_work: u64 = outs.iter().map(|o| o.small_work).sum();

    // 3. Cells into place (only when some cell holds several rows).
    let mut cell_obs = Vec::new();
    if !trivial {
        for (gi, o) in outs.iter().enumerate() {
            let (src, dst) = (row_bounds[gi], seg_off[groups[gi] as usize] as usize);
            if src != dst {
                cell_codes.copy_within(src * nb..(src + o.cells) * nb, dst * nb);
            }
        }
        cell_codes.truncate(ncells * nb);
        cell_codes.shrink_to_fit();
        let mut co = alloc_uninit::<u32>(ncells + 1)?;
        let cb: Vec<usize> = groups.iter().map(|&s| seg_off[s as usize] as usize).collect();
        let pieces = split_bounds(&mut co[..ncells], &cb);
        pool.install(|| {
            pieces.into_par_iter().zip(&outs).enumerate().for_each(|(gi, (co, o))| match &o.starts {
                Some(st) => co.iter_mut().zip(st).for_each(|(d, &s)| {
                    d.write(s);
                }),
                None => co.iter_mut().enumerate().for_each(|(k, d)| {
                    d.write((row_bounds[gi] + k) as u32);
                }),
            })
        });
        co[ncells].write(n as u32);
        // SAFETY: each group's piece holds its cells, all written above, and the last entry.
        cell_obs = unsafe { assume_init(co) };
    }
    // SAFETY: the groups wrote the codes of all ncells cells, now at 0..ncells*nb.
    let cell_codes = unsafe { assume_init(cell_codes) };
    drop(outs);
    // Every cell code below its block's level count: the operator kernels index the reduced
    // vectors with them unchecked ([`Structure::codes_within`]).
    let code_bound: Vec<usize> = blocks.iter().map(|b| b.g).collect();
    let in_range = nb == 0
        || pool.install(|| {
            cell_codes.par_chunks(CHUNK * nb).all(|ch| ch.chunks_exact(nb).all(|cc| cc.iter().zip(&code_bound).all(|(&c, &g)| (c as usize) < g)))
        });
    if !in_range {
        return Err(RfError::Internal("cell code out of range of its level count".into()));
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
        code_bound,
        blocks,
        m,
        m_small,
        parts,
        acc_bytes_max: crate::operator::ACC_BYTES_MAX,
        small_work,
    };
    // Dense small-block work sum_seg s^2 too large: every block is treated as large.
    if st.m_small > 0 && small_work > 64 * st.ncells as u64 {
        st.blocks.iter_mut().for_each(|b| b.small = false);
        st.m_small = 0;
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

/// Segment boundaries grouping consecutive segments into >~ CHUNK rows (for parallel passes).
pub(crate) fn obs_groups(seg_obs_off: &[u32]) -> Vec<u32> {
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

/// Raw pointer shared by tasks that write disjoint positions.
#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}
impl<T> SyncPtr<T> {
    #[inline]
    fn get(self) -> *mut T {
        self.0
    }
}

/// `0, 1, ..., n - 1`.
fn identity(n: usize, pool: &Pool) -> Result<Vec<u32>, RfError> {
    let mut v = alloc_uninit::<u32>(n)?;
    pool.install(|| {
        v.par_chunks_mut(CHUNK).enumerate().for_each(|(c, ch)| {
            ch.iter_mut().zip((c * CHUNK) as u32..).for_each(|(x, i)| {
                x.write(i);
            });
        })
    });
    // SAFETY: every chunk wrote each of its elements.
    Ok(unsafe { assume_init(v) })
}

/// Stable counting sort of observation ids by `codes` (`g` levels): the g + 1 level offsets and
/// the ids, ascending within each level.
pub(crate) fn scatter_by_code(codes: &[u32], g: usize, pool: &Pool) -> Result<(Vec<u32>, Vec<u32>), RfError> {
    counting_sort(codes.len(), g, |i| codes[i], pool)
}

/// Stable counting sort of `0..n` by `code(i) < g`: the g + 1 level offsets and the ids,
/// ascending within each level. The result is unique, so the number of parts (which follows the
/// thread count) cannot change it.
fn counting_sort(n: usize, g: usize, code: impl Fn(usize) -> u32 + Sync, pool: &Pool) -> Result<(Vec<u32>, Vec<u32>), RfError> {
    let nc = n.div_ceil(CHUNK);
    let mut off = alloc_par(g + 1, 0u32, pool)?;
    // Codes already ascending (data sorted by this factor): the ids are the identity and the
    // offsets are the run starts, recorded while the order is checked. A descent stops the check
    // and resets the partial offsets.
    let op = SyncPtr(off.as_mut_ptr());
    let sorted = n > 0
        && pool.install(|| {
            (0..nc).into_par_iter().all(|c| {
                let r = chunk_range(c, n, CHUNK);
                for i in r.start.max(1)..r.end {
                    let (a, b) = (code(i - 1) as usize, code(i) as usize);
                    if b < a || b >= g {
                        return false;
                    }
                    for k in a + 1..=b {
                        // SAFETY: k < g. The stores are atomic because the chunks of unsorted
                        // codes may write the same level concurrently.
                        unsafe { AtomicU32::from_ptr(op.get().add(k)).store(i as u32, Ordering::Relaxed) };
                    }
                }
                true
            })
        });
    if sorted && (code(n - 1) as usize) < g {
        let last = code(n - 1) as usize;
        off[last + 1..].fill(n as u32);
        return Ok((off, identity(n, pool)?));
    }
    off.fill(0);
    let mut ids = alloc_uninit::<u32>(n)?;
    let nt = pool.threads().min(nc).max(1);
    // Parts: one per thread while their histograms (parts x g counters) stay within 2n.
    let parts = nt.min(2 * n / g.max(1)).max(1);
    if nt == 1 {
        for i in 0..n {
            off[code(i) as usize + 1] += 1;
        }
        for k in 0..g {
            off[k + 1] += off[k];
        }
        // off[k] is the cursor of level k; afterwards it holds the end of k (= start of k + 1).
        for i in 0..n {
            let c = &mut off[code(i) as usize];
            ids[*c as usize].write(i as u32);
            *c += 1;
        }
        off.copy_within(0..g, 1);
        off[0] = 0;
        // SAFETY: the level cursors cover 0..n, each position once.
        return Ok((off, unsafe { assume_init(ids) }));
    }
    if parts < nt || n < 64 * g * parts {
        // Many levels: the parts would share the cache lines of every level.
        return bucket_sort(n, g, &code, off, ids, nt, pool);
    }
    let bnd: Vec<usize> = (0..=parts).map(|j| j * n / parts).collect();
    let mut hist = alloc_par(parts * g, 0u32, pool)?;
    pool.install(|| {
        hist.par_chunks_mut(g).enumerate().for_each(|(j, h)| {
            for i in bnd[j]..bnd[j + 1] {
                h[code(i) as usize] += 1;
            }
        })
    });
    // Per level: exclusive prefix over the parts (in place) and the level total (into off).
    const KC: usize = 4096;
    let hp = SyncPtr(hist.as_mut_ptr());
    let row = |j: usize, k0: usize, len: usize| {
        // SAFETY: concurrent tasks touch disjoint level ranges k0..k0 + len of every part row.
        unsafe { std::slice::from_raw_parts_mut(hp.get().add(j * g + k0), len) }
    };
    let sums: Vec<u32> = pool.install(|| {
        off[..g]
            .par_chunks_mut(KC)
            .enumerate()
            .map(|(r, o)| {
                for j in 0..parts {
                    for (x, t) in row(j, r * KC, o.len()).iter_mut().zip(o.iter_mut()) {
                        let c = *x;
                        *x = *t;
                        *t += c;
                    }
                }
                o.iter().sum::<u32>()
            })
            .collect()
    });
    let mut base = Vec::with_capacity(sums.len());
    let mut acc = 0u32;
    for s in sums {
        base.push(acc);
        acc += s;
    }
    // Level offsets, then the part cursors.
    pool.install(|| {
        off[..g].par_chunks_mut(KC).enumerate().for_each(|(r, o)| {
            let mut run = base[r];
            for t in o.iter_mut() {
                let c = *t;
                *t = run;
                run += c;
            }
            for j in 0..parts {
                row(j, r * KC, o.len()).iter_mut().zip(o.iter()).for_each(|(x, s)| *x += s);
            }
        })
    });
    off[g] = n as u32;
    let ip = SyncPtr(ids.as_mut_ptr().cast::<u32>());
    pool.install(|| {
        hist.par_chunks_mut(g).enumerate().for_each(|(j, cur)| {
            for i in bnd[j]..bnd[j + 1] {
                let c = &mut cur[code(i) as usize];
                // SAFETY: the cursors of all (level, part) pairs cover 0..n disjointly.
                unsafe { *ip.get().add(*c as usize) = i as u32 };
                *c += 1;
            }
        })
    });
    // SAFETY: as above, every position was written once.
    Ok((off, unsafe { assume_init(ids) }))
}

/// [`counting_sort`] for many levels (parallel): the (code, id) pairs are first scattered stably
/// into buckets of 2^shift consecutive levels -- large regions per (bucket, part) -- and every
/// bucket is then counting-sorted on its own. `off` is zeroed (g + 1), `ids` has n slots.
fn bucket_sort(
    n: usize,
    g: usize,
    code: &(impl Fn(usize) -> u32 + Sync),
    mut off: Vec<u32>,
    mut ids: Vec<MaybeUninit<u32>>,
    parts: usize,
    pool: &Pool,
) -> Result<(Vec<u32>, Vec<u32>), RfError> {
    // About 16 buckets per thread.
    let shift = bits_for(g.div_ceil(16 * pool.threads()));
    let q = g.div_ceil(1 << shift);
    let bnd: Vec<usize> = (0..=parts).map(|j| j * n / parts).collect();
    let mut hist = alloc_vec(parts * q, 0u32)?;
    pool.install(|| {
        hist.par_chunks_mut(q).enumerate().for_each(|(j, h)| {
            for i in bnd[j]..bnd[j + 1] {
                let c = code(i);
                assert!((c as usize) < g, "code out of range");
                h[(c >> shift) as usize] += 1;
            }
        })
    });
    // Cursors in (bucket, part) order, and the bucket starts.
    let mut bstart = Vec::with_capacity(q + 1);
    let mut acc = 0u32;
    for b in 0..q {
        bstart.push(acc as usize);
        for j in 0..parts {
            let c = hist[j * q + b];
            hist[j * q + b] = acc;
            acc += c;
        }
    }
    bstart.push(n);
    let mut tmp = alloc_uninit::<u64>(n)?;
    let tp = SyncPtr(tmp.as_mut_ptr().cast::<u64>());
    pool.install(|| {
        hist.par_chunks_mut(q).enumerate().for_each(|(j, cur)| {
            for i in bnd[j]..bnd[j + 1] {
                let c = code(i);
                let p = &mut cur[(c >> shift) as usize];
                // SAFETY: the (bucket, part) cursors cover 0..n disjointly.
                unsafe { *tp.get().add(*p as usize) = (u64::from(c) << 32) | i as u64 };
                *p += 1;
            }
        })
    });
    // SAFETY: as above, every position was written once.
    let tmp = unsafe { assume_init(tmp) };
    let level_bounds: Vec<usize> = (0..=q).map(|b| (b << shift).min(g)).collect();
    let off_pieces = split_bounds(&mut off[..g], &level_bounds);
    let id_pieces = split_bounds(&mut ids, &bstart);
    pool.install(|| {
        off_pieces.into_par_iter().zip(id_pieces).enumerate().for_each_init(Vec::new, |cnt, (b, (o, d))| {
            let (lo, base) = (level_bounds[b], bstart[b]);
            cnt.clear();
            cnt.resize(o.len(), 0u32);
            let pairs = &tmp[base..bstart[b + 1]];
            for &p in pairs {
                cnt[(p >> 32) as usize - lo] += 1;
            }
            let mut run = 0u32;
            for (c, s) in cnt.iter_mut().zip(o.iter_mut()) {
                *s = (base as u32) + run;
                let k = *c;
                *c = run;
                run += k;
            }
            for &p in pairs {
                let c = &mut cnt[(p >> 32) as usize - lo];
                d[*c as usize].write(p as u32);
                *c += 1;
            }
        })
    });
    off[g] = n as u32;
    // SAFETY: every bucket wrote each slot of its id range.
    Ok((off, unsafe { assume_init(ids) }))
}

/// Packed sort key layout: bit shift and width of each block's code (block 0 most significant)
/// and the total code width; `None` when the codes do not fit 96 bits (comparator sort).
struct KeyLayout {
    shifts: Vec<u32>,
    bits: Vec<u32>,
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
    Some(KeyLayout { shifts, bits, width })
}

/// Row sort key: the packed block codes above the 32-bit observation id.
trait Packed: Copy + Ord {
    fn pack(o: u32, bcodes: &[&[u32]], shifts: &[u32]) -> Self;
    fn obs(self) -> u32;
    /// The code part (shared by the rows of a cell).
    fn cell(self) -> Self;
    fn code(self, shift: u32, bits: u32) -> u32;
}

macro_rules! packed {
    ($t:ty) => {
        impl Packed for $t {
            #[inline]
            fn pack(o: u32, bcodes: &[&[u32]], shifts: &[u32]) -> Self {
                let mut k: $t = 0;
                for (c, &s) in bcodes.iter().zip(shifts) {
                    k |= <$t>::from(c[o as usize]) << s;
                }
                (k << 32) | <$t>::from(o)
            }
            #[inline]
            fn obs(self) -> u32 {
                self as u32
            }
            #[inline]
            fn cell(self) -> Self {
                self >> 32
            }
            #[inline]
            fn code(self, shift: u32, bits: u32) -> u32 {
                ((self >> (32 + shift)) as u32) & ((1u64 << bits) - 1) as u32
            }
        }
    };
}
packed!(u64);
packed!(u128);

/// Inputs of the per-segment pass of [`build`].
struct SegPass<'a> {
    seg_obs_off: &'a [u32],
    bcodes: &'a [&'a [u32]],
    layout: Option<KeyLayout>,
    /// Coordinate offsets of the small blocks (blocks `0..small_off.len()`).
    small_off: Vec<u32>,
    m_small: usize,
    nb: usize,
}

/// Cells of one group of segments.
struct GroupCells {
    cells: usize,
    /// First row of each cell; `None` while every cell holds one row (cell k starts at row k).
    starts: Option<Vec<u32>>,
    /// sum_seg s^2 over the group's segments.
    small_work: u64,
}

/// Cell writer of one group: codes into `cc` (local cell index), first rows, small coordinates.
struct CellOut<'p> {
    p: &'p SegPass<'p>,
    base: usize,
    out: GroupCells,
    mark: Vec<u32>,
    stamp: u32,
    distinct: u64,
}

impl<'p> CellOut<'p> {
    fn new(p: &'p SegPass<'p>, base: usize) -> Self {
        CellOut { p, base, out: GroupCells { cells: 0, starts: None, small_work: 0 }, mark: vec![0; p.m_small], stamp: 1, distinct: 0 }
    }

    /// A cell starts at local row `row`; `code(b)` is its code of block b.
    #[inline]
    fn open(&mut self, row: usize, cc: &mut [MaybeUninit<u32>], code: impl Fn(usize) -> u32) {
        let nb = self.p.nb;
        let c = self.out.cells;
        for (b, d) in cc[c * nb..(c + 1) * nb].iter_mut().enumerate() {
            let x = code(b);
            d.write(x);
            // Small blocks come first.
            if let Some(&o) = self.p.small_off.get(b) {
                let m = &mut self.mark[(o + x) as usize];
                if *m != self.stamp {
                    *m = self.stamp;
                    self.distinct += 1;
                }
            }
        }
        if let Some(st) = &mut self.out.starts {
            st.push((self.base + row) as u32);
        }
        self.out.cells += 1;
    }

    /// A row joins the current cell.
    #[inline]
    fn join(&mut self) {
        if self.out.starts.is_none() {
            self.out.starts = Some((0..self.out.cells).map(|k| (self.base + k) as u32).collect());
        }
    }

    fn end_segment(&mut self) {
        self.out.small_work += self.distinct * self.distinct;
        self.distinct = 0;
        self.stamp += 1;
    }
}

/// Key buffers reused by one worker across groups.
#[derive(Default)]
struct KeyBufs {
    k64: Vec<u64>,
    k128: Vec<u128>,
}

impl SegPass<'_> {
    /// Local row range of segment `s` in a group starting at row `base`.
    fn rows(&self, s: usize, base: usize) -> Range<usize> {
        self.seg_obs_off[s] as usize - base..self.seg_obs_off[s + 1] as usize - base
    }

    /// Packed-key pass over the segments `segs` (rows from `base`); `cnt[s]` gets the cells of s.
    fn group<K: Packed>(
        &self,
        segs: Range<usize>,
        base: usize,
        pis: &mut [u32],
        cnt: &mut [u32],
        cc: &mut [MaybeUninit<u32>],
        keys: &mut Vec<K>,
    ) -> GroupCells {
        let kl = self.layout.as_ref().expect("packed layout");
        keys.clear();
        keys.extend(pis.iter().map(|&o| K::pack(o, self.bcodes, &kl.shifts)));
        let mut w = CellOut::new(self, base);
        for s in segs.clone() {
            let r = self.rows(s, base);
            let ks = &mut keys[r.clone()];
            if !ks.is_sorted() {
                ks.sort_unstable();
                pis[r.clone()].iter_mut().zip(ks.iter()).for_each(|(d, k)| *d = k.obs());
            }
            let c0 = w.out.cells;
            for (i, &k) in ks.iter().enumerate() {
                if i == 0 || k.cell() != ks[i - 1].cell() {
                    w.open(r.start + i, cc, |b| k.code(kl.shifts[b], kl.bits[b]));
                } else {
                    w.join();
                }
            }
            cnt[s - segs.start] = (w.out.cells - c0) as u32;
            w.end_segment();
        }
        w.out
    }

    /// [`Self::group`] specialized to `NB` blocks (the common small counts): one loop over the
    /// rows that checks the order as it goes (a segment found out of order is rolled back, sorted
    /// and redone), and counts the distinct codes of block 0 -- the most significant key part,
    /// so ascending within a segment -- by their changes rather than through marks.
    fn group_c<K: Packed, const NB: usize>(
        &self,
        segs: Range<usize>,
        base: usize,
        pis: &mut [u32],
        cnt: &mut [u32],
        cc: &mut [MaybeUninit<u32>],
        keys: &mut Vec<K>,
    ) -> GroupCells {
        let kl = self.layout.as_ref().expect("packed layout");
        let bc: [&[u32]; NB] = std::array::from_fn(|b| self.bcodes[b]);
        let shifts: [u32; NB] = std::array::from_fn(|b| kl.shifts[b]);
        let bits: [u32; NB] = std::array::from_fn(|b| kl.bits[b]);
        let nsmall = self.small_off.len();
        keys.clear();
        keys.extend(pis.iter().map(|&o| K::pack(o, &bc, &shifts)));
        let mut mark = vec![0u32; if nsmall > 1 { self.m_small } else { 0 }];
        let mut stamp = 0u32;
        let (mut cells, mut starts, mut work) = (0usize, None::<Vec<u32>>, 0u64);
        for s in segs.clone() {
            let r = self.rows(s, base);
            let (c0, had_starts) = (cells, starts.is_some());
            let distinct = 'attempt: loop {
                stamp += 1;
                let mut distinct = 0u64;
                let ks = &keys[r.clone()];
                for (i, &k) in ks.iter().enumerate() {
                    if i > 0 && k <= ks[i - 1] {
                        // Out of order: undo the segment's cells, sort it and redo it.
                        cells = c0;
                        match &mut starts {
                            Some(st) if had_starts => st.truncate(c0),
                            _ => starts = None,
                        }
                        let ks = &mut keys[r.clone()];
                        ks.sort_unstable();
                        pis[r.clone()].iter_mut().zip(ks.iter()).for_each(|(d, k)| *d = k.obs());
                        continue 'attempt;
                    }
                    if i > 0 && k.cell() == ks[i - 1].cell() {
                        if starts.is_none() {
                            starts = Some((0..cells).map(|c| (base + c) as u32).collect());
                        }
                        continue;
                    }
                    let codes: [u32; NB] = std::array::from_fn(|b| k.code(shifts[b], bits[b]));
                    let dst: &mut [MaybeUninit<u32>; NB] = (&mut cc[cells * NB..(cells + 1) * NB]).try_into().expect("NB codes");
                    for (d, &x) in dst.iter_mut().zip(&codes) {
                        d.write(x);
                    }
                    if nsmall > 0 && (i == 0 || codes[0] != ks[i - 1].code(shifts[0], bits[0])) {
                        distinct += 1;
                    }
                    for b in 1..nsmall.min(NB) {
                        let m = &mut mark[(self.small_off[b] + codes[b]) as usize];
                        if *m != stamp {
                            *m = stamp;
                            distinct += 1;
                        }
                    }
                    if let Some(st) = &mut starts {
                        st.push((base + r.start + i) as u32);
                    }
                    cells += 1;
                }
                break distinct;
            };
            cnt[s - segs.start] = (cells - c0) as u32;
            work += distinct * distinct;
        }
        GroupCells { cells, starts, small_work: work }
    }

    /// Comparator pass (codes wider than 96 bits).
    fn group_cmp(&self, segs: Range<usize>, base: usize, pis: &mut [u32], cnt: &mut [u32], cc: &mut [MaybeUninit<u32>]) -> GroupCells {
        let bc = self.bcodes;
        let cmp = |a: &u32, b: &u32| {
            for c in bc {
                match c[*a as usize].cmp(&c[*b as usize]) {
                    std::cmp::Ordering::Equal => {}
                    o => return o,
                }
            }
            a.cmp(b)
        };
        let mut w = CellOut::new(self, base);
        for s in segs.clone() {
            let r = self.rows(s, base);
            let seg = &mut pis[r.clone()];
            if !seg.is_sorted_by(|a, b| cmp(a, b).is_le()) {
                seg.sort_unstable_by(cmp);
            }
            let c0 = w.out.cells;
            for i in 0..seg.len() {
                let o = seg[i] as usize;
                if i == 0 || bc.iter().any(|c| c[seg[i - 1] as usize] != c[o]) {
                    w.open(r.start + i, cc, |b| bc[b][o]);
                } else {
                    w.join();
                }
            }
            cnt[s - segs.start] = (w.out.cells - c0) as u32;
            w.end_segment();
        }
        w.out
    }
}

/// Cells sorted by (level of block `b`, cell) with per-level offsets.
fn level_perm(st: &Structure, b: usize, pool: &Pool) -> Result<(Vec<u32>, Vec<u32>), RfError> {
    let (nb, cc) = (st.nb, &st.cell_codes);
    let (off, perm) = counting_sort(st.ncells, st.blocks[b].g, |c| cc[c * nb + b], pool)?;
    Ok((perm, off))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::par::with_pool;
    use crate::rng::SplitMix64;

    fn term(codes: Vec<u32>, g: usize) -> Term {
        Term { factors: vec![0], codes, g, intercept: true, slopes: vec![], slope_vars: vec![], save: false }
    }

    /// Checks `st` against the definitions with plain sorts.
    fn check(terms: &[Term], n: usize, small_limit: usize, st: &Structure) {
        let fe1 = with_pool(1, |pool| elimination_index(terms, pool)).unwrap().unwrap();
        assert_eq!(st.fe1, fe1);
        let c1 = |o: usize| fe1.map_or(0, |f| terms[f].codes[o]);
        let bt: Vec<usize> = st.blocks.iter().map(|b| b.term).collect();
        let tuple = |o: usize| -> Vec<u32> { std::iter::once(c1(o)).chain(bt.iter().map(|&q| terms[q].codes[o])).collect() };
        let mut pi: Vec<u32> = (0..n as u32).collect();
        pi.sort_by_key(|&o| (tuple(o as usize), o));
        assert_eq!(st.pi, pi);
        let (mut seg_off, mut starts, mut cc) = (vec![0u32], Vec::new(), Vec::new());
        for row in 0..n {
            let o = pi[row] as usize;
            if row == 0 || tuple(pi[row - 1] as usize) != tuple(o) {
                while seg_off.len() <= c1(o) as usize {
                    seg_off.push(starts.len() as u32);
                }
                starts.push(row as u32);
                cc.extend_from_slice(&tuple(o)[1..]);
            }
        }
        let ncells = starts.len();
        while seg_off.len() <= st.g1 {
            seg_off.push(ncells as u32);
        }
        assert_eq!(st.seg_off, seg_off);
        assert_eq!(st.ncells, ncells);
        assert_eq!(st.cell_codes, cc);
        starts.push(n as u32);
        assert_eq!(st.cell_obs_off, (ncells != n).then_some(starts));
        // Small classification as documented, then sum_seg s^2.
        let mut by_g: Vec<usize> = bt.clone();
        by_g.sort_by_key(|&q| (terms[q].g, q));
        let mut small = vec![false; terms.len()];
        let mut total = 0;
        for &q in &by_g {
            if total + terms[q].g > small_limit {
                break;
            }
            total += terms[q].g;
            small[q] = true;
        }
        let mut work = 0u64;
        if total > 0 {
            for s in 0..st.g1 {
                let mut d: Vec<usize> = Vec::new();
                for c in seg_off[s] as usize..seg_off[s + 1] as usize {
                    for (b, blk) in st.blocks.iter().enumerate() {
                        if small[blk.term] {
                            d.push(blk.off + cc[c * bt.len() + b] as usize);
                        }
                    }
                }
                d.sort_unstable();
                d.dedup();
                work += (d.len() as u64).pow(2);
            }
        }
        assert_eq!(st.small_work, work);
        let demote = total > 0 && work > 64 * ncells as u64;
        for (b, blk) in st.blocks.iter().enumerate() {
            assert_eq!(blk.small, small[blk.term] && !demote);
            if blk.small {
                assert!(blk.perm.is_empty());
                continue;
            }
            let mut perm: Vec<u32> = (0..ncells as u32).collect();
            perm.sort_by_key(|&c| (cc[c as usize * bt.len() + b], c));
            let mut off = vec![0u32; blk.g + 1];
            for c in 0..ncells {
                off[cc[c * bt.len() + b] as usize + 1] += 1;
            }
            for k in 0..blk.g {
                off[k + 1] += off[k];
            }
            assert_eq!(blk.perm, perm);
            assert_eq!(blk.perm_off, off);
        }
    }

    fn codes(n: usize, g: usize, rng: &mut SplitMix64) -> Vec<u32> {
        (0..n).map(|_| (rng.next_u64() % g as u64) as u32).collect()
    }

    /// Rows reordered by their code tuples (data sorted by FE1, then the other terms).
    fn sorted(terms: &[Term]) -> Vec<Term> {
        let n = terms[0].codes.len();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by_key(|&o| terms.iter().map(|t| t.codes[o]).collect::<Vec<_>>());
        terms.iter().map(|t| term(idx.iter().map(|&o| t.codes[o]).collect(), t.g)).collect()
    }

    #[test]
    fn build_matches_reference() {
        let mut rng = SplitMix64::new(7);
        let mut cases: Vec<(Vec<Term>, Vec<usize>)> = Vec::new();
        // Unsorted with distinct cells, and with repeated cells.
        let n = 6000;
        let a = vec![term(codes(n, 600, &mut rng), 600), term(codes(n, 20, &mut rng), 20), term(codes(n, 150, &mut rng), 150)];
        let b = vec![term(codes(n, 9, &mut rng), 9), term(codes(n, 40, &mut rng), 40), term(codes(n, 3, &mut rng), 3), term(codes(n, 2, &mut rng), 2)];
        cases.push((sorted(&a), vec![4096, 50, 0]));
        cases.push((a, vec![4096, 50, 0]));
        cases.push((sorted(&b), vec![4096, 3, 0]));
        cases.push((b, vec![4096, 3, 0]));
        // Dense small work above the limit (blocks demoted), a single term, no term.
        cases.push((vec![term(codes(n, 2, &mut rng), 2), term(codes(n, 1500, &mut rng), 1500)], vec![4096]));
        cases.push((vec![term(codes(n, 700, &mut rng), 700)], vec![4096]));
        cases.push((Vec::new(), vec![4096]));
        // 33-96 bit keys with two and three blocks, and five blocks (generic pass).
        let n = 6000;
        let wide_g = |rng: &mut SplitMix64, k: u32| term(codes(n, 6, rng).iter().map(|&c| c << 14).collect(), 1 << k);
        let k2 = vec![term(codes(n, 900, &mut rng), 1 << 20), wide_g(&mut rng, 17), wide_g(&mut rng, 17)];
        let k3 = vec![term(codes(n, 900, &mut rng), 1 << 20), term(codes(n, 3, &mut rng), 3), wide_g(&mut rng, 17), wide_g(&mut rng, 18)];
        cases.push((sorted(&k2), vec![0]));
        cases.push((k2, vec![0]));
        cases.push((k3, vec![4096, 3, 0]));
        let five: Vec<Term> = [400, 3, 4, 5, 6, 2].iter().map(|&g| term(codes(n, g, &mut rng), g)).collect();
        cases.push((sorted(&five), vec![4096, 7, 0]));
        cases.push((five, vec![4096, 7, 0]));
        // Codes wider than 96 bits (comparator path; FE1 is the first wide term).
        let mut wide = vec![term(codes(3000, 3000, &mut rng), 3000)];
        for _ in 0..6 {
            wide.push(term(codes(3000, 4, &mut rng).iter().map(|&c| c << 18).collect(), 1 << 20));
        }
        cases.push((wide, vec![0]));
        // Several parallel groups and counting-sort parts.
        let n = 200_000;
        let big = vec![term(codes(n, 20_000, &mut rng), 20_000), term(codes(n, 10, &mut rng), 10), term(codes(n, 5000, &mut rng), 5000)];
        cases.push((sorted(&big), vec![4096]));
        cases.push((big, vec![4096]));
        for (terms, limits) in &cases {
            let n = terms.first().map_or(500, |t| t.codes.len());
            for &limit in limits {
                for threads in [1, 3, 8] {
                    let st = with_pool(threads, |pool| build(terms, n, limit, pool).unwrap()).unwrap();
                    check(terms, n, limit, &st);
                }
            }
        }
    }

    #[test]
    fn scatter_is_a_stable_counting_sort() {
        let mut rng = SplitMix64::new(11);
        let n = 300_000;
        // Few levels (per-part histograms), and many levels (buckets) at various thread counts.
        let mut inputs = vec![(codes(n, 7, &mut rng), 7), (codes(n, 500, &mut rng), 500), (codes(n, 50_000, &mut rng), 50_000)];
        inputs.push((codes(n, 290_000, &mut rng), 300_000));
        let mut s = codes(n, 1000, &mut rng);
        s.sort_unstable();
        inputs.push((s, 1200));
        inputs.push((Vec::new(), 4));
        for (c, g) in &inputs {
            let mut ids: Vec<u32> = (0..c.len() as u32).collect();
            ids.sort_by_key(|&i| c[i as usize]);
            let mut off = vec![0u32; g + 1];
            for &k in c {
                off[k as usize + 1] += 1;
            }
            for k in 0..*g {
                off[k + 1] += off[k];
            }
            for threads in [1, 4, 16] {
                let got = with_pool(threads, |pool| scatter_by_code(c, *g, pool).unwrap()).unwrap();
                assert_eq!(got, (off.clone(), ids.clone()));
            }
        }
    }

    /// Three comparable terms with dense coupling: the eliminated term minimizes
    /// E = sum_o sum_s d_{s,o}^2 (brute force), here not the default (first of the largest).
    #[test]
    fn elimination_minimizes_assembly_work() {
        let n = 4000usize;
        let x: Vec<u32> = (0..n).map(|i| (i / 40) as u32).collect();
        let y: Vec<u32> = (0..n).map(|i| (i % 100) as u32).collect();
        let z: Vec<u32> = (0..n).map(|i| ((i * 7) % 80) as u32).collect();
        let terms = vec![term(x, 100), term(y, 100), term(z, 80)];
        let e = |f: usize| -> u64 {
            (0..3)
                .filter(|&o| o != f)
                .map(|o| {
                    let mut sets = vec![std::collections::BTreeSet::new(); terms[f].g];
                    for i in 0..n {
                        sets[terms[f].codes[i] as usize].insert(terms[o].codes[i]);
                    }
                    sets.iter().map(|s| (s.len() * s.len()) as u64).sum::<u64>()
                })
                .sum()
        };
        let best = (0..3).min_by_key(|&f| e(f)).unwrap();
        assert_ne!(Some(best), fe1_index(&terms));
        for threads in [1, 4] {
            let got = with_pool(threads, |pool| elimination_index(&terms, pool)).unwrap().unwrap();
            assert_eq!(got, Some(best));
            let st = with_pool(threads, |pool| build(&terms, n, 0, pool)).unwrap().unwrap();
            check(&terms, n, 0, &st);
        }
    }
}
