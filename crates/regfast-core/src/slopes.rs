//! Heterogeneous slopes `fe#c.x`, `fe##c.x`, `fe##c.(x z)` (plan Phase 7).
//!
//! [`SlopeSolver`] implements [`Absorber`] whenever some absorb term has slopes or no intercept.
//! It follows plan D5 on an observation-level structure: rows are sorted by the eliminated term
//! E (FE1 = [`fe1_index`]; without an intercept-only term, the term with the most levels) and
//! every row is its own "cell". Level g of a term with design row h_i = (1?, x_i…) of width b owns
//! b consecutive reduced coordinates.
//!
//! - **Elimination.** E is eliminated exactly per level with a generalized inverse M_g⁺ of
//!   M_g = Σ_{i∈g} w_i h_i h_i'. reghdfe convention: with an intercept, the centered slope block
//!   is inverted by a symmetric sweep that drops pivots ≤ [`LEVEL_TOL`]·Σw x², so a level whose
//!   slope has no within variation gets slope 0 and intercept = mean.
//! - **Reduced operator** (per segment g of E): u_i = Σ_q h_{q,i}'P[q, g_q(i)],
//!   t = M_g⁺ Σ_i w_i h_{E,i} u_i', z_i = w_i (u_i − h_{E,i}'t), Y[q, g_q(i)] += h_{q,i} z_i'.
//!   The right-hand side is the same kernel with u_i := v_i; the residualization pass replaces
//!   v_i by d_i − h_{E,i}'t with d_i = v_i − u_i(x) and t the projection coefficients of d (α_E).
//! - **Preconditioner.** The exact dense factor of the small blocks (jointly ≤ `small_limit`
//!   coordinates, and only while the deterministic cost model of [`small_cost`] finds the dense
//!   factor cheap next to a few matvecs), per-level b×b block Jacobi on the large blocks,
//!   escalated once (the `cg` probe) to an exact sparse Cholesky of a large block when its
//!   assembly work Σ_s d_s²·b² and its nonzeros fit the [`SolverConfig`] budgets. approx-chol is
//!   never used here.
//! - **Kernels.** Row kernels work on column groups of compile-time width W ≤ [`MAX_W`]
//!   (`[f64; W]` rows: no per-row heap buffers or dynamic-length loops). Every row stores the
//!   first reduced coordinate of its level in each block (`coord`), so a row touches P and Y
//!   with one load/add per design slot. The level permutations of the large blocks (needed only
//!   by escalation and by the n×W scratch path) are built on first use.
//!
//! FeCoefs layout: p = m + G_E·b_E rows, `[x (m×R); α_E (G_E·b_E × R)]` (level-major, slots
//! intercept then slopes). Deterministic for any thread count: fixed task partitions, task-order
//! combines and level pulls, as the intercept-only solver; every cost-model decision depends on
//! the data only, never on the thread count.

use std::ops::Range;
use std::sync::OnceLock;
use std::time::Instant;

use faer::Mat;
use rayon::prelude::*;

use crate::absorber::{Absorber, FeCoefs, SolveInfo, SolveOpts, fe1_index};
use crate::api::Ctx;
use crate::cg::{self, CgSystem};
use crate::error::RfError;
use crate::operator::{Csr, REDUNDANT};
use crate::par::{CHUNK, Pool, alloc_vec, sum_chunks};
use crate::precond::dense::Dense;
use crate::precond::sparse_chol::SparseChol;
use crate::precond::{SOLVE_COLS, solve_tasks};
use crate::session::Term;
use crate::solve::{SolverConfig, log_phase};
use crate::structure::{obs_groups, scatter_by_code, split_bounds};

/// Relative pivot tolerance of the per-level generalized inverses of the eliminated term.
pub const LEVEL_TOL: f64 = 1e-12;
/// Relative pivot tolerance of the block-Jacobi inverses (as the dense factor's grounding).
const JACOBI_TOL: f64 = 1e-10;
/// Widest column group of the row kernels; wider blocks run in groups of [`WIDE_GROUP`].
pub const MAX_W: usize = 12;
const WIDE_GROUP: usize = 8;

// Cost model of the small-block choice: single-thread seconds on the reference machine (Ryzen 9
// 9950X), fitted on the fixest benchmark panels. Independent of the thread count (plan D3).
/// Matvec cost per row and per (design slot + column).
const C_ROW: f64 = 1e-9;
/// Dense small-block assembly cost per unit of Σ_s d_s² + n·b_s² (d_s: small coordinates of
/// segment s, b_s: small design slots per row).
const C_SMALL_ASM: f64 = 1e-9;
/// Dense factorization cost per multiply-add.
const C_FLOP: f64 = 1.2e-11;
/// The small set is factored densely while its cost is at most max(SMALL_FLOOR, SMALL_ITERS ×
/// matvec): cheap in absolute terms, or cheaper than the few iterations exactness can save.
const SMALL_FLOOR: f64 = 1e-3;
const SMALL_ITERS: f64 = 8.0;

/// Column groups of an r-column block: one group up to [`MAX_W`] columns, else groups of
/// [`WIDE_GROUP`].
fn col_groups(r: usize) -> Vec<Range<usize>> {
    if r <= MAX_W { vec![0..r] } else { (0..r).step_by(WIDE_GROUP).map(|j| j..(j + WIDE_GROUP).min(r)).collect() }
}

/// Width of the widest column group of an r-column block.
pub fn group_width(r: usize) -> usize {
    if r <= MAX_W { r } else { WIDE_GROUP }
}

#[inline(always)]
fn at<const W: usize>(v: &[f64], c: usize) -> &[f64; W] {
    v[c..c + W].try_into().expect("W-wide slice")
}

#[inline(always)]
fn at_mut<const W: usize>(v: &mut [f64], c: usize) -> &mut [f64; W] {
    (&mut v[c..c + W]).try_into().expect("W-wide slice")
}

/// Generalized inverse of the symmetric PSD k×k row-major `a`, in place, by the symmetric sweep.
/// Pivot p is dropped (row and column p of the result are 0) unless its Schur complement given
/// the earlier kept pivots exceeds `tol·refd[p]`.
pub fn sweep_ginv(a: &mut [f64], k: usize, refd: &[f64], tol: f64) {
    for p in 0..k {
        let d = a[p * k + p];
        if !(d > tol * refd[p] && d > 0.0 && d.is_finite()) {
            for j in 0..k {
                a[p * k + j] = 0.0;
                a[j * k + p] = 0.0;
            }
            continue;
        }
        let inv = 1.0 / d;
        for j in 0..k {
            a[p * k + j] *= inv;
        }
        for i in 0..k {
            if i == p {
                continue;
            }
            let c = a[i * k + p];
            if c == 0.0 {
                continue;
            }
            for j in 0..k {
                a[i * k + j] -= c * a[p * k + j];
            }
            a[i * k + p] = -c * inv;
        }
        a[p * k + p] = inv;
    }
}

/// Design row (1?, x…) of a term.
#[inline]
fn design(icpt: bool, x: &[f64], out: &mut [f64]) {
    if icpt {
        out[0] = 1.0;
        out[1..].copy_from_slice(x);
    } else {
        out.copy_from_slice(x);
    }
}

/// One non-eliminated term.
#[derive(Clone, Debug)]
pub struct SBlock {
    /// Session term index.
    pub term: usize,
    /// Levels.
    pub g: usize,
    pub icpt: bool,
    /// Number of slopes.
    pub ns: usize,
    /// Coordinates per level (intercept + slopes).
    pub b: usize,
    /// First design slot of this block within a row of [`SlopeStructure::sc`]/`sv` (slots
    /// s0..s0+b: intercept, then slopes).
    pub s0: usize,
    /// Offset of level 0 in the reduced coordinate vector.
    pub off: usize,
    pub small: bool,
}

/// Level-sorted row lists of the large blocks and the E level of every row (built on first use:
/// escalation, diagonal blocks too large for accumulators, and the n×W scratch path).
pub struct LevelIndex {
    /// Level of E of every row.
    pub row_seg: Vec<u32>,
    /// Per large block: rows sorted by (level, row), and offsets per level (g + 1 entries).
    pub perm: Vec<Vec<u32>>,
    pub perm_off: Vec<Vec<u32>>,
}

pub struct SlopeStructure {
    pub n: usize,
    /// Kept-observation index of each row.
    pub pi: Vec<u32>,
    /// Session index of the eliminated term E, its levels, intercept and slope count, width.
    pub e: usize,
    pub ge: usize,
    pub e_icpt: bool,
    pub e_ns: usize,
    pub be: usize,
    /// Row offsets per level of E (ge + 1 entries).
    pub seg_off: Vec<u32>,
    /// Number of blocks.
    pub nb: usize,
    /// Design slots per row: every block of width b owns b consecutive slots (intercept, then
    /// slopes), small blocks first. Slot j of row i has reduced coordinate `sc[i·k + j]`
    /// (level coordinate off + g·b plus the slot within the level) and value `sv[i·k + j]` (1
    /// for intercepts). The small blocks own the first `nslots_small` slots.
    pub k: usize,
    pub sc: Vec<u32>,
    pub sv: Vec<f64>,
    pub nslots_small: usize,
    /// Slope values of E per row (n × e_ns).
    pub he: Vec<f64>,
    /// Small blocks first, then large ones, each in term order.
    pub blocks: Vec<SBlock>,
    pub nsmall: usize,
    /// Reduced dimension Σ g·b and the small coordinates `0..m_small`.
    pub m: usize,
    pub m_small: usize,
    /// Rows of the longest segment of E.
    pub max_seg: usize,
    /// Fixed partition of the segments into parts of ≳ [`CHUNK`] rows (boundaries).
    pub parts: Vec<u32>,
    /// Largest reduced vector (bytes, one column group) scattered entirely through accumulators.
    pub acc_bytes_max: usize,
    index: OnceLock<LevelIndex>,
}

impl SlopeStructure {
    #[inline]
    fn rows(&self, s: usize) -> Range<usize> {
        self.seg_off[s] as usize..self.seg_off[s + 1] as usize
    }

    #[inline]
    fn he_raw(&self, i: usize) -> &[f64] {
        &self.he[i * self.e_ns..(i + 1) * self.e_ns]
    }

    /// Slope values of block `blk` in row i.
    #[inline]
    fn hq_raw(&self, i: usize, blk: &SBlock) -> &[f64] {
        let base = i * self.k + blk.s0 + usize::from(blk.icpt);
        &self.sv[base..base + blk.ns]
    }

    /// First reduced coordinate (off + g·b) of row i's level in block q.
    #[inline]
    fn coord(&self, i: usize, q: usize) -> usize {
        self.sc[i * self.k + self.blocks[q].s0] as usize
    }

    /// Level of row i in block q.
    #[inline]
    pub fn code(&self, i: usize, q: usize) -> usize {
        let blk = &self.blocks[q];
        (self.coord(i, q) - blk.off) / blk.b
    }

    pub fn has_large(&self) -> bool {
        self.nsmall < self.nb
    }

    /// Groups the parts so that `ntasks × acc_len ≤ 2²²` (segment boundaries).
    fn tasks(&self, acc_len: usize) -> Vec<u32> {
        let np = self.parts.len() - 1;
        let per = if acc_len == 0 { 1 } else { (np * acc_len).div_ceil(1 << 22).max(1) };
        let mut t: Vec<u32> = (0..np).step_by(per).map(|k| self.parts[k]).collect();
        t.push(self.parts[np]);
        t
    }

    /// `k` tasks of nearly equal row counts (segment boundaries).
    fn even_tasks(&self, k: usize) -> Vec<u32> {
        let k = k.clamp(1, self.ge.max(1));
        let mut t = vec![0u32];
        for j in 1..k {
            let target = (self.n * j / k) as u32;
            let s = self.seg_off[..self.ge].partition_point(|&c| c < target) as u32;
            if s > *t.last().expect("nonempty") {
                t.push(s);
            }
        }
        t.push(self.ge as u32);
        t
    }

    /// Whether every block is scattered through per-task accumulators for `r` columns.
    fn accumulates_all(&self, r: usize) -> bool {
        !self.has_large() || self.m * group_width(r) * 8 <= self.acc_bytes_max
    }

    /// Whether the diagonal blocks of large block `q` are accumulated in row order.
    fn diag_in_row_order(&self, q: usize) -> bool {
        let blk = &self.blocks[q];
        blk.g * blk.b * blk.b * 8 <= self.acc_bytes_max
    }

    /// Row boundaries of the parts (times `scale`).
    fn part_rows(&self, scale: usize) -> Vec<usize> {
        self.parts.iter().map(|&s| self.seg_off[s as usize] as usize * scale).collect()
    }

    /// The level index, built on first use.
    pub fn index(&self, pool: &Pool) -> Result<&LevelIndex, RfError> {
        if let Some(ix) = self.index.get() {
            return Ok(ix);
        }
        let ix = build_index(self, pool)?;
        Ok(self.index.get_or_init(|| ix))
    }

    /// The level index; callers have built it with [`SlopeStructure::index`].
    fn built_index(&self) -> &LevelIndex {
        self.index.get().expect("level index built before use")
    }
}

/// Row offsets per level when `codes` is non-decreasing (rows already in level order), else
/// `None`.
fn sorted_offsets(codes: &[u32], g: usize, pool: &Pool) -> Result<Option<Vec<u32>>, RfError> {
    let n = codes.len();
    // Per chunk: sortedness (including the step into the next chunk) and level changes.
    let pieces: Vec<Option<Vec<u32>>> = pool.install(|| {
        (0..n.div_ceil(CHUNK))
            .into_par_iter()
            .map(|c| {
                let lo = c * CHUNK;
                let hi = (lo + CHUNK + 1).min(n);
                let mut starts = Vec::new();
                if lo == 0 {
                    starts.push(0);
                }
                for i in lo.max(1)..hi {
                    if codes[i] < codes[i - 1] {
                        return None;
                    }
                    if codes[i] != codes[i - 1] && i < lo + CHUNK {
                        starts.push(i as u32);
                    }
                }
                Some(starts)
            })
            .collect()
    });
    if pieces.iter().any(Option::is_none) || codes.last().is_some_and(|&c| c as usize >= g) {
        return Ok(None);
    }
    let mut off = alloc_vec(g + 1, n as u32)?;
    let mut next = 0usize;
    for &i in pieces.iter().flatten().flatten() {
        let c = codes[i as usize] as usize;
        while next <= c {
            off[next] = i;
            next += 1;
        }
    }
    Ok(Some(off))
}

/// Dense-factor cost of a small set with `ms` coordinates and `bs` design slots per row:
/// assembly Σ_s min(rows_s·bs, ms)² + n·bs² plus the factorization.
fn small_cost(seg_off: &[u32], ms: usize, bs: usize) -> f64 {
    let n = *seg_off.last().expect("nonempty") as u64;
    let work: u64 = seg_off.windows(2).map(|w| (u64::from(w[1] - w[0]) * bs as u64).min(ms as u64).pow(2)).sum::<u64>() + n * (bs * bs) as u64;
    C_SMALL_ASM * work as f64 + C_FLOP * (ms as f64).powi(3) / 3.0
}

/// Builds the observation-level structure.
pub fn build_structure(terms: &[Term], n: usize, small_limit: usize, pool: &Pool) -> Result<SlopeStructure, RfError> {
    if terms.is_empty() {
        return Err(RfError::Internal("slopes: no absorb terms".into()));
    }
    if terms.iter().any(|t| !t.intercept && t.slopes.is_empty()) {
        return Err(RfError::InvalidInput("absorb term without intercept or slopes".into()));
    }
    if terms.iter().any(|t| t.codes.len() != n || t.slopes.iter().any(|x| x.len() != n)) {
        return Err(RfError::InvalidInput("term length differs from the kept sample".into()));
    }
    let e = fe1_index(terms).unwrap_or_else(|| {
        let mut best = 0;
        for (q, t) in terms.iter().enumerate() {
            if t.g > terms[best].g {
                best = q;
            }
        }
        best
    });
    let te = &terms[e];
    let ge = te.g;
    let (e_icpt, e_ns) = (te.intercept, te.slopes.len());
    let be = usize::from(e_icpt) + e_ns;

    // Rows sorted by (level of E, observation): the identity when the data are already sorted.
    let (seg_off, pi, identity) = match sorted_offsets(&te.codes, ge, pool)? {
        Some(off) => {
            let pi: Vec<u32> = pool.install(|| (0..n as u32).into_par_iter().collect());
            (off, pi, true)
        }
        None => {
            let (seg_off, mut pi) = scatter_by_code(&te.codes, ge, pool)?;
            let parts = obs_groups(&seg_off);
            let row_bounds: Vec<usize> = parts.iter().map(|&s| seg_off[s as usize] as usize).collect();
            let pp = split_bounds(&mut pi, &row_bounds);
            pool.install(|| {
                pp.into_par_iter().enumerate().for_each(|(k, pis)| {
                    let base = row_bounds[k];
                    for s in parts[k] as usize..parts[k + 1] as usize {
                        pis[seg_off[s] as usize - base..seg_off[s + 1] as usize - base].sort_unstable();
                    }
                })
            });
            (seg_off, pi, false)
        }
    };
    let parts = obs_groups(&seg_off);

    // Blocks: small (ascending coordinates while the total stays ≤ small_limit and the dense
    // factor stays cheap, see `small_cost`) first.
    let others: Vec<usize> = (0..terms.len()).filter(|&q| q != e).collect();
    let width = |q: usize| usize::from(terms[q].intercept) + terms[q].slopes.len();
    let slots = be + others.iter().map(|&q| width(q)).sum::<usize>();
    let matvec = C_ROW * n as f64 * (slots + 2) as f64;
    let mut by_size = others.clone();
    by_size.sort_by_key(|&q| (terms[q].g * width(q), q));
    let mut small_set = vec![false; terms.len()];
    let (mut total, mut bs) = (0usize, 0usize);
    for &q in &by_size {
        let c = terms[q].g * width(q);
        if total + c > small_limit || small_cost(&seg_off, total + c, bs + width(q)) > SMALL_FLOOR.max(SMALL_ITERS * matvec) {
            break;
        }
        total += c;
        bs += width(q);
        small_set[q] = true;
    }
    let mut order: Vec<usize> = others.iter().copied().filter(|&q| small_set[q]).collect();
    let nsmall = order.len();
    order.extend(others.iter().copied().filter(|&q| !small_set[q]));
    let mut blocks = Vec::with_capacity(order.len());
    let (mut off, mut s0) = (0usize, 0usize);
    for &q in &order {
        let t = &terms[q];
        let b = width(q);
        blocks.push(SBlock { term: q, g: t.g, icpt: t.intercept, ns: t.slopes.len(), b, s0, off, small: small_set[q] });
        off += t.g * b;
        s0 += b;
    }
    let m = off;
    if m > u32::MAX as usize {
        return Err(RfError::InvalidInput("too many fixed-effect coefficients".into()));
    }
    let m_small: usize = blocks[..nsmall].iter().map(|b| b.g * b.b).sum();
    let nb = blocks.len();
    let k = s0;
    let nslots_small = blocks[..nsmall].iter().map(|b| b.b).sum();

    // Slot coordinates and values, and E's slopes, in row order.
    let mut sc = alloc_vec(n * k, 0u32)?;
    let mut sv = alloc_vec(n * k, 1.0f64)?;
    let mut he = alloc_vec(n * e_ns, 0.0f64)?;
    let obs = |i: usize| if identity { i } else { pi[i] as usize };
    pool.install(|| {
        if k > 0 {
            sc.par_chunks_mut(CHUNK * k).zip(sv.par_chunks_mut(CHUNK * k)).enumerate().for_each(|(c, (scc, svc))| {
                for (j, (sr, vr)) in scc.chunks_exact_mut(k).zip(svc.chunks_exact_mut(k)).enumerate() {
                    let o = obs(c * CHUNK + j);
                    for blk in &blocks {
                        let tb = &terms[blk.term];
                        let c0 = (blk.off + tb.codes[o] as usize * blk.b) as u32;
                        for d in 0..blk.b {
                            sr[blk.s0 + d] = c0 + d as u32;
                        }
                        let first = blk.s0 + usize::from(blk.icpt);
                        for (s, x) in tb.slopes.iter().enumerate() {
                            vr[first + s] = x[o];
                        }
                    }
                }
            });
        }
        if e_ns > 0 {
            he.par_chunks_mut(CHUNK * e_ns).enumerate().for_each(|(c, hc)| {
                for (j, hr) in hc.chunks_exact_mut(e_ns).enumerate() {
                    let o = obs(c * CHUNK + j);
                    for (s, x) in te.slopes.iter().enumerate() {
                        hr[s] = x[o];
                    }
                }
            });
        }
    });

    let max_seg = seg_off.windows(2).map(|w| (w[1] - w[0]) as usize).max().unwrap_or(0);
    Ok(SlopeStructure {
        n,
        pi,
        e,
        ge,
        e_icpt,
        e_ns,
        be,
        seg_off,
        nb,
        k,
        sc,
        sv,
        nslots_small,
        he,
        blocks,
        nsmall,
        m,
        m_small,
        max_seg,
        parts,
        acc_bytes_max: crate::operator::ACC_BYTES_MAX,
        index: OnceLock::new(),
    })
}

/// Builds the level index: the E level of every row and the level permutations of the large
/// blocks (rows sorted by (level, row)).
fn build_index(st: &SlopeStructure, pool: &Pool) -> Result<LevelIndex, RfError> {
    let mut row_seg = alloc_vec(st.n, 0u32)?;
    {
        let rb = st.part_rows(1);
        let rp = split_bounds(&mut row_seg, &rb);
        pool.install(|| {
            rp.into_par_iter().enumerate().for_each(|(k, rs)| {
                let base = rb[k];
                for s in st.parts[k] as usize..st.parts[k + 1] as usize {
                    rs[st.rows(s).start - base..st.rows(s).end - base].iter_mut().for_each(|v| *v = s as u32);
                }
            })
        });
    }
    let (mut perms, mut offs) = (Vec::new(), Vec::new());
    for q in st.nsmall..st.nb {
        let col: Vec<u32> = pool.install(|| (0..st.n).into_par_iter().map(|i| st.code(i, q) as u32).collect());
        let (poff, mut perm) = scatter_by_code(&col, st.blocks[q].g, pool)?;
        let lg = obs_groups(&poff);
        let lb: Vec<usize> = lg.iter().map(|&g| poff[g as usize] as usize).collect();
        let pieces = split_bounds(&mut perm, &lb);
        pool.install(|| {
            pieces.into_par_iter().enumerate().for_each(|(k, p)| {
                let base = lb[k];
                for g in lg[k] as usize..lg[k + 1] as usize {
                    p[poff[g] as usize - base..poff[g + 1] as usize - base].sort_unstable();
                }
            })
        });
        perms.push(perm);
        offs.push(poff);
    }
    Ok(LevelIndex { row_seg, perm: perms, perm_off: offs })
}

/// Weight-dependent quantities (refreshed by `set_weights`).
pub struct SlopeWeights {
    /// Row weights (`None` = all ones).
    pub w_rows: Option<Vec<f64>>,
    /// M_g⁺ of every level of E (ge × be², row-major blocks).
    pub minv: Vec<f64>,
    /// Per block: the diagonal b×b blocks of S per level (g × b²).
    pub dblk: Vec<Vec<f64>>,
    /// Per block: their generalized inverses (block Jacobi).
    pub jinv: Vec<Vec<f64>>,
    /// Redundant coordinates (no information beyond E).
    pub redundant: Vec<bool>,
    pub sw: f64,
}

impl SlopeWeights {
    #[inline]
    fn w(&self, i: usize) -> f64 {
        match &self.w_rows {
            Some(w) => w[i],
            None => 1.0,
        }
    }
}

/// Per-level generalized inverse of M_g (reghdfe convention, see the module docs).
fn level_minv(st: &SlopeStructure, ws_w: &Option<Vec<f64>>, s: usize, out: &mut [f64], sc: &mut Vec<f64>) {
    let (be, ns) = (st.be, st.e_ns);
    let w = |i: usize| ws_w.as_ref().map_or(1.0, |v| v[i]);
    let rows = st.rows(s);
    out.iter_mut().for_each(|v| *v = 0.0);
    sc.clear();
    sc.resize(ns * ns + 2 * ns, 0.0);
    let (g, rest) = sc.split_at_mut(ns * ns);
    let (xbar, refd) = rest.split_at_mut(ns);
    if st.e_icpt {
        let mut sw = 0.0;
        for i in rows.clone() {
            let wi = w(i);
            sw += wi;
            for (k, x) in st.he_raw(i).iter().enumerate() {
                xbar[k] += wi * x;
                refd[k] += wi * x * x;
            }
        }
        xbar.iter_mut().for_each(|v| *v /= sw);
        for i in rows {
            let wi = w(i);
            let x = st.he_raw(i);
            for k in 0..ns {
                let dk = wi * (x[k] - xbar[k]);
                for l in 0..=k {
                    g[k * ns + l] += dk * (x[l] - xbar[l]);
                }
            }
        }
        for k in 0..ns {
            for l in 0..k {
                g[l * ns + k] = g[k * ns + l];
            }
        }
        sweep_ginv(g, ns, refd, LEVEL_TOL);
        // [[1/W + x̄'G x̄, −(G x̄)'], [−G x̄, G]].
        let mut gx = vec![0.0; ns];
        for k in 0..ns {
            gx[k] = (0..ns).map(|l| g[k * ns + l] * xbar[l]).sum();
        }
        out[0] = 1.0 / sw + (0..ns).map(|k| xbar[k] * gx[k]).sum::<f64>();
        for k in 0..ns {
            out[1 + k] = -gx[k];
            out[(1 + k) * be] = -gx[k];
            for l in 0..ns {
                out[(1 + k) * be + 1 + l] = g[k * ns + l];
            }
        }
    } else {
        for i in rows {
            let wi = w(i);
            let x = st.he_raw(i);
            for k in 0..ns {
                for l in 0..=k {
                    g[k * ns + l] += wi * x[k] * x[l];
                }
            }
        }
        for k in 0..ns {
            refd[k] = g[k * ns + k];
            for l in 0..k {
                g[l * ns + k] = g[k * ns + l];
            }
        }
        sweep_ginv(g, ns, refd, LEVEL_TOL);
        out.copy_from_slice(g);
    }
}

/// Computes the weight state for row weights `w_rows` (row order).
pub fn slope_weights(st: &SlopeStructure, w_rows: Option<Vec<f64>>, pool: &Pool) -> Result<SlopeWeights, RfError> {
    let be2 = st.be * st.be;
    let sw = match &w_rows {
        Some(w) => sum_chunks(pool, st.n, CHUNK, 1, |r, acc| acc[0] += w[r].iter().sum::<f64>())[0],
        None => st.n as f64,
    };
    let mut minv = alloc_vec(st.ge * be2, 0.0)?;
    if st.e_icpt && st.be == 1 {
        // Intercept-only E: M_s⁺ = 1/Σw.
        pool.install(|| {
            minv.par_chunks_mut(4096).enumerate().for_each(|(c, out)| {
                for (k, o) in out.iter_mut().enumerate() {
                    let rows = st.rows(c * 4096 + k);
                    let w = match &w_rows {
                        Some(w) => w[rows].iter().sum::<f64>(),
                        None => rows.len() as f64,
                    };
                    *o = 1.0 / w;
                }
            })
        });
    } else {
        pool.install(|| {
            minv.par_chunks_mut(be2).enumerate().with_min_len(256).for_each_init(Vec::new, |sc, (s, out)| level_minv(st, &w_rows, s, out, sc))
        });
    }
    // Diagonal blocks / block-Jacobi inverses of the large blocks. Small blocks get empty
    // entries; their redundancy flags are set from the assembled dense block (`ensure_basic`).
    let mut dblk = Vec::with_capacity(st.nb);
    let mut jinv = Vec::with_capacity(st.nb);
    let mut redundant = alloc_vec(st.m, false)?;
    for (q, blk) in st.blocks.iter().enumerate() {
        if blk.small {
            dblk.push(Vec::new());
            jinv.push(Vec::new());
            continue;
        }
        let b = blk.b;
        let b2 = b * b;
        let (gram, mut d) = if st.diag_in_row_order(q) { diag_rows(st, &w_rows, &minv, q, pool)? } else { diag_levels(st, &w_rows, &minv, q, pool)? };
        let mut j = alloc_vec(blk.g * b2, 0.0)?;
        let red = &mut redundant[blk.off..blk.off + blk.g * b];
        pool.install(|| {
            d.par_chunks_mut(b2).zip(j.par_chunks_mut(b2)).zip(red.par_chunks_mut(b)).zip(gram.par_chunks(b2)).with_min_len(64).for_each_init(
                || vec![0.0; b],
                |refd, (((dl, jl), rl), gl)| {
                    // On input dl = Σ_s C' M_s⁺ C; the diagonal block is gram − dl.
                    for (dv, gv) in dl.iter_mut().zip(gl) {
                        *dv = gv - *dv;
                    }
                    for a in 0..b {
                        rl[a] = !(dl[a * b + a] > REDUNDANT * gl[a * b + a]);
                    }
                    jl.copy_from_slice(dl);
                    for a in 0..b {
                        refd[a] = dl[a * b + a];
                        if rl[a] {
                            for bb in 0..b {
                                jl[a * b + bb] = 0.0;
                                jl[bb * b + a] = 0.0;
                            }
                        }
                    }
                    sweep_ginv(jl, b, refd, JACOBI_TOL);
                },
            )
        });
        dblk.push(d);
        jinv.push(j);
    }
    Ok(SlopeWeights { w_rows, minv, dblk, jinv, redundant, sw })
}

/// cmc += C' M⁺ C for C (be × b) and M⁺ (be × be).
#[inline]
fn add_cmc(c: &[f64], mv: &[f64], be: usize, b: usize, cmc: &mut [f64]) {
    for a in 0..b {
        for bb in 0..b {
            let mut acc = 0.0;
            for l in 0..be {
                let mut t = 0.0;
                for l2 in 0..be {
                    t += mv[l * be + l2] * c[l2 * b + bb];
                }
                acc += c[l * b + a] * t;
            }
            cmc[a * b + bb] += acc;
        }
    }
}

/// Per level of large block `q`: (Σ w h h', Σ_s C_s' M_s⁺ C_s) (g × b² each), accumulated in row
/// order through per-task accumulators combined in task order. The task count depends on n and
/// the accumulator size only (≥ 4 accumulator lengths of rows per task, ≤ 64 MiB in total).
fn diag_rows(st: &SlopeStructure, w_rows: &Option<Vec<f64>>, minv: &[f64], q: usize, pool: &Pool) -> Result<(Vec<f64>, Vec<f64>), RfError> {
    let blk = &st.blocks[q];
    let (be, b) = (st.be, blk.b);
    let (b2, be2) = (b * b, be * be);
    let len = blk.g * b2;
    let nt_want = (st.n / (4 * len).max(1)).min((1 << 23) / (2 * len).max(1)).clamp(1, 64);
    let tb = st.even_tasks(nt_want);
    // Intercept-only E: h_E = 1, so C = Σ w h_q.
    let simple_e = st.e_icpt && be == 1;
    let accs: Vec<Vec<f64>> = pool.install(|| {
        (0..tb.len() - 1)
            .into_par_iter()
            .map(|t| {
                let mut acc = vec![0.0; 2 * len];
                let segs = tb[t] as usize..tb[t + 1] as usize;
                if simple_e && (1..=4).contains(&b) {
                    let wr = w_rows.as_deref();
                    match b {
                        1 => diag_task_simple::<1>(st, blk, wr, minv, segs, &mut acc),
                        2 => diag_task_simple::<2>(st, blk, wr, minv, segs, &mut acc),
                        3 => diag_task_simple::<3>(st, blk, wr, minv, segs, &mut acc),
                        _ => diag_task_simple::<4>(st, blk, wr, minv, segs, &mut acc),
                    }
                    return acc;
                }
                let (gram, cmc) = acc.split_at_mut(len);
                let mut items: Vec<(u32, u32)> = Vec::new();
                let (mut c, mut he) = (vec![0.0; be * b], vec![0.0; be]);
                for s in segs {
                    items.clear();
                    items.extend(st.rows(s).map(|i| (st.sc[i * st.k + blk.s0], i as u32)));
                    items.sort_unstable();
                    let mv = &minv[s * be2..(s + 1) * be2];
                    let mut k = 0;
                    while k < items.len() {
                        let c0 = items[k].0;
                        // g·b² = (c0 − off)·b.
                        let gb2 = (c0 as usize - blk.off) * b;
                        c.iter_mut().for_each(|v| *v = 0.0);
                        let gl = &mut gram[gb2..gb2 + b2];
                        while k < items.len() && items[k].0 == c0 {
                            let i = items[k].1 as usize;
                            let wi = w_rows.as_ref().map_or(1.0, |v| v[i]);
                            // The block's slot values are its design row (1?, x…).
                            let hq = &st.sv[i * st.k + blk.s0..i * st.k + blk.s0 + b];
                            for a in 0..b {
                                let wa = wi * hq[a];
                                for bb in 0..b {
                                    gl[a * b + bb] += wa * hq[bb];
                                }
                            }
                            if simple_e {
                                for a in 0..b {
                                    c[a] += wi * hq[a];
                                }
                            } else {
                                design(st.e_icpt, st.he_raw(i), &mut he);
                                for l in 0..be {
                                    for a in 0..b {
                                        c[l * b + a] += wi * he[l] * hq[a];
                                    }
                                }
                            }
                            k += 1;
                        }
                        add_cmc(&c, mv, be, b, &mut cmc[gb2..gb2 + b2]);
                    }
                }
                acc
            })
            .collect()
    });
    let mut out = alloc_vec(2 * len, 0.0)?;
    pool.install(|| {
        out.par_chunks_mut(4096).enumerate().for_each(|(ci, o)| {
            let (base, len) = (ci * 4096, o.len());
            for acc in &accs {
                for (v, a) in o.iter_mut().zip(&acc[base..base + len]) {
                    *v += a;
                }
            }
        })
    });
    let cmc = out.split_off(len);
    Ok((out, cmc))
}

/// [`diag_rows`] task for an intercept-only E (M_s⁺ = 1/W_s) and a block of width `B`: the same
/// sums as the general loop, with the per-level vectors in registers. Rows of a level are
/// grouped by a quadratic scan in segments of ≤ 64 rows (no sort), by sorting otherwise; either
/// way a level's rows are summed in row order.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn diag_task_simple<const B: usize>(st: &SlopeStructure, blk: &SBlock, wr: Option<&[f64]>, minv: &[f64], segs: Range<usize>, acc: &mut [f64]) {
    let len = blk.g * B * B;
    let (gram, cmc) = acc.split_at_mut(len);
    let (k, s0, off) = (st.k, blk.s0, blk.off);
    let sv = &st.sv[..];
    let mut items: Vec<(u32, u32)> = Vec::with_capacity(st.max_seg);
    // One level of a segment: gram += Σ w h h', cmc += C C'/W_s with C = Σ w h.
    let mut level = |c0: u32, rows: &[usize], mv: f64| {
        let gb2 = (c0 as usize - off) * B;
        let mut c = [0.0; B];
        let gl: &mut [f64] = &mut gram[gb2..gb2 + B * B];
        for &i in rows {
            let wi = wr.map_or(1.0, |w| w[i]);
            let hq: &[f64; B] = sv[i * k + s0..i * k + s0 + B].try_into().expect("B slots");
            for a in 0..B {
                let wa = wi * hq[a];
                for bb in 0..B {
                    gl[a * B + bb] += wa * hq[bb];
                }
                c[a] += wa;
            }
        }
        // C' M_s⁺ C as `add_cmc` with be = 1.
        let cl = &mut cmc[gb2..gb2 + B * B];
        for a in 0..B {
            for bb in 0..B {
                cl[a * B + bb] += c[a] * (mv * c[bb]);
            }
        }
    };
    let mut cc = [0u32; 64];
    let mut grp = [0usize; 64];
    let mut grows: Vec<usize> = Vec::with_capacity(st.max_seg);
    for s in segs {
        let rows = st.rows(s);
        let mv = minv[s];
        if rows.len() <= 64 {
            let n = rows.len();
            for (kk, i) in rows.clone().enumerate() {
                cc[kk] = st.sc[i * k + s0];
            }
            let mut done = 0u64;
            for a0 in 0..n {
                if done >> a0 & 1 == 1 {
                    continue;
                }
                let c0 = cc[a0];
                let mut m = 0;
                for a1 in a0..n {
                    if cc[a1] == c0 {
                        done |= 1 << a1;
                        grp[m] = rows.start + a1;
                        m += 1;
                    }
                }
                level(c0, &grp[..m], mv);
            }
            continue;
        }
        items.clear();
        items.extend(rows.map(|i| (st.sc[i * k + s0], i as u32)));
        items.sort_unstable();
        let mut j = 0;
        while j < items.len() {
            let c0 = items[j].0;
            grows.clear();
            grows.extend(items[j..].iter().take_while(|it| it.0 == c0).map(|it| it.1 as usize));
            j += grows.len();
            level(c0, &grows, mv);
        }
    }
}

/// [`diag_rows`] by level pulls through the level index (blocks too large for accumulators).
fn diag_levels(st: &SlopeStructure, w_rows: &Option<Vec<f64>>, minv: &[f64], q: usize, pool: &Pool) -> Result<(Vec<f64>, Vec<f64>), RfError> {
    let ix = st.index(pool)?;
    let blk = &st.blocks[q];
    let (be, b) = (st.be, blk.b);
    let (b2, be2) = (b * b, be * be);
    let (perm, poff) = (&ix.perm[q - st.nsmall], &ix.perm_off[q - st.nsmall]);
    let mut gram = alloc_vec(blk.g * b2, 0.0)?;
    let mut cmc = alloc_vec(blk.g * b2, 0.0)?;
    pool.install(|| {
        gram.par_chunks_mut(b2).zip(cmc.par_chunks_mut(b2)).enumerate().with_min_len(64).for_each_init(
            || (vec![0.0; be * b], vec![0.0; be], vec![0.0; b]),
            |(c, he, hq), (g, (gl, dl))| {
                let rows = &perm[poff[g] as usize..poff[g + 1] as usize];
                let mut k = 0;
                while k < rows.len() {
                    let s = ix.row_seg[rows[k] as usize] as usize;
                    c.iter_mut().for_each(|v| *v = 0.0);
                    while k < rows.len() && ix.row_seg[rows[k] as usize] as usize == s {
                        let i = rows[k] as usize;
                        let wi = w_rows.as_ref().map_or(1.0, |v| v[i]);
                        design(st.e_icpt, st.he_raw(i), he);
                        design(blk.icpt, st.hq_raw(i, blk), hq);
                        for a in 0..b {
                            for bb in 0..b {
                                gl[a * b + bb] += wi * hq[a] * hq[bb];
                            }
                        }
                        for l in 0..be {
                            for a in 0..b {
                                c[l * b + a] += wi * he[l] * hq[a];
                            }
                        }
                        k += 1;
                    }
                    add_cmc(c, &minv[s * be2..(s + 1) * be2], be, b, dl);
                }
            },
        )
    });
    Ok((gram, cmc))
}

/// Per-segment aggregation of one block: for segment s, entries `off[s]..off[s+1]` hold the
/// distinct levels (ascending), C = Σ w h_E h_q' (be × b) and M_s⁺C.
struct BlockAgg {
    off: Vec<u32>,
    codes: Vec<u32>,
    c: Vec<f64>,
    mc: Vec<f64>,
    /// Σ_s d_s².
    work: u64,
}

/// Kernels bound to a structure and a weight state.
#[derive(Clone, Copy)]
pub struct SOps<'a> {
    pub st: &'a SlopeStructure,
    pub ws: &'a SlopeWeights,
}

/// Reusable buffers of the scatter passes: the per-task accumulators and the n × W scratch of
/// the level pulls (kept across the PCG iterations so no pass allocates or page-faults them).
#[derive(Default)]
pub struct PassScratch {
    acc: Vec<f64>,
    zbuf: Vec<f64>,
}

/// Grows `v` to at least `len` entries (fallible; contents unspecified).
fn grow(v: &mut Vec<f64>, len: usize) -> Result<(), RfError> {
    if v.len() < len {
        v.try_reserve_exact(len - v.len()).map_err(|_| RfError::OutOfMemory)?;
        v.resize(len, 0.0);
    }
    Ok(())
}

/// Per-task scratch of the segment kernels: the segment rows `d` (sized for the longest
/// segment, so the row loops contain no allocation call and keep their accumulators in
/// registers) and the E projection accumulators `a`, `t` (be × W).
struct Seg<const W: usize> {
    d: Vec<[f64; W]>,
    a: Vec<[f64; W]>,
    t: Vec<[f64; W]>,
}

impl<const W: usize> Seg<W> {
    fn new(st: &SlopeStructure) -> Self {
        Seg { d: vec![[0.0; W]; st.max_seg], a: vec![[0.0; W]; st.be], t: vec![[0.0; W]; st.be] }
    }
}

/// Read-only inputs of one scatter-pass task.
#[derive(Clone, Copy)]
struct TaskIn<'a> {
    src: &'a [f64],
    r: usize,
    j0: usize,
    sc: &'a [u32],
    sv: &'a [f64],
    seg_off: &'a [u32],
    minv: &'a [f64],
    wr: Option<&'a [f64]>,
    /// Slots scattered through the accumulators (all, or those of the small blocks).
    nk_acc: usize,
}

/// One task of [`SOps::scatter_pass`] over the segments `segs`: returns the per-column
/// [Σwv², Σwv, Σw·e²] when `STATS`. A separate non-inlined function with the accumulator as its
/// own `&mut` argument, so the compiler knows the accumulator stores never touch the slot
/// arrays or the inputs (they stay in registers instead of being reloaded after every store).
#[inline(never)]
fn seg_task<const W: usize, const K: usize, const APPLY: bool, const STATS: bool>(
    ops: SOps,
    segs: Range<usize>,
    ins: &TaskIn,
    acc: &mut [f64],
    mut zp: Option<&mut [f64]>,
) -> [[f64; W]; 3] {
    let TaskIn { src, r, j0, sc, sv, seg_off, minv, wr, nk_acc } = *ins;
    let st = ops.st;
    let k = st.k;
    let nk = nslots::<K>(st);
    let fused = st.e_icpt && st.be == 1;
    let row0 = seg_off[segs.start] as usize;
    let mut ex = [[0.0; W]; 3];
    let mut sg = Seg::<W>::new(st);
    let Seg { d: dbuf, a, t } = &mut sg;
    let load = |i: usize| -> [f64; W] {
        if APPLY {
            let base = i * k;
            let mut u = [0.0; W];
            for (&c, &x) in sc[base..base + nk].iter().zip(&sv[base..base + nk]) {
                let pr = at::<W>(src, c as usize * r + j0);
                for j in 0..W {
                    u[j] += x * pr[j];
                }
            }
            u
        } else {
            *at::<W>(src, i * r + j0)
        }
    };
    let scatter = |i: usize, z: &[f64; W], acc: &mut [f64]| {
        let base = i * k;
        let mut put = |cnt: usize| {
            for (&c, &x) in sc[base..base + cnt].iter().zip(&sv[base..base + cnt]) {
                let a = at_mut::<W>(acc, c as usize * W);
                for j in 0..W {
                    a[j] += x * z[j];
                }
            }
        };
        // A compile-time slot count when every slot is accumulated.
        if nk_acc == nk { put(nk) } else { put(nk_acc) }
    };
    for s in segs {
        let rows = seg_off[s] as usize..seg_off[s + 1] as usize;
        let d = &mut dbuf[..rows.len()];
        if fused {
            let mut m = [0.0; W];
            for (kk, i) in rows.clone().enumerate() {
                let v = load(i);
                let w = wr.map_or(1.0, |w| w[i]);
                for j in 0..W {
                    m[j] += w * v[j];
                    if STATS {
                        ex[0][j] += w * v[j] * v[j];
                        ex[1][j] += w * v[j];
                    }
                }
                d[kk] = v;
            }
            let inv = minv[s];
            for v in m.iter_mut() {
                *v *= inv;
            }
            for (kk, i) in rows.enumerate() {
                let w = wr.map_or(1.0, |w| w[i]);
                let mut z = d[kk];
                for j in 0..W {
                    let e = z[j] - m[j];
                    if STATS {
                        ex[2][j] += w * e * e;
                    }
                    z[j] = w * e;
                }
                scatter(i, &z, acc);
                if let Some(zp) = zp.as_deref_mut() {
                    *at_mut::<W>(zp, (i - row0) * W) = z;
                }
            }
            continue;
        }
        for (kk, i) in rows.clone().enumerate() {
            let v = load(i);
            if STATS {
                let w = wr.map_or(1.0, |w| w[i]);
                for j in 0..W {
                    ex[0][j] += w * v[j] * v[j];
                    ex[1][j] += w * v[j];
                }
            }
            d[kk] = v;
        }
        ops.seg_project::<W>(s, rows.clone(), d, a, t);
        for (kk, i) in rows.enumerate() {
            let w = wr.map_or(1.0, |w| w[i]);
            let mut z = d[kk];
            for j in 0..W {
                if STATS {
                    ex[2][j] += w * z[j] * z[j];
                }
                z[j] *= w;
            }
            scatter(i, &z, acc);
            if let Some(zp) = zp.as_deref_mut() {
                *at_mut::<W>(zp, (i - row0) * W) = z;
            }
        }
    }
    ex
}

/// One task of [`SOps::residualize_rows`] over the segments `segs` (rows of `vs`, columns
/// j0..j0+W of r): v_i ← d_i − h_{E,i}'t with d = v − u(x); writes α_E into `alpha` (segment
/// rows of be × r) when given. Not inlined, for the same aliasing reason as [`seg_task`].
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn resid_task<const W: usize, const K: usize>(ops: SOps, segs: Range<usize>, vs: &mut [f64], r: usize, j0: usize, x: &[f64], mut alpha: Option<&mut [f64]>) {
    let st = ops.st;
    let (sc, sv, seg_off, minv) = (&st.sc[..], &st.sv[..], &st.seg_off[..], &ops.ws.minv[..]);
    let (k, be) = (st.k, st.be);
    let nk = nslots::<K>(st);
    let wr = ops.ws.w_rows.as_deref();
    let fused = st.e_icpt && be == 1;
    let (s0, row0) = (segs.start, seg_off[segs.start] as usize);
    let load = |vs: &[f64], i: usize| -> [f64; W] {
        let mut d = *at::<W>(vs, (i - row0) * r + j0);
        let base = i * k;
        let mut u = [0.0; W];
        for (&c, &h) in sc[base..base + nk].iter().zip(&sv[base..base + nk]) {
            let pr = at::<W>(x, c as usize * r + j0);
            for j in 0..W {
                u[j] += h * pr[j];
            }
        }
        for j in 0..W {
            d[j] -= u[j];
        }
        d
    };
    let mut sg = Seg::<W>::new(st);
    let Seg { d: dbuf, a: pa, t } = &mut sg;
    for s in segs {
        let rows = seg_off[s] as usize..seg_off[s + 1] as usize;
        let d = &mut dbuf[..rows.len()];
        if fused {
            // Intercept-only E: t = Σ w d / Σ w.
            let mut m = [0.0; W];
            for (kk, i) in rows.clone().enumerate() {
                let dv = load(vs, i);
                let w = wr.map_or(1.0, |w| w[i]);
                for j in 0..W {
                    m[j] += w * dv[j];
                }
                d[kk] = dv;
            }
            let inv = minv[s];
            for v in m.iter_mut() {
                *v *= inv;
            }
            for (kk, i) in rows.enumerate() {
                let out = at_mut::<W>(vs, (i - row0) * r + j0);
                for j in 0..W {
                    out[j] = d[kk][j] - m[j];
                }
            }
            t[0] = m;
        } else {
            for (kk, i) in rows.clone().enumerate() {
                d[kk] = load(vs, i);
            }
            ops.seg_project::<W>(s, rows.clone(), d, pa, t);
            for (kk, i) in rows.enumerate() {
                *at_mut::<W>(vs, (i - row0) * r + j0) = d[kk];
            }
        }
        if let Some(a) = alpha.as_deref_mut() {
            for (l, tl) in t.iter().enumerate() {
                let at0 = ((s - s0) * be + l) * r + j0;
                a[at0..at0 + W].copy_from_slice(tl);
            }
        }
    }
}

/// Calls `$s.$f::<W, K>(args)` with the column-group width W = `$w` ∈ 1..=[`MAX_W`] as a
/// compile-time constant (the per-row loops unroll into registers).
macro_rules! by_width {
    ($w:expr, $k:tt, $s:ident . $f:ident ( $($a:expr),* )) => {
        match $w {
            1 => $s.$f::<1, $k>($($a),*),
            2 => $s.$f::<2, $k>($($a),*),
            3 => $s.$f::<3, $k>($($a),*),
            4 => $s.$f::<4, $k>($($a),*),
            5 => $s.$f::<5, $k>($($a),*),
            6 => $s.$f::<6, $k>($($a),*),
            7 => $s.$f::<7, $k>($($a),*),
            8 => $s.$f::<8, $k>($($a),*),
            9 => $s.$f::<9, $k>($($a),*),
            10 => $s.$f::<10, $k>($($a),*),
            11 => $s.$f::<11, $k>($($a),*),
            12 => $s.$f::<12, $k>($($a),*),
            w => unreachable!("column group of width {w}"),
        }
    };
}

/// [`by_width`] with K = `$k` design slots as a compile-time constant for 1 ≤ K ≤ 4 (K = 0:
/// the slot count is read at run time).
macro_rules! by_shape {
    ($w:expr, $k:expr, $s:ident . $f:ident ( $($a:expr),* )) => {
        match $k {
            1 => by_width!($w, 1, $s.$f($($a),*)),
            2 => by_width!($w, 2, $s.$f($($a),*)),
            3 => by_width!($w, 3, $s.$f($($a),*)),
            4 => by_width!($w, 4, $s.$f($($a),*)),
            _ => by_width!($w, 0, $s.$f($($a),*)),
        }
    };
}

/// The slot count of a kernel instance for `K` (0 = all slots, read at run time).
#[inline(always)]
fn nslots<const K: usize>(st: &SlopeStructure) -> usize {
    if K == 0 { st.k } else { K }
}

impl SOps<'_> {
    #[inline(always)]
    fn w_at(&self, i: usize) -> f64 {
        match &self.ws.w_rows {
            Some(w) => w[i],
            None => 1.0,
        }
    }

    /// u = Σ_q h_{q,i}'P[q, g_q(i)] for the columns j0..j0+W of the m × r block `p`.
    #[inline(always)]
    fn u_row<const W: usize, const K: usize>(&self, i: usize, p: &[f64], r: usize, j0: usize) -> [f64; W] {
        let st = self.st;
        let nk = nslots::<K>(st);
        let (cs, vs) = (&st.sc[i * st.k..i * st.k + nk], &st.sv[i * st.k..i * st.k + nk]);
        let mut u = [0.0; W];
        for (&c, &x) in cs.iter().zip(vs) {
            let pr = at::<W>(p, c as usize * r + j0);
            for j in 0..W {
                u[j] += x * pr[j];
            }
        }
        u
    }

    /// Projects E out of the rows `d` of segment s: t = M_s⁺ Σ_i w_i h_{E,i} d_i' (be × W, in
    /// `t`) and d_i −= h_{E,i}'t. `a` is scratch (be × W).
    #[inline(always)]
    fn seg_project<const W: usize>(&self, s: usize, rows: Range<usize>, d: &mut [[f64; W]], a: &mut [[f64; W]], t: &mut [[f64; W]]) {
        let (st, ws) = (self.st, self.ws);
        let be = st.be;
        if st.e_icpt && be == 1 {
            // Intercept-only E: t = Σ w d / Σ w, d_i −= t.
            let mut m = [0.0; W];
            match &ws.w_rows {
                None => {
                    for dr in d.iter() {
                        for j in 0..W {
                            m[j] += dr[j];
                        }
                    }
                }
                Some(w) => {
                    for (dr, &wi) in d.iter().zip(&w[rows]) {
                        for j in 0..W {
                            m[j] += wi * dr[j];
                        }
                    }
                }
            }
            let inv = ws.minv[s];
            for v in m.iter_mut() {
                *v *= inv;
            }
            for dr in d.iter_mut() {
                for j in 0..W {
                    dr[j] -= m[j];
                }
            }
            t[0] = m;
            return;
        }
        a.iter_mut().for_each(|v| *v = [0.0; W]);
        let l0 = usize::from(st.e_icpt);
        for (dr, i) in d.iter().zip(rows.clone()) {
            let w = self.w_at(i);
            if st.e_icpt {
                for j in 0..W {
                    a[0][j] += w * dr[j];
                }
            }
            for (k, &x) in st.he_raw(i).iter().enumerate() {
                let c = w * x;
                for j in 0..W {
                    a[l0 + k][j] += c * dr[j];
                }
            }
        }
        let mv = &ws.minv[s * be * be..(s + 1) * be * be];
        for l in 0..be {
            let mut tr = [0.0; W];
            for l2 in 0..be {
                let c = mv[l * be + l2];
                if c != 0.0 {
                    for j in 0..W {
                        tr[j] += c * a[l2][j];
                    }
                }
            }
            t[l] = tr;
        }
        for (dr, i) in d.iter_mut().zip(rows) {
            if st.e_icpt {
                for j in 0..W {
                    dr[j] -= t[0][j];
                }
            }
            for (k, &x) in st.he_raw(i).iter().enumerate() {
                for j in 0..W {
                    dr[j] -= x * t[l0 + k][j];
                }
            }
        }
    }

    /// Fused pass over the columns j0..j0+W: per segment, row i is loaded (APPLY: u_i from the
    /// m × r block P = `src`; else row i of the n × r block `src`), E is projected out and
    /// z_i = w_i d_i is scattered into those columns of `y` (m × r; fully overwritten there).
    /// With `STATS`, returns [Σwv² (W), Σwv (W), Σw·e² (W)] of the loaded values v and the
    /// E-residuals e.
    #[allow(clippy::too_many_arguments)]
    fn scatter_pass<const W: usize, const K: usize, const APPLY: bool, const STATS: bool>(
        &self,
        r: usize,
        j0: usize,
        src: &[f64],
        y: &mut [f64],
        sc: &mut PassScratch,
        pool: &Pool,
    ) -> Result<Vec<f64>, RfError> {
        let st = self.st;
        let all = st.accumulates_all(r);
        let macc = if all { st.m } else { st.m_small };
        let nk_acc = if all { st.k } else { st.nslots_small };
        let acc_len = macc * W;
        let tb = st.tasks(acc_len);
        let nt = tb.len() - 1;
        grow(&mut sc.acc, nt * acc_len)?;
        let accs: Vec<&mut [f64]> = if acc_len == 0 { (0..nt).map(|_| &mut [][..]).collect() } else { sc.acc[..nt * acc_len].chunks_mut(acc_len).collect() };
        let zp: Vec<Option<&mut [f64]>> = if all {
            (0..nt).map(|_| None).collect()
        } else {
            grow(&mut sc.zbuf, st.n * W)?;
            let rb: Vec<usize> = tb.iter().map(|&s| st.seg_off[s as usize] as usize * W).collect();
            split_bounds(&mut sc.zbuf[..st.n * W], &rb).into_iter().map(Some).collect()
        };
        let ws = self.ws;
        let extras: Vec<[[f64; W]; 3]> = pool.install(|| {
            accs.into_par_iter()
                .zip(zp)
                .enumerate()
                .map(|(t, (acc, zp))| {
                    acc.iter_mut().for_each(|v| *v = 0.0);
                    let segs = tb[t] as usize..tb[t + 1] as usize;
                    let ins = TaskIn { src, r, j0, sc: &st.sc, sv: &st.sv, seg_off: &st.seg_off, minv: &ws.minv, wr: ws.w_rows.as_deref(), nk_acc };
                    seg_task::<W, K, APPLY, STATS>(*self, segs, &ins, acc, zp)
                })
                .collect()
        });
        // Task-order combine into the columns j0..j0+W of the accumulated coordinates.
        const CB: usize = 1024;
        let accs = &sc.acc[..nt * acc_len];
        pool.install(|| {
            y[..macc * r].par_chunks_mut(CB * r).enumerate().for_each(|(ci, out)| {
                let len = out.len() / r * W;
                let base = ci * CB * W;
                let mut sum = vec![0.0; len];
                for t in 0..nt {
                    for (o, v) in sum.iter_mut().zip(&accs[t * acc_len + base..t * acc_len + base + len]) {
                        *o += v;
                    }
                }
                for (orow, srow) in out.chunks_exact_mut(r).zip(sum.chunks_exact(W)) {
                    orow[j0..j0 + W].copy_from_slice(srow);
                }
            })
        });
        if !all {
            self.pull_large::<W>(&sc.zbuf, r, j0, y, pool);
        }
        let mut extra = vec![0.0; if STATS { 3 * W } else { 0 }];
        if STATS {
            for ex in &extras {
                for (k, e) in ex.iter().enumerate() {
                    for j in 0..W {
                        extra[k * W + j] += e[j];
                    }
                }
            }
        }
        Ok(extra)
    }

    /// Y[q, g] = Σ_{i ∈ rows of g} h_{q,i} z_i (columns j0..j0+W) for every large block, by
    /// level pulls from the n × W scratch.
    fn pull_large<const W: usize>(&self, zbuf: &[f64], r: usize, j0: usize, y: &mut [f64], pool: &Pool) {
        let st = self.st;
        let ix = st.built_index();
        for (k, blk) in st.blocks[st.nsmall..].iter().enumerate() {
            let (perm, poff) = (&ix.perm[k], &ix.perm_off[k]);
            let ys = &mut y[blk.off * r..(blk.off + blk.g * blk.b) * r];
            pool.install(|| {
                ys.par_chunks_mut(blk.b * r).enumerate().with_min_len(64).for_each(|(g, out)| {
                    let rows = &perm[poff[g] as usize..poff[g + 1] as usize];
                    let mut slot = 0;
                    if blk.icpt {
                        let mut sm = [0.0; W];
                        for &i in rows {
                            let z = at::<W>(zbuf, i as usize * W);
                            for j in 0..W {
                                sm[j] += z[j];
                            }
                        }
                        out[j0..j0 + W].copy_from_slice(&sm);
                        slot = 1;
                    }
                    for sl in 0..blk.ns {
                        let mut sm = [0.0; W];
                        for &i in rows {
                            let i = i as usize;
                            let x = st.sv[i * st.k + blk.s0 + slot + sl];
                            let z = at::<W>(zbuf, i * W);
                            for j in 0..W {
                                sm[j] += x * z[j];
                            }
                        }
                        out[(slot + sl) * r + j0..(slot + sl) * r + j0 + W].copy_from_slice(&sm);
                    }
                })
            });
        }
    }

    /// Y = S·P (m × r row-major).
    pub fn apply(&self, p: &[f64], r: usize, y: &mut [f64], sc: &mut PassScratch, pool: &Pool) -> Result<(), RfError> {
        if !self.st.accumulates_all(r) {
            self.st.index(pool)?;
        }
        for g in col_groups(r) {
            by_shape!(g.len(), self.st.k, self.apply_n(p, r, g.start, y, sc, pool))?;
        }
        Ok(())
    }

    fn apply_n<const W: usize, const K: usize>(&self, p: &[f64], r: usize, j0: usize, y: &mut [f64], sc: &mut PassScratch, pool: &Pool) -> Result<Vec<f64>, RfError> {
        self.scatter_pass::<W, K, true, false>(r, j0, p, y, sc, pool)
    }

    /// Reduced right-hand side of the row block `v` (n × r): (rhs, [Σwv², Σwv, Σw e²] per column)
    /// with e the residual after E alone.
    pub fn rhs(&self, v: &[f64], r: usize, sc: &mut PassScratch, pool: &Pool) -> Result<(Vec<f64>, Vec<f64>), RfError> {
        if !self.st.accumulates_all(r) {
            self.st.index(pool)?;
        }
        let mut y = alloc_vec(self.st.m * r, 0.0)?;
        let mut ex = vec![0.0; 3 * r];
        for g in col_groups(r) {
            let w = g.len();
            let e = by_shape!(w, self.st.k, self.rhs_n(v, r, g.start, &mut y, sc, pool))?;
            for k in 0..3 {
                ex[k * r + g.start..k * r + g.end].copy_from_slice(&e[k * w..(k + 1) * w]);
            }
        }
        Ok((y, ex))
    }

    fn rhs_n<const W: usize, const K: usize>(&self, v: &[f64], r: usize, j0: usize, y: &mut [f64], sc: &mut PassScratch, pool: &Pool) -> Result<Vec<f64>, RfError> {
        self.scatter_pass::<W, K, false, true>(r, j0, v, y, sc, pool)
    }

    /// In place: v_i ← d_i − h_{E,i}'t with d = v − u(x); writes α_E (ge·be × r) when given.
    pub fn residualize_rows(&self, v: &mut [f64], r: usize, x: &[f64], mut alpha: Option<&mut [f64]>, pool: &Pool) {
        for g in col_groups(r) {
            by_shape!(g.len(), self.st.k, self.residualize_rows_n(v, r, g.start, x, alpha.as_deref_mut(), pool));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn residualize_rows_n<const W: usize, const K: usize>(&self, v: &mut [f64], r: usize, j0: usize, x: &[f64], alpha: Option<&mut [f64]>, pool: &Pool) {
        let st = self.st;
        let be = st.be;
        let rb = st.part_rows(r);
        let vp = split_bounds(v, &rb);
        let ap: Vec<Option<&mut [f64]>> = match alpha {
            Some(a) => {
                let ab: Vec<usize> = st.parts.iter().map(|&s| s as usize * be * r).collect();
                split_bounds(a, &ab).into_iter().map(Some).collect()
            }
            None => (0..vp.len()).map(|_| None).collect(),
        };
        pool.install(|| {
            vp.into_par_iter().zip(ap).enumerate().for_each(|(k, (vs, a))| {
                resid_task::<W, K>(*self, st.parts[k] as usize..st.parts[k + 1] as usize, vs, r, j0, x, a);
            })
        });
    }

    /// FE fit per row for one coefficient column (x: m, alpha: ge·be).
    pub fn fit_rows(&self, x: &[f64], alpha: &[f64], pool: &Pool) -> Vec<f64> {
        let st = self.st;
        let be = st.be;
        let mut out = vec![0.0; st.n];
        let rb = st.part_rows(1);
        let op = split_bounds(&mut out, &rb);
        pool.install(|| {
            op.into_par_iter().enumerate().for_each(|(k, o)| {
                let row0 = rb[k];
                let mut he = vec![0.0; be];
                for s in st.parts[k] as usize..st.parts[k + 1] as usize {
                    for i in st.rows(s) {
                        design(st.e_icpt, st.he_raw(i), &mut he);
                        let f: f64 = (0..be).map(|l| he[l] * alpha[s * be + l]).sum();
                        o[i - row0] = if st.m > 0 { f + self.u_row::<1, 0>(i, x, 1, 0)[0] } else { f };
                    }
                }
            })
        });
        out
    }

    /// Dense S restricted to the small coordinates (m_s × m_s, row-major) and the diagonal of
    /// the level Gram Σ w h_c² of every small coordinate (the redundancy reference).
    pub fn assemble_small(&self, pool: &Pool) -> Result<(Vec<f64>, Vec<f64>), RfError> {
        let (st, ws) = (self.st, self.ws);
        let ms = st.m_small;
        let be = st.be;
        let small = &st.blocks[..st.nsmall];
        let bsmall: usize = small.iter().map(|b| b.b).sum();
        let nrb = (ms * ms).div_ceil(1 << 21).max(1);
        let rows_per = ms.div_ceil(nrb);
        let work: u64 = (0..st.ge).map(|s| ((st.rows(s).len() * bsmall).min(ms) as u64).pow(2)).sum::<u64>() + (st.n * bsmall * bsmall) as u64;
        let nt_want = (work / (4 * (rows_per * ms) as u64).max(1)).clamp(1, 64) as usize;
        let tb = st.even_tasks(nt_want);
        let nt = tb.len() - 1;
        let jobs: Vec<(usize, usize)> = (0..nrb).flat_map(|rbk| (0..nt).map(move |t| (rbk, t))).collect();
        let accs: Vec<(Vec<f64>, Vec<f64>)> = pool.install(|| {
            jobs.par_iter()
                .map(|&(rbk, t)| {
                    let k0 = rbk * rows_per;
                    let k1 = ((rbk + 1) * rows_per).min(ms);
                    let mut acc = vec![0.0; (k1 - k0) * ms];
                    let mut gd = vec![0.0; k1 - k0];
                    // Per segment: C = Σ w h_E h_c (be values per touched small coordinate c),
                    // in a dense scratch reset on first touch (marked by the segment index).
                    let mut cs = vec![0.0; ms * be];
                    let mut mark = vec![u32::MAX; ms];
                    let mut touched: Vec<u32> = Vec::with_capacity(ms);
                    let nss = st.nslots_small;
                    let mut mc = vec![0.0; be];
                    let l0 = usize::from(st.e_icpt);
                    for s in tb[t] as usize..tb[t + 1] as usize {
                        touched.clear();
                        for i in st.rows(s) {
                            let w = self.w_at(i);
                            let xe = st.he_raw(i);
                            let base = i * st.k;
                            let (rcs, rvs) = (&st.sc[base..base + nss], &st.sv[base..base + nss]);
                            for (&c, &v) in rcs.iter().zip(rvs) {
                                let cu = c as usize;
                                let cc = &mut cs[cu * be..(cu + 1) * be];
                                let wv = w * v;
                                // First touch in this segment: assign, else accumulate.
                                let first = mark[cu] != s as u32;
                                if first {
                                    mark[cu] = s as u32;
                                    touched.push(c);
                                }
                                if st.e_icpt {
                                    cc[0] = if first { wv } else { cc[0] + wv };
                                }
                                for (k, &x) in xe.iter().enumerate() {
                                    cc[l0 + k] = if first { wv * x } else { cc[l0 + k] + wv * x };
                                }
                                if cu >= k0 && cu < k1 {
                                    let row = &mut acc[(cu - k0) * ms..(cu - k0 + 1) * ms];
                                    for (&c2, &v2) in rcs.iter().zip(rvs) {
                                        if c2 <= c {
                                            row[c2 as usize] += wv * v2;
                                        }
                                    }
                                    gd[cu - k0] += wv * v;
                                }
                            }
                        }
                        // Subtract C' M_s⁺ C.
                        let mv = &ws.minv[s * be * be..(s + 1) * be * be];
                        for &c in &touched {
                            let cu = c as usize;
                            if cu < k0 || cu >= k1 {
                                continue;
                            }
                            let row = &mut acc[(cu - k0) * ms..(cu - k0 + 1) * ms];
                            if be == 1 {
                                let mc0 = mv[0] * cs[cu];
                                for &c2 in &touched {
                                    if c2 <= c {
                                        row[c2 as usize] -= mc0 * cs[c2 as usize];
                                    }
                                }
                                continue;
                            }
                            for l in 0..be {
                                mc[l] = (0..be).map(|l2| mv[l * be + l2] * cs[cu * be + l2]).sum();
                            }
                            for &c2 in &touched {
                                if c2 <= c {
                                    let c2 = c2 as usize;
                                    let v: f64 = mc.iter().zip(&cs[c2 * be..(c2 + 1) * be]).map(|(p, q)| p * q).sum();
                                    row[c2] -= v;
                                }
                            }
                        }
                    }
                    (acc, gd)
                })
                .collect()
        });
        let mut full = alloc_vec(ms * ms, 0.0)?;
        let mut gram = alloc_vec(ms, 0.0)?;
        pool.install(|| {
            full.par_chunks_mut(ms.max(1)).zip(gram.par_iter_mut()).enumerate().for_each(|(k, (row, gk))| {
                let rbk = k / rows_per;
                let k0 = rbk * rows_per;
                for (job, (acc, gd)) in jobs.iter().zip(&accs) {
                    if job.0 == rbk {
                        for (o, v) in row.iter_mut().zip(&acc[(k - k0) * ms..(k - k0 + 1) * ms]) {
                            *o += v;
                        }
                        *gk += gd[k - k0];
                    }
                }
            })
        });
        for k in 0..ms {
            for l in 0..k {
                full[l * ms + k] = full[k * ms + l];
            }
        }
        Ok((full, gram))
    }

    /// Per-segment aggregation of block `q`.
    fn block_agg(&self, q: usize, pool: &Pool) -> Result<BlockAgg, RfError> {
        let (st, ws) = (self.st, self.ws);
        let blk = &st.blocks[q];
        let (be, b) = (st.be, blk.b);
        let eb = be * b;
        let np = st.parts.len() - 1;
        type Piece = (Vec<u32>, Vec<u32>, Vec<f64>, Vec<f64>);
        let pieces: Vec<Piece> = pool.install(|| {
            (0..np)
                .into_par_iter()
                .map(|k| {
                    let mut counts = Vec::new();
                    let mut codes = Vec::new();
                    let mut cv = Vec::new();
                    let mut mcv = Vec::new();
                    let mut items: Vec<(u32, u32)> = Vec::new();
                    let (mut he, mut hq) = (vec![0.0; be], vec![0.0; b]);
                    for s in st.parts[k] as usize..st.parts[k + 1] as usize {
                        items.clear();
                        items.extend(st.rows(s).map(|i| (st.code(i, q) as u32, i as u32)));
                        items.sort_unstable();
                        let before = codes.len();
                        let mut a = 0;
                        while a < items.len() {
                            let g = items[a].0;
                            let at = cv.len();
                            cv.resize(at + eb, 0.0);
                            while a < items.len() && items[a].0 == g {
                                let i = items[a].1 as usize;
                                let w = ws.w(i);
                                design(st.e_icpt, st.he_raw(i), &mut he);
                                design(blk.icpt, st.hq_raw(i, blk), &mut hq);
                                for l in 0..be {
                                    for c in 0..b {
                                        cv[at + l * b + c] += w * he[l] * hq[c];
                                    }
                                }
                                a += 1;
                            }
                            codes.push(g);
                            let mv = &ws.minv[s * be * be..(s + 1) * be * be];
                            for l in 0..be {
                                for c in 0..b {
                                    let v: f64 = (0..be).map(|l2| mv[l * be + l2] * cv[at + l2 * b + c]).sum();
                                    mcv.push(v);
                                }
                            }
                        }
                        counts.push((codes.len() - before) as u32);
                    }
                    (counts, codes, cv, mcv)
                })
                .collect()
        });
        let mut off = alloc_vec(st.ge + 1, 0u32)?;
        let (mut codes, mut c, mut mc) = (Vec::new(), Vec::new(), Vec::new());
        let mut s = 0;
        let mut work = 0u64;
        for (counts, cd, cv, mcv) in pieces {
            for d in counts {
                off[s + 1] = off[s] + d;
                work += u64::from(d) * u64::from(d);
                s += 1;
            }
            codes.extend(cd);
            c.extend(cv);
            mc.extend(mcv);
        }
        Ok(BlockAgg { off, codes, c, mc, work })
    }

    /// Level neighbours of every level of block `q` (sharing a segment), including itself.
    fn level_degrees(&self, q: usize, agg: &BlockAgg, ix: &LevelIndex, pool: &Pool) -> Vec<u32> {
        let st = self.st;
        let blk = &st.blocks[q];
        let (perm, poff) = (&ix.perm[q - st.nsmall], &ix.perm_off[q - st.nsmall]);
        pool.install(|| {
            (0..blk.g)
                .into_par_iter()
                .with_min_len(256)
                .map_init(
                    || vec![u32::MAX; blk.g],
                    |mark, g| {
                        let stamp = g as u32;
                        mark[g] = stamp;
                        let mut cnt = 1u32;
                        let mut last = u32::MAX;
                        for &i in &perm[poff[g] as usize..poff[g + 1] as usize] {
                            let s = ix.row_seg[i as usize];
                            if s == last {
                                continue;
                            }
                            last = s;
                            for &g2 in &agg.codes[agg.off[s as usize] as usize..agg.off[s as usize + 1] as usize] {
                                if mark[g2 as usize] != stamp {
                                    mark[g2 as usize] = stamp;
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

    /// Exact S_qq of large block `q` over its non-redundant coordinates (full symmetric CSR) and
    /// the kept coordinates (relative to the block); `None` when the assembly work or the
    /// nonzeros exceed the budgets.
    fn assemble_block(&self, q: usize, cfg: &SolverConfig, pool: &Pool) -> Result<Option<(Csr, Vec<u32>)>, RfError> {
        let (st, ws) = (self.st, self.ws);
        let blk = &st.blocks[q];
        let (be, b) = (st.be, blk.b);
        let (b2, eb) = (b * b, be * b);
        let agg = self.block_agg(q, pool)?;
        if agg.work.saturating_mul(b2 as u64) > cfg.max_assembly_work {
            return Ok(None);
        }
        let ix = st.index(pool)?;
        let (perm, poff) = (&ix.perm[q - st.nsmall], &ix.perm_off[q - st.nsmall]);
        let deg = self.level_degrees(q, &agg, ix, pool);
        let nnz_levels: u64 = deg.iter().map(|&d| u64::from(d)).sum();
        if nnz_levels.saturating_mul(b2 as u64) > cfg.max_block_nnz {
            return Ok(None);
        }
        let mut lptr = alloc_vec(blk.g + 1, 0usize)?;
        for g in 0..blk.g {
            lptr[g + 1] = lptr[g] + deg[g] as usize;
        }
        let nnz = lptr[blk.g];
        let mut nbr = alloc_vec(nnz, 0u32)?;
        let mut vals = alloc_vec(nnz * b2, 0.0f64)?;
        let chunks: Vec<usize> = (0..=blk.g.div_ceil(512)).map(|k| (k * 512).min(blk.g)).collect();
        let nb_b: Vec<usize> = chunks.iter().map(|&g| lptr[g]).collect();
        let vb: Vec<usize> = nb_b.iter().map(|&p| p * b2).collect();
        let np = split_bounds(&mut nbr, &nb_b);
        let vp = split_bounds(&mut vals, &vb);
        let dq = &ws.dblk[q];
        pool.install(|| {
            np.into_par_iter().zip(vp).enumerate().for_each_init(
                || (vec![u32::MAX; blk.g], vec![0.0; blk.g * b2], Vec::<u32>::new()),
                |(mark, acc, touched), (k, (nbs, vs))| {
                    let base = lptr[chunks[k]];
                    for g in chunks[k]..chunks[k + 1] {
                        let stamp = g as u32;
                        touched.clear();
                        touched.push(g as u32);
                        mark[g] = stamp;
                        acc[g * b2..(g + 1) * b2].copy_from_slice(&dq[g * b2..(g + 1) * b2]);
                        let mut last = u32::MAX;
                        for &i in &perm[poff[g] as usize..poff[g + 1] as usize] {
                            let s = ix.row_seg[i as usize];
                            if s == last {
                                continue;
                            }
                            last = s;
                            let r = agg.off[s as usize] as usize..agg.off[s as usize + 1] as usize;
                            let lc = &agg.codes[r.clone()];
                            let me = r.start + lc.binary_search(&(g as u32)).expect("level present in its segment");
                            let mcg = &agg.mc[me * eb..(me + 1) * eb];
                            for e2 in r {
                                let g2 = agg.codes[e2] as usize;
                                if g2 == g {
                                    continue;
                                }
                                if mark[g2] != stamp {
                                    mark[g2] = stamp;
                                    touched.push(g2 as u32);
                                    acc[g2 * b2..(g2 + 1) * b2].iter_mut().for_each(|v| *v = 0.0);
                                }
                                let c2 = &agg.c[e2 * eb..(e2 + 1) * eb];
                                let out = &mut acc[g2 * b2..(g2 + 1) * b2];
                                for a in 0..b {
                                    for bb in 0..b {
                                        let mut v = 0.0;
                                        for l in 0..be {
                                            v += mcg[l * b + a] * c2[l * b + bb];
                                        }
                                        out[a * b + bb] -= v;
                                    }
                                }
                            }
                        }
                        touched.sort_unstable();
                        let at = lptr[g] - base;
                        for (j, &g2) in touched.iter().enumerate() {
                            nbs[at + j] = g2;
                            vs[(at + j) * b2..(at + j + 1) * b2].copy_from_slice(&acc[g2 as usize * b2..(g2 as usize + 1) * b2]);
                        }
                    }
                },
            )
        });
        // Coordinate CSR over the kept coordinates.
        let red = &ws.redundant[blk.off..blk.off + blk.g * b];
        let mut newid = vec![u32::MAX; blk.g * b];
        let mut keep = Vec::new();
        for (c, &rd) in red.iter().enumerate() {
            if !rd {
                newid[c] = keep.len() as u32;
                keep.push(c as u32);
            }
        }
        let mut ptr = Vec::with_capacity(keep.len() + 1);
        ptr.push(0u32);
        let mut col = Vec::new();
        let mut val = Vec::new();
        for &c in &keep {
            let (g, a) = (c as usize / b, c as usize % b);
            for e in lptr[g]..lptr[g + 1] {
                let g2 = nbr[e] as usize;
                for bb in 0..b {
                    let id = newid[g2 * b + bb];
                    if id != u32::MAX {
                        col.push(id);
                        val.push(vals[e * b2 + a * b + bb]);
                    }
                }
            }
            ptr.push(col.len() as u32);
        }
        Ok(Some((Csr { n: keep.len(), ptr, col, val }, keep)))
    }
}

/// Preconditioner of one large block.
pub enum LargePre {
    /// Per-level b×b block Jacobi.
    Jacobi,
    /// Exact sparse Cholesky of the block over its kept coordinates.
    Chol { keep: Vec<u32>, f: Box<SparseChol> },
}

pub struct SPre {
    pub small: Option<Dense>,
    /// One entry per large block.
    pub large: Vec<LargePre>,
}

impl SPre {
    fn apply(&self, ops: SOps, r: &[f64], z: &mut [f64], w: usize, pool: &Pool) {
        let st = ops.st;
        if let Some(d) = &self.small {
            d.apply(r, z, st.m_small, w, pool);
        }
        for (k, lp) in self.large.iter().enumerate() {
            let LargePre::Jacobi = lp else { continue };
            let q = st.nsmall + k;
            let blk = &st.blocks[q];
            let (off, b) = (blk.off, blk.b);
            let zs = &mut z[off * w..(off + blk.g * b) * w];
            let rs = &r[off * w..(off + blk.g * b) * w];
            let jv = &ops.ws.jinv[q];
            pool.install(|| {
                zs.par_chunks_mut(b * w).zip(rs.par_chunks(b * w)).enumerate().with_min_len(256).for_each(|(g, (zo, ri))| {
                    let jl = &jv[g * b * b..(g + 1) * b * b];
                    for a in 0..b {
                        let out = &mut zo[a * w..(a + 1) * w];
                        out.iter_mut().for_each(|v| *v = 0.0);
                        for c in 0..b {
                            let f = jl[a * b + c];
                            if f != 0.0 {
                                for (o, x) in out.iter_mut().zip(&ri[c * w..(c + 1) * w]) {
                                    *o += f * x;
                                }
                            }
                        }
                    }
                })
            });
        }
        // Factored blocks are independent: each (block, column group) is a sequential solve
        // task, and the tasks run concurrently.
        let tasks = solve_tasks(&self.large, w, |lp| matches!(lp, LargePre::Chol { .. }).then_some(SOLVE_COLS));
        if tasks.is_empty() {
            return;
        }
        let chol = |k: usize| match &self.large[k] {
            LargePre::Chol { keep, f } => (keep, f),
            LargePre::Jacobi => unreachable!("Jacobi blocks have no solve task"),
        };
        let sols: Vec<Vec<f64>> = pool.install(|| {
            tasks
                .par_iter()
                .map(|(k, cols)| {
                    let (keep, f) = chol(*k);
                    let off = st.blocks[st.nsmall + k].off;
                    let mut rhs = Mat::<f64>::from_fn(keep.len(), cols.len(), |i, j| r[(off + keep[i] as usize) * w + cols.start + j]);
                    f.solve(&mut rhs);
                    let mut sol = Vec::with_capacity(keep.len() * cols.len());
                    for j in 0..cols.len() {
                        sol.extend((0..keep.len()).map(|i| rhs[(i, j)]));
                    }
                    sol
                })
                .collect()
        });
        for (k, lp) in self.large.iter().enumerate() {
            if let LargePre::Chol { .. } = lp {
                let blk = &st.blocks[st.nsmall + k];
                z[blk.off * w..(blk.off + blk.g * blk.b) * w].iter_mut().for_each(|v| *v = 0.0);
            }
        }
        for ((k, cols), sol) in tasks.iter().zip(&sols) {
            let (keep, _) = chol(*k);
            let off = st.blocks[st.nsmall + k].off;
            for (j, c) in cols.clone().enumerate() {
                for (i, &kk) in keep.iter().enumerate() {
                    z[(off + kk as usize) * w + c] = sol[j * keep.len() + i];
                }
            }
        }
    }
}

/// Escalated preconditioner of large block `q`: exact sparse Cholesky within the budgets.
fn escalate_block(ops: SOps, q: usize, cfg: &SolverConfig, pool: &Pool, ctx: &Ctx) -> Result<LargePre, RfError> {
    let t = Instant::now();
    let out = if cfg.sparse_chol {
        match ops.assemble_block(q, cfg, pool)? {
            Some((csr, keep)) => match SparseChol::try_factor(&csr, pool)? {
                Some(f) => LargePre::Chol { keep, f: Box::new(f) },
                None => LargePre::Jacobi,
            },
            None => LargePre::Jacobi,
        }
    } else {
        LargePre::Jacobi
    };
    if ctx.verbose >= 2 {
        let blk = &ops.st.blocks[q];
        ctx.log(&format!(
            "regfast: escalate slope block {q} (G={} b={}) -> {} in {:.4}s",
            blk.g,
            blk.b,
            if matches!(out, LargePre::Chol { .. }) { "sparse-chol" } else { "block-jacobi" },
            t.elapsed().as_secs_f64()
        ));
    }
    Ok(out)
}

struct SSys<'a> {
    ops: SOps<'a>,
    pre: &'a mut SPre,
    scratch: PassScratch,
    w: usize,
    can_escalate: bool,
    large_fresh: bool,
    cfg: &'a SolverConfig,
}

impl CgSystem for SSys<'_> {
    fn matvec(&mut self, p: &[f64], q: &mut [f64], pool: &Pool) -> Result<(), RfError> {
        self.ops.apply(p, self.w, q, &mut self.scratch, pool)
    }

    fn precond(&mut self, r: &[f64], z: &mut [f64], pool: &Pool) {
        self.pre.apply(self.ops, r, z, self.w, pool);
    }

    fn exact(&self) -> bool {
        let st = self.ops.st;
        (st.m_small == st.m && self.pre.small.is_some())
            || (st.nsmall == 0 && st.nb == 1 && self.large_fresh && matches!(self.pre.large.first(), Some(LargePre::Chol { .. })))
    }

    fn can_escalate(&self) -> bool {
        self.can_escalate
    }

    fn escalate(&mut self, pool: &Pool, ctx: &Ctx) -> Result<bool, RfError> {
        self.can_escalate = false;
        let mut changed = false;
        for k in 0..self.pre.large.len() {
            if matches!(self.pre.large[k], LargePre::Jacobi) {
                let lp = escalate_block(self.ops, self.ops.st.nsmall + k, self.cfg, pool, ctx)?;
                changed |= !matches!(lp, LargePre::Jacobi);
                self.pre.large[k] = lp;
            }
        }
        Ok(changed)
    }
}

/// FE solver for designs with heterogeneous slopes (see the module docs).
pub struct SlopeSolver {
    st: SlopeStructure,
    ws: SlopeWeights,
    pre: SPre,
    /// Session terms: (intercept, slope count, levels) and FE1 for `term_values`.
    term_info: Vec<(bool, usize, usize)>,
    fe1: Option<usize>,
    basic_fresh: bool,
    escalated: bool,
    large_fresh: bool,
    base_iters: Option<usize>,
    need_refactor: bool,
    cfg: SolverConfig,
}

impl SlopeSolver {
    /// Builds the solver for session `terms` (kept order) and kept-order `weights`.
    pub fn new(terms: &[Term], weights: Option<&[f64]>, n: usize, cfg: SolverConfig, pool: &Pool, ctx: &Ctx) -> Result<SlopeSolver, RfError> {
        let mut t = Instant::now();
        if weights.is_some_and(|w| w.len() != n) {
            return Err(RfError::InvalidInput("weight length differs from the kept sample".into()));
        }
        let mut st = build_structure(terms, n, cfg.small_limit, pool)?;
        st.acc_bytes_max = cfg.acc_bytes_max;
        log_phase(ctx, &mut t, "slope structure");
        ctx.check()?;
        let w_rows = match weights {
            Some(w) => {
                let mut v = alloc_vec(n, 0.0)?;
                for (o, &k) in v.iter_mut().zip(&st.pi) {
                    *o = w[k as usize];
                }
                Some(v)
            }
            None => None,
        };
        let ws = slope_weights(&st, w_rows, pool)?;
        log_phase(ctx, &mut t, "slope weights/diag");
        if ctx.verbose >= 2 {
            let blocks: Vec<String> = st
                .blocks
                .iter()
                .map(|b| format!("{}:{}x{}{}", b.term, b.g, b.b, if b.small { "s" } else { "L" }))
                .collect();
            ctx.log(&format!(
                "regfast: n={} E=term {} (G={} b={}) m={} (small {}) blocks [{}]",
                st.n,
                st.e,
                st.ge,
                st.be,
                st.m,
                st.m_small,
                blocks.join(" ")
            ));
        }
        let nl = st.nb - st.nsmall;
        Ok(SlopeSolver {
            pre: SPre { small: None, large: (0..nl).map(|_| LargePre::Jacobi).collect() },
            term_info: terms.iter().map(|t| (t.intercept, t.slopes.len(), t.g)).collect(),
            fe1: fe1_index(terms),
            st,
            ws,
            basic_fresh: false,
            escalated: false,
            large_fresh: true,
            base_iters: None,
            need_refactor: false,
            cfg,
        })
    }

    /// Kernels bound to the current structure and weights.
    pub fn ops(&self) -> SOps<'_> {
        SOps { st: &self.st, ws: &self.ws }
    }

    pub fn structure(&self) -> &SlopeStructure {
        &self.st
    }

    /// Names of the current large-block preconditioners.
    pub fn precond_kinds(&self) -> Vec<&'static str> {
        self.pre.large.iter().map(|p| if matches!(p, LargePre::Chol { .. }) { "sparse-chol" } else { "block-jacobi" }).collect()
    }

    /// Rebuilds the escalated factors for the current weights at the next `residualize`.
    pub fn request_refactor(&mut self) {
        if self.escalated {
            self.need_refactor = true;
        }
    }

    fn ensure_basic(&mut self, pool: &Pool) -> Result<(), RfError> {
        if self.basic_fresh {
            return Ok(());
        }
        let ms = self.st.m_small;
        self.pre.small = if ms > 0 {
            let (a, gram) = SOps { st: &self.st, ws: &self.ws }.assemble_small(pool)?;
            for (k, red) in self.ws.redundant[..ms].iter_mut().enumerate() {
                *red = !(a[k * ms + k] > REDUNDANT * gram[k]);
            }
            Some(Dense::factor(&a, ms, &self.ws.redundant[..ms], pool)?)
        } else {
            None
        };
        self.basic_fresh = true;
        Ok(())
    }

    fn refresh_large(&mut self, pool: &Pool, ctx: &Ctx) -> Result<(), RfError> {
        let ops = SOps { st: &self.st, ws: &self.ws };
        for k in 0..self.pre.large.len() {
            let q = self.st.nsmall + k;
            let refreshed = match &mut self.pre.large[k] {
                LargePre::Jacobi => true,
                LargePre::Chol { keep, f } => match ops.assemble_block(q, &self.cfg, pool)? {
                    Some((csr, k2)) if k2 == *keep => f.refactor(&csr, pool)?,
                    _ => false,
                },
            };
            if !refreshed {
                self.pre.large[k] = escalate_block(ops, q, &self.cfg, pool, ctx)?;
            }
            ctx.check()?;
        }
        self.large_fresh = true;
        self.base_iters = None;
        Ok(())
    }
}

impl Absorber for SlopeSolver {
    fn n(&self) -> usize {
        self.st.n
    }

    fn row_order(&self) -> &[u32] {
        &self.st.pi
    }

    fn weights(&self) -> Option<&[f64]> {
        self.ws.w_rows.as_deref()
    }

    fn set_weights(&mut self, w: Option<Vec<f64>>, pool: &Pool, ctx: &Ctx) -> Result<(), RfError> {
        if w.as_ref().is_some_and(|v| v.len() != self.st.n || v.iter().any(|x| !(x.is_finite() && *x > 0.0))) {
            return Err(RfError::InvalidInput("weights must be positive and finite".into()));
        }
        ctx.check()?;
        self.ws = slope_weights(&self.st, w, pool)?;
        self.basic_fresh = false;
        if self.escalated {
            self.large_fresh = false;
        }
        Ok(())
    }

    fn residualize(
        &mut self,
        z: &mut [f64],
        r: usize,
        o: &SolveOpts,
        warm: Option<&FeCoefs>,
        pool: &Pool,
        ctx: &Ctx,
    ) -> Result<SolveInfo, RfError> {
        let (n, m) = (self.st.n, self.st.m);
        let pe = self.st.ge * self.st.be;
        if z.len() != n * r {
            return Err(RfError::InvalidInput("residualize: block size differs from n·R".into()));
        }
        if warm.is_some_and(|wc| wc.r != r || wc.p != m + pe) {
            return Err(RfError::InvalidInput("residualize: warm start has a different shape".into()));
        }
        if r == 0 {
            return Ok(SolveInfo { iterations: 1, escalated: false, coefs: o.want_coefs.then(|| FeCoefs { p: m + pe, r, data: Vec::new() }), exact: false });
        }
        ctx.check()?;
        let mut t = Instant::now();
        let mut x = match warm {
            Some(wc) => wc.data[..m * r].to_vec(),
            None => alloc_vec(m * r, 0.0)?,
        };
        let mut iterations = 1;
        let mut escalated_now = false;
        if m > 0 {
            self.ensure_basic(pool)?;
            if self.need_refactor {
                self.refresh_large(pool, ctx)?;
                self.need_refactor = false;
            }
            log_phase(ctx, &mut t, "preconditioner");
            let ops = SOps { st: &self.st, ws: &self.ws };
            let mut scratch = PassScratch::default();
            let (rhs, ex) = ops.rhs(z, r, &mut scratch, pool)?;
            log_phase(ctx, &mut t, "rhs pass");
            ctx.check()?;
            let (wsq, wsum, ss1) = (&ex[..r], &ex[r..2 * r], &ex[2 * r..]);
            let sw = self.ws.sw;
            let ss0: Vec<f64> = (0..r)
                .map(|j| {
                    let s = wsq[j] - wsum[j] * wsum[j] / sw;
                    if s > 1e-15 * wsq[j] { s } else { 0.0 }
                })
                .collect();
            for j in 0..r {
                if ss0[j] == 0.0 {
                    for i in 0..m {
                        x[i * r + j] = 0.0;
                    }
                }
            }
            let can_escalate = !self.escalated && self.st.has_large() && self.cfg.sparse_chol;
            let mut sys = SSys {
                ops: SOps { st: &self.st, ws: &self.ws },
                pre: &mut self.pre,
                scratch,
                w: r,
                can_escalate,
                large_fresh: self.large_fresh,
                cfg: &self.cfg,
            };
            let out = cg::pcg(&mut sys, &rhs, &mut x, r, ss1, &ss0, o.tol, o.maxiter, pool, ctx)?;
            iterations = out.iterations.max(1);
            if out.escalated || !sys.can_escalate && can_escalate {
                self.escalated = true;
                self.large_fresh = true;
                self.base_iters = None;
            }
            escalated_now = out.escalated;
            if ctx.verbose >= 2 {
                ctx.log(&format!("regfast: pcg {} iterations (escalated: {})", out.iterations, out.escalated));
            }
            if self.escalated {
                match self.base_iters {
                    None => self.base_iters = Some(iterations),
                    Some(base) if !self.large_fresh && iterations > 30.max(2 * base) => self.need_refactor = true,
                    Some(_) => {}
                }
            }
            log_phase(ctx, &mut t, "pcg");
            ctx.check()?;
        }
        let mut alpha = if o.want_coefs { Some(alloc_vec(pe * r, 0.0)?) } else { None };
        SOps { st: &self.st, ws: &self.ws }.residualize_rows(z, r, &x, alpha.as_deref_mut(), pool);
        log_phase(ctx, &mut t, "residualize pass");
        let coefs = alpha.map(|a| {
            let mut data = x;
            data.extend_from_slice(&a);
            FeCoefs { p: m + pe, r, data }
        });
        Ok(SolveInfo { iterations, escalated: escalated_now, coefs, exact: false })
    }

    fn fit_rows(&self, c: &FeCoefs, col: usize, pool: &Pool) -> Vec<f64> {
        let v = c.col(col);
        let m = self.st.m;
        SOps { st: &self.st, ws: &self.ws }.fit_rows(&v[..m], &v[m..], pool)
    }

    fn term_values(&self, c: &FeCoefs, col: usize) -> Vec<Vec<f64>> {
        let v = c.col(col);
        let st = &self.st;
        // Per term: slot vectors.
        let mut vals: Vec<Vec<Vec<f64>>> = vec![Vec::new(); self.term_info.len()];
        vals[st.e] = (0..st.be).map(|k| (0..st.ge).map(|g| v[st.m + g * st.be + k]).collect()).collect();
        let mut by_term = vec![usize::MAX; self.term_info.len()];
        for (q, blk) in st.blocks.iter().enumerate() {
            by_term[blk.term] = q;
            vals[blk.term] = (0..blk.b).map(|k| (0..blk.g).map(|g| v[blk.off + g * blk.b + k]).collect()).collect();
        }
        if let Some(f) = self.fe1 {
            for (tq, &(icpt, ns, g)) in self.term_info.iter().enumerate() {
                if tq == f || !icpt || ns > 0 {
                    continue;
                }
                let q = by_term[tq];
                let mut nr = vec![0.0; g];
                for i in 0..st.n {
                    nr[st.code(i, q)] += self.ws.w(i);
                }
                let mean: f64 = vals[tq][0].iter().zip(&nr).map(|(a, w)| a * w).sum::<f64>() / self.ws.sw;
                vals[tq][0].iter_mut().for_each(|a| *a -= mean);
                vals[f][0].iter_mut().for_each(|a| *a += mean);
            }
        }
        vals.into_iter().flatten().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    #[test]
    fn sweep_inverts_and_drops_dependent_pivots() {
        let mut rng = SplitMix64::new(3);
        let k = 4;
        let b: Vec<f64> = (0..k * k).map(|_| rng.next_f64() - 0.5).collect();
        let mut a = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                a[i * k + j] = (0..k).map(|l| b[i * k + l] * b[j * k + l]).sum::<f64>() + if i == j { 0.1 } else { 0.0 };
            }
        }
        let refd: Vec<f64> = (0..k).map(|i| a[i * k + i]).collect();
        let mut g = a.clone();
        sweep_ginv(&mut g, k, &refd, 1e-12);
        for i in 0..k {
            for j in 0..k {
                let v: f64 = (0..k).map(|l| a[i * k + l] * g[l * k + j]).sum();
                assert!((v - f64::from(u8::from(i == j))).abs() < 1e-10, "A·A⁻¹ ({i},{j}) = {v}");
            }
        }
        // Column 2 = column 0 + column 1: dropped, A G A = A.
        let x: Vec<[f64; 3]> = (0..6).map(|_| [rng.next_f64(), rng.next_f64(), 0.0]).map(|[p, q, _]| [p, q, p + q]).collect();
        let mut a3 = vec![0.0; 9];
        for r in &x {
            for i in 0..3 {
                for j in 0..3 {
                    a3[i * 3 + j] += r[i] * r[j];
                }
            }
        }
        let refd3: Vec<f64> = (0..3).map(|i| a3[i * 4]).collect();
        let mut g3 = a3.clone();
        sweep_ginv(&mut g3, 3, &refd3, 1e-12);
        assert!((0..3).all(|j| g3[2 * 3 + j] == 0.0 && g3[j * 3 + 2] == 0.0));
        for i in 0..3 {
            for j in 0..3 {
                let v: f64 = (0..3).flat_map(|l| (0..3).map(move |m| (l, m))).map(|(l, m)| a3[i * 3 + l] * g3[l * 3 + m] * a3[m * 3 + j]).sum();
                assert!((v - a3[i * 3 + j]).abs() < 1e-10 * a3[0].abs().max(1.0), "AGA ({i},{j})");
            }
        }
    }
}
