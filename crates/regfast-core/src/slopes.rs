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
//!   coordinates), per-level b×b block Jacobi on the large blocks, escalated once (the `cg` probe)
//!   to an exact sparse Cholesky of a large block when its assembly work Σ_s d_s²·b² and its
//!   nonzeros fit the [`SolverConfig`] budgets. approx-chol is never used here.
//!
//! FeCoefs layout: p = m + G_E·b_E rows, `[x (m×R); α_E (G_E·b_E × R)]` (level-major, slots
//! intercept then slopes). Deterministic for any thread count: fixed task partitions, task-order
//! combines and level pulls, as the intercept-only solver.

use std::ops::Range;
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
    /// Offset of this term's slopes within a row of [`SlopeStructure::h`].
    pub hoff: usize,
    /// Offset of level 0 in the reduced coordinate vector.
    pub off: usize,
    pub small: bool,
    /// Rows sorted by (level, row).
    pub perm: Vec<u32>,
    /// `perm` offsets per level (g + 1 entries).
    pub perm_off: Vec<u32>,
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
    /// Level of E of every row.
    pub row_seg: Vec<u32>,
    /// Number of blocks.
    pub nb: usize,
    /// Codes of every block per row (row-major n × nb).
    pub codes: Vec<u32>,
    /// Slope values per row (row-major n × hw): E's slopes first, then each block's at `hoff`.
    pub hw: usize,
    pub h: Vec<f64>,
    /// Small blocks first, then large ones, each in term order.
    pub blocks: Vec<SBlock>,
    pub nsmall: usize,
    /// Reduced dimension Σ g·b and the small coordinates `0..m_small`.
    pub m: usize,
    pub m_small: usize,
    /// Fixed partition of the segments into parts of ≳ [`CHUNK`] rows (boundaries).
    pub parts: Vec<u32>,
    /// Largest reduced vector (bytes, all R columns) scattered entirely through accumulators.
    pub acc_bytes_max: usize,
}

impl SlopeStructure {
    #[inline]
    fn rows(&self, s: usize) -> Range<usize> {
        self.seg_off[s] as usize..self.seg_off[s + 1] as usize
    }

    #[inline]
    fn he_raw(&self, i: usize) -> &[f64] {
        &self.h[i * self.hw..i * self.hw + self.e_ns]
    }

    #[inline]
    fn hq_raw(&self, i: usize, blk: &SBlock) -> &[f64] {
        let base = i * self.hw + blk.hoff;
        &self.h[base..base + blk.ns]
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
        !self.has_large() || self.m * r * 8 <= self.acc_bytes_max
    }
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

    // Rows sorted by (level of E, observation).
    let (seg_off, mut pi) = scatter_by_code(&te.codes, ge, pool)?;
    let parts = obs_groups(&seg_off);
    let row_bounds: Vec<usize> = parts.iter().map(|&s| seg_off[s as usize] as usize).collect();
    let mut row_seg = alloc_vec(n, 0u32)?;
    {
        let pp = split_bounds(&mut pi, &row_bounds);
        let rp = split_bounds(&mut row_seg, &row_bounds);
        pool.install(|| {
            pp.into_par_iter().zip(rp).enumerate().for_each(|(k, (pis, rs))| {
                let base = row_bounds[k];
                for s in parts[k] as usize..parts[k + 1] as usize {
                    let r = seg_off[s] as usize - base..seg_off[s + 1] as usize - base;
                    pis[r.clone()].sort_unstable();
                    rs[r].iter_mut().for_each(|v| *v = s as u32);
                }
            })
        });
    }

    // Blocks: small (ascending coordinates while the total stays ≤ small_limit) first.
    let others: Vec<usize> = (0..terms.len()).filter(|&q| q != e).collect();
    let width = |q: usize| usize::from(terms[q].intercept) + terms[q].slopes.len();
    let mut by_size = others.clone();
    by_size.sort_by_key(|&q| (terms[q].g * width(q), q));
    let mut small_set = vec![false; terms.len()];
    let mut total = 0usize;
    for &q in &by_size {
        let c = terms[q].g * width(q);
        if total + c > small_limit {
            break;
        }
        total += c;
        small_set[q] = true;
    }
    let mut order: Vec<usize> = others.iter().copied().filter(|&q| small_set[q]).collect();
    let nsmall = order.len();
    order.extend(others.iter().copied().filter(|&q| !small_set[q]));
    let mut blocks = Vec::with_capacity(order.len());
    let (mut off, mut hoff) = (0usize, e_ns);
    for &q in &order {
        let t = &terms[q];
        let b = width(q);
        blocks.push(SBlock {
            term: q,
            g: t.g,
            icpt: t.intercept,
            ns: t.slopes.len(),
            b,
            hoff,
            off,
            small: small_set[q],
            perm: Vec::new(),
            perm_off: Vec::new(),
        });
        off += t.g * b;
        hoff += t.slopes.len();
    }
    let m = off;
    let m_small: usize = blocks[..nsmall].iter().map(|b| b.g * b.b).sum();
    let nb = blocks.len();
    let hw = hoff;

    // Codes and slope values in row order.
    let mut codes = alloc_vec(n * nb, 0u32)?;
    let mut h = alloc_vec(n * hw, 0.0f64)?;
    pool.install(|| {
        if nb > 0 {
            codes.par_chunks_mut(CHUNK * nb).enumerate().for_each(|(c, cc)| {
                for (k, row) in cc.chunks_exact_mut(nb).enumerate() {
                    let o = pi[c * CHUNK + k] as usize;
                    for (v, blk) in row.iter_mut().zip(&blocks) {
                        *v = terms[blk.term].codes[o];
                    }
                }
            });
        }
        if hw > 0 {
            h.par_chunks_mut(CHUNK * hw).enumerate().for_each(|(c, hc)| {
                for (k, hr) in hc.chunks_exact_mut(hw).enumerate() {
                    let o = pi[c * CHUNK + k] as usize;
                    for (s, x) in te.slopes.iter().enumerate() {
                        hr[s] = x[o];
                    }
                    for blk in &blocks {
                        for (s, x) in terms[blk.term].slopes.iter().enumerate() {
                            hr[blk.hoff + s] = x[o];
                        }
                    }
                }
            });
        }
    });

    // Level-sorted rows of every large block (small blocks are covered by the dense factor).
    for q in nsmall..nb {
        let col: Vec<u32> = pool.install(|| (0..n).into_par_iter().map(|i| codes[i * nb + q]).collect());
        let (poff, mut perm) = scatter_by_code(&col, blocks[q].g, pool)?;
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
        blocks[q].perm = perm;
        blocks[q].perm_off = poff;
    }

    Ok(SlopeStructure {
        n,
        pi,
        e,
        ge,
        e_icpt,
        e_ns,
        be,
        seg_off,
        row_seg,
        nb,
        codes,
        hw,
        h,
        blocks,
        nsmall,
        m,
        m_small,
        parts,
        acc_bytes_max: crate::operator::ACC_BYTES_MAX,
    })
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
    pool.install(|| {
        minv.par_chunks_mut(be2).enumerate().with_min_len(256).for_each_init(Vec::new, |sc, (s, out)| level_minv(st, &w_rows, s, out, sc))
    });
    // Diagonal blocks / block-Jacobi inverses of the large blocks. Small blocks get empty
    // entries; their redundancy flags are set from the assembled dense block (`ensure_basic`).
    let mut dblk = Vec::with_capacity(st.nb);
    let mut jinv = Vec::with_capacity(st.nb);
    let mut redundant = alloc_vec(st.m, false)?;
    for blk in &st.blocks {
        if blk.small {
            dblk.push(Vec::new());
            jinv.push(Vec::new());
            continue;
        }
        let b = blk.b;
        let b2 = b * b;
        let mut d = alloc_vec(blk.g * b2, 0.0)?;
        let mut j = alloc_vec(blk.g * b2, 0.0)?;
        let red = &mut redundant[blk.off..blk.off + blk.g * b];
        pool.install(|| {
            d.par_chunks_mut(b2).zip(j.par_chunks_mut(b2)).zip(red.par_chunks_mut(b)).enumerate().with_min_len(64).for_each_init(
                || (vec![0.0; b2], vec![0.0; st.be * b], vec![0.0; st.be], vec![0.0; b], vec![0.0; b]),
                |(gram, c, he, hq, refd), (g, ((dl, jl), rl))| {
                    gram.iter_mut().for_each(|v| *v = 0.0);
                    dl.iter_mut().for_each(|v| *v = 0.0);
                    let rows = &blk.perm[blk.perm_off[g] as usize..blk.perm_off[g + 1] as usize];
                    let mut k = 0;
                    while k < rows.len() {
                        let s = st.row_seg[rows[k] as usize] as usize;
                        c.iter_mut().for_each(|v| *v = 0.0);
                        while k < rows.len() && st.row_seg[rows[k] as usize] as usize == s {
                            let i = rows[k] as usize;
                            let wi = w_rows.as_ref().map_or(1.0, |v| v[i]);
                            design(st.e_icpt, st.he_raw(i), he);
                            design(blk.icpt, st.hq_raw(i, blk), hq);
                            for a in 0..b {
                                for bb in 0..b {
                                    gram[a * b + bb] += wi * hq[a] * hq[bb];
                                }
                            }
                            for l in 0..st.be {
                                for a in 0..b {
                                    c[l * b + a] += wi * he[l] * hq[a];
                                }
                            }
                            k += 1;
                        }
                        // dl += C' M_s⁺ C.
                        let mv = &minv[s * be2..(s + 1) * be2];
                        for a in 0..b {
                            for bb in 0..b {
                                let mut acc = 0.0;
                                for l in 0..st.be {
                                    let mut t = 0.0;
                                    for l2 in 0..st.be {
                                        t += mv[l * st.be + l2] * c[l2 * b + bb];
                                    }
                                    acc += c[l * b + a] * t;
                                }
                                dl[a * b + bb] += acc;
                            }
                        }
                    }
                    for (dv, gv) in dl.iter_mut().zip(gram.iter()) {
                        *dv = gv - *dv;
                    }
                    for a in 0..b {
                        rl[a] = !(dl[a * b + a] > REDUNDANT * gram[a * b + a]);
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

/// Per-task scratch of the segment kernels.
struct Scratch {
    d: Vec<f64>,
    a: Vec<f64>,
    t: Vec<f64>,
    he: Vec<f64>,
}

impl Scratch {
    fn new(be: usize, r: usize) -> Scratch {
        Scratch { d: Vec::new(), a: vec![0.0; be * r], t: vec![0.0; be * r], he: vec![0.0; be] }
    }
}

/// Calls `$s.$f::<RC>(args)` with RC = `$r` for 1 ≤ r ≤ 12 (the column count becomes a
/// compile-time constant, so the per-row loops unroll) and RC = 0 (dynamic) otherwise.
macro_rules! by_width {
    ($r:expr, $s:ident . $f:ident ( $($a:expr),* )) => {
        match $r {
            1 => $s.$f::<1>($($a),*),
            2 => $s.$f::<2>($($a),*),
            3 => $s.$f::<3>($($a),*),
            4 => $s.$f::<4>($($a),*),
            5 => $s.$f::<5>($($a),*),
            6 => $s.$f::<6>($($a),*),
            7 => $s.$f::<7>($($a),*),
            8 => $s.$f::<8>($($a),*),
            9 => $s.$f::<9>($($a),*),
            10 => $s.$f::<10>($($a),*),
            11 => $s.$f::<11>($($a),*),
            12 => $s.$f::<12>($($a),*),
            _ => $s.$f::<0>($($a),*),
        }
    };
}

/// The column count of a kernel monomorphized for `RC` (0 = dynamic `r`).
#[inline(always)]
const fn width<const RC: usize>(r: usize) -> usize {
    if RC == 0 { r } else { RC }
}

impl SOps<'_> {
    /// u = Σ_q h_{q,i}'P[q, g_q(i)] (r columns).
    #[inline(always)]
    fn u_row<const RC: usize>(&self, i: usize, p: &[f64], r: usize, u: &mut [f64]) {
        let r = width::<RC>(r);
        let st = self.st;
        u.iter_mut().for_each(|v| *v = 0.0);
        let codes = &st.codes[i * st.nb..(i + 1) * st.nb];
        for (blk, &g) in st.blocks.iter().zip(codes) {
            let mut c = (blk.off + g as usize * blk.b) * r;
            if blk.icpt {
                for (uj, pj) in u.iter_mut().zip(&p[c..c + r]) {
                    *uj += pj;
                }
                c += r;
            }
            for &x in st.hq_raw(i, blk) {
                for (uj, pj) in u.iter_mut().zip(&p[c..c + r]) {
                    *uj += x * pj;
                }
                c += r;
            }
        }
    }

    /// acc[q, g_q(i)] += h_{q,i} z for the first `nblk` blocks (acc indexed from coordinate 0).
    #[inline(always)]
    fn scatter_row<const RC: usize>(&self, i: usize, z: &[f64], r: usize, acc: &mut [f64], nblk: usize) {
        let r = width::<RC>(r);
        let st = self.st;
        let codes = &st.codes[i * st.nb..i * st.nb + nblk];
        for (blk, &g) in st.blocks[..nblk].iter().zip(codes) {
            let mut c = (blk.off + g as usize * blk.b) * r;
            if blk.icpt {
                for (a, zj) in acc[c..c + r].iter_mut().zip(z) {
                    *a += zj;
                }
                c += r;
            }
            for &x in st.hq_raw(i, blk) {
                for (a, zj) in acc[c..c + r].iter_mut().zip(z) {
                    *a += x * zj;
                }
                c += r;
            }
        }
    }

    /// Projects out E within segment s: with d (rows × r) on input, t = M_s⁺ Σ w h_E d' and
    /// d_i −= h_{E,i}'t. `sc.t` holds t (be × r) on return.
    #[inline(always)]
    fn seg_project<const RC: usize>(&self, s: usize, rows: Range<usize>, d: &mut [f64], r: usize, sc: &mut Scratch) {
        let r = width::<RC>(r);
        let (st, ws) = (self.st, self.ws);
        let be = st.be;
        let Scratch { a, t, he, .. } = sc;
        if st.e_icpt && be == 1 {
            // Intercept-only E: t = Σ w d / Σ w (M_s⁺ = 1/Σw), d_i −= t.
            a.iter_mut().for_each(|v| *v = 0.0);
            match &ws.w_rows {
                None => {
                    for dr in d.chunks_exact(r) {
                        for (av, dv) in a.iter_mut().zip(dr) {
                            *av += dv;
                        }
                    }
                }
                Some(w) => {
                    for (dr, &wi) in d.chunks_exact(r).zip(&w[rows]) {
                        for (av, dv) in a.iter_mut().zip(dr) {
                            *av += wi * dv;
                        }
                    }
                }
            }
            let inv = ws.minv[s];
            for (tv, av) in t.iter_mut().zip(a.iter()) {
                *tv = inv * av;
            }
            for dr in d.chunks_exact_mut(r) {
                for (dv, tv) in dr.iter_mut().zip(t.iter()) {
                    *dv -= tv;
                }
            }
            return;
        }
        a.iter_mut().for_each(|v| *v = 0.0);
        for (k, i) in rows.clone().enumerate() {
            let w = ws.w(i);
            design(st.e_icpt, st.he_raw(i), he);
            let dr = &d[k * r..(k + 1) * r];
            for l in 0..be {
                let c = w * he[l];
                for (av, dv) in a[l * r..(l + 1) * r].iter_mut().zip(dr) {
                    *av += c * dv;
                }
            }
        }
        let mv = &ws.minv[s * be * be..(s + 1) * be * be];
        for l in 0..be {
            let tr = &mut t[l * r..(l + 1) * r];
            tr.iter_mut().for_each(|v| *v = 0.0);
            for l2 in 0..be {
                let c = mv[l * be + l2];
                if c != 0.0 {
                    for (tv, av) in tr.iter_mut().zip(&a[l2 * r..(l2 + 1) * r]) {
                        *tv += c * av;
                    }
                }
            }
        }
        for (k, i) in rows.enumerate() {
            design(st.e_icpt, st.he_raw(i), he);
            let dr = &mut d[k * r..(k + 1) * r];
            for l in 0..be {
                let c = he[l];
                for (dv, tv) in dr.iter_mut().zip(&t[l * r..(l + 1) * r]) {
                    *dv -= c * tv;
                }
            }
        }
    }

    /// Fused pass: per segment, `fill` loads d (rows × r), E is projected out, z_i = w_i d_i is
    /// scattered into `y` (m × r; fully overwritten). With `stats`, returns per column
    /// [Σwv², Σwv, Σw·e²] of the loaded values v and the E-residuals e.
    fn scatter_pass<const RC: usize>(
        &self,
        r: usize,
        y: &mut [f64],
        zbuf: &mut [f64],
        stats: bool,
        pool: &Pool,
        fill: impl Fn(usize, Range<usize>, &mut [f64]) + Sync,
    ) -> Vec<f64> {
        let r = width::<RC>(r);
        let st = self.st;
        let all = st.accumulates_all(r);
        let nacc = if all { st.nb } else { st.nsmall };
        let acc_len = if all { st.m * r } else { st.m_small * r };
        let tb = st.tasks(acc_len);
        let nt = tb.len() - 1;
        let zp: Vec<Option<&mut [f64]>> = if all {
            (0..nt).map(|_| None).collect()
        } else {
            let rb: Vec<usize> = tb.iter().map(|&s| st.seg_off[s as usize] as usize * r).collect();
            split_bounds(zbuf, &rb).into_iter().map(Some).collect()
        };
        let ne = if stats { 3 * r } else { 0 };
        // Intercept-only E: the projection is fused into the two row loops below.
        let fused = st.e_icpt && st.be == 1;
        let results: Vec<(Vec<f64>, Vec<f64>)> = pool.install(|| {
            zp.into_par_iter()
                .enumerate()
                .map(|(t, mut zp)| {
                    let mut acc = vec![0.0; acc_len];
                    let mut ex = vec![0.0; ne];
                    let mut sc = Scratch::new(st.be, r);
                    let row0 = st.seg_off[tb[t] as usize] as usize;
                    for s in tb[t] as usize..tb[t + 1] as usize {
                        let rows = st.rows(s);
                        let len = rows.len() * r;
                        if sc.d.len() < len {
                            sc.d.resize(len, 0.0);
                        }
                        let d = &mut sc.d[..len];
                        fill(s, rows.clone(), d);
                        if fused {
                            let a = &mut sc.a;
                            a.iter_mut().for_each(|v| *v = 0.0);
                            for (dr, i) in d.chunks_exact(r).zip(rows.clone()) {
                                let w = self.ws.w(i);
                                for j in 0..r {
                                    let v = dr[j];
                                    a[j] += w * v;
                                    if stats {
                                        ex[j] += w * v * v;
                                        ex[r + j] += w * v;
                                    }
                                }
                            }
                            let inv = self.ws.minv[s];
                            a.iter_mut().for_each(|v| *v *= inv);
                            for (z, i) in d.chunks_exact_mut(r).zip(rows) {
                                let w = self.ws.w(i);
                                for j in 0..r {
                                    let e = z[j] - a[j];
                                    if stats {
                                        ex[2 * r + j] += w * e * e;
                                    }
                                    z[j] = w * e;
                                }
                                self.scatter_row::<RC>(i, z, r, &mut acc, nacc);
                                if let Some(zp) = zp.as_deref_mut() {
                                    zp[(i - row0) * r..(i - row0 + 1) * r].copy_from_slice(z);
                                }
                            }
                            continue;
                        }
                        if stats {
                            for (k, i) in rows.clone().enumerate() {
                                let w = self.ws.w(i);
                                for (j, &v) in d[k * r..(k + 1) * r].iter().enumerate() {
                                    ex[j] += w * v * v;
                                    ex[r + j] += w * v;
                                }
                            }
                        }
                        let mut d = std::mem::take(&mut sc.d);
                        self.seg_project::<RC>(s, rows.clone(), &mut d[..len], r, &mut sc);
                        for (k, i) in rows.enumerate() {
                            let w = self.ws.w(i);
                            let z = &mut d[k * r..(k + 1) * r];
                            if stats {
                                for (j, &v) in z.iter().enumerate() {
                                    ex[2 * r + j] += w * v * v;
                                }
                            }
                            z.iter_mut().for_each(|v| *v *= w);
                            self.scatter_row::<RC>(i, z, r, &mut acc, nacc);
                            if let Some(zp) = zp.as_deref_mut() {
                                zp[(i - row0) * r..(i - row0 + 1) * r].copy_from_slice(z);
                            }
                        }
                        sc.d = d;
                    }
                    (acc, ex)
                })
                .collect()
        });
        pool.install(|| {
            y[..acc_len].par_chunks_mut(4096).enumerate().for_each(|(ci, out)| {
                let base = ci * 4096;
                let len = out.len();
                out.iter_mut().for_each(|v| *v = 0.0);
                for (acc, _) in &results {
                    for (o, v) in out.iter_mut().zip(&acc[base..base + len]) {
                        *o += v;
                    }
                }
            })
        });
        if !all {
            self.pull_large::<RC>(zbuf, r, y, pool);
        }
        let mut extra = vec![0.0; ne];
        for (_, ex) in &results {
            for (a, v) in extra.iter_mut().zip(ex) {
                *a += v;
            }
        }
        extra
    }

    /// Y[q, g] = Σ_{i ∈ rows of g} h_{q,i} z_i for every large block (level pulls).
    fn pull_large<const RC: usize>(&self, zbuf: &[f64], r: usize, y: &mut [f64], pool: &Pool) {
        let r = width::<RC>(r);
        let st = self.st;
        for blk in &st.blocks[st.nsmall..] {
            let ys = &mut y[blk.off * r..(blk.off + blk.g * blk.b) * r];
            pool.install(|| {
                ys.par_chunks_mut(blk.b * r).enumerate().with_min_len(64).for_each(|(g, out)| {
                    out.iter_mut().for_each(|v| *v = 0.0);
                    for &i in &blk.perm[blk.perm_off[g] as usize..blk.perm_off[g + 1] as usize] {
                        let i = i as usize;
                        let z = &zbuf[i * r..(i + 1) * r];
                        let mut c = 0;
                        if blk.icpt {
                            for (o, zj) in out[..r].iter_mut().zip(z) {
                                *o += zj;
                            }
                            c = r;
                        }
                        for &x in st.hq_raw(i, blk) {
                            for (o, zj) in out[c..c + r].iter_mut().zip(z) {
                                *o += x * zj;
                            }
                            c += r;
                        }
                    }
                })
            });
        }
    }

    pub fn needs_zbuf(&self, r: usize) -> bool {
        !self.st.accumulates_all(r)
    }

    /// Y = S·P (m × r row-major).
    pub fn apply(&self, p: &[f64], r: usize, y: &mut [f64], zbuf: &mut [f64], pool: &Pool) {
        by_width!(r, self.apply_n(p, r, y, zbuf, pool))
    }

    fn apply_n<const RC: usize>(&self, p: &[f64], r: usize, y: &mut [f64], zbuf: &mut [f64], pool: &Pool) {
        let r = width::<RC>(r);
        self.scatter_pass::<RC>(r, y, zbuf, false, pool, |_, rows, d| {
            for (k, i) in rows.enumerate() {
                self.u_row::<RC>(i, p, r, &mut d[k * r..(k + 1) * r]);
            }
        });
    }

    /// Reduced right-hand side of the row block `v` (n × r): (rhs, Σwv², Σwv, Σw e²) with e the
    /// residual after E alone.
    pub fn rhs(&self, v: &[f64], r: usize, zbuf: &mut [f64], pool: &Pool) -> Result<(Vec<f64>, Vec<f64>), RfError> {
        let mut y = alloc_vec(self.st.m * r, 0.0)?;
        let ex = by_width!(r, self.rhs_n(v, r, &mut y, zbuf, pool));
        Ok((y, ex))
    }

    fn rhs_n<const RC: usize>(&self, v: &[f64], r: usize, y: &mut [f64], zbuf: &mut [f64], pool: &Pool) -> Vec<f64> {
        let r = width::<RC>(r);
        self.scatter_pass::<RC>(r, y, zbuf, true, pool, |_, rows, d| d.copy_from_slice(&v[rows.start * r..rows.end * r]))
    }

    /// In place: v_i ← d_i − h_{E,i}'t with d = v − u(x); writes α_E (ge·be × r) when given.
    pub fn residualize_rows(&self, v: &mut [f64], r: usize, x: &[f64], alpha: Option<&mut [f64]>, pool: &Pool) {
        by_width!(r, self.residualize_rows_n(v, r, x, alpha, pool))
    }

    fn residualize_rows_n<const RC: usize>(&self, v: &mut [f64], r: usize, x: &[f64], alpha: Option<&mut [f64]>, pool: &Pool) {
        let r = width::<RC>(r);
        let st = self.st;
        let be = st.be;
        let rb: Vec<usize> = st.parts.iter().map(|&s| st.seg_off[s as usize] as usize * r).collect();
        let vp = split_bounds(v, &rb);
        let ap: Vec<Option<&mut [f64]>> = match alpha {
            Some(a) => {
                let ab: Vec<usize> = st.parts.iter().map(|&s| s as usize * be * r).collect();
                split_bounds(a, &ab).into_iter().map(Some).collect()
            }
            None => (0..vp.len()).map(|_| None).collect(),
        };
        pool.install(|| {
            vp.into_par_iter().zip(ap).enumerate().for_each(|(k, (vs, mut a))| {
                let (s0, s1) = (st.parts[k] as usize, st.parts[k + 1] as usize);
                let row0 = st.seg_off[s0] as usize;
                let mut sc = Scratch::new(be, r);
                let mut u = vec![0.0; r];
                for s in s0..s1 {
                    let rows = st.rows(s);
                    let d = &mut vs[(rows.start - row0) * r..(rows.end - row0) * r];
                    if st.m > 0 {
                        for (kk, i) in rows.clone().enumerate() {
                            self.u_row::<RC>(i, x, r, &mut u);
                            for (dv, uv) in d[kk * r..(kk + 1) * r].iter_mut().zip(&u) {
                                *dv -= uv;
                            }
                        }
                    }
                    self.seg_project::<RC>(s, rows, d, r, &mut sc);
                    if let Some(a) = a.as_deref_mut() {
                        a[(s - s0) * be * r..(s - s0 + 1) * be * r].copy_from_slice(&sc.t);
                    }
                }
            })
        });
    }

    /// FE fit per row for one coefficient column (x: m, alpha: ge·be).
    pub fn fit_rows(&self, x: &[f64], alpha: &[f64], pool: &Pool) -> Vec<f64> {
        let st = self.st;
        let be = st.be;
        let mut out = vec![0.0; st.n];
        pool.install(|| {
            out.par_chunks_mut(CHUNK).enumerate().for_each(|(c, ch)| {
                let mut he = vec![0.0; be];
                let mut u = [0.0];
                for (k, o) in ch.iter_mut().enumerate() {
                    let i = c * CHUNK + k;
                    let s = st.row_seg[i] as usize;
                    design(st.e_icpt, st.he_raw(i), &mut he);
                    let mut f: f64 = (0..be).map(|l| he[l] * alpha[s * be + l]).sum();
                    if st.m > 0 {
                        self.u_row::<1>(i, x, 1, &mut u);
                        f += u[0];
                    }
                    *o = f;
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
        let bsmall: usize = st.blocks[..st.nsmall].iter().map(|b| b.b).sum();
        let nrb = (ms * ms).div_ceil(1 << 21).max(1);
        let rows_per = ms.div_ceil(nrb);
        let work: u64 = (0..st.ge).map(|s| ((st.rows(s).len() * bsmall).min(ms) as u64).pow(2)).sum::<u64>() + (st.n * bsmall * bsmall) as u64;
        let nt_want = (work / (4 * (rows_per * ms) as u64).max(1)).clamp(1, 64) as usize;
        let tb = st.even_tasks(nt_want);
        let nt = tb.len() - 1;
        let jobs: Vec<(usize, usize)> = (0..nrb).flat_map(|rbk| (0..nt).map(move |t| (rbk, t))).collect();
        let accs: Vec<Vec<f64>> = pool.install(|| {
            jobs.par_iter()
                .map(|&(rbk, t)| {
                    let k0 = rbk * rows_per;
                    let k1 = ((rbk + 1) * rows_per).min(ms);
                    let mut acc = vec![0.0; (k1 - k0) * ms];
                    // Row's small coordinates and values; per-segment (coord, row, value) items.
                    let mut rc: Vec<(u32, f64)> = Vec::new();
                    let mut items: Vec<(u32, u32, f64)> = Vec::new();
                    let mut hes: Vec<f64> = Vec::new();
                    let mut agg: Vec<u32> = Vec::new();
                    let mut cs: Vec<f64> = Vec::new();
                    let mut mcs: Vec<f64> = Vec::new();
                    let mut he = vec![0.0; be];
                    for s in tb[t] as usize..tb[t + 1] as usize {
                        let rows = st.rows(s);
                        items.clear();
                        hes.clear();
                        for (kr, i) in rows.clone().enumerate() {
                            let w = ws.w(i);
                            design(st.e_icpt, st.he_raw(i), &mut he);
                            hes.extend(he.iter().map(|v| v * w));
                            rc.clear();
                            for (q, blk) in st.blocks[..st.nsmall].iter().enumerate() {
                                let base = blk.off + st.codes[i * st.nb + q] as usize * blk.b;
                                let mut c = base;
                                if blk.icpt {
                                    rc.push((c as u32, 1.0));
                                    c += 1;
                                }
                                for &x in st.hq_raw(i, blk) {
                                    rc.push((c as u32, x));
                                    c += 1;
                                }
                            }
                            for &(c, v) in &rc {
                                items.push((c, kr as u32, v));
                                let c = c as usize;
                                if c >= k0 && c < k1 {
                                    let row = &mut acc[(c - k0) * ms..(c - k0 + 1) * ms];
                                    for &(c2, v2) in &rc {
                                        if (c2 as usize) <= c {
                                            row[c2 as usize] += w * v * v2;
                                        }
                                    }
                                }
                            }
                        }
                        items.sort_unstable_by_key(|e| (e.0, e.1));
                        agg.clear();
                        cs.clear();
                        let mut a = 0;
                        while a < items.len() {
                            let c = items[a].0;
                            let at = cs.len();
                            cs.resize(at + be, 0.0);
                            while a < items.len() && items[a].0 == c {
                                let (kr, v) = (items[a].1 as usize, items[a].2);
                                for l in 0..be {
                                    cs[at + l] += hes[kr * be + l] * v;
                                }
                                a += 1;
                            }
                            agg.push(c);
                        }
                        let mv = &ws.minv[s * be * be..(s + 1) * be * be];
                        mcs.clear();
                        mcs.resize(cs.len(), 0.0);
                        for e in 0..agg.len() {
                            for l in 0..be {
                                mcs[e * be + l] = (0..be).map(|l2| mv[l * be + l2] * cs[e * be + l2]).sum();
                            }
                        }
                        for (e, &c) in agg.iter().enumerate() {
                            let c = c as usize;
                            if c < k0 || c >= k1 {
                                continue;
                            }
                            let row = &mut acc[(c - k0) * ms..(c - k0 + 1) * ms];
                            let mc = &mcs[e * be..(e + 1) * be];
                            for (e2, &c2) in agg[..=e].iter().enumerate() {
                                let v: f64 = mc.iter().zip(&cs[e2 * be..(e2 + 1) * be]).map(|(p, q)| p * q).sum();
                                row[c2 as usize] -= v;
                            }
                        }
                    }
                    acc
                })
                .collect()
        });
        let mut full = alloc_vec(ms * ms, 0.0)?;
        pool.install(|| {
            full.par_chunks_mut(ms.max(1)).enumerate().for_each(|(k, row)| {
                let rbk = k / rows_per;
                let k0 = rbk * rows_per;
                for (job, acc) in jobs.iter().zip(&accs) {
                    if job.0 == rbk {
                        for (o, v) in row.iter_mut().zip(&acc[(k - k0) * ms..(k - k0 + 1) * ms]) {
                            *o += v;
                        }
                    }
                }
            })
        });
        for k in 0..ms {
            for l in 0..k {
                full[l * ms + k] = full[k * ms + l];
            }
        }
        let gram = sum_chunks(pool, st.n, CHUNK, ms, |rows, acc| {
            for i in rows {
                let w = ws.w(i);
                for (q, blk) in st.blocks[..st.nsmall].iter().enumerate() {
                    let mut c = blk.off + st.codes[i * st.nb + q] as usize * blk.b;
                    if blk.icpt {
                        acc[c] += w;
                        c += 1;
                    }
                    for &x in st.hq_raw(i, blk) {
                        acc[c] += w * x * x;
                        c += 1;
                    }
                }
            }
        });
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
                        items.extend(st.rows(s).map(|i| (st.codes[i * st.nb + q], i as u32)));
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
    fn level_degrees(&self, q: usize, agg: &BlockAgg, pool: &Pool) -> Vec<u32> {
        let st = self.st;
        let blk = &st.blocks[q];
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
                        for &i in &blk.perm[blk.perm_off[g] as usize..blk.perm_off[g + 1] as usize] {
                            let s = st.row_seg[i as usize];
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
        let deg = self.level_degrees(q, &agg, pool);
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
                        for &i in &blk.perm[blk.perm_off[g] as usize..blk.perm_off[g + 1] as usize] {
                            let s = st.row_seg[i as usize];
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
    zbuf: Vec<f64>,
    w: usize,
    can_escalate: bool,
    large_fresh: bool,
    cfg: &'a SolverConfig,
}

impl CgSystem for SSys<'_> {
    fn matvec(&mut self, p: &[f64], q: &mut [f64], pool: &Pool) -> Result<(), RfError> {
        self.ops.apply(p, self.w, q, &mut self.zbuf, pool);
        Ok(())
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
            return Ok(SolveInfo { iterations: 1, escalated: false, coefs: o.want_coefs.then(|| FeCoefs { p: m + pe, r, data: Vec::new() }) });
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
            let mut zbuf = if ops.needs_zbuf(r) { alloc_vec(n * r, 0.0)? } else { Vec::new() };
            let (rhs, ex) = ops.rhs(z, r, &mut zbuf, pool)?;
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
                zbuf,
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
        Ok(SolveInfo { iterations, escalated: escalated_now, coefs })
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
                    nr[st.codes[i * st.nb + q] as usize] += self.ws.w(i);
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
