//! Exact leverage of the fixed-effect columns (SE design contract 3c).
//!
//! `h_i = w_i d_i'(D'WD)^+ d_i`, where `d_i` is row `i` of the matrix `D` of every fixed-effect
//! column: the intercept dummies `1[level]` and the varying-slope columns `x * 1[level]` of every
//! term. This is the fixed-effect part of fixest 0.14.2's exact `hatvalues` (`P_fixef + P_slope`,
//! methods.R `hatvalues.fixest`) and of Stata `regress ..., vce(hc2)` with explicit dummies; the
//! full-model leverage adds `w_i a~_i'(X~'WX~)^-1 a~_i` for the FE-residualized regressors (FWL).
//!
//! Method (direct, no iteration, no approximation):
//! 1. `A = D'WD` is assembled over all columns (entries from co-occurrence within rows; every
//!    pair of columns that share a row is a structural entry even when its value is 0) and
//!    scaled to unit diagonal (`S A S`, `S = diag(A)^-1/2`; zero columns get scale 0).
//! 2. The connected-component redundancies between the intercepts of the finest term and of
//!    each other term are removed exactly, from the graph alone ([`component_columns`]).
//! 3. faer AMD ordering and supernodal `L D L'`. A pivot below the tolerance marks a column
//!    that is dependent on the columns eliminated before it (further redundancies with 3+
//!    terms, nested terms, slopes constant within a level, zero columns, ...): faer substitutes
//!    [`DROP`] for it, so its column of `L` and its updates vanish, and the column is then
//!    dropped exactly (`L` column set to 0, `D^+ = 0`). `G = L^-T D^+ L^-1` satisfies
//!    `A G A = A`, and `d' G d` is the same for every generalized inverse `G` when `d` is in
//!    range(`A`), which every `d_i` is (`A = sum_i w_i d_i d_i'`).
//! 4. Takahashi selected inversion in supernodal form: `Z = G` on the pattern of `L`, top-down
//!    over the elimination tree. Two columns sharing a row are an entry of `A`, hence of `L`'s
//!    pattern, so `h_i = w_i sum_{p,q in cols(i)} v_p v_q Z_pq` needs only selected entries.
//!
//! Pivot tolerance: on the unit-diagonal scale a pivot is the squared sine of the angle (in the
//! W inner product) between a column and the span of the columns eliminated before it. For a
//! slope column eliminated after its own intercept it is exactly the ratio of the centered to
//! the uncentered weighted sum of squares of the slope variable within the level, the quantity
//! reghdfe compares with 1e-12 to declare a slope degenerate; [`PIVOT_TOL`] uses the same
//! threshold. A zero pivot accumulated over `m` updates rounds to up to a few `m * eps`, so the
//! threshold of each column is raised to `PIVOT_ROUNDING * m` ([`PIVOT_ROUNDING`]): 1e-12 up
//! to ~300 updates, ~4e-9 for a pivot updated 1e6 times. Legitimately small pivots of weakly
//! connected graphs (a low-weight observation bridging two groups: ~1e-5) stay far above it
//! (tests `weak_bridge_is_kept`, `nested_and_duplicated_terms`, `perf_fe_hat` sweeps).
//!
//! Determinism: the result is a function of the input only. Assembly sums each entry over its
//! rows in kept order, faer's factorization does not depend on the thread count, and each
//! supernode of the selected inverse is computed by one fixed sequence of operations (chosen by
//! its size) from its ancestors' finished blocks, whatever the schedule.

use std::sync::atomic::{AtomicBool, Ordering};

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::linalg::matmul::matmul;
use faer::linalg::matmul::triangular::{BlockStructure, matmul as tri_matmul};
use faer::linalg::triangular_inverse::invert_unit_lower_triangular;
use faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place;
use faer::perm::PermRef;
use faer::prelude::{Reborrow, ReborrowMut};
use faer::sparse::linalg::cholesky::supernodal::SymbolicSupernodalCholesky;
use faer::sparse::linalg::cholesky::{CholeskySymbolicParams, SymbolicCholeskyRaw, SymmetricOrdering, factorize_symbolic_cholesky};
use faer::sparse::linalg::{SupernodalThreshold, amd};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Accum, Mat, MatMut, MatRef, Par, Side};
use rayon::prelude::*;

use crate::api::Ctx;
use crate::error::RfError;
use crate::par::{self, CHUNK, Pool};

/// One fixed-effect term, kept order: level codes `< g`, an intercept dummy per level when
/// `intercept`, and one slope column `x * 1[level]` per entry of `slopes`.
pub(crate) struct FeCols<'a> {
    pub codes: &'a [u32],
    pub g: usize,
    pub intercept: bool,
    pub slopes: Vec<&'a [f64]>,
}

/// Pivots at or below this value (unit-diagonal scale) are dropped as dependent columns
/// (see the module documentation); raised per column by [`PIVOT_ROUNDING`].
pub(crate) const PIVOT_TOL: f64 = 1e-12;

/// Rounding allowance per update of a pivot: a pivot that is exactly zero is computed as
/// `1 - sum_k l_jk^2 d_k` over its `m` updates (`sum <= 1` on the unit-diagonal scale), with an
/// error bounded by about `m * eps` and observed up to ~4 `m * eps` (balanced 1e6 x 10 panel,
/// m = 1e6: the redundant pivot rounds to ~1e-9). Column j's threshold is raised to at least
/// `16 eps * m_j`.
const PIVOT_ROUNDING: f64 = 16.0 * f64::EPSILON;

/// Value faer substitutes for a dropped pivot: the column of `L` becomes rounding noise
/// divided by 1e150 (then set to 0) and its Schur-complement updates vanish.
const DROP: f64 = 1e150;

/// Memory budget: entries of the Cholesky factor `L` (its selected inverse has as many).
/// 2^27 entries = 1 GiB each for `L` and `Z`; dense temporaries of the selected inversion
/// add at most a few times the largest front. Larger problems return an error.
pub(crate) const MAX_FACTOR_ENTRIES: usize = 1 << 27;

/// Factorization multiply-adds from which faer's numeric factorization runs in parallel
/// (as `precond::sparse_chol`).
const PAR_FACTOR_FLOPS: f64 = 1e10;

/// Supernodes whose order (columns + below-diagonal rows) reaches this are processed one at a
/// time with parallel dense kernels (bounding the dense temporaries); smaller supernodes are
/// processed concurrently, each sequentially.
const BIG_FRONT: usize = 1024;

/// Multiply-adds of a supernode below which the selected inversion uses plain loops instead of
/// faer's dense kernels (call overhead dominates for the many tiny supernodes of panels).
const SMALL_NODE_FLOPS: usize = 1 << 15;

/// AMD dense-row threshold: columns of degree > AMD_DENSE * sqrt(ncol) are ordered last
/// without AMD bookkeeping. With faer's default (10), the period columns of a 1e6-row,
/// 1e5 x 1e3 panel (degree ~1e3 each) make AMD take 2.2 s; with 1 it takes 0.015 s for the
/// same factor, and the AKM fill is unchanged (`perf_fe_hat`).
const AMD_DENSE: f64 = 1.0;

#[derive(Clone, Copy)]
struct Tuning {
    max_entries: usize,
    par_factor_flops: f64,
    big_front: usize,
    pivot_tol: f64,
    /// Remove the component redundancies exactly before factorizing ([`component_columns`]).
    predrop: bool,
    /// AMD: rows of degree > amd_dense * sqrt(ncol) are ordered last.
    amd_dense: f64,
}

const TUNING: Tuning = Tuning { max_entries: MAX_FACTOR_ENTRIES, par_factor_flops: PAR_FACTOR_FLOPS, big_front: BIG_FRONT, pivot_tol: PIVOT_TOL, predrop: true, amd_dense: AMD_DENSE };

/// Exact diagonal of the weighted projection on all fixed-effect columns (intercept dummies and
/// slope columns of every term): `h_i = w_i d_i'(D'WD)^+ d_i`, kept order. Invariant to
/// redundant columns. Errors (no approximation) when the factor exceeds [`MAX_FACTOR_ENTRIES`].
pub(crate) fn fe_hat(terms: &[FeCols], w: Option<&[f64]>, n: usize, ctx: &Ctx, pool: &Pool) -> Result<Vec<f64>, RfError> {
    fe_hat_with(terms, w, n, ctx, pool, TUNING).map(|(h, _)| h)
}

/// Diagnostics of the general path (tests and `verbose`).
#[derive(Clone, Copy, Debug, Default)]
struct Info {
    columns: usize,
    nnz_a: usize,
    factor_entries: usize,
    dropped: usize,
}

fn fe_hat_with(terms: &[FeCols], w: Option<&[f64]>, n: usize, ctx: &Ctx, pool: &Pool, tune: Tuning) -> Result<(Vec<f64>, Info), RfError> {
    validate(terms, w, n, pool)?;
    let active: Vec<&FeCols> = terms.iter().filter(|t| t.g > 0 && (t.intercept || !t.slopes.is_empty())).collect();
    if n == 0 || active.is_empty() {
        return Ok((par::alloc_vec(n, 0.0)?, Info::default()));
    }
    if active.len() == 1 && usize::from(active[0].intercept) + active[0].slopes.len() == 1 {
        return Ok((one_column(active[0], w, n, pool)?, Info::default()));
    }
    general(&active, w, n, ctx, pool, tune)
}

fn validate(terms: &[FeCols], w: Option<&[f64]>, n: usize, pool: &Pool) -> Result<(), RfError> {
    if n > u32::MAX as usize {
        return Err(RfError::InvalidInput("leverage: more than 2^32 observations".into()));
    }
    for t in terms {
        if t.codes.len() != n || t.slopes.iter().any(|s| s.len() != n) {
            return Err(RfError::Internal("leverage: term length differs from the number of observations".into()));
        }
        let g = t.g;
        if !pool.install(|| t.codes.par_chunks(CHUNK).all(|c| c.iter().all(|&v| (v as usize) < g))) {
            return Err(RfError::Internal("leverage: level code out of range".into()));
        }
        for s in &t.slopes {
            if !pool.install(|| s.par_chunks(CHUNK).all(|c| c.iter().all(|v| v.is_finite()))) {
                return Err(RfError::InvalidInput("leverage: slope variable has non-finite values".into()));
            }
        }
    }
    if let Some(w) = w {
        if w.len() != n {
            return Err(RfError::Internal("leverage: weight length differs from the number of observations".into()));
        }
        if !pool.install(|| w.par_chunks(CHUNK).all(|c| c.iter().all(|v| v.is_finite() && *v >= 0.0))) {
            return Err(RfError::InvalidInput("leverage: weights must be finite and non-negative".into()));
        }
    }
    Ok(())
}

/// One term with one column per level (intercept only, or a single slope without intercept):
/// `A` is diagonal and `h_i = w_i v_i^2 / sum_{level} w v^2` (0 for a zero column).
fn one_column(t: &FeCols, w: Option<&[f64]>, n: usize, pool: &Pool) -> Result<Vec<f64>, RfError> {
    let v = |i: usize| if t.intercept { 1.0 } else { t.slopes[0][i] };
    let mut tot = par::alloc_vec(t.g, 0.0)?;
    for i in 0..n {
        tot[t.codes[i] as usize] += w.map_or(1.0, |w| w[i]) * v(i) * v(i);
    }
    let mut h = par::alloc_vec(n, 0.0)?;
    pool.install(|| {
        h.par_chunks_mut(CHUNK).enumerate().for_each(|(c, out)| {
            for (k, o) in out.iter_mut().enumerate() {
                let i = c * CHUNK + k;
                let m = w.map_or(1.0, |w| w[i]) * v(i) * v(i);
                let tg = tot[t.codes[i] as usize];
                *o = if m > 0.0 { m / tg } else { 0.0 };
            }
        })
    });
    Ok(h)
}

/// A term in the internal column layout: columns `off + level * b + slot`, slot 0 the
/// intercept when present, then the slopes.
struct Term<'a> {
    codes: &'a [u32],
    g: usize,
    b: usize,
    off: usize,
    intercept: bool,
    slopes: &'a [&'a [f64]],
}

impl Term<'_> {
    #[inline(always)]
    fn val(&self, slot: usize, i: usize) -> f64 {
        if self.intercept {
            if slot == 0 { 1.0 } else { self.slopes[slot - 1][i] }
        } else {
            self.slopes[slot][i]
        }
    }
}

/// Lower triangle (CSC, sorted rows, diagonal first) of `A = D'WD`.
struct Lower {
    col_ptr: Vec<u32>,
    rows: Vec<u32>,
    vals: Vec<f64>,
}

/// Columns emitted by one chunk of levels of one term.
struct Part {
    counts: Vec<u32>,
    rows: Vec<u32>,
    vals: Vec<f64>,
}

/// Rows of each level of `codes` (counting sort, kept order within a level).
fn group_rows(codes: &[u32], g: usize) -> Result<(Vec<u32>, Vec<u32>), RfError> {
    let mut start = par::alloc_vec(g + 1, 0u32)?;
    for &c in codes {
        start[c as usize + 1] += 1;
    }
    for l in 0..g {
        start[l + 1] += start[l];
    }
    let mut next = start.clone();
    let mut idx = par::alloc_vec(codes.len(), 0u32)?;
    for (i, &c) in codes.iter().enumerate() {
        let p = &mut next[c as usize];
        idx[*p as usize] = i as u32;
        *p += 1;
    }
    Ok((start, idx))
}

/// Consecutive level ranges of at least [`CHUNK`] rows (the last may be smaller).
fn level_chunks(start: &[u32]) -> Vec<(usize, usize)> {
    let g = start.len() - 1;
    let mut out = Vec::new();
    let mut lo = 0;
    while lo < g {
        let mut hi = lo + 1;
        while hi < g && (start[hi] - start[lo]) < CHUNK as u32 {
            hi += 1;
        }
        out.push((lo, hi));
        lo = hi;
    }
    out
}

/// Columns of levels `lo..hi` of term `t`: the level's own block (rows `slot..b`), then for
/// every later term its co-occurring levels in ascending order.
fn assemble_levels(terms: &[Term], t: usize, start: &[u32], idx: &[u32], w: Option<&[f64]>, lo: usize, hi: usize) -> Part {
    let tm = &terms[t];
    let later = &terms[t + 1..];
    let b = tm.b;
    let mut part = Part { counts: Vec::with_capacity((hi - lo) * b), rows: Vec::new(), vals: Vec::new() };
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    let mut agg_codes: Vec<Vec<u32>> = vec![Vec::new(); later.len()];
    let mut agg_vals: Vec<Vec<f64>> = vec![Vec::new(); later.len()];
    let mut own = vec![0.0; b * b];
    let mut vs = vec![0.0; b];
    for gl in lo..hi {
        let lr = &idx[start[gl] as usize..start[gl + 1] as usize];
        own.fill(0.0);
        for &i in lr {
            let i = i as usize;
            let wi = w.map_or(1.0, |w| w[i]);
            for (s, v) in vs.iter_mut().enumerate() {
                *v = wi * tm.val(s, i);
            }
            for s in 0..b {
                for s2 in s..b {
                    own[s * b + s2] += vs[s] * tm.val(s2, i);
                }
            }
        }
        for (k, r) in later.iter().enumerate() {
            pairs.clear();
            pairs.extend(lr.iter().map(|&i| (r.codes[i as usize], i)));
            pairs.sort_unstable();
            let (codes, av) = (&mut agg_codes[k], &mut agg_vals[k]);
            codes.clear();
            av.clear();
            let br = r.b;
            for &(c, i) in &pairs {
                if codes.last() != Some(&c) {
                    codes.push(c);
                    av.resize(av.len() + b * br, 0.0);
                }
                let base = av.len() - b * br;
                let i = i as usize;
                let wi = w.map_or(1.0, |w| w[i]);
                for s in 0..b {
                    let v = wi * tm.val(s, i);
                    for s2 in 0..br {
                        av[base + s * br + s2] += v * r.val(s2, i);
                    }
                }
            }
        }
        for s in 0..b {
            let before = part.rows.len();
            for s2 in s..b {
                part.rows.push((tm.off + gl * b + s2) as u32);
                part.vals.push(own[s * b + s2]);
            }
            for (k, r) in later.iter().enumerate() {
                let br = r.b;
                for (ci, &c) in agg_codes[k].iter().enumerate() {
                    for s2 in 0..br {
                        part.rows.push((r.off + c as usize * br + s2) as u32);
                        part.vals.push(agg_vals[k][ci * b * br + s * br + s2]);
                    }
                }
            }
            part.counts.push((part.rows.len() - before) as u32);
        }
    }
    part
}

fn assemble(terms: &[Term], ncol: usize, w: Option<&[f64]>, pool: &Pool, ctx: &Ctx) -> Result<Lower, RfError> {
    let mut col_ptr = par::alloc_vec(ncol + 1, 0u32)?;
    let mut rows: Vec<u32> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    let mut col = 0usize;
    for t in 0..terms.len() {
        ctx.check()?;
        let (start, idx) = group_rows(terms[t].codes, terms[t].g)?;
        let chunks = level_chunks(&start);
        let parts: Vec<Part> = pool.install(|| chunks.par_iter().map(|&(lo, hi)| assemble_levels(terms, t, &start, &idx, w, lo, hi)).collect());
        for p in parts {
            if rows.len() + p.rows.len() > u32::MAX as usize {
                return Err(RfError::InvalidInput("leverage: the fixed-effect cross-product matrix has more than 2^32 entries".into()));
            }
            rows.try_reserve(p.rows.len()).map_err(|_| RfError::OutOfMemory)?;
            vals.try_reserve(p.vals.len()).map_err(|_| RfError::OutOfMemory)?;
            let base = rows.len() as u32;
            let mut acc = base;
            for &c in &p.counts {
                acc += c;
                col += 1;
                col_ptr[col] = acc;
            }
            rows.extend_from_slice(&p.rows);
            vals.extend_from_slice(&p.vals);
        }
    }
    debug_assert_eq!(col, ncol);
    Ok(Lower { col_ptr, rows, vals })
}

/// Intercept columns removed exactly before factorizing. The anchor is the first term with an
/// intercept (internal order: most levels). For every other term q with an intercept and every
/// connected component C of the anchor-q bipartite graph (rows with positive weight),
/// `sum_{q levels in C} 1[level] = 1_C = sum_{anchor levels in C} 1[level]` on those rows; the
/// lowest q level of C is removed. Each relation contains exactly one removed column, so the
/// span of the columns is unchanged, and these zero pivots (the bulk of the rank deficiency,
/// with the largest rounding: sums over a whole component) never reach the factorization.
fn component_columns(terms: &[Term], w: Option<&[f64]>) -> Result<Vec<usize>, RfError> {
    let Some(a) = terms.iter().position(|t| t.intercept) else { return Ok(Vec::new()) };
    let anchor = &terms[a];
    let mut out = Vec::new();
    for (q, tq) in terms.iter().enumerate() {
        if q == a || !tq.intercept {
            continue;
        }
        // Union-find over anchor levels 0..ga and q levels ga..ga+gq.
        let ga = anchor.g;
        let mut parent: Vec<u32> = par::alloc_vec(ga + tq.g, 0u32)?;
        for (k, p) in parent.iter_mut().enumerate() {
            *p = k as u32;
        }
        let find = |parent: &mut [u32], mut x: u32| {
            while parent[x as usize] != x {
                let gp = parent[parent[x as usize] as usize];
                parent[x as usize] = gp;
                x = gp;
            }
            x
        };
        let mut touched = vec![false; tq.g];
        for i in 0..anchor.codes.len() {
            if w.is_some_and(|w| w[i] <= 0.0) {
                continue;
            }
            let lq = tq.codes[i] as usize;
            touched[lq] = true;
            let ra = find(&mut parent, anchor.codes[i]);
            let rq = find(&mut parent, (ga + lq) as u32);
            if ra != rq {
                let (lo, hi) = (ra.min(rq), ra.max(rq));
                parent[hi as usize] = lo;
            }
        }
        let mut seen = vec![false; ga + tq.g];
        for (l, &t) in touched.iter().enumerate() {
            if t {
                let r = find(&mut parent, (ga + l) as u32) as usize;
                if !seen[r] {
                    seen[r] = true;
                    out.push(tq.off + l * tq.b);
                }
            }
        }
    }
    Ok(out)
}

/// Raw pointer shared by the workers of the selected inversion; every supernode writes only
/// its own block and reads only finished ancestors' blocks.
#[derive(Clone, Copy)]
struct SyncPtr(*mut f64);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}

/// Supernodal structure and values for the selected inversion.
struct Selinv<'a> {
    sn: &'a SymbolicSupernodalCholesky<u32>,
    l: &'a [f64],
    dinv: &'a [f64],
    z: SyncPtr,
    col_super: &'a [u32],
    failed: AtomicBool,
}

impl Selinv<'_> {
    #[inline]
    fn range(&self, s: usize) -> (usize, usize) {
        (self.sn.supernode_begin()[s] as usize, self.sn.supernode_end()[s] as usize)
    }

    #[inline]
    fn pattern(&self, s: usize) -> &[u32] {
        self.sn.supernode(s).pattern()
    }

    #[inline]
    fn val_off(&self, s: usize) -> usize {
        self.sn.col_ptr_for_val()[s] as usize
    }

    /// The finished block of supernode `s` (column-major, (nc + nr) x nc).
    ///
    /// # Safety
    /// `s` must be finished and not written concurrently.
    unsafe fn zblock(&self, s: usize) -> &[f64] {
        let (b, e) = self.range(s);
        let len = (e - b + self.pattern(s).len()) * (e - b);
        unsafe { std::slice::from_raw_parts(self.z.0.add(self.val_off(s)), len) }
    }

    /// Z_RR of supernode `s` (nr x nr, column-major, both triangles) from its ancestors.
    ///
    /// # Safety
    /// Every ancestor of `s` must be finished.
    unsafe fn gather(&self, pat: &[u32], zrr: &mut [f64], pos: &mut Vec<usize>) -> bool {
        let nr = pat.len();
        let mut k = 0;
        while k < nr {
            let t = self.col_super[pat[k] as usize] as usize;
            let (bt, et) = self.range(t);
            let patt = self.pattern(t);
            let nct = et - bt;
            let ldt = nct + patt.len();
            let mut k2 = k;
            while k2 < nr && (pat[k2] as usize) < et {
                k2 += 1;
            }
            pos.clear();
            let mut lo = 0;
            for &r in &pat[k2..] {
                let p = lo + patt[lo..].partition_point(|&x| x < r);
                if p >= patt.len() || patt[p] != r {
                    return false;
                }
                pos.push(p);
                lo = p + 1;
            }
            let zt = unsafe { self.zblock(t) };
            for kk in k..k2 {
                let col = &zt[(pat[kk] as usize - bt) * ldt..][..ldt];
                for m in kk..k2 {
                    let v = col[pat[m] as usize - bt];
                    zrr[kk * nr + m] = v;
                    zrr[m * nr + kk] = v;
                }
                for (mi, &p) in pos.iter().enumerate() {
                    let m = k2 + mi;
                    let v = col[nct + p];
                    zrr[kk * nr + m] = v;
                    zrr[m * nr + kk] = v;
                }
            }
            k = k2;
        }
        true
    }

    /// Computes the block of supernode `s`:
    /// `Y = L21 L11^-1`, `Z_RJ = -Z_RR Y`, `Z_JJ = L11^-T D^+ L11^-1 - Y' Z_RJ`.
    ///
    /// # Safety
    /// Every ancestor of `s` must be finished and nothing else may access block `s`.
    unsafe fn node(&self, s: usize, par: Par) {
        let (b, e) = self.range(s);
        let nc = e - b;
        let pat = self.pattern(s);
        let nr = pat.len();
        let ld = nc + nr;
        let off = self.val_off(s);
        let l = &self.l[off..off + ld * nc];
        let dinv = &self.dinv[b..e];
        let mut zrr = vec![0.0; nr * nr];
        let mut pos = Vec::new();
        if !unsafe { self.gather(pat, &mut zrr, &mut pos) } {
            self.failed.store(true, Ordering::Relaxed);
            return;
        }
        let zs = unsafe { std::slice::from_raw_parts_mut(self.z.0.add(off), ld * nc) };
        if nr * nr * nc + nc * nc * nc + nr * nc * nc < SMALL_NODE_FLOPS {
            node_small(l, dinv, &zrr, zs, nc, nr);
        } else {
            node_dense(l, dinv, &zrr, zs, nc, nr, par);
        }
    }
}

fn node_small(l: &[f64], dinv: &[f64], zrr: &[f64], zs: &mut [f64], nc: usize, nr: usize) {
    let ld = nc + nr;
    let lv = |i: usize, j: usize| l[j * ld + i];
    // Y (nr x nc, column-major): Y L11 = L21.
    let mut y = vec![0.0; nr * nc];
    for r in 0..nr {
        for j in (0..nc).rev() {
            let mut v = lv(nc + r, j);
            for k in j + 1..nc {
                v -= y[k * nr + r] * lv(k, j);
            }
            y[j * nr + r] = v;
        }
    }
    // Z_RJ = -Z_RR Y into the bottom block.
    for j in 0..nc {
        for r in 0..nr {
            let mut v = 0.0;
            for m in 0..nr {
                v += zrr[m * nr + r] * y[j * nr + m];
            }
            zs[j * ld + nc + r] = -v;
        }
    }
    // T = L11^-1 (unit lower, column-major).
    let mut tinv = vec![0.0; nc * nc];
    for j in 0..nc {
        tinv[j * nc + j] = 1.0;
        for i in j + 1..nc {
            let mut v = 0.0;
            for k in j..i {
                v -= lv(i, k) * tinv[j * nc + k];
            }
            tinv[j * nc + i] = v;
        }
    }
    // Z_JJ = T' D^+ T - Y' Z_RJ (lower triangle: the only one read).
    for j in 0..nc {
        for i in j..nc {
            let mut v = 0.0;
            for k in i.max(j)..nc {
                v += tinv[i * nc + k] * dinv[k] * tinv[j * nc + k];
            }
            for r in 0..nr {
                v -= y[i * nr + r] * zs[j * ld + nc + r];
            }
            zs[j * ld + i] = v;
        }
    }
}

fn node_dense(l: &[f64], dinv: &[f64], zrr: &[f64], zs: &mut [f64], nc: usize, nr: usize, par: Par) {
    let ld = nc + nr;
    let lm = MatRef::from_column_major_slice(l, ld, nc);
    let l11 = lm.subrows(0, nc);
    let l21 = lm.subrows(nc, nr);
    let mut y = Mat::<f64>::zeros(nr, nc);
    y.as_mut().copy_from(l21);
    solve_unit_upper_triangular_in_place(l11.transpose(), y.as_mut().transpose_mut(), par);
    let zrr = MatRef::from_column_major_slice(zrr, nr, nr);
    let zm = MatMut::from_column_major_slice_mut(zs, ld, nc);
    let (mut ztop, mut zbot) = zm.split_at_row_mut(nc);
    matmul(zbot.rb_mut(), Accum::Replace, zrr, y.as_ref(), -1.0, par);
    // B = (D^+)^1/2 L11^-1 (D^+ >= 0), Z_JJ = B'B - Y' Z_RJ: one nc x nc temporary, lower
    // triangles only (the only ones read).
    let mut b = Mat::<f64>::zeros(nc, nc);
    invert_unit_lower_triangular(b.as_mut(), l11, par);
    for j in 0..nc {
        b[(j, j)] = 1.0;
    }
    for j in 0..nc {
        for k in j..nc {
            b[(k, j)] *= dinv[k].sqrt();
        }
    }
    tri_matmul(ztop.rb_mut(), BlockStructure::TriangularLower, Accum::Replace, b.transpose(), BlockStructure::TriangularUpper, b.as_ref(), BlockStructure::TriangularLower, 1.0, par);
    tri_matmul(ztop.rb_mut(), BlockStructure::TriangularLower, Accum::Add, y.transpose(), BlockStructure::Rectangular, zbot.rb(), BlockStructure::Rectangular, -1.0, par);
}

fn general(active: &[&FeCols], w: Option<&[f64]>, n: usize, ctx: &Ctx, pool: &Pool, tune: Tuning) -> Result<(Vec<f64>, Info), RfError> {
    let t0 = std::time::Instant::now();
    // Internal column order: terms by decreasing number of levels, so the per-level row lists
    // sorted during assembly are those of the finest term.
    let mut order: Vec<usize> = (0..active.len()).collect();
    order.sort_by_key(|&k| std::cmp::Reverse(active[k].g));
    let mut terms = Vec::with_capacity(active.len());
    let mut ncol = 0usize;
    for &k in &order {
        let t = active[k];
        let b = usize::from(t.intercept) + t.slopes.len();
        terms.push(Term { codes: t.codes, g: t.g, b, off: ncol, intercept: t.intercept, slopes: &t.slopes });
        ncol = ncol.checked_add(t.g * b).ok_or_else(|| RfError::InvalidInput("leverage: too many fixed-effect columns".into()))?;
    }
    if ncol >= u32::MAX as usize {
        return Err(RfError::InvalidInput("leverage: more than 2^32 fixed-effect columns".into()));
    }
    let lower = assemble(&terms, ncol, w, pool, ctx)?;
    ctx.check()?;
    let Lower { col_ptr, rows, mut vals } = lower;
    let mut scale = par::alloc_vec(ncol, 0.0)?;
    for p in 0..ncol {
        let d = vals[col_ptr[p] as usize];
        scale[p] = if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 };
    }
    if tune.predrop {
        for p in component_columns(&terms, w)? {
            scale[p] = 0.0;
        }
    }
    let nnz_a = rows.len();
    let t_assemble = t0.elapsed().as_secs_f64();

    let sym = SymbolicSparseColMatRef::<u32>::new_checked(ncol, ncol, &col_ptr, None, &rows);
    let mut fwd = par::alloc_vec(ncol, 0u32)?;
    let mut inv = par::alloc_vec(ncol, 0u32)?;
    {
        let mut mem = MemBuffer::try_new(amd::order_maybe_unsorted_scratch::<u32>(ncol, nnz_a)).map_err(|_| RfError::OutOfMemory)?;
        amd::order_maybe_unsorted(&mut fwd, &mut inv, sym, amd::Control { dense: tune.amd_dense, aggressive: true }, MemStack::new(&mut mem))
            .map_err(|e| RfError::Internal(format!("leverage: AMD ordering failed: {e:?}")))?;
    }
    let t_amd = t0.elapsed().as_secs_f64();
    let params = CholeskySymbolicParams { supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SUPERNODAL, ..Default::default() };
    let symbolic = pool
        .install(|| factorize_symbolic_cholesky(sym, Side::Lower, SymmetricOrdering::Custom(PermRef::new_checked(&fwd, &inv, ncol)), params))
        .map_err(|e| RfError::Internal(format!("leverage: symbolic factorization failed: {e:?}")))?;
    let len_val = symbolic.len_val();
    if len_val > tune.max_entries {
        return Err(RfError::InvalidInput(format!(
            "exact leverage of the fixed effects needs a Cholesky factor with {len_val} entries, above the memory budget of {} entries ({:.1} GiB for the factor and its selected inverse); HC2/HC3 are not available for this fixed-effects structure",
            tune.max_entries,
            (2 * len_val * 8) as f64 / (1u64 << 30) as f64
        )));
    }
    let SymbolicCholeskyRaw::Supernodal(sn) = symbolic.raw() else {
        return Err(RfError::Internal("leverage: expected a supernodal factorization".into()));
    };
    let ns = sn.n_supernodes();
    let flops = supernodal_flops(sn);
    let t_symbolic = t0.elapsed().as_secs_f64();
    ctx.check()?;

    // Per-column pivot tolerance PIVOT_TOL * 4^k >= PIVOT_ROUNDING * m_j (m_j = updates summed
    // into pivot j = off-diagonal entries of row j of L), applied by scaling column j by 2^-k:
    // powers of two leave every rounding of the factorization unchanged, so the single faer
    // threshold compares pivot_j * 4^-k with PIVOT_TOL.
    {
        let mut updates = vec![0u64; ncol];
        for s in 0..ns {
            let (b, e) = (sn.supernode_begin()[s] as usize, sn.supernode_end()[s] as usize);
            for (k, u) in updates[b..e].iter_mut().enumerate() {
                *u += k as u64;
            }
            for &r in sn.supernode(s).pattern() {
                updates[r as usize] += (e - b) as u64;
            }
        }
        for (j, &m) in updates.iter().enumerate() {
            let mut k = 0;
            while tune.pivot_tol * 4f64.powi(k) < PIVOT_ROUNDING * m as f64 {
                k += 1;
            }
            scale[fwd[j] as usize] *= 0.5f64.powi(k);
        }
    }
    for p in 0..ncol {
        for k in col_ptr[p] as usize..col_ptr[p + 1] as usize {
            vals[k] *= scale[p] * scale[rows[k] as usize];
        }
    }

    let mut lval = par::alloc_vec(len_val, 0.0)?;
    {
        let signs = vec![1i8; ncol];
        let reg = LdltRegularization { dynamic_regularization_signs: Some(&signs), dynamic_regularization_delta: DROP, dynamic_regularization_epsilon: tune.pivot_tol };
        let fpar = if pool.threads() > 1 && flops >= tune.par_factor_flops { Par::rayon(pool.threads()) } else { Par::Seq };
        let ok = pool.install(|| {
            let a = SparseColMatRef::new(sym, &vals);
            let mut buf = MemBuffer::new(symbolic.factorize_numeric_ldlt_scratch::<f64>(fpar, Default::default()));
            symbolic.factorize_numeric_ldlt(&mut lval, a, Side::Lower, reg, fpar, MemStack::new(&mut buf), Default::default()).is_ok()
        });
        if !ok {
            return Err(RfError::Internal("leverage: numeric factorization failed".into()));
        }
    }
    drop(vals);
    drop(rows);
    let t_numeric = t0.elapsed().as_secs_f64();
    ctx.check()?;

    // Dropped pivots: zero their column of L, D^+ = 0.
    let mut dinv = par::alloc_vec(ncol, 0.0)?;
    let mut col_super = par::alloc_vec(ncol, 0u32)?;
    let mut dropped = 0usize;
    for s in 0..ns {
        let (b, e) = (sn.supernode_begin()[s] as usize, sn.supernode_end()[s] as usize);
        let nc = e - b;
        let ld = nc + sn.supernode(s).pattern().len();
        let off = sn.col_ptr_for_val()[s] as usize;
        for j in 0..nc {
            col_super[b + j] = s as u32;
            let col = &mut lval[off + j * ld..off + (j + 1) * ld];
            let d = col[j];
            if d >= 0.5 * DROP {
                dropped += 1;
                col[j + 1..].fill(0.0);
            } else {
                dinv[b + j] = 1.0 / d;
            }
        }
    }
    // Supernodal elimination tree: the parent holds the first below-diagonal row.
    let mut parent = vec![u32::MAX; ns];
    for (s, ps) in parent.iter_mut().enumerate() {
        if let Some(&r) = sn.supernode(s).pattern().first() {
            let p = col_super[r as usize];
            if p as usize <= s {
                return Err(RfError::Internal("leverage: supernodes are not in postorder".into()));
            }
            *ps = p;
        }
    }

    // Selected inversion. Supernodes of order >= big_front, and their ancestors, are processed
    // one at a time from the root down (parents have larger indices) with parallel dense
    // kernels; the remaining subtrees then run concurrently.
    let mut zval = par::alloc_vec(len_val, 0.0)?;
    let threads = pool.threads();
    let mut big = vec![false; ns];
    for s in 0..ns {
        let (b, e) = (sn.supernode_begin()[s] as usize, sn.supernode_end()[s] as usize);
        if e - b + sn.supernode(s).pattern().len() >= tune.big_front || big[s] {
            big[s] = true;
            let mut p = parent[s];
            while p != u32::MAX && !big[p as usize] {
                big[p as usize] = true;
                p = parent[p as usize];
            }
        }
    }
    let mut child_ptr = vec![0u32; ns + 1];
    for s in 0..ns {
        if parent[s] != u32::MAX && !big[s] {
            child_ptr[parent[s] as usize + 1] += 1;
        }
    }
    for s in 0..ns {
        child_ptr[s + 1] += child_ptr[s];
    }
    let mut child = vec![0u32; child_ptr[ns] as usize];
    let mut fill = child_ptr.clone();
    let mut roots = Vec::new();
    for s in 0..ns {
        if big[s] {
            continue;
        }
        if parent[s] == u32::MAX || big[parent[s] as usize] {
            roots.push(s as u32);
        } else {
            let p = parent[s] as usize;
            child[fill[p] as usize] = s as u32;
            fill[p] += 1;
        }
    }
    let si = Selinv { sn, l: &lval, dinv: &dinv, z: SyncPtr(zval.as_mut_ptr()), col_super: &col_super, failed: AtomicBool::new(false) };
    let dpar = if threads > 1 { Par::rayon(threads) } else { Par::Seq };
    for s in (0..ns).rev() {
        if big[s] {
            // SAFETY: parents have larger indices and are finished; nothing runs concurrently.
            pool.install(|| unsafe { si.node(s, dpar) });
        }
    }
    ctx.check()?;
    {
        let (si, roots) = (&si, &roots[..]);
        let (child_ptr, child) = (&child_ptr[..], &child[..]);
        pool.install(|| rayon::scope(|sc| spawn_subtrees(sc, roots, si, child_ptr, child)));
    }
    if si.failed.load(Ordering::Relaxed) {
        return Err(RfError::Internal("leverage: selected-inverse pattern lookup failed".into()));
    }
    let t_selinv = t0.elapsed().as_secs_f64();
    ctx.check()?;

    // h_i = w_i sum_{p,q} u_p u_q Z_pq (scaled, permuted coordinates).
    let map = |p: usize| inv[p];
    let cmax: usize = terms.iter().map(|t| t.b).sum();
    let lookup = |p: usize, q: usize| -> Option<f64> {
        let (lo, hi) = if p < q { (p, q) } else { (q, p) };
        let t = col_super[lo] as usize;
        let (b, e) = (sn.supernode_begin()[t] as usize, sn.supernode_end()[t] as usize);
        let pat = sn.supernode(t).pattern();
        let nc = e - b;
        let base = sn.col_ptr_for_val()[t] as usize + (lo - b) * (nc + pat.len());
        if hi < e {
            Some(zval[base + hi - b])
        } else {
            pat.binary_search(&(hi as u32)).ok().map(|k| zval[base + nc + k])
        }
    };
    let mut h = par::alloc_vec(n, 0.0)?;
    let ok = pool.install(|| {
        h.par_chunks_mut(CHUNK).enumerate().all(|(c, out)| {
            let mut cols: Vec<(usize, f64)> = Vec::with_capacity(cmax);
            for (k, o) in out.iter_mut().enumerate() {
                let i = c * CHUNK + k;
                let wi = w.map_or(1.0, |w| w[i]);
                if wi == 0.0 {
                    *o = 0.0;
                    continue;
                }
                cols.clear();
                for t in &terms {
                    let base = t.off + t.codes[i] as usize * t.b;
                    for s in 0..t.b {
                        let sc = scale[base + s];
                        if sc != 0.0 {
                            cols.push((map(base + s) as usize, t.val(s, i) * sc));
                        }
                    }
                }
                let mut acc = 0.0;
                for a in 0..cols.len() {
                    let (pa, ua) = cols[a];
                    let Some(zaa) = lookup(pa, pa) else { return false };
                    acc += ua * ua * zaa;
                    for &(pb, ub) in &cols[a + 1..] {
                        let Some(zab) = lookup(pa, pb) else { return false };
                        acc += 2.0 * ua * ub * zab;
                    }
                }
                *o = wi * acc;
            }
            true
        })
    });
    if !ok {
        return Err(RfError::Internal("leverage: column pair outside the factor pattern".into()));
    }
    let info = Info { columns: ncol, nnz_a, factor_entries: len_val, dropped };
    if ctx.verbose >= 2 {
        ctx.log(&format!(
            "leverage: {} FE columns, {} entries in A, {} in L ({ns} supernodes, {flops:.3e} flops), {} dependent columns; assemble {t_assemble:.3}s amd {:.3}s symbolic {:.3}s numeric {:.3}s selinv {:.3}s diag {:.3}s",
            info.columns,
            info.nnz_a,
            info.factor_entries,
            info.dropped,
            t_amd - t_assemble,
            t_symbolic - t_amd,
            t_numeric - t_symbolic,
            t_selinv - t_numeric,
            t0.elapsed().as_secs_f64() - t_selinv
        ));
    }
    Ok((h, info))
}

/// Supernodes per spawned task of [`visit`]: panels have one leaf supernode per unit, and a
/// task per leaf costs more than the leaf.
const SPAWN_GROUP: usize = 64;

/// Spawns the subtrees rooted at `nodes`, [`SPAWN_GROUP`] roots per task.
fn spawn_subtrees<'s>(sc: &rayon::Scope<'s>, nodes: &'s [u32], si: &'s Selinv<'s>, child_ptr: &'s [u32], child: &'s [u32]) {
    for group in nodes.chunks(SPAWN_GROUP) {
        sc.spawn(move |sc| {
            for &c in group {
                visit(sc, c as usize, si, child_ptr, child);
            }
        });
    }
}

/// A supernode, then its subtree: a single child is continued in this task (chains do not
/// grow the stack), several are spawned in groups.
fn visit<'s>(sc: &rayon::Scope<'s>, mut s: usize, si: &'s Selinv<'s>, child_ptr: &'s [u32], child: &'s [u32]) {
    loop {
        // SAFETY: the parent finished before this task was spawned or continued.
        unsafe { si.node(s, Par::Seq) };
        let ch = &child[child_ptr[s] as usize..child_ptr[s + 1] as usize];
        match ch {
            [] => return,
            [only] => s = *only as usize,
            _ => return spawn_subtrees(sc, ch, si, child_ptr, child),
        }
    }
}

/// Multiply-adds of the supernodal factorization (as `precond::sparse_chol::SparseChol::flops`).
fn supernodal_flops(sn: &SymbolicSupernodalCholesky<u32>) -> f64 {
    let sq = |x: f64| x * (x + 1.0) * (2.0 * x + 1.0) / 6.0;
    (0..sn.n_supernodes())
        .map(|s| {
            let ncols = (sn.supernode_end()[s] - sn.supernode_begin()[s]) as f64;
            let below = sn.supernode(s).pattern().len() as f64;
            sq(ncols + below) - sq(below)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::par::with_pool;
    use crate::rng::SplitMix64;
    use crate::vcov::sym_eigen;

    /// Owned test data for one FE structure.
    struct Data {
        n: usize,
        codes: Vec<Vec<u32>>,
        g: Vec<usize>,
        intercept: Vec<bool>,
        slopes: Vec<Vec<Vec<f64>>>,
        w: Option<Vec<f64>>,
    }

    impl Data {
        fn new(n: usize) -> Data {
            Data { n, codes: Vec::new(), g: Vec::new(), intercept: Vec::new(), slopes: Vec::new(), w: None }
        }

        fn term(mut self, codes: Vec<u32>, intercept: bool, slopes: Vec<Vec<f64>>) -> Data {
            assert_eq!(codes.len(), self.n);
            self.g.push(codes.iter().map(|&c| c as usize + 1).max().unwrap_or(0));
            self.codes.push(codes);
            self.intercept.push(intercept);
            self.slopes.push(slopes);
            self
        }

        fn weights(mut self, w: Vec<f64>) -> Data {
            self.w = Some(w);
            self
        }

        fn cols(&self) -> Vec<FeCols<'_>> {
            (0..self.codes.len())
                .map(|t| FeCols { codes: &self.codes[t], g: self.g[t], intercept: self.intercept[t], slopes: self.slopes[t].iter().map(|s| s.as_slice()).collect() })
                .collect()
        }

        fn wi(&self, i: usize) -> f64 {
            self.w.as_ref().map_or(1.0, |w| w[i])
        }
    }

    fn uniform_codes(rng: &mut SplitMix64, n: usize, g: usize) -> Vec<u32> {
        (0..n).map(|_| (rng.next_u64() % g as u64) as u32).collect()
    }

    fn normals(rng: &mut SplitMix64, n: usize, shift: f64) -> Vec<f64> {
        (0..n).map(|_| shift + (rng.next_f64() - 0.5) * 3.0 + (rng.next_f64() - 0.5)).collect()
    }

    /// Dense oracle: explicit columns D, A = D'WD scaled to unit diagonal, pseudo-inverse by
    /// symmetric (Jacobi) eigendecomposition with cutoff 1e-11 * lambda_max.
    struct Oracle {
        cols: Vec<Vec<f64>>,
        scale: Vec<f64>,
        pinv: Vec<f64>,
        rank: usize,
    }

    impl Oracle {
        fn new(d: &Data) -> Oracle {
            let n = d.n;
            let mut cols: Vec<Vec<f64>> = Vec::new();
            for t in 0..d.codes.len() {
                for l in 0..d.g[t] {
                    let ind = |i: usize| d.codes[t][i] as usize == l;
                    if d.intercept[t] {
                        cols.push((0..n).map(|i| if ind(i) { 1.0 } else { 0.0 }).collect());
                    }
                    for s in &d.slopes[t] {
                        cols.push((0..n).map(|i| if ind(i) { s[i] } else { 0.0 }).collect());
                    }
                }
            }
            let p = cols.len();
            let mut a = vec![0.0; p * p];
            for j in 0..p {
                for k in 0..=j {
                    let v: f64 = (0..n).map(|i| d.wi(i) * cols[j][i] * cols[k][i]).sum();
                    a[j * p + k] = v;
                    a[k * p + j] = v;
                }
            }
            let scale: Vec<f64> = (0..p).map(|j| if a[j * p + j] > 0.0 { 1.0 / a[j * p + j].sqrt() } else { 0.0 }).collect();
            for j in 0..p {
                for k in 0..p {
                    a[j * p + k] *= scale[j] * scale[k];
                }
            }
            let (ev, vecs) = sym_eigen(&a, p);
            let lmax = ev.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
            let mut pinv = vec![0.0; p * p];
            let mut rank = 0;
            for e in 0..p {
                if ev[e] > 1e-11 * lmax {
                    rank += 1;
                    for j in 0..p {
                        for k in 0..p {
                            pinv[j * p + k] += vecs[j * p + e] * vecs[k * p + e] / ev[e];
                        }
                    }
                }
            }
            Oracle { cols, scale, pinv, rank }
        }

        fn hat(&self, d: &Data) -> Vec<f64> {
            let p = self.cols.len();
            (0..d.n)
                .map(|i| {
                    let u: Vec<f64> = (0..p).map(|j| self.cols[j][i] * self.scale[j]).collect();
                    let mut acc = 0.0;
                    for j in 0..p {
                        if u[j] != 0.0 {
                            for k in 0..p {
                                acc += u[j] * self.pinv[j * p + k] * u[k];
                            }
                        }
                    }
                    d.wi(i) * acc
                })
                .collect()
        }

        /// x minus its weighted projection on every FE column.
        fn residualize(&self, d: &Data, x: &[f64]) -> Vec<f64> {
            let p = self.cols.len();
            let b: Vec<f64> = (0..p).map(|j| self.scale[j] * (0..d.n).map(|i| d.wi(i) * self.cols[j][i] * x[i]).sum::<f64>()).collect();
            let c: Vec<f64> = (0..p).map(|j| self.scale[j] * (0..p).map(|k| self.pinv[j * p + k] * b[k]).sum::<f64>()).collect();
            (0..d.n).map(|i| x[i] - (0..p).map(|j| self.cols[j][i] * c[j]).sum::<f64>()).collect()
        }

        fn deficiency(&self) -> usize {
            self.cols.len() - self.rank
        }
    }

    fn run(d: &Data, threads: usize, tune: Tuning) -> Result<(Vec<f64>, Info), RfError> {
        let cols = d.cols();
        with_pool(threads, |pool| fe_hat_with(&cols, d.w.as_deref(), d.n, &Ctx::quiet(threads), pool, tune)).unwrap()
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// fe_hat vs the dense oracle at 1e-10 absolute, and the number of dropped columns equals
    /// the rank deficiency (general path only).
    fn check(d: &Data) -> (Vec<f64>, Info) {
        check_with(d, 1e-10, 2, TUNING)
    }

    fn check_with(d: &Data, tol: f64, threads: usize, tune: Tuning) -> (Vec<f64>, Info) {
        let o = Oracle::new(d);
        let want = o.hat(d);
        let (h, info) = run(d, threads, tune).unwrap();
        let diff = max_abs_diff(&h, &want);
        eprintln!("max |h - oracle| = {diff:e} (dropped {})", info.dropped);
        assert!(diff <= tol, "max |h - oracle| = {diff:e}");
        // Without the exact component pre-drop, the pivot tolerance alone finds the same rank.
        let (h2, info2) = run(d, threads, Tuning { predrop: false, ..tune }).unwrap();
        let diff2 = max_abs_diff(&h2, &want);
        assert!(diff2 <= tol, "without pre-drop: max |h - oracle| = {diff2:e}");
        assert_eq!(info2.dropped, info.dropped);
        if info.columns > 0 {
            assert_eq!(info.dropped, o.deficiency(), "dropped columns vs oracle rank deficiency");
        }
        (h, info)
    }

    #[test]
    fn one_fe_closed_form() {
        let mut rng = SplitMix64::new(1);
        let n = 120;
        let mut f = uniform_codes(&mut rng, n, 9);
        f[7] = 9; // singleton level
        let w: Vec<f64> = (0..n).map(|_| 0.2 + rng.next_f64()).collect();
        let d = Data::new(n).term(f, true, vec![]).weights(w);
        let (h, _) = check(&d);
        assert!((h[7] - 1.0).abs() < 1e-14);
        let total: f64 = h.iter().sum();
        assert!((total - 10.0).abs() < 1e-12, "trace = number of levels");
        // The crate entry point, and an empty FE set.
        let cols = d.cols();
        let h2 = with_pool(3, |pool| fe_hat(&cols, d.w.as_deref(), n, &Ctx::quiet(3), pool)).unwrap().unwrap();
        assert_eq!(h, h2);
        let none = with_pool(1, |pool| fe_hat(&[], None, n, &Ctx::quiet(1), pool)).unwrap().unwrap();
        assert!(none.iter().all(|&v| v == 0.0) && none.len() == n);
    }

    #[test]
    fn two_fe_components_and_singleton() {
        let mut rng = SplitMix64::new(2);
        let n = 400;
        // Three components: f1 blocks [0,15), [15,30), [30,40) with their own f2 levels.
        let mut f1 = uniform_codes(&mut rng, n, 40);
        let f2: Vec<u32> = f1
            .iter()
            .map(|&a| {
                let r = (rng.next_u64() % 5) as u32;
                if a < 15 { r } else if a < 30 { 5 + r } else { 10 + r % 3 }
            })
            .collect();
        f1[11] = 40; // singleton level of f1, joined to an existing f2 level
        let w: Vec<f64> = (0..n).map(|_| 0.5 + 1.5 * rng.next_f64()).collect();
        let d = Data::new(n).term(f1, true, vec![]).term(f2, true, vec![]).weights(w);
        let (h, info) = check(&d);
        assert_eq!(info.dropped, 3);
        assert!((h[11] - 1.0).abs() < 1e-12, "singleton h = {}", h[11]);
    }

    #[test]
    fn three_fe_weighted() {
        let mut rng = SplitMix64::new(3);
        let n = 500;
        let f1 = uniform_codes(&mut rng, n, 45);
        let f2 = uniform_codes(&mut rng, n, 12);
        let f3 = uniform_codes(&mut rng, n, 6);
        let w: Vec<f64> = (0..n).map(|_| 0.1 + 3.0 * rng.next_f64()).collect();
        let d = Data::new(n).term(f1, true, vec![]).term(f2, true, vec![]).term(f3, true, vec![]).weights(w);
        let (_, info) = check(&d);
        assert_eq!(info.dropped, 2);
    }

    #[test]
    fn nested_and_duplicated_terms() {
        // f2 = f1 relabeled (exact duplicate columns), f3 = f1 / 4 (f1 nested in f3): every
        // pivot of the redundant columns is a rounded zero.
        let mut rng = SplitMix64::new(4);
        let n = 300;
        let f1 = uniform_codes(&mut rng, n, 24);
        let f2: Vec<u32> = f1.iter().map(|&a| (a * 7) % 24).collect();
        let f3: Vec<u32> = f1.iter().map(|&a| a / 4).collect();
        let f4 = uniform_codes(&mut rng, n, 5);
        let w: Vec<f64> = (0..n).map(|_| 0.3 + rng.next_f64()).collect();
        let d = Data::new(n).term(f1, true, vec![]).term(f2, true, vec![]).term(f3, true, vec![]).term(f4, true, vec![]).weights(w);
        let (_, info) = check(&d);
        assert_eq!(info.dropped, 24 + 6 + 1);
    }

    #[test]
    fn slopes_with_degenerate_levels() {
        // f1 + f2[x] + f3[[x2]]: x constant within f2 levels 0 and 1 (slope column equal to its
        // intercept column times a constant), x2 = 0 on f3 level 2 (zero column), and a level of
        // f2 with a single observation.
        let mut rng = SplitMix64::new(5);
        let n = 360;
        let f1 = uniform_codes(&mut rng, n, 30);
        let mut f2 = uniform_codes(&mut rng, n, 8);
        f2[5] = 8;
        let f3 = uniform_codes(&mut rng, n, 5);
        let mut x = normals(&mut rng, n, 2.0);
        let mut x2 = normals(&mut rng, n, -1.0);
        for i in 0..n {
            match f2[i] {
                0 => x[i] = 3.5,
                1 => x[i] = -0.25,
                _ => {}
            }
            if f3[i] == 2 {
                x2[i] = 0.0;
            }
        }
        let w: Vec<f64> = (0..n).map(|_| 0.5 + rng.next_f64()).collect();
        let d = Data::new(n).term(f1.clone(), true, vec![]).term(f2.clone(), true, vec![x.clone()]).term(f3.clone(), false, vec![x2.clone()]).weights(w.clone());
        check(&d);
        // The same without weights, and a single term with intercept and two slopes.
        let d = Data::new(n).term(f1, true, vec![]).term(f2.clone(), true, vec![x.clone()]).term(f3, false, vec![x2.clone()]);
        check(&d);
        let d = Data::new(n).term(f2.clone(), true, vec![x.clone(), x2.clone()]).weights(w.clone());
        check(&d);
        // Slope-only terms only (fe[[x]] alone, and two of them).
        let d = Data::new(n).term(f2.clone(), false, vec![x.clone()]).weights(w);
        check(&d);
        let f4 = uniform_codes(&mut rng, n, 7);
        let d = Data::new(n).term(f2, false, vec![x]).term(f4, false, vec![x2]);
        check(&d);
    }

    #[test]
    fn weak_bridge_is_kept() {
        // Two groups joined by one observation of weight 1e-4: a legitimately small pivot. The
        // bridge row's leverage is exactly 1 (it is the only link between the groups).
        let mut rng = SplitMix64::new(6);
        let n = 200;
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        for i in 0..n {
            let grp = (i % 2) as u32;
            f1.push(grp * 5 + (rng.next_u64() % 5) as u32);
            f2.push(grp * 3 + (rng.next_u64() % 3) as u32);
        }
        let mut w: Vec<f64> = (0..n).map(|_| 0.5 + rng.next_f64()).collect();
        f1[0] = 0;
        f2[0] = 3;
        w[0] = 1e-4;
        let d = Data::new(n).term(f1, true, vec![]).term(f2, true, vec![]).weights(w);
        // The bridge pivot is ~1e-5 on the unit-diagonal scale, so both fe_hat and the oracle
        // lose ~5 digits on the bridge row (rounding / pivot).
        let (h, info) = check_with(&d, 1e-9, 2, TUNING);
        assert_eq!(info.dropped, 1);
        assert!((h[0] - 1.0).abs() < 1e-9, "bridge h = {}", h[0]);
    }

    #[test]
    fn dense_fronts_match_oracle() {
        // A few hundred columns: the trailing front is processed by the dense (faer) kernels,
        // and with a small big_front by the sequential top-down pass as well.
        let mut rng = SplitMix64::new(10);
        let n = 1500;
        let f1 = uniform_codes(&mut rng, n, 150);
        let f2 = uniform_codes(&mut rng, n, 40);
        let f3 = uniform_codes(&mut rng, n, 6);
        let x = normals(&mut rng, n, 1.5);
        let w: Vec<f64> = (0..n).map(|_| 0.5 + rng.next_f64()).collect();
        let d = Data::new(n).term(f1, true, vec![]).term(f2, true, vec![x]).term(f3, true, vec![]).weights(w);
        let o = Oracle::new(&d);
        let want = o.hat(&d);
        for (threads, tune) in [(4, TUNING), (4, Tuning { par_factor_flops: 0.0, big_front: 16, ..TUNING }), (1, Tuning { big_front: 16, ..TUNING })] {
            let (h, info) = run(&d, threads, tune).unwrap();
            let diff = max_abs_diff(&h, &want);
            eprintln!("dense fronts: threads {threads}: max |h - oracle| = {diff:e}");
            assert!(diff <= 1e-10, "max |h - oracle| = {diff:e}");
            assert_eq!(info.dropped, o.deficiency());
        }
    }

    #[test]
    fn zero_weight_rows() {
        let mut rng = SplitMix64::new(7);
        let n = 200;
        let f1 = uniform_codes(&mut rng, n, 20);
        let f2 = uniform_codes(&mut rng, n, 6);
        let mut w: Vec<f64> = (0..n).map(|_| 0.5 + rng.next_f64()).collect();
        for i in (0..n).step_by(9) {
            w[i] = 0.0;
        }
        let d = Data::new(n).term(f1, true, vec![]).term(f2, true, vec![]).weights(w);
        let (h, _) = check(&d);
        assert!(h.iter().step_by(9).all(|&v| v == 0.0));
    }

    #[test]
    fn bit_identical_across_threads() {
        let mut rng = SplitMix64::new(8);
        let n = 40_000;
        let f1 = uniform_codes(&mut rng, n, 4000);
        let f2 = uniform_codes(&mut rng, n, 300);
        let f3 = uniform_codes(&mut rng, n, 20);
        let x = normals(&mut rng, n, 1.0);
        let w: Vec<f64> = (0..n).map(|_| 0.5 + rng.next_f64()).collect();
        let d = Data::new(n).term(f1, true, vec![]).term(f2, true, vec![x]).term(f3, true, vec![]).weights(w);
        // Forced parallel factorization and parallel dense kernels on small fronts.
        let forced = Tuning { par_factor_flops: 0.0, big_front: 64, ..TUNING };
        let (h1, _) = run(&d, 1, TUNING).unwrap();
        for (threads, tune) in [(1, forced), (3, forced), (8, forced), (8, TUNING)] {
            let (h, _) = run(&d, threads, tune).unwrap();
            assert!(h.iter().zip(&h1).all(|(a, b)| a.to_bits() == b.to_bits()), "threads {threads} differ");
        }
        // Leverages sum to the rank of the FE columns.
        let (h, info) = run(&d, 4, TUNING).unwrap();
        let rank = (info.columns - info.dropped) as f64;
        let sum: f64 = h.iter().sum();
        assert!((sum - rank).abs() < 1e-7 * rank, "sum h = {sum}, rank = {rank}");
    }

    #[test]
    fn budget_is_an_error() {
        let mut rng = SplitMix64::new(9);
        let n = 100;
        let d = Data::new(n).term(uniform_codes(&mut rng, n, 10), true, vec![]).term(uniform_codes(&mut rng, n, 4), true, vec![]);
        let err = run(&d, 1, Tuning { max_entries: 5, ..TUNING }).unwrap_err();
        assert!(matches!(&err, RfError::InvalidInput(m) if m.contains("memory budget")), "{err:?}");
    }

    /// Fixture rows: f1, f2, f3 (1-based dense), x, x2, w, h (fixest hatvalues, exact = TRUE).
    fn fixture(name: &str) -> (Vec<[u32; 3]>, Vec<[f64; 4]>) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/leverage").join(format!("fixest_{name}.csv"));
        let text = std::fs::read_to_string(&path).unwrap();
        let mut fs = Vec::new();
        let mut vs = Vec::new();
        for line in text.lines().skip(1) {
            let f: Vec<&str> = line.split(',').collect();
            fs.push([f[0].parse::<u32>().unwrap() - 1, f[1].parse::<u32>().unwrap() - 1, f[2].parse::<u32>().unwrap() - 1]);
            vs.push([f[3].parse().unwrap(), f[4].parse().unwrap(), f[5].parse().unwrap(), f[6].parse().unwrap()]);
        }
        (fs, vs)
    }

    /// fe_hat + w x~^2 / sum(w x~^2) (x~ = x residualized on every FE column) vs fixest.
    fn check_fixest(name: &str, build: impl Fn(Data, &[Vec<u32>], &[f64]) -> Data, weighted: bool) {
        let (fs, vs) = fixture(name);
        let n = fs.len();
        let f: Vec<Vec<u32>> = (0..3).map(|k| fs.iter().map(|r| r[k]).collect()).collect();
        let x: Vec<f64> = vs.iter().map(|r| r[0]).collect();
        let x2: Vec<f64> = vs.iter().map(|r| r[1]).collect();
        let mut d = build(Data::new(n), &f, &x2);
        if weighted {
            d = d.weights(vs.iter().map(|r| r[2]).collect());
        }
        let (h, _) = run(&d, 2, TUNING).unwrap();
        let o = Oracle::new(&d);
        let oracle = max_abs_diff(&h, &o.hat(&d));
        assert!(oracle <= 1e-10, "{name}: max |h - oracle| = {oracle:e}");
        let xt = o.residualize(&d, &x);
        let sxx: f64 = (0..n).map(|i| d.wi(i) * xt[i] * xt[i]).sum();
        let diff = (0..n).map(|i| (h[i] + d.wi(i) * xt[i] * xt[i] / sxx - vs[i][3]).abs()).fold(0.0, f64::max);
        eprintln!("{name}: max |h - fixest| = {diff:e}, max |fe_hat - oracle| = {oracle:e}");
        assert!(diff <= 1e-10, "{name}: max |h - fixest| = {diff:e}");
    }

    #[test]
    fn fixest_hatvalues_2fe_components() {
        check_fixest("2fe_w_disc", |d, f, _| d.term(f[0].clone(), true, vec![]).term(f[1].clone(), true, vec![]), true);
    }

    #[test]
    fn fixest_hatvalues_3fe() {
        check_fixest("3fe_w", |d, f, _| d.term(f[0].clone(), true, vec![]).term(f[1].clone(), true, vec![]).term(f[2].clone(), true, vec![]), true);
    }

    #[test]
    fn fixest_hatvalues_varying_slope() {
        check_fixest("slope_w", |d, f, x2| d.term(f[0].clone(), true, vec![]).term(f[1].clone(), true, vec![x2.to_vec()]), true);
        check_fixest("slope1", |d, f, x2| d.term(f[0].clone(), true, vec![x2.to_vec()]), false);
    }

    fn densify(v: &mut [u32]) -> usize {
        let mut u: Vec<u32> = v.to_vec();
        u.sort_unstable();
        u.dedup();
        for c in v.iter_mut() {
            *c = u.binary_search(c).unwrap() as u32;
        }
        u.len()
    }

    fn proc_status(key: &str) -> String {
        std::fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines().find(|l| l.starts_with(key)).map(|l| l.trim().to_string())).unwrap_or_default()
    }

    /// Timing and memory on large graphs (one case per process for a meaningful VmHWM):
    /// `LEV_CASE=panel1m|akm1m|panel10m LEV_THREADS=16 [LEV_SWEEP=1] cargo test --release -p
    /// fereg-core --lib leverage::tests::perf_fe_hat -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn perf_fe_hat() {
        let case = std::env::var("LEV_CASE").unwrap_or_else(|_| "panel1m".into());
        let threads: usize = std::env::var("LEV_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
        let mut rng = SplitMix64::new(42);
        let mut d = match case.as_str() {
            // 1e6 rows, 1e5 units x 10 observations, time drawn from 1000 periods.
            "panel1m" => {
                let n = 1_000_000;
                let unit: Vec<u32> = (0..n as u32).map(|i| i / 10).collect();
                let mut time = uniform_codes(&mut rng, n, 1000);
                densify(&mut time);
                Data::new(n).term(unit, true, vec![]).term(time, true, vec![])
            }
            // AKM: 1e5 workers x 10 years, 5e4 firms of skewed size, move probability 0.1;
            // worker + firm + year.
            "akm1m" => {
                let (workers, years, firms) = (100_000usize, 10usize, 50_000f64);
                let n = workers * years;
                let draw = |rng: &mut SplitMix64| (firms * rng.next_f64().powi(2)) as u32;
                let mut firm = Vec::with_capacity(n);
                for _ in 0..workers {
                    let mut f = draw(&mut rng);
                    for _ in 0..years {
                        if rng.next_f64() < 0.1 {
                            f = draw(&mut rng);
                        }
                        firm.push(f);
                    }
                }
                densify(&mut firm);
                let worker: Vec<u32> = (0..n as u32).map(|i| i / years as u32).collect();
                let year: Vec<u32> = (0..n as u32).map(|i| i % years as u32).collect();
                Data::new(n).term(worker, true, vec![]).term(firm, true, vec![]).term(year, true, vec![])
            }
            // 1e7 rows: 1e6 units x 10 periods, balanced.
            "panel10m" => {
                let n = 10_000_000;
                Data::new(n).term((0..n as u32).map(|i| i / 10).collect(), true, vec![]).term((0..n as u32).map(|i| i % 10).collect(), true, vec![])
            }
            other => panic!("unknown LEV_CASE {other}"),
        };
        if std::env::var("LEV_WEIGHTS").is_ok() {
            d.w = Some((0..d.n).map(|_| 0.5 + rng.next_f64()).collect());
        }
        let rss_before = proc_status("VmRSS");
        let cols = d.cols();
        let log = |s: &str| eprintln!("{s}");
        let ctx = Ctx { threads, verbose: 2, interrupt: &|| false, log: &log };
        let amd_dense = std::env::var("LEV_AMD_DENSE").ok().and_then(|s| s.parse().ok()).unwrap_or(AMD_DENSE);
        let tune = Tuning { predrop: std::env::var("LEV_NO_PREDROP").is_err(), amd_dense, ..TUNING };
        let t = std::time::Instant::now();
        let (h, info) = with_pool(threads, |pool| fe_hat_with(&cols, d.w.as_deref(), d.n, &ctx, pool, tune)).unwrap().unwrap();
        let secs = t.elapsed().as_secs_f64();
        // Compensated (Neumaier) sum: trace = rank.
        let (mut sum, mut comp) = (0.0f64, 0.0f64);
        for &v in &h {
            let t = sum + v;
            comp += if sum.abs() >= v.abs() { (sum - t) + v } else { (v - t) + sum };
            sum = t;
        }
        let sum = sum + comp;
        if case == "panel10m" {
            // Balanced two-way panel: h = 1/T + 1/N - 1/(NT) for every row.
            let exact = 1.0 / 10.0 + 1.0 / 1e6 - 1.0 / 1e7;
            eprintln!("panel10m: max |h - closed form| = {:e}", h.iter().map(|v| (v - exact).abs()).fold(0.0, f64::max));
        }
        eprintln!(
            "PERF case={case} threads={threads} predrop={} n={} secs={secs:.3} cols={} nnzA={} factor={} dropped={} sum_h={sum:.6} rank={} | before: {rss_before} | after: {} {}",
            tune.predrop,
            d.n,
            info.columns,
            info.nnz_a,
            info.factor_entries,
            info.dropped,
            info.columns - info.dropped,
            proc_status("VmRSS"),
            proc_status("VmHWM")
        );
        if std::env::var("LEV_SWEEP").is_ok() {
            for tol in [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4] {
                let r = with_pool(threads, |pool| fe_hat_with(&cols, d.w.as_deref(), d.n, &Ctx::quiet(threads), pool, Tuning { pivot_tol: tol, ..tune })).unwrap().unwrap();
                eprintln!("SWEEP case={case} pivot_tol={tol:e} dropped={} max|dh|={:e}", r.1.dropped, max_abs_diff(&r.0, &h));
            }
        }
    }
}
