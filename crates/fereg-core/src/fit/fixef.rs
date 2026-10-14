//! Fixed-effect values of the fit layer: fixest's normalization (`fixef()`, with
//! `attr(, "references")`, also used by `ssc(K.exact = TRUE)`) and pyfixest's (`fixef()`).
//!
//! A fit keeps the per-level values of its fixed effects in the solver's normalization
//! ([`FeLevels`], cheap: one value per level and slot); [`fixef_values`] renormalizes them on
//! demand ([`super::FitState::fixef`]) or at fit time (`FitSpec::keep_fixef`).

use std::sync::atomic::{AtomicU32, Ordering};

use rayon::prelude::*;

use super::state::StateFe;
use super::{FixefNorm, FixefValues};
use crate::factor::{self, Codes};
use crate::error::RfError;
use crate::par::{CHUNK, Pool};

/// Fixed-effect values retained by a fit.
pub(crate) struct FeLevels {
    /// Per term slot (term order; each term's intercept slot, then one per slope) and level of
    /// the solved sample: a solution of the fixed-effect part in the solver's normalization.
    pub slots: Vec<Vec<f64>>,
    /// Set when the solved sample is a strict subset of the reported one.
    pub subset: Option<FeSubset>,
}

/// Pyfixest-flavor logit: IRLS ran without the rows of the levels whose outcomes are all 0 or
/// all 1 (their linear predictor diverges), the reported sample keeps them.
pub(crate) struct FeSubset {
    /// Positions (kept order of the reported sample) of the solved rows, ascending.
    pub rows: Vec<u32>,
    /// Per term slot and level of the reported sample: the value reported for levels without a
    /// solved row (+inf / -inf for intercept slots of levels whose outcomes are all 1 / all 0, else 0).
    pub limits: Vec<Vec<f64>>,
}

/// Output buffer written at disjoint indices from parallel chunks.
#[derive(Clone, Copy)]
struct Scatter(*mut u32);
// SAFETY: every index is written by exactly one chunk (see `level_rows`).
unsafe impl Send for Scatter {}
unsafe impl Sync for Scatter {}

impl Scatter {
    /// # Safety
    /// `i` is inside the buffer and no other thread writes it.
    unsafe fn write(&self, i: usize, v: u32) {
        // SAFETY: guaranteed by the caller.
        unsafe { *self.0.add(i) = v };
    }
}

/// Rows of every level, ascending within a level (CSR: level `l` owns `rows[off[l]..off[l+1]]`).
/// Few levels (g x chunks <= n): per-chunk histograms, each chunk scattering its rows in order
/// into its own range of every level. Many levels: an atomic counting sort (little contention)
/// and a per-level sort. Neither depends on the thread count.
fn level_rows(codes: &[u32], g: usize, pool: &Pool) -> (Vec<u32>, Vec<u32>) {
    let n = codes.len();
    if pool.install(|| codes.par_windows(2).all(|w| w[0] <= w[1])) {
        let off = pool.install(|| (0..=g).into_par_iter().map(|l| codes.partition_point(|&v| (v as usize) < l) as u32).collect());
        return (off, pool.install(|| (0..n as u32).into_par_iter().collect()));
    }
    let nch = n.div_ceil(CHUNK);
    if g.saturating_mul(nch) <= n {
        let mut starts: Vec<Vec<u32>> = pool.install(|| {
            codes
                .par_chunks(CHUNK)
                .map(|c| {
                    let mut h = vec![0u32; g];
                    c.iter().for_each(|&v| h[v as usize] += 1);
                    h
                })
                .collect()
        });
        // Exclusive prefix over (level, chunk): chunk c's rows of level l start at starts[c][l].
        let mut off = Vec::with_capacity(g + 1);
        let mut pos = 0u32;
        for l in 0..g {
            off.push(pos);
            for s in starts.iter_mut() {
                let count = s[l];
                s[l] = pos;
                pos += count;
            }
        }
        off.push(pos);
        let mut rows = vec![0u32; n];
        let out = Scatter(rows.as_mut_ptr());
        pool.install(|| {
            codes.par_chunks(CHUNK).zip(starts.par_iter_mut()).enumerate().for_each(|(c, (chunk, cursor))| {
                for (k, &v) in chunk.iter().enumerate() {
                    // SAFETY: the ranges of the chunks within each level are disjoint and inside `rows`.
                    unsafe { out.write(cursor[v as usize] as usize, (c * CHUNK + k) as u32) };
                    cursor[v as usize] += 1;
                }
            })
        });
        return (off, rows);
    }
    let counts: Vec<AtomicU32> = (0..g).map(|_| AtomicU32::new(0)).collect();
    pool.install(|| {
        codes.par_chunks(CHUNK).for_each(|c| {
            for &v in c {
                counts[v as usize].fetch_add(1, Ordering::Relaxed);
            }
        })
    });
    let mut off = Vec::with_capacity(g + 1);
    off.push(0u32);
    for c in &counts {
        off.push(off[off.len() - 1] + c.load(Ordering::Relaxed));
    }
    let cursor: Vec<AtomicU32> = off[..g].iter().map(|&o| AtomicU32::new(o)).collect();
    let mut rows = vec![0u32; n];
    let out = Scatter(rows.as_mut_ptr());
    pool.install(|| {
        codes.par_chunks(CHUNK).enumerate().for_each(|(c, chunk)| {
            for (k, &v) in chunk.iter().enumerate() {
                let p = cursor[v as usize].fetch_add(1, Ordering::Relaxed);
                // SAFETY: every position is handed out once by the atomic cursors.
                unsafe { out.write(p as usize, (c * CHUNK + k) as u32) };
            }
        })
    });
    let mut segs: Vec<&mut [u32]> = Vec::with_capacity(g);
    let mut rest: &mut [u32] = &mut rows;
    for l in 0..g {
        let (head, tail) = rest.split_at_mut((off[l + 1] - off[l]) as usize);
        segs.push(head);
        rest = tail;
    }
    pool.install(|| segs.par_iter_mut().for_each(|s| s.sort_unstable()));
    (off, rows)
}

/// A set of row indices with O(log_64 n) insert and minimum: a bitset per level, each word of a
/// level summarized by one bit of the level above, up to a single word. Deletion is lazy:
/// [`MinSet::first_valid`] drops the invalid entries it meets on the way to the minimum.
struct MinSet {
    levels: Vec<Vec<u64>>,
}

impl MinSet {
    fn new(n: usize) -> MinSet {
        let mut levels = vec![vec![0u64; n.div_ceil(64).max(1)]];
        while levels[levels.len() - 1].len() > 1 {
            let below = levels[levels.len() - 1].len();
            levels.push(vec![0u64; below.div_ceil(64)]);
        }
        MinSet { levels }
    }

    fn insert(&mut self, i: usize) {
        let mut i = i;
        for level in &mut self.levels {
            let was = level[i / 64];
            level[i / 64] = was | 1 << (i % 64);
            if was != 0 {
                return;
            }
            i /= 64;
        }
    }

    /// The smallest element for which `valid` holds, after removing the smaller ones (a word
    /// of the bottom level at a time).
    fn first_valid(&mut self, valid: impl Fn(usize) -> bool) -> Option<usize> {
        loop {
            let top_level = self.levels.len() - 1;
            let top = self.levels[top_level][0];
            if top == 0 {
                return None;
            }
            // Index of the bottom-level word holding the minimum.
            let mut w = 0;
            if top_level > 0 {
                w = top.trailing_zeros() as usize;
                for level in self.levels[1..top_level].iter().rev() {
                    w = w * 64 + level[w].trailing_zeros() as usize;
                }
            }
            let mut word = self.levels[0][w];
            while word != 0 {
                let i = w * 64 + word.trailing_zeros() as usize;
                if valid(i) {
                    self.levels[0][w] = word;
                    return Some(i);
                }
                word &= word - 1;
            }
            // The whole word was invalid: clear it and its summary bits.
            self.levels[0][w] = 0;
            let mut i = w;
            for level in &mut self.levels[1..] {
                level[i / 64] &= !(1 << (i % 64));
                if level[i / 64] != 0 {
                    break;
                }
                i /= 64;
            }
        }
    }
}

/// First row of every level (u32::MAX for levels without rows), in parallel.
fn first_rows(codes: &[u32], g: usize, pool: &Pool) -> Vec<u32> {
    let first: Vec<AtomicU32> = (0..g).map(|_| AtomicU32::new(u32::MAX)).collect();
    pool.install(|| {
        codes.par_chunks(CHUNK).enumerate().for_each(|(c, chunk)| {
            for (k, &v) in chunk.iter().enumerate() {
                // Plain read first: once a level's first row is known, later rows only read the
                // shared line (no contention on few-level terms).
                let (slot, row) = (&first[v as usize], (c * CHUNK + k) as u32);
                if row < slot.load(Ordering::Relaxed) {
                    slot.fetch_min(row, Ordering::Relaxed);
                }
            }
        })
    });
    first.into_iter().map(AtomicU32::into_inner).collect()
}

/// fixest's `cpp_get_fe_gnl` (misc_funs.cpp) over `codes.len()` intercept-only terms on the
/// estimation rows (estimation order): the number of levels per term set to 0 as references
/// and, given the FE sum of every row (`sum`), the values of every level. The choice is
/// structural:
/// - pick an observation: among those not yet determined, the first one (lowest row) with the
///   most terms already determined (initially row 0);
/// - on it, the first undetermined term is left free and every other undetermined term's level
///   becomes a reference (0);
/// - propagate: an observation with all terms but one determined determines the remaining level
///   as its sum minus the other terms' values;
/// - repeat until every level is determined.
///
/// The references do not depend on the order of the propagation (the set of determined levels
/// is its closure). Rows waiting with r determined terms (1 <= r <= q - 2) are kept in one
/// [`MinSet`] per r (lazy deletion), rows with q - 1 in a stack; the per-term row lists are
/// built in parallel. The state of a row is two bytes: the number of its determined terms and
/// their XOR, which names the last undetermined one (q <= 255). For two terms this is one
/// reference in the second term per connected component.
fn get_fe_gnl(codes: &[&[u32]], sizes: &[usize], sum: Option<&[f64]>, pool: &Pool) -> (Vec<usize>, Vec<Vec<f64>>) {
    let q = codes.len();
    let mut refs = vec![0usize; q];
    let mut values: Vec<Vec<f64>> = sizes.iter().map(|&g| vec![0.0; if sum.is_some() { g } else { 0 }]).collect();
    if q == 0 {
        return (refs, values);
    }
    let n = codes[0].len();
    if q == 1 {
        if let Some(s) = sum {
            let first = first_rows(codes[0], sizes[0], pool);
            values[0].iter_mut().zip(&first).filter(|(_, f)| **f != u32::MAX).for_each(|(v, &f)| *v = s[f as usize]);
        }
        return (refs, values);
    }
    assert!(q <= 255, "at most 255 fixed-effect terms");
    let lists: Vec<(Vec<u32>, Vec<u32>)> = pool.install(|| codes.par_iter().zip(sizes).map(|(c, &g)| level_rows(c, g, pool)).collect());
    // Low byte: number of determined terms; high byte: XOR of their indices.
    let mut st = vec![0u16; n];
    let xor_all = (0..q as u16).fold(0, |a, t| a ^ t);
    let last = q as u16 - 1;
    let mut done: Vec<Vec<bool>> = sizes.iter().map(|&g| vec![false; g]).collect();
    // Rows that reached r determined terms (1 <= r <= q - 2), with lazy deletion: a row stays in
    // the set of r after moving on, and is removed when it surfaces as its minimum.
    let mut waiting: Vec<MinSet> = (1..q - 1).map(|_| MinSet::new(n)).collect();
    let mut queue: Vec<u32> = Vec::new();
    // Marks level l of term t determined: each of its rows gains one determined term.
    let mark = |t: usize, l: usize, st: &mut [u16], done: &mut [Vec<bool>], waiting: &mut [MinSet], queue: &mut Vec<u32>| {
        done[t][l] = true;
        let (off, rows) = &lists[t];
        for &i in &rows[off[l] as usize..off[l + 1] as usize] {
            let s = &mut st[i as usize];
            *s = (*s + 1) ^ ((t as u16) << 8);
            let r = *s & 0xff;
            if r < last {
                waiting[r as usize - 1].insert(i as usize);
            } else if r == last {
                queue.push(i);
            }
        }
    };
    let mut cursor = 0usize;
    loop {
        let pick = waiting.iter_mut().enumerate().rev().find_map(|(r, set)| set.first_valid(|i| (st[i] & 0xff) as usize == r + 1)).or_else(|| {
            while cursor < n && st[cursor] != 0 {
                cursor += 1;
            }
            (cursor < n).then_some(cursor)
        });
        let Some(i) = pick else { return (refs, values) };
        let mut first = true;
        for t in 0..q {
            let l = codes[t][i] as usize;
            if done[t][l] {
                continue;
            }
            if first {
                first = false;
            } else {
                refs[t] += 1;
                mark(t, l, &mut st, &mut done, &mut waiting, &mut queue);
            }
        }
        while let Some(i) = queue.pop() {
            let i = i as usize;
            // A row whose last term got determined through another row is complete.
            if st[i] & 0xff != last {
                continue;
            }
            let t = ((st[i] >> 8) ^ xor_all) as usize;
            let l = codes[t][i] as usize;
            if let Some(s) = sum {
                let others: f64 = (0..q).filter(|&u| u != t).map(|u| values[u][codes[u][i] as usize]).sum();
                values[t][l] = s[i] - others;
            }
            mark(t, l, &mut st, &mut done, &mut waiting, &mut queue);
        }
    }
}

/// Number of levels per term that fixest sets to 0 as references (see [`get_fe_gnl`]).
pub(crate) fn references(codes: &[&[u32]], sizes: &[usize], pool: &Pool) -> Vec<usize> {
    get_fe_gnl(codes, sizes, None, pool).0
}

/// fixest `cpp_constant_dum`: levels whose slope variable is constant (term with intercept) or
/// identically zero (slope-only term), i.e. whose slope coefficient is not identified.
fn constant_slope_levels(codes: &[u32], g: usize, x: &[f64], intercept: bool) -> usize {
    let mut first = vec![f64::NAN; g];
    let mut varies = vec![false; g];
    for (&c, &v) in codes.iter().zip(x) {
        let c = c as usize;
        if intercept {
            if first[c].is_nan() {
                first[c] = v;
            } else if v != first[c] {
                varies[c] = true;
            }
        } else if v != 0.0 {
            varies[c] = true;
        }
    }
    varies.iter().filter(|v| !**v).count()
}

/// Union-find with path halving.
fn find(parent: &mut [u32], mut a: u32) -> u32 {
    while parent[a as usize] != a {
        parent[a as usize] = parent[parent[a as usize] as usize];
        a = parent[a as usize];
    }
    a
}

/// pyfixest's normalization of intercept terms `vals` (the first free, every other term's first
/// level pinned at 0, as its `C(f1) + C(f2) + ...` dummies without intercept) with the remaining
/// indeterminacy resolved by minimum norm, as its `lsqr` solve does. The indeterminacy is taken
/// to be spanned by the shifts v_{t,c} = +1 on the first term's levels and -1 on term t's levels
/// of each connected component c of the (first term, term t) level graph: each is in the null
/// space of the dummies (every row has both levels in c or neither), so the fit is unchanged,
/// and they span it for two terms and for the usual designs with more (e.g. worker, firm, year).
fn min_norm(codes: &[&[u32]], sizes: &[usize], vals: &mut [Vec<f64>]) {
    let q = codes.len();
    if q < 2 {
        return;
    }
    let g1 = sizes[0];
    // Per term t >= 1: component of every first-term level and of every term-t level, and the
    // index of each component's unknown shift (`None`: the pinned component).
    let mut c1: Vec<Vec<usize>> = Vec::with_capacity(q - 1);
    let mut ct: Vec<Vec<usize>> = Vec::with_capacity(q - 1);
    let mut var: Vec<Vec<Option<usize>>> = Vec::with_capacity(q - 1);
    let mut nvar = 0usize;
    for t in 1..q {
        let mut parent: Vec<u32> = (0..(g1 + sizes[t]) as u32).collect();
        for (&a, &b) in codes[0].iter().zip(codes[t]) {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, (g1 + b as usize) as u32));
            if ra != rb {
                parent[rb as usize] = ra;
            }
        }
        let mut ids = vec![usize::MAX; g1 + sizes[t]];
        let mut nc = 0;
        let comp: Vec<usize> = (0..g1 + sizes[t])
            .map(|v| {
                let root = find(&mut parent, v as u32) as usize;
                if ids[root] == usize::MAX {
                    ids[root] = nc;
                    nc += 1;
                }
                ids[root]
            })
            .collect();
        let (a, b) = comp.split_at(g1);
        // Pin: shift the pinned level's component so that its first level is 0.
        if sizes[t] > 0 {
            let (c0, s) = (b[0], vals[t][0]);
            vals[t].iter_mut().zip(b).filter(|(_, c)| **c == c0).for_each(|(v, _)| *v -= s);
            vals[0].iter_mut().zip(a).filter(|(_, c)| **c == c0).for_each(|(v, _)| *v += s);
        }
        let v: Vec<Option<usize>> = (0..nc)
            .map(|c| {
                (sizes[t] == 0 || c != b[0]).then(|| {
                    nvar += 1;
                    nvar - 1
                })
            })
            .collect();
        c1.push(a.to_vec());
        ct.push(b.to_vec());
        var.push(v);
    }
    if nvar == 0 {
        return;
    }
    // Minimize ||vals + sum s_v v||^2 over the shifts s: normal equations A s = b with
    // (A s)_{t,c} = sum_{first-term l in c} S_l + n_{t,c}*s_{t,c}, S_l = sum_t s_{t,c_t(l)}, and
    // b_{t,c} = sum_{term-t m in c} vals_t[m] - sum_{first-term l in c} vals_1[l]; Jacobi-preconditioned
    // CG (A is diagonal, and one iteration exact, for two terms).
    let mut rhs = vec![0.0; nvar];
    let mut diag = vec![0.0; nvar];
    for t in 0..q - 1 {
        for (l, &c) in c1[t].iter().enumerate() {
            if let Some(j) = var[t][c] {
                rhs[j] -= vals[0][l];
                diag[j] += 1.0;
            }
        }
        for (m, &c) in ct[t].iter().enumerate() {
            if let Some(j) = var[t][c] {
                rhs[j] += vals[t + 1][m];
                diag[j] += 1.0;
            }
        }
    }
    let apply = |s: &[f64], out: &mut [f64]| {
        out.fill(0.0);
        for l in 0..g1 {
            let sl: f64 = (0..q - 1).filter_map(|t| var[t][c1[t][l]].map(|j| s[j])).sum();
            (0..q - 1).filter_map(|t| var[t][c1[t][l]]).for_each(|j| out[j] += sl);
        }
        for t in 0..q - 1 {
            for &c in &ct[t] {
                if let Some(j) = var[t][c] {
                    out[j] += s[j];
                }
            }
        }
    };
    let mut s = vec![0.0; nvar];
    let mut r = rhs.clone();
    let mut z: Vec<f64> = r.iter().zip(&diag).map(|(r, d)| r / d).collect();
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let stop = 1e-30 * rhs.iter().map(|v| v * v).sum::<f64>();
    let mut ap = vec![0.0; nvar];
    for _ in 0..10 * nvar + 100 {
        if r.iter().map(|v| v * v).sum::<f64>() <= stop {
            break;
        }
        apply(&p, &mut ap);
        let alpha = rz / p.iter().zip(&ap).map(|(a, b)| a * b).sum::<f64>();
        s.iter_mut().zip(&p).for_each(|(x, p)| *x += alpha * p);
        r.iter_mut().zip(&ap).for_each(|(x, a)| *x -= alpha * a);
        z.iter_mut().zip(r.iter().zip(&diag)).for_each(|(z, (r, d))| *z = r / d);
        let rz_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz_new / rz;
        rz = rz_new;
        p.iter_mut().zip(&z).for_each(|(p, z)| *p = z + beta * *p);
    }
    for l in 0..g1 {
        vals[0][l] += (0..q - 1).filter_map(|t| var[t][c1[t][l]].map(|j| s[j])).sum::<f64>();
    }
    for t in 0..q - 1 {
        for (m, &c) in ct[t].iter().enumerate() {
            if let Some(j) = var[t][c] {
                vals[t + 1][m] -= s[j];
            }
        }
    }
}

/// Normalized values per term slot (term order) and references of the intercept terms (in
/// order), for terms with codes `codes`, levels `sizes`, intercept flags `intercept` and
/// `nslopes` slopes, from solved values `slots` over the same rows.
fn normalize(
    codes: &[&[u32]],
    sizes: &[usize],
    intercept: &[bool],
    nslopes: &[usize],
    mut slots: Vec<Vec<f64>>,
    norm: FixefNorm,
    pool: &Pool,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    // Slot index of every intercept term.
    let mut slot = 0;
    let mut icpt: Vec<(usize, usize)> = Vec::new();
    for (q, (&ic, &ns)) in intercept.iter().zip(nslopes).enumerate() {
        if ic {
            icpt.push((q, slot));
        }
        slot += usize::from(ic) + ns;
    }
    let ic: Vec<&[u32]> = icpt.iter().map(|&(q, _)| codes[q]).collect();
    let is: Vec<usize> = icpt.iter().map(|&(q, _)| sizes[q]).collect();
    let mut refs = vec![0usize; icpt.len()];
    if icpt.len() >= 2 {
        match norm {
            FixefNorm::Fixest => {
                let n = ic[0].len();
                let sum: Vec<f64> = pool.install(|| {
                    (0..n).into_par_iter().with_min_len(CHUNK).map(|i| icpt.iter().zip(&ic).map(|(&(_, s), c)| slots[s][c[i] as usize]).sum()).collect()
                });
                let (r, vals) = get_fe_gnl(&ic, &is, Some(&sum), pool);
                refs = r;
                for (&(_, s), v) in icpt.iter().zip(vals) {
                    slots[s] = v;
                }
            }
            FixefNorm::MinNorm => {
                let mut vals: Vec<Vec<f64>> = icpt.iter().map(|&(_, s)| std::mem::take(&mut slots[s])).collect();
                min_norm(&ic, &is, &mut vals);
                for (&(_, s), v) in icpt.iter().zip(vals) {
                    slots[s] = v;
                }
                refs.iter_mut().zip(&is).skip(1).for_each(|(r, &g)| *r = usize::from(g > 0));
            }
        }
    }
    (slots, refs)
}

/// [`FixefValues`] of every term slot under `norm`. `terms` hold the codes over the reported
/// rows `obs` (input rows, kept order); `fe` was solved over those rows, or over the rows
/// `fe.subset.rows` only. Intercept terms are renormalized; slope coefficients are reported
/// as solved.
pub(crate) fn fixef_values(terms: &[StateFe], obs: &[u32], fe: &FeLevels, norm: FixefNorm, pool: &Pool) -> Result<Vec<FixefValues>, RfError> {
    let intercept: Vec<bool> = terms.iter().map(|t| t.intercept).collect();
    let nslopes: Vec<usize> = terms.iter().map(|t| t.slopes.len()).collect();
    let (slots, refs) = match &fe.subset {
        None => {
            let codes: Vec<&[u32]> = terms.iter().map(|t| t.codes.as_slice()).collect();
            let sizes: Vec<usize> = terms.iter().map(|t| t.levels).collect();
            normalize(&codes, &sizes, &intercept, &nslopes, fe.slots.clone(), norm, pool)
        }
        Some(sub) => {
            // The solved codes are the reported ones restricted to the solved rows and renumbered
            // densely in order (both number levels in ascending order of the factor values).
            let solved: Vec<Codes> = terms
                .iter()
                .map(|t| factor::restrict(&Codes { codes: t.codes.clone(), n: t.levels as u32 }, &sub.rows, pool))
                .collect::<Result<_, _>>()?;
            let codes: Vec<&[u32]> = solved.iter().map(|c| c.codes.as_slice()).collect();
            let sizes: Vec<usize> = solved.iter().map(|c| c.n as usize).collect();
            let (vals, refs) = normalize(&codes, &sizes, &intercept, &nslopes, fe.slots.clone(), norm, pool);
            // Solved level -> reported level, per term.
            let level_of: Vec<Vec<u32>> = terms
                .iter()
                .zip(&solved)
                .map(|(t, s)| {
                    let mut of = vec![0u32; s.n as usize];
                    s.codes.iter().zip(&sub.rows).for_each(|(&l, &k)| of[l as usize] = t.codes[k as usize]);
                    of
                })
                .collect();
            let slot_term = terms.iter().enumerate().flat_map(|(q, t)| std::iter::repeat_n(q, usize::from(t.intercept) + t.slopes.len()));
            let full = vals
                .into_iter()
                .zip(slot_term)
                .zip(&sub.limits)
                .map(|((v, q), lim)| {
                    let mut out = lim.clone();
                    level_of[q].iter().zip(v).for_each(|(&l, x)| out[l as usize] = x);
                    out
                })
                .collect();
            (full, refs)
        }
    };
    let mut out = Vec::new();
    let mut slots = slots.into_iter();
    let mut next_ref = refs.into_iter();
    for (q, t) in terms.iter().enumerate() {
        let first_obs: Vec<u32> = first_rows(&t.codes, t.levels, pool).into_iter().map(|k| obs[k as usize]).collect();
        if t.intercept {
            let values = slots.next().expect("intercept slot");
            out.push(FixefValues { term: q, slope: None, values, first_obs: first_obs.clone(), references: next_ref.next().expect("reference count") });
        }
        for (&j, x) in t.slope_ids.iter().zip(&t.slopes) {
            let values = slots.next().expect("slope slot");
            let references = constant_slope_levels(&t.codes, t.levels, x, t.intercept);
            out.push(FixefValues { term: q, slope: Some(j), values, first_obs: first_obs.clone(), references });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::par::with_pool;

    #[test]
    fn two_terms_give_one_reference_per_component() {
        // Components {a0, b0, b1}, {a1, b2}, {a2, b3}.
        let a = [0u32, 0, 1, 2];
        let b = [0u32, 1, 2, 3];
        with_pool(2, |pool| {
            assert_eq!(references(&[&a, &b], &[3, 4], pool), vec![0, 3]);
            assert_eq!(references(&[&a], &[3], pool), vec![0]);
        })
        .unwrap();
    }

    /// The three layouts (sorted codes, per-chunk histograms, atomic counting sort) list the
    /// rows of every level in ascending order at any thread count.
    #[test]
    fn level_rows_layouts() {
        let n = 100_000;
        let mut rng = crate::rng::SplitMix64::new(3);
        let random = |g: u64, rng: &mut crate::rng::SplitMix64| -> Vec<u32> { (0..n).map(|_| (rng.next_u64() % g) as u32).collect() };
        let mut sorted = random(700, &mut rng);
        sorted.sort_unstable();
        for (codes, g) in [(sorted, 700), (random(50, &mut rng), 50), (random(60_000, &mut rng), 60_000)] {
            let mut want_off = vec![0u32; g + 1];
            codes.iter().for_each(|&v| want_off[v as usize + 1] += 1);
            (0..g).for_each(|l| want_off[l + 1] += want_off[l]);
            let mut want_rows: Vec<u32> = (0..n as u32).collect();
            want_rows.sort_by_key(|&i| codes[i as usize]);
            for threads in [1, 4] {
                let got = with_pool(threads, |pool| level_rows(&codes, g, pool)).unwrap();
                assert_eq!(got, (want_off.clone(), want_rows.clone()), "g = {g}, threads = {threads}");
            }
        }
    }

    #[test]
    fn min_set_orders_rows() {
        for n in [50, 300_000] {
            let mut s = MinSet::new(n);
            assert_eq!(s.first_valid(|_| true), None);
            let items: Vec<usize> = [n - 1, n * 3 / 4, n / 2 + 1, n / 2, 3].into_iter().collect();
            items.iter().for_each(|&i| s.insert(i));
            assert_eq!(s.first_valid(|_| true), Some(3));
            assert_eq!(s.first_valid(|i| i > 3), Some(n / 2));
            assert_eq!(s.first_valid(|i| i > n / 2 + 1), Some(n * 3 / 4));
            // Invalid entries below the minimum were dropped; valid ones stay.
            assert_eq!(s.first_valid(|_| true), Some(n * 3 / 4));
            assert_eq!(s.first_valid(|i| i == n - 1), Some(n - 1));
            assert_eq!(s.first_valid(|_| false), None);
            assert_eq!(s.first_valid(|_| true), None);
        }
    }

    /// The propagation with waiting sets reproduces the straightforward implementation (heap
    /// buckets with lazy deletion, the previous code) on three and four random terms.
    #[test]
    fn references_match_reference_implementation() {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        fn slow(codes: &[&[u32]], sizes: &[usize]) -> Vec<usize> {
            let q = codes.len();
            let n = codes[0].len();
            let mut refs = vec![0; q];
            let mut done: Vec<Vec<bool>> = sizes.iter().map(|&g| vec![false; g]).collect();
            let mut rs = vec![0usize; n];
            let mut heaps: Vec<BinaryHeap<Reverse<usize>>> = (0..q).map(|_| BinaryHeap::new()).collect();
            let mut queue = Vec::new();
            let mark = |t: usize, l: usize, rs: &mut [usize], done: &mut [Vec<bool>], heaps: &mut [BinaryHeap<Reverse<usize>>], queue: &mut Vec<usize>| {
                done[t][l] = true;
                for i in (0..n).filter(|&i| codes[t][i] as usize == l) {
                    rs[i] += 1;
                    if rs[i] == q - 1 {
                        queue.push(i);
                    } else if rs[i] < q - 1 {
                        heaps[rs[i]].push(Reverse(i));
                    }
                }
            };
            let mut cursor = 0;
            loop {
                let mut pick = None;
                for r in (1..q - 1).rev() {
                    while let Some(&Reverse(i)) = heaps[r].peek() {
                        if rs[i] == r {
                            pick = Some(i);
                            break;
                        }
                        heaps[r].pop();
                    }
                    if pick.is_some() {
                        break;
                    }
                }
                if pick.is_none() {
                    while cursor < n && rs[cursor] != 0 {
                        cursor += 1;
                    }
                    if cursor == n {
                        return refs;
                    }
                    pick = Some(cursor);
                }
                let i = pick.unwrap();
                let mut first = true;
                for t in 0..q {
                    let l = codes[t][i] as usize;
                    if !done[t][l] {
                        if first {
                            first = false;
                        } else {
                            refs[t] += 1;
                            mark(t, l, &mut rs, &mut done, &mut heaps, &mut queue);
                        }
                    }
                }
                while let Some(i) = queue.pop() {
                    if rs[i] == q - 1 {
                        let t = (0..q).find(|&t| !done[t][codes[t][i] as usize]).unwrap();
                        mark(t, codes[t][i] as usize, &mut rs, &mut done, &mut heaps, &mut queue);
                    }
                }
            }
        }
        let mut rng = crate::rng::SplitMix64::new(7);
        for (n, g) in [(60usize, [9u32, 5, 4, 3]), (400, [80, 30, 12, 7]), (3000, [900, 40, 300, 2])] {
            let cols: Vec<Vec<u32>> = g.iter().map(|&gq| (0..n).map(|_| (rng.next_u64() % gq as u64) as u32).collect()).collect();
            for q in [3usize, 4] {
                // Densify each term (codes must be the present levels).
                let dense: Vec<Vec<u32>> = cols[..q]
                    .iter()
                    .map(|c| {
                        let mut lv: Vec<u32> = c.clone();
                        lv.sort_unstable();
                        lv.dedup();
                        c.iter().map(|v| lv.binary_search(v).unwrap() as u32).collect()
                    })
                    .collect();
                let sizes: Vec<usize> = dense.iter().map(|c| *c.iter().max().unwrap() as usize + 1).collect();
                let codes: Vec<&[u32]> = dense.iter().map(Vec::as_slice).collect();
                for threads in [1, 4] {
                    let fast = with_pool(threads, |pool| references(&codes, &sizes, pool)).unwrap();
                    assert_eq!(fast, slow(&codes, &sizes), "n = {n}, q = {q}");
                }
            }
        }
    }

    #[test]
    fn min_norm_pins_and_minimizes() {
        // Two components: {a0, a1, b0, b1} (b0 pinned) and {a2, b2} (free shift).
        let a = [0u32, 1, 0, 2];
        let b = [0u32, 1, 1, 2];
        let mut v = vec![vec![1.0, 2.0, 5.0], vec![0.5, 1.0, -1.0]];
        let sum = |v: &[Vec<f64>], i: usize| v[0][a[i] as usize] + v[1][b[i] as usize];
        let before: Vec<f64> = (0..4).map(|i| sum(&v, i)).collect();
        min_norm(&[&a, &b], &[3, 3], &mut v);
        assert_eq!(v[1][0], 0.0);
        for (i, s) in before.iter().enumerate() {
            assert!((sum(&v, i) - s).abs() < 1e-12);
        }
        // a2 + b2 = 4 split evenly.
        assert!((v[0][2] - 2.0).abs() < 1e-12 && (v[1][2] - 2.0).abs() < 1e-12);
    }
}
