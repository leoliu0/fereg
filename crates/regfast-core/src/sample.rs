//! Estimation sample (plan §1.3): reghdfe's cyclic singleton removal, extended with the GLM
//! `fe` separation rule.
//!
//! Each pass counts the level sizes of one term over the current sample and drops the
//! observations of offending levels. Without frequency weights the counts are exact integers
//! (atomic histograms, parallel); with fweights, or Poisson depvars with negative values, the
//! serial f64 path keeps the exact weighted-sum semantics. Both give the same sample.

use std::sync::atomic::{AtomicU32, Ordering};

use rayon::prelude::*;

use crate::api::Family;
use crate::par::{CHUNK, Pool};

/// GLM `fe` separation: Poisson drops FE groups with Σy = 0; logit also drops groups with Σy = count.
#[derive(Clone, Copy, Debug)]
pub struct SepRule<'a> {
    /// Depvar over the same observations as the codes (logit: any nonzero value counts as 1).
    pub y: &'a [f64],
    pub family: Family,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DropCounts {
    /// Observations dropped because their FE group had count 1.
    pub singletons: u64,
    /// Observations dropped by the `fe` separation rule.
    pub fe_separated: u64,
}

/// One absorbed term: codes over all observations and its number of levels.
#[derive(Clone, Copy, Debug)]
pub struct TermCodes<'a> {
    pub codes: &'a [u32],
    pub g: usize,
}

/// Iteratively drops singleton observations (when `singletons`) and `fe`-separated observations
/// (when `sep` is given) until no term changes, cycling over terms in absorb order exactly as
/// reghdfe does. `alive` is the current sample and is updated in place. With fweights, group
/// counts are weight sums.
pub fn drop_singletons(
    terms: &[TermCodes],
    alive: &mut [bool],
    fw: Option<&[f64]>,
    sep: Option<SepRule>,
    singletons: bool,
    pool: &Pool,
) -> DropCounts {
    let mut out = DropCounts::default();
    let nterms = terms.len();
    if nterms == 0 || (!singletons && sep.is_none()) {
        return out;
    }
    // Integer counting is exact unless fweights or a Poisson depvar with negative values make
    // the group sums genuinely real-valued.
    let serial = fw.is_some()
        || sep.is_some_and(|r| r.family == Family::Poisson && r.y.iter().any(|v| !(*v >= 0.0)));
    let mut remaining = nterms;
    let mut i = 0usize;
    while remaining > 0 {
        let t = terms[i % nterms];
        i += 1;
        let (s, f) = if serial { pass_serial(t, alive, fw, sep, singletons) } else { pass_parallel(t, alive, sep, singletons, pool) };
        out.singletons += s;
        out.fe_separated += f;
        remaining = if s + f > 0 { nterms - 1 } else { remaining - 1 };
    }
    out
}

fn separated(family: Family, ysum: f64, count: f64) -> bool {
    match family {
        Family::Poisson => ysum == 0.0,
        Family::Logit => ysum == 0.0 || ysum == count,
        Family::Gaussian => false,
    }
}

/// Levels with at most this many entries are counted in per-worker local histograms (exact
/// integer counts, so the merge order cannot change the result); larger ones use atomics.
const LOCAL_HIST_MAX: usize = 1 << 14;

/// Per-level counts of alive observations and of those with a nonzero outcome.
fn level_counts(t: TermCodes, alive: &[bool], sep: Option<SepRule>, pool: &Pool) -> (Vec<u32>, Vec<u32>) {
    let with_y = sep.is_some();
    let width = if with_y { 2 * t.g } else { t.g };
    if t.g <= LOCAL_HIST_MAX {
        let h = pool.install(|| {
            t.codes
                .par_chunks(CHUNK)
                .zip(alive.par_chunks(CHUNK))
                .enumerate()
                .fold(
                    || vec![0u32; width],
                    |mut h, (ci, (codes, al))| {
                        let base = ci * CHUNK;
                        for (k, (&c, &a)) in codes.iter().zip(al).enumerate() {
                            if a {
                                h[c as usize] += 1;
                                if let Some(rule) = sep {
                                    h[t.g + c as usize] += u32::from(rule.y[base + k] != 0.0);
                                }
                            }
                        }
                        h
                    },
                )
                .reduce(
                    || vec![0u32; width],
                    |mut a, b| {
                        a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                        a
                    },
                )
        });
        let pos = if with_y { h[t.g..].to_vec() } else { Vec::new() };
        let mut count = h;
        count.truncate(t.g);
        return (count, pos);
    }
    let count: Vec<AtomicU32> = (0..t.g).map(|_| AtomicU32::new(0)).collect();
    let pos: Vec<AtomicU32> = if with_y { (0..t.g).map(|_| AtomicU32::new(0)).collect() } else { Vec::new() };
    pool.install(|| {
        t.codes.par_chunks(CHUNK).zip(alive.par_chunks(CHUNK)).enumerate().for_each(|(ci, (codes, al))| {
            let base = ci * CHUNK;
            for (k, (&c, &a)) in codes.iter().zip(al).enumerate() {
                if a {
                    count[c as usize].fetch_add(1, Ordering::Relaxed);
                    if let Some(rule) = sep {
                        if rule.y[base + k] != 0.0 {
                            pos[c as usize].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }
        })
    });
    (count.into_iter().map(AtomicU32::into_inner).collect(), pos.into_iter().map(AtomicU32::into_inner).collect())
}

fn pass_parallel(t: TermCodes, alive: &mut [bool], sep: Option<SepRule>, singletons: bool, pool: &Pool) -> (u64, u64) {
    let (count, pos) = level_counts(t, alive, sep, pool);
    // Verdict per level: 0 keep, 1 singleton, 2 separated.
    let verdict: Vec<u8> = (0..t.g)
        .map(|g| {
            let k = count[g];
            if k == 0 {
                0
            } else if singletons && k == 1 {
                1
            } else if sep.is_some_and(|rule| separated(rule.family, f64::from(pos[g]), f64::from(k))) {
                2
            } else {
                0
            }
        })
        .collect();
    if verdict.iter().all(|v| *v == 0) {
        return (0, 0);
    }
    let parts: Vec<(u64, u64)> = pool.install(|| {
        alive
            .par_chunks_mut(CHUNK)
            .zip(t.codes.par_chunks(CHUNK))
            .map(|(al, codes)| {
                let (mut s, mut f) = (0u64, 0u64);
                for (a, &c) in al.iter_mut().zip(codes) {
                    if *a {
                        match verdict[c as usize] {
                            1 => {
                                *a = false;
                                s += 1;
                            }
                            2 => {
                                *a = false;
                                f += 1;
                            }
                            _ => {}
                        }
                    }
                }
                (s, f)
            })
            .collect()
    });
    parts.into_iter().fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
}

fn pass_serial(t: TermCodes, alive: &mut [bool], fw: Option<&[f64]>, sep: Option<SepRule>, singletons: bool) -> (u64, u64) {
    let mut count = vec![0.0; t.g];
    let mut ysum = if sep.is_some() { vec![0.0; t.g] } else { Vec::new() };
    for (obs, &c) in t.codes.iter().enumerate() {
        if alive[obs] {
            let w = fw.map_or(1.0, |w| w[obs]);
            count[c as usize] += w;
            if let Some(rule) = sep {
                let y = match rule.family {
                    Family::Logit => f64::from(u8::from(rule.y[obs] != 0.0)),
                    _ => rule.y[obs],
                };
                ysum[c as usize] += w * y;
            }
        }
    }
    let (mut s, mut f) = (0u64, 0u64);
    for (obs, &c) in t.codes.iter().enumerate() {
        if !alive[obs] {
            continue;
        }
        let k = count[c as usize];
        if singletons && k == 1.0 {
            alive[obs] = false;
            s += 1;
        } else if let Some(rule) = sep {
            if separated(rule.family, ysum[c as usize], k) {
                alive[obs] = false;
                f += 1;
            }
        }
    }
    (s, f)
}
