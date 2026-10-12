//! Factor encoding (plan §1.2): dense codes 0..n in ascending value order.
//!
//! Numeric factors take a direct-table path when every value is an integer and the value range
//! is at most max(2n, 2²⁰); otherwise values are mapped to order-preserving u64 keys, sorted,
//! deduplicated and looked up by binary search. Every path is deterministic (the codes are a
//! function of the values only), whatever the thread count.

use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use crate::api::FactorValues;
use crate::fit::FactorCol;
use crate::error::RfError;
use crate::par::{CHUNK, Pool, alloc_uninit, alloc_vec, assume_init, chunk_range, map_chunks};

/// Dense level codes of one factor (or interaction) over some set of observations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Codes {
    pub codes: Vec<u32>,
    /// Number of levels; every code is `< n`.
    pub n: u32,
}

/// Order-preserving u64 key of a finite f64; −0.0 and 0.0 share a key.
pub fn f64_key(x: f64) -> u64 {
    let x = if x == 0.0 { 0.0 } else { x };
    let bits = x.to_bits();
    if bits >> 63 == 1 { !bits } else { bits | (1 << 63) }
}

/// Largest value range handled by a direct table for `n` observations.
fn table_limit(n: usize) -> u64 {
    (2 * n as u64).max(1 << 20)
}

/// Encodes a factor. Numeric codes follow numeric order; strings follow byte-lexicographic order.
pub fn factorize(v: &FactorValues, pool: &Pool) -> Result<Codes, RfError> {
    match *v {
        FactorValues::Num(x) => factorize_num(x, pool),
        FactorValues::Str { bytes, offsets } => factorize_str(bytes, offsets, pool),
    }
}

#[derive(Clone, Copy)]
struct Scan {
    min: f64,
    max: f64,
    nan: bool,
    int: bool,
}

fn factorize_num(x: &[f64], pool: &Pool) -> Result<Codes, RfError> {
    let n = x.len();
    if n == 0 {
        return Ok(Codes { codes: Vec::new(), n: 0 });
    }
    if let Some(c) = ascending_runs(x, pool)? {
        return Ok(c);
    }
    let parts = map_chunks(pool, n, CHUNK, |_, r| scan(&x[r]));
    let s = parts.iter().fold(Scan { min: f64::INFINITY, max: f64::NEG_INFINITY, nan: false, int: true }, |a, b| Scan {
        min: a.min.min(b.min),
        max: a.max.max(b.max),
        nan: a.nan | b.nan,
        int: a.int & b.int,
    });
    if s.nan {
        return Err(RfError::InvalidInput("missing values in factor".into()));
    }
    if s.int && s.max - s.min < table_limit(n) as f64 {
        return direct_table_num(x, s.min, (s.max - s.min) as usize + 1, pool);
    }
    let keys: Vec<u64> = pool.install(|| x.par_iter().map(|&v| f64_key(v)).collect());
    sort_encode(&keys, pool)
}

/// Range, NaN and integrality of `x`. Four independent branch-free lanes, so the loop
/// vectorizes; NaN fails every comparison (it is not an integer), so it is only searched for
/// when some value is not an integer.
fn scan(x: &[f64]) -> Scan {
    let (mut min, mut max, mut bad) = ([f64::INFINITY; 4], [f64::NEG_INFINITY; 4], [0u64; 4]);
    let quads = x.chunks_exact(4);
    let rest = quads.remainder();
    for q in quads {
        for k in 0..4 {
            let v = q[k];
            min[k] = if v < min[k] { v } else { min[k] };
            max[k] = if v > max[k] { v } else { max[k] };
            bad[k] |= u64::from(!((v == v.trunc()) & (v.abs() < 9.0e15)));
        }
    }
    for &v in rest {
        min[0] = if v < min[0] { v } else { min[0] };
        max[0] = if v > max[0] { v } else { max[0] };
        bad[0] |= u64::from(!((v == v.trunc()) & (v.abs() < 9.0e15)));
    }
    let int = bad.iter().all(|b| *b == 0);
    Scan {
        min: min.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        max: max.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        nan: !int && x.iter().any(|v| v.is_nan()),
        int,
    }
}

/// Sets a presence flag. One worker: a plain store. Several: only when unset, so the shared
/// flags' cache lines are not written over and over by every worker.
#[inline]
fn mark(p: &AtomicBool, single: bool) {
    if single || !p.load(Ordering::Relaxed) {
        p.store(true, Ordering::Relaxed);
    }
}

/// Rank of every slot among the present ones (slot order), and their number.
fn ranks(present: &[AtomicBool]) -> Result<(Vec<u32>, u32), RfError> {
    let mut map = alloc_vec(present.len(), 0u32)?;
    let mut next = 0u32;
    for (m, p) in map.iter_mut().zip(present) {
        *m = next;
        next += u32::from(p.load(Ordering::Relaxed));
    }
    Ok((map, next))
}

/// [`direct_table`] for integer values `lo..lo + range` (slice loops; x − lo is an exact
/// nonnegative integer, so the cheap i64 conversion applies).
fn direct_table_num(x: &[f64], lo: f64, range: usize, pool: &Pool) -> Result<Codes, RfError> {
    let single = pool.threads() == 1;
    let present: Vec<AtomicBool> = (0..range).map(|_| AtomicBool::new(false)).collect();
    pool.install(|| {
        x.par_chunks(CHUNK).for_each(|xs| xs.iter().for_each(|&v| mark(&present[(v - lo) as i64 as usize], single)))
    });
    let (map, n) = ranks(&present)?;
    drop(present);
    let mut codes = alloc_uninit::<u32>(x.len())?;
    pool.install(|| {
        codes.par_chunks_mut(CHUNK).zip(x.par_chunks(CHUNK)).for_each(|(cs, xs)| {
            for (c, &v) in cs.iter_mut().zip(xs) {
                c.write(map[(v - lo) as i64 as usize]);
            }
        })
    });
    // SAFETY: every chunk wrote each of its elements.
    Ok(Codes { codes: unsafe { assume_init(codes) }, n })
}

/// Codes of ascending values (data sorted by this factor): the runs of equal values numbered in
/// one pass. `None` as soon as a value descends or is NaN. Each part numbers its runs from 0 and
/// is shifted afterwards, so the codes do not depend on the number of parts.
fn ascending_runs(x: &[f64], pool: &Pool) -> Result<Option<Codes>, RfError> {
    // Unsorted data almost always shows within its first values.
    if !x[..x.len().min(256)].is_sorted_by(|a, b| a <= b) {
        return Ok(None);
    }
    let n = x.len();
    let size = n.div_ceil(pool.threads().min(n.div_ceil(CHUNK)).max(1));
    let mut codes = alloc_uninit::<u32>(n)?;
    let last: Vec<Option<u32>> = pool.install(|| {
        codes
            .par_chunks_mut(size)
            .enumerate()
            .map(|(j, cs)| {
                let xs = &x[j * size..j * size + cs.len()];
                let mut prev = xs[0];
                let mut k = 0u32;
                // Blocks of 256, branch free inside, so unsorted data stops after one block.
                for (cb, xb) in cs.chunks_mut(256).zip(xs.chunks(256)) {
                    let mut bad = prev.is_nan();
                    for (c, &v) in cb.iter_mut().zip(xb) {
                        let step = v != prev;
                        bad |= step & !(v > prev);
                        k += u32::from(step);
                        prev = v;
                        c.write(k);
                    }
                    if bad {
                        return None;
                    }
                }
                Some(k)
            })
            .collect()
    });
    let mut base = Vec::with_capacity(last.len());
    let mut next = 0u32;
    for (j, l) in last.iter().enumerate() {
        let Some(l) = *l else { return Ok(None) };
        if j > 0 {
            let (a, b) = (x[j * size - 1], x[j * size]);
            if b < a {
                return Ok(None);
            }
            next += u32::from(b != a);
        }
        base.push(next);
        next += l;
    }
    // SAFETY: every part ran to completion, writing each of its elements.
    let mut codes = unsafe { assume_init(codes) };
    if base.iter().any(|&b| b > 0) {
        pool.install(|| {
            codes.par_chunks_mut(size).zip(&base).for_each(|(cs, &b)| cs.iter_mut().for_each(|c| *c += b));
        });
    }
    Ok(Some(Codes { codes, n: next + 1 }))
}

/// Codes via a presence table over `0..range` (`slot(i)` is observation i's table index).
fn direct_table(n: usize, range: usize, slot: impl Fn(usize) -> usize + Sync, pool: &Pool) -> Result<Codes, RfError> {
    let single = pool.threads() == 1;
    let present: Vec<AtomicBool> = (0..range).map(|_| AtomicBool::new(false)).collect();
    pool.install(|| (0..n.div_ceil(CHUNK)).into_par_iter().for_each(|c| chunk_range(c, n, CHUNK).for_each(|i| mark(&present[slot(i)], single))));
    let (map, next) = ranks(&present)?;
    drop(present);
    let mut codes = alloc_uninit::<u32>(n)?;
    pool.install(|| {
        codes.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, cs)| {
            for (k, c) in cs.iter_mut().enumerate() {
                c.write(map[slot(ci * CHUNK + k)]);
            }
        })
    });
    // SAFETY: every chunk wrote each of its elements.
    Ok(Codes { codes: unsafe { assume_init(codes) }, n: next })
}

fn sort_encode(keys: &[u64], pool: &Pool) -> Result<Codes, RfError> {
    let mut uniq = alloc_vec(0, 0u64)?;
    uniq.try_reserve_exact(keys.len()).map_err(|_| RfError::OutOfMemory)?;
    uniq.extend_from_slice(keys);
    pool.install(|| uniq.par_sort_unstable());
    uniq.dedup();
    let codes = pool.install(|| {
        keys.par_iter().with_min_len(4096).map(|k| uniq.binary_search(k).expect("level present") as u32).collect()
    });
    Ok(Codes { codes, n: uniq.len() as u32 })
}

fn factorize_str(bytes: &[u8], offsets: &[u32], pool: &Pool) -> Result<Codes, RfError> {
    let n = offsets.len().saturating_sub(1);
    let get = |i: usize| &bytes[offsets[i] as usize..offsets[i + 1] as usize];
    let mut uniq: Vec<&[u8]> = pool.install(|| (0..n).into_par_iter().map(get).collect());
    pool.install(|| uniq.par_sort_unstable());
    uniq.dedup();
    let codes = pool.install(|| {
        (0..n).into_par_iter().with_min_len(1024).map(|i| uniq.binary_search(&get(i)).expect("level present") as u32).collect()
    });
    Ok(Codes { codes, n: uniq.len() as u32 })
}

/// Interaction of several factors (`a#b#c`): codes follow lexicographic order of (a, b, c)
/// over the combinations that occur.
pub fn combine(parts: &[&Codes], pool: &Pool) -> Result<Codes, RfError> {
    match parts {
        [] => Err(RfError::Internal("combine: no factors".into())),
        [one] => Ok((*one).clone()),
        _ => {
            let mut radix: u128 = 1;
            for p in parts {
                radix *= u128::from(p.n.max(1));
                if radix >= 1u128 << 63 {
                    return Err(RfError::InvalidInput("interaction has too many levels".into()));
                }
            }
            let n = parts[0].codes.len();
            let key = |i: usize| parts.iter().fold(0u64, |acc, p| acc * u64::from(p.n) + u64::from(p.codes[i]));
            if radix as u64 <= table_limit(n) {
                direct_table(n, (radix as usize).max(1), |i| key(i) as usize, pool)
            } else {
                let keys: Vec<u64> = pool.install(|| (0..n).into_par_iter().with_min_len(4096).map(key).collect());
                sort_encode(&keys, pool)
            }
        }
    }
}

/// [`compact`] given the ascending indices of the kept observations (parallel): the kept codes
/// are gathered, then renumbered only if some level vanished.
pub fn restrict(codes: &Codes, kept: &[u32], pool: &Pool) -> Result<Codes, RfError> {
    let single = pool.threads() == 1;
    let present: Vec<AtomicBool> = (0..codes.n).map(|_| AtomicBool::new(false)).collect();
    let mut out = alloc_uninit::<u32>(kept.len())?;
    pool.install(|| {
        out.par_chunks_mut(CHUNK).zip(kept.par_chunks(CHUNK)).for_each(|(o, ks)| {
            for (x, &i) in o.iter_mut().zip(ks) {
                let c = codes.codes[i as usize];
                x.write(c);
                mark(&present[c as usize], single);
            }
        })
    });
    // SAFETY: every chunk wrote each of its elements.
    let mut out = unsafe { assume_init(out) };
    let (new_id, next) = ranks(&present)?;
    if next < codes.n {
        pool.install(|| out.par_chunks_mut(CHUNK).for_each(|o| o.iter_mut().for_each(|c| *c = new_id[*c as usize])));
    }
    Ok(Codes { codes: out, n: next })
}

/// Encodes a host factor column of the `fit` layer over all its rows. Missing entries (non-finite
/// values, the integer NA sentinels, out-of-range codes) share one extra level; callers drop
/// those rows and [`restrict`] the codes to the estimation sample. Without missing entries the
/// codes are dense: numeric levels in ascending order, pre-coded levels in the host's order
/// (unused levels removed).
pub fn factorize_col(col: &FactorCol, pool: &Pool) -> Result<Codes, RfError> {
    match *col {
        FactorCol::F64(x) => {
            let finite = |v: f64| v.is_finite();
            if pool.install(|| x.par_chunks(CHUNK).all(|c| c.iter().all(|&v| finite(v)))) {
                return factorize_num(x, pool);
            }
            let parts = map_chunks(pool, x.len(), CHUNK, |_, r| {
                x[r].iter().filter(|v| v.is_finite()).fold((f64::INFINITY, f64::NEG_INFINITY, true), |(lo, hi, int), &v| {
                    (lo.min(v), hi.max(v), int && v == v.trunc() && v.abs() < 9.0e15)
                })
            });
            let (lo, hi, int) = parts.iter().fold((f64::INFINITY, f64::NEG_INFINITY, true), |a, b| (a.0.min(b.0), a.1.max(b.1), a.2 && b.2));
            if lo > hi {
                return direct_table(x.len(), 1, |_| 0, pool);
            }
            if int && hi - lo < table_limit(x.len()) as f64 {
                let slot = |i: usize| if x[i].is_finite() { (x[i] - lo) as i64 as usize + 1 } else { 0 };
                return direct_table(x.len(), (hi - lo) as usize + 2, slot, pool);
            }
            // Key 0 is the NaN bit pattern with every bit set, never the key of a finite value.
            let keys: Vec<u64> = pool.install(|| x.par_iter().map(|&v| if v.is_finite() { f64_key(v) } else { 0 }).collect());
            sort_encode(&keys, pool)
        }
        FactorCol::I32(x) => factorize_int(x.len(), |i| (x[i] != i32::MIN).then(|| i64::from(x[i])), pool),
        FactorCol::I64(x) => factorize_int(x.len(), |i| (x[i] != i64::MIN).then_some(x[i]), pool),
        FactorCol::Codes { codes, nlevels, base } => {
            let slot = |i: usize| {
                let l = i64::from(codes[i]) - i64::from(base);
                if (0..i64::from(nlevels)).contains(&l) { l as usize + 1 } else { 0 }
            };
            direct_table(codes.len(), nlevels as usize + 1, slot, pool)
        }
    }
}

/// [`factorize_col`] for integer values (`None` = missing): a direct table over the value range
/// when it is small, else sorted order-preserving keys (missing = key 0, which no value maps to
/// because the i64 sentinel `i64::MIN` is itself missing).
fn factorize_int(n: usize, value: impl Fn(usize) -> Option<i64> + Sync, pool: &Pool) -> Result<Codes, RfError> {
    let parts = map_chunks(pool, n, CHUNK, |_, r| r.filter_map(&value).fold((i64::MAX, i64::MIN), |(lo, hi), v| (lo.min(v), hi.max(v))));
    let (lo, hi) = parts.iter().fold((i64::MAX, i64::MIN), |a, b| (a.0.min(b.0), a.1.max(b.1)));
    if lo > hi {
        return direct_table(n, 1, |_| 0, pool);
    }
    let range = i128::from(hi) - i128::from(lo);
    if range < i128::from(table_limit(n)) {
        return direct_table(n, range as usize + 2, |i| value(i).map_or(0, |v| (v - lo) as usize + 1), pool);
    }
    let keys: Vec<u64> = pool.install(|| (0..n).into_par_iter().with_min_len(4096).map(|i| value(i).map_or(0, |v| (v ^ i64::MIN) as u64)).collect());
    sort_encode(&keys, pool)
}

/// Restricts `codes` to the observations with `keep[i]` and renumbers the surviving levels
/// densely, preserving their order.
pub fn compact(codes: &Codes, keep: &[bool]) -> Codes {
    // Nothing dropped: dense codes stay dense (every level occurs), so the codes are unchanged.
    if keep.len() == codes.codes.len() && keep.iter().all(|k| *k) {
        return codes.clone();
    }
    let mut new_id = vec![u32::MAX; codes.n as usize];
    for (&c, &k) in codes.codes.iter().zip(keep) {
        if k {
            new_id[c as usize] = 0;
        }
    }
    let mut next = 0u32;
    for id in new_id.iter_mut() {
        if *id == 0 {
            *id = next;
            next += 1;
        }
    }
    let kept = codes.codes.iter().zip(keep).filter(|(_, k)| **k).map(|(&c, _)| new_id[c as usize]).collect();
    Codes { codes: kept, n: next }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::par::with_pool;

    fn simple(keys: &[u64]) -> Codes {
        let mut u = keys.to_vec();
        u.sort_unstable();
        u.dedup();
        Codes { codes: keys.iter().map(|k| u.binary_search(k).unwrap() as u32).collect(), n: u.len() as u32 }
    }

    #[test]
    fn numeric_paths_agree_with_sorting() {
        with_pool(4, |pool| {
            let mut s = crate::rng::SplitMix64::new(3);
            let small: Vec<f64> = (0..50_000).map(|_| (s.next_u64() % 97) as f64 - 40.0).collect();
            let wide: Vec<f64> = (0..50_000).map(|_| (s.next_u64() % 1000) as f64 * 1e9).collect();
            let frac: Vec<f64> = (0..50_000).map(|_| (s.next_u64() % 300) as f64 * 0.25 - 7.0).collect();
            for x in [&small, &wide, &frac] {
                let got = factorize(&FactorValues::Num(x), pool).unwrap();
                let want = simple(&x.iter().map(|&v| f64_key(v)).collect::<Vec<_>>());
                assert_eq!(got, want);
            }
            let a = factorize(&FactorValues::Num(&small), pool).unwrap();
            let b = factorize(&FactorValues::Num(&frac), pool).unwrap();
            let got = combine(&[&a, &b], pool).unwrap();
            let want = simple(&(0..a.codes.len()).map(|i| u64::from(a.codes[i]) * 1000 + u64::from(b.codes[i])).collect::<Vec<_>>());
            assert_eq!(got, want);
            let nan = [1.0, f64::NAN];
            assert!(factorize(&FactorValues::Num(&nan), pool).is_err());
            let bytes = b"bbaab";
            let offsets = [0u32, 1, 2, 3, 4, 5];
            let s = factorize(&FactorValues::Str { bytes, offsets: &offsets }, pool).unwrap();
            assert_eq!(s, Codes { codes: vec![1, 1, 0, 0, 1], n: 2 });
        })
        .unwrap();
    }

    #[test]
    fn fast_paths_agree_with_sorting_for_any_threads() {
        let mut s = crate::rng::SplitMix64::new(5);
        let n = 100_000;
        let mut sorted: Vec<f64> = (0..n).map(|_| (s.next_u64() % 5000) as f64).collect();
        sorted.sort_by(f64::total_cmp);
        let mut late_descent = sorted.clone();
        late_descent[n - 1] = -1.0;
        let mut sorted_frac: Vec<f64> = (0..n).map(|_| (s.next_u64() % 700) as f64 * 0.3 - 9.0).collect();
        sorted_frac.sort_by(f64::total_cmp);
        let zeros = vec![-1.0, -0.0, 0.0, -0.0, 0.0, 2.5, 2.5, 7.0];
        // Table edges: x[0] at the top / bottom of a range just below the table limit.
        let lim = table_limit(n) as f64;
        let mut top: Vec<f64> = (0..n).map(|_| 5e6 - (s.next_u64() % lim as u64) as f64).collect();
        top[0] = 5e6;
        top[1] = 5e6 - lim + 1.0;
        let bottom: Vec<f64> = top.iter().map(|v| -v).collect();
        let mut over = top.clone();
        over[2] = 5e6 - lim;
        for x in [&sorted, &late_descent, &sorted_frac, &zeros, &top, &bottom, &over] {
            let want = simple(&x.iter().map(|&v| f64_key(v)).collect::<Vec<_>>());
            for threads in [1, 3, 8] {
                let got = with_pool(threads, |pool| factorize(&FactorValues::Num(x), pool).unwrap()).unwrap();
                assert_eq!(got, want);
            }
        }
        let mut nan_end = sorted.clone();
        nan_end[n - 1] = f64::NAN;
        for threads in [1, 3] {
            assert!(with_pool(threads, |pool| factorize(&FactorValues::Num(&nan_end), pool).is_err()).unwrap());
        }
        // restrict ≡ compact.
        let c = Codes { codes: (0..n).map(|_| (s.next_u64() % 3000) as u32).collect(), n: 3000 };
        let keep: Vec<bool> = (0..n).map(|_| s.next_u64() % 7 != 0).collect();
        let kept: Vec<u32> = (0..n as u32).filter(|&i| keep[i as usize]).collect();
        for threads in [1, 4] {
            assert_eq!(with_pool(threads, |pool| restrict(&c, &kept, pool).unwrap()).unwrap(), compact(&c, &keep));
        }
    }
}
