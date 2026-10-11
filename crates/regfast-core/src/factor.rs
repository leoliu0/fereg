//! Factor encoding (plan §1.2): dense codes 0..n in ascending value order.
//!
//! Numeric factors take a direct-table path when every value is an integer and the value range
//! is at most max(2n, 2²⁰); otherwise values are mapped to order-preserving u64 keys, sorted,
//! deduplicated and looked up by binary search. Every path is deterministic (the codes are a
//! function of the values only), whatever the thread count.

use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use crate::api::FactorValues;
use crate::error::RfError;
use crate::par::{CHUNK, Pool, alloc_vec, map_chunks};

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
    let parts = map_chunks(pool, n, CHUNK, |_, r| {
        let mut s = Scan { min: f64::INFINITY, max: f64::NEG_INFINITY, nan: false, int: true };
        for &v in &x[r] {
            if v.is_nan() {
                s.nan = true;
                continue;
            }
            s.min = s.min.min(v);
            s.max = s.max.max(v);
            s.int &= v == v.trunc() && v.abs() < 9.0e15;
        }
        s
    });
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
        let lo = s.min;
        let range = (s.max - s.min) as usize + 1;
        return direct_table(n, range, |i| (x[i] - lo) as usize, pool);
    }
    let keys: Vec<u64> = pool.install(|| x.par_iter().map(|&v| f64_key(v)).collect());
    sort_encode(&keys, pool)
}

/// Codes via a presence table over `0..range` (`slot(i)` is observation i's table index).
fn direct_table(n: usize, range: usize, slot: impl Fn(usize) -> usize + Sync, pool: &Pool) -> Result<Codes, RfError> {
    let present: Vec<AtomicBool> = (0..range).map(|_| AtomicBool::new(false)).collect();
    pool.install(|| {
        (0..n).into_par_iter().with_min_len(4096).for_each(|i| {
            let p = &present[slot(i)];
            if !p.load(Ordering::Relaxed) {
                p.store(true, Ordering::Relaxed);
            }
        })
    });
    let mut map = alloc_vec(range, 0u32)?;
    let mut next = 0u32;
    for (m, p) in map.iter_mut().zip(&present) {
        if p.load(Ordering::Relaxed) {
            *m = next;
            next += 1;
        }
    }
    drop(present);
    let codes = pool.install(|| (0..n).into_par_iter().with_min_len(4096).map(|i| map[slot(i)]).collect());
    Ok(Codes { codes, n: next })
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

/// Restricts `codes` to the observations with `keep[i]` and renumbers the surviving levels
/// densely, preserving their order.
pub fn compact(codes: &Codes, keep: &[bool]) -> Codes {
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
}
