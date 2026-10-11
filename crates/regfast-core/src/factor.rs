//! Factor encoding (plan §1.2): dense codes 0..n in ascending value order.
//!
//! Simple sort-based implementation; CoreSolver replaces it with the parallel direct-table /
//! sort paths of §1.2 keeping these signatures.

use crate::api::FactorValues;
use crate::error::RfError;
use crate::par::Pool;

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

/// Encodes a factor. Numeric codes follow numeric order; strings follow byte-lexicographic order.
pub fn factorize(v: &FactorValues, _pool: &Pool) -> Result<Codes, RfError> {
    match *v {
        FactorValues::Num(x) => {
            if x.iter().any(|v| v.is_nan()) {
                return Err(RfError::InvalidInput("missing values in factor".into()));
            }
            let keys: Vec<u64> = x.iter().map(|&v| f64_key(v)).collect();
            Ok(encode_keys(&keys))
        }
        FactorValues::Str { bytes, offsets } => {
            let n = offsets.len().saturating_sub(1);
            let get = |i: usize| &bytes[offsets[i] as usize..offsets[i + 1] as usize];
            let mut uniq: Vec<&[u8]> = (0..n).map(get).collect();
            uniq.sort_unstable();
            uniq.dedup();
            let codes = (0..n).map(|i| uniq.binary_search(&get(i)).expect("level present") as u32).collect();
            Ok(Codes { codes, n: uniq.len() as u32 })
        }
    }
}

fn encode_keys(keys: &[u64]) -> Codes {
    let mut uniq = keys.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    let codes = keys.iter().map(|k| uniq.binary_search(k).expect("level present") as u32).collect();
    Codes { codes, n: uniq.len() as u32 }
}

/// Interaction of several factors (`a#b#c`): codes follow lexicographic order of (a, b, c)
/// over the combinations that occur.
pub fn combine(parts: &[&Codes], _pool: &Pool) -> Result<Codes, RfError> {
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
            let keys: Vec<u64> = (0..n)
                .map(|i| parts.iter().fold(0u64, |acc, p| acc * u64::from(p.n) + u64::from(p.codes[i])))
                .collect();
            Ok(encode_keys(&keys))
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
