//! Estimation sample (plan §1.3): reghdfe's cyclic singleton removal, extended with the GLM
//! `fe` separation rule.

use crate::api::Family;
use crate::par::Pool;

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
    _pool: &Pool,
) -> DropCounts {
    let mut out = DropCounts::default();
    let nterms = terms.len();
    if nterms == 0 || (!singletons && sep.is_none()) {
        return out;
    }
    let mut remaining = nterms;
    let mut i = 0usize;
    let mut count: Vec<f64> = Vec::new();
    let mut ysum: Vec<f64> = Vec::new();
    while remaining > 0 {
        let t = terms[i % nterms];
        i += 1;
        count.clear();
        count.resize(t.g, 0.0);
        if sep.is_some() {
            ysum.clear();
            ysum.resize(t.g, 0.0);
        }
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
        let mut dropped = false;
        for (obs, &c) in t.codes.iter().enumerate() {
            if !alive[obs] {
                continue;
            }
            let k = count[c as usize];
            if singletons && k == 1.0 {
                alive[obs] = false;
                out.singletons += 1;
                dropped = true;
            } else if let Some(rule) = sep {
                let s = ysum[c as usize];
                let separated = match rule.family {
                    Family::Poisson => s == 0.0,
                    Family::Logit => s == 0.0 || s == k,
                    Family::Gaussian => false,
                };
                if separated {
                    alive[obs] = false;
                    out.fe_separated += 1;
                    dropped = true;
                }
            }
        }
        remaining = if dropped { nterms - 1 } else { remaining - 1 };
    }
    out
}
