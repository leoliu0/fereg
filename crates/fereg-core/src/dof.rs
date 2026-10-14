//! Degrees of freedom absorbed by the fixed effects: reghdfe 6.13.1 `estimate_dof`
//! (reghdfe.mata l.4187-4440; plan Phase 3.5).
//!
//! Slots ("extended absvars"): one per term intercept, then one per slope, in absorb order;
//! a slot has K = number of levels of its term. Under `noabsorb` there is one slot `_cons`
//! with K = 1 (reghdfe absorbs a one-level `_cons` factor).

use crate::api::DofOptions;
use crate::session::{ClusterDim, Term};
use crate::stats::stdev;

/// One row of `e(dof_table)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DofSlot {
    /// Categories (levels of the term).
    pub k: u64,
    /// Redundant coefficients.
    pub m: u64,
    pub exact: bool,
    pub nested: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dof {
    pub slots: Vec<DofSlot>,
    pub df_a: u64,
    pub df_a_initial: u64,
    pub df_a_redundant: u64,
    pub df_a_nested: u64,
    /// Mobility group (1-based) of each kept observation for the first pair of remaining intercept
    /// terms, when requested and such a pair exists.
    pub groupvar: Option<Vec<u32>>,
}

impl Dof {
    /// `e(dof_table)` rows: (K, M, K-M, inexact, nested), row-major n_slotsx5.
    pub fn table(&self) -> Vec<f64> {
        self.slots
            .iter()
            .flat_map(|s| [s.k as f64, s.m as f64, (s.k - s.m) as f64, f64::from(u8::from(!s.exact)), f64::from(u8::from(s.nested))])
            .collect()
    }
}

/// Term view used by the dof rules (owned codes for the virtual `_cons` term).
struct DofTerm<'a> {
    codes: std::borrow::Cow<'a, [u32]>,
    g: usize,
    intercept: bool,
    slopes: &'a [Vec<f64>],
}

/// reghdfe `estimate_dof`. `clusters` are the cluster dimensions of a cluster VCE (empty
/// otherwise); `n` is the number of kept observations. With `reghdfe5` (ppmlhdfe 2.3.3, which
/// runs reghdfe5's `estimate_dof`), the "first remaining intercept is exact" flag goes to the
/// slot whose position equals that term's position (reghdfe5.mata l.1446 indexes the extended
/// list with the absvar index), which differs from the term's intercept slot when slope slots
/// precede it; only the `e(dof_table)` flags change.
pub fn estimate_dof(terms: &[Term], clusters: &[ClusterDim], n: usize, opts: &DofOptions, want_groupvar: bool, reghdfe5: bool) -> Dof {
    let dterms: Vec<DofTerm> = if terms.is_empty() {
        vec![DofTerm { codes: std::borrow::Cow::Owned(vec![0; n]), g: 1, intercept: true, slopes: &[] }]
    } else {
        terms
            .iter()
            .map(|t| DofTerm { codes: std::borrow::Cow::Borrowed(&t.codes), g: t.g, intercept: t.intercept, slopes: &t.slopes })
            .collect()
    };

    // Slots and the intercept index (term of each intercept slot, None for slope slots).
    let mut slots = Vec::new();
    let mut intercept_index: Vec<Option<usize>> = Vec::new();
    let mut slope_slot_of: Vec<Option<usize>> = Vec::new(); // first slope slot of each term
    for (q, t) in dterms.iter().enumerate() {
        if t.intercept {
            slots.push(DofSlot { k: t.g as u64, m: 0, exact: false, nested: false });
            intercept_index.push(Some(q));
        }
        slope_slot_of.push((!t.slopes.is_empty()).then_some(slots.len()));
        for _ in t.slopes {
            slots.push(DofSlot { k: t.g as u64, m: 0, exact: false, nested: false });
            intercept_index.push(None);
        }
    }
    let mut df_a_nested = 0u64;

    // (a) Intercept terms that are cluster variables or nested within one (dof_update_nested).
    // reghdfe first matches absvar names against clustervar names, then tests nesting per
    // cluster dimension; equal partitions are nested, so a single nesting test per pair gives
    // the same set.
    if !clusters.is_empty() && opts.clusters {
        for c in clusters {
            if intercept_index.iter().all(Option::is_none) {
                break;
            }
            for (i, slot) in slots.iter_mut().enumerate() {
                let Some(q) = intercept_index[i] else { continue };
                if slot.exact {
                    continue;
                }
                if nested_within(&dterms[q].codes, dterms[q].g, &c.codes) {
                    slot.m = slot.k;
                    slot.exact = true;
                    slot.nested = true;
                    df_a_nested += slot.k;
                    intercept_index[i] = None;
                }
            }
        }
    }

    // (b) Every remaining intercept but the first has at least one redundant coefficient
    // (including the first when some FE is nested within a cluster).
    let remaining = intercept_index.iter().filter(|x| x.is_some()).count();
    if remaining > 1 || (remaining > 0 && df_a_nested > 0) {
        let mut skip_next = df_a_nested == 0;
        for (i, slot) in slots.iter_mut().enumerate() {
            if intercept_index[i].is_none() {
                continue;
            }
            if skip_next {
                skip_next = false;
                continue;
            }
            slot.m = 1;
        }
    }

    // (c) The first remaining intercept is exact.
    if let Some(i) = intercept_index.iter().position(Option::is_some) {
        let at = if reghdfe5 { intercept_index[i].expect("intercept slot") } else { i };
        slots[at].exact = true;
    }

    // (d) Mobility groups between pairs of remaining intercepts.
    let mut groupvar = None;
    if opts.pairwise || opts.firstpair {
        let idx: Vec<(usize, usize)> = intercept_index.iter().enumerate().filter_map(|(i, q)| q.map(|q| (i, q))).collect();
        let mut pair_count = 0;
        'outer: for a in 0..idx.len() {
            for b in a + 1..idx.len() {
                let (_, qa) = idx[a];
                let (j, qb) = idx[b];
                pair_count += 1;
                let save = pair_count == 1 && want_groupvar;
                let (m, gv) = mobility_groups(&dterms[qa].codes, dterms[qa].g, &dterms[qb].codes, dterms[qb].g, save);
                if save {
                    groupvar = gv;
                }
                slots[j].m = slots[j].m.max(m);
                if pair_count == 1 {
                    slots[j].exact = true;
                }
                if opts.firstpair {
                    break 'outer;
                }
            }
        }
    }

    // (e) Slopes that are constant (term with intercept) or zero (term without) within a level.
    if opts.continuous {
        for (q, t) in dterms.iter().enumerate() {
            let Some(first) = slope_slot_of[q] else { continue };
            for (s, x) in t.slopes.iter().enumerate() {
                let cnt = redundant_slope_levels(&t.codes, t.g, x, t.intercept);
                if cnt > 0 {
                    slots[first + s].m = cnt;
                }
            }
        }
    }

    let df_a_initial: u64 = slots.iter().map(|s| s.k).sum();
    let df_a_redundant: u64 = slots.iter().map(|s| s.m).sum();
    Dof { slots, df_a: df_a_initial - df_a_redundant, df_a_initial, df_a_redundant, df_a_nested, groupvar }
}

/// ftools `Factor::nested_within`: every level of the term maps to a single cluster value.
pub(crate) fn nested_within(codes: &[u32], g: usize, cluster: &[u32]) -> bool {
    let mut map = vec![u32::MAX; g];
    for (&c, &k) in codes.iter().zip(cluster) {
        let slot = &mut map[c as usize];
        if *slot == u32::MAX {
            *slot = k;
        } else if *slot != k {
            return false;
        }
    }
    true
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let p = parent[x as usize];
        parent[x as usize] = parent[p as usize];
        x = parent[x as usize];
    }
    x
}

/// Connected components of the bipartite graph between the levels of two terms (reghdfe
/// `BipartiteGraph::init_zigzag`, via union-find). With `save`, also returns the component of
/// each observation numbered 1.. in order of the smallest first-term level of each component
/// (the order in which reghdfe's zigzag search discovers them).
pub(crate) fn mobility_groups(a: &[u32], ga: usize, b: &[u32], gb: usize, save: bool) -> (u64, Option<Vec<u32>>) {
    let mut parent: Vec<u32> = (0..(ga + gb) as u32).collect();
    // The root of the previous observation's first-term level is cached (panels sorted by the
    // first term repeat it); it stays a root until a union attaches it, which updates it.
    let (mut last, mut rx) = (u32::MAX, 0u32);
    for (&x, &y) in a.iter().zip(b) {
        if x != last {
            last = x;
            rx = find(&mut parent, x);
        }
        let ry = find(&mut parent, ga as u32 + y);
        if rx != ry {
            // Deterministic: attach the larger root under the smaller.
            let (lo, hi) = if rx < ry { (rx, ry) } else { (ry, rx) };
            parent[hi as usize] = lo;
            rx = lo;
        }
    }
    let mut id = vec![0u32; ga + gb];
    let mut next = 0u32;
    for l in 0..ga as u32 {
        let r = find(&mut parent, l) as usize;
        if id[r] == 0 {
            next += 1;
            id[r] = next;
        }
    }
    let gv = save.then(|| a.iter().map(|&x| id[find(&mut parent, x) as usize]).collect());
    (u64::from(next), gv)
}

/// reghdfe `dof_update_cvars` for one slope: the variable is standardized (reghdfe
/// `standardize_data`), then a level is redundant when max - min <= 1e-6 (term with intercept) or
/// |min| + |max| <= 1e-6 (term without; reghdfe sums `abs(colminmax())`, not sum|x| as the plan says).
pub(crate) fn redundant_slope_levels(codes: &[u32], g: usize, x: &[f64], intercept: bool) -> u64 {
    let sd = stdev(x);
    let mut lo = vec![f64::INFINITY; g];
    let mut hi = vec![f64::NEG_INFINITY; g];
    for (&c, &v) in codes.iter().zip(x) {
        let v = v / sd;
        let c = c as usize;
        lo[c] = lo[c].min(v);
        hi[c] = hi[c].max(v);
    }
    lo.iter()
        .zip(&hi)
        .filter(|(l, h)| if intercept { *h - *l <= 1e-6 } else { l.abs() + h.abs() <= 1e-6 })
        .count() as u64
}
