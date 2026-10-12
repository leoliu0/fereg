//! Fixed-effect values of the fit layer: fixest's normalization (`fixef()`, with
//! `attr(, "references")`, also used by `ssc(K.exact = TRUE)`) and pyfixest's (`fixef()`).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::{FitInput, FixefNorm, FixefValues};
use crate::session::Term;

/// fixest's `cpp_get_fe_gnl` (misc_funs.cpp) over `codes.len()` intercept-only terms on the
/// estimation rows (estimation order): the number of levels per term set to 0 as references
/// and, given the FE sum of every row (`sum`), the values of every level. The choice is
/// structural:
/// - pick an observation: the first one (lowest row) among those with the most terms already
///   determined (initially row 0);
/// - on it, the first undetermined term is left free and every other undetermined term's level
///   becomes a reference (0);
/// - propagate: an observation with all terms but one determined determines the remaining level
///   as its sum minus the other terms' values;
/// - repeat until every level is determined.
///
/// For two terms this is one reference in the second term per connected component.
fn get_fe_gnl(codes: &[&[u32]], sizes: &[usize], sum: Option<&[f64]>) -> (Vec<usize>, Vec<Vec<f64>>) {
    let q = codes.len();
    let mut refs = vec![0usize; q];
    let mut values: Vec<Vec<f64>> = sizes.iter().map(|&g| vec![0.0; if sum.is_some() { g } else { 0 }]).collect();
    if q == 0 {
        return (refs, values);
    }
    let n = codes[0].len();
    if q == 1 {
        if let Some(s) = sum {
            let mut seen = vec![false; sizes[0]];
            for (i, &c) in codes[0].iter().enumerate() {
                if !std::mem::replace(&mut seen[c as usize], true) {
                    values[0][c as usize] = s[i];
                }
            }
        }
        return (refs, values);
    }
    // Rows of each level, ascending (CSR per term).
    let lists: Vec<(Vec<u32>, Vec<u32>)> = codes
        .iter()
        .zip(sizes)
        .map(|(c, &g)| {
            let mut off = vec![0u32; g + 1];
            c.iter().for_each(|&v| off[v as usize + 1] += 1);
            for l in 0..g {
                off[l + 1] += off[l];
            }
            let mut pos = off.clone();
            let mut rows = vec![0u32; n];
            for (i, &v) in c.iter().enumerate() {
                rows[pos[v as usize] as usize] = i as u32;
                pos[v as usize] += 1;
            }
            (off, rows)
        })
        .collect();
    let mut done: Vec<Vec<bool>> = sizes.iter().map(|&g| vec![false; g]).collect();
    // Determined terms per row; rows with q − 1 determined wait in `queue`, rows with r ≤ q − 2
    // (r ≥ 1) in `buckets[r]` (lazy deletion: stale entries have a different count).
    let mut rs = vec![0u8; n];
    let mut queue: Vec<u32> = Vec::new();
    let mut buckets: Vec<BinaryHeap<Reverse<u32>>> = (0..q - 1).map(|_| BinaryHeap::new()).collect();
    let mark = |t: usize, l: usize, rs: &mut [u8], done: &mut [Vec<bool>], queue: &mut Vec<u32>, buckets: &mut [BinaryHeap<Reverse<u32>>]| {
        done[t][l] = true;
        let (off, rows) = &lists[t];
        for &i in &rows[off[l] as usize..off[l + 1] as usize] {
            rs[i as usize] += 1;
            let r = rs[i as usize] as usize;
            if r == q - 1 {
                queue.push(i);
            } else if r < q - 1 {
                buckets[r].push(Reverse(i));
            }
        }
    };
    let mut cursor = 0usize;
    loop {
        // Most-determined bucket first, lowest row within it.
        let mut pick = None;
        for r in (1..q - 1).rev() {
            while let Some(&Reverse(i)) = buckets[r].peek() {
                if rs[i as usize] as usize == r {
                    pick = Some(i as usize);
                    break;
                }
                buckets[r].pop();
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
                return (refs, values);
            }
            pick = Some(cursor);
        }
        let i = pick.expect("picked");
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
                mark(t, l, &mut rs, &mut done, &mut queue, &mut buckets);
            }
        }
        while let Some(i) = queue.pop() {
            let i = i as usize;
            if rs[i] as usize != q - 1 {
                continue;
            }
            let t = (0..q).find(|&t| !done[t][codes[t][i] as usize]).expect("one undetermined term");
            let l = codes[t][i] as usize;
            if let Some(s) = sum {
                let others: f64 = (0..q).filter(|&u| u != t).map(|u| values[u][codes[u][i] as usize]).sum();
                values[t][l] = s[i] - others;
            }
            mark(t, l, &mut rs, &mut done, &mut queue, &mut buckets);
        }
    }
}

/// Number of levels per term that fixest sets to 0 as references (see [`get_fe_gnl`]).
pub(crate) fn references(codes: &[&[u32]], sizes: &[usize]) -> Vec<usize> {
    get_fe_gnl(codes, sizes, None).0
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
/// level pinned at 0, as its `C(f1) + C(f2) + …` dummies without intercept) with the remaining
/// indeterminacy resolved by minimum norm, as its `lsqr` solve does. The indeterminacy is taken
/// to be spanned by the shifts v_{t,c} = +1 on the first term's levels and −1 on term t's levels
/// of each connected component c of the (first term, term t) level graph: each is in the null
/// space of the dummies (every row has both levels in c or neither), so the fit is unchanged,
/// and they span it for two terms and for the usual designs with more (e.g. worker, firm, year).
fn min_norm(codes: &[&[u32]], sizes: &[usize], vals: &mut [Vec<f64>]) {
    let q = codes.len();
    if q < 2 {
        return;
    }
    let g1 = sizes[0];
    // Per term t ≥ 1: component of every first-term level and of every term-t level, and the
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
    // Minimize ‖vals + Σ s_v v‖² over the shifts s: normal equations A s = b with
    // (A s)_{t,c} = Σ_{first-term l in c} S_l + n_{t,c}·s_{t,c}, S_l = Σ_t s_{t,c_t(l)}, and
    // b_{t,c} = Σ_{term-t m in c} vals_t[m] − Σ_{first-term l in c} vals_1[l]; Jacobi-preconditioned
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

/// Builds [`FixefValues`] from the absorber's per-term, per-slot values (`slots`: for each term
/// in `input.fe` order, the intercept slot if any, then one slot per slope) under `norm`.
/// Intercept terms are renormalized; slope coefficients are reported as solved.
pub(crate) fn fixef_values(input: &FitInput, terms: &[Term], mut slots: Vec<Vec<f64>>, kept_index: &[u32], norm: FixefNorm) -> Vec<FixefValues> {
    // Slot index of every intercept term.
    let mut slot = 0;
    let mut icpt: Vec<(usize, usize)> = Vec::new();
    for (q, t) in terms.iter().enumerate() {
        if t.intercept {
            icpt.push((q, slot));
        }
        slot += usize::from(t.intercept) + t.slopes.len();
    }
    let codes: Vec<&[u32]> = icpt.iter().map(|&(q, _)| terms[q].codes.as_slice()).collect();
    let sizes: Vec<usize> = icpt.iter().map(|&(q, _)| terms[q].g).collect();
    let mut refs = vec![0usize; icpt.len()];
    if icpt.len() >= 2 {
        match norm {
            FixefNorm::Fixest => {
                let n = codes[0].len();
                let sum: Vec<f64> = (0..n).map(|i| icpt.iter().zip(&codes).map(|(&(_, s), c)| slots[s][c[i] as usize]).sum()).collect();
                let (r, vals) = get_fe_gnl(&codes, &sizes, Some(&sum));
                refs = r;
                for (&(_, s), v) in icpt.iter().zip(vals) {
                    slots[s] = v;
                }
            }
            FixefNorm::MinNorm => {
                let mut vals: Vec<Vec<f64>> = icpt.iter().map(|&(_, s)| std::mem::take(&mut slots[s])).collect();
                min_norm(&codes, &sizes, &mut vals);
                for (&(_, s), v) in icpt.iter().zip(vals) {
                    slots[s] = v;
                }
                refs.iter_mut().zip(&sizes).skip(1).for_each(|(r, &g)| *r = usize::from(g > 0));
            }
        }
    }
    let mut out = Vec::new();
    let mut slots = slots.into_iter();
    let mut next_ref = refs.into_iter();
    for (q, (t, ft)) in terms.iter().zip(&input.fe).enumerate() {
        let mut first_obs = vec![u32::MAX; t.g];
        for (k, &c) in t.codes.iter().enumerate() {
            if first_obs[c as usize] == u32::MAX {
                first_obs[c as usize] = kept_index[k];
            }
        }
        if t.intercept {
            let values = slots.next().expect("intercept slot");
            out.push(FixefValues { term: q, slope: None, values, first_obs: first_obs.clone(), references: next_ref.next().expect("reference count") });
        }
        for (j, x) in ft.slopes.iter().zip(&t.slopes) {
            let values = slots.next().expect("slope slot");
            let references = constant_slope_levels(&t.codes, t.g, x, t.intercept);
            out.push(FixefValues { term: q, slope: Some(*j), values, first_obs: first_obs.clone(), references });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_terms_give_one_reference_per_component() {
        // Components {a0, b0, b1}, {a1, b2}, {a2, b3}.
        let a = [0u32, 0, 1, 2];
        let b = [0u32, 1, 2, 3];
        assert_eq!(references(&[&a, &b], &[3, 4]), vec![0, 3]);
        assert_eq!(references(&[&a], &[3]), vec![0]);
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
