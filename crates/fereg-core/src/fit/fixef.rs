//! Fixed-effect references in fixest's convention (`fixef()`, `attr(, "references")`, used by
//! `ssc(K.exact = TRUE)`).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Number of levels per term that fixest's `cpp_get_fe_gnl` (misc_funs.cpp) sets to 0 when it
/// recovers the coefficients of `codes.len()` intercept-only terms over the estimation rows (in
/// estimation order). The choice is structural:
/// - pick an observation: the first one (lowest row) among those with the most terms already
///   determined (initially row 0);
/// - on it, the first undetermined term is left free and every other undetermined term's level
///   becomes a reference (0);
/// - propagate: an observation with all terms but one determined determines the remaining level;
/// - repeat until every level is determined.
///
/// For two terms this is one reference in the second term per connected component.
pub(crate) fn references(codes: &[&[u32]], sizes: &[usize]) -> Vec<usize> {
    let q = codes.len();
    let mut refs = vec![0usize; q];
    if q < 2 {
        return refs;
    }
    let n = codes[0].len();
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
                return refs;
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
            if rs[i as usize] as usize != q - 1 {
                continue;
            }
            let t = (0..q).find(|&t| !done[t][codes[t][i as usize] as usize]).expect("one undetermined term");
            mark(t, codes[t][i as usize] as usize, &mut rs, &mut done, &mut queue, &mut buckets);
        }
    }
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
}
