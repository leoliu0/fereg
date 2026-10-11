//! Block-Jacobi preconditioner over the reduced system S (plan D5): an exact dense factor of
//! the small super-block, and per large block either Jacobi or (after escalation) an exact
//! sparse Cholesky or an approximate Cholesky of the block Laplacian.

pub mod approx_chol;
pub mod dense;
pub mod jacobi;
pub mod sparse_chol;

use std::ops::Range;

use faer::Mat;
use rayon::prelude::*;

use crate::par::Pool;
use crate::structure::Structure;

/// Right-hand-side columns per sparse-Cholesky solve task. Fixed (not derived from the thread
/// count) so that every column is solved with the same grouping at any thread count (D3).
pub const SOLVE_COLS: usize = 4;

/// (block, column range) solve tasks for the factored blocks among `large`: sparse-Cholesky
/// blocks in groups of [`SOLVE_COLS`] columns, approximate-Cholesky blocks one column each.
pub fn solve_tasks<T>(large: &[T], w: usize, kind: impl Fn(&T) -> Option<usize>) -> Vec<(usize, Range<usize>)> {
    let mut tasks = Vec::new();
    for (b, bp) in large.iter().enumerate() {
        if let Some(cols) = kind(bp) {
            tasks.extend((0..w.div_ceil(cols)).map(|c| (b, c * cols..((c + 1) * cols).min(w))));
        }
    }
    tasks
}

/// Preconditioner of one large block.
pub enum BlockPrecond {
    Jacobi,
    SparseChol { keep: Vec<u32>, f: Box<sparse_chol::SparseChol> },
    ApproxChol { keep: Vec<u32>, f: Box<approx_chol::ApproxChol> },
}

impl BlockPrecond {
    pub fn name(&self) -> &'static str {
        match self {
            BlockPrecond::Jacobi => "jacobi",
            BlockPrecond::SparseChol { .. } => "sparse-chol",
            BlockPrecond::ApproxChol { .. } => "approx-chol",
        }
    }
}

pub struct Precond {
    pub small: Option<dense::Dense>,
    /// 1/diag(S) (0 for redundant coordinates); used by Jacobi blocks.
    pub inv_diag: Vec<f64>,
    /// One entry per large block (blocks `ns..nb` of the structure).
    pub large: Vec<BlockPrecond>,
}

impl Precond {
    /// z = M⁻¹ r for row-major m × w blocks.
    pub fn apply(&self, st: &Structure, r: &[f64], z: &mut [f64], w: usize, pool: &Pool) {
        jacobi::apply(&self.inv_diag, r, z, w, pool);
        if let Some(d) = &self.small {
            d.apply(r, z, st.m_small, w, pool);
        }
        let ns = st.blocks.iter().filter(|b| b.small).count();
        // The factored blocks are independent: every (block, column group) is solved
        // sequentially in its own task, and the tasks run concurrently.
        let tasks = solve_tasks(&self.large, w, |bp| match bp {
            BlockPrecond::Jacobi => None,
            BlockPrecond::SparseChol { .. } => Some(SOLVE_COLS),
            BlockPrecond::ApproxChol { .. } => Some(1),
        });
        if tasks.is_empty() {
            return;
        }
        let sols: Vec<Vec<f64>> = pool.install(|| {
            tasks
                .par_iter()
                .map(|(b, cols)| {
                    let off = st.blocks[ns + b].off;
                    match &self.large[*b] {
                        BlockPrecond::SparseChol { keep, f } => {
                            let mut rhs =
                                Mat::<f64>::from_fn(keep.len(), cols.len(), |i, j| r[(off + keep[i] as usize) * w + cols.start + j]);
                            f.solve(&mut rhs);
                            let mut sol = Vec::with_capacity(keep.len() * cols.len());
                            for j in 0..cols.len() {
                                sol.extend((0..keep.len()).map(|i| rhs[(i, j)]));
                            }
                            sol
                        }
                        BlockPrecond::ApproxChol { keep, f } => {
                            let mut col: Vec<f64> = keep.iter().map(|&k| r[(off + k as usize) * w + cols.start]).collect();
                            f.solve_col(&mut col);
                            col
                        }
                        BlockPrecond::Jacobi => unreachable!("Jacobi blocks have no solve task"),
                    }
                })
                .collect()
        });
        for (b, bp) in self.large.iter().enumerate() {
            if !matches!(bp, BlockPrecond::Jacobi) {
                let blk = &st.blocks[ns + b];
                z[blk.off * w..(blk.off + blk.g) * w].iter_mut().for_each(|v| *v = 0.0);
            }
        }
        for ((b, cols), sol) in tasks.iter().zip(&sols) {
            let keep = match &self.large[*b] {
                BlockPrecond::SparseChol { keep, .. } | BlockPrecond::ApproxChol { keep, .. } => keep,
                BlockPrecond::Jacobi => unreachable!("Jacobi blocks have no solve task"),
            };
            let off = st.blocks[ns + b].off;
            for (j, c) in cols.clone().enumerate() {
                for (i, &k) in keep.iter().enumerate() {
                    z[(off + k as usize) * w + c] = sol[j * keep.len() + i];
                }
            }
        }
    }
}
