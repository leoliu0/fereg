//! Block-Jacobi preconditioner over the reduced system S (plan D5): an exact dense factor of
//! the small super-block, and per large block either Jacobi or (after escalation) an exact
//! sparse Cholesky or an approximate Cholesky of the block Laplacian. Alternatively a single
//! exact sparse Cholesky of the joint S over all blocks (cross-block coupling included), which
//! replaces the block-Jacobi preconditioner entirely.

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
    /// `rebuild`: estimated cost of assembling and factoring the block again for new weights
    /// (single-thread seconds of the solver's deterministic cost model).
    SparseChol { keep: Vec<u32>, f: Box<sparse_chol::SparseChol>, rebuild: f64 },
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

/// Exact sparse Cholesky of the joint S over the kept (non-redundant) coordinates of all blocks.
pub struct Joint {
    /// Kept global coordinates (ascending).
    pub keep: Vec<u32>,
    pub f: Box<sparse_chol::SparseChol>,
}

pub struct Precond {
    pub small: Option<dense::Dense>,
    /// 1/diag(S) (0 for redundant coordinates); used by Jacobi blocks.
    pub inv_diag: Vec<f64>,
    /// One entry per large block (blocks `ns..nb` of the structure).
    pub large: Vec<BlockPrecond>,
    /// When set, M = the joint factor (the block preconditioners are unused).
    pub joint: Option<Joint>,
}

impl Precond {
    /// z = M^-1 r for row-major m x w blocks.
    pub fn apply(&self, st: &Structure, r: &[f64], z: &mut [f64], w: usize, pool: &Pool) {
        if let Some(j) = &self.joint {
            apply_joint(j, r, z, w, pool);
            return;
        }
        let has_jacobi_large = self.large.iter().any(|bp| matches!(bp, BlockPrecond::Jacobi));
        if let Some(d) = &self.small {
            if has_jacobi_large {
                let ms = st.m_small;
                jacobi::apply(&self.inv_diag[ms..], &r[ms * w..], &mut z[ms * w..], w, pool);
            }
            d.apply(r, z, st.m_small, w, pool);
        } else if has_jacobi_large || st.m_small > 0 {
            jacobi::apply(&self.inv_diag, r, z, w, pool);
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
        enum TaskSol {
            Mat(Mat<f64>),
            Col(Vec<f64>),
        }
        let solve_one = |(b, cols): &(usize, Range<usize>)| {
            let off = st.blocks[ns + *b].off;
            match &self.large[*b] {
                BlockPrecond::SparseChol { keep, f, .. } => {
                    let mut rhs =
                        Mat::<f64>::from_fn(keep.len(), cols.len(), |i, j| r[(off + keep[i] as usize) * w + cols.start + j]);
                    f.solve(&mut rhs);
                    TaskSol::Mat(rhs)
                }
                BlockPrecond::ApproxChol { keep, f } => {
                    let mut col: Vec<f64> = keep.iter().map(|&k| r[(off + k as usize) * w + cols.start]).collect();
                    f.solve_col(&mut col);
                    TaskSol::Col(col)
                }
                BlockPrecond::Jacobi => unreachable!("Jacobi blocks have no solve task"),
            }
        };
        let sols: Vec<TaskSol> = if tasks.len() <= 1 || pool.threads() <= 1 {
            tasks.iter().map(solve_one).collect()
        } else {
            pool.install(|| tasks.par_iter().map(solve_one).collect())
        };
        for (b, bp) in self.large.iter().enumerate() {
            let (BlockPrecond::SparseChol { keep, .. } | BlockPrecond::ApproxChol { keep, .. }) = bp else { continue };
            let blk = &st.blocks[ns + b];
            if keep.len() < blk.g {
                z[blk.off * w..(blk.off + blk.g) * w].fill(0.0);
            }
        }
        for ((b, cols), sol) in tasks.iter().zip(&sols) {
            let keep = match &self.large[*b] {
                BlockPrecond::SparseChol { keep, .. } | BlockPrecond::ApproxChol { keep, .. } => keep,
                BlockPrecond::Jacobi => unreachable!("Jacobi blocks have no solve task"),
            };
            let off = st.blocks[ns + b].off;
            match sol {
                TaskSol::Mat(rhs) => {
                    for (j, c) in cols.clone().enumerate() {
                        let col = rhs.col_as_slice(j);
                        for (&k, &v) in keep.iter().zip(col) {
                            z[(off + k as usize) * w + c] = v;
                        }
                    }
                }
                TaskSol::Col(col) => {
                    let c = cols.start;
                    for (&k, &v) in keep.iter().zip(col) {
                        z[(off + k as usize) * w + c] = v;
                    }
                }
            }
        }
    }
}

/// z = joint^-1 r on the kept coordinates (redundant ones get 0). Column groups of
/// [`SOLVE_COLS`] are solved sequentially in concurrent tasks (thread-count invariant).
fn apply_joint(j: &Joint, r: &[f64], z: &mut [f64], w: usize, pool: &Pool) {
    let keep = &j.keep;
    let groups: Vec<Range<usize>> = (0..w.div_ceil(SOLVE_COLS)).map(|c| c * SOLVE_COLS..((c + 1) * SOLVE_COLS).min(w)).collect();
    let solve_group = |cols: &Range<usize>| {
        let mut rhs = Mat::<f64>::from_fn(keep.len(), cols.len(), |i, jj| r[keep[i] as usize * w + cols.start + jj]);
        j.f.solve(&mut rhs);
        rhs
    };
    let sols: Vec<Mat<f64>> = if groups.len() <= 1 || pool.threads() <= 1 {
        groups.iter().map(solve_group).collect()
    } else {
        pool.install(|| groups.par_iter().map(solve_group).collect())
    };
    if keep.len() * w < z.len() {
        z.fill(0.0);
    }
    for (cols, sol) in groups.iter().zip(&sols) {
        for (jj, c) in cols.clone().enumerate() {
            let col = sol.col_as_slice(jj);
            for (&k, &v) in keep.iter().zip(col) {
                z[k as usize * w + c] = v;
            }
        }
    }
}
