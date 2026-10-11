//! Block-Jacobi preconditioner over the reduced system S (plan D5): an exact dense factor of
//! the small super-block, and per large block either Jacobi or (after escalation) an exact
//! sparse Cholesky or an approximate Cholesky of the block Laplacian.

pub mod approx_chol;
pub mod dense;
pub mod jacobi;
pub mod sparse_chol;

use faer::Mat;

use crate::par::Pool;
use crate::structure::Structure;

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
        for (b, bp) in self.large.iter().enumerate() {
            let off = st.blocks[ns + b].off;
            match bp {
                BlockPrecond::Jacobi => {}
                BlockPrecond::SparseChol { keep, f } => {
                    let mut rhs = Mat::<f64>::from_fn(keep.len(), w, |i, j| r[(off + keep[i] as usize) * w + j]);
                    f.solve(&mut rhs, pool);
                    let g = st.blocks[ns + b].g;
                    z[off * w..(off + g) * w].iter_mut().for_each(|v| *v = 0.0);
                    for (i, &k) in keep.iter().enumerate() {
                        for j in 0..w {
                            z[(off + k as usize) * w + j] = rhs[(i, j)];
                        }
                    }
                }
                BlockPrecond::ApproxChol { keep, f } => {
                    let mut cols: Vec<Vec<f64>> =
                        (0..w).map(|j| keep.iter().map(|&k| r[(off + k as usize) * w + j]).collect()).collect();
                    f.solve(&mut cols, pool);
                    let g = st.blocks[ns + b].g;
                    z[off * w..(off + g) * w].iter_mut().for_each(|v| *v = 0.0);
                    for (j, col) in cols.iter().enumerate() {
                        for (i, &k) in keep.iter().enumerate() {
                            z[(off + k as usize) * w + j] = col[i];
                        }
                    }
                }
            }
        }
    }
}
