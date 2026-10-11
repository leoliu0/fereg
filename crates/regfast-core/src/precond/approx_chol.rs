//! Approximate Cholesky of a block Laplacian (plan D5 (b) 2) via `approx-chol` 0.6.0
//! (`Config::default()`, seed 0). Solves return the zero-mean solution per connected block.

use rayon::prelude::*;

use crate::error::RfError;
use crate::operator::Csr;
use crate::par::Pool;

pub struct ApproxChol {
    factor: approx_chol::Factor<f64>,
    n: usize,
}

impl ApproxChol {
    pub fn factor(a: &Csr) -> Result<ApproxChol, RfError> {
        let csr = approx_chol::CsrRef::new(&a.ptr, &a.col, &a.val, a.n as u32)
            .map_err(|e| RfError::Internal(format!("approx-chol input: {e}")))?;
        let factor = approx_chol::factorize(csr).map_err(|e| RfError::Internal(format!("approx-chol: {e}")))?;
        Ok(ApproxChol { factor, n: a.n })
    }

    /// Solves column by column (columns in parallel). `cols[j]` holds the kept coordinates of column j.
    pub fn solve(&self, cols: &mut [Vec<f64>], pool: &Pool) {
        pool.install(|| {
            cols.par_iter_mut().for_each(|b| {
                let mut work = vec![0.0; self.factor.n()];
                self.factor.solve_into(&b[..self.n], &mut work).expect("factor dimension");
                b[..self.n].copy_from_slice(&work[..self.n]);
            })
        });
    }
}
