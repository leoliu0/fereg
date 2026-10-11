//! Approximate Cholesky of a block Laplacian (plan D5 (b) 2) via `approx-chol` 0.6.0
//! (`Config::default()`, seed 0). Solves return the zero-mean solution per connected block.

use crate::error::RfError;
use crate::operator::Csr;

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

    /// Solves one column in place (`col` holds the kept coordinates of the block).
    pub fn solve_col(&self, col: &mut [f64]) {
        let mut work = vec![0.0; self.factor.n()];
        self.factor.solve_into(&col[..self.n], &mut work).expect("factor dimension");
        col[..self.n].copy_from_slice(&work[..self.n]);
    }
}
