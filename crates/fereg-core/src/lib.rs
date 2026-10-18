//! fereg-core: high-dimensional fixed-effects estimation (OLS, IV/2SLS, Poisson PML, logit).
//!
//! The solver eliminates the largest fixed effect exactly and runs preconditioned conjugate
//! gradient on cell-compressed sufficient statistics (see `structure`, `operator`, `cg`).
//! The Stata plugin's estimators (`ols`, `iv`, `glm`) follow reghdfe / ivreghdfe / ppmlhdfe
//! conventions; `fit` is the host-neutral layer of the Python and R packages (fixest / pyfixest
//! conventions).

#[cfg(test)]
extern crate self as fereg_core;

pub mod absorber;
pub mod api;
pub mod bias;
pub mod cg;
pub mod conley;
pub mod crv3;
pub mod dof;
pub mod error;
pub mod factor;
pub mod fe_recovery;
pub mod fit;
pub mod gather;
pub mod glm;
pub mod hac;
pub mod iv;
pub mod leverage;
pub mod ols;
pub mod operator;
pub mod par;
pub mod precond;
pub mod reference;
pub mod rng;
pub mod sample;
pub mod session;
pub mod slopes;
pub mod solve;
pub mod stats;
pub mod structure;
pub mod timing;
pub mod vcov;

pub use api::*;
pub use session::{AbsorberKind, ClusterDim, DropReport, Session, Term};

/// Crate version; the plugin reports it and every ado file checks it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
