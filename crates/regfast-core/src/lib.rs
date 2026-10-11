//! regfast-core: high-dimensional fixed-effects estimation (OLS, IV/2SLS, Poisson PML, logit).
//!
//! The solver eliminates the largest fixed effect exactly and runs preconditioned conjugate
//! gradient on cell-compressed sufficient statistics (see `structure`, `operator`, `cg`).
//! Numbers follow reghdfe / ivreghdfe / ppmlhdfe conventions.

pub mod absorber;
pub mod api;
pub mod cg;
pub mod dof;
pub mod error;
pub mod factor;
pub mod fe_recovery;
pub mod gather;
pub mod glm;
pub mod iv;
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
