//! Error type and its mapping to Stata return codes.

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum RfError {
    /// Bad input or unsupported request (r(198)).
    InvalidInput(String),
    /// Iteration cap reached (r(430)).
    Convergence(usize),
    /// An allocation failed (r(909)).
    OutOfMemory,
    /// Fewer than two observations remain (r(2001)).
    TooFewObs,
    /// The user pressed Break (r(1)).
    Interrupted,
    /// A bug (r(9999)).
    Internal(String),
    /// A specific Stata return code with its message (e.g. 481, 459).
    Code(i32, String),
}

impl RfError {
    /// Stata return code.
    pub fn rc(&self) -> i32 {
        match self {
            RfError::InvalidInput(_) => 198,
            RfError::Convergence(_) => 430,
            RfError::OutOfMemory => 909,
            RfError::TooFewObs => 2001,
            RfError::Interrupted => 1,
            RfError::Internal(_) => 9999,
            RfError::Code(rc, _) => *rc,
        }
    }
}

impl fmt::Display for RfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RfError::InvalidInput(msg) => write!(f, "{msg}"),
            RfError::Convergence(iter) => write!(f, "convergence not achieved in {iter} iterations"),
            RfError::OutOfMemory => write!(f, "regfast: insufficient memory"),
            RfError::TooFewObs => write!(f, "insufficient observations"),
            RfError::Interrupted => write!(f, "--Break--"),
            RfError::Internal(msg) => write!(f, "regfast internal error: {msg}"),
            RfError::Code(_, msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for RfError {}
