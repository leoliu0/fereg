//! The session kept between `prepare` and an estimation command (plan §4.4).

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use regfast_core::{RfError, Session};

/// A prepared estimation sample.
pub struct Stored {
    pub token: u64,
    /// Stata observation (1-based) of every input observation given to `Session::prepare`.
    pub obs: Vec<i32>,
    /// `_N` at prepare time (guards against data changes between the calls).
    pub nobs: usize,
    /// Kept observations at prepare time; an estimator that drops more (GLM separation) makes
    /// [`crate::est::finish`] zero touse for them.
    pub n_kept: usize,
    pub session: Session,
    /// The model variables read by `prepare` (protocol `nmodel`), row-major: row k holds their
    /// `nmodel` values at `obs[k]`; estimation commands gather the variables they map to them
    /// (`cached=`) instead of reading them again.
    pub model: Vec<f64>,
    pub nmodel: usize,
}

static SESSION: Mutex<Option<Stored>> = Mutex::new(None);
static COUNTER: AtomicU64 = AtomicU64::new(0);

fn slot() -> std::sync::MutexGuard<'static, Option<Stored>> {
    SESSION.lock().unwrap_or_else(|e| e.into_inner())
}

/// Discards any stored session (frees its memory).
pub fn clear() {
    let old = slot().take();
    drop(old);
}

/// A fresh token: time-seeded, unique within the process, below 2^53 (exact as a Stata number).
pub fn new_token() -> u64 {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    let c = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut x = nanos ^ c.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    // SplitMix64 finaliser.
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x & ((1u64 << 53) - 1)).max(1)
}

pub fn put(s: Stored) {
    *slot() = Some(s);
}

/// Takes (consumes) the stored session if `token` matches; any stored session is discarded.
pub fn take(token: &str) -> Result<Stored, RfError> {
    let stale = || RfError::InvalidInput("regfast: stale plugin session; rerun the command".into());
    let stored = slot().take().ok_or_else(stale)?;
    if token.parse::<u64>().ok() != Some(stored.token) {
        return Err(stale());
    }
    if crate::spi::nobs() != stored.nobs {
        return Err(stale());
    }
    Ok(stored)
}
