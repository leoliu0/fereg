//! Scoped thread pool and deterministic parallel primitives (plan D3, D4).
//!
//! Rules every kernel follows:
//! - Orchestration runs on the calling thread; only kernels enter [`Pool::install`].
//! - Work is split into fixed-size chunks ([`CHUNK`]) independent of the thread count, and
//!   per-chunk results are combined in chunk order, so results are bit-identical for any
//!   number of threads.

use std::mem::MaybeUninit;
use std::ops::Range;

use rayon::prelude::*;

use crate::error::RfError;

/// Fixed reduction/scatter chunk size (rows or cells).
pub const CHUNK: usize = 16_384;

/// A live scoped worker pool. Every worker is joined when [`with_pool`] returns.
pub struct Pool<'a> {
    pool: &'a rayon::ThreadPool,
}

impl Pool<'_> {
    pub fn threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Runs `f` inside the pool (rayon parallel iterators inside `f` use these workers).
    pub fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        self.pool.install(f)
    }

    /// The underlying rayon pool, for APIs that need it (e.g. `faer::Par::rayon`).
    pub fn rayon(&self) -> &rayon::ThreadPool {
        self.pool
    }
}

/// Resolves a requested thread count (0 = all logical CPUs).
pub fn resolve_threads(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        threads
    }
}

/// Threads worth starting for `work` rows or cells: at most one per [`CHUNK`], since chunks are
/// the unit of parallel work. A 16-thread scoped pool costs ≈0.2 ms to build and join, which
/// dominates small problems. Results do not depend on the thread count (D3), so capping is safe.
pub fn threads_for(requested: usize, work: usize) -> usize {
    resolve_threads(requested).min(work.div_ceil(CHUNK)).max(1)
}

/// Runs `body` on the calling thread with a scoped pool of `threads` workers (0 = all CPUs).
/// No worker outlives this call.
pub fn with_pool<R>(threads: usize, body: impl FnOnce(&Pool) -> R) -> Result<R, RfError> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(resolve_threads(threads))
        .thread_name(|i| format!("regfast-{i}"))
        .build_scoped(|thread| thread.run(), |pool| body(&Pool { pool }))
        .map_err(|e| RfError::Internal(format!("cannot start thread pool: {e}")))
}

/// Number of chunks of size `chunk` covering `0..n`.
pub fn n_chunks(n: usize, chunk: usize) -> usize {
    n.div_ceil(chunk)
}

/// Range of chunk `c`.
pub fn chunk_range(c: usize, n: usize, chunk: usize) -> Range<usize> {
    let lo = c * chunk;
    lo..(lo + chunk).min(n)
}

/// Maps every chunk of `0..n` in parallel; the results are returned in chunk order.
pub fn map_chunks<T: Send>(
    pool: &Pool,
    n: usize,
    chunk: usize,
    f: impl Fn(usize, Range<usize>) -> T + Sync,
) -> Vec<T> {
    let nc = n_chunks(n, chunk);
    pool.install(|| (0..nc).into_par_iter().map(|c| f(c, chunk_range(c, n, chunk))).collect())
}

/// Deterministic sum of a length-`k` accumulator: `f` adds the contribution of a chunk into a
/// zeroed buffer; chunk buffers are combined in chunk order.
pub fn sum_chunks(
    pool: &Pool,
    n: usize,
    chunk: usize,
    k: usize,
    f: impl Fn(Range<usize>, &mut [f64]) + Sync,
) -> Vec<f64> {
    let parts = map_chunks(pool, n, chunk, |_, range| {
        let mut acc = vec![0.0; k];
        f(range, &mut acc);
        acc
    });
    let mut out = vec![0.0; k];
    for part in parts {
        for (o, v) in out.iter_mut().zip(part) {
            *o += v;
        }
    }
    out
}

/// Fallible allocation of `n` copies of `value` (plan D4: large buffers return r(909) on failure).
pub fn alloc_vec<T: Clone>(n: usize, value: T) -> Result<Vec<T>, RfError> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| RfError::OutOfMemory)?;
    v.resize(n, value);
    Ok(v)
}

/// Fallible buffer of `n` uninitialized values (no zeroing pass): kernels write every element,
/// typically in parallel (so the first touch of the pages is spread over the workers), then
/// [`assume_init`] it.
pub fn alloc_uninit<T>(n: usize) -> Result<Vec<MaybeUninit<T>>, RfError> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| RfError::OutOfMemory)?;
    // SAFETY: the capacity is at least n, and `MaybeUninit` values need no initialization.
    unsafe { v.set_len(n) };
    Ok(v)
}

/// The values of a fully written [`alloc_uninit`] buffer.
///
/// # Safety
/// Every element of `v` must have been written.
pub unsafe fn assume_init<T>(v: Vec<MaybeUninit<T>>) -> Vec<T> {
    let mut v = std::mem::ManuallyDrop::new(v);
    // SAFETY: same allocation, capacity and length; `MaybeUninit<T>` has the layout of `T`, and
    // the caller guarantees that every element is initialized.
    unsafe { Vec::from_raw_parts(v.as_mut_ptr().cast::<T>(), v.len(), v.capacity()) }
}

/// [`alloc_vec`] filled in parallel chunks instead of serially on the calling thread.
pub fn alloc_par<T: Copy + Send + Sync>(n: usize, value: T, pool: &Pool) -> Result<Vec<T>, RfError> {
    let mut v = alloc_uninit(n)?;
    if n <= CHUNK {
        v.fill(MaybeUninit::new(value));
    } else {
        pool.install(|| v.par_chunks_mut(CHUNK).for_each(|c| c.fill(MaybeUninit::new(value))));
    }
    // SAFETY: every element was written above.
    Ok(unsafe { assume_init(v) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_chunks_is_thread_count_invariant() {
        let x: Vec<f64> = (0..100_003).map(|i| ((i * 7919) % 1000) as f64 * 1e-3 + 1e-9 * i as f64).collect();
        let run = |t| {
            with_pool(t, |pool| {
                sum_chunks(pool, x.len(), 1000, 2, |r, acc| {
                    for &v in &x[r] {
                        acc[0] += v;
                        acc[1] += v * v;
                    }
                })
            })
            .unwrap()
        };
        let a = run(1);
        assert_eq!(a, run(3));
        assert_eq!(a, run(8));
    }
}
