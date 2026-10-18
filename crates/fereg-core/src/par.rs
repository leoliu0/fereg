//! Scoped thread pool and deterministic parallel primitives (plan D3, D4).
//!
//! Rules every kernel follows:
//! - Orchestration runs on the calling thread; only kernels enter [`Pool::install`].
//! - Work is split into fixed-size chunks ([`CHUNK`]) independent of the thread count, and
//!   per-chunk results are combined in chunk order, so results are bit-identical for any
//!   number of threads.
//! - Every thread fereg starts (pool workers, [`spawn_scoped`]) runs with the floating-point
//!   control state of the thread that starts it ([`FpEnv`]).

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
/// the unit of parallel work. A 16-thread scoped pool costs ~=0.2 ms to build and join, which
/// dominates small problems. Results do not depend on the thread count (D3), so capping is safe.
pub fn threads_for(requested: usize, work: usize) -> usize {
    resolve_threads(requested).min(work.div_ceil(CHUNK)).max(1)
}

/// Runs `body` on the calling thread with a scoped pool of `threads` workers (0 = all CPUs).
/// No worker outlives this call. Workers take the calling thread's [`FpEnv`].
pub fn with_pool<R>(threads: usize, body: impl FnOnce(&Pool) -> R) -> Result<R, RfError> {
    let env = FpEnv::current();
    rayon::ThreadPoolBuilder::new()
        .num_threads(resolve_threads(threads))
        .thread_name(|i| format!("fereg-{i}"))
        .build_scoped(
            move |thread| {
                env.apply();
                thread.run()
            },
            |pool| body(&Pool { pool }),
        )
        .map_err(|e| RfError::Internal(format!("cannot start thread pool: {e}")))
}

/// Spawns `f` on a new thread of the scope `s`, with the calling thread's [`FpEnv`].
pub(crate) fn spawn_scoped<'scope, T: Send + 'scope>(
    s: &'scope std::thread::Scope<'scope, '_>,
    f: impl FnOnce() -> T + Send + 'scope,
) -> std::thread::ScopedJoinHandle<'scope, T> {
    let env = FpEnv::current();
    s.spawn(move || {
        env.apply();
        f()
    })
}

/// Floating-point control state of a thread: the x87 control word and MXCSR (x86-64; nothing
/// elsewhere). POSIX threads inherit it from the thread that creates them; Windows threads start
/// from the system default instead (x87 precision 53 bits), which is not the state of the R main
/// thread (mingw-w64 programs start with 64-bit x87 precision). The C runtime's x87 code, such as
/// mingw-w64's `exp()` behind the logit link, then rounds differently on fereg's threads than on
/// the calling thread. Serial code runs on either (the calling thread is one of the CRV3 refit and
/// jackknife subpanel workers, and which subpanel it takes is a race), so without this state the
/// results of the R package on Windows depend on the thread count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FpEnv {
    #[cfg(target_arch = "x86_64")]
    x87: u16,
    #[cfg(target_arch = "x86_64")]
    mxcsr: u32,
}

impl FpEnv {
    /// The calling thread's state.
    pub(crate) fn current() -> FpEnv {
        #[cfg(target_arch = "x86_64")]
        {
            let mut x87 = 0u16;
            let mut mxcsr = 0u32;
            // SAFETY: both instructions only store a control register into the local.
            unsafe {
                std::arch::asm!("fnstcw word ptr [{}]", in(reg) &mut x87, options(nostack, preserves_flags));
                std::arch::asm!("stmxcsr dword ptr [{}]", in(reg) &mut mxcsr, options(nostack, preserves_flags));
            }
            FpEnv { x87, mxcsr }
        }
        #[cfg(not(target_arch = "x86_64"))]
        FpEnv {}
    }

    /// Makes this the calling thread's state.
    pub(crate) fn apply(self) {
        // SAFETY: the values were read from a thread of this process, so this thread's code runs
        // in a floating-point environment that code already runs in there.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::asm!("fldcw word ptr [{}]", in(reg) &self.x87, options(nostack, preserves_flags, readonly));
            std::arch::asm!("ldmxcsr dword ptr [{}]", in(reg) &self.mxcsr, options(nostack, preserves_flags, readonly));
        }
    }
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

/// Width of a kernel instance: the const `R`, or the runtime `r` when `R = 0`.
#[inline(always)]
pub(crate) fn width<const R: usize>(r: usize) -> usize {
    if R == 0 { r } else { R }
}

/// Calls `f::<R, B>(args)` with `R` = `$r` as a constant (1..=8; 0 = runtime width, see
/// [`width`]) and the const bool `B` = `$b`, so per-row loops over the columns are unrolled and
/// a flag-dependent branch (e.g. unit weights, where w*v = v exactly) disappears.
macro_rules! dispatch_width {
    ($r:expr, $b:expr, $f:ident($($a:expr),* $(,)?)) => {
        match ($r, $b) {
            (1, true) => $f::<1, true>($($a),*),
            (1, false) => $f::<1, false>($($a),*),
            (2, true) => $f::<2, true>($($a),*),
            (2, false) => $f::<2, false>($($a),*),
            (3, true) => $f::<3, true>($($a),*),
            (3, false) => $f::<3, false>($($a),*),
            (4, true) => $f::<4, true>($($a),*),
            (4, false) => $f::<4, false>($($a),*),
            (5, true) => $f::<5, true>($($a),*),
            (5, false) => $f::<5, false>($($a),*),
            (6, true) => $f::<6, true>($($a),*),
            (6, false) => $f::<6, false>($($a),*),
            (7, true) => $f::<7, true>($($a),*),
            (7, false) => $f::<7, false>($($a),*),
            (8, true) => $f::<8, true>($($a),*),
            (8, false) => $f::<8, false>($($a),*),
            (_, true) => $f::<0, true>($($a),*),
            (_, false) => $f::<0, false>($($a),*),
        }
    };
}
pub(crate) use dispatch_width;
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

/// `src` copied into `buf` when it has the same length (no allocation, its pages are already
/// mapped), else into a new buffer; parallel chunks either way.
pub fn copy_into(buf: Option<Vec<f64>>, src: &[f64], pool: &Pool) -> Result<Vec<f64>, RfError> {
    let mut out = match buf {
        Some(b) if b.len() == src.len() => b,
        _ => {
            let mut v = alloc_uninit(src.len())?;
            pool.install(|| v.par_chunks_mut(CHUNK).zip(src.par_chunks(CHUNK)).for_each(|(o, s)| {
                for (o, &x) in o.iter_mut().zip(s) {
                    *o = MaybeUninit::new(x);
                }
            }));
            // SAFETY: the zip covers every element (same length as `src`).
            return Ok(unsafe { assume_init(v) });
        }
    };
    pool.install(|| out.par_chunks_mut(CHUNK).zip(src.par_chunks(CHUNK)).for_each(|(o, s)| o.copy_from_slice(s)));
    Ok(out)
}

/// Every value is finite and strictly positive (valid observation weights).
pub fn all_positive(v: &[f64], pool: &Pool) -> bool {
    pool.install(|| v.par_chunks(CHUNK).all(|c| c.iter().all(|x| x.is_finite() && *x > 0.0)))
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

    /// Pool workers and scoped threads run with the creating thread's x87 precision (64 bits here,
    /// as on the R main thread on Windows, where new threads would start with 53 bits).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn threads_take_the_creators_floating_point_state() {
        let saved = FpEnv::current();
        let caller = FpEnv { x87: saved.x87 | 0x0300, ..saved };
        caller.apply();
        let here = FpEnv::current();
        let workers = with_pool(3, |pool| pool.rayon().broadcast(|_| FpEnv::current())).unwrap();
        let spawned = std::thread::scope(|s| spawn_scoped(s, FpEnv::current).join().unwrap());
        saved.apply();
        assert_eq!(here, caller);
        assert_eq!(workers, vec![caller; 3]);
        assert_eq!(spawned, caller);
    }
}
