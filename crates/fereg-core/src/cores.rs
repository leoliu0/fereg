//! Worker placement on Linux: one pool worker per physical core.
//!
//! The scheduler wakes a thread near its waker and stays within the waker's last-level-cache
//! domain: there it takes an idle hardware thread of a busy core before an idle core of another
//! domain. On CPUs with several domains (AMD Zen with two or more CCDs, several sockets) the
//! workers of a pool so share cores after every wake-up while other cores idle, until the load
//! balancer moves them milliseconds later, and every kernel with one fixed task per worker waits
//! for the slowest one. Measured on a Ryzen 9 9950X3D (2 x 8 cores) with 16 workers: a matvec
//! task took 2.2-2.4 ms alone on its core and 3.5-3.7 ms next to another worker, and whole
//! estimations ran 5-25% longer than with one worker bound to each core (bench/TUNING.md).
//!
//! A pool with exactly as many workers as the calling thread may use physical cores therefore
//! binds worker i to the hardware threads of core i ([`worker_sets`]). One worker per core
//! leaves no choice of cores: another workload shares a core with exactly one worker, as it
//! would under any placement. Pools of every other size are left to the scheduler (choosing
//! some of the cores would stack the workers of concurrent processes on the same ones). The
//! calling thread is never bound, and the workers end with the pool. `FEREG_PIN=0` in the
//! environment turns the binding off. The module is compiled on 64-bit Linux only.

use std::sync::LazyLock;

// glibc and musl; `cpu_set_t` is an array of `unsigned long`.
unsafe extern "C" {
    fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut u64) -> i32;
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
}

/// A set of logical CPUs: the kernel's `cpu_set_t` (1024 bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CpuSet([u64; CpuSet::BITS / 64]);

impl CpuSet {
    const BITS: usize = 1024;
    const EMPTY: CpuSet = CpuSet([0; CpuSet::BITS / 64]);

    fn contains(&self, cpu: usize) -> bool {
        self.0[cpu / 64] >> (cpu % 64) & 1 == 1
    }

    fn insert(&mut self, cpu: usize) {
        self.0[cpu / 64] |= 1 << (cpu % 64);
    }

    /// The CPUs the calling thread may run on.
    fn allowed() -> Option<CpuSet> {
        let mut set = CpuSet::EMPTY;
        // SAFETY: the buffer holds the `size_of_val(&set.0)` bytes the call may write.
        let rc = unsafe { sched_getaffinity(0, size_of_val(&set.0), set.0.as_mut_ptr()) };
        (rc == 0).then_some(set)
    }

    /// Restricts the calling thread to this set (best effort: a refusal leaves it where it is).
    pub(crate) fn bind(&self) {
        // SAFETY: the call reads `size_of_val(&self.0)` bytes of the set.
        unsafe { sched_setaffinity(0, size_of_val(&self.0), self.0.as_ptr()) };
    }
}

const UNKNOWN: u16 = u16::MAX;

/// For every logical CPU the first hardware thread of its core ([`UNKNOWN`] where sysfs does not
/// say); `None` when `FEREG_PIN=0`. Read once per process.
static CORE_OF: LazyLock<Option<Vec<u16>>> = LazyLock::new(|| {
    if std::env::var_os("FEREG_PIN").is_some_and(|v| v == "0") {
        return None;
    }
    let core_of = (0..CpuSet::BITS)
        .map(|cpu| {
            // "0,16" or "0-1": the hardware threads of the core, ascending.
            std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"))
                .ok()
                .and_then(|s| s.trim().split([',', '-']).next().and_then(|first| first.parse::<u16>().ok()))
                .filter(|&first| usize::from(first) < CpuSet::BITS)
                .unwrap_or(UNKNOWN)
        })
        .collect();
    Some(core_of)
});

/// The allowed CPUs of every physical core the calling thread may use, cores in ascending order
/// of their first allowed CPU; `None` when the topology is unknown or binding is off.
fn core_sets() -> Option<Vec<CpuSet>> {
    let core_of = CORE_OF.as_ref()?;
    let allowed = CpuSet::allowed()?;
    let mut slot = [UNKNOWN; CpuSet::BITS];
    let mut sets: Vec<CpuSet> = Vec::new();
    for cpu in (0..CpuSet::BITS).filter(|&cpu| allowed.contains(cpu)) {
        let core = usize::from(core_of[cpu]);
        if core >= CpuSet::BITS {
            return None;
        }
        if slot[core] == UNKNOWN {
            slot[core] = sets.len() as u16;
            sets.push(CpuSet::EMPTY);
        }
        sets[usize::from(slot[core])].insert(cpu);
    }
    Some(sets)
}

/// The CPU sets of the workers of a pool of `threads` workers, one physical core each, when the
/// calling thread may use exactly `threads` physical cores; `None` = no binding.
pub(crate) fn worker_sets(threads: usize) -> Option<Vec<CpuSet>> {
    if threads < 2 {
        return None;
    }
    core_sets().filter(|sets| sets.len() == threads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::par::with_pool;

    fn workers(threads: usize) -> Vec<CpuSet> {
        with_pool(threads, |pool| pool.rayon().broadcast(|_| CpuSet::allowed().expect("affinity"))).unwrap()
    }

    /// A pool with one worker per core binds each worker to its own core; any other size, and
    /// the calling thread, keep the caller's CPUs.
    #[test]
    fn one_worker_per_core_only_when_the_pool_has_as_many_workers_as_cores() {
        let here = CpuSet::allowed().expect("affinity");
        // No topology in sysfs (some sandboxes), or FEREG_PIN=0: nothing to check.
        let Some(cores) = core_sets() else { return };
        let ncores = cores.len();
        if ncores >= 2 {
            assert_eq!(workers(ncores), cores);
            let mut union = CpuSet::EMPTY;
            for (i, c) in cores.iter().enumerate() {
                assert!(c.0.iter().zip(&union.0).all(|(a, b)| a & b == 0), "core {i} overlaps an earlier one");
                union.0.iter_mut().zip(&c.0).for_each(|(u, a)| *u |= a);
            }
            assert_eq!(union, here);
        }
        assert_eq!(workers(ncores + 1), vec![here; ncores + 1]);
        assert_eq!(workers(1), vec![here]);
        assert_eq!(CpuSet::allowed(), Some(here));
    }
}
