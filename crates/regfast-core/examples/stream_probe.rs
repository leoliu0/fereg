//! Streaming read bandwidth (plan Phase 0.4 / D2): 16 threads sum a 4 GiB f64 array.
//!
//! `cargo run --release -p regfast-core --example stream_probe [threads] [GiB]`

use std::time::Instant;

use rayon::prelude::*;
use regfast_core::par::with_pool;

/// Best-of-`reps` parallel read bandwidth in GB/s over `gib` GiB with `threads` workers.
fn probe(threads: usize, gib: usize, reps: usize) -> f64 {
    let n = gib * (1 << 30) / 8;
    with_pool(threads, |pool| {
        // First touch inside the pool so pages are spread like the kernels that read them.
        let v: Vec<f64> = pool.install(|| (0..n).into_par_iter().map(|i| (i & 1023) as f64).collect());
        let mut best = 0.0f64;
        for _ in 0..reps {
            let t = Instant::now();
            let s: f64 = pool.install(|| {
                v.par_chunks(1 << 16)
                    .map(|c| {
                        // Eight independent accumulators: bound by memory, not add latency.
                        let mut acc = [0.0f64; 8];
                        for q in c.chunks_exact(8) {
                            for (a, x) in acc.iter_mut().zip(q) {
                                *a += x;
                            }
                        }
                        acc.iter().sum::<f64>()
                    })
                    .sum()
            });
            let secs = t.elapsed().as_secs_f64();
            std::hint::black_box(s);
            best = best.max((n * 8) as f64 / secs / 1e9);
        }
        best
    })
    .expect("thread pool")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let threads = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(16);
    let gib = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let bw = probe(threads, gib, 7);
    println!("stream_probe: threads={threads} size={gib} GiB read bandwidth = {bw:.1} GB/s");
}
