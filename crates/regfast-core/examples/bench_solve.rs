//! Solver benchmark on the fixest benchmark structure (plan Phase 0.4 / D2 roofline).
//!
//! n obs; nb_year = 10; indiv = n/10 (obs i: indiv i/10, year i%10); firms = indiv/23;
//! simple: random firm per obs; difficult: firm = i mod nb_firm. Columns: N(0,1) + FE effects.
//! Times `Session::prepare`, `gather` and `residualize` (cold: a fresh session per run) and
//! compares the residualize time with the D2 budget at the measured bandwidth.
//!
//! `cargo run --release -p regfast-core --example bench_solve [--n 1e6,1e7] [--reps 3] [--bw GB/s] [--verbose]`

use std::cell::RefCell;
use std::time::Instant;

use rayon::prelude::*;
use regfast_core::absorber::SolveOpts;
use regfast_core::par::with_pool;
use regfast_core::rng::SplitMix64;
use regfast_core::{AbsorbTerm, Ctx, Family, FactorValues, Prep, Session, Vce, WeightKind};

const NB_YEAR: usize = 10;
const INDIV_PER_FIRM: usize = 23;

fn normal(rng: &mut SplitMix64) -> f64 {
    let u1 = rng.next_f64().max(1e-300);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * rng.next_f64()).cos()
}

struct Data {
    indiv: Vec<f64>,
    year: Vec<f64>,
    firm: Vec<f64>,
    /// y, x1..x10.
    cols: Vec<Vec<f64>>,
}

fn generate(n: usize, difficult: bool) -> Data {
    let nindiv = n / NB_YEAR;
    let nfirm = (nindiv / INDIV_PER_FIRM).max(1);
    let seed = n as u64 * 100 + 42 + u64::from(difficult);
    let indiv: Vec<f64> = (0..n).into_par_iter().map(|i| (i / NB_YEAR) as f64).collect();
    let year: Vec<f64> = (0..n).into_par_iter().map(|i| (i % NB_YEAR) as f64).collect();
    let firm: Vec<f64> = if difficult {
        (0..n).into_par_iter().map(|i| (i % nfirm) as f64).collect()
    } else {
        (0..n).into_par_iter().map(|i| (SplitMix64::stream(seed, i as u64).next_u64() % nfirm as u64) as f64).collect()
    };
    let mut rng = SplitMix64::new(seed ^ 0xFE);
    let fe_i: Vec<f64> = (0..nindiv).map(|_| normal(&mut rng)).collect();
    let fe_f: Vec<f64> = (0..nfirm).map(|_| normal(&mut rng)).collect();
    let fe_y: Vec<f64> = (0..NB_YEAR).map(|_| normal(&mut rng)).collect();
    let cols = (0..11u64)
        .map(|j| {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut r = SplitMix64::stream(seed + 1000 + j, i as u64);
                    normal(&mut r) + fe_i[i / NB_YEAR] + fe_y[i % NB_YEAR] + fe_f[firm[i] as usize]
                })
                .collect()
        })
        .collect();
    Data { indiv, year, firm, cols }
}

fn stream_bandwidth(threads: usize) -> f64 {
    let n = (2usize << 30) / 8;
    with_pool(threads, |pool| {
        let v: Vec<f64> = pool.install(|| (0..n).into_par_iter().map(|i| (i & 1023) as f64).collect());
        let mut best = 0.0f64;
        for _ in 0..5 {
            let t = Instant::now();
            let s: f64 = pool.install(|| {
                v.par_chunks(1 << 16)
                    .map(|c| {
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
    .expect("pool")
}

struct Run {
    prepare: f64,
    gather: f64,
    resid: f64,
    iterations: usize,
    escalated: bool,
    ncells: usize,
    log: Vec<String>,
}

/// `fe`: "2" (indiv + year), "3" (+ firm) or "3s" (indiv + year + firm##c.x2, plan G7).
fn run_once(d: &Data, fe: &str, r: usize, threads: usize, verbose: bool) -> Run {
    let log = RefCell::new(Vec::new());
    let sink = |s: &str| log.borrow_mut().push(s.to_string());
    let never = || false;
    // Phase lines are always captured (cheap); printed only with --verbose.
    let _ = verbose;
    let ctx = Ctx { threads, verbose: 2, interrupt: &never, log: &sink };
    let n = d.indiv.len();
    let factors = vec![FactorValues::Num(&d.indiv), FactorValues::Num(&d.year), FactorValues::Num(&d.firm)];
    let nfe: usize = fe[..1].parse().expect("fe");
    let slopes = fe.ends_with('s');
    let prep = Prep {
        n,
        factors,
        absorb: (0..nfe)
            .map(|q| AbsorbTerm { factors: vec![q], intercept: true, slopes: if slopes && q == 2 { vec![0] } else { vec![] }, save: false })
            .collect(),
        slope_vars: if slopes { vec![&d.cols[2]] } else { vec![] },
        weights: None,
        weight_kind: WeightKind::None,
        vce: Vce::Unadjusted,
        drop_singletons: true,
        family: Family::Gaussian,
        y_for_sep: None,
        sep_fe: false,
    };
    let t = Instant::now();
    let mut s = Session::prepare(prep, &ctx).expect("prepare");
    let prepare = t.elapsed().as_secs_f64();
    // Kept-order columns (no singletons in this design: identity).
    assert_eq!(s.n(), n);
    let refs: Vec<&[f64]> = d.cols[..r].iter().map(Vec::as_slice).collect();
    let t = Instant::now();
    let mut rows = s.gather(&refs, &ctx).expect("gather");
    let gather = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let info = with_pool(threads, |pool| {
        let o = SolveOpts { tol: 1e-8, maxiter: 16_000, want_coefs: false };
        s.absorber_mut().residualize(&mut rows, r, &o, None, pool, &ctx)
    })
    .expect("pool")
    .expect("residualize");
    let resid = t.elapsed().as_secs_f64();
    std::hint::black_box(&rows);
    let lines = log.borrow().clone();
    let ncells = lines
        .iter()
        .find_map(|l| l.split("cells=").nth(1).and_then(|v| v.split_whitespace().next()).and_then(|v| v.parse().ok()))
        .unwrap_or(n);
    Run { prepare, gather, resid, iterations: info.iterations, escalated: info.escalated, ncells, log: lines }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let ns: Vec<usize> = get("--n").unwrap_or_else(|| "1e6,1e7".into()).split(',').map(|v| v.parse::<f64>().expect("n") as usize).collect();
    let reps: usize = get("--reps").and_then(|v| v.parse().ok()).unwrap_or(3);
    let verbose = args.iter().any(|a| a == "--verbose");
    let fes: Vec<String> = get("--fe").unwrap_or_else(|| "2,3".into()).split(',').map(str::to_string).collect();
    let bw16 = get("--bw").and_then(|v| v.parse().ok()).unwrap_or_else(|| stream_bandwidth(16));
    let bw1 = get("--bw1").and_then(|v| v.parse().ok()).unwrap_or_else(|| stream_bandwidth(1));
    println!("bandwidth: {bw16:.1} GB/s (16 threads), {bw1:.1} GB/s (1 thread); tol 1e-8; best of {reps} cold runs");
    println!(
        "{:>9} {:>9} {:>3} {:>3} {:>3} | {:>8} {:>8} {:>8} {:>8} | {:>5} {:>4} | {:>8} {:>6}",
        "n", "type", "FE", "T", "R", "prepare", "gather", "resid", "total", "iters", "esc", "budget", "ratio"
    );
    for &n in &ns {
        for difficult in [false, true] {
            let d = generate(n, difficult);
            for fe in &fes {
                let nfe: usize = fe[..1].parse().expect("fe");
                let mut threads = vec![16];
                if n <= 1_000_000 {
                    threads.push(1);
                }
                for &t in &threads {
                    for r in [2usize, 11] {
                        let mut best: Option<Run> = None;
                        for k in 0..reps {
                            let run = run_once(&d, fe, r, t, verbose && k == 0);
                            if k == 0 && verbose {
                                for l in &run.log {
                                    eprintln!("    {l}");
                                }
                            }
                            let tot = run.prepare + run.gather + run.resid;
                            if best.as_ref().is_none_or(|b| tot < b.prepare + b.gather + b.resid) {
                                best = Some(run);
                            }
                        }
                        let b = best.expect("reps ≥ 1");
                        // D2 budget of the residualize call: two streaming row passes (read; read +
                        // write) plus the cell codes twice, and per iteration one pass over the cells.
                        let q1 = (nfe - 1) as f64;
                        let bytes = n as f64 * (r as f64 * 24.0) + 2.0 * b.ncells as f64 * q1 * 4.0;
                        let iter_bytes = b.ncells as f64 * (q1 * 4.0 + r as f64 * 16.0);
                        let bw = if t == 1 { bw1 } else { bw16 };
                        let budget = 1.5 * bytes / (bw * 1e9) + b.iterations as f64 * 1.5 * iter_bytes / (bw * 1e9);
                        println!(
                            "{:>9} {:>9} {:>3} {:>3} {:>3} | {:>8.4} {:>8.4} {:>8.4} {:>8.4} | {:>5} {:>4} | {:>8.4} {:>6.2}",
                            n,
                            if difficult { "difficult" } else { "simple" },
                            fe,
                            t,
                            r,
                            b.prepare,
                            b.gather,
                            b.resid,
                            b.prepare + b.gather + b.resid,
                            b.iterations,
                            if b.escalated { "yes" } else { "no" },
                            budget,
                            b.resid / budget
                        );
                    }
                }
            }
        }
    }
}
