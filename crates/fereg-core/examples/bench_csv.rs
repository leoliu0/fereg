//! `fit::fit` (the R and Python host layer) on a benchmark CSV (bench/gen_data.py layout): one
//! model, median time over `--reps` fits, the estimation rows, the IRLS iterations, the first
//! coefficient and a hash of the bits of the coefficients and variance matrix (equal hashes =
//! bit-identical estimates). `--verbose` adds one fit with per-phase timings on stderr first.
//!
//! `cargo run --release -p fereg-core --example bench_csv -- <file.csv> --y y --x x1,x2
//!   --fe indiv_id,year,firm_id [--model ols|poisson|logit] [--endog xe --inst z1,z2]
//!   [--slope firm_id:x2] [--vcov iid|hetero|cluster:a+b] [--flavor fixest|pyfixest]
//!   [--reps 5] [--threads 1] [--verbose]`
//!
//! Factor columns are read as i64 (the Python package's parquet path); `--slope f:v` adds the
//! term `f[v]` (intercept and slope).

use std::time::Instant;

use rayon::prelude::*;
use fereg_core::Ctx;
use fereg_core::fit::{self, FactorCol, FeTerm, FitInput, FitSpec, Flavor, Model, VcovKind};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let list = |k: &str| get(k).map(|v| v.split(',').map(str::to_string).collect::<Vec<_>>()).unwrap_or_default();
    let path = args.get(1).expect("usage: bench_csv <file.csv> --y y --x x1 --fe a,b [options]");
    let reps: usize = get("--reps").and_then(|v| v.parse().ok()).unwrap_or(5);
    let threads: usize = get("--threads").and_then(|v| v.parse().ok()).unwrap_or(1);
    let model = match get("--model").as_deref().unwrap_or("ols") {
        "ols" => Model::Ols,
        "poisson" => Model::Poisson,
        "logit" => Model::Logit,
        m => panic!("unknown model {m}"),
    };
    let flavor = match get("--flavor").as_deref().unwrap_or("fixest") {
        "fixest" => Flavor::Fixest,
        "pyfixest" => Flavor::Pyfixest,
        f => panic!("unknown flavor {f}"),
    };
    let y = get("--y").expect("--y");
    let (x, endog, inst, fe) = (list("--x"), list("--endog"), list("--inst"), list("--fe"));
    let slope = get("--slope").map(|s| {
        let (f, v) = s.split_once(':').expect("--slope fe:var");
        (f.to_string(), v.to_string())
    });
    let vcov = get("--vcov").unwrap_or_else(|| "iid".into());
    let cluster: Vec<String> = vcov.strip_prefix("cluster:").map(|c| c.split('+').map(str::to_string).collect()).unwrap_or_default();

    let mut factor_names: Vec<String> = fe.clone();
    for c in cluster.iter().chain(slope.iter().map(|(f, _)| f)) {
        if !factor_names.contains(c) {
            factor_names.push(c.clone());
        }
    }
    let num_names: Vec<String> =
        [vec![y.clone()], x.clone(), endog.clone(), inst.clone(), slope.iter().map(|(_, v)| v.clone()).collect()].concat();

    let text = std::fs::read_to_string(path).expect("read the CSV");
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("header").split(',').map(|h| h.trim_matches('"')).collect();
    let pos = |n: &str| header.iter().position(|h| *h == n).unwrap_or_else(|| panic!("no column {n}"));
    let (num_at, fac_at): (Vec<usize>, Vec<usize>) =
        (num_names.iter().map(|n| pos(n)).collect(), factor_names.iter().map(|n| pos(n)).collect());
    let rows: Vec<&str> = lines.collect();
    let parsed: Vec<(Vec<f64>, Vec<i64>)> = rows
        .par_iter()
        .map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            (num_at.iter().map(|&j| f[j].parse().expect("number")).collect(), fac_at.iter().map(|&j| f[j].parse().expect("integer")).collect())
        })
        .collect();
    let num: Vec<Vec<f64>> = (0..num_at.len()).map(|j| parsed.iter().map(|r| r.0[j]).collect()).collect();
    let fac: Vec<Vec<i64>> = (0..fac_at.len()).map(|j| parsed.iter().map(|r| r.1[j]).collect()).collect();
    drop(parsed);

    let col = |n: &str| &num[num_names.iter().position(|m| m == n).unwrap()][..];
    let fidx = |n: &str| factor_names.iter().position(|m| m == n).unwrap();
    let mut terms: Vec<FeTerm> = fe.iter().map(|f| FeTerm { factors: vec![fidx(f)], intercept: true, slopes: vec![] }).collect();
    if let Some((f, _)) = &slope {
        terms.push(FeTerm { factors: vec![fidx(f)], intercept: true, slopes: vec![0] });
    }
    let input = FitInput {
        n: rows.len(),
        y: col(&y),
        x: x.iter().map(|n| col(n)).collect(),
        endog: endog.iter().map(|n| col(n)).collect(),
        inst: inst.iter().map(|n| col(n)).collect(),
        factors: fac.iter().map(|c| FactorCol::I64(c)).collect(),
        factor_names: factor_names.clone(),
        fe: terms,
        slopes: slope.iter().map(|(_, v)| col(v)).collect(),
        cluster: cluster.iter().map(|c| vec![fidx(c)]).collect(),
        ..Default::default()
    };
    let vcov_kind = match vcov.as_str() {
        "iid" => VcovKind::Iid,
        "hetero" => VcovKind::Hetero,
        _ => VcovKind::Cluster,
    };
    let spec = FitSpec { model, flavor, vcov: vcov_kind, keep_fixef: false, ..Default::default() };

    if args.iter().any(|a| a == "--verbose") {
        let log = |s: &str| eprintln!("{s}");
        let never = || false;
        fit::fit(&input, &spec, &Ctx { threads, verbose: 2, interrupt: &never, log: &log }).expect("fit");
    }
    let ctx = Ctx::quiet(threads);
    let mut times = Vec::new();
    let mut last = None;
    for _ in 0..reps {
        let t = Instant::now();
        let out = fit::fit(&input, &spec, &ctx).expect("fit");
        times.push(t.elapsed().as_secs_f64());
        last = Some(out);
    }
    times.sort_by(f64::total_cmp);
    let out = last.expect("one run");
    // FNV-1a over the bits of the coefficients and the variance matrix: equal hashes = bit-identical.
    let hash = out.coef.iter().chain(&out.vcov.vcov).fold(0xcbf2_9ce4_8422_2325u64, |h, v| (h ^ v.to_bits()).wrapping_mul(0x0100_0000_01b3));
    println!(
        "{path} threads={threads} median of {reps}: {:.4}s (min {:.4}s) nobs={} irls={} coef0={:.15e} se0={:.15e} bits={hash:016x}",
        times[times.len() / 2],
        times[0],
        out.nobs,
        out.stats.irls_iterations,
        out.coef[0],
        out.vcov.vcov.first().map_or(f64::NAN, |v| v.sqrt())
    );
}
