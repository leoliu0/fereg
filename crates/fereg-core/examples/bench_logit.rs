//! `fit::fit` logit `binary_y ~ x1 | indiv_id + year` on a benchmark CSV (bench/gen_data.py
//! layout) under the fixest and pyfixest flavors: estimation rows, IRLS iterations and the
//! median time of the fit.
//!
//! `cargo run --release -p fereg-core --example bench_logit -- <file.csv> [--reps 5] [--threads 1]`

use std::time::Instant;

use fereg_core::Ctx;
use fereg_core::fit::{self, FactorCol, FeTerm, FitInput, FitSpec, Flavor, Model};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let path = args.get(1).expect("usage: bench_logit <file.csv> [--reps 5] [--threads 1]");
    let reps: usize = get("--reps").and_then(|v| v.parse().ok()).unwrap_or(5);
    let threads: usize = get("--threads").and_then(|v| v.parse().ok()).unwrap_or(1);

    let text = std::fs::read_to_string(path).expect("read the CSV");
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("header").split(',').map(|h| h.trim_matches('"')).collect();
    let names = ["indiv_id", "year", "binary_y", "x1"];
    let at: Vec<usize> = names.iter().map(|n| header.iter().position(|h| h == n).unwrap_or_else(|| panic!("no column {n}"))).collect();
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        for (c, &j) in cols.iter_mut().zip(&at) {
            c.push(fields[j].parse().expect("number"));
        }
    }
    let input = FitInput {
        n: cols[0].len(),
        y: &cols[2],
        x: vec![&cols[3]],
        factors: vec![FactorCol::F64(&cols[0]), FactorCol::F64(&cols[1])],
        factor_names: vec!["indiv_id".into(), "year".into()],
        fe: (0..2).map(|q| FeTerm { factors: vec![q], intercept: true, slopes: vec![] }).collect(),
        ..Default::default()
    };
    let ctx = Ctx::quiet(threads);
    println!("{path}, threads = {threads}, median of {reps} runs");
    println!("{:>9} {:>9} {:>6} {:>10} {:>22}", "flavor", "nobs", "iter", "seconds", "x1");
    for flavor in [Flavor::Fixest, Flavor::Pyfixest] {
        let spec = FitSpec { model: Model::Logit, flavor, keep_fixef: false, ..Default::default() };
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
        println!("{:>9} {:>9} {:>6} {:>10.4} {:>22.15e}", format!("{flavor:?}"), out.nobs, out.stats.irls_iterations, times[times.len() / 2], out.coef[0]);
    }
}
