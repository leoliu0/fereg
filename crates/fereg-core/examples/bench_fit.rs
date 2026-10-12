//! `fit::fit` (host layer) against the Stata plugin path `Session::prepare` + `gather` +
//! `Session::ols` on the same data (fixest benchmark structure, as `bench_solve`), fit's
//! Poisson against `Session::glm_parts` under the fixest convention, and the cost of the
//! fixed-effect values.
//!
//! n obs; nb_year = 10; indiv = n/10 (obs i: indiv i/10, year i%10); firm: random per obs among
//! indiv/23 firms; y, x1..x10 = N(0,1) + FE effects; count outcome yp = round(exp(0.3·x1 − 0.5)).
//! Models `y ~ x1..xk | indiv + year`, iid, residuals requested on both paths;
//! `yp ~ x1..xk | indiv + year`; `y ~ x1 | indiv + year (+ firm)` for the fixed-effect values.
//!
//! `cargo run --release -p fereg-core --example bench_fit [--n 1e7] [--reps 5] [--threads 16] [--verbose] [--fixef-only]`
//! (`--verbose`: one extra run of each OLS path per k with per-phase timings on stderr first;
//! `--fixef-only`: only the fixed-effect values).

use std::time::Instant;

use rayon::prelude::*;
use fereg_core::fit::{self, FactorCol, FeTerm, FitInput, FitSpec};
use fereg_core::rng::SplitMix64;
use fereg_core::{AbsorbTerm, Ctx, Family, FactorValues, GlmConvention, GlmOptions, Guess, Layout, Prep, Separation, Session, Spec, Vce, WeightKind};

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
    yp: Vec<f64>,
}

fn generate(n: usize) -> Data {
    let nindiv = n / 10;
    let seed = n as u64 * 100 + 42;
    let indiv: Vec<f64> = (0..n).into_par_iter().map(|i| (i / 10) as f64).collect();
    let year: Vec<f64> = (0..n).into_par_iter().map(|i| (i % 10) as f64).collect();
    let nfirm = (nindiv / 23).max(1) as u64;
    let firm: Vec<f64> = (0..n).into_par_iter().map(|i| (SplitMix64::stream(seed + 7, i as u64).next_u64() % nfirm) as f64).collect();
    let mut rng = SplitMix64::new(seed ^ 0xFE);
    let fe_i: Vec<f64> = (0..nindiv).map(|_| normal(&mut rng)).collect();
    let fe_y: Vec<f64> = (0..10).map(|_| normal(&mut rng)).collect();
    let cols = (0..11u64)
        .map(|j| {
            (0..n)
                .into_par_iter()
                .map(|i| normal(&mut SplitMix64::stream(seed + 1000 + j, i as u64)) + fe_i[i / 10] + fe_y[i % 10])
                .collect()
        })
        .collect::<Vec<Vec<f64>>>();
    let yp = cols[1].par_iter().map(|x| (0.3 * x - 0.5).exp().round()).collect();
    Data { indiv, year, firm, cols, yp }
}

fn session_path(d: &Data, k: usize, ctx: &Ctx) -> Vec<f64> {
    let n = d.indiv.len();
    let prep = Prep {
        n,
        factors: vec![FactorValues::Num(&d.indiv), FactorValues::Num(&d.year)],
        absorb: (0..2).map(|q| AbsorbTerm { factors: vec![q], intercept: true, slopes: vec![], save: false }).collect(),
        slope_vars: vec![],
        weights: None,
        weight_kind: WeightKind::None,
        vce: Vce::Unadjusted,
        drop_singletons: true,
        family: Family::Gaussian,
        y_for_sep: None,
        sep_fe: false,
    };
    let mut s = Session::prepare(prep, ctx).expect("prepare");
    let cols: Vec<&[f64]> = d.cols[..=k].iter().map(Vec::as_slice).collect();
    let rows = s.gather(&cols, ctx).expect("gather");
    let spec = Spec { want_resid: true, ..Spec::default() };
    let res = s.ols(rows, &Layout::ols(k), &spec, ctx).expect("ols");
    res.matrix("b").expect("b").data.clone()
}

fn fit_path(d: &Data, k: usize, ctx: &Ctx) -> Vec<f64> {
    let input = FitInput {
        n: d.indiv.len(),
        y: &d.cols[0],
        x: d.cols[1..=k].iter().map(Vec::as_slice).collect(),
        factors: vec![FactorCol::F64(&d.indiv), FactorCol::F64(&d.year)],
        factor_names: vec!["indiv".into(), "year".into()],
        fe: (0..2).map(|q| FeTerm { factors: vec![q], intercept: true, slopes: vec![] }).collect(),
        ..Default::default()
    };
    let spec = FitSpec { keep_fixef: false, keep_state: false, ..FitSpec::default() };
    fit::fit(&input, &spec, ctx).expect("fit").coef
}

fn session_poisson(d: &Data, k: usize, ctx: &Ctx) -> Vec<f64> {
    let n = d.indiv.len();
    let prep = Prep {
        n,
        factors: vec![FactorValues::Num(&d.indiv), FactorValues::Num(&d.year)],
        absorb: (0..2).map(|q| AbsorbTerm { factors: vec![q], intercept: true, slopes: vec![], save: false }).collect(),
        slope_vars: vec![],
        weights: None,
        weight_kind: WeightKind::None,
        vce: Vce::Unadjusted,
        drop_singletons: true,
        family: Family::Poisson,
        y_for_sep: Some(&d.yp),
        sep_fe: true,
    };
    let mut s = Session::prepare(prep, ctx).expect("prepare");
    // `gather` takes kept-order columns (the Stata plugin reads only the kept observations).
    let kept = s.kept_index();
    let own: Vec<Vec<f64>> =
        std::iter::once(&d.yp).chain(&d.cols[1..=k]).map(|c| kept.par_iter().map(|&i| c[i as usize]).collect()).collect();
    let cols: Vec<&[f64]> = own.iter().map(Vec::as_slice).collect();
    let rows = s.gather(&cols, ctx).expect("gather");
    let g = GlmOptions { family: Family::Poisson, tol: 1e-8, maxiter: 25, itol: 1e-8, separation: Separation::default(), guess: Guess::Simple, log: false };
    let (_, parts) = s.glm_parts(rows, &Layout::ols(k), None, &Spec::default(), &g, GlmConvention::Fixest, ctx).expect("glm");
    parts.coef
}

fn fit_poisson(d: &Data, k: usize, ctx: &Ctx) -> Vec<f64> {
    let input = FitInput {
        n: d.indiv.len(),
        y: &d.yp,
        x: d.cols[1..=k].iter().map(Vec::as_slice).collect(),
        factors: vec![FactorCol::F64(&d.indiv), FactorCol::F64(&d.year)],
        factor_names: vec!["indiv".into(), "year".into()],
        fe: (0..2).map(|q| FeTerm { factors: vec![q], intercept: true, slopes: vec![] }).collect(),
        ..Default::default()
    };
    let spec = FitSpec { model: fit::Model::Poisson, keep_fixef: false, keep_state: false, ..FitSpec::default() };
    fit::fit(&input, &spec, ctx).expect("fit").coef
}

fn med(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Median times of `a` and `b` over `reps` alternating runs; their coefficients must agree.
fn compare(reps: usize, a: impl Fn() -> Vec<f64>, b: impl Fn() -> Vec<f64>) -> (f64, f64) {
    let (mut ta, mut tb) = (Vec::new(), Vec::new());
    for _ in 0..reps {
        let t = Instant::now();
        let x = a();
        ta.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        let y = b();
        tb.push(t.elapsed().as_secs_f64());
        assert!(x.iter().zip(&y).all(|(a, b)| (a - b).abs() <= 1e-6 * a.abs().max(1e-3)), "coefficients differ: {x:?} {y:?}");
    }
    (med(&mut ta), med(&mut tb))
}

/// `y ~ x1 | indiv + year (+ firm)`: time of the fit without fixed-effect values (lean, and
/// keeping the state), with them under each normalization, and of `FitState::fixef` on demand.
fn fixef_times(d: &Data, nfe: usize, reps: usize, ctx: &Ctx) -> [f64; 6] {
    let factors = [&d.indiv, &d.year, &d.firm];
    let input = FitInput {
        n: d.indiv.len(),
        y: &d.cols[0],
        x: vec![&d.cols[1]],
        factors: factors[..nfe].iter().map(|f| FactorCol::F64(f)).collect(),
        factor_names: ["indiv", "year", "firm"][..nfe].iter().map(|s| s.to_string()).collect(),
        fe: (0..nfe).map(|q| FeTerm { factors: vec![q], intercept: true, slopes: vec![] }).collect(),
        ..Default::default()
    };
    let time = |f: &dyn Fn()| {
        let mut t: Vec<f64> = (0..reps)
            .map(|_| {
                let s = Instant::now();
                f();
                s.elapsed().as_secs_f64()
            })
            .collect();
        med(&mut t)
    };
    let run = |spec: &FitSpec| time(&|| drop(std::hint::black_box(fit::fit(&input, spec, ctx).expect("fit"))));
    let lean = FitSpec { keep_fixef: false, keep_state: false, ..FitSpec::default() };
    let with_state = FitSpec { keep_state: true, ..lean.clone() };
    let state = fit::fit(&input, &with_state, ctx).expect("fit").state.expect("state");
    let on_demand = |norm| time(&|| drop(std::hint::black_box(state.fixef(norm, ctx).expect("fixef"))));
    [
        run(&lean),
        run(&with_state),
        run(&FitSpec { keep_fixef: true, fixef_norm: fit::FixefNorm::Fixest, ..lean.clone() }),
        run(&FitSpec { keep_fixef: true, fixef_norm: fit::FixefNorm::MinNorm, ..lean.clone() }),
        on_demand(fit::FixefNorm::Fixest),
        on_demand(fit::FixefNorm::MinNorm),
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let n = get("--n").map_or(10_000_000, |v| v.parse::<f64>().expect("n") as usize);
    let reps: usize = get("--reps").and_then(|v| v.parse().ok()).unwrap_or(5);
    let threads: usize = get("--threads").and_then(|v| v.parse().ok()).unwrap_or(16);
    let d = generate(n);
    let ctx = Ctx::quiet(threads);
    if args.iter().any(|a| a == "--verbose") {
        let log = |s: &str| eprintln!("{s}");
        let vctx = Ctx { verbose: 2, log: &log, ..Ctx::quiet(threads) };
        for k in [1usize, 10] {
            eprintln!("--- Session, k = {k}");
            session_path(&d, k, &vctx);
            eprintln!("--- fit, k = {k}");
            fit_path(&d, k, &vctx);
        }
    }
    println!("n = {n}, threads = {threads}, median of {reps} runs (seconds)");
    if !args.iter().any(|a| a == "--fixef-only") {
        println!("{:>11} {:>12} {:>12} {:>7}", "model", "Session", "fit", "ratio");
        for k in [1usize, 10] {
            let (s, f) = compare(reps, || session_path(&d, k, &ctx), || fit_path(&d, k, &ctx));
            println!("{:>11} {s:>12.4} {f:>12.4} {:>7.3}", format!("ols k={k}"), f / s);
        }
        for k in [1usize, 10] {
            let (s, f) = compare(reps, || session_poisson(&d, k, &ctx), || fit_poisson(&d, k, &ctx));
            println!("{:>11} {s:>12.4} {f:>12.4} {:>7.3}", format!("pois k={k}"), f / s);
        }
    }
    println!("FE values: fit time (lean, with state, fixef at fit time) and FitState::fixef time");
    println!("{:>11} {:>9} {:>9} {:>9} {:>9} {:>11} {:>11}", "", "lean", "state", "fixest", "minnorm", "lazy fixest", "lazy minnorm");
    for nfe in [2usize, 3] {
        let v = fixef_times(&d, nfe, reps, &ctx);
        println!("{:>11} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>11.4} {:>11.4}", format!("{nfe} FE"), v[0], v[1], v[2], v[3], v[4], v[5]);
    }
}
