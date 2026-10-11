//! Plan D3 / V-1: identical bits for identical input at any thread count.

use regfast_core::absorber::{Absorber, SolveOpts};
use regfast_core::gather::gather_rows;
use regfast_core::par::with_pool;
use regfast_core::rng::SplitMix64;
use regfast_core::solve::{FeSolver, SolverConfig};
use regfast_core::{AbsorbTerm, Ctx, Family, FactorValues, Prep, Session, Vce, WeightKind};

struct Data {
    factors: Vec<Vec<f64>>,
    w: Option<Vec<f64>>,
    cols: Vec<Vec<f64>>,
}

fn normal(rng: &mut SplitMix64) -> f64 {
    let u1 = rng.next_f64().max(1e-300);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * rng.next_f64()).cos()
}

fn finish(factors: Vec<Vec<f64>>, weighted: bool, r: usize, seed: u64) -> Data {
    let n = factors[0].len();
    let mut rng = SplitMix64::new(seed);
    let effects: Vec<Vec<f64>> = factors.iter().map(|f| (0..=f.iter().fold(0.0f64, |m, v| m.max(*v)) as usize).map(|_| normal(&mut rng)).collect()).collect();
    let cols = (0..r)
        .map(|_| (0..n).map(|i| normal(&mut rng) + factors.iter().zip(&effects).map(|(f, e)| e[f[i] as usize]).sum::<f64>()).collect())
        .collect();
    let w = weighted.then(|| (0..n).map(|_| 0.5 + rng.next_f64()).collect());
    Data { factors, w, cols }
}

fn random(n: usize, gs: &[usize], weighted: bool, seed: u64) -> Data {
    let mut rng = SplitMix64::new(seed);
    let factors = gs.iter().map(|&g| (0..n).map(|_| (rng.next_u64() % g as u64) as f64).collect()).collect();
    finish(factors, weighted, 3, seed + 1)
}

/// Ring band: indiv i at `width` consecutive firms; random years.
fn band(nfirm: usize, nindiv: usize, width: usize) -> Data {
    let mut rng = SplitMix64::new(77);
    let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..nindiv {
        for t in 0..width {
            a.push(i as f64);
            b.push((rng.next_u64() % 10) as f64);
            c.push(((i * nfirm / nindiv + t) % nfirm) as f64);
        }
    }
    finish(vec![a, b, c], false, 2, 78)
}

fn prep(d: &Data) -> Prep<'_> {
    Prep {
        n: d.factors[0].len(),
        factors: d.factors.iter().map(|f| FactorValues::Num(f)).collect(),
        absorb: (0..d.factors.len()).map(|q| AbsorbTerm { factors: vec![q], intercept: true, slopes: vec![], save: false }).collect(),
        slope_vars: vec![],
        weights: d.w.as_deref(),
        weight_kind: if d.w.is_some() { WeightKind::Pweight } else { WeightKind::None },
        vce: Vce::Unadjusted,
        drop_singletons: true,
        family: Family::Gaussian,
        y_for_sep: None,
        sep_fe: false,
    }
}

/// (row order, residualized rows, FE coefficients, iterations, escalated) at `threads`.
fn run(d: &Data, cfg: SolverConfig, threads: usize) -> (Vec<u32>, Vec<u64>, Vec<u64>, usize, bool) {
    let ctx = Ctx::quiet(threads);
    let s = Session::prepare(prep(d), &ctx).unwrap();
    let kept: Vec<Vec<f64>> = d.cols.iter().map(|c| s.kept_index().iter().map(|&k| c[k as usize]).collect()).collect();
    let refs: Vec<&[f64]> = kept.iter().map(Vec::as_slice).collect();
    let r = refs.len();
    with_pool(threads, |pool| {
        let mut fs = FeSolver::new(s.terms(), s.weights(), s.n(), cfg, pool, &ctx).unwrap();
        let mut rows = gather_rows(fs.row_order(), &refs, pool).unwrap();
        let o = SolveOpts { tol: 1e-10, maxiter: 10_000, want_coefs: true };
        let info = fs.residualize(&mut rows, r, &o, None, pool, &ctx).unwrap();
        let coefs = info.coefs.unwrap();
        (
            fs.row_order().to_vec(),
            rows.iter().map(|v| v.to_bits()).collect(),
            coefs.data.iter().map(|v| v.to_bits()).collect(),
            info.iterations,
            info.escalated,
        )
    })
    .unwrap()
}

fn check(d: &Data, cfg: SolverConfig) -> (usize, bool) {
    let base = run(d, cfg, 1);
    for t in [4, 16] {
        let other = run(d, cfg, t);
        assert!(base.0 == other.0, "row order differs at {t} threads");
        assert!(base.1 == other.1, "residualized rows differ at {t} threads");
        assert!(base.2 == other.2, "FE coefficients differ at {t} threads");
        assert_eq!(base.3, other.3);
    }
    (base.3, base.4)
}

#[test]
fn two_fe_with_large_block() {
    let d = random(200_000, &[20_000, 5_000], false, 1);
    check(&d, SolverConfig::default());
}

#[test]
fn band_graph_with_escalation() {
    let d = band(2_000, 40_000, 5);
    let (_, escalated) = check(&d, SolverConfig { small_limit: 0 });
    assert!(escalated, "the band graph should exercise the escalated preconditioner");
}

#[test]
fn weighted_three_fe_with_repeated_cells() {
    let d = random(300_000, &[2_000, 8, 30], true, 3);
    check(&d, SolverConfig::default());
    check(&d, SolverConfig { small_limit: 0 });
}
