//! Plan D3 / V-1: identical bits for identical input at any thread count.

use fereg_core::absorber::{Absorber, SolveOpts};
use fereg_core::gather::gather_rows;
use fereg_core::par::with_pool;
use fereg_core::rng::SplitMix64;
use fereg_core::slopes::SlopeSolver;
use fereg_core::solve::{FeSolver, JointPolicy, SolverConfig};
use fereg_core::{AbsorbTerm, Ctx, Family, FactorValues, Prep, Session, Vce, WeightKind};

struct Data {
    factors: Vec<Vec<f64>>,
    w: Option<Vec<f64>>,
    cols: Vec<Vec<f64>>,
    /// When set, the last term is `factor##c.x` with these slope values.
    slope: Option<Vec<f64>>,
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
    Data { factors, w, cols, slope: None }
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
    let nf = d.factors.len();
    Prep {
        n: d.factors[0].len(),
        factors: d.factors.iter().map(|f| FactorValues::Num(f)).collect(),
        absorb: (0..nf)
            .map(|q| AbsorbTerm {
                factors: vec![q],
                intercept: true,
                slopes: if d.slope.is_some() && q == nf - 1 { vec![0] } else { vec![] },
                save: false,
            })
            .collect(),
        slope_vars: d.slope.iter().map(Vec::as_slice).collect(),
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
        let mut fs: Box<dyn Absorber> = if d.slope.is_some() {
            Box::new(SlopeSolver::new(s.terms(), s.weights(), s.n(), cfg, pool, &ctx).unwrap())
        } else {
            Box::new(FeSolver::new(s.terms(), s.weights(), s.n(), cfg, pool, &ctx).unwrap())
        };
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
    check(&d, SolverConfig { acc_bytes_max: 0, ..SolverConfig::default() });
}

#[test]
fn band_graph_with_escalation() {
    let d = band(2_000, 40_000, 5);
    let (_, escalated) = check(&d, SolverConfig { small_limit: 0, ..SolverConfig::default() });
    assert!(escalated, "the band graph should exercise the escalated preconditioner");
    let (_, escalated) = check(&d, SolverConfig { small_limit: 0, acc_bytes_max: 0, ..SolverConfig::default() });
    assert!(escalated, "the band graph should exercise the escalated preconditioner");
    // Clique-sampled Laplacian + approx-chol (seeded sampling must not depend on threads).
    let (_, escalated) = check(&d, SolverConfig { small_limit: 0, max_block_nnz: 0, ..SolverConfig::default() });
    assert!(escalated, "the band graph should exercise the escalated preconditioner");
}

#[test]
fn weighted_three_fe_with_repeated_cells() {
    let d = random(300_000, &[2_000, 8, 30], true, 3);
    check(&d, SolverConfig::default());
    check(&d, SolverConfig { small_limit: 0, acc_bytes_max: 0, ..SolverConfig::default() });
}

#[test]
fn joint_factor_with_coupled_blocks() {
    // Two large blocks (3000 and 800 levels) and a small one (10), weighted: the joint factor
    // (faer AMD + supernodal LDL^T inside the pool) must give the same bits at any thread count.
    let d = random(200_000, &[20_000, 3_000, 800, 10], true, 5);
    let cfg = SolverConfig { small_limit: 100, ..SolverConfig::default() };
    let (iterations, escalated) = check(&d, SolverConfig { joint: JointPolicy::Always, ..cfg });
    assert!(escalated && iterations == 1, "joint factor: {iterations} iterations");
    // Auto: whatever the cost model decides, the decision and the bits are thread invariant.
    check(&d, cfg);
    // Band graph: year small, firm large (2000 levels); the joint factor before the first iteration.
    let d = band(2_000, 40_000, 5);
    let (iterations, escalated) = check(&d, SolverConfig { joint: JointPolicy::Always, small_limit: 100, ..SolverConfig::default() });
    assert!(escalated && iterations == 1, "band joint factor: {iterations} iterations");
}

#[test]
fn band_graph_with_firm_slopes() {
    let mut d = band(2_000, 40_000, 5);
    let mut rng = SplitMix64::new(91);
    d.slope = Some((0..d.factors[0].len()).map(|_| normal(&mut rng)).collect());
    d.w = Some((0..d.factors[0].len()).map(|_| 0.5 + rng.next_f64()).collect());
    check(&d, SolverConfig::default());
    let (_, escalated) = check(&d, SolverConfig { small_limit: 0, ..SolverConfig::default() });
    assert!(escalated, "the slope band should exercise the escalated preconditioner");
    check(&d, SolverConfig { small_limit: 0, acc_bytes_max: 0, ..SolverConfig::default() });
}
