//! FeSolver (plan D5) against the dense explicit-dummy oracle (plan §2.5).

use regfast_core::absorber::{Absorber, FeCoefs, SolveInfo, SolveOpts};
use regfast_core::gather::{gather_rows, to_kept_order};
use regfast_core::par::with_pool;
use regfast_core::rng::SplitMix64;
use regfast_core::solve::{FeSolver, SolverConfig};
use regfast_core::{AbsorbTerm, AbsorberKind, Ctx, Family, FactorValues, Prep, RfError, Session, Vce, WeightKind};

struct Data {
    factors: Vec<Vec<f64>>,
    /// Each absorb term: factor indices (interaction when > 1).
    terms: Vec<Vec<usize>>,
    w: Option<Vec<f64>>,
    seed: u64,
}

fn normal(rng: &mut SplitMix64) -> f64 {
    let u1 = rng.next_f64().max(1e-300);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn prep(d: &Data) -> Prep<'_> {
    let n = d.factors.first().map_or_else(|| d.w.as_ref().map_or(0, Vec::len), Vec::len);
    Prep {
        n,
        factors: d.factors.iter().map(|f| FactorValues::Num(f)).collect(),
        absorb: d.terms.iter().map(|t| AbsorbTerm { factors: t.clone(), intercept: true, slopes: vec![], save: true }).collect(),
        slope_vars: vec![],
        weights: d.w.as_deref(),
        weight_kind: if d.w.is_some() { WeightKind::Aweight } else { WeightKind::None },
        vce: Vce::Unadjusted,
        drop_singletons: true,
        family: Family::Gaussian,
        y_for_sep: None,
        sep_fe: false,
    }
}

/// Random columns (kept order): FE components of every term plus noise.
fn columns(s: &Session, r: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = SplitMix64::new(seed);
    let n = s.n();
    (0..r)
        .map(|_| {
            let mut v: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
            for t in s.terms() {
                let eff: Vec<f64> = (0..t.g).map(|_| 3.0 * normal(&mut rng)).collect();
                for i in 0..n {
                    v[i] += eff[t.codes[i] as usize];
                }
            }
            v.iter_mut().for_each(|x| *x += 5.0);
            v
        })
        .collect()
}

fn opts() -> SolveOpts {
    SolveOpts { tol: 1e-12, maxiter: 10_000, want_coefs: true }
}

struct Pair {
    dense: Session,
    fast: FeSolver,
    cols: Vec<Vec<f64>>,
}

fn setup(d: &Data, cfg: SolverConfig, r: usize) -> Pair {
    let ctx = Ctx::quiet(4);
    let dense = Session::prepare_with(prep(d), &ctx, AbsorberKind::Dense).unwrap();
    let cols = columns(&dense, r, d.seed);
    let terms = dense.terms().to_vec();
    let fast = with_pool(4, |pool| FeSolver::new(&terms, dense.weights(), dense.n(), cfg, pool, &ctx)).unwrap().unwrap();
    Pair { dense, fast, cols }
}

/// Residualizes `cols` with both absorbers; returns (fast kept-order, dense kept-order, infos).
fn run(p: &mut Pair, warm: Option<&FeCoefs>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, SolveInfo, SolveInfo) {
    let ctx = Ctx::quiet(4);
    let r = p.cols.len();
    let refs: Vec<&[f64]> = p.cols.iter().map(Vec::as_slice).collect();
    with_pool(4, |pool| {
        let mut rows_d = gather_rows(p.dense.row_order(), &refs, pool).unwrap();
        let info_d = p.dense.absorber_mut().residualize(&mut rows_d, r, &opts(), None, pool, &ctx).unwrap();
        let mut rows_f = gather_rows(p.fast.row_order(), &refs, pool).unwrap();
        let info_f = p.fast.residualize(&mut rows_f, r, &opts(), warm, pool, &ctx).unwrap();
        let split = |rows: &[f64], order: &[u32]| -> Vec<Vec<f64>> {
            (0..r).map(|j| to_kept_order(order, &(0..order.len()).map(|i| rows[i * r + j]).collect::<Vec<_>>())).collect()
        };
        (split(&rows_f, p.fast.row_order()), split(&rows_d, p.dense.row_order()), info_f, info_d)
    })
    .unwrap()
}

fn max_rel(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let scale = y.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
            x.iter().zip(y).fold(0.0f64, |m, (p, q)| m.max((p - q).abs())) / scale
        })
        .fold(0.0, f64::max)
}

fn check(d: &Data, cfg: SolverConfig, r: usize) -> SolveInfo {
    let mut p = setup(d, cfg, r);
    let (f, dn, info, _) = run(&mut p, None);
    let err = max_rel(&f, &dn);
    assert!(err <= 1e-9, "max rel error {err:e} (iterations {}, escalated {}, {cfg:?}, seed {})", info.iterations, info.escalated, d.seed);
    info
}

fn random_codes(n: usize, g: usize, rng: &mut SplitMix64) -> Vec<f64> {
    (0..n).map(|_| (rng.next_u64() % g as u64) as f64).collect()
}

fn bipartite(n: usize, gs: &[usize], weighted: bool, seed: u64) -> Data {
    let mut rng = SplitMix64::new(seed);
    let factors: Vec<Vec<f64>> = gs.iter().map(|&g| random_codes(n, g, &mut rng)).collect();
    let w = weighted.then(|| (0..n).map(|_| 0.5 + rng.next_f64()).collect());
    Data { terms: (0..gs.len()).map(|q| vec![q]).collect(), factors, w, seed }
}

/// Ring graph: indiv i works at firms i mod F and (i + 1) mod F.
fn ring(nfirm: usize, nindiv: usize) -> Data {
    let mut indiv = Vec::new();
    let mut firm = Vec::new();
    for i in 0..nindiv {
        for k in 0..2 {
            indiv.push(i as f64);
            firm.push(((i + k) % nfirm) as f64);
        }
    }
    Data { factors: vec![indiv, firm], terms: vec![vec![0], vec![1]], w: None, seed: 7 }
}

/// fixest "difficult" design: indiv × year panel, firm = obs mod F (band graph on a ring).
fn difficult(nindiv: usize, nyear: usize, nfirm: usize, weighted: bool) -> Data {
    let n = nindiv * nyear;
    let indiv: Vec<f64> = (0..n).map(|i| (i / nyear) as f64).collect();
    let year: Vec<f64> = (0..n).map(|i| (i % nyear) as f64).collect();
    let firm: Vec<f64> = (0..n).map(|i| (i % nfirm) as f64).collect();
    let mut rng = SplitMix64::new(11);
    let w = weighted.then(|| (0..n).map(|_| 0.5 + rng.next_f64()).collect());
    Data { factors: vec![indiv, year, firm], terms: vec![vec![0], vec![1], vec![2]], w, seed: 13 }
}

fn all_large() -> SolverConfig {
    SolverConfig { small_limit: 0 }
}

#[test]
fn random_bipartite_two_and_three_fe() {
    for (gs, seed) in [(vec![300, 80], 1), (vec![250, 40, 120], 2), (vec![60, 25], 3)] {
        let d = bipartite(2000, &gs, false, seed);
        check(&d, SolverConfig::default(), 3);
        check(&d, all_large(), 3);
        check(&d, SolverConfig { small_limit: 60 }, 2);
    }
}

#[test]
fn weighted_observations() {
    let d = bipartite(1800, &[200, 30, 90], true, 4);
    check(&d, SolverConfig::default(), 2);
    check(&d, all_large(), 2);
    let d = difficult(150, 10, 300, true);
    check(&d, all_large(), 2);
}

#[test]
fn ring_graph_escalates_and_converges_fast() {
    let d = ring(400, 800);
    let info = check(&d, all_large(), 2);
    assert!(info.escalated, "ring graph must escalate");
    assert!(info.iterations <= 30, "ring graph took {} iterations", info.iterations);
}

/// Band graph on a ring of firms: indiv i works at `width` consecutive firms starting at i/2,
/// one observation per firm, each in a random year.
fn band(nfirm: usize, nindiv: usize, width: usize, nyear: usize, weighted: bool) -> Data {
    let mut rng = SplitMix64::new(31);
    let (mut indiv, mut firm, mut year) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..nindiv {
        for t in 0..width {
            indiv.push(i as f64);
            firm.push(((i * nfirm / nindiv + t) % nfirm) as f64);
            year.push((rng.next_u64() % nyear as u64) as f64);
        }
    }
    let n = indiv.len();
    let w = weighted.then(|| (0..n).map(|_| 0.5 + rng.next_f64()).collect());
    Data { factors: vec![indiv, year, firm], terms: vec![vec![0], vec![1], vec![2]], w, seed: 32 }
}

#[test]
fn band_graph_with_year_escalates() {
    let d = band(400, 800, 3, 10, false);
    let info = check(&d, SolverConfig { small_limit: 50 }, 3);
    assert!(info.escalated, "band graph must escalate ({} iterations)", info.iterations);
    assert!(info.iterations <= 60, "band graph took {} iterations", info.iterations);
    check(&d, all_large(), 2);
    check(&d, SolverConfig::default(), 2);
    check(&band(400, 800, 3, 10, true), SolverConfig { small_limit: 50 }, 2);
}

#[test]
fn disconnected_components() {
    let mut rng = SplitMix64::new(5);
    let n = 1600;
    let mut a = Vec::new();
    let mut b = Vec::new();
    for i in 0..n {
        let side = i % 2;
        a.push((side * 100 + (rng.next_u64() % 100) as usize) as f64);
        b.push((side * 40 + (rng.next_u64() % 40) as usize) as f64);
    }
    let d = Data { factors: vec![a, b], terms: vec![vec![0], vec![1]], w: None, seed: 6 };
    check(&d, SolverConfig::default(), 2);
    check(&d, all_large(), 2);
}

#[test]
fn nested_fe() {
    // group ⊃ indiv: the group block is redundant given FE1 = indiv.
    let mut rng = SplitMix64::new(8);
    let n = 2000;
    let indiv: Vec<f64> = (0..n).map(|_| (rng.next_u64() % 200) as f64).collect();
    let group: Vec<f64> = indiv.iter().map(|v| (*v as u64 / 5) as f64).collect();
    let year = random_codes(n, 12, &mut rng);
    let d = Data { factors: vec![indiv, group, year], terms: vec![vec![0], vec![1], vec![2]], w: None, seed: 9 };
    check(&d, SolverConfig::default(), 2);
    check(&d, all_large(), 2);
    // Duplicate term and an interaction term.
    let d2 = Data { factors: d.factors.clone(), terms: vec![vec![0], vec![0], vec![1, 2]], w: None, seed: 10 };
    check(&d2, SolverConfig::default(), 2);
    check(&d2, all_large(), 2);
}

#[test]
fn large_plus_small_blocks_and_repeated_cells() {
    let d = bipartite(3000, &[100, 12, 400], false, 12);
    let info = check(&d, SolverConfig { small_limit: 50 }, 4);
    assert!(info.iterations >= 1);
    // Few levels: cells hold many observations.
    let d = bipartite(2000, &[40, 15, 9], true, 14);
    check(&d, SolverConfig::default(), 3);
    check(&d, all_large(), 3);
}

#[test]
fn noabsorb_and_single_fe_are_closed_form() {
    let mut rng = SplitMix64::new(15);
    let n = 500;
    let w: Vec<f64> = (0..n).map(|_| 0.5 + rng.next_f64()).collect();
    let d = Data { factors: vec![], terms: vec![], w: Some(w), seed: 16 };
    let info = check(&d, SolverConfig::default(), 2);
    assert_eq!(info.iterations, 1);
    let d = bipartite(1000, &[70], true, 17);
    let info = check(&d, SolverConfig::default(), 2);
    assert_eq!(info.iterations, 1);
}

#[test]
fn set_weights_with_warm_start() {
    for (d, cfg) in [(bipartite(1500, &[150, 20, 60], false, 18), all_large()), (difficult(150, 10, 300, false), SolverConfig { small_limit: 20 })] {
        let mut p = setup(&d, cfg, 2);
        let (f, dn, info0, _) = run(&mut p, None);
        assert!(max_rel(&f, &dn) <= 1e-9);
        let coefs = info0.coefs.unwrap();
        let ctx = Ctx::quiet(4);
        let mut rng = SplitMix64::new(19);
        let n = p.dense.n();
        let wk: Vec<f64> = (0..n).map(|_| 0.3 + 2.0 * rng.next_f64()).collect();
        with_pool(4, |pool| {
            let wd: Vec<f64> = p.dense.row_order().iter().map(|&k| wk[k as usize]).collect();
            p.dense.absorber_mut().set_weights(Some(wd), pool, &ctx).unwrap();
            let wf: Vec<f64> = p.fast.row_order().iter().map(|&k| wk[k as usize]).collect();
            p.fast.set_weights(Some(wf), pool, &ctx).unwrap();
        })
        .unwrap();
        for _ in 0..2 {
            let (f, dn, info, _) = run(&mut p, Some(&coefs));
            let err = max_rel(&f, &dn);
            assert!(err <= 1e-9, "after set_weights: {err:e} ({} iterations)", info.iterations);
        }
    }
}

/// fit_rows reproduces z − z̃, and term_values match the dense oracle on connected designs.
#[test]
fn fit_rows_and_term_values_parity() {
    let designs = [
        (bipartite(2000, &[150, 30], true, 20), SolverConfig::default()),
        (bipartite(2000, &[150, 30], false, 21), all_large()),
        (bipartite(3000, &[120, 10, 40], false, 22), SolverConfig::default()),
        (bipartite(3000, &[120, 10, 40], true, 23), all_large()),
    ];
    for (d, cfg) in designs {
        let mut p = setup(&d, cfg, 2);
        let (f, _, info_f, info_d) = run(&mut p, None);
        let (cf, cd) = (info_f.coefs.unwrap(), info_d.coefs.unwrap());
        let n = p.dense.n();
        with_pool(4, |pool| {
            for j in 0..2 {
                // fit = z − z̃ (fast, kept order).
                let fit = to_kept_order(p.fast.row_order(), &p.fast.fit_rows(&cf, j, pool));
                let scale = p.cols[j].iter().fold(0.0f64, |m, v| m.max(v.abs()));
                for i in 0..n {
                    assert!((p.cols[j][i] - f[j][i] - fit[i]).abs() <= 1e-9 * scale, "fit_rows mismatch");
                }
                let fit_d = p.dense.absorber().fit_rows(&cd, j, pool);
                for i in 0..n {
                    assert!((fit[i] - fit_d[i]).abs() <= 1e-9 * scale);
                }
                let tf = p.fast.term_values(&cf, j);
                let td = p.dense.absorber().term_values(&cd, j);
                assert_eq!(tf.len(), td.len());
                for (a, b) in tf.iter().zip(&td) {
                    assert_eq!(a.len(), b.len());
                    for (x, y) in a.iter().zip(b) {
                        assert!((x - y).abs() <= 1e-8 * scale, "term_values {x} vs {y}");
                    }
                }
            }
            // combine() is linear: fit of [1, −1] = fit_0 − fit_1.
            let comb = cf.combine(&[1.0, -1.0]);
            let a = p.fast.fit_rows(&comb, 0, pool);
            let b0 = p.fast.fit_rows(&cf, 0, pool);
            let b1 = p.fast.fit_rows(&cf, 1, pool);
            for i in 0..n {
                assert!((a[i] - (b0[i] - b1[i])).abs() <= 1e-9 * (1.0 + b0[i].abs()));
            }
        })
        .unwrap();
    }
}

#[test]
fn session_fast_path_matches_dense() {
    let d = difficult(300, 10, 500, true);
    let ctx = Ctx::quiet(3);
    let mut fast = Session::prepare(prep(&d), &ctx).unwrap();
    let mut dense = Session::prepare_with(prep(&d), &ctx, AbsorberKind::Dense).unwrap();
    assert_eq!(fast.keep(), dense.keep());
    let cols = columns(&dense, 2, 3);
    let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
    let mut rf = fast.gather(&refs, &ctx).unwrap();
    let mut rd = dense.gather(&refs, &ctx).unwrap();
    with_pool(3, |pool| {
        fast.absorber_mut().residualize(&mut rf, 2, &opts(), None, pool, &ctx).unwrap();
        dense.absorber_mut().residualize(&mut rd, 2, &opts(), None, pool, &ctx).unwrap();
    })
    .unwrap();
    let split = |rows: &[f64], order: &[u32]| -> Vec<Vec<f64>> {
        (0..2).map(|j| to_kept_order(order, &(0..order.len()).map(|i| rows[i * 2 + j]).collect::<Vec<_>>())).collect()
    };
    let err = max_rel(&split(&rf, fast.row_order()), &split(&rd, dense.row_order()));
    assert!(err <= 1e-9, "{err:e}");
}

#[test]
fn slopes_are_rejected_and_maxiter_is_enforced() {
    let mut rng = SplitMix64::new(30);
    let n = 400;
    let a = random_codes(n, 20, &mut rng);
    let x: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    let prep = Prep {
        n,
        factors: vec![FactorValues::Num(&a)],
        absorb: vec![AbsorbTerm { factors: vec![0], intercept: true, slopes: vec![0], save: false }],
        slope_vars: vec![&x],
        weights: None,
        weight_kind: WeightKind::None,
        vce: Vce::Unadjusted,
        drop_singletons: true,
        family: Family::Gaussian,
        y_for_sep: None,
        sep_fe: false,
    };
    match Session::prepare(prep, &Ctx::quiet(1)) {
        Err(RfError::InvalidInput(m)) => assert_eq!(m, "slopes not yet supported"),
        _ => panic!("slopes must be rejected"),
    }
    let mut p = setup(&ring(400, 800), all_large(), 1);
    let refs: Vec<&[f64]> = p.cols.iter().map(Vec::as_slice).collect();
    let ctx = Ctx::quiet(2);
    with_pool(2, |pool| {
        let mut rows = gather_rows(p.fast.row_order(), &refs, pool).unwrap();
        let o = SolveOpts { tol: 1e-12, maxiter: 5, want_coefs: false };
        assert_eq!(p.fast.residualize(&mut rows, 1, &o, None, pool, &ctx).unwrap_err(), RfError::Convergence(5));
    })
    .unwrap();
}
