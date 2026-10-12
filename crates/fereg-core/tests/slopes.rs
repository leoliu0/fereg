//! SlopeSolver (plan Phase 7) against the dense explicit-dummy oracle.

use fereg_core::absorber::{Absorber, FeCoefs, SolveInfo, SolveOpts};
use fereg_core::gather::{gather_rows, to_kept_order};
use fereg_core::par::with_pool;
use fereg_core::rng::SplitMix64;
use fereg_core::slopes::SlopeSolver;
use fereg_core::solve::SolverConfig;
use fereg_core::{AbsorbTerm, AbsorberKind, Ctx, Family, FactorValues, Prep, Session, Vce, WeightKind};

/// (factor indices, intercept, slope variable indices).
type TermSpec = (Vec<usize>, bool, Vec<usize>);

struct Design {
    factors: Vec<Vec<f64>>,
    slopes: Vec<Vec<f64>>,
    terms: Vec<TermSpec>,
    w: Option<Vec<f64>>,
    seed: u64,
}

fn normal(rng: &mut SplitMix64) -> f64 {
    let u1 = rng.next_f64().max(1e-300);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * rng.next_f64()).cos()
}

/// indiv × year panel; firm random (or a band on a ring); slopes x (constant within every 7th
/// indiv: degenerate levels) and z.
fn panel(nindiv: usize, nyear: usize, nfirm: usize, band: bool, weighted: bool, seed: u64) -> Design {
    let mut rng = SplitMix64::new(seed);
    let n = nindiv * nyear;
    let indiv: Vec<f64> = (0..n).map(|i| (i / nyear) as f64).collect();
    let year: Vec<f64> = (0..n).map(|i| (i % nyear) as f64).collect();
    let firm: Vec<f64> = (0..n)
        .map(|i| if band { ((i / nyear * nfirm / nindiv + i % 3) % nfirm) as f64 } else { (rng.next_u64() % nfirm as u64) as f64 })
        .collect();
    let x: Vec<f64> = (0..n).map(|i| if (i / nyear) % 7 == 0 { 1.5 } else { 2.0 + normal(&mut rng) }).collect();
    let z: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
    let w = weighted.then(|| (0..n).map(|_| 0.5 + rng.next_f64()).collect());
    Design { factors: vec![indiv, year, firm], slopes: vec![x, z], terms: Vec::new(), w, seed }
}

fn with_terms(mut d: Design, terms: &[TermSpec]) -> Design {
    d.terms = terms.to_vec();
    d
}

fn prep(d: &Design) -> Prep<'_> {
    Prep {
        n: d.factors[0].len(),
        factors: d.factors.iter().map(|f| FactorValues::Num(f)).collect(),
        absorb: d.terms.iter().map(|(f, i, s)| AbsorbTerm { factors: f.clone(), intercept: *i, slopes: s.clone(), save: true }).collect(),
        slope_vars: d.slopes.iter().map(Vec::as_slice).collect(),
        weights: d.w.as_deref(),
        weight_kind: if d.w.is_some() { WeightKind::Aweight } else { WeightKind::None },
        vce: Vce::Unadjusted,
        drop_singletons: true,
        family: Family::Gaussian,
        y_for_sep: None,
        sep_fe: false,
    }
}

/// Random columns (kept order): noise plus intercept and slope effects of every term.
fn columns(s: &Session, r: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = SplitMix64::new(seed);
    let n = s.n();
    (0..r)
        .map(|_| {
            let mut v: Vec<f64> = (0..n).map(|_| normal(&mut rng) + 5.0).collect();
            for t in s.terms() {
                if t.intercept {
                    let eff: Vec<f64> = (0..t.g).map(|_| 3.0 * normal(&mut rng)).collect();
                    for i in 0..n {
                        v[i] += eff[t.codes[i] as usize];
                    }
                }
                for x in &t.slopes {
                    let eff: Vec<f64> = (0..t.g).map(|_| normal(&mut rng)).collect();
                    for i in 0..n {
                        v[i] += eff[t.codes[i] as usize] * x[i];
                    }
                }
            }
            v
        })
        .collect()
}

fn opts() -> SolveOpts {
    SolveOpts { tol: 1e-12, maxiter: 10_000, want_coefs: true }
}

struct Pair {
    dense: Session,
    fast: SlopeSolver,
    cols: Vec<Vec<f64>>,
}

fn setup(d: &Design, cfg: SolverConfig, r: usize) -> Pair {
    let ctx = Ctx::quiet(4);
    let dense = Session::prepare_with(prep(d), &ctx, AbsorberKind::Dense).unwrap();
    let cols = columns(&dense, r, d.seed);
    let terms = dense.terms().to_vec();
    let fast = with_pool(4, |pool| SlopeSolver::new(&terms, dense.weights(), dense.n(), cfg, pool, &ctx)).unwrap().unwrap();
    Pair { dense, fast, cols }
}

/// Kept-order columns of a row-order block.
fn split(rows: &[f64], order: &[u32], r: usize) -> Vec<Vec<f64>> {
    (0..r).map(|j| to_kept_order(order, &(0..order.len()).map(|i| rows[i * r + j]).collect::<Vec<_>>())).collect()
}

/// Residualizes with both absorbers: (fast, dense) kept-order residuals and the fast info.
fn run(p: &mut Pair, warm: Option<&FeCoefs>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, SolveInfo) {
    let ctx = Ctx::quiet(4);
    let r = p.cols.len();
    let refs: Vec<&[f64]> = p.cols.iter().map(Vec::as_slice).collect();
    with_pool(4, |pool| {
        let mut rows_d = gather_rows(p.dense.row_order(), &refs, pool).unwrap();
        p.dense.absorber_mut().residualize(&mut rows_d, r, &opts(), None, pool, &ctx).unwrap();
        let mut rows_f = gather_rows(p.fast.row_order(), &refs, pool).unwrap();
        let info = p.fast.residualize(&mut rows_f, r, &opts(), warm, pool, &ctx).unwrap();
        (split(&rows_f, p.fast.row_order(), r), split(&rows_d, p.dense.row_order(), r), info)
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

/// Checks `cfg` and the same configuration with large blocks pulled through the n×R scratch.
fn check(d: &Design, cfg: SolverConfig, r: usize) -> SolveInfo {
    let mut first = None;
    for c in [cfg, SolverConfig { acc_bytes_max: 0, ..cfg }] {
        let mut p = setup(d, c, r);
        let (f, dn, info) = run(&mut p, None);
        let err = max_rel(&f, &dn);
        assert!(err <= 1e-9, "max rel error {err:e} (iterations {}, escalated {}, {c:?}, terms {:?})", info.iterations, info.escalated, d.terms);
        first.get_or_insert(info);
    }
    first.expect("ran")
}

fn limit(small_limit: usize) -> SolverConfig {
    SolverConfig { small_limit, ..SolverConfig::default() }
}

const INDIV: usize = 0;
const YEAR: usize = 1;
const FIRM: usize = 2;
const X: usize = 0;
const Z: usize = 1;

fn t(f: usize, icpt: bool, s: &[usize]) -> TermSpec {
    (vec![f], icpt, s.to_vec())
}

#[test]
fn intercepts_plus_firm_slopes() {
    let d = with_terms(panel(200, 8, 40, false, false, 1), &[t(INDIV, true, &[]), t(YEAR, true, &[]), t(FIRM, true, &[X])]);
    let info = check(&d, SolverConfig::default(), 3);
    assert!(info.iterations <= 3, "dense exact path took {} iterations", info.iterations);
    check(&d, limit(0), 3);
    check(&d, limit(30), 2);
}

#[test]
fn slope_terms_with_and_without_intercepts() {
    let base = || panel(150, 8, 30, false, true, 2);
    // indiv##c.x with year: E = year (the only intercept-only term), indiv is a width-2 block.
    let d = with_terms(base(), &[t(INDIV, true, &[X]), t(YEAR, true, &[])]);
    check(&d, SolverConfig::default(), 2);
    check(&d, limit(0), 2);
    // firm#c.x (no intercept) with indiv.
    let d = with_terms(base(), &[t(FIRM, false, &[X]), t(INDIV, true, &[])]);
    check(&d, SolverConfig::default(), 2);
    check(&d, limit(0), 2);
    // No intercept-only term: E = indiv#c.x (most levels), firm#c.z a block.
    let d = with_terms(base(), &[t(INDIV, false, &[X]), t(FIRM, false, &[Z])]);
    check(&d, SolverConfig::default(), 2);
    check(&d, limit(0), 2);
    // Two slopes on two terms plus an interaction term.
    let mut d = with_terms(base(), &[t(INDIV, true, &[]), t(FIRM, true, &[X, Z])]);
    d.terms.push((vec![YEAR, FIRM], true, vec![]));
    check(&d, SolverConfig::default(), 2);
    check(&d, limit(0), 2);
}

#[test]
fn single_slope_term_is_closed_form() {
    let d = with_terms(panel(200, 6, 10, false, true, 3), &[t(INDIV, true, &[X, Z])]);
    let info = check(&d, SolverConfig::default(), 2);
    assert_eq!(info.iterations, 1);
    let d = with_terms(panel(200, 6, 10, false, false, 4), &[t(INDIV, false, &[X])]);
    assert_eq!(check(&d, SolverConfig::default(), 1).iterations, 1);
}

#[test]
fn band_firm_slopes_escalate_to_sparse_cholesky() {
    let d = with_terms(panel(400, 6, 300, true, false, 5), &[t(INDIV, true, &[]), t(YEAR, true, &[]), t(FIRM, true, &[X])]);
    let info = check(&d, limit(20), 2);
    assert!(info.escalated, "band design must escalate ({} iterations)", info.iterations);
    assert!(info.iterations <= 40, "band design took {} iterations", info.iterations);
    // Budgets exceeded: block Jacobi only, still accurate to the tolerance.
    let info = check(&d, SolverConfig { max_block_nnz: 0, ..limit(20) }, 1);
    assert!(!info.escalated, "nothing to escalate to within a zero budget");
}

#[test]
fn set_weights_warm_start_and_refactor() {
    let d = with_terms(panel(300, 6, 200, true, true, 6), &[t(INDIV, true, &[]), t(FIRM, true, &[X])]);
    for cfg in [SolverConfig::default(), limit(0), SolverConfig { acc_bytes_max: 0, ..limit(0) }] {
        let mut p = setup(&d, cfg, 2);
        let (f, dn, info0) = run(&mut p, None);
        assert!(max_rel(&f, &dn) <= 1e-9);
        let coefs = {
            let ctx = Ctx::quiet(4);
            let refs: Vec<&[f64]> = p.cols.iter().map(Vec::as_slice).collect();
            with_pool(4, |pool| {
                let mut rows = gather_rows(p.fast.row_order(), &refs, pool).unwrap();
                p.fast.residualize(&mut rows, 2, &opts(), None, pool, &ctx).unwrap().coefs.unwrap()
            })
            .unwrap()
        };
        let mut rng = SplitMix64::new(61);
        for round in 0..3 {
            let wk: Vec<f64> = (0..p.dense.n()).map(|_| 0.3 + 2.0 * rng.next_f64()).collect();
            let ctx = Ctx::quiet(4);
            with_pool(4, |pool| {
                let wf: Vec<f64> = p.fast.row_order().iter().map(|&k| wk[k as usize]).collect();
                p.fast.set_weights(Some(wf), pool, &ctx).unwrap();
                let wd: Vec<f64> = p.dense.row_order().iter().map(|&k| wk[k as usize]).collect();
                p.dense.absorber_mut().set_weights(Some(wd), pool, &ctx).unwrap();
            })
            .unwrap();
            if round == 2 {
                p.fast.request_refactor();
            }
            let (f, dn, info) = run(&mut p, Some(&coefs));
            let err = max_rel(&f, &dn);
            assert!(err <= 1e-9, "after set_weights (round {round}, {cfg:?}): {err:e} ({} iterations, first {})", info.iterations, info0.iterations);
        }
    }
}

#[test]
fn fit_rows_and_term_values_reconstruct_the_fit() {
    let d = with_terms(panel(150, 8, 25, false, true, 7), &[t(INDIV, true, &[]), t(YEAR, true, &[]), t(FIRM, true, &[X, Z])]);
    for cfg in [SolverConfig::default(), limit(0)] {
        let mut p = setup(&d, cfg, 2);
        let ctx = Ctx::quiet(4);
        let refs: Vec<&[f64]> = p.cols.iter().map(Vec::as_slice).collect();
        let (resid, coefs, fits) = with_pool(4, |pool| {
            let mut rows = gather_rows(p.fast.row_order(), &refs, pool).unwrap();
            let coefs = p.fast.residualize(&mut rows, 2, &opts(), None, pool, &ctx).unwrap().coefs.unwrap();
            let fits: Vec<Vec<f64>> = (0..2).map(|j| to_kept_order(p.fast.row_order(), &p.fast.fit_rows(&coefs, j, pool))).collect();
            (split(&rows, p.fast.row_order(), 2), coefs, fits)
        })
        .unwrap();
        let terms = p.dense.terms().to_vec();
        for j in 0..2 {
            let scale = p.cols[j].iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for i in 0..p.dense.n() {
                let e = (p.cols[j][i] - resid[j][i] - fits[j][i]).abs();
                assert!(e <= 1e-9 * scale, "fit_rows: row {i} col {j}: {e:e}");
            }
            let vals = p.fast.term_values(&coefs, j);
            assert_eq!(vals.len(), terms.iter().map(|t| usize::from(t.intercept) + t.slopes.len()).sum::<usize>());
            let mut slot = 0;
            let mut recon = vec![0.0; p.dense.n()];
            for t in &terms {
                if t.intercept {
                    for i in 0..p.dense.n() {
                        recon[i] += vals[slot][t.codes[i] as usize];
                    }
                    slot += 1;
                }
                for x in &t.slopes {
                    for i in 0..p.dense.n() {
                        recon[i] += vals[slot][t.codes[i] as usize] * x[i];
                    }
                    slot += 1;
                }
            }
            for i in 0..p.dense.n() {
                assert!((recon[i] - fits[j][i]).abs() <= 1e-9 * scale, "term_values: row {i}");
            }
            // Normalization: the year intercept has weighted mean 0.
            let w = p.dense.weights().unwrap();
            let year = &terms[1];
            let mean: f64 = (0..p.dense.n()).map(|i| w[i] * vals[1][year.codes[i] as usize]).sum::<f64>() / w.iter().sum::<f64>();
            assert!(mean.abs() <= 1e-9 * scale, "year mean {mean:e}");
        }
    }
}

#[test]
fn session_fast_path_handles_slopes() {
    let d = with_terms(panel(150, 8, 25, false, false, 8), &[t(INDIV, true, &[]), t(FIRM, true, &[X])]);
    let ctx = Ctx::quiet(3);
    let fast = Session::prepare(prep(&d), &ctx).unwrap();
    let dense = Session::prepare_with(prep(&d), &ctx, AbsorberKind::Dense).unwrap();
    assert_eq!(fast.keep(), dense.keep());
    let cols = columns(&dense, 2, 9);
    let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
    let mut fast = fast;
    let mut dense = dense;
    let (rf, rd) = with_pool(3, |pool| {
        let mut rf = gather_rows(fast.row_order(), &refs, pool).unwrap();
        fast.absorber_mut().residualize(&mut rf, 2, &opts(), None, pool, &ctx).unwrap();
        let mut rd = gather_rows(dense.row_order(), &refs, pool).unwrap();
        dense.absorber_mut().residualize(&mut rd, 2, &opts(), None, pool, &ctx).unwrap();
        (rf, rd)
    })
    .unwrap();
    let err = max_rel(&split(&rf, fast.row_order(), 2), &split(&rd, dense.row_order(), 2));
    assert!(err <= 1e-9, "{err:e}");
}
