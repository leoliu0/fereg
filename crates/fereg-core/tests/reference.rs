//! The dense oracle must be an exact W-orthogonal projection onto the FE space.

use fereg_core::absorber::SolveOpts;
use fereg_core::par::with_pool;
use fereg_core::rng::SplitMix64;
use fereg_core::{AbsorbTerm, AbsorberKind, Ctx, Family, FactorValues, Prep, Session, Vce, WeightKind};

fn panel(n: usize, g1: usize, g2: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = SplitMix64::new(seed);
    let a: Vec<f64> = (0..n).map(|_| (rng.next_u64() % g1 as u64) as f64).collect();
    let b: Vec<f64> = (0..n).map(|_| (rng.next_u64() % g2 as u64) as f64).collect();
    let x: Vec<f64> = (0..n).map(|_| rng.next_f64() - 0.5).collect();
    let w: Vec<f64> = (0..n).map(|_| 0.5 + rng.next_f64()).collect();
    (a, b, x, w)
}

#[test]
fn dense_residuals_are_w_orthogonal_to_every_dummy_and_slope_column() {
    let n = 600;
    let (a, b, x, w) = panel(n, 40, 7, 1);
    let term = |f: usize, intercept: bool, slopes: Vec<usize>| AbsorbTerm { factors: vec![f], intercept, slopes, save: true };
    let prep = Prep {
        n,
        factors: vec![FactorValues::Num(&a), FactorValues::Num(&b)],
        absorb: vec![term(0, true, vec![]), term(1, true, vec![0])],
        slope_vars: vec![&x],
        weights: Some(&w),
        weight_kind: WeightKind::Aweight,
        vce: Vce::Unadjusted,
        drop_singletons: true,
        family: Family::Gaussian,
        y_for_sep: None,
        sep_fe: false,
    };
    let ctx = Ctx::quiet(2);
    let mut s = Session::prepare_with(prep, &ctx, AbsorberKind::Dense).unwrap();
    let nk = s.n();
    let y: Vec<f64> = (0..nk).map(|i| (i as f64).sin() + 3.0).collect();
    let mut rows = s.gather(&[&y], &ctx).unwrap();
    let orig = rows.clone();
    let terms = s.terms().to_vec();
    let wk = s.weights().unwrap().to_vec();
    with_pool(2, |pool| {
        let ab = s.absorber_mut();
        let opts = SolveOpts { tol: 1e-12, maxiter: 100, want_coefs: true };
        let info = ab.residualize(&mut rows, 1, &opts, None, pool, &ctx).unwrap();
        // D'W z~ = 0 for every column of D.
        for t in &terms {
            let mut ic = vec![0.0; t.g];
            let mut sc = vec![0.0; t.g];
            for i in 0..nk {
                ic[t.codes[i] as usize] += wk[i] * rows[i];
                if let Some(xs) = t.slopes.first() {
                    sc[t.codes[i] as usize] += wk[i] * xs[i] * rows[i];
                }
            }
            assert!(ic.iter().chain(&sc).all(|v| v.abs() < 1e-10), "not orthogonal");
        }
        // z = z~ + fit.
        let coefs = info.coefs.unwrap();
        let fit = ab.fit_rows(&coefs, 0, pool);
        for i in 0..nk {
            assert!((orig[i] - rows[i] - fit[i]).abs() < 1e-10);
        }
        // Idempotent.
        let before = rows.clone();
        ab.residualize(&mut rows, 1, &opts, None, pool, &ctx).unwrap();
        assert!(rows.iter().zip(&before).all(|(p, q)| (p - q).abs() < 1e-10));
    })
    .unwrap();
}

#[test]
fn singletons_cycle_until_no_term_changes() {
    // Obs 3 is a singleton in factor b; dropping it makes obs 2 a singleton in factor a.
    let a = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
    let b = [0.0, 1.0, 0.0, 2.0, 1.0, 0.0];
    let prep = Prep {
        n: 6,
        factors: vec![FactorValues::Num(&a), FactorValues::Num(&b)],
        absorb: vec![
            AbsorbTerm { factors: vec![0], intercept: true, slopes: vec![], save: false },
            AbsorbTerm { factors: vec![1], intercept: true, slopes: vec![], save: false },
        ],
        slope_vars: vec![],
        weights: None,
        weight_kind: WeightKind::None,
        vce: Vce::Unadjusted,
        drop_singletons: true,
        family: Family::Gaussian,
        y_for_sep: None,
        sep_fe: false,
    };
    let s = Session::prepare_with(prep, &Ctx::quiet(1), AbsorberKind::Dense).unwrap();
    assert_eq!(s.keep(), &[true, true, false, false, true, true]);
    assert_eq!(s.num_singletons(), 2);
}
