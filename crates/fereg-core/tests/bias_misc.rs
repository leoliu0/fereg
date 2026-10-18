//! Bias correction: normalization identity, fixed-effect re-solve, thread-count determinism,
//! errors, and (ignored) timings.

mod bias_common;

use bias_common::*;
use fereg_core::Ctx;
use fereg_core::fit::{
    self, ApeAt, ApeOptions, BiasMethod, BiasOptions, FactorCol, FeTerm, FitInput, Flavor, JackknifeOptions, Model, Normalization, Permutation, Sampling, Structure,
    Variant, VcovKind, WhichBias,
};
use fereg_core::rng::SplitMix64;

/// A simulated classic panel: n_i individuals x n_t periods, regressors x1, x2 (continuous), d
/// (binary), y logit with individual and time effects. `gap` deletes rows at random.
struct Panel {
    i: Vec<f64>,
    t: Vec<f64>,
    y: Vec<f64>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    d: Vec<f64>,
}

fn normal(r: &mut SplitMix64) -> f64 {
    let (u, v) = (r.next_f64().max(1e-300), r.next_f64());
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

fn panel(n_i: usize, n_t: usize, gap: f64, seed: u64) -> Panel {
    let mut r = SplitMix64::new(seed);
    let a: Vec<f64> = (0..n_i).map(|_| 0.8 * normal(&mut r)).collect();
    let g: Vec<f64> = (0..n_t).map(|_| 0.5 * normal(&mut r)).collect();
    let mut p = Panel { i: vec![], t: vec![], y: vec![], x1: vec![], x2: vec![], d: vec![] };
    for i in 0..n_i {
        for t in 0..n_t {
            let x1 = normal(&mut r) + 0.3 * a[i];
            let x2 = normal(&mut r);
            let d = f64::from(r.next_f64() < 0.4);
            let e = 0.7 * x1 - 0.5 * x2 + 0.6 * d + a[i] + g[t];
            let u = r.next_f64();
            let y = f64::from(u < 1.0 / (1.0 + (-e).exp()));
            if r.next_f64() < gap {
                continue;
            }
            p.i.push(i as f64 + 1.0);
            p.t.push(t as f64 + 1.0);
            p.y.push(y);
            p.x1.push(x1);
            p.x2.push(x2);
            p.d.push(d);
        }
    }
    p
}

fn two_way(p: &Panel) -> FitInput<'_> {
    FitInput {
        n: p.y.len(),
        y: &p.y,
        x: vec![&p.x1, &p.x2, &p.d],
        factors: vec![FactorCol::F64(&p.i), FactorCol::F64(&p.t)],
        factor_names: vec!["i".into(), "t".into()],
        fe: vec![FeTerm { factors: vec![0], intercept: true, slopes: vec![] }, FeTerm { factors: vec![1], intercept: true, slopes: vec![] }],
        ..Default::default()
    }
}

fn jk(variant: Variant) -> BiasMethod {
    BiasMethod::Jackknife(JackknifeOptions { variant, partitions: vec![] })
}

#[test]
fn logitfe_equals_alpaca_on_balanced_panels() {
    let p = panel(150, 8, 0.0, 11);
    let input = two_way(&p);
    let ctx = Ctx::quiet(4);
    let sp = spec(Flavor::Fixest);
    for l in [0usize, 1, 2] {
        for which in [WhichBias::Both, WhichBias::Individual, WhichBias::Time] {
            if which == WhichBias::Time && l > 0 {
                continue;
            }
            let a = BiasOptions { bandwidth: l, which, ..BiasOptions::default() };
            let b = BiasOptions { normalization: Normalization::Logitfe, panel: Some(0), time: Some(1), ..a.clone() };
            let ra = fit::bias_corr(&input, &sp, &a, &ctx).unwrap();
            let rb = fit::bias_corr(&input, &sp, &b, &ctx).unwrap();
            // Only whole individuals are dropped: the estimation panel stays balanced.
            check(&format!("L={l} {which:?}"), &rb.report.bias_term, &ra.report.bias_term, 1e-10);
            let apes = |o: &BiasOptions, c: &[f64]| {
                fit::apes(&input, &sp, ApeAt::Corrected { options: o, coef: c, ape_jackknife: None }, &ApeOptions::default(), &ctx).unwrap()
            };
            let (aa, ab) = (apes(&a, &ra.fit.coef), apes(&b, &rb.fit.coef));
            check(&format!("L={l} {which:?} ape bias"), ab.bias_term.as_ref().unwrap(), aa.bias_term.as_ref().unwrap(), 1e-9);
        }
    }
}

#[test]
fn fe_resolve_is_exact() {
    let p = panel(300, 6, 0.1, 5);
    let input = two_way(&p);
    let ctx = Ctx::quiet(4);
    for method in [BiasMethod::Analytical, jk(Variant::Ss2)] {
        let bf = fit::bias_corr(&input, &spec(Flavor::Fixest), &BiasOptions { method, ..BiasOptions::default() }, &ctx).unwrap();
        let resid = bf.fit.resid.as_ref().unwrap();
        for q in 0..2 {
            let col = if q == 0 { &p.i } else { &p.t };
            let mut s = std::collections::BTreeMap::<i64, f64>::new();
            for (r, &i) in bf.fit.obs.iter().enumerate() {
                *s.entry(col[i as usize] as i64).or_default() += resid[r];
            }
            let worst = s.values().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(worst < 1e-10, "FE score {worst:e}");
        }
        // The reported fixed effects reproduce eta - x'beta.
        let fx = &bf.fit.fixef;
        let eta = bf.fit.eta.as_ref().unwrap();
        let lv = |q: usize, v: f64| {
            let col = if q == 0 { &p.i } else { &p.t };
            let first = &fx[q].first_obs;
            let pos = first.iter().position(|&o| col[o as usize] == v).unwrap();
            fx[q].values[pos]
        };
        for (r, &i) in bf.fit.obs.iter().enumerate().step_by(37) {
            let i = i as usize;
            let xb = bf.fit.coef[0] * p.x1[i] + bf.fit.coef[1] * p.x2[i] + bf.fit.coef[2] * p.d[i];
            let fe = lv(0, p.i[i]) + lv(1, p.t[i]);
            assert!((xb + fe - eta[r]).abs() < 1e-9, "fixef + xb vs eta");
        }
    }
}

fn identical_across_threads(p: &Panel, methods: &[BiasOptions]) {
    let input = two_way(p);
    let sp = FitSpec2::new();
    for o in methods {
        let run = |t: usize| {
            let ctx = Ctx::quiet(t);
            let b = fit::bias_corr(&input, &sp.0, o, &ctx).unwrap();
            let at = ApeAt::Corrected { options: o, coef: &b.fit.coef, ape_jackknife: b.report.ape_jackknife.as_deref() };
            let a = fit::apes(&input, &sp.0, at, &ApeOptions { n_pop: Some(1e4), weak_exo: true, ..ApeOptions::default() }, &ctx).unwrap();
            (b, a)
        };
        let ((b1, a1), (b16, a16)) = (run(1), run(16));
        assert_eq!(b1.report, b16.report, "{:?}", o.method);
        assert_eq!(b1.fit.coef, b16.fit.coef);
        assert_eq!(b1.fit.vcov, b16.fit.vcov);
        assert_eq!(b1.fit.eta, b16.fit.eta);
        assert_eq!(b1.fit.fixef, b16.fit.fixef);
        assert_eq!(format!("{:?}", b1.fit.stats), format!("{:?}", b16.fit.stats));
        assert_eq!(a1, a16);
    }
}

#[test]
fn bit_identical_across_threads() {
    // 40 000 rows (> 2 * CHUNK): every kernel runs on a multi-thread pool.
    let p = panel(5000, 8, 0.0, 3);
    assert!(p.y.len() >= 2 * fereg_core::par::CHUNK);
    let perm_i: Vec<u32> = (0..5000).map(|i| (i * 7 + 3) % 5000).collect();
    identical_across_threads(
        &p,
        &[
            BiasOptions { bandwidth: 1, ..BiasOptions::default() },
            BiasOptions { method: jk(Variant::Ss2), ..BiasOptions::default() },
            BiasOptions {
                method: BiasMethod::Jackknife(JackknifeOptions {
                    variant: Variant::Ss1,
                    partitions: vec![
                        Permutation { individual: Some(perm_i), period: Some(vec![3, 1, 0, 7, 6, 5, 4, 2]) },
                        Permutation { individual: None, period: Some(vec![7, 6, 5, 4, 3, 2, 1, 0]) },
                    ],
                }),
                ..BiasOptions::default()
            },
        ],
    );
    // Delete-one variants (many concurrent subpanel fits) on a small panel.
    identical_across_threads(&panel(120, 7, 0.1, 3), &[BiasOptions { method: jk(Variant::Jj), ..BiasOptions::default() }]);
}

#[test]
fn rows_missing_a_cluster_value_stay_out_of_subpanels() {
    // Fixest flavor drops rows with a missing cluster value from the estimation (and N*); the
    // jackknife subpanels must not bring them back.
    let p = panel(80, 6, 0.0, 41);
    let cl: Vec<f64> = (0..p.y.len()).map(|r| if r % 17 == 5 { f64::NAN } else { (r % 9) as f64 }).collect();
    let mut with = two_way(&p);
    with.factors.push(FactorCol::F64(&cl));
    with.factor_names.push("cl".into());
    with.cluster = vec![vec![2]];
    let keep: Vec<usize> = (0..p.y.len()).filter(|&r| cl[r].is_finite()).collect();
    let sub = |v: &[f64]| keep.iter().map(|&r| v[r]).collect::<Vec<f64>>();
    let q = Panel { i: sub(&p.i), t: sub(&p.t), y: sub(&p.y), x1: sub(&p.x1), x2: sub(&p.x2), d: sub(&p.d) };
    let without = two_way(&q);
    let ctx = Ctx::quiet(4);
    let sp = fit::FitSpec { vcov: VcovKind::Cluster, ..spec(Flavor::Fixest) };
    for variant in [Variant::Ss2, Variant::Jj] {
        let o = BiasOptions { method: jk(variant), ..BiasOptions::default() };
        let a = fit::bias_corr(&with, &sp, &o, &ctx).unwrap();
        let b = fit::bias_corr(&without, &spec(Flavor::Fixest), &o, &ctx).unwrap();
        check(&format!("{variant:?} coef"), &a.fit.coef, &b.fit.coef, 1e-10);
        check(&format!("{variant:?} ape"), a.report.ape_jackknife.as_ref().unwrap(), b.report.ape_jackknife.as_ref().unwrap(), 1e-10);
        assert_eq!(a.report.n_full, b.report.n_full);
    }
}

#[test]
fn logitfe_term_order_does_not_matter() {
    // Fit layer, logitfe normalization with lags(1) (weak exogeneity forced): fe = [t, id] gives
    // the numbers of fe = [id, t] (the lag groups are the panel variable's term).
    let p = panel(150, 8, 0.1, 17);
    let a_in = two_way(&p);
    let mut b_in = two_way(&p);
    b_in.fe.reverse();
    let ctx = Ctx::quiet(4);
    let sp = spec(Flavor::Fixest);
    let o = BiasOptions { bandwidth: 1, normalization: Normalization::Logitfe, panel: Some(0), time: Some(1), ..BiasOptions::default() };
    let ao = ApeOptions { n_pop: Some(f64::INFINITY), ..ApeOptions::default() };
    let run = |input: &FitInput| {
        let b = fit::bias_corr(input, &sp, &o, &ctx).unwrap();
        let a = fit::apes(input, &sp, ApeAt::Corrected { options: &o, coef: &b.fit.coef, ape_jackknife: None }, &ao, &ctx).unwrap();
        (b.fit.coef, a)
    };
    let ((ca, aa), (cb, ab)) = (run(&a_in), run(&b_in));
    assert!(aa.weak_exo_used && aa.adj == 1.0);
    check("coef", &cb, &ca, 1e-9);
    check("ape", &ab.delta, &aa.delta, 1e-9);
    check_mat("ape vcov", &ab.vcov, &aa.vcov, 1e-9);
}

/// Spec wrapper (keeps the test bodies short).
struct FitSpec2(fit::FitSpec);
impl FitSpec2 {
    fn new() -> Self {
        FitSpec2(spec(Flavor::Fixest))
    }
}

fn err(r: Result<fit::BiasFit, fereg_core::RfError>) -> String {
    match r {
        Ok(_) => panic!("expected an error"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn errors() {
    let p = panel(60, 6, 0.0, 9);
    let input = two_way(&p);
    let ctx = Ctx::quiet(2);
    let sp = spec(Flavor::Fixest);
    let d = BiasOptions::default();
    // Not a logit model.
    let e = err(fit::bias_corr(&input, &fit::FitSpec { model: Model::Poisson, ..sp.clone() }, &d, &ctx));
    assert!(e.contains("only available for logit"), "{e}");
    // Varying slopes.
    let mut sl = two_way(&p);
    sl.slopes = vec![&p.x2];
    sl.fe[0].slopes = vec![0];
    let e = err(fit::bias_corr(&sl, &sp, &d, &ctx));
    assert!(e.contains("intercept-only"), "{e}");
    // Number of terms vs structure.
    let mut three = two_way(&p);
    three.fe.push(FeTerm { factors: vec![0, 1], intercept: true, slopes: vec![] });
    let e = err(fit::bias_corr(&three, &sp, &d, &ctx));
    assert!(e.contains("one- or two-way"), "{e}");
    let mut one = two_way(&p);
    one.fe.truncate(1);
    let e = err(fit::bias_corr(&one, &sp, &BiasOptions { structure: Structure::Network, ..d.clone() }, &ctx));
    assert!(e.contains("two- or three-way"), "{e}");
    // Network: only both biases (bias_corr and corrected APEs).
    for which in [WhichBias::Individual, WhichBias::Time] {
        let o = BiasOptions { structure: Structure::Network, which, ..d.clone() };
        let e = err(fit::bias_corr(&input, &sp, &o, &ctx));
        assert_eq!(e, "network structure: only the correction of both biases is available");
        let e = fit::apes(&input, &sp, ApeAt::Corrected { options: &o, coef: &[0.0, 0.0, 0.0], ape_jackknife: None }, &ApeOptions::default(), &ctx).err().unwrap();
        assert_eq!(e.to_string(), "network structure: only the correction of both biases is available");
    }
    // Jackknife on a network structure.
    let e = err(fit::bias_corr(&input, &sp, &BiasOptions { structure: Structure::Network, method: jk(Variant::Ss2), ..d.clone() }, &ctx));
    assert!(e.contains("not available for the network"), "{e}");
    // Bandwidth too large (6 periods).
    let e = err(fit::bias_corr(&input, &sp, &BiasOptions { bandwidth: 6, ..d.clone() }, &ctx));
    assert!(e.contains("too large"), "{e}");
    let e = err(fit::bias_corr(&input, &sp, &BiasOptions { bandwidth: 6, normalization: Normalization::Logitfe, panel: Some(0), time: Some(1), ..d.clone() }, &ctx));
    assert!(e.contains("too large"), "{e}");
    // CRV3 at fit time.
    let e = err(fit::bias_corr(&input, &fit::FitSpec { vcov: VcovKind::Crv3, ..sp.clone() }, &d, &ctx));
    assert!(e.contains("CRV3 is not available after bias correction"), "{e}");
    // Time bias on a one-way model.
    let e = err(fit::bias_corr(&one, &sp, &BiasOptions { which: WhichBias::Time, ..d.clone() }, &ctx));
    assert!(e.contains("time fixed effect"), "{e}");
    // Multiple partitions with a delete-one variant.
    let e = err(fit::bias_corr(
        &input,
        &sp,
        &BiasOptions {
            method: BiasMethod::Jackknife(JackknifeOptions { variant: Variant::Jj, partitions: vec![Permutation { individual: None, period: Some((0..6).collect()) }] }),
            ..d.clone()
        },
        &ctx,
    ));
    assert!(e.contains("ss1 and ss2"), "{e}");
    // A one-FE jackknife needing time halves without a time variable.
    let e = err(fit::bias_corr(&one, &sp, &BiasOptions { method: jk(Variant::Ss2), ..d.clone() }, &ctx));
    assert!(e.contains("time variable"), "{e}");
    // Options without effect are errors.
    let e = err(fit::bias_corr(&input, &sp, &BiasOptions { method: jk(Variant::Ss2), bandwidth: 1, ..d.clone() }, &ctx));
    assert!(e.contains("has no effect"), "{e}");
    let e = err(fit::bias_corr(&input, &sp, &BiasOptions { which: WhichBias::Time, bandwidth: 1, ..d.clone() }, &ctx));
    assert!(e.contains("has no effect"), "{e}");
    let ind_only = |variant| BiasOptions {
        which: WhichBias::Individual,
        method: BiasMethod::Jackknife(JackknifeOptions { variant, partitions: vec![Permutation { individual: Some((0..60).rev().collect()), period: None }] }),
        ..d.clone()
    };
    for v in [Variant::Ss1, Variant::Ss2] {
        let e = err(fit::bias_corr(&input, &sp, &ind_only(v), &ctx));
        assert!(e.contains("multiple partitions have no effect"), "{e}");
    }
    // Missing values in the time / panel variable on estimation rows.
    let mut tna = p.t.clone();
    tna[7] = f64::NAN;
    let mut one_na = two_way(&p);
    one_na.fe.truncate(1);
    one_na.factors[1] = FactorCol::F64(&tna);
    let e = err(fit::bias_corr(&one_na, &sp, &BiasOptions { method: jk(Variant::Jj), time: Some(1), ..d.clone() }, &ctx));
    assert_eq!(e, "the time variable has missing values in the estimation sample");
    let mut ina = p.i.clone();
    ina[3] = f64::NAN;
    let mut lf_na = two_way(&p);
    lf_na.fe.truncate(1);
    lf_na.fe[0].factors = vec![1];
    lf_na.factors.push(FactorCol::F64(&ina));
    lf_na.factor_names.push("id2".into());
    let e = err(fit::bias_corr(&lf_na, &sp, &BiasOptions { normalization: Normalization::Logitfe, panel: Some(2), time: Some(1), ..d.clone() }, &ctx));
    assert_eq!(e, "the panel variable has missing values in the estimation sample");
    // Frequency weights.
    let wts = vec![2.0; p.y.len()];
    let mut fw = two_way(&p);
    fw.weights = Some(&wts);
    let e = err(fit::bias_corr(&fw, &fit::FitSpec { flavor: Flavor::Pyfixest, weights_type: fit::WeightsType::Fweights, ..sp.clone() }, &d, &ctx));
    assert_eq!(e, "bias correction and APEs do not support frequency weights");
    // n_pop: +Inf = infinite population (adj 1); NaN / negative are errors.
    let inf = fit::apes(&input, &sp, ApeAt::Uncorrected, &ApeOptions { n_pop: Some(f64::INFINITY), ..ApeOptions::default() }, &ctx).unwrap();
    assert_eq!(inf.adj, 1.0);
    for bad in [f64::NAN, -1.0, f64::NEG_INFINITY] {
        let e = fit::apes(&input, &sp, ApeAt::Uncorrected, &ApeOptions { n_pop: Some(bad), ..ApeOptions::default() }, &ctx).err().unwrap();
        assert!(e.to_string().contains("n_pop"), "{e}");
    }
    // n.pop below N*: note and adj = 0.
    let a = fit::apes(&input, &sp, ApeAt::Uncorrected, &ApeOptions { n_pop: Some(10.0), ..ApeOptions::default() }, &ctx).unwrap();
    assert_eq!(a.adj, 0.0);
    assert!(a.notes[0].contains("Correction factor set to zero"), "{:?}", a.notes);
    let b = fit::apes(&input, &sp, ApeAt::Uncorrected, &ApeOptions::default(), &ctx).unwrap();
    assert_eq!(a.vcov, b.vcov);
}

#[test]
fn subpanel_losing_a_coefficient_is_named() {
    // z is non-zero only for individuals in the second half: in individual half 1 it is all
    // zero (collinear), so ss2 fails there.
    let p = panel(80, 6, 0.0, 21);
    let z: Vec<f64> = p.i.iter().zip(&p.x2).map(|(&i, &x)| if i > 41.0 { x } else { 0.0 }).collect();
    let mut input = two_way(&p);
    input.x.push(&z);
    let e = err(fit::bias_corr(&input, &spec(Flavor::Fixest), &BiasOptions { method: jk(Variant::Ss2), ..BiasOptions::default() }, &Ctx::quiet(4)));
    assert!(e.contains("jackknife, subpanel individual half 1") && e.contains("not estimable"), "{e}");
}

#[test]
fn one_way_jackknife_uses_the_time_variable() {
    let p = panel(100, 6, 0.0, 33);
    let mut input = two_way(&p);
    input.fe.truncate(1);
    let ctx = Ctx::quiet(4);
    let sp = spec(Flavor::Fixest);
    // Individual effects only: ss2 = 2b - mean(per-individual time halves); ss1/sj/jj: time dimension.
    let b = fit::bias_corr(&input, &sp, &BiasOptions { method: jk(Variant::Jj), time: Some(1), ..BiasOptions::default() }, &ctx).unwrap();
    assert_eq!(b.report.which, WhichBias::Individual);
    assert_eq!(b.report.n_fits, 6);
    let s = fit::bias_corr(&input, &sp, &BiasOptions { method: jk(Variant::Sj), time: Some(1), ..BiasOptions::default() }, &ctx).unwrap();
    assert_eq!(s.fit.coef, b.fit.coef);
}

/// Timings (run with `--ignored --nocapture`).
#[test]
#[ignore]
fn timings() {
    let p = panel(100_000, 10, 0.0, 1);
    let input = two_way(&p);
    let ctx = Ctx::quiet(16);
    let sp = fit::FitSpec { glm_tol: 1e-10, fixef_tol: 1e-10, ..spec(Flavor::Fixest) };
    let t = std::time::Instant::now();
    let f = fit::fit(&input, &sp, &ctx).unwrap();
    println!("1e6 rows two-way fit: {:.2}s", t.elapsed().as_secs_f64());
    let t = std::time::Instant::now();
    let b = fit::bias_corr(&input, &sp, &BiasOptions { bandwidth: 1, ..BiasOptions::default() }, &ctx).unwrap();
    println!("1e6 rows analytical L=1 (incl. refit): {:.2}s (resolve {} it)", t.elapsed().as_secs_f64(), b.report.resolve_iterations);
    let t = std::time::Instant::now();
    fit::apes(&input, &sp, ApeAt::Uncorrected, &ApeOptions { n_pop: Some(2e6), weak_exo: true, ..ApeOptions::default() }, &ctx).unwrap();
    println!("1e6 rows APEs (incl. refit): {:.2}s", t.elapsed().as_secs_f64());
    let o = BiasOptions { bandwidth: 1, ..BiasOptions::default() };
    let t = std::time::Instant::now();
    fit::apes(&input, &sp, ApeAt::Corrected { options: &o, coef: &b.fit.coef, ape_jackknife: None }, &ApeOptions { n_pop: Some(2e6), sampling: Sampling::Independence, ..ApeOptions::default() }, &ctx).unwrap();
    println!("1e6 rows corrected APEs (incl. refit + resolve): {:.2}s", t.elapsed().as_secs_f64());
    let t = std::time::Instant::now();
    fit::bias_corr(&input, &sp, &BiasOptions { method: jk(Variant::Ss2), ..BiasOptions::default() }, &ctx).unwrap();
    println!("1e6 rows ss2: {:.2}s", t.elapsed().as_secs_f64());
    drop(f);
    let p = panel(2_000, 10, 0.0, 2);
    let input = two_way(&p);
    let t = std::time::Instant::now();
    let b = fit::bias_corr(&input, &sp, &BiasOptions { method: jk(Variant::Jj), ..BiasOptions::default() }, &ctx).unwrap();
    println!("2e4 rows (2000 x 10) jj: {:.2}s, {} subpanel fits", t.elapsed().as_secs_f64(), b.report.n_fits);
}

/// Smoke at scale: 1e7 rows (1e6 x 10), 16 threads (run with `--ignored --nocapture`).
#[test]
#[ignore]
fn smoke_1e7() {
    let t = std::time::Instant::now();
    let p = panel(1_000_000, 10, 0.0, 7);
    println!("1e7 data: {:.1}s", t.elapsed().as_secs_f64());
    let input = two_way(&p);
    let ctx = Ctx::quiet(16);
    let sp = fit::FitSpec { glm_tol: 1e-10, fixef_tol: 1e-10, keep_fixef: false, ..spec(Flavor::Fixest) };
    let t = std::time::Instant::now();
    fit::fit(&input, &sp, &ctx).unwrap();
    println!("1e7 fit: {:.2}s", t.elapsed().as_secs_f64());
    let o = BiasOptions { bandwidth: 1, ..BiasOptions::default() };
    let t = std::time::Instant::now();
    let b = fit::bias_corr(&input, &sp, &o, &ctx).unwrap();
    println!("1e7 analytical L=1 (incl. refit): {:.2}s, resolve {} iterations", t.elapsed().as_secs_f64(), b.report.resolve_iterations);
    // FE score at the re-solved point.
    let resid = b.fit.resid.as_ref().unwrap();
    let mut si = vec![0.0f64; 1_000_001];
    let mut st = vec![0.0f64; 11];
    for (r, &i) in b.fit.obs.iter().enumerate() {
        si[p.i[i as usize] as usize] += resid[r];
        st[p.t[i as usize] as usize] += resid[r];
    }
    let score = si.iter().chain(&st).fold(0.0f64, |m, v| m.max(v.abs()));
    println!("1e7 FE score max |sum_g (y - mu)|: {score:e}");
    let t = std::time::Instant::now();
    let a = fit::apes(&input, &sp, ApeAt::Corrected { options: &o, coef: &b.fit.coef, ape_jackknife: None }, &ApeOptions { n_pop: Some(2e7), ..ApeOptions::default() }, &ctx).unwrap();
    println!("1e7 corrected APEs with n_pop and weak exogeneity (incl. refit + resolve): {:.2}s ({:?})", t.elapsed().as_secs_f64(), a.delta);
    let t = std::time::Instant::now();
    let a0 = fit::apes(&input, &sp, ApeAt::Uncorrected, &ApeOptions { n_pop: Some(2e7), weak_exo: true, ..ApeOptions::default() }, &ctx).unwrap();
    println!("1e7 uncorrected APEs (incl. refit): {:.2}s ({:?})", t.elapsed().as_secs_f64(), a0.delta);
    let t = std::time::Instant::now();
    let j = fit::bias_corr(&input, &sp, &BiasOptions { method: jk(Variant::Ss2), ..BiasOptions::default() }, &ctx).unwrap();
    println!("1e7 ss2: {:.2}s, resolve {} iterations", t.elapsed().as_secs_f64(), j.report.resolve_iterations);
}
