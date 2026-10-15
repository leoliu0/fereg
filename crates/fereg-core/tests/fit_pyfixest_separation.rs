//! pyfixest's separation checks in the fit layer ([`Flavor::Pyfixest`]): the `fe` check on the
//! gaussian `feglm` ([`FitSpec::gaussian_glm`]) and without singleton removal
//! ([`FixefRm::PerfectOnly`]), the `ir` check (`relu`) for logit and gaussian, and no check
//! without fixed effects. The Python tests compare the same cases with pyfixest itself.

use fereg_core::Ctx;
use fereg_core::api::Separation;
use fereg_core::fit::{self, FactorCol, FeTerm, FitInput, FitOutput, FitSpec, FixefRm, Flavor, Model};

/// 60 rows, f with 12 levels (5 rows each) plus one singleton level (row 60), g with 3 levels.
/// y: level f = 0 all zero, d = 1 rows zero (separated by d only), others positive.
struct Data {
    y: Vec<f64>,
    yg: Vec<f64>,
    x: Vec<f64>,
    d: Vec<f64>,
    f: Vec<f64>,
    g: Vec<f64>,
}

fn data() -> Data {
    let n = 61;
    let f: Vec<f64> = (0..n).map(|i| if i == 60 { 99.0 } else { (i / 5) as f64 }).collect();
    let g: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    let x: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64 / 10.0 - 0.4).collect();
    let d: Vec<f64> = (0..n).map(|i| if i % 9 == 4 { 1.0 } else { 0.0 }).collect();
    // zeros: level f = 0, the d = 1 rows, and every 7th row (not separated)
    let y: Vec<f64> = (0..n).map(|i| if f[i] == 0.0 || d[i] == 1.0 || i % 7 == 3 { 0.0 } else { 1.0 + ((i * 5) % 3) as f64 }).collect();
    let yg: Vec<f64> = (0..n).map(|i| if f[i] == 1.0 { -1.0 } else { 2.0 + x[i] + ((i * 3) % 5) as f64 / 4.0 }).collect();
    Data { y, yg, x, d, f, g }
}

fn run(dt: &Data, y: &[f64], with_d: bool, fe: bool, spec: FitSpec) -> FitOutput {
    let mut x = vec![dt.x.as_slice()];
    if with_d {
        x.push(&dt.d);
    }
    let input = FitInput {
        n: y.len(),
        y,
        x,
        factors: if fe { vec![FactorCol::F64(&dt.f), FactorCol::F64(&dt.g)] } else { Vec::new() },
        factor_names: if fe { vec!["f".into(), "g".into()] } else { Vec::new() },
        fe: if fe {
            vec![FeTerm { factors: vec![0], intercept: true, slopes: Vec::new() }, FeTerm { factors: vec![1], intercept: true, slopes: Vec::new() }]
        } else {
            Vec::new()
        },
        ..Default::default()
    };
    fit::fit(&input, &FitSpec { flavor: Flavor::Pyfixest, glm_tol: 1e-10, glm_maxiter: 100, ..spec }, &Ctx::quiet(2)).unwrap()
}

#[test]
fn gaussian_glm_fe_check() {
    let dt = data();
    let base = FitSpec { model: Model::Ols, fixef_rm: FixefRm::Perfect, ..Default::default() };
    // feols (gaussian_glm off): only the singleton goes
    let ols = run(&dt, &dt.yg, false, true, base.clone());
    assert_eq!((ols.dropped.singletons, ols.dropped.perfect_fit), (1, 0));
    // feglm(family="gaussian"): level f = 1 (all <= 0) goes too; df keep its level
    let glm = run(&dt, &dt.yg, false, true, FitSpec { gaussian_glm: true, ..base.clone() });
    assert_eq!((glm.dropped.singletons, glm.dropped.perfect_fit, glm.nobs), (1, 5, 55));
    assert_eq!(glm.nparams, 1 + 12 + 3 - 1);
    // the fixest flavor ignores it
    let fixest = fit::fit(
        &FitInput {
            n: 61,
            y: &dt.yg,
            x: vec![&dt.x],
            factors: vec![FactorCol::F64(&dt.f)],
            factor_names: vec!["f".into()],
            fe: vec![FeTerm { factors: vec![0], intercept: true, slopes: Vec::new() }],
            ..Default::default()
        },
        &FitSpec { gaussian_glm: true, ..base },
        &Ctx::quiet(2),
    )
    .unwrap();
    assert_eq!(fixest.dropped.perfect_fit, 0);
}

#[test]
fn perfect_only_keeps_singletons() {
    let dt = data();
    for model in [Model::Poisson, Model::Logit] {
        let y: Vec<f64> = if model == Model::Logit { dt.y.iter().map(|v| f64::from(*v > 0.0)).collect() } else { dt.y.clone() };
        let out = run(&dt, &y, false, true, FitSpec { model, fixef_rm: FixefRm::PerfectOnly, ..Default::default() });
        assert_eq!((out.dropped.singletons, out.dropped.perfect_fit, out.nobs), (0, 5, 56), "{model:?}");
    }
}

#[test]
fn ir_check_logit_gaussian_and_no_fe() {
    let dt = data();
    let relu = Separation { relu: true, mu: false };
    let yb: Vec<f64> = dt.y.iter().map(|v| f64::from(*v > 0.0)).collect();
    let pois = run(&dt, &dt.y, true, true, FitSpec { model: Model::Poisson, fixef_rm: FixefRm::Perfect, separation: relu, ..Default::default() });
    assert!(pois.dropped.separated > 0);
    for (model, gaussian_glm) in [(Model::Logit, false), (Model::Ols, true)] {
        let out = run(&dt, &yb, true, true, FitSpec { model, gaussian_glm, fixef_rm: FixefRm::Perfect, separation: relu, ..Default::default() });
        assert_eq!((out.dropped.perfect_fit, out.dropped.separated, out.nobs), (pois.dropped.perfect_fit, pois.dropped.separated, pois.nobs), "{model:?}");
        // no singleton re-check after the separation drops
        assert_eq!(out.dropped.singletons, 1);
    }
    // without fixed effects pyfixest runs no separation check
    let nofe = run(&dt, &dt.y, false, false, FitSpec { model: Model::Poisson, fixef_rm: FixefRm::Perfect, separation: relu, ..Default::default() });
    assert_eq!((nofe.dropped.separated, nofe.dropped.perfect_fit, nofe.nobs), (0, 0, 61));
}
