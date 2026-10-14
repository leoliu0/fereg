//! `fit` Poisson and logit against fixest 0.14.2 (tests/data/fit_fixest_glm.json, written by
//! gen_fit_fixtures.R with glm.tol = 1e-12) and pyfixest 0.60.0 (tests/data/fit_pyfixest_glm.json).

mod fit_common;

use fereg_core::Ctx;
use fereg_core::fit::{self, Flavor, Model};
use fit_common::*;

/// The GLM core follows fixest's IRLS path (starting values, step halving, stopping rule), so
/// every fixest field, the variance matrices included, matches at this tolerance.
const RTOL: f64 = 1e-8;
/// pyfixest's IRLS takes another path (mu_0 = (y + ybar)/2, no step halving); its variance matrices
/// come from the IRLS weights at the start of its last iteration and so carry that iteration's
/// distance to the optimum (~=1e-6 relative here). Coefficients and deviances use [`RTOL`].
const RTOL_PY_V: f64 = 5e-6;

#[test]
fn fixest_glm_cases() {
    let cases = read_json("fit_fixest_glm.json").list();
    assert!(cases.len() >= 18);
    for j in &cases {
        let (case, out) = check_case(j, Flavor::Fixest, FactorKind::F64, 4, RTOL);
        let e = j.at("expect");
        let names: Vec<String> = out.coef_source.iter().map(|s| case.source_name(s, Flavor::Fixest)).collect();
        check_coef(j.at("name").str(), &out.coef, &names, &e.at("coef").nums(), &e.at("coef_names").strs(), RTOL);
        for (k, v) in [("deviance", out.stats.deviance), ("ll", out.stats.loglik), ("ll_null", out.stats.ll_null)] {
            assert_close(&format!("{} {k}", j.at("name").str()), v, e.at(k).num(), RTOL);
        }
        let name = j.at("name").str();
        assert!(out.stats.converged, "{name}: converged");
        // yp is all zero in g 38-39 and yb constant in g 36-37: perfect-fit rows with a g term.
        if name.ends_with("fe2_iid") {
            assert!(out.dropped.perfect_fit > 0, "{name}: perfect-fit rows dropped");
        }
    }
}

#[test]
fn glm_bit_identical_across_threads() {
    let mut cases = read_json("fit_fixest_glm.json").list();
    cases.extend(read_json("fit_pyfixest_glm.json").list());
    for name in ["pois_fe2_cl_c3_h", "logit_fe2_w", "pois_nofe_off_w_cl", "pois_sl", "py_logit_fe2_cl_g"] {
        let j = cases.iter().find(|c| c.at("name").str() == name).unwrap();
        let case = Case::new(j, FactorKind::F64);
        let flavor = if name.starts_with("py_") { Flavor::Pyfixest } else { Flavor::Fixest };
        let run = |t| fit::fit(&case.input(), &fit::FitSpec { keep_fixef: true, ..case.spec(flavor) }, &Ctx::quiet(t)).unwrap();
        let (a, b) = (run(1), run(16));
        assert_eq!(a.coef, b.coef, "{name}");
        assert_eq!(a.vcov, b.vcov, "{name}");
        assert_eq!(a.eta, b.eta, "{name}");
        assert_eq!(a.fixef, b.fixef, "{name}");
        // Debug strings: NaN fields (r2 family) compare equal there.
        assert_eq!(format!("{:?}", a.stats), format!("{:?}", b.stats), "{name}");
    }
}

#[test]
fn glm_input_errors() {
    let cases = read_json("fit_fixest_glm.json").list();
    let j = cases.iter().find(|c| c.at("name").str() == "pois_fe2_iid").unwrap();
    let case = Case::new(j, FactorKind::F64);
    let input = case.input();
    let spec = case.spec(Flavor::Fixest);
    let err = |input: &fit::FitInput, spec: &fit::FitSpec| fit::fit(input, spec, &Ctx::quiet(2)).err().expect("an error").to_string();

    let neg: Vec<f64> = input.y.iter().map(|v| if *v == 0.0 { -1.0 } else { *v }).collect();
    assert!(err(&fit::FitInput { y: &neg, ..input.clone() }, &spec).contains("non-negative"));
    let frac: Vec<f64> = input.y.iter().map(|v| v / 10.0).collect();
    let logit = fit::FitSpec { model: Model::Logit, ..spec.clone() };
    assert!(err(&fit::FitInput { y: &frac, ..input.clone() }, &logit).contains("0 or 1"));
    let nofe = fit::FitInput { fe: Vec::new(), ..input.clone() };
    assert!(err(&nofe, &fit::FitSpec { intercept: false, ..spec.clone() }).contains("need an intercept"));
    let x1 = input.x[0];
    assert!(err(&fit::FitInput { endog: vec![x1], inst: vec![x1], ..input.clone() }, &spec).contains("linear models"));
}

/// Lean fits keep no state and no per-observation outputs; the estimates do not change.
#[test]
fn glm_lean() {
    let cases = read_json("fit_fixest_glm.json").list();
    let j = cases.iter().find(|c| c.at("name").str() == "pois_fe2_off_w").unwrap();
    let case = Case::new(j, FactorKind::F64);
    let full = fit::fit(&case.input(), &case.spec(Flavor::Fixest), &Ctx::quiet(4)).unwrap();
    let spec = fit::FitSpec { keep_state: false, keep_resid: false, keep_fixef: false, ..case.spec(Flavor::Fixest) };
    let lean = fit::fit(&case.input(), &spec, &Ctx::quiet(4)).unwrap();
    assert!(lean.state.is_none() && lean.eta.is_none() && lean.irls_weights.is_none() && lean.fixef.is_empty());
    assert_eq!(full.coef, lean.coef);
    assert_eq!(full.vcov, lean.vcov);
}

/// Pyfixest flavor against pyfixest 0.60.0 (tests/data/fit_pyfixest_glm.json, iwls_tol = 1e-12):
/// its single-pass `fe` separation (logit keeps all-one levels) and degrees of freedom.
#[test]
fn pyfixest_glm_cases() {
    let cases = read_json("fit_pyfixest_glm.json").list();
    assert!(cases.len() >= 9);
    for j in &cases {
        let (case, out) = check_case(j, Flavor::Pyfixest, FactorKind::F64, 4, RTOL_PY_V);
        let e = j.at("expect");
        let name = j.at("name").str();
        let names: Vec<String> = out.coef_source.iter().map(|s| case.source_name(s, Flavor::Pyfixest)).collect();
        check_coef(name, &out.coef, &names, &e.at("coef").nums(), &e.at("coef_names").strs(), RTOL);
        assert_close(&format!("{name} deviance"), out.stats.deviance, e.at("deviance").num(), RTOL);
    }
}

/// Pyfixest-flavor logit: the rows of the levels whose outcomes are all 1 (pyfixest keeps them)
/// are left out of IRLS and reported at the bound, with fixed effects +/-inf on their levels; the
/// reported sample is pyfixest's (checked against the fixtures above).
#[test]
fn pyfixest_logit_left_out_rows() {
    let cases = read_json("fit_pyfixest_glm.json").list();
    for name in ["py_logit_fe2_iid", "py_logit_fe1_hetero"] {
        let j = cases.iter().find(|c| c.at("name").str() == name).unwrap();
        let case = Case::new(j, FactorKind::F64);
        let input = case.input();
        let spec = fit::FitSpec { keep_fixef: true, ..case.spec(Flavor::Pyfixest) };
        let out = fit::fit(&input, &spec, &Ctx::quiet(4)).unwrap();
        let eta = out.eta.as_ref().unwrap();
        let bound = eta.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let left_out: Vec<usize> = (0..out.nobs).filter(|&k| eta[k].abs() == bound).collect();
        assert!(bound > 30.0 && !left_out.is_empty(), "{name}: rows at the bound");
        for &k in &left_out {
            let i = out.obs[k] as usize;
            assert_eq!(input.y[i], 1.0, "{name}: only all-one levels are left out");
            assert_eq!(eta[k], bound, "{name}: eta at the upper bound");
            assert_eq!(out.irls_weights.as_ref().unwrap()[k], (1.0 - out.fitted.as_ref().unwrap()[k]) * out.fitted.as_ref().unwrap()[k]);
        }
        // Their level of the first term diverges; the other rows' fixed effects are finite.
        let g = &out.fixef[0];
        let infinite = g.values.iter().filter(|v| **v == f64::INFINITY).count();
        assert!(infinite > 0 && g.values.iter().all(|v| v.is_finite() || *v == f64::INFINITY), "{name}: fixef {:?}", g.values);
        let state = out.state.as_ref().unwrap();
        assert_eq!(state.fixef(spec.fixef_norm, &Ctx::quiet(2)).unwrap(), out.fixef);
    }
}
