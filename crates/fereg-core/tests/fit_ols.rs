//! `fit` OLS against fixest 0.14.2 (tests/data/fit_fixest.json, written by gen_fit_fixtures.R)
//! and pyfixest 0.60.0 (tests/data/fit_pyfixest.json, written by gen_fit_fixtures.py).

mod fit_common;

use fereg_core::Ctx;
use fereg_core::fit::{self, CoefSource, FitOutput, Flavor};
use fit_common::*;

const RTOL: f64 = 1e-8;

fn coef_names(case: &Case, out: &FitOutput) -> Vec<String> {
    let x = case.json.at("x").strs();
    out.coef_source
        .iter()
        .map(|s| match s {
            CoefSource::Intercept => "(Intercept)".to_string(),
            CoefSource::X(j) => x[*j].clone(),
            other => panic!("unexpected source {other:?}"),
        })
        .collect()
}

fn check_fixest(j: &J, kind: FactorKind, threads: usize) -> FitOutput {
    let name = j.at("name").str();
    let case = Case::new(j, kind);
    let e = j.at("expect");
    let out = fit::fit(&case.input(), &case.spec(Flavor::Fixest), &Ctx::quiet(threads)).unwrap_or_else(|err| panic!("{name}: {err}"));
    let what = |s: &str| format!("{name} {s}");
    assert_eq!(coef_names(&case, &out), e.at("coef_names").strs(), "{name}: coefficient names");
    assert_vec(&what("coef"), &out.coef, &e.at("coef").nums(), RTOL, 1e-12);
    let se: Vec<f64> = (0..out.coef.len()).map(|a| out.vcov.vcov[a * out.coef.len() + a].sqrt()).collect();
    assert_vec(&what("se"), &se, &e.at("se").nums(), RTOL, 1e-12);
    let scale = e.at("vcov").nums().iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert_vec(&what("vcov"), &out.vcov.vcov, &e.at("vcov").nums(), RTOL, scale);
    assert_eq!(out.nobs as f64, e.at("nobs").num(), "{name}: nobs");
    assert_eq!(out.nparams as f64, e.at("nparams").num(), "{name}: nparams");
    assert_eq!(out.vcov.df_k, e.at("df_k").num(), "{name}: df_k");
    assert_eq!(out.vcov.df_t, Some(e.at("df_t").num()), "{name}: df_t");
    let g: Vec<f64> = out.vcov.n_clusters.iter().map(|&g| g as f64).collect();
    assert_eq!(g, e.at("G").nums(), "{name}: clusters");
    let x = j.at("x").strs();
    let collin: Vec<String> = out.collinear.iter().map(|s| if let CoefSource::X(i) = s { x[*i].clone() } else { panic!() }).collect();
    assert_eq!(collin, e.at("collin").strs(), "{name}: collinear");
    let obs: Vec<f64> = out.obs.iter().map(|&i| f64::from(i) + 1.0).collect();
    assert_eq!(obs, e.at("obs").nums(), "{name}: obs");
    let s = &out.stats;
    for (k, v) in [
        ("r2", s.r2),
        ("ar2", s.ar2),
        ("wr2", s.wr2),
        ("war2", s.war2),
        ("pr2", s.pseudo_r2),
        ("apr2", s.apr2),
        ("sq_cor", s.sq_cor),
        ("rmse", s.rmse),
        ("ll", s.loglik),
        ("ll_null", s.ll_null),
        ("ssr", s.ssr),
        ("ssr_null", s.ssr_null),
        ("ssr_fe_only", s.ssr_fe_only),
        ("sigma2", s.sigma2),
        ("f_stat", s.f_test.stat),
        ("f_df1", s.f_test.df1),
        ("f_df2", s.f_test.df2),
    ] {
        assert_close(&what(k), v, e.at(k).num(), RTOL);
    }
    assert_close(&what("f_p"), s.f_test.p, e.at("f_p").num(), 1e-6);
    let resid = e.at("resid").nums();
    if !resid.is_empty() {
        let scale = resid.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert_vec(&what("resid"), out.resid.as_ref().unwrap(), &resid, RTOL, scale);
        let fitted = e.at("fitted").nums();
        let scale = fitted.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert_vec(&what("fitted"), out.fitted.as_ref().unwrap(), &fitted, RTOL, scale);
    }
    out
}

#[test]
fn fixest_cases() {
    let cases = read_json("fit_fixest.json").list();
    assert!(cases.len() > 40);
    for j in &cases {
        check_fixest(j, FactorKind::F64, 4);
    }
}

#[test]
fn factor_encodings_agree() {
    let cases = read_json("fit_fixest.json").list();
    for name in ["fe2_cl_c1_c3", "inter_w", "fe3_tw_exact", "nacluster"] {
        let j = cases.iter().find(|c| c.at("name").str() == name).unwrap();
        for kind in [FactorKind::I32, FactorKind::I64, FactorKind::Codes] {
            check_fixest(j, kind, 3);
        }
    }
}

#[test]
fn bit_identical_across_threads() {
    let cases = read_json("fit_fixest.json").list();
    for name in ["fe2_cl_c1_c3_h", "fe1_w_cl", "fe3_tw"] {
        let j = cases.iter().find(|c| c.at("name").str() == name).unwrap();
        let a = check_fixest(j, FactorKind::F64, 1);
        let b = check_fixest(j, FactorKind::F64, 16);
        assert_eq!(a.coef, b.coef, "{name}");
        assert_eq!(a.vcov, b.vcov, "{name}");
        assert_eq!(a.resid, b.resid, "{name}");
        assert_eq!(a.stats, b.stats, "{name}");
    }
}

/// Plan §1 toy example: the printed six digits of fixest.
#[test]
fn toy_example() {
    let cases = read_json("fit_fixest.json").list();
    for (name, se) in [
        ("toy_iid", 0.235702),
        ("toy_hetero", 0.222222),
        ("toy_cl_g", 0.248452),
        ("toy_cl_g_full", 0.351364),
        ("toy_cl_g_none", 0.222222),
        ("toy_tw", 0.261891),
    ] {
        let j = cases.iter().find(|c| c.at("name").str() == name).unwrap();
        let out = check_fixest(j, FactorKind::F64, 1);
        assert!((out.vcov.vcov[0].sqrt() - se).abs() < 1e-6, "{name}");
        assert_eq!(out.nparams, 4);
    }
}

#[test]
fn pyfixest_cases() {
    let cases = read_json("fit_pyfixest.json").list();
    assert!(cases.len() >= 20);
    for j in &cases {
        let name = j.at("name").str();
        let case = Case::new(j, FactorKind::F64);
        let res = fit::fit(&case.input(), &case.spec(Flavor::Pyfixest), &Ctx::quiet(4));
        if let Some(err) = j.get("error") {
            assert_eq!(err.str(), "NanInClusterVarError");
            let msg = res.err().unwrap_or_else(|| panic!("{name}: expected an error")).to_string();
            assert!(msg.contains("cluster variable cna"), "{name}: {msg}");
            continue;
        }
        let out = res.unwrap_or_else(|err| panic!("{name}: {err}"));
        let e = j.at("expect");
        let what = |s: &str| format!("{name} {s}");
        assert_eq!(coef_names(&case, &out), e.at("coef_names").strs(), "{name}: coefficient names");
        assert_vec(&what("coef"), &out.coef, &e.at("coef").nums(), RTOL, 1e-12);
        let k = out.coef.len();
        let se: Vec<f64> = (0..k).map(|a| out.vcov.vcov[a * k + a].sqrt()).collect();
        assert_vec(&what("se"), &se, &e.at("se").nums(), RTOL, 1e-12);
        let scale = e.at("vcov").nums().iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert_vec(&what("vcov"), &out.vcov.vcov, &e.at("vcov").nums(), RTOL, scale);
        assert_eq!(out.nobs as f64, e.at("nobs").num(), "{name}: nobs");
        assert_eq!(out.vcov.df_k, e.at("df_k").num(), "{name}: df_k");
        assert_eq!(out.vcov.df_t, Some(e.at("df_t").num()), "{name}: df_t");
        let x = j.at("x").strs();
        let collin: Vec<String> = out.collinear.iter().map(|s| if let CoefSource::X(i) = s { x[*i].clone() } else { panic!() }).collect();
        assert_eq!(collin, e.at("collin").strs(), "{name}: collinear");
        let s = &out.stats;
        for (key, v) in [("r2", s.r2), ("ar2", s.ar2), ("wr2", s.wr2), ("war2", s.war2), ("rmse", s.rmse)] {
            assert_close(&what(key), v, e.at(key).num(), RTOL);
        }
        let resid = e.at("resid").nums();
        if !resid.is_empty() && j.get("weights").is_some_and(|w| matches!(w, J::Null)) {
            let scale = resid.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert_vec(&what("resid"), out.resid.as_ref().unwrap(), &resid, RTOL, scale);
        }
    }
}

/// Row accounting: missing values per group, zero weights, the extra drop mask and singletons.
#[test]
fn dropped_rows() {
    let csv = Csv::read("fit_d1.csv");
    let n = csv.n();
    let mut w = csv.col("w").to_vec();
    w[30] = 0.0;
    w[31] = 0.0;
    w[10] = 0.0; // y is missing too (R row 11): counts as missing only
    let mut drop = vec![false; n];
    drop[40] = true;
    drop[22] = true; // x1 is missing too (R row 23)
    let g = FactorData::new(csv.col("g"), FactorKind::F64);
    let h = FactorData::new(csv.col("h"), FactorKind::I32);
    let cna = FactorData::new(csv.col("cna"), FactorKind::Codes);
    let input = fit::FitInput {
        n,
        y: csv.col("y"),
        x: vec![csv.col("x1"), csv.col("x2")],
        factors: vec![g.col(), h.col(), cna.col()],
        factor_names: vec!["g".into(), "h".into(), "cna".into()],
        fe: vec![fit::FeTerm { factors: vec![0], intercept: true, slopes: vec![] }, fit::FeTerm { factors: vec![1], intercept: true, slopes: vec![] }],
        weights: Some(&w),
        cluster: vec![vec![2]],
        drop: Some(&drop),
        ..Default::default()
    };
    let spec = fit::FitSpec { vcov: fit::VcovKind::Cluster, ..Default::default() };
    let out = fit::fit(&input, &spec, &Ctx::quiet(2)).unwrap();
    let missing = |c: &str| csv.col(c).iter().filter(|v| v.is_nan()).count();
    let d = &out.dropped;
    assert_eq!((d.na_by.y, d.na_by.x, d.na_by.fe, d.na_by.cluster, d.na_by.extra, d.na_by.weights), (3, 2, 2, missing("cna"), 2, 0));
    let na_rows = (0..n)
        .filter(|&i| drop[i] || ["y", "x1", "x2", "g", "h", "cna"].iter().any(|c| csv.col(c)[i].is_nan()))
        .count();
    assert_eq!(d.na, na_rows);
    assert_eq!(d.zero_weight, 2);
    assert!(d.singletons >= 4, "singleton levels 41..44 of g (row 4 has a missing cluster)");
    assert_eq!(d.perfect_fit + d.separated, 0);
    assert_eq!(out.nobs, n - d.na - d.zero_weight - d.singletons);
    assert!(out.obs.windows(2).all(|p| p[0] < p[1]));
    assert!(!out.obs.contains(&30) && !out.obs.contains(&40) && !out.obs.contains(&0));

    // Errors are plain messages, never panics.
    let mut neg = csv.col("w").to_vec();
    neg[100] = -1.0;
    let bad = fit::FitInput { weights: Some(&neg), ..input.clone() };
    assert!(fit::fit(&bad, &spec, &Ctx::quiet(1)).err().unwrap().to_string().contains("negative"));
    let short = [1.0, 2.0];
    let bad = fit::FitInput { x: vec![&short], ..input.clone() };
    assert!(fit::fit(&bad, &spec, &Ctx::quiet(1)).is_err());
    let xg = csv.col("xg");
    let bad = fit::FitInput { x: vec![xg], ..input.clone() };
    assert!(fit::fit(&bad, &spec, &Ctx::quiet(1)).err().unwrap().to_string().contains("collinear"));
}
