//! `fit` IV/2SLS against fixest 0.14.2 (tests/data/fit_fixest_iv.json) and pyfixest 0.60.0
//! (the IV cases of tests/data/fit_pyfixest_b.json), both written by gen_fit_fixtures.*.

mod fit_common;

use fereg_core::Ctx;
use fereg_core::fit::{self, FitOutput, Flavor, TestStat};
use fit_common::*;

const RTOL: f64 = 1e-8;

fn check_test(what: &str, t: &TestStat, e: &J) {
    for (k, v) in [("stat", t.stat), ("df1", t.df1), ("df2", t.df2)] {
        assert_close(&format!("{what} {k}"), v, e.at(k).num(), RTOL);
    }
    assert_close(&format!("{what} p"), t.p, e.at("p").num(), 1e-6);
}

fn check_fixest_iv(j: &J, threads: usize) -> FitOutput {
    let name = j.at("name").str().to_string();
    let (case, out) = check_case(j, Flavor::Fixest, FactorKind::F64, threads, RTOL);
    let e = j.at("expect").at("iv");
    let iv = out.iv.as_ref().unwrap_or_else(|| panic!("{name}: IV statistics"));
    assert_close(&format!("{name} ssr_iv"), iv.ssr_iv, e.at("ssr_iv").num(), RTOL);
    check_test(&format!("{name} wh"), &iv.wu_hausman, e.at("wh"));
    check_test(&format!("{name} sargan"), &iv.sargan, e.at("sargan"));
    let firsts = e.at("first").list();
    assert_eq!(iv.first_stage.len(), firsts.len());
    for (f, ef) in iv.first_stage.iter().zip(&firsts) {
        let what = format!("{name} first stage {}", f.endog);
        let names: Vec<String> = f.coef_source.iter().map(|s| case.source_name(s, Flavor::Fixest)).collect();
        let want = ef.at("coef_names").strs();
        assert_eq!(names, want, "{what}: names");
        check_coef(&what, &f.coef, &names, &ef.at("coef").nums(), &want, RTOL);
        check_vcov(&what, &f.vcov, &names, ef, &want, RTOL);
        let collin: Vec<String> = f.collinear.iter().map(|s| case.source_name(s, Flavor::Fixest)).collect();
        assert_eq!(collin, ef.at("collin").strs(), "{what}: collinear");
        check_test(&format!("{what} f"), &f.f_test, ef.at("f"));
        check_test(&format!("{what} wald"), &f.wald, ef.at("wald"));
    }
    out
}

#[test]
fn fixest_iv_cases() {
    let cases = read_json("fit_fixest_iv.json").list();
    assert!(cases.len() >= 15);
    for j in &cases {
        check_fixest_iv(j, 4);
    }
}

#[test]
fn pyfixest_iv_cases() {
    let cases = read_json("fit_pyfixest_b.json").list();
    let mut n = 0;
    for j in cases.iter().filter(|c| !c.at("endog").list().is_empty()) {
        let name = j.at("name").str().to_string();
        // pyfixest reports weighted residuals scaled by √w: only unweighted residuals are compared.
        let (case, out) = check_case(j, Flavor::Pyfixest, FactorKind::F64, 4, RTOL);
        let e = j.at("expect").at("first");
        let f = &out.iv.as_ref().unwrap().first_stage[0];
        let names: Vec<String> = f.coef_source.iter().map(|s| case.source_name(s, Flavor::Pyfixest)).collect();
        let want = e.at("coef_names").strs();
        check_coef(&name, &f.coef, &names, &e.at("coef").nums(), &want, RTOL);
        check_vcov(&format!("{name} first stage"), &f.vcov, &names, e, &want, RTOL);
        assert_close(&format!("{name} first-stage F"), f.wald.stat, e.at("f_stat").num(), RTOL);
        assert_close(&format!("{name} first-stage p"), f.wald.p, e.at("f_p").num(), 1e-6);
        n += 1;
    }
    assert!(n >= 7);
}

#[test]
fn iv_bit_identical_across_threads() {
    let cases = read_json("fit_fixest_iv.json").list();
    for name in ["iv_fe2_cl_c3_h", "iv_fe1_two_endog_cl", "iv_nofe_hetero_w"] {
        let j = cases.iter().find(|c| c.at("name").str() == name).unwrap();
        let a = check_fixest_iv(j, 1);
        let b = check_fixest_iv(j, 16);
        assert_eq!(a.coef, b.coef, "{name}");
        assert_eq!(a.vcov, b.vcov, "{name}");
        // Debug strings: NaN fields (Sargan df2, within R² without FE) compare equal there.
        assert_eq!(format!("{:?}", a.iv), format!("{:?}", b.iv), "{name}");
        assert_eq!(a.resid, b.resid, "{name}");
        assert_eq!(format!("{:?}", a.stats), format!("{:?}", b.stats), "{name}");
    }
}

#[test]
fn iv_input_errors() {
    let cases = read_json("fit_fixest_iv.json").list();
    let j = cases.iter().find(|c| c.at("name").str() == "iv_fe1_two_endog").unwrap();
    let case = Case::new(j, FactorKind::F64);
    let spec = case.spec(Flavor::Fixest);
    let mut input = case.input();
    input.inst.truncate(1);
    let err = fit::fit(&input, &spec, &Ctx::quiet(2)).err().expect("under-identified").to_string();
    assert!(err.contains("at least as many instruments"), "{err}");
    let mut input = case.input();
    let xg = case.csv.col("xg");
    input.inst = vec![xg, xg, xg];
    let err = fit::fit(&input, &spec, &Ctx::quiet(2)).err().expect("irrelevant instruments").to_string();
    assert!(err.contains("no explanatory power"), "{err}");
}
