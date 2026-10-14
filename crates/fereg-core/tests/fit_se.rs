//! HC2, HC3, Newey-West, Driscoll-Kraay, Conley and CRV3 in the fit layer, at fit time and
//! post hoc (`FitState::vcov`), against fixest 0.14.2 (tests/data/fit_se_fixest.json) and
//! pyfixest 0.60.0 (tests/data/fit_se_pyfixest.json, fit_se_pyconley.json), written by
//! gen_se_fixtures.R and gen_se_fixtures.py on the unbalanced panel with gaps tests/data/fit_se.csv.

mod fit_common;

use fereg_core::Ctx;
use fereg_core::fit::{self, FactorCol, FeTerm, FitInput, FitOutput, FitSpec, FixefRm, Flavor, VcovKind, VcovOutput};
use fit_common::*;

const RTOL: f64 = 1e-8;
/// pyfixest's GLM variance matrices use the IRLS weights at the start of its last iteration and so
/// carry that iteration's distance to the optimum (as in fit_glm.rs); coefficients use [`RTOL`].
const RTOL_PY_GLM_V: f64 = 5e-6;

fn find<'a>(cases: &'a [J], name: &str) -> &'a J {
    cases.iter().find(|c| c.at("name").str() == name).unwrap_or_else(|| panic!("no case {name}"))
}

/// The request of a case without its estimation-time variance (the post-hoc fits use iid).
fn iid_spec(case: &Case, flavor: Flavor) -> FitSpec {
    FitSpec { vcov: VcovKind::Iid, ..case.spec(flavor) }
}

/// Runs a case at fit time (every reported field) and, unless it is CRV3 or drops rows for a
/// missing variance variable, post hoc from an iid fit. Returns the fit and the post-hoc variance.
fn run(j: &J, flavor: Flavor, threads: usize) -> Option<(FitOutput, Option<VcovOutput>)> {
    let name = j.at("name").str().to_string();
    let case = Case::new(j, FactorKind::F64);
    if let Some(e) = j.get("error") {
        let err = fit::fit(&case.input(), &case.spec(flavor), &Ctx::quiet(threads)).err().unwrap_or_else(|| panic!("{name}: expected an error"));
        // fixest prefixes the call, pyfixest the exception class.
        let msg = e.str().replace("\\n", "\n");
        let want = msg.rsplit_once(": \n").map_or_else(|| msg.split_once(": ").map_or(msg.as_str(), |x| x.1), |x| x.1).trim();
        assert_eq!(err.to_string(), want, "{name}: error");
        return None;
    }
    let glm = j.get("model").is_some_and(|m| m.str() != "ols");
    let rtol = if glm && name.starts_with("py_") { RTOL_PY_GLM_V } else { RTOL };
    let (case, out) = check_case(j, flavor, FactorKind::F64, threads, rtol);
    let e = j.at("expect");
    check_coef(&name, &out.coef, &out.coef_source.iter().map(|s| case.source_name(s, flavor)).collect::<Vec<_>>(), &e.at("coef").nums(), &e.at("coef_names").strs(), RTOL);
    if let Some(first) = e.get("first") {
        let iv = out.iv.as_ref().unwrap_or_else(|| panic!("{name}: IV statistics"));
        for (f, ef) in iv.first_stage.iter().zip(first.list()) {
            let what = format!("{name} first stage {}", f.endog);
            let names: Vec<String> = f.coef_source.iter().map(|s| case.source_name(s, flavor)).collect();
            let want = ef.at("coef_names").strs();
            check_coef(&what, &f.coef, &names, &ef.at("coef").nums(), &want, RTOL);
            check_vcov(&what, &f.vcov, &names, &ef, &want, RTOL);
        }
    }
    let kind = case.spec(flavor).vcov;
    let dropped = out.dropped.na_by.cluster > 0;
    let posthoc = if kind == VcovKind::Crv3 || dropped {
        None
    } else {
        let base = fit::fit(&case.input(), &iid_spec(&case, flavor), &Ctx::quiet(threads)).unwrap();
        let input = case.input();
        let data = fit::VcovData {
            factors: &input.factors,
            factor_names: &case.factor_names,
            cluster: &input.cluster,
            time: input.time,
            panel: input.panel,
            lat: input.lat,
            lon: input.lon,
        };
        let v = base.state.as_ref().unwrap().vcov(kind, &case.spec(flavor).ssc, &data, &Ctx::quiet(threads)).unwrap_or_else(|err| panic!("{name} post-hoc: {err}"));
        let names: Vec<String> = base.coef_source.iter().map(|s| case.source_name(s, flavor)).collect();
        check_vcov(&format!("{name} post-hoc"), &v, &names, e, &e.at("coef_names").strs(), rtol);
        Some(v)
    };
    Some((out, posthoc))
}

#[test]
fn fixest_cases() {
    let cases = read_json("fit_se_fixest.json").list();
    assert!(cases.len() >= 80);
    run_all(&cases, Flavor::Fixest);
}

/// Runs every case, then reports all the failing ones.
fn run_all(cases: &[J], flavor: Flavor) {
    let failed: Vec<&str> = cases
        .iter()
        .filter(|j| std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(j, flavor, 4))).is_err())
        .map(|j| j.at("name").str())
        .collect();
    assert!(failed.is_empty(), "failed cases: {failed:?}");
}

#[test]
fn pyfixest_cases() {
    let mut cases = read_json("fit_se_pyfixest.json").list();
    cases.extend(read_json("fit_se_pyconley.json").list());
    assert!(cases.len() >= 70);
    run_all(&cases, Flavor::Pyfixest);
}

/// Bit-identical variance matrices (fit time and post hoc) at 1, 4 and 16 threads for every kind.
#[test]
fn bit_identical_across_threads() {
    let fx = read_json("fit_se_fixest.json").list();
    let py = read_json("fit_se_pyfixest.json").list();
    let picks = [
        (&fx, Flavor::Fixest, "hc2_sl"),
        (&fx, Flavor::Fixest, "hc3_w_fe2"),
        (&fx, Flavor::Fixest, "nw_fe2"),
        (&fx, Flavor::Fixest, "nwts_fe2"),
        (&fx, Flavor::Fixest, "dk_fe3"),
        (&fx, Flavor::Fixest, "conley_px_fe2"),
        (&fx, Flavor::Fixest, "crv3_c3_h_fe2"),
        (&fx, Flavor::Fixest, "iv_nw_fe2"),
        (&fx, Flavor::Fixest, "pois_crv3_fe2"),
        (&fx, Flavor::Fixest, "logi_hc2_fe2"),
        (&py, Flavor::Pyfixest, "py_nw_fe2"),
        (&py, Flavor::Pyfixest, "py_dk_fe2"),
        (&py, Flavor::Pyfixest, "py_hc3_fw_fe2"),
        (&py, Flavor::Pyfixest, "py_crv3_tw_fe2"),
    ];
    for (cases, flavor, name) in picks {
        let j = find(cases, name);
        let (a, pa) = run(j, flavor, 1).unwrap();
        for threads in [4, 16] {
            let (b, pb) = run(j, flavor, threads).unwrap();
            assert_eq!(a.vcov, b.vcov, "{name}: {threads} threads");
            assert_eq!(pa, pb, "{name}: post-hoc, {threads} threads");
            if let (Some(x), Some(y)) = (&a.iv, &b.iv) {
                for (f, g) in x.first_stage.iter().zip(&y.first_stage) {
                    assert_eq!(f.vcov, g.vcov, "{name}: first stage, {threads} threads");
                }
            }
        }
    }
}

fn err_of(input: &FitInput, spec: &FitSpec) -> String {
    fit::fit(input, spec, &Ctx::quiet(2)).err().expect("an error").to_string()
}

/// Missing values of variance variables: fixest drops the rows (counted with the `vcov` NA group),
/// pyfixest errors; post hoc both error, fixest with its message.
#[test]
fn missing_variance_variables() {
    let fx = read_json("fit_se_fixest.json").list();
    let j = find(&fx, "nw_tna_fe2");
    let case = Case::new(j, FactorKind::F64);
    let input = case.input();
    let out = fit::fit(&input, &case.spec(Flavor::Fixest), &Ctx::quiet(2)).unwrap();
    assert_eq!(out.dropped.na, 4);
    assert_eq!(out.dropped.na_by.cluster, 4);
    let err = err_of(&input, &case.spec(Flavor::Pyfixest));
    assert!(err.contains("missing values in the time variable tna"), "{err}");
    let base = fit::fit(&input, &iid_spec(&case, Flavor::Fixest), &Ctx::quiet(2)).unwrap();
    assert_eq!(base.dropped.na, 0, "an unused variance variable does not drop rows");
    let data = fit::VcovData { factors: &input.factors, factor_names: &case.factor_names, time: input.time, panel: input.panel, ..Default::default() };
    let err = base.state.as_ref().unwrap().vcov(case.spec(Flavor::Fixest).vcov, &Default::default(), &data, &Ctx::quiet(2)).err().unwrap().to_string();
    assert!(err.starts_with("The variable 'tna' used to estimate the VCOV (employed as time) has NA values"), "{err}");

    let j = find(&fx, "conley_latna_fe1");
    let case = Case::new(j, FactorKind::F64);
    let input = case.input();
    let out = fit::fit(&input, &case.spec(Flavor::Fixest), &Ctx::quiet(2)).unwrap();
    assert_eq!(out.dropped.na_by.cluster, 3);
    let err = err_of(&input, &case.spec(Flavor::Pyfixest));
    assert!(err.contains("missing values in the latitude"), "{err}");
}

/// Requests that cannot be computed, with the references' messages.
#[test]
fn invalid_requests() {
    let fx = read_json("fit_se_fixest.json").list();
    let case = Case::new(find(&fx, "nw_fe2"), FactorKind::F64);
    let input = case.input();
    let fixest = case.spec(Flavor::Fixest);
    let py = case.spec(Flavor::Pyfixest);
    let nw = |lag| FitSpec { vcov: VcovKind::NeweyWest { lag }, ..fixest.clone() };
    assert!(err_of(&FitInput { time: None, ..input.clone() }, &nw(None)).contains("need a time variable"));
    let conley = FitSpec { vcov: VcovKind::Conley { cutoff: None, pixel: 0.0, spherical: false }, ..fixest.clone() };
    assert!(err_of(&input, &conley).contains("latitude and longitude"));
    assert!(err_of(&input, &FitSpec { vcov: VcovKind::Crv3, ..fixest.clone() }).contains("cluster dimensions"));
    // pyfixest's time-series Newey-West has no default lag.
    let ts = FitInput { panel: None, ..input.clone() };
    assert_eq!(
        err_of(&ts, &FitSpec { vcov: VcovKind::NeweyWest { lag: None }, ..py.clone() }),
        "We have not yet implemented the default Newey-West HAC lag. Please provide a lag value via the `vcov_kwargs`."
    );
    // Duplicates: (unit, time) with units g, and time duplicates without a panel.
    let mut names = case.factor_names.clone();
    names.push("g".into());
    let gcol = FactorData::new(case.csv.col("g"), FactorKind::F64);
    let mut factors = input.factors.clone();
    factors.push(gcol.col());
    let dup = FitInput { factors: factors.clone(), factor_names: names.clone(), panel: Some(factors.len() - 1), ..input.clone() };
    assert!(err_of(&dup, &nw(Some(1))).contains("there are (unit x time) duplicates"));
    assert_eq!(err_of(&dup, &FitSpec { vcov: VcovKind::NeweyWest { lag: Some(1) }, ..py.clone() }), "There are duplicate time periods for the same panel id. This is not supported for HAC SEs.");
    assert_eq!(err_of(&dup, &FitSpec { vcov: VcovKind::DriscollKraay { lag: Some(1) }, ..py.clone() }), "There are duplicate time periods for the same panel id. This is not supported for HAC SEs.");
    assert!(fit::fit(&dup, &FitSpec { vcov: VcovKind::DriscollKraay { lag: Some(1) }, ..fixest.clone() }, &Ctx::quiet(2)).is_ok(), "fixest's DK ignores the panel");
    assert!(err_of(&ts, &nw(Some(1))).contains("there are time duplicates"));
    assert_eq!(err_of(&ts, &FitSpec { vcov: VcovKind::NeweyWest { lag: Some(1) }, ..py.clone() }), "There are duplicate time periods in the data. This is not supported for HAC SEs.");
    // Post-hoc CRV3 re-estimates.
    let out = fit::fit(&input, &fixest, &Ctx::quiet(2)).unwrap();
    let cl = [vec![0]];
    let data = fit::VcovData { factors: &input.factors, factor_names: &case.factor_names, cluster: &cl, ..Default::default() };
    let err = out.state.as_ref().unwrap().vcov(VcovKind::Crv3, &Default::default(), &data, &Ctx::quiet(2)).err().unwrap().to_string();
    assert!(err.contains("re-estimates the model"), "{err}");
}

/// fixest's error when a leverage is 1 (a fixed-effect level with one observation, kept with
/// `fixef.rm = "none"`), with its first-row quirk and row enumeration.
#[test]
fn hc2_leverage_one() {
    for k in [2usize, 9] {
        let n = 9 + k;
        let g: Vec<f64> = (0..n).map(|i| if i < 9 { (i / 3) as f64 } else { (i - 6) as f64 }).collect();
        let x: Vec<f64> = (0..n).map(|i| ((i * 7) % 5) as f64 + 0.1 * i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| ((i * 3) % 4) as f64 - 0.2 * i as f64).collect();
        let input = FitInput {
            n,
            y: &y,
            x: vec![&x],
            factors: vec![FactorCol::F64(&g)],
            factor_names: vec!["g".into()],
            fe: vec![FeTerm { factors: vec![0], intercept: true, slopes: Vec::new() }],
            ..Default::default()
        };
        let spec = FitSpec { vcov: VcovKind::Hc2, fixef_rm: FixefRm::None, ..Default::default() };
        let err = err_of(&input, &spec);
        let rows = if k == 2 { "10 and 11".to_string() } else { "10, 11, 12, 13, 14, 15 and 3 others".to_string() };
        assert_eq!(
            err,
            format!(
                "When calculating the diagonals of the projection matrix, 10 values were found to equal 1. This is most likely due to dummy variables/fixed-effects with only 1 observation, so that removing that observation would make that coefficient non-estimable. \nThe problematic rows are: {rows}."
            )
        );
    }
}

/// Timings of the heavy kinds at 1e6 rows (1e5 units x 10 periods, 1e4 firms), 16 threads:
/// `cargo test -p fereg-core --release --test fit_se timing_1m -- --ignored --nocapture`.
#[test]
#[ignore]
fn timing_1m() {
    let (units, periods) = (100_000usize, 10usize);
    let n = units * periods;
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut unif = move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let id: Vec<f64> = (0..n).map(|i| (i / periods) as f64).collect();
    let t: Vec<f64> = (0..n).map(|i| (i % periods) as f64).collect();
    let firm: Vec<f64> = (0..n).map(|_| (unif() * 10_000.0).floor()).collect();
    let cl: Vec<f64> = id.iter().map(|&u| (u as usize % 50) as f64).collect();
    let ulat: Vec<f64> = (0..units).map(|_| 30.0 + 15.0 * unif()).collect();
    let ulon: Vec<f64> = (0..units).map(|_| -100.0 + 25.0 * unif()).collect();
    let lat: Vec<f64> = (0..n).map(|i| ulat[i / periods]).collect();
    let lon: Vec<f64> = (0..n).map(|i| ulon[i / periods]).collect();
    let x1: Vec<f64> = (0..n).map(|_| unif() - 0.5).collect();
    let x2: Vec<f64> = (0..n).map(|_| unif() - 0.5).collect();
    let y: Vec<f64> = (0..n).map(|i| x1[i] - 0.5 * x2[i] + (id[i] % 7.0) * 0.1 + (firm[i] % 5.0) * 0.2 + unif()).collect();
    let yp: Vec<f64> = y.iter().map(|v| (v * 2.0).max(0.0).floor()).collect();
    let input = FitInput {
        n,
        y: &y,
        x: vec![&x1, &x2],
        factors: vec![FactorCol::F64(&id), FactorCol::F64(&firm), FactorCol::F64(&t), FactorCol::F64(&cl)],
        factor_names: vec!["id".into(), "firm".into(), "t".into(), "cl".into()],
        fe: vec![
            FeTerm { factors: vec![0], intercept: true, slopes: Vec::new() },
            FeTerm { factors: vec![1], intercept: true, slopes: Vec::new() },
        ],
        cluster: vec![vec![3]],
        time: Some(2),
        panel: Some(0),
        lat: Some(&lat),
        lon: Some(&lon),
        ..Default::default()
    };
    let ctx = Ctx::quiet(16);
    let kinds = [
        ("iid", VcovKind::Iid),
        ("hc2", VcovKind::Hc2),
        ("nw", VcovKind::NeweyWest { lag: None }),
        ("dk", VcovKind::DriscollKraay { lag: None }),
        ("conley 50km", VcovKind::Conley { cutoff: Some(50.0), pixel: 0.0, spherical: false }),
        ("crv3 G=50", VcovKind::Crv3),
    ];
    for model in [fit::Model::Ols, fit::Model::Poisson] {
        let yv: &[f64] = if model == fit::Model::Ols { &y } else { &yp };
        let input = FitInput { y: yv, ..input.clone() };
        for (name, kind) in kinds {
            let spec = FitSpec { model, vcov: kind, keep_fixef: false, ..Default::default() };
            let t0 = std::time::Instant::now();
            let out = fit::fit(&input, &spec, &ctx).unwrap();
            println!("{model:?} {name}: fit {:.3} s (se x1 {:.3e})", t0.elapsed().as_secs_f64(), out.vcov.vcov[0].sqrt());
        }
    }
}

/// Max relative differences of the standard errors and variance matrices (fit time) per flavor and
/// kind: `cargo test -p fereg-core --release --test fit_se parity_report -- --ignored --nocapture`.
#[test]
#[ignore]
fn parity_report() {
    let mut py = read_json("fit_se_pyfixest.json").list();
    py.extend(read_json("fit_se_pyconley.json").list());
    for (flavor, cases) in [(Flavor::Fixest, read_json("fit_se_fixest.json").list()), (Flavor::Pyfixest, py)] {
        let mut worst: Vec<(String, f64, f64, String)> = Vec::new();
        for j in cases.iter().filter(|j| j.get("error").is_none()) {
            let case = Case::new(j, FactorKind::F64);
            let spec = case.spec(flavor);
            let out = fit::fit(&case.input(), &spec, &Ctx::quiet(4)).unwrap();
            let e = j.at("expect");
            let names: Vec<String> = out.coef_source.iter().map(|s| case.source_name(s, flavor)).collect();
            let want_names = e.at("coef_names").strs();
            let k = names.len();
            let p: Vec<usize> = want_names.iter().map(|w| names.iter().position(|n| n == w).unwrap()).collect();
            let want = e.at("vcov").nums();
            let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let dv = p.iter().flat_map(|&a| p.iter().map(move |&b| (a, b))).zip(&want).map(|((a, b), w)| (out.vcov.vcov[a * k + b] - w).abs() / scale).fold(0.0, f64::max);
            let ds = p.iter().zip(e.at("se").nums()).map(|(&a, w)| (out.vcov.vcov[a * k + a].sqrt() - w).abs() / w.abs()).fold(0.0, f64::max);
            let model = j.get("model").map_or("ols", J::str).to_string();
            let iv = if j.at("endog").list().is_empty() { "" } else { " IV" };
            let label = match spec.vcov {
                VcovKind::Hc2 | VcovKind::Hc3 => "HC2/HC3",
                VcovKind::Crv3 => "CRV3",
                VcovKind::NeweyWest { .. } => "NW",
                VcovKind::DriscollKraay { .. } => "DK",
                VcovKind::Conley { .. } => "Conley",
                _ => "other",
            };
            let key = format!("{label} {model}{iv}");
            match worst.iter_mut().find(|w| w.0 == key) {
                Some(w) if ds.max(dv) > w.1.max(w.2) => *w = (key, ds, dv, j.at("name").str().to_string()),
                Some(_) => {}
                None => worst.push((key, ds, dv, j.at("name").str().to_string())),
            }
        }
        for (key, ds, dv, name) in worst {
            println!("{flavor:?} {key}: max rel diff se {ds:.1e}, vcov {dv:.1e} (worst case {name})");
        }
    }
}
