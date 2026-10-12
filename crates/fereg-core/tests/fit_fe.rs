//! `fit` varying slopes, fixed-effect values and post-hoc variance matrices against fixest 0.14.2
//! (tests/data/fit_fixest_fe.json, the `fixef` of fit_fixest_iv.json and fit_fixest_glm.json) and
//! pyfixest 0.60.0 (the `fixef` and post-hoc cases of tests/data/fit_pyfixest_b.json).

mod fit_common;

use fereg_core::Ctx;
use fereg_core::fit::{self, FitOutput, FixefNorm, Flavor, VcovKind};
use fit_common::*;

const RTOL: f64 = 1e-8;

/// Our fixed effects as (fixest-style name, levels as the numbers of their labels, values,
/// references).
fn ours(case: &Case, out: &FitOutput) -> Vec<(String, Vec<Vec<f64>>, Vec<f64>, usize)> {
    out.fixef
        .iter()
        .map(|f| {
            let term = &case.fe[f.term];
            let fname = term.factors.iter().map(|&q| case.factor_names[q].clone()).collect::<Vec<_>>().join("^");
            let name = match f.slope {
                None => fname,
                Some(s) => format!("{fname}[[{}]]", case.slope_names[s]),
            };
            let labels = f.first_obs.iter().map(|&i| term.factors.iter().map(|&q| case.csv.col(&case.factor_names[q])[i as usize]).collect()).collect();
            (name, labels, f.values.clone(), f.references)
        })
        .collect()
}

fn label_numbers(s: &str) -> Vec<f64> {
    s.split('_').map(|p| p.parse::<f64>().unwrap_or_else(|_| panic!("label {s}"))).collect()
}

/// Per-row sum of the fixed effects (intercepts plus slope × variable) over the estimation rows.
fn fe_sum(case: &Case, out: &FitOutput, fixef: &[(usize, Option<usize>, Vec<f64>, Vec<Vec<f64>>)]) -> Vec<f64> {
    out.obs
        .iter()
        .map(|&i| {
            fixef
                .iter()
                .map(|(t, slope, values, labels)| {
                    let term = &case.fe[*t];
                    let key: Vec<f64> = term.factors.iter().map(|&q| case.csv.col(&case.factor_names[q])[i as usize]).collect();
                    let l = labels.iter().position(|x| *x == key).unwrap_or_else(|| panic!("level {key:?}"));
                    values[l] * slope.map_or(1.0, |s| case.csv.col(&case.slope_names[s])[i as usize])
                })
                .sum()
        })
        .collect()
}

/// fixest's fixed effects: values per level (by label) and references; with varying slopes
/// fixest normalizes the intercepts by its own iterations, so there only the slopes and the
/// per-row sums are compared.
fn check_fixest_fixef(j: &J, case: &Case, out: &FitOutput) {
    let name = j.at("name").str();
    let Some(want) = j.at("expect").get("fixef") else { return };
    let ours = ours(case, out);
    let slopes = case.fe.iter().any(|t| !t.slopes.is_empty());
    let mut want_sum_terms = Vec::new();
    let mut our_sum_terms = Vec::new();
    for w in want.list() {
        let wname = w.at("name").str();
        let (k, (_, labels, values, refs)) = ours.iter().enumerate().find(|(_, o)| o.0 == wname).unwrap_or_else(|| panic!("{name}: no fixef {wname}"));
        let f = &out.fixef[k];
        let wl: Vec<Vec<f64>> = w.at("levels").strs().iter().map(|s| label_numbers(s)).collect();
        let wv = w.at("values").nums();
        assert_eq!(wl.len(), labels.len(), "{name} {wname}: levels");
        want_sum_terms.push((f.term, f.slope, wv.clone(), wl.clone()));
        our_sum_terms.push((f.term, f.slope, values.clone(), labels.clone()));
        if slopes && f.slope.is_none() {
            continue;
        }
        let scale = wv.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (l, v) in wl.iter().zip(&wv) {
            let p = labels.iter().position(|x| x == l).unwrap_or_else(|| panic!("{name} {wname}: level {l:?}"));
            assert!(close(values[p], *v, RTOL, scale), "{name} {wname} level {l:?}: got {:e}, want {v:e}", values[p]);
        }
        let wr = w.at("references").nums().first().copied().unwrap_or(0.0);
        assert_eq!(*refs as f64, wr, "{name} {wname}: references");
    }
    let a = fe_sum(case, out, &our_sum_terms);
    let b = fe_sum(case, out, &want_sum_terms);
    let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert_vec(&format!("{name} FE sum"), &a, &b, RTOL, scale);
}

#[test]
fn fixest_fe_cases() {
    let mut cases = read_json("fit_fixest_fe.json").list();
    for file in ["fit_fixest_iv.json", "fit_fixest_glm.json"] {
        cases.extend(read_json(file).list().into_iter().filter(|c| c.at("expect").get("fixef").is_some()));
    }
    assert!(cases.len() >= 16);
    for j in &cases {
        let (case, out) = check_case(j, Flavor::Fixest, FactorKind::F64, 4, RTOL);
        check_fixest_fixef(j, &case, &out);
    }
}

/// The fixed effects add up to the fitted values net of the regressors at every row (GLM: the
/// linear predictor net of the offset and the regressors).
#[test]
fn fixef_sum_identity() {
    let mut cases = read_json("fit_fixest_fe.json").list();
    cases.extend(read_json("fit_fixest_glm.json").list());
    for name in ["sl_g_s_h_iid", "sl_only", "fx_fe3_comp", "fx_inter", "pois_sl", "pois_sl_only", "pois_fe2_off_w", "logit_fe1_off"] {
        let j = cases.iter().find(|c| c.at("name").str() == name).unwrap();
        let case = Case::new(j, FactorKind::F64);
        let input = case.input();
        for norm in [FixefNorm::Fixest, FixefNorm::MinNorm] {
            let spec = fit::FitSpec { fixef_norm: norm, ..case.spec(Flavor::Fixest) };
            let out = fit::fit(&input, &spec, &Ctx::quiet(3)).unwrap();
            let terms: Vec<(usize, Option<usize>, Vec<f64>, Vec<Vec<f64>>)> =
                ours(&case, &out).into_iter().zip(&out.fixef).map(|((_, labels, values, _), f)| (f.term, f.slope, values, labels)).collect();
            let sums = fe_sum(&case, &out, &terms);
            let glm = case.model() != fit::Model::Ols;
            let base = if glm { out.eta.as_ref().unwrap() } else { out.fitted.as_ref().unwrap() };
            let net: Vec<f64> = out
                .obs
                .iter()
                .enumerate()
                .map(|(k, &i)| {
                    let xb: f64 = out
                        .coef_source
                        .iter()
                        .zip(&out.coef)
                        .map(|(s, b)| match s {
                            fit::CoefSource::Intercept => *b,
                            fit::CoefSource::X(x) => b * input.x[*x][i as usize],
                            _ => unreachable!(),
                        })
                        .sum();
                    base[k] - xb - input.offset.map_or(0.0, |o| o[i as usize])
                })
                .collect();
            let scale = net.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert_vec(&format!("{name} {norm:?}"), &sums, &net, 1e-9, scale);
        }
    }
}

#[test]
fn pyfixest_fixef_and_posthoc() {
    let cases = read_json("fit_pyfixest_b.json").list();
    let mut n = 0;
    for j in cases.iter().filter(|c| c.at("endog").list().is_empty() || c.at("name").str() == "py_posthoc_iv") {
        let name = j.at("name").str();
        let mut j = j.clone();
        if let J::Obj(fields) = &mut j {
            // pyfixest's fixef() is the minimum-norm solution: the case runs with that normalization.
            fields.retain(|(k, _)| k != "fixef_norm");
        }
        let case = Case::new(&j, FactorKind::F64);
        let spec = fit::FitSpec { fixef_norm: FixefNorm::MinNorm, ..case.spec(Flavor::Pyfixest) };
        let out = fit::fit(&case.input(), &spec, &Ctx::quiet(4)).unwrap_or_else(|e| panic!("{name}: {e}"));
        let (_, base) = check_case(&j, Flavor::Pyfixest, FactorKind::F64, 4, RTOL);
        assert_eq!(base.coef, out.coef);
        let Some(want) = j.at("expect").get("fixef") else {
            n += 1;
            continue;
        };
        let ours = ours(&case, &out);
        for (q, w) in want.list().iter().enumerate() {
            let wname = w.at("name").str();
            let (_, labels, values, refs) = ours.iter().find(|o| o.0 == wname).unwrap_or_else(|| panic!("{name}: no fixef {wname}"));
            let wv = w.at("values").nums();
            let wl: Vec<Vec<f64>> = w.at("levels").strs().iter().map(|s| label_numbers(s)).collect();
            // pyfixest omits the pinned first level of every term after the first.
            assert_eq!(wl.len() + usize::from(q > 0), labels.len(), "{name} {wname}: levels");
            if q > 0 {
                assert_eq!((values[0], *refs), (0.0, 1), "{name} {wname}: pinned level");
            }
            let scale = wv.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for (l, v) in wl.iter().zip(&wv) {
                let p = labels.iter().position(|x| x == l).unwrap_or_else(|| panic!("{name} {wname}: level {l:?}"));
                assert!(close(values[p], *v, 1e-7, scale), "{name} {wname} level {l:?}: got {:e}, want {v:e}", values[p]);
            }
        }
        n += 1;
    }
    assert!(n >= 7);
}

/// Post-hoc variance matrices: lean fits keep no state; a missing cluster value on an
/// estimation row is an error; the threads do not change the result.
#[test]
fn posthoc_requests() {
    let cases = read_json("fit_fixest_fe.json").list();
    let j = cases.iter().find(|c| c.at("name").str() == "fx_nofe_posthoc").unwrap();
    let case = Case::new(j, FactorKind::F64);
    let input = case.input();
    let lean = fit::fit(&input, &fit::FitSpec { keep_state: false, ..case.spec(Flavor::Fixest) }, &Ctx::quiet(2)).unwrap();
    assert!(lean.state.is_none());
    let out = fit::fit(&input, &case.spec(Flavor::Fixest), &Ctx::quiet(2)).unwrap();
    let state = out.state.as_ref().unwrap();
    assert_eq!(state.nobs(), out.nobs);
    let h = FactorData::new(case.csv.col("h"), FactorKind::F64);
    let names = vec!["h".to_string()];
    let err = state.vcov(VcovKind::Cluster, &Default::default(), &[h.col()], &names, &[vec![0]], &Ctx::quiet(2)).err().unwrap().to_string();
    assert!(err.contains("missing values"), "{err}");
    let c3 = FactorData::new(case.csv.col("c3"), FactorKind::Codes);
    let k = FactorData::new(case.csv.col("k"), FactorKind::I32);
    let names = vec!["c3".to_string(), "k".to_string()];
    let cols = [c3.col(), k.col()];
    let a = state.vcov(VcovKind::Cluster, &Default::default(), &cols, &names, &[vec![0], vec![1], vec![0, 1]], &Ctx::quiet(1)).unwrap();
    let b = state.vcov(VcovKind::Cluster, &Default::default(), &cols, &names, &[vec![0], vec![1], vec![0, 1]], &Ctx::quiet(16)).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.n_clusters.len(), 3);
}

/// `FitState::fixef` (fixed effects on demand) equals the values computed at fit time, under
/// both normalizations, and a fit without `keep_fixef` returns none.
#[test]
fn lazy_fixef_matches_eager() {
    let mut cases = read_json("fit_fixest_fe.json").list();
    for file in ["fit_fixest_iv.json", "fit_fixest_glm.json", "fit_pyfixest_b.json", "fit_pyfixest_glm.json"] {
        cases.extend(read_json(file).list().into_iter().filter(|c| c.get("error").is_none() && !c.at("fe").list().is_empty()));
    }
    assert!(cases.len() >= 40);
    for j in &cases {
        let name = j.at("name").str();
        let case = Case::new(j, FactorKind::F64);
        let input = case.input();
        let flavor = if name.starts_with("py_") { Flavor::Pyfixest } else { Flavor::Fixest };
        let lazy = fit::fit(&input, &fit::FitSpec { keep_fixef: false, ..case.spec(flavor) }, &Ctx::quiet(3)).unwrap();
        assert!(lazy.fixef.is_empty(), "{name}");
        for norm in [FixefNorm::Fixest, FixefNorm::MinNorm] {
            let eager = fit::fit(&input, &fit::FitSpec { keep_fixef: true, fixef_norm: norm, ..case.spec(flavor) }, &Ctx::quiet(3)).unwrap();
            assert_eq!(eager.coef, lazy.coef, "{name}");
            assert_eq!(lazy.state.as_ref().unwrap().fixef(norm, &Ctx::quiet(1)).unwrap(), eager.fixef, "{name} {norm:?}");
        }
    }
}
