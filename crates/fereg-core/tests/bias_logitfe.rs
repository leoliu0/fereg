//! Logitfe normalization and the jackknife against Stata logitfe 3.0.3 (tests/data/bias/logitfe.json,
//! written by gen_logitfe.do + gen_logitfe_indep.R), through the fit layer with
//! `Normalization::Logitfe` (the Stata path's kernels).
//!
//! Cases with source "logitfe" are logitfe's live numbers: logitfe stops its Newton iterations at
//! Stata's default tolerance, so they are compared at 1e-6 (points) / 1e-4 (jackknife
//! combinations, which amplify that error). Cases with source "indep" are where logitfe is wrong
//! (design sec. 8.4): lags(2) (bug B1: psi_{t-1} for every lag), the dimension swaps of js / sj and
//! of the single-bias jj / double variants (B3), ss2 with the individual-effect bias only (B4:
//! logitfe crashes), and multiple() with explicit partitions; their expectations come from the
//! generator's independent R computation (exact MLE fits of every subpanel, design formulas) and are
//! compared at 1e-8 / 1e-7. APE variances (`ape_vcov`) follow alpaca / Fernandez-Val and Weidner
//! (design sec. 6), not logitfe (bug B2), and come from the independent computation too.

mod bias_common;

use bias_common::*;
use fereg_core::Ctx;
use fereg_core::fit::{self, ApeAt, ApeOptions, BiasMethod, BiasOptions, Flavor, JackknifeOptions, Normalization, Permutation, Variant, WhichBias};

fn variant(s: &str) -> Variant {
    match s {
        "ss1" => Variant::Ss1,
        "ss2" => Variant::Ss2,
        "js" => Variant::Js,
        "sj" => Variant::Sj,
        "jj" => Variant::Jj,
        "double" => Variant::Double,
        v => panic!("variant {v}"),
    }
}

fn perm(j: &J) -> Option<Vec<u32>> {
    match j {
        J::Null => None,
        v => Some(v.nums().iter().map(|&x| x as u32).collect()),
    }
}

#[test]
fn logitfe_cases() {
    let j = read_json("bias/logitfe.json");
    let ctx = Ctx::quiet(8);
    let mut lines = Vec::new();
    for (flavor, c) in [Flavor::Fixest, Flavor::Pyfixest].into_iter().flat_map(|f| j.at("cases").list().into_iter().map(move |c| (f, c))) {
        let sp = spec(flavor);
        let (panel, time) = (c.at("panel").str().to_string(), c.at("time").str().to_string());
        let case = BCase::new(&c, &[&panel, &time]);
        let name = format!("{flavor:?}/{}", case.name);
        let input = case.input();
        let indep = c.at("source").str() == "indep";
        let method = c.at("method").str();
        let jack = method == "jackknife";
        let tol = match (indep, jack) {
            (true, false) => 1e-8,
            (true, true) => 1e-7,
            (false, false) => 1e-6,
            (false, true) => 1e-4,
        };
        let e = c.at("expect");
        let out = fit::fit(&input, &sp, &ctx).unwrap();
        if flavor == Flavor::Fixest {
            assert_eq!(out.nobs, c.at("n").num() as usize, "{name} n");
        }
        let worst_unc = check(&format!("{name} coef_uncorr"), &out.coef, &e.at("coef_uncorr").nums(), if indep { 1e-8 } else { 1e-6 });
        let unc = fit::apes(&input, &sp, ApeAt::Uncorrected, &ApeOptions::default(), &ctx).unwrap();
        check(&format!("{name} ape_uncorr"), &unc.delta, &e.at("ape_uncorr").nums(), if indep { 1e-8 } else { 1e-6 });
        let which = match (c.at("ibias").bool(), c.at("tbias").bool()) {
            (true, false) => WhichBias::Individual,
            (false, true) => WhichBias::Time,
            _ => WhichBias::Both,
        };
        // logitfe resets lags() to 0 when the individual-effect bias is not corrected; fereg
        // refuses the pointless option.
        let lags = if which == WhichBias::Time { 0 } else { c.at("lags").num() as usize };
        let (coef, ape) = match method {
            "none" => (out.coef.clone(), unc.delta.clone()),
            _ => {
                let m = if jack {
                    let partitions = c
                        .get("partitions")
                        .map_or(Vec::new(), |p| p.list().iter().map(|q| Permutation { individual: perm(q.at("individual")), period: perm(q.at("period")) }).collect());
                    BiasMethod::Jackknife(JackknifeOptions { variant: variant(c.at("variant").str()), partitions })
                } else {
                    BiasMethod::Analytical
                };
                let o = BiasOptions {
                    method: m,
                    normalization: Normalization::Logitfe,
                    which,
                    bandwidth: lags,
                    panel: Some(case.factor(&panel)),
                    time: Some(case.factor(&time)),
                    ..BiasOptions::default()
                };
                let b = fit::bias_corr(&input, &sp, &o, &ctx).unwrap_or_else(|er| panic!("{name}: {er}"));
                let ape = match &b.report.ape_jackknife {
                    Some(a) => a.clone(),
                    None => {
                        let at = ApeAt::Corrected { options: &o, coef: &b.fit.coef, ape_jackknife: None };
                        fit::apes(&input, &sp, at, &ApeOptions::default(), &ctx).unwrap().delta
                    }
                };
                (b.fit.coef, ape)
            }
        };
        let rc = check(&format!("{name} coef"), &coef, &e.at("coef").nums(), tol);
        let ra = check(&format!("{name} ape"), &ape, &e.at("ape").nums(), tol);
        let mut rv = 0.0f64;
        for v in c.get("ape_vcov").map_or(Vec::new(), J::list) {
            if v.at("rows").str() != "all" {
                continue;
            }
            let adj = v.at("adj").num();
            // adj = 1 (logitfe's default, infinite population) through a huge n_pop.
            let n_pop = match v.at("population") {
                J::Null if adj == 1.0 => Some(1e300),
                J::Null => None,
                p => Some(p.num()),
            };
            let ao = ApeOptions { n_pop, weak_exo: v.at("weak_exo").bool(), ..ApeOptions::default() };
            let a = fit::apes(&input, &sp, ApeAt::Uncorrected, &ao, &ctx).unwrap();
            assert!((a.adj - adj).abs() < 1e-15, "{name} adj");
            rv = rv.max(check_mat(&format!("{name} ape_vcov adj={adj} weak={}", ao.weak_exo), &a.vcov, &v.at("vcov").nums(), 1e-8));
        }
        lines.push(format!("{name} [{}]: coef_uncorr {worst_unc:.1e} coef {rc:.1e} ape {ra:.1e} ape_vcov {rv:.1e}", c.at("source").str()));
    }
    println!("{}", lines.join("\n"));
}
