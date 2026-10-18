//! Analytical bias correction and APEs (alpaca normalization) against alpaca 0.3.5 with its exact
//! control (tests/data/bias/alpaca.json, written by gen_alpaca.R). Quantities after the fixed-effect
//! re-estimation at the corrected coefficients are compared with the generator's independent exact
//! re-solve (alpaca's own re-solve is imprecise, alpaca spec Q2). APE variances use the production
//! row set (all N* rows); alpaca's retained-rows variant is checked by the crate's unit tests.

mod bias_common;

use bias_common::*;
use fereg_core::Ctx;
use fereg_core::fit::{self, ApeAt, ApeOptions, BiasOptions, Flavor, Sampling, Ssc, Structure, VcovData, VcovKind, WhichBias};

/// Coefficients, bias terms and everything from the uncorrected fit.
const RTOL: f64 = 1e-8;
/// After the re-estimation of the fixed effects (vs the exact re-solve of the generator).
const RTOL_C: f64 = 1e-7;

fn structure(c: &J) -> Structure {
    if c.at("structure").str() == "network" { Structure::Network } else { Structure::Classic }
}

fn opts(c: &J, l: usize, which: WhichBias) -> BiasOptions {
    BiasOptions { structure: structure(c), bandwidth: l, which, ..BiasOptions::default() }
}

/// Largest |sum_g (y - mu)| over every FE group of a fit (kept rows).
fn fe_score(case: &BCase, out: &fit::FitOutput) -> f64 {
    let resid = out.resid.as_ref().unwrap();
    let mut worst = 0.0f64;
    for q in 0..case.fe.len() {
        let col = &case.factors[q];
        let mut sums: std::collections::BTreeMap<i64, f64> = Default::default();
        for (r, &i) in out.obs.iter().enumerate() {
            *sums.entry(col[i as usize] as i64).or_default() += resid[r];
        }
        worst = sums.values().fold(worst, |m, v| m.max(v.abs()));
    }
    worst
}

#[test]
fn alpaca_cases() {
    let j = read_json("bias/alpaca.json");
    let ctx = Ctx::quiet(4);
    let mut report = Vec::new();
    for flavor in [Flavor::Fixest, Flavor::Pyfixest] {
        for c in j.at("cases").list() {
            let case = BCase::new(&c, &[]);
            let name = format!("{:?}/{}", flavor, case.name);
            let input = case.input();
            let sp = spec(flavor);
            let out = fit::fit(&input, &sp, &ctx).unwrap();
            let r0 = check(&format!("{name} coef"), &out.coef, &c.at("feglm").at("coef").nums(), RTOL);
            if flavor == Flavor::Fixest {
                assert_eq!(out.nobs, c.at("n").num() as usize, "{name} n");
            }
            let mut worst = [r0, 0.0, 0.0, 0.0];
            let mut bc_coef: Vec<(usize, Vec<f64>)> = Vec::new();
            for b in c.at("bc").list() {
                let l = b.at("L").num() as usize;
                let bf = fit::bias_corr(&input, &sp, &opts(&c, l, WhichBias::Both), &ctx).unwrap_or_else(|e| panic!("{name} L={l}: {e}"));
                let tag = format!("{name} bc L={l}");
                worst[1] = worst[1].max(check(&format!("{tag} bias_term"), &bf.report.bias_term, &b.at("bias_term").nums(), RTOL));
                worst[1] = worst[1].max(check(&format!("{tag} coef"), &bf.fit.coef, &b.at("coef").nums(), RTOL));
                check(&format!("{tag} coef_uncorr"), &bf.report.coef_uncorr, &b.at("coef_uncorr").nums(), RTOL);
                let score = fe_score(&case, &bf.fit);
                assert!(score < 1e-9, "{tag}: FE score {score:e}");
                if flavor == Flavor::Fixest {
                    let eta_sum: f64 = bf.fit.eta.as_ref().unwrap().iter().sum();
                    worst[2] = worst[2].max(check(&format!("{tag} eta sum"), &[eta_sum], &[b.at("eta_c_sum").num()], RTOL_C));
                    worst[2] = worst[2].max(check_mat(&format!("{tag} vcov hessian"), &bf.fit.vcov.vcov, &b.at("vcov_hessian_c").nums(), RTOL_C));
                    let st = bf.fit.state.as_ref().unwrap();
                    let data = VcovData { factors: &input.factors, factor_names: &input.factor_names, ..Default::default() };
                    let sw = st.vcov(VcovKind::Hetero, &Ssc { k_adj: false, g_adj: false, ..Ssc::default() }, &data, &ctx).unwrap();
                    worst[2] = worst[2].max(check_mat(&format!("{tag} vcov sandwich"), &sw.vcov, &b.at("vcov_sandwich_c").nums(), RTOL_C));
                    let crv3 = st.vcov(VcovKind::Crv3, &Ssc::default(), &data, &ctx).err().unwrap();
                    assert_eq!(crv3.to_string(), "CRV3 is not available after bias correction");
                }
                bc_coef.push((l, bf.fit.coef.clone()));
            }
            for b in c.get("bc_which").map_or(Vec::new(), J::list) {
                let l = b.at("L").num() as usize;
                let which = match b.at("which").str() {
                    "individual" => WhichBias::Individual,
                    "time" => WhichBias::Time,
                    _ => WhichBias::Both,
                };
                if which == WhichBias::Time && l > 0 {
                    // alpaca has no bias choice; fereg refuses a bandwidth without effect.
                    let e = fit::bias_corr(&input, &sp, &opts(&c, l, which), &ctx).err().unwrap();
                    assert!(e.to_string().contains("has no effect"), "{e}");
                    continue;
                }
                let bf = fit::bias_corr(&input, &sp, &opts(&c, l, which), &ctx).unwrap();
                let tag = format!("{name} bc {which:?} L={l}");
                worst[1] = worst[1].max(check(&format!("{tag} bias_term"), &bf.report.bias_term, &b.at("bias_term").nums(), RTOL));
                check(&format!("{tag} coef"), &bf.fit.coef, &b.at("coef").nums(), RTOL);
            }
            for a in c.at("apes").list() {
                if a.at("rows").str() != "all" {
                    continue;
                }
                let l = a.at("L").num() as usize;
                let ao = ApeOptions {
                    n_pop: match a.at("n_pop") {
                        J::Null => None,
                        v => Some(v.num()),
                    },
                    structure: structure(&c),
                    sampling: if a.at("sampling").str() == "independence" { Sampling::Independence } else { Sampling::Unrestricted },
                    weak_exo: a.at("weak_exo").bool(),
                };
                let bo = opts(&c, l, WhichBias::Both);
                let is_bc = a.at("object").str() == "bc";
                let coef = bc_coef.iter().find(|(x, _)| *x == l).map(|(_, v)| v.clone()).unwrap_or_default();
                let at = if is_bc { ApeAt::Corrected { options: &bo, coef: &coef, ape_jackknife: None } } else { ApeAt::Uncorrected };
                let got = fit::apes(&input, &sp, at, &ao, &ctx).unwrap_or_else(|e| panic!("{name} apes: {e}"));
                let tag = format!("{name} ape {} L={l} npop={:?} {:?} weak={}", a.at("object").str(), ao.n_pop, ao.sampling, ao.weak_exo);
                let tol = if is_bc { RTOL_C } else { RTOL };
                assert_eq!(got.weak_exo_used, a.at("weak_exo_used").bool(), "{tag} weak_exo_used");
                assert_eq!(got.binary, c.at("binary").list().iter().map(J::bool).collect::<Vec<_>>(), "{tag} binary");
                worst[3] = worst[3].max(check(&format!("{tag} delta"), &got.delta, &a.at("delta").nums(), tol));
                worst[3] = worst[3].max(check_mat(&format!("{tag} vcov"), &got.vcov, &a.at("vcov").nums(), tol));
                if is_bc {
                    worst[3] = worst[3].max(check_mat(&format!("{tag} bias_term"), got.bias_term.as_ref().unwrap(), &a.at("bias_term").nums(), tol));
                }
            }
            report.push(format!("{name}: coef {:.1e} bias {:.1e} corrected-point {:.1e} apes {:.1e}", worst[0], worst[1], worst[2], worst[3]));
        }
    }
    println!("{}", report.join("\n"));
}
