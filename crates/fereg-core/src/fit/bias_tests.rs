//! alpaca's retained-rows APE variance (the `#[cfg(test)]` row set) against alpaca 0.3.5 on data
//! with dropped groups: checks the machinery shared with the production all-rows variance (delta
//! part, S(.), D'D, the weak-exogeneity covariance C).

#[path = "../../tests/bias_common/mod.rs"]
mod bias_common;

use bias_common::*;

use super::{ApeAt, ApeOptions, BiasOptions, Sampling, Structure, apes_retained, bias_corr};
use crate::api::Ctx;
use crate::fit::{Flavor, fit};

#[test]
fn retained_rows_match_alpaca() {
    let j = read_json("bias/alpaca.json");
    let ctx = Ctx::quiet(4);
    let mut worst = (0.0f64, 0.0f64);
    for c in j.at("cases").list() {
        let case = BCase::new(&c, &[]);
        let input = case.input();
        let sp = spec(Flavor::Fixest);
        let structure = if c.at("structure").str() == "network" { Structure::Network } else { Structure::Classic };
        assert!(fit(&input, &sp, &ctx).unwrap().nobs < c.at("n_full").num() as usize, "{}: rows dropped", case.name);
        for a in c.at("apes").list() {
            if a.at("rows").str() != "retained" {
                continue;
            }
            let l = a.at("L").num() as usize;
            let bo = BiasOptions { structure, bandwidth: l, ..BiasOptions::default() };
            let is_bc = a.at("object").str() == "bc";
            let coef = if is_bc { bias_corr(&input, &sp, &bo, &ctx).unwrap().fit.coef } else { Vec::new() };
            let at = if is_bc { ApeAt::Corrected { options: &bo, coef: &coef, ape_jackknife: None } } else { ApeAt::Uncorrected };
            let ao = ApeOptions {
                n_pop: match a.at("n_pop") {
                    J::Null => None,
                    v => Some(v.num()),
                },
                structure,
                sampling: if a.at("sampling").str() == "independence" { Sampling::Independence } else { Sampling::Unrestricted },
                weak_exo: a.at("weak_exo").bool(),
            };
            let got = apes_retained(&input, &sp, at, &ao, &ctx).unwrap();
            let tag = format!("{} {} L={l} {:?} {:?} {}", case.name, a.at("object").str(), ao.n_pop, ao.sampling, ao.weak_exo);
            // The uncorrected point is alpaca's own output (its one-way fit is 3.6e-9 from the MLE,
            // which fereg and fixest agree on to 2e-15); the corrected one the exact re-solve.
            let tol = if is_bc { 1e-7 } else { 1e-8 };
            worst.0 = worst.0.max(check(&format!("{tag} delta"), &got.delta, &a.at("delta").nums(), tol));
            worst.1 = worst.1.max(check_mat(&format!("{tag} vcov"), &got.vcov, &a.at("vcov").nums(), tol));
        }
    }
    println!("retained rows vs alpaca: delta {:.1e}, vcov {:.1e}", worst.0, worst.1);
}
