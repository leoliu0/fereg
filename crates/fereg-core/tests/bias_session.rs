//! The Stata path: the kernels of `fereg_core::bias` after `Session::glm_parts` under the ppmlhdfe
//! convention, with the session's absorber and a jackknife closure that refits subpanels through
//! their own sessions (what `cmd_glm.rs` does), against logitfe 3.0.3 (tests/data/bias/logitfe.json).

mod bias_common;

use bias_common::*;
use fereg_core::bias::{self, ApeSpec, BiasSpec, FeCodes, FullSample, JkPanel, JkSpec, LogitData, LogitfePanel, Normalization, Sampling, Structure, SubFit, Variant, WhichBias};
use fereg_core::{
    AbsorbTerm, Ctx, Family, FactorValues, GlmCollinear, GlmConvention, GlmOptions, GlmParts, Guess, Layout, Prep, RfError, Separation, Session, Spec, Vce, WeightKind,
};

/// One Stata-style fit of `y ~ x | id + t` on the rows `rows` of the data (logit, ppmlhdfe
/// convention): the session (absorber alive) and its parts.
fn stata_fit(csv: &Csv, rows: &[usize], x: &[&str], ctx: &Ctx) -> Result<(Session, GlmParts), RfError> {
    let sub = |c: &str| rows.iter().map(|&r| csv.col(c)[r]).collect::<Vec<f64>>();
    let (id, t, y) = (sub("id"), sub("t"), sub("y"));
    let prep = Prep {
        n: rows.len(),
        factors: vec![FactorValues::Num(&id), FactorValues::Num(&t)],
        absorb: (0..2).map(|q| AbsorbTerm { factors: vec![q], intercept: true, slopes: vec![], save: false }).collect(),
        slope_vars: vec![],
        weights: None,
        weight_kind: WeightKind::None,
        vce: Vce::Unadjusted,
        drop_singletons: true,
        family: Family::Logit,
        y_for_sep: Some(&y),
        sep_fe: true,
    };
    let mut s = Session::prepare(prep, ctx)?;
    let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
    let mut cols = vec![kept.iter().map(|&i| y[i]).collect::<Vec<f64>>()];
    cols.extend(x.iter().map(|c| {
        let v = sub(c);
        kept.iter().map(|&i| v[i]).collect::<Vec<f64>>()
    }));
    let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
    let block = s.gather(&refs, ctx)?;
    let opts = GlmOptions { family: Family::Logit, tol: 1e-14, maxiter: 100, itol: 1e-13, separation: Separation::default(), guess: Guess::Simple, log: false, collinear: GlmCollinear::Ppmlhdfe };
    let spec = Spec { tol: 1e-13, ..Spec::default() };
    let (_, parts) = s.glm_parts(block, &Layout::ols(x.len()), None, &spec, &opts, GlmConvention::Ppmlhdfe, ctx)?;
    Ok((s, parts))
}

/// Kept-order columns of a fit.
struct KeptCols {
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
    id: Vec<u32>,
    n_id: usize,
    period: Vec<i64>,
}

fn kept_cols(csv: &Csv, rows: &[usize], s: &Session, x: &[&str]) -> KeptCols {
    let at = |c: &str| s.kept_index().iter().map(|&i| csv.col(c)[rows[i as usize]]).collect::<Vec<f64>>();
    let id = s.terms()[0].codes.clone();
    KeptCols { y: at("y"), x: x.iter().map(|c| at(c)).collect(), n_id: s.terms()[0].g, id, period: at("t").iter().map(|&v| v as i64).collect() }
}

#[test]
fn stata_path_matches_logitfe() {
    let j = read_json("bias/logitfe.json");
    let case = |n: &str| j.at("cases").list().into_iter().find(|c| c.at("name").str() == n).unwrap();
    let csv = Csv::read("bias/l_static.csv");
    let xs = ["x1", "x2", "x3", "d"];
    let ctx = Ctx::quiet(8);
    let all: Vec<usize> = (0..csv.n()).collect();
    let n_full = all.len();
    let (mut s, parts) = stata_fit(&csv, &all, &xs, &ctx).unwrap();
    let coef: Vec<f64> = parts.coef.clone();
    let kc = kept_cols(&csv, &all, &s, &xs);
    let fe: Vec<(Vec<u32>, usize)> = s.terms().iter().map(|t| (t.codes.clone(), t.g)).collect();
    let d = LogitData {
        y: &kc.y,
        weights: None,
        offset: None,
        x: kc.x.iter().map(Vec::as_slice).collect(),
        fe: fe.iter().map(|(c, g)| FeCodes { codes: c, g: *g }).collect(),
        n_full,
    };
    check("uncorrected", &coef, &case("s_noc").at("expect").at("coef").nums(), 1e-6);
    let panel = LogitfePanel { individual: &kc.id, n_individuals: kc.n_id, period: &kc.period, individual_effects: true, time_effects: true };
    for (name, l) in [("s_an_l0", 0usize), ("s_an_l1", 1), ("s_an_l2", 2)] {
        let spec = BiasSpec { normalization: Normalization::Logitfe, structure: Structure::Classic, which: WhichBias::Both, bandwidth: l, order: None, logitfe: Some(panel) };
        let c = bias::analytical(&d, s.absorber_mut(), &coef, &parts.eta, &spec, &ctx).unwrap();
        // d() / savefe at the corrected coefficients: the FE values sum to eta_c - x'beta_c.
        let fev = bias::saved_fe(s.absorber(), s.terms(), &c.fe_coefs);
        for r in (0..kc.y.len()).step_by(41) {
            let xb: f64 = kc.x.iter().zip(&c.coef).map(|(x, b)| x[r] * b).sum();
            assert!((fev[0][r] + fev[1][r] + xb - c.eta[r]).abs() < 1e-9, "{name}: saved FE");
        }
        let e = case(name);
        let tol = if e.at("source").str() == "indep" { 1e-8 } else { 1e-6 };
        check(&format!("{name} coef"), &c.coef, &e.at("expect").at("coef").nums(), tol);
        // Corrected APEs: point at (beta_c, eta_c) with the bias term, no variance.
        let a = ApeSpec { structure: Structure::Classic, sampling: Sampling::Independence, weak_exo: false, adj: 0.0, correction: Some(spec), full: None, want_vcov: false };
        let ap = bias::apes(&d, s.absorber_mut(), &c.coef, &c.eta, &a, &ctx).unwrap();
        check(&format!("{name} ape"), &ap.delta, &e.at("expect").at("ape").nums(), tol);
    }
    // APE variance at the uncorrected fit, logitfe's default adj = 1, all N* rows.
    let full_kept: Vec<u32> = {
        let mut v = vec![u32::MAX; n_full];
        s.kept_index().iter().enumerate().for_each(|(k, &i)| v[i as usize] = k as u32);
        v
    };
    let dense = |c: &str| {
        let v = csv.col(c);
        let mut u: Vec<f64> = v.to_vec();
        u.sort_by(f64::total_cmp);
        u.dedup();
        (v.iter().map(|x| u.binary_search_by(|y| y.total_cmp(x)).unwrap() as u32).collect::<Vec<u32>>(), u.len())
    };
    let (fid, gid) = dense("id");
    let (ft, gt) = dense("t");
    let order: Vec<i64> = csv.col("t").iter().map(|&v| v as i64).collect();
    let full = FullSample { kept: &full_kept, fe: vec![FeCodes { codes: &fid, g: gid }, FeCodes { codes: &ft, g: gt }], order: Some(&order) };
    let a = ApeSpec { structure: Structure::Classic, sampling: Sampling::Independence, weak_exo: false, adj: 1.0, correction: None, full: Some(full), want_vcov: true };
    let ap = bias::apes(&d, s.absorber_mut(), &coef, &parts.eta, &a, &ctx).unwrap();
    let want = case("s_noc").at("ape_vcov").list().into_iter().find(|v| v.at("adj").num() == 1.0 && v.at("rows").str() == "all").unwrap();
    check_mat("ape_vcov adj=1", &ap.vcov, &want.at("vcov").nums(), 1e-8);

    // Jackknife ss2 and jj through a closure refitting each subpanel with its own session.
    let binary = bias::binary(&d.x);
    let ape0 = bias::ape_point(&d.x, &binary, &coef, &parts.eta, n_full as f64);
    let full_fit = SubFit { coef: coef.clone(), ape: ape0 };
    let jk_panel = JkPanel { individual: Some(&fid), n_individuals: gid, period: Some(&ft), n_periods: gt, double: None };
    let fit = |_: &bias::Subpanel, keep: &[bool], fctx: &Ctx| -> Result<SubFit, RfError> {
        let rows: Vec<usize> = (0..n_full).filter(|&r| keep[r]).collect();
        let (s, p) = stata_fit(&csv, &rows, &xs, fctx)?;
        if p.status.iter().any(|&st| st != 0) {
            return Err(RfError::InvalidInput("a regressor is collinear in this subpanel".into()));
        }
        let k = kept_cols(&csv, &rows, &s, &xs);
        let x: Vec<&[f64]> = k.x.iter().map(Vec::as_slice).collect();
        Ok(SubFit { ape: bias::ape_point(&x, &binary, &p.coef, &p.eta, rows.len() as f64), coef: p.coef })
    };
    for (name, variant) in [("s_jk_ss2", Variant::Ss2), ("s_jk_jj", Variant::Jj)] {
        let spec = JkSpec { variant, which: WhichBias::Both, partitions: &[], bytes_per_fit: 0 };
        let out = bias::jackknife(&jk_panel, &spec, &full_fit, fit, &ctx).unwrap();
        let e = case(name);
        check(&format!("{name} coef"), &out.coef, &e.at("expect").at("coef").nums(), 1e-4);
        check(&format!("{name} ape"), &out.ape, &e.at("expect").at("ape").nums(), 1e-4);
        // Deterministic across thread counts.
        let again = bias::jackknife(&jk_panel, &spec, &full_fit, fit, &Ctx::quiet(1)).unwrap();
        assert_eq!(again, out, "{name} threads");
    }
}
