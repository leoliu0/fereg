//! Tests for `fereg_core::engine` (`ols_fit`, `iv_fit`, `glm_fit`).

use fereg_core::engine::{CollinearMode, EngineVce, GlmEngineOpts, IvEngineOpts, OlsEngineOpts, glm_fit, iv_fit, ols_fit};
use fereg_core::{AbsorbTerm, AbsorberKind, Ctx, DofOptions, FactorValues, Family, Layout, Prep, Session, Spec, Vce, WeightKind};

fn make_session(n: usize, absorb: Vec<AbsorbTerm>, factors: Vec<FactorValues<'_>>, vce: Vce) -> Session {
    let p = Prep {
        n,
        factors,
        absorb,
        slope_vars: Vec::new(),
        weights: None,
        weight_kind: WeightKind::None,
        vce,
        drop_singletons: false,
        family: Family::Gaussian,
        y_for_sep: None,
        sep_fe: false,
    };
    Session::prepare_with(p, &Ctx::quiet(1), AbsorberKind::Dense).expect("prepare session")
}

#[test]
fn ols_noabsorb_centered_large_offset_and_noconstant() {
    let ctx = Ctx::quiet(1);
    let n = 40usize;
    let mut rows = Vec::with_capacity(n * 3);
    let mut rows_shifted = Vec::with_capacity(n * 3);
    for i in 0..n {
        let x1_base = (i % 10) as f64;
        let x1 = 2000.0 + x1_base;
        let x2 = ((i * 7) % 13) as f64 * 0.5;
        let noise = ((i * 3) % 11) as f64 - 5.0;
        let y = 1.5 + 0.7 * x1_base - 1.2 * x2 + 0.01 * noise;
        rows.extend_from_slice(&[y, x1, x2]);
        rows_shifted.extend_from_slice(&[y, x1_base, x2]);
    }

    // Centered OLS with large offset (2000..2009): matches shifted regression to machine precision.
    let mut sess = make_session(n, Vec::new(), Vec::new(), Vce::Unadjusted);
    let out_c = ols_fit(&mut sess, rows.clone(), 2, &OlsEngineOpts::default(), &ctx).expect("centered ols");
    let mut sess_s = make_session(n, Vec::new(), Vec::new(), Vce::Unadjusted);
    let out_s = ols_fit(&mut sess_s, rows_shifted, 2, &OlsEngineOpts::default(), &ctx).expect("shifted ols");
    assert_eq!(out_c.kept, vec![0, 1]);
    assert_eq!(out_c.ic, 0);
    assert!(out_c.bread_has_cons);
    assert!((out_c.beta[0] - out_s.beta[0]).abs() < 1e-10, "beta[0]={} vs {}", out_c.beta[0], out_s.beta[0]);
    assert!((out_c.beta[1] - out_s.beta[1]).abs() < 1e-10, "beta[1]={} vs {}", out_c.beta[1], out_s.beta[1]);

    // Uncentered (`noconstant`) OLS solves y = b1 * x1 + b2 * x2 without demeaning.
    let mut sess_nc = make_session(n, Vec::new(), Vec::new(), Vce::Unadjusted);
    let opts_nc = OlsEngineOpts { has_constant: false, ..OlsEngineOpts::default() };
    let out_nc = ols_fit(&mut sess_nc, rows.clone(), 2, &opts_nc, &ctx).expect("nocons ols");
    assert!(!out_nc.bread_has_cons);
    assert_eq!(out_nc.b0, 0.0);
    assert_eq!(out_nc.msize, 2);
    assert!((out_nc.beta[0] - 0.7).abs() > 1e-4);

    // Greedy vs Natural collinearity with scaled collinear column (`x3 = 0.5 * x1`).
    let mut rows_col = Vec::with_capacity(n * 4);
    for i in 0..n {
        let row = &rows[i * 3..(i + 1) * 3];
        rows_col.extend_from_slice(&[row[0], row[1], row[2], 0.5 * row[1]]);
    }
    let mut s_nat = make_session(n, Vec::new(), Vec::new(), Vce::Unadjusted);
    let out_nat = ols_fit(&mut s_nat, rows_col.clone(), 3, &OlsEngineOpts { collinear: CollinearMode::Natural, ..OlsEngineOpts::default() }, &ctx).unwrap();
    assert_eq!(out_nat.kept, vec![0, 1]);
    assert_eq!(out_nat.status[2], 3);

    let mut s_grd = make_session(n, Vec::new(), Vec::new(), Vce::Unadjusted);
    let out_grd = ols_fit(&mut s_grd, rows_col, 3, &OlsEngineOpts { collinear: CollinearMode::Greedy, ..OlsEngineOpts::default() }, &ctx).unwrap();
    assert_eq!(out_grd.kept.len(), 2);
}

#[test]
fn ols_collinear_and_fe_d_and_report_constant() {
    let ctx = Ctx::quiet(1);
    let n = 60usize;
    let fe_raw: Vec<f64> = (0..n).map(|i| (i / 6) as f64).collect();
    let cl_raw: Vec<f64> = (0..n).map(|i| (i / 12) as f64).collect();
    let factors = vec![FactorValues::Num(&fe_raw), FactorValues::Num(&cl_raw)];
    let absorb = vec![AbsorbTerm { factors: vec![0], intercept: true, slopes: Vec::new(), save: false }];
    let mut y = Vec::with_capacity(n);
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut x3 = Vec::with_capacity(n);
    for i in 0..n {
        let g = (i / 6) as f64;
        let v1 = ((i % 7) as f64) * 1.3;
        let v2 = 2.0 * v1; // raw collinear with x1
        let v3 = g * 3.0; // invariant within FE -> absorbed!
        y.push(1.0 + 0.5 * v1 + 2.0 * g + ((i % 3) as f64) * 0.2);
        x1.push(v1);
        x2.push(v2);
        x3.push(v3);
    }
    let mut sess = make_session(n, absorb, factors, Vce::Cluster(vec![vec![1]]));
    let rows = sess.gather(&[&y, &x1, &x2, &x3], &ctx).unwrap();
    let opts = OlsEngineOpts {
        vce: EngineVce::Cluster { per_subset_g_adj: false },
        report_constant: true,
        check_raw_collinear: true,
        want_d: true,
        want_resid: true,
        dof: DofOptions::default(),
        ..OlsEngineOpts::default()
    };
    let out = ols_fit(&mut sess, rows, 3, &opts, &ctx).expect("fe ols");
    assert_eq!(out.kept, vec![0]);
    assert!(out.bread_has_cons);
    assert_eq!(out.msize, 2); // [x1, _cons]
    assert!(out.raw_omitted[1]); // x2 raw collinear
    assert!(!out.raw_omitted[2]); // x3 varies overall, absorbed by FE!
    assert_eq!(out.status[1], 3);
    assert_eq!(out.status[2], 2);
    let d = out.d.as_ref().unwrap();
    let e = out.resid.as_ref().unwrap();
    assert_eq!(d.len(), n);
    assert_eq!(e.len(), n);
    assert!(d.iter().sum::<f64>().abs() < 1e-8);
}

#[test]
fn ols_multiway_cluster_per_subset_g_adj_and_newey_dk() {
    let ctx = Ctx::quiet(1);
    let n = 50usize;
    let c1: Vec<f64> = (0..n).map(|i| (i / 10) as f64).collect();
    let c2: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
    let time_vals: Vec<f64> = (0..n).map(|i| ((i % 10) + 1) as f64).collect();
    let time_codes: Vec<u32> = (0..n).map(|i| (i % 10) as u32).collect();
    let panel_codes: Vec<u32> = (0..n).map(|i| (i / 10) as u32).collect();
    let factors = vec![FactorValues::Num(&c1), FactorValues::Num(&c2)];
    let mut rows = Vec::with_capacity(n * 2);
    for i in 0..n {
        let x = ((i * 3) % 11) as f64 + 0.1 * i as f64;
        let y = 2.0 + 1.5 * x + ((i % 5) as f64) * 0.4;
        rows.extend_from_slice(&[y, x]);
    }

    // Two-way cluster with per-subset G/(G-1) (`regress, vce(cluster c1 c2)`).
    let mut sess = make_session(n, Vec::new(), factors, Vce::Cluster(vec![vec![0], vec![1]]));
    let out_mw = ols_fit(
        &mut sess,
        rows.clone(),
        1,
        &OlsEngineOpts { vce: EngineVce::Cluster { per_subset_g_adj: true }, ..OlsEngineOpts::default() },
        &ctx,
    )
    .expect("mw");
    assert_eq!(out_mw.subset_counts.len(), 3); // {1}, {2}, {1,2}

    // Newey-West over panels (`lag(2)`).
    let mut sess_nw = make_session(n, Vec::new(), Vec::new(), Vce::Unadjusted);
    let out_nw = ols_fit(
        &mut sess_nw,
        rows.clone(),
        1,
        &OlsEngineOpts {
            vce: EngineVce::NeweyWest { lag: 2, time: time_vals, delta: 1.0, panel: Some((panel_codes, 5)) },
            ..OlsEngineOpts::default()
        },
        &ctx,
    )
    .expect("newey");
    assert_eq!(out_nw.seen_obs, 40); // 5 panels * 8 lag-2 pairs

    // Driscoll-Kraay (`lag(2)`).
    let mut sess_dk = make_session(n, Vec::new(), Vec::new(), Vce::Unadjusted);
    let out_dk = ols_fit(
        &mut sess_dk,
        rows,
        1,
        &OlsEngineOpts {
            vce: EngineVce::DriscollKraay { lag: 2, time_codes, n_periods: 10 },
            ..OlsEngineOpts::default()
        },
        &ctx,
    )
    .expect("dkraay");
    assert!(out_dk.v_unscaled[0] > 0.0);
}

#[test]
fn iv_fit_ivreg2_modes_and_collinear_constant() {
    let ctx = Ctx::quiet(1);
    let n = 60usize;
    let id: Vec<f64> = (0..n).map(|i| (i / 6) as f64).collect();
    let factors = vec![FactorValues::Num(&id)];
    let cl = vec![vec![0usize]];
    // Layout: [y, ex1, x1, z1, z2]
    let mut rows = Vec::with_capacity(n * 5);
    for i in 0..n {
        let z1 = ((i * 5) % 17) as f64 / 5.0 - 1.5;
        let z2 = ((i * 11) % 19) as f64 / 6.0 - 1.5;
        let ex1 = ((i * 7) % 13) as f64 / 4.0 - 1.5;
        let err = ((i % 7) as f64 - 3.0) * 0.25;
        let x1 = 0.6 * z1 + 0.4 * z2 + 0.3 * err + 0.1 * ex1;
        let y = 1.2 + 1.5 * x1 - 0.8 * ex1 + err;
        rows.extend_from_slice(&[y, ex1, x1, z1, z2]);
    }
    let lay = Layout { nx: 1, nendog: 1, ninst: 2 };

    // 1. Default asymptotic (`small = false`, `report_constant = true`) vs `small = true`.
    let mut s_nosm = make_session(n, Vec::new(), Vec::new(), Vce::Unadjusted);
    let r_nosm = iv_fit(&mut s_nosm, rows.clone(), &lay, &IvEngineOpts { small: false, report_constant: true, ..IvEngineOpts::default() }, &ctx).unwrap();
    let mut s_sm = make_session(n, Vec::new(), Vec::new(), Vce::Unadjusted);
    let r_sm = iv_fit(&mut s_sm, rows.clone(), &lay, &IvEngineOpts { small: true, report_constant: true, ..IvEngineOpts::default() }, &ctx).unwrap();

    let v_nosm = &r_nosm.matrices.iter().find(|m| m.name == "V").unwrap().data;
    let v_sm = &r_sm.matrices.iter().find(|m| m.name == "V").unwrap().data;
    assert_eq!(v_nosm.len(), 9); // 3x3: [x1, ex1, _cons]
    let ratio = v_sm[0] / v_nosm[0];
    assert!((ratio - (n as f64 / (n - 3) as f64)).abs() < 1e-11, "ratio={ratio}");
    // F matches across small/nosmall!
    let f_nosm = r_nosm.scalars.iter().find(|(k, _)| k == "F").unwrap().1;
    let f_sm = r_sm.scalars.iter().find(|(k, _)| k == "F").unwrap().1;
    assert!((f_nosm - f_sm).abs() < 1e-10);

    // 2. Cluster `small = false` vs `small = true` ratio = (G/(G-1)) * ((N-1)/(N-K)).
    let mut sc1 = make_session(n, Vec::new(), factors.clone(), Vce::Cluster(cl.clone()));
    let rc1 = iv_fit(&mut sc1, rows.clone(), &lay, &IvEngineOpts { small: false, report_constant: true, ..IvEngineOpts::default() }, &ctx).unwrap();
    let mut sc2 = make_session(n, Vec::new(), factors, Vce::Cluster(cl));
    let rc2 = iv_fit(&mut sc2, rows, &lay, &IvEngineOpts { small: true, report_constant: true, ..IvEngineOpts::default() }, &ctx).unwrap();
    let vc1 = rc1.matrices.iter().find(|m| m.name == "V").unwrap().data[0];
    let vc2 = rc2.matrices.iter().find(|m| m.name == "V").unwrap().data[0];
    let cl_ratio = (10.0 / 9.0) * ((n - 1) as f64 / (n - 3) as f64);
    assert!((vc2 / vc1 - cl_ratio).abs() < 1e-10, "vc2/vc1={}", vc2 / vc1);

    // 3. Indicator pair `d1 + d2 == 1` drops `d2` and keeps `_cons`.
    let mut rows_ind = Vec::with_capacity(n * 6);
    for i in 0..n {
        let d1 = if i < n / 2 { 1.0 } else { 0.0 };
        let d2 = 1.0 - d1;
        let z1 = ((i * 5) % 17) as f64 / 5.0 - 1.5;
        let z2 = ((i * 11) % 19) as f64 / 6.0 - 1.5;
        let x1 = 0.6 * z1 + 0.4 * z2 + 0.2 * d1;
        let y = 1.0 + 1.5 * x1 - 0.5 * d1 + ((i % 5) as f64) * 0.1;
        rows_ind.extend_from_slice(&[y, d1, d2, x1, z1, z2]);
    }
    let mut si = make_session(n, Vec::new(), Vec::new(), Vce::Unadjusted);
    let ri = iv_fit(&mut si, rows_ind, &Layout { nx: 2, nendog: 1, ninst: 2 }, &IvEngineOpts { small: false, report_constant: true, ..IvEngineOpts::default() }, &ctx).unwrap();
    assert_eq!(ri.status[0], 0);
    assert_eq!(ri.status[1], 3); // d2 dropped!
    let bi = &ri.matrices.iter().find(|m| m.name == "b").unwrap().data;
    assert_eq!(bi.len(), 4); // [x1, d1, d2(=0), _cons]
    assert_ne!(bi[3], 0.0);
}

#[test]
fn glm_fit_poisson_oim_and_noconstant() {
    let ctx = Ctx::quiet(1);
    let n = 50usize;
    let mut rows = Vec::with_capacity(n * 3);
    for i in 0..n {
        let x1 = ((i % 7) as f64 - 3.0) * 0.3;
        let x2 = ((i * 3) % 5) as f64 * 0.2;
        let y = ((i % 4) + 1) as f64;
        rows.extend_from_slice(&[y, x1, x2]);
    }
    // With constant, OIM (`Vce::Unadjusted` NOT coerced to Robust!).
    let mut s1 = make_session(n, Vec::new(), Vec::new(), Vce::Unadjusted);
    let g1 = glm_fit(&mut s1, rows.clone(), 2, None, &GlmEngineOpts::default(), &ctx).expect("poisson oim");
    let v1 = &g1.res.matrices.iter().find(|m| m.name == "V").unwrap().data;
    assert_eq!(g1.msize, 3);
    // For OIM, posted V equals bread!
    for (a, b) in v1.iter().zip(&g1.bread) {
        assert!((a - b).abs() < 1e-12, "oim v={a} bread={b}");
    }

    // Without constant (`noconstant = true`).
    let mut s2 = make_session(n, Vec::new(), Vec::new(), Vce::Unadjusted);
    let g2 = glm_fit(
        &mut s2,
        rows,
        2,
        None,
        &GlmEngineOpts { noconstant: true, ..GlmEngineOpts::default() },
        &ctx,
    )
    .expect("poisson noconstant");
    assert_eq!(g2.msize, 2);
    let b2 = &g2.res.matrices.iter().find(|m| m.name == "b").unwrap().data;
    assert_eq!(b2.len(), 2);
    let ll0 = g2.res.scalars.iter().find(|(k, _)| k == "ll_0").unwrap().1;
    assert!(ll0.is_nan());
}

#[test]
fn ols_constant_only_hc_and_single_cluster_guards() {
    let ctx = Ctx::quiet(1);
    let n = 12usize;
    let y: Vec<f64> = (0..n).map(|i| 2.0 + ((i % 5) as f64) * 0.4).collect();
    let one_cl = vec![1.0; n];

    // Constant-only OLS, HC2, HC3 (`nx == 0`, `has_constant == true`).
    for vce in [EngineVce::Ols, EngineVce::Hc2, EngineVce::Hc3] {
        let mut s = make_session(n, Vec::new(), Vec::new(), Vce::Unadjusted);
        let out = ols_fit(&mut s, y.clone(), 0, &OlsEngineOpts { vce, ..OlsEngineOpts::default() }, &ctx).unwrap();
        assert_eq!(out.msize, 1);
        assert!(out.v_unscaled[0].is_finite() && out.v_unscaled[0] > 0.0);
    }

    // Single-cluster (`g == 1`) with `per_subset_g_adj = true` returns finite zeroed covariance.
    let factors = vec![FactorValues::Num(&one_cl)];
    let mut s1 = make_session(n, Vec::new(), factors, Vce::Cluster(vec![vec![0]]));
    let out1 = ols_fit(
        &mut s1,
        y.clone(),
        0,
        &OlsEngineOpts { vce: EngineVce::Cluster { per_subset_g_adj: true }, ..OlsEngineOpts::default() },
        &ctx,
    )
    .unwrap();
    assert_eq!(out1.v_unscaled, vec![0.0]);
    assert_eq!(out1.n_clust, Some(1.0));

    // GLM single-cluster returns zeroed V instead of dividing by zero.
    let y_bin: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let factors_g = vec![FactorValues::Num(&one_cl)];
    let mut sg = make_session(n, Vec::new(), factors_g, Vce::Cluster(vec![vec![0]]));
    let gg = glm_fit(
        &mut sg,
        y_bin,
        0,
        None,
        &GlmEngineOpts { family: Family::Logit, ..GlmEngineOpts::default() },
        &ctx,
    )
    .unwrap();
    let vg = &gg.res.matrices.iter().find(|m| m.name == "V").unwrap().data;
    assert_eq!(vg, &[0.0]);
}

#[test]
fn ols_natural_uncentred_omits_year2_next_to_year() {
    let ctx = Ctx::quiet(1);
    let n = 120usize;
    let mut rows = Vec::with_capacity(n * 4);
    for i in 0..n {
        let yr = 1990.0 + (i % 30) as f64;
        let yr2 = yr * yr;
        let x = ((i % 7) as f64) * 0.5 - 1.5;
        let y = 1.0 + 0.8 * x + 0.01 * (yr - 2000.0);
        rows.extend_from_slice(&[y, x, yr, yr2]);
    }
    let mut s_nat = make_session(n, Vec::new(), Vec::new(), Vce::Unadjusted);
    let out_nat = ols_fit(&mut s_nat, rows.clone(), 3, &OlsEngineOpts::default(), &ctx).unwrap();
    assert_eq!(out_nat.status, vec![0, 0, 0]);

    let mut s_unc = make_session(n, Vec::new(), Vec::new(), Vce::Unadjusted);
    let out_unc = ols_fit(
        &mut s_unc,
        rows,
        3,
        &OlsEngineOpts { collinear: CollinearMode::NaturalUncentred, ..OlsEngineOpts::default() },
        &ctx,
    )
    .unwrap();
    assert_eq!(out_unc.status[0], 0);
    assert_eq!(out_unc.status[1], 0);
    assert_ne!(out_unc.status[2], 0);
}

#[test]
fn glm_ml_ltol_and_init_mean_trace() {
    let ctx = Ctx::quiet(1);
    let n = 40usize;
    let mut rows = Vec::with_capacity(n * 2);
    for i in 0..n {
        let x = ((i % 7) as f64 - 3.0) * 0.4;
        let y = f64::from(u8::from(i % 3 == 0 || x > 0.5));
        rows.extend_from_slice(&[y, x]);
    }
    let mut s = make_session(n, Vec::new(), Vec::new(), Vce::Unadjusted);
    let out = glm_fit(
        &mut s,
        rows,
        1,
        None,
        &GlmEngineOpts {
            family: Family::Logit,
            init_mean: true,
            ml_ltol: Some(1e-7),
            ml_ll_const: 0.0,
            tol: 1e-10,
            itol: 1e-12,
            ..GlmEngineOpts::default()
        },
        &ctx,
    )
    .expect("glm logit ml_ltol");
    assert!(out.converged);
    assert!(out.dev_trace.len() >= 2);
    assert!(out.dev_trace[0] > *out.dev_trace.last().unwrap());
    assert!(out.dev_trace.windows(2).all(|w| w[1] <= w[0] + 1e-9));
}

#[test]
fn iv_fit_shifted_mean_aweight() {
    let ctx = Ctx::quiet(1);
    let n = 40usize;
    let mut rows = Vec::with_capacity(n * 5);
    let mut wt = Vec::with_capacity(n);
    for i in 0..n {
        let yr = 1950.0 + ((i * 11) % 37) as f64;
        let z1 = (i as f64 * 0.3).sin();
        let z2 = (i as f64 * 0.7).cos();
        let x = 10.0 + 0.7 * z1 - 0.4 * z2 + 0.05 * (yr - 1960.0);
        let y = -0.5 + 1.25 * x + 0.02 * (yr - 1960.0);
        rows.extend_from_slice(&[y, yr, x, z1, z2]);
        wt.push(1.0 + (i % 3) as f64 * 0.5);
    }
    let prep = Prep {
        n,
        factors: Vec::new(),
        absorb: Vec::new(),
        slope_vars: Vec::new(),
        weights: Some(&wt),
        weight_kind: WeightKind::Aweight,
        vce: Vce::Unadjusted,
        drop_singletons: false,
        family: Family::Gaussian,
        y_for_sep: None,
        sep_fe: false,
    };
    let mut sess = Session::prepare_with(prep, &ctx, AbsorberKind::Dense).unwrap();
    let res = iv_fit(
        &mut sess,
        rows,
        &Layout { nx: 1, nendog: 1, ninst: 2 },
        &IvEngineOpts { small: true, report_constant: true, ..IvEngineOpts::default() },
        &ctx,
    )
    .unwrap();
    let b = &res.matrices.iter().find(|m| m.name == "b").unwrap().data;
    assert!((b[0] - 1.25).abs() < 1e-13, "x slope={}", b[0]);
    assert!((b[1] - 0.02).abs() < 1e-13, "yr slope={}", b[1]);
    assert!((b[2] - (-0.5 - 0.02 * 1960.0)).abs() < 1e-10, "cons={}", b[2]);
}

#[test]
fn ols_fit_fweight_hc2_hc3_matches_expanded() {
    let ctx = Ctx::quiet(1);
    let base_rows = [
        [10.0, 1.0, 2.0, 3.0],
        [14.0, 2.5, 1.0, 1.0],
        [19.0, 3.0, 4.0, 2.0],
        [11.0, 0.5, 3.0, 4.0],
        [23.0, 4.0, 2.0, 2.0],
        [16.0, 2.0, 5.0, 1.0],
    ];
    let fw: Vec<f64> = base_rows.iter().map(|r| r[3]).collect();
    let rows_w: Vec<f64> = base_rows.iter().flat_map(|r| [r[0], r[1], r[2]]).collect();
    let mut rows_exp = Vec::new();
    for r in &base_rows {
        for _ in 0..(r[3] as usize) {
            rows_exp.extend_from_slice(&[r[0], r[1], r[2]]);
        }
    }
    let n_exp = rows_exp.len() / 3;
    for vce_kind in [EngineVce::Hc2, EngineVce::Hc3] {
        let p_w = Prep {
            n: base_rows.len(),
            factors: Vec::new(),
            absorb: Vec::new(),
            slope_vars: Vec::new(),
            weights: Some(&fw),
            weight_kind: WeightKind::Fweight,
            vce: match vce_kind {
                EngineVce::Hc2 => Vce::Hc2,
                _ => Vce::Hc3,
            },
            drop_singletons: false,
            family: Family::Gaussian,
            y_for_sep: None,
            sep_fe: false,
        };
        let mut s_w = Session::prepare_with(p_w, &ctx, AbsorberKind::Dense).unwrap();
        let o_w = ols_fit(&mut s_w, rows_w.clone(), 2, &OlsEngineOpts { vce: vce_kind.clone(), ..OlsEngineOpts::default() }, &ctx).unwrap();

        let mut s_e = make_session(n_exp, Vec::new(), Vec::new(), match vce_kind {
            EngineVce::Hc2 => Vce::Hc2,
            _ => Vce::Hc3,
        });
        let o_e = ols_fit(&mut s_e, rows_exp.clone(), 2, &OlsEngineOpts { vce: vce_kind, ..OlsEngineOpts::default() }, &ctx).unwrap();

        for (a, b) in o_w.beta.iter().zip(&o_e.beta) {
            assert!((a - b).abs() < 1e-12, "beta fw={a} exp={b}");
        }
        assert!((o_w.b0 - o_e.b0).abs() < 1e-12);
        for (a, b) in o_w.v_unscaled.iter().zip(&o_e.v_unscaled) {
            assert!((a - b).abs() < 1e-12, "v fw={a} exp={b}");
        }
    }
}

#[test]
fn ols_few_clusters_missing_f_warning() {
    let ctx = Ctx::quiet(1);
    let n = 120usize;
    let fe: Vec<f64> = (0..n).map(|i| ((i / 6) + 1) as f64).collect();
    let cl: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 3 + 1) as f64).collect();
    let mut rows = Vec::with_capacity(n * 4);
    for i in 0..n {
        let g = (i / 6) as f64;
        let x1 = (i as f64 * 0.37).sin();
        let x2 = (i as f64 * 0.83).cos();
        let x3 = (i as f64 * 1.41).sin();
        let y = 1.0 + x1 + 0.5 * x2 + 0.2 * g + (i as f64 * 2.13).cos() * 0.5;
        rows.extend_from_slice(&[y, x1, x2, x3]);
    }
    let factors = vec![FactorValues::Num(&fe), FactorValues::Num(&cl)];
    let absorb = vec![AbsorbTerm { factors: vec![0], intercept: true, slopes: Vec::new(), save: false }];
    let mut sess = make_session(n, absorb, factors, Vce::Cluster(vec![vec![1]]));
    let res = sess.ols(rows, &Layout::ols(3), &Spec::default(), &ctx).unwrap();
    let f_val = res.scalars.iter().find(|(k, _)| k == "F").unwrap().1;
    assert!(f_val.is_nan(), "expected NaN F, got {f_val}");
    assert!(res.notes.iter().any(|n| n.contains("warning: missing F statistic")), "notes={:?}", res.notes);
}
