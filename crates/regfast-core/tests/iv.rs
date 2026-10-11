//! IV/2SLS (plan Phase 5).
//!
//! Ground truth: ivreghdfe 1.1.4 (reghdfe 6.13.1, ranktest 2.0.04) results at tol(1e-14) with
//! `ffirst savefirst`, saved by `tests/data/gen_iv_fixtures.do`: an nlswork subset
//! (`ln_wage age (tenure = union south [msp])`, absorb(idcode year), × unadjusted / robust /
//! cluster idcode / cluster idcode year × none/aw/fw/pw) and simulated data with two endogenous
//! regressors. b rtol 1e-7, V and every statistic rtol 1e-6, integers exact.
//!
//! Every test runs over [`KINDS`].

use regfast_core::{
    AbsorbTerm, AbsorberKind, Ctx, EstResult, Family, FactorValues, IvOptions, Layout, Prep, RfError, Session, Spec, Vce, WeightKind,
};

const KINDS: &[AbsorberKind] = &[AbsorberKind::Dense];

// ---------------------------------------------------------------------------------------------
// Minimal JSON reader (fixtures are flat objects of numbers, strings, arrays and null).

#[derive(Clone, Debug)]
enum J {
    Null,
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    fn get(&self, k: &str) -> Option<&J> {
        match self {
            J::Obj(v) => v.iter().find(|(a, _)| a == k).map(|(_, b)| b),
            _ => None,
        }
    }
    fn num(&self) -> f64 {
        match self {
            J::Num(x) => *x,
            J::Null => f64::NAN,
            _ => panic!("not a number: {self:?}"),
        }
    }
    fn nums(&self) -> Vec<f64> {
        match self {
            J::Arr(v) => v.iter().map(J::num).collect(),
            _ => panic!("not an array"),
        }
    }
}

fn parse_json(s: &str) -> J {
    fn ws(b: &[u8], i: &mut usize) {
        while *i < b.len() && b[*i].is_ascii_whitespace() {
            *i += 1;
        }
    }
    fn val(b: &[u8], i: &mut usize) -> J {
        ws(b, i);
        match b[*i] {
            b'{' => {
                *i += 1;
                let mut out = Vec::new();
                loop {
                    ws(b, i);
                    if b[*i] == b'}' {
                        *i += 1;
                        return J::Obj(out);
                    }
                    let J::Str(k) = val(b, i) else { panic!("bad key") };
                    ws(b, i);
                    assert_eq!(b[*i], b':');
                    *i += 1;
                    let v = val(b, i);
                    out.push((k, v));
                    ws(b, i);
                    if b[*i] == b',' {
                        *i += 1;
                    }
                }
            }
            b'[' => {
                *i += 1;
                let mut out = Vec::new();
                loop {
                    ws(b, i);
                    if b[*i] == b']' {
                        *i += 1;
                        return J::Arr(out);
                    }
                    out.push(val(b, i));
                    ws(b, i);
                    if b[*i] == b',' {
                        *i += 1;
                    }
                }
            }
            b'"' => {
                *i += 1;
                let st = *i;
                while b[*i] != b'"' {
                    *i += 1;
                }
                *i += 1;
                J::Str(String::from_utf8(b[st..*i - 1].to_vec()).unwrap())
            }
            b'n' => {
                *i += 4;
                J::Null
            }
            _ => {
                let st = *i;
                while *i < b.len() && (b[*i].is_ascii_digit() || b"+-.eE".contains(&b[*i])) {
                    *i += 1;
                }
                J::Num(std::str::from_utf8(&b[st..*i]).unwrap().parse().unwrap())
            }
        }
    }
    let mut i = 0;
    val(s.as_bytes(), &mut i)
}

fn data_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

struct Csv {
    names: Vec<String>,
    cols: Vec<Vec<f64>>,
}

impl Csv {
    fn read(name: &str) -> Csv {
        let text = std::fs::read_to_string(data_dir().join(name)).unwrap();
        let mut lines = text.lines();
        let names: Vec<String> = lines.next().unwrap().split(',').map(str::to_string).collect();
        let mut cols = vec![Vec::new(); names.len()];
        for l in lines.filter(|l| !l.is_empty()) {
            for (j, v) in l.split(',').enumerate() {
                cols[j].push(v.parse::<f64>().unwrap());
            }
        }
        Csv { names, cols }
    }
    fn col(&self, n: &str) -> &[f64] {
        &self.cols[self.names.iter().position(|x| x == n).unwrap_or_else(|| panic!("no column {n}"))]
    }
}

// ---------------------------------------------------------------------------------------------
// Running a model.

struct Model<'a> {
    factors: Vec<&'a [f64]>,
    absorb: Vec<AbsorbTerm>,
    y: &'a [f64],
    exog: Vec<&'a [f64]>,
    endog: Vec<&'a [f64]>,
    inst: Vec<&'a [f64]>,
    weights: Option<(&'a [f64], WeightKind)>,
    vce: Vce,
    spec: Spec,
}

fn term(f: usize) -> AbsorbTerm {
    AbsorbTerm { factors: vec![f], intercept: true, slopes: vec![], save: false }
}

fn run(m: &Model, kind: AbsorberKind, threads: usize) -> (Session, Result<EstResult, RfError>) {
    let prep = Prep {
        n: m.y.len(),
        factors: m.factors.iter().map(|f| FactorValues::Num(f)).collect(),
        absorb: m.absorb.clone(),
        slope_vars: vec![],
        weights: m.weights.map(|w| w.0),
        weight_kind: m.weights.map_or(WeightKind::None, |w| w.1),
        vce: m.vce.clone(),
        drop_singletons: true,
        family: Family::Gaussian,
        y_for_sep: None,
        sep_fe: false,
    };
    let ctx = Ctx::quiet(threads);
    let mut s = Session::prepare_with(prep, &ctx, kind).unwrap();
    let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
    let pick = |v: &[f64]| -> Vec<f64> { kept.iter().map(|&i| v[i]).collect() };
    let cols: Vec<Vec<f64>> =
        std::iter::once(m.y).chain(m.exog.iter().copied()).chain(m.endog.iter().copied()).chain(m.inst.iter().copied()).map(pick).collect();
    let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
    let rows = s.gather(&refs, &ctx).unwrap();
    let lay = Layout { nx: m.exog.len(), nendog: m.endog.len(), ninst: m.inst.len() };
    let res = s.iv(rows, &lay, &m.spec, &ctx);
    (s, res)
}

fn spec() -> Spec {
    Spec { tol: 1e-14, iv: IvOptions { first: true, ffirst: true }, ..Spec::default() }
}

// ---------------------------------------------------------------------------------------------
// Fixture descriptions.

struct Fixture {
    name: String,
    data: &'static str,
    factors: &'static [&'static str],
    absorb: &'static [&'static str],
    y: &'static str,
    exog: &'static [&'static str],
    endog: &'static [&'static str],
    inst: &'static [&'static str],
    weight: Option<(&'static str, WeightKind)>,
    cluster: &'static [&'static str],
    robust: bool,
}

fn fixtures() -> Vec<Fixture> {
    let mut out = Vec::new();
    let models: [(&str, &[&str]); 3] = [("a", &["union", "south"]), ("b", &["union", "south", "msp"]), ("j", &["union"])];
    for (tag, inst) in models {
        for v in ["unadj", "robust", "cl1", "cl2"] {
            for wt in ["none", "aw", "fw", "pw"] {
                if tag == "j" && wt != "none" {
                    continue;
                }
                out.push(Fixture {
                    name: format!("nls_{tag}_{v}_{wt}"),
                    data: "iv_nls.csv",
                    factors: &["idcode", "year"],
                    absorb: &["idcode", "year"],
                    y: "ln_wage",
                    exog: &["age"],
                    endog: &["tenure"],
                    inst,
                    weight: match wt {
                        "aw" => Some(("w", WeightKind::Aweight)),
                        "fw" => Some(("fw", WeightKind::Fweight)),
                        "pw" => Some(("w", WeightKind::Pweight)),
                        _ => None,
                    },
                    cluster: match v {
                        "cl1" => &["idcode"],
                        "cl2" => &["idcode", "year"],
                        _ => &[],
                    },
                    robust: v == "robust",
                });
            }
        }
    }
    let sim = |name: &str, absorb: &'static [&'static str], exog: &'static [&'static str], endog: &'static [&'static str], inst: &'static [&'static str]| Fixture {
        name: name.to_string(),
        data: "iv_sim.csv",
        factors: &["id", "year", "c1", "c2"],
        absorb,
        y: "y",
        exog,
        endog,
        inst,
        weight: None,
        cluster: &[],
        robust: false,
    };
    let idyear: &[&str] = &["id", "year"];
    let z4: &[&str] = &["z1", "z2", "z3", "z4"];
    let e12: &[&str] = &["e1", "e2"];
    let x1: &[&str] = &["x1"];
    out.push(sim("sim_k2_unadj", idyear, x1, e12, z4));
    out.push(Fixture { robust: true, ..sim("sim_k2_robust", idyear, x1, e12, z4) });
    out.push(Fixture { cluster: &["c1"], weight: Some(("w", WeightKind::Aweight)), ..sim("sim_k2_aw_cl1", idyear, x1, e12, z4) });
    out.push(Fixture { cluster: &["c1", "c2"], weight: Some(("w", WeightKind::Pweight)), ..sim("sim_k2_pw_cl2", idyear, x1, e12, z4) });
    out.push(Fixture { robust: true, ..sim("sim_k2_noexog_robust", idyear, &[], e12, &["z1", "z2", "z3"]) });
    out.push(Fixture { cluster: &["c1"], ..sim("sim_k2_justid_cl1", idyear, x1, e12, &["z1", "z2"]) });
    out.push(Fixture { cluster: &["c1"], ..sim("sim_k1_1fe_cl1", &["id"], x1, &["e1"], &["z1", "z2", "z3"]) });
    out.push(Fixture { cluster: &["id"], ..sim("sim_k1_cl_eq_fe", idyear, x1, &["e1"], &["z1", "z2", "z3"]) });
    out.push(Fixture { cluster: &["id"], ..sim("sim_k1_dfa0", &["id"], x1, &["e1"], &["z1", "z2", "z3"]) });
    out
}

const INT_SCALARS: &[&str] = &[
    "N", "df_r", "df_m", "sdofminus", "dofminus", "center", "partial_ct", "partialcons", "cons", "nocollin", "inexog_ct", "exexog_ct",
    "endog_ct", "Fdf1", "Fdf2", "rankxx", "rankzz", "rankS", "rankV", "jdf", "sargandf", "N_clust", "N_clust1", "N_clust2", "df_a",
    "df_a_initial", "df_a_redundant", "df_a_nested", "num_singletons", "N_full", "iddf", "ardf", "ardf_r", "sstatdf",
];
/// Posted by the ado (`N_hdfe`, `N_hdfe_extended`, `drop_singletons`).
const SKIP_SCALARS: &[&str] = &["N_hdfe", "N_hdfe_extended", "drop_singletons"];

fn rel_err(a: f64, b: f64, atol: f64) -> f64 {
    if a.is_nan() && b.is_nan() {
        return 0.0;
    }
    let d = (a - b).abs();
    if d <= atol { 0.0 } else { d / a.abs().max(b.abs()) }
}

#[derive(Default)]
struct MaxErr {
    b: f64,
    v: f64,
    stats: f64,
}

/// Max relative error of `got` vs `want` with an absolute floor of 1e-10·max|want| (entries that
/// are zero up to rounding, e.g. tiny p-values next to large F statistics).
fn vec_err(what: &str, got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().filter(|v| v.is_finite()).fold(0.0f64, |m, v| m.max(v.abs()));
    got.iter().zip(want).map(|(a, b)| rel_err(*a, *b, 1e-10 * scale)).fold(0.0, f64::max)
}

fn check_fixture(f: &Fixture, kind: AbsorberKind, max: &mut MaxErr) {
    let data = Csv::read(f.data);
    let text = std::fs::read_to_string(data_dir().join(format!("iv_{}.json", f.name))).unwrap();
    let j = parse_json(&text);
    let fidx = |n: &str| f.factors.iter().position(|x| *x == n).unwrap();
    let m = Model {
        factors: f.factors.iter().map(|n| data.col(n)).collect(),
        absorb: f.absorb.iter().map(|n| term(fidx(n))).collect(),
        y: data.col(f.y),
        exog: f.exog.iter().map(|n| data.col(n)).collect(),
        endog: f.endog.iter().map(|n| data.col(n)).collect(),
        inst: f.inst.iter().map(|n| data.col(n)).collect(),
        weights: f.weight.map(|(w, k)| (data.col(w), k)),
        vce: if !f.cluster.is_empty() {
            Vce::Cluster(f.cluster.iter().map(|n| vec![fidx(n)]).collect())
        } else if f.robust {
            Vce::Robust
        } else {
            Vce::Unadjusted
        },
        spec: spec(),
    };
    let (s, res) = run(&m, kind, 4);
    let res = res.unwrap_or_else(|e| panic!("{}: {e}", f.name));
    let what = format!("{} [{kind:?}]", f.name);

    let sample = j.get("sample").unwrap().nums();
    let keep: Vec<f64> = s.keep().iter().map(|&k| f64::from(u8::from(k))).collect();
    assert_eq!(keep, sample, "{what}: e(sample)");
    assert!(res.status.iter().all(|&st| st == 0), "{what}: status {:?}", res.status);

    let mat = |k: &str| &res.matrix(k).unwrap_or_else(|| panic!("{what}: missing matrix {k}")).data;
    // Coefficients are compared without an absolute floor.
    let eb = mat("b").iter().zip(&j.get("b").unwrap().nums()).map(|(a, b)| rel_err(*a, *b, 0.0)).fold(0.0, f64::max);
    assert!(eb <= 1e-7, "{what}: b rel err {eb:e}");
    max.b = max.b.max(eb);
    let ev = vec_err(&format!("{what}: V"), mat("V"), &j.get("V").unwrap().nums());
    assert!(ev <= 1e-6, "{what}: V rel err {ev:e}");
    max.v = max.v.max(ev);
    assert_eq!(mat("dof_table"), &j.get("dof_table").unwrap().nums(), "{what}: dof_table");
    let mut keys = vec!["S", "W", "first", "ccev", "cdev"];
    let k1 = f.endog.len();
    let first_keys: Vec<String> = (1..=k1).flat_map(|i| [format!("bfirst{i}"), format!("Vfirst{i}"), format!("sfirst{i}")]).collect();
    keys.extend(first_keys.iter().map(String::as_str));
    for k in keys {
        let want = j.get(k).unwrap_or_else(|| panic!("{what}: fixture lacks {k}")).nums();
        let e = vec_err(&format!("{what}: {k}"), mat(k), &want);
        assert!(e <= 1e-6, "{what}: {k} rel err {e:e}\n got {:?}\nwant {want:?}", mat(k));
        max.stats = max.stats.max(e);
    }

    let J::Obj(scal) = j.get("scalars").unwrap() else { panic!() };
    for (name, val) in scal {
        if SKIP_SCALARS.contains(&name.as_str()) {
            continue;
        }
        let want = val.num();
        let got = res.scalar(name).unwrap_or_else(|| panic!("{what}: scalar {name} not produced"));
        if INT_SCALARS.contains(&name.as_str()) {
            assert!(got == want || (got.is_nan() && want.is_nan()), "{what}: {name} got {got} want {want}");
        } else {
            let e = rel_err(got, want, 1e-300);
            assert!(e <= 1e-6, "{what}: {name} got {got:e} want {want:e} (rel err {e:e})");
            max.stats = max.stats.max(e);
        }
    }
    for (name, _) in &res.scalars {
        // `ic` (solver iterations) feeds the ado's convergence message only.
        assert!(name == "ic" || scal.iter().any(|(k, _)| k == name), "{what}: produced {name} which ivreghdfe does not post");
    }
}

#[test]
fn b_ivreghdfe_ground_truth() {
    for &kind in KINDS {
        let mut max = MaxErr::default();
        let fx = fixtures();
        for f in &fx {
            check_fixture(f, kind, &mut max);
        }
        eprintln!("[{kind:?}] {} ivreghdfe fixtures: max rel err b {:.2e}, V {:.2e}, statistics {:.2e}", fx.len(), max.b, max.v, max.stats);
    }
}

// ---------------------------------------------------------------------------------------------
// Errors, Stock–Yogo macros, collinearity and thread invariance.

fn sim_model<'a>(d: &'a Csv, endog: &[&str], inst: &[&str], vce: Vce) -> Model<'a> {
    Model {
        factors: vec![d.col("id"), d.col("year"), d.col("c1"), d.col("c2")],
        absorb: vec![term(0), term(1)],
        y: d.col("y"),
        exog: vec![d.col("x1")],
        endog: endog.iter().map(|n| d.col(n)).collect(),
        inst: inst.iter().map(|n| d.col(n)).collect(),
        weights: None,
        vce,
        spec: spec(),
    }
}

#[test]
fn a_errors() {
    let d = Csv::read("iv_sim.csv");
    for &kind in KINDS {
        let m = sim_model(&d, &["e1", "e2"], &["z1"], Vce::Unadjusted);
        let err = run(&m, kind, 2).1.unwrap_err();
        assert_eq!(err.rc(), 481, "{err}");
        let m = sim_model(&d, &["e1"], &["z1", "z2"], Vce::Cluster(vec![vec![0], vec![2], vec![3]]));
        let err = run(&m, kind, 2).1.unwrap_err();
        assert_eq!(err.rc(), 198, "{err}");
    }
}

#[test]
fn a_stock_yogo_and_collinear_instrument() {
    let d = Csv::read("iv_sim.csv");
    let z_dup: Vec<f64> = d.col("z1").iter().zip(d.col("z2")).map(|(a, b)| a + 2.0 * b).collect();
    for &kind in KINDS {
        let m = sim_model(&d, &["e1"], &["z1", "z2", "z3"], Vce::Robust);
        let res = run(&m, kind, 2).1.unwrap();
        // L1 = 3, K1 = 1: ivreg2 shows IV bias 5% = 13.91 and IV size 10% = 22.30.
        assert_eq!(res.macro_value("sy_ivbias5"), Some("13.91"));
        assert_eq!(res.macro_value("sy_ivsize10"), Some("22.3"));
        // A linear combination of instruments is dropped as collinear; results equal the model without it.
        let mut m2 = sim_model(&d, &["e1"], &["z1", "z2", "z3"], Vce::Robust);
        m2.inst.push(&z_dup);
        let res2 = run(&m2, kind, 2).1.unwrap();
        assert_eq!(res2.status, vec![0, 0, 0, 0, 0, 3]);
        let (b1, b2) = (&res.matrix("b").unwrap().data, &res2.matrix("b").unwrap().data);
        for (a, b) in b1.iter().zip(b2) {
            assert!((a - b).abs() <= 1e-9 * a.abs().max(1.0), "{a} vs {b}");
        }
        assert_eq!(res2.scalar("exexog_ct"), Some(3.0));

        // An endogenous regressor collinear with the exogenous one is dropped: K1 = 1, e(first)
        // keeps a missing column for it and only bfirst1 is produced; b equals the K1 = 1 model.
        let x1_twice: Vec<f64> = d.col("x1").iter().map(|v| 2.0 * v).collect();
        let mut m3 = sim_model(&d, &["e1"], &["z1", "z2", "z3"], Vce::Robust);
        m3.endog.push(&x1_twice);
        let res3 = run(&m3, kind, 2).1.unwrap();
        assert_eq!(res3.status, vec![0, 0, 3, 0, 0, 0]);
        assert_eq!(res3.scalar("endog_ct"), Some(1.0));
        let b3 = &res3.matrix("b").unwrap().data;
        assert_eq!(b3.len(), 3);
        assert!((b3[0] - b1[0]).abs() <= 1e-9 * b1[0].abs() && b3[1] == 0.0 && (b3[2] - b1[1]).abs() <= 1e-9 * b1[1].abs());
        let first = &res3.matrix("first").unwrap().data;
        assert!(first[0].is_finite() && first[1].is_nan());
        assert!(res3.matrix("bfirst1").is_some() && res3.matrix("bfirst2").is_none());
        assert!(res3.matrix("ccev").unwrap().data[1].is_nan());
    }
}

#[test]
fn a_thread_count_invariance() {
    let d = Csv::read("iv_sim.csv");
    // The dense oracle factorizes with faer inside the pool and is not thread-count invariant.
    for &kind in KINDS.iter().filter(|k| **k != AbsorberKind::Dense) {
        let m = sim_model(&d, &["e1", "e2"], &["z1", "z2", "z3", "z4"], Vce::Cluster(vec![vec![2], vec![3]]));
        let a = run(&m, kind, 1).1.unwrap();
        let b = run(&m, kind, 5).1.unwrap();
        assert_eq!(a, b, "results differ between 1 and 5 threads");
    }
}
