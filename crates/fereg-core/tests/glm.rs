//! GLM (plan Phase 6): ground truth from Stata, saved by `tests/data/gen_glm_fixtures.do`.
//!
//! - Poisson: ppmlhdfe 2.3.3 on the ppmlhdfe test-suite datasets plus a simulated gravity panel
//!   (`glm_p_*.json`): N, sample, separated/singleton counts, b (rtol 1e-7), V (rtol 1e-6),
//!   ll, ll_0, deviance, r2_p, chi2, dof, and d / saved-FE sums where requested.
//! - Logit: Stata `logit y x i.fe1 i.fe2` (`glm_l_*.json`): N, sample, β and V_ββ, ll, ll_0, r2_p.
//! - fixest convention and [`GlmParts`]: fixest 0.14.2 (`glm_fx_*.json`, see that section).
//!
//! Every test runs over [`KINDS`]. `cargo test --test glm -- --nocapture` prints the largest
//! observed relative error per fixture.

use fereg_core::{
    AbsorbTerm, AbsorberKind, Ctx, EstResult, Family, FactorValues, GlmConvention, GlmOptions, GlmParts, Guess, Layout, Prep, RfError, Separation,
    Session, Spec, Vce, WeightKind,
};

const KINDS: &[AbsorberKind] = &[AbsorberKind::Dense, AbsorberKind::Fast];

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
    fn str(&self) -> &str {
        match self {
            J::Str(s) => s,
            _ => panic!("not a string"),
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

fn fixture(name: &str) -> J {
    parse_json(&std::fs::read_to_string(data_dir().join(format!("glm_{name}.json"))).unwrap())
}

struct Csv {
    names: Vec<String>,
    cols: Vec<Vec<f64>>,
}

impl Csv {
    fn read(name: &str) -> Csv {
        let text = std::fs::read_to_string(data_dir().join(format!("glm_{name}.csv"))).unwrap();
        let mut lines = text.lines();
        let names: Vec<String> = lines.next().unwrap().split(',').map(str::to_string).collect();
        let mut cols = vec![Vec::new(); names.len()];
        for l in lines.filter(|l| !l.is_empty()) {
            for (j, v) in l.split(',').enumerate() {
                cols[j].push(if v == "null" { f64::NAN } else { v.parse::<f64>().unwrap() });
            }
        }
        Csv { names, cols }
    }
    fn col(&self, n: &str) -> &[f64] {
        &self.cols[self.names.iter().position(|x| x == n).unwrap_or_else(|| panic!("no column {n}"))]
    }
}

// ---------------------------------------------------------------------------------------------
// Comparison helpers: relative error with an absolute floor tied to the scale of the reference.

#[derive(Default)]
struct Errs {
    worst: Vec<(String, f64)>,
}

impl Errs {
    fn vec(&mut self, what: &str, got: &[f64], want: &[f64], rtol: f64) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let scale = want.iter().filter(|v| v.is_finite()).fold(0.0f64, |m, v| m.max(v.abs()));
        let atol = 1e-10 * scale + 1e-14;
        let mut worst = 0.0f64;
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            let err = (a - b).abs() / (atol / rtol + a.abs().max(b.abs()));
            worst = worst.max(err);
            assert!(err <= rtol, "{what}[{i}]: got {a:e}, want {b:e} (rel {err:e})");
        }
        self.worst.push((what.to_string(), worst));
    }
    fn scalar(&mut self, what: &str, got: f64, want: f64, rtol: f64) {
        self.vec(what, &[got], &[want], rtol);
    }
    /// |got − want| ≤ tol·max|want| (for vectors such as d whose entries can be near zero).
    fn vec_scaled(&mut self, what: &str, got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
        let mut worst = 0.0f64;
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            let err = (a - b).abs() / scale;
            worst = worst.max(err);
            assert!(err <= tol, "{what}[{i}]: got {a:e}, want {b:e} (err/scale {err:e})");
        }
        self.worst.push((format!("{what} (scaled)"), worst));
    }
    fn report(&self, name: &str) {
        let (w, e) = self.worst.iter().fold(("", 0.0f64), |acc, (k, v)| if *v > acc.1 { (k.as_str(), *v) } else { acc });
        println!("{name:28} max rel err {e:9.2e} ({w})");
    }
}

fn scalar(r: &EstResult, k: &str) -> f64 {
    r.scalar(k).unwrap_or_else(|| panic!("missing scalar {k}"))
}

fn mat<'a>(r: &'a EstResult, k: &str) -> &'a [f64] {
    &r.matrix(k).unwrap_or_else(|| panic!("missing matrix {k}")).data
}

// ---------------------------------------------------------------------------------------------
// Specs.

#[derive(Clone, Copy)]
enum Off {
    None,
    Offset(&'static str),
}

#[derive(Clone)]
struct G {
    fixture: &'static str,
    data: &'static str,
    family: Family,
    y: &'static str,
    x: Vec<&'static str>,
    /// Absorb terms (each an interaction of factor columns); `save` flags per term.
    absorb: Vec<(Vec<&'static str>, bool)>,
    /// `None` = default VCE (Poisson robust, logit oim).
    cluster: Option<Vec<Vec<&'static str>>>,
    robust: bool,
    weights: Option<(&'static str, WeightKind)>,
    off: Off,
    singletons: bool,
    sep_fe: bool,
    relu: bool,
    mu: bool,
    tol: f64,
    maxiter: usize,
    guess: Guess,
    /// First input observation used (0-based), for `if _n > k` samples.
    first_obs: usize,
    /// Fixture variable holding d, compared with our d.
    d_var: Option<&'static str>,
    /// Fixture variables holding saved FEs, whose sum is compared with the sum of ours.
    fe_vars: Vec<&'static str>,
    /// Expected (num_separated, num_singletons) when the fixture was produced on a pre-trimmed
    /// sample (N_full is then N + both).
    counts: Option<(f64, f64)>,
    /// Scalars not compared, with the reason in the spec list.
    skip: Vec<&'static str>,
    /// Expected values replacing ppmlhdfe's when its reported scalar is inexact.
    exact: Vec<(&'static str, f64)>,
    /// Tolerance of d relative to max|d|.
    d_tol: f64,
    /// Relative tolerance of V (1e-6 unless the design is ill-conditioned).
    v_tol: f64,
}

fn poisson(fixture: &'static str, data: &'static str, y: &'static str, x: &[&'static str], absorb: &[&'static str]) -> G {
    G {
        fixture,
        data,
        family: Family::Poisson,
        y,
        x: x.to_vec(),
        absorb: absorb.iter().map(|a| (vec![*a], false)).collect(),
        cluster: None,
        robust: false,
        weights: None,
        off: Off::None,
        singletons: true,
        sep_fe: true,
        relu: true,
        mu: false,
        tol: 1e-10,
        maxiter: 10_000,
        guess: Guess::Simple,
        first_obs: 0,
        d_var: None,
        fe_vars: Vec::new(),
        counts: None,
        skip: Vec::new(),
        exact: Vec::new(),
        d_tol: 1e-6,
        v_tol: 1e-6,
    }
}

impl G {
    /// ppmlhdfe `keepsingletons`: no singleton dropping and no `fe` separation.
    fn keepsing(mut self) -> G {
        self.singletons = false;
        self.sep_fe = false;
        self
    }
    fn tol(mut self, t: f64) -> G {
        self.tol = t;
        self
    }
    fn cluster(mut self, c: &[&[&'static str]]) -> G {
        self.cluster = Some(c.iter().map(|d| d.to_vec()).collect());
        self
    }
    fn w(mut self, v: &'static str, k: WeightKind) -> G {
        self.weights = Some((v, k));
        self
    }
    fn offset(mut self, v: &'static str) -> G {
        self.off = Off::Offset(v);
        self
    }
}

fn logit(fixture: &'static str, y: &'static str, absorb: &[&'static str]) -> G {
    let mut g = poisson(fixture, "logit", y, &["x1", "x2"], absorb);
    g.family = Family::Logit;
    g.relu = false;
    g.tol = 1e-12;
    g
}

struct Run {
    session: Session,
    res: EstResult,
    /// Input observations (CSV rows) used.
    obs: Vec<usize>,
}

/// Session, data block, offset and options of spec `g` (everything [`Session::glm`] needs).
struct Setup {
    session: Session,
    rows: Vec<f64>,
    offset: Option<Vec<f64>>,
    spec: Spec,
    opts: GlmOptions,
    obs: Vec<usize>,
}

fn setup(g: &G, csv: &Csv, kind: AbsorberKind, threads: usize) -> Result<Setup, RfError> {
    let nall = csv.col(g.y).len();
    let obs: Vec<usize> = (g.first_obs..nall).collect();
    let pick = |name: &str| -> Vec<f64> { obs.iter().map(|&i| csv.col(name)[i]).collect() };
    // Factor columns: every variable named in absorb terms and cluster dimensions.
    let mut fnames: Vec<&str> = Vec::new();
    for (t, _) in &g.absorb {
        for f in t {
            if !fnames.contains(f) {
                fnames.push(f);
            }
        }
    }
    for d in g.cluster.iter().flatten() {
        for f in d {
            if !fnames.contains(f) {
                fnames.push(f);
            }
        }
    }
    let fcols: Vec<Vec<f64>> = fnames.iter().map(|f| pick(f)).collect();
    let idx = |f: &str| fnames.iter().position(|x| *x == f).unwrap();
    let absorb: Vec<AbsorbTerm> = g
        .absorb
        .iter()
        .map(|(t, save)| AbsorbTerm { factors: t.iter().map(|f| idx(f)).collect(), intercept: true, slopes: vec![], save: *save })
        .collect();
    let vce = match &g.cluster {
        Some(c) => Vce::Cluster(c.iter().map(|d| d.iter().map(|f| idx(f)).collect()).collect()),
        None if g.robust => Vce::Robust,
        None => Vce::Unadjusted,
    };
    let y = pick(g.y);
    let w = g.weights.map(|(v, _)| pick(v));
    let prep = Prep {
        n: obs.len(),
        factors: fcols.iter().map(|c| FactorValues::Num(c)).collect(),
        absorb,
        slope_vars: vec![],
        weights: w.as_deref(),
        weight_kind: g.weights.map_or(WeightKind::None, |w| w.1),
        vce,
        drop_singletons: g.singletons,
        family: g.family,
        y_for_sep: Some(&y),
        sep_fe: g.sep_fe,
    };
    let ctx = Ctx::quiet(threads);
    let session = Session::prepare_with(prep, &ctx, kind)?;
    let kept: Vec<usize> = session.kept_index().iter().map(|&i| i as usize).collect();
    let kc = |v: &[f64]| -> Vec<f64> { kept.iter().map(|&i| v[i]).collect() };
    let cols: Vec<Vec<f64>> = std::iter::once(kc(&y)).chain(g.x.iter().map(|x| kc(&pick(x)))).collect();
    let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
    let rows = session.gather(&refs, &ctx)?;
    let offset = match g.off {
        Off::None => None,
        Off::Offset(v) => Some(session.gather(&[&kc(&pick(v))], &ctx)?),
    };
    let spec = Spec { want_d: g.d_var.is_some() || !g.fe_vars.is_empty(), want_fe: !g.fe_vars.is_empty(), ..Spec::default() };
    let opts = GlmOptions {
        family: g.family,
        tol: g.tol,
        maxiter: g.maxiter,
        itol: 1e-9,
        separation: Separation { relu: g.relu, mu: g.mu },
        guess: g.guess,
        log: false,
    };
    Ok(Setup { session, rows, offset, spec, opts, obs })
}

fn run(g: &G, csv: &Csv, kind: AbsorberKind, threads: usize) -> Result<Run, RfError> {
    let Setup { mut session, rows, offset, spec, opts, obs } = setup(g, csv, kind, threads)?;
    let res = session.glm(rows, &Layout::ols(g.x.len()), offset, &spec, &opts, &Ctx::quiet(threads))?;
    Ok(Run { session, res, obs })
}

/// Fixture sample restricted to the observations used, vs the session's keep mask.
fn check_sample(g: &G, run: &Run, fx: &J) {
    let want = fx.get("sample").unwrap().nums();
    let got: Vec<f64> = run.session.keep().iter().map(|&k| f64::from(u8::from(k))).collect();
    let want_used: Vec<f64> = run.obs.iter().map(|&i| want[i]).collect();
    assert_eq!(got, want_used, "{}: e(sample)", g.fixture);
    // Observations outside the used range must be out of the Stata sample too.
    assert!((0..g.first_obs).all(|i| want[i] == 0.0));
}

fn check_poisson(g: &G, kind: AbsorberKind) {
    let csv = Csv::read(g.data);
    let fx = fixture(g.fixture);
    let run = run(g, &csv, kind, 3).unwrap_or_else(|e| panic!("{}: {e}", g.fixture));
    let sc = |k: &str| fx.get("scalars").unwrap().get(k).unwrap_or_else(|| panic!("{}: fixture lacks {k}", g.fixture)).num();
    let r = &run.res;
    let mut e = Errs::default();
    check_sample(g, &run, &fx);
    for k in ["N", "rank", "df_m", "df", "df_a"] {
        assert_eq!(scalar(r, k), sc(k), "{}: e({k})", g.fixture);
    }
    let (nsep, nsing) = g.counts.unwrap_or_else(|| (sc("num_separated"), sc("num_singletons")));
    assert_eq!(scalar(r, "num_separated"), nsep, "{}: e(num_separated)", g.fixture);
    assert_eq!(scalar(r, "num_singletons"), nsing, "{}: e(num_singletons)", g.fixture);
    assert_eq!(scalar(r, "N_full"), sc("N") + nsep + nsing, "{}: e(N_full)", g.fixture);
    if fx.get("scalars").unwrap().get("N_clust").is_some() {
        assert_eq!(scalar(r, "N_clust"), sc("N_clust"), "{}: e(N_clust)", g.fixture);
    }
    let colnames = fx.get("colnames").unwrap().str().to_string();
    let omitted: Vec<bool> = colnames.split_whitespace().map(|c| c.starts_with("o.") || c.contains("o.")).collect();
    for (j, &o) in omitted.iter().take(g.x.len()).enumerate() {
        assert_eq!(r.status[j] != 0, o, "{}: omitted status of {}", g.fixture, g.x[j]);
    }
    e.vec(&format!("{} b", g.fixture), mat(r, "b"), &fx.get("b").unwrap().nums(), 1e-7);
    e.vec(&format!("{} V", g.fixture), mat(r, "V"), &fx.get("V").unwrap().nums(), g.v_tol);
    let want = |k: &str| g.exact.iter().find(|(n, _)| *n == k).map_or_else(|| sc(k), |(_, v)| *v);
    for k in ["ll", "ll_0", "deviance", "chi2", "r2_p", "rss", "rmse"] {
        if !g.skip.contains(&k) {
            e.scalar(k, scalar(r, k), want(k), 1e-6);
        }
    }
    if let Some(dv) = g.d_var {
        let want: Vec<f64> = run.obs.iter().map(|&i| fx.get(dv).unwrap().nums()[i]).filter(|v| !v.is_nan()).collect();
        e.vec_scaled("d", r.d.as_deref().unwrap(), &want, g.d_tol);
        // Our d solves the score equations of every FE: Σ_{i∈g} w_i(y_i − μ_i) = 0 with
        // μ = exp(x'b + _cons + d + offset).
        let viol = score_violation(g, &run, &csv);
        assert!(viol <= 1e-9, "{}: FE score equations violated by {viol:e}", g.fixture);
        e.worst.push(("FE score equations".into(), viol));
    }
    if !g.fe_vars.is_empty() {
        let n = run.session.n();
        let kept: Vec<usize> = run.session.kept_index().iter().map(|&i| run.obs[i as usize]).collect();
        let want: Vec<f64> =
            kept.iter().map(|&i| g.fe_vars.iter().map(|v| fx.get(v).unwrap().nums()[i]).sum::<f64>()).collect();
        let got: Vec<f64> = (0..n).map(|i| r.fe.iter().map(|f| f[i]).sum::<f64>()).collect();
        e.vec_scaled("sum of saved FEs", &got, &want, 1e-6);
        // Σ saved FEs = d when every term is saved.
        e.vec("d vs saved FEs", r.d.as_deref().unwrap(), &got, 1e-9);
    }
    e.report(g.fixture);
}

/// Largest violation of the FE score equations Σ_{i∈g} w_i(y_i − μ_i) = 0 (relative to
/// max(Σ w y, 1)), with μ = exp(x'b + _cons + d + offset) from our reported b and d.
fn score_violation(g: &G, run: &Run, csv: &Csv) -> f64 {
    let r = &run.res;
    let b = mat(r, "b");
    let k = g.x.len();
    let d = r.d.as_deref().unwrap();
    let rows: Vec<usize> = run.session.kept_index().iter().map(|&i| run.obs[i as usize]).collect();
    let y = csv.col(g.y);
    let mut worst = 0.0f64;
    for t in run.session.terms() {
        let mut s = vec![0.0; t.g];
        let mut tot = vec![0.0; t.g];
        for (kk, &i) in rows.iter().enumerate() {
            let mut eta = b[k] + d[kk];
            for (j, x) in g.x.iter().enumerate() {
                eta += b[j] * csv.col(x)[i];
            }
            if let Off::Offset(v) = g.off {
                eta += csv.col(v)[i];
            }
            let w = g.weights.map_or(1.0, |(v, _)| csv.col(v)[i]);
            s[t.codes[kk] as usize] += w * (y[i] - eta.exp());
            tot[t.codes[kk] as usize] += w * y[i];
        }
        for (a, b) in s.iter().zip(&tot) {
            worst = worst.max(a.abs() / b.max(1.0));
        }
    }
    worst
}

fn check_logit(g: &G, kind: AbsorberKind) {
    let csv = Csv::read(g.data);
    let fx = fixture(g.fixture);
    let run = run(g, &csv, kind, 3).unwrap_or_else(|e| panic!("{}: {e}", g.fixture));
    let sc = |k: &str| fx.get("scalars").unwrap().get(k).unwrap().num();
    let r = &run.res;
    let mut e = Errs::default();
    check_sample(g, &run, &fx);
    assert_eq!(scalar(r, "N"), sc("N"), "{}: e(N)", g.fixture);
    let kx = g.x.len();
    let bw = fx.get("b").unwrap().nums();
    let vw = fx.get("V").unwrap().nums();
    let p = bw.len();
    let b = mat(r, "b");
    let v = mat(r, "V");
    let k_all = kx + 1;
    if g.absorb.is_empty() {
        // Without fixed effects the constant is Stata's.
        e.vec(&format!("{} b", g.fixture), b, &bw, 1e-7);
        e.vec(&format!("{} V", g.fixture), v, &vw, 1e-6);
    } else {
        e.vec(&format!("{} b", g.fixture), &b[..kx], &bw[..kx], 1e-7);
        let vb: Vec<f64> = (0..kx * kx).map(|ab| v[(ab / kx) * k_all + ab % kx]).collect();
        let vwb: Vec<f64> = (0..kx * kx).map(|ab| vw[(ab / kx) * p + ab % kx]).collect();
        e.vec(&format!("{} V", g.fixture), &vb, &vwb, 1e-6);
    }
    for k in ["ll", "ll_0", "r2_p"] {
        e.scalar(k, scalar(r, k), sc(k), 1e-7);
    }
    e.scalar("deviance", scalar(r, "deviance"), -2.0 * sc("ll"), 1e-7);
    e.report(g.fixture);
}

fn poisson_specs() -> Vec<G> {
    let mut v = vec![
        {
            let mut g = poisson("p_problem1_keepsing_mu", "problem1", "y", &["x1", "x2"], &["i", "j"]).keepsing();
            g.relu = false;
            g.mu = true;
            g
        },
        {
            // Fixture: ppmlhdfe on the sample its own test expects (its ReLU check gives up on
            // the full data); ReLU here drops 4 separated obs, then 1 new singleton.
            let mut g = poisson("p_problem1", "problem1", "y", &["x1", "x2"], &["i", "j"]);
            g.counts = Some((4.0, 1.0));
            g
        },
        poisson("p_problem2", "problem2", "y", &["x1", "x2"], &["i", "j"]),
        poisson("p_problem3", "problem3", "y", &["x1", "x2", "x3"], &[]).tol(1e-12),
        {
            let mut g = poisson("p_problem4", "problem4", "y", &["x1", "x2", "x3"], &[]).tol(1e-12);
            g.maxiter = 200;
            g
        },
        poisson("p_problem5", "problem5", "y", &["x1", "x2"], &["i", "j"]).tol(1e-12),
        {
            // ppmlhdfe's e(ll) and e(deviance) are off by 4e-3 and 8e-3 here (the sample has
            // one observation with x = −716). The values below are the Poisson ll and deviance
            // recomputed in double precision from ppmlhdfe's own e(b); r2_p is not compared.
            let mut g = poisson("p_problem6", "problem6", "y", &["x"], &[]);
            g.exact = vec![("ll", -1486.4665049560613), ("deviance", 2136.1091000805777)];
            g.skip = vec!["r2_p"];
            g
        },
        poisson("p_hard1", "hard1", "y", &[], &["id1", "id2"]),
        poisson("p_hard2_keepsing", "hard2", "FTA", &[], &["ijk_fe", "jkt_fe", "ikt_fe"]).keepsing().tol(1e-12),
        poisson("p_hard3", "hard3", "y", &[], &["id1", "id2"]),
        poisson("p_hard5", "hard5", "y", &["x1", "x2", "x3", "x4"], &[]),
        poisson("p_hard5b", "hard5", "y", &["x2", "x3", "x4"], &[]),
        {
            let mut g = poisson("p_hard4", "hard5", "y", &["x2", "x3", "x4"], &[]);
            g.first_obs = 2;
            g
        },
        poisson("p_collinear1", "collinear1", "y", &["x", "z"], &["im", "ex"]),
        poisson("p_collinear2", "collinear2", "y", &["x1", "x2"], &[]),
        poisson("p_collinear7", "collinear2", "y", &["x1"], &[]),
        poisson("p_collinear3", "collinear3", "y", &["x1", "x2"], &[]),
        poisson("p_collinear4", "collinear4", "y", &["x1", "x2"], &[]),
        poisson("p_collinear5", "collinear5", "y", &["x1", "x2"], &[]),
        {
            // x1 and x2 are nearly collinear (x2 = x1 + e/18000): ppmlhdfe finds V numerically
            // singular and reports a missing chi2 (whether the sweep flags it is rounding noise),
            // and V (entries ≈ 1e6) agrees with ppmlhdfe only to 9.7e-7 (Dense) / 1.6e-6 (Fast).
            let mut g = poisson("p_collinear6", "collinear6", "y", &["x1", "x2"], &[]).tol(1e-12);
            g.skip = vec!["chi2"];
            g.v_tol = 1e-5;
            g
        },
    ];
    let auto = |f: &'static str, absorb: &[&'static str]| poisson(f, "auto", "price", &["weight", "gear_ratio", "length"], absorb);
    v.push({
        let mut g = auto("p_twoway_robust", &["turn", "trunk"]).keepsing().tol(1e-12);
        g.robust = true;
        g
    });
    v.push(auto("p_twoway_cluster2", &["turn", "trunk"]).keepsing().tol(1e-12).cluster(&[&["turn"], &["trunk"]]));
    v.push(auto("p_twoway_cluster_turn", &["turn", "trunk"]).tol(1e-12).cluster(&[&["turn"]]));
    v.push(auto("p_fw_noa", &[]).w("displacement", WeightKind::Fweight));
    v.push(auto("p_fw_cluster", &[]).w("displacement", WeightKind::Fweight).cluster(&[&["turn"]]));
    v.push(auto("p_fw_turn_cluster", &[]).w("turn", WeightKind::Fweight).cluster(&[&["trunk"]]));
    v.push(auto("p_fw_absorb", &["turn"]).w("displacement", WeightKind::Fweight).keepsing());
    v.push(auto("p_pw_noa", &[]).w("displacement", WeightKind::Pweight));
    v.push(auto("p_pw_cluster", &[]).w("displacement", WeightKind::Pweight).cluster(&[&["turn"]]));
    v.push(auto("p_pw_absorb", &["turn"]).w("displacement", WeightKind::Pweight).keepsing());
    v.push(auto("p_exposure_noa", &[]).offset("lnhead").tol(1e-12));
    v.push(auto("p_exposure_absorb", &["turn"]).offset("lnhead").keepsing());
    v.push(auto("p_offset_noa", &[]).offset("mpg").tol(1e-12));
    v.push(auto("p_offset_fw", &[]).offset("mpg").w("W", WeightKind::Fweight));
    v.push(auto("p_offset_absorb", &["turn"]).offset("mpg").keepsing());
    v.push(poisson("p_noreg_noa", "auto", "price", &[], &[]));
    v.push(poisson("p_noreg_absorb", "auto", "price", &[], &["turn"]).keepsing().tol(1e-12));
    v.push({
        let mut g = poisson("p_savefe", "auto", "price", &["weight", "gear_ratio"], &["turn", "foreign"]).keepsing();
        g.robust = true;
        g.absorb.iter_mut().for_each(|t| t.1 = true);
        g.d_var = Some("dfe");
        g.fe_vars = vec!["FE1", "FE2"];
        g
    });
    v.push({
        let mut g = auto("p_d_twoway", &["turn", "trunk"]).tol(1e-12);
        g.d_var = Some("dd");
        g
    });
    v.push({
        let mut g = auto("p_guess_ols", &["turn"]);
        g.guess = Guess::Ols;
        g
    });
    v.push({
        let mut g = poisson("p_gravity", "gravity", "trade", &["rta", "x2"], &[]).cluster(&[&["exp", "imp"]]);
        g.absorb = vec![(vec!["exp", "year"], false), (vec!["imp", "year"], false), (vec!["exp", "imp"], false)];
        g.d_var = Some("dg");
        // ppmlhdfe's d satisfies the FE score equations only to ≈5e-7 (relative) here.
        g.d_tol = 1e-5;
        g
    });
    v
}

fn logit_specs() -> Vec<G> {
    let mut v = vec![
        logit("l_oim", "y", &["fe1", "fe2"]),
        {
            let mut g = logit("l_robust", "y", &["fe1", "fe2"]);
            g.robust = true;
            g
        },
        logit("l_cluster", "y", &["fe1", "fe2"]).cluster(&[&["c"]]),
        logit("l_fw", "y", &["fe1", "fe2"]).w("fw", WeightKind::Fweight),
        logit("l_pw", "y", &["fe1", "fe2"]).w("pw", WeightKind::Pweight),
        logit("l_onefe", "y", &["fe1"]),
        logit("l_noabsorb", "y", &[]),
        logit("l_allzero_groups", "y2", &["fe1", "fe2"]),
    ];
    // Logit separation: the `fe` rule only (plan default).
    v.iter_mut().for_each(|g| g.mu = false);
    v
}

#[test]
fn poisson_matches_ppmlhdfe() {
    for &kind in KINDS {
        each(poisson_specs(), |g| check_poisson(g, kind));
    }
}

#[test]
fn logit_matches_stata_logit_with_dummies() {
    for &kind in KINDS {
        each(logit_specs(), |g| check_logit(g, kind));
    }
}

/// Runs every spec, then fails listing all specs that failed.
fn each(specs: Vec<G>, f: impl Fn(&G)) {
    let failed: Vec<&str> = specs
        .iter()
        .filter(|g| std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(g))).is_err())
        .map(|g| g.fixture)
        .collect();
    assert!(failed.is_empty(), "failed fixtures: {failed:?}");
}

#[test]
fn logit_mu_rule_is_harmless_without_separation() {
    for &kind in KINDS {
        let mut g = logit("l_oim", "y", &["fe1", "fe2"]);
        g.mu = true;
        check_logit(&g, kind);
    }
}

#[test]
fn all_zero_depvar_is_insufficient_observations() {
    let csv = Csv::read("problem3");
    let mut g = poisson("p_problem3", "problem3", "y", &["x1"], &[]);
    // Rows 1-3 have y = 0: keep only those.
    g.first_obs = 0;
    let mut zero = Csv { names: csv.names.clone(), cols: csv.cols.iter().map(|c| c[..3].to_vec()).collect() };
    zero.cols[0] = vec![1.0, 2.0, 3.0];
    for &kind in KINDS {
        match run(&g, &zero, kind, 2) {
            Err(e) => assert_eq!(e.rc(), 2001, "{e}"),
            Ok(_) => panic!("all-zero depvar must fail"),
        }
    }
}

#[test]
fn negative_depvar_is_rejected() {
    let mut csv = Csv::read("problem2");
    let y = csv.names.iter().position(|n| n == "y").unwrap();
    csv.cols[y][1] = -1.0;
    let g = poisson("p_problem2", "problem2", "y", &["x1", "x2"], &["i", "j"]);
    for &kind in KINDS {
        assert_eq!(run(&g, &csv, kind, 1).err().map(|e| e.rc()), Some(459));
    }
}

/// D3: identical bits for any thread count. The dense reference absorber runs faer's
/// multithreaded SVD, whose rounding depends on the thread count, so with it only `noabsorb`
/// models (no SVD work) are checked; the fast absorber is checked on FE models.
#[test]
fn results_do_not_depend_on_threads() {
    for &kind in KINDS {
        let specs = if kind == AbsorberKind::Dense {
            vec![
                poisson("p_fw_cluster", "auto", "price", &["weight", "gear_ratio", "length"], &[])
                    .w("displacement", WeightKind::Fweight)
                    .cluster(&[&["turn"]]),
                logit("l_noabsorb", "y", &[]).cluster(&[&["c"], &["fe2"]]),
            ]
        } else {
            vec![poisson_specs().pop().unwrap(), logit("l_cluster", "y", &["fe1", "fe2"]).cluster(&[&["c"]])]
        };
        for g in specs {
            let csv = Csv::read(g.data);
            let a = run(&g, &csv, kind, 1).unwrap().res;
            let b = run(&g, &csv, kind, 7).unwrap().res;
            assert_eq!(mat(&a, "b"), mat(&b, "b"), "{}", g.fixture);
            assert_eq!(mat(&a, "V"), mat(&b, "V"), "{}", g.fixture);
            assert_eq!(a.scalars, b.scalars, "{}", g.fixture);
        }
    }
}

/// Synthetic panel large enough for several fixed chunks (n = 70 000), Poisson with zeros and
/// logit; fast absorber only (beyond the dense reference's size limit).
fn synthetic() -> Csv {
    let n = 70_000usize;
    let mut rng = fereg_core::rng::SplitMix64::new(0x61F1);
    let a1: Vec<f64> = (0..2000).map(|_| rng.next_f64() - 0.5).collect();
    let a2: Vec<f64> = (0..40).map(|_| 0.5 * (rng.next_f64() - 0.5)).collect();
    let names = ["y", "yb", "x1", "x2", "fe1", "fe2", "c"].map(str::to_string).to_vec();
    let mut cols = vec![Vec::with_capacity(n); names.len()];
    for i in 0..n {
        let (f1, f2) = (i % 2000, (i * 7) % 40);
        let x1 = rng.next_f64() * 2.0 - 1.0;
        let x2 = rng.next_f64() + 0.3 * x1;
        let eta = 0.4 * x1 - 0.3 * x2 + a1[f1] + a2[f2];
        let u = rng.next_f64();
        // A count with mean ≈ exp(η) and many zeros (geometric draw).
        let p = 1.0 / (1.0 + eta.exp());
        let y = (u.ln() / (1.0 - p).ln()).floor().min(50.0);
        let yb = f64::from(u8::from(rng.next_f64() < 1.0 / (1.0 + (-eta).exp())));
        for (c, v) in cols.iter_mut().zip([y, yb, x1, x2, f1 as f64, f2 as f64, (i % 300) as f64]) {
            c.push(v);
        }
    }
    Csv { names, cols }
}

#[test]
fn large_problems_are_thread_count_invariant() {
    let csv = synthetic();
    let mut p = poisson("synthetic poisson", "-", "y", &["x1", "x2"], &["fe1", "fe2"]).cluster(&[&["c"], &["fe2"]]);
    p.mu = true;
    let mut l = logit("synthetic logit", "yb", &["fe1", "fe2"]).cluster(&[&["c"]]);
    l.tol = 1e-10;
    for &kind in KINDS.iter().filter(|k| **k != AbsorberKind::Dense) {
        for g in [&p, &l] {
            let a = run(g, &csv, kind, 1).unwrap().res;
            let b = run(g, &csv, kind, 4).unwrap().res;
            assert_eq!(mat(&a, "b"), mat(&b, "b"), "{}", g.fixture);
            assert_eq!(mat(&a, "V"), mat(&b, "V"), "{}", g.fixture);
            assert_eq!(a.scalars, b.scalars, "{}", g.fixture);
            assert_eq!(scalar(&a, "converged"), 1.0);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// fixest convention ([`GlmConvention::Fixest`]) and [`GlmParts`]: ground truth from fixest 0.14.2
// `fepois` / `feglm(family = "logit")` at glm.tol = 1e-12 (`tests/data/gen_glm_parts_fixtures.R`,
// data `glm_parts.csv`, fixtures `glm_fx_*.json`): sample, coefficients (rtol 1e-7), deviance,
// log-likelihood, sumFE, IRLS iterations, and the inverse Hessian inv(xt'·diag(w)·xt) (rtol 1e-7
// vs fixest's cov.iid and vcov(iid)·(n − K)/(n − 1); 1e-6 vs the one at the converged estimate).

const FX_MODELS: &[&str] = &["fe1", "fe2", "fe3", "w", "off", "w_off", "nofe", "nofe_w_off"];

struct FxRun {
    session: Session,
    res: EstResult,
    parts: Option<GlmParts>,
}

/// Fixture model `fx` with every FE term saved, fixest-style sample rules (singletons and the
/// perfect-fit rule) and no extra separation checks; `conv` = None runs plain [`Session::glm`].
fn run_fx(fx: &J, csv: &Csv, kind: AbsorberKind, maxiter: usize, conv: Option<GlmConvention>) -> Result<FxRun, RfError> {
    let family = if fx.get("family").unwrap().str() == "poisson" { Family::Poisson } else { Family::Logit };
    let fe: Vec<&str> = match fx.get("fe").unwrap() {
        J::Arr(v) => v.iter().map(J::str).collect(),
        _ => panic!("fe"),
    };
    let fcols: Vec<&[f64]> = fe.iter().map(|f| csv.col(f)).collect();
    let y = csv.col(fx.get("y").unwrap().str());
    let weighted = fx.get("weights").unwrap().num() == 1.0;
    let prep = Prep {
        n: y.len(),
        factors: fcols.iter().map(|c| FactorValues::Num(c)).collect(),
        absorb: (0..fe.len()).map(|q| AbsorbTerm { factors: vec![q], intercept: true, slopes: vec![], save: true }).collect(),
        slope_vars: vec![],
        weights: weighted.then(|| csv.col("w")),
        weight_kind: if weighted { WeightKind::Aweight } else { WeightKind::None },
        vce: Vce::Unadjusted,
        drop_singletons: true,
        family,
        y_for_sep: Some(y),
        sep_fe: true,
    };
    let ctx = Ctx::quiet(3);
    let mut session = Session::prepare_with(prep, &ctx, kind)?;
    let kept: Vec<usize> = session.kept_index().iter().map(|&i| i as usize).collect();
    let kc = |v: &[f64]| -> Vec<f64> { kept.iter().map(|&i| v[i]).collect() };
    let cols = [kc(y), kc(csv.col("x1")), kc(csv.col("x2"))];
    let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
    let rows = session.gather(&refs, &ctx)?;
    let offset = if fx.get("offset").unwrap().num() == 1.0 { Some(session.gather(&[&kc(csv.col("off"))], &ctx)?) } else { None };
    let opts = GlmOptions {
        family,
        tol: 1e-12,
        maxiter,
        itol: 1e-12,
        separation: Separation::default(),
        guess: Guess::Simple,
        log: false,
    };
    let (res, parts) = match conv {
        Some(c) => session.glm_parts(rows, &Layout::ols(2), offset, &Spec::default(), &opts, c, &ctx).map(|(r, p)| (r, Some(p)))?,
        None => (session.glm(rows, &Layout::ols(2), offset, &Spec::default(), &opts, &ctx)?, None),
    };
    Ok(FxRun { session, res, parts })
}

/// Inverse of a small symmetric positive definite matrix (Gauss–Jordan).
fn inv_small(a: &[f64], k: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut inv: Vec<f64> = (0..k * k).map(|i| f64::from(u8::from(i / k == i % k))).collect();
    for c in 0..k {
        let p = m[c * k + c];
        for j in 0..k {
            m[c * k + j] /= p;
            inv[c * k + j] /= p;
        }
        for r in (0..k).filter(|&r| r != c) {
            let f = m[r * k + c];
            for j in 0..k {
                m[r * k + j] -= f * m[c * k + j];
                inv[r * k + j] -= f * inv[c * k + j];
            }
        }
    }
    inv
}

fn check_fx(name: &str, kind: AbsorberKind) {
    let fx = fixture(&format!("fx_{name}"));
    let csv = Csv::read("parts");
    let FxRun { session, res, parts } = run_fx(&fx, &csv, kind, 25, Some(GlmConvention::Fixest)).unwrap();
    let parts = parts.unwrap();
    let mut e = Errs::default();
    let n = session.n();
    assert!(parts.converged, "{name}: not converged in {} iterations", parts.iterations);
    assert_eq!(n as f64, fx.get("nobs").unwrap().num(), "{name}: nobs");
    let obs: Vec<f64> = session.kept_index().iter().map(|&i| f64::from(i) + 1.0).collect();
    assert_eq!(obs, fx.get("obs").unwrap().nums(), "{name}: sample");
    assert_eq!(parts.status, vec![0, 0]);
    assert_eq!(parts.k, 2);
    for v in [&parts.eta, &parts.mu, &parts.w, &parts.fe_sum, &parts.working_resid] {
        assert_eq!(v.len(), n);
    }
    assert_eq!(parts.xt.len(), 2 * n);

    // Coefficients (fixest reports `(Intercept)` first when there is no FE).
    let want = fx.get("coef").unwrap().nums();
    let nofe = want.len() == 3;
    if nofe {
        e.scalar("cons", parts.cons, want[0], 1e-7);
    }
    e.vec("coef", &parts.coef, &want[want.len() - 2..], 1e-7);
    e.scalar("deviance", parts.deviance, fx.get("deviance").unwrap().num(), 1e-9);
    let poisson = fx.get("family").unwrap().str() == "poisson";
    let weighted = fx.get("weights").unwrap().num() == 1.0;
    // fixest's weighted logit log-likelihood treats the weights as binomial trial counts
    // (binomial()$aic with rounded weights); ours is Σ w·[y ln μ + (1 − y) ln(1 − μ)].
    if poisson || !weighted {
        e.scalar("ll", scalar(&res, "ll"), fx.get("loglik").unwrap().num(), 1e-9);
    }

    // η = x'β + fe_sum + offset, and the saved FE slots add up to fe_sum.
    let kept: Vec<usize> = session.kept_index().iter().map(|&i| i as usize).collect();
    let has_off = fx.get("offset").unwrap().num() == 1.0;
    let rebuilt: Vec<f64> = kept
        .iter()
        .zip(&parts.fe_sum)
        .map(|(&i, f)| f + parts.coef[0] * csv.col("x1")[i] + parts.coef[1] * csv.col("x2")[i] + if has_off { csv.col("off")[i] } else { 0.0 })
        .collect();
    e.vec_scaled("eta identity", &rebuilt, &parts.eta, 1e-10);
    if nofe {
        assert!(parts.fe.is_empty());
        e.vec_scaled("fe_sum = cons", &parts.fe_sum, &vec![parts.cons; n], 1e-9);
    } else {
        let sum: Vec<f64> = (0..n).map(|i| parts.fe.iter().map(|s| s[i]).sum()).collect();
        e.vec_scaled("sum of fe slots", &sum, &parts.fe_sum, 1e-9);
        e.vec_scaled("sumFE", &parts.fe_sum, &fx.get("sumFE").unwrap().nums(), 1e-7);
    }
    let mu_from_eta: Vec<f64> = parts.eta.iter().map(|&t| if poisson { t.exp() } else { 1.0 / (1.0 + (-t).exp()) }).collect();
    e.vec("mu", &parts.mu, &mu_from_eta, 1e-12);

    // inv(xt'·diag(w)·xt) is fixest's cov.iid (slope block), i.e. vcov(iid)/((n−1)/(n−K)).
    let mut h = [0.0; 4];
    for i in 0..n {
        let (a, b) = (parts.xt[2 * i], parts.xt[2 * i + 1]);
        h[0] += parts.w[i] * a * a;
        h[1] += parts.w[i] * a * b;
        h[3] += parts.w[i] * b * b;
    }
    h[2] = h[1];
    let inv = inv_small(&h, 2);
    let slope = |m: &[f64]| -> Vec<f64> {
        let k = if nofe { 3 } else { 2 };
        let o = k - 2;
        vec![m[o * k + o], m[o * k + o + 1], m[(o + 1) * k + o], m[(o + 1) * k + o + 1]]
    };
    e.vec("cov.iid", &inv, &slope(&fx.get("cov_iid").unwrap().nums()), 1e-7);
    let nf = n as f64;
    let adj = (nf - 1.0) / (nf - fx.get("nparams").unwrap().num());
    let v_iid: Vec<f64> = slope(&fx.get("vcov_iid").unwrap().nums()).iter().map(|v| v / adj).collect();
    e.vec("vcov iid / adj", &inv, &v_iid, 1e-7);
    // Like fixest, `w` comes from the start of the last iteration, whose μ is accurate to about
    // sqrt(glm.tol): 3e-7 relative from the Hessian at the converged estimate on the logits.
    e.vec("inverse Hessian at the estimate", &inv, &slope(&fx.get("cov_iid_conv").unwrap().nums()), 1e-6);
    // Same starting values, step halving and stopping rule as fixest: the same IRLS path.
    let fx_iter = fx.get("iterations").unwrap().num();
    assert_eq!(parts.iterations as f64, fx_iter, "{name}: IRLS iterations");
    e.report(&format!("fx_{name} {kind:?} it={}/{fx_iter}", parts.iterations));
}

#[test]
fn fixest_convention_matches_fixest() {
    let mut failed = Vec::new();
    for fam in ["p", "l"] {
        for m in FX_MODELS {
            let name = format!("{fam}_{m}");
            for &kind in KINDS {
                if std::panic::catch_unwind(|| check_fx(&name, kind)).is_err() {
                    failed.push(format!("{name} {kind:?}"));
                }
            }
        }
    }
    assert!(failed.is_empty(), "failed: {failed:?}");
}

/// Reaching the IRLS cap is a note with `converged` = false, not an error.
#[test]
fn fixest_convention_reports_non_convergence() {
    let fx = fixture("fx_p_fe2");
    let csv = Csv::read("parts");
    let FxRun { res, parts, .. } = run_fx(&fx, &csv, AbsorberKind::Fast, 2, Some(GlmConvention::Fixest)).unwrap();
    let parts = parts.unwrap();
    assert!(!parts.converged);
    assert_eq!(parts.iterations, 2);
    assert_eq!(scalar(&res, "converged"), 0.0);
    assert!(res.notes.iter().any(|s| s.starts_with("Absence of convergence")), "{:?}", res.notes);
}

/// `glm_parts` under the ppmlhdfe convention returns exactly `glm`'s results, and its parts agree.
#[test]
fn parts_do_not_change_ppmlhdfe_results() {
    let fx = fixture("fx_p_w_off");
    let csv = Csv::read("parts");
    let plain = run_fx(&fx, &csv, AbsorberKind::Fast, 10_000, None).unwrap();
    let with = run_fx(&fx, &csv, AbsorberKind::Fast, 10_000, Some(GlmConvention::Ppmlhdfe)).unwrap();
    assert_eq!(plain.res, with.res);
    let parts = with.parts.unwrap();
    assert_eq!(parts.coef, mat(&with.res, "b")[..2]);
    assert_eq!(parts.cons, mat(&with.res, "b")[2]);
    assert!(parts.converged);
}

/// The separation check alone (feppml `tagsep()`): it flags exactly the observations the fit
/// drops as separated, with a positive certificate there and 0 elsewhere, and leaves the
/// session unchanged; the fit reports a complete check.
#[test]
fn separation_check_alone_matches_the_fit() {
    let specs = [
        poisson("p_problem2", "problem2", "y", &["x1", "x2"], &["i", "j"]),
        poisson("p_problem5", "problem5", "y", &["x1", "x2"], &["i", "j"]).tol(1e-12),
        poisson("p_hard3", "hard3", "y", &[], &["id1", "id2"]),
    ];
    for g in &specs {
        let csv = Csv::read(g.data);
        for &kind in KINDS {
            let Setup { mut session, rows, offset, spec, opts, .. } = setup(g, &csv, kind, 2).unwrap();
            let kept_before = session.kept_index().to_vec();
            let chk = session.glm_separation(rows, &Layout::ols(g.x.len()), offset, &spec, &opts, &Ctx::quiet(2)).unwrap();
            // The session is left without the flagged observations (and singletons that leaves).
            let left: std::collections::HashSet<u32> = session.kept_index().iter().copied().collect();
            assert!(left.len() < kept_before.len() || chk.num_separated == 0, "{}: session not reduced", g.fixture);
            for (k, s) in chk.separated.iter().enumerate() {
                assert!(!(*s && left.contains(&kept_before[k])), "{}: flagged row left in the session", g.fixture);
            }
            assert!(chk.complete);
            let fx = fixture(g.fixture);
            assert_eq!(chk.num_separated as f64, fx.get("scalars").unwrap().get("num_separated").unwrap().num(), "{}", g.fixture);
            for (s, z) in chk.separated.iter().zip(&chk.certificate) {
                assert_eq!(*s, *z > 0.0, "{}: certificate", g.fixture);
            }
            // The fit drops the flagged observations (and possibly singletons created by that).
            let fit = run(g, &csv, kind, 2).unwrap();
            assert_eq!(scalar(&fit.res, "separation_complete"), 1.0);
            let after: std::collections::HashSet<u32> = fit.session.kept_index().iter().copied().collect();
            for (k, s) in chk.separated.iter().enumerate() {
                if *s {
                    assert!(!after.contains(&kept_before[k]), "{}: flagged row kept by the fit", g.fixture);
                }
            }
        }
    }
}

/// Field-panel shape of the feppml `tagsep()` report (firm + industry×year FEs, Poisson): firms
/// with all-zero outcomes, an all-zero industry×year cell, observations separated by a
/// regressor that is positive only where y = 0, and singletons left behind by the drops.
struct SepPanel {
    firm: Vec<f64>,
    indyr: Vec<f64>,
    y: Vec<f64>,
    x1: Vec<f64>,
    x2: Vec<f64>,
}

fn sep_panel() -> SepPanel {
    let mut rng = fereg_core::rng::SplitMix64::new(20261012);
    let normal = |rng: &mut fereg_core::rng::SplitMix64| {
        let u1 = rng.next_f64().max(1e-300);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * rng.next_f64()).cos()
    };
    let (nfirm, nyear, nind) = (120usize, 8usize, 5usize);
    let a: Vec<f64> = (0..nfirm).map(|_| 0.5 * normal(&mut rng)).collect();
    let c: Vec<f64> = (0..nind * nyear).map(|_| 0.5 * normal(&mut rng)).collect();
    let mut p = SepPanel { firm: vec![], indyr: vec![], y: vec![], x1: vec![], x2: vec![] };
    for f in 0..nfirm {
        let ind = f % nind;
        for t in 0..nyear {
            if rng.next_f64() > 0.85 {
                continue;
            }
            let cell = ind * nyear + t;
            let x1 = normal(&mut rng);
            let mu = (0.4 * x1 + a[f] + c[cell]).exp();
            // Poisson draw by inversion.
            let (mut k, mut pk, u) = (0.0, (-mu).exp(), rng.next_f64());
            let mut cdf = pk;
            while u > cdf && k < 1e3 {
                k += 1.0;
                pk *= mu / k;
                cdf += pk;
            }
            let all_zero = f % 9 == 0 || cell == 3;
            let y = if all_zero { 0.0 } else { k };
            p.firm.push(f as f64);
            p.indyr.push(cell as f64);
            p.y.push(y);
            p.x1.push(x1);
            p.x2.push(if y == 0.0 && f % 7 == 3 && !all_zero { 1.0 } else { 0.0 });
        }
    }
    p
}

/// Rows of `alive` (over the panel) that the `fe` rule removes, iterated with singleton removal
/// to a fixed point (plain loops: the expected values of the test below). Returns the rows
/// removed with y = 0 and updates `alive`.
fn fe_cascade(p: &SepPanel, alive: &mut [bool]) -> Vec<bool> {
    let n = p.y.len();
    let mut out = vec![false; n];
    loop {
        let mut changed = false;
        for f in [&p.firm, &p.indyr] {
            let mut cnt = std::collections::HashMap::<i64, (usize, f64)>::new();
            for i in (0..n).filter(|&i| alive[i]) {
                let e = cnt.entry(f[i] as i64).or_default();
                e.0 += 1;
                e.1 += p.y[i];
            }
            for i in 0..n {
                let (c, s) = cnt.get(&(f[i] as i64)).copied().unwrap_or_default();
                if alive[i] && (c == 1 || s == 0.0) {
                    alive[i] = false;
                    out[i] = p.y[i] == 0.0;
                    changed = true;
                }
            }
        }
        if !changed {
            return out;
        }
    }
}

fn sep_session(p: &SepPanel, sep_fe: bool, kind: AbsorberKind, ctx: &Ctx) -> (Session, Vec<f64>) {
    let prep = Prep {
        n: p.y.len(),
        factors: vec![FactorValues::Num(&p.firm), FactorValues::Num(&p.indyr)],
        absorb: (0..2).map(|q| AbsorbTerm { factors: vec![q], intercept: true, slopes: vec![], save: false }).collect(),
        slope_vars: vec![],
        weights: None,
        weight_kind: WeightKind::None,
        vce: Vce::Robust,
        drop_singletons: true,
        family: Family::Poisson,
        y_for_sep: Some(&p.y),
        sep_fe,
    };
    let session = Session::prepare_with(prep, ctx, kind).unwrap();
    let kept: Vec<usize> = session.kept_index().iter().map(|&i| i as usize).collect();
    let cols: Vec<Vec<f64>> = [&p.y, &p.x1, &p.x2].iter().map(|v| kept.iter().map(|&i| v[i]).collect()).collect();
    let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
    let rows = session.gather(&refs, ctx).unwrap();
    (session, rows)
}

/// The separation check without the `fe` rule at prepare (feppml `tagsep()`, as ppmlhdfe) finds
/// every observation of all-zero FE groups, including those that become all-zero or singletons
/// only after other separated observations are dropped, plus those separated by a regressor;
/// the fit with `relu` alone reaches the same sample and estimates as with `fe relu`.
#[test]
fn separation_check_without_fe_rule_finds_all_zero_groups() {
    let p = sep_panel();
    let n = p.y.len();
    let ctx = Ctx::quiet(2);
    let opts = GlmOptions {
        family: Family::Poisson,
        tol: 1e-10,
        maxiter: 10_000,
        itol: 1e-12,
        separation: Separation { relu: true, mu: false },
        guess: Guess::Simple,
        log: false,
    };
    for &kind in KINDS {
        let (mut session, rows) = sep_session(&p, false, kind, &ctx);
        let kept: Vec<usize> = session.kept_index().iter().map(|&i| i as usize).collect();
        let chk = session.glm_separation(rows, &Layout::ols(2), None, &Spec::default(), &GlmOptions { tol: 1e-8, itol: 1e-9, ..opts.clone() }, &ctx).unwrap();
        assert!(chk.complete);

        // Expected: the fe cascade on the prepared sample, the rows with x2 = 1 left after it,
        // then the cascade again.
        let mut alive = vec![false; n];
        kept.iter().for_each(|&i| alive[i] = true);
        let mut want = fe_cascade(&p, &mut alive);
        let nfe = want.iter().filter(|v| **v).count();
        for i in 0..n {
            if alive[i] && p.x2[i] > 0.0 {
                want[i] = true;
                alive[i] = false;
            }
        }
        for (w, c) in want.iter_mut().zip(fe_cascade(&p, &mut alive)) {
            *w |= c;
        }
        let got: Vec<bool> = {
            let mut g = vec![false; n];
            kept.iter().zip(&chk.separated).for_each(|(&i, &s)| g[i] = s);
            g
        };
        assert!(nfe > 0 && want.iter().filter(|v| **v).count() > nfe, "the panel exercises both rules");
        assert_eq!(got, want, "{kind:?}: separated observations");
        assert_eq!(chk.num_separated, want.iter().filter(|v| **v).count());
        for (s, z) in chk.separated.iter().zip(&chk.certificate) {
            assert_eq!(*s, *z > 0.0, "{kind:?}: certificate");
        }

        // Estimation: `relu` alone gives the sample and estimates of `fe relu`.
        let fit = |sep_fe: bool| {
            let (mut session, rows) = sep_session(&p, sep_fe, kind, &ctx);
            let res = session.glm(rows, &Layout::ols(2), None, &Spec::default(), &opts, &ctx).unwrap();
            (session.kept_index().to_vec(), res)
        };
        let (kept_a, a) = fit(false);
        let (kept_b, b) = fit(true);
        assert_eq!(kept_a, kept_b, "{kind:?}: estimation sample");
        assert_eq!(scalar(&a, "N_full"), scalar(&b, "N_full"));
        let (ba, bb) = (mat(&a, "b"), mat(&b, "b"));
        for (x, y) in ba.iter().zip(bb) {
            assert!((x - y).abs() <= 1e-8 * (1.0 + y.abs()), "{kind:?}: b {ba:?} vs {bb:?}");
        }
    }
}
