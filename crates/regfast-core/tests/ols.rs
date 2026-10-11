//! OLS (plan Phase 3.10).
//!
//! A. An independent explicit-dummy dense reference (eigen pseudo-inverse projections, plain
//!    inverses, brute-force cluster sums, rank-based dof) implementing reghdfe's formulas,
//!    compared at rtol 1e-9 over VCE × weight combinations and special cases.
//! B. Ground truth: reghdfe 6.13.1 results (tol 1e-14) saved by `tests/data/gen_ols_fixtures.do`.
//!
//! Every test runs over [`KINDS`].

use std::collections::HashMap;

use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use regfast_core::rng::SplitMix64;
use regfast_core::{
    AbsorbTerm, AbsorberKind, Ctx, DofOptions, EstResult, Family, FactorValues, Layout, Prep, Session, Spec, Vce, WeightKind,
};

const KINDS: &[AbsorberKind] = &[AbsorberKind::Dense, AbsorberKind::Fast];
/// Absorbers that support heterogeneous slopes (the Fast absorber gains them in Phase 7).
const SLOPE_KINDS: &[AbsorberKind] = &[AbsorberKind::Dense];

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
// Comparison helpers.

fn close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= atol + rtol * a.abs().max(b.abs())
}

fn assert_vec(what: &str, got: &[f64], want: &[f64], rtol: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().filter(|v| v.is_finite()).fold(0.0f64, |m, v| m.max(v.abs()));
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(close(*a, *b, rtol, 1e-10 * scale), "{what}[{i}]: got {a:e}, want {b:e}");
    }
}

fn scalar(r: &EstResult, k: &str) -> f64 {
    r.scalar(k).unwrap_or_else(|| panic!("missing scalar {k}"))
}

fn mat<'a>(r: &'a EstResult, k: &str) -> &'a [f64] {
    &r.matrix(k).unwrap_or_else(|| panic!("missing matrix {k}")).data
}

// ---------------------------------------------------------------------------------------------
// Problem description shared by A and B.

#[derive(Clone)]
struct Model<'a> {
    /// Factor variables (index = factor id).
    factors: Vec<&'a [f64]>,
    slope_vars: Vec<&'a [f64]>,
    absorb: Vec<AbsorbTerm>,
    y: &'a [f64],
    x: Vec<&'a [f64]>,
    weights: Option<(&'a [f64], WeightKind)>,
    vce: Vce,
    singletons: bool,
    spec: Spec,
}

fn term(factors: &[usize], intercept: bool, slopes: &[usize], save: bool) -> AbsorbTerm {
    AbsorbTerm { factors: factors.to_vec(), intercept, slopes: slopes.to_vec(), save }
}

fn run(m: &Model, kind: AbsorberKind, threads: usize) -> (Session, EstResult) {
    let n = m.y.len();
    let prep = Prep {
        n,
        factors: m.factors.iter().map(|f| FactorValues::Num(f)).collect(),
        absorb: m.absorb.clone(),
        slope_vars: m.slope_vars.clone(),
        weights: m.weights.map(|w| w.0),
        weight_kind: m.weights.map_or(WeightKind::None, |w| w.1),
        vce: m.vce.clone(),
        drop_singletons: m.singletons,
        family: Family::Gaussian,
        y_for_sep: None,
        sep_fe: false,
    };
    let ctx = Ctx::quiet(threads);
    let mut s = Session::prepare_with(prep, &ctx, kind).unwrap();
    let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
    let pick = |v: &[f64]| -> Vec<f64> { kept.iter().map(|&i| v[i]).collect() };
    let cols: Vec<Vec<f64>> = std::iter::once(pick(m.y)).chain(m.x.iter().map(|x| pick(x))).collect();
    let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
    let rows = s.gather(&refs, &ctx).unwrap();
    let res = s.ols(rows, &Layout::ols(m.x.len()), &m.spec, &ctx).unwrap();
    (s, res)
}

// ---------------------------------------------------------------------------------------------
// A. Independent dense reference.

fn mat_from(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f64) -> Mat<f64> {
    Mat::<f64>::from_fn(rows, cols, f)
}

/// Moore–Penrose pseudo-inverse of a symmetric PSD matrix via its eigendecomposition; also
/// returns the numerical rank.
fn pinv_sym(a: &Mat<f64>) -> (Mat<f64>, usize) {
    let p = a.nrows();
    let evd = a.self_adjoint_eigen(Side::Lower).unwrap();
    let (u, s) = (evd.U(), evd.S().column_vector());
    let smax = (0..p).map(|i| s[i].abs()).fold(0.0, f64::max);
    let mut out = Mat::<f64>::zeros(p, p);
    let mut rank = 0;
    for e in 0..p {
        if s[e] > 1e-10 * smax {
            rank += 1;
            for i in 0..p {
                for j in 0..p {
                    out[(i, j)] += u[(i, e)] * u[(j, e)] / s[e];
                }
            }
        }
    }
    (out, rank)
}

fn inverse(a: &Mat<f64>) -> Mat<f64> {
    let k = a.nrows();
    a.partial_piv_lu().solve(Mat::<f64>::identity(k, k))
}

fn sdev(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt().max(1e-3)
}

/// Eigen-clip V on reghdfe's standardized scale (s = scale per coefficient).
fn ref_fix_psd(v: &mut Mat<f64>, s: &[f64], rc: bool) -> bool {
    let k = v.nrows();
    let clip = |a: &Mat<f64>| -> Option<Mat<f64>> {
        let n = a.nrows();
        let evd = a.self_adjoint_eigen(Side::Lower).unwrap();
        let (u, l) = (evd.U(), evd.S().column_vector());
        if (0..n).all(|i| l[i] >= 0.0) {
            return None;
        }
        Some(mat_from(n, n, |i, j| (0..n).map(|e| u[(i, e)] * l[e].max(0.0) * u[(j, e)]).sum()))
    };
    let vs = mat_from(k, k, |i, j| v[(i, j)] * s[i] * s[j]);
    let Some(mut fixed) = clip(&vs) else { return false };
    if rc {
        let kb = k - 1;
        let sub = mat_from(kb, kb, |i, j| vs[(i, j)]);
        let sub = clip(&sub).unwrap_or(sub);
        for i in 0..kb {
            for j in 0..kb {
                fixed[(i, j)] = sub[(i, j)];
            }
        }
    }
    *v = mat_from(k, k, |i, j| fixed[(i, j)] / (s[i] * s[j]));
    true
}

struct RefOut {
    b: Vec<f64>,
    v: Vec<f64>,
    df_a: f64,
    df_r: f64,
    rss: f64,
    tss: f64,
    tss_within: f64,
    f: f64,
    resid: Vec<f64>,
    d: Vec<f64>,
}

/// Reference OLS on the kept sample (kept order). `keep_cols` are the regressors expected to
/// survive collinearity checks; `nested_k` is Σ levels of intercept terms nested in a cluster.
fn reference(s: &Session, m: &Model, keep_cols: &[usize], nested_k: f64) -> RefOut {
    let kept: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
    let n = kept.len();
    let pick = |v: &[f64]| -> Vec<f64> { kept.iter().map(|&i| v[i]).collect() };
    let y = pick(m.y);
    let xs: Vec<Vec<f64>> = keep_cols.iter().map(|&j| pick(m.x[j])).collect();
    let k = xs.len();
    let (raw_w, kind) = match m.weights {
        Some((w, kd)) => (pick(w), kd),
        None => (vec![1.0; n], WeightKind::None),
    };
    let sumw: f64 = raw_w.iter().sum();
    let big_n = if kind == WeightKind::Fweight { sumw } else { n as f64 };
    let wv: Vec<f64> = match kind {
        WeightKind::Aweight | WeightKind::Pweight => raw_w.iter().map(|w| w * n as f64 / sumw).collect(),
        _ => raw_w.clone(),
    };

    // Explicit dummies.
    let terms = s.terms();
    let mut dcols: Vec<Vec<f64>> = Vec::new();
    if terms.is_empty() {
        dcols.push(vec![1.0; n]);
    }
    for t in terms {
        if t.intercept {
            for g in 0..t.g {
                dcols.push(t.codes.iter().map(|&c| f64::from(u8::from(c as usize == g))).collect());
            }
        }
        for x in &t.slopes {
            for g in 0..t.g {
                dcols.push(t.codes.iter().zip(x).map(|(&c, &v)| if c as usize == g { v } else { 0.0 }).collect());
            }
        }
    }
    let p = dcols.len();
    let dwd = mat_from(p, p, |a, b| (0..n).map(|i| wv[i] * dcols[a][i] * dcols[b][i]).sum());
    let (pinv, rank_d) = pinv_sym(&dwd);
    let resid_on_d = |v: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let dwv: Vec<f64> = (0..p).map(|a| (0..n).map(|i| wv[i] * dcols[a][i] * v[i]).sum()).collect();
        let alpha: Vec<f64> = (0..p).map(|a| (0..p).map(|b| pinv[(a, b)] * dwv[b]).sum()).collect();
        let fit: Vec<f64> = (0..n).map(|i| (0..p).map(|a| dcols[a][i] * alpha[a]).sum()).collect();
        ((0..n).map(|i| v[i] - fit[i]).collect(), fit)
    };
    let (yt, _) = resid_on_d(&y);
    let xt: Vec<Vec<f64>> = xs.iter().map(|x| resid_on_d(x).0).collect();

    let xx = mat_from(k, k, |a, b| (0..n).map(|i| wv[i] * xt[a][i] * xt[b][i]).sum());
    let xy: Vec<f64> = (0..k).map(|a| (0..n).map(|i| wv[i] * xt[a][i] * yt[i]).sum()).collect();
    let inv = inverse(&xx);
    let beta: Vec<f64> = (0..k).map(|a| (0..k).map(|b| inv[(a, b)] * xy[b]).sum()).collect();
    let e: Vec<f64> = (0..n).map(|i| yt[i] - (0..k).map(|a| beta[a] * xt[a][i]).sum::<f64>()).collect();
    let rss: f64 = (0..n).map(|i| wv[i] * e[i] * e[i]).sum();

    let has_intercept = terms.is_empty() || terms.iter().any(|t| t.intercept);
    let rc = has_intercept && !m.spec.noconstant;
    let wmean = |v: &[f64]| (0..n).map(|i| raw_w[i] * v[i]).sum::<f64>() / sumw;
    let ybar = wmean(&y);
    let xbar: Vec<f64> = xs.iter().map(|x| wmean(x)).collect();
    let b0 = ybar - (0..k).map(|a| xbar[a] * beta[a]).sum::<f64>();
    let n_c = match kind {
        WeightKind::Aweight | WeightKind::Pweight => n as f64,
        _ => sumw,
    };
    let ms = k + usize::from(rc);
    // Bordered matrix [[X̃'WX̃ + n_c x̄x̄', n_c x̄], [n_c x̄', n_c]]; its inverse is reghdfe's inv_ext.
    let bread = if rc {
        let a = mat_from(ms, ms, |i, j| match (i < k, j < k) {
            (true, true) => xx[(i, j)] + n_c * xbar[i] * xbar[j],
            (true, false) => n_c * xbar[i],
            (false, true) => n_c * xbar[j],
            (false, false) => n_c,
        });
        inverse(&a)
    } else {
        inv.clone()
    };
    let design = |i: usize| -> Vec<f64> {
        let mut a: Vec<f64> = (0..k).map(|c| xt[c][i] + if rc { xbar[c] } else { 0.0 }).collect();
        if rc {
            a.push(1.0);
        }
        a
    };

    // Exact for ≤ 2 intercept terms; nested terms are fully redundant (when every term is nested,
    // reghdfe's df_a is 0 while rank − Σ nested would be −1).
    let df_a = (rank_d as f64 - nested_k).max(0.0);
    let df_m = k as f64;
    let mut df_r = big_n - df_a - df_m;
    let mut v = match &m.vce {
        Vce::Unadjusted if kind != WeightKind::Pweight => mat_from(ms, ms, |i, j| rss / df_r * bread[(i, j)]),
        Vce::Unadjusted | Vce::Robust => {
            let mut meat = Mat::<f64>::zeros(ms, ms);
            for i in 0..n {
                let a = design(i);
                let mi = match kind {
                    WeightKind::None => e[i] * e[i],
                    WeightKind::Fweight => e[i] * e[i] * raw_w[i],
                    _ => (e[i] * wv[i]).powi(2),
                };
                for r in 0..ms {
                    for c in 0..ms {
                        meat[(r, c)] += mi * a[r] * a[c];
                    }
                }
            }
            let q = big_n / (big_n - df_a - df_m);
            let out = &bread * &meat * &bread;
            mat_from(ms, ms, |i, j| out[(i, j)] * q)
        }
        Vce::Cluster(_) => {
            let dims: Vec<&[u32]> = s.clusters().iter().map(|c| c.codes.as_slice()).collect();
            let qd = dims.len();
            let mut meat = Mat::<f64>::zeros(ms, ms);
            for mask in 1u32..(1 << qd) {
                let sign = if mask.count_ones() % 2 == 1 { 1.0 } else { -1.0 };
                let mut sums: HashMap<Vec<u32>, Vec<f64>> = HashMap::new();
                for i in 0..n {
                    let key: Vec<u32> = (0..qd).filter(|d| mask >> d & 1 == 1).map(|d| dims[d][i]).collect();
                    let a = design(i);
                    let sc = e[i] * wv[i];
                    let ent = sums.entry(key).or_insert_with(|| vec![0.0; ms]);
                    for r in 0..ms {
                        ent[r] += sc * a[r];
                    }
                }
                for sg in sums.values() {
                    for r in 0..ms {
                        for c in 0..ms {
                            meat[(r, c)] += sign * sg[r] * sg[c];
                        }
                    }
                }
            }
            let g_min = s.clusters().iter().map(|c| c.g).min().unwrap() as f64;
            let nested_adj = if nested_k > 0.0 { 1.0 } else { 0.0 };
            let q = (big_n - 1.0) / (big_n - nested_adj - df_m - df_a) * g_min / (g_min - 1.0);
            let out = &bread * &meat * &bread;
            let mut v = mat_from(ms, ms, |i, j| out[(i, j)] * q);
            if qd > 1 {
                let sy = sdev(&y);
                let sc: Vec<f64> = xs.iter().map(|x| sdev(x) / sy).chain(rc.then_some(1.0 / sy)).collect();
                ref_fix_psd(&mut v, &sc, rc);
            }
            df_r = df_r.min(g_min - 1.0);
            v
        }
    };
    if (0..ms).any(|i| (0..ms).any(|j| v[(i, j)].is_nan())) {
        v = Mat::<f64>::zeros(ms, ms);
    }
    let f = if k > 0 {
        let vk = mat_from(k, k, |i, j| v[(i, j)]);
        let vi = inverse(&vk);
        (0..k).map(|a| (0..k).map(|b| beta[a] * vi[(a, b)] * beta[b]).sum::<f64>()).sum::<f64>() / df_m
    } else {
        f64::NAN
    };
    let tss: f64 = if has_intercept {
        let scale = if matches!(kind, WeightKind::Aweight | WeightKind::Pweight) { n as f64 / sumw } else { 1.0 };
        (0..n).map(|i| raw_w[i] * (y[i] - ybar).powi(2)).sum::<f64>() * scale
    } else {
        let scale = if matches!(kind, WeightKind::Aweight | WeightKind::Pweight) { n as f64 / sumw } else { 1.0 };
        (0..n).map(|i| raw_w[i] * y[i] * y[i]).sum::<f64>() * scale
    };
    let tss_within: f64 = (0..n).map(|i| wv[i] * yt[i] * yt[i]).sum();
    let mut b = beta.clone();
    if rc {
        b.push(b0);
    }
    let b0r = if rc { b0 } else { 0.0 };
    let d: Vec<f64> = (0..n).map(|i| y[i] - (0..k).map(|a| beta[a] * xs[a][i]).sum::<f64>() - b0r - e[i]).collect();
    RefOut { b, v: (0..ms * ms).map(|ij| v[(ij / ms, ij % ms)]).collect(), df_a, df_r, rss, tss, tss_within, f, resid: e, d }
}

struct SimData {
    id: Vec<f64>,
    year: Vec<f64>,
    firm: Vec<f64>,
    region: Vec<f64>,
    c1: Vec<f64>,
    x: Vec<Vec<f64>>,
    slope: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
    fw: Vec<f64>,
}

fn sim(n: usize, seed: u64) -> SimData {
    let mut r = SplitMix64::new(seed);
    let mut normal = move || {
        let u1 = r.next_f64().max(1e-300);
        let u2 = r.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let mut r2 = SplitMix64::new(seed ^ 0xABCD);
    let mut unif = move || r2.next_f64();
    let id: Vec<f64> = (0..n).map(|i| if i < n - 6 { (i / 6) as f64 } else { (1000 + i) as f64 }).collect();
    let year: Vec<f64> = (0..n).map(|i| ((i * 5) % 7) as f64).collect();
    let firm: Vec<f64> = (0..n).map(|_| (unif() * 12.0).floor()).collect();
    let region: Vec<f64> = id.iter().map(|v| (v / 9.0).floor()).collect();
    let c1: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let x1: Vec<f64> = (0..n).map(|_| normal()).collect();
    let x2: Vec<f64> = (0..n).map(|_| normal()).collect();
    let x3: Vec<f64> = (0..n).map(|i| normal() + 0.4 * x1[i]).collect();
    let xa: Vec<f64> = (0..n).map(|i| (id[i] % 4.0) + year[i] * 0.5).collect(); // in the FE span
    let xc: Vec<f64> = (0..n).map(|i| x1[i] - 3.0 * x2[i]).collect(); // collinear with x1, x2
    let slope: Vec<f64> = (0..n).map(|_| 1.0 + 2.0 * unif()).collect();
    let fe_id: Vec<f64> = (0..2000).map(|_| normal()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 0.8 * x1[i] - 0.4 * x2[i] + 0.2 * x3[i] + fe_id[(id[i] as usize) % 2000] + 0.1 * year[i] + 0.3 * firm[i] * slope[i] / 10.0 + normal())
        .collect();
    let w: Vec<f64> = (0..n).map(|_| 0.5 + unif()).collect();
    let fw: Vec<f64> = (0..n).map(|_| 1.0 + (unif() * 3.0).floor()).collect();
    SimData { id, year, firm, region, c1, x: vec![x1, x2, x3, xa, xc], slope, y, w, fw }
}

impl SimData {
    fn model(&self, xs: &[usize], absorb: Vec<AbsorbTerm>, weights: Option<WeightKind>, vce: Vce) -> Model<'_> {
        Model {
            factors: vec![&self.id, &self.year, &self.firm, &self.region, &self.c1],
            slope_vars: vec![&self.slope],
            absorb,
            y: &self.y,
            x: xs.iter().map(|&j| self.x[j].as_slice()).collect(),
            weights: weights.map(|k| (if k == WeightKind::Fweight { self.fw.as_slice() } else { self.w.as_slice() }, k)),
            vce,
            singletons: true,
            spec: Spec { tol: 1e-12, ..Spec::default() },
        }
    }
}

const ID: usize = 0;
const YEAR: usize = 1;
const FIRM: usize = 2;
const REGION: usize = 3;
const C1: usize = 4;

fn check_against_reference(m: &Model, expect_status: &[u8], nested_k: f64, kind: AbsorberKind, what: &str) -> (Session, EstResult) {
    let (s, res) = run(m, kind, 3);
    assert_eq!(res.status, expect_status, "{what}: status");
    let keep: Vec<usize> = expect_status.iter().enumerate().filter(|(_, s)| **s == 0).map(|(j, _)| j).collect();
    let r = reference(&s, m, &keep, nested_k);
    // Map the full b/V onto the kept coefficients (+ _cons).
    let nx = m.x.len();
    let rc = scalar(&res, "report_constant") == 1.0;
    let k_all = nx + usize::from(rc);
    let pos: Vec<usize> = keep.iter().copied().chain(rc.then_some(nx)).collect();
    let b = mat(&res, "b");
    let v = mat(&res, "V");
    assert_eq!(b.len(), k_all);
    let bk: Vec<f64> = pos.iter().map(|&p| b[p]).collect();
    let vk: Vec<f64> = pos.iter().flat_map(|&a| pos.iter().map(move |&c| v[a * k_all + c])).collect();
    assert_vec(&format!("{what}: b"), &bk, &r.b, 1e-9);
    assert_vec(&format!("{what}: V"), &vk, &r.v, 1e-9);
    for (j, st) in expect_status.iter().enumerate() {
        if *st != 0 {
            assert_eq!(b[j], 0.0);
            assert!((0..k_all).all(|c| v[j * k_all + c] == 0.0));
        }
    }
    assert_eq!(scalar(&res, "df_a"), r.df_a, "{what}: df_a");
    assert_eq!(scalar(&res, "df_r"), r.df_r, "{what}: df_r");
    for (name, want) in [("rss", r.rss), ("tss", r.tss), ("tss_within", r.tss_within)] {
        assert!(close(scalar(&res, name), want, 1e-9, 0.0), "{what}: {name} {} vs {want}", scalar(&res, name));
    }
    if r.f.is_finite() && scalar(&res, "F").is_finite() {
        assert!(close(scalar(&res, "F"), r.f, 1e-8, 0.0), "{what}: F {} vs {}", scalar(&res, "F"), r.f);
    }
    if let Some(e) = &res.resid {
        assert_vec(&format!("{what}: resid"), e, &r.resid, 1e-9);
    }
    if let Some(d) = &res.d {
        assert_vec(&format!("{what}: d"), d, &r.d, 1e-9);
    }
    (s, res)
}

#[test]
fn a_vce_by_weight_grid() {
    let dat = sim(900, 7);
    let absorb = vec![term(&[ID], true, &[], false), term(&[YEAR], true, &[], false)];
    let vces: Vec<(Vce, &str, f64)> = vec![
        (Vce::Unadjusted, "unadjusted", 0.0),
        (Vce::Robust, "robust", 0.0),
        (Vce::Cluster(vec![vec![FIRM]]), "cluster firm", 0.0),
        (Vce::Cluster(vec![vec![FIRM], vec![C1]]), "cluster firm c1", 0.0),
    ];
    let weights = [None, Some(WeightKind::Aweight), Some(WeightKind::Fweight), Some(WeightKind::Pweight)];
    for &kind in KINDS {
        for (vce, vname, nested) in &vces {
            for w in weights {
                let mut m = dat.model(&[0, 1, 2], absorb.clone(), w, vce.clone());
                m.spec.want_resid = true;
                m.spec.want_d = true;
                check_against_reference(&m, &[0, 0, 0], *nested, kind, &format!("{vname} {w:?}"));
            }
        }
    }
}

#[test]
fn a_fe_nested_in_cluster() {
    let dat = sim(900, 11);
    let absorb = vec![term(&[ID], true, &[], false), term(&[YEAR], true, &[], false)];
    for &kind in KINDS {
        for vce in [Vce::Cluster(vec![vec![REGION]]), Vce::Cluster(vec![vec![ID]]), Vce::Cluster(vec![vec![REGION], vec![YEAR]])] {
            let m = dat.model(&[0, 1, 2], absorb.clone(), Some(WeightKind::Aweight), vce.clone());
            let (s, _) = run(&m, kind, 2);
            let k_id = s.terms()[0].g as f64;
            // With both id and year nested (cluster region & year), both are fully redundant.
            let nested = if matches!(&vce, Vce::Cluster(d) if d.len() == 2) { k_id + s.terms()[1].g as f64 } else { k_id };
            let (_, res) = check_against_reference(&m, &[0, 0, 0], nested, kind, &format!("nested {vce:?}"));
            assert_eq!(scalar(&res, "df_a_nested"), nested);
        }
    }
}

#[test]
fn a_absorbed_and_collinear_regressors() {
    let dat = sim(900, 13);
    let absorb = vec![term(&[ID], true, &[], false), term(&[YEAR], true, &[], false)];
    for &kind in KINDS {
        for vce in [Vce::Unadjusted, Vce::Cluster(vec![vec![FIRM]])] {
            let m = dat.model(&[0, 3, 1, 4, 2], absorb.clone(), None, vce);
            let (_, res) = check_against_reference(&m, &[0, 2, 0, 3, 0], 0.0, kind, "absorbed/collinear");
            assert_eq!(scalar(&res, "rank"), 3.0);
            assert!(res.notes.iter().all(|n| !n.contains("collinear with the fixed effects")), "per-variable notes belong to the ado");
        }
    }
}

#[test]
fn a_noabsorb_and_noconstant() {
    let dat = sim(700, 17);
    for &kind in KINDS {
        for vce in [Vce::Unadjusted, Vce::Robust, Vce::Cluster(vec![vec![FIRM]])] {
            let m = dat.model(&[0, 1, 2], vec![], Some(WeightKind::Fweight), vce.clone());
            let (_, res) = check_against_reference(&m, &[0, 0, 0], 0.0, kind, &format!("noabsorb {vce:?}"));
            assert_eq!(mat(&res, "dof_table"), &[1.0, 0.0, 1.0, 0.0, 0.0]);
            let mut m = dat.model(&[0, 1, 2], vec![term(&[ID], true, &[], false), term(&[YEAR], true, &[], false)], None, vce.clone());
            m.spec.noconstant = true;
            let (_, res) = check_against_reference(&m, &[0, 0, 0], 0.0, kind, &format!("noconstant {vce:?}"));
            assert_eq!(scalar(&res, "report_constant"), 0.0);
            assert_eq!(mat(&res, "b").len(), 3);
        }
    }
}

#[test]
fn a_slopes_savefe_and_d() {
    let dat = sim(900, 19);
    for &kind in SLOPE_KINDS {
        let absorb = vec![term(&[ID], true, &[], true), term(&[FIRM], true, &[0], true)];
        let mut m = dat.model(&[0, 1, 2], absorb, Some(WeightKind::Aweight), Vce::Robust);
        m.spec.want_d = true;
        m.spec.want_fe = true;
        m.spec.want_resid = true;
        let (s, res) = check_against_reference(&m, &[0, 0, 0], 0.0, kind, "slopes");
        let d = res.d.as_ref().unwrap();
        assert_eq!(res.fe.len(), 3, "id, firm, firm#c.slope");
        let slope = &s.terms()[1].slopes[0];
        for i in 0..s.n() {
            let sum = res.fe[0][i] + res.fe[1][i] + res.fe[2][i] * slope[i];
            assert!((sum - d[i]).abs() < 1e-8 * (1.0 + d[i].abs()), "Σ saved FE != d at {i}");
        }
        // dof table: id exact, firm intercept redundant = mobility groups, slope slot 0.
        let dt = mat(&res, "dof_table");
        assert_eq!(dt.len(), 15);
    }
}

#[test]
fn a_groupvar_matches_bfs_components() {
    // Two disconnected blocks of (id, firm).
    let n = 400;
    let id: Vec<f64> = (0..n).map(|i| (i / 4) as f64).collect();
    let firm: Vec<f64> = (0..n).map(|i| ((i * 7) % 5) as f64 + if i / 4 >= 50 { 10.0 } else { 0.0 }).collect();
    let year: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    let mut r = SplitMix64::new(3);
    let x: Vec<f64> = (0..n).map(|_| r.next_f64()).collect();
    let y: Vec<f64> = (0..n).map(|i| x[i] + r.next_f64() + id[i] * 0.01).collect();
    for &kind in KINDS {
        let mut m = Model {
            factors: vec![&firm, &id, &year],
            slope_vars: vec![],
            absorb: vec![term(&[0], true, &[], false), term(&[1], true, &[], false), term(&[2], true, &[], false)],
            y: &y,
            x: vec![&x],
            weights: None,
            vce: Vce::Unadjusted,
            singletons: true,
            spec: Spec { tol: 1e-12, want_groupvar: true, ..Spec::default() },
        };
        let (s, res) = run(&m, kind, 2);
        let gv = res.groupvar.as_ref().unwrap();
        // First pair is (firm, id): components numbered by the smallest firm level of each.
        let firm_k: Vec<f64> = s.kept_index().iter().map(|&i| firm[i as usize]).collect();
        for (i, g) in gv.iter().enumerate() {
            assert_eq!(*g, if firm_k[i] >= 10.0 { 2.0 } else { 1.0 });
        }
        let dt = mat(&res, "dof_table");
        assert_eq!(dt[5 + 1], 2.0, "id redundant = 2 mobility groups");
        m.spec.dof = DofOptions::none();
        let (_, res) = run(&m, kind, 2);
        assert_eq!(mat(&res, "dof_table")[5 + 1], 1.0);
    }
}

#[test]
fn a_no_regressors() {
    let dat = sim(700, 29);
    let absorb = vec![term(&[ID], true, &[], false), term(&[YEAR], true, &[], false)];
    for &kind in KINDS {
        for vce in [Vce::Unadjusted, Vce::Robust, Vce::Cluster(vec![vec![FIRM], vec![C1]])] {
            let mut m = dat.model(&[], absorb.clone(), Some(WeightKind::Aweight), vce.clone());
            m.spec.want_resid = true;
            let (_, res) = check_against_reference(&m, &[], 0.0, kind, &format!("no regressors {vce:?}"));
            assert_eq!(mat(&res, "b").len(), 1);
            assert!(scalar(&res, "F").is_nan());
            m.spec.noconstant = true;
            let (_, res) = run(&m, kind, 2);
            assert!(mat(&res, "b").is_empty() && mat(&res, "V").is_empty());
        }
    }
}

#[test]
fn a_thread_count_invariance() {
    let dat = sim(900, 23);
    let absorb = vec![term(&[ID], true, &[], false), term(&[YEAR], true, &[], false)];
    // The dense oracle's SVD uses faer's parallelism inside the pool, so its bits depend on the
    // thread count; the invariance contract applies to the production absorber.
    for &kind in KINDS.iter().filter(|k| **k != AbsorberKind::Dense) {
        let m = dat.model(&[0, 1, 2], absorb.clone(), Some(WeightKind::Aweight), Vce::Cluster(vec![vec![FIRM], vec![C1]]));
        let (_, a) = run(&m, kind, 1);
        for t in [2, 4, 7] {
            let (_, b) = run(&m, kind, t);
            assert_eq!(mat(&a, "b"), mat(&b, "b"));
            assert_eq!(mat(&a, "V"), mat(&b, "V"));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// B. reghdfe ground truth.

const FACTORS: [&str; 6] = ["id", "year", "firm", "region", "c1", "c2"];

struct Fixture {
    name: &'static str,
    y: &'static str,
    x: &'static [&'static str],
    /// (factor names, intercept, slope names)
    absorb: &'static [(&'static [&'static str], bool, &'static [&'static str])],
    vce: &'static [&'static str],
    robust: bool,
    weight: Option<(&'static str, WeightKind)>,
    singletons: bool,
    noconstant: bool,
    dof: DofOptions,
    extras: bool,
    status: &'static [u8],
}

const DOF: DofOptions = DofOptions { firstpair: false, pairwise: true, clusters: true, continuous: true };
const IDYEAR: &[(&[&str], bool, &[&str])] = &[(&["id"], true, &[]), (&["year"], true, &[])];

fn fx(name: &'static str, x: &'static [&'static str], absorb: &'static [(&'static [&'static str], bool, &'static [&'static str])]) -> Fixture {
    Fixture {
        name,
        y: "y",
        x,
        absorb,
        vce: &[],
        robust: false,
        weight: None,
        singletons: true,
        noconstant: false,
        dof: DOF,
        extras: false,
        status: &[],
    }
}

fn fixtures() -> Vec<Fixture> {
    let x3: &[&str] = &["x1", "x2", "x3"];
    let x2: &[&str] = &["x1", "x2"];
    vec![
        Fixture { extras: true, ..fx("s01_unadj", x3, IDYEAR) },
        Fixture { robust: true, ..fx("s02_robust_3fe", x3, &[(&["id"], true, &[]), (&["year"], true, &[]), (&["firm"], true, &[])]) },
        Fixture { vce: &["id"], ..fx("s03_cluster_eq_fe", x2, IDYEAR) },
        Fixture { vce: &["region"], ..fx("s04_fe_nested_in_cluster", x3, IDYEAR) },
        Fixture { vce: &["c1", "c2"], ..fx("s05_twoway_psd", x3, &[(&["firm"], true, &[]), (&["year"], true, &[])]) },
        Fixture { vce: &["firm"], weight: Some(("w", WeightKind::Aweight)), extras: true, ..fx("s06_aw_cluster", x3, IDYEAR) },
        Fixture { robust: true, weight: Some(("fw", WeightKind::Fweight)), ..fx("s07_fw_robust", x3, IDYEAR) },
        Fixture { weight: Some(("w", WeightKind::Pweight)), ..fx("s08_pw", x3, &[(&["id"], true, &[]), (&["year"], true, &[]), (&["firm"], true, &[])]) },
        Fixture { status: &[0, 0, 2, 3], ..fx("s09_absorbed_collinear", &["x1", "x2", "xa", "xc"], IDYEAR) },
        fx("s10_interaction", x2, &[(&["id"], true, &[]), (&["firm", "year"], true, &[])]),
        Fixture { vce: &["firm"], ..fx("s11_noabsorb_cluster", x3, &[]) },
        Fixture { noconstant: true, ..fx("s12_noconstant", x2, IDYEAR) },
        Fixture { singletons: false, vce: &["id", "year"], ..fx("s13_keepsing_twoway", x2, IDYEAR) },
        Fixture { extras: true, ..fx("s14_groupvar", x2, &[(&["id"], true, &[]), (&["firm"], true, &[]), (&["year"], true, &[])]) },
        Fixture {
            vce: &["firm"],
            // `dofadjustments(firstpair)` replaces the default list: no `clusters` nesting rule.
            dof: DofOptions { firstpair: true, pairwise: false, clusters: false, continuous: false },
            ..fx("s15_firstpair_cluster", x2, &[(&["id"], true, &[]), (&["year"], true, &[]), (&["firm"], true, &[])])
        },
        fx("s16_slopes", x2, &[(&["id"], true, &[]), (&["firm"], true, &["x4"])]),
        Fixture { y: "y2", ..fx("s17_y_absorbed", x2, IDYEAR) },
        Fixture { vce: &["c1", "c2"], weight: Some(("fw", WeightKind::Fweight)), ..fx("s18_fw_twoway", x3, &[(&["firm"], true, &[]), (&["year"], true, &[])]) },
    ]
}

const INT_SCALARS: &[&str] = &[
    "N", "rank", "df_r", "df_m", "df_a", "df_a_initial", "df_a_redundant", "df_a_nested", "N_clust", "N_clustervars", "N_clust1", "N_clust2",
    "num_singletons", "N_full", "report_constant", "converged",
];
const FLOAT_SCALARS: &[&str] =
    &["tss", "tss_within", "rss", "mss", "rmse", "F", "ll", "ll_0", "r2", "r2_within", "r2_a", "r2_a_within", "sumweights"];
/// Set by the ado (`N_hdfe`, `N_hdfe_extended`, `drop_singletons`) or solver specific (`ic`).
const SKIP_SCALARS: &[&str] = &["N_hdfe", "N_hdfe_extended", "drop_singletons", "ic"];

#[test]
fn b_reghdfe_ground_truth() {
    let data = Csv::read("ols_data.csv");
    let factor_cols: Vec<&[f64]> = FACTORS.iter().map(|f| data.col(f)).collect();
    let fidx = |n: &str| FACTORS.iter().position(|f| *f == n).unwrap();
    for &kind in KINDS {
        for f in fixtures() {
            if !SLOPE_KINDS.contains(&kind) && f.absorb.iter().any(|(_, icpt, sl)| !*icpt || !sl.is_empty()) {
                continue;
            }
            let text = std::fs::read_to_string(data_dir().join(format!("ols_{}.json", f.name))).unwrap();
            let j = parse_json(&text);
            let mut slope_names: Vec<&str> = Vec::new();
            let absorb: Vec<AbsorbTerm> = f
                .absorb
                .iter()
                .map(|(fs, icpt, sl)| {
                    let slopes = sl
                        .iter()
                        .map(|s| {
                            slope_names.push(s);
                            slope_names.len() - 1
                        })
                        .collect::<Vec<_>>();
                    AbsorbTerm { factors: fs.iter().map(|n| fidx(n)).collect(), intercept: *icpt, slopes, save: false }
                })
                .collect();
            let vce = if !f.vce.is_empty() {
                Vce::Cluster(f.vce.iter().map(|n| vec![fidx(n)]).collect())
            } else if f.robust {
                Vce::Robust
            } else {
                Vce::Unadjusted
            };
            let m = Model {
                factors: factor_cols.clone(),
                slope_vars: slope_names.iter().map(|s| data.col(s)).collect(),
                absorb,
                y: data.col(f.y),
                x: f.x.iter().map(|x| data.col(x)).collect(),
                weights: f.weight.map(|(w, k)| (data.col(w), k)),
                vce,
                singletons: f.singletons,
                spec: Spec {
                    tol: 1e-14,
                    dof: f.dof,
                    noconstant: f.noconstant,
                    want_resid: f.extras,
                    want_d: f.extras,
                    want_groupvar: f.extras,
                    ..Spec::default()
                },
            };
            let (s, res) = run(&m, kind, 4);
            let what = format!("{} [{kind:?}]", f.name);

            // Sample.
            let sample = j.get("sample").unwrap().nums();
            let keep: Vec<f64> = s.keep().iter().map(|&k| f64::from(u8::from(k))).collect();
            assert_eq!(keep, sample, "{what}: e(sample)");

            // Status vs omitted names. `_ms_findomitted` (run by reghdfe and by our ado) also marks
            // coefficients with b = 0 and V = 0 as omitted (e.g. when the depvar is absorbed).
            let colnames: Vec<&str> = j.get("colnames").unwrap().str().split_whitespace().collect();
            let (b, v) = (mat(&res, "b"), mat(&res, "V"));
            for (k, st) in res.status.iter().enumerate() {
                let zero = b[k] == 0.0 && v[k * b.len() + k] == 0.0;
                assert_eq!(*st != 0 || zero, colnames[k].starts_with("o."), "{what}: status of {}", colnames[k]);
            }
            if !f.status.is_empty() {
                assert_eq!(res.status, f.status, "{what}: status codes");
            }

            assert_vec(&format!("{what}: b"), mat(&res, "b"), &j.get("b").unwrap().nums(), 1e-7);
            assert_vec(&format!("{what}: V"), mat(&res, "V"), &j.get("V").unwrap().nums(), 1e-6);
            assert_eq!(mat(&res, "dof_table"), j.get("dof_table").unwrap().nums().as_slice(), "{what}: dof_table");

            let J::Obj(scal) = j.get("scalars").unwrap() else { panic!() };
            for (name, val) in scal {
                if SKIP_SCALARS.contains(&name.as_str()) {
                    continue;
                }
                let want = val.num();
                let got = res.scalar(name).unwrap_or_else(|| panic!("{what}: scalar {name} not produced"));
                if INT_SCALARS.contains(&name.as_str()) {
                    assert_eq!(got, want, "{what}: {name}");
                } else {
                    assert!(FLOAT_SCALARS.contains(&name.as_str()), "{what}: unexpected scalar {name}");
                    assert!(close(got, want, 1e-6, 0.0), "{what}: {name} got {got:e} want {want:e}");
                }
            }
            for name in res.scalars.iter().map(|(k, _)| k.as_str()) {
                if name == "y_absorbed" {
                    continue;
                }
                assert!(scal.iter().any(|(k, _)| k == name), "{what}: produced {name} which reghdfe does not post");
            }
            assert_eq!(scalar(&res, "y_absorbed"), f64::from(u8::from(f.name == "s17_y_absorbed")), "{what}: y_absorbed");

            // Per-observation outputs (kept order ↔ sample obs).
            let kept_vals = |key: &str| -> Option<Vec<f64>> {
                j.get(key).map(|v| v.nums().into_iter().zip(&sample).filter(|(_, s)| **s == 1.0).map(|(v, _)| v).collect())
            };
            let tag = &f.name[1..3];
            if let Some(r) = kept_vals(&format!("r{tag}")) {
                assert_vec(&format!("{what}: resid"), res.resid.as_ref().unwrap(), &r, 1e-6);
            }
            if let Some(d) = kept_vals(&format!("d{tag}")) {
                assert_vec(&format!("{what}: d"), res.d.as_ref().unwrap(), &d, 1e-6);
            }
            if let Some(g) = kept_vals(&format!("gv{tag}")) {
                assert_eq!(res.groupvar.as_ref().unwrap(), &g, "{what}: groupvar");
            }
        }
    }
}
