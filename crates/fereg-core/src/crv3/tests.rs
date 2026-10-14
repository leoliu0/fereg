use std::time::Instant;

use super::*;
use crate::par::with_pool;
use crate::rng::SplitMix64;

fn normal(s: &mut SplitMix64) -> f64 {
    let u1 = s.next_f64().max(1e-300);
    let u2 = s.next_f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn poisson(s: &mut SplitMix64, mu: f64) -> f64 {
    let l = (-mu).exp();
    let (mut k, mut p) = (0.0, 1.0);
    loop {
        p *= s.next_f64();
        if p <= l {
            return k;
        }
        k += 1.0;
    }
}

/// Simulated sample: `g` clusters; `nested` (3 levels per cluster) and `nested2` (2 per
/// cluster) are nested in the cluster, `cross` (12 levels) is not.
struct Sim {
    n: usize,
    g: usize,
    cluster: Vec<u32>,
    nested: Vec<u32>,
    nested2: Vec<u32>,
    cross: Vec<u32>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    slope: Vec<f64>,
    z1: Vec<f64>,
    z2: Vec<f64>,
    endog: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
    ybin: Vec<f64>,
    ycount: Vec<f64>,
    off: Vec<f64>,
}

const CROSS: usize = 12;

fn sim(n: usize, g: usize, seed: u64) -> Sim {
    let mut s = SplitMix64::new(seed);
    let mut d = Sim {
        n,
        g,
        cluster: Vec::new(),
        nested: Vec::new(),
        nested2: Vec::new(),
        cross: Vec::new(),
        x1: Vec::new(),
        x2: Vec::new(),
        slope: Vec::new(),
        z1: Vec::new(),
        z2: Vec::new(),
        endog: Vec::new(),
        y: Vec::new(),
        w: Vec::new(),
        ybin: Vec::new(),
        ycount: Vec::new(),
        off: Vec::new(),
    };
    let shock: Vec<f64> = (0..g).map(|_| normal(&mut s)).collect();
    let fe_c: Vec<f64> = (0..CROSS).map(|_| normal(&mut s)).collect();
    for i in 0..n {
        let c = (i % g) as u32;
        let nested = c * 3 + (s.next_u64() % 3) as u32;
        let nested2 = c * 2 + (s.next_u64() % 2) as u32;
        let cross = (s.next_u64() % CROSS as u64) as u32;
        let a = 0.3 * ((nested % 7) as f64 - 3.0) + fe_c[cross as usize];
        let x1 = normal(&mut s) + 0.2 * a;
        let x2 = normal(&mut s) + 0.5 * shock[c as usize];
        let slope = 1.0 + s.next_f64();
        let z1 = normal(&mut s);
        let z2 = normal(&mut s);
        let u = normal(&mut s) + 0.5 * shock[c as usize];
        let endog = 0.6 * z1 + 0.4 * z2 + 0.3 * x1 + 0.5 * u + normal(&mut s);
        let y = 1.0 + 0.5 * x1 - 0.3 * x2 + 0.7 * endog + a + 0.2 * slope * ((nested % 5) as f64) + u;
        let eta = 0.2 + 0.4 * x1 - 0.3 * x2 + 0.5 * a;
        let off = 0.1 * normal(&mut s);
        let pb = 1.0 / (1.0 + (-(eta + off)).exp());
        d.ybin.push(if s.next_f64() < pb { 1.0 } else { 0.0 });
        d.ycount.push(poisson(&mut s, (eta + off).exp()));
        d.cluster.push(c);
        d.nested.push(nested);
        d.nested2.push(nested2);
        d.cross.push(cross);
        d.x1.push(x1);
        d.x2.push(x2);
        d.slope.push(slope);
        d.z1.push(z1);
        d.z2.push(z2);
        d.endog.push(endog);
        d.y.push(y);
        d.w.push(0.5 + 1.5 * s.next_f64());
        d.off.push(off);
    }
    d
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Fe {
    None,
    Nested,
    NestedTwo,
    NestedSlope,
    Cross,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    model: JkModel,
    iv: bool,
    weighted: bool,
    offset: bool,
    fe: Fe,
}

impl Sim {
    fn fe(&self, fe: Fe) -> Vec<FeCols<'_>> {
        fn t(codes: &[u32], g: usize) -> FeCols<'_> {
            FeCols { codes, g, intercept: true, slopes: Vec::new() }
        }
        match fe {
            Fe::None => Vec::new(),
            Fe::Nested => vec![t(&self.nested, 3 * self.g)],
            Fe::NestedTwo => vec![t(&self.nested, 3 * self.g), t(&self.nested2, 2 * self.g)],
            Fe::NestedSlope => vec![FeCols { codes: &self.nested, g: 3 * self.g, intercept: true, slopes: vec![&self.slope] }],
            Fe::Cross => vec![t(&self.cross, CROSS)],
        }
    }

    fn problem<'a>(&'a self, c: Case, beta: &'a [f64]) -> JkProblem<'a> {
        let y: &[f64] = match c.model {
            JkModel::Linear => &self.y,
            JkModel::Poisson => &self.ycount,
            JkModel::Logit => &self.ybin,
        };
        JkProblem {
            model: c.model,
            n: self.n,
            y,
            x: vec![&self.x1, &self.x2],
            endog: if c.iv { vec![&self.endog] } else { Vec::new() },
            inst: if c.iv { vec![&self.z1, &self.z2] } else { Vec::new() },
            intercept: c.fe == Fe::None,
            weights: c.weighted.then_some(self.w.as_slice()),
            offset: c.offset.then_some(self.off.as_slice()),
            fe: self.fe(c.fe),
            beta,
            fixef_tol: 1e-12,
            fixef_maxiter: 100_000,
            glm_tol: 1e-12,
            glm_maxiter: 100,
            collin_tol: 1e-9,
        }
    }

    /// pyfixest-style brute force: `fit::fit` on the data with a factor column of the FE and a
    /// drop mask per cluster; coefficients in [`JkProblem::beta`] order.
    fn brute(&self, c: Case) -> Vec<f64> {
        let zero = vec![0.0; 4];
        let p = self.problem(c, &zero);
        let k = n_coef(&p);
        let want = sources(&p);
        let cross: Vec<f64> = self.cross.iter().map(|&v| v as f64 * 10.0 + 3.0).collect();
        let (factors, fe) = if c.fe == Fe::Cross {
            (vec![FactorCol::F64(&cross)], vec![FeTerm { factors: vec![0], intercept: true, slopes: Vec::new() }])
        } else {
            assert_eq!(c.fe, Fe::None);
            (Vec::new(), Vec::new())
        };
        let spec = FitSpec {
            model: match c.model {
                JkModel::Linear => Model::Ols,
                JkModel::Poisson => Model::Poisson,
                JkModel::Logit => Model::Logit,
            },
            intercept: true,
            fixef_tol: p.fixef_tol,
            fixef_maxiter: p.fixef_maxiter,
            glm_tol: p.glm_tol,
            glm_maxiter: p.glm_maxiter,
            keep_resid: false,
            keep_fixef: false,
            keep_state: false,
            ..FitSpec::default()
        };
        let mut out = Vec::with_capacity(self.g * k);
        for g in 0..self.g {
            let drop: Vec<bool> = self.cluster.iter().map(|&c| c as usize == g).collect();
            let input = FitInput {
                n: self.n,
                y: p.y,
                x: p.x.clone(),
                endog: p.endog.clone(),
                inst: p.inst.clone(),
                factor_names: factors.iter().map(|_| "cross".to_string()).collect(),
                factors: factors.clone(),
                fe: fe.clone(),
                weights: p.weights,
                offset: p.offset,
                drop: Some(&drop),
                ..FitInput::default()
            };
            let f = fit::fit(&input, &spec, &Ctx::quiet(2)).unwrap();
            out.extend(want.iter().map(|s| f.coef[f.coef_source.iter().position(|t| t == s).unwrap()]));
        }
        out
    }
}

fn run(p: &JkProblem, cluster: &[u32], g: usize, threads: usize, path: Path, workers: Option<usize>) -> Result<Vec<f64>, RfError> {
    with_pool(threads, |pool| leave_cluster_out_with(p, cluster, g, &Ctx::quiet(threads), pool, path, workers)).unwrap()
}

/// max over coefficients j of max_g |a - b| / max_g |b| (G x k blocks).
fn max_rel(a: &[f64], b: &[f64], k: usize) -> f64 {
    assert_eq!(a.len(), b.len());
    (0..k)
        .map(|j| {
            let scale = b.iter().skip(j).step_by(k).fold(0.0f64, |m, v| m.max(v.abs()));
            a.iter().zip(b).skip(j).step_by(k).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max) / scale
        })
        .fold(0.0, f64::max)
}

#[test]
fn closed_form_equals_refit() {
    let s = sim(2400, 24, 7);
    for weighted in [false, true] {
        for iv in [false, true] {
            for fe in [Fe::None, Fe::Nested, Fe::NestedTwo, Fe::NestedSlope] {
                let c = Case { model: JkModel::Linear, iv, weighted, offset: false, fe };
                let zero = [0.0; 4];
                let p = s.problem(c, &zero[..n_coef(&s.problem(c, &zero))]);
                assert!(closed_form_applies(&p, &s.cluster), "{c:?}");
                let k = n_coef(&p);
                let a = run(&p, &s.cluster, s.g, 4, Path::Auto, None).unwrap();
                let b = run(&p, &s.cluster, s.g, 4, Path::Refit, None).unwrap();
                let e = max_rel(&a, &b, k);
                eprintln!("closed vs refit {c:?}: max rel {e:e}");
                assert!(e <= 1e-9, "{c:?}: {e:e}");
            }
        }
    }
    let c = Case { model: JkModel::Linear, iv: false, weighted: false, offset: false, fe: Fe::Cross };
    assert!(!closed_form_applies(&s.problem(c, &[0.0; 2]), &s.cluster));
}

#[test]
fn iv_and_glm_match_brute_force() {
    let s = sim(1800, 15, 11);
    let cases = [
        Case { model: JkModel::Linear, iv: true, weighted: true, offset: false, fe: Fe::Cross },
        Case { model: JkModel::Linear, iv: true, weighted: true, offset: false, fe: Fe::None },
        Case { model: JkModel::Linear, iv: false, weighted: false, offset: false, fe: Fe::Cross },
        Case { model: JkModel::Logit, iv: false, weighted: false, offset: false, fe: Fe::Cross },
        Case { model: JkModel::Logit, iv: false, weighted: true, offset: true, fe: Fe::None },
        Case { model: JkModel::Poisson, iv: false, weighted: true, offset: true, fe: Fe::Cross },
    ];
    for c in cases {
        let zero = [0.0; 4];
        let k = n_coef(&s.problem(c, &zero));
        let p = s.problem(c, &zero[..k]);
        let got = run(&p, &s.cluster, s.g, 4, Path::Auto, None).unwrap();
        let want = s.brute(c);
        let e = max_rel(&got, &want, k);
        eprintln!("brute force {c:?}: max rel {e:e}");
        assert!(e <= 1e-9, "{c:?}: {e:e}");
    }
}

#[test]
fn bit_identical_for_any_thread_count() {
    let s = sim(1500, 12, 3);
    let zero = [0.0; 4];
    let cases = [
        (Case { model: JkModel::Poisson, iv: false, weighted: true, offset: true, fe: Fe::Cross }, Path::Auto),
        (Case { model: JkModel::Linear, iv: true, weighted: true, offset: false, fe: Fe::NestedTwo }, Path::Auto),
        (Case { model: JkModel::Linear, iv: true, weighted: true, offset: false, fe: Fe::NestedTwo }, Path::Refit),
    ];
    for (c, path) in cases {
        let k = n_coef(&s.problem(c, &zero));
        let p = s.problem(c, &zero[..k]);
        let a = run(&p, &s.cluster, s.g, 1, path, None).unwrap();
        for (threads, workers) in [(3, None), (8, None), (8, Some(1)), (8, Some(3))] {
            let b = run(&p, &s.cluster, s.g, threads, path, workers).unwrap();
            assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{c:?} {path:?} threads {threads} workers {workers:?}");
        }
    }
}

#[test]
fn lost_coefficient_is_an_error() {
    let mut s = sim(1200, 12, 5);
    // x2 is zero outside cluster 4: without cluster 4 its coefficient is not estimable.
    s.x2.iter_mut().zip(&s.cluster).for_each(|(v, &c)| {
        if c != 4 {
            *v = 0.0;
        }
    });
    for (c, path) in [
        (Case { model: JkModel::Linear, iv: false, weighted: true, offset: false, fe: Fe::None }, Path::Auto),
        (Case { model: JkModel::Linear, iv: true, weighted: false, offset: false, fe: Fe::Nested }, Path::Auto),
        (Case { model: JkModel::Linear, iv: false, weighted: false, offset: false, fe: Fe::Cross }, Path::Auto),
        (Case { model: JkModel::Poisson, iv: false, weighted: false, offset: false, fe: Fe::Cross }, Path::Auto),
    ] {
        let zero = [0.0; 4];
        let k = n_coef(&s.problem(c, &zero));
        let p = s.problem(c, &zero[..k]);
        for (threads, workers) in [(1, None), (6, None)] {
            match run(&p, &s.cluster, s.g, threads, path, workers) {
                Err(RfError::InvalidInput(m)) => assert!(m.contains("x[1]") && m.contains("cluster 4 "), "{c:?}: {m}"),
                other => panic!("{c:?}: expected an error, got {other:?}"),
            }
        }
    }
}

#[test]
fn interrupt_stops_the_refits() {
    let s = sim(1200, 12, 9);
    let zero = [0.0; 2];
    let p = s.problem(Case { model: JkModel::Poisson, iv: false, weighted: false, offset: false, fe: Fe::Cross }, &zero);
    for threads in [1, 4] {
        let calls = Cell::new(0usize);
        let interrupt = || {
            calls.set(calls.get() + 1);
            calls.get() > 40
        };
        let ctx = Ctx { threads, verbose: 0, interrupt: &interrupt, log: &discard };
        let r = with_pool(threads, |pool| leave_cluster_out(&p, &s.cluster, s.g, &ctx, pool)).unwrap();
        assert_eq!(r, Err(RfError::Interrupted), "threads {threads}");
    }
}

#[test]
fn meat_is_centered_outer_product_sum() {
    let betas = [1.0, 2.0, 0.5, 1.0, 1.5, 3.0];
    let beta = [1.0, 2.0];
    // d = (0, 0), (-0.5, -1), (0.5, 1)
    assert_eq!(jackknife_meat(&betas, &beta, 2), vec![0.5, 1.0, 1.0, 2.0]);
    assert!(jackknife_meat(&[], &[], 0).is_empty());
}

// ---------------------------------------------------------------------------------------------
// pyfixest 0.60.0 reference (tests/data/crv3/gen_crv3_fixtures.py)

fn data_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/crv3")
}

/// Numeric CSV with a header.
fn read_csv(name: &str) -> (Vec<String>, Vec<Vec<f64>>) {
    let text = std::fs::read_to_string(data_dir().join(name)).unwrap();
    let mut lines = text.lines();
    let names: Vec<String> = lines.next().unwrap().split(',').map(str::to_string).collect();
    let mut cols = vec![Vec::new(); names.len()];
    for l in lines.filter(|l| !l.is_empty()) {
        l.split(',').zip(&mut cols).for_each(|(v, c)| c.push(v.parse::<f64>().unwrap()));
    }
    (names, cols)
}

#[test]
fn matches_pyfixest() {
    let (names, cols) = read_csv("data.csv");
    let col = |n: &str| cols[names.iter().position(|x| x == n).unwrap()].as_slice();
    let expected = std::fs::read_to_string(data_dir().join("expected.txt")).unwrap();
    let field = |case: &str, key: &str| -> Vec<f64> {
        let line = expected.lines().find(|l| l.split(' ').take(2).eq([case, key])).unwrap_or_else(|| panic!("{case} {key}"));
        line.split(' ').skip(2).map(|v| v.parse().unwrap()).collect()
    };
    let n = col("y").len();
    let fe1: Vec<u32> = col("fe1").iter().map(|&v| v as u32).collect();
    let cl: Vec<u32> = col("cl").iter().map(|&v| v as u32).collect();
    let g_count = *cl.iter().max().unwrap() as usize + 1;
    let fe1_g = *fe1.iter().max().unwrap() as usize + 1;
    let fe1_f: Vec<f64> = fe1.iter().map(|&v| v as f64).collect();
    // (case, model, y, weighted, fe)
    let cases = [
        ("ols_nofe", JkModel::Linear, "y", false, false),
        ("ols_nofe_w", JkModel::Linear, "y", true, false),
        ("ols_fe", JkModel::Linear, "y", false, true),
        ("ols_fe_w", JkModel::Linear, "y", true, true),
        ("pois_fe", JkModel::Poisson, "yp", false, true),
        ("pois_fe_w", JkModel::Poisson, "yp", true, true),
    ];
    for (case, model, yname, weighted, has_fe) in cases {
        let y = col(yname);
        let weights = weighted.then(|| col("w"));
        // Full-sample beta (tight tolerances).
        let input = FitInput {
            n,
            y,
            x: vec![col("x1"), col("x2")],
            factors: if has_fe { vec![FactorCol::F64(&fe1_f)] } else { Vec::new() },
            factor_names: if has_fe { vec!["fe1".into()] } else { Vec::new() },
            fe: if has_fe { vec![FeTerm { factors: vec![0], intercept: true, slopes: Vec::new() }] } else { Vec::new() },
            weights,
            ..FitInput::default()
        };
        let spec = FitSpec {
            model: if model == JkModel::Linear { Model::Ols } else { Model::Poisson },
            fixef_tol: 1e-13,
            glm_tol: 1e-13,
            glm_maxiter: 100,
            keep_state: false,
            ..FitSpec::default()
        };
        let full = fit::fit(&input, &spec, &Ctx::quiet(4)).unwrap();
        assert_eq!(full.nobs, n, "{case}");
        let p = JkProblem {
            model,
            n,
            y,
            x: vec![col("x1"), col("x2")],
            endog: Vec::new(),
            inst: Vec::new(),
            intercept: !has_fe,
            weights,
            offset: None,
            fe: if has_fe { vec![FeCols { codes: &fe1, g: fe1_g, intercept: true, slopes: Vec::new() }] } else { Vec::new() },
            beta: &full.coef,
            fixef_tol: 1e-13,
            fixef_maxiter: 100_000,
            glm_tol: 1e-13,
            glm_maxiter: 100,
            collin_tol: 1e-9,
        };
        let k = full.coef.len();
        let betas = run(&p, &cl, g_count, 4, Path::Auto, None).unwrap();
        let meat = jackknife_meat(&betas, &full.coef, k);
        let (nn, df_k, gg) = (field(case, "N")[0], field(case, "df_k")[0], field(case, "G")[0]);
        assert_eq!(gg as usize, g_count);
        let ssc = (nn - 1.0) / (nn - df_k) * gg / (gg - 1.0);
        let v: Vec<f64> = meat.iter().map(|m| ssc * m).collect();
        let want = field(case, "vcov");
        let coef_want = field(case, "coef");
        let coef_err = full.coef.iter().zip(&coef_want).map(|(a, b)| ((a - b) / b).abs()).fold(0.0, f64::max);
        let err = v.iter().zip(&want).map(|(a, b)| ((a - b) / b).abs()).fold(0.0, f64::max);
        eprintln!("pyfixest {case}: coef max rel {coef_err:e}, vcov max rel {err:e}");
        assert!(err <= 1e-8, "{case}: vcov max rel {err:e}\n got {v:?}\nwant {want:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Timings: `cargo test -p fereg-core --release crv3::tests::timing_1m -- --ignored --nocapture`

#[test]
#[ignore]
fn timing_1m() {
    let n = 1_000_000;
    let threads = 16;
    let mut s = SplitMix64::new(42);
    // 100k individuals x 10 years; clusters group individuals (so the year FE is not nested).
    let indiv: Vec<u32> = (0..n as u32).map(|i| i / 10).collect();
    let year: Vec<u32> = (0..n as u32).map(|i| i % 10).collect();
    let fi: Vec<f64> = (0..n / 10).map(|_| normal(&mut s)).collect();
    let x1: Vec<f64> = (0..n).map(|i| normal(&mut s) + 0.3 * fi[i / 10]).collect();
    let x2: Vec<f64> = (0..n).map(|_| normal(&mut s)).collect();
    let y: Vec<f64> = (0..n).map(|i| 0.5 * x1[i] - 0.2 * x2[i] + fi[i / 10] + 0.1 * (i % 10) as f64 + normal(&mut s)).collect();
    let yc: Vec<f64> = (0..n).map(|i| poisson(&mut s, (0.1 + 0.2 * x1[i] - 0.1 * x2[i] + 0.3 * fi[i / 10]).exp())).collect();
    let beta = [0.0; 2];
    let problem = |model: JkModel, two_fe: bool| JkProblem {
        model,
        n,
        y: if model == JkModel::Linear { &y } else { &yc },
        x: vec![&x1, &x2],
        endog: Vec::new(),
        inst: Vec::new(),
        intercept: false,
        weights: None,
        offset: None,
        fe: if two_fe {
            vec![FeCols { codes: &indiv, g: n / 10, intercept: true, slopes: Vec::new() }, FeCols { codes: &year, g: 10, intercept: true, slopes: Vec::new() }]
        } else {
            vec![FeCols { codes: &indiv, g: n / 10, intercept: true, slopes: Vec::new() }]
        },
        beta: &beta,
        fixef_tol: 1e-8,
        fixef_maxiter: 16_000,
        glm_tol: 1e-8,
        glm_maxiter: 25,
        collin_tol: 1e-9,
    };
    let time = |what: &str, p: &JkProblem, g: usize, path: Path, workers: Option<usize>| {
        let cl: Vec<u32> = indiv.iter().map(|&i| (i as usize % g) as u32).collect();
        let t = Instant::now();
        let r = run(p, &cl, g, threads, path, workers).unwrap();
        let secs = t.elapsed().as_secs_f64();
        eprintln!("timing n=1e6 threads={threads} G={g} {what} workers={workers:?}: {secs:.3} s ({:.2} ms per cluster)", 1e3 * secs / g as f64);
        r
    };
    let ols2 = problem(JkModel::Linear, true);
    let pois2 = problem(JkModel::Poisson, true);
    let ols1 = problem(JkModel::Linear, false);
    // Single fit for reference.
    for t in [1, 16] {
        let cl: Vec<u32> = indiv.iter().map(|&i| i % 2).collect();
        let st = Instant::now();
        let _ = with_pool(t, |pool| refits(&ols2, &cl, 2, &Ctx::quiet(t), pool, Some(1))).unwrap().unwrap();
        eprintln!("timing n=1e6 one OLS refit on {t} threads: {:.3} s", st.elapsed().as_secs_f64() / 2.0);
    }
    for g in [50, 5000] {
        let a = time("OLS 2FE refit", &ols2, g, Path::Auto, None);
        if g == 50 {
            let b = time("OLS 2FE refit", &ols2, g, Path::Auto, Some(1));
            assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
            time("OLS 2FE refit", &ols2, g, Path::Auto, Some(4));
        }
        time("Poisson 2FE refit", &pois2, g, Path::Auto, None);
        time("OLS 1FE (nested) closed form", &ols1, g, Path::Auto, None);
    }
}
