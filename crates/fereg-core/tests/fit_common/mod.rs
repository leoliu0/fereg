//! Shared helpers of the `fit_*` tests: a minimal JSON reader, the fixture CSV reader, and the
//! translation of a fixture case into a `FitInput` / `FitSpec`.
#![allow(dead_code)]

use fereg_core::Ctx;
use fereg_core::fit::{self, CoefSource, FactorCol, FeTerm, FitInput, FitOutput, FitSpec, Flavor, GDf, KFixef, Model, Ssc, TDf, VcovKind, VcovOutput, WeightsType};

#[derive(Clone, Debug)]
pub enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    pub fn get(&self, k: &str) -> Option<&J> {
        match self {
            J::Obj(v) => v.iter().find(|(a, _)| a == k).map(|(_, b)| b),
            _ => None,
        }
    }
    pub fn at(&self, k: &str) -> &J {
        self.get(k).unwrap_or_else(|| panic!("missing key {k}"))
    }
    pub fn num(&self) -> f64 {
        match self {
            J::Num(x) => *x,
            J::Null => f64::NAN,
            J::Arr(v) if v.len() == 1 => v[0].num(),
            _ => panic!("not a number: {self:?}"),
        }
    }
    /// An array (a scalar counts as a one-element array: jsonlite unboxes them).
    pub fn list(&self) -> Vec<J> {
        match self {
            J::Arr(v) => v.clone(),
            J::Null => Vec::new(),
            x => vec![x.clone()],
        }
    }
    pub fn nums(&self) -> Vec<f64> {
        self.list().iter().map(J::num).collect()
    }
    pub fn str(&self) -> &str {
        match self {
            J::Str(s) => s,
            _ => panic!("not a string: {self:?}"),
        }
    }
    pub fn strs(&self) -> Vec<String> {
        self.list().iter().map(|s| s.str().to_string()).collect()
    }
    pub fn bool(&self) -> bool {
        match self {
            J::Bool(b) => *b,
            J::Arr(v) if v.len() == 1 => v[0].bool(),
            _ => panic!("not a bool: {self:?}"),
        }
    }
}

pub fn parse_json(s: &str) -> J {
    fn ws(b: &[u8], i: &mut usize) {
        while *i < b.len() && b[*i].is_ascii_whitespace() {
            *i += 1;
        }
    }
    fn val(b: &[u8], i: &mut usize) -> J {
        ws(b, i);
        match b[*i] {
            b'{' | b'[' => {
                let obj = b[*i] == b'{';
                *i += 1;
                let (mut o, mut a) = (Vec::new(), Vec::new());
                loop {
                    ws(b, i);
                    if b[*i] == b'}' || b[*i] == b']' {
                        *i += 1;
                        return if obj { J::Obj(o) } else { J::Arr(a) };
                    }
                    if obj {
                        let J::Str(k) = val(b, i) else { panic!("bad key") };
                        ws(b, i);
                        assert_eq!(b[*i], b':');
                        *i += 1;
                        o.push((k, val(b, i)));
                    } else {
                        a.push(val(b, i));
                    }
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
            b't' => {
                *i += 4;
                J::Bool(true)
            }
            b'f' => {
                *i += 5;
                J::Bool(false)
            }
            _ => {
                let st = *i;
                while *i < b.len() && (b[*i].is_ascii_alphanumeric() || b"+-.".contains(&b[*i])) {
                    *i += 1;
                }
                let s = std::str::from_utf8(&b[st..*i]).unwrap();
                J::Num(match s {
                    "NaN" | "NA" => f64::NAN,
                    "Inf" => f64::INFINITY,
                    "-Inf" => f64::NEG_INFINITY,
                    _ => s.parse().unwrap_or_else(|_| panic!("bad number {s}")),
                })
            }
        }
    }
    let mut i = 0;
    val(s.as_bytes(), &mut i)
}

pub fn data_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

pub fn read_json(name: &str) -> J {
    parse_json(&std::fs::read_to_string(data_dir().join(name)).unwrap())
}

/// Numeric CSV with a header; `NA` (or an empty field) = NaN.
pub struct Csv {
    pub names: Vec<String>,
    pub cols: Vec<Vec<f64>>,
}

impl Csv {
    pub fn read(name: &str) -> Csv {
        let text = std::fs::read_to_string(data_dir().join(name)).unwrap();
        let mut lines = text.lines();
        let names: Vec<String> = lines.next().unwrap().split(',').map(|s| s.trim_matches('"').to_string()).collect();
        let mut cols = vec![Vec::new(); names.len()];
        for line in lines.filter(|l| !l.is_empty()) {
            for (c, v) in cols.iter_mut().zip(line.split(',')) {
                c.push(if v == "NA" || v.is_empty() { f64::NAN } else { v.parse().unwrap() });
            }
        }
        Csv { names, cols }
    }
    pub fn col(&self, n: &str) -> &[f64] {
        &self.cols[self.names.iter().position(|x| x == n).unwrap_or_else(|| panic!("no column {n}"))]
    }
    pub fn n(&self) -> usize {
        self.cols[0].len()
    }
}

/// Factor columns re-encoded as each [`FactorCol`] variant (the fit must not depend on it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactorKind {
    F64,
    I32,
    I64,
    /// Pre-coded, levels in descending value order with two unused levels, base 1.
    Codes,
}

/// Owned per-factor storage behind the `FactorCol` borrows.
pub enum FactorData {
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Codes(Vec<i32>, u32),
}

impl FactorData {
    pub fn new(v: &[f64], kind: FactorKind) -> FactorData {
        match kind {
            FactorKind::F64 => FactorData::F64(v.to_vec()),
            FactorKind::I32 => FactorData::I32(v.iter().map(|x| if x.is_nan() { i32::MIN } else { *x as i32 }).collect()),
            FactorKind::I64 => FactorData::I64(v.iter().map(|x| if x.is_nan() { i64::MIN } else { *x as i64 }).collect()),
            FactorKind::Codes => {
                let mut lv: Vec<i64> = v.iter().filter(|x| !x.is_nan()).map(|x| *x as i64).collect();
                lv.sort_unstable();
                lv.dedup();
                lv.reverse();
                // Unused levels at positions 0 and 2; missing = 0 (below base).
                let mut levels = vec![i64::MIN];
                levels.extend(lv);
                levels.insert(2.min(levels.len()), i64::MIN + 1);
                let codes = v
                    .iter()
                    .map(|x| if x.is_nan() { 0 } else { levels.iter().position(|l| *l == *x as i64).unwrap() as i32 + 1 })
                    .collect();
                FactorData::Codes(codes, levels.len() as u32)
            }
        }
    }
    pub fn col(&self) -> FactorCol<'_> {
        match self {
            FactorData::F64(v) => FactorCol::F64(v),
            FactorData::I32(v) => FactorCol::I32(v),
            FactorData::I64(v) => FactorCol::I64(v),
            FactorData::Codes(c, n) => FactorCol::Codes { codes: c, nlevels: *n, base: 1 },
        }
    }
}

/// A fixture case's columns, owned (the `FitInput` borrows from it).
pub struct Case {
    pub json: J,
    pub csv: Csv,
    pub factor_names: Vec<String>,
    pub factors: Vec<FactorData>,
    pub fe: Vec<FeTerm>,
    pub slope_names: Vec<String>,
    pub cluster: Vec<Vec<usize>>,
}

/// Position of `name` in `names`, appended when new.
pub fn index_of(name: &str, names: &mut Vec<String>) -> usize {
    names.iter().position(|n| n == name).unwrap_or_else(|| {
        names.push(name.to_string());
        names.len() - 1
    })
}

impl Case {
    pub fn new(json: &J, kind: FactorKind) -> Case {
        let csv = Csv::read(json.at("data").str());
        let mut factor_names: Vec<String> = Vec::new();
        let mut slope_names: Vec<String> = Vec::new();
        // A term is a list of factor names (intercept only) or {factors, slopes, intercept}.
        let fe: Vec<FeTerm> = json
            .at("fe")
            .list()
            .iter()
            .map(|t| match t.get("factors") {
                Some(f) => FeTerm {
                    factors: f.strs().iter().map(|f| index_of(f, &mut factor_names)).collect(),
                    intercept: t.at("intercept").bool(),
                    slopes: t.at("slopes").strs().iter().map(|s| index_of(s, &mut slope_names)).collect(),
                },
                None => FeTerm { factors: t.strs().iter().map(|f| index_of(f, &mut factor_names)).collect(), intercept: true, slopes: Vec::new() },
            })
            .collect();
        let cluster: Vec<Vec<usize>> =
            json.at("cluster").list().iter().map(|d| d.strs().iter().map(|f| index_of(f, &mut factor_names)).collect()).collect();
        // Post-hoc cluster, time and panel variables are input factors too (unused by the fit
        // itself).
        for p in json.get("posthoc").map_or(Vec::new(), J::list) {
            p.at("cluster").list().iter().flat_map(|d| d.strs()).for_each(|f| {
                index_of(&f, &mut factor_names);
            });
        }
        for p in std::iter::once(json.clone()).chain(json.get("posthoc").map_or(Vec::new(), J::list)) {
            for key in ["time", "panel"] {
                if let Some(J::Str(f)) = p.get(key) {
                    index_of(f, &mut factor_names);
                }
            }
        }
        let factors = factor_names.iter().map(|f| FactorData::new(csv.col(f), kind)).collect();
        Case { json: json.clone(), csv, factor_names, factors, fe, slope_names, cluster }
    }

    fn cols(&self, key: &str) -> Vec<&[f64]> {
        self.json.get(key).map_or(Vec::new(), |v| v.strs().iter().map(|x| self.csv.col(x)).collect())
    }

    pub fn input(&self) -> FitInput<'_> {
        let j = &self.json;
        FitInput {
            n: self.csv.n(),
            y: self.csv.col(j.at("y").str()),
            x: self.cols("x"),
            endog: self.cols("endog"),
            inst: self.cols("inst"),
            factors: self.factors.iter().map(FactorData::col).collect(),
            factor_names: self.factor_names.clone(),
            fe: self.fe.clone(),
            slopes: self.slope_names.iter().map(|s| self.csv.col(s)).collect(),
            weights: match j.get("weights") {
                Some(J::Str(w)) => Some(self.csv.col(w)),
                _ => None,
            },
            offset: match j.get("offset") {
                Some(J::Str(o)) => Some(self.csv.col(o)),
                _ => None,
            },
            cluster: self.cluster.clone(),
            time: self.factor_of(j, "time"),
            panel: self.factor_of(j, "panel"),
            lat: self.col_of(j, "lat"),
            lon: self.col_of(j, "lon"),
            ..Default::default()
        }
    }

    /// Factor index of the variable named by `key` of `j` (time, panel), if any.
    pub fn factor_of(&self, j: &J, key: &str) -> Option<usize> {
        match j.get(key) {
            Some(J::Str(f)) => Some(self.factor_names.iter().position(|n| n == f).expect("factor registered")),
            _ => None,
        }
    }

    /// Numeric column named by `key` of `j` (lat, lon), if any.
    pub fn col_of(&self, j: &J, key: &str) -> Option<&[f64]> {
        match j.get(key) {
            Some(J::Str(c)) => Some(self.csv.col(c)),
            _ => None,
        }
    }

    pub fn spec(&self, flavor: Flavor) -> FitSpec {
        let j = &self.json;
        FitSpec {
            model: self.model(),
            flavor,
            intercept: j.at("intercept").bool(),
            vcov: parse_vcov(j.at("vcov")),
            ssc: parse_ssc(j.at("ssc")),
            weights_type: match j.get("weights_type") {
                Some(J::Str(t)) if t == "fweights" => WeightsType::Fweights,
                _ => WeightsType::Aweights,
            },
            fixef_tol: 1e-12,
            glm_tol: 1e-12,
            glm_maxiter: 100,
            ..Default::default()
        }
    }

    pub fn model(&self) -> Model {
        match self.json.get("model").map(J::str) {
            Some("poisson") => Model::Poisson,
            Some("logit") => Model::Logit,
            _ => Model::Ols,
        }
    }

    /// Name of a coefficient as the fixture reports it (fixest names the second-stage
    /// endogenous regressors `fit_<name>`, pyfixest by the variable).
    pub fn source_name(&self, s: &CoefSource, flavor: Flavor) -> String {
        let j = &self.json;
        match s {
            CoefSource::Intercept => "(Intercept)".to_string(),
            CoefSource::X(i) => j.at("x").strs()[*i].clone(),
            CoefSource::Inst(i) => j.at("inst").strs()[*i].clone(),
            CoefSource::Endog(i) => {
                let e = j.at("endog").strs()[*i].clone();
                if flavor == Flavor::Fixest { format!("fit_{e}") } else { e }
            }
        }
    }

    /// The post-hoc variance requests: (kind, ssc, cluster dimensions as factor indices).
    pub fn posthoc(&self) -> Vec<(VcovKind, Ssc, Vec<Vec<usize>>)> {
        let mut names = self.factor_names.clone();
        self.json
            .get("posthoc")
            .map_or(Vec::new(), J::list)
            .iter()
            .map(|p| {
                let cl = p.at("cluster").list().iter().map(|d| d.strs().iter().map(|f| index_of(f, &mut names)).collect()).collect();
                (parse_vcov(p.at("vcov")), parse_ssc(p.at("ssc")), cl)
            })
            .collect()
    }
}

/// A fixture's variance request: a string (`iid`, `hetero`, `hc2`, `hc3`, `crv3`, anything else =
/// cluster) or an object `{type: NW | DK | conley, lag, cutoff, pixel, spherical}`.
pub fn parse_vcov(v: &J) -> VcovKind {
    let opt_lag = |o: &J| match o.get("lag") {
        Some(J::Num(l)) => Some(*l as usize),
        _ => None,
    };
    match v {
        J::Obj(_) => match v.at("type").str() {
            "NW" => VcovKind::NeweyWest { lag: opt_lag(v) },
            "DK" => VcovKind::DriscollKraay { lag: opt_lag(v) },
            "conley" => VcovKind::Conley {
                cutoff: match v.get("cutoff") {
                    Some(J::Num(c)) => Some(*c),
                    _ => None,
                },
                pixel: match v.get("pixel") {
                    Some(J::Num(p)) => *p,
                    _ => 0.0,
                },
                spherical: matches!(v.get("spherical"), Some(J::Bool(true))),
            },
            t => panic!("unknown vcov type {t}"),
        },
        _ => match v.str() {
            "iid" => VcovKind::Iid,
            "hetero" => VcovKind::Hetero,
            "hc2" => VcovKind::Hc2,
            "hc3" => VcovKind::Hc3,
            "crv3" => VcovKind::Crv3,
            _ => VcovKind::Cluster,
        },
    }
}

pub fn parse_ssc(s: &J) -> Ssc {
    Ssc {
        k_adj: s.at("K.adj").bool(),
        k_fixef: match s.at("K.fixef").str() {
            "none" => KFixef::None,
            "full" => KFixef::Full,
            _ => KFixef::Nonnested,
        },
        k_exact: s.at("K.exact").bool(),
        g_adj: s.at("G.adj").bool(),
        g_df: if s.at("G.df").str() == "min" { GDf::Min } else { GDf::Conventional },
        t_df: match s.at("t.df") {
            J::Str(t) if t == "min" => TDf::Min,
            J::Str(_) => TDf::Conventional,
            x => TDf::Fixed(x.num()),
        },
    }
}

/// Compares a variance matrix over the coefficients `names` (our order) with a fixture entry
/// holding `vcov`, `se`, `df_k`, `df_t` and optionally `G`, over `want_names`.
pub fn check_vcov(what: &str, v: &VcovOutput, names: &[String], e: &J, want_names: &[String], rtol: f64) {
    let k = names.len();
    assert_eq!(k, want_names.len(), "{what}: coefficient count");
    let p: Vec<usize> = want_names.iter().map(|w| names.iter().position(|n| n == w).unwrap_or_else(|| panic!("{what}: no coefficient {w}"))).collect();
    let se: Vec<f64> = p.iter().map(|&a| v.vcov[a * k + a].sqrt()).collect();
    assert_vec(&format!("{what} se"), &se, &e.at("se").nums(), rtol, 1e-12);
    let want = e.at("vcov").nums();
    let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let got: Vec<f64> = p.iter().flat_map(|&a| p.iter().map(move |&b| v.vcov[a * k + b])).collect();
    assert_vec(&format!("{what} vcov"), &got, &want, rtol, scale);
    assert_eq!(v.df_k, e.at("df_k").num(), "{what}: df_k");
    // GLMs report normal inference (fixest still attaches its df.t).
    if let Some(t) = v.df_t {
        assert_eq!(t, e.at("df_t").num(), "{what}: df_t");
    }
    if let Some(t) = e.get("n_periods") {
        assert_eq!(v.n_periods.map(|t| t as f64), Some(t.num()), "{what}: n_periods");
    }
    if let Some(l) = e.get("lag") {
        let ours = match v.kind {
            VcovKind::NeweyWest { lag } | VcovKind::DriscollKraay { lag } => lag,
            _ => None,
        };
        assert_eq!(ours.map(|l| l as f64), Some(l.num()), "{what}: lag");
    }
    if let Some(c) = e.get("cutoff") {
        let VcovKind::Conley { cutoff: Some(ours), .. } = v.kind else { panic!("{what}: no resolved cutoff") };
        assert_close(&format!("{what} cutoff"), ours, c.num(), 1e-12);
    }
    if let Some(g) = e.get("G") {
        let ours: Vec<f64> = v.n_clusters.iter().map(|&g| g as f64).collect();
        assert_eq!(ours, g.nums(), "{what}: clusters");
    }
}

/// Compares coefficients by name (the fixture's order may differ from ours).
pub fn check_coef(what: &str, coef: &[f64], names: &[String], want: &[f64], want_names: &[String], rtol: f64) {
    assert_eq!(names.len(), want_names.len(), "{what}: coefficients {names:?} vs {want_names:?}");
    let got: Vec<f64> = want_names.iter().map(|w| coef[names.iter().position(|n| n == w).unwrap_or_else(|| panic!("{what}: no coefficient {w}"))]).collect();
    assert_vec(&format!("{what} coef"), &got, want, rtol, 1e-12);
}

/// Runs a fixture case and checks every reported field against the fixture.
pub fn check_case(j: &J, flavor: Flavor, kind: FactorKind, threads: usize, rtol: f64) -> (Case, FitOutput) {
    let name = j.at("name").str().to_string();
    let case = Case::new(j, kind);
    let e = j.at("expect");
    let out = fit::fit(&case.input(), &case.spec(flavor), &Ctx::quiet(threads)).unwrap_or_else(|err| panic!("{name}: {err}"));
    let what = |s: &str| format!("{name} {s}");
    let names: Vec<String> = out.coef_source.iter().map(|s| case.source_name(s, flavor)).collect();
    let want_names = e.at("coef_names").strs();
    check_coef(&name, &out.coef, &names, &e.at("coef").nums(), &want_names, rtol);
    check_vcov(&name, &out.vcov, &names, e, &want_names, rtol);
    assert_eq!(out.nobs as f64, e.at("nobs").num(), "{name}: nobs");
    let glm = case.model() != Model::Ols;
    assert_eq!(out.vcov.df_t.is_none(), glm, "{name}: normal inference iff GLM");
    if let Some(np) = e.get("nparams") {
        assert_eq!(out.nparams as f64, np.num(), "{name}: nparams");
    }
    let collin: Vec<String> = out.collinear.iter().map(|s| case.source_name(s, flavor)).collect();
    assert_eq!(collin, e.at("collin").strs(), "{name}: collinear");
    if let Some(obs) = e.get("obs") {
        let ours: Vec<f64> = out.obs.iter().map(|&i| f64::from(i) + 1.0).collect();
        assert_eq!(ours, obs.nums(), "{name}: obs");
    }
    let s = &out.stats;
    for (k, v) in [
        ("r2", s.r2),
        ("ar2", s.ar2),
        ("wr2", s.wr2),
        ("war2", s.war2),
        ("pr2", s.pseudo_r2),
        ("apr2", s.apr2),
        ("sq_cor", s.sq_cor),
        ("rmse", s.rmse),
        ("ll", s.loglik),
        ("ll_null", s.ll_null),
        ("ssr", s.ssr),
        ("ssr_null", s.ssr_null),
        ("ssr_fe_only", s.ssr_fe_only),
        ("sigma2", s.sigma2),
        ("deviance", s.deviance),
        ("f_stat", s.f_test.stat),
        ("f_df1", s.f_test.df1),
        ("f_df2", s.f_test.df2),
    ] {
        // fixest leaves ssr and ssr_null unset for GLMs (ours: response residuals).
        if glm && (k == "ssr" || k == "ssr_null") {
            continue;
        }
        if let Some(x) = e.get(k) {
            assert_close(&what(k), v, x.num(), rtol);
        }
    }
    if let Some(p) = e.get("f_p") {
        assert_close(&what("f_p"), s.f_test.p, p.num(), 1e-6);
    }
    let resid = e.get("resid").map_or(Vec::new(), J::nums);
    if !resid.is_empty() {
        let scale = resid.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert_vec(&what("resid"), out.resid.as_ref().unwrap(), &resid, rtol, scale);
        if let Some(f) = e.get("fitted") {
            let fitted = f.nums();
            let scale = fitted.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert_vec(&what("fitted"), out.fitted.as_ref().unwrap(), &fitted, rtol, scale);
        }
    }
    for (k, ours) in [("eta", &out.eta), ("irls_weights", &out.irls_weights)] {
        let want = e.get(k).map_or(Vec::new(), J::nums);
        if !want.is_empty() {
            let ours = ours.as_ref().unwrap();
            assert_eq!(ours.len(), want.len(), "{}: length", what(k));
            // Separated observations (|eta| > 20, mu at a bound) have a diverging eta whose value
            // depends on when IRLS stops; their mu and weights are compared through `fitted`.
            let (a, b): (Vec<f64>, Vec<f64>) = ours.iter().zip(&want).filter(|(_, w)| k != "eta" || w.abs() <= 20.0).map(|(a, b)| (*a, *b)).unzip();
            let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert_vec(&what(k), &a, &b, rtol, scale);
        }
    }
    let posthoc = case.posthoc();
    let requests = j.get("posthoc").map_or(Vec::new(), J::list);
    for ((p, (kind, ssc, cl)), req) in e.get("posthoc").map_or(Vec::new(), J::list).iter().zip(posthoc).zip(&requests) {
        let input = case.input();
        let data = fit::VcovData {
            factors: &input.factors,
            factor_names: &case.factor_names,
            cluster: &cl,
            time: case.factor_of(req, "time"),
            panel: case.factor_of(req, "panel"),
            lat: case.col_of(req, "lat"),
            lon: case.col_of(req, "lon"),
        };
        let v = out.state.as_ref().expect("state").vcov(kind, &ssc, &data, &Ctx::quiet(threads)).unwrap_or_else(|err| panic!("{name} post-hoc: {err}"));
        check_vcov(&what("post-hoc"), &v, &names, p, &want_names, rtol);
    }
    (case, out)
}

/// |a - b| <= rtol*max(|b|, floor). An expected NaN (JSON null: jsonlite writes NA, NaN and +/-Inf
/// as null) matches any non-finite value.
pub fn close(a: f64, b: f64, rtol: f64, floor: f64) -> bool {
    (b.is_nan() && !a.is_finite()) || (a - b).abs() <= rtol * b.abs().max(floor)
}

pub fn assert_close(what: &str, a: f64, b: f64, rtol: f64) {
    assert!(close(a, b, rtol, 1e-300), "{what}: got {a:e}, want {b:e}");
}

pub fn assert_vec(what: &str, a: &[f64], b: &[f64], rtol: f64, floor: f64) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(close(*x, *y, rtol, floor), "{what}[{i}]: got {x:e}, want {y:e}");
    }
}
