//! Shared helpers of the `fit_*` tests: a minimal JSON reader, the fixture CSV reader, and the
//! translation of a fixture case into a `FitInput` / `FitSpec`.
#![allow(dead_code)]

use fereg_core::fit::{FactorCol, FeTerm, FitInput, FitSpec, Flavor, GDf, KFixef, Ssc, TDf, VcovKind, WeightsType};

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
    pub cluster: Vec<Vec<usize>>,
}

impl Case {
    pub fn new(json: &J, kind: FactorKind) -> Case {
        let csv = Csv::read(json.at("data").str());
        let mut factor_names: Vec<String> = Vec::new();
        let index = |name: &str, names: &mut Vec<String>| -> usize {
            names.iter().position(|n| n == name).unwrap_or_else(|| {
                names.push(name.to_string());
                names.len() - 1
            })
        };
        let fe: Vec<FeTerm> = json
            .at("fe")
            .list()
            .iter()
            .map(|t| FeTerm { factors: t.strs().iter().map(|f| index(f, &mut factor_names)).collect(), intercept: true, slopes: Vec::new() })
            .collect();
        let cluster: Vec<Vec<usize>> =
            json.at("cluster").list().iter().map(|d| d.strs().iter().map(|f| index(f, &mut factor_names)).collect()).collect();
        let factors = factor_names.iter().map(|f| FactorData::new(csv.col(f), kind)).collect();
        Case { json: json.clone(), csv, factor_names, factors, fe, cluster }
    }

    pub fn input(&self) -> FitInput<'_> {
        let j = &self.json;
        FitInput {
            n: self.csv.n(),
            y: self.csv.col(j.at("y").str()),
            x: j.at("x").strs().iter().map(|x| self.csv.col(x)).collect(),
            factors: self.factors.iter().map(FactorData::col).collect(),
            factor_names: self.factor_names.clone(),
            fe: self.fe.clone(),
            weights: match j.get("weights") {
                Some(J::Str(w)) => Some(self.csv.col(w)),
                _ => None,
            },
            cluster: self.cluster.clone(),
            ..Default::default()
        }
    }

    pub fn spec(&self, flavor: Flavor) -> FitSpec {
        let j = &self.json;
        let s = j.at("ssc");
        let ssc = Ssc {
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
        };
        FitSpec {
            flavor,
            intercept: j.at("intercept").bool(),
            vcov: match j.at("vcov").str() {
                "iid" => VcovKind::Iid,
                "hetero" => VcovKind::Hetero,
                _ => VcovKind::Cluster,
            },
            ssc,
            weights_type: match j.get("weights_type") {
                Some(J::Str(t)) if t == "fweights" => WeightsType::Fweights,
                _ => WeightsType::Aweights,
            },
            fixef_tol: 1e-12,
            ..Default::default()
        }
    }
}

/// |a − b| ≤ rtol·max(|b|, floor). An expected NaN (JSON null: jsonlite writes NA, NaN and ±Inf
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
