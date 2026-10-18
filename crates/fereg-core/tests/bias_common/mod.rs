//! Shared helpers of the bias-correction tests: fixture cases (tests/data/bias) as fit-layer inputs.
#![allow(dead_code)]

#[path = "../fit_common/mod.rs"]
pub mod fit_common;

#[allow(unused_imports)]
pub use fit_common::{Csv, J, read_json};

use fereg_core::fit::{FactorCol, FeTerm, FitInput, FitSpec, Flavor, Model, Ssc, VcovKind};

/// A fixture case: data, outcome, regressors, FE columns, extra factor columns (panel / time).
pub struct BCase {
    pub name: String,
    pub csv: Csv,
    pub y: String,
    pub x: Vec<String>,
    /// Factor columns: the FE columns (in order), then the extra ones.
    pub factor_names: Vec<String>,
    pub factors: Vec<Vec<f64>>,
    pub fe: Vec<FeTerm>,
    pub weights: Option<String>,
    pub offset: Option<String>,
}

impl BCase {
    pub fn new(j: &J, extra: &[&str]) -> BCase {
        let csv = Csv::read(j.at("data").str());
        let mut factor_names: Vec<String> = j.at("fe").strs();
        let fe = (0..factor_names.len()).map(|q| FeTerm { factors: vec![q], intercept: true, slopes: Vec::new() }).collect();
        for e in extra {
            if !factor_names.iter().any(|f| f == e) {
                factor_names.push(e.to_string());
            }
        }
        let factors = factor_names.iter().map(|f| csv.col(f).to_vec()).collect();
        let opt = |k: &str| match j.get(k) {
            Some(J::Str(v)) => Some(v.clone()),
            _ => None,
        };
        BCase {
            name: j.at("name").str().to_string(),
            csv,
            y: j.at("y").str().to_string(),
            x: j.at("x").strs(),
            factor_names,
            factors,
            fe,
            weights: opt("weights"),
            offset: opt("offset"),
        }
    }

    pub fn factor(&self, name: &str) -> usize {
        self.factor_names.iter().position(|f| f == name).unwrap_or_else(|| panic!("no factor {name}"))
    }

    pub fn input(&self) -> FitInput<'_> {
        FitInput {
            n: self.csv.n(),
            y: self.csv.col(&self.y),
            x: self.x.iter().map(|x| self.csv.col(x)).collect(),
            factors: self.factors.iter().map(|v| FactorCol::F64(v)).collect(),
            factor_names: self.factor_names.clone(),
            fe: self.fe.clone(),
            weights: self.weights.as_ref().map(|w| self.csv.col(w)),
            offset: self.offset.as_ref().map(|o| self.csv.col(o)),
            ..Default::default()
        }
    }
}

/// Logit spec at tight tolerances (alpaca's exact control reaches the MLE).
pub fn spec(flavor: Flavor) -> FitSpec {
    FitSpec {
        model: Model::Logit,
        flavor,
        vcov: VcovKind::Iid,
        ssc: Ssc { k_adj: false, ..Ssc::default() },
        glm_tol: 1e-13,
        glm_maxiter: 100,
        fixef_tol: 1e-13,
        keep_fixef: true,
        ..FitSpec::default()
    }
}

/// Largest relative difference |a - b| / max(|b|, floor).
pub fn max_rel(a: &[f64], b: &[f64], floor: f64) -> f64 {
    assert_eq!(a.len(), b.len(), "lengths");
    a.iter().zip(b).map(|(x, y)| (x - y).abs() / y.abs().max(floor)).fold(0.0, f64::max)
}

#[track_caller]
pub fn check(what: &str, a: &[f64], b: &[f64], rtol: f64) -> f64 {
    // Matrices: relative to the largest entry (off-diagonal covariances can be ~0).
    let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let r = max_rel(a, b, 1e-12 * scale.max(1e-300));
    assert!(r <= rtol, "{what}: max rel diff {r:e} > {rtol:e}\n  got  {a:?}\n  want {b:?}");
    r
}

/// Matrix comparison relative to the largest absolute entry.
#[track_caller]
pub fn check_mat(what: &str, a: &[f64], b: &[f64], rtol: f64) -> f64 {
    let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let r = a.iter().zip(b).map(|(x, y)| (x - y).abs() / scale).fold(0.0, f64::max);
    assert!(r <= rtol, "{what}: max diff / max entry {r:e} > {rtol:e}\n  got  {a:?}\n  want {b:?}");
    r
}
