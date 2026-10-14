//! Inputs of the HC2/HC3, Newey-West, Driscoll-Kraay, Conley and CRV3 variance matrices over the
//! estimation rows, shared by the fit-time variance matrices and [`super::FitState::vcov`]: the
//! time / panel codes and coordinates read from input-length columns, the leverage of the fixed
//! effects, and the leave-one-cluster-out jackknife.

use rayon::prelude::*;

use super::input::{factor_missing, interaction_kept_codes};
use super::vcov::{self as fvcov, Extra};
use super::{Flavor, Model, VcovData, VcovKind};
use crate::api::Ctx;
use crate::crv3::{self, JkModel, JkProblem};
use crate::error::RfError;
use crate::factor::Codes;
use crate::leverage::{self, FeCols};
use crate::par::Pool;

/// Time / panel codes and coordinates over the estimation rows (kept order) that `kind` reads.
#[derive(Default)]
pub(crate) struct VcovInputs {
    pub time: Option<Codes>,
    pub panel: Option<Codes>,
    pub coords: Option<(Vec<f64>, Vec<f64>)>,
}

impl VcovInputs {
    /// Reads them at the input rows `rows` (kept order). A missing value is an error (fixest's
    /// message in its flavor): the fit drops or rejects such rows, a post-hoc request cannot.
    pub(crate) fn read(kind: VcovKind, flavor: Flavor, data: &VcovData, rows: &[u32], pool: &Pool) -> Result<VcovInputs, RfError> {
        let missing = |name: &str, label: &str| {
            RfError::InvalidInput(match flavor {
                Flavor::Fixest => fvcov::fixest_na_message(name, label),
                Flavor::Pyfixest => format!("the {label} variable {name} has missing values on estimation rows"),
            })
        };
        let codes = |f: usize, label: &str| -> Result<Codes, RfError> {
            if pool.install(|| rows.par_iter().any(|&i| factor_missing(&data.factors[f], i as usize))) {
                return Err(missing(&data.factor_names[f], label));
            }
            interaction_kept_codes(data.factors, &[f], rows, pool)
        };
        let mut out = VcovInputs::default();
        match kind {
            VcovKind::NeweyWest { .. } | VcovKind::DriscollKraay { .. } => {
                let nw = matches!(kind, VcovKind::NeweyWest { .. });
                if let Some(p) = data.panel.filter(|_| nw || flavor == Flavor::Pyfixest) {
                    out.panel = Some(codes(p, "panel ID")?);
                }
                if let Some(t) = data.time {
                    out.time = Some(codes(t, "time")?);
                }
            }
            VcovKind::Conley { .. } => {
                let read = |c: &[f64], label: &str| -> Result<Vec<f64>, RfError> {
                    let v: Vec<f64> = pool.install(|| rows.par_iter().map(|&i| c[i as usize]).collect());
                    if v.iter().any(|x| !x.is_finite()) {
                        return Err(missing(label, label));
                    }
                    Ok(v)
                };
                if let (Some(lat), Some(lon)) = (data.lat, data.lon) {
                    out.coords = Some((read(lat, "latitude")?, read(lon, "longitude")?));
                }
            }
            _ => {}
        }
        Ok(out)
    }

    /// The compute-side view, without the leverage and the jackknife.
    pub(crate) fn extra(&self, has_fe: bool, intercept: Option<usize>) -> Extra<'_> {
        Extra {
            time: self.time.as_ref().map(|c| (c.codes.as_slice(), c.n as usize)),
            panel: self.panel.as_ref().map(|c| (c.codes.as_slice(), c.n as usize)),
            coords: self.coords.as_ref().map(|(a, b)| (a.as_slice(), b.as_slice())),
            has_fe,
            intercept,
            ..Default::default()
        }
    }
}

/// Exact leverage of the fixed-effect columns (intercept dummies and slopes of every term), kept
/// order: `terms` are (codes, levels, intercept, slope values), `w` the weights of the variance
/// matrix (IRLS weights for GLMs).
pub(crate) fn fe_leverage(terms: &[(&[u32], usize, bool, Vec<&[f64]>)], w: Option<&[f64]>, n: usize, ctx: &Ctx, pool: &Pool) -> Result<Vec<f64>, RfError> {
    let cols: Vec<FeCols> = terms.iter().map(|(codes, g, intercept, slopes)| FeCols { codes, g: *g, intercept: *intercept, slopes: slopes.clone() }).collect();
    leverage::fe_hat(&cols, w, n, ctx, pool)
}

/// Kept-order values of an input column at the input rows `rows`.
pub(crate) fn kept_values(col: &[f64], rows: &[u32], pool: &Pool) -> Vec<f64> {
    pool.install(|| rows.par_iter().map(|&i| col[i as usize]).collect())
}

/// The model of a leave-one-cluster-out refit (`Linear` is 2SLS when the problem has endogenous
/// regressors).
pub(crate) fn jk_model(model: Model) -> JkModel {
    match model {
        Model::Ols => JkModel::Linear,
        Model::Poisson => JkModel::Poisson,
        Model::Logit => JkModel::Logit,
    }
}

/// CRV3 jackknife meat of one cluster code vector: leave-one-cluster-out coefficients
/// ([`crv3::leave_cluster_out`]) centered at the full-sample coefficients (pyfixest).
pub(crate) fn jackknife_meat(p: &JkProblem, codes: &[u32], g: usize, ctx: &Ctx, pool: &Pool) -> Result<Vec<f64>, RfError> {
    let betas = crv3::leave_cluster_out(p, codes, g, ctx, pool)?;
    Ok(crv3::jackknife_meat(&betas, p.beta, p.beta.len()))
}

/// The estimation rows of a CRV3 refit problem (kept order): owned columns, borrowed
/// fixed-effect terms.
pub(crate) struct JkData<'a> {
    pub y: Vec<f64>,
    pub x: Vec<Vec<f64>>,
    pub endog: Vec<Vec<f64>>,
    pub inst: Vec<Vec<f64>>,
    pub weights: Option<Vec<f64>>,
    pub offset: Option<Vec<f64>>,
    pub fe: Vec<FeCols<'a>>,
}

impl<'a> JkData<'a> {
    /// Columns `y`, `x`, `endog`, `inst`, `offset` (input length) at the input rows `rows`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        y: &[f64],
        x: &[&[f64]],
        endog: &[&[f64]],
        inst: &[&[f64]],
        offset: Option<&[f64]>,
        weights: Option<&[f64]>,
        fe: Vec<FeCols<'a>>,
        rows: &[u32],
        pool: &Pool,
    ) -> JkData<'a> {
        let kv = |c: &[f64]| kept_values(c, rows, pool);
        JkData {
            y: kv(y),
            x: x.iter().map(|c| kv(c)).collect(),
            endog: endog.iter().map(|c| kv(c)).collect(),
            inst: inst.iter().map(|c| kv(c)).collect(),
            weights: weights.map(<[f64]>::to_vec),
            offset: offset.map(kv),
            fe,
        }
    }

    pub(crate) fn problem<'b>(&'b self, model: JkModel, intercept: bool, beta: &'b [f64], spec: &super::FitSpec) -> JkProblem<'b> {
        JkProblem {
            model,
            n: self.y.len(),
            y: &self.y,
            x: self.x.iter().map(Vec::as_slice).collect(),
            endog: self.endog.iter().map(Vec::as_slice).collect(),
            inst: self.inst.iter().map(Vec::as_slice).collect(),
            intercept,
            weights: self.weights.as_deref(),
            offset: self.offset.as_deref(),
            fe: self.fe.iter().map(|f| FeCols { codes: f.codes, g: f.g, intercept: f.intercept, slopes: f.slopes.clone() }).collect(),
            beta,
            fixef_tol: spec.fixef_tol,
            fixef_maxiter: spec.fixef_maxiter,
            glm_tol: spec.glm_tol,
            glm_maxiter: spec.glm_maxiter,
            collin_tol: spec.collin_tol,
        }
    }
}
