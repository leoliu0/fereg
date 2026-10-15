//! The data a fit retains for post-hoc variance matrices and fixed-effect values
//! ([`super::FitState`]): the residualized row block with the design and residual combos, the
//! bread, the weights, the row order, the fixed-effect codes and slopes (nesting tests, fixest's
//! `K.exact`, the leverage of HC2/HC3) and the solved fixed-effect values.

use std::sync::OnceLock;

use rayon::prelude::*;

use super::extra::{VcovInputs, fe_leverage};
use super::fixef::{self, FeLevels};
use super::input::{factor_missing, interaction_kept_codes};
use super::vcov::{self, ClusterInfo, FeInfo, KindVars, Model};
use super::{FitInput, FixefNorm, FixefValues, Flavor, Ssc, VcovData, VcovKind, VcovOutput};
use crate::api::Ctx;
use crate::error::RfError;
use crate::factor::Codes;
use crate::par::{self, Pool};
use crate::session::Term;

/// One fixed-effect term over the estimation rows (kept order).
pub(crate) struct StateFe {
    pub codes: Vec<u32>,
    /// Levels in the degrees of freedom (pyfixest's `feglm` counts those present before its
    /// `fe` separation check; otherwise `levels`).
    pub g: usize,
    /// Levels present: every code is below it.
    pub levels: usize,
    pub intercept: bool,
    /// Data-column names of its factors.
    pub names: Vec<String>,
    /// Slope values per row (kept order) and their indices into `FitInput::slopes`.
    pub slopes: Vec<Vec<f64>>,
    pub slope_ids: Vec<usize>,
}

impl StateFe {
    pub(crate) fn info(&self) -> FeInfo<'_> {
        FeInfo { codes: &self.codes, g: self.g, intercept: self.intercept, names: self.names.iter().map(String::as_str).collect() }
    }
}

/// The session's terms (moved) as state terms, with `g[q]` levels in the degrees of freedom.
pub(crate) fn state_terms(terms: Vec<Term>, input: &FitInput, g: &[usize]) -> Vec<StateFe> {
    terms
        .into_iter()
        .zip(&input.fe)
        .zip(g)
        .map(|((t, ft), &g)| StateFe {
            codes: t.codes,
            g,
            levels: t.g,
            intercept: t.intercept,
            names: ft.factors.iter().map(|&f| input.factor_names[f].clone()).collect(),
            slopes: t.slopes,
            slope_ids: ft.slopes.clone(),
        })
        .collect()
}

/// fixest references per term (`K.exact`): over the intercept terms, 0 for slope-only terms.
pub(crate) fn term_refs(fe: &[FeInfo], pool: &Pool) -> Vec<usize> {
    let icpt: Vec<&FeInfo> = fe.iter().filter(|t| t.intercept).collect();
    let codes: Vec<&[u32]> = icpt.iter().map(|t| t.codes).collect();
    let sizes: Vec<usize> = icpt.iter().map(|t| t.g).collect();
    let mut refs = fixef::references(&codes, &sizes, pool).into_iter();
    fe.iter().map(|t| if t.intercept { refs.next().unwrap_or(0) } else { 0 }).collect()
}

pub(crate) struct State {
    pub flavor: Flavor,
    pub n: f64,
    pub nparams: usize,
    pub bread: Vec<f64>,
    pub iid_scale: f64,
    pub normal: bool,
    /// Residualized row block (row order, `r` columns).
    pub rows: Vec<f64>,
    pub r: usize,
    /// Reported regressors as combos of the block columns (coefficient order).
    pub design: Vec<Vec<f64>>,
    /// Score residual as a combo.
    pub resid: Vec<f64>,
    /// Weights in kept order.
    pub weights: Option<Vec<f64>>,
    pub fweights: bool,
    /// Kept index of each row.
    pub row_order: Vec<u32>,
    /// Input row of each estimation row (kept order).
    pub obs: Vec<u32>,
    pub fe: Vec<StateFe>,
    /// Solved fixed-effect values (`None` without fixed-effect terms).
    pub fixef: Option<FeLevels>,
    /// IV second stage (HC2/HC3 are not defined).
    pub iv: bool,
    /// Design index of the intercept, if one is reported.
    pub intercept: Option<usize>,
    /// Leverage of the fixed-effect columns (kept order), computed by the first HC2/HC3 request.
    pub fe_hat: OnceLock<Vec<f64>>,
}

impl State {
    /// Codes over the estimation rows of the cluster dimension `dim` (factor indices into
    /// `data.factors`); a missing value on an estimation row is an error.
    fn cluster_codes(&self, data: &VcovData, dim: &[usize], pool: &Pool) -> Result<Codes, RfError> {
        for &f in dim {
            if pool.install(|| self.obs.par_iter().any(|&i| factor_missing(&data.factors[f], i as usize))) {
                return Err(RfError::InvalidInput(format!("the cluster variable {} has missing values on estimation rows", data.factor_names[f])));
            }
        }
        interaction_kept_codes(data.factors, dim, &self.obs, pool)
    }

    /// Leverage of the fixed-effect columns under the variance weights (kept order).
    fn fe_hat(&self, ctx: &Ctx, pool: &Pool) -> Result<&[f64], RfError> {
        if let Some(h) = self.fe_hat.get() {
            return Ok(h);
        }
        let terms: Vec<(&[u32], usize, bool, Vec<&[f64]>)> =
            self.fe.iter().map(|t| (t.codes.as_slice(), t.levels, t.intercept, t.slopes.iter().map(Vec::as_slice).collect())).collect();
        let h = fe_leverage(&terms, self.weights.as_deref(), self.obs.len(), ctx, pool)?;
        Ok(self.fe_hat.get_or_init(|| h))
    }

    pub(crate) fn vcov(&self, kind: VcovKind, ssc: &Ssc, data: &VcovData, ctx: &Ctx) -> Result<VcovOutput, RfError> {
        let nobs = self.obs.len();
        let rows_needed = self.obs.last().map_or(0, |&i| i as usize + 1);
        let (factors, names, cluster) = (data.factors, data.factor_names, data.cluster);
        if factors.len() != names.len()
            || factors.iter().any(|f| f.len() < rows_needed)
            || data.lat.is_some_and(|c| c.len() < rows_needed)
            || data.lon.is_some_and(|c| c.len() < rows_needed)
        {
            return Err(RfError::InvalidInput("one name per factor is required, and the variables must have the rows of the original input".into()));
        }
        if cluster.len() > 4 || cluster.iter().any(|d| d.is_empty() || d.iter().any(|&f| f >= factors.len())) {
            return Err(RfError::InvalidInput("cluster dimensions must reference between one and four valid factors".into()));
        }
        if data.time.is_some_and(|f| f >= factors.len()) || data.panel.is_some_and(|f| f >= factors.len()) {
            return Err(RfError::InvalidInput("the time or panel variable references an invalid factor".into()));
        }
        let vars = KindVars { cluster: !cluster.is_empty(), time: data.time.is_some(), panel: data.panel.is_some(), coords: data.lat.is_some() && data.lon.is_some() };
        vcov::check_kind(kind, self.flavor, self.iv, &vars)?;
        if kind == VcovKind::Crv3 {
            return Err(RfError::InvalidInput("a CRV3 variance matrix re-estimates the model: estimate it with vcov = CRV3".into()));
        }
        par::with_pool(par::threads_for(ctx.threads, nobs), |pool| {
            let dims: Vec<Codes> =
                if kind.is_clustered() { cluster.iter().map(|d| self.cluster_codes(data, d, pool)).collect::<Result<_, _>>()? } else { Vec::new() };
            let cl_info: Vec<ClusterInfo> = dims
                .iter()
                .zip(cluster)
                .map(|(c, d)| ClusterInfo { codes: &c.codes, g: c.n as usize, names: d.iter().map(|&f| names[f].as_str()).collect() })
                .collect();
            let inputs = VcovInputs::read(kind, self.flavor, data, &self.obs, pool)?;
            let mut extra = inputs.extra(!self.fe.is_empty(), self.intercept);
            if matches!(kind, VcovKind::Hc2 | VcovKind::Hc3) && !self.fe.is_empty() {
                extra.fe_hat = Some(self.fe_hat(ctx, pool)?);
            }
            let w_rows = match &self.weights {
                Some(w) => Some(crate::vcov::to_row_order(w, &self.row_order, pool)?),
                None => None,
            };
            let model = Model {
                flavor: self.flavor,
                n: self.n,
                nparams: self.nparams,
                bread: &self.bread,
                iid_scale: self.iid_scale,
                normal: self.normal,
                rows: &self.rows,
                r: self.r,
                design: &self.design,
                resid: &self.resid,
                w: w_rows.as_deref(),
                fweights: self.fweights,
                row_order: &self.row_order,
            };
            let fe_info: Vec<FeInfo> = self.fe.iter().map(StateFe::info).collect();
            let refs = || term_refs(&fe_info, pool);
            vcov::compute(&model, kind, ssc, &fe_info, &cl_info, &extra, &refs, pool).map(|(v, _)| v)
        })?
    }

    pub(crate) fn fixef(&self, norm: FixefNorm, ctx: &Ctx) -> Result<Vec<FixefValues>, RfError> {
        let Some(fe) = &self.fixef else { return Ok(Vec::new()) };
        par::with_pool(par::threads_for(ctx.threads, self.obs.len()), |pool| fixef::fixef_values(&self.fe, &self.obs, fe, norm, pool))?
    }

    pub(crate) fn fixef_references(&self, order: &[usize], ctx: &Ctx) -> Result<Vec<usize>, RfError> {
        let q = self.fe.len();
        let mut seen = vec![false; q];
        if order.len() != q || order.iter().any(|&t| t >= q || std::mem::replace(&mut seen[t], true)) {
            return Err(RfError::InvalidInput("the term order must be a permutation of the fixed-effect terms".into()));
        }
        let icpt: Vec<usize> = order.iter().copied().filter(|&t| self.fe[t].intercept).collect();
        let codes: Vec<&[u32]> = icpt.iter().map(|&t| self.fe[t].codes.as_slice()).collect();
        let sizes: Vec<usize> = icpt.iter().map(|&t| self.fe[t].levels).collect();
        let refs = par::with_pool(par::threads_for(ctx.threads, self.obs.len()), |pool| fixef::references(&codes, &sizes, pool))?;
        let mut out = vec![0; q];
        icpt.iter().zip(refs).for_each(|(&t, r)| out[t] = r);
        Ok(out)
    }
}
