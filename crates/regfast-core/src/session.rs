//! Estimation session: sample fixing, compaction and the absorber (plan §1.1).
//!
//! `Session::prepare` encodes factors, removes singletons (and GLM `fe`-separated groups),
//! compacts every per-observation array to the kept sample, and builds the [`Absorber`].
//! Estimators (`ols`, `iv`, `glm`) are implemented in their own modules as `impl Session`.

use crate::absorber::Absorber;
use crate::api::{Ctx, Family, FactorValues, Prep, Vce, WeightKind};
use crate::error::RfError;
use crate::factor::{self, Codes};
use crate::gather;
use crate::par::{self, Pool};
use crate::reference::DenseAbsorber;
use crate::sample::{self, SepRule, TermCodes};

/// One absorbed term restricted to the kept sample.
#[derive(Clone, Debug, PartialEq)]
pub struct Term {
    /// Sorted, deduplicated factor indices (into `Prep::factors`).
    pub factors: Vec<usize>,
    /// Dense codes per kept observation (kept order).
    pub codes: Vec<u32>,
    /// Number of levels present in the kept sample.
    pub g: usize,
    pub intercept: bool,
    /// Slope values per kept observation (kept order), one vector per slope.
    pub slopes: Vec<Vec<f64>>,
    /// Indices of the slopes into `Prep::slope_vars`.
    pub slope_vars: Vec<usize>,
    pub save: bool,
}

/// One cluster dimension restricted to the kept sample.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClusterDim {
    /// Sorted, deduplicated factor indices (into `Prep::factors`).
    pub factors: Vec<usize>,
    pub codes: Vec<u32>,
    pub g: usize,
}

/// Which absorber implementation backs the session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbsorberKind {
    /// The production solver (plan D5).
    Fast,
    /// Explicit dummies; small problems only (tests, debugging).
    Dense,
}

/// What [`Session::drop_obs`] removed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DropReport {
    pub requested: u64,
    pub singletons: u64,
    pub fe_separated: u64,
}

pub struct Session {
    pub(crate) n_input: usize,
    pub(crate) keep: Vec<bool>,
    pub(crate) kept_index: Vec<u32>,
    pub(crate) terms: Vec<Term>,
    pub(crate) clusters: Vec<ClusterDim>,
    pub(crate) vce: Vce,
    pub(crate) weight_kind: WeightKind,
    /// Raw weights per kept observation (kept order).
    pub(crate) weights: Option<Vec<f64>>,
    /// GLM depvar per kept observation (kept order), for re-running the `fe` rule.
    pub(crate) y_sep: Option<Vec<f64>>,
    pub(crate) family: Family,
    pub(crate) drop_singletons: bool,
    pub(crate) sep_fe: bool,
    pub(crate) num_singletons: u64,
    pub(crate) num_fe_separated: u64,
    pub(crate) kind: AbsorberKind,
    pub(crate) absorber: Box<dyn Absorber>,
}

impl Session {
    /// Builds a session with the production absorber.
    pub fn prepare(p: Prep, ctx: &Ctx) -> Result<Session, RfError> {
        Self::prepare_with(p, ctx, AbsorberKind::Fast)
    }

    pub fn prepare_with(p: Prep, ctx: &Ctx, kind: AbsorberKind) -> Result<Session, RfError> {
        validate(&p)?;
        par::with_pool(par::threads_for(ctx.threads, p.n), |pool| prepare_in_pool(&p, ctx, kind, pool))?
    }

    /// Kept observations.
    pub fn n(&self) -> usize {
        self.kept_index.len()
    }

    /// Input observations given to `prepare`.
    pub fn n_input(&self) -> usize {
        self.n_input
    }

    /// Keep mask over the input observations.
    pub fn keep(&self) -> &[bool] {
        &self.keep
    }

    /// Input index of each kept observation (ascending).
    pub fn kept_index(&self) -> &[u32] {
        &self.kept_index
    }

    /// Kept-observation index of each solver row; estimator data blocks use this order.
    pub fn row_order(&self) -> &[u32] {
        self.absorber.row_order()
    }

    /// True when at least half of consecutive rows map to kept observations at most 8 apart,
    /// i.e. reading the source data in row order is close to sequential.
    pub fn row_order_is_local(&self) -> bool {
        gather::is_local(self.row_order())
    }

    /// Kept-order columns → row-major block in [`Session::row_order`].
    pub fn gather(&self, cols: &[&[f64]], ctx: &Ctx) -> Result<Vec<f64>, RfError> {
        for c in cols {
            if c.len() != self.n() {
                return Err(RfError::InvalidInput("gather: column length differs from the kept sample".into()));
            }
        }
        par::with_pool(par::threads_for(ctx.threads, self.n()), |pool| gather::gather_rows(self.row_order(), cols, pool))?
    }

    /// Kept-order row-major block (R columns) → row-major block in [`Session::row_order`].
    pub fn permute_rows(&self, kept_rows: Vec<f64>, r: usize, ctx: &Ctx) -> Result<Vec<f64>, RfError> {
        if kept_rows.len() != self.n() * r {
            return Err(RfError::InvalidInput("permute_rows: block size differs from the kept sample".into()));
        }
        par::with_pool(par::threads_for(ctx.threads, self.n()), |pool| gather::permute_rows(self.row_order(), &kept_rows, r, pool))?
    }

    pub fn num_singletons(&self) -> u64 {
        self.num_singletons
    }

    /// Observations removed by the GLM `fe` separation rule.
    pub fn num_fe_separated(&self) -> u64 {
        self.num_fe_separated
    }

    pub fn terms(&self) -> &[Term] {
        &self.terms
    }

    pub fn clusters(&self) -> &[ClusterDim] {
        &self.clusters
    }

    pub fn vce(&self) -> &Vce {
        &self.vce
    }

    pub fn weight_kind(&self) -> WeightKind {
        self.weight_kind
    }

    /// Raw weights per kept observation (kept order).
    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    pub fn family(&self) -> Family {
        self.family
    }

    pub fn absorber_kind(&self) -> AbsorberKind {
        self.kind
    }

    /// The absorbed space contains the constant (`noabsorb`, or some term has an intercept).
    pub fn has_intercept(&self) -> bool {
        self.terms.is_empty() || self.terms.iter().any(|t| t.intercept)
    }

    /// Low-level access: residualize arbitrary row-order blocks against the absorbed FEs.
    pub fn absorber(&self) -> &dyn Absorber {
        self.absorber.as_ref()
    }

    pub fn absorber_mut(&mut self) -> &mut dyn Absorber {
        self.absorber.as_mut()
    }

    /// Drops the kept observations flagged in `drop` (kept order), then re-runs singleton removal
    /// and the `fe` rule, compacts, and rebuilds the absorber. Row order may change.
    pub fn drop_obs(&mut self, drop: &[bool], ctx: &Ctx) -> Result<DropReport, RfError> {
        if drop.len() != self.n() {
            return Err(RfError::InvalidInput("drop_obs: mask length differs from the kept sample".into()));
        }
        par::with_pool(par::threads_for(ctx.threads, self.n()), |pool| self.drop_obs_in_pool(drop, ctx, pool))?
    }

    fn drop_obs_in_pool(&mut self, drop: &[bool], ctx: &Ctx, pool: &Pool) -> Result<DropReport, RfError> {
        let mut alive: Vec<bool> = drop.iter().map(|d| !d).collect();
        let requested = drop.iter().filter(|d| **d).count() as u64;
        let term_codes: Vec<TermCodes> = self.terms.iter().map(|t| TermCodes { codes: &t.codes, g: t.g }).collect();
        let fw = (self.weight_kind == WeightKind::Fweight).then(|| self.weights.as_deref()).flatten();
        let sep = (self.sep_fe).then(|| self.y_sep.as_deref().map(|y| SepRule { y, family: self.family })).flatten();
        let counts = sample::drop_singletons(&term_codes, &mut alive, fw, sep, self.drop_singletons, pool);
        let kept = alive.iter().filter(|a| **a).count();
        if kept < 2 {
            return Err(RfError::TooFewObs);
        }
        let mut new_index = Vec::with_capacity(kept);
        for (k, &a) in alive.iter().enumerate() {
            if a {
                new_index.push(self.kept_index[k]);
            } else {
                self.keep[self.kept_index[k] as usize] = false;
            }
        }
        for t in &mut self.terms {
            let c = factor::compact(&Codes { codes: std::mem::take(&mut t.codes), n: t.g as u32 }, &alive);
            t.codes = c.codes;
            t.g = c.n as usize;
            for s in &mut t.slopes {
                *s = filter(s, &alive);
            }
        }
        for cd in &mut self.clusters {
            let c = factor::compact(&Codes { codes: std::mem::take(&mut cd.codes), n: cd.g as u32 }, &alive);
            cd.codes = c.codes;
            cd.g = c.n as usize;
        }
        self.weights = self.weights.as_ref().map(|w| filter(w, &alive));
        self.y_sep = self.y_sep.as_ref().map(|y| filter(y, &alive));
        self.kept_index = new_index;
        self.num_singletons += counts.singletons;
        self.num_fe_separated += counts.fe_separated;
        ctx.check()?;
        self.absorber = build_absorber(self.kind, &self.terms, self.weights.as_deref(), self.n(), pool, ctx)?;
        Ok(DropReport { requested, singletons: counts.singletons, fe_separated: counts.fe_separated })
    }
}

fn filter(v: &[f64], alive: &[bool]) -> Vec<f64> {
    v.iter().zip(alive).filter(|(_, a)| **a).map(|(x, _)| *x).collect()
}

fn validate(p: &Prep) -> Result<(), RfError> {
    let bad = |m: &str| Err(RfError::InvalidInput(m.to_string()));
    if p.n >= u32::MAX as usize {
        return bad("too many observations (limit 4294967294)");
    }
    if p.factors.iter().any(|f| f.len() != p.n) || p.slope_vars.iter().any(|s| s.len() != p.n) {
        return bad("factor or slope variable length differs from n");
    }
    for t in &p.absorb {
        if t.factors.is_empty() || t.factors.iter().any(|&f| f >= p.factors.len()) {
            return bad("absorb term references an invalid factor");
        }
        if t.slopes.iter().any(|&s| s >= p.slope_vars.len()) {
            return bad("absorb term references an invalid slope variable");
        }
        if !t.intercept && t.slopes.is_empty() {
            return bad("absorb term has neither intercept nor slopes");
        }
    }
    if let Vce::Cluster(dims) = &p.vce {
        if dims.is_empty() || dims.iter().any(|d| d.is_empty() || d.iter().any(|&f| f >= p.factors.len())) {
            return bad("cluster dimension references an invalid factor");
        }
    }
    match (p.weight_kind, p.weights) {
        (WeightKind::None, None) => {}
        (WeightKind::None, Some(_)) | (_, None) => return bad("weights and weight kind disagree"),
        (_, Some(w)) => {
            if w.len() != p.n {
                return bad("weight length differs from n");
            }
            if w.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
                return bad("weights must be positive and finite");
            }
        }
    }
    if p.slope_vars.iter().any(|s| s.iter().any(|v| !v.is_finite())) {
        return bad("missing values in slope variable");
    }
    if p.sep_fe && p.y_for_sep.is_none_or(|y| y.len() != p.n) {
        return bad("fe separation needs the depvar");
    }
    Ok(())
}

fn sorted_set(v: &[usize]) -> Vec<usize> {
    let mut s = v.to_vec();
    s.sort_unstable();
    s.dedup();
    s
}

fn prepare_in_pool(p: &Prep, ctx: &Ctx, kind: AbsorberKind, pool: &Pool) -> Result<Session, RfError> {
    let mut t = std::time::Instant::now();
    let codes: Vec<Codes> = p.factors.iter().map(|f: &FactorValues| factor::factorize(f, pool)).collect::<Result<_, _>>()?;
    crate::solve::log_phase(ctx, &mut t, "factorize");
    ctx.check()?;
    let interact = |fs: &[usize]| -> Result<Codes, RfError> {
        let parts: Vec<&Codes> = fs.iter().map(|&f| &codes[f]).collect();
        factor::combine(&parts, pool)
    };
    let term_codes: Vec<Codes> = p.absorb.iter().map(|t| interact(&t.factors)).collect::<Result<_, _>>()?;
    let cluster_codes: Vec<(Vec<usize>, Codes)> = match &p.vce {
        Vce::Cluster(dims) => dims.iter().map(|d| Ok((sorted_set(d), interact(d)?))).collect::<Result<_, RfError>>()?,
        _ => Vec::new(),
    };
    ctx.check()?;

    let mut alive = vec![true; p.n];
    let tc: Vec<TermCodes> = term_codes.iter().map(|c| TermCodes { codes: &c.codes, g: c.n as usize }).collect();
    let fw = (p.weight_kind == WeightKind::Fweight).then_some(p.weights).flatten();
    let sep = p.sep_fe.then(|| SepRule { y: p.y_for_sep.expect("validated"), family: p.family });
    let counts = sample::drop_singletons(&tc, &mut alive, fw, sep, p.drop_singletons, pool);
    crate::solve::log_phase(ctx, &mut t, "singletons");
    let kept_index: Vec<u32> = (0..p.n as u32).filter(|&i| alive[i as usize]).collect();
    if kept_index.len() < 2 {
        return Err(RfError::TooFewObs);
    }
    ctx.check()?;

    let terms: Vec<Term> = p
        .absorb
        .iter()
        .zip(&term_codes)
        .map(|(t, c)| {
            let c = factor::compact(c, &alive);
            Term {
                factors: sorted_set(&t.factors),
                codes: c.codes,
                g: c.n as usize,
                intercept: t.intercept,
                slopes: t.slopes.iter().map(|&s| filter(p.slope_vars[s], &alive)).collect(),
                slope_vars: t.slopes.clone(),
                save: t.save,
            }
        })
        .collect();
    let clusters: Vec<ClusterDim> = cluster_codes
        .into_iter()
        .map(|(factors, c)| {
            let c = factor::compact(&c, &alive);
            ClusterDim { factors, codes: c.codes, g: c.n as usize }
        })
        .collect();
    let weights = p.weights.map(|w| filter(w, &alive));
    let y_sep = p.y_for_sep.map(|y| filter(y, &alive));
    let n = kept_index.len();
    crate::solve::log_phase(ctx, &mut t, "compact");
    let absorber = build_absorber(kind, &terms, weights.as_deref(), n, pool, ctx)?;
    Ok(Session {
        n_input: p.n,
        keep: alive,
        kept_index,
        terms,
        clusters,
        vce: p.vce.clone(),
        weight_kind: p.weight_kind,
        weights,
        y_sep,
        family: p.family,
        drop_singletons: p.drop_singletons,
        sep_fe: p.sep_fe,
        num_singletons: counts.singletons,
        num_fe_separated: counts.fe_separated,
        kind,
        absorber,
    })
}

fn build_absorber(
    kind: AbsorberKind,
    terms: &[Term],
    weights: Option<&[f64]>,
    n: usize,
    pool: &Pool,
    ctx: &Ctx,
) -> Result<Box<dyn Absorber>, RfError> {
    match kind {
        AbsorberKind::Dense => Ok(Box::new(DenseAbsorber::new(terms, weights, n)?)),
        AbsorberKind::Fast => crate::solve::build(terms, weights, n, pool, ctx),
    }
}
