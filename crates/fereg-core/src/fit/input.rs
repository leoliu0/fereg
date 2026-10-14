//! Estimation sample of the fit layer: input validation, missing values, weights, factor codes,
//! singleton / perfect-fit removal, and the [`Session`] over the kept rows.

use rayon::prelude::*;

use super::{Dropped, FactorCol, FitInput, FitSpec, FixefRm, Flavor, Model, NaCounts, VcovKind};
use crate::api::{AbsorbTerm, Ctx, Family, Vce, WeightKind};
use crate::error::RfError;
use crate::factor::{self, Codes};
use crate::par::{CHUNK, Pool};
use crate::sample::{self, SepRule, TermCodes};
use crate::session::{AbsorberKind, Parts, Session, term_and_cluster_codes};

/// Bits of the per-row missing mask, one per [`NaCounts`] group.
const Y: u16 = 1;
const X: u16 = 1 << 1;
const ENDOG: u16 = 1 << 2;
const INST: u16 = 1 << 3;
const FE: u16 = 1 << 4;
const CLUSTER: u16 = 1 << 5;
const WEIGHTS: u16 = 1 << 6;
const OFFSET: u16 = 1 << 7;
const EXTRA: u16 = 1 << 8;
/// Time, panel, latitude or longitude of the requested variance matrix (counted with `CLUSTER`).
const VCOV: u16 = 1 << 9;
const GROUPS: usize = 9;

/// The kept sample and its session.
pub(crate) struct Sample {
    pub session: Session,
    pub dropped: Dropped,
    /// Pyfixest flavor with its `fe` separation check: levels per intercept term before that
    /// check (pyfixest's `feglm` keeps these counts in its degrees of freedom).
    pub levels_before_separation: Option<Vec<usize>>,
    /// Pyfixest-flavor logit: the rows of the reported sample left out of the session.
    pub excluded: Option<Excluded>,
}

/// Rows of the reported sample left out of the session (Pyfixest-flavor logit): those of the
/// levels whose remaining outcomes are all 0 or all 1 (recursively, as fixest's perfect-fit
/// rule). pyfixest keeps them, but their linear predictor diverges in its IRLS and they carry
/// no information on beta: IRLS runs without them (exactly the limit pyfixest approaches) and the
/// outputs report them at the bound.
pub(crate) struct Excluded {
    /// Input rows left out (ascending).
    pub rows: Vec<u32>,
    /// Input rows of the reported sample before any GLM-internal drop (ascending), and the
    /// codes over them of every term and cluster dimension.
    pub reported: Vec<u32>,
    pub terms: Vec<Codes>,
    pub clusters: Vec<Codes>,
}

pub(crate) fn validate(input: &FitInput, spec: &FitSpec) -> Result<(), RfError> {
    let bad = |m: String| Err(RfError::InvalidInput(m));
    let n = input.n;
    if n >= u32::MAX as usize {
        return bad("too many observations (limit 4294967294)".into());
    }
    let cols = std::iter::once(input.y).chain(input.x.iter().copied()).chain(input.endog.iter().copied()).chain(input.inst.iter().copied());
    if cols.chain(input.slopes.iter().copied()).chain(input.weights).chain(input.offset).any(|c| c.len() != n)
        || input.factors.iter().any(|f| f.len() != n)
        || input.drop.is_some_and(|d| d.len() != n)
    {
        return bad("every input column must have n rows".into());
    }
    if input.factor_names.len() != input.factors.len() {
        return bad("one name per factor is required".into());
    }
    if input.fe.len() > 255 {
        return bad("at most 255 fixed-effect terms are supported".into());
    }
    for t in &input.fe {
        if t.factors.is_empty() || t.factors.iter().any(|&f| f >= input.factors.len()) {
            return bad("a fixed-effect term references an invalid factor".into());
        }
        if t.slopes.iter().any(|&s| s >= input.slopes.len()) {
            return bad("a fixed-effect term references an invalid slope variable".into());
        }
        if !t.intercept && t.slopes.is_empty() {
            return bad("a fixed-effect term has neither an intercept nor slopes".into());
        }
    }
    if input.cluster.iter().any(|d| d.is_empty() || d.iter().any(|&f| f >= input.factors.len())) {
        return bad("a cluster dimension references an invalid factor".into());
    }
    if input.cluster.len() > 4 {
        return bad("at most four cluster dimensions are supported".into());
    }
    if input.time.is_some_and(|f| f >= input.factors.len()) || input.panel.is_some_and(|f| f >= input.factors.len()) {
        return bad("the time or panel variable references an invalid factor".into());
    }
    if input.lat.is_some_and(|c| c.len() != n) || input.lon.is_some_and(|c| c.len() != n) {
        return bad("every input column must have n rows".into());
    }
    let vars = super::vcov::KindVars {
        cluster: !input.cluster.is_empty(),
        time: input.time.is_some(),
        panel: input.panel.is_some(),
        coords: input.lat.is_some() && input.lon.is_some(),
    };
    super::vcov::check_kind(spec.vcov, spec.flavor, !input.endog.is_empty(), &vars)?;
    if spec.model == Model::Ols && input.offset.is_some() {
        return bad("an offset is only supported for Poisson and logit models".into());
    }
    if !input.endog.is_empty() && spec.model != Model::Ols {
        return bad("instrumental variables are only supported for linear models".into());
    }
    if input.inst.len() < input.endog.len() || (input.endog.is_empty() && !input.inst.is_empty()) {
        return bad("IV estimation needs at least as many instruments as endogenous regressors".into());
    }
    if !(spec.collin_tol >= 0.0) || !(spec.fixef_tol > 0.0) || spec.fixef_maxiter == 0 {
        return bad("invalid tolerance or iteration limit".into());
    }
    Ok(())
}

/// Whether row `i` of a factor column is missing.
pub(crate) fn factor_missing(col: &FactorCol, i: usize) -> bool {
    match *col {
        FactorCol::F64(v) => !v[i].is_finite(),
        FactorCol::I32(v) => v[i] == i32::MIN,
        FactorCol::I64(v) => v[i] == i64::MIN,
        FactorCol::Codes { codes, nlevels, base } => !(0..i64::from(nlevels)).contains(&(i64::from(codes[i]) - i64::from(base))),
    }
}

/// Dense codes of a factor column over the rows `kept` (ascending input rows, none missing).
pub(crate) fn factor_kept_codes(col: &FactorCol, kept: &[u32], pool: &Pool) -> Result<Codes, RfError> {
    factor::restrict(&factor::factorize_col(col, pool)?, kept, pool)
}

/// Dense codes over the input rows `kept` of the interaction of the factor columns `dim`
/// (combined in `dim` order, as the session's terms and clusters).
pub(crate) fn interaction_kept_codes(factors: &[FactorCol], dim: &[usize], kept: &[u32], pool: &Pool) -> Result<Codes, RfError> {
    let mut parts: Vec<Codes> = dim.iter().map(|&f| factor_kept_codes(&factors[f], kept, pool)).collect::<Result<_, _>>()?;
    if parts.len() == 1 {
        return Ok(parts.pop().expect("one part"));
    }
    factor::combine(&parts.iter().collect::<Vec<_>>(), pool)
}

/// Missing entries of a factor column within `range`, as `bit` in `mask`.
fn mark_factor(col: &FactorCol, range: std::ops::Range<usize>, mask: &mut [u16], bit: u16) {
    mask.iter_mut().zip(range).for_each(|(m, i)| *m |= if factor_missing(col, i) { bit } else { 0 });
}

fn mark_values(v: &[f64], range: std::ops::Range<usize>, mask: &mut [u16], bit: u16) {
    mask.iter_mut().zip(&v[range]).for_each(|(m, x)| *m |= if x.is_finite() { 0 } else { bit });
}

#[derive(Default)]
struct ChunkCounts {
    groups: [usize; GROUPS],
    na: usize,
    zero_weight: usize,
    negative_weight: bool,
    /// Rows missing only their cluster value (Pyfixest flavor: an error).
    cluster_only: bool,
    /// Rows missing only variance-matrix variables, at least one of time / panel / coordinates.
    vcov_only: bool,
}

/// The time / panel factors and the coordinates the requested variance matrix reads (fixest
/// drops rows missing them at estimation; a variable the request does not use is not checked).
struct VcovVars<'a> {
    factors: Vec<(usize, &'static str)>,
    coords: Vec<(&'a [f64], &'static str)>,
}

fn vcov_vars<'a>(input: &FitInput<'a>, spec: &FitSpec) -> VcovVars<'a> {
    let mut factors = Vec::new();
    let mut coords = Vec::new();
    match spec.vcov {
        VcovKind::NeweyWest { .. } | VcovKind::DriscollKraay { .. } => {
            let nw = matches!(spec.vcov, VcovKind::NeweyWest { .. });
            // pyfixest's Driscoll-Kraay reads the panel variable for its duplicate check.
            if let Some(p) = input.panel.filter(|_| nw || spec.flavor == Flavor::Pyfixest) {
                factors.push((p, "panel ID"));
            }
            factors.extend(input.time.map(|t| (t, "time")));
        }
        VcovKind::Conley { .. } => {
            coords.extend(input.lat.map(|c| (c, "latitude")));
            coords.extend(input.lon.map(|c| (c, "longitude")));
        }
        _ => {}
    }
    VcovVars { factors, coords }
}

/// Missing-value and zero-weight scan: returns the keep mask over the input rows and the counts.
fn scan(input: &FitInput, spec: &FitSpec, pool: &Pool) -> Result<(Vec<bool>, Dropped), RfError> {
    let flavor = spec.flavor;
    let vv = vcov_vars(input, spec);
    let n = input.n;
    let mut fe_factors: Vec<usize> = input.fe.iter().flat_map(|t| t.factors.iter().copied()).collect();
    fe_factors.sort_unstable();
    fe_factors.dedup();
    let mut fe_slopes: Vec<usize> = input.fe.iter().flat_map(|t| t.slopes.iter().copied()).collect();
    fe_slopes.sort_unstable();
    fe_slopes.dedup();
    let mut cl_factors: Vec<usize> = input.cluster.iter().flatten().copied().collect();
    cl_factors.sort_unstable();
    cl_factors.dedup();
    let mut alive = crate::par::alloc_par(n, false, pool)?;
    let parts: Vec<ChunkCounts> = pool.install(|| {
        alive
            .par_chunks_mut(CHUNK)
            .enumerate()
            .map(|(c, al)| {
                let range = c * CHUNK..c * CHUNK + al.len();
                let mut mask = vec![0u16; al.len()];
                mark_values(input.y, range.clone(), &mut mask, Y);
                input.x.iter().for_each(|v| mark_values(v, range.clone(), &mut mask, X));
                input.endog.iter().for_each(|v| mark_values(v, range.clone(), &mut mask, ENDOG));
                input.inst.iter().for_each(|v| mark_values(v, range.clone(), &mut mask, INST));
                fe_factors.iter().for_each(|&f| mark_factor(&input.factors[f], range.clone(), &mut mask, FE));
                fe_slopes.iter().for_each(|&s| mark_values(input.slopes[s], range.clone(), &mut mask, FE));
                cl_factors.iter().for_each(|&f| mark_factor(&input.factors[f], range.clone(), &mut mask, CLUSTER));
                vv.factors.iter().for_each(|&(f, _)| mark_factor(&input.factors[f], range.clone(), &mut mask, VCOV));
                vv.coords.iter().for_each(|&(c, _)| mark_values(c, range.clone(), &mut mask, VCOV));
                if let Some(w) = input.weights {
                    mark_values(w, range.clone(), &mut mask, WEIGHTS);
                }
                if let Some(o) = input.offset {
                    mark_values(o, range.clone(), &mut mask, OFFSET);
                }
                if let Some(d) = input.drop {
                    mask.iter_mut().zip(&d[range.clone()]).for_each(|(m, &x)| *m |= if x { EXTRA } else { 0 });
                }
                let mut out = ChunkCounts::default();
                for (g, count) in out.groups.iter_mut().enumerate() {
                    let bits = if 1 << g == CLUSTER { CLUSTER | VCOV } else { 1 << g };
                    *count = mask.iter().filter(|&&m| m & bits != 0).count();
                }
                let w = input.weights.map(|w| &w[range.clone()]);
                for (k, (a, &m)) in al.iter_mut().zip(&mask).enumerate() {
                    let wk = w.map_or(1.0, |w| w[k]);
                    out.negative_weight |= wk < 0.0;
                    if m != 0 {
                        out.na += 1;
                        out.cluster_only |= m == CLUSTER;
                        out.vcov_only |= m & VCOV != 0 && m & !(CLUSTER | VCOV) == 0;
                    } else if wk == 0.0 {
                        out.zero_weight += 1;
                    } else {
                        *a = true;
                    }
                }
                out
            })
            .collect()
    });
    let mut d = Dropped::default();
    let mut g = [0usize; GROUPS];
    let (mut negative, mut cluster_only, mut vcov_only) = (false, false, false);
    for p in &parts {
        d.na += p.na;
        d.zero_weight += p.zero_weight;
        negative |= p.negative_weight;
        cluster_only |= p.cluster_only;
        vcov_only |= p.vcov_only;
        g.iter_mut().zip(&p.groups).for_each(|(a, b)| *a += b);
    }
    if negative {
        return Err(RfError::InvalidInput("weights must not be negative".into()));
    }
    if flavor == Flavor::Pyfixest && cluster_only {
        let name = cluster_missing_name(input).unwrap_or_default();
        return Err(RfError::InvalidInput(format!(
            "missing values in the cluster variable {name}: clustered inference does not support missing cluster values"
        )));
    }
    if flavor == Flavor::Pyfixest && vcov_only {
        let n = input.n;
        let name = vv
            .factors
            .iter()
            .find(|&&(f, _)| (0..n).any(|i| factor_missing(&input.factors[f], i)))
            .map(|&(f, label)| format!("the {label} variable {}", input.factor_names[f]))
            .or_else(|| vv.coords.iter().find(|(c, _)| c.iter().any(|x| !x.is_finite())).map(|(_, label)| format!("the {label}")))
            .unwrap_or_default();
        return Err(RfError::InvalidInput(format!("missing values in {name}: the variance matrix does not support missing values")));
    }
    d.na_by = NaCounts { y: g[0], x: g[1], endog: g[2], inst: g[3], fe: g[4], cluster: g[5], weights: g[6], offset: g[7], extra: g[8] };
    Ok((alive, d))
}

/// Name of the first cluster dimension with a missing value (error messages; serial).
fn cluster_missing_name(input: &FitInput) -> Option<String> {
    input.cluster.iter().find_map(|dim| {
        let mut mask = vec![0u16; input.n];
        dim.iter().for_each(|&f| mark_factor(&input.factors[f], 0..input.n, &mut mask, CLUSTER));
        mask.contains(&CLUSTER).then(|| dim.iter().map(|&f| input.factor_names[f].as_str()).collect::<Vec<_>>().join("^"))
    })
}

/// Builds the estimation sample: missing values and zero weights, then singleton and (GLM)
/// perfect-fit removal over the intercept terms (fixest skips slope-only terms), then the session.
pub(crate) fn build(input: &FitInput, spec: &FitSpec, ctx: &Ctx, pool: &Pool) -> Result<Sample, RfError> {
    let mut t = std::time::Instant::now();
    let (mut alive, mut dropped) = scan(input, spec, pool)?;
    crate::solve::log_phase(ctx, &mut t, "fit: missing values");
    ctx.check()?;

    let mut used = vec![false; input.factors.len()];
    input.fe.iter().flat_map(|t| &t.factors).chain(input.cluster.iter().flatten()).for_each(|&f| used[f] = true);
    let codes: Vec<Codes> = input
        .factors
        .iter()
        .zip(&used)
        .map(|(f, &u)| if u { factor::factorize_col(f, pool) } else { Ok(Codes { codes: Vec::new(), n: 0 }) })
        .collect::<Result<_, _>>()?;
    let term_factors: Vec<&[usize]> = input.fe.iter().map(|t| t.factors.as_slice()).collect();
    let (term_codes, cluster_codes) = term_and_cluster_codes(codes, &term_factors, &input.cluster, pool)?;
    crate::solve::log_phase(ctx, &mut t, "fit: factorize");
    ctx.check()?;

    let family = match spec.model {
        Model::Ols => Family::Gaussian,
        Model::Poisson => Family::Poisson,
        Model::Logit => Family::Logit,
    };
    let singletons = spec.fixef_rm != FixefRm::None;
    let sep_fe = spec.fixef_rm == FixefRm::Perfect && family != Family::Gaussian;
    let tc: Vec<TermCodes> = input
        .fe
        .iter()
        .zip(&term_codes)
        .filter(|(t, _)| t.intercept)
        .map(|(_, c)| TermCodes { codes: &c.codes, g: c.n as usize })
        .collect();
    // fixest removes perfect-fit levels inside the singleton loop; pyfixest runs its `fe` check
    // once afterwards.
    let pyfixest_sep = sep_fe && spec.flavor == Flavor::Pyfixest;
    let sep = (sep_fe && !pyfixest_sep).then_some(SepRule { y: input.y, family });
    let mut counts = sample::drop_singletons(&tc, &mut alive, None, sep, singletons, pool);
    let levels_before_separation = pyfixest_sep.then(|| {
        let (separated, levels) = pyfixest_separation(&tc, &mut alive, input.y);
        counts.fe_separated = separated as u64;
        levels
    });
    let excluded = if pyfixest_sep && family == Family::Logit {
        let reported_mask = alive.clone();
        sample::drop_singletons(&tc, &mut alive, None, Some(SepRule { y: input.y, family }), false, pool);
        let rows: Vec<u32> = pool.install(|| (0..input.n as u32).into_par_iter().filter(|&i| reported_mask[i as usize] && !alive[i as usize]).collect());
        if rows.is_empty() {
            None
        } else {
            let reported: Vec<u32> = pool.install(|| (0..input.n as u32).into_par_iter().filter(|&i| reported_mask[i as usize]).collect());
            let on_reported = |c: &Codes| factor::restrict(c, &reported, pool);
            Some(Excluded {
                terms: term_codes.iter().map(on_reported).collect::<Result<_, _>>()?,
                clusters: cluster_codes.iter().map(|(_, c)| on_reported(c)).collect::<Result<_, _>>()?,
                rows,
                reported,
            })
        }
    } else {
        None
    };
    crate::solve::log_phase(ctx, &mut t, "fit: singletons");
    dropped.singletons = counts.singletons as usize;
    dropped.perfect_fit = counts.fe_separated as usize;
    let left_out = excluded.as_ref().map_or(0, |e| e.rows.len());
    let kept = input.n - dropped.na - dropped.zero_weight - dropped.singletons - dropped.perfect_fit - left_out;
    if kept < 2 {
        return Err(RfError::TooFewObs);
    }

    // GLM fixed-effect values come from the session's saved terms (linear models read the
    // absorber's coefficients directly).
    let save = spec.model != Model::Ols && (spec.keep_fixef || spec.keep_state);
    let absorb: Vec<AbsorbTerm> =
        input.fe.iter().map(|t| AbsorbTerm { factors: t.factors.clone(), intercept: t.intercept, slopes: t.slopes.clone(), save }).collect();
    let parts = Parts {
        n_input: input.n,
        alive,
        dropped: kept < input.n,
        absorb: &absorb,
        term_codes,
        cluster_codes,
        slope_vars: &input.slopes,
        weights: input.weights,
        weight_kind: if input.weights.is_some() { WeightKind::Aweight } else { WeightKind::None },
        vce: if input.cluster.is_empty() { Vce::Unadjusted } else { Vce::Cluster(input.cluster.clone()) },
        y_sep: sep_fe.then_some(input.y),
        family,
        drop_singletons: singletons,
        sep_fe,
        counts,
    };
    let session = Session::from_parts(parts, AbsorberKind::Fast, pool, ctx)?;
    Ok(Sample { session, dropped, levels_before_separation, excluded })
}

/// pyfixest's `fe` separation check (fepois_.py `_check_for_separation_fe`): unless every
/// outcome is positive, drop the rows of every level, of any intercept term, whose outcomes are
/// all zero; one pass, no singleton re-check (logit too: all-one levels stay). Returns the rows
/// dropped and the levels present per term before the check.
fn pyfixest_separation(tc: &[TermCodes], alive: &mut [bool], y: &[f64]) -> (usize, Vec<usize>) {
    let levels: Vec<usize> = tc
        .iter()
        .map(|t| {
            let mut seen = vec![false; t.g];
            t.codes.iter().zip(alive.iter()).filter(|(_, a)| **a).for_each(|(&c, _)| seen[c as usize] = true);
            seen.iter().filter(|s| **s).count()
        })
        .collect();
    if alive.iter().zip(y).all(|(a, &v)| !*a || v > 0.0) {
        return (0, levels);
    }
    let mut drop = vec![false; alive.len()];
    for t in tc {
        let mut positive = vec![false; t.g];
        for ((&c, &a), &v) in t.codes.iter().zip(alive.iter()).zip(y) {
            if a && v > 0.0 {
                positive[c as usize] = true;
            }
        }
        for ((&c, &a), d) in t.codes.iter().zip(alive.iter()).zip(drop.iter_mut()) {
            *d |= a && !positive[c as usize];
        }
    }
    let mut n = 0;
    for (a, d) in alive.iter_mut().zip(drop) {
        if *a && d {
            *a = false;
            n += 1;
        }
    }
    (n, levels)
}
