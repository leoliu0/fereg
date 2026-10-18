//! CRV3: the leave-one-cluster-out jackknife (fereg SE design, contract 3d).
//!
//! pyfixest 0.60.0 `_vcov_crv3_fast` / `_vcov_crv3_slow` (feols_.py): for every cluster g the
//! model is re-estimated without g's rows (fixed effects re-estimated, an exact jackknife, no
//! one-step approximation), and the meat sum_g (beta_(-g) - beta)(beta_(-g) - beta)' is centered
//! at the full-sample beta. The front ends apply their small-sample factors to [`jackknife_meat`].
//!
//! beta_(-g) comes from one of two paths:
//! - Closed form: linear models (OLS, 2SLS; weighted or not) without FE terms or with every FE
//!   term nested in the cluster. Leaving g out then removes whole FE levels, so the residualized
//!   columns of the remaining rows are the full-sample residualized columns restricted to them.
//!   The columns are residualized once by the crate's FE solver (`solve::build` over the
//!   estimation rows: the absorber `fit::ols` residualizes with through its session, at the
//!   problem's `fixef_tol`), and beta_(-g) solves the normal equations of the weighted Gram of
//!   all rows minus cluster g's own (pyfixest's fast path, generalized to nested FEs and 2SLS).
//! - Refit: everything else, and GLMs always. `fit::fit` on the estimation rows with g's rows
//!   dropped, fixest flavor, `FixefRm::Perfect` (singletons and perfect-fit levels of the
//!   leave-out sample are dropped as a fresh fit drops them; they do not move beta), iid variance.
//!   Logit refits are logit (pyfixest refits every GLM with `fepois`, a bug not replicated).
//!
//! Both paths use fixest's collinearity rule (at [`JkProblem::collin_tol`]) on the
//! leave-out sample; a coefficient of the full model that it removes, or that the refit loses
//! otherwise, is an error naming the cluster and the coefficient.
//!
//! Parallelism: the closed form runs its passes on the pool and solves the clusters in parallel.
//! Refits run concurrently, one cluster per worker and each fit on `threads / workers` threads:
//! concurrent fits scale far better than one fit spread over many threads (`timing_1m` test,
//! 1e6 rows, two FEs, 16 threads: 5.1 ms per cluster with 16 single-threaded fits in flight,
//! 19.4 ms with one 16-thread fit at a time). The number of concurrent fits is capped by the
//! clusters and by a working-memory budget. Every beta_(-g) depends on g's data only, and every
//! fit is bit-identical for any thread count, so the result is too.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use rayon::prelude::*;

use crate::absorber::SolveOpts;
use crate::api::Ctx;
use crate::error::RfError;
use crate::fit::{self, CoefSource, FactorCol, FeTerm, FitInput, FitOutput, FitSpec, Flavor, FixefRm, Model, VcovKind};
use crate::gather::{self, Col};
use crate::leverage::FeCols;
use crate::par::{self, Pool};
use crate::session::Term;
use crate::vcov::chol_collin;

/// `Linear` = OLS, or 2SLS when `endog` is not empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JkModel {
    Linear,
    Poisson,
    Logit,
}

pub(crate) struct JkProblem<'a> {
    pub model: JkModel,
    /// Estimation rows, kept order; every slice below has `n` entries.
    pub n: usize,
    pub y: &'a [f64],
    pub x: Vec<&'a [f64]>,
    pub endog: Vec<&'a [f64]>,
    pub inst: Vec<&'a [f64]>,
    /// Coefficient on a constant (no FE term with an intercept).
    pub intercept: bool,
    pub weights: Option<&'a [f64]>,
    /// GLM offset (link scale).
    pub offset: Option<&'a [f64]>,
    pub fe: Vec<FeCols<'a>>,
    /// Full-sample coefficients, order: intercept?, x, endog.
    pub beta: &'a [f64],
    pub fixef_tol: f64,
    pub fixef_maxiter: usize,
    pub glm_tol: f64,
    pub glm_maxiter: usize,
    /// Collinearity tolerance of the leave-out fits (fixest `cpp_cholesky` rule, as the full fit).
    pub collin_tol: f64,
}

/// beta_(-g) for every cluster g (`g_count` x k row-major, coefficient order of
/// [`JkProblem::beta`]). `cluster`: codes over the estimation rows (kept order), `< g_count`,
/// every code with at least one row.
pub(crate) fn leave_cluster_out(p: &JkProblem, cluster: &[u32], g_count: usize, ctx: &Ctx, pool: &Pool) -> Result<Vec<f64>, RfError> {
    leave_cluster_out_with(p, cluster, g_count, ctx, pool, Path::Auto, None)
}

/// sum_g (beta_(-g) - beta)(beta_(-g) - beta)' (k x k row-major), in cluster order.
pub(crate) fn jackknife_meat(betas: &[f64], beta: &[f64], k: usize) -> Vec<f64> {
    let mut m = vec![0.0; k * k];
    if k == 0 {
        return m;
    }
    let mut d = vec![0.0; k];
    for b in betas.chunks_exact(k) {
        d.iter_mut().zip(b.iter().zip(beta)).for_each(|(d, (b, c))| *d = b - c);
        for i in 0..k {
            for j in 0..k {
                m[i * k + j] += d[i] * d[j];
            }
        }
    }
    m
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    /// Closed form when it applies, refits otherwise.
    Auto,
    /// Refits even where the closed form applies (the tests compare the two paths).
    #[cfg(test)]
    Refit,
}

/// Working-memory budget of the concurrent refits.
const REFIT_MEMORY: usize = 16 << 30;

/// Number of coefficients: intercept?, x, endog.
fn n_coef(p: &JkProblem) -> usize {
    usize::from(p.intercept) + p.x.len() + p.endog.len()
}

/// Coefficient sources in [`JkProblem::beta`] order.
fn sources(p: &JkProblem) -> Vec<CoefSource> {
    let mut s = Vec::with_capacity(n_coef(p));
    if p.intercept {
        s.push(CoefSource::Intercept);
    }
    s.extend((0..p.x.len()).map(CoefSource::X));
    s.extend((0..p.endog.len()).map(CoefSource::Endog));
    s
}

/// Position of a coefficient in [`JkProblem::beta`] order.
fn position(p: &JkProblem, s: CoefSource) -> usize {
    let icpt = usize::from(p.intercept);
    match s {
        CoefSource::Intercept => 0,
        CoefSource::X(j) => icpt + j,
        CoefSource::Endog(j) => icpt + p.x.len() + j,
        CoefSource::Inst(j) => icpt + p.x.len() + p.endog.len() + j,
    }
}

fn label(s: CoefSource) -> String {
    match s {
        CoefSource::Intercept => "(Intercept)".into(),
        CoefSource::X(j) => format!("x[{j}]"),
        CoefSource::Endog(j) => format!("endog[{j}]"),
        CoefSource::Inst(j) => format!("inst[{j}]"),
    }
}

fn not_estimable(s: CoefSource, g: usize) -> RfError {
    RfError::InvalidInput(format!(
        "CRV3: coefficient {} is not estimable when cluster {g} (0-based cluster code) is left out",
        label(s)
    ))
}

fn validate(p: &JkProblem, cluster: &[u32], g_count: usize) -> Result<(), RfError> {
    let bad = |m: &str| Err(RfError::InvalidInput(format!("CRV3: {m}")));
    let n = p.n;
    let cols = std::iter::once(p.y).chain(p.x.iter().copied()).chain(p.endog.iter().copied()).chain(p.inst.iter().copied());
    if cols.chain(p.weights).chain(p.offset).chain(p.fe.iter().flat_map(|f| f.slopes.iter().copied())).any(|c| c.len() != n)
        || p.fe.iter().any(|f| f.codes.len() != n)
        || cluster.len() != n
    {
        return bad("every column must have one entry per estimation row");
    }
    if p.beta.len() != n_coef(p) {
        return bad("beta must have one entry per coefficient");
    }
    if p.model != JkModel::Linear && !p.endog.is_empty() {
        return bad("instrumental variables are only supported for linear models");
    }
    if p.model == JkModel::Linear && p.offset.is_some() {
        return bad("an offset is only supported for Poisson and logit models");
    }
    if p.inst.len() < p.endog.len() || (p.endog.is_empty() && !p.inst.is_empty()) {
        return bad("IV estimation needs at least as many instruments as endogenous regressors");
    }
    if p.intercept && p.fe.iter().any(|f| f.intercept) {
        return bad("no intercept coefficient with an FE term that has an intercept");
    }
    if p.fe.iter().any(|f| (!f.intercept && f.slopes.is_empty()) || f.codes.iter().any(|&c| c as usize >= f.g)) {
        return bad("invalid FE term");
    }
    if g_count < 2 {
        return bad("at least two clusters are required");
    }
    let mut count = vec![0usize; g_count];
    for &c in cluster {
        match count.get_mut(c as usize) {
            Some(x) => *x += 1,
            None => return bad("cluster code out of range"),
        }
    }
    if let Some(g) = count.iter().position(|&c| c == 0) {
        return Err(RfError::InvalidInput(format!("CRV3: cluster {g} has no observations")));
    }
    Ok(())
}

fn leave_cluster_out_with(
    p: &JkProblem,
    cluster: &[u32],
    g_count: usize,
    ctx: &Ctx,
    pool: &Pool,
    path: Path,
    workers: Option<usize>,
) -> Result<Vec<f64>, RfError> {
    validate(p, cluster, g_count)?;
    if n_coef(p) == 0 {
        return Ok(Vec::new());
    }
    if path == Path::Auto && closed_form_applies(p, cluster) {
        closed_form(p, cluster, g_count, ctx, pool)
    } else {
        refits(p, cluster, g_count, ctx, pool, workers)
    }
}

/// Linear model, every FE term nested in the cluster (ftools `nested_within`).
fn closed_form_applies(p: &JkProblem, cluster: &[u32]) -> bool {
    p.model == JkModel::Linear && p.fe.iter().all(|f| crate::dof::nested_within(f.codes, f.g, cluster))
}

// ---------------------------------------------------------------------------------------------
// Closed form

/// Columns of the residualized row block: y, (1), x..., endog..., inst...
struct Lay {
    r: usize,
    icpt: bool,
    nx: usize,
    ne: usize,
    nl: usize,
}

impl Lay {
    fn one(&self) -> usize {
        1
    }
    fn x(&self, j: usize) -> usize {
        1 + usize::from(self.icpt) + j
    }
    fn endog(&self, j: usize) -> usize {
        1 + usize::from(self.icpt) + self.nx + j
    }
    fn inst(&self, j: usize) -> usize {
        1 + usize::from(self.icpt) + self.nx + self.ne + j
    }
    fn unit(&self, c: usize) -> Vec<f64> {
        let mut u = vec![0.0; self.r];
        u[c] = 1.0;
        u
    }

    /// Coefficients (fixest order: intercept, endog, x; the caller maps them) of the linear
    /// model on the Gram `c` (r x r) of its columns, as `fit::ols` computes them: 2SLS first
    /// stages on Z = [(1), inst, x] and the second stage on [(1), fitted endog, x], each with
    /// fixest's collinearity rule. Err = a second-stage regressor removed for collinearity.
    fn solve(&self, c: &[f64], tol: f64) -> Result<Vec<(CoefSource, f64)>, CoefSource> {
        let r = self.r;
        let cross = |a: &[f64], b: &[f64]| -> f64 {
            let mut s = 0.0;
            for (i, &ai) in a.iter().enumerate().filter(|(_, v)| **v != 0.0) {
                let row = &c[i * r..(i + 1) * r];
                s += ai * row.iter().zip(b).filter(|(_, bj)| **bj != 0.0).map(|(g, bj)| g * bj).sum::<f64>();
            }
            s
        };
        let gram = |d: &[Vec<f64>]| -> Vec<f64> { d.iter().flat_map(|a| d.iter().map(|b| cross(a, b))).collect() };
        let mut design: Vec<(CoefSource, Vec<f64>)> = Vec::new();
        if self.icpt {
            design.push((CoefSource::Intercept, self.unit(self.one())));
        }
        if self.ne > 0 {
            let mut z: Vec<Vec<f64>> = Vec::new();
            if self.icpt {
                z.push(self.unit(self.one()));
            }
            z.extend((0..self.nl).map(|j| self.unit(self.inst(j))));
            z.extend((0..self.nx).map(|j| self.unit(self.x(j))));
            let (excl, inv) = chol_collin(&gram(&z), z.len(), tol);
            let kept: Vec<usize> = (0..z.len()).filter(|&a| !excl[a]).collect();
            let m = kept.len();
            for j in 0..self.ne {
                let t = self.unit(self.endog(j));
                let zt: Vec<f64> = kept.iter().map(|&a| cross(&z[a], &t)).collect();
                let mut fitted = vec![0.0; r];
                for (ai, &a) in kept.iter().enumerate() {
                    let pa: f64 = (0..m).map(|b| inv[ai * m + b] * zt[b]).sum();
                    fitted.iter_mut().zip(&z[a]).for_each(|(f, v)| *f += pa * v);
                }
                design.push((CoefSource::Endog(j), fitted));
            }
        }
        design.extend((0..self.nx).map(|j| (CoefSource::X(j), self.unit(self.x(j)))));
        let d: Vec<Vec<f64>> = design.iter().map(|(_, v)| v.clone()).collect();
        let k = d.len();
        let (excl, inv) = chol_collin(&gram(&d), k, tol);
        if let Some(j) = excl.iter().position(|&e| e) {
            return Err(design[j].0);
        }
        let y = self.unit(0);
        let xty: Vec<f64> = d.iter().map(|a| cross(a, &y)).collect();
        Ok((0..k).map(|a| (design[a].0, (0..k).map(|b| inv[a * k + b] * xty[b]).sum())).collect())
    }
}

/// Codes made dense (levels without rows removed, order kept) and the number of levels.
fn dense_codes(codes: &[u32], g: usize) -> (Vec<u32>, usize) {
    let mut map = vec![u32::MAX; g];
    codes.iter().for_each(|&c| map[c as usize] = 0);
    let mut next = 0u32;
    for m in map.iter_mut().filter(|m| **m == 0) {
        *m = next;
        next += 1;
    }
    if next as usize == g { (codes.to_vec(), g) } else { (codes.iter().map(|&c| map[c as usize]).collect(), next as usize) }
}

fn fe_terms(p: &JkProblem) -> Vec<Term> {
    let mut s0 = 0;
    p.fe.iter()
        .enumerate()
        .map(|(q, f)| {
            let (codes, g) = dense_codes(f.codes, f.g);
            let slope_vars: Vec<usize> = (s0..s0 + f.slopes.len()).collect();
            s0 += f.slopes.len();
            Term {
                factors: vec![q],
                codes,
                g,
                intercept: f.intercept,
                slopes: f.slopes.iter().map(|s| s.to_vec()).collect(),
                slope_vars,
                save: false,
            }
        })
        .collect()
}

fn closed_form(p: &JkProblem, cluster: &[u32], g_count: usize, ctx: &Ctx, pool: &Pool) -> Result<Vec<f64>, RfError> {
    let n = p.n;
    let lay = Lay { r: 1 + usize::from(p.intercept) + p.x.len() + p.endog.len() + p.inst.len(), icpt: p.intercept, nx: p.x.len(), ne: p.endog.len(), nl: p.inst.len() };
    let r = lay.r;
    let k = n_coef(p);
    let mut cols = vec![Col::Values(p.y)];
    if p.intercept {
        cols.push(Col::One);
    }
    cols.extend(p.x.iter().chain(&p.endog).chain(&p.inst).map(|v| Col::Values(v)));

    // Row block in the absorber's row order, residualized once on the FEs.
    let mut absorber = if p.fe.is_empty() { None } else { Some(crate::solve::build(&fe_terms(p), p.weights, n, pool, ctx)?) };
    let order: Vec<u32> = match &absorber {
        Some(a) => a.row_order().to_vec(),
        None => (0..n as u32).collect(),
    };
    let mut rows = gather::gather_input(&order, None, &cols, pool)?;
    if let Some(a) = absorber.as_mut() {
        let opts = SolveOpts { tol: p.fixef_tol, maxiter: p.fixef_maxiter, want_coefs: false };
        a.residualize(&mut rows, r, &opts, None, pool, ctx)?;
    }
    ctx.check()?;
    let w: Option<&[f64]> = match &absorber {
        Some(a) => a.weights(),
        None => p.weights,
    };
    let cl_rows: Vec<u32> = pool.install(|| order.par_iter().map(|&i| cluster[i as usize]).collect());
    let (full, _) = crate::stats::gram(&rows, r, w, pool);
    let (off, ids) = crate::structure::scatter_by_code(&cl_rows, g_count, pool)?;
    ctx.check()?;

    let tol = p.collin_tol;
    let mut out = par::alloc_vec(g_count * k, 0.0)?;
    let errs: Vec<Option<RfError>> = pool.install(|| {
        out.par_chunks_mut(k)
            .enumerate()
            .map(|(g, o)| {
                // Cluster g's own Gram (its rows in ascending row order), subtracted from the full one.
                let mut cg = vec![0.0; r * r];
                for &i in &ids[off[g] as usize..off[g + 1] as usize] {
                    let i = i as usize;
                    let row = &rows[i * r..(i + 1) * r];
                    let wi = w.map_or(1.0, |w| w[i]);
                    for a in 0..r {
                        let wa = wi * row[a];
                        for b in a..r {
                            cg[a * r + b] += wa * row[b];
                        }
                    }
                }
                let mut c = vec![0.0; r * r];
                for a in 0..r {
                    for b in a..r {
                        let v = full[a * r + b] - cg[a * r + b];
                        c[a * r + b] = v;
                        c[b * r + a] = v;
                    }
                }
                match lay.solve(&c, tol) {
                    Ok(coef) => {
                        coef.into_iter().for_each(|(s, v)| o[position(p, s)] = v);
                        None
                    }
                    Err(s) => Some(not_estimable(s, g)),
                }
            })
            .collect()
    });
    match errs.into_iter().flatten().next() {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

// ---------------------------------------------------------------------------------------------
// Refits

/// Working memory of one refit (estimate): the gathered and residualized row blocks, the factor
/// and solver arrays, the drop mask. Calibrated on 1e6 rows, two FEs, two regressors: 16
/// concurrent OLS / Poisson refits peak at ~190 bytes per row each; this gives 224.
fn refit_bytes(p: &JkProblem) -> usize {
    let r = 2 + p.x.len() + p.endog.len() + p.inst.len() + usize::from(p.weights.is_some()) + usize::from(p.offset.is_some());
    let fe: usize = p.fe.iter().map(|f| 1 + f.slopes.len()).sum();
    p.n * (16 * r + 32 * fe + 96)
}

/// Concurrent refits: one per thread, at most one per cluster, within [`REFIT_MEMORY`].
fn refit_workers(threads: usize, g_count: usize, bytes_per_fit: usize) -> usize {
    threads.min(g_count).min(REFIT_MEMORY / bytes_per_fit.max(1)).max(1)
}

fn discard(_: &str) {}

/// Adds the cluster to the message of a refit error.
fn refit_error(e: RfError, g: usize) -> RfError {
    let at = |m: &str| format!("CRV3, refit without cluster {g} (0-based cluster code): {m}");
    match e {
        RfError::InvalidInput(m) => RfError::InvalidInput(at(&m)),
        RfError::TooFewObs => RfError::InvalidInput(at("insufficient observations")),
        RfError::Code(rc, m) => RfError::Code(rc, at(&m)),
        e => e,
    }
}

/// beta_(-g) of one refit in [`JkProblem::beta`] order.
fn refit_coefs(out: Result<FitOutput, RfError>, want: &[CoefSource], g: usize, p: &JkProblem) -> Result<Vec<f64>, RfError> {
    let out = out.map_err(|e| refit_error(e, g))?;
    if p.model != JkModel::Linear && !out.stats.converged {
        return Err(RfError::InvalidInput(format!(
            "CRV3, refit without cluster {g} (0-based cluster code): no convergence in {} IRLS iterations",
            p.glm_maxiter
        )));
    }
    want.iter().map(|s| out.coef_source.iter().position(|t| t == s).map(|a| out.coef[a]).ok_or_else(|| not_estimable(*s, g))).collect()
}

fn refits(p: &JkProblem, cluster: &[u32], g_count: usize, ctx: &Ctx, pool: &Pool, workers: Option<usize>) -> Result<Vec<f64>, RfError> {
    let n = p.n;
    let k = n_coef(p);
    let (off, ids) = crate::structure::scatter_by_code(cluster, g_count, pool)?;
    let codes: Vec<Vec<i32>> = p.fe.iter().map(|f| pool.install(|| f.codes.par_iter().map(|&c| c as i32).collect())).collect();
    let mut slopes: Vec<&[f64]> = Vec::new();
    let mut fe: Vec<FeTerm> = Vec::new();
    for (q, f) in p.fe.iter().enumerate() {
        let s0 = slopes.len();
        slopes.extend(f.slopes.iter().copied());
        fe.push(FeTerm { factors: vec![q], intercept: f.intercept, slopes: (s0..slopes.len()).collect() });
    }
    let base = FitInput {
        n,
        y: p.y,
        x: p.x.clone(),
        endog: p.endog.clone(),
        inst: p.inst.clone(),
        factors: codes.iter().zip(&p.fe).map(|(c, f)| FactorCol::Codes { codes: c, nlevels: f.g as u32, base: 0 }).collect(),
        factor_names: (0..p.fe.len()).map(|q| format!("fe{q}")).collect(),
        fe,
        slopes,
        weights: p.weights,
        offset: p.offset,
        cluster: Vec::new(),
        drop: None,
        ..Default::default()
    };
    let spec = FitSpec {
        model: match p.model {
            JkModel::Linear => Model::Ols,
            JkModel::Poisson => Model::Poisson,
            JkModel::Logit => Model::Logit,
        },
        flavor: Flavor::Fixest,
        intercept: p.intercept,
        vcov: VcovKind::Iid,
        fixef_rm: FixefRm::Perfect,
        fixef_tol: p.fixef_tol,
        fixef_maxiter: p.fixef_maxiter,
        glm_tol: p.glm_tol,
        glm_maxiter: p.glm_maxiter,
        collin_tol: p.collin_tol,
        keep_resid: false,
        keep_fixef: false,
        keep_state: false,
        ..FitSpec::default()
    };
    let want = sources(p);
    // One leave-out fit; `mask` is all false before and after.
    let one = |g: usize, mask: &mut [bool], fctx: &Ctx| -> Result<Vec<f64>, RfError> {
        let rows = &ids[off[g] as usize..off[g + 1] as usize];
        rows.iter().for_each(|&i| mask[i as usize] = true);
        let out = fit::fit(&FitInput { drop: Some(&*mask), ..base.clone() }, &spec, fctx);
        rows.iter().for_each(|&i| mask[i as usize] = false);
        refit_coefs(out, &want, g, p)
    };

    let threads = pool.threads();
    let workers = workers.unwrap_or_else(|| refit_workers(threads, g_count, refit_bytes(p))).clamp(1, g_count);
    let mut out = par::alloc_vec(g_count * k, 0.0)?;
    if workers == 1 {
        let fctx = Ctx { threads, verbose: 0, interrupt: ctx.interrupt, log: &discard };
        let mut mask = par::alloc_vec(n, false)?;
        for g in 0..g_count {
            ctx.check()?;
            out[g * k..(g + 1) * k].copy_from_slice(&one(g, &mut mask, &fctx)?);
        }
        return Ok(out);
    }

    // Workers claim clusters in ascending order. After a failure at cluster f no cluster above f
    // is started (fits above f in flight are interrupted), and every cluster below f was claimed
    // before and completes, so the error reported (the lowest failing cluster) is deterministic.
    let next = AtomicUsize::new(0);
    let failed = AtomicUsize::new(usize::MAX);
    let stop = AtomicBool::new(false);
    type Done = Vec<(usize, Result<Vec<f64>, RfError>)>;
    let work = |tid: usize, poll: &dyn Fn() -> bool| -> Result<Done, RfError> {
        let t = (threads / workers + usize::from(tid < threads % workers)).max(1);
        let mut mask = par::alloc_vec(n, false)?;
        let cur = Cell::new(0usize);
        let interrupt = || stop.load(Ordering::Relaxed) || cur.get() > failed.load(Ordering::Relaxed) || poll();
        let fctx = Ctx { threads: t, verbose: 0, interrupt: &interrupt, log: &discard };
        let mut done = Vec::new();
        loop {
            let g = next.fetch_add(1, Ordering::Relaxed);
            if g >= g_count || g > failed.load(Ordering::Relaxed) || stop.load(Ordering::Relaxed) {
                return Ok(done);
            }
            cur.set(g);
            let res = one(g, &mut mask, &fctx);
            if res.is_err() {
                failed.fetch_min(g, Ordering::Relaxed);
            }
            done.push((g, res));
        }
    };
    let interrupted = AtomicBool::new(false);
    // The calling thread is worker 0 and the only one that polls the caller's interrupt.
    let main_poll = || {
        let i = (ctx.interrupt)();
        if i {
            interrupted.store(true, Ordering::Relaxed);
            stop.store(true, Ordering::Relaxed);
        }
        i
    };
    let results: Vec<Result<Done, RfError>> = std::thread::scope(|s| {
        let work = &work;
        let handles: Vec<_> = (1..workers).map(|tid| par::spawn_scoped(s, move || work(tid, &|| false))).collect();
        let mine = work(0, &main_poll);
        if mine.is_err() {
            stop.store(true, Ordering::Relaxed);
        }
        while !handles.iter().all(|h| h.is_finished()) {
            main_poll();
            std::thread::sleep(Duration::from_millis(1));
        }
        let mut all = vec![mine];
        for h in handles {
            all.push(h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)));
        }
        all
    });
    if interrupted.load(Ordering::Relaxed) {
        return Err(RfError::Interrupted);
    }
    let mut slots: Vec<Option<Result<Vec<f64>, RfError>>> = (0..g_count).map(|_| None).collect();
    for done in results {
        for (g, res) in done? {
            slots[g] = Some(res);
        }
    }
    let f = failed.load(Ordering::Relaxed);
    if f < g_count {
        if let Some(Some(Err(e))) = slots.get_mut(f).map(Option::take) {
            return Err(e);
        }
        return Err(RfError::Internal("CRV3: lost the error of a failed refit".into()));
    }
    for (g, slot) in slots.into_iter().enumerate() {
        match slot {
            Some(Ok(b)) => out[g * k..(g + 1) * k].copy_from_slice(&b),
            Some(Err(e)) => return Err(e),
            None => return Err(RfError::Internal(format!("CRV3: cluster {g} was not refitted"))),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
