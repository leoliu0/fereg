//! GLM estimation by IRLS: Poisson PML with ppmlhdfe 2.3.3 conventions and logit MLE with
//! absorbed fixed effects (plan Phase 6).
//!
//! Flow (ppmlhdfe `init_variables`, `init_separation`, `solve`):
//! 1. y (Poisson, when the absorbed space holds the constant) and X are divided by their
//!    unweighted sample sd (reghdfe `reghdfe_standardize`, floored at 1e-3); results are scaled
//!    back at the end.
//! 2. Regressors collinear with the fixed effects (Σx̃² / Σx² ≤ min(1e-6, itol/10)) or with each
//!    other (sweep of X̃'WX̃ with the true weights) are dropped (ppmlhdfe `remove_collinears`).
//! 3. Poisson `relu` separation (`simplex` is an alias), in rounds ([`Session::glm_separate`]):
//!    the `fe` rule when prepare did not apply it, then the ReLU check; drop, re-run singletons
//!    (and the `fe` rule when prepare applied it, [`Session::drop_obs`]), remap the rows, re-run
//!    step 2, and repeat until a ReLU check finds nothing (ppmlhdfe runs it once). When the
//!    check cannot certify its result, the `mu` rule runs in step 4 as a fallback and the fit
//!    reports `separation_complete` = 0 with a note.
//! 4. IRLS ([`irls::irls`]); with the `mu` rule, separated observations are dropped as in 3 and
//!    IRLS is re-run from the current μ.
//! 5. Final VCE on the last IRLS weights (reghdfe5 `reghdfe_solve_ols(..., "vce_asymptotic")`):
//!    w_v = w·N/Σw; robust multiplier (e·w_v)² (÷ fw with fweights) and factor N/(N−1);
//!    cluster scores e·w_v and factor G/(G−1); multiway PSD fix on the whole matrix. Logit `oim`
//!    is the inverse Hessian. `_cons` is the regression-through-the-means constant.

pub mod family;
pub mod irls;
pub mod separation;
pub(crate) mod wls;

use rayon::prelude::*;

use crate::absorber::SolveOpts;
use crate::api::{Ctx, EstResult, Family, GlmConvention, GlmOptions, GlmParts, Guess, Layout, NamedMat, SeparationCheck, Spec, Vce, WeightKind};
use crate::dof;
use crate::error::RfError;
use crate::fe_recovery;
use crate::ols::{self, STATUS_ABSORBED, STATUS_COLLINEAR, STATUS_OK, missing_if_nonfinite, sub_block};
use crate::par::{self, CHUNK, Pool};
use crate::sample::{self, SepRule, TermCodes};
use crate::session::{DropReport, Session, Term};
use crate::stats;
use crate::vcov::{self, ClusterCodes, INVSYM_TOL};

use irls::{IrlsCfg, IrlsOut, fmt_e};

/// Working data of a GLM fit, in the absorber's row order.
pub(crate) struct GlmData {
    pub family: Family,
    pub n: usize,
    /// Depvar (Poisson: divided by sd_y; logit: 0/1).
    pub y: Vec<f64>,
    /// Number of regressors still in the model.
    pub k: usize,
    /// n×k row-major, each column divided by its sd.
    pub x: Vec<f64>,
    /// Original regressor index (0-based) of each column of `x`.
    pub cols: Vec<usize>,
    pub sd_x: Vec<f64>,
    pub off: Option<Vec<f64>>,
    /// True (user) weights.
    pub tw: Option<Vec<f64>>,
}

/// Outcome of [`Session::glm_separate`].
struct SepRounds {
    /// Separated observations: kept position at entry and certificate value (the ReLU
    /// prediction, or 1 for the `fe` rule).
    tagged: Vec<(u32, f64)>,
    /// Every ReLU round certified its result.
    complete: bool,
    /// ReLU iterations over all rounds.
    iterations: usize,
}

/// reghdfe `reghdfe_standardize` (fast formula), floored at 1e-3.
fn standardize_sd(v: impl Iterator<Item = f64>) -> f64 {
    let (mut n, mut s, mut ss) = (0.0f64, 0.0f64, 0.0f64);
    for x in v {
        n += 1.0;
        s += x;
        ss += x * x;
    }
    ((ss - s * s / n) / (n - 1.0)).sqrt().max(1e-3)
}

fn wmean(v: &[f64], w: Option<&[f64]>, pool: &Pool) -> f64 {
    let acc = par::sum_chunks(pool, v.len(), CHUNK, 2, |range, acc| {
        for i in range {
            let wi = w.map_or(1.0, |w| w[i]);
            acc[0] += wi * v[i];
            acc[1] += wi;
        }
    });
    acc[0] / acc[1]
}

fn col_sumsq(x: &[f64], k: usize, pool: &Pool) -> Vec<f64> {
    let n = x.len().checked_div(k).unwrap_or(0);
    par::sum_chunks(pool, n, CHUNK, k, |range, acc| {
        for i in range {
            for (a, v) in acc.iter_mut().zip(&x[i * k..(i + 1) * k]) {
                *a += v * v;
            }
        }
    })
}

fn plural(n: usize) -> &'static str {
    if n > 1 { "s" } else { "" }
}

/// Whether ppmlhdfe's MAP partial-out is exact for these terms: one term's column space contains
/// every other term's, so all the projections commute and one sweep solves the system (its
/// realized accuracy is at machine precision). A single term (reghdfe `G == 1`) is the trivial
/// case; redundant terms nested in another (`absorb(firm industry)`, `absorb(i##c.t c)`) are the
/// others.
fn map_exact(terms: &[Term]) -> bool {
    terms.len() <= 1 || terms.iter().enumerate().any(|(a, ta)| terms.iter().enumerate().all(|(b, tb)| a == b || spans(ta, tb)))
}

/// The columns of term `b` lie in the span of those of term `a`: every level of `a` lies within
/// one level of `b`, and `a` has an intercept and each slope wherever `b` does.
fn spans(a: &Term, b: &Term) -> bool {
    (a.intercept || !b.intercept)
        && b.slope_vars.iter().all(|v| a.slope_vars.contains(v))
        && a.g >= b.g
        && dof::nested_within(&a.codes, a.g, &b.codes)
}

/// Input checks shared by the GLM entry points.
fn validate(se: &Session, rows: &[f64], lay: &Layout, offset: Option<&[f64]>, s: &Spec, g: &GlmOptions) -> Result<(), RfError> {
    if lay.nendog != 0 || lay.ninst != 0 {
        return Err(RfError::InvalidInput("glm: layout has endogenous variables or instruments".into()));
    }
    if rows.len() != se.n() * lay.ncols() {
        return Err(RfError::InvalidInput("glm: data block size differs from N·R".into()));
    }
    if offset.is_some_and(|o| o.len() != se.n()) {
        return Err(RfError::InvalidInput("glm: offset length differs from N".into()));
    }
    if g.family == Family::Gaussian {
        return Err(RfError::InvalidInput("glm: family must be poisson or logit".into()));
    }
    if g.family == Family::Logit && g.separation.relu {
        return Err(RfError::InvalidInput("glm: the relu/simplex separation check is only available for Poisson".into()));
    }
    if s.want_resid {
        return Err(RfError::InvalidInput("glm: residuals are not available; use d and predict".into()));
    }
    if s.noconstant {
        return Err(RfError::InvalidInput("glm: noconstant is not allowed".into()));
    }
    if !(g.tol > 0.0 && g.tol <= 1.0 && g.itol > 0.0 && g.maxiter > 0) {
        return Err(RfError::InvalidInput("glm: tolerances must be in (0, 1] and maxiter positive".into()));
    }
    Ok(())
}

impl Session {
    /// GLM on `rows` (N×R row-major in [`Session::row_order`], layout `[y, x...]`); `offset` in row
    /// order. Observations dropped by separation are removed from the session (see
    /// [`Session::keep`]); per-observation outputs refer to the final kept sample. ppmlhdfe /
    /// Stata `logit` conventions ([`GlmConvention::Ppmlhdfe`]).
    pub fn glm(&mut self, rows: Vec<f64>, lay: &Layout, offset: Option<Vec<f64>>, s: &Spec, g: &GlmOptions, ctx: &Ctx) -> Result<EstResult, RfError> {
        self.glm_checked(rows, lay, offset, s, g, GlmConvention::Ppmlhdfe, false, ctx).map(|(res, _)| res)
    }

    /// [`Session::glm`] under convention `conv`, also returning the per-observation [`GlmParts`].
    #[allow(clippy::too_many_arguments)]
    pub fn glm_parts(
        &mut self,
        rows: Vec<f64>,
        lay: &Layout,
        offset: Option<Vec<f64>>,
        s: &Spec,
        g: &GlmOptions,
        conv: GlmConvention,
        ctx: &Ctx,
    ) -> Result<(EstResult, GlmParts), RfError> {
        let (res, parts) = self.glm_checked(rows, lay, offset, s, g, conv, true, ctx)?;
        Ok((res, parts.ok_or_else(|| RfError::Internal("glm: parts were not computed".into()))?))
    }

    /// The Poisson separation check alone on `rows` (layout `[y, x...]`, row order), after the
    /// same preparation as [`Session::glm`] (standardization, collinear regressors dropped):
    /// the rounds of [`Session::glm_separate`]; `s.maxiter` caps the FE solves. Nothing is
    /// estimated. The separated observations are dropped from the session, which is left on the
    /// reduced sample; the returned vectors are in kept order over the sample at the call.
    pub fn glm_separation(
        &mut self,
        rows: Vec<f64>,
        lay: &Layout,
        offset: Option<Vec<f64>>,
        s: &Spec,
        g: &GlmOptions,
        ctx: &Ctx,
    ) -> Result<SeparationCheck, RfError> {
        validate(self, &rows, lay, offset.as_deref(), s, g)?;
        if g.family != Family::Poisson {
            return Err(RfError::InvalidInput("glm: the separation check is only available for Poisson".into()));
        }
        let n0 = self.n();
        par::with_pool(par::threads_for(ctx.threads, n0), |pool| {
            let (mut d, mut status, _) = self.glm_data(rows, lay.nx, offset, g, s.maxiter, pool, ctx)?;
            let rounds = self.glm_separate(&mut d, &mut status, g.itol, s.maxiter, pool, ctx)?;
            let mut separated = vec![false; n0];
            let mut certificate = vec![0.0; n0];
            for &(k, z) in &rounds.tagged {
                separated[k as usize] = true;
                certificate[k as usize] = z;
            }
            Ok(SeparationCheck {
                num_separated: rounds.tagged.len(),
                separated,
                certificate,
                complete: rounds.complete,
                iterations: rounds.iterations,
            })
        })?
    }

    #[allow(clippy::too_many_arguments)]
    fn glm_checked(
        &mut self,
        rows: Vec<f64>,
        lay: &Layout,
        offset: Option<Vec<f64>>,
        s: &Spec,
        g: &GlmOptions,
        conv: GlmConvention,
        want_parts: bool,
        ctx: &Ctx,
    ) -> Result<(EstResult, Option<GlmParts>), RfError> {
        validate(self, &rows, lay, offset.as_deref(), s, g)?;
        // The iteration log and drop messages are SMCL for the Stata plugin; fixest mode reports
        // through EstResult::notes and the returned counts instead.
        let muted = Ctx { threads: ctx.threads, verbose: ctx.verbose, interrupt: ctx.interrupt, log: &|_: &str| {} };
        let ctx = if conv == GlmConvention::Fixest { &muted } else { ctx };
        par::with_pool(par::threads_for(ctx.threads, self.n()), |pool| self.glm_in_pool(rows, lay.nx, offset, s, g, conv, want_parts, ctx, pool))?
    }

    /// ppmlhdfe `remove_collinears` on the current sample, with the true weights.
    fn glm_collinear(&mut self, d: &mut GlmData, status: &mut [u8], itol: f64, inner_maxiter: usize, pool: &Pool, ctx: &Ctx) -> Result<(), RfError> {
        let k = d.k;
        if k == 0 {
            return Ok(());
        }
        self.absorber.set_weights(d.tw.clone(), pool, ctx)?;
        let mut xt = d.x.clone();
        let opts = SolveOpts { tol: itol, maxiter: inner_maxiter, want_coefs: false };
        self.absorber.residualize(&mut xt, k, &opts, None, pool, ctx)?;
        let absorbed = ols::stage1_absorbed(&col_sumsq(&xt, k, pool), &col_sumsq(&d.x, k, pool), ols::collinear_tol(itol));
        if absorbed.iter().any(|a| *a) {
            pool.install(|| {
                xt.par_chunks_mut(k).for_each(|row| {
                    for (v, a) in row.iter_mut().zip(&absorbed) {
                        if *a {
                            *v = 0.0;
                        }
                    }
                })
            });
        }
        let gram = stats::gram(&xt, k, d.tw.as_deref(), pool).0;
        let all: Vec<usize> = (0..k).collect();
        let (dropped, _) = ols::stage2_collinear(&gram, k, &all, &vec![1.0; k]);
        let keep: Vec<usize> = (0..k).filter(|&j| !absorbed[j] && !dropped[j]).collect();
        for j in 0..k {
            if absorbed[j] {
                status[d.cols[j]] = STATUS_ABSORBED;
            } else if dropped[j] {
                status[d.cols[j]] = STATUS_COLLINEAR;
            }
        }
        if keep.len() < k {
            let n = d.n;
            let mut x = par::alloc_vec(n * keep.len(), 0.0)?;
            let kk = keep.len();
            if kk > 0 {
                pool.install(|| {
                    x.par_chunks_mut(kk).enumerate().for_each(|(i, row)| {
                        for (o, &j) in row.iter_mut().zip(&keep) {
                            *o = d.x[i * k + j];
                        }
                    })
                });
            }
            d.x = x;
            d.cols = keep.iter().map(|&j| d.cols[j]).collect();
            d.sd_x = keep.iter().map(|&j| d.sd_x[j]).collect();
            d.k = kk;
        }
        Ok(())
    }

    /// Rows (row order) that the `fe` rule separates once the `seed` rows are dropped: rows with
    /// y = 0 in groups of an intercept term whose outcomes are all zero, iterated together with
    /// singleton removal (when the session drops singletons) to a fixed point. Each such group's
    /// indicator certifies it on the sample left by the earlier steps; y > 0 rows the iteration
    /// removes are singletons, not separated, and are left to [`Session::drop_obs`].
    fn fe_rule_rows(&self, y: &[f64], seed: &[bool], pool: &Pool) -> Vec<bool> {
        let order = self.row_order();
        let n = order.len();
        let mut yk = vec![0.0; n];
        let mut alive = vec![true; n];
        for (row, &kk) in order.iter().enumerate() {
            yk[kk as usize] = y[row];
            alive[kk as usize] = !seed[row];
        }
        let terms: Vec<TermCodes> = self.terms.iter().filter(|t| t.intercept).map(|t| TermCodes { codes: &t.codes, g: t.g }).collect();
        let fw = (self.weight_kind == WeightKind::Fweight).then_some(self.weights.as_deref()).flatten();
        sample::drop_singletons(&terms, &mut alive, fw, Some(SepRule { y: &yk, family: Family::Poisson }), self.drop_singletons, pool);
        order.iter().enumerate().map(|(row, &kk)| !alive[kk as usize] && !seed[row] && y[row] == 0.0).collect()
    }

    /// Poisson separation before IRLS, in rounds until a ReLU check finds nothing: the `fe` rule
    /// (when the session did not apply it at prepare: [`Session::fe_rule_rows`]) on the sample
    /// left by the previous round, then the ReLU check ([`separation::relu`]). ReLU converges
    /// to some certificate of separation, not necessarily one of maximal support (on a field
    /// panel without the `fe` rule it left rows of all-zero firm groups in the sample, and which
    /// ones depended on rounding), so the rows it finds are dropped and the check is repeated
    /// on the rest; certificates of successive rounds combine into one for all of them. Every
    /// round drops its rows ([`Session::glm_drop`]) and re-runs the collinearity check.
    #[allow(clippy::too_many_arguments)]
    fn glm_separate(
        &mut self,
        d: &mut GlmData,
        status: &mut [u8],
        itol: f64,
        inner_maxiter: usize,
        pool: &Pool,
        ctx: &Ctx,
    ) -> Result<SepRounds, RfError> {
        // Kept position (at entry) of every current row.
        let mut origin: Vec<f64> = self.row_order().iter().map(|&k| f64::from(k)).collect();
        let mut out = SepRounds { tagged: Vec::new(), complete: true, iterations: 0 };
        let mut seed = vec![false; d.n];
        let mut cert = vec![0.0; d.n];
        loop {
            if !self.sep_fe {
                let fe = self.fe_rule_rows(&d.y, &seed, pool);
                let nfe = fe.iter().filter(|v| **v).count();
                if nfe > 0 {
                    ctx.log(&format!("{{txt}}({nfe} observation{} separated by a fixed effect: all-zero outcomes within the group)", plural(nfe)));
                }
                for ((s, c), f) in seed.iter_mut().zip(cert.iter_mut()).zip(fe) {
                    if f {
                        *s = true;
                        *c = 1.0;
                    }
                }
            }
            if seed.iter().any(|v| *v) {
                out.tagged.extend((0..d.n).filter(|&i| seed[i]).map(|i| (origin[i] as u32, cert[i])));
                self.glm_drop(d, &seed, &mut [&mut origin], pool, ctx)?;
                self.glm_collinear(d, status, itol, inner_maxiter, pool, ctx)?;
            }
            let r = separation::relu(self.absorber.as_mut(), d, inner_maxiter, pool, ctx)?;
            out.iterations += r.iterations;
            out.complete &= r.complete;
            match r.separated {
                Some(sep) => {
                    seed = sep;
                    cert = r.certificate;
                }
                None => return Ok(out),
            }
        }
    }

    /// Drops the separated rows (row order) through [`Session::drop_obs`] and remaps `d` and the
    /// `extra` row vectors to the new row order.
    fn glm_drop(&mut self, d: &mut GlmData, sep: &[bool], extra: &mut [&mut Vec<f64>], pool: &Pool, ctx: &Ctx) -> Result<DropReport, RfError> {
        let old_kept = self.kept_index.clone();
        let old_order: Vec<u32> = self.row_order().to_vec();
        let mut drop = vec![false; old_kept.len()];
        for (row, &s) in sep.iter().enumerate() {
            if s {
                drop[old_order[row] as usize] = true;
            }
        }
        let rep = self.drop_obs(&drop, ctx)?;
        // Old kept position of every new kept observation, then old row of every new row.
        let mut pos = Vec::with_capacity(self.n());
        let mut j = 0usize;
        for &inp in &self.kept_index {
            while old_kept[j] != inp {
                j += 1;
            }
            pos.push(j);
            j += 1;
        }
        let mut inv = vec![0usize; old_kept.len()];
        for (row, &kk) in old_order.iter().enumerate() {
            inv[kk as usize] = row;
        }
        let map: Vec<usize> = self.row_order().iter().map(|&kk| inv[pos[kk as usize]]).collect();
        let pick = |v: &[f64]| -> Vec<f64> { map.iter().map(|&m| v[m]).collect() };
        let k = d.k;
        d.y = pick(&d.y);
        d.x = map.iter().flat_map(|&m| d.x[m * k..(m + 1) * k].iter().copied()).collect();
        d.off = d.off.as_deref().map(pick);
        d.tw = self.weights.as_deref().map(|w| vcov::to_row_order(w, self.row_order(), pool)).transpose()?;
        d.n = map.len();
        for v in extra.iter_mut() {
            **v = pick(v);
        }
        let more = rep.singletons + rep.fe_separated;
        if more > 0 {
            ctx.log(&format!("{{txt}}(dropped {more} singleton observation{})", plural(more as usize)));
        }
        Ok(rep)
    }

    /// Splits [y, x…] (row order), validates and standardizes them (y for Poisson, x always),
    /// and drops regressors collinear with the fixed effects or each other. Returns the working
    /// data, the status per regressor sent, and sd_y.
    #[allow(clippy::too_many_arguments)]
    fn glm_data(
        &mut self,
        rows: Vec<f64>,
        nx: usize,
        offset: Option<Vec<f64>>,
        g: &GlmOptions,
        inner_maxiter: usize,
        pool: &Pool,
        ctx: &Ctx,
    ) -> Result<(GlmData, Vec<u8>, f64), RfError> {
        let fam = g.family;
        let n0 = self.n();
        let r0 = 1 + nx;

        // Split [y, x…] and standardize.
        let mut y: Vec<f64> = rows.iter().step_by(r0).copied().collect();
        let mut x = par::alloc_vec(n0 * nx, 0.0)?;
        if nx > 0 {
            pool.install(|| x.par_chunks_mut(nx).zip(rows.par_chunks(r0)).for_each(|(o, row)| o.copy_from_slice(&row[1..])));
        }
        drop(rows);
        if y.iter().chain(&x).any(|v| !v.is_finite()) || offset.as_ref().is_some_and(|o| o.iter().any(|v| !v.is_finite())) {
            return Err(RfError::InvalidInput("glm: missing values in the data".into()));
        }
        match fam {
            Family::Poisson => {
                if y.iter().any(|&v| v < 0.0) {
                    return Err(RfError::Code(459, "dependent variable must be greater than or equal to zero".into()));
                }
            }
            _ => y.iter_mut().for_each(|v| *v = f64::from(u8::from(*v != 0.0))),
        }
        // Dividing y by sd_y shifts η by −ln sd_y, which only a free constant absorbs: without
        // one in the absorbed space (slope-only terms) y keeps its scale.
        let sd_y = if fam == Family::Poisson && self.has_intercept() { standardize_sd(y.iter().copied()) } else { 1.0 };
        if sd_y != 1.0 {
            y.iter_mut().for_each(|v| *v /= sd_y);
            let min_pos = y.iter().copied().filter(|v| *v > 0.0).fold(f64::INFINITY, f64::min);
            if min_pos <= 1e-6 {
                ctx.log(&format!("{{err}}warning: dependent variable takes very low values after standardizing ({min_pos:e})"));
            }
        }
        let sd_x: Vec<f64> = (0..nx).map(|j| standardize_sd((0..n0).map(|i| x[i * nx + j]))).collect();
        if nx > 0 {
            pool.install(|| {
                x.par_chunks_mut(nx).for_each(|row| {
                    for (v, s) in row.iter_mut().zip(&sd_x) {
                        *v /= s;
                    }
                })
            });
        }
        let tw = self.weights.as_deref().map(|w| vcov::to_row_order(w, self.row_order(), pool)).transpose()?;
        let mut d = GlmData { family: fam, n: n0, y, k: nx, x, cols: (0..nx).collect(), sd_x, off: offset, tw };
        let mut status = vec![STATUS_OK; nx];
        self.glm_collinear(&mut d, &mut status, g.itol, inner_maxiter, pool, ctx)?;
        ctx.check()?;
        Ok((d, status, sd_y))
    }

    #[allow(clippy::too_many_arguments)]
    fn glm_in_pool(
        &mut self,
        rows: Vec<f64>,
        nx: usize,
        offset: Option<Vec<f64>>,
        s: &Spec,
        g: &GlmOptions,
        conv: GlmConvention,
        want_parts: bool,
        ctx: &Ctx,
        pool: &Pool,
    ) -> Result<(EstResult, Option<GlmParts>), RfError> {
        let fam = g.family;
        let inner_maxiter = s.maxiter;
        let (mut d, mut status, sd_y) = self.glm_data(rows, nx, offset, g, inner_maxiter, pool, ctx)?;

        // Separation (relu; `fe` ran at prepare).
        let mut num_sep = 0u64;
        // False when the ReLU check could not certify its result; the IRLS-step `mu` rule then
        // runs as a fallback and the fit reports the incomplete check.
        let mut sep_complete = true;
        if g.separation.relu {
            let rounds = self.glm_separate(&mut d, &mut status, g.itol, inner_maxiter, pool, ctx)?;
            sep_complete = rounds.complete;
            num_sep += rounds.tagged.len() as u64;
        }

        // Initial values of μ: fixest's `family$initialize` in fixest mode (Poisson y + 0.1, on
        // the standardized scale; binomial (w·y + 1/2)/(w + 1) with the prior weights), else
        // ppmlhdfe `guess()` and, for logit, (y + 1/2)/2.
        let fixest = conv == GlmConvention::Fixest;
        let mut mu: Vec<f64> = match (fam, g.guess) {
            (Family::Poisson, _) if fixest => d.y.iter().map(|y| y + 0.1 / sd_y).collect(),
            (Family::Logit, _) if fixest => {
                let tw = d.tw.as_deref();
                d.y.iter().enumerate().map(|(i, y)| tw.map_or((y + 0.5) / 2.0, |t| (t[i] * y + 0.5) / (t[i] + 1.0))).collect()
            }
            (Family::Logit, _) => d.y.iter().map(|y| (y + 0.5) / 2.0).collect(),
            (_, Guess::Simple) => {
                let ybar = wmean(&d.y, d.tw.as_deref(), pool);
                d.y.iter().map(|y| 0.5 * (y + ybar)).collect()
            }
            (_, Guess::Ols) => self.glm_ols_guess(&d, g, inner_maxiter, pool, ctx)?,
        };

        // IRLS.
        let highest = g.itol.min(0.1 * g.tol).max(1e-12);
        let mut cfg = IrlsCfg {
            tol: g.tol,
            maxiter: g.maxiter,
            highest,
            start_tol: irls::START_INNER_TOL.max(g.tol),
            inner_maxiter,
            exact: map_exact(&self.terms),
            log: g.log,
            sd_y,
            check_sep: g.separation.mu || !sep_complete,
            fixest,
        };
        let mut eta = vec![0.0; d.n];
        let mut out: IrlsOut = irls::irls(self.absorber.as_mut(), &d, &mut mu, &mut eta, &cfg, pool, ctx)?;
        let mut iter = out.iter;
        let mut subiter = out.subiter;
        if out.n_sep > 0 {
            let n_sep = out.n_sep;
            ctx.log(&format!("{{txt}}(IRLS step detected {n_sep} separated observation{})", plural(n_sep)));
            num_sep += n_sep as u64;
            let sep = std::mem::take(&mut out.separated);
            cfg.start_tol = out.tol_in;
            drop(out);
            self.glm_drop(&mut d, &sep, &mut [&mut mu, &mut eta], pool, ctx)?;
            self.glm_collinear(&mut d, &mut status, g.itol, inner_maxiter, pool, ctx)?;
            cfg.check_sep = false;
            cfg.exact = map_exact(&self.terms);
            out = irls::irls(self.absorber.as_mut(), &d, &mut mu, &mut eta, &cfg, pool, ctx)?;
            iter += out.iter;
            subiter += out.subiter;
        }
        if g.log {
            ctx.log("{txt}{hline 108}");
            ctx.log("{txt}(legend: {res}p{txt}: exact partial-out   {res}s{txt}: exact solver   {res}h{txt}: step-halving   {res}o{txt}: epsilon below tolerance)");
        }
        ctx.log(&format!("{{txt}}Converged in {iter} iterations and {subiter} HDFE sub-iterations (tol = {})", fmt_e(g.tol, 1)));
        ctx.check()?;
        let fin = Finish { d: &d, out: &out, mu: &mu, eta: &eta, nx, sd_y, num_sep, sep_complete, iter, subiter, conv, want_parts };
        self.glm_finish(&fin, status, s, g, pool)
    }

    /// ppmlhdfe `guess(ols)`: OLS of ln(y + ȳ/100) on X and the fixed effects (true weights,
    /// tolerance max(1e-4, tol)); μ = exp(ln(y + 1) − residual).
    fn glm_ols_guess(&mut self, d: &GlmData, g: &GlmOptions, inner_maxiter: usize, pool: &Pool, ctx: &Ctx) -> Result<Vec<f64>, RfError> {
        let (n, k) = (d.n, d.k);
        let r = 1 + k;
        let ybar = wmean(&d.y, d.tw.as_deref(), pool);
        self.absorber.set_weights(d.tw.clone(), pool, ctx)?;
        let mut block = par::alloc_vec(n * r, 0.0)?;
        pool.install(|| {
            block.par_chunks_mut(r).enumerate().for_each(|(i, row)| {
                row[0] = (d.y[i] + ybar / 100.0).ln();
                row[1..].copy_from_slice(&d.x[i * k..(i + 1) * k]);
            })
        });
        let opts = SolveOpts { tol: irls::START_INNER_TOL.max(g.tol), maxiter: inner_maxiter, want_coefs: false };
        self.absorber.residualize(&mut block, r, &opts, None, pool, ctx)?;
        let xcols: Vec<usize> = (1..r).collect();
        let b = wls::qr_coefs(&block, r, 0, &xcols, d.tw.as_deref(), pool);
        Ok((0..n)
            .map(|i| {
                let row = &block[i * r..(i + 1) * r];
                let e = row[0] - b.iter().zip(&row[1..]).map(|(b, x)| b * x).sum::<f64>();
                ((d.y[i] + 1.0).ln() - e).exp()
            })
            .collect())
    }

    /// Final β, VCE, statistics and outputs (ppmlhdfe `GLM::solve` after IRLS).
    fn glm_finish(
        &mut self,
        f: &Finish,
        mut status: Vec<u8>,
        s: &Spec,
        g: &GlmOptions,
        pool: &Pool,
    ) -> Result<(EstResult, Option<GlmParts>), RfError> {
        let Finish { d, out, mu, eta, nx, sd_y, num_sep, sep_complete, iter, subiter, conv, want_parts } = *f;
        let fam = d.family;
        let n = d.n;
        let k = d.k;
        let r = 1 + k;
        let block = &out.block;
        let w = &out.w;
        let tw = d.tw.as_deref();
        let fw = self.weight_kind == WeightKind::Fweight;
        let big_n =
            if fw { tw.map_or(n as f64, |t| par::sum_chunks(pool, n, CHUNK, 1, |rg, a| a[0] += t[rg].iter().sum::<f64>())[0]) } else { n as f64 };
        let sumw = par::sum_chunks(pool, n, CHUNK, 1, |rg, a| a[0] += w[rg].iter().sum::<f64>())[0];
        let cw = big_n / sumw;
        let wv: Vec<f64> = w.iter().map(|v| v * cw).collect();
        let dof = dof::estimate_dof(&self.terms, &self.clusters, n, &s.dof, s.want_groupvar, true);

        // reghdfe_rmcoll on the final weighted Gram (normally drops nothing).
        let gfull = stats::gram(block, r, Some(&wv), pool).0;
        let gxx: Vec<f64> = (0..k * k).map(|ab| gfull[(1 + ab / k) * r + 1 + ab % k]).collect();
        let (inv_xx, dropped) = vcov::invsym(&gxx, k, INVSYM_TOL);
        let mut beta = out.beta.clone();
        let mut resid_owned: Option<Vec<f64>> = None;
        if dropped.iter().any(|v| *v) {
            let kept_cols: Vec<usize> = (0..k).filter(|&j| !dropped[j]).map(|j| j + 1).collect();
            let bk = wls::qr_coefs(block, r, 0, &kept_cols, Some(w), pool);
            beta = vec![0.0; k];
            for (c, b) in kept_cols.iter().zip(bk) {
                beta[c - 1] = b;
            }
            resid_owned = Some(
                (0..n)
                    .map(|i| {
                        let row = &block[i * r..(i + 1) * r];
                        row[0] - beta.iter().zip(&row[1..]).map(|(b, x)| b * x).sum::<f64>()
                    })
                    .collect(),
            );
            for j in 0..k {
                if dropped[j] {
                    status[d.cols[j]] = STATUS_COLLINEAR;
                }
            }
        }
        let e: &[f64] = resid_owned.as_deref().unwrap_or(&out.resid);
        let rank = (0..k).filter(|&j| !dropped[j]).count();
        let df_m = rank as f64;

        // Constant: regression through the IRLS-weighted means.
        let xbar: Vec<f64> = if k == 0 {
            Vec::new()
        } else {
            let acc = par::sum_chunks(pool, n, CHUNK, k, |range, acc| {
                for i in range {
                    for (a, v) in acc.iter_mut().zip(&d.x[i * k..(i + 1) * k]) {
                        *a += w[i] * v;
                    }
                }
            });
            acc.iter().map(|a| a / sumw).collect()
        };
        let lin_mean = match fam {
            // ppmlhdfe: mean(log(mu)) − mean(offset).
            Family::Poisson => {
                let lm: Vec<f64> = mu.iter().map(|m| m.ln()).collect();
                wmean(&lm, Some(w), pool) - d.off.as_deref().map_or(0.0, |o| wmean(o, Some(w), pool))
            }
            _ => {
                let lm: Vec<f64> = (0..n).map(|i| eta[i] - d.off.as_deref().map_or(0.0, |o| o[i])).collect();
                wmean(&lm, Some(w), pool)
            }
        };
        let b0_s = lin_mean - xbar.iter().zip(&beta).map(|(m, b)| m * b).sum::<f64>();
        let msize = k + 1;
        let bread = vcov::inv_ext(&inv_xx, k, &xbar, big_n);

        // VCE on the standardized-x scale. ppmlhdfe forces robust for Poisson and pweights;
        // fixest mode keeps the requested VCE (the fit layer computes its own).
        let vce = match (&self.vce, fam, self.weight_kind, conv) {
            (Vce::Unadjusted, Family::Poisson, _, GlmConvention::Ppmlhdfe) | (Vce::Unadjusted, _, WeightKind::Pweight, GlmConvention::Ppmlhdfe) => {
                Vce::Robust
            }
            (v, ..) => v.clone(),
        };
        let fill_row = |i: usize, a: &mut [f64]| {
            let row = &block[i * r..(i + 1) * r];
            for j in 0..k {
                a[j] = row[1 + j] + xbar[j];
            }
            a[k] = 1.0;
        };
        let mut notes = Vec::new();
        if !out.converged {
            notes.push(format!(
                "Absence of convergence: maximum number of IRLS iterations reached ({}). Final deviance: {}.",
                g.maxiter,
                out.deviance * sd_y
            ));
        }
        if !sep_complete {
            notes.push(
                "warning: the ReLU separation check did not converge; the IRLS-step (mu) rule was applied instead, and separated observations may remain in the sample (e(separation_complete) = 0)"
                    .to_string(),
            );
        }
        let mut df_r = big_n - dof.df_a as f64 - df_m;
        let mut n_clust: Option<(f64, Vec<f64>)> = None;
        let mut v_s = match &vce {
            Vce::Unadjusted => bread.iter().map(|v| v * cw).collect::<Vec<f64>>(),
            Vce::Robust => {
                let meat = vcov::robust_meat(n, msize, pool, |i, a| {
                    fill_row(i, a);
                    let sc = e[i] * wv[i];
                    if fw { sc * sc / tw.map_or(1.0, |t| t[i]) } else { sc * sc }
                });
                vcov::sandwich(&bread, &meat, msize, big_n / (big_n - 1.0))
            }
            Vce::Cluster(_) => {
                let dims: Vec<ClusterCodes> = self.clusters.iter().map(|c| ClusterCodes { codes: &c.codes, g: c.g }).collect();
                let meat = vcov::multiway_meat(&dims, self.absorber.row_order(), msize, pool, |i, a| {
                    fill_row(i, a);
                    e[i] * wv[i]
                })?;
                let g_min = self.clusters.iter().map(|c| c.g).min().unwrap_or(0) as f64;
                let mut v = vcov::sandwich(&bread, &meat, msize, g_min / (g_min - 1.0));
                if self.clusters.len() > 1 && vcov::fix_psd(&mut v, msize, false) {
                    notes.push("Warning: VCV matrix was non-positive semi-definite; adjustment from Cameron, Gelbach & Miller applied.".to_string());
                }
                df_r = df_r.min(g_min - 1.0);
                n_clust = Some((g_min, self.clusters.iter().map(|c| c.g as f64).collect()));
                v
            }
        };

        // Wald test on the non-constant coefficients.
        let kept_idx: Vec<usize> = (0..k).filter(|&j| !dropped[j]).collect();
        let mut wald = f64::NAN;
        if !kept_idx.is_empty() {
            let vkk = sub_block(&v_s, msize, &kept_idx);
            let kk = kept_idx.len();
            let (inv_v, dv) = vcov::invsym(&vkk, kk, INVSYM_TOL);
            if dv.iter().any(|v| *v) {
                notes.push("warning: missing F statistic; dropped variables due to collinearity or too few clusters".to_string());
            } else {
                let bk: Vec<f64> = kept_idx.iter().map(|&j| beta[j]).collect();
                wald = (0..kk).map(|a| (0..kk).map(|b| bk[a] * inv_v[a * kk + b] * bk[b]).sum::<f64>()).sum::<f64>() / df_m;
                if !wald.is_finite() {
                    notes.push("warning: missing F statistic".to_string());
                    wald = f64::NAN;
                }
            }
        }
        if v_s.iter().any(|v| v.is_nan()) {
            v_s.iter_mut().for_each(|v| *v = 0.0);
        }
        let chi2 = wald * df_m;
        // ppmlhdfe leaves reghdfe5's e(rss) and e(rmse) of the final weighted least squares.
        let rss = par::sum_chunks(pool, n, CHUNK, 1, |range, acc| {
            for i in range {
                acc[0] += wv[i] * e[i] * e[i];
            }
        })[0];
        let used_df_r = big_n - dof.df_a as f64 - df_m - dof.df_a_nested as f64;
        let rmse = if used_df_r != 0.0 { (rss / used_df_r).sqrt() } else { rss.sqrt() };

        // Undo the standardization.
        let sx: Vec<f64> = d.sd_x.iter().copied().chain(std::iter::once(1.0)).collect();
        let mut b_raw: Vec<f64> = beta.iter().chain(std::iter::once(&b0_s)).zip(&sx).map(|(b, s)| b / s).collect();
        if fam == Family::Poisson {
            b_raw[k] += sd_y.ln();
        }
        let v_raw: Vec<f64> = (0..msize * msize).map(|ab| v_s[ab] / (sx[ab / msize] * sx[ab % msize])).collect();
        let k_all = nx + 1;
        let pos: Vec<usize> = d.cols.iter().copied().chain(std::iter::once(nx)).collect();
        let mut b_out = vec![0.0; k_all];
        let mut v_out = vec![0.0; k_all * k_all];
        for (a, &pa) in pos.iter().enumerate() {
            if a < k && dropped[a] {
                continue;
            }
            b_out[pa] = b_raw[a];
            for (bidx, &pb) in pos.iter().enumerate() {
                if bidx < k && dropped[bidx] {
                    continue;
                }
                v_out[pa * k_all + pb] = v_raw[a * msize + bidx];
            }
        }

        // Log-likelihoods.
        let (ll, ll_0) = match fam {
            Family::Poisson => {
                let ll0mu = wmean(&d.y, tw, pool);
                let ln_sd = sd_y.ln();
                let acc = par::sum_chunks(pool, n, CHUNK, 2, |range, acc| {
                    for i in range {
                        let wi = tw.map_or(1.0, |t| t[i]);
                        let y = d.y[i];
                        if y == 0.0 {
                            acc[0] -= wi * mu[i] * sd_y;
                            acc[1] -= wi * ll0mu * sd_y;
                        } else {
                            let ys = y * sd_y;
                            let lg = family::ln_gamma(ys + 1.0);
                            acc[0] += wi * (ys * (eta[i] + ln_sd) - mu[i] * sd_y - lg);
                            acc[1] += wi * (y * (sd_y * (ll0mu.ln() + ln_sd)) - ll0mu * sd_y - lg);
                        }
                    }
                });
                (acc[0], acc[1])
            }
            _ => {
                let p = wmean(&d.y, tw, pool);
                let acc = par::sum_chunks(pool, n, CHUNK, 2, |range, acc| {
                    for i in range {
                        let wi = tw.map_or(1.0, |t| t[i]);
                        let y = d.y[i];
                        acc[0] += wi * family::logit_ll(y, eta[i]);
                        acc[1] += wi * if y > 0.0 { p.ln() } else { (1.0 - p).ln() };
                    }
                });
                (acc[0], acc[1])
            }
        };

        let (num_singletons, num_separated) = match fam {
            Family::Poisson => (self.num_singletons + self.num_fe_separated, num_sep),
            _ => (self.num_singletons, self.num_fe_separated + num_sep),
        };
        let mut res = EstResult { status, notes, ..EstResult::default() };
        let scal = |res: &mut EstResult, name: &str, v: f64| res.set_scalar(name, missing_if_nonfinite(v));
        scal(&mut res, "N", big_n);
        scal(&mut res, "rank", df_m);
        if fam == Family::Poisson {
            scal(&mut res, "df", df_r);
            scal(&mut res, "rss", rss);
            scal(&mut res, "rmse", rmse);
        }
        scal(&mut res, "df_m", df_m);
        scal(&mut res, "ll", ll);
        scal(&mut res, "ll_0", ll_0);
        scal(&mut res, "deviance", out.deviance * sd_y);
        scal(&mut res, "r2_p", 1.0 - ll / ll_0);
        scal(&mut res, "chi2", chi2);
        scal(&mut res, "converged", f64::from(u8::from(out.converged)));
        scal(&mut res, "ic", iter as f64);
        scal(&mut res, "ic2", subiter as f64);
        scal(&mut res, "num_separated", num_separated as f64);
        scal(&mut res, "separation_complete", f64::from(u8::from(sep_complete)));
        scal(&mut res, "num_singletons", num_singletons as f64);
        scal(&mut res, "num_fe_separated", self.num_fe_separated as f64);
        scal(&mut res, "N_full", big_n + (num_singletons + num_separated) as f64);
        if let Some((g_min, list)) = &n_clust {
            scal(&mut res, "N_clust", *g_min);
            scal(&mut res, "N_clustervars", list.len() as f64);
            for (i, v) in list.iter().enumerate() {
                scal(&mut res, &format!("N_clust{}", i + 1), *v);
            }
        }
        scal(&mut res, "df_a", dof.df_a as f64);
        scal(&mut res, "df_a_initial", dof.df_a_initial as f64);
        scal(&mut res, "df_a_redundant", dof.df_a_redundant as f64);
        scal(&mut res, "df_a_nested", dof.df_a_nested as f64);
        let cons = b_out[nx];
        res.matrices.push(NamedMat { name: "b".into(), rows: 1, cols: k_all, data: b_out });
        res.matrices.push(NamedMat { name: "V".into(), rows: k_all, cols: k_all, data: v_out });
        res.matrices.push(NamedMat { name: "dof_table".into(), rows: dof.slots.len(), cols: 5, data: dof.table() });

        // d = Σ FE (weighted mean 0 under the final IRLS weights) and saved FEs.
        let want_coefs = s.want_d || (s.want_fe && self.terms.iter().any(|t| t.save));
        if want_coefs {
            let coefs = out.coefs.as_ref().ok_or_else(|| RfError::Internal("absorber returned no FE coefficients".into()))?;
            let mut c = vec![0.0; r];
            c[0] = 1.0;
            for j in 0..k {
                c[1 + j] = -beta[j];
            }
            let combined = coefs.combine(&c);
            let fit = self.absorber.fit_rows(&combined, 0, pool);
            let shift = wmean(&fit, Some(w), pool);
            if s.want_d {
                res.d = Some(fe_recovery::d_values(self.absorber.as_ref(), &combined, shift, pool)?);
            }
            if s.want_fe {
                res.fe = fe_recovery::saved_fe(self.absorber.as_ref(), &self.terms, &combined, shift);
            }
        }
        if s.want_groupvar {
            res.groupvar = Some(match &dof.groupvar {
                Some(gv) => gv.iter().map(|&v| f64::from(v)).collect(),
                None => vec![f64::NAN; n],
            });
        }
        let parts = if want_parts { Some(self.glm_parts_out(f, &beta, &dropped, e, &res.status, cons, pool)?) } else { None };
        Ok((res, parts))
    }

    /// [`GlmParts`] in kept order on the original scale (see its field docs). `beta` and `dropped`
    /// are over the columns of `d.x` (standardized scale), `e` the working residuals (row order).
    #[allow(clippy::too_many_arguments)]
    fn glm_parts_out(
        &self,
        f: &Finish,
        beta: &[f64],
        dropped: &[bool],
        e: &[f64],
        status: &[u8],
        cons: f64,
        pool: &Pool,
    ) -> Result<GlmParts, RfError> {
        let Finish { d, out, mu, eta, nx, sd_y, iter, .. } = *f;
        let (n, k) = (d.n, d.k);
        let r = 1 + k;
        let order = self.absorber.row_order();
        // y was divided by sd_y (Poisson): μ, the IRLS weights and the deviance scale by sd_y and
        // η shifts by ln sd_y; x'β, the offset and the working residuals are scale-free.
        let ln_sd = sd_y.ln();
        let kept: Vec<usize> = (0..k).filter(|&j| !dropped[j]).collect();
        let kk = kept.len();
        let mut coef = vec![f64::NAN; nx];
        for &j in &kept {
            coef[d.cols[j]] = beta[j] / d.sd_x[j];
        }
        let off = d.off.as_deref();
        let mut fe_sum = par::alloc_vec(n, 0.0)?;
        pool.install(|| {
            fe_sum.par_iter_mut().enumerate().for_each(|(i, v)| {
                let xb: f64 = beta.iter().zip(&d.x[i * k..(i + 1) * k]).map(|(b, x)| b * x).sum();
                *v = eta[i] - off.map_or(0.0, |o| o[i]) - xb + ln_sd;
            })
        });
        let scaled = |v: &[f64], a: f64, b: f64| -> Result<Vec<f64>, RfError> {
            let s: Vec<f64> = v.iter().map(|x| a * x + b).collect();
            fe_recovery::to_kept_order(&s, order, pool)
        };
        // Kept-order row-major xt: kept position p holds solver row inv[p].
        let mut inv = vec![0u32; n];
        for (row, &p) in order.iter().enumerate() {
            inv[p as usize] = row as u32;
        }
        let mut xt = par::alloc_vec(n * kk, 0.0)?;
        if kk > 0 {
            let block = &out.block;
            pool.install(|| {
                xt.par_chunks_mut(kk).zip(inv.par_iter()).for_each(|(o, &row)| {
                    let src = &block[row as usize * r + 1..(row as usize + 1) * r];
                    for (v, &j) in o.iter_mut().zip(&kept) {
                        *v = src[j] * d.sd_x[j];
                    }
                })
            });
        }
        let fe = if self.terms.iter().any(|t| t.save) {
            let coefs = out.coefs.as_ref().ok_or_else(|| RfError::Internal("absorber returned no FE coefficients".into()))?;
            let c: Vec<f64> = std::iter::once(1.0).chain(beta.iter().map(|b| -b)).collect();
            fe_recovery::saved_fe(self.absorber.as_ref(), &self.terms, &coefs.combine(&c), -ln_sd)
        } else {
            Vec::new()
        };
        Ok(GlmParts {
            coef,
            status: status.to_vec(),
            cons,
            eta: scaled(eta, 1.0, ln_sd)?,
            mu: scaled(mu, sd_y, 0.0)?,
            w: scaled(&out.w, sd_y, 0.0)?,
            k: kk,
            xt,
            working_resid: fe_recovery::to_kept_order(e, order, pool)?,
            fe_sum: fe_recovery::to_kept_order(&fe_sum, order, pool)?,
            fe,
            deviance: out.deviance * sd_y,
            iterations: iter,
            converged: out.converged,
        })
    }
}

/// Outcome of the IRLS phase, handed to [`Session::glm_finish`].
#[derive(Clone, Copy)]
struct Finish<'a> {
    d: &'a GlmData,
    out: &'a IrlsOut,
    mu: &'a [f64],
    eta: &'a [f64],
    nx: usize,
    sd_y: f64,
    num_sep: u64,
    sep_complete: bool,
    iter: usize,
    subiter: usize,
    conv: GlmConvention,
    want_parts: bool,
}
