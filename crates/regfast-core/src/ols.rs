//! OLS with reghdfe 6.13.1 conventions (plan Phase 3; reghdfe.mata `reghdfe_solve_ols`
//! l.3567-3760, `check_collinear_with_fe` l.2212, `reghdfe_rmcoll` l.3765, VCE l.3892-4186).
//!
//! reghdfe standardizes every variable by its unweighted sample sd before partialling out and runs
//! the collinearity sweep, the VCE, the PSD fix and the F test on that scale, then undoes the
//! scaling. Residualization is linear, so here the raw columns are residualized and the Gram
//! matrices are rescaled afterwards; all scale-sensitive decisions use the standardized matrices.

use rayon::prelude::*;

use crate::absorber::SolveOpts;
use crate::api::{Ctx, EstResult, Layout, NamedMat, Spec, Vce, WeightKind};
use crate::dof::{self, Dof};
use crate::error::RfError;
use crate::fe_recovery;
use crate::par::{self, CHUNK, Pool};
use crate::session::Session;
use crate::stats::{self, ColStats};
use crate::vcov::{self, ClusterCodes, INVSYM_TOL};

/// Regressor status codes of [`EstResult::status`].
pub(crate) const STATUS_OK: u8 = 0;
pub(crate) const STATUS_ABSORBED: u8 = 2;
pub(crate) const STATUS_COLLINEAR: u8 = 3;

/// reghdfe `collinear_tol = min(1e-6, tolerance/10)`.
pub(crate) fn collinear_tol(tol: f64) -> f64 {
    1e-6f64.min(tol / 10.0)
}

/// Stage 1 (reghdfe `check_collinear_with_fe`): column j is explained by the fixed effects when
/// Σz̃_j² / Σx_j² ≤ `ctol` (unweighted, raw uncentered denominator). A zero denominator gives a
/// missing ratio, which Stata treats as "not collinear" (the sweep then drops the column).
pub(crate) fn stage1_absorbed(ss_resid: &[f64], ss_raw: &[f64], ctol: f64) -> Vec<bool> {
    ss_resid.iter().zip(ss_raw).map(|(a, b)| a / b <= ctol).collect()
}

/// Stage 2 (reghdfe `reghdfe_rmcoll`, Mata `invsym(xx, 1..K)`): sweeps the columns `cols` of the
/// r×r Gram `g` in order on the standardized scale (entry (a,b) divided by sd_a·sd_b).
/// Returns (collinear flag per entry of `cols`, generalized inverse of the standardized block with
/// zero rows/columns for collinear entries, |cols|×|cols|).
pub(crate) fn stage2_collinear(g: &[f64], r: usize, cols: &[usize], sd: &[f64]) -> (Vec<bool>, Vec<f64>) {
    let k = cols.len();
    let gs: Vec<f64> = (0..k * k).map(|ab| {
        let (a, b) = (cols[ab / k], cols[ab % k]);
        g[a * r + b] / (sd[a] * sd[b])
    })
    .collect();
    let (inv, dropped) = vcov::invsym(&gs, k, INVSYM_TOL);
    (dropped, inv)
}

/// Extracts the sub-block of a k×k matrix on `keep` (indices into 0..k).
pub(crate) fn sub_block(m: &[f64], k: usize, keep: &[usize]) -> Vec<f64> {
    keep.iter().flat_map(|&a| keep.iter().map(move |&b| m[a * k + b])).collect()
}

/// Weights as used by reghdfe's regression formulas.
pub(crate) struct Weights {
    pub kind: WeightKind,
    /// Kept observations (rows).
    pub n_obs: usize,
    /// Σ raw weights (N_obs when unweighted): `e(sumweights)`.
    pub sumw: f64,
    /// N: Σfw with fweights, N_obs otherwise.
    pub n: f64,
    /// Raw weights in row order.
    pub raw: Option<Vec<f64>>,
    /// Regression weights w_v in row order: w·N_obs/Σw for aw/pw, w for fw, `None` unweighted.
    pub wv: Option<Vec<f64>>,
}

impl Weights {
    /// `kept` are the raw weights in kept order; `row_order` maps rows to kept observations.
    pub(crate) fn new(kind: WeightKind, kept: Option<&[f64]>, row_order: &[u32], pool: &Pool) -> Result<Weights, RfError> {
        let n_obs = row_order.len();
        let raw = kept.map(|w| vcov::to_row_order(w, row_order, pool)).transpose()?;
        let sumw = raw.as_deref().map_or(n_obs as f64, |w| par::sum_chunks(pool, w.len(), CHUNK, 1, |r, acc| acc[0] += w[r].iter().sum::<f64>())[0]);
        let n = if kind == WeightKind::Fweight { sumw } else { n_obs as f64 };
        let wv = match (kind, &raw) {
            (WeightKind::Aweight | WeightKind::Pweight, Some(w)) => {
                let f = n_obs as f64 / sumw;
                let mut v = par::alloc_vec(w.len(), 0.0)?;
                pool.install(|| v.par_iter_mut().zip(w.par_iter()).for_each(|(o, x)| *o = x * f));
                Some(v)
            }
            (WeightKind::Fweight, Some(w)) => Some(w.clone()),
            _ => None,
        };
        Ok(Weights { kind, n_obs, sumw, n, raw, wv })
    }

    /// w_v of row i (1 when unweighted).
    #[inline]
    pub(crate) fn wv(&self, i: usize) -> f64 {
        self.wv.as_ref().map_or(1.0, |w| w[i])
    }

    /// reghdfe `tmp_N` of the constant extension: N_obs for aw/pw, Σfw (= N) otherwise.
    pub(crate) fn n_const(&self) -> f64 {
        match self.kind {
            WeightKind::Aweight | WeightKind::Pweight => self.n_obs as f64,
            _ => self.sumw,
        }
    }

    /// reghdfe robust multiplier m_i given the residual: e² (none), e²·w (fw), (e·w_v)² (aw/pw).
    #[inline]
    pub(crate) fn robust_mult(&self, i: usize, e: f64) -> f64 {
        match self.kind {
            WeightKind::None => e * e,
            WeightKind::Fweight => e * e * self.wv(i),
            WeightKind::Aweight | WeightKind::Pweight => {
                let x = e * self.wv(i);
                x * x
            }
        }
    }
}

/// reghdfe TSS of each column: Σw(x − x̄_w)² with an intercept in the model, else Σw·x², scaled
/// by N_obs/Σw for aw/pw.
pub(crate) fn tss_of(cs: &ColStats, w: &Weights, has_intercept: bool) -> Vec<f64> {
    let scale = match w.kind {
        WeightKind::Aweight | WeightKind::Pweight => w.n_obs as f64 / w.sumw,
        _ => 1.0,
    };
    let base = if has_intercept { &cs.wss } else { &cs.wss_unc };
    base.iter().map(|v| v * scale).collect()
}

/// VCE flavour after reghdfe's rule "pweight + unadjusted → robust".
pub(crate) fn effective_vce(vce: &Vce, kind: WeightKind) -> Vce {
    match vce {
        Vce::Unadjusted if kind == WeightKind::Pweight => Vce::Robust,
        v => v.clone(),
    }
}

/// Stata turns every non-finite result into missing.
pub(crate) fn missing_if_nonfinite(x: f64) -> f64 {
    if x.is_finite() { x } else { f64::NAN }
}

impl Session {
    /// OLS on `rows` (N×R row-major in [`Session::row_order`], layout `[y, x...]`).
    pub fn ols(&mut self, rows: Vec<f64>, lay: &Layout, s: &Spec, ctx: &Ctx) -> Result<EstResult, RfError> {
        if lay.nendog != 0 || lay.ninst != 0 {
            return Err(RfError::InvalidInput("ols: layout has endogenous variables or instruments".into()));
        }
        if rows.len() != self.n() * lay.ncols() {
            return Err(RfError::InvalidInput("ols: data block size differs from N·R".into()));
        }
        par::with_pool(par::threads_for(ctx.threads, self.n()), |pool| self.ols_in_pool(rows, lay.nx, s, ctx, pool))?
    }

    fn ols_in_pool(&mut self, mut rows: Vec<f64>, nx: usize, s: &Spec, ctx: &Ctx, pool: &Pool) -> Result<EstResult, RfError> {
        let n = self.n();
        let r = 1 + nx;
        let has_intercept = self.has_intercept();
        let report_constant = has_intercept && !s.noconstant;
        let w = Weights::new(self.weight_kind, self.weights.as_deref(), self.absorber.row_order(), pool)?;
        let big_n = w.n;

        // Degrees of freedom do not depend on the data columns.
        let dof: Dof = dof::estimate_dof(&self.terms, &self.clusters, n, &s.dof, s.want_groupvar);
        ctx.check()?;

        // Pass A: raw column statistics.
        let cs = stats::col_stats(&rows, r, w.raw.as_deref(), pool);
        let tss_all = tss_of(&cs, &w, has_intercept);
        let sd = &cs.sd;
        ctx.check()?;

        // Residualize every column against the fixed effects.
        let want_coefs = s.want_d || (s.want_fe && self.terms.iter().any(|t| t.save));
        let opts = SolveOpts { tol: s.tol, maxiter: s.maxiter, want_coefs };
        let info = self.absorber.residualize(&mut rows, r, &opts, None, pool, ctx)?;
        ctx.check()?;

        // Pass B: weighted Gram of the residualized columns.
        let (mut g, ss_res) = stats::gram(&rows, r, w.wv.as_deref(), pool);

        // Collinearity, stage 1 (with the fixed effects).
        let ctol = collinear_tol(s.tol);
        let absorbed = stage1_absorbed(&ss_res, &cs.sumsq, ctol);
        let mut status = vec![STATUS_OK; nx];
        let y_absorbed = nx >= 1 && absorbed[0];
        if y_absorbed {
            // reghdfe zeroes the partialled-out depvar so tss_within, ll and ll_0 are exact.
            pool.install(|| rows.par_chunks_mut(r).for_each(|row| row[0] = 0.0));
            for j in 0..r {
                g[j] = 0.0;
                g[j * r] = 0.0;
            }
        }
        for j in 1..r {
            if absorbed[j] {
                status[j - 1] = STATUS_ABSORBED;
            }
        }

        // Stage 2: sweep the remaining regressors on the standardized scale.
        let cand: Vec<usize> = (1..r).filter(|&j| status[j - 1] == STATUS_OK).collect();
        let (dropped, inv_cand) = stage2_collinear(&g, r, &cand, sd);
        for (c, &j) in cand.iter().enumerate() {
            if dropped[c] {
                status[j - 1] = STATUS_COLLINEAR;
            }
        }
        let keep_pos: Vec<usize> = (0..cand.len()).filter(|&c| !dropped[c]).collect();
        let kept: Vec<usize> = keep_pos.iter().map(|&c| cand[c]).collect();
        let k = kept.len();
        let inv_s = sub_block(&inv_cand, cand.len(), &keep_pos);

        // Coefficients on the standardized scale, then raw.
        let sd_y = sd[0];
        let xy_s: Vec<f64> = kept.iter().map(|&j| g[j * r] / (sd[j] * sd_y)).collect();
        let b_s: Vec<f64> = (0..k).map(|a| (0..k).map(|b| inv_s[a * k + b] * xy_s[b]).sum()).collect();
        let beta: Vec<f64> = (0..k).map(|a| b_s[a] * sd_y / sd[kept[a]]).collect();

        // Pass C: residuals e = ỹ − X̃β (materialized only when needed) and rss = Σ w_v e².
        let vce = effective_vce(&self.vce, w.kind);
        let need_e = s.want_resid || !matches!(vce, Vce::Unadjusted);
        let resid_of = |i: usize| -> f64 {
            let row = &rows[i * r..(i + 1) * r];
            let mut e = row[0];
            for (a, &j) in kept.iter().enumerate() {
                e -= beta[a] * row[j];
            }
            e
        };
        let e: Option<Vec<f64>> = if need_e {
            let mut e = par::alloc_vec(n, 0.0)?;
            pool.install(|| e.par_iter_mut().enumerate().for_each(|(i, v)| *v = resid_of(i)));
            Some(e)
        } else {
            None
        };
        let rss = par::sum_chunks(pool, n, CHUNK, 1, |range, acc| {
            for i in range {
                let ei = e.as_ref().map_or_else(|| resid_of(i), |e| e[i]);
                acc[0] += w.wv(i) * ei * ei;
            }
        })[0];
        let rss_s = rss / (sd_y * sd_y);
        ctx.check()?;

        // Constant (regression through the means) and the extended bread.
        let means_s: Vec<f64> = kept.iter().map(|&j| cs.wmean[j] / sd[j]).collect();
        let ybar_s = cs.wmean[0] / sd_y;
        let b0_s = ybar_s - means_s.iter().zip(&b_s).map(|(m, b)| m * b).sum::<f64>();
        let msize = k + usize::from(report_constant);
        let bread = if report_constant { vcov::inv_ext(&inv_s, k, &means_s, w.n_const()) } else { inv_s.clone() };
        let mut b_full_s = b_s.clone();
        if report_constant {
            b_full_s.push(b0_s);
        }

        // VCE on the standardized scale.
        let df_a = dof.df_a as f64;
        let df_m = k as f64;
        let mut df_r = big_n - df_a - df_m;
        let mut notes = Vec::new();
        let mut v_s = vec![0.0; msize * msize];
        let mut n_clust: Option<(f64, Vec<f64>)> = None;
        // Raw-scale design row a_i = (x̃_i + x̄ if constant, 1 if constant); `scale` converts a raw
        // meat to the standardized one: M_s(a,b) = M(a,b) / (sc_a·sc_b·sd_y²).
        let xbar: Vec<f64> = kept.iter().map(|&j| cs.wmean[j]).collect();
        let fill_row = |i: usize, a: &mut [f64]| {
            let row = &rows[i * r..(i + 1) * r];
            for (p, &j) in kept.iter().enumerate() {
                a[p] = row[j] + if report_constant { xbar[p] } else { 0.0 };
            }
            if report_constant {
                a[k] = 1.0;
            }
        };
        let sc: Vec<f64> = kept.iter().map(|&j| sd[j]).chain(report_constant.then_some(1.0)).collect();
        let standardize_meat = |m: &mut [f64]| {
            for a in 0..msize {
                for b in 0..msize {
                    m[a * msize + b] /= sc[a] * sc[b] * sd_y * sd_y;
                }
            }
        };
        if msize > 0 {
            match &vce {
                Vce::Unadjusted => {
                    let adj = big_n / df_r;
                    v_s = bread.iter().map(|x| (rss_s / big_n) * adj * x).collect();
                }
                Vce::Robust => {
                    let e = e.as_deref().expect("residuals materialized for robust VCE");
                    let mut meat = vcov::robust_meat(n, msize, pool, |i, a| {
                        fill_row(i, a);
                        w.robust_mult(i, e[i])
                    });
                    standardize_meat(&mut meat);
                    v_s = vcov::sandwich(&bread, &meat, msize, big_n / (big_n - df_a - df_m));
                }
                Vce::Cluster(_) => {
                    let e = e.as_deref().expect("residuals materialized for cluster VCE");
                    let dims: Vec<ClusterCodes> = self.clusters.iter().map(|c| ClusterCodes { codes: &c.codes, g: c.g }).collect();
                    let mut meat = vcov::multiway_meat(&dims, self.absorber.row_order(), msize, pool, |i, a| {
                        fill_row(i, a);
                        e[i] * w.wv(i)
                    })?;
                    standardize_meat(&mut meat);
                    let g_min = self.clusters.iter().map(|c| c.g).min().unwrap_or(0) as f64;
                    let nested_adj = if dof.df_a_nested > 0 { 1.0 } else { 0.0 };
                    let adj = (big_n - 1.0) / (big_n - nested_adj - df_m - df_a) * g_min / (g_min - 1.0);
                    v_s = vcov::sandwich(&bread, &meat, msize, adj);
                    if self.clusters.len() > 1 && vcov::fix_psd(&mut v_s, msize, report_constant) {
                        notes.push("Warning: VCV matrix was non-positive semi-definite; adjustment from Cameron, Gelbach & Miller applied.".to_string());
                    }
                }
            }
        }
        if let Vce::Cluster(_) = &vce {
            let g_min = self.clusters.iter().map(|c| c.g).min().unwrap_or(0) as f64;
            df_r = df_r.min(g_min - 1.0);
            n_clust = Some((g_min, self.clusters.iter().map(|c| c.g as f64).collect()));
        }
        ctx.check()?;

        // Wald F on the non-constant coefficients (standardized scale, as reghdfe).
        let mut f_stat = f64::NAN;
        if k > 0 {
            let vkk = sub_block(&v_s, msize, &(0..k).collect::<Vec<_>>());
            let (inv_v, dropped_v) = vcov::invsym(&vkk, k, INVSYM_TOL);
            if dropped_v.iter().any(|&d| d) {
                notes.push("warning: missing F statistic; dropped variables due to collinearity or too few clusters".to_string());
            } else {
                let q: f64 = (0..k).map(|a| (0..k).map(|b| b_s[a] * inv_v[a * k + b] * b_s[b]).sum::<f64>()).sum();
                f_stat = q / df_m;
                if !f_stat.is_finite() {
                    notes.push("warning: missing F statistic".to_string());
                    f_stat = f64::NAN;
                }
            }
        }
        if v_s.iter().any(|x| x.is_nan()) {
            v_s.iter_mut().for_each(|x| *x = 0.0);
        }

        // Undo the standardization.
        let sx: Vec<f64> = sc.iter().map(|v| v / sd_y).collect();
        let b_raw: Vec<f64> = b_full_s.iter().zip(&sx).map(|(b, s)| b / s).collect();
        let v_raw: Vec<f64> = (0..msize * msize).map(|ab| v_s[ab] / (sx[ab / msize] * sx[ab % msize])).collect();
        let b0 = if report_constant { b_raw[k] } else { 0.0 };

        // Expand to every regressor sent (+ _cons): omitted entries are 0.
        let k_all = nx + usize::from(report_constant);
        let pos: Vec<usize> = kept.iter().map(|&j| j - 1).chain(report_constant.then_some(nx)).collect();
        let mut b_out = vec![0.0; k_all];
        let mut v_out = vec![0.0; k_all * k_all];
        for (a, &pa) in pos.iter().enumerate() {
            b_out[pa] = b_raw[a];
            for (b, &pb) in pos.iter().enumerate() {
                v_out[pa * k_all + pb] = v_raw[a * msize + b];
            }
        }

        // Fit statistics.
        let tss = tss_all[0];
        let tss_within = g[0];
        let used_df_r = big_n - df_a - df_m - dof.df_a_nested as f64;
        let fit = stats::fit_stats(big_n, rss, tss, tss_within, used_df_r, df_m, has_intercept);

        let mut res = EstResult { status, notes, ..EstResult::default() };
        let scal = |res: &mut EstResult, name: &str, v: f64| res.set_scalar(name, missing_if_nonfinite(v));
        scal(&mut res, "N", big_n);
        scal(&mut res, "rank", df_m);
        scal(&mut res, "df_r", df_r);
        scal(&mut res, "df_m", df_m);
        scal(&mut res, "tss", tss);
        scal(&mut res, "tss_within", tss_within);
        scal(&mut res, "rss", rss);
        scal(&mut res, "mss", fit.mss);
        scal(&mut res, "rmse", fit.rmse);
        scal(&mut res, "F", f_stat);
        scal(&mut res, "ll", fit.ll);
        scal(&mut res, "ll_0", fit.ll_0);
        scal(&mut res, "r2", fit.r2);
        scal(&mut res, "r2_within", fit.r2_within);
        scal(&mut res, "r2_a", fit.r2_a);
        scal(&mut res, "r2_a_within", fit.r2_a_within);
        scal(&mut res, "sumweights", w.sumw);
        scal(&mut res, "ic", info.iterations as f64);
        scal(&mut res, "converged", 1.0);
        scal(&mut res, "report_constant", f64::from(u8::from(report_constant)));
        if let Some((g_min, list)) = &n_clust {
            scal(&mut res, "N_clust", *g_min);
            scal(&mut res, "N_clustervars", list.len() as f64);
            for (i, v) in list.iter().enumerate() {
                scal(&mut res, &format!("N_clust{}", i + 1), *v);
            }
        }
        scal(&mut res, "df_a", df_a);
        scal(&mut res, "df_a_initial", dof.df_a_initial as f64);
        scal(&mut res, "df_a_redundant", dof.df_a_redundant as f64);
        scal(&mut res, "df_a_nested", dof.df_a_nested as f64);
        scal(&mut res, "num_singletons", self.num_singletons as f64);
        // reghdfe: e(N_full) = e(N) + num_singletons (e(N) is Σfw with fweights).
        scal(&mut res, "N_full", big_n + self.num_singletons as f64);
        scal(&mut res, "y_absorbed", f64::from(u8::from(y_absorbed)));

        res.matrices.push(NamedMat { name: "b".into(), rows: 1, cols: k_all, data: b_out });
        res.matrices.push(NamedMat { name: "V".into(), rows: k_all, cols: k_all, data: v_out });
        res.matrices.push(NamedMat { name: "dof_table".into(), rows: dof.slots.len(), cols: 5, data: dof.table() });

        // Per-observation outputs, kept order.
        let row_order = self.absorber.row_order();
        if s.want_resid {
            let e = e.as_deref().expect("residuals materialized when requested");
            res.resid = Some(fe_recovery::to_kept_order(e, row_order, pool)?);
        }
        if want_coefs {
            let coefs = info.coefs.as_ref().ok_or_else(|| RfError::Internal("absorber returned no FE coefficients".into()))?;
            let mut c = vec![0.0; r];
            c[0] = 1.0;
            for (a, &j) in kept.iter().enumerate() {
                c[j] = -beta[a];
            }
            let combined = coefs.combine(&c);
            if s.want_d {
                res.d = Some(fe_recovery::d_values(self.absorber.as_ref(), &combined, b0, pool)?);
            }
            if s.want_fe {
                res.fe = fe_recovery::saved_fe(self.absorber.as_ref(), &self.terms, &combined, b0);
            }
        }
        if s.want_groupvar {
            res.groupvar = Some(match &dof.groupvar {
                Some(gv) => gv.iter().map(|&v| f64::from(v)).collect(),
                // reghdfe creates no group variable without a pair of intercept terms.
                None => vec![f64::NAN; n],
            });
        }
        Ok(res)
    }
}
