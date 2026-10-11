//! IV/2SLS with ivreghdfe 1.1.4 conventions (plan Phase 5).
//!
//! ivreghdfe partials the fixed effects out of [y, endog, exog, inst] and then runs ivreg2 on the
//! residualized data with `small` and `noconstant` forced and `sdofminus = max(df_a, 1)`
//! (ivreghdfe.ado l.258-269, 663-712). Everything below mirrors ivreg2's Mata code
//! (`s_crossprods`, `s_gmm1s`, `s_iegmm`, `s_ffirst`, ivreghdfe.ado l.5188-6496), livreg2's
//! `m_omega` (variance of the moment conditions) and ranktest 2.0.04 (`s_jstat`, `m_svd`).
//!
//! Notation: X = [endog, exog] (coefficient order of e(b)), Z = [inst, exog], N = Σfw or the
//! number of rows, w_v = regression weights (w·N_obs/Σw for aw/pw, fw, or 1); every
//! cross-product is weighted by w_v (ivreg2's `wf*wvar`).
//!
//! Weighting of the moment-condition covariance (`m_omega`, livreg2 1.2.01 l.240-546, checked
//! against the ivreghdfe fixtures): robust meat Σ m_i z_i z_i' with m_i = e² (none), e²·w (fw),
//! (e·w_v)² (aw/pw); cluster scores Σ_{i∈g} e_i·w_v,i·z_i. These are exactly reghdfe's
//! [`Weights::robust_mult`] and cluster score, so no variant was needed.

mod dense;
mod dist;
mod first;
mod rank;
mod stock_yogo;

use rayon::prelude::*;

use crate::absorber::SolveOpts;
use crate::api::{Ctx, EstResult, Layout, NamedMat, Spec, Vce};
use crate::dof;
use crate::error::RfError;
use crate::fe_recovery;
use crate::ols::{STATUS_ABSORBED, STATUS_COLLINEAR, Weights, collinear_tol, effective_vce, missing_if_nonfinite, stage1_absorbed, stage2_collinear};
use crate::par::{self, CHUNK, Pool};
use crate::session::Session;
use crate::stats;
use crate::vcov::{self, ClusterCodes};

use dense::M;

/// Variance flavour of the moment conditions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Mode {
    Iid,
    Robust,
    /// Cluster-robust with G_min clusters (one or two dimensions).
    Cluster { g: f64 },
}

/// Streaming passes over the residualized rows.
pub(crate) struct Passes<'a> {
    rows: &'a [f64],
    r: usize,
    w: &'a Weights,
    pool: &'a Pool<'a>,
    mode: Mode,
    dims: Vec<ClusterCodes<'a>>,
    row_order: &'a [u32],
}

impl Passes<'_> {
    fn n(&self) -> usize {
        self.rows.len() / self.r
    }

    /// Linear combinations of row i: out[c] = Σ_j row_j·coef[j, c].
    #[inline]
    fn combine(row: &[f64], coef: &M, out: &mut [f64]) {
        out.iter_mut().for_each(|v| *v = 0.0);
        for (j, &x) in row.iter().enumerate() {
            if x == 0.0 {
                continue;
            }
            let cr = &coef.d[j * coef.c..(j + 1) * coef.c];
            for (o, c) in out.iter_mut().zip(cr) {
                *o += x * c;
            }
        }
    }

    /// Σ_i w_v,i·e_i e_i' (m×m) with e_i = coef'·row_i (`quadcross(e, wf*wvar, e)`).
    pub(crate) fn cross(&self, coef: &M) -> M {
        let m = coef.c;
        let (rows, r, w) = (self.rows, self.r, self.w);
        let mut acc = par::sum_chunks(self.pool, self.n(), CHUNK, m * m, |range, acc| {
            let mut e = vec![0.0; m];
            for i in range {
                Self::combine(&rows[i * r..(i + 1) * r], coef, &mut e);
                let wi = w.wv(i);
                for a in 0..m {
                    let wa = wi * e[a];
                    for b in a..m {
                        acc[a * m + b] += wa * e[b];
                    }
                }
            }
        });
        for a in 0..m {
            for b in 0..a {
                acc[a * m + b] = acc[b * m + a];
            }
        }
        M { r: m, c: m, d: acc }
    }

    /// Σ_i w_v,i·e_i (m) with e_i = coef'·row_i.
    fn wsum(&self, coef: &M) -> Vec<f64> {
        let m = coef.c;
        let (rows, r, w) = (self.rows, self.r, self.w);
        par::sum_chunks(self.pool, self.n(), CHUNK, m, |range, acc| {
            let mut e = vec![0.0; m];
            for i in range {
                Self::combine(&rows[i * r..(i + 1) * r], coef, &mut e);
                let wi = w.wv(i);
                for (a, v) in acc.iter_mut().zip(&e) {
                    *a += wi * v;
                }
            }
        })
    }

    /// Un-normalized robust or cluster meat of the moments e ⊗ z (`m_omega` × N), (m·l)², with
    /// e_i = ce'·row_i (m residuals) and z_i = cz'·row_i (l instruments); index c·l + k.
    pub(crate) fn meat(&self, ce: &M, cz: &M) -> Result<M, RfError> {
        let (m, l) = (ce.c, cz.c);
        let dim = m * l;
        let (rows, r, w) = (self.rows, self.r, self.w);
        let fill = |i: usize, a: &mut [f64]| {
            let row = &rows[i * r..(i + 1) * r];
            let mut stack = [0.0f64; 128];
            let mut heap;
            let buf: &mut [f64] = if m + l <= stack.len() {
                &mut stack[..m + l]
            } else {
                heap = vec![0.0; m + l];
                &mut heap
            };
            let (e, z) = buf.split_at_mut(m);
            Self::combine(row, ce, e);
            Self::combine(row, cz, z);
            for c in 0..m {
                for k in 0..l {
                    a[c * l + k] = e[c] * z[k];
                }
            }
        };
        let d = match self.mode {
            Mode::Iid => return Err(RfError::Internal("iv: meat requested for an iid VCE".into())),
            Mode::Robust => vcov::robust_meat(self.n(), dim, self.pool, |i, a| {
                fill(i, a);
                w.robust_mult(i, 1.0)
            }),
            Mode::Cluster { .. } => vcov::multiway_meat(&self.dims, self.row_order, dim, self.pool, |i, a| {
                fill(i, a);
                w.wv(i)
            })?,
        };
        Ok(M { r: dim, c: dim, d })
    }
}

/// Coefficients (over the R row columns) of `target − Σ_k b_k·col_k`, as an R×1 matrix.
pub(crate) fn resid_coef(r: usize, target: usize, cols: &[usize], b: &[f64]) -> M {
    let mut c = M::zeros(r, 1);
    c.d[target] += 1.0;
    for (&j, &bj) in cols.iter().zip(b) {
        c.d[j] -= bj;
    }
    c
}

/// R×l selector of the row columns `cols`.
pub(crate) fn selector(r: usize, cols: &[usize]) -> M {
    let mut s = M::zeros(r, cols.len());
    for (k, &j) in cols.iter().enumerate() {
        s.set(j, k, 1.0);
    }
    s
}

/// Scatters a square matrix over kept items into an n×n row-major matrix at positions `pos`
/// (zeros elsewhere, as ivreg2's `AddOmitted`).
fn scatter_sym(m: &M, pos: &[usize], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for (a, &pa) in pos.iter().enumerate() {
        for (b, &pb) in pos.iter().enumerate() {
            out[pa * n + pb] = m.at(a, b);
        }
    }
    out
}

/// Inputs shared by the diagnostics (ranktest, first stage).
pub(crate) struct Model<'a> {
    /// Weighted Gram of all residualized columns (R×R).
    pub g: &'a M,
    pub en: &'a [usize],
    pub ex: &'a [usize],
    pub iz: &'a [usize],
    pub n: f64,
    pub sdof: f64,
    pub mode: Mode,
}

impl Model<'_> {
    pub fn l(&self) -> usize {
        self.iz.len() + self.ex.len()
    }
    pub fn zcols(&self) -> Vec<usize> {
        self.iz.iter().chain(self.ex).copied().collect()
    }
    pub fn xcols(&self) -> Vec<usize> {
        self.en.iter().chain(self.ex).copied().collect()
    }
}

impl Session {
    /// 2SLS on `rows` (N×R row-major in [`Session::row_order`], layout `[y, exog..., endog...,
    /// inst...]`). e(b) is ordered endog then exog (ivreghdfe), without a constant;
    /// [`EstResult::status`] follows the layout order (exog, endog, inst).
    pub fn iv(&mut self, rows: Vec<f64>, lay: &Layout, s: &Spec, ctx: &Ctx) -> Result<EstResult, RfError> {
        if rows.len() != self.n() * lay.ncols() {
            return Err(RfError::InvalidInput("iv: data block size differs from N·R".into()));
        }
        if let Vce::Cluster(d) = &self.vce
            && d.len() > 2
        {
            return Err(RfError::Code(198, "regfast: IV supports at most two cluster variables".into()));
        }
        par::with_pool(ctx.threads, |pool| self.iv_in_pool(rows, lay, s, ctx, pool))?
    }

    fn iv_in_pool(&mut self, mut rows: Vec<f64>, lay: &Layout, s: &Spec, ctx: &Ctx, pool: &Pool) -> Result<EstResult, RfError> {
        let n = self.n();
        let r = lay.ncols();
        let (nx, ne, ni) = (lay.nx, lay.nendog, lay.ninst);
        let w = Weights::new(self.weight_kind, self.weights.as_deref(), self.absorber.row_order(), pool)?;
        let big_n = w.n;
        let dof = dof::estimate_dof(&self.terms, &self.clusters, n, &s.dof, s.want_groupvar);
        // ivreghdfe l.670: absorb_ct = df_a, or 1 when every FE is nested in a cluster.
        let sdof = dof.df_a.max(1) as f64;
        ctx.check()?;

        let cs = stats::col_stats(&rows, r, w.raw.as_deref(), pool);
        let want_coefs = s.want_d || (s.want_fe && self.terms.iter().any(|t| t.save));
        let opts = SolveOpts { tol: s.tol, maxiter: s.maxiter, want_coefs };
        let info = self.absorber.residualize(&mut rows, r, &opts, None, pool, ctx)?;
        ctx.check()?;
        let (graw, ss_res) = stats::gram(&rows, r, w.wv.as_deref(), pool);
        let gm = M { r, c: r, d: graw };

        // Collinearity: stage 1 (with the FEs) on every column, stage 2 on Z = [exog, inst] and
        // then on X = [exog, endog], so excluded instruments and endogenous regressors are the
        // ones dropped (ivreg2 drops the rightmost collinear variable).
        let exog = |j: usize| 1 + j;
        let endog = |j: usize| 1 + nx + j;
        let inst = |j: usize| 1 + nx + ne + j;
        let absorbed = stage1_absorbed(&ss_res, &cs.sumsq, collinear_tol(s.tol));
        let mut status = vec![0u8; r - 1];
        for c in 1..r {
            if absorbed[c] {
                status[c - 1] = STATUS_ABSORBED;
            }
        }
        let alive = |status: &[u8], c: usize| status[c - 1] == 0;
        let zc: Vec<usize> = (0..nx).map(exog).chain((0..ni).map(inst)).filter(|&c| alive(&status, c)).collect();
        let (dz, _) = stage2_collinear(&gm.d, r, &zc, &cs.sd);
        for (&c, &d) in zc.iter().zip(&dz) {
            if d {
                status[c - 1] = STATUS_COLLINEAR;
            }
        }
        let ex: Vec<usize> = (0..nx).map(exog).filter(|&c| alive(&status, c)).collect();
        let xc: Vec<usize> = ex.iter().copied().chain((0..ne).map(endog).filter(|&c| alive(&status, c))).collect();
        let (dx, _) = stage2_collinear(&gm.d, r, &xc, &cs.sd);
        for (&c, &d) in xc.iter().zip(&dx) {
            if d {
                status[c - 1] = STATUS_COLLINEAR;
            }
        }
        let ex: Vec<usize> = ex.into_iter().filter(|&c| alive(&status, c)).collect();
        let en: Vec<usize> = (0..ne).map(endog).filter(|&c| alive(&status, c)).collect();
        let iz: Vec<usize> = (0..ni).map(inst).filter(|&c| alive(&status, c)).collect();
        let (k1, k2, l1) = (en.len(), ex.len(), iz.len());
        if l1 < k1 {
            return Err(RfError::Code(
                481,
                "equation not identified; must have at least as many instruments not in the regression as there are instrumented variables"
                    .into(),
            ));
        }
        let xcols: Vec<usize> = en.iter().chain(&ex).copied().collect();
        let zcols: Vec<usize> = iz.iter().chain(&ex).copied().collect();
        let (k, l) = (k1 + k2, l1 + k2);

        let vce = effective_vce(&self.vce, w.kind);
        let mode = match vce {
            Vce::Unadjusted => Mode::Iid,
            Vce::Robust => Mode::Robust,
            Vce::Cluster(_) => Mode::Cluster { g: self.clusters.iter().map(|c| c.g).min().unwrap_or(0) as f64 },
        };
        let passes = Passes {
            rows: &rows,
            r,
            w: &w,
            pool,
            mode,
            dims: self.clusters.iter().map(|c| ClusterCodes { codes: &c.codes, g: c.g }).collect(),
            row_order: self.absorber.row_order(),
        };

        // Cross-products (s_crossprods) and 2SLS (s_gmm1s).
        let xx = gm.pick(&xcols, &xcols);
        let xz = gm.pick(&xcols, &zcols);
        let zz = gm.pick(&zcols, &zcols);
        let zy = gm.pick(&zcols, &[0]);
        let zzinv = dense::spd_inv(&zz);
        let xz_zzinv = xz.mul(&zzinv);
        let mut xpzx = xz_zzinv.mul(&xz.t());
        xpzx.symmetrize();
        let xpzx_inv = dense::spd_inv(&xpzx);
        let xxinv = dense::spd_inv(&xx);
        let beta = xpzx_inv.mul(&xz_zzinv.mul(&zy)).d;
        let ce = resid_coef(r, 0, &xcols, &beta);
        let ee = passes.cross(&ce).d[0];
        let rss = ee;
        let ysum = passes.wsum(&selector(r, &[0]))[0];
        let sz = selector(r, &zcols);
        ctx.check()?;

        let nk = big_n - k as f64 - sdof;
        let sigmasq = rss / big_n;
        let (v, omega, rank_s, rank_v) = match mode {
            Mode::Iid => {
                let v = xpzx_inv.scale(rss / nk);
                let omega = zz.scale(sigmasq / big_n);
                let rank_v = v.diag().iter().filter(|d| **d != 0.0).count();
                (v, omega, dense::ginv(&zz).1, rank_v)
            }
            Mode::Robust | Mode::Cluster { .. } => {
                let s_raw = passes.meat(&ce, &sz)?;
                let bmat = xpzx_inv.mul(&xz_zzinv);
                let mut v = bmat.mul(&s_raw).mul(&bmat.t());
                v.symmetrize();
                let adj = match mode {
                    Mode::Cluster { g } => (big_n - 1.0) / nk * g / (g - 1.0),
                    _ => big_n / nk,
                };
                let v = v.scale(adj);
                let omega = s_raw.scale(1.0 / big_n);
                let rank_s = dense::ginv(&omega).1;
                let rank_v = dense::ginv(&v).1;
                (v, omega, rank_s, rank_v)
            }
        };
        let wmat = zzinv.scale(big_n * big_n / rss);

        // Sargan (iid) / Hansen J from two-step efficient GMM residuals (s_gmm1s, s_iegmm).
        let jdf = l as f64 - k as f64;
        let zx = xz.t();
        let mut j = if l > k {
            match mode {
                Mode::Iid => {
                    let ze = zy.sub(&zx.mul(&M::col(&beta)));
                    big_n * zzinv.quad(&ze.d) / rss
                }
                _ => {
                    let oinv = dense::spd_inv(&omega);
                    let a = xz.mul(&oinv);
                    let mut aux3 = a.mul(&zx);
                    aux3.symmetrize();
                    let b2s = dense::spd_solve(&aux3, &a.mul(&zy));
                    let ze2 = zy.sub(&zx.mul(&b2s));
                    oinv.quad(&ze2.d) / big_n
                }
            }
        } else {
            0.0
        };
        if mode != Mode::Iid && rank_s < l {
            j = f64::NAN;
        }

        // Wald F on all coefficients (ivreg2 `test` on the posted V; `small` → F = χ²/K).
        let df_r = match mode {
            Mode::Cluster { g } => g - 1.0,
            _ => nk,
        };
        let f_stat = if k > 0 {
            let (vinv, _) = dense::ginv(&v);
            vinv.quad(&beta) / k as f64
        } else {
            f64::NAN
        };

        // Fit statistics (ivreghdfe l.1242-1305 with `noconstant`).
        let yy = gm.at(0, 0);
        let ym = ysum / big_n;
        let yyc = yy - big_n * ym * ym;
        let r2u = 1.0 - rss / yy;
        let r2c = 1.0 - rss / yyc;
        let ll = -0.5 * (big_n * (2.0 * std::f64::consts::PI).ln() + big_n * (rss / big_n).ln() + big_n);
        ctx.check()?;

        let model = Model { g: &gm, en: &en, ex: &ex, iz: &iz, n: big_n, sdof, mode };
        let mut res = EstResult { status, ..EstResult::default() };
        let scal = |res: &mut EstResult, name: &str, v: f64| res.set_scalar(name, missing_if_nonfinite(v));
        scal(&mut res, "N", big_n);
        scal(&mut res, "df_r", df_r);
        scal(&mut res, "df_m", k as f64);
        scal(&mut res, "sdofminus", sdof);
        scal(&mut res, "dofminus", 0.0);
        scal(&mut res, "center", 0.0);
        scal(&mut res, "partial_ct", sdof);
        scal(&mut res, "partialcons", sdof);
        scal(&mut res, "cons", 0.0);
        scal(&mut res, "nocollin", 0.0);
        scal(&mut res, "inexog_ct", k2 as f64);
        scal(&mut res, "exexog_ct", l1 as f64);
        scal(&mut res, "endog_ct", k1 as f64);
        scal(&mut res, "r2", r2u);
        scal(&mut res, "r2u", r2u);
        scal(&mut res, "r2c", r2c);
        scal(&mut res, "r2_a", 1.0 - (1.0 - r2u) * big_n / nk);
        scal(&mut res, "rmse", (rss / nk).sqrt());
        scal(&mut res, "rss", rss);
        scal(&mut res, "mss", yy - rss);
        scal(&mut res, "yy", yy);
        scal(&mut res, "yyc", yyc);
        scal(&mut res, "ll", ll);
        scal(&mut res, "F", f_stat);
        scal(&mut res, "Fp", dist::ftail(k as f64, df_r, f_stat));
        scal(&mut res, "Fdf1", k as f64);
        scal(&mut res, "Fdf2", df_r);
        scal(&mut res, "condxx", dense::cond_sym(&xx));
        scal(&mut res, "condzz", dense::cond_sym(&zz));
        scal(&mut res, "rankxx", k as f64);
        scal(&mut res, "rankzz", l as f64);
        scal(&mut res, "rankS", rank_s as f64);
        scal(&mut res, "rankV", rank_v as f64);
        scal(&mut res, "j", j);
        scal(&mut res, "jdf", jdf);
        if j != 0.0 && j.is_finite() {
            scal(&mut res, "jp", dist::chi2tail(jdf, j));
        }
        if mode == Mode::Iid {
            scal(&mut res, "sargan", j);
            scal(&mut res, "sargandf", jdf);
            if j != 0.0 && j.is_finite() {
                scal(&mut res, "sarganp", dist::chi2tail(jdf, j));
            }
        }
        if let Mode::Cluster { g } = mode {
            scal(&mut res, "N_clust", g);
            if self.clusters.len() == 2 {
                scal(&mut res, "N_clust1", self.clusters[0].g as f64);
                scal(&mut res, "N_clust2", self.clusters[1].g as f64);
            }
        }
        scal(&mut res, "df_a", dof.df_a as f64);
        scal(&mut res, "df_a_initial", dof.df_a_initial as f64);
        scal(&mut res, "df_a_redundant", dof.df_a_redundant as f64);
        scal(&mut res, "df_a_nested", dof.df_a_nested as f64);
        scal(&mut res, "num_singletons", self.num_singletons as f64);
        scal(&mut res, "N_full", big_n + self.num_singletons as f64);
        // Solver iterations, for the ado's "converged in # iterations" line (not an e() scalar).
        scal(&mut res, "ic", info.iterations as f64);

        // Identification statistics (ranktest) and Stock–Yogo critical values.
        if k1 > 0 {
            let id = rank::id_stats(&model, &passes)?;
            scal(&mut res, "idstat", id.idstat);
            scal(&mut res, "iddf", id.iddf);
            scal(&mut res, "idp", dist::chi2tail(id.iddf, id.idstat));
            scal(&mut res, "cd", id.cd);
            scal(&mut res, "cdf", id.cdf);
            scal(&mut res, "widstat", id.widstat);
            if let Some(rkf) = id.rkf {
                scal(&mut res, "rkf", rkf);
            }
            // Padded to one entry per endogenous regressor sent (missing beyond K1).
            let (ccev, cdev) = id.ev.unwrap_or_default();
            let pad = |v: Vec<f64>| (0..ne).map(|i| v.get(i).copied().unwrap_or(f64::NAN)).collect::<Vec<_>>();
            res.matrices.push(NamedMat { name: "ccev".into(), rows: 1, cols: ne, data: pad(ccev) });
            res.matrices.push(NamedMat { name: "cdev".into(), rows: 1, cols: ne, data: pad(cdev) });
            // Stock–Yogo critical values for (L1, K1) as `sy_<table>` and for a single
            // endogenous regressor (the `ffirst` summary) as `sy1_<table>`; display only.
            for (name, _, table) in stock_yogo::IV_TABLES {
                for (prefix, k) in [("sy", k1), ("sy1", 1)] {
                    let cv = table(l1, k);
                    if cv.is_finite() {
                        res.macros.push((format!("{prefix}_{name}"), format!("{cv}")));
                    }
                }
            }
        }
        ctx.check()?;

        // First-stage statistics and regressions, in sent coordinates: e(first) is 21×nendog with
        // missing columns for dropped endogenous regressors; bfirst#/Vfirst#/sfirst# exist for
        // the kept ones only (# = position among the endogenous regressors sent), over Z sent.
        let pos_x: Vec<usize> = xcols.iter().map(|&c| if c > nx { c - 1 - nx } else { ne + c - 1 }).collect();
        let pos_z: Vec<usize> = zcols.iter().map(|&c| if c > nx { c - 1 - nx - ne } else { ni + c - 1 }).collect();
        if k1 > 0 && (s.iv.first || s.iv.ffirst) {
            let fs = first::first_stage(&model, &passes, &xxinv, &xpzx_inv)?;
            let mut fm = vec![f64::NAN; 21 * ne];
            for (i, &pe) in pos_x[..k1].iter().enumerate() {
                for row in 0..21 {
                    fm[row * ne + pe] = fs.firstmat.at(row, i);
                }
            }
            res.matrices.push(NamedMat { name: "first".into(), rows: 21, cols: ne, data: fm });
            for (name, v) in [
                ("arf", fs.arf),
                ("arfp", dist::ftail(fs.ardf, fs.ardf_r, fs.arf)),
                ("archi2", fs.archi2),
                ("archi2p", dist::chi2tail(fs.ardf, fs.archi2)),
                ("ardf", fs.ardf),
                ("ardf_r", fs.ardf_r),
                ("sstat", fs.sstat),
                ("sstatp", dist::chi2tail(fs.sstatdf, fs.sstat)),
                ("sstatdf", fs.sstatdf),
            ] {
                scal(&mut res, name, v);
            }
            if s.iv.first {
                let lz = ni + nx;
                for (eq, &pe) in fs.eqs.iter().zip(&pos_x[..k1]) {
                    let mut b = vec![0.0; lz];
                    for (a, &p) in pos_z.iter().enumerate() {
                        b[p] = eq.b[a];
                    }
                    res.matrices.push(NamedMat { name: format!("bfirst{}", pe + 1), rows: 1, cols: lz, data: b });
                    res.matrices.push(NamedMat { name: format!("Vfirst{}", pe + 1), rows: lz, cols: lz, data: scatter_sym(&eq.v, &pos_z, lz) });
                    res.matrices.push(NamedMat { name: format!("sfirst{}", pe + 1), rows: 1, cols: 4, data: eq.stats.to_vec() });
                }
            }
        }

        // e(b), e(V) over every regressor sent (endog then exog); omitted entries are 0.
        let kall = ne + nx;
        let mut b_out = vec![0.0; kall];
        for (a, &p) in pos_x.iter().enumerate() {
            b_out[p] = beta[a];
        }
        res.matrices.push(NamedMat { name: "b".into(), rows: 1, cols: kall, data: b_out });
        res.matrices.push(NamedMat { name: "V".into(), rows: kall, cols: kall, data: scatter_sym(&v, &pos_x, kall) });
        // e(S), e(W) over every instrument sent [inst, exog]; omitted entries are 0.
        let lz = ni + nx;
        res.matrices.push(NamedMat { name: "S".into(), rows: lz, cols: lz, data: scatter_sym(&omega, &pos_z, lz) });
        res.matrices.push(NamedMat { name: "W".into(), rows: lz, cols: lz, data: scatter_sym(&wmat, &pos_z, lz) });
        res.matrices.push(NamedMat { name: "dof_table".into(), rows: dof.slots.len(), cols: 5, data: dof.table() });

        // Per-observation outputs, kept order.
        let row_order = self.absorber.row_order();
        if s.want_resid {
            let mut e = par::alloc_vec(n, 0.0)?;
            pool.install(|| {
                e.par_iter_mut().enumerate().for_each(|(i, v)| {
                    let row = &rows[i * r..(i + 1) * r];
                    *v = row.iter().zip(&ce.d).map(|(x, c)| x * c).sum();
                })
            });
            res.resid = Some(fe_recovery::to_kept_order(&e, row_order, pool)?);
        }
        if want_coefs {
            let coefs = info.coefs.as_ref().ok_or_else(|| RfError::Internal("absorber returned no FE coefficients".into()))?;
            let combined = coefs.combine(&ce.d);
            if s.want_d {
                res.d = Some(fe_recovery::d_values(self.absorber.as_ref(), &combined, 0.0, pool)?);
            }
            if s.want_fe {
                res.fe = fe_recovery::saved_fe(self.absorber.as_ref(), &self.terms, &combined, 0.0);
            }
        }
        if s.want_groupvar {
            res.groupvar = Some(match &dof.groupvar {
                Some(gv) => gv.iter().map(|&v| f64::from(v)).collect(),
                None => vec![f64::NAN; n],
            });
        }
        Ok(res)
    }
}
