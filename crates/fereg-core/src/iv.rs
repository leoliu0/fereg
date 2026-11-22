//! IV/2SLS with ivreghdfe 1.1.4 conventions (plan Phase 5).
//!
//! ivreghdfe partials the fixed effects out of [y, endog, exog, inst] and then runs ivreg2 on the
//! residualized data with `small` and `noconstant` forced and `sdofminus = max(df_a, 1)`
//! (ivreghdfe.ado l.258-269, 663-712). Everything below mirrors ivreg2's Mata code
//! (`s_crossprods`, `s_gmm1s`, `s_iegmm`, `s_ffirst`, ivreghdfe.ado l.5188-6496), livreg2's
//! `m_omega` (variance of the moment conditions) and ranktest 2.0.04 (`s_jstat`, `m_svd`).
//!
//! Notation: X = [endog, exog] (coefficient order of e(b)), Z = [inst, exog], N = sum fw or the
//! number of rows, w_v = regression weights (w*N_obs/sum w for aw/pw, fw, or 1); every
//! cross-product is weighted by w_v (ivreg2's `wf*wvar`).
//!
//! Weighting of the moment-condition covariance (`m_omega`, livreg2 1.2.01 l.240-546, checked
//! against the ivreghdfe fixtures): robust meat sum m_i z_i z_i' with m_i = e^2 (none), e^2*w (fw),
//! (e*w_v)^2 (aw/pw); cluster scores sum_{i in g} e_i*w_v,i*z_i. These are exactly reghdfe's
//! [`Weights::robust_mult`] and cluster score, so no variant was needed.

mod dense;
mod dist;
mod first;
mod rank;
mod stock_yogo;

use rayon::prelude::*;

use crate::absorber::SolveOpts;
use crate::api::{Bandwidth, ConleyVce, Ctx, EstResult, KernelVce, Layout, NamedMat, Psd, Spec, Vce};
use crate::crv3::{JkModel, JkProblem};
use crate::hac::{self, Kernel};
use crate::leverage::FeCols;
use crate::ols::stata_vce::{self, KernelPlan};
use crate::dof;
use crate::error::RfError;
use crate::fe_recovery;
use crate::ols::{STATUS_ABSORBED, STATUS_COLLINEAR, Weights, collinear_tol, effective_vce, kept_columns, missing_if_nonfinite, stage1_absorbed, stage2_collinear};
use crate::par::{self, CHUNK, Pool};
use crate::session::{Session, VceData};
use crate::stats;
use crate::vcov::{self, ClusterCodes};

use dense::M;

/// Variance flavour of the moment conditions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Mode {
    Iid,
    /// Heteroskedasticity-robust (also HAC and Conley).
    Robust,
    /// Cluster-robust with G_min clusters (one or two dimensions; also kernel + cluster).
    Cluster { g: f64 },
    /// ivreg2 AC: kernel without robust (homoskedastic autocorrelation; `kiefer`).
    Ac,
}

/// Source of the robust/cluster meat beyond reghdfe's (m_omega's kernel branches, Conley).
pub(crate) enum Extra<'a> {
    Plain,
    Kernel(&'a KernelPlan),
    Conley { cv: &'a ConleyVce, vdata: &'a VceData, cutoff: std::cell::Cell<f64> },
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
    extra: Extra<'a>,
    /// ivreg2's N (sum fw or the number of rows).
    big_n: f64,
}

/// Scratch length for the per-row linear combinations of [`Passes::meat`] kept on the stack.
const STACK_COMB: usize = 128;

/// sum_j a_j*b_j in index order, accumulated in a register.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

impl Passes<'_> {
    fn n(&self) -> usize {
        self.rows.len() / self.r
    }

    /// Linear combinations of row i: out[c] = sum_j row_j*coef[j, c], with `coef_t` the c x R
    /// transpose of coef (each combination is one register dot product).
    #[inline]
    fn combine(row: &[f64], coef_t: &[f64], out: &mut [f64]) {
        for (o, col) in out.iter_mut().zip(coef_t.chunks_exact(row.len())) {
            *o = dot(row, col);
        }
    }

    /// sum_i w_v,i*e_i e_i' (m x m) with e_i = coef'*row_i (`quadcross(e, wf*wvar, e)`).
    pub(crate) fn cross(&self, coef: &M) -> M {
        let m = coef.c;
        let (rows, r, w) = (self.rows, self.r, self.w);
        let coef_t = coef.t().d;
        let mut acc = par::sum_chunks(self.pool, self.n(), CHUNK, m * m, |range, acc| {
            let mut e = vec![0.0; m];
            for i in range {
                Self::combine(&rows[i * r..(i + 1) * r], &coef_t, &mut e);
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

    /// sum_i w_v,i*e_i (m) with e_i = coef'*row_i.
    fn wsum(&self, coef: &M) -> Vec<f64> {
        let m = coef.c;
        let (rows, r, w) = (self.rows, self.r, self.w);
        let coef_t = coef.t().d;
        par::sum_chunks(self.pool, self.n(), CHUNK, m, |range, acc| {
            let mut e = vec![0.0; m];
            for i in range {
                Self::combine(&rows[i * r..(i + 1) * r], &coef_t, &mut e);
                let wi = w.wv(i);
                for (a, v) in acc.iter_mut().zip(&e) {
                    *a += wi * v;
                }
            }
        })
    }

    /// Un-normalized robust or cluster meat of the moments e (x) z (`m_omega` x N), (m*l)^2, with
    /// e_i = ce'*row_i (m residuals) and z_i = cz'*row_i (l instruments); index c*l + k.
    pub(crate) fn meat(&self, ce: &M, cz: &M) -> Result<M, RfError> {
        self.meat_for(ce, cz, false)
    }

    /// [`Passes::meat`] as ivreghdfe's `ranktest` calls compute it: with a kernel VCE they pass
    /// `bw()` but no `kernel()` or `psd0`/`psda` (ivreghdfe.ado 1741-1752: `kernopt` is never
    /// set), so the kernel is Bartlett at the same bandwidth and no PSD repair applies. Under
    /// `kiefer` not even `bw()` is passed (bw is still 0 when `bwopt` is built, ivreghdfe.ado
    /// 3920-3933): ranktest gets only `robust` (pweights) and the cluster options, so the meat
    /// is the plain robust or cluster one.
    pub(crate) fn meat_rank(&self, ce: &M, cz: &M) -> Result<M, RfError> {
        self.meat_for(ce, cz, true)
    }

    /// `kiefer`: ivreghdfe's ranktest calls get no `bw()`.
    fn ranktest_kiefer(&self) -> bool {
        matches!(&self.extra, Extra::Kernel(p) if p.kiefer)
    }

    /// ivreghdfe's ranktest calls run homoskedastic (`kiefer` without robust or cluster).
    pub(crate) fn ranktest_iid(&self) -> bool {
        self.ranktest_kiefer() && self.mode == Mode::Ac
    }

    fn meat_for(&self, ce: &M, cz: &M, ranktest: bool) -> Result<M, RfError> {
        let (m, l) = (ce.c, cz.c);
        let dim = m * l;
        let (rows, r, w) = (self.rows, self.r, self.w);
        // [ce cz] transposed: the m residual combinations, then the l instrument combinations.
        let mut ez_t = ce.t().d;
        ez_t.extend_from_slice(&cz.t().d);
        let ez_t = &ez_t;
        let fill = |i: usize, a: &mut [f64]| {
            let row = &rows[i * r..(i + 1) * r];
            let mut stack = [std::mem::MaybeUninit::<f64>::uninit(); STACK_COMB];
            let heap;
            let ez: &[f64] = if m + l <= STACK_COMB {
                for (s, col) in stack.iter_mut().zip(ez_t.chunks_exact(r)) {
                    s.write(dot(row, col));
                }
                // SAFETY: the loop above initialized the first m + l entries.
                unsafe { std::slice::from_raw_parts(stack.as_ptr().cast::<f64>(), m + l) }
            } else {
                heap = ez_t.chunks_exact(r).map(|col| dot(row, col)).collect::<Vec<f64>>();
                &heap
            };
            let (e, z) = ez.split_at(m);
            for c in 0..m {
                for k in 0..l {
                    a[c * l + k] = e[c] * z[k];
                }
            }
        };
        let extra = if ranktest && self.ranktest_kiefer() { &Extra::Plain } else { &self.extra };
        let d = match (extra, self.mode) {
            (_, Mode::Iid) => return Err(RfError::Internal("iv: meat requested for an iid VCE".into())),
            (Extra::Kernel(plan), _) => {
                let mut s = plan.meat(m, l, self.big_n, ranktest, self.pool, |i, buf| {
                    let row = &rows[i * r..(i + 1) * r];
                    for (b, col) in buf.iter_mut().zip(ez_t.chunks_exact(r)) {
                        *b = dot(row, col);
                    }
                    w.wv(i)
                })?;
                if !ranktest {
                    stata_vce::apply_psd(&mut s, dim, plan.psd);
                }
                s
            }
            (Extra::Conley { cv, vdata, cutoff }, _) => {
                let (s, c) = stata_vce::conley_meat(cv, vdata, self.row_order, dim, self.pool, |i, a| {
                    fill(i, a);
                    w.wv(i)
                })?;
                cutoff.set(c);
                s
            }
            (Extra::Plain, Mode::Robust) => vcov::robust_meat(self.n(), dim, self.pool, |i, a| {
                fill(i, a);
                w.robust_mult(i, 1.0)
            }),
            (Extra::Plain, Mode::Cluster { .. }) => vcov::multiway_meat(&self.dims, self.row_order, dim, self.pool, |i, a| {
                fill(i, a);
                w.wv(i)
            })?,
            (Extra::Plain, Mode::Ac) => return Err(RfError::Internal("iv: AC without a kernel".into())),
        };
        Ok(M { r: dim, c: dim, d })
    }
}

/// Coefficients (over the R row columns) of `target - sum_k b_k*col_k`, as an Rx1 matrix.
pub(crate) fn resid_coef(r: usize, target: usize, cols: &[usize], b: &[f64]) -> M {
    let mut c = M::zeros(r, 1);
    c.d[target] += 1.0;
    for (&j, &bj) in cols.iter().zip(b) {
        c.d[j] -= bj;
    }
    c
}

/// R x l selector of the row columns `cols`.
pub(crate) fn selector(r: usize, cols: &[usize]) -> M {
    let mut s = M::zeros(r, cols.len());
    for (k, &j) in cols.iter().enumerate() {
        s.set(j, k, 1.0);
    }
    s
}

/// Scatters a square matrix over kept items into an n x n row-major matrix at positions `pos`
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
    /// Weighted Gram of all residualized columns (R x R).
    pub g: &'a M,
    pub en: &'a [usize],
    pub ex: &'a [usize],
    pub iz: &'a [usize],
    pub n: f64,
    pub sdof: f64,
    pub dofm: f64,
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
    /// 2SLS on `rows` (N x R row-major in [`Session::row_order`], layout `[y, exog..., endog...,
    /// inst...]`). e(b) is ordered endog then exog (ivreghdfe), without a constant;
    /// [`EstResult::status`] follows the layout order (exog, endog, inst).
    pub fn iv(&mut self, rows: Vec<f64>, lay: &Layout, s: &Spec, ctx: &Ctx) -> Result<EstResult, RfError> {
        if rows.len() != self.n() * lay.ncols() {
            return Err(RfError::InvalidInput("iv: data block size differs from N*R".into()));
        }
        match &self.vce {
            Vce::Cluster(d) | Vce::Crv3(d) if d.len() > 2 => {
                return Err(RfError::Code(198, "fereg: IV supports at most two cluster variables".into()));
            }
            Vce::Hc2 | Vce::Hc3 => return Err(RfError::Code(198, "vce() only supports cluster, robust, and unadjusted".into())),
            _ => {}
        }
        par::with_pool(par::threads_for(ctx.threads, self.n()), |pool| self.iv_in_pool(rows, lay, s, ctx, pool))?
    }

    fn iv_in_pool(&mut self, rows_in: Vec<f64>, lay: &Layout, s: &Spec, ctx: &Ctx, pool: &Pool) -> Result<EstResult, RfError> {
        let n = self.n();
        let r_in = lay.ncols();
        let (nx_in, ne, ni) = (lay.nx, lay.nendog, lay.ninst);
        let no_absorb_const = self.terms.is_empty() && !s.noconstant && s.iv.report_constant;
        let no_absorb_nocons = self.terms.is_empty() && s.noconstant;
        let skip_absorber = no_absorb_const || no_absorb_nocons;
        let cons_col = no_absorb_const.then_some(1 + nx_in);
        let nx = nx_in + usize::from(no_absorb_const);
        let r = r_in + usize::from(no_absorb_const);
        let mut rows = if let Some(cc) = cons_col {
            let mut ext = par::alloc_uninit::<f64>(n * r)?;
            for (i, dst) in ext.chunks_exact_mut(r).enumerate() {
                let src = &rows_in[i * r_in..(i + 1) * r_in];
                for j in 0..cc {
                    dst[j].write(src[j]);
                }
                dst[cc].write(1.0);
                for j in cc..r_in {
                    dst[j + 1].write(src[j]);
                }
            }
            // SAFETY: all `n * r` slots written.
            unsafe { par::assume_init(ext) }
        } else {
            rows_in
        };

        let w = Weights::new(self.weight_kind, self.weights.as_deref(), self.absorber.row_order(), pool)?;
        let big_n = w.n;
        let dof = dof::estimate_dof(&self.terms, &self.clusters, n, &s.dof, s.want_groupvar, false, pool);
        let sdof = if !self.terms.is_empty() || !skip_absorber {
            s.iv.sdofminus.unwrap_or_else(|| dof.df_a.max(1) as f64)
        } else {
            s.iv.sdofminus.unwrap_or(0.0)
        };
        let dofm = s.iv.dofminus;
        let vce = effective_vce(&self.vce, w.kind);
        // `vce(dkraay [#])` on IV is ivreghdfe's dkraay(#): Bartlett kernel, cluster on time.
        let dk_kernel = match &vce {
            Vce::Dkraay { time, bw } => Some(KernelVce {
                kernel: Kernel::Bartlett,
                bw: Bandwidth::Fixed(bw.unwrap_or_else(|| hac::bw_reghdfe_dkraay(self.clusters[0].g)) as f64),
                robust: true,
                cluster: vec![vec![time.time]],
                kiefer: false,
                psd: Psd::None,
                time: time.clone(),
            }),
            _ => None,
        };
        let mut kplan = match (&vce, &dk_kernel) {
            (Vce::Kernel(kv), _) | (_, Some(kv)) => Some(KernelPlan::new(kv, &self.vdata, w.kind, self.absorber.row_order(), pool)?),
            _ => None,
        };
        let raw_cols = if matches!(vce, Vce::Crv3(_)) { Some(kept_columns(&rows, r, self.absorber.row_order())?) } else { None };
        ctx.check()?;
        let cs = stats::col_stats(&rows, r, w.raw.as_deref(), pool);
        let want_coefs = !self.terms.is_empty() && (s.want_d || (s.want_fe && self.terms.iter().any(|t| t.save)));
        let info = if skip_absorber {
            crate::absorber::SolveInfo { iterations: 0, escalated: false, exact: true, coefs: None }
        } else {
            let opts = SolveOpts { tol: s.tol, maxiter: s.maxiter, want_coefs };
            self.absorber.residualize(&mut rows, r, &opts, None, pool, ctx)?
        };
        ctx.check()?;
        let (graw, ss_res) = stats::gram(&rows, r, w.wv.as_deref(), pool);
        let gm = M { r, c: r, d: graw };
        let gm_c_opt = if no_absorb_const {
            let cc = cons_col.expect("no_absorb_const sets cons_col");
            let wmeans = &cs.wmean;
            let mut gc = par::sum_chunks(pool, n, CHUNK, r * r, |range, acc| {
                let mut d = vec![0.0; r];
                for i in range {
                    let row = &rows[i * r..(i + 1) * r];
                    for j in 0..r {
                        d[j] = if j == cc { 0.0 } else { row[j] - wmeans[j] };
                    }
                    let wi = w.wv(i);
                    for a in 0..r {
                        let wa = wi * d[a];
                        for b in a..r {
                            acc[a * r + b] += wa * d[b];
                        }
                    }
                }
            });
            for a in 0..r {
                for b in 0..a {
                    gc[a * r + b] = gc[b * r + a];
                }
            }
            gc[cc * r + cc] = w.n_const();
            Some(M { r, c: r, d: gc })
        } else {
            None
        };

        // Collinearity: stage 1 (with the FEs) on every column, stage 2 on Z = [cons?, exog, inst]
        // and then on X = [cons?, exog, endog], so `_cons` stays if collinear with indicators, and
        // excluded instruments / endogenous regressors are dropped rightmost.
        let exog = |j: usize| 1 + j;
        let endog = |j: usize| 1 + nx + j;
        let inst = |j: usize| 1 + nx + ne + j;
        let mut col_status = vec![0u8; r - 1];
        if !skip_absorber {
            let absorbed = stage1_absorbed(&ss_res, &cs.sumsq, collinear_tol(s.tol));
            for c in 1..r {
                if absorbed[c] {
                    col_status[c - 1] = STATUS_ABSORBED;
                }
            }
        }
        let alive = |st: &[u8], c: usize| st[c - 1] == 0;
        let zc: Vec<usize> = cons_col
            .into_iter()
            .chain((0..nx_in).map(exog))
            .chain((0..ni).map(inst))
            .filter(|&c| alive(&col_status, c))
            .collect();
        let gm_collin = gm_c_opt.as_ref().unwrap_or(&gm);
        let (dz, _) = stage2_collinear(&gm_collin.d, r, &zc, &cs.sd);
        for (&c, &d) in zc.iter().zip(&dz) {
            if d {
                col_status[c - 1] = STATUS_COLLINEAR;
            }
        }
        let ex: Vec<usize> = (0..nx).map(exog).filter(|&c| alive(&col_status, c)).collect();
        let xc: Vec<usize> = cons_col
            .into_iter()
            .filter(|&c| alive(&col_status, c))
            .chain(ex.iter().copied().filter(|&c| Some(c) != cons_col))
            .chain((0..ne).map(endog).filter(|&c| alive(&col_status, c)))
            .collect();
        let (dx, _) = stage2_collinear(&gm_collin.d, r, &xc, &cs.sd);
        for (&c, &d) in xc.iter().zip(&dx) {
            if d {
                col_status[c - 1] = STATUS_COLLINEAR;
            }
        }
        let ex: Vec<usize> = ex.into_iter().filter(|&c| alive(&col_status, c)).collect();
        let en: Vec<usize> = (0..ne).map(endog).filter(|&c| alive(&col_status, c)).collect();
        let iz: Vec<usize> = (0..ni).map(inst).filter(|&c| alive(&col_status, c)).collect();
        let cons_kept = cons_col.is_some_and(|c| alive(&col_status, c));
        let status: Vec<u8> = match cons_col {
            Some(cc) => col_status.iter().enumerate().filter(|&(idx, _)| idx != cc - 1).map(|(_, &s)| s).collect(),
            None => col_status,
        };
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

        let g_min = self.clusters.iter().map(|c| c.g).min().unwrap_or(0) as f64;
        let mode = match (&vce, &kplan) {
            (_, Some(p)) => match p.n_clust() {
                Some(g) => Mode::Cluster { g },
                None if p.mode == stata_vce::KernelMode::Ac => Mode::Ac,
                None => Mode::Robust,
            },
            (Vce::Unadjusted, _) => Mode::Iid,
            (Vce::Cluster(_) | Vce::Crv3(_), _) => Mode::Cluster { g: g_min },
            _ => Mode::Robust,
        };
        let dims: Vec<ClusterCodes> = self.clusters.iter().map(|c| ClusterCodes { codes: &c.codes, g: c.g }).collect();
        let base = Passes { rows: &rows, r, w: &w, pool, mode, dims, row_order: self.absorber.row_order(), extra: Extra::Plain, big_n };

        // Cross-products (s_crossprods) and 2SLS (s_gmm1s).
        let xx = gm.pick(&xcols, &xcols);
        let xz = gm.pick(&xcols, &zcols);
        let zz = gm.pick(&zcols, &zcols);
        let zy = gm.pick(&zcols, &[0]);
        let (zzinv, xpzx_inv, xxinv, beta, bmat) = if let (Some(gm_c), Some(cc)) = (&gm_c_opt, cons_col)
            && cons_kept
        {
            debug_assert_eq!(xcols.last(), Some(&cc));
            debug_assert_eq!(zcols.last(), Some(&cc));
            let k_nc = k - 1;
            let l_nc = l - 1;
            let xcols_nc = &xcols[..k_nc];
            let zcols_nc = &zcols[..l_nc];
            let n_c = w.n_const();
            let xbar_x: Vec<f64> = xcols_nc.iter().map(|&c| cs.wmean[c]).collect();
            let zbar_z: Vec<f64> = zcols_nc.iter().map(|&c| cs.wmean[c]).collect();
            let zzinv_nc = if l_nc > 0 { dense::spd_inv(&gm_c.pick(zcols_nc, zcols_nc)) } else { M::zeros(0, 0) };
            let xxinv_nc = if k_nc > 0 { dense::spd_inv(&gm_c.pick(xcols_nc, xcols_nc)) } else { M::zeros(0, 0) };
            let (xpzx_inv_nc, beta_nc, bmat_nc) = if k_nc > 0 {
                let xz_nc = gm_c.pick(xcols_nc, zcols_nc);
                let zy_nc = gm_c.pick(zcols_nc, &[0]);
                let xz_zzinv_nc = xz_nc.mul(&zzinv_nc);
                let mut xpzx_nc = xz_zzinv_nc.mul(&xz_nc.t());
                xpzx_nc.symmetrize();
                let xpzx_inv_nc = dense::spd_inv(&xpzx_nc);
                let bmat_nc = xpzx_inv_nc.mul(&xz_zzinv_nc);
                let beta_nc = bmat_nc.mul(&zy_nc).d;
                (xpzx_inv_nc, beta_nc, bmat_nc)
            } else {
                (M::zeros(0, 0), Vec::new(), M::zeros(0, l_nc))
            };
            let b0 = cs.wmean[0] - xbar_x.iter().zip(&beta_nc).map(|(m, b)| m * b).sum::<f64>();
            let mut beta = Vec::with_capacity(k);
            beta.extend_from_slice(&beta_nc);
            beta.push(b0);
            let zzinv = M { r: l, c: l, d: vcov::inv_ext(&zzinv_nc.d, l_nc, &zbar_z, n_c) };
            let xxinv = M { r: k, c: k, d: vcov::inv_ext(&xxinv_nc.d, k_nc, &xbar_x, n_c) };
            let xpzx_inv = M { r: k, c: k, d: vcov::inv_ext(&xpzx_inv_nc.d, k_nc, &xbar_x, n_c) };
            // B_raw = T_X^{-1} * diag(bmat_nc, 1/n_c) * T_Z'^{-1}.
            let bz: Vec<f64> = (0..k_nc).map(|a| (0..l_nc).map(|j| bmat_nc.at(a, j) * zbar_z[j]).sum()).collect();
            let mut bmat = M::zeros(k, l);
            for a in 0..k_nc {
                for j in 0..l_nc {
                    bmat.set(a, j, bmat_nc.at(a, j));
                }
                bmat.set(a, l_nc, -bz[a]);
            }
            for j in 0..l_nc {
                let row_c = -(0..k_nc).map(|a| xbar_x[a] * bmat_nc.at(a, j)).sum::<f64>();
                bmat.set(k_nc, j, row_c);
            }
            let corner = 1.0 / n_c + (0..k_nc).map(|a| xbar_x[a] * bz[a]).sum::<f64>();
            bmat.set(k_nc, l_nc, corner);
            (zzinv, xpzx_inv, xxinv, beta, bmat)
        } else {
            let zzinv = dense::spd_inv(&zz);
            let xz_zzinv = xz.mul(&zzinv);
            let mut xpzx = xz_zzinv.mul(&xz.t());
            xpzx.symmetrize();
            let xpzx_inv = dense::spd_inv(&xpzx);
            let xxinv = dense::spd_inv(&xx);
            let bmat = xpzx_inv.mul(&xz_zzinv);
            let beta = bmat.mul(&zy).d;
            (zzinv, xpzx_inv, xxinv, beta, bmat)
        };
        let ce = resid_coef(r, 0, &xcols, &beta);
        let ee = base.cross(&ce).d[0];
        let rss = ee;
        let ysum = base.wsum(&selector(r, &[0]))[0];
        let sz = selector(r, &zcols);
        ctx.check()?;

        // ivreghdfe `abw` on the 2SLS residuals and the row sums of Z (no weights).
        if let Some(p) = kplan.as_mut().filter(|p| p.auto_pending) {
            let f: Vec<f64> = (0..n)
                .map(|i| {
                    let row = &rows[i * r..(i + 1) * r];
                    dot(row, &ce.d) * zcols.iter().map(|&c| row[c]).sum::<f64>()
                })
                .collect();
            p.resolve_auto(&f, big_n, pool);
        }
        let extra = match (&vce, &kplan) {
            (_, Some(p)) => Extra::Kernel(p),
            (Vce::Conley(cv), _) => Extra::Conley { cv, vdata: &self.vdata, cutoff: std::cell::Cell::new(f64::NAN) },
            _ => Extra::Plain,
        };
        let passes = Passes { extra, ..base };

        let nk = big_n - k as f64 - dofm - sdof;
        let sigmasq = rss / (big_n - dofm);
        let (mut v, mut v_small, omega, rank_s, mut rank_v) = match mode {
            Mode::Iid => {
                let v_sm = xpzx_inv.scale(rss / nk);
                let v = if s.iv.small { v_sm.clone() } else { xpzx_inv.scale(sigmasq) };
                let omega = zz.scale(sigmasq / big_n);
                let rank_v = v.diag().iter().filter(|d| **d != 0.0).count();
                (v, v_sm, omega, dense::ginv(&zz).1, rank_v)
            }
            Mode::Robust | Mode::Cluster { .. } | Mode::Ac => {
                let s_raw = passes.meat(&ce, &sz)?;
                let mut v_unadj = bmat.mul(&s_raw).mul(&bmat.t());
                v_unadj.symmetrize();
                let small_adj = match mode {
                    Mode::Cluster { g } => (big_n - 1.0) / (big_n - k as f64 - sdof) * g / (g - 1.0),
                    _ => big_n / nk,
                };
                let nosmall_adj = match mode {
                    Mode::Cluster { .. } => 1.0,
                    _ => big_n / (big_n - dofm),
                };
                let v_sm = v_unadj.scale(small_adj);
                let v = if s.iv.small { v_sm.clone() } else { v_unadj.scale(nosmall_adj) };
                let omega = match mode {
                    Mode::Cluster { .. } => s_raw.scale(1.0 / big_n),
                    _ => s_raw.scale(1.0 / (big_n - dofm)),
                };
                let rank_s = dense::ginv(&omega).1;
                let rank_v = dense::ginv(&v).1;
                (v, v_sm, omega, rank_s, rank_v)
            }
        };
        // CRV3: the jackknife replaces V (S and the diagnostics stay cluster-robust).
        if let Some(cols) = &raw_cols {
            let col = |c: usize| cols[c].as_slice();
            let fe: Vec<FeCols> =
                self.terms.iter().map(|t| FeCols { codes: &t.codes, g: t.g, intercept: t.intercept, slopes: t.slopes.iter().map(Vec::as_slice).collect() }).collect();
            let icpt = self.terms.is_empty() && (!no_absorb_nocons);
            let ex_nocl: Vec<usize> = ex.iter().copied().filter(|&c| Some(c) != cons_col).collect();
            // beta is ordered [endog, exog, _cons?]; the refits use [constant?, exog, endog].
            let b_of = |c: usize| beta[xcols.iter().position(|&x| x == c).expect("regressor")];
            let b0 = cs.wmean[0] - xcols.iter().zip(&beta).filter(|&(&c, _)| Some(c) != cons_col).map(|(&c, b)| cs.wmean[c] * b).sum::<f64>();
            let jk_beta: Vec<f64> = icpt.then_some(b0).into_iter().chain(ex_nocl.iter().map(|&c| b_of(c))).chain(en.iter().map(|&c| b_of(c))).collect();
            let p = JkProblem {
                model: JkModel::Linear,
                n,
                y: col(0),
                x: ex_nocl.iter().map(|&c| col(c)).collect(),
                endog: en.iter().map(|&c| col(c)).collect(),
                inst: iz.iter().map(|&c| col(c)).collect(),
                intercept: icpt,
                weights: self.weights.as_deref(),
                offset: None,
                fe,
                beta: &jk_beta,
                fixef_tol: s.tol,
                fixef_maxiter: s.maxiter,
                glm_tol: 1e-10,
                glm_maxiter: 10_000,
                collin_tol: crate::fit::FitSpec::default().collin_tol,
            };
            let skip = usize::from(icpt);
            let k2_nc = ex_nocl.len();
            let augment = |_: &[u32], _: usize, betas: Vec<f64>| -> Vec<f64> {
                betas.chunks_exact(skip + k1 + k2_nc).flat_map(|b| {
                    let mut out: Vec<f64> = b[skip + k2_nc..].iter().chain(&b[skip..skip + k2_nc]).copied().collect();
                    if cons_kept {
                        out.push(b[0]);
                    }
                    out
                }).collect()
            };
            let mut b_crv3: Vec<f64> = (0..k1 + k2_nc).map(|i| beta[i]).collect();
            if cons_kept {
                b_crv3.push(b_of(cons_col.expect("cons")));
            }
            let meat = stata_vce::crv3_meat(&p, &self.clusters, &b_crv3, &augment, ctx, pool)?;
            let q = if s.iv.small { (big_n - 1.0) / (big_n - k as f64 - sdof) * g_min / (g_min - 1.0) } else { 1.0 };
            v = M { r: k, c: k, d: meat.iter().map(|x| q * x).collect() };
            v_small = M { r: k, c: k, d: meat.iter().map(|x| ((big_n - 1.0) / (big_n - k as f64 - sdof) * g_min / (g_min - 1.0)) * x).collect() };
            rank_v = dense::ginv(&v).1;
        }
        // Stata posts V only when it is exactly symmetric.
        v.symmetrize();
        v_small.symmetrize();
        let wmat = zzinv.scale(big_n * (big_n - dofm) / rss);

        // Sargan (iid) / Hansen J from two-step efficient GMM residuals (s_gmm1s, s_iegmm).
        let jdf = l as f64 - k as f64;
        let zx = xz.t();
        let mut j = if l > k {
            match mode {
                Mode::Iid => {
                    let ze = zy.sub(&zx.mul(&M::col(&beta)));
                    (big_n - dofm) * zzinv.quad(&ze.d) / rss
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

        // Wald F on all non-constant coefficients (`test`: small -> F = chi^2/fdf1 using v_small).
        let cons_f = if cons_kept { 1.0 } else { 0.0 };
        let k_nc = k - usize::from(cons_kept);
        let fdf1 = k_nc as f64;
        let fdf2 = match mode {
            Mode::Cluster { g } => g - 1.0,
            _ => nk,
        };
        let df_r = if s.iv.small { fdf2 } else { f64::NAN };
        let f_stat = if k_nc > 0 {
            let non_cons: Vec<usize> = (0..k).filter(|&i| Some(xcols[i]) != cons_col).collect();
            let v_nc = v_small.pick(&non_cons, &non_cons);
            let b_nc: Vec<f64> = non_cons.iter().map(|&i| beta[i]).collect();
            let (vinv, _) = dense::ginv(&v_nc);
            vinv.quad(&b_nc) / fdf1
        } else {
            f64::NAN
        };

        // Fit statistics.
        let yy = gm.at(0, 0);
        let ym = ysum / big_n;
        let yyc = yy - big_n * ym * ym;
        let r2u = 1.0 - rss / yy;
        let r2c = 1.0 - rss / yyc;
        let use_centered_r2 = self.terms.is_empty() && !s.noconstant;
        let r2_main = if use_centered_r2 { r2c } else { r2u };
        let mss = if use_centered_r2 { yyc - rss } else { yy - rss };
        let r2_a = 1.0 - (1.0 - r2_main) * (big_n - if use_centered_r2 { 1.0 } else { 0.0 }) / nk;
        let rmse = if s.iv.small { (rss / nk).sqrt() } else { sigmasq.sqrt() };
        let df_m = if skip_absorber { fdf1 + sdof } else { k as f64 };
        let partial_ct = if skip_absorber { sdof } else { sdof };
        let ll = -0.5 * (big_n * (2.0 * std::f64::consts::PI).ln() + big_n * (rss / big_n).ln() + big_n);
        ctx.check()?;

        let model = Model { g: &gm, en: &en, ex: &ex, iz: &iz, n: big_n, sdof, dofm, mode };
        let mut res = EstResult { status, ..EstResult::default() };
        let scal = |res: &mut EstResult, name: &str, v: f64| res.set_scalar(name, missing_if_nonfinite(v));
        scal(&mut res, "N", big_n);
        scal(&mut res, "df_r", df_r);
        scal(&mut res, "df_m", df_m);
        scal(&mut res, "sdofminus", sdof);
        scal(&mut res, "dofminus", dofm);
        scal(&mut res, "center", 0.0);
        scal(&mut res, "partial_ct", partial_ct);
        scal(&mut res, "partialcons", if !self.terms.is_empty() || !skip_absorber { sdof } else { 0.0 });
        scal(&mut res, "cons", cons_f);
        scal(&mut res, "nocollin", 0.0);
        scal(&mut res, "inexog_ct", k2 as f64);
        scal(&mut res, "exexog_ct", l1 as f64);
        scal(&mut res, "endog_ct", k1 as f64);
        scal(&mut res, "r2", r2_main);
        scal(&mut res, "r2u", r2u);
        scal(&mut res, "r2c", r2c);
        scal(&mut res, "r2_a", r2_a);
        scal(&mut res, "rmse", rmse);
        scal(&mut res, "rss", rss);
        scal(&mut res, "mss", mss);
        scal(&mut res, "yy", yy);
        scal(&mut res, "yyc", yyc);
        scal(&mut res, "ll", ll);
        scal(&mut res, "F", f_stat);
        scal(&mut res, "Fp", dist::ftail(fdf1, fdf2, f_stat));
        scal(&mut res, "Fdf1", fdf1);
        scal(&mut res, "Fdf2", fdf2);
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
        // ivreghdfe.ado 2120: e(sargan*) whenever neither robust nor cluster was requested (AC
        // kernels included), e(j*) always.
        if matches!(mode, Mode::Iid | Mode::Ac) {
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
        if let Some(p) = &kplan {
            scal(&mut res, "bw", p.bw);
            scal(&mut res, "hac_T", p.span);
            if matches!(vce, Vce::Dkraay { .. }) {
                scal(&mut res, "dkraay_bw", p.bw);
            }
            res.macros.push(("kernel".into(), p.kernel.name().into()));
            res.macros.push(("bwchoice".into(), p.bwchoice.clone()));
            res.notes.extend(p.notes.iter().cloned());
        }
        if let Extra::Conley { cutoff, .. } = &passes.extra {
            scal(&mut res, "cutoff", cutoff.get());
        }
        scal(&mut res, "df_a", dof.df_a as f64);
        scal(&mut res, "df_a_initial", dof.df_a_initial as f64);
        scal(&mut res, "df_a_redundant", dof.df_a_redundant as f64);
        scal(&mut res, "df_a_nested", dof.df_a_nested as f64);
        scal(&mut res, "num_singletons", self.num_singletons as f64);
        scal(&mut res, "N_full", big_n + self.num_singletons as f64);
        // Solver iterations, for the ado's "converged in # iterations" line (not an e() scalar).
        scal(&mut res, "ic", info.iterations as f64);

        // Identification statistics (ranktest) and Stock-Yogo critical values.
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
            // Stock-Yogo critical values for (L1, K1) as `sy_<table>` and for a single
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

        // First-stage statistics and regressions, in sent coordinates: e(first) is 21xnendog with
        // missing columns for dropped endogenous regressors; bfirst#/Vfirst#/sfirst# exist for
        // the kept ones only (# = position among the endogenous regressors sent), over Z sent.
        let pos_x: Vec<usize> = xcols.iter().map(|&c| {
            if c > nx {
                c - 1 - nx
            } else {
                ne + c - 1
            }
        }).collect();
        let pos_z: Vec<usize> = zcols.iter().map(|&c| {
            if c > nx {
                c - 1 - nx - ne
            } else {
                ni + c - 1
            }
        }).collect();
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

        // e(b), e(V) over every regressor sent (endog then exog, plus `_cons` when `no_absorb_const`); omitted entries are 0.
        let kall = ne + nx;
        let mut b_out = vec![0.0; kall];
        for (a, &p) in pos_x.iter().enumerate() {
            b_out[p] = beta[a];
        }
        res.matrices.push(NamedMat { name: "b".into(), rows: 1, cols: kall, data: b_out });
        res.matrices.push(NamedMat { name: "V".into(), rows: kall, cols: kall, data: scatter_sym(&v, &pos_x, kall) });
        res.matrices.push(NamedMat { name: "bread".into(), rows: kall, cols: kall, data: scatter_sym(&xpzx_inv, &pos_x, kall) });
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
