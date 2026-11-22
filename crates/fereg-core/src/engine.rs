//! Engine entry points for native statistical commands built over `fereg-core`.
//!
//! Exposes lower-level linear, IV/2SLS, and canonical GLM estimation kernels with configurable
//! collinearity policies (`Natural`, `Greedy`, `KeepAll`), Stata `invsym` tolerances, unscaled
//! sandwich/bread variance matrices (so each caller applies its exact finite-sample or large-sample
//! degrees-of-freedom factor), true `noabsorb` + `noconstant` paths without absorber demeaning, and
//! time-series / panel HAC covariance estimators (Newey-West and Driscoll-Kraay).

use rayon::prelude::*;

use crate::absorber::SolveOpts;
use crate::api::{Ctx, DofOptions, EstResult, Family, GlmCollinear, GlmConvention, GlmOptions, Guess, IvOptions, Layout, Separation, Spec, WeightKind};
use crate::dof;
use crate::error::RfError;
use crate::fe_recovery;
use crate::glm::GlmEngineExtra;
use crate::hac;
use crate::ols::{STATUS_ABSORBED, STATUS_COLLINEAR, STATUS_OK, Weights, stage1_absorbed, sub_block};
use crate::ols::stata_vce;
use crate::par::{self, CHUNK, Pool};
use crate::session::Session;
use crate::stats;
use crate::vcov::{self, ClusterCodes, INVSYM_TOL};

/// Collinearity sweep strategy for [`ols_fit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollinearMode {
    /// Natural left-to-right column sweep (drop later collinear columns).
    Natural,
    /// Natural left-to-right column sweep with uncentred diagonal reference (`_rmcoll` / `newey`).
    NaturalUncentred,
    /// Greedy largest-remaining-diagonal pivot sweep (Stata `regress` without FV operators).
    Greedy,
    /// Do not drop columns during the sweep (invert kept block directly with `INVSYM_TOL`).
    KeepAll,
}

/// Variance-covariance estimator requested from [`ols_fit`] or [`glm_fit`].
#[derive(Clone, Debug)]
pub enum EngineVce {
    Ols,
    Robust,
    Hc2,
    Hc3,
    /// Cluster-robust (1-way or multiway CGM).
    /// - `per_subset_g_adj = false`: no subset factor inside the meat (`reghdfe`/`xtreg`/`areg` apply
    ///   `G_min/(G_min-1)` outside).
    /// - `per_subset_g_adj = true`: Stata `regress` / CGM applies `G_S/(G_S-1)` per subset `S` inside
    ///   the inclusion-exclusion sum; `subset_counts` is populated on [`OlsEngineOut`].
    Cluster { per_subset_g_adj: bool },
    /// Newey-West HAC (`newey`): calendar lag pairs `t_a - t_b == j * delta` within each panel,
    /// Bartlett kernel weights `1 - j / (lag + 1)`.
    NeweyWest {
        lag: usize,
        /// Time values in `Session` kept order.
        time: Vec<f64>,
        /// Panel time step (`tsdelta`, typically `1.0`).
        delta: f64,
        /// Optional `(panel_codes, n_panels)` in `Session` kept order.
        panel: Option<(Vec<u32>, usize)>,
    },
    /// Driscoll-Kraay HAC (`xtscc`, Hoechle 2007): sum scores within each period, then Bartlett
    /// kernel across dense period lags `0..=lag`.
    DriscollKraay {
        lag: usize,
        /// Dense `0..n_periods` period code in `Session` kept order.
        time_codes: Vec<u32>,
        n_periods: usize,
    },
}

/// Options for [`ols_fit`].
#[derive(Clone, Debug)]
pub struct OlsEngineOpts {
    pub vce: EngineVce,
    pub collinear: CollinearMode,
    /// Relative pivot tolerance (`pivot <= tol * orig_diag`). Defaults to `1e-9` (Stata `regress`).
    pub collin_tol: f64,
    /// Include an intercept when `Session::terms` is empty. Ignored (always `false` for within
    /// demeaning) when FEs are absorbed by `Session::terms`.
    pub has_constant: bool,
    /// When `Session::terms` is non-empty, compute grand-mean intercept `b0` and include the
    /// `_cons` row/col in `bread` and `v_unscaled` (as in Stata `areg` and `xtreg, fe`).
    pub report_constant: bool,
    /// When `terms` is non-empty, check collinearity on raw `[1?, X]` before absorbing FEs so
    /// callers can distinguish raw-collinear columns from FE-absorbed ones.
    pub check_raw_collinear: bool,
    /// FE solver tolerance (`reghdfe`/`areg`).
    pub tol: f64,
    /// FE solver max iterations.
    pub maxiter: usize,
    /// Degrees-of-freedom options used when `terms` is non-empty.
    pub dof: DofOptions,
    pub want_resid: bool,
    pub want_d: bool,
    pub want_fe: bool,
    pub want_groupvar: bool,
}

impl Default for OlsEngineOpts {
    fn default() -> Self {
        Self {
            vce: EngineVce::Ols,
            collinear: CollinearMode::Natural,
            collin_tol: 1e-9,
            has_constant: true,
            report_constant: true,
            check_raw_collinear: false,
            tol: 1e-8,
            maxiter: 16_000,
            dof: DofOptions::default(),
            want_resid: false,
            want_d: false,
            want_fe: false,
            want_groupvar: false,
        }
    }
}

/// Output of [`ols_fit`], in `kept` order (unscaled by finite-sample DoF multipliers except
/// `per_subset_g_adj` when explicitly requested for `Cluster`).
#[derive(Clone, Debug)]
pub struct OlsEngineOut {
    /// Slope coefficients (length `nx`, `0.0` for omitted/absorbed columns).
    pub beta: Vec<f64>,
    /// Intercept (`0.0` when `!has_constant` or FEs absorbed; for absorbed FEs, callers compute
    /// Stata-style constant from grand weighted means if needed).
    pub b0: f64,
    /// Per-regressor status in `0..nx`: `0` ok, `2` absorbed by FEs, `3` collinear.
    pub status: Vec<u8>,
    /// Subset of collinear columns omitted during raw pre-check (`check_raw_collinear`).
    pub raw_omitted: Vec<bool>,
    /// Kept regressor indices (`0..nx`).
    pub kept: Vec<usize>,
    /// Number of kept non-constant regressors (`kept.len()`).
    pub rank: usize,
    /// Effective observation count (`sum fw` for fweights, `n_obs` otherwise).
    pub n: f64,
    /// Sum of raw weights (`n_obs` when unweighted).
    pub sumw: f64,
    pub df_a: f64,
    pub df_a_initial: f64,
    pub df_a_redundant: f64,
    pub df_a_nested: f64,
    pub rss: f64,
    pub tss: f64,
    pub tss_within: f64,
    /// Bread `(X' W X)^{-1}` of size `msize x msize`, ordered `[kept..., _cons?]` on the raw scale,
    /// where `msize = kept.len() + usize::from(has_constant && terms.is_empty())`.
    pub bread: Vec<f64>,
    /// Number of columns in `bread` / `v_unscaled` / `meat_raw` (`msize`).
    pub msize: usize,
    /// True if `bread` / `v_unscaled` include `_cons` as their last row/col.
    pub bread_has_cons: bool,
    /// Unscaled covariance matrix `bread * meat_raw * bread` (size `msize x msize`).
    /// For `EngineVce::Ols`, equals `bread` (NOT multiplied by `rss / df_r`).
    pub v_unscaled: Vec<f64>,
    /// Raw meat matrix (size `msize x msize`), when computed (`None` for `EngineVce::Ols`).
    pub meat_raw: Option<Vec<f64>>,
    pub n_clust: Option<f64>,
    pub clust_counts: Vec<f64>,
    /// For `Cluster { per_subset_g_adj: true }`: `(1-based dim indices, n_groups)` per subset.
    pub subset_counts: Vec<(Vec<usize>, usize)>,
    /// Minimum pair count across lags `0..=lag` for `NeweyWest` (equals `n_obs` otherwise).
    pub seen_obs: usize,
    pub ic: usize,
    pub resid: Option<Vec<f64>>,
    pub d: Option<Vec<f64>>,
    pub fe: Vec<Vec<f64>>,
    pub dof_table: Vec<f64>,
    pub y_absorbed: bool,
    /// Weighted means of `[y, x...]` (length `1 + nx`, in original column coordinates).
    pub wmean: Vec<f64>,
    /// Grand weighted sum of `(y - X_kept b_kept - b0)^2` already equal to `rss`.
    pub df_dof_rows: usize,
}

/// Collinearity sweep + Cholesky inverse on an `m x m` symmetric PSD matrix `a` (row-major).
/// Returns `(dropped, inv)` where `inv` is `m x m` with zeroed rows/cols for dropped entries.
/// `base`, when provided, replaces the diagonal of `a` as the denominator for `tol * orig[j]`.
fn sweep_collinear(a: &[f64], m: usize, mode: CollinearMode, tol: f64, base: Option<&[f64]>) -> (Vec<bool>, Vec<f64>) {
    if m == 0 {
        return (Vec::new(), Vec::new());
    }
    match mode {
        CollinearMode::KeepAll => {
            let (inv, dropped) = vcov::invsym(a, m, INVSYM_TOL);
            (dropped, inv)
        }
        CollinearMode::Natural => {
            // Standardize by diagonal sqrt before invsym so scale differences cannot distort
            // pivots, then rescale back. Note ratio d_new / orig is scale-invariant whenever
            // orig > 0.
            let sd: Vec<f64> = (0..m).map(|j| {
                let d = a[j * m + j];
                if d > 0.0 && d.is_finite() { d.sqrt() } else { 1.0 }
            }).collect();
            let gs: Vec<f64> = (0..m * m).map(|ab| a[ab] / (sd[ab / m] * sd[ab % m])).collect();
            let (inv_s, dropped) = vcov::invsym(&gs, m, tol);
            let mut inv = vec![0.0; m * m];
            for i in 0..m {
                if dropped[i] {
                    continue;
                }
                for j in 0..m {
                    if !dropped[j] {
                        inv[i * m + j] = inv_s[i * m + j] / (sd[i] * sd[j]);
                    }
                }
            }
            (dropped, inv)
        }
        CollinearMode::Greedy | CollinearMode::NaturalUncentred => {
            // Pivoted Cholesky matching Stata `regress` / `_rmcoll`: standardize/compare against
            // `base` (uncentred diag) when provided, else `a[j,j]`.
            let natural = mode == CollinearMode::NaturalUncentred;
            let orig: Vec<f64> = base.map_or_else(|| (0..m).map(|j| a[j * m + j]).collect(), <[f64]>::to_vec);
            let mut mat = a.to_vec();
            let mut active = vec![true; m];
            let mut order = Vec::new();
            let mut lpiv = Vec::new();
            let mut lcols: Vec<Vec<(usize, f64)>> = Vec::new();
            let tie_rel = 1e-10;
            loop {
                let mut best: Option<usize> = None;
                for j in 0..m {
                    if !active[j] {
                        continue;
                    }
                    let rem = mat[j * m + j];
                    if orig[j] <= 0.0 || !rem.is_finite() || rem <= tol * orig[j] {
                        active[j] = false;
                        continue;
                    }
                    match best {
                        Some(b) if natural || mat[b * m + b] * (1.0 + tie_rel) >= rem => {}
                        _ => best = Some(j),
                    }
                }
                let Some(p) = best else { break };
                let lpp = mat[p * m + p].sqrt();
                active[p] = false;
                let mut col = Vec::new();
                for j in 0..m {
                    if active[j] {
                        let v = if j < p { mat[p * m + j] } else { mat[j * m + p] };
                        col.push((j, v / lpp));
                    }
                }
                for (xi, &(j, lj)) in col.iter().enumerate() {
                    let row = &mut mat[j * m..(j + 1) * m];
                    for &(l, ll) in &col[..=xi] {
                        row[l] -= lj * ll;
                    }
                }
                order.push(p);
                lpiv.push(lpp);
                lcols.push(col);
            }
            let mut dropped = vec![true; m];
            for &p in &order {
                dropped[p] = false;
            }
            let r = order.len();
            if r == 0 {
                return (dropped, vec![0.0; m * m]);
            }
            // Equilibrate the kept submatrix by diagonal sqrt before Cholesky inversion so
            // multi-scale designs (e.g. year and year^2) retain full precision.
            let sd: Vec<f64> = order.iter().map(|&p| {
                let d = a[p * m + p];
                if d > 0.0 && d.is_finite() { d.sqrt() } else { 1.0 }
            }).collect();
            let mut l_eq = vec![0.0; r * r];
            let mut eq_ok = true;
            for j in 0..r {
                let pj = order[j];
                let mut d = a[pj * m + pj] / (sd[j] * sd[j]);
                for k in 0..j {
                    d -= l_eq[j * r + k] * l_eq[j * r + k];
                }
                if !d.is_finite() || d <= 0.0 {
                    eq_ok = false;
                    break;
                }
                let lj = d.sqrt();
                l_eq[j * r + j] = lj;
                for i in (j + 1)..r {
                    let pi = order[i];
                    let mut s = a[pi * m + pj] / (sd[i] * sd[j]);
                    for k in 0..j {
                        s -= l_eq[i * r + k] * l_eq[j * r + k];
                    }
                    l_eq[i * r + j] = s / lj;
                }
            }
            let lmat = if eq_ok {
                l_eq
            } else {
                // Fall back to the already-computed un-equilibrated Cholesky factor.
                let mut raw = vec![0.0; r * r];
                let mut pos = vec![usize::MAX; m];
                for (i, &c) in order.iter().enumerate() {
                    pos[c] = i;
                }
                for (b, col) in lcols.iter().enumerate() {
                    raw[b * r + b] = lpiv[b];
                    for &(j, v) in col {
                        let a_pos = pos[j];
                        if a_pos != usize::MAX {
                            raw[a_pos * r + b] = v;
                        }
                    }
                }
                raw
            };
            // Forward substitution for L^{-1} (lower triangular).
            let mut linv = vec![0.0; r * r];
            for j in 0..r {
                linv[j * r + j] = 1.0 / lmat[j * r + j];
                for i in (j + 1)..r {
                    let mut s = 0.0;
                    for k in j..i {
                        s += lmat[i * r + k] * linv[k * r + j];
                    }
                    linv[i * r + j] = -s / lmat[i * r + i];
                }
            }
            let mut inv = vec![0.0; m * m];
            for a_i in 0..r {
                let pa = order[a_i];
                for b_i in a_i..r {
                    let pb = order[b_i];
                    let mut s = 0.0;
                    for row in b_i..r {
                        s += linv[row * r + a_i] * linv[row * r + b_i];
                    }
                    let val = if eq_ok { s / (sd[a_i] * sd[b_i]) } else { s };
                    inv[pa * m + pb] = val;
                    inv[pb * m + pa] = val;
                }
            }
            (dropped, inv)
        }
    }
}

/// Extends an inverse block `inv` (`k x k`, centered around weighted means `xbar` with total weight `n_c`)
/// to `[kept..., _cons]`. Handles `k == 0` as `[1 / n_c]`.
fn extend_const_inv(inv: &[f64], k: usize, xbar: &[f64], n_c: f64) -> Vec<f64> {
    if k == 0 {
        return vec![if n_c > 0.0 { 1.0 / n_c } else { 0.0 }];
    }
    vcov::inv_ext(inv, k, xbar, n_c)
}

/// Core OLS / FE-OLS estimation engine over an initialized `Session`.
///
/// `rows` has size `sess.n() * (1 + nx)` in [`Session::row_order`] with layout `[y, x_0..x_{nx-1}]`.
/// When `sess.terms()` is empty, skips the iterative FE solver entirely and performs exact weighted
/// mean centering (if `opts.has_constant`) or uncentered cross-products (if `!opts.has_constant`).
pub fn ols_fit(sess: &mut Session, mut rows: Vec<f64>, nx: usize, opts: &OlsEngineOpts, ctx: &Ctx) -> Result<OlsEngineOut, RfError> {
    let n = sess.n();
    let r = 1 + nx;
    if rows.len() != n * r {
        return Err(RfError::InvalidInput("ols_fit: data block size differs from N * (1 + nx)".into()));
    }
    let pool_cell = par::with_pool(par::threads_for(ctx.threads, n), |pool| ols_fit_in_pool(sess, &mut rows, nx, opts, ctx, pool))?;
    pool_cell
}

fn ols_fit_in_pool(sess: &mut Session, rows: &mut [f64], nx: usize, opts: &OlsEngineOpts, ctx: &Ctx, pool: &Pool) -> Result<OlsEngineOut, RfError> {
    let n = sess.n();
    let r = 1 + nx;
    let has_terms = !sess.terms.is_empty();
    let has_const = !has_terms && opts.has_constant;
    let w = Weights::new(sess.weight_kind, sess.weights.as_deref(), sess.absorber.row_order(), pool)?;
    let big_n = w.n;

    let dof = if has_terms {
        dof::estimate_dof(&sess.terms, &sess.clusters, n, &opts.dof, opts.want_groupvar, false, pool)
    } else {
        dof::Dof {
            df_a: 0,
            df_a_initial: 0,
            df_a_redundant: 0,
            df_a_nested: 0,
            slots: Vec::new(),
            groupvar: None,
        }
    };

    ctx.check()?;
    let cs = stats::col_stats(rows, r, w.raw.as_deref(), pool);
    let tss_all = crate::ols::tss_of(&cs, &w, has_terms || opts.has_constant);

    // Optional pre-FE raw collinearity check so callers can distinguish raw collinearity from FE
    // absorption cleanly (`areg` / `xtreg, fe` notes).
    let mut raw_omitted = vec![false; nx];
    let mut status = vec![STATUS_OK; nx];
    if has_terms && opts.check_raw_collinear && nx > 0 {
        let mut raw_centered = rows.to_vec();
        let wmeans = &cs.wmean;
        if n <= CHUNK || pool.threads() <= 1 {
            for row in raw_centered.chunks_exact_mut(r) {
                for j in 0..r {
                    row[j] -= wmeans[j];
                }
            }
        } else {
            pool.install(|| {
                raw_centered.par_chunks_mut(CHUNK * r).for_each(|rc| {
                    for row in rc.chunks_exact_mut(r) {
                        for j in 0..r {
                            row[j] -= wmeans[j];
                        }
                    }
                })
            });
        }
        let g_raw = stats::gram(&raw_centered, r, w.wv.as_deref(), pool).0;
        let x_block: Vec<f64> = (0..nx * nx).map(|ab| g_raw[(1 + ab / nx) * r + (1 + ab % nx)]).collect();
        let (dropped_raw, _) = sweep_collinear(&x_block, nx, opts.collinear, opts.collin_tol, None);
        for j in 0..nx {
            if dropped_raw[j] {
                raw_omitted[j] = true;
                status[j] = STATUS_COLLINEAR;
            }
        }
    }

    // Residualize or center.
    let want_coefs = has_terms && (opts.want_d || (opts.want_fe && sess.terms.iter().any(|t| t.save)));
    let (ic, solve_coefs) = if has_terms {
        let sopts = SolveOpts { tol: opts.tol, maxiter: opts.maxiter, want_coefs };
        let info = sess.absorber.residualize(rows, r, &sopts, None, pool, ctx)?;
        (info.iterations, info.coefs)
    } else {
        if has_const {
            let wmeans = &cs.wmean;
            if n <= CHUNK || pool.threads() <= 1 {
                for row in rows.chunks_exact_mut(r) {
                    for j in 0..r {
                        row[j] -= wmeans[j];
                    }
                }
            } else {
                pool.install(|| {
                    rows.par_chunks_mut(CHUNK * r).for_each(|rc| {
                        for row in rc.chunks_exact_mut(r) {
                            for j in 0..r {
                                row[j] -= wmeans[j];
                            }
                        }
                    })
                });
            }
        }
        (0, None)
    };
    ctx.check()?;

    let (mut g, ss_res) = stats::gram(rows, r, w.wv.as_deref(), pool);

    // Stage 1 (with FEs): check if dependent variable or remaining regressors are absorbed by FEs.
    let mut y_absorbed = false;
    if has_terms {
        let ctol = crate::ols::collinear_tol(opts.tol);
        let absorbed = stage1_absorbed(&ss_res, &cs.sumsq, ctol);
        y_absorbed = nx >= 1 && absorbed[0];
        if y_absorbed {
            pool.install(|| rows.par_chunks_mut(r).for_each(|row| row[0] = 0.0));
            for j in 0..r {
                g[j] = 0.0;
                g[j * r] = 0.0;
            }
        }
        for j in 0..nx {
            if status[j] == STATUS_OK && absorbed[1 + j] {
                status[j] = STATUS_ABSORBED;
            }
        }
    }

    // Stage 2: collinearity sweep over surviving regressors on raw-centered (or uncentered) Gram.
    let cand: Vec<usize> = (0..nx).filter(|&j| status[j] == STATUS_OK).collect();
    let kc = cand.len();
    let g_cand: Vec<f64> = (0..kc * kc).map(|ab| g[(1 + cand[ab / kc]) * r + (1 + cand[ab % kc])]).collect();
    let uncentred: Option<Vec<f64>> = (opts.collinear == CollinearMode::NaturalUncentred).then(|| {
        cand.iter()
            .enumerate()
            .map(|(ci, &j)| {
                let s = g_cand[ci * kc + ci];
                if has_const { s + w.n_const() * cs.wmean[1 + j] * cs.wmean[1 + j] } else { s }
            })
            .collect()
    });
    let (dropped_c, inv_cand) = sweep_collinear(&g_cand, kc, opts.collinear, opts.collin_tol, uncentred.as_deref());
    for (ci, &j) in cand.iter().enumerate() {
        if dropped_c[ci] {
            status[j] = STATUS_COLLINEAR;
        }
    }
    let keep_pos: Vec<usize> = (0..kc).filter(|&ci| !dropped_c[ci]).collect();
    let kept: Vec<usize> = keep_pos.iter().map(|&ci| cand[ci]).collect();
    let k = kept.len();
    let inv_xx = sub_block(&inv_cand, kc, &keep_pos);

    // Solve beta on raw scale.
    let xy: Vec<f64> = kept.iter().map(|&j| g[(1 + j) * r]).collect();
    let beta_kept: Vec<f64> = (0..k).map(|a| (0..k).map(|b| inv_xx[a * k + b] * xy[b]).sum()).collect();
    let mut beta = vec![0.0; nx];
    for (a, &j) in kept.iter().enumerate() {
        beta[j] = beta_kept[a];
    }

    // Constant & extended bread `(X' W X)^{-1}` in `[kept..., _cons?]`.
    let xbar_kept: Vec<f64> = kept.iter().map(|&j| cs.wmean[1 + j]).collect();
    let bread_has_cons = has_const || (has_terms && opts.report_constant);
    let b0 = if bread_has_cons { cs.wmean[0] - xbar_kept.iter().zip(&beta_kept).map(|(m, b)| m * b).sum::<f64>() } else { 0.0 };
    let msize = k + usize::from(bread_has_cons);
    let bread = if bread_has_cons { extend_const_inv(&inv_xx, k, &xbar_kept, w.n_const()) } else { inv_xx.clone() };

    // Residuals e = y~ - X~ beta (in `row_order`), plus rss = sum w_v e^2.
    let need_e = opts.want_resid || !matches!(opts.vce, EngineVce::Ols);
    let resid_of = |i: usize| -> f64 {
        let row = &rows[i * r..(i + 1) * r];
        let mut e = row[0];
        for (a, &j) in kept.iter().enumerate() {
            e -= beta_kept[a] * row[1 + j];
        }
        e
    };
    let e_row: Option<Vec<f64>> = if need_e {
        let mut ev = par::alloc_uninit::<f64>(n)?;
        if n <= CHUNK || pool.threads() <= 1 {
            for (i, slot) in ev.iter_mut().enumerate() {
                slot.write(resid_of(i));
            }
        } else {
            pool.install(|| {
                ev.par_chunks_mut(CHUNK).enumerate().for_each(|(ci, ec)| {
                    let base = ci * CHUNK;
                    for (q, slot) in ec.iter_mut().enumerate() {
                        slot.write(resid_of(base + q));
                    }
                })
            });
        }
        // SAFETY: every index `0..n` was written above.
        Some(unsafe { par::assume_init(ev) })
    } else {
        None
    };

    let rss_gram = || {
        let gy: f64 = kept.iter().zip(&beta_kept).map(|(&j, b)| b * g[1 + j]).sum();
        let gxx: f64 = kept.iter().zip(&beta_kept).map(|(&ja, ba)| ba * kept.iter().zip(&beta_kept).map(|(&jb, bb)| bb * g[(1 + ja) * r + (1 + jb)]).sum::<f64>()).sum();
        g[0] - 2.0 * gy + gxx
    };
    let rss = match e_row.is_none().then(rss_gram) {
        Some(v) if v.is_finite() && v > 1e-6 * g[0] => v,
        _ => par::sum_chunks(pool, n, CHUNK, 1, |range, acc| {
            for i in range {
                let ei = e_row.as_ref().map_or_else(|| resid_of(i), |e| e[i]);
                acc[0] += w.wv(i) * ei * ei;
            }
        })[0],
    };
    ctx.check()?;

    let row_order = sess.absorber.row_order();
    let res_e = |i: usize| e_row.as_deref().expect("residuals computed")[i];
    let fill_row = |i: usize, a: &mut [f64]| {
        let row = &rows[i * r..(i + 1) * r];
        for (p, &j) in kept.iter().enumerate() {
            a[p] = if bread_has_cons { row[1 + j] + xbar_kept[p] } else { row[1 + j] };
        }
        if bread_has_cons {
            a[k] = 1.0;
        }
    };

    let mut seen_obs = n;
    let mut n_clust = None;
    let mut clust_counts = Vec::new();
    let mut subset_counts = Vec::new();

    let (v_unscaled, meat_raw) = match &opts.vce {
        EngineVce::Ols => (bread.clone(), None),
        EngineVce::Robust => {
            let meat = vcov::robust_meat(n, msize, pool, |i, a| {
                fill_row(i, a);
                w.robust_mult(i, res_e(i))
            });
            let v = vcov::sandwich(&bread, &meat, msize, 1.0);
            (v, Some(meat))
        }
        EngineVce::Hc2 | EngineVce::Hc3 => {
            let hc3 = matches!(opts.vce, EngineVce::Hc3);
            let w_kept = w.raw.as_deref().map(|wt| fe_recovery::to_kept_order(wt, row_order, pool)).transpose()?;
            let h_fe = stata_vce::fe_leverage(&sess.terms, opts.has_constant || has_terms, w_kept.as_deref(), n, ctx, pool)?;
            let h_fe = vcov::to_row_order(&h_fe, row_order, pool)?;
            let fw = w.kind == WeightKind::Fweight;
            let meat = vcov::robust_meat(n, msize, pool, |i, a| {
                let row = &rows[i * r..(i + 1) * r];
                let q: f64 = kept.iter().enumerate().map(|(p, &ja)| {
                    row[1 + ja] * kept.iter().enumerate().map(|(c, &jb)| inv_xx[p * k + c] * row[1 + jb]).sum::<f64>()
                }).sum();
                fill_row(i, a);
                let wv_i = w.wv(i);
                let hi = h_fe[i] + wv_i * q;
                let s = res_e(i) * wv_i;
                let tau = if fw { wv_i } else { 1.0 };
                stata_vce::hc_mult(hc3, s, tau, hi)
            });
            let v = vcov::sandwich(&bread, &meat, msize, 1.0);
            (v, Some(meat))
        }
        EngineVce::Cluster { per_subset_g_adj } => {
            if sess.clusters.is_empty() {
                return Err(RfError::InvalidInput("ols_fit: Cluster VCE requires cluster dimensions on Session".into()));
            }
            let dims: Vec<ClusterCodes> = sess.clusters.iter().map(|c| ClusterCodes { codes: &c.codes, g: c.g }).collect();
            let g_min = dims.iter().map(|c| c.g).min().unwrap_or(0) as f64;
            n_clust = Some(g_min);
            clust_counts = dims.iter().map(|c| c.g as f64).collect();
            if msize == 0 {
                return Err(RfError::InvalidInput("ols_fit: Cluster VCE requires at least one parameter".into()));
            }
            if *per_subset_g_adj && dims.len() > 1 {
                let q = dims.len();
                let owned: Vec<crate::factor::Codes> = dims.iter().map(|d| crate::factor::Codes { codes: d.codes.to_vec(), n: d.g as u32 }).collect();
                for size in 1..=q {
                    let mut comb = Vec::new();
                    fn gen_comb(start: usize, rem: usize, q: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
                        if rem == 0 {
                            out.push(cur.clone());
                            return;
                        }
                        for i in start..=(q - rem) {
                            cur.push(i);
                            gen_comb(i + 1, rem - 1, q, cur, out);
                            cur.pop();
                        }
                    }
                    gen_comb(0, size, q, &mut Vec::new(), &mut comb);
                    for sub in comb {
                        let g = if sub.len() == 1 {
                            dims[sub[0]].g
                        } else {
                            let parts: Vec<&crate::factor::Codes> = sub.iter().map(|&d| &owned[d]).collect();
                            crate::factor::combine(&parts, pool)?.n as usize
                        };
                        subset_counts.push((sub.iter().map(|&i| i + 1).collect(), g));
                    }
                }
                if subset_counts.iter().any(|&(_, g)| g <= 1) {
                    (vec![0.0; msize * msize], Some(vec![0.0; msize * msize]))
                } else {
                    let scale = |g: usize| g as f64 / (g as f64 - 1.0);
                    let meat = vcov::multiway_meat_scaled(&dims, row_order, msize, pool, scale, |i, a| {
                        fill_row(i, a);
                        res_e(i) * w.wv(i)
                    })?;
                    let v = vcov::sandwich(&bread, &meat, msize, 1.0);
                    (v, Some(meat))
                }
            } else if *per_subset_g_adj && dims[0].g <= 1 {
                subset_counts.push((vec![1], dims[0].g));
                (vec![0.0; msize * msize], Some(vec![0.0; msize * msize]))
            } else {
                let scale = if *per_subset_g_adj && dims.len() == 1 {
                    let g = dims[0].g;
                    subset_counts.push((vec![1], g));
                    g as f64 / (g as f64 - 1.0)
                } else {
                    1.0
                };
                let meat = vcov::multiway_meat(&dims, row_order, msize, pool, |i, a| {
                    fill_row(i, a);
                    res_e(i) * w.wv(i)
                })?;
                let meat = if scale == 1.0 { meat } else { meat.into_iter().map(|v| v * scale).collect() };
                let v = vcov::sandwich(&bread, &meat, msize, 1.0);
                (v, Some(meat))
            }
        }
        EngineVce::NeweyWest { lag, time, delta, panel } => {
            let t_ord = vcov::to_row_order(time, row_order, pool)?;
            let p_ord = match panel {
                Some((pc, _)) => Some(vcov::to_row_order(pc, row_order, pool)?),
                None => None,
            };
            let (meat, seen) = newey_west_meat(n, msize, *lag, &t_ord, *delta, p_ord.as_deref(), pool, |i, a| {
                fill_row(i, a);
                res_e(i) * w.wv(i)
            })?;
            seen_obs = seen;
            let v = vcov::sandwich(&bread, &meat, msize, 1.0);
            (v, Some(meat))
        }
        EngineVce::DriscollKraay { lag, time_codes, n_periods } => {
            let t_ord = vcov::to_row_order(time_codes, row_order, pool)?;
            let hsum = hac::slot_sums(&t_ord, *n_periods, msize, pool, |i, a| {
                fill_row(i, a);
                res_e(i) * w.wv(i)
            })?;
            let pos: Vec<i64> = (0..(*n_periods as i64)).collect();
            let c = hac::lag_cross_dense(&hsum, &pos, msize, (*lag).min(n_periods.saturating_sub(1)));
            let wts = hac::bartlett_weights(*lag);
            let meat = hac::hac_combine(&c, &wts[..c.len().min(wts.len())], msize);
            let v = vcov::sandwich(&bread, &meat, msize, 1.0);
            (v, Some(meat))
        }
    };

    let resid = if opts.want_resid {
        let ev = e_row.as_deref().expect("residuals computed when want_resid is set");
        Some(fe_recovery::to_kept_order(ev, row_order, pool)?)
    } else {
        None
    };

    let (d, fe) = if want_coefs {
        let coefs = solve_coefs.as_ref().ok_or_else(|| RfError::Internal("ols_fit: missing FE coefficients".into()))?;
        let mut c = vec![0.0; r];
        c[0] = 1.0;
        for (a, &j) in kept.iter().enumerate() {
            c[1 + j] = -beta_kept[a];
        }
        let combined = coefs.combine(&c);
        let d = if opts.want_d { Some(fe_recovery::d_values(sess.absorber.as_ref(), &combined, b0, pool)?) } else { None };
        let fe = if opts.want_fe { fe_recovery::saved_fe(sess.absorber.as_ref(), &sess.terms, &combined, b0) } else { Vec::new() };
        (d, fe)
    } else {
        (None, Vec::new())
    };

    Ok(OlsEngineOut {
        beta,
        b0,
        status,
        raw_omitted,
        kept,
        rank: k,
        n: big_n,
        sumw: w.sumw,
        df_a: dof.df_a as f64,
        df_a_initial: dof.df_a_initial as f64,
        df_a_redundant: dof.df_a_redundant as f64,
        df_a_nested: dof.df_a_nested as f64,
        rss,
        tss: tss_all[0],
        tss_within: g[0],
        bread,
        msize,
        bread_has_cons,
        v_unscaled,
        meat_raw,
        n_clust,
        clust_counts,
        subset_counts,
        seen_obs,
        ic,
        resid,
        d,
        fe,
        dof_table: dof.table(),
        y_absorbed,
        wmean: cs.wmean,
        df_dof_rows: n,
    })
}

/// Computes Stata `newey`-compatible Bartlett HAC meat matrix over sorted calendar pairs within
/// each panel (`times[t] - times[s] == j * delta`). Returns `(meat, min_pairs_seen)`.
fn newey_west_meat(
    n: usize,
    m: usize,
    lag: usize,
    times: &[f64],
    delta: f64,
    panel: Option<&[u32]>,
    pool: &Pool,
    score: impl Fn(usize, &mut [f64]) -> f64 + Sync,
) -> Result<(Vec<f64>, usize), RfError> {
    if n == 0 {
        return Err(RfError::Code(2000, "no observations".into()));
    }
    if m == 0 {
        return Ok((Vec::new(), n));
    }
    // Materialize scaled score vectors `z_i = s_i * score_scale_i` once in row order.
    let mut z = vec![0.0; n * m];
    if n <= CHUNK || pool.threads() <= 1 {
        for i in 0..n {
            let s = score(i, &mut z[i * m..(i + 1) * m]);
            for v in &mut z[i * m..(i + 1) * m] {
                *v *= s;
            }
        }
    } else {
        pool.install(|| {
            z.par_chunks_mut(CHUNK * m).enumerate().for_each(|(ci, zc)| {
                let base = ci * CHUNK;
                for (q, row) in zc.chunks_exact_mut(m).enumerate() {
                    let s = score(base + q, row);
                    for v in row.iter_mut() {
                        *v *= s;
                    }
                }
            })
        });
    }

    // Group and sort rows by (panel_code, time) to find exact calendar lag pairs in O(N log N).
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        let pa = panel.map_or(0, |p| p[a]);
        let pb = panel.map_or(0, |p| p[b]);
        pa.cmp(&pb).then_with(|| times[a].total_cmp(&times[b])).then_with(|| a.cmp(&b))
    });

    let mut meat = vec![0.0; m * m];
    let mut seen = n;
    for j in 0..=lag {
        let mut tt = vec![0.0; m * m];
        if j == 0 {
            tt = par::sum_chunks(pool, n, CHUNK, m * m, |range, acc| {
                for i in range {
                    let zi = &z[i * m..(i + 1) * m];
                    for a in 0..m {
                        let za = zi[a];
                        if za == 0.0 {
                            continue;
                        }
                        for b in a..m {
                            acc[a * m + b] += za * zi[b];
                        }
                    }
                }
            });
            for a in 0..m {
                for b in 0..a {
                    tt[a * m + b] = tt[b * m + a];
                }
            }
        } else {
            let mut count = 0usize;
            let target = (j as f64) * delta;
            let tol = 1e-9 * target.abs().max(1.0);
            let mut start = 0usize;
            while start < n {
                let p_cur = panel.map_or(0, |p| p[idx[start]]);
                let mut end = start + 1;
                while end < n && panel.map_or(0, |p| p[idx[end]]) == p_cur {
                    end += 1;
                }
                // Two-pointer scan inside [start..end) sorted by `times`.
                let mut sp_lo = start;
                for t_pos in start..end {
                    let t_idx = idx[t_pos];
                    let t_val = times[t_idx];
                    let want = t_val - target;
                    while sp_lo < t_pos && times[idx[sp_lo]] < want - tol {
                        sp_lo += 1;
                    }
                    let mut sp_scan = sp_lo;
                    while sp_scan < t_pos && (times[idx[sp_scan]] - want).abs() <= tol {
                        let s_idx = idx[sp_scan];
                        count += 1;
                        let zt = &z[t_idx * m..(t_idx + 1) * m];
                        let zs = &z[s_idx * m..(s_idx + 1) * m];
                        for a in 0..m {
                            let zsa = zs[a];
                            for b in 0..m {
                                tt[a * m + b] += zsa * zt[b];
                            }
                        }
                        sp_scan += 1;
                    }
                }
                start = end;
            }
            match count {
                0 => return Err(RfError::Code(2000, "no observations".into())),
                1 => return Err(RfError::Code(2001, "insufficient observations".into())),
                _ => seen = count,
            }
        }
        let scale = 1.0 - (j as f64) / ((1 + lag) as f64);
        for a in 0..m {
            for b in 0..m {
                let mj = (tt[a * m + b] + tt[b * m + a]) * scale;
                meat[a * m + b] = if j == 0 { mj * 0.5 } else { meat[a * m + b] + mj };
            }
        }
    }
    Ok((meat, seen))
}

/// Options for [`iv_fit`].
#[derive(Clone, Debug)]
pub struct IvEngineOpts {
    pub small: bool,
    pub report_constant: bool,
    pub noconstant: bool,
    pub sdofminus: Option<f64>,
    pub dofminus: f64,
    pub first: bool,
    pub ffirst: bool,
    pub tol: f64,
    pub maxiter: usize,
    pub dof: DofOptions,
    pub want_resid: bool,
    pub want_d: bool,
    pub want_fe: bool,
}

impl Default for IvEngineOpts {
    fn default() -> Self {
        Self {
            small: false,
            report_constant: true,
            noconstant: false,
            sdofminus: None,
            dofminus: 0.0,
            first: false,
            ffirst: false,
            tol: 1e-8,
            maxiter: 16_000,
            dof: DofOptions::default(),
            want_resid: false,
            want_d: false,
            want_fe: false,
        }
    }
}

/// 2SLS / IV estimation entry point delegating to [`Session::iv`].
pub fn iv_fit(sess: &mut Session, rows: Vec<f64>, lay: &Layout, opts: &IvEngineOpts, ctx: &Ctx) -> Result<EstResult, RfError> {
    let spec = Spec {
        tol: opts.tol,
        maxiter: opts.maxiter,
        dof: opts.dof.clone(),
        noconstant: opts.noconstant,
        want_resid: opts.want_resid,
        want_d: opts.want_d,
        want_fe: opts.want_fe,
        want_groupvar: false,
        iv: IvOptions {
            first: opts.first,
            ffirst: opts.ffirst,
            small: opts.small,
            report_constant: opts.report_constant,
            sdofminus: opts.sdofminus,
            dofminus: opts.dofminus,
        },
    };
    sess.iv(rows, lay, &spec, ctx)
}

/// Options for [`glm_fit`].
#[derive(Clone, Debug)]
pub struct GlmEngineOpts {
    pub family: Family,
    pub noconstant: bool,
    pub tol: f64,
    pub itol: f64,
    pub maxiter: usize,
    pub fe_maxiter: usize,
    pub dof: DofOptions,
    pub guess: Guess,
    pub separation: Separation,
    pub convention: GlmConvention,
    pub collinear: GlmCollinear,
    pub keep_unadjusted_vce: bool,
    pub want_d: bool,
    pub want_fe: bool,
    pub log: bool,
    pub init_mean: bool,
    pub ml_ltol: Option<f64>,
    pub ml_ll_const: f64,
    /// Optional initial fitted means `mu` (unstandardized `y` scale, in `Session` kept order).
    pub init_mu: Option<Vec<f64>>,
}

impl Default for GlmEngineOpts {
    fn default() -> Self {
        Self {
            family: Family::Poisson,
            noconstant: false,
            tol: 1e-8,
            itol: 1e-8,
            maxiter: 10_000,
            fe_maxiter: 16_000,
            dof: DofOptions::default(),
            guess: Guess::Simple,
            separation: Separation::default(),
            convention: GlmConvention::Ppmlhdfe,
            collinear: GlmCollinear::Ppmlhdfe,
            keep_unadjusted_vce: true,
            want_d: false,
            want_fe: false,
            log: false,
            init_mean: false,
            ml_ltol: None,
            ml_ll_const: 0.0,
            init_mu: None,
        }
    }
}

/// Full GLM estimation result returned by [`glm_fit`], in kept observation order on the original
/// data scale.
pub struct GlmEngineOut {
    pub res: EstResult,
    /// Inverse Hessian / Fisher information on `[kept..., _cons?]` (unscaled OIM covariance).
    pub bread: Vec<f64>,
    /// Linear predictor `eta = offset + X beta + FE` in final kept order (original `y` scale).
    pub eta: Vec<f64>,
    /// Mean prediction `mu` in final kept order (original `y` scale).
    pub mu: Vec<f64>,
    /// Final IRLS weights in final kept order.
    pub w_irls: Vec<f64>,
    /// Working residuals in final kept order.
    pub resid: Vec<f64>,
    pub deviance: f64,
    pub dev_trace: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub msize: usize,
}

/// Canonical Poisson / Logit GLM engine over `Session`, supporting true `noconstant` (no FE
/// residualization when `Session::terms` is empty) and `OIM` (`keep_unadjusted_vce = true`).
pub fn glm_fit(sess: &mut Session, rows: Vec<f64>, nx: usize, offset: Option<Vec<f64>>, opts: &GlmEngineOpts, ctx: &Ctx) -> Result<GlmEngineOut, RfError> {
    let spec = Spec {
        tol: opts.itol,
        maxiter: opts.fe_maxiter,
        dof: opts.dof.clone(),
        noconstant: opts.noconstant,
        want_resid: false,
        want_d: opts.want_d,
        want_fe: opts.want_fe,
        want_groupvar: false,
        iv: IvOptions::default(),
    };
    let g = GlmOptions {
        family: opts.family,
        tol: opts.tol,
        itol: opts.itol,
        maxiter: opts.maxiter,
        separation: opts.separation,
        guess: opts.guess,
        collinear: opts.collinear,
        log: opts.log,
    };
    let (res, extra): (EstResult, GlmEngineExtra) = sess.glm_for_engine(
        rows,
        &Layout::ols(nx),
        offset,
        &spec,
        &g,
        opts.convention,
        opts.keep_unadjusted_vce,
        (opts.init_mean, opts.ml_ltol, opts.ml_ll_const, opts.init_mu.as_deref()),
        ctx,
    )?;
    Ok(GlmEngineOut {
        res,
        bread: extra.bread,
        eta: extra.eta,
        mu: extra.mu,
        w_irls: extra.w_irls,
        resid: extra.resid,
        deviance: extra.deviance,
        dev_trace: extra.dev_trace,
        iterations: extra.iterations,
        converged: extra.converged,
        msize: extra.msize,
    })
}
