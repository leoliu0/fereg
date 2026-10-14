//! Variance estimators of the Stata layer beyond unadjusted / robust / cluster, shared by
//! [`Session::ols`](crate::Session::ols), [`Session::iv`](crate::Session::iv) and the GLM
//! estimators:
//! - Stata `regress` `vce(hc2)` / `vce(hc3)` (eq. 10/12) with the leverage of the full design;
//! - reghdfe 6.13.1 `vce(dkraay [bw])` (`reghdfe_vce_dkraay`, reghdfe.mata l.5721-5859):
//!   positional periods, Bartlett weights, bandwidth = lags + 1;
//! - ivreg2's kernel-robust variances (livreg2 `m_omega`, `m_calckw`, `s_vkernel`; ivreghdfe
//!   `ivparse` and `s_abw`): AC, HAC, Driscoll–Kraay (cluster on time + kernel) and Thompson
//!   (cluster on panel and time + kernel), calendar lags on the tsset time (step tdelta, within
//!   the panel when xtset), `psd0`/`psda`;
//! - fixest's Conley meat;
//! - the leave-one-cluster-out jackknife (CRV3), by inclusion–exclusion over several dimensions.
//!
//! Meats are un-normalized sums ("N x m_omega"), in the convention of `vcov::robust_meat`:
//! the callers' small-sample factors divide by N where the reference does.

use crate::api::{Bandwidth, ConleyVce, Ctx, KernelVce, Psd, WeightKind};
use crate::conley::{self, ConleyOpts};
use crate::crv3::{self, JkProblem};
use crate::error::RfError;
use crate::factor::{self, Codes};
use crate::hac::{self, Kernel, Panel, PanelError};
use crate::leverage::{self, FeCols};
use crate::par::{self, CHUNK, Pool};
use crate::session::{ClusterDim, Term, VceData};
use crate::vcov;

/// Scratch length kept on the stack by [`with_ez`].
const STACK_EZ: usize = 128;

/// Calls `f` with the `n` values written by `fill` (on the stack when they fit).
#[inline]
fn with_ez<R>(n: usize, fill: impl FnOnce(&mut [f64]) -> f64, f: impl FnOnce(&[f64], f64) -> R) -> R {
    if n <= STACK_EZ {
        let mut buf = [0.0f64; STACK_EZ];
        let s = fill(&mut buf[..n]);
        f(&buf[..n], s)
    } else {
        let mut buf = vec![0.0; n];
        let s = fill(&mut buf);
        f(&buf, s)
    }
}

/// a = e (x) z (index c*l + k) for the [e, z] values `ez` (`me` residual combinations).
#[inline]
fn kron_row(ez: &[f64], me: usize, a: &mut [f64]) {
    let (e, z) = ez.split_at(me);
    let l = z.len();
    for (c, &ec) in e.iter().enumerate() {
        for (k, &zk) in z.iter().enumerate() {
            a[c * l + k] = ec * zk;
        }
    }
}

/// kron(a, b) of row-major square matrices (p x p and q x q).
fn kron(a: &[f64], p: usize, b: &[f64], q: usize) -> Vec<f64> {
    let d = p * q;
    let mut out = vec![0.0; d * d];
    for i in 0..p {
        for j in 0..p {
            let x = a[i * p + j];
            for k in 0..q {
                for l in 0..q {
                    out[(i * q + k) * d + j * q + l] = x * b[k * q + l];
                }
            }
        }
    }
    out
}

/// Block [r0..r0+p) x [r0..r0+p) of an m x m matrix.
fn block(m_: &[f64], m: usize, r0: usize, p: usize) -> Vec<f64> {
    (0..p * p).map(|ab| m_[(r0 + ab / p) * m + r0 + ab % p]).collect()
}

/// ivreg2 `psd0` / `psda` on a symmetric matrix (Mata `_makesymmetric` copies the lower
/// triangle, then negative eigenvalues become 0 or their absolute value).
pub(crate) fn apply_psd(s: &mut [f64], k: usize, psd: Psd) {
    if psd == Psd::None || k == 0 {
        return;
    }
    for i in 0..k {
        for j in i + 1..k {
            s[i * k + j] = s[j * k + i];
        }
    }
    let (lam, u) = vcov::sym_eigen(s, k);
    let lam: Vec<f64> = lam.iter().map(|&l| if psd == Psd::Abs { l.abs() } else { l.max(0.0) }).collect();
    s.iter_mut().for_each(|x| *x = 0.0);
    for (e, &l) in lam.iter().enumerate() {
        if l == 0.0 {
            continue;
        }
        for i in 0..k {
            let ui = u[i * k + e] * l;
            for j in 0..k {
                s[i * k + j] += ui * u[j * k + e];
            }
        }
    }
    for i in 0..k {
        for j in i + 1..k {
            s[i * k + j] = s[j * k + i];
        }
    }
}

/// k-subsets of 0..q in lexicographic order, for k = 1..=q (reghdfe's multiway order).
fn all_subsets(q: usize) -> Vec<Vec<usize>> {
    fn rec(start: usize, q: usize, k: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for i in start..q {
            cur.push(i);
            rec(i + 1, q, k, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    for k in 1..=q {
        rec(0, q, k, &mut Vec::new(), &mut out);
    }
    out
}

// ---------------------------------------------------------------------------------------------
// HC2 / HC3

/// Leverage of the absorbed part of the design (kept order): `leverage::fe_hat` of every term,
/// or w_i / sum w for the constant alone (`noabsorb`). `w` are the raw weights (kept order).
pub(crate) fn fe_leverage(terms: &[Term], has_intercept: bool, w: Option<&[f64]>, n: usize, ctx: &Ctx, pool: &Pool) -> Result<Vec<f64>, RfError> {
    if terms.is_empty() {
        if !has_intercept {
            return par::alloc_vec(n, 0.0);
        }
        return Ok(match w {
            Some(w) => {
                let sw = par::sum_chunks(pool, n, CHUNK, 1, |r, a| a[0] += w[r].iter().sum::<f64>())[0];
                w.iter().map(|v| v / sw).collect()
            }
            None => vec![1.0 / n as f64; n],
        });
    }
    let cols: Vec<FeCols> = terms
        .iter()
        .map(|t| FeCols { codes: &t.codes, g: t.g, intercept: t.intercept, slopes: t.slopes.iter().map(Vec::as_slice).collect() })
        .collect();
    leverage::fe_hat(&cols, w, n, ctx, pool)
}

/// Stata `regress` HC2/HC3 multiplier of the design row given the robust score s = e*w_v:
/// HC2 s^2/(tau - h), HC3 tau*s^2/(tau - h)^2 (eq. 10/12), tau = the frequency weight or 1, h
/// the leverage of the weighted design. Observations with tau - h <= sqrt(eps)*tau (residual 0
/// in exact arithmetic) contribute nothing.
#[inline]
pub(crate) fn hc_mult(hc3: bool, s: f64, tau: f64, h: f64) -> f64 {
    let d = tau - h;
    if d <= f64::EPSILON.sqrt() * tau {
        return 0.0;
    }
    if hc3 { tau * s * s / (d * d) } else { s * s / d }
}

// ---------------------------------------------------------------------------------------------
// reghdfe Driscoll–Kraay

/// reghdfe `reghdfe_vce_dkraay` meat on the dense time codes (positional periods, gaps
/// ignored): h_t = sum_{i in t} s_i a_i, M = sum_t h_t h_t' + sum_{j=1}^{L} (1 - j/(L+1))
/// (Omega_j + Omega_j'), L = bw - 1 (default bw `floor(4(T/100)^(2/9)) + 1`, reduced to T - 1
/// lags with reghdfe's warning). Returns (meat, bandwidth used = L + 1, notes).
pub(crate) fn dkraay_reghdfe(
    time: &ClusterDim,
    bw: Option<usize>,
    row_order: &[u32],
    m: usize,
    pool: &Pool,
    row: impl Fn(usize, &mut [f64]) -> f64 + Sync,
) -> Result<(Vec<f64>, usize, Vec<String>), RfError> {
    let t = time.g;
    let bw = bw.unwrap_or_else(|| hac::bw_reghdfe_dkraay(t));
    let mut notes = Vec::new();
    let mut lags = bw.saturating_sub(1);
    if bw == 0 {
        notes.push("Warning: bandwidth must be >= 1; setting to 1 (0 lags)".to_string());
    }
    if lags >= t {
        notes.push(format!("Warning: bandwidth {bw} too large for {t} time periods; reducing to {t}"));
        lags = t - 1;
    }
    let codes = vcov::to_row_order(&time.codes, row_order, pool)?;
    let h = hac::slot_sums(&codes, t, m, pool, row)?;
    let pos: Vec<i64> = (0..t as i64).collect();
    let c = hac::lag_cross_dense(&h, &pos, m, lags);
    let mut meat = hac::hac_combine(&c, &hac::bartlett_weights(lags), m);
    if m > 0 && h.is_empty() {
        meat = vec![0.0; m * m];
    }
    Ok((meat, lags + 1, notes))
}

// ---------------------------------------------------------------------------------------------
// ivreg2 kernels

/// m_omega branch of a kernel VCE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KernelMode {
    /// No robust, no cluster: homoskedastic autocorrelation (also `kiefer`).
    Ac,
    /// Robust, no cluster.
    Hac,
    /// Cluster on the time variable + kernel.
    DriscollKraay,
    /// Cluster on the panel and the time variable + kernel.
    Thompson,
}

/// A kernel VCE fixed on the estimation sample (row order).
pub(crate) struct KernelPlan {
    pub mode: KernelMode,
    pub kernel: Kernel,
    /// Bandwidth used (bw, kiefer's T, or the `bw(auto)` result once resolved).
    pub bw: f64,
    /// `bw(auto)` still to be resolved by [`KernelPlan::resolve_auto`].
    pub auto_pending: bool,
    /// ivreghdfe `e(bwchoice)`.
    pub bwchoice: String,
    /// Time span T = tmax - tmin + 1 (time units).
    pub span: f64,
    tmin: f64,
    tmax: f64,
    tdelta: f64,
    /// ivreg2 `psd0` / `psda` (applied by the callers on ivreg2's coordinates, [`apply_psd`]).
    pub psd: Psd,
    /// `kiefer`: ivreghdfe's ranktest calls then get no `bw()` (bw is still 0 when `bwopt` is
    /// built, ivreghdfe.ado 3920-3933), so the identification statistics are those without a
    /// kernel (iid, or robust/cluster when pweights or a cluster are given).
    pub kiefer: bool,
    /// Lag weights of lags 1..=TAU (index l-1).
    weights: Vec<f64>,
    /// Weights of ivreghdfe's ranktest calls: Bartlett at the same bw (ivreghdfe passes
    /// `bw()` but never `kernel()`, `psd0`/`psda` to ranktest: its `kernopt` local is unset).
    rank_weights: Vec<f64>,
    /// Rows by (panel, position) for within-panel lags (AC, HAC, Thompson, auto).
    panel: Option<Panel>,
    /// Row-order panel codes and their count (xtset).
    panel_codes: Option<(Vec<u32>, usize)>,
    /// Row-order period slot (calendar position) and the number of slots (DK, Thompson).
    slots: Vec<u32>,
    n_slots: usize,
    /// Distinct periods in the sample (DK cluster count).
    pub n_periods: usize,
    /// Messages ivreg2 prints from Mata.
    pub notes: Vec<String>,
}

fn code(rc: i32, msg: &str) -> RfError {
    RfError::Code(rc, msg.to_string())
}

impl KernelPlan {
    /// Validates the request against the sample and fixes the lag structure. `w_kind`: ivreg2
    /// refuses frequency weights with a kernel.
    pub(crate) fn new(kv: &KernelVce, vdata: &VceData, w_kind: WeightKind, row_order: &[u32], pool: &Pool) -> Result<KernelPlan, RfError> {
        if w_kind == WeightKind::Fweight {
            return Err(code(101, "fweights not allowed (data are -tsset-)"));
        }
        let time = vdata.time.as_deref().ok_or_else(|| RfError::Internal("kernel VCE without time values".into()))?;
        let time = vcov::to_row_order(time, row_order, pool)?;
        let tdelta = kv.time.delta;
        let pos = hac::pos_calendar(&time, tdelta)?;
        let tmin = time.iter().copied().fold(f64::INFINITY, f64::min);
        let tmax = time.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let span = tmax - tmin + 1.0;
        let panel_codes = match &vdata.panel {
            Some((c, g)) => Some((vcov::to_row_order(c, row_order, pool)?, *g)),
            None => None,
        };

        let sorted = |d: &[usize]| {
            let mut s = d.to_vec();
            s.sort_unstable();
            s.dedup();
            s
        };
        let tdim = vec![kv.time.time];
        let pdim = kv.time.panel.map(|p| vec![p]);
        let mode = match kv.cluster.as_slice() {
            [] if kv.robust => KernelMode::Hac,
            [] => KernelMode::Ac,
            [a] if sorted(a) == tdim => KernelMode::DriscollKraay,
            [_] => return Err(code(198, "Error: cluster kernel-robust requires clustering on tsset time variable.")),
            [a, b] if pdim.as_ref().is_some_and(|p| (sorted(a) == *p && sorted(b) == tdim) || (sorted(b) == *p && sorted(a) == tdim)) => {
                KernelMode::Thompson
            }
            _ => return Err(code(198, "Error: cluster kernel-robust requires clustering on tsset panel & time vars.")),
        };

        let kernel = if kv.kiefer { Kernel::Truncated } else { kv.kernel };
        let auto = !kv.kiefer && kv.bw == Bandwidth::Auto;
        let bw = if kv.kiefer {
            // ivreghdfe checks bw (still 0) against the time span before kiefer sets bw = T.
            span
        } else {
            match kv.bw {
                Bandwidth::Fixed(b) => {
                    if !(b > 0.0) {
                        return Err(code(198, "invalid bandwidth in option bw() - must be real > 0"));
                    }
                    if b > (span - 1.0) / tdelta {
                        return Err(code(198, "invalid bandwidth in option bw() - cannot exceed timespan of data"));
                    }
                    b
                }
                Bandwidth::Auto => {
                    if kv.time.panel.is_some() {
                        return Err(code(198, "Automatic bandwidth selection not available for panel data"));
                    }
                    f64::NAN
                }
            }
        };

        let need_panel = auto || matches!(mode, KernelMode::Ac | KernelMode::Hac | KernelMode::Thompson);
        let panel = if need_panel {
            let unit = panel_codes.as_ref().map(|(c, _)| c.as_slice());
            match Panel::build(unit, &pos, pool) {
                Ok(p) => Some(p),
                Err(PanelError::Duplicate) => {
                    return Err(code(451, if unit.is_some() { "repeated time values within panel" } else { "repeated time values in sample" }));
                }
                Err(PanelError::Rf(e)) => return Err(e),
            }
        } else {
            None
        };

        let n_slots = pos.iter().copied().max().map_or(0, |p| p as usize + 1);
        let mut seen = vec![false; n_slots];
        pos.iter().for_each(|&p| seen[p as usize] = true);
        let n_periods = seen.iter().filter(|s| **s).count();
        let slots: Vec<u32> = if matches!(mode, KernelMode::DriscollKraay | KernelMode::Thompson) {
            pos.iter().map(|&p| p as u32).collect()
        } else {
            Vec::new()
        };

        let mut plan = KernelPlan {
            mode,
            kernel,
            bw,
            auto_pending: auto,
            bwchoice: String::new(),
            span,
            tmin,
            tmax,
            tdelta,
            psd: kv.psd,
            kiefer: kv.kiefer,
            weights: Vec::new(),
            rank_weights: Vec::new(),
            panel,
            panel_codes,
            slots,
            n_slots,
            n_periods,
            notes: Vec::new(),
        };
        if !auto {
            plan.set_weights();
        }
        Ok(plan)
    }

    fn set_weights(&mut self) {
        self.weights = hac::ivreg2_lag_weights(self.kernel, self.bw, hac::ivreg2_spectral_span(self.tmin, self.tmax, self.tdelta));
        self.rank_weights = hac::ivreg2_lag_weights(Kernel::Bartlett, self.bw, 0);
    }

    /// ivreghdfe `abw` / `s_abw` on f_i = u_i * sum_j z_ij (row order; 2SLS residual times the
    /// row sum of the instruments, no weights), N = `nobs`.
    pub(crate) fn resolve_auto(&mut self, f: &[f64], nobs: f64, pool: &Pool) {
        if !self.auto_pending {
            return;
        }
        let p = self.panel.as_ref().expect("bw(auto) builds the time order");
        let fs: Vec<f64> = p.order().iter().map(|&i| f[i as usize]).collect();
        match hac::bw_ivreg2_auto(self.kernel, &fs, p.pos(), nobs, self.span, pool) {
            Some(bw) => {
                self.bw = bw;
                self.bwchoice = "Automatic bw selection according to Newey-West (1994)".to_string();
            }
            None => {
                let k = self.kernel.name();
                self.notes.push(format!("Error: kernel {k} not compatible with bw(auto)"));
                self.bw = 1.0;
                self.bwchoice = format!("Kernel {k} not compatible with bw(auto); bw=1 (default)");
            }
        }
        self.auto_pending = false;
        self.set_weights();
    }

    /// Number of clusters of the small-sample factor: periods (DK), min(panels, periods)
    /// (Thompson); None without a cluster.
    pub(crate) fn n_clust(&self) -> Option<f64> {
        match self.mode {
            KernelMode::DriscollKraay => Some(self.n_periods as f64),
            KernelMode::Thompson => Some(self.panel_codes.as_ref().map_or(0, |(_, g)| *g).min(self.n_periods) as f64),
            _ => None,
        }
    }

    /// Un-normalized m_omega (N x shat, (me*l)^2) of the moments e (x) z, before `psd0`/`psda`.
    /// `ez(i, buf)` writes [e_1..e_me, z_1..z_l] of row i into `buf` and returns its regression
    /// weight w_v (ivreg2 `wf*wvar`); `big_n` is ivreg2's N. `ranktest`: the Bartlett weights
    /// ivreghdfe's ranktest calls use.
    pub(crate) fn meat(&self, me: usize, l: usize, big_n: f64, ranktest: bool, pool: &Pool, ez: impl Fn(usize, &mut [f64]) -> f64 + Sync) -> Result<Vec<f64>, RfError> {
        debug_assert!(!self.auto_pending, "bw(auto) not resolved");
        let dim = me * l;
        let n = self.n_rows();
        let weights = if ranktest { &self.rank_weights } else { &self.weights };
        let tau = weights.len();
        let kron_score = |i: usize, a: &mut [f64]| -> f64 { with_ez(me + l, |b| ez(i, b), |b, w| {
            kron_row(b, me, a);
            w
        }) };
        let hac_panel = || -> Result<Vec<f64>, RfError> {
            let p = self.panel.as_ref().expect("panel built for within-panel lags");
            let c = hac::lag_cross(p, tau, dim, pool, kron_score)?;
            Ok(hac::hac_combine(&c, weights, dim))
        };
        let dk = || -> Result<Vec<f64>, RfError> {
            let h = hac::slot_sums(&self.slots, self.n_slots, dim, pool, kron_score)?;
            if h.is_empty() {
                return Ok(vec![0.0; dim * dim]);
            }
            let pos: Vec<i64> = (0..self.n_slots as i64).collect();
            let c = hac::lag_cross_dense(&h, &pos, dim, tau);
            Ok(hac::hac_combine(&c, weights, dim))
        };
        let s = match self.mode {
            KernelMode::Hac => hac_panel()?,
            KernelMode::DriscollKraay => dk()?,
            KernelMode::Thompson => {
                let (codes, g) = self.panel_codes.as_ref().expect("Thompson needs the panel");
                let s1 = vcov::cluster_meat(codes, *g, dim, pool, kron_score)?;
                let s2 = dk()?;
                let s3 = hac_panel()?;
                s1.iter().zip(&s2).zip(&s3).map(|((a, b), c)| a + b - c).collect()
            }
            KernelMode::Ac => {
                // sigma2 # ZZ + sum_tau kw (sigmahat_tau # ZZ_tau + transpose), with
                // sigma2 = e'We/N and sigmahat_tau = sum of lagged pairs w_t w_s e_t e_s'/N.
                let ml = me + l;
                let base = vcov::robust_meat(n, ml, pool, |i, a| ez(i, a));
                let mut s = kron(&block(&base, ml, 0, me).iter().map(|v| v / big_n).collect::<Vec<_>>(), me, &block(&base, ml, me, l), l);
                let p = self.panel.as_ref().expect("panel built for within-panel lags");
                let c = hac::lag_cross(p, tau, ml, pool, |i, a| ez(i, a))?;
                for (ct, &kw) in c.iter().skip(1).zip(weights) {
                    if kw == 0.0 {
                        continue;
                    }
                    let see: Vec<f64> = (0..me * me).map(|ab| ct[(ab / me) * ml + ab % me] / big_n).collect();
                    let szz: Vec<f64> = (0..l * l).map(|ab| ct[(me + ab / l) * ml + me + ab % l]).collect();
                    let g = kron(&see, me, &szz, l);
                    for a in 0..dim {
                        for b in 0..dim {
                            s[a * dim + b] += kw * (g[a * dim + b] + g[b * dim + a]);
                        }
                    }
                }
                s
            }
        };
        Ok(s)
    }

    fn n_rows(&self) -> usize {
        self.panel.as_ref().map_or(self.slots.len(), Panel::n)
    }
}

// ---------------------------------------------------------------------------------------------
// Conley

/// fixest Conley meat on the session's coordinates (row order); returns (meat, cutoff km).
pub(crate) fn conley_meat(
    cv: &ConleyVce,
    vdata: &VceData,
    row_order: &[u32],
    m: usize,
    pool: &Pool,
    row: impl Fn(usize, &mut [f64]) -> f64 + Sync,
) -> Result<(Vec<f64>, f64), RfError> {
    let (lat, lon) = vdata.coords.as_ref().ok_or_else(|| RfError::Internal("Conley VCE without coordinates".into()))?;
    let lat = vcov::to_row_order(lat, row_order, pool)?;
    let lon = vcov::to_row_order(lon, row_order, pool)?;
    let opts = ConleyOpts { cutoff_km: cv.cutoff, pixel_km: cv.pixel, spherical: cv.spherical };
    conley::conley_meat(&lat, &lon, &opts, m, pool, row)
}

// ---------------------------------------------------------------------------------------------
// CRV3

/// Jackknife meat over the cluster dimensions: sum over non-empty subsets S of the dimensions
/// (reghdfe's order) of (-1)^{|S|+1} sum_g (b_(-g) - b)(b_(-g) - b)', the clusters of S being the
/// intersection of its dimensions. `augment(codes, g, betas)` turns the G x k leave-out
/// coefficients of [`crv3::leave_cluster_out`] (order of `p.beta`) into the G x k' rows compared
/// with `beta` (k' entries; e.g. adds the reported constant).
pub(crate) fn crv3_meat(
    p: &JkProblem,
    dims: &[ClusterDim],
    beta: &[f64],
    augment: &dyn Fn(&[u32], usize, Vec<f64>) -> Vec<f64>,
    ctx: &Ctx,
    pool: &Pool,
) -> Result<Vec<f64>, RfError> {
    let k = beta.len();
    let mut meat = vec![0.0; k * k];
    for subset in all_subsets(dims.len()) {
        let sign = if subset.len() % 2 == 1 { 1.0 } else { -1.0 };
        let combined;
        let (codes, g): (&[u32], usize) = if let [d] = subset[..] {
            (&dims[d].codes, dims[d].g)
        } else {
            let owned: Vec<Codes> = subset.iter().map(|&d| Codes { codes: dims[d].codes.clone(), n: dims[d].g as u32 }).collect();
            combined = factor::combine(&owned.iter().collect::<Vec<_>>(), pool)?;
            (&combined.codes, combined.n as usize)
        };
        ctx.check()?;
        let betas = crv3::leave_cluster_out(p, codes, g, ctx, pool)?;
        let rows = augment(codes, g, betas);
        let part = crv3::jackknife_meat(&rows, beta, k);
        meat.iter_mut().zip(&part).for_each(|(a, v)| *a += sign * v);
    }
    Ok(meat)
}

/// Per-cluster sums of w and w*v_j (kept order, `cols` columns), G x (1 + cols), and the totals.
pub(crate) fn cluster_wsums(codes: &[u32], g: usize, w: Option<&[f64]>, cols: &[&[f64]]) -> (Vec<f64>, Vec<f64>) {
    let c = 1 + cols.len();
    let mut s = vec![0.0; g * c];
    for (i, &gi) in codes.iter().enumerate() {
        let wi = w.map_or(1.0, |w| w[i]);
        let r = &mut s[gi as usize * c..(gi as usize + 1) * c];
        r[0] += wi;
        for (rv, col) in r[1..].iter_mut().zip(cols) {
            *rv += wi * col[i];
        }
    }
    let mut tot = vec![0.0; c];
    for r in s.chunks_exact(c) {
        tot.iter_mut().zip(r).for_each(|(t, v)| *t += v);
    }
    (s, tot)
}

/// Adds the regression-through-the-means constant of each leave-out sample to the jackknife
/// rows: b0_(-g) = ybar_(-g) - xbar_(-g)'b_(-g), means weighted by `w` (reghdfe's reported
/// constant of a refit; for GLMs the working response and the IRLS weights). `betas` are G x k
/// on the columns `x` (k = x.len()); returns G x (k + 1).
pub(crate) fn add_constant(codes: &[u32], g: usize, betas: Vec<f64>, w: Option<&[f64]>, y: &[f64], x: &[&[f64]]) -> Vec<f64> {
    let k = x.len();
    let mut cols: Vec<&[f64]> = vec![y];
    cols.extend_from_slice(x);
    let (s, tot) = cluster_wsums(codes, g, w, &cols);
    let c = 2 + k;
    let mut out = Vec::with_capacity(g * (k + 1));
    for (gi, b) in betas.chunks_exact(k.max(1)).take(g).enumerate() {
        let r = &s[gi * c..(gi + 1) * c];
        let sw = tot[0] - r[0];
        let mut b0 = (tot[1] - r[1]) / sw;
        for j in 0..k {
            b0 -= b[j] * (tot[2 + j] - r[2 + j]) / sw;
        }
        out.extend_from_slice(&b[..k]);
        out.push(b0);
    }
    if k == 0 {
        // No regressors: one constant per cluster.
        out.clear();
        for gi in 0..g {
            let r = &s[gi * c..(gi + 1) * c];
            out.push((tot[1] - r[1]) / (tot[0] - r[0]));
        }
    }
    out
}
