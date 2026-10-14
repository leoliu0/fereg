//! Kernel (HAC) variance building blocks shared by every front end: ivreg2's kernel weights and
//! lag windows, the reference lag/bandwidth rules (fixest, pyfixest, reghdfe, ivreg2), and the
//! lagged score cross products behind Newey–West, Driscoll–Kraay and ivreg2 HAC meats.
//!
//! Pairing is on caller-supplied integer positions: a pair at lag l is two rows of the same unit
//! whose positions differ by l. The references define positions differently:
//! - fixest: dense rank of the time variable over the estimation sample (the codes of the time
//!   factor, whose levels ascend);
//! - pyfixest: index within the unit after sorting by (unit, time)
//!   ([`Panel::within_unit_positions`]);
//! - Stata (ivreg2 `L#.` operators on tsset/xtset data): (t - tmin) / tdelta ([`pos_calendar`]).
//!
//! Inputs follow `vcov.rs`: `row(i, a)` fills the design row a_i (length m) of row i and returns
//! the score multiplier s_i; the score is s_i*a_i. Matrices are row-major m x m. Every kernel splits
//! its work into chunks whose sizes depend only on the data and combines them in chunk order, so
//! results are bit-identical for any thread count.

use std::ops::Range;

use rayon::prelude::*;

use crate::error::RfError;
use crate::par::{self, CHUNK, Pool};
use crate::structure;
use crate::vcov;

/// ivreg2 kernels (`s_vkernel` list).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Bartlett,
    Parzen,
    Truncated,
    TukeyHanning,
    TukeyHamming,
    QuadraticSpectral,
    Daniell,
    Tent,
}

impl Kernel {
    /// ivreg2 `s_vkernel`: the spelling is lower-cased (ASCII) and stripped of leading blanks
    /// only, then must equal one of ivreg2's abbreviations or names; "" is Bartlett.
    pub fn parse(s: &str) -> Option<Kernel> {
        Some(match s.trim_start_matches(' ').to_ascii_lowercase().as_str() {
            "" | "bar" | "bartlett" => Kernel::Bartlett,
            "par" | "parzen" => Kernel::Parzen,
            "tru" | "truncated" => Kernel::Truncated,
            "thann" | "tukey-hanning" => Kernel::TukeyHanning,
            "thamm" | "tukey-hamming" => Kernel::TukeyHamming,
            "qua" | "qs" | "quadratic-spectral" | "quadratic spectral" => Kernel::QuadraticSpectral,
            "dan" | "danielle" => Kernel::Daniell,
            "ten" | "tent" => Kernel::Tent,
            _ => return None,
        })
    }

    /// ivreg2's display name (`strproper` of its list entry; e(kernel)).
    pub fn name(self) -> &'static str {
        match self {
            Kernel::Bartlett => "Bartlett",
            Kernel::Parzen => "Parzen",
            Kernel::Truncated => "Truncated",
            Kernel::TukeyHanning => "Tukey-Hanning",
            Kernel::TukeyHamming => "Tukey-Hamming",
            Kernel::QuadraticSpectral => "Quadratic Spectral",
            Kernel::Daniell => "Danielle",
            Kernel::Tent => "Tent",
        }
    }

    /// m_omega's "spectral" window (every lag of the time span) rather than lags 1..bw.
    pub fn is_spectral(self) -> bool {
        matches!(self, Kernel::QuadraticSpectral | Kernel::Daniell | Kernel::Tent)
    }

    /// ivreg2 notes "kernel=%s and bw=1 implies zero lags used" for these kernels (weight 0 at
    /// tau = bw).
    pub fn zero_lags_at_bw1(self) -> bool {
        matches!(self, Kernel::Bartlett | Kernel::Parzen | Kernel::TukeyHanning | Kernel::TukeyHamming)
    }
}

/// ivreg2 `m_calckw(tau, bw, kernel)`, operation by operation (karg = tau/bw; Tent uses
/// cos(tau*karg) as written in ivreg2).
pub fn kernel_weight(kernel: Kernel, tau: f64, bw: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let karg = tau / bw;
    match kernel {
        Kernel::Truncated => 1.0,
        Kernel::Bartlett => 1.0 - karg,
        Kernel::Parzen => {
            if karg <= 0.5 {
                1.0 - 6.0 * (karg * karg) + 6.0 * (karg * karg * karg)
            } else {
                let d = 1.0 - karg;
                2.0 * (d * d * d)
            }
        }
        Kernel::TukeyHanning => 0.5 + 0.5 * (pi * karg).cos(),
        Kernel::TukeyHamming => 0.54 + 0.46 * (pi * karg).cos(),
        Kernel::Tent => 2.0 * (1.0 - (tau * karg).cos()) / (karg * karg),
        Kernel::Daniell => (pi * karg).sin() / (pi * karg),
        Kernel::QuadraticSpectral => {
            let a = 6.0 * pi * karg / 5.0;
            25.0 / (12.0 * (pi * pi) * (karg * karg)) * ((a).sin() / a - a.cos())
        }
    }
}

/// Weights of lags 1..=TAU of ivreg2 `m_omega` (index l-1): lag kernels use TAU = bw (so
/// Truncated and Tukey-Hamming weigh lag floor(bw) too, while Bartlett, Parzen and
/// Tukey-Hanning give lag bw weight 0), spectral kernels TAU = `span` (see
/// [`ivreg2_spectral_span`]). Zero weights are kept (m_omega skips them).
pub fn ivreg2_lag_weights(kernel: Kernel, bw: f64, span: usize) -> Vec<f64> {
    let tau_max = if kernel.is_spectral() { span as f64 } else { bw };
    if !tau_max.is_finite() || tau_max < 1.0 {
        return Vec::new();
    }
    (1..=tau_max.floor() as usize).map(|tau| kernel_weight(kernel, tau as f64, bw)).collect()
}

/// TAU of m_omega's spectral window, `T/tdelta - 1` with T = tmax - tmin + 1 in time units (the
/// loop runs tau = 1..=floor(TAU)).
pub fn ivreg2_spectral_span(tmin: f64, tmax: f64, tdelta: f64) -> usize {
    let tau = (tmax - tmin + 1.0) / tdelta - 1.0;
    if tau.is_finite() && tau >= 1.0 { tau.floor() as usize } else { 0 }
}

/// Bartlett weights 1 - l/(lag+1) of lags l = 1..=lag (index l-1): fixest, pyfixest, reghdfe
/// (reghdfe's `1 - j/bw` with lag = bw - 1).
pub fn bartlett_weights(lag: usize) -> Vec<f64> {
    (1..=lag).map(|l| 1.0 - l as f64 / (lag as f64 + 1.0)).collect()
}

/// floor(T^(1/4)): default lag of fixest panel Newey–West and Driscoll–Kraay
/// (`floor(n_time^(1/4))`) and of pyfixest (`int(floor(T**0.25))`), T distinct time values.
pub fn lag_quarter_root(t: usize) -> usize {
    (t as f64).powf(0.25).floor() as usize
}

/// reghdfe `reghdfe_dkraay_default_bw(T)` = floor(4*(T/100)^(2/9)) + 1 (bandwidth = lags + 1).
pub fn bw_reghdfe_dkraay(t: usize) -> usize {
    (4.0 * (t as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize + 1
}

/// The largest lag fixest actually uses: its C++ kernels cap the number of weights (lags
/// 0..=lag) at T-1 (`cpp_newey_west_panel`, `cpp_driscoll_kraay`; T distinct times) or N-1
/// (`cpp_newey_west`, N rows), so a requested lag >= T-1 uses lags up to T-2.
pub fn lag_fixest_clip(lag: usize, t: usize) -> usize {
    lag.min(t.saturating_sub(2))
}

/// Stata positions: (t - tmin)/tdelta, the units of ivreg2's `L#.` operators on tsset/xtset
/// time. Every value must lie on the tdelta grid.
pub(crate) fn pos_calendar(time: &[f64], tdelta: f64) -> Result<Vec<i64>, RfError> {
    if !(tdelta.is_finite() && tdelta > 0.0) {
        return Err(RfError::InvalidInput("time delta must be positive".into()));
    }
    let tmin = time.iter().copied().fold(f64::INFINITY, f64::min);
    time.iter()
        .map(|&t| {
            let p = (t - tmin) / tdelta;
            if p.is_finite() && p == p.round() && p < 9.0e18 {
                Ok(p as i64)
            } else {
                Err(RfError::InvalidInput("time values are not on the grid of the time delta".into()))
            }
        })
        .collect()
}

/// Failure of [`Panel::build`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PanelError {
    /// A (unit, position) pair occurs twice; callers raise their reference's message.
    Duplicate,
    Rf(RfError),
}

impl From<RfError> for PanelError {
    fn from(e: RfError) -> Self {
        PanelError::Rf(e)
    }
}

/// Rows grouped by unit, the rows of a unit in strictly increasing position.
pub(crate) struct Panel {
    /// Row indices sorted by (unit, position).
    order: Vec<u32>,
    /// Position of `order[k]`.
    pos: Vec<i64>,
    /// Offsets of the (non-empty) units in `order`: units + 1 entries, from 0 to n.
    starts: Vec<u32>,
    /// Largest position difference within a unit.
    span: u64,
}

impl Panel {
    /// Groups the rows by `unit` (row-order codes; `None` = one unit) and sorts each unit by
    /// `pos`. A repeated (unit, pos) is [`PanelError::Duplicate`].
    pub(crate) fn build(unit: Option<&[u32]>, pos: &[i64], pool: &Pool) -> Result<Panel, PanelError> {
        let n = pos.len();
        if unit.is_some_and(|u| u.len() != n) {
            return Err(RfError::Internal("Panel::build: unit and pos lengths differ".into()).into());
        }
        if u32::try_from(n).is_err() {
            return Err(RfError::InvalidInput("too many observations for a panel".into()).into());
        }
        let key = |i: usize| (unit.map_or(0, |u| u[i]), pos[i]);
        let sorted = pool.install(|| (1..n).into_par_iter().with_min_len(CHUNK).all(|i| key(i - 1) < key(i)));
        let order: Vec<u32> = if sorted {
            pool.install(|| (0..n as u32).into_par_iter().collect())
        } else {
            let mut o: Vec<u32> = pool.install(|| (0..n as u32).into_par_iter().collect());
            // The keys are unique unless there is a duplicate (an error), so the order is unique.
            pool.install(|| o.par_sort_unstable_by_key(|&i| key(i as usize)));
            if pool.install(|| o.par_windows(2).any(|w| key(w[0] as usize) == key(w[1] as usize))) {
                return Err(PanelError::Duplicate);
            }
            o
        };
        let spos: Vec<i64> = pool.install(|| order.par_iter().map(|&i| pos[i as usize]).collect());
        let mut starts: Vec<u32> = vec![0];
        if let Some(u) = unit {
            let inner: Vec<u32> = pool.install(|| {
                (1..n).into_par_iter().with_min_len(CHUNK).filter(|&k| u[order[k] as usize] != u[order[k - 1] as usize]).map(|k| k as u32).collect()
            });
            starts.extend(inner);
        }
        if n > 0 {
            starts.push(n as u32);
        }
        let span = starts.windows(2).map(|w| spos[w[1] as usize - 1].abs_diff(spos[w[0] as usize])).max().unwrap_or(0);
        Ok(Panel { order, pos: spos, starts, span })
    }

    pub(crate) fn n(&self) -> usize {
        self.order.len()
    }

    /// Number of non-empty units.
    #[cfg(test)]
    pub(crate) fn n_units(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    /// Row indices sorted by (unit, position).
    pub(crate) fn order(&self) -> &[u32] {
        &self.order
    }

    /// Positions, in [`Panel::order`].
    pub(crate) fn pos(&self) -> &[i64] {
        &self.pos
    }

    /// Unit offsets in [`Panel::order`] (units + 1 entries).
    #[cfg(test)]
    pub(crate) fn starts(&self) -> &[u32] {
        &self.starts
    }

    /// Largest position difference within a unit (no pair has a larger lag).
    #[cfg(test)]
    pub(crate) fn span(&self) -> u64 {
        self.span
    }

    /// pyfixest positions: the index of each row within its unit (row shifts after sorting by
    /// (unit, time); calendar gaps ignored). Build the panel on any strictly increasing time key
    /// (e.g. the time factor's codes) first, so duplicates are detected on (unit, time).
    pub(crate) fn within_unit_positions(mut self) -> Panel {
        let mut span = 0u64;
        for w in self.starts.windows(2) {
            let (a, b) = (w[0] as usize, w[1] as usize);
            for (k, p) in self.pos[a..b].iter_mut().enumerate() {
                *p = k as i64;
            }
            span = span.max((b - a - 1) as u64);
        }
        self.span = span;
        self
    }

    /// Lagged cross products of the rows `r` (positions in `order`) with their partners at
    /// lags 0..=eff, which lie at most `eff` positions earlier in `order`.
    fn lag_chunk(&self, r: Range<usize>, eff: usize, m: usize, row: &(impl Fn(usize, &mut [f64]) -> f64 + Sync)) -> Vec<f64> {
        let mm = m * m;
        let mut acc = vec![0.0; (eff + 1) * mm];
        let mut u = self.starts.partition_point(|&s| s as usize <= r.start) - 1;
        let lo = r.start.saturating_sub(eff).max(self.starts[u] as usize);
        let mut buf = vec![0.0; (r.end - lo) * m];
        let mut a = vec![0.0; m];
        for (k, x) in (lo..r.end).zip(buf.chunks_exact_mut(m)) {
            let s = row(self.order[k] as usize, &mut a);
            for (x, v) in x.iter_mut().zip(&a) {
                *x = s * v;
            }
        }
        let mut us = self.starts[u] as usize;
        let mut ue = self.starts[u + 1] as usize;
        for k in r {
            while k >= ue {
                u += 1;
                us = ue;
                ue = self.starts[u + 1] as usize;
            }
            let first = us.max(lo);
            let pk = self.pos[k];
            let xk = &buf[(k - lo) * m..(k - lo + 1) * m];
            let mut j = k;
            loop {
                // Positions increase strictly within the unit: d >= k - j.
                let d = pk.abs_diff(self.pos[j]);
                if d > eff as u64 {
                    break;
                }
                let xj = &buf[(j - lo) * m..(j - lo + 1) * m];
                let c = &mut acc[d as usize * mm..(d as usize + 1) * mm];
                for (crow, &xr) in c.chunks_exact_mut(m).zip(xk) {
                    for (cv, &xc) in crow.iter_mut().zip(xj) {
                        *cv += xr * xc;
                    }
                }
                if j == first {
                    break;
                }
                j -= 1;
            }
        }
        acc
    }
}

/// Bytes of chunk accumulators of [`lag_cross`] alive at once.
const LAG_BUDGET: usize = 256 << 20;

/// C_l = sum over pairs (t, s) of the same unit with pos_t - pos_s = l of (s_t a_t)(s_s a_s)'
/// (m x m, row-major), for l = 0..=L with L = min(max_lag, [`Panel::span`]): no pair has a
/// larger lag, so the higher C_l are zero and omitted.
///
/// Cost O(n * (L+1) * m^2) at most, memory O(n*m) for the chunk buffers plus (L+1) m^2 per live
/// chunk. Chunks of the sorted rows (smaller when L is large) are combined in order.
pub(crate) fn lag_cross(p: &Panel, max_lag: usize, m: usize, pool: &Pool, row: impl Fn(usize, &mut [f64]) -> f64 + Sync) -> Result<Vec<Vec<f64>>, RfError> {
    let n = p.n();
    let eff = max_lag.min(usize::try_from(p.span).unwrap_or(usize::MAX));
    let mm = m * m;
    let per = (eff + 1)
        .checked_mul(mm)
        .ok_or_else(|| RfError::InvalidInput("too many lags".into()))?;
    let mut total = par::alloc_vec(per, 0.0)?;
    if n > 0 && m > 0 {
        let chunk = (CHUNK / (1 + eff / 16)).max(256);
        let nch = par::n_chunks(n, chunk);
        let batch = (LAG_BUDGET / (8 * per).max(1)).max(1);
        let mut c0 = 0;
        while c0 < nch {
            let c1 = (c0 + batch).min(nch);
            let parts: Vec<Vec<f64>> = pool.install(|| (c0..c1).into_par_iter().map(|c| p.lag_chunk(par::chunk_range(c, n, chunk), eff, m, &row)).collect());
            for part in parts {
                for (t, v) in total.iter_mut().zip(&part) {
                    *t += v;
                }
            }
            c0 = c1;
        }
    }
    Ok(total.chunks_exact(mm.max(1)).take(eff + 1).map(<[f64]>::to_vec).collect())
}

/// [`lag_cross`] on dense slot sums `h` (T x m, row-major; slot t at position `pos[t]`, strictly
/// increasing), with every multiplier 1: Driscoll–Kraay and other time-sum meats. Returns
/// C_0..=C_L, L = min(max_lag, pos[T-1] - pos[0]).
pub(crate) fn lag_cross_dense(h: &[f64], pos: &[i64], m: usize, max_lag: usize) -> Vec<Vec<f64>> {
    let t = pos.len();
    debug_assert_eq!(h.len(), t * m);
    debug_assert!(pos.windows(2).all(|w| w[0] < w[1]));
    let span = if t > 0 { pos[t - 1].abs_diff(pos[0]) } else { 0 };
    let eff = max_lag.min(usize::try_from(span).unwrap_or(usize::MAX));
    let mm = m * m;
    let mut c = vec![vec![0.0; mm]; eff + 1];
    for k in 0..t {
        let xk = &h[k * m..(k + 1) * m];
        for j in (0..=k).rev() {
            let d = pos[k].abs_diff(pos[j]);
            if d > eff as u64 {
                break;
            }
            let xj = &h[j * m..(j + 1) * m];
            for (crow, &xr) in c[d as usize].chunks_exact_mut(m.max(1)).zip(xk) {
                for (cv, &xc) in crow.iter_mut().zip(xj) {
                    *cv += xr * xc;
                }
            }
        }
    }
    c
}

/// C_0 + sum_l w[l-1] (C_l + C_l') over the lags present in both `c` and `w` (m x m).
pub(crate) fn hac_combine(c: &[Vec<f64>], w: &[f64], m: usize) -> Vec<f64> {
    let mut out = c.first().cloned().unwrap_or_else(|| vec![0.0; m * m]);
    for (cl, &wl) in c.iter().skip(1).zip(w) {
        if wl == 0.0 {
            continue;
        }
        for i in 0..m {
            for j in 0..m {
                out[i * m + j] += wl * (cl[i * m + j] + cl[j * m + i]);
            }
        }
    }
    out
}

/// Slots per task of the grouped path of [`slot_sums`].
const SLOT_CHUNK: usize = 1024;

/// Per-slot score sums H (t x m, row-major): H_c = sum of s_i a_i over the rows with code c
/// (`codes` in row order, values < t). Streamed over the rows when the sums are small
/// ([`vcov::stream_score_sums`]), otherwise summed slot by slot over the rows grouped by code.
pub(crate) fn slot_sums(codes: &[u32], t: usize, m: usize, pool: &Pool, row: impl Fn(usize, &mut [f64]) -> f64 + Sync) -> Result<Vec<f64>, RfError> {
    if m == 0 || t == 0 {
        return Ok(Vec::new());
    }
    if let Some(s) = vcov::stream_score_sums(codes, t, m, pool, &row) {
        return Ok(s);
    }
    let (off, ids) = structure::scatter_by_code(codes, t, pool)?;
    let mut out = par::alloc_vec(t * m, 0.0)?;
    pool.install(|| {
        out.par_chunks_mut(m * SLOT_CHUNK).enumerate().for_each(|(c, ch)| {
            let mut a = vec![0.0; m];
            for (q, slot) in ch.chunks_exact_mut(m).enumerate() {
                let g = c * SLOT_CHUNK + q;
                for &i in &ids[off[g] as usize..off[g + 1] as usize] {
                    let s = row(i as usize, &mut a);
                    for (h, v) in slot.iter_mut().zip(&a) {
                        *h += s * v;
                    }
                }
            }
        })
    });
    Ok(out)
}

/// fixest 0.14.2 default lag of the time-series Newey–West VCOV: `floor(sandwich::bwNeweyWest(x,
/// weights = 1))` on the scores in time order (`vcov_newey_west_internal`). `order` lists the rows
/// in time order; `drop_col` is the intercept column, which fixest drops when there are other
/// columns (pass `None` otherwise). See [`bw_newey_west`] for the procedure.
pub(crate) fn lag_fixest_ts(order: &[u32], m: usize, drop_col: Option<usize>, pool: &Pool, row: impl Fn(usize, &mut [f64]) -> f64 + Sync) -> Result<usize, RfError> {
    Ok(bw_newey_west(order, m, drop_col, pool, &row)?.floor() as usize)
}

/// sandwich 3.1.3 `bwNeweyWest(x, kernel = "Bartlett", weights = 1, prewhite = 1, ar.method =
/// "ols")` on the n x k score matrix u (rows `order`, column `drop_col` removed):
/// 1. VAR(1) prewhitening by `ar.ols(u, order.max = 1, aic = FALSE, demean = FALSE)`: columns
///    scaled by their sd, A = (Y'X)(X'X)^-1 on consecutive rows, residuals rescaled;
/// 2. hw_t = sum of the residual row t (weights 1), t = 2..n; n' = n - 1;
/// 3. sigma_j = sum_t hw_t hw_{t+j} / n', j = 0..=floor(3 (n/100)^(2/9));
/// 4. 1.1447 ((s1/s0)^2)^(1/3) n^(1/3) with s0 = sigma_0 + 2 sum sigma_j, s1 = 2 sum j sigma_j.
pub(crate) fn bw_newey_west(order: &[u32], m: usize, drop_col: Option<usize>, pool: &Pool, row: &(impl Fn(usize, &mut [f64]) -> f64 + Sync)) -> Result<f64, RfError> {
    const FAILED: &str = "VAR(1) prewhitening of estimating functions failed";
    let n = order.len();
    if drop_col.is_some_and(|d| d >= m) {
        return Err(RfError::Internal("bw_newey_west: drop_col out of range".into()));
    }
    let k = m - usize::from(drop_col.is_some());
    if k == 0 || n < 3 {
        return Err(RfError::InvalidInput(FAILED.into()));
    }
    // Row t of u (time order).
    let urow = |t: usize, a: &mut [f64], out: &mut [f64]| {
        let s = row(order[t] as usize, a);
        let mut q = 0;
        for (j, &v) in a.iter().enumerate() {
            if Some(j) != drop_col {
                out[q] = s * v;
                q += 1;
            }
        }
    };
    let nf = n as f64;
    // Column sd (R `var`: mean, then centered sum of squares / (n-1)).
    let sums = par::sum_chunks(pool, n, CHUNK, k, |r, acc| {
        let (mut a, mut x) = (vec![0.0; m], vec![0.0; k]);
        for t in r {
            urow(t, &mut a, &mut x);
            for (s, v) in acc.iter_mut().zip(&x) {
                *s += v;
            }
        }
    });
    let mean0: Vec<f64> = sums.iter().map(|s| s / nf).collect();
    let dev = par::sum_chunks(pool, n, CHUNK, 2 * k, |r, acc| {
        let (mut a, mut x) = (vec![0.0; m], vec![0.0; k]);
        for t in r {
            urow(t, &mut a, &mut x);
            for j in 0..k {
                let d = x[j] - mean0[j];
                acc[j] += d;
                acc[k + j] += d * d;
            }
        }
    });
    let sc: Vec<f64> = (0..k)
        .map(|j| {
            let s = ((dev[k + j] - dev[j] * dev[j] / nf) / (nf - 1.0)).max(0.0).sqrt();
            if s == 0.0 { 1.0 } else { s }
        })
        .collect();
    // X'X and Y'X of the scaled series on consecutive rows (X = row t-1, Y = row t).
    let cross = par::sum_chunks(pool, n - 1, CHUNK, 2 * k * k, |r, acc| {
        let mut a = vec![0.0; m];
        let (mut prev, mut cur) = (vec![0.0; k], vec![0.0; k]);
        urow(r.start, &mut a, &mut prev);
        prev.iter_mut().zip(&sc).for_each(|(x, s)| *x /= s);
        for t in r.start + 1..r.end + 1 {
            urow(t, &mut a, &mut cur);
            cur.iter_mut().zip(&sc).for_each(|(x, s)| *x /= s);
            for i in 0..k {
                for j in 0..k {
                    acc[i * k + j] += prev[i] * prev[j];
                    acc[k * k + i * k + j] += cur[i] * prev[j];
                }
            }
            std::mem::swap(&mut prev, &mut cur);
        }
    });
    let (xx, yx) = cross.split_at(k * k);
    if !full_rank_qr(xx, k, 1e-7) {
        return Err(RfError::InvalidInput(FAILED.into()));
    }
    let p = lu_inverse(xx, k).ok_or_else(|| RfError::InvalidInput(FAILED.into()))?;
    let mut ar = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            ar[i * k + j] = (0..k).map(|l| yx[i * k + l] * p[l * k + j]).sum();
        }
    }
    // hw_t: sums of the rescaled VAR residuals, t = 1..n-1 (stored at t-1).
    let np = n - 1;
    let mut hw = par::alloc_vec(np, 0.0)?;
    pool.install(|| {
        hw.par_chunks_mut(CHUNK).enumerate().for_each(|(c, out)| {
            let mut a = vec![0.0; m];
            let (mut prev, mut cur) = (vec![0.0; k], vec![0.0; k]);
            let t0 = c * CHUNK + 1;
            urow(t0 - 1, &mut a, &mut prev);
            prev.iter_mut().zip(&sc).for_each(|(x, s)| *x /= s);
            for (q, h) in out.iter_mut().enumerate() {
                urow(t0 + q, &mut a, &mut cur);
                cur.iter_mut().zip(&sc).for_each(|(x, s)| *x /= s);
                let mut sum = 0.0;
                for j in 0..k {
                    let fit: f64 = (0..k).map(|l| prev[l] * ar[j * k + l]).sum();
                    sum += (cur[j] - fit) * sc[j];
                }
                *h = sum;
                std::mem::swap(&mut prev, &mut cur);
            }
        })
    });
    let mlag = (3.0 * (nf / 100.0).powf(2.0 / 9.0)).floor() as usize;
    let sig = par::sum_chunks(pool, np, CHUNK, mlag + 1, |r, acc| {
        for t in r {
            for (j, s) in acc.iter_mut().enumerate() {
                if t + j < np {
                    *s += hw[t] * hw[t + j];
                }
            }
        }
    });
    let npf = np as f64;
    let sigma: Vec<f64> = sig.iter().map(|s| s / npf).collect();
    let s0 = sigma[0] + 2.0 * sigma[1..].iter().sum::<f64>();
    let s1 = 2.0 * sigma[1..].iter().enumerate().map(|(j, s)| (j + 1) as f64 * s).sum::<f64>();
    let qrate = 1.0 / 3.0;
    let ratio = s1 / s0;
    let rval = 1.1447 * (ratio * ratio).powf(qrate) * nf.powf(qrate);
    if rval.is_finite() && rval >= 0.0 {
        Ok(rval)
    } else {
        Err(RfError::InvalidInput("the default Newey-West lag cannot be computed (the scores have no variance)".into()))
    }
}

/// R `qr(a)$rank == k` (LINPACK dqrdc2, tolerance `tol`): no column of the k x k matrix `a`
/// keeps less than `tol` of its norm after removing its projection on the previous kept columns.
fn full_rank_qr(a: &[f64], k: usize, tol: f64) -> bool {
    let cols: Vec<Vec<f64>> = (0..k).map(|j| (0..k).map(|i| a[i * k + j]).collect()).collect();
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(k);
    for c in cols {
        let norm0 = c.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut v = c;
        for b in &basis {
            let d: f64 = v.iter().zip(b).map(|(x, y)| x * y).sum();
            v.iter_mut().zip(b).for_each(|(x, y)| *x -= d * y);
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !(norm >= tol * norm0) || norm == 0.0 {
            return false;
        }
        v.iter_mut().for_each(|x| *x /= norm);
        basis.push(v);
    }
    true
}

/// Inverse of a k x k matrix by LU with partial pivoting (LAPACK `dgesv` on the identity, as R
/// `solve`); `None` when singular.
fn lu_inverse(a: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..k).collect();
    for c in 0..k {
        let p = (c..k).max_by(|&x, &y| lu[x * k + c].abs().total_cmp(&lu[y * k + c].abs()))?;
        if lu[p * k + c] == 0.0 {
            return None;
        }
        if p != c {
            for j in 0..k {
                lu.swap(p * k + j, c * k + j);
            }
            perm.swap(p, c);
        }
        for r in c + 1..k {
            let f = lu[r * k + c] / lu[c * k + c];
            lu[r * k + c] = f;
            for j in c + 1..k {
                lu[r * k + j] -= f * lu[c * k + j];
            }
        }
    }
    let mut inv = vec![0.0; k * k];
    for col in 0..k {
        let mut x: Vec<f64> = (0..k).map(|i| if perm[i] == col { 1.0 } else { 0.0 }).collect();
        for i in 0..k {
            for j in 0..i {
                x[i] -= lu[i * k + j] * x[j];
            }
        }
        for i in (0..k).rev() {
            for j in i + 1..k {
                x[i] -= lu[i * k + j] * x[j];
            }
            x[i] /= lu[i * k + i];
        }
        for i in 0..k {
            inv[i * k + col] = x[i];
        }
    }
    Some(inv)
}

/// ivreg2 / ivreghdfe `s_abw` (Newey–West 1994 automatic bandwidth) for Bartlett, Parzen and
/// Quadratic Spectral; `None` for other kernels (ivreg2 then prints "Error: kernel X not
/// compatible with bw(auto)" and uses bw = 1).
///
/// `f` is the series f_t = u_t * sum_j Z_tj (2SLS residual times the row sum of the excluded and
/// included instruments, constant excluded, no weights) in time order with calendar positions
/// `pos` (strictly increasing); `nobs` = N and `tobs` = time span tmax - tmin + 1 in time units
/// (not divided by tdelta). Returns the bandwidth optlag + 1 rounded as ivreg2 passes it through
/// `strofreal()` (Stata `%9.0g`: 7 significant digits).
pub(crate) fn bw_ivreg2_auto(kernel: Kernel, f: &[f64], pos: &[i64], nobs: f64, tobs: f64, pool: &Pool) -> Option<f64> {
    let (expo, q, cgamma) = match kernel {
        Kernel::Bartlett => (2.0 / 9.0, 1, 1.1447),
        Kernel::Parzen => (4.0 / 25.0, 2, 2.6614),
        Kernel::QuadraticSpectral => (2.0 / 25.0, 2, 1.3221),
        _ => return None,
    };
    debug_assert_eq!(f.len(), pos.len());
    let mstar = (20.0 * (tobs / 100.0).powf(expo)).trunc().max(0.0);
    let ms = mstar as usize;
    let n = f.len();
    let sums = par::sum_chunks(pool, n, CHUNK, ms + 1, |r, acc| {
        for t in r {
            for s in (0..=t).rev() {
                let d = pos[t].abs_diff(pos[s]);
                if d > ms as u64 {
                    break;
                }
                acc[d as usize] += f[t] * f[s];
            }
        }
    });
    let sigmahat: Vec<f64> = sums.iter().map(|s| s / nobs).collect();
    let mut shatq = 0.0;
    let mut shat0 = sigmahat[0];
    for (j, s) in sigmahat.iter().enumerate().skip(1) {
        let jq = if q == 1 { j as f64 } else { (j * j) as f64 };
        shatq += 2.0 * s * jq;
        shat0 += 2.0 * s;
    }
    let expon = 1.0 / f64::from(2 * q + 1);
    let r = shatq / shat0;
    let gammahat = cgamma * (r * r).powf(expon);
    let m = gammahat * tobs.powf(expon);
    // Mata: missing (here non-finite) is larger than every number in min().
    let optlag = if !m.is_finite() {
        mstar
    } else if kernel == Kernel::QuadraticSpectral {
        m.min(mstar)
    } else {
        m.trunc().min(mstar)
    };
    Some(stata_strofreal(optlag + 1.0))
}

/// Value of Mata `strofreal(x)` read back as a number: Stata's default `%9.0g` keeps 7
/// significant digits for 1 <= |x| < 1e7 (the range of bandwidths).
fn stata_strofreal(x: f64) -> f64 {
    format!("{x:.6e}").parse().unwrap_or(x)
}

#[cfg(test)]
pub(crate) mod testdata {
    //! Fixture access for the hac/conley unit tests (`tests/data/hac`, generated by the scripts
    //! committed there).
    use std::collections::HashMap;
    use std::path::PathBuf;

    pub(crate) fn path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/hac").join(name)
    }

    /// A CSV file with a header line (no quoting).
    pub(crate) struct Csv {
        pub header: Vec<String>,
        pub rows: Vec<Vec<String>>,
    }

    impl Csv {
        pub(crate) fn read(name: &str) -> Csv {
            Csv::read_sep(name, ',')
        }

        pub(crate) fn read_sep(name: &str, sep: char) -> Csv {
            let text = std::fs::read_to_string(path(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
            let mut lines = text.lines();
            let header = lines.next().expect("header").split(sep).map(|s| s.trim().to_string()).collect();
            let rows = lines.filter(|l| !l.is_empty()).map(|l| l.split(sep).map(|s| s.to_string()).collect()).collect();
            Csv { header, rows }
        }

        pub(crate) fn idx(&self, col: &str) -> usize {
            self.header.iter().position(|h| h == col).unwrap_or_else(|| panic!("no column {col}"))
        }

        pub(crate) fn col(&self, col: &str) -> Vec<f64> {
            let j = self.idx(col);
            self.rows.iter().map(|r| num(&r[j])).collect()
        }
    }

    pub(crate) fn num(s: &str) -> f64 {
        let s = s.trim();
        s.parse().unwrap_or_else(|_| panic!("not a number: {s:?}"))
    }

    /// Long-format references `case,name,i,j,value`.
    pub(crate) struct Refs(HashMap<(String, String), Vec<(usize, usize, f64)>>);

    impl Refs {
        pub(crate) fn read(name: &str) -> Refs {
            let csv = Csv::read(name);
            let mut map: HashMap<(String, String), Vec<(usize, usize, f64)>> = HashMap::new();
            for r in &csv.rows {
                map.entry((r[0].clone(), r[1].clone())).or_default().push((r[2].trim().parse().unwrap(), r[3].trim().parse().unwrap(), num(&r[4])));
            }
            Refs(map)
        }

        pub(crate) fn has(&self, case: &str, name: &str) -> bool {
            self.0.contains_key(&(case.to_string(), name.to_string()))
        }

        pub(crate) fn scalar(&self, case: &str, name: &str) -> f64 {
            self.0.get(&(case.to_string(), name.to_string())).unwrap_or_else(|| panic!("missing {case}/{name}"))[0].2
        }

        /// (rows, cols, row-major values).
        pub(crate) fn mat(&self, case: &str, name: &str) -> (usize, usize, Vec<f64>) {
            let v = self.0.get(&(case.to_string(), name.to_string())).unwrap_or_else(|| panic!("missing {case}/{name}"));
            let r = v.iter().map(|e| e.0).max().unwrap() + 1;
            let c = v.iter().map(|e| e.1).max().unwrap() + 1;
            let mut out = vec![f64::NAN; r * c];
            for &(i, j, x) in v {
                out[i * c + j] = x;
            }
            (r, c, out)
        }
    }

    /// Weighted least squares by the normal equations: (beta, residuals, (X'WX)^-1) for the
    /// row-major n x k design `x`.
    pub(crate) fn ols(x: &[f64], y: &[f64], w: Option<&[f64]>, k: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = y.len();
        let mut xtx = vec![0.0; k * k];
        let mut xty = vec![0.0; k];
        for i in 0..n {
            let wi = w.map_or(1.0, |w| w[i]);
            let xi = &x[i * k..(i + 1) * k];
            for a in 0..k {
                xty[a] += wi * xi[a] * y[i];
                for b in 0..k {
                    xtx[a * k + b] += wi * xi[a] * xi[b];
                }
            }
        }
        let (inv, dropped) = crate::vcov::invsym(&xtx, k, 1e-13);
        assert!(!dropped.iter().any(|d| *d));
        let beta: Vec<f64> = (0..k).map(|a| (0..k).map(|b| inv[a * k + b] * xty[b]).sum()).collect();
        let e = (0..n).map(|i| y[i] - (0..k).map(|a| x[i * k + a] * beta[a]).sum::<f64>()).collect();
        (beta, e, inv)
    }

    /// Max over entries of |a - b| / sqrt(|b_ii b_jj|) (k x k): the error relative to the scale of
    /// the variances involved.
    pub(crate) fn rel_err_v(a: &[f64], b: &[f64], k: usize) -> f64 {
        let mut worst = 0.0f64;
        for i in 0..k {
            for j in 0..k {
                let s = (b[i * k + i] * b[j * k + j]).abs().sqrt();
                worst = worst.max((a[i * k + j] - b[i * k + j]).abs() / s);
            }
        }
        worst
    }
}

#[cfg(test)]
mod tests {
    use super::testdata::{Csv, Refs, num, ols, rel_err_v};
    use super::*;
    use crate::rng::SplitMix64;

    fn pool_run<R: Send>(threads: usize, f: impl FnOnce(&Pool) -> R) -> R {
        par::with_pool(threads, f).unwrap()
    }

    /// fixest positions from raw time values: dense rank (0-based, `to_integer(time, sorted =
    /// TRUE)`) and the number of distinct values T. Hosts pass the time factor's codes instead.
    fn pos_dense_rank(time: &[f64], pool: &Pool) -> Result<(Vec<i64>, usize), RfError> {
        let c = crate::factor::factorize(&crate::api::FactorValues::Num(time), pool)?;
        let pos = pool.install(|| c.codes.par_iter().map(|&x| i64::from(x)).collect());
        Ok((pos, c.n as usize))
    }

    #[test]
    fn kernel_weights_match_mata_calckw() {
        let csv = Csv::read("stata_kw.csv");
        let mut worst = 0.0f64;
        for r in &csv.rows {
            let k = Kernel::parse(&r[0]).unwrap();
            assert_eq!(k.name(), r[0]);
            let (bw, tau, kw) = (num(&r[1]), num(&r[2]), num(&r[3]));
            let ours = kernel_weight(k, tau, bw);
            let err = (ours - kw).abs() / kw.abs().max(1e-300);
            assert!((ours - kw).abs() <= 1e-15 * kw.abs().max(1.0) + 1e-17, "{} bw={bw} tau={tau}: {ours} vs {kw}", r[0]);
            if kw != 0.0 {
                worst = worst.max(err);
            }
        }
        assert_eq!(csv.rows.len(), 8 * 8 * 15);
        eprintln!("m_calckw: max rel diff {worst:e}");
    }

    #[test]
    fn kernel_spellings_match_s_vkernel() {
        let csv = Csv::read_sep("stata_kernels.csv", '|');
        for r in &csv.rows {
            let got = Kernel::parse(&r[0]).map(Kernel::name).unwrap_or("");
            assert_eq!(got, r[1], "spelling {:?}", r[0]);
        }
    }

    #[test]
    fn lag_windows() {
        // Lag kernels: lags 1..=floor(bw); Bartlett/Parzen/Tukey-Hanning zero at tau = bw.
        let w = ivreg2_lag_weights(Kernel::Bartlett, 3.0, 99);
        assert_eq!(w, vec![1.0 - 1.0 / 3.0, 1.0 - 2.0 / 3.0, 0.0]);
        assert_eq!(ivreg2_lag_weights(Kernel::Truncated, 2.5, 99), vec![1.0, 1.0]);
        assert_eq!(*ivreg2_lag_weights(Kernel::Parzen, 4.0, 99).last().unwrap(), 0.0);
        assert!((ivreg2_lag_weights(Kernel::TukeyHamming, 2.0, 99)[1] - 0.08).abs() < 1e-15);
        assert_eq!(ivreg2_lag_weights(Kernel::QuadraticSpectral, 1.7, 7).len(), 7);
        assert_eq!(ivreg2_spectral_span(1980.0, 2010.0, 2.0), 14); // 31/2 - 1 = 14.5
        assert_eq!(ivreg2_spectral_span(1.0, 12.0, 1.0), 11);
        assert_eq!(bartlett_weights(2), vec![1.0 - 1.0 / 3.0, 1.0 - 2.0 / 3.0]);
        assert_eq!(lag_quarter_root(12), 1);
        assert_eq!(lag_quarter_root(16), 2);
        assert_eq!(lag_quarter_root(81), 3);
        assert_eq!(bw_reghdfe_dkraay(12), 3); // floor(4 * 0.12^(2/9)) + 1 = floor(2.497) + 1
        assert_eq!(bw_reghdfe_dkraay(100), 5);
        assert_eq!(lag_fixest_clip(20, 12), 10);
        assert_eq!(lag_fixest_clip(3, 12), 3);
    }

    /// Random design rows and multipliers keyed by row.
    fn rand_row(seed: u64, m: usize) -> impl Fn(usize, &mut [f64]) -> f64 + Sync {
        move |i, a| {
            let mut g = SplitMix64::stream(seed, i as u64);
            for x in a.iter_mut().take(m) {
                *x = g.next_f64() * 2.0 - 1.0;
            }
            g.next_f64() - 0.3
        }
    }

    fn brute_lag_cross(unit: Option<&[u32]>, pos: &[i64], max_lag: usize, m: usize, row: &impl Fn(usize, &mut [f64]) -> f64) -> Vec<Vec<f64>> {
        let n = pos.len();
        let x: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut a = vec![0.0; m];
                let s = row(i, &mut a);
                a.iter().map(|v| s * v).collect()
            })
            .collect();
        let mut c = vec![vec![0.0; m * m]; max_lag + 1];
        for t in 0..n {
            for s in 0..n {
                let same = unit.is_none_or(|u| u[t] == u[s]);
                let d = pos[t] - pos[s];
                if same && d >= 0 && d as usize <= max_lag {
                    for i in 0..m {
                        for j in 0..m {
                            c[d as usize][i * m + j] += x[t][i] * x[s][j];
                        }
                    }
                }
            }
        }
        c
    }

    /// Unbalanced panel with gaps and a scrambled row order: (unit, pos).
    fn random_panel(seed: u64, units: u32, periods: i64, p_keep: f64) -> (Vec<u32>, Vec<i64>) {
        let mut g = SplitMix64::new(seed);
        let mut rows = Vec::new();
        for u in 0..units {
            for t in 0..periods {
                if g.next_f64() < p_keep {
                    rows.push((u * 7 % units, t * 3 - 5));
                }
            }
        }
        for i in (1..rows.len()).rev() {
            let j = (g.next_u64() % (i as u64 + 1)) as usize;
            rows.swap(i, j);
        }
        rows.into_iter().unzip()
    }

    fn assert_cross_eq(got: &[Vec<f64>], want: &[Vec<f64>]) {
        let scale = want[0].iter().fold(1.0f64, |a, v| a.max(v.abs()));
        for (l, w) in want.iter().enumerate() {
            let g = got.get(l).cloned().unwrap_or_else(|| vec![0.0; w.len()]);
            for (a, b) in g.iter().zip(w) {
                assert!((a - b).abs() <= 1e-12 * scale, "lag {l}: {a} vs {b}");
            }
        }
        for g in got.iter().skip(want.len()) {
            assert!(g.iter().all(|v| *v == 0.0));
        }
    }

    #[test]
    fn lag_cross_matches_brute_force() {
        let m = 3;
        for (seed, units, periods, keep) in [(1u64, 30u32, 15i64, 0.7), (2, 5, 60, 0.5), (3, 200, 4, 0.9), (4, 1, 300, 0.6)] {
            let (unit, pos) = random_panel(seed, units, periods, keep);
            let row = rand_row(seed, m);
            let single = units == 1;
            let p = pool_run(4, |pool| Panel::build(if single { None } else { Some(&unit) }, &pos, pool)).unwrap();
            assert_eq!(p.n(), pos.len());
            let span = p.span() as usize;
            // Small lags, lags beyond the span (every lag within a unit: spectral windows).
            for max_lag in [0, 3, 6, span, span + 40] {
                let want = brute_lag_cross(if single { None } else { Some(&unit) }, &pos, max_lag, m, &row);
                let got = pool_run(4, |pool| lag_cross(&p, max_lag, m, pool, &row)).unwrap();
                assert_eq!(got.len(), max_lag.min(span) + 1);
                assert_cross_eq(&got, &want);
            }
        }
    }

    #[test]
    fn lag_cross_dense_matches_brute_force() {
        let m = 2;
        let pos: Vec<i64> = vec![0, 1, 2, 5, 6, 9, 10, 11, 20];
        let row = rand_row(9, m);
        let h: Vec<f64> = (0..pos.len())
            .flat_map(|i| {
                let mut a = vec![0.0; m];
                let s = row(i, &mut a);
                a.into_iter().map(move |v| s * v)
            })
            .collect();
        for max_lag in [0, 1, 4, 20, 30] {
            let want = brute_lag_cross(None, &pos, max_lag, m, &row);
            assert_cross_eq(&lag_cross_dense(&h, &pos, m, max_lag), &want);
        }
    }

    #[test]
    fn panel_duplicates_detected() {
        let unit = vec![0u32, 1, 0, 1, 0];
        let pos = vec![3i64, 3, 1, 4, 3];
        assert_eq!(pool_run(2, |pool| Panel::build(Some(&unit), &pos, pool).err()), Some(PanelError::Duplicate));
        assert_eq!(pool_run(2, |pool| Panel::build(None, &[1, 2, 1], pool).err()), Some(PanelError::Duplicate));
        // Same position in different units is fine; sorted input takes the fast path.
        let p = pool_run(2, |pool| Panel::build(Some(&[0, 0, 1, 1]), &[1, 2, 1, 2], pool)).unwrap();
        assert_eq!((p.n_units(), p.order(), p.starts()), (2, &[0u32, 1, 2, 3][..], &[0u32, 2, 4][..]));
        let p = pool_run(2, |pool| Panel::build(Some(&unit[..4]), &pos[..4], pool)).unwrap();
        assert_eq!(p.order(), &[2, 0, 1, 3]);
        assert_eq!(p.pos(), &[1, 3, 3, 4]);
        assert_eq!(p.span(), 2);
        let q = p.within_unit_positions();
        assert_eq!((q.pos(), q.span()), (&[0i64, 1, 0, 1][..], 1));
    }

    #[test]
    fn deterministic_across_threads() {
        let m = 3;
        let (unit, pos) = random_panel(11, 4000, 12, 0.85);
        let row = rand_row(11, m);
        let run = |threads: usize, max_lag: usize| {
            pool_run(threads, |pool| {
                let p = Panel::build(Some(&unit), &pos, pool).unwrap();
                let c = lag_cross(&p, max_lag, m, pool, &row).unwrap();
                let codes: Vec<u32> = pos.iter().map(|&t| ((t + 5) / 3) as u32).collect();
                let small = slot_sums(&codes, 12, m, pool, &row).unwrap();
                let big_codes: Vec<u32> = (0..pos.len()).map(|i| i as u32).collect();
                let big = slot_sums(&big_codes, 300_000, m, pool, &row).unwrap();
                (c, small, big)
            })
        };
        for max_lag in [2, 40] {
            let base = run(1, max_lag);
            for t in [4, 16] {
                let other = run(t, max_lag);
                let bits = |x: &(Vec<Vec<f64>>, Vec<f64>, Vec<f64>)| -> Vec<u64> {
                    x.0.iter().flatten().chain(&x.1).chain(&x.2).map(|v| v.to_bits()).collect()
                };
                assert_eq!(bits(&base), bits(&other), "threads {t}, max_lag {max_lag}");
            }
        }
    }

    #[test]
    fn slot_sums_both_paths() {
        let m = 2;
        let n = 70_000;
        let row = rand_row(5, m);
        for g in [7usize, 300_000] {
            let codes: Vec<u32> = (0..n).map(|i| ((i * 7919) % g.min(n)) as u32).collect();
            let got = pool_run(4, |pool| slot_sums(&codes, g, m, pool, &row)).unwrap();
            let mut want = vec![0.0; g * m];
            let mut a = vec![0.0; m];
            for i in 0..n {
                let s = row(i, &mut a);
                for j in 0..m {
                    want[codes[i] as usize * m + j] += s * a[j];
                }
            }
            for (x, y) in got.iter().zip(&want) {
                assert!((x - y).abs() <= 1e-11 * (1.0 + y.abs()));
            }
        }
    }

    /// Row-major n x 3 design [1, x1, x2] (intercept first, as fixest/pyfixest) or [x1, x2, 1]
    /// (Stata).
    fn design(csv: &Csv, xs: &[&str], cons_last: bool) -> Vec<f64> {
        let cols: Vec<Vec<f64>> = xs.iter().map(|c| csv.col(c)).collect();
        let n = cols[0].len();
        let mut out = Vec::with_capacity(n * (xs.len() + 1));
        for i in 0..n {
            if !cons_last {
                out.push(1.0);
            }
            out.extend(cols.iter().map(|c| c[i]));
            if cons_last {
                out.push(1.0);
            }
        }
        out
    }

    struct Fit {
        k: usize,
        x: Vec<f64>,
        e: Vec<f64>,
        w: Option<Vec<f64>>,
        bread: Vec<f64>,
        beta: Vec<f64>,
    }

    impl Fit {
        fn new(csv: &Csv, y: &str, xs: &[&str], w: Option<Vec<f64>>, cons_last: bool) -> Fit {
            let k = xs.len() + 1;
            let x = design(csv, xs, cons_last);
            let (beta, e, bread) = ols(&x, &csv.col(y), w.as_deref(), k);
            Fit { k, x, e, w, bread, beta }
        }

        /// Score multiplier w_i e_i (times `wf`) with design row x_i.
        fn row(&self, wf: f64) -> impl Fn(usize, &mut [f64]) -> f64 + Sync + '_ {
            move |i, a| {
                a.copy_from_slice(&self.x[i * self.k..(i + 1) * self.k]);
                self.e[i] * self.w.as_ref().map_or(1.0, |w| w[i]) * wf
            }
        }
    }

    fn check_v(what: &str, refs: &Refs, case: &str, fit: &Fit, meat: &[f64], scale: f64) -> f64 {
        let (_, _, b) = refs.mat(case, "b");
        for (x, y) in fit.beta.iter().zip(&b) {
            assert!((x - y).abs() <= 1e-10 * (1.0 + y.abs()), "{what} {case}: beta {x} vs {y}");
        }
        let v = vcov::sandwich(&fit.bread, meat, fit.k, scale);
        let (_, _, want) = refs.mat(case, "V");
        let err = rel_err_v(&v, &want, fit.k);
        assert!(err <= 1e-10, "{what} {case}: V rel err {err:e}\n{v:?}\n{want:?}");
        err
    }

    #[test]
    fn fixest_newey_west_and_driscoll_kraay() {
        let refs = Refs::read("fixest_ref.csv");
        let panel = Csv::read("panel.csv");
        let ts = Csv::read("ts.csv");
        let mut worst = 0.0f64;
        pool_run(4, |pool| {
            let id_codes: Vec<u32> = panel.col("id").iter().map(|&v| v as u32).collect();
            for (case, data, lag, weighted, kind) in [
                ("nw_panel_L2", &panel, Some(2), false, "panel"),
                ("nw_panel_def", &panel, None, false, "panel"),
                ("nw_panel_L20", &panel, Some(20), false, "panel"),
                ("nw_panel_w_L3", &panel, Some(3), true, "panel"),
                ("dk_L2", &panel, Some(2), false, "dk"),
                ("dk_def", &panel, None, false, "dk"),
                ("dk_L11", &panel, Some(11), false, "dk"),
                ("nw_ts_def", &ts, None, false, "ts"),
                ("nw_ts_L4", &ts, Some(4), false, "ts"),
                ("nw_ts_w_def", &ts, None, true, "ts"),
            ] {
                let fit = Fit::new(data, "y", &["x1", "x2"], weighted.then(|| data.col("w")), false);
                let row = fit.row(1.0);
                let n = fit.e.len();
                let (pos, t) = pos_dense_rank(&data.col("t"), pool).unwrap();
                assert_eq!(t as f64, refs.scalar(case, "T"));
                let k = fit.k;
                let (meat, lag) = match kind {
                    "panel" => {
                        let lag = lag.unwrap_or_else(|| lag_quarter_root(t));
                        let p = Panel::build(Some(&id_codes), &pos, pool).unwrap();
                        let c = lag_cross(&p, lag_fixest_clip(lag, t), k, pool, &row).unwrap();
                        (hac_combine(&c, &bartlett_weights(lag), k), lag)
                    }
                    "dk" => {
                        let lag = lag.unwrap_or_else(|| lag_quarter_root(t));
                        let codes: Vec<u32> = pos.iter().map(|&p| p as u32).collect();
                        let h = slot_sums(&codes, t, k, pool, &row).unwrap();
                        let slots: Vec<i64> = (0..t as i64).collect();
                        let c = lag_cross_dense(&h, &slots, k, lag_fixest_clip(lag, t));
                        (hac_combine(&c, &bartlett_weights(lag), k), lag)
                    }
                    _ => {
                        let p = Panel::build(None, &pos, pool).unwrap();
                        let lag = match lag {
                            Some(l) => l,
                            None => lag_fixest_ts(p.order(), k, Some(0), pool, &row).unwrap(),
                        };
                        let c = lag_cross(&p, lag_fixest_clip(lag, n), k, pool, &row).unwrap();
                        (hac_combine(&c, &bartlett_weights(lag), k), lag)
                    }
                };
                assert_eq!(lag as f64, refs.scalar(case, "lag"), "{case}: lag");
                let (nf, kf, tf) = (n as f64, refs.scalar(case, "K"), t as f64);
                let ssc = tf / (tf - 1.0) * (nf - 1.0) / (nf - kf);
                worst = worst.max(check_v("fixest", &refs, case, &fit, &meat, ssc));
            }
        });
        eprintln!("fixest NW/DK: max rel diff {worst:e}");
    }

    #[test]
    fn fixest_time_series_default_lag() {
        let refs = Refs::read("fixest_ref.csv");
        let csv = Csv::read("fixest_bw_scores.csv");
        let mut worst = 0.0f64;
        let cases: Vec<String> = {
            let mut c: Vec<String> = csv.rows.iter().map(|r| r[0].clone()).collect();
            c.dedup();
            c
        };
        assert_eq!(cases.len(), 12);
        for case in &cases {
            let rows: Vec<&Vec<String>> = csv.rows.iter().filter(|r| &r[0] == case).collect();
            let k = (2..csv.header.len()).take_while(|&j| !rows[0][j].trim().is_empty()).count();
            let s: Vec<f64> = rows.iter().flat_map(|r| (0..k).map(|j| num(&r[2 + j]))).collect();
            // Scores in a scrambled row order (time order given by `order`), with an intercept
            // column in front that fixest drops.
            let n = rows.len();
            let order: Vec<u32> = (0..n as u32).map(|t| (t * 7 + 3) % n as u32).collect();
            let mut at = vec![0usize; n];
            for (t, &i) in order.iter().enumerate() {
                at[i as usize] = t;
            }
            let row = |i: usize, a: &mut [f64]| {
                a[0] = 1.0;
                a[1..].copy_from_slice(&s[at[i] * k..(at[i] + 1) * k]);
                1.0
            };
            assert!(n % 7 != 0, "`order` must be a permutation");
            for threads in [1, 4] {
                let raw = pool_run(threads, |pool| bw_newey_west(&order, k + 1, Some(0), pool, &row)).unwrap();
                let want = refs.scalar(case, "raw");
                let err = (raw - want).abs() / want;
                assert!(err <= 1e-10, "{case}: {raw} vs {want}");
                worst = worst.max(err);
                let lag = pool_run(threads, |pool| lag_fixest_ts(&order, k + 1, Some(0), pool, &row)).unwrap();
                assert_eq!(lag as f64, refs.scalar(case, "lag"), "{case}");
            }
        }
        eprintln!("bwNeweyWest: max rel diff {worst:e}");
    }

    #[test]
    fn pyfixest_newey_west_and_driscoll_kraay() {
        let refs = Refs::read("pyfixest_ref.csv");
        let panel = Csv::read("panel.csv");
        let ts = Csv::read("ts.csv");
        let mut worst = 0.0f64;
        pool_run(4, |pool| {
            let id_codes: Vec<u32> = panel.col("id").iter().map(|&v| v as u32).collect();
            for (case, data, lag, weighted, kind) in [
                ("nw_panel_L2", &panel, Some(2), false, "nw"),
                ("nw_panel_def", &panel, None, false, "nw"),
                ("nw_panel_L20", &panel, Some(20), false, "nw"),
                ("nw_panel_w_L3", &panel, Some(3), true, "nw"),
                ("nw_ts_L4", &ts, Some(4), false, "ts"),
                ("nw_ts_w_L2", &ts, Some(2), true, "ts"),
                ("dk_L2", &panel, Some(2), false, "dk"),
                ("dk_def", &panel, None, false, "dk"),
            ] {
                let fit = Fit::new(data, "y", &["x1", "x2"], weighted.then(|| data.col("w")), false);
                let row = fit.row(1.0);
                let k = fit.k;
                let (rank, t) = pos_dense_rank(&data.col("t"), pool).unwrap();
                let lag = lag.unwrap_or_else(|| lag_quarter_root(t));
                let meat = match kind {
                    "nw" | "ts" => {
                        let unit = (kind == "nw").then_some(&id_codes[..]);
                        let p = Panel::build(unit, &rank, pool).unwrap().within_unit_positions();
                        let c = lag_cross(&p, lag, k, pool, &row).unwrap();
                        hac_combine(&c, &bartlett_weights(lag), k)
                    }
                    _ => {
                        // Duplicates are checked on (panel, time); lags are consecutive distinct times.
                        assert!(Panel::build(Some(&id_codes), &rank, pool).is_ok());
                        let codes: Vec<u32> = rank.iter().map(|&p| p as u32).collect();
                        let h = slot_sums(&codes, t, k, pool, &row).unwrap();
                        let slots: Vec<i64> = (0..t as i64).collect();
                        hac_combine(&lag_cross_dense(&h, &slots, k, lag), &bartlett_weights(lag), k)
                    }
                };
                let (nf, dfk, tf) = (refs.scalar(case, "N"), refs.scalar(case, "df_k"), refs.scalar(case, "T"));
                assert_eq!(tf, t as f64);
                let ssc = (nf - 1.0) / (nf - dfk) * tf / (tf - 1.0);
                worst = worst.max(check_v("pyfixest", &refs, case, &fit, &meat, ssc));
            }
        });
        eprintln!("pyfixest NW/DK: max rel diff {worst:e}");
    }

    #[test]
    fn ivreg2_hac_variances() {
        let refs = Refs::read("stata_ref.csv");
        let ts = Csv::read("ts.csv");
        let panel = Csv::read("panel.csv");
        let specs: [(&str, f64); 10] = [
            ("bartlett", 3.0),
            ("parzen", 2.5),
            ("truncated", 2.0),
            ("thann", 3.0),
            ("thamm", 2.0),
            ("qs", 1.7),
            ("dan", 2.0),
            ("tent", 2.0),
            ("bartlett", 1.0),
            ("parzen", 4.7),
        ];
        let mut cases: Vec<(String, &Csv, &str, Kernel, f64, bool, f64, bool)> = specs
            .iter()
            .enumerate()
            .map(|(i, &(k, bw))| (format!("hac_ts_{}", i + 1), &ts, "t", Kernel::parse(k).unwrap(), bw, false, 1.0, false))
            .collect();
        cases.push(("hac_ts_aw".into(), &ts, "t", Kernel::Bartlett, 3.0, true, 1.0, false));
        cases.push(("hac_d2_parzen".into(), &ts, "t2", Kernel::Parzen, 3.0, false, 2.0, false));
        cases.push(("hac_d2_tent".into(), &ts, "t2", Kernel::Tent, 2.5, false, 2.0, false));
        cases.push(("hac_panel_bartlett".into(), &panel, "t", Kernel::Bartlett, 3.0, false, 1.0, true));
        cases.push(("hac_panel_truncated".into(), &panel, "t", Kernel::Truncated, 2.0, false, 1.0, true));
        cases.push(("hac_panel_qs".into(), &panel, "t", Kernel::QuadraticSpectral, 2.2, false, 1.0, true));
        let mut worst = 0.0f64;
        pool_run(4, |pool| {
            for (case, data, tv, kernel, bw, aw, tdelta, is_panel) in &cases {
                assert_eq!(refs.scalar(case, "bw"), *bw, "{case}");
                let w = aw.then(|| data.col("w"));
                let mut fit = Fit::new(data, "y", &["x1", "x2"], w.clone(), true);
                let n = fit.e.len() as f64;
                // ivreg2: aweights enter as w * wf, wf = N / sum(w); V = (X'WX)^-1 S (X'WX)^-1
                // with the bread on w * wf.
                let wf = w.as_ref().map_or(1.0, |w| n / w.iter().sum::<f64>());
                fit.bread.iter_mut().for_each(|b| *b /= wf);
                let row = fit.row(wf);
                let time = data.col(tv);
                let pos = pos_calendar(&time, *tdelta).unwrap();
                let tmin = time.iter().copied().fold(f64::INFINITY, f64::min);
                let tmax = time.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let wts = ivreg2_lag_weights(*kernel, *bw, ivreg2_spectral_span(tmin, tmax, *tdelta));
                let id_codes: Vec<u32> = if *is_panel { data.col("id").iter().map(|&v| v as u32).collect() } else { Vec::new() };
                let p = Panel::build(is_panel.then_some(&id_codes[..]), &pos, pool).unwrap();
                let c = lag_cross(&p, wts.len(), fit.k, pool, &row).unwrap();
                let meat = hac_combine(&c, &wts, fit.k);
                worst = worst.max(check_v("ivreg2", &refs, case, &fit, &meat, 1.0));
            }
        });
        eprintln!("ivreg2 HAC: max rel diff {worst:e}");
    }

    #[test]
    fn ivreg2_automatic_bandwidth() {
        let refs = Refs::read("stata_ref.csv");
        let ts = Csv::read("ts.csv");
        let auto = Csv::read("ts_auto.csv");
        const XS: &[&str] = &["x1", "x2"];
        const XA: &[&str] = &["xa", "xb"];
        let mut cases: Vec<(String, &Csv, &str, &str, &[&str], f64, Kernel)> = Vec::new();
        for (k, name) in [(Kernel::Bartlett, "bartlett"), (Kernel::Parzen, "parzen"), (Kernel::QuadraticSpectral, "qs")] {
            cases.push((format!("abw_{name}"), &ts, "y", "t", XS, 1.0, k));
            cases.push((format!("abwhdfe_{name}"), &ts, "y", "t", XS, 1.0, k));
            for (tv, d) in [("t", 1.0), ("t3", 3.0)] {
                for y in ["ya", "yb", "yc"] {
                    cases.push((format!("auto_{tv}_{y}_{name}"), &auto, y, tv, XA, d, k));
                }
            }
        }
        for (k, name) in [(Kernel::Bartlett, "bartlett"), (Kernel::QuadraticSpectral, "qs")] {
            cases.push((format!("abwd2_{name}"), &ts, "y", "t2", XS, 2.0, k));
        }
        cases.push(("abwac_qs".into(), &ts, "y", "t", XS, 1.0, Kernel::QuadraticSpectral));
        cases.push(("autohdfe_t_yb_qs".into(), &auto, "yb", "t", XA, 1.0, Kernel::QuadraticSpectral));
        cases.push(("autoac_t_yb_parzen".into(), &auto, "yb", "t", XA, 1.0, Kernel::Parzen));
        let mut below_cap = 0;
        pool_run(4, |pool| {
            for (case, data, y, tv, xs, tdelta, kernel) in &cases {
                let fit = Fit::new(data, y, xs, None, true);
                let time = data.col(tv);
                let pos = pos_calendar(&time, *tdelta).unwrap();
                let tmin = time.iter().copied().fold(f64::INFINITY, f64::min);
                let tmax = time.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let kx = xs.len();
                let f: Vec<f64> = (0..fit.e.len()).map(|i| fit.e[i] * (0..kx).map(|j| fit.x[i * fit.k + j]).sum::<f64>()).collect();
                let bw = bw_ivreg2_auto(*kernel, &f, &pos, fit.e.len() as f64, tmax - tmin + 1.0, pool).unwrap();
                let want = refs.scalar(case, "bw");
                assert_eq!(bw, want, "{case}");
                let mstar = (20.0 * ((tmax - tmin + 1.0) / 100.0).powf(match kernel {
                    Kernel::Bartlett => 2.0 / 9.0,
                    Kernel::Parzen => 4.0 / 25.0,
                    _ => 2.0 / 25.0,
                }))
                .trunc();
                if bw < mstar + 1.0 {
                    below_cap += 1;
                }
                // The HAC variance at that bandwidth.
                if refs.has(case, "V") && !case.starts_with("abwac") && !case.starts_with("autoac") {
                    let wts = ivreg2_lag_weights(*kernel, bw, ivreg2_spectral_span(tmin, tmax, *tdelta));
                    let p = Panel::build(None, &pos, pool).unwrap();
                    let row = fit.row(1.0);
                    let c = lag_cross(&p, wts.len(), fit.k, pool, &row).unwrap();
                    check_v("ivreg2 auto", &refs, case, &fit, &hac_combine(&c, &wts, fit.k), 1.0);
                }
            }
        });
        assert!(below_cap >= 6, "only {below_cap} automatic bandwidths below the cap");
        let other = pool_run(1, |pool| bw_ivreg2_auto(Kernel::Truncated, &[1.0, 2.0], &[0, 1], 2.0, 2.0, pool));
        assert_eq!(other, None);
    }

    /// Scale check (run explicitly): 10M rows x 10 columns, Bartlett lag 50 on a 1M-unit panel,
    /// every lag of 1M units x 10 periods, and Driscoll–Kraay slot sums.
    #[test]
    #[ignore]
    fn timing_10m() {
        let (n, m) = (10_000_000usize, 10usize);
        let per = 10usize;
        let unit: Vec<u32> = (0..n).map(|i| (i / per) as u32).collect();
        let pos: Vec<i64> = (0..n).map(|i| ((i % per) * 6) as i64).collect();
        let row = |i: usize, a: &mut [f64]| {
            let mut g = SplitMix64::stream(3, i as u64);
            for x in a.iter_mut() {
                *x = g.next_f64();
            }
            1.0
        };
        pool_run(0, |pool| {
            let t0 = std::time::Instant::now();
            let p = Panel::build(Some(&unit), &pos, pool).unwrap();
            eprintln!("build (sorted): {:?}", t0.elapsed());
            let scrambled: Vec<u32> = (0..n).map(|i| ((i * 7_654_321) % n / per) as u32).collect();
            let t0 = std::time::Instant::now();
            let _ = Panel::build(Some(&scrambled), &pos, pool);
            eprintln!("build (scrambled): {:?}", t0.elapsed());
            let ts_pos: Vec<i64> = (0..n as i64).collect();
            let ts = Panel::build(None, &ts_pos, pool).unwrap();
            let t0 = std::time::Instant::now();
            let c = lag_cross(&ts, 50, m, pool, row).unwrap();
            eprintln!("lag_cross time series, lag 50: {:?} ({} lags)", t0.elapsed(), c.len());
            let t0 = std::time::Instant::now();
            let c = lag_cross(&p, usize::MAX, m, pool, row).unwrap();
            eprintln!("lag_cross panel, every lag (span {}): {:?}", c.len() - 1, t0.elapsed());
            let codes: Vec<u32> = (0..n).map(|i| (i % 1000) as u32).collect();
            let t0 = std::time::Instant::now();
            let _ = slot_sums(&codes, 1000, m, pool, row).unwrap();
            eprintln!("slot_sums 1000 slots: {:?}", t0.elapsed());
            let t0 = std::time::Instant::now();
            let lag = lag_fixest_ts(ts.order(), m, None, pool, row).unwrap();
            eprintln!("lag_fixest_ts: {:?} (lag {lag})", t0.elapsed());
        });
    }
}
