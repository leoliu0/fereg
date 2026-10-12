//! Dense explicit-dummy absorber: the oracle for every core test (plan §1.1b).
//!
//! D has one column per (term, slot, level): an intercept dummy and/or slope×dummy columns
//! (a single column of ones under `noabsorb`). Residualization uses the thin SVD of √W·D with
//! rank cutoff σ > 1e-11·σ_max·max(n, p), so collinear and disconnected designs are handled
//! exactly (minimum-norm coefficients). Only for small problems (n·p ≤ 2·10⁷).

use faer::Mat;

use crate::absorber::{Absorber, FeCoefs, SolveInfo, SolveOpts, fe1_index};
use crate::api::Ctx;
use crate::error::RfError;
use crate::par::Pool;
use crate::session::Term;

#[derive(Clone, Debug)]
struct Slot {
    /// Column offset of level 0.
    off: usize,
    g: usize,
}

struct Basis {
    rank: usize,
    /// n×rank, column j at `u[j*n..]`.
    u: Vec<f64>,
    s: Vec<f64>,
    /// p×rank, column j at `v[j*p..]`.
    v: Vec<f64>,
}

pub struct DenseAbsorber {
    n: usize,
    order: Vec<u32>,
    w: Option<Vec<f64>>,
    p: usize,
    /// n×p row-major design.
    d: Vec<f64>,
    /// Slots of each session term (intercept first, then slopes).
    term_slots: Vec<Vec<Slot>>,
    /// Codes of each term (row order = kept order), for normalization.
    term_codes: Vec<Vec<u32>>,
    /// Intercept-only terms (normalized in `term_values`).
    intercept_only: Vec<bool>,
    fe1: Option<usize>,
    basis: Option<Basis>,
}

impl DenseAbsorber {
    pub fn new(terms: &[Term], weights: Option<&[f64]>, n: usize) -> Result<Self, RfError> {
        let mut p = 0usize;
        let mut term_slots = Vec::with_capacity(terms.len());
        for t in terms {
            let mut slots = Vec::new();
            for _ in 0..(usize::from(t.intercept) + t.slopes.len()) {
                slots.push(Slot { off: p, g: t.g });
                p += t.g;
            }
            term_slots.push(slots);
        }
        let constant_only = terms.is_empty();
        if constant_only {
            p = 1;
        }
        if n.saturating_mul(p) > 20_000_000 {
            return Err(RfError::InvalidInput(format!(
                "DenseAbsorber is for small problems only (n = {n}, dummy columns = {p})"
            )));
        }
        let mut d = vec![0.0; n * p];
        if constant_only {
            d.iter_mut().for_each(|v| *v = 1.0);
        }
        for (t, slots) in terms.iter().zip(&term_slots) {
            let mut k = 0;
            if t.intercept {
                for i in 0..n {
                    d[i * p + slots[0].off + t.codes[i] as usize] = 1.0;
                }
                k = 1;
            }
            for (s, x) in t.slopes.iter().enumerate() {
                let off = slots[k + s].off;
                for i in 0..n {
                    d[i * p + off + t.codes[i] as usize] = x[i];
                }
            }
        }
        Ok(DenseAbsorber {
            n,
            order: (0..n as u32).collect(),
            w: weights.map(<[f64]>::to_vec),
            p,
            d,
            term_slots,
            term_codes: terms.iter().map(|t| t.codes.clone()).collect(),
            intercept_only: terms.iter().map(|t| t.intercept && t.slopes.is_empty()).collect(),
            fe1: fe1_index(terms),
            basis: None,
        })
    }

    fn ensure_basis(&mut self, pool: &Pool) -> Result<(), RfError> {
        if self.basis.is_none() {
            let (n, p) = (self.n, self.p);
            let sw: Vec<f64> = match &self.w {
                Some(w) => w.iter().map(|v| v.sqrt()).collect(),
                None => vec![1.0; n],
            };
            let a = Mat::<f64>::from_fn(n, p, |i, j| sw[i] * self.d[i * p + j]);
            // Inside the pool so faer's rayon parallelism uses our scoped workers (plan D4).
            let svd = pool.install(|| a.thin_svd()).map_err(|e| RfError::Internal(format!("dense SVD failed: {e:?}")))?;
            let (u, s, v) = (svd.U(), svd.S().column_vector(), svd.V());
            let k = s.nrows();
            let smax = if k > 0 { s[0] } else { 0.0 };
            let cut = 1e-11 * smax * n.max(p) as f64;
            let rank = (0..k).take_while(|&i| s[i] > cut).count();
            let mut bu = vec![0.0; n * rank];
            let mut bv = vec![0.0; p * rank];
            for j in 0..rank {
                for i in 0..n {
                    bu[j * n + i] = u[(i, j)];
                }
                for i in 0..p {
                    bv[j * p + i] = v[(i, j)];
                }
            }
            self.basis = Some(Basis { rank, u: bu, s: (0..rank).map(|i| s[i]).collect(), v: bv });
        }
        Ok(())
    }
}

impl Absorber for DenseAbsorber {
    fn n(&self) -> usize {
        self.n
    }

    fn row_order(&self) -> &[u32] {
        &self.order
    }

    fn weights(&self) -> Option<&[f64]> {
        self.w.as_deref()
    }

    fn set_weights(&mut self, w: Option<Vec<f64>>, _pool: &Pool, _ctx: &Ctx) -> Result<(), RfError> {
        if let Some(v) = &w {
            if v.len() != self.n || v.iter().any(|x| !(x.is_finite() && *x > 0.0)) {
                return Err(RfError::InvalidInput("weights must be positive and finite".into()));
            }
        }
        self.w = w;
        self.basis = None;
        Ok(())
    }

    fn residualize(
        &mut self,
        z: &mut [f64],
        r: usize,
        o: &SolveOpts,
        _warm: Option<&FeCoefs>,
        pool: &Pool,
        ctx: &Ctx,
    ) -> Result<SolveInfo, RfError> {
        let (n, p) = (self.n, self.p);
        if z.len() != n * r {
            return Err(RfError::InvalidInput("residualize: block size differs from n·R".into()));
        }
        ctx.check()?;
        let sw: Vec<f64> = match &self.w {
            Some(w) => w.iter().map(|v| v.sqrt()).collect(),
            None => vec![1.0; n],
        };
        self.ensure_basis(pool)?;
        let basis = self.basis.as_ref().expect("ensured above");
        let mut coefs = vec![0.0; p * r];
        for col in 0..r {
            // α = V Σ⁻¹ U'(√w z)
            let mut alpha = vec![0.0; p];
            for j in 0..basis.rank {
                let uj = &basis.u[j * n..(j + 1) * n];
                let t: f64 = (0..n).map(|i| uj[i] * sw[i] * z[i * r + col]).sum::<f64>() / basis.s[j];
                let vj = &basis.v[j * p..(j + 1) * p];
                for (a, &vv) in alpha.iter_mut().zip(vj) {
                    *a += vv * t;
                }
            }
            for i in 0..n {
                let fit: f64 = self.d[i * p..(i + 1) * p].iter().zip(&alpha).map(|(a, b)| a * b).sum();
                z[i * r + col] -= fit;
            }
            for (k, a) in alpha.into_iter().enumerate() {
                coefs[k * r + col] = a;
            }
        }
        Ok(SolveInfo { iterations: 1, escalated: false, coefs: o.want_coefs.then_some(FeCoefs { p, r, data: coefs }), exact: true })
    }

    fn fit_rows(&self, c: &FeCoefs, col: usize, _pool: &Pool) -> Vec<f64> {
        let alpha = c.col(col);
        (0..self.n).map(|i| self.d[i * self.p..(i + 1) * self.p].iter().zip(&alpha).map(|(a, b)| a * b).sum()).collect()
    }

    fn term_values(&self, c: &FeCoefs, col: usize) -> Vec<Vec<f64>> {
        let alpha = c.col(col);
        let mut vals: Vec<Vec<Vec<f64>>> = self
            .term_slots
            .iter()
            .map(|slots| slots.iter().map(|s| alpha[s.off..s.off + s.g].to_vec()).collect())
            .collect();
        if let Some(f) = self.fe1 {
            let wsum: f64 = self.w.as_ref().map_or(self.n as f64, |w| w.iter().sum());
            for q in 0..vals.len() {
                if q == f || !self.intercept_only[q] {
                    continue;
                }
                let codes = &self.term_codes[q];
                let mean: f64 = (0..self.n)
                    .map(|i| self.w.as_ref().map_or(1.0, |w| w[i]) * vals[q][0][codes[i] as usize])
                    .sum::<f64>()
                    / wsum;
                vals[q][0].iter_mut().for_each(|v| *v -= mean);
                vals[f][0].iter_mut().for_each(|v| *v += mean);
            }
        }
        vals.into_iter().flatten().collect()
    }
}
