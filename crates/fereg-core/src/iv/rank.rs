//! Under- and weak-identification statistics as ivreghdfe obtains them from ranktest 2.0.04
//! (`ranktest (endog) (inst), partial(exog) full noconstant [wald] [robust|cluster()]`;
//! ivreghdfe.ado l.1727-1840, ranktest.ado `s_jstat` l.1010-1931 and `m_svd` l.2083-2137).
//!
//! iid: Anderson canonical-correlations LM = N·λ_min and Cragg–Donald F. Robust/cluster:
//! Kleibergen–Paap rk LM (underidentification) and rk Wald F (weak identification); with one
//! endogenous regressor ranktest's rank-0 block applies, with several the SVD algorithm.

use super::dense::{self, M};
use super::{Mode, Model, Passes};
use crate::error::RfError;

pub(crate) struct IdStats {
    pub idstat: f64,
    pub iddf: f64,
    pub cd: f64,
    pub cdf: f64,
    pub widstat: f64,
    /// KP rk Wald F (robust/cluster only).
    pub rkf: Option<f64>,
    /// Squared canonical correlations and cc²/(1−cc²), descending (`e(ccev)`, `e(cdev)`).
    pub ev: Option<(Vec<f64>, Vec<f64>)>,
}

/// Endog and excluded instruments with the exogenous regressors partialled out.
struct Partialled {
    /// R×K1 and R×L1 coefficients of the partialled columns over the row columns.
    ce: M,
    cz: M,
    qyy: M,
    qzz: M,
    qzy: M,
}

fn partial_out(m: &Model) -> Partialled {
    let (g, en, ex, iz) = (m.g, m.en, m.ex, m.iz);
    let r = g.r;
    let (k1, l1, k2) = (en.len(), iz.len(), ex.len());
    let x1x1 = g.pick(en, en);
    let z1z1 = g.pick(iz, iz);
    let z1x1 = g.pick(iz, en);
    let mut ce = M::zeros(r, k1);
    let mut cz = M::zeros(r, l1);
    for (c, &j) in en.iter().enumerate() {
        ce.set(j, c, 1.0);
    }
    for (c, &j) in iz.iter().enumerate() {
        cz.set(j, c, 1.0);
    }
    let n = m.n;
    if k2 == 0 {
        return Partialled { ce, cz, qyy: x1x1.scale(1.0 / n), qzz: z1z1.scale(1.0 / n), qzy: z1x1.scale(1.0 / n) };
    }
    let x2x2 = g.pick(ex, ex);
    let x2x1 = g.pick(ex, en);
    let x2z1 = g.pick(ex, iz);
    let by = dense::spd_solve(&x2x2, &x2x1);
    let bz = dense::spd_solve(&x2x2, &x2z1);
    for (c, _) in en.iter().enumerate() {
        for (k, &j) in ex.iter().enumerate() {
            ce.set(j, c, -by.at(k, c));
        }
    }
    for (c, _) in iz.iter().enumerate() {
        for (k, &j) in ex.iter().enumerate() {
            cz.set(j, c, -bz.at(k, c));
        }
    }
    let mut qyy = x1x1.sub(&x2x1.t().mul(&by)).scale(1.0 / n);
    let mut qzz = z1z1.sub(&x2z1.t().mul(&bz)).scale(1.0 / n);
    qyy.symmetrize();
    qzz.symmetrize();
    let qzy = z1x1.sub(&x2z1.t().mul(&by)).scale(1.0 / n);
    Partialled { ce, cz, qyy, qzz, qzy }
}

/// KP statistic of the test rank = K1−1 (`r(chi2)` of `ranktest …, full`) given the
/// un-normalized meat of vhat ⊗ z.
fn kp_stat(p: &Partialled, shat_raw: &M, n: f64) -> f64 {
    let (k1, l1) = (p.qyy.r, p.qzz.r);
    let gvec = p.qzy.vec(); // vec(Qzy): index c·L1 + k
    if k1 == 1 {
        // Rank-0 block (ranktest l.1600-1628): j = gbar'·invsym(Ŝ0)·gbar·N with Ŝ0 = meat/N.
        let (inv, _) = dense::ginv(&shat_raw.scale(1.0 / n));
        return inv.quad(&gvec) * n;
    }
    let ly = dense::chol(&p.qyy).expect("Qyy positive definite");
    let lz = dense::chol(&p.qzz).expect("Qzz positive definite");
    let lyi = dense::tri_lower_inv(&ly);
    let lzi = dense::tri_lower_inv(&lz);
    let pihat = dense::spd_solve(&p.qzz, &p.qzy);
    // kpthat = rQzz'·pihat·irQyy with irQyy = (Ly')⁻¹.
    let that = lz.t().mul(&pihat).mul(&lyi.t());
    let (u, _s, v) = dense::svd_full(&that);
    let kron = lyi.kron(&lzi);
    let mut kpvar = kron.mul(&shat_raw.scale(1.0 / n)).mul(&kron.t());
    kpvar.symmetrize();
    let kk = k1 - 1;
    let u22 = u.block(kk, l1, kk, l1);
    let v22 = v.block(kk, k1, kk, k1);
    let u22h = dense::sqrtm_psd(&u22.mul(&u22.t()));
    let v22h = dense::sqrtm_psd(&v22.mul(&v22.t()));
    let ucol = u.block(0, l1, kk, l1);
    let vcol = v.block(0, k1, kk, k1);
    let aq = ucol.mul(&dense::inv_general(&u22).unwrap_or_else(|| dense::ginv(&u22).0)).mul(&u22h);
    let bq = v22h.mul(&dense::inv_general(&v22.t()).unwrap_or_else(|| dense::ginv(&v22.t()).0)).mul(&vcol.t());
    let t = bq.kron(&aq.t());
    let lab = t.mul(&M::col(&that.vec()));
    let mut vlab = t.mul(&kpvar).mul(&t.t());
    vlab.symmetrize();
    let (vinv, _) = dense::ginv(&vlab);
    vinv.quad(&lab.d) * n
}

pub(crate) fn id_stats(m: &Model, passes: &Passes) -> Result<IdStats, RfError> {
    let (k1, l1, l) = (m.en.len(), m.iz.len(), m.l() as f64);
    let n = m.n;
    let p = partial_out(m);

    // Canonical correlations: eigenvalues of Qyy^{-1/2}·Qzy'·Qzz⁻¹·Qzy·Qyy^{-1/2}.
    let ly = dense::chol(&p.qyy);
    let ev = ly.map(|ly| {
        let lyi = dense::tri_lower_inv(&ly);
        let a = lyi.mul(&p.qzy.t());
        let mut mat = a.mul(&dense::spd_solve(&p.qzz, &p.qzy)).mul(&lyi.t());
        mat.symmetrize();
        let (vals, _) = dense::sym_eig(&mat);
        let ccev: Vec<f64> = vals.iter().map(|v| v.max(0.0)).collect();
        let cdev: Vec<f64> = ccev.iter().map(|c| c / (1.0 - c)).collect();
        (ccev, cdev)
    });
    // ranktest error (ivreghdfe l.1758-1770): missing statistics when collinear (cc = 1).
    let Some((ccev, cdev)) = ev.filter(|(cc, _)| cc[0] < 1.0) else {
        return Ok(IdStats {
            idstat: f64::NAN,
            iddf: f64::NAN,
            cd: f64::NAN,
            cdf: f64::NAN,
            widstat: f64::NAN,
            rkf: (m.mode != Mode::Iid).then_some(f64::NAN),
            ev: None,
        });
    };
    let cd = cdev[k1 - 1];
    let cdf = cd * (n - m.sdof - l) / l1 as f64;
    let iddf = (l1 - k1 + 1) as f64;
    let (idstat, rkf) = match m.mode {
        Mode::Iid => (n * ccev[k1 - 1], None),
        Mode::Robust | Mode::Cluster { .. } => {
            let lm = kp_stat(&p, &passes.meat(&p.ce, &p.cz)?, n);
            // Wald version: vhat = y − z·pihat.
            let pihat = dense::spd_solve(&p.qzz, &p.qzy);
            let cw = p.ce.sub(&p.cz.mul(&pihat));
            let wald = kp_stat(&p, &passes.meat(&cw, &p.cz)?, n);
            let rkf = match m.mode {
                Mode::Cluster { g } => wald / (n - 1.0) * (n - l - m.sdof) * (g - 1.0) / g / l1 as f64,
                _ => wald / n * (n - l - m.sdof) / l1 as f64,
            };
            (lm, Some(rkf))
        }
    };
    let widstat = rkf.unwrap_or(cdf);
    Ok(IdStats { idstat, iddf, cd, cdf, widstat, rkf, ev: Some((ccev, cdev)) })
}
