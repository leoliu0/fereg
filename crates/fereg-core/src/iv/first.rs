//! First-stage statistics (`ffirst`) and first-stage regressions (`first`): ivreg2's `s_ffirst`
//! (ivreghdfe.ado l.6150-6493) and `PostFirstRF` (l.3156-3372).

use super::dense::{self, M};
use super::{Mode, Model, Passes, resid_coef, selector};
use crate::error::RfError;

/// One first-stage regression (coefficients over Z = [inst, exog]).
pub(crate) struct FirstEq {
    pub b: Vec<f64>,
    /// Small-sample adjusted VCE (`PostFirstRF`).
    pub v: M,
    /// (e(rmse), e(df_r), e(df_m), e(N)) of the posted regression.
    pub stats: [f64; 4],
}

pub(crate) struct FirstStage {
    /// 21×K1, rows rmse sheapr2 pr2 F df df_r pvalue SWF SWFdf1 SWFdf2 SWFp SWchi2 SWchi2p SWr2
    /// APF APFdf1 APFdf2 APFp APchi2 APchi2p APr2.
    pub firstmat: M,
    pub arf: f64,
    pub archi2: f64,
    pub ardf: f64,
    pub ardf_r: f64,
    pub sstat: f64,
    pub sstatdf: f64,
    pub eqs: Vec<FirstEq>,
}

/// Wald statistic Rb'·RVR⁻¹·Rb (Mata `cholsolve`).
fn wald(rb: &[f64], rvr: &M) -> f64 {
    dense::spd_inv(rvr).quad(rb)
}

pub(crate) fn first_stage(m: &Model, passes: &Passes, xxinv: &M, xpzx_inv: &M) -> Result<FirstStage, RfError> {
    let g = m.g;
    let r = g.r;
    let (k1, l1, k2) = (m.en.len(), m.iz.len(), m.ex.len());
    let (l, k) = (l1 + k2, k1 + k2);
    let n = m.n;
    let zcols = m.zcols();
    let xcols = m.xcols();
    let ycols: Vec<usize> = std::iter::once(0).chain(m.en.iter().copied()).collect();
    let sz = selector(r, &zcols);
    let robust = m.mode != Mode::Iid;
    let nl = n - l as f64 - m.sdof;
    let (fdf_r, scale_f) = match m.mode {
        // F = Wald/N·df_r/df, or Wald/(N−1)·(N−L−sdof)·(G−1)/G/df when clustered.
        Mode::Cluster { g } => (g - 1.0, (nl) * (g - 1.0) / g / (n - 1.0)),
        _ => (nl, nl / n),
    };

    // Reduced form (col 0) and first stages: bz = (Z'Z)⁻¹Z'[y, X1].
    let zz = g.pick(&zcols, &zcols);
    let zzinv = dense::spd_inv(&zz);
    let zyall = g.pick(&zcols, &ycols);
    let bz = zzinv.mul(&zyall);
    let m1 = k1 + 1;
    let mut call = M::zeros(r, m1);
    for (c, &yc) in ycols.iter().enumerate() {
        let col = resid_coef(r, yc, &zcols, &(0..l).map(|a| bz.at(a, c)).collect::<Vec<_>>());
        for j in 0..r {
            call.set(j, c, col.d[j]);
        }
    }
    let ee = passes.cross(&call);
    let sigmasqall = ee.scale(1.0 / n);
    let rmseall: Vec<f64> = ee.diag().iter().map(|v| (v / nl).sqrt()).collect();
    // Vr: (I ⊗ (Z'Z)⁻¹)·S·(I ⊗ (Z'Z)⁻¹), S the (robust) meat of eall ⊗ z; iid: (ee/N) ⊗ (Z'Z)⁻¹.
    let vr = if robust {
        let s = passes.meat(&call, &sz)?;
        let kz = M::eye(m1).kron(&zzinv);
        let mut v = kz.mul(&s).mul(&kz);
        v.symmetrize();
        v
    } else {
        sigmasqall.kron(&zzinv)
    };
    let vblock = |eq: usize, len: usize| vr.block(eq * l, eq * l + len, eq * l, eq * l + len);

    // Anderson–Rubin test from the reduced form.
    let rb: Vec<f64> = (0..l1).map(|a| bz.at(a, 0)).collect();
    let archi2 = wald(&rb, &vblock(0, l1));
    let ardf = l1 as f64;
    let (ardf_r, arf) = match m.mode {
        Mode::Cluster { g } => (g - 1.0, archi2 / (n - 1.0) * nl / g * (g - 1.0) / ardf),
        _ => (nl, archi2 / n * nl / ardf),
    };

    // Stock–Wright S: J of the model with zero endog coefficients, exog partialled out.
    let cs = if k2 > 0 {
        let x2x2 = g.pick(m.ex, m.ex);
        let by = dense::spd_solve(&x2x2, &g.pick(m.ex, &[0]));
        resid_coef(r, 0, m.ex, &by.d)
    } else {
        resid_coef(r, 0, &[], &[])
    };
    let ze: Vec<f64> = (0..l).map(|a| (0..r).map(|j| g.at(zcols[a], j) * cs.d[j]).sum()).collect();
    let sstat = if robust {
        let s = passes.meat(&cs, &sz)?;
        dense::spd_inv(&s).quad(&ze)
    } else {
        let es = passes.cross(&cs).d[0];
        n * zzinv.quad(&ze) / es
    };

    // Per endogenous regressor.
    let shea = |i: usize| xxinv.at(i, i) / xpzx_inv.at(i, i);
    let zzinv_l1 = zzinv.block(0, l1, 0, l1);
    let r2_of = |w: f64| (w / n) / (1.0 + w / n);
    let mut firstmat = M::zeros(21, k1);
    let mut eqs = Vec::with_capacity(k1);
    for i in 0..k1 {
        let b: Vec<f64> = (0..l).map(|a| bz.at(a, i + 1)).collect();
        let rb = &b[..l1];
        let classical = wald(rb, &zzinv_l1.scale(sigmasqall.at(i + 1, i + 1)));
        let pr2 = r2_of(classical);
        let wd = if robust { wald(rb, &vblock(i + 1, l1)) } else { classical };
        let df = l1 as f64;
        let f = wd * scale_f / df;
        let pvalue = super::dist::ftail(df, fdf_r, f);
        let col: [f64; 21] = if k1 == 1 {
            let chi2p = super::dist::chi2tail(df, wd);
            [rmseall[i + 1], shea(i), pr2, f, df, fdf_r, pvalue, f, df, fdf_r, pvalue, wd, chi2p, pr2, f, df, fdf_r, pvalue, wd, chi2p, pr2]
        } else {
            // Angrist–Pischke and Sanderson–Windmeijer conditional statistics.
            let sel: Vec<usize> = (0..k).filter(|&a| a != i).collect();
            // Fitted X: Xhat = Z·Bx with Bx = [bz endog columns, selectors of the exog in Z].
            let bx = M::from_fn(l, k, |a, c| if c < k1 { bz.at(a, c + 1) } else { f64::from(u8::from(a == l1 + c - k1)) });
            let zx1i: Vec<f64> = (0..l).map(|a| zyall.at(a, i + 1)).collect();
            let qxhxh = bx.t().mul(&zz).mul(&bx);
            let qxhx1 = bx.t().mul(&M::col(&zx1i));
            let qxhxhi = qxhxh.pick(&sel, &sel);
            let qxhx1i = qxhx1.pick(&sel, &[0]);
            let b1 = dense::ginv(&qxhxhi).0.mul(&qxhx1i).d;
            let bx_sel = bx.pick(&(0..l).collect::<Vec<_>>(), &sel);
            let c: Vec<f64> = bx_sel.mul(&M::col(&b1)).d;
            let zzc = zz.mul(&M::col(&c)).d;
            let b2: Vec<f64> = zzinv.mul(&M::col(&zx1i.iter().zip(&zzc).map(|(a, b)| a - b).collect::<Vec<_>>())).d;
            let ap_coef: Vec<f64> = c.iter().zip(&b2).map(|(a, b)| a + b).collect();
            let ap_e = resid_coef(r, m.en[i], &zcols, &ap_coef);
            let ap_rb = &b2[..l1];
            // SW: residuals with the actual X.
            let xsel: Vec<usize> = sel.iter().map(|&a| xcols[a]).collect();
            let e1 = resid_coef(r, m.en[i], &xsel, &b1);
            let qze1: Vec<f64> = (0..l).map(|a| (0..r).map(|j| g.at(zcols[a], j) * e1.d[j]).sum()).collect();
            let sb2 = zzinv.mul(&M::col(&qze1)).d;
            let mut sw_e = e1.clone();
            for (a, &zc) in zcols.iter().enumerate() {
                sw_e.d[zc] -= sb2[a];
            }
            let sw_rb = &sb2[..l1];
            let both = M::from_fn(r, 2, |j, cc| if cc == 0 { ap_e.d[j] } else { sw_e.d[j] });
            let ee2 = passes.cross(&both);
            let ap_cls = wald(ap_rb, &zzinv_l1.scale(ee2.at(0, 0) / n));
            let sw_cls = wald(sw_rb, &zzinv_l1.scale(ee2.at(1, 1) / n));
            let (ap_r2, sw_r2) = (r2_of(ap_cls), r2_of(sw_cls));
            let (ap_w, sw_w) = if robust {
                let rob = |e: &M, rb: &[f64]| -> Result<f64, RfError> {
                    let s = passes.meat(e, &sz)?;
                    let mut vi = zzinv.mul(&s).mul(&zzinv);
                    vi.symmetrize();
                    Ok(wald(rb, &vi.block(0, l1, 0, l1)))
                };
                (rob(&ap_e, ap_rb)?, rob(&sw_e, sw_rb)?)
            } else {
                (ap_cls, sw_cls)
            };
            let fdf1 = (l1 - k1 + 1) as f64;
            let (apf, swf) = (ap_w * scale_f / fdf1, sw_w * scale_f / fdf1);
            let ftail = |f| super::dist::ftail(fdf1, fdf_r, f);
            let chi2p = |w| super::dist::chi2tail(fdf1, w);
            [
                rmseall[i + 1],
                shea(i),
                pr2,
                f,
                df,
                fdf_r,
                pvalue,
                swf,
                fdf1,
                fdf_r,
                ftail(swf),
                sw_w,
                chi2p(sw_w),
                sw_r2,
                apf,
                fdf1,
                fdf_r,
                ftail(apf),
                ap_w,
                chi2p(ap_w),
                ap_r2,
            ]
        };
        for (row, v) in col.iter().enumerate() {
            firstmat.set(row, i, *v);
        }

        // Posted first-stage regression (PostFirstRF small-sample adjustment).
        let adj = match m.mode {
            Mode::Cluster { g } => (n - 1.0) / nl * g / (g - 1.0),
            _ => n / nl,
        };
        eqs.push(FirstEq { b, v: vblock(i + 1, l).scale(adj), stats: [rmseall[i + 1], nl, l as f64, n] });
    }
    Ok(FirstStage { firstmat, arf, archi2, ardf, ardf_r, sstat, sstatdf: l1 as f64, eqs })
}
