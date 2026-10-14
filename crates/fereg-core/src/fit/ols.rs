//! Linear models in the fit layer: OLS and IV/2SLS (fixest `feols`, pyfixest `feols` / `Feiv`).
//! One gather of the input columns into solver row order, one residualization and one Gram
//! pass; every regression afterwards (the IV first stages, the second stage, the auxiliary
//! regressions of the F, Wu-Hausman and Sargan statistics) works on that Gram, and one more pass
//! over the rows evaluates all residual sums of squares and the residuals.

use rayon::prelude::*;

use super::design::{self, Reg, regress, residual_pass, unit};
use super::extra::{self, JkData, VcovInputs, fe_leverage, jk_model};
use super::input::Sample;
use super::fixef::{self, FeLevels};
use super::state::{State, state_terms, term_refs};
use super::vcov::{self, ClusterInfo, FeInfo, Jackknife, Model};
use super::{CoefSource, FitInput, FitOutput, FitSpec, FitStats, Flavor, IvFirstStage, IvStats, TestStat, VcovData, VcovKind, WeightsType, dist};
use crate::absorber::SolveOpts;
use crate::api::Ctx;
use crate::error::RfError;
use crate::gather::Col;
use crate::par::{self, CHUNK, Pool};
use crate::leverage::FeCols;
use crate::session::Session;
use crate::stats;

/// fixest `nparams`: kept coefficients plus FE parameters (fixest_env.R): every intercept term
/// counts G*(1 + #slopes), a slope-only term G*#slopes, minus one per intercept term beyond the first.
pub(crate) fn nparams(k: usize, session: &Session) -> usize {
    let sizes: Vec<usize> = session.terms().iter().map(|t| t.g).collect();
    nparams_with(k, session.terms(), &sizes)
}

/// [`nparams`] with `sizes[q]` levels for term q.
pub(crate) fn nparams_with(k: usize, terms: &[crate::session::Term], sizes: &[usize]) -> usize {
    let n_icpt = terms.iter().filter(|t| t.intercept).count();
    let fe: usize = terms.iter().zip(sizes).map(|(t, g)| g * (usize::from(t.intercept) + t.slopes.len())).sum();
    k + fe - n_icpt.saturating_sub(1)
}

/// Weighted mean and sum w(y - ybar_w)^2 of the input y over the kept rows (kept order weights),
/// chunked with Chan's merge in chunk order. Returns (sum w, ybar_w, ssr_null).
pub(crate) fn y_stats(y: &[f64], kept: &[u32], w: Option<&[f64]>, pool: &Pool) -> (f64, f64, f64) {
    let parts = par::map_chunks(pool, kept.len(), CHUNK, |_, range| {
        let wi = |i: usize| w.map_or(1.0, |w| w[i]);
        let sw: f64 = range.clone().map(wi).sum();
        let mean = range.clone().map(|i| wi(i) * y[kept[i] as usize]).sum::<f64>() / sw;
        let m2: f64 = range.map(|i| wi(i) * (y[kept[i] as usize] - mean).powi(2)).sum();
        (sw, mean, m2)
    });
    let mut it = parts.into_iter();
    let first = it.next().unwrap_or((0.0, 0.0, 0.0));
    let (sw, mean, m2) = it.fold(first, |(sa, ma, qa), (sb, mb, qb)| {
        let s = sa + sb;
        let d = mb - ma;
        (s, ma + d * sb / s, qa + qb + d * d * sa * sb / s)
    });
    (sw, mean, m2)
}

/// Partial sums of one chunk for [`sq_cor`]: count, means of y and f, centered sum y^2, sum f^2, sum yf.
pub(crate) fn cor_part(pairs: impl Iterator<Item = (f64, f64)> + Clone) -> [f64; 6] {
    let (mut c, mut sy, mut sf) = (0.0, 0.0, 0.0);
    for (y, f) in pairs.clone() {
        c += 1.0;
        sy += y;
        sf += f;
    }
    let (my, mf) = (sy / c, sf / c);
    let (mut syy, mut sff, mut syf) = (0.0, 0.0, 0.0);
    for (y, f) in pairs {
        let (dy, df) = (y - my, f - mf);
        syy += dy * dy;
        sff += df * df;
        syf += dy * df;
    }
    [c, my, mf, syy, sff, syf]
}

/// Squared correlation of y and f from chunk partials ([`cor_part`], merged with Chan's formula
/// in chunk order); NaN when f is constant (fixest `sq.cor`).
pub(crate) fn sq_cor(parts: Vec<[f64; 6]>) -> f64 {
    let mut it = parts.into_iter();
    let mut a = it.next().unwrap_or([0.0; 6]);
    for b in it {
        let n = a[0] + b[0];
        let (dy, df) = (b[1] - a[1], b[2] - a[2]);
        let f = a[0] * b[0] / n;
        a = [n, a[1] + dy * b[0] / n, a[2] + df * b[0] / n, a[3] + b[3] + dy * dy * f, a[4] + b[4] + df * df * f, a[5] + b[5] + dy * df * f];
    }
    if a[4] == 0.0 { f64::NAN } else { a[5] * a[5] / (a[3] * a[4]) }
}

/// Fitted values y - e (kept order) and their squared correlation with y, in one chunked pass.
fn fitted_sq_cor(y: &[f64], kept: &[u32], e: &[f64], pool: &Pool) -> Result<(Vec<f64>, f64), RfError> {
    let mut fitted = par::alloc_vec(kept.len(), 0.0)?;
    let parts: Vec<[f64; 6]> = pool.install(|| {
        fitted
            .par_chunks_mut(CHUNK)
            .zip(kept.par_chunks(CHUNK).zip(e.par_chunks(CHUNK)))
            .map(|(f, (kc, ec))| {
                for ((f, &i), &ei) in f.iter_mut().zip(kc).zip(ec) {
                    *f = y[i as usize] - ei;
                }
                cor_part(kc.iter().map(|&i| y[i as usize]).zip(f.iter().copied()))
            })
            .collect()
    });
    Ok((fitted, sq_cor(parts)))
}

/// Columns of the row block: [y, endog..., (1), inst..., x...].
struct Layout {
    r: usize,
    m: usize,
    icpt: bool,
    l: usize,
}

impl Layout {
    fn endog(&self, j: usize) -> usize {
        1 + j
    }
    fn one(&self) -> usize {
        1 + self.m
    }
    fn inst(&self, j: usize) -> usize {
        1 + self.m + usize::from(self.icpt) + j
    }
    fn x(&self, j: usize) -> usize {
        1 + self.m + usize::from(self.icpt) + self.l + j
    }
}

/// `a*b` for a k-vector and a k x k row-major inverse: c'*A*c.
fn quad_form(c: &[f64], a: &[f64]) -> f64 {
    let k = c.len();
    (0..k).map(|i| c[i] * (0..k).map(|j| a[i * k + j] * c[j]).sum::<f64>()).sum()
}

/// fixest `.wald` on the coefficients `idx` of a model with variance `v` (k x k): stat = c'V^-1 c / q
/// with V^-1 by `cpp_cholesky` (tolerance 1e-10; dropped coefficients are left out), p from
/// F(df1, df2). Pyfixest: c'*pinv(V)*c / q with p from chi^2 (q) on c'*pinv(V)*c.
fn first_stage_wald(flavor: Flavor, coef: &[f64], v: &[f64], idx: &[usize], df1: f64, df2: f64) -> TestStat {
    let k = coef.len();
    if idx.is_empty() {
        return TestStat::NA;
    }
    let sub: Vec<f64> = idx.iter().flat_map(|&a| idx.iter().map(move |&b| v[a * k + b])).collect();
    let q = idx.len();
    match flavor {
        Flavor::Fixest => {
            let (excl, inv) = crate::vcov::chol_collin(&sub, q, 1e-10);
            let c: Vec<f64> = idx.iter().zip(&excl).filter(|(_, e)| !**e).map(|(&a, _)| coef[a]).collect();
            if c.is_empty() {
                return TestStat { stat: 1e6, p: dist::pf_upper(1e6, df1, df2), df1, df2 };
            }
            let stat = quad_form(&c, &inv) / c.len() as f64;
            TestStat { stat, p: dist::pf_upper(stat, df1, df2), df1, df2 }
        }
        Flavor::Pyfixest => {
            let c: Vec<f64> = idx.iter().map(|&a| coef[a]).collect();
            let w = quad_form(&c, &design::pinv_sym(&sub, q));
            TestStat { stat: w / q as f64, p: dist::pchisq_upper(w, q as f64), df1: q as f64, df2: f64::NAN }
        }
    }
}

/// The input columns of the sources `which` selects, in `sources` order (CRV3 refit problems).
fn pick<'c>(sources: &[CoefSource], which: fn(&CoefSource) -> Option<usize>, cols: &[&'c [f64]]) -> Vec<&'c [f64]> {
    sources.iter().filter_map(which).map(|j| cols[j]).collect()
}

fn x_of(s: &CoefSource) -> Option<usize> {
    if let CoefSource::X(j) = s { Some(*j) } else { None }
}

fn endog_of(s: &CoefSource) -> Option<usize> {
    if let CoefSource::Endog(j) = s { Some(*j) } else { None }
}

fn inst_of(s: &CoefSource) -> Option<usize> {
    if let CoefSource::Inst(j) = s { Some(*j) } else { None }
}

/// One regressor of the reported model: its source, the combo regressed on (IV: the fitted
/// endogenous variable) and the combo of the original variable (IV: the endogenous variable).
struct Regressor {
    source: CoefSource,
    design: Vec<f64>,
    actual: Vec<f64>,
}

pub(crate) fn linear(input: &FitInput, spec: &FitSpec, sample: Sample, ctx: &Ctx, pool: &Pool) -> Result<FitOutput, RfError> {
    let Sample { mut session, dropped, .. } = sample;
    let mut t = std::time::Instant::now();
    let n = session.n();
    let has_fe = !input.fe.is_empty();
    let intercept = spec.intercept && !input.fe.iter().any(|t| t.intercept);
    let iv = !input.endog.is_empty();
    let lay = Layout { r: 1 + input.endog.len() + usize::from(intercept) + input.inst.len() + input.x.len(), m: input.endog.len(), icpt: intercept, l: input.inst.len() };
    let r = lay.r;

    // Row block in solver row order, residualized on the fixed effects.
    let mut cols = vec![Col::Values(input.y)];
    cols.extend(input.endog.iter().map(|v| Col::Values(v)));
    if intercept {
        cols.push(Col::One);
    }
    cols.extend(input.inst.iter().chain(&input.x).map(|v| Col::Values(v)));
    let mut rows = session.gather_input(&cols, pool)?;
    crate::solve::log_phase(ctx, &mut t, "fit: gather");
    let (sumw, _, ssr_null) = y_stats(input.y, session.kept_index(), session.weights(), pool);
    ctx.check()?;
    let want_coefs = (spec.keep_fixef || spec.keep_state) && has_fe;
    let (fe_iterations, coefs) = if has_fe {
        let opts = SolveOpts { tol: spec.fixef_tol, maxiter: spec.fixef_maxiter, want_coefs };
        let info = session.absorber_mut().residualize(&mut rows, r, &opts, None, pool, ctx)?;
        (info.iterations, info.coefs)
    } else {
        (0, None)
    };
    let mut t = std::time::Instant::now();
    let w_rows = session.absorber().weights();
    let (g, _) = stats::gram(&rows, r, w_rows, pool);
    // pyfixest drops collinear columns before applying the weights: unweighted Gram.
    let g_u = (spec.flavor == Flavor::Pyfixest && w_rows.is_some()).then(|| stats::gram(&rows, r, None, pool).0);
    let gc = g_u.as_deref();
    crate::solve::log_phase(ctx, &mut t, "fit: gram");
    let tol = spec.collin_tol;
    let y = unit(r, 0);

    // Exogenous regressors in fixest order (intercept first).
    let mut exog: Vec<Regressor> = Vec::new();
    if intercept {
        exog.push(Regressor { source: CoefSource::Intercept, design: unit(r, lay.one()), actual: unit(r, lay.one()) });
    }
    exog.extend((0..input.x.len()).map(|j| Regressor { source: CoefSource::X(j), design: unit(r, lay.x(j)), actual: unit(r, lay.x(j)) }));
    let exog_design: Vec<Vec<f64>> = exog.iter().map(|x| x.design.clone()).collect();

    // IV first stages: each endogenous variable on Z = [(1), inst..., x...].
    struct First {
        reg: Reg,
        no_inst: Reg,
    }
    let mut z_sources: Vec<CoefSource> = Vec::new();
    let mut z_design: Vec<Vec<f64>> = Vec::new();
    let mut firsts: Vec<First> = Vec::new();
    // The second-stage regressors in fixest order: [(1), fit_endog..., x...].
    let mut second: Vec<Regressor> = Vec::new();
    if iv {
        if intercept {
            z_sources.push(CoefSource::Intercept);
            z_design.push(unit(r, lay.one()));
        }
        z_sources.extend((0..lay.l).map(CoefSource::Inst));
        z_design.extend((0..lay.l).map(|j| unit(r, lay.inst(j))));
        z_sources.extend((0..input.x.len()).map(CoefSource::X));
        z_design.extend((0..input.x.len()).map(|j| unit(r, lay.x(j))));
        for j in 0..lay.m {
            let target = unit(r, lay.endog(j));
            let reg = regress(&g, gc, r, &target, &z_design, tol)?;
            let no_inst = regress(&g, gc, r, &target, &exog_design, tol)?;
            firsts.push(First { reg, no_inst });
        }
        if intercept {
            second.push(exog.remove(0));
        }
        for (j, f) in firsts.iter().enumerate() {
            let target = unit(r, lay.endog(j));
            let fitted: Vec<f64> = target.iter().zip(&f.reg.resid).map(|(a, b)| a - b).collect();
            second.push(Regressor { source: CoefSource::Endog(j), design: fitted, actual: target });
        }
        second.append(&mut exog);
    } else {
        second = exog;
    }
    let design2: Vec<Vec<f64>> = second.iter().map(|x| x.design.clone()).collect();
    let reg2 = regress(&g, gc, r, &y, &design2, tol)?;
    if !design2.is_empty() && reg2.kept.is_empty() {
        return Err(RfError::InvalidInput("all the regressors are collinear (with the fixed effects)".into()));
    }
    // Corrected residual (IV: with the original endogenous variables) and uncorrected one.
    let mut e = y.clone();
    for (&j, &b) in reg2.kept.iter().zip(&reg2.coef) {
        e.iter_mut().zip(&second[j].actual).for_each(|(v, a)| *v -= b * a);
    }
    let e2 = reg2.resid.clone();

    // Auxiliary IV regressions: y on the exogenous regressors (F of the endogenous block),
    // Wu-Hausman (y on [endog..., (1), fit..., x...]) and Sargan (corrected residual on Z).
    let mut aux: Vec<Vec<f64>> = vec![e.clone()];
    let (mut wh_reg, mut sargan_reg) = (None, None);
    if iv {
        aux.push(e2.clone());
        aux.extend(firsts.iter().map(|f| f.reg.resid.clone()));
        aux.extend(firsts.iter().map(|f| f.no_inst.resid.clone()));
        let wh_design: Vec<Vec<f64>> = (0..lay.m).map(|j| unit(r, lay.endog(j))).chain(design2.iter().cloned()).collect();
        let wh = regress(&g, gc, r, &y, &wh_design, tol)?;
        aux.push(wh.resid.clone());
        wh_reg = Some(wh);
        if lay.l > lay.m {
            let s = regress(&g, gc, r, &e, &z_design, tol)?;
            aux.push(s.resid.clone());
            sargan_reg = Some(s);
        }
    }
    // The uncorrected residual gives the fitted values (and sq.cor); IV residuals are corrected.
    let keep_resid = spec.keep_resid;
    // Kept-order residuals: e2 (uncorrected; combo 1 for IV, equal to e = combo 0 otherwise)
    // always, the corrected IV residual e when kept.
    let mut e2_kept = par::alloc_vec(n, 0.0)?;
    let mut e_kept = if keep_resid && iv { par::alloc_vec(n, 0.0)? } else { Vec::new() };
    let mut store = vec![(usize::from(iv), e2_kept.as_mut_slice())];
    if keep_resid && iv {
        store.push((0, e_kept.as_mut_slice()));
    }
    let sums = residual_pass(&rows, r, w_rows, &aux, sargan_reg.is_some().then_some(0), store, session.row_order(), pool);
    crate::solve::log_phase(ctx, &mut t, "fit: residuals");
    ctx.check()?;
    let ssr = sums.wss[0];
    let sse = sums.uss[0];
    let m = lay.m;
    let (ssr_iv, ssr_first, ssr_no_inst) = if iv { (sums.wss[1], &sums.wss[2..2 + m], &sums.wss[2 + m..2 + 2 * m]) } else { (ssr, &[][..], &[][..]) };
    for j in 0..m {
        if ssr_first[j] < 1e-10 {
            return Err(RfError::InvalidInput(format!(
                "endogenous regressor {} is fully explained by the exogenous variables and the instruments",
                j + 1
            )));
        }
        if (ssr_first[j] - ssr_no_inst[j]).abs() < 1e-10 {
            return Err(RfError::InvalidInput(format!("the instruments have no explanatory power for endogenous regressor {}", j + 1)));
        }
    }

    // Reported coefficients: intercept, x..., endog... (the contract's order).
    let rank = |s: &CoefSource| match s {
        CoefSource::Intercept => (0, 0),
        CoefSource::X(j) => (1, *j),
        CoefSource::Endog(j) => (2, *j),
        CoefSource::Inst(j) => (3, *j),
    };
    let mut order: Vec<usize> = (0..reg2.kept.len()).collect();
    order.sort_by_key(|&a| rank(&second[reg2.kept[a]].source));
    let k = order.len();
    let kk = reg2.kept.len();
    let coef: Vec<f64> = order.iter().map(|&a| reg2.coef[a]).collect();
    let coef_source: Vec<CoefSource> = order.iter().map(|&a| second[reg2.kept[a]].source).collect();
    let inv2 = &reg2.inv;
    let bread: Vec<f64> = order.iter().flat_map(|&a| order.iter().map(move |&b| inv2[a * kk + b])).collect();
    let design_out: Vec<Vec<f64>> = order.iter().map(|&a| second[reg2.kept[a]].design.clone()).collect();
    let mut collinear: Vec<CoefSource> = second.iter().zip(&reg2.excluded).filter(|(_, x)| **x).map(|(s, _)| s.source).collect();
    if let Some(f) = firsts.first() {
        for (s, _) in z_sources.iter().zip(&f.reg.excluded).filter(|(_, x)| **x) {
            if !collinear.contains(s) {
                collinear.push(*s);
            }
        }
    }

    // Variance matrix (through the same state as post-hoc requests).
    let fweights = spec.flavor == Flavor::Pyfixest && spec.weights_type == WeightsType::Fweights && w_rows.is_some();
    let n_eff = if fweights { sumw } else { n as f64 };
    let np = nparams(k, &session);
    let terms = session.terms();
    let fe_names = |ft: &super::FeTerm| -> Vec<&str> { ft.factors.iter().map(|&f| input.factor_names[f].as_str()).collect() };
    let fe_info: Vec<FeInfo> =
        terms.iter().zip(&input.fe).map(|(t, ft)| FeInfo { codes: &t.codes, g: t.g, intercept: t.intercept, names: fe_names(ft) }).collect();
    let cl_info: Vec<ClusterInfo> = session
        .clusters()
        .iter()
        .zip(&input.cluster)
        .map(|(c, dim)| ClusterInfo { codes: &c.codes, g: c.g, names: dim.iter().map(|&f| input.factor_names[f].as_str()).collect() })
        .collect();
    let refs = || term_refs(&fe_info, pool);
    let model = Model {
        flavor: spec.flavor,
        n: n_eff,
        nparams: np,
        bread: &bread,
        iid_scale: ssr / (n_eff - 1.0),
        normal: false,
        rows: &rows,
        r,
        design: &design_out,
        resid: &e,
        w: w_rows,
        fweights,
        row_order: session.row_order(),
    };
    // Inputs of the HC2/HC3, HAC, Conley and CRV3 variance matrices.
    let kept_index = session.kept_index();
    let vdata = VcovData {
        factors: &input.factors,
        factor_names: &input.factor_names,
        cluster: &input.cluster,
        time: input.time,
        panel: input.panel,
        lat: input.lat,
        lon: input.lon,
    };
    let inputs = VcovInputs::read(spec.vcov, spec.flavor, &vdata, kept_index, pool)?;
    let hc = matches!(spec.vcov, VcovKind::Hc2 | VcovKind::Hc3);
    let fe_hat = if hc && has_fe {
        let terms: Vec<(&[u32], usize, bool, Vec<&[f64]>)> =
            terms.iter().map(|t| (t.codes.as_slice(), t.g, t.intercept, t.slopes.iter().map(Vec::as_slice).collect())).collect();
        Some(fe_leverage(&terms, session.weights(), n, ctx, pool)?)
    } else {
        None
    };
    let crv3 = spec.vcov == VcovKind::Crv3;
    let fe_cols = || -> Vec<FeCols> {
        terms.iter().map(|t| FeCols { codes: &t.codes, g: t.g, intercept: t.intercept, slopes: t.slopes.iter().map(Vec::as_slice).collect() }).collect()
    };
    // The 2SLS refits use the instruments the first stages kept.
    let z_kept: Vec<CoefSource> = firsts.first().map_or(Vec::new(), |f| f.reg.kept.iter().map(|&a| z_sources[a]).collect());
    let jk_data = crv3.then(|| {
        JkData::new(
            input.y,
            &pick(&coef_source, x_of, &input.x),
            &pick(&coef_source, endog_of, &input.endog),
            &pick(&z_kept, inst_of, &input.inst),
            None,
            session.weights(),
            fe_cols(),
            kept_index,
            pool,
        )
    });
    let jk_problem = jk_data.as_ref().map(|d| d.problem(jk_model(spec.model), intercept, &coef, spec));
    let jk_fn = jk_problem.as_ref().map(|p| move |codes: &[u32], g: usize| extra::jackknife_meat(p, codes, g, ctx, pool));
    let intercept_idx = coef_source.iter().position(|s| *s == CoefSource::Intercept);
    let mut extra2 = inputs.extra(has_fe, intercept_idx);
    extra2.fe_hat = fe_hat.as_deref();
    extra2.jackknife = jk_fn.as_ref().map(|f| f as &Jackknife);
    let (vcov, notes) = vcov::compute(&model, spec.vcov, &spec.ssc, &fe_info, &cl_info, &extra2, &refs, pool)?;
    crate::solve::log_phase(ctx, &mut t, "fit: vcov");

    // IV statistics.
    let nf = n as f64;
    let iv_stats = if iv {
        let f0 = &firsts[0].reg;
        let inst_kept: Vec<usize> = f0.kept.iter().enumerate().filter(|(_, j)| matches!(z_sources[**j], CoefSource::Inst(_))).map(|(a, _)| a).collect();
        let np1 = nparams(f0.kept.len(), &session);
        // pyfixest's first stage clusters by the first cluster variable only.
        let cl_first = if spec.flavor == Flavor::Pyfixest { &cl_info[..cl_info.len().min(1)] } else { &cl_info[..] };
        let mut first_stage = Vec::with_capacity(m);
        let mut df2_first = f64::NAN;
        for (j, f) in firsts.iter().enumerate() {
            let design1: Vec<Vec<f64>> = f.reg.kept.iter().map(|&a| z_design[a].clone()).collect();
            let model1 = Model { nparams: np1, bread: &f.reg.inv, iid_scale: ssr_first[j] / (n_eff - 1.0), design: &design1, resid: &f.reg.resid, ..model };
            // The first stage of endogenous variable j: [(1), inst..., x...] (kept) on endog_j.
            let sources1: Vec<CoefSource> = f.reg.kept.iter().map(|&a| z_sources[a]).collect();
            let jk_data1 = crv3.then(|| {
                let mut x1 = pick(&sources1, inst_of, &input.inst);
                x1.extend(pick(&sources1, x_of, &input.x));
                JkData::new(input.endog[j], &x1, &[], &[], None, session.weights(), fe_cols(), kept_index, pool)
            });
            let jk_problem1 = jk_data1.as_ref().map(|d| d.problem(jk_model(spec.model), intercept, &f.reg.coef, spec));
            let jk_fn1 = jk_problem1.as_ref().map(|p| move |codes: &[u32], g: usize| extra::jackknife_meat(p, codes, g, ctx, pool));
            let mut extra1 = inputs.extra(has_fe, sources1.iter().position(|s| *s == CoefSource::Intercept));
            extra1.jackknife = jk_fn1.as_ref().map(|f| f as &Jackknife);
            let (v1, _) = vcov::compute(&model1, spec.vcov, &spec.ssc, &fe_info, cl_first, &extra1, &refs, pool)?;
            if j == 0 {
                df2_first = (nf - v1.df_k).max(1.0);
            }
            let df1 = inst_kept.len() as f64;
            let stat = ((ssr_no_inst[j] - ssr_first[j]) / df1) / (ssr_first[j] / df2_first);
            let f_test = TestStat { stat, p: dist::pf_upper(stat, df1, df2_first), df1, df2: df2_first };
            let wald = first_stage_wald(spec.flavor, &f.reg.coef, &v1.vcov, &inst_kept, df1, df2_first);
            first_stage.push(IvFirstStage {
                endog: j,
                coef_source: f.reg.kept.iter().map(|&a| z_sources[a]).collect(),
                coef: f.reg.coef.clone(),
                vcov: v1,
                collinear: z_sources.iter().zip(&f.reg.excluded).filter(|(_, x)| **x).map(|(s, _)| *s).collect(),
                f_test,
                wald,
            });
        }
        let base = 2 + 2 * m;
        let wh = wh_reg.as_ref().expect("IV");
        let wu_hausman = if wh.excluded.contains(&true) {
            TestStat::NA
        } else {
            let df1 = m as f64;
            let df2 = nf - (np + m) as f64;
            let kw = wh.kept.len();
            let qui: Vec<usize> = (m + usize::from(intercept)..2 * m + usize::from(intercept)).collect();
            let s2 = sums.wss[base] / df2;
            let sub: Vec<f64> = qui.iter().flat_map(|&a| qui.iter().map(move |&b| wh.inv[a * kw + b] * s2)).collect();
            let (_, sub_inv) = crate::vcov::chol_collin(&sub, m, f64::MIN_POSITIVE);
            let c: Vec<f64> = qui.iter().map(|&a| wh.coef[a]).collect();
            let stat = quad_form(&c, &sub_inv) / df1;
            TestStat { stat, p: dist::pf_upper(stat, df1, df2), df1, df2 }
        };
        let sargan = if sargan_reg.is_some() {
            let df = (lay.l - m) as f64;
            let stat = nf * (1.0 - sums.wss[base + 1] / sums.centered);
            TestStat { stat, p: dist::pchisq_upper(stat, df), df1: df, df2: f64::NAN }
        } else {
            TestStat::NA
        };
        Some(IvStats { first_stage, wu_hausman, sargan, ssr_iv })
    } else {
        None
    };

    // Per-observation outputs (estimation order) and fit statistics.
    let kept_index = session.kept_index();
    let (fitted, cor2) = fitted_sq_cor(input.y, kept_index, &e2_kept, pool)?;
    let resid = match (keep_resid, iv) {
        (false, _) => None,
        (true, false) => Some(e2_kept),
        (true, true) => Some(e_kept),
    };
    let fitted = keep_resid.then_some(fitted);
    let weighted = w_rows.is_some();
    let ssr_fe_only = if has_fe { g[0] } else { f64::NAN };
    let only_intercept = np == 1 && intercept && k == 1;
    let n_stat = if spec.flavor == Flavor::Pyfixest { n_eff } else { nf };
    let di = if has_fe || intercept { 1.0 } else { 0.0 };
    // fixest's R^2 family uses the uncorrected second-stage residuals for IV, unweighted
    // (fitstats.R `r2`: `cpp_ssq(x$iv_residuals)`); its F test the weighted ones.
    let ssr_r2 = if iv { sums.uss[1] } else { ssr };
    let (mut r2, mut ar2) = match spec.flavor {
        _ if only_intercept && spec.flavor == Flavor::Fixest => (f64::NAN, f64::NAN),
        Flavor::Fixest => (1.0 - ssr_r2 / ssr_null * (nf - di) / (nf - 1.0), 1.0 - ssr_r2 / ssr_null * (nf - di) / (nf - np as f64)),
        Flavor::Pyfixest => {
            let hi = if spec.intercept { 1.0 } else { 0.0 };
            (1.0 - ssr / ssr_null, 1.0 - ssr / ssr_null * (n_eff - hi) / (n_eff - np as f64))
        }
    };
    let nb_fe = (np - k) as f64;
    let (mut wr2, mut war2) = if has_fe && k > 0 {
        (1.0 - ssr_r2 / ssr_fe_only, 1.0 - ssr_r2 / ssr_fe_only * (n_stat - nb_fe) / (n_stat - np as f64))
    } else {
        (f64::NAN, f64::NAN)
    };
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    let loglik = -0.5 * nf * (1.0 + ln2pi + (sse / nf).ln());
    let wn = if weighted { sumw } else { nf };
    let ll_null = -0.5 * wn * (1.0 + ln2pi + (ssr_null / wn).ln());
    let (pseudo_r2, apr2) =
        if only_intercept { (f64::NAN, f64::NAN) } else { (1.0 - loglik / ll_null, 1.0 - (loglik - np as f64 + 1.0) / ll_null) };
    let mut rmse = match spec.flavor {
        Flavor::Fixest => (ssr / wn).sqrt(),
        Flavor::Pyfixest => (ssr / n_eff).sqrt(),
    };
    if iv && spec.flavor == Flavor::Pyfixest {
        // pyfixest reports no R^2 or RMSE for IV.
        (r2, ar2, wr2, war2, rmse) = (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let f_test = match vcov.df_t {
        Some(df2) => {
            let df1 = vcov.df_k - 1.0;
            let stat = ((ssr_null - ssr_iv) / df1) / (ssr_iv / df2);
            TestStat { stat, p: dist::pf_upper(stat, df1, df2), df1, df2 }
        }
        None => TestStat::NA,
    };
    let stats = FitStats {
        ssr,
        ssr_null,
        ssr_fe_only,
        sigma2: ssr / (nf - np as f64),
        rmse,
        r2,
        ar2,
        wr2,
        war2,
        loglik,
        ll_null,
        deviance: ssr,
        null_deviance: ssr_null,
        pseudo_r2,
        apr2,
        sq_cor: cor2,
        f_test,
        fe_iterations,
        irls_iterations: 0,
        converged: true,
    };

    // Fixed effects (fixest `sumFE`): the FE part of y - X beta with the original endogenous
    // variables, i.e. of the corrected residual's combo. For IV this equals fixest's
    // y - Uhat beta - e_2 with Uhat the first-stage fitted values including the first-stage FE.
    let fe_levels = coefs.as_ref().map(|c| FeLevels { slots: session.absorber().term_values(&c.combine(&e), 0), subset: None });
    let obs = kept_index.to_vec();
    let fe_sizes: Vec<usize> = terms.iter().map(|t| t.g).collect();
    let row_order = if spec.keep_state { session.row_order().to_vec() } else { Vec::new() };
    let weights = session.weights.take();
    let state_fe = state_terms(std::mem::take(&mut session.terms), input, &fe_sizes);
    drop(session);
    let fixef = match &fe_levels {
        Some(l) if spec.keep_fixef => fixef::fixef_values(&state_fe, &obs, l, spec.fixef_norm, pool)?,
        _ => Vec::new(),
    };
    crate::solve::log_phase(ctx, &mut t, "fit: outputs");
    let state = spec.keep_state.then(|| super::FitState {
        inner: State {
            flavor: spec.flavor,
            n: n_eff,
            nparams: np,
            bread,
            iid_scale: ssr / (n_eff - 1.0),
            normal: false,
            rows,
            r,
            design: design_out,
            resid: e,
            weights,
            fweights,
            row_order,
            obs: obs.clone(),
            fe: state_fe,
            fixef: fe_levels,
            iv,
            intercept: intercept_idx,
            fe_hat: fe_hat.map(std::sync::OnceLock::from).unwrap_or_default(),
        },
    });
    Ok(FitOutput {
        nobs: n,
        nobs_origin: input.n,
        sumw,
        obs,
        dropped,
        coef_source,
        coef,
        collinear,
        vcov,
        nparams: np,
        fe_sizes,
        stats,
        iv: iv_stats,
        resid,
        fitted,
        eta: None,
        irls_weights: None,
        fixef,
        notes,
        state,
    })
}

#[cfg(test)]
mod tests {
    use super::design::cross;

    #[test]
    fn cross_matches_dense_product() {
        let g = [2.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 4.0];
        let a = [1.0, 0.0, -2.0];
        let b = [0.5, 1.0, 0.0];
        let dense: f64 = (0..3).map(|i| (0..3).map(|j| a[i] * g[i * 3 + j] * b[j]).sum::<f64>()).sum();
        assert!((cross(&g, 3, &a, &b) - dense).abs() < 1e-15);
    }
}
