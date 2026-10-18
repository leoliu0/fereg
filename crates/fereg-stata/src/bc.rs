//! felogit's bias corrections and average partial effects (logitfe 3.0.3 options): the `bc=` keys
//! of `cmd=glm` (see `PROTOCOL.md`). The uncorrected fit is `Session::glm_parts` (ppmlhdfe
//! convention); the kernels of `fereg_core::bias` then run on the session's absorber, and the
//! jackknife refits every subpanel through its own session with prepare's sample rules.

use fereg_core::absorber::fe1_index;
use fereg_core::bias::{
    self, ApeSpec, BiasSpec, FeCodes, FullSample, JkPanel, JkSpec, LogitData, LogitfePanel, Normalization, Permutation, Sampling, Structure, SubFit, Subpanel,
    Variant, WhichBias,
};
use fereg_core::{
    AbsorbTerm, AbsorberKind, Ctx, DofOptions, EstResult, FactorValues, Family, GlmConvention, GlmOptions, GlmParts, Layout, Prep, RfError, Session, Spec, Vce,
    WeightKind,
};

use crate::args::Args;
use crate::io;
use crate::store::Stored;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Ape,
    Analytical,
    Jackknife(Variant),
}

/// The `bc=` keys of one glm call.
#[derive(Debug)]
pub struct BcKeys {
    pub method: Method,
    pub network: bool,
    /// Classic: role of each absorbed term in absorb order (`b'i'` panel, `b't'` time).
    pub effects: Vec<u8>,
    pub which: WhichBias,
    pub lags: usize,
    pub multiple: usize,
    pub perm_i: bool,
    pub perm_t: bool,
    pub adj: f64,
    pub tdelta: f64,
    /// Number of bias-correction variables in the varlist.
    pub nbc: usize,
    pub b0: String,
    pub b2: String,
    pub v2: String,
    /// `xnames=` (jackknife): the sent regressors' names.
    pub xnames: Vec<String>,
}

fn bad<T>(m: impl Into<String>) -> Result<T, RfError> {
    Err(RfError::InvalidInput(format!("fereg plugin: {}", m.into())))
}

/// Parses the bias-correction keys; `None` without `bc=` (the other keys are then left unread,
/// so `Args::finish` rejects them).
pub fn keys(a: &Args, family: Family) -> Result<Option<BcKeys>, RfError> {
    let Some(bc) = a.str_opt("bc") else {
        return Ok(None);
    };
    if family != Family::Logit {
        return bad("bc= needs family=logit");
    }
    let jk = a.str_opt("jk");
    let method = match (bc, jk) {
        ("ape", None) => Method::Ape,
        ("analytical", None) => Method::Analytical,
        ("jackknife", Some(v)) => Method::Jackknife(match v {
            "ss1" => Variant::Ss1,
            "ss2" => Variant::Ss2,
            "js" => Variant::Js,
            "sj" => Variant::Sj,
            "jj" => Variant::Jj,
            "double" => Variant::Double,
            _ => return bad(format!("invalid jk `{v}`")),
        }),
        ("jackknife", None) => return bad("bc=jackknife needs jk="),
        ("ape" | "analytical", Some(_)) => return bad("jk= needs bc=jackknife"),
        _ => return bad(format!("invalid bc `{bc}` (ape|analytical|jackknife)")),
    };
    let network = match a.str_req("structure")? {
        "classic" => false,
        "network" => true,
        v => return bad(format!("invalid structure `{v}` (classic|network)")),
    };
    let effects: Vec<u8> = if network {
        Vec::new()
    } else {
        match a.str_req("effects")? {
            e @ ("it" | "ti" | "i" | "t") => e.bytes().collect(),
            v => return bad(format!("invalid effects `{v}` (it|ti|i|t)")),
        }
    };
    let which = match a.str_or("which", "both") {
        "both" => WhichBias::Both,
        "individual" => WhichBias::Individual,
        "time" => WhichBias::Time,
        v => return bad(format!("invalid which `{v}` (both|individual|time)")),
    };
    let lags: usize = a.num_or("lags", 0)?;
    let multiple: usize = a.num_or("multiple", 0)?;
    let (perm_i, perm_t) = if multiple > 0 {
        match a.str_req("perm")? {
            "it" => (true, true),
            "i" => (true, false),
            "t" => (false, true),
            v => return bad(format!("invalid perm `{v}` (it|i|t)")),
        }
    } else {
        (false, false)
    };
    let adj: f64 = a.num_or("adj", 1.0)?;
    let tdelta: f64 = if network { 1.0 } else { a.num_or("tdelta", 1.0)? };
    let nbc: usize = a.num_req("nbc")?;
    let (b0, b2, v2) = (a.str_req("b0")?.to_string(), a.str_req("b2")?.to_string(), a.str_req("V2")?.to_string());
    if network && matches!(method, Method::Jackknife(_)) {
        return bad("jk= is not available with structure=network");
    }
    if network && which != WhichBias::Both {
        return bad("structure=network needs which=both");
    }
    if lags > 0 && method != Method::Analytical {
        return bad("lags= needs bc=analytical");
    }
    if multiple > 0 && !matches!(method, Method::Jackknife(Variant::Ss1 | Variant::Ss2)) {
        return bad("multiple= needs jk=ss1 or jk=ss2");
    }
    if !(adj >= 0.0 && adj.is_finite()) || !(tdelta > 0.0 && tdelta.is_finite()) {
        return bad("adj= must be finite and >= 0, tdelta= finite and > 0");
    }
    let per = usize::from(perm_i) + usize::from(perm_t);
    if !network && nbc != 2 + multiple * per {
        return bad(format!("nbc={nbc} does not match the classic structure (2 + multiple x perm = {})", 2 + multiple * per));
    }
    // Regressor names (comma list, one per sent regressor): only for jackknife error messages.
    let xnames: Vec<String> = match method {
        Method::Jackknife(_) => a.str_req("xnames")?.split(',').map(str::to_string).collect(),
        _ => Vec::new(),
    };
    Ok(Some(BcKeys { method, network, effects, which, lags, multiple, perm_i, perm_t, adj, tdelta, nbc, b0, b2, v2, xnames }))
}

/// Dense codes of `v` (ascending values) and the number of distinct values.
fn dense(v: &[f64]) -> (Vec<u32>, usize) {
    let mut u = v.to_vec();
    u.sort_by(f64::total_cmp);
    u.dedup();
    (v.iter().map(|x| u.binary_search_by(|y| y.total_cmp(x)).expect("value present") as u32).collect(), u.len())
}

/// Integer time keys `(t - t0)/delta`; errors on values that are not on the grid.
fn periods(t: &[f64], t0: f64, delta: f64) -> Result<Vec<i64>, RfError> {
    t.iter()
        .map(|&v| {
            let p = (v - t0) / delta;
            let r = p.round();
            if (p - r).abs() > 1e-6 || r.abs() > 9.0e15 {
                bad("the time variable must hold integer values (in units of its xtset delta)")
            } else {
                Ok(r as i64)
            }
        })
        .collect()
}

/// Rank (0-based) of each unit's permuted id among the units: unit `u` has value `vals[u]`.
fn ranks(vals: &[f64]) -> Result<Vec<u32>, RfError> {
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    idx.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
    if idx.windows(2).any(|w| vals[w[0]] == vals[w[1]]) {
        return bad("multiple(): the permuted ids of a partition are not distinct");
    }
    let mut r = vec![0u32; vals.len()];
    for (k, &u) in idx.iter().enumerate() {
        r[u] = k as u32;
    }
    Ok(r)
}

/// The permuted id of each unit (its value at the unit's first row).
fn unit_values(codes: &[u32], g: usize, col: &[f64]) -> Vec<f64> {
    let mut v = vec![f64::NAN; g];
    for (c, x) in codes.iter().zip(col) {
        if v[*c as usize].is_nan() {
            v[*c as usize] = *x;
        }
    }
    v
}

/// Everything the subpanel refits need (data over the N* pre-drop rows).
struct SubData<'a> {
    y: &'a [f64],
    x: &'a [Vec<f64>],
    offset: Option<&'a [f64]>,
    /// Factor values per absorbed term (classic: the panel or the time variable).
    fe: Vec<&'a [f64]>,
    drop_singletons: bool,
    sep_fe: bool,
    kind: AbsorberKind,
    spec: Spec,
    glm: GlmOptions,
    binary: Vec<bool>,
    /// Stata names of the regressors (coefficient order), for error messages.
    names: Vec<String>,
    /// Pre-drop rows kept by the full fit (a subpanel keeping all of them has the full sample).
    full_kept: Vec<usize>,
    full: SubFit,
    n_full: usize,
}

impl SubData<'_> {
    fn fit(&self, keep: &[bool], ctx: &Ctx) -> Result<SubFit, RfError> {
        let rows: Vec<usize> = (0..keep.len()).filter(|&r| keep[r]).collect();
        let n_s = rows.len();
        // Same sample as the full fit (fereg's drops are iterated): no refit (core API sec. 3).
        if self.full_kept.iter().all(|&r| keep[r]) {
            let scale = self.n_full as f64 / n_s as f64;
            return Ok(SubFit { coef: self.full.coef.clone(), ape: self.full.ape.iter().map(|a| a * scale).collect() });
        }
        let sub = |v: &[f64]| rows.iter().map(|&r| v[r]).collect::<Vec<f64>>();
        let y = sub(self.y);
        let fac: Vec<Vec<f64>> = self.fe.iter().map(|v| sub(v)).collect();
        let prep = Prep {
            n: n_s,
            factors: fac.iter().map(|v| FactorValues::Num(v)).collect(),
            absorb: (0..fac.len()).map(|q| AbsorbTerm { factors: vec![q], intercept: true, slopes: vec![], save: false }).collect(),
            slope_vars: vec![],
            weights: None,
            weight_kind: WeightKind::None,
            vce: Vce::Unadjusted,
            drop_singletons: self.drop_singletons,
            family: Family::Logit,
            y_for_sep: Some(&y),
            sep_fe: self.sep_fe,
        };
        let mut s = Session::prepare_with(prep, ctx, self.kind)?;
        let kept: Vec<usize> = s.kept_index().iter().map(|&i| rows[i as usize]).collect();
        let at = |v: &[f64]| kept.iter().map(|&r| v[r]).collect::<Vec<f64>>();
        let mut cols = vec![at(self.y)];
        cols.extend(self.x.iter().map(|c| at(c)));
        let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
        let block = s.gather(&refs, ctx)?;
        let off = match self.offset {
            Some(o) => Some(s.gather(&[&at(o)], ctx)?),
            None => None,
        };
        let kx = self.x.len();
        let (_, p) = s.glm_parts(block, &Layout::ols(kx), off, &self.spec, &self.glm, GlmConvention::Ppmlhdfe, ctx)?;
        if let Some(j) = p.status.iter().position(|&st| st != 0) {
            return Err(RfError::InvalidInput(format!(
                "the coefficient of regressor {} ({}) is not estimable (collinear or a perfect predictor in this subpanel)",
                j + 1,
                self.names[j]
            )));
        }
        if !p.converged {
            return Err(RfError::Code(430, format!("no convergence in {} IRLS iterations", p.iterations)));
        }
        // Rows of the subpanel fit (its final kept sample) in kept order.
        let kept: Vec<usize> = s.kept_index().iter().map(|&i| rows[i as usize]).collect();
        let xs: Vec<Vec<f64>> = self.x.iter().map(|c| kept.iter().map(|&r| c[r]).collect()).collect();
        let xr: Vec<&[f64]> = xs.iter().map(Vec::as_slice).collect();
        Ok(SubFit { ape: bias::ape_point(&xr, &self.binary, &p.coef, &p.eta, n_s as f64), coef: p.coef })
    }
}

/// Inputs of [`run`] from the glm command.
pub struct GlmCall<'a> {
    pub nx: usize,
    pub has_offset: bool,
    pub spec: &'a Spec,
    pub glm: &'a GlmOptions,
    pub parallel: bool,
}

/// The bias correction / APEs after the uncorrected fit: overwrites `res`'s `b` (corrected
/// coefficients and `_cons` at the corrected point), `d` and `fe` (fixed effects re-estimated at
/// the corrected coefficients), appends the notes and the scalar `bc_nfits`, and stores the
/// Stata matrices `b0`, `b2`, `V2`.
pub fn run(bk: &BcKeys, stored: &mut Stored, res: &mut EstResult, parts: &GlmParts, call: &GlmCall, ctx: &Ctx) -> Result<(), RfError> {
    let nx = call.nx;
    let s = &mut stored.session;
    let n_full = stored.obs.len();
    let nterms = s.terms().len();
    if matches!(bk.method, Method::Jackknife(_)) && bk.xnames.len() != nx {
        return bad(format!("xnames= lists {} names for {nx} regressors", bk.xnames.len()));
    }
    if s.terms().iter().any(|t| !t.intercept || !t.slopes.is_empty()) {
        return bad("bias correction and APEs need intercept-only fixed effects (no varying slopes)");
    }
    if bk.network {
        if !(2..=3).contains(&nterms) {
            return bad("structure=network needs two or three absorbed terms");
        }
        if bk.nbc != nterms + usize::from(bk.lags > 0) {
            return bad(format!("nbc={} does not match the network structure", bk.nbc));
        }
    } else if bk.effects.len() != nterms {
        return bad("effects= must list one role per absorbed term");
    }

    // y, the regressors, the offset and the bias-correction variables over the N* rows.
    let first_bc = nx + 1 + usize::from(call.has_offset);
    let m = first_bc + bk.nbc;
    let vars: Vec<i32> = (1..=m as i32).collect();
    let blk = io::read_rows(&vars, &stored.obs, ctx.threads, call.parallel)?;
    if blk.iter().any(|v| v.is_nan()) {
        return Err(RfError::Code(416, "fereg: missing values in estimation variables of the sample".into()));
    }
    let col = |c: usize| (0..n_full).map(|r| blk[r * m + c]).collect::<Vec<f64>>();
    let y_full: Vec<f64> = col(0).iter().map(|&v| f64::from(u8::from(v != 0.0))).collect();
    let kept_x: Vec<usize> = (0..nx).filter(|&j| parts.status[j] == 0).collect();
    let x_full: Vec<Vec<f64>> = kept_x.iter().map(|&j| col(1 + j)).collect();
    let off_full = call.has_offset.then(|| col(nx + 1));
    let bcv: Vec<Vec<f64>> = (0..bk.nbc).map(|c| col(first_bc + c)).collect();
    drop(blk);

    // Kept rows (the final sample of the uncorrected fit), kept order.
    let kidx: Vec<usize> = s.kept_index().iter().map(|&i| i as usize).collect();
    let kept_of = |v: &[f64]| kidx.iter().map(|&r| v[r]).collect::<Vec<f64>>();
    let y = kept_of(&y_full);
    let xk: Vec<Vec<f64>> = x_full.iter().map(|c| kept_of(c)).collect();
    let off = off_full.as_deref().map(kept_of);
    let mut pre_kept = vec![u32::MAX; n_full];
    for (k, &r) in kidx.iter().enumerate() {
        pre_kept[r] = k as u32;
    }
    let coef: Vec<f64> = kept_x.iter().map(|&j| parts.coef[j]).collect();
    let fe_codes: Vec<(Vec<u32>, usize)> = s.terms().iter().map(|t| (t.codes.clone(), t.g)).collect();
    let d = LogitData {
        y: &y,
        weights: None,
        offset: off.as_deref(),
        x: xk.iter().map(Vec::as_slice).collect(),
        fe: fe_codes.iter().map(|(c, g)| FeCodes { codes: c, g: *g }).collect(),
        n_full,
    };

    // Panel identifiers. Classic: xtset panel and time variables; network: group codes per term.
    let structure = if bk.network { Structure::Network } else { Structure::Classic };
    let (id_full, n_pre, per_full, t_pre);
    let (id_kept, n_id_kept, cal_kept);
    let mut pre_fe: Vec<(Vec<u32>, usize)> = Vec::new();
    let mut order_kept: Option<Vec<i64>> = None;
    let mut order_full: Option<Vec<i64>> = None;
    if bk.network {
        for c in &bcv[..nterms] {
            pre_fe.push(dense(c));
        }
        if bk.lags > 0 {
            let t = &bcv[nterms];
            order_full = Some(periods(t, 0.0, 1.0)?);
            order_kept = Some(periods(&kept_of(t), 0.0, 1.0)?);
        }
        (id_full, n_pre, per_full, t_pre) = (Vec::new(), 0, Vec::new(), 0);
        (id_kept, n_id_kept, cal_kept) = (Vec::new(), 0, Vec::new());
    } else {
        let (iv, tv) = (&bcv[0], &bcv[1]);
        (id_full, n_pre) = dense(iv);
        (per_full, t_pre) = dense(tv);
        let t0 = tv.iter().copied().fold(f64::INFINITY, f64::min);
        let cal_full = periods(tv, t0, bk.tdelta)?;
        cal_kept = kidx.iter().map(|&r| cal_full[r]).collect::<Vec<i64>>();
        (id_kept, n_id_kept) = dense(&kept_of(iv));
        for &e in &bk.effects {
            pre_fe.push(if e == b'i' { (id_full.clone(), n_pre) } else { (per_full.clone(), t_pre) });
        }
        order_full = Some(cal_full);
    }
    let logitfe = (!bk.network).then(|| LogitfePanel {
        individual: &id_kept,
        n_individuals: n_id_kept,
        period: &cal_kept,
        individual_effects: bk.effects.contains(&b'i'),
        time_effects: bk.effects.contains(&b't'),
    });
    // The lag-group term (core API rev2): classic, the xtset panel variable's term in absorb order
    // (none in a time-effects-only model, which has no lags); network, the pair term (third).
    let lag_term = if bk.network { (nterms == 3).then_some(2) } else { bk.effects.iter().position(|&e| e == b'i') };
    let spec = BiasSpec {
        normalization: if bk.network { Normalization::Alpaca } else { Normalization::Logitfe },
        structure,
        which: bk.which,
        bandwidth: bk.lags,
        lag_term,
        order: order_kept.as_deref(),
        logitfe,
    };
    let full = FullSample {
        kept: &pre_kept,
        fe: pre_fe.iter().map(|(c, g)| FeCodes { codes: c, g: *g }).collect(),
        order: order_full.as_deref(),
    };

    // 1. APEs and their variance at the uncorrected fit (e(V2) in every case, as logitfe).
    let unc = bias::apes(
        &d,
        s.absorber_mut(),
        &coef,
        &parts.eta,
        &ApeSpec {
            structure,
            sampling: Sampling::Independence,
            weak_exo: bk.lags > 0,
            lag_term,
            adj: bk.adj,
            correction: None,
            full: Some(full),
            want_vcov: true,
        },
        ctx,
    )?;

    // 2. The correction: coefficients, linear predictor and FE values at the corrected point.
    let want_fe = !res.fe.is_empty();
    let mut corrected: Option<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>)> = None;
    let mut ape = unc.delta.clone();
    let mut nfits = 0usize;
    match bk.method {
        Method::Ape => {}
        Method::Analytical => {
            let c = bias::analytical(&d, s.absorber_mut(), &coef, &parts.eta, &spec, ctx)?;
            let fev = if want_fe { bias::saved_fe(s.absorber(), s.terms(), &c.fe_coefs) } else { Vec::new() };
            let a = ApeSpec {
                structure,
                sampling: Sampling::Independence,
                weak_exo: false,
                adj: 0.0,
                correction: Some(spec),
                full: None,
                want_vcov: false,
                lag_term,
            };
            ape = bias::apes(&d, s.absorber_mut(), &c.coef, &c.eta, &a, ctx)?.delta;
            res.notes.extend(c.notes);
            corrected = Some((c.coef, c.eta, fev));
        }
        Method::Jackknife(variant) => {
            let (dbl1, dbl2, nj);
            let double = if variant == Variant::Double {
                let (iv, tv) = (&bcv[0], &bcv[1]);
                let uniq = |v: &[f64]| {
                    let mut u = v.to_vec();
                    u.sort_by(f64::total_cmp);
                    u.dedup();
                    u
                };
                let ut = uniq(tv);
                let common: Vec<f64> = uniq(iv).into_iter().filter(|v| ut.binary_search_by(|t| t.total_cmp(v)).is_ok()).collect();
                if common.is_empty() {
                    return Err(RfError::InvalidInput(
                        "Invalid -double- option: no observations with the same index for the panel and time variables".into(),
                    ));
                }
                let pos = |v: &f64| common.binary_search_by(|c| c.total_cmp(v)).map_or(u32::MAX, |k| k as u32);
                dbl1 = iv.iter().map(pos).collect::<Vec<u32>>();
                dbl2 = tv.iter().map(pos).collect::<Vec<u32>>();
                nj = common.len();
                Some((dbl1.as_slice(), dbl2.as_slice(), nj))
            } else {
                None
            };
            let panel = JkPanel { individual: Some(&id_full), n_individuals: n_pre, period: Some(&per_full), n_periods: t_pre, double };
            // multiple(R): partition r's permuted ids follow the panel and time variables.
            let per = usize::from(bk.perm_i) + usize::from(bk.perm_t);
            let mut perms = Vec::with_capacity(bk.multiple);
            for r in 0..bk.multiple {
                let mut c = 2 + r * per;
                let mut p = Permutation::default();
                if bk.perm_i {
                    p.individual = Some(ranks(&unit_values(&id_full, n_pre, &bcv[c]))?);
                    c += 1;
                }
                if bk.perm_t {
                    p.period = Some(ranks(&unit_values(&per_full, t_pre, &bcv[c]))?);
                }
                perms.push(p);
            }
            let fe_vals: Vec<&[f64]> = bk.effects.iter().map(|&e| if e == b'i' { bcv[0].as_slice() } else { bcv[1].as_slice() }).collect();
            let sub = SubData {
                y: &y_full,
                x: &x_full,
                offset: off_full.as_deref(),
                fe: fe_vals,
                drop_singletons: stored.drop_singletons,
                sep_fe: stored.sep_fe,
                kind: s.absorber_kind(),
                spec: Spec { want_resid: false, want_d: false, want_fe: false, want_groupvar: false, dof: DofOptions::none(), ..call.spec.clone() },
                glm: GlmOptions { log: false, ..call.glm.clone() },
                binary: bias::binary(&d.x),
                names: kept_x.iter().map(|&j| bk.xnames[j].clone()).collect(),
                full_kept: kidx.clone(),
                full: SubFit { coef: coef.clone(), ape: unc.delta.clone() },
                n_full,
            };
            let bytes_per_fit = n_full.saturating_mul(x_full.len() + nterms + 8).saturating_mul(8 * 6);
            let jk = bias::jackknife(
                &panel,
                &JkSpec { variant, which: bk.which, partitions: &perms, bytes_per_fit },
                &sub.full,
                |_: &Subpanel, keep: &[bool], fctx: &Ctx| sub.fit(keep, fctx),
                ctx,
            )
            .map_err(|e| match e {
                RfError::Interrupted | RfError::OutOfMemory => e,
                e if e.to_string().starts_with("jackknife, subpanel") => RfError::Code(498, e.to_string()),
                e => e,
            })?;
            // FE re-estimation at the jackknife coefficients, from eta + X(beta_J - beta).
            let start: Vec<f64> = (0..y.len())
                .map(|i| parts.eta[i] + xk.iter().zip(jk.coef.iter().zip(&coef)).map(|(x, (bj, b))| x[i] * (bj - b)).sum::<f64>())
                .collect();
            let r = bias::resolve_fe(&d, s.absorber_mut(), &jk.coef, &start, ctx)?;
            let fev = if want_fe { bias::saved_fe(s.absorber(), s.terms(), &r.fe_coefs) } else { Vec::new() };
            ape = jk.ape;
            nfits = jk.n_fits;
            res.notes.extend(jk.notes);
            corrected = Some((jk.coef, r.eta, fev));
        }
    }

    // 3. Results. b0 = the uncorrected b; b gets the corrected coefficients and the constant at
    // the corrected point; d and the saved FEs follow.
    let bm = res.matrices.iter_mut().find(|m| m.name == "b").ok_or_else(|| RfError::Internal("glm returned no b".into()))?;
    if bm.data.len() != nx + 1 {
        return Err(RfError::Internal("glm b has the wrong size".into()));
    }
    let b0 = bm.data.clone();
    if let Some((cc, eta_c, fev)) = &corrected {
        // fe_sum = eta_c - offset - x'beta_c; _cons = its mean under the logit weights mu(1 - mu).
        let fe_sum: Vec<f64> = (0..y.len())
            .map(|i| eta_c[i] - off.as_ref().map_or(0.0, |o| o[i]) - xk.iter().zip(cc).map(|(x, b)| x[i] * b).sum::<f64>())
            .collect();
        let (mut sw, mut swf) = (0.0, 0.0);
        for (e, f) in eta_c.iter().zip(&fe_sum) {
            let mu = 1.0 / (1.0 + (-e).exp());
            let w = mu * (1.0 - mu);
            sw += w;
            swf += w * f;
        }
        let cons = swf / sw;
        for (k, &j) in kept_x.iter().enumerate() {
            bm.data[j] = cc[k];
        }
        bm.data[nx] = cons;
        if res.d.is_some() {
            res.d = Some(fe_sum.iter().map(|f| f - cons).collect());
        }
        if want_fe {
            let shift = fe1_index(s.terms()).or_else(|| s.terms().iter().position(|t| t.intercept));
            res.fe = s
                .terms()
                .iter()
                .enumerate()
                .filter(|(_, t)| t.save)
                .map(|(q, _)| if Some(q) == shift { fev[q].iter().map(|v| v - cons).collect() } else { fev[q].clone() })
                .collect();
        }
    }
    let mut b2 = vec![0.0; nx];
    let mut v2 = vec![0.0; nx * nx];
    let kk = kept_x.len();
    for (a, &ja) in kept_x.iter().enumerate() {
        b2[ja] = ape[a];
        for (c, &jc) in kept_x.iter().enumerate() {
            v2[ja * nx + jc] = unc.vcov[a * kk + c];
        }
    }
    io::store_matrix(&bk.b0, 1, nx + 1, &b0)?;
    io::store_matrix(&bk.b2, 1, nx, &b2)?;
    io::store_matrix(&bk.v2, nx, nx, &v2)?;
    res.set_scalar("bc_nfits", nfits as f64);
    Ok(())
}
