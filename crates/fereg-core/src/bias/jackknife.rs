//! Split-panel and delete-one jackknife (design sec. 7): the subpanel plan of every variant and
//! bias choice, the subpanel fits through a host closure (concurrent, deterministic), and the
//! combination of coefficients and APEs.
//!
//! Subpanels are keep-masks over the pre-drop rows (the N* rows). Individuals are 0..N_pre and
//! periods 0..T_pre (time order) over those rows, units later dropped for a constant outcome
//! included. Halves of m units: the first holds 1-based indices <= ceil(m/2), the second those
//! > m/2 (m even) or >= ceil(m/2) (m odd: the middle unit is in both), as logitfe.
//!
//! Concurrency follows `crv3::refits`: one subpanel fit per worker, each with `threads / workers`
//! threads, subpanels claimed in ascending order; after a failure no higher subpanel starts, so the
//! error reported (the lowest failing subpanel) is deterministic; every fit depends only on its
//! mask, so the result is the same for any thread count when the closure is.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use super::{Variant, WhichBias, invalid};
use crate::api::Ctx;
use crate::error::RfError;
use crate::par;

/// One partition of `multiple(R)`: `individual[i]` is the new 0-based index of individual i,
/// `period[t]` that of period t; `None` = that dimension is not permuted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Permutation {
    pub individual: Option<Vec<u32>>,
    pub period: Option<Vec<u32>>,
}

/// Panel indices of the pre-drop rows.
#[derive(Clone, Copy, Debug, Default)]
pub struct JkPanel<'a> {
    /// Individual index of every pre-drop row (below `n_individuals`); `None` when the model has no
    /// individual dimension.
    pub individual: Option<&'a [u32]>,
    pub n_individuals: usize,
    /// Period index (time order, below `n_periods`).
    pub period: Option<&'a [u32]>,
    pub n_periods: usize,
    /// [`Variant::Double`]: per pre-drop row, the index of its FE1 value and of its FE2 value among
    /// the J ids that occur as both (`u32::MAX` when the value is not one of them), and J.
    pub double: Option<(&'a [u32], &'a [u32], usize)>,
}

/// The jackknife request.
#[derive(Clone, Copy, Debug)]
pub struct JkSpec<'a> {
    pub variant: Variant,
    /// Already resolved against the model (one FE term: only its bias).
    pub which: WhichBias,
    /// `multiple(R)` partitions (ss1 / ss2 only); empty = none.
    pub partitions: &'a [Permutation],
    /// Working memory of one subpanel fit, to cap concurrent fits at 16 GiB (0 = no cap).
    pub bytes_per_fit: usize,
}

/// Estimates of one fit: coefficients and APEs (APE divided by the fit's own N*).
#[derive(Clone, Debug, PartialEq)]
pub struct SubFit {
    pub coef: Vec<f64>,
    pub ape: Vec<f64>,
}

/// A subpanel handed to the fit closure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Subpanel {
    /// 0-based position in the engine's list.
    pub index: usize,
    /// Name used in error messages, e.g. "individual half 1", "without period 3".
    pub label: String,
}

/// Result of [`jackknife`].
#[derive(Clone, Debug, PartialEq)]
pub struct JkOut {
    pub coef: Vec<f64>,
    pub ape: Vec<f64>,
    /// Subpanel fits run.
    pub n_fits: usize,
    pub notes: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Individual half (1 or 2).
    Ih(u8),
    /// Global time half.
    Th(u8),
    /// Per-individual time half.
    Ph(u8),
    /// Quadrant (individual half, time half).
    Q(u8, u8),
    DropInd(u32),
    DropPer(u32),
    DropId(u32),
}

#[derive(Clone, Copy, Debug)]
struct Sub {
    kind: Kind,
    part: Option<usize>,
}

/// estimate = c * b_full - sum_g weight_g * mean(fits of group g).
struct Plan {
    c: f64,
    groups: Vec<(f64, Vec<Kind>)>,
}

fn halves(f: fn(u8) -> Kind) -> Vec<Kind> {
    vec![f(1), f(2)]
}

fn plan(variant: Variant, which: WhichBias, multiple: bool, n: usize, t: usize, j: usize) -> Plan {
    use WhichBias::*;
    let (nf, tf, jf) = (n as f64, t as f64, j as f64);
    let drop_ind = || (0..n as u32).map(Kind::DropInd).collect::<Vec<_>>();
    let drop_per = || (0..t as u32).map(Kind::DropPer).collect::<Vec<_>>();
    let one = |c: f64, g: Vec<(f64, Vec<Kind>)>| Plan { c, groups: g };
    match (variant, which) {
        (Variant::Ss2, Both) => one(3.0, vec![(1.0, halves(Kind::Ih)), (1.0, halves(Kind::Ph))]),
        (Variant::Ss2, Individual) if multiple => one(2.0, vec![(1.0, halves(Kind::Th))]),
        (Variant::Ss2, Individual) => one(2.0, vec![(1.0, halves(Kind::Ph))]),
        (Variant::Ss1, Both) => one(2.0, vec![(1.0, vec![Kind::Q(1, 1), Kind::Q(1, 2), Kind::Q(2, 1), Kind::Q(2, 2)])]),
        (Variant::Ss1 | Variant::Js, Individual) => one(2.0, vec![(1.0, halves(Kind::Th))]),
        (Variant::Ss2 | Variant::Ss1 | Variant::Sj, Time) => one(2.0, vec![(1.0, halves(Kind::Ih))]),
        (Variant::Js, Both) => one(nf + 1.0, vec![(nf - 1.0, drop_ind()), (1.0, halves(Kind::Th))]),
        (Variant::Js | Variant::Jj | Variant::Double, Time) => one(nf, vec![(nf - 1.0, drop_ind())]),
        (Variant::Sj, Both) => one(tf + 1.0, vec![(tf - 1.0, drop_per()), (1.0, halves(Kind::Ih))]),
        (Variant::Sj | Variant::Jj | Variant::Double, Individual) => one(tf, vec![(tf - 1.0, drop_per())]),
        (Variant::Jj, Both) => one(nf + tf - 1.0, vec![(nf - 1.0, drop_ind()), (tf - 1.0, drop_per())]),
        (Variant::Double, Both) => one(jf, vec![(jf - 1.0, (0..j as u32).map(Kind::DropId).collect())]),
    }
}

/// Dimensions a kind splits on: (individual, period).
fn uses(k: Kind) -> (bool, bool) {
    match k {
        Kind::Ih(_) | Kind::DropInd(_) => (true, false),
        Kind::Th(_) | Kind::Ph(_) | Kind::DropPer(_) => (false, true),
        Kind::Q(..) => (true, true),
        Kind::DropId(_) => (false, false),
    }
}

/// Needs: (individual ids, period ids, double ids).
fn needs(k: Kind) -> (bool, bool, bool) {
    match k {
        Kind::Ih(_) | Kind::DropInd(_) => (true, false, false),
        Kind::Th(_) | Kind::DropPer(_) => (false, true, false),
        Kind::Ph(_) | Kind::Q(..) => (true, true, false),
        Kind::DropId(_) => (false, false, true),
    }
}

fn label(s: &Sub) -> String {
    let base = match s.kind {
        Kind::Ih(a) => format!("individual half {a}"),
        Kind::Th(b) => format!("time half {b}"),
        Kind::Ph(a) => format!("per-individual time half {a}"),
        Kind::Q(a, b) => format!("quadrant (individual half {a}, time half {b})"),
        Kind::DropInd(i) => format!("without individual {}", i + 1),
        Kind::DropPer(t) => format!("without period {}", t + 1),
        Kind::DropId(c) => format!("without id {}", c + 1),
    };
    match s.part {
        Some(r) => format!("{base} of partition {}", r + 1),
        None => base,
    }
}

/// Whether 0-based unit `idx` of `m` lies in half `a`.
fn in_half(idx: u32, m: usize, a: u8) -> bool {
    let one = idx as usize + 1;
    let ceil = m.div_ceil(2);
    if a == 1 { one <= ceil } else if m % 2 == 0 { one > m / 2 } else { one >= ceil }
}

fn check_perm(p: &[u32], m: usize, what: &str) -> Result<(), RfError> {
    let mut seen = vec![false; m];
    if p.len() != m || p.iter().any(|&v| v as usize >= m || std::mem::replace(&mut seen[v as usize], true)) {
        return invalid(format!("jackknife: every {what} permutation must be a permutation of 0..{m}"));
    }
    Ok(())
}

/// Keep-mask of a subpanel over the pre-drop rows.
fn mask(s: &Sub, panel: &JkPanel, parts: &[Permutation], n_rows: usize) -> Vec<bool> {
    let perm = s.part.map(|r| &parts[r]);
    let ind = |r: usize| {
        let i = panel.individual.expect("checked")[r];
        perm.and_then(|p| p.individual.as_ref()).map_or(i, |p| p[i as usize])
    };
    let per = |r: usize| {
        let t = panel.period.expect("checked")[r];
        perm.and_then(|p| p.period.as_ref()).map_or(t, |p| p[t as usize])
    };
    let (n, t) = (panel.n_individuals, panel.n_periods);
    match s.kind {
        Kind::Ih(a) => (0..n_rows).map(|r| in_half(ind(r), n, a)).collect(),
        Kind::Th(b) => (0..n_rows).map(|r| in_half(per(r), t, b)).collect(),
        Kind::Q(a, b) => (0..n_rows).map(|r| in_half(ind(r), n, a) && in_half(per(r), t, b)).collect(),
        Kind::DropInd(i) => (0..n_rows).map(|r| panel.individual.expect("checked")[r] != i).collect(),
        Kind::DropPer(q) => (0..n_rows).map(|r| panel.period.expect("checked")[r] != q).collect(),
        Kind::DropId(c) => {
            let (a, b, _) = panel.double.expect("checked");
            (0..n_rows).map(|r| a[r] != c && b[r] != c).collect()
        }
        Kind::Ph(a) => {
            // Rows of each individual in (permuted) period order, ties in row order.
            let indiv = panel.individual.expect("checked");
            let (off, ids) = super::groups(indiv, n);
            let mut keep = vec![false; n_rows];
            let mut rows: Vec<u32> = Vec::new();
            for i in 0..n {
                rows.clear();
                rows.extend_from_slice(&ids[off[i]..off[i + 1]]);
                rows.sort_by_key(|&r| per(r as usize));
                let ti = rows.len();
                for (pos, &r) in rows.iter().enumerate() {
                    keep[r as usize] = in_half(pos as u32, ti, a);
                }
            }
            keep
        }
    }
}

/// Working-memory budget of concurrent subpanel fits.
const FIT_MEMORY: usize = 16 << 30;

fn discard(_: &str) {}

/// Adds the subpanel to the message of a fit error.
fn sub_error(e: RfError, label: &str) -> RfError {
    let at = |m: &str| format!("jackknife, subpanel {label}: {m}");
    match e {
        RfError::InvalidInput(m) => RfError::InvalidInput(at(&m)),
        RfError::TooFewObs => RfError::InvalidInput(at("insufficient observations")),
        RfError::Convergence(it) => RfError::Code(430, at(&format!("convergence not achieved in {it} iterations"))),
        RfError::Code(rc, m) => RfError::Code(rc, at(&m)),
        RfError::Internal(m) => RfError::Internal(at(&m)),
        e => e,
    }
}

/// The jackknife estimate of `spec.variant` / `spec.which` (design sec. 7) from the full-sample
/// estimates `full` (coefficients and APEs of the uncorrected fit) and the subpanel fits of
/// `fit`, called with the subpanel, its keep-mask over the pre-drop rows and a context whose
/// `threads` the fit should use. A closure error is returned prefixed with the subpanel.
pub fn jackknife<F>(panel: &JkPanel, spec: &JkSpec, full: &SubFit, fit: F, ctx: &Ctx) -> Result<JkOut, RfError>
where
    F: Fn(&Subpanel, &[bool], &Ctx) -> Result<SubFit, RfError> + Sync,
{
    let n_rows = panel.individual.or(panel.period).map(<[u32]>::len).or(panel.double.map(|d| d.0.len()));
    let Some(n_rows) = n_rows else { return invalid("jackknife: no panel dimension given") };
    if panel.individual.is_some_and(|v| v.len() != n_rows || v.iter().any(|&c| c as usize >= panel.n_individuals))
        || panel.period.is_some_and(|v| v.len() != n_rows || v.iter().any(|&c| c as usize >= panel.n_periods))
        || panel.double.is_some_and(|(a, b, j)| a.len() != n_rows || b.len() != n_rows || a.iter().chain(b).any(|&c| c != u32::MAX && c as usize >= j))
    {
        return invalid("jackknife: panel indices must have one entry per pre-drop row, below their counts");
    }
    if full.coef.len() != full.ape.len() {
        return invalid("jackknife: one APE per coefficient is required");
    }
    let parts = spec.partitions;
    let notes = Vec::new();
    if !parts.is_empty() && !matches!(spec.variant, Variant::Ss1 | Variant::Ss2) {
        return invalid("jackknife: multiple partitions are only available with the ss1 and ss2 variants");
    }
    for p in parts {
        if let Some(v) = &p.individual {
            check_perm(v, panel.n_individuals, "individual")?;
        }
        if let Some(v) = &p.period {
            check_perm(v, panel.n_periods, "period")?;
        }
    }
    let j_double = panel.double.map_or(0, |d| d.2);
    let multiple = !parts.is_empty();
    let pl = plan(spec.variant, spec.which, multiple, panel.n_individuals, panel.n_periods, j_double);
    if multiple {
        let (ui, ut) = pl.groups.iter().flat_map(|g| g.1.iter()).fold((false, false), |a, &k| (a.0 || uses(k).0, a.1 || uses(k).1));
        let effective = parts.iter().any(|p| (ui && p.individual.is_some()) || (ut && p.period.is_some()));
        if !effective {
            let dim = match (ui, ut) {
                (true, true) => "individual and time dimensions",
                (true, false) => "individual dimension",
                _ => "time dimension",
            };
            return invalid(format!("multiple partitions have no effect: the subpanels of this bias split only the {dim}, which no partition permutes"));
        }
    }
    for k in pl.groups.iter().flat_map(|g| g.1.iter()) {
        let (ni, np, nd) = needs(*k);
        if ni && panel.individual.is_none() {
            return invalid("jackknife: this variant needs the individual identifier");
        }
        if np && panel.period.is_none() {
            return invalid("jackknife: this variant needs a time variable");
        }
        if nd && panel.double.is_none() {
            return invalid("jackknife: the double variant needs the fixed-effect ids");
        }
    }
    if spec.variant == Variant::Double && spec.which == WhichBias::Both && j_double < 2 {
        return invalid("jackknife: the double variant needs at least two ids that occur in both fixed effects");
    }
    let mut subs = Vec::new();
    let nparts = if multiple { parts.len() } else { 1 };
    for r in 0..nparts {
        for g in &pl.groups {
            for &k in &g.1 {
                subs.push(Sub { kind: k, part: multiple.then_some(r) });
            }
        }
    }
    let p = full.coef.len();
    let fits = run(&subs, panel, parts, n_rows, p, spec.bytes_per_fit, &fit, ctx)?;
    // Combine per partition (subpanels are listed partition by partition, groups in plan order).
    let per_part: usize = pl.groups.iter().map(|g| g.1.len()).sum();
    let mut coef = vec![0.0; p];
    let mut ape = vec![0.0; p];
    for r in 0..nparts {
        let mut at = r * per_part;
        let mut c: Vec<f64> = full.coef.iter().map(|b| pl.c * b).collect();
        let mut a: Vec<f64> = full.ape.iter().map(|b| pl.c * b).collect();
        for (wgt, kinds) in &pl.groups {
            let m = kinds.len() as f64;
            let mut mc = vec![0.0; p];
            let mut ma = vec![0.0; p];
            for f in &fits[at..at + kinds.len()] {
                mc.iter_mut().zip(&f.coef).for_each(|(s, v)| *s += v);
                ma.iter_mut().zip(&f.ape).for_each(|(s, v)| *s += v);
            }
            at += kinds.len();
            c.iter_mut().zip(&mc).for_each(|(x, s)| *x -= wgt * s / m);
            a.iter_mut().zip(&ma).for_each(|(x, s)| *x -= wgt * s / m);
        }
        coef.iter_mut().zip(&c).for_each(|(x, v)| *x += v / nparts as f64);
        ape.iter_mut().zip(&a).for_each(|(x, v)| *x += v / nparts as f64);
    }
    Ok(JkOut { coef, ape, n_fits: subs.len(), notes })
}

/// Runs every subpanel fit (results in subpanel order).
#[allow(clippy::too_many_arguments)]
fn run<F>(subs: &[Sub], panel: &JkPanel, parts: &[Permutation], n_rows: usize, p: usize, bytes: usize, fit: &F, ctx: &Ctx) -> Result<Vec<SubFit>, RfError>
where
    F: Fn(&Subpanel, &[bool], &Ctx) -> Result<SubFit, RfError> + Sync,
{
    let count = subs.len();
    let threads = par::resolve_threads(ctx.threads);
    let cap = FIT_MEMORY.checked_div(bytes).map_or(usize::MAX, |c| c.max(1));
    let workers = threads.min(count).min(cap).max(1);
    let one = |i: usize, fctx: &Ctx| -> Result<SubFit, RfError> {
        let s = &subs[i];
        let lab = label(s);
        let m = mask(s, panel, parts, n_rows);
        let out = fit(&Subpanel { index: i, label: lab.clone() }, &m, fctx).map_err(|e| sub_error(e, &lab))?;
        if out.coef.len() != p || out.ape.len() != p {
            return Err(RfError::InvalidInput(format!("jackknife, subpanel {lab}: the fit returned the wrong number of estimates")));
        }
        Ok(out)
    };
    if workers == 1 {
        let fctx = Ctx { threads, verbose: 0, interrupt: ctx.interrupt, log: &discard };
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            ctx.check()?;
            out.push(one(i, &fctx)?);
        }
        return Ok(out);
    }
    let next = AtomicUsize::new(0);
    let failed = AtomicUsize::new(usize::MAX);
    let stop = AtomicBool::new(false);
    type Done = Vec<(usize, Result<SubFit, RfError>)>;
    let work = |tid: usize, poll: &dyn Fn() -> bool| -> Done {
        let t = (threads / workers + usize::from(tid < threads % workers)).max(1);
        let cur = Cell::new(0usize);
        let interrupt = || stop.load(Ordering::Relaxed) || cur.get() > failed.load(Ordering::Relaxed) || poll();
        let fctx = Ctx { threads: t, verbose: 0, interrupt: &interrupt, log: &discard };
        let mut done = Vec::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= count || i > failed.load(Ordering::Relaxed) || stop.load(Ordering::Relaxed) {
                return done;
            }
            cur.set(i);
            let res = one(i, &fctx);
            if res.is_err() {
                failed.fetch_min(i, Ordering::Relaxed);
            }
            done.push((i, res));
        }
    };
    let interrupted = AtomicBool::new(false);
    let main_poll = || {
        let i = (ctx.interrupt)();
        if i {
            interrupted.store(true, Ordering::Relaxed);
            stop.store(true, Ordering::Relaxed);
        }
        i
    };
    let results: Vec<Done> = std::thread::scope(|s| {
        let work = &work;
        let handles: Vec<_> = (1..workers).map(|tid| par::spawn_scoped(s, move || work(tid, &|| false))).collect();
        let mine = work(0, &main_poll);
        while !handles.iter().all(|h| h.is_finished()) {
            main_poll();
            std::thread::sleep(Duration::from_millis(1));
        }
        let mut all = vec![mine];
        for h in handles {
            all.push(h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)));
        }
        all
    });
    if interrupted.load(Ordering::Relaxed) {
        return Err(RfError::Interrupted);
    }
    let mut slots: Vec<Option<Result<SubFit, RfError>>> = (0..count).map(|_| None).collect();
    for done in results {
        for (i, res) in done {
            slots[i] = Some(res);
        }
    }
    let f = failed.load(Ordering::Relaxed);
    if f < count {
        if let Some(Some(Err(e))) = slots.get_mut(f).map(Option::take) {
            return Err(e);
        }
        return Err(RfError::Internal("jackknife: lost the error of a failed subpanel fit".into()));
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(i, s)| match s {
            Some(r) => r,
            None => Err(RfError::Internal(format!("jackknife: subpanel {i} was not fitted"))),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_follow_logitfe() {
        // m = 5: first 1..=3, second 3..=5; m = 4: first 1..=2, second 3..=4.
        let h = |m: usize, a: u8| (0..m as u32).filter(|&i| in_half(i, m, a)).map(|i| i + 1).collect::<Vec<_>>();
        assert_eq!(h(5, 1), vec![1, 2, 3]);
        assert_eq!(h(5, 2), vec![3, 4, 5]);
        assert_eq!(h(4, 1), vec![1, 2]);
        assert_eq!(h(4, 2), vec![3, 4]);
        assert_eq!(h(1, 2), vec![1]);
    }

    #[test]
    fn plan_counts() {
        let p = plan(Variant::Jj, WhichBias::Both, false, 10, 4, 0);
        assert_eq!(p.c, 13.0);
        assert_eq!(p.groups[0].1.len(), 10);
        assert_eq!(p.groups[1].1.len(), 4);
        let p = plan(Variant::Ss2, WhichBias::Individual, true, 10, 4, 0);
        assert_eq!(p.groups[0].1, vec![Kind::Th(1), Kind::Th(2)]);
    }
}
