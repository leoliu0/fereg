//! Recovery of the fixed effects (plan Phase 3.9): d = sum FE per observation and saved FE values
//! per term slot, all in kept-observation order.
//!
//! Given the FE coefficients of every residualized column, the FE fit of a linear combination of
//! columns is the same combination of their coefficients ([`FeCoefs::combine`]). With
//! c = [1, -beta...] the fit is that of y - X beta, i.e. y - X beta - e, and reghdfe's
//! d = y - xb - _cons - e is that fit minus the reported constant.

use rayon::prelude::*;

use crate::absorber::{Absorber, FeCoefs, fe1_index};
use crate::error::RfError;
use crate::par::{self, Pool};
use crate::session::Term;

/// Row-order values -> kept order (`out[row_order[k]] = v[k]`).
pub(crate) fn to_kept_order(v: &[f64], row_order: &[u32], pool: &Pool) -> Result<Vec<f64>, RfError> {
    scale_to_kept_order(v, 1.0, 0.0, row_order, pool)
}

/// `out[row_order[k]] = a * v[k] + b`, without a zeroing pass.
pub(crate) fn scale_to_kept_order(v: &[f64], a: f64, b: f64, row_order: &[u32], pool: &Pool) -> Result<Vec<f64>, RfError> {
    assert_eq!(v.len(), row_order.len(), "to_kept_order: length mismatch");
    let n = v.len();
    let mut out = par::alloc_uninit::<f64>(n)?;
    let ptr = SendPtr(out.as_mut_ptr().cast::<f64>());
    let tr = |x: f64| if a == 1.0 && b == 0.0 { x } else if b == 0.0 { a * x } else { a * x + b };
    if n <= par::CHUNK || pool.threads() <= 1 {
        for (&x, &k) in v.iter().zip(row_order) {
            debug_assert!((k as usize) < n);
            // SAFETY: row_order is a permutation of 0..n, so every slot is written exactly once.
            unsafe { *ptr.0.add(k as usize) = tr(x) };
        }
    } else {
        pool.install(|| {
            v.par_chunks(par::CHUNK).zip(row_order.par_chunks(par::CHUNK)).for_each(|(vc, oc)| {
                let p = ptr;
                for (&x, &k) in vc.iter().zip(oc) {
                    debug_assert!((k as usize) < n);
                    // SAFETY: row_order is a permutation of 0..n, so every slot is written once.
                    unsafe { *p.0.add(k as usize) = tr(x) };
                }
            })
        });
    }
    // SAFETY: row_order is a permutation of 0..n, so every element 0..n was initialized.
    Ok(unsafe { par::assume_init(out) })
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// d per kept observation: FE fit of the combined coefficient column minus `b0`.
pub(crate) fn d_values(absorber: &dyn Absorber, combined: &FeCoefs, b0: f64, pool: &Pool) -> Result<Vec<f64>, RfError> {
    let mut fit = absorber.fit_rows(combined, 0, pool);
    if b0 != 0.0 {
        pool.install(|| fit.par_iter_mut().for_each(|v| *v -= b0));
    }
    to_kept_order(&fit, absorber.row_order(), pool)
}

/// Saved FE values: for each term with `save` (absorb order), its intercept slot (if any) then its
/// slope slots (the per-level slope coefficient, as reghdfe stores it), per kept observation.
/// Values come from [`Absorber::term_values`] (normalized: every intercept-only term except FE1
/// has weighted mean 0). The reported constant `b0` is subtracted from FE1's intercept (or,
/// without FE1, from the first intercept slot), so that sum intercepts + sum slope*x = d when every
/// term is saved; reghdfe's own split across terms is path dependent, only d is unique.
pub(crate) fn saved_fe(absorber: &dyn Absorber, terms: &[Term], combined: &FeCoefs, b0: f64) -> Vec<Vec<f64>> {
    let mut vals = absorber.term_values(combined, 0);
    let shift_term = fe1_index(terms).or_else(|| terms.iter().position(|t| t.intercept));
    let mut out = Vec::new();
    let mut slot = 0;
    for (q, t) in terms.iter().enumerate() {
        let nslots = usize::from(t.intercept) + t.slopes.len();
        if t.intercept && Some(q) == shift_term && b0 != 0.0 {
            vals[slot].iter_mut().for_each(|v| *v -= b0);
        }
        if t.save {
            for lv in &vals[slot..slot + nslots] {
                out.push(t.codes.iter().map(|&c| lv[c as usize]).collect());
            }
        }
        slot += nslots;
    }
    out
}
