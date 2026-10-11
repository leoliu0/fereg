//! FE solver (plan D5, Phase 2): `FeSolver` implements [`crate::absorber::Absorber`].
//!
//! `residualize` runs (1) the fused rhs pass over the rows (cell sums, b1, Σwv², Σwv),
//! (2) lockstep PCG on the reduced system S x = rhs over the non-FE1 blocks, (3) the fused
//! back-substitution / in-place residualization pass (F is never materialized).
//!
//! FeCoefs layout: p = m + G1 rows, `[x (m×R); α1 (G1×R)]`, where x stacks the non-FE1 blocks
//! (small blocks first, then large ones, each in term order; see [`crate::structure`]).
//! Under `noabsorb` G1 = 1 and α1 is the weighted mean.

use std::time::Instant;

use crate::absorber::{Absorber, FeCoefs, SolveInfo, SolveOpts};
use crate::api::Ctx;
use crate::cg::{self, CgSystem};
use crate::error::RfError;
use crate::operator::{Ops, Rhs, WeightState, weight_state};
use crate::par::{CHUNK, Pool, alloc_vec, sum_chunks};
use crate::precond::{BlockPrecond, Precond, approx_chol::ApproxChol, dense::Dense, jacobi, sparse_chol::SparseChol};
use crate::session::Term;
use crate::structure::{self, Structure};

/// Assembly-work limit E_b = Σ_s d_s² for exact assembly of a large block.
pub const MAX_ASSEMBLY_WORK: u64 = 400_000_000;
/// Structural nonzeros limit of an exactly assembled block.
pub const MAX_BLOCK_NNZ: u64 = 30_000_000;

/// Solver policy knobs (defaults = plan D5).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverConfig {
    /// Total levels of the blocks factored jointly and exactly (dense).
    pub small_limit: usize,
    /// Largest reduced vector (bytes, all R columns) scattered entirely through per-task
    /// accumulators; larger large blocks use the M×R scratch and level pulls.
    pub acc_bytes_max: usize,
    /// Assembly-work limit E_b for exact assembly of a large block.
    pub max_assembly_work: u64,
    /// Structural nonzeros limit of an exactly assembled block.
    pub max_block_nnz: u64,
    /// Try the exact sparse Cholesky of an assembled block (else approx-chol of it).
    pub sparse_chol: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            small_limit: structure::SMALL_LIMIT,
            acc_bytes_max: crate::operator::ACC_BYTES_MAX,
            max_assembly_work: MAX_ASSEMBLY_WORK,
            max_block_nnz: MAX_BLOCK_NNZ,
            sparse_chol: true,
        }
    }
}

/// Builds the production absorber for the kept sample (`weights` in kept order).
pub fn build(
    terms: &[Term],
    weights: Option<&[f64]>,
    n: usize,
    pool: &Pool,
    ctx: &Ctx,
) -> Result<Box<dyn Absorber>, RfError> {
    Ok(Box::new(FeSolver::new(terms, weights, n, SolverConfig::default(), pool, ctx)?))
}

pub struct FeSolver {
    st: Structure,
    ws: WeightState,
    pre: Precond,
    /// `pre.small` / `pre.inv_diag` match the current weights.
    basic_fresh: bool,
    /// The large blocks were escalated (once per solver).
    escalated: bool,
    /// Escalated factors match the current weights.
    large_fresh: bool,
    /// CG count of the first solve after the last (re)factorization.
    base_iters: Option<usize>,
    need_refactor: bool,
    cfg: SolverConfig,
}

fn log_phase(ctx: &Ctx, t: &mut Instant, what: &str) {
    if ctx.verbose >= 2 {
        ctx.log(&format!("regfast: {what:<28} {:>9.4}s", t.elapsed().as_secs_f64()));
    }
    *t = Instant::now();
}

impl FeSolver {
    /// Builds the solver for session `terms` (kept order codes) and kept-order `weights`.
    pub fn new(terms: &[Term], weights: Option<&[f64]>, n: usize, cfg: SolverConfig, pool: &Pool, ctx: &Ctx) -> Result<FeSolver, RfError> {
        let mut t = Instant::now();
        if weights.is_some_and(|w| w.len() != n) {
            return Err(RfError::InvalidInput("weight length differs from the kept sample".into()));
        }
        if terms.iter().any(|t| t.codes.len() != n) {
            return Err(RfError::InvalidInput("term codes length differs from the kept sample".into()));
        }
        let mut st = structure::build(terms, n, cfg.small_limit, pool)?;
        st.acc_bytes_max = cfg.acc_bytes_max;
        log_phase(ctx, &mut t, "structure");
        ctx.check()?;
        let w_rows = match weights {
            Some(w) => {
                let mut v = alloc_vec(n, 0.0)?;
                for (o, &k) in v.iter_mut().zip(&st.pi) {
                    *o = w[k as usize];
                }
                Some(v)
            }
            None => None,
        };
        let ws = weight_state(&st, w_rows, pool)?;
        log_phase(ctx, &mut t, "weights/diag");
        let nl = st.blocks.iter().filter(|b| !b.small).count();
        if ctx.verbose >= 2 {
            let blocks: Vec<String> =
                st.blocks.iter().map(|b| format!("{}:{}{}", b.term, b.g, if b.small { "s" } else { "L" })).collect();
            ctx.log(&format!(
                "regfast: n={} cells={} G1={} m={} (small {}) blocks [{}]",
                st.n,
                st.ncells,
                st.g1,
                st.m,
                st.m_small,
                blocks.join(" ")
            ));
        }
        Ok(FeSolver {
            pre: Precond { small: None, inv_diag: Vec::new(), large: (0..nl).map(|_| BlockPrecond::Jacobi).collect() },
            st,
            ws,
            basic_fresh: false,
            escalated: false,
            large_fresh: true,
            base_iters: None,
            need_refactor: false,
            cfg,
        })
    }

    pub fn structure(&self) -> &Structure {
        &self.st
    }

    /// Names of the current large-block preconditioners.
    pub fn precond_kinds(&self) -> Vec<&'static str> {
        self.pre.large.iter().map(BlockPrecond::name).collect()
    }

    /// Rebuilds the escalated large-block factors for the current weights at the next
    /// `residualize` (numeric refactorization; the automatic rule does this when a solve with a
    /// stale factor needs more than max(30, 2 × the post-factorization count) iterations).
    pub fn request_refactor(&mut self) {
        if self.escalated {
            self.need_refactor = true;
        }
    }

    fn ensure_basic(&mut self, pool: &Pool) -> Result<(), RfError> {
        if self.basic_fresh {
            return Ok(());
        }
        self.pre.inv_diag = jacobi::inverse_diagonal(&self.ws.diag, &self.ws.redundant);
        self.pre.small = if self.st.m_small > 0 {
            let ops = Ops { st: &self.st, ws: &self.ws };
            let a = ops.assemble_small(pool)?;
            Some(Dense::factor(&a, self.st.m_small, &self.ws.redundant[..self.st.m_small], pool)?)
        } else {
            None
        };
        self.basic_fresh = true;
        Ok(())
    }

    fn refresh_large(&mut self, pool: &Pool, ctx: &Ctx) -> Result<(), RfError> {
        let ops = Ops { st: &self.st, ws: &self.ws };
        let ns = self.st.blocks.len() - self.pre.large.len();
        let mut cell_seg = None;
        for i in 0..self.pre.large.len() {
            let b = ns + i;
            let keep = kept_levels(&ops, b);
            let refreshed = match &mut self.pre.large[i] {
                BlockPrecond::Jacobi => true,
                BlockPrecond::SparseChol { keep: k0, f } if *k0 == keep => {
                    let agg = ops.seg_agg(b, pool)?;
                    if cell_seg.is_none() {
                        cell_seg = Some(self.st.cell_segments(pool)?);
                    }
                    let cs = cell_seg.as_deref().expect("set above");
                    let nnz = ops.block_row_nnz(b, &agg, cs, pool);
                    let csr = ops.assemble_block(b, &agg, cs, &nnz, &keep, pool)?;
                    f.refactor(&csr, pool)?
                }
                _ => false,
            };
            if !refreshed {
                self.pre.large[i] = escalate_block(&ops, b, &self.cfg, &mut cell_seg, pool, ctx)?;
            }
            ctx.check()?;
        }
        self.large_fresh = true;
        self.base_iters = None;
        Ok(())
    }

    fn wsum_b1sq(&self, b1: &[f64], r: usize, pool: &Pool) -> Vec<f64> {
        let n1 = &self.ws.n1;
        sum_chunks(pool, self.st.g1, CHUNK, r, |range, acc| {
            for s in range {
                for j in 0..r {
                    let v = b1[s * r + j];
                    acc[j] += v * v / n1[s];
                }
            }
        })
    }
}

/// Non-redundant levels of block `b`.
fn kept_levels(ops: &Ops, b: usize) -> Vec<u32> {
    let blk = &ops.st.blocks[b];
    (0..blk.g).filter(|&g| !ops.ws.redundant[blk.off + g]).map(|g| g as u32).collect()
}

/// Escalated preconditioner of large block `b` (plan D5 (b)).
fn escalate_block(ops: &Ops, b: usize, cfg: &SolverConfig, cell_seg: &mut Option<Vec<u32>>, pool: &Pool, ctx: &Ctx) -> Result<BlockPrecond, RfError> {
    let t = Instant::now();
    let keep = kept_levels(ops, b);
    let g = ops.st.blocks[b].g;
    let agg = ops.seg_agg(b, pool)?;
    ctx.check()?;
    let mut exact_csr = None;
    let mut nnz_total = 0u64;
    if agg.work <= cfg.max_assembly_work {
        if cell_seg.is_none() {
            *cell_seg = Some(ops.st.cell_segments(pool)?);
        }
        let cs = cell_seg.as_deref().expect("set above");
        let row_nnz = ops.block_row_nnz(b, &agg, cs, pool);
        nnz_total = row_nnz.iter().map(|&v| u64::from(v)).sum();
        if nnz_total <= cfg.max_block_nnz {
            exact_csr = Some(ops.assemble_block(b, &agg, cs, &row_nnz, &keep, pool)?);
        }
    }
    ctx.check()?;
    let out = match exact_csr {
        Some(csr) => match if cfg.sparse_chol { SparseChol::try_factor(&csr, pool)? } else { None } {
            Some(f) => BlockPrecond::SparseChol { keep, f: Box::new(f) },
            None => match ApproxChol::factor(&csr) {
                Ok(f) => BlockPrecond::ApproxChol { keep, f: Box::new(f) },
                Err(_) => BlockPrecond::Jacobi,
            },
        },
        None => {
            let csr = ops.sample_block(&agg, g, &keep, pool)?;
            match ApproxChol::factor(&csr) {
                Ok(f) => BlockPrecond::ApproxChol { keep, f: Box::new(f) },
                Err(_) => BlockPrecond::Jacobi,
            }
        }
    };
    if ctx.verbose >= 2 {
        ctx.log(&format!(
            "regfast: escalate block {b} (G={g}): E={} nnz={} -> {} in {:.4}s",
            agg.work,
            nnz_total,
            out.name(),
            t.elapsed().as_secs_f64()
        ));
    }
    Ok(out)
}

struct System<'a> {
    ops: Ops<'a>,
    pre: &'a mut Precond,
    zbuf: Vec<f64>,
    w: usize,
    can_escalate: bool,
    large_fresh: bool,
    cfg: &'a SolverConfig,
}

impl CgSystem for System<'_> {
    fn matvec(&mut self, p: &[f64], q: &mut [f64], pool: &Pool) -> Result<(), RfError> {
        self.ops.apply(p, self.w, q, &mut self.zbuf, pool);
        Ok(())
    }

    fn precond(&mut self, r: &[f64], z: &mut [f64], pool: &Pool) {
        self.pre.apply(self.ops.st, r, z, self.w, pool);
    }

    fn exact(&self) -> bool {
        let st = self.ops.st;
        // M = S on the whole reduced space: every block is in the dense small super-block, or the
        // only block is large with a current exact sparse factor.
        (st.m_small == st.m && self.pre.small.is_some())
            || (st.nb == 1 && self.large_fresh && matches!(self.pre.large.first(), Some(BlockPrecond::SparseChol { .. })))
    }

    fn can_escalate(&self) -> bool {
        self.can_escalate
    }

    fn escalate(&mut self, pool: &Pool, ctx: &Ctx) -> Result<bool, RfError> {
        self.can_escalate = false;
        let ns = self.ops.st.blocks.len() - self.pre.large.len();
        let mut cell_seg = None;
        let mut changed = false;
        for i in 0..self.pre.large.len() {
            if matches!(self.pre.large[i], BlockPrecond::Jacobi) {
                let bp = escalate_block(&self.ops, ns + i, self.cfg, &mut cell_seg, pool, ctx)?;
                changed |= !matches!(bp, BlockPrecond::Jacobi);
                self.pre.large[i] = bp;
            }
        }
        Ok(changed)
    }
}

impl Absorber for FeSolver {
    fn n(&self) -> usize {
        self.st.n
    }

    fn row_order(&self) -> &[u32] {
        &self.st.pi
    }

    fn weights(&self) -> Option<&[f64]> {
        self.ws.w_rows.as_deref()
    }

    fn set_weights(&mut self, w: Option<Vec<f64>>, pool: &Pool, ctx: &Ctx) -> Result<(), RfError> {
        if w.as_ref().is_some_and(|v| v.len() != self.st.n || v.iter().any(|x| !(x.is_finite() && *x > 0.0))) {
            return Err(RfError::InvalidInput("weights must be positive and finite".into()));
        }
        ctx.check()?;
        self.ws = weight_state(&self.st, w, pool)?;
        self.basic_fresh = false;
        if self.escalated {
            self.large_fresh = false;
        }
        Ok(())
    }

    fn residualize(
        &mut self,
        z: &mut [f64],
        r: usize,
        o: &SolveOpts,
        warm: Option<&FeCoefs>,
        pool: &Pool,
        ctx: &Ctx,
    ) -> Result<SolveInfo, RfError> {
        let (n, m, g1) = (self.st.n, self.st.m, self.st.g1);
        if z.len() != n * r {
            return Err(RfError::InvalidInput("residualize: block size differs from n·R".into()));
        }
        if warm.is_some_and(|wc| wc.r != r || wc.p != m + g1) {
            return Err(RfError::InvalidInput("residualize: warm start has a different shape".into()));
        }
        if r == 0 {
            return Ok(SolveInfo { iterations: 1, escalated: false, coefs: o.want_coefs.then(|| FeCoefs { p: m + g1, r, data: Vec::new() }) });
        }
        ctx.check()?;
        let mut t = Instant::now();
        self.ensure_basic(pool)?;
        if self.need_refactor {
            self.refresh_large(pool, ctx)?;
            self.need_refactor = false;
        }
        log_phase(ctx, &mut t, "preconditioner");
        let ops = Ops { st: &self.st, ws: &self.ws };
        let mut zbuf = if ops.needs_zbuf(r) { alloc_vec(self.st.ncells * r, 0.0)? } else { Vec::new() };
        let Rhs { rhs, b1, wsq, wsum } = ops.rhs(z, r, &mut zbuf, pool)?;
        log_phase(ctx, &mut t, "rhs pass");
        ctx.check()?;
        let sw = self.ws.sw;
        let ss0: Vec<f64> = (0..r)
            .map(|j| {
                let s = wsq[j] - wsum[j] * wsum[j] / sw;
                if s > 1e-15 * wsq[j] { s } else { 0.0 }
            })
            .collect();
        let b1sq = self.wsum_b1sq(&b1, r, pool);
        let ss1: Vec<f64> = (0..r).map(|j| wsq[j] - b1sq[j]).collect();
        let mut x = match warm {
            Some(wc) => wc.data[..m * r].to_vec(),
            None => alloc_vec(m * r, 0.0)?,
        };
        for j in 0..r {
            if ss0[j] == 0.0 {
                for i in 0..m {
                    x[i * r + j] = 0.0;
                }
            }
        }
        let mut iterations = 1;
        let mut escalated_now = false;
        if m > 0 {
            let can_escalate = !self.escalated && self.st.has_large();
            let mut sys = System {
                ops: Ops { st: &self.st, ws: &self.ws },
                pre: &mut self.pre,
                zbuf,
                w: r,
                can_escalate,
                large_fresh: self.large_fresh,
                cfg: &self.cfg,
            };
            let out = cg::pcg(&mut sys, &rhs, &mut x, r, &ss1, &ss0, o.tol, o.maxiter, pool, ctx)?;
            iterations = out.iterations.max(1);
            if out.escalated || !sys.can_escalate && can_escalate {
                self.escalated = true;
                self.large_fresh = true;
                self.base_iters = None;
            }
            escalated_now = out.escalated;
            if ctx.verbose >= 2 {
                ctx.log(&format!("regfast: pcg {} iterations (escalated: {})", out.iterations, out.escalated));
            }
            if self.escalated {
                match self.base_iters {
                    None => self.base_iters = Some(iterations),
                    Some(base) if !self.large_fresh && iterations > 30.max(2 * base) => self.need_refactor = true,
                    Some(_) => {}
                }
            }
        }
        log_phase(ctx, &mut t, "pcg");
        ctx.check()?;
        let mut alpha = if o.want_coefs { Some(alloc_vec(g1 * r, 0.0)?) } else { None };
        Ops { st: &self.st, ws: &self.ws }.residualize_rows(z, r, &x, &b1, alpha.as_deref_mut(), pool);
        log_phase(ctx, &mut t, "residualize pass");
        let coefs = alpha.map(|a| {
            let mut data = x;
            data.extend_from_slice(&a);
            FeCoefs { p: m + g1, r, data }
        });
        Ok(SolveInfo { iterations, escalated: escalated_now, coefs })
    }

    fn fit_rows(&self, c: &FeCoefs, col: usize, pool: &Pool) -> Vec<f64> {
        let v = c.col(col);
        let m = self.st.m;
        Ops { st: &self.st, ws: &self.ws }.fit_rows(&v[..m], &v[m..], pool)
    }

    fn term_values(&self, c: &FeCoefs, col: usize) -> Vec<Vec<f64>> {
        let v = c.col(col);
        let st = &self.st;
        let Some(fe1) = st.fe1 else { return Vec::new() };
        let nterms = st.blocks.len() + 1;
        let mut out: Vec<Vec<f64>> = vec![Vec::new(); nterms];
        let mut alpha = v[st.m..].to_vec();
        for blk in &st.blocks {
            let mut vals = v[blk.off..blk.off + blk.g].to_vec();
            let mean: f64 = vals.iter().zip(&self.ws.nr[blk.off..blk.off + blk.g]).map(|(a, w)| a * w).sum::<f64>() / self.ws.sw;
            vals.iter_mut().for_each(|a| *a -= mean);
            alpha.iter_mut().for_each(|a| *a += mean);
            out[blk.term] = vals;
        }
        out[fe1] = alpha;
        out
    }
}
