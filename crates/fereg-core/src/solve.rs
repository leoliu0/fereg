//! FE solver (plan D5, Phase 2): `FeSolver` implements [`crate::absorber::Absorber`].
//!
//! `residualize` runs (1) the fused rhs pass over the rows (cell sums, b1, sum wv^2, sum wv),
//! (2) lockstep PCG on the reduced system S x = rhs over the non-FE1 blocks, (3) the fused
//! back-substitution / in-place residualization pass (F is never materialized).
//!
//! FeCoefs layout: p = m + G1 rows, `[x (m x R); alpha1 (G1xR)]`, where x stacks the non-FE1 blocks
//! (small blocks first, then large ones, each in term order; see [`crate::structure`]).
//! Under `noabsorb` G1 = 1 and alpha1 is the weighted mean.

use std::time::Instant;

use rayon::prelude::*;

use crate::absorber::{Absorber, FeCoefs, SolveInfo, SolveOpts, fe1_index};
use crate::api::Ctx;
use crate::cg::{self, CgSystem};
use crate::error::RfError;
use crate::operator::{JointCells, Ops, Rhs, Scratch, SegAgg, WeightState, weight_state};
use crate::par::{self, CHUNK, Pool, alloc_vec, sum_chunks};
use crate::precond::dense::{self, Dense, SmallWork};
use crate::precond::{BlockPrecond, Joint, Precond, approx_chol::ApproxChol, jacobi, sparse_chol::SparseChol};
use crate::session::Term;
use crate::slopes::SlopeSolver;
use crate::structure::{self, Structure};

/// Assembly-work limit E_b = sum_s d_s^2 for exact assembly of a large block.
pub const MAX_ASSEMBLY_WORK: u64 = 400_000_000;
/// Structural nonzeros limit of an exactly assembled block.
pub const MAX_BLOCK_NNZ: u64 = 30_000_000;
/// Factor entries limit of the joint factor (memory: 8 bytes each).
pub const MAX_JOINT_LEN: usize = 100_000_000;

/// When to factor the joint reduced matrix S over all blocks exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointPolicy {
    /// Never (block-Jacobi preconditioner with per-block escalation only).
    Off,
    /// At the PCG probes (iterations 6 and 12 after each start), when the cost model predicts a
    /// saving: the joint build cost plus two exact iterations vs the projected remaining ones.
    Auto,
    /// Before the first iteration whenever the hard budgets allow (tests).
    Always,
}

/// Solver policy knobs (defaults = plan D5).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverConfig {
    /// Total levels of the blocks factored jointly and exactly (dense).
    pub small_limit: usize,
    /// Largest reduced vector (bytes, all R columns) scattered entirely through per-task
    /// accumulators; larger large blocks use the M x R scratch and level pulls.
    pub acc_bytes_max: usize,
    /// Assembly-work limit E_b for exact assembly of a large block.
    pub max_assembly_work: u64,
    /// Structural nonzeros limit of an exactly assembled block.
    pub max_block_nnz: u64,
    /// Try the exact sparse Cholesky of an assembled block (else approx-chol of it).
    pub sparse_chol: bool,
    /// Joint exact factor of S over all blocks (needs `sparse_chol`; same work/nnz budgets).
    pub joint: JointPolicy,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            small_limit: structure::SMALL_LIMIT,
            acc_bytes_max: crate::operator::ACC_BYTES_MAX,
            max_assembly_work: MAX_ASSEMBLY_WORK,
            max_block_nnz: MAX_BLOCK_NNZ,
            sparse_chol: true,
            joint: JointPolicy::Auto,
        }
    }
}

// Deterministic cost model: single-thread seconds on the reference machine (Ryzen 9 9950X),
// fitted on the 1e6-observation AKM / fixest panels with `verbose(2)` phase timings. It never
// depends on the thread count, so the solver path and therefore the result bits are the same at
// any number of threads (plan D3). Single-thread costs are the relevant scale: the matvec and the
// assembly scale ~5x on 16 threads while AMD and small factorizations do not, so at 16 threads
// the joint factor is relatively dearer, but PCG times there are far below every competitor.
/// Matvec cost per cell and per (block code + column).
const C_CELL: f64 = 2.8e-9;
/// Vector work per reduced coordinate and column (PCG updates, dots, Jacobi).
const C_COORD: f64 = 4e-9;
/// Sparse triangular solve cost per factor entry and column (forward + backward pass).
const C_SOLVE: f64 = 3e-10;
/// Dense small-block solve cost per entry and column.
const C_DENSE: f64 = 1e-9;
/// Joint assembly cost per unit of work (row walks + cells x blocks + dense small block) and
/// per structural nonzero (sorting and writing the rows).
const C_ASM: f64 = 1.2e-9;
const C_ASM_NNZ: f64 = 5e-8;
/// AMD cost per structural nonzero of the large-coordinate submatrix (the small coordinates are
/// ordered last without AMD), and symbolic analysis cost per structural nonzero.
const C_AMD: f64 = 1.5e-7;
const C_SYM: f64 = 1e-8;
/// Numeric factorization cost per multiply-add and per factor entry.
const C_FLOP: f64 = 1.2e-11;
const C_ENTRY: f64 = 6e-9;
/// Accept the joint factor when its remaining cost is below this share of the projected PCG
/// cost (model uncertainty margin).
const JOINT_MARGIN: f64 = 0.8;
/// Deferred dense small block (every block small, R small): a PCG iteration on cell data that fit
/// in the cache costs C_CELL_CACHED per cell and (block code + column) instead of C_CELL, and
/// the dense assembly costs C_SMALL_ASM per unit of sum_s d_s^2 (fitted at 1e5 observations,
/// 3 FEs: 0.23 ms per Jacobi iteration at R = 2, 3.0 ms assembly). The factor is deferred when
/// it costs more than DEFER_ITERS such iterations.
const C_CELL_CACHED: f64 = 6e-10;
const CACHE_BYTES: usize = 32 << 20;
const C_SMALL_ASM: f64 = 8e-10;
const DEFER_ITERS: f64 = 8.0;
/// Sparse small-factor path: assembly of the dense S_SS, scan and CSR extraction, per entry
/// of S_SS (0.058 s at m_s = 4067).
const C_SCAN: f64 = 3.5e-9;

/// Structural statistics of the joint factor (weight independent), filled stage by stage.
#[derive(Clone, Copy, Debug, Default)]
struct JointStats {
    /// 2*sum_s d_s*l_s + cells x blocks + dense small-block work (l_s: large codes of segment s).
    work: Option<u64>,
    /// sum_s d_s^2 (the assembly-work budget measure).
    seg_work: Option<u64>,
    /// Structural nonzeros of the joint matrix and of its large-coordinate submatrix.
    nnz: Option<u64>,
    amd_nnz: Option<u64>,
    len_val: Option<usize>,
    flops: Option<f64>,
}

/// Build stages of the joint factor, in order; costs of finished stages are sunk.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    Assemble,
    Analyze,
    Factor,
}

/// Cost of the joint build from `from` on, from the statistics known so far (unknown ones
/// count 0: a lower bound until they are measured).
fn joint_build_cost(js: &JointStats, from: Stage) -> f64 {
    let opt = |v: Option<u64>, c: f64| v.map_or(0.0, |v| c * v as f64);
    let assemble = opt(js.work, C_ASM) + opt(js.nnz, C_ASM_NNZ);
    let analyze = opt(js.amd_nnz, C_AMD) + opt(js.nnz, C_SYM);
    let factor = js.flops.map_or(0.0, |f| C_FLOP * f) + opt(js.len_val.map(|v| v as u64), C_ENTRY);
    let mut c = factor;
    if from <= Stage::Analyze {
        c += analyze;
    }
    if from <= Stage::Assemble {
        c += assemble;
    }
    c
}

/// Cost of a numeric refresh of the joint factor (same pattern: assembly + factorization).
fn joint_refresh_cost(js: &JointStats) -> f64 {
    joint_build_cost(js, Stage::Assemble) - joint_build_cost(js, Stage::Analyze) + joint_build_cost(js, Stage::Factor)
}

/// Cost of one solve with the joint factor for `r` columns.
fn joint_solve_cost(js: &JointStats, r: usize) -> f64 {
    js.len_val.map_or(0.0, |l| C_SOLVE * (l * r) as f64)
}

/// Cost of one PCG iteration without the preconditioner solve.
fn matvec_cost(st: &Structure, r: usize) -> f64 {
    C_CELL * (st.ncells * (st.nb + r)) as f64 + C_COORD * (st.m * r) as f64
}

/// Cost of one block-Jacobi preconditioner application.
fn block_precond_cost(st: &Structure, pre: &Precond, r: usize) -> f64 {
    let ms = st.m_small as f64;
    let factored: usize = pre.large.iter().map(|bp| if let BlockPrecond::SparseChol { f, .. } = bp { f.len_val() } else { 0 }).sum();
    C_DENSE * ms * ms * r as f64 + C_SOLVE * (factored * r) as f64
}

/// Cost of one PCG iteration with the Jacobi preconditioner on every coordinate, for the
/// deferral decision of the dense small block.
fn jacobi_iter_cost(st: &Structure, r: usize) -> f64 {
    let c = if st.ncells * (st.nb * 4 + r * 16) <= CACHE_BYTES { C_CELL_CACHED } else { C_CELL };
    c * (st.ncells * (st.nb + r)) as f64 + C_COORD * (st.m * r) as f64
}

/// Cost of assembling (unless the weight pass already did) and factoring the small block.
/// `known`: the representation of an earlier factorization of this structure ([`Dense::work`];
/// it does not change with the weights). Without one, a sparse factor is expected when sum_s d_s^2
/// bounds the nonzeros of S_SS below [`dense::SPARSE_DENSITY`].
fn small_build_cost(st: &Structure, ws: &WeightState, known: Option<SmallWork>) -> f64 {
    let asm = if ws.small_s.is_some() { 0.0 } else { C_SMALL_ASM * st.small_work as f64 };
    let ms = st.m_small as f64;
    let sparse_expected = (st.small_work as f64) <= dense::SPARSE_DENSITY * ms * ms;
    let factor = match known {
        Some(SmallWork::Sparse { flops, entries }) => C_SCAN * ms * ms + C_FLOP * flops + C_ENTRY * entries as f64,
        Some(SmallWork::Dense) => C_FLOP * ms.powi(3) / 3.0,
        None if sparse_expected => C_SCAN * ms * ms,
        None => C_FLOP * ms.powi(3) / 3.0,
    };
    asm + factor
}

/// One application of the small factor to `r` columns. Dense: two triangular solves, measured
/// at 1-16 threads (m_s 512-4096) as m_s^2*(1.5e-10 + 3.5e-11*r) (matrix traffic, then flops per
/// column); sparse: the factor entries are streamed twice per column.
fn small_apply_cost(st: &Structure, work: SmallWork, r: usize) -> f64 {
    match work {
        SmallWork::Dense => {
            let ms = st.m_small as f64;
            ms * ms * (1.5e-10 + 3.5e-11 * r as f64)
        }
        SmallWork::Sparse { entries, .. } => C_SOLVE * (entries * r) as f64,
    }
}

/// Share of a solve's PCG cost that a refresh of stale block factors is expected to save. On the
/// 6.7M-trip taxi Poisson model (two coupled large blocks with exact factors) refreshing them
/// before every IRLS solve cut the PCG iterations from 654 to 361.
const STALE_GAIN: f64 = 0.3;

/// A stale small factor is always refactored once the weights moved by more than this ratio
/// range since its factorization (e.g. the switch to and from the ReLU separation weights,
/// 1 vs ~7e7): PCG would need on the order of sqrt(ratio) iterations, more than any observed count
/// predicts. Gradual IRLS changes stay far below it between refactorizations.
const MAX_SMALL_DRIFT: f64 = 1e6;

/// max_i(b_i / a_i) / min_i(b_i / a_i) over the rows (`None` = all ones).
fn ratio_range(a: Option<&[f64]>, b: Option<&[f64]>, pool: &Pool) -> f64 {
    let n = a.or(b).map_or(0, <[f64]>::len);
    if n == 0 {
        return 1.0;
    }
    let (lo, hi) = pool.install(|| {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let q = b.map_or(1.0, |b| b[i]) / a.map_or(1.0, |a| a[i]);
                (q, q)
            })
            .reduce(|| (f64::INFINITY, 0.0), |x, y| (x.0.min(y.0), x.1.max(y.1)))
    });
    hi / lo
}

/// Small-block factor iteration counts, for the stale-factor decision after weight changes.
#[derive(Clone, Copy, Debug, Default)]
struct SmallIters {
    /// Iterations of the last solve that started with a factor for its own weights.
    fresh: Option<usize>,
    /// Iterations of the last two solves with a stale factor (weights changed since
    /// factorization), most recent first.
    stale: Option<usize>,
    stale_prev: Option<usize>,
}

impl SmallIters {
    /// Extra iterations the next solve with the stale factor is expected to cost: the last
    /// stale count, extrapolated linearly from the one before (the factor drifts further from
    /// the weights with every change), minus the fresh count; [`cg::WINDOW`] before any stale
    /// solve.
    fn stale_extra(&self) -> usize {
        let Some(s) = self.stale else { return cg::WINDOW };
        let next = match self.stale_prev {
            Some(p) => (2 * s).saturating_sub(p).max(s),
            None => s,
        };
        next.saturating_sub(self.fresh.unwrap_or(0))
    }

    fn record_stale(&mut self, iterations: usize) {
        self.stale_prev = self.stale;
        self.stale = Some(iterations);
    }

    /// A new factor: its staleness is learned again.
    fn reset_stale(&mut self) {
        self.stale = None;
        self.stale_prev = None;
    }
}

/// The dense factor of the small super-block for the current weights.
fn small_factor(ops: &Ops, pool: &Pool) -> Result<Dense, RfError> {
    let (st, ws) = (ops.st, ops.ws);
    let built;
    let a = match ws.small_s.as_deref() {
        Some(a) => a,
        None => {
            built = ops.assemble_small(pool)?;
            &built
        }
    };
    Dense::factor(a, st.m_small, &ws.redundant[..st.m_small], pool)
}

/// Joint factor state of a solver.
#[derive(Default)]
struct JointState {
    stats: JointStats,
    /// Over a hard budget or failed numerically: never retried.
    failed: bool,
    /// Iterations of the last solve with a stale joint factor (weights changed since its
    /// factorization): the prediction for the next stale solve, kept across refreshes.
    stale_iters: Option<usize>,
    /// Structural lookups and row capacities of the joint matrix (weight independent; kept for
    /// later attempts and refreshes).
    pattern: Option<(JointCells, Vec<u32>)>,
    /// Per-segment aggregates for the current weights, kept between attempts of one solve.
    agg: Option<SegAgg>,
}

/// Builds the production absorber for the kept sample (`weights` in kept order): [`FeSolver`]
/// for intercept-only designs, [`SlopeSolver`] when some term has slopes or no intercept.
pub fn build(
    terms: &[Term],
    weights: Option<&[f64]>,
    n: usize,
    pool: &Pool,
    ctx: &Ctx,
) -> Result<Box<dyn Absorber>, RfError> {
    if terms.iter().any(|t| !t.slopes.is_empty() || !t.intercept) {
        return Ok(Box::new(SlopeSolver::new(terms, weights, n, SolverConfig::default(), pool, ctx)?));
    }
    Ok(Box::new(FeSolver::new(terms, weights, n, SolverConfig::default(), pool, ctx)?))
}

pub struct FeSolver {
    st: Structure,
    /// FE1 of the FE normalization ([`fe1_index`]); the solver may eliminate another term.
    norm_fe1: Option<usize>,
    ws: WeightState,
    pre: Precond,
    /// `pre.inv_diag` matches the current weights.
    basic_fresh: bool,
    /// `pre.small` (when set) was factored for the current weights. After a weight change the
    /// old factor is kept as a preconditioner while that is cheaper than refactoring (see
    /// [`FeSolver::refresh_small`]).
    small_fresh: bool,
    small_iters: SmallIters,
    /// [`Dense::work`] of the last small factor built by this solver.
    small_work: Option<SmallWork>,
    /// Bound on max_i(w_i / w_i at factorization) / min_i(same) for the small factor: the
    /// product of the per-change ratio ranges since it was built.
    small_drift: f64,
    /// The large blocks were escalated (once per solver).
    escalated: bool,
    /// Escalated (block or joint) factors match the current weights.
    large_fresh: bool,
    /// CG count of the first solve after the last (re)factorization.
    base_iters: Option<usize>,
    need_refactor: bool,
    /// PCG iterations of the last solve (the stale-factor refresh decision).
    last_iters: Option<usize>,
    /// A deferred solve had to build the dense small factor at a probe (Jacobi converges too
    /// slowly on this graph): later solves, e.g. the next IRLS weights, build it up front.
    small_upfront: bool,
    joint: JointState,
    cfg: SolverConfig,
    /// Kernel scratch reused across solves.
    scratch: Scratch,
}

pub(crate) fn log_phase(ctx: &Ctx, t: &mut Instant, what: &str) {
    if ctx.verbose >= 2 {
        ctx.log(&format!("fereg: {what:<28} {:>9.4}s", t.elapsed().as_secs_f64()));
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
                pool.install(|| {
                    v.par_chunks_mut(CHUNK).zip(st.pi.par_chunks(CHUNK)).for_each(|(o, p)| {
                        for (o, &k) in o.iter_mut().zip(p) {
                            *o = w[k as usize];
                        }
                    })
                });
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
                "fereg: n={} cells={} G1={} m={} (small {}) blocks [{}]",
                st.n,
                st.ncells,
                st.g1,
                st.m,
                st.m_small,
                blocks.join(" ")
            ));
        }
        Ok(FeSolver {
            norm_fe1: fe1_index(terms),
            pre: Precond { small: None, inv_diag: Vec::new(), large: (0..nl).map(|_| BlockPrecond::Jacobi).collect(), joint: None },
            st,
            ws,
            basic_fresh: false,
            small_fresh: false,
            small_iters: SmallIters::default(),
            small_work: None,
            small_drift: 1.0,
            escalated: false,
            large_fresh: true,
            base_iters: None,
            need_refactor: false,
            last_iters: None,
            small_upfront: false,
            joint: JointState::default(),
            cfg,
            scratch: Scratch::default(),
        })
    }

    pub fn structure(&self) -> &Structure {
        &self.st
    }

    /// Names of the current large-block preconditioners (`["joint"]` with the joint factor).
    pub fn precond_kinds(&self) -> Vec<&'static str> {
        if self.pre.joint.is_some() {
            return vec!["joint"];
        }
        self.pre.large.iter().map(BlockPrecond::name).collect()
    }

    /// Rebuilds the escalated large-block (or joint) factors for the current weights at the next
    /// `residualize` (numeric refactorization; the automatic rules do this when a solve with a
    /// stale block factor needs more than max(30, 2 x the post-factorization count) iterations,
    /// or when the cost model prefers refreshing a stale joint factor to iterating with it).
    pub fn request_refactor(&mut self) {
        if self.escalated || self.pre.joint.is_some() {
            self.need_refactor = true;
        }
    }

    /// Whether the joint factor may still be tried.
    fn joint_allowed(&self) -> bool {
        joint_allowed(&self.cfg, &self.joint) && self.pre.joint.is_none() && self.st.has_large()
    }

    /// Current inverse diagonal, and the dense small factor unless `defer_small` (it is then
    /// built at a PCG probe if the iterations turn out dearer, see [`System::probe`]). A stale
    /// small factor kept by [`FeSolver::refresh_small`] is used as is.
    fn ensure_basic(&mut self, defer_small: bool, pool: &Pool) -> Result<(), RfError> {
        if !self.basic_fresh {
            self.pre.inv_diag = jacobi::inverse_diagonal(&self.ws.diag, &self.ws.redundant);
            self.basic_fresh = true;
        }
        if !defer_small && self.pre.small.is_none() && self.st.m_small > 0 {
            let d = small_factor(&Ops { st: &self.st, ws: &self.ws }, pool)?;
            self.small_work = Some(d.work());
            self.pre.small = Some(d);
            self.small_fresh = true;
            self.small_drift = 1.0;
        }
        Ok(())
    }

    /// After a weight change, keeps the dense small factor of earlier weights as the small-block
    /// preconditioner (PCG then needs a few more iterations instead of being exact) unless
    /// refactoring costs less than the extra iterations it is expected to save
    /// ([`SmallIters::stale_extra`]), or the factored coordinates changed. Dropping it makes
    /// [`FeSolver::ensure_basic`] (or the deferral) build a fresh one.
    fn refresh_small(&mut self, r: usize, ctx: &Ctx) {
        let Some(d) = &self.pre.small else { return };
        if self.small_fresh {
            return;
        }
        let (st, ws) = (&self.st, &self.ws);
        let same_coords = d.keep().iter().copied().eq((0..st.m_small).filter(|&i| !ws.redundant[i] && ws.diag[i] > 0.0).map(|i| i as u32));
        let build = small_build_cost(st, ws, Some(d.work()));
        let stale = self.small_iters.stale_extra() as f64 * (jacobi_iter_cost(st, r) + small_apply_cost(st, d.work(), r));
        if ctx.verbose >= 2 {
            ctx.log(&format!("fereg: stale small factor: refactor est {build:.4}s vs extra pcg est {stale:.4}s"));
        }
        if !same_coords || build <= stale || self.small_drift > MAX_SMALL_DRIFT {
            // The new factor starts a new cycle: its staleness is re-learned from the next
            // stale solve (weights drift less as IRLS converges).
            self.pre.small = None;
            self.small_iters.reset_stale();
        }
    }

    /// Whether this solve starts with Jacobi on the small block instead of its dense factor:
    /// every block is small (no large-block escalation or joint factor involved), no factor (not
    /// even a stale one) is available, no earlier deferral of this solver had to build it, and
    /// it costs more than [`DEFER_ITERS`] iterations.
    fn defer_small(&self, r: usize) -> bool {
        let st = &self.st;
        st.m_small > 0
            && !st.has_large()
            && !self.small_upfront
            && self.pre.small.is_none()
            && small_build_cost(st, &self.ws, self.small_work) > DEFER_ITERS * jacobi_iter_cost(st, r)
    }

    /// Escalated block factors that are stale (the weights changed since factorization, as
    /// between IRLS iterations) are refactored before a solve when that costs at most
    /// [`STALE_GAIN`] of the last solve's PCG cost. Only exact sparse factors have a cost
    /// estimate; with an approximate one the iteration-count rule of `residualize` decides.
    fn stale_refresh_pays(&self, r: usize, ctx: &Ctx) -> bool {
        if !self.escalated || self.large_fresh || self.pre.joint.is_some() {
            return false;
        }
        let Some(last) = self.last_iters else { return false };
        let mut rebuild = 0.0;
        for bp in &self.pre.large {
            match bp {
                BlockPrecond::Jacobi => {}
                BlockPrecond::SparseChol { rebuild: c, .. } => rebuild += c,
                BlockPrecond::ApproxChol { .. } => return false,
            }
        }
        let pcg = last as f64 * (matvec_cost(&self.st, r) + block_precond_cost(&self.st, &self.pre, r));
        let pays = rebuild <= STALE_GAIN * pcg;
        if ctx.verbose >= 2 {
            ctx.log(&format!("fereg: stale block factors: refresh est {rebuild:.4}s vs last pcg est {pcg:.4}s -> {}", if pays { "refresh" } else { "keep" }));
        }
        pays
    }

    fn refresh_large(&mut self, pool: &Pool, ctx: &Ctx) -> Result<(), RfError> {
        if self.pre.joint.is_some() {
            self.refresh_joint(pool, ctx)?;
            if self.pre.joint.is_some() {
                return Ok(());
            }
        }
        let ops = Ops { st: &self.st, ws: &self.ws };
        let ns = self.st.blocks.len() - self.pre.large.len();
        let mut cell_seg = None;
        for i in 0..self.pre.large.len() {
            let b = ns + i;
            let keep = kept_levels(&ops, b);
            let refreshed = match &mut self.pre.large[i] {
                BlockPrecond::Jacobi => true,
                BlockPrecond::SparseChol { keep: k0, f, .. } if *k0 == keep => {
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

    /// Numeric refresh of the joint factor for the current weights (same pattern), or a rebuild
    /// when the kept coordinates changed. On failure the joint factor is dropped for good and the
    /// block preconditioner (with its one-time escalation) takes over again.
    fn refresh_joint(&mut self, pool: &Pool, ctx: &Ctx) -> Result<(), RfError> {
        let t = Instant::now();
        let ops = Ops { st: &self.st, ws: &self.ws };
        let keep = kept_coords(&ops);
        let Some(j) = self.pre.joint.as_mut() else { return Ok(()) };
        let mut ok = false;
        if j.keep == keep {
            let agg = ops.joint_seg_agg(pool)?;
            if self.joint.pattern.is_none() {
                let jc = ops.joint_cells(pool)?;
                let (row_nnz, _, _) = ops.joint_row_nnz(&agg, &jc, pool);
                self.joint.pattern = Some((jc, row_nnz));
            }
            let (jc, row_nnz) = self.joint.pattern.as_ref().expect("set above");
            let csr = ops.assemble_joint(&agg, jc, row_nnz, &keep, pool)?;
            ok = j.f.refactor(&csr, pool)?;
        }
        if ctx.verbose >= 2 {
            ctx.log(&format!("fereg: joint factor refresh: {} in {:.4}s", if ok { "refactored" } else { "pattern changed or failed" }, t.elapsed().as_secs_f64()));
        }
        if !ok {
            self.pre.joint = None;
            let ops = Ops { st: &self.st, ws: &self.ws };
            self.pre.joint = build_joint(&ops, &self.cfg, &mut self.joint, f64::INFINITY, 1, "refresh", pool, ctx)?;
            if self.pre.joint.is_none() {
                self.joint.failed = true;
                self.escalated = false;
            }
        }
        self.large_fresh = true;
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

/// Non-redundant reduced coordinates (global, ascending).
fn kept_coords(ops: &Ops) -> Vec<u32> {
    (0..ops.st.m).filter(|&k| !ops.ws.redundant[k]).map(|k| k as u32).collect()
}

fn joint_allowed(cfg: &SolverConfig, js: &JointState) -> bool {
    cfg.joint != JointPolicy::Off && cfg.sparse_chol && !js.failed
}

/// Builds the exact joint factor of S when the cost model accepts it: the remaining build cost
/// plus the solves of the (at most two) remaining iterations, 2*(matvec + joint solve), must be
/// at most [`JOINT_MARGIN`] x `without` (the projected PCG cost with the current preconditioner;
/// `f64::INFINITY` = always within the hard budgets). The check runs before every stage with
/// the statistics known so far (assembly work -> nnz -> factor size and flops, cached in `js`)
/// and without the costs already sunk, so an unaffordable factor is rejected as early as
/// possible. Hard budget overruns (assembly work, nnz, factor entries) and numerical failure
/// mark the joint factor as failed for good.
#[allow(clippy::too_many_arguments)]
fn build_joint(ops: &Ops, cfg: &SolverConfig, js: &mut JointState, without: f64, r: usize, why: &str, pool: &Pool, ctx: &Ctx) -> Result<Option<Joint>, RfError> {
    let t = Instant::now();
    let mv = matvec_cost(ops.st, r);
    let force = cfg.joint == JointPolicy::Always;
    let cost = |s: &JointStats, from: Stage| joint_build_cost(s, from) + 2.0 * (mv + joint_solve_cost(s, r));
    let budget = JOINT_MARGIN * without;
    let affordable = |s: &JointStats, from: Stage| force || cost(s, from) <= budget;
    let mut times = [0.0f64; 6];
    let mut lap = Instant::now();
    let mut built = None;
    // Per-segment aggregates (current weights) and the structural pattern survive a rejection,
    // so a later probe in the same solve does not recompute them.
    let mut agg = js.agg.take();
    let mut pattern = js.pattern.take();
    let what: &str = 'stages: {
        if !affordable(&js.stats, Stage::Assemble) {
            break 'stages "rejected (cost)";
        }
        let st = ops.st;
        // Structural work first (a cheap counting pass): an unaffordable factor is rejected
        // before the per-segment aggregates are built.
        if js.stats.work.is_none() {
            let (seg_work, dl) = ops.joint_seg_counts(pool);
            let small_work = if st.m_small > 0 { st.small_work } else { 0 };
            js.stats.seg_work = Some(seg_work);
            js.stats.work = Some(2 * dl + (st.ncells * st.nb) as u64 + small_work);
        }
        times[0] = split(&mut lap);
        ctx.check()?;
        if js.stats.seg_work.is_some_and(|w| w > cfg.max_assembly_work) {
            js.failed = true;
            break 'stages "over the assembly budget";
        }
        if !affordable(&js.stats, Stage::Assemble) {
            break 'stages "rejected (cost)";
        }
        if agg.is_none() {
            agg = Some(ops.joint_seg_agg(pool)?);
        }
        let a = agg.as_ref().expect("set above");
        times[0] += split(&mut lap);
        if pattern.is_none() {
            let jc = ops.joint_cells(pool)?;
            times[1] = split(&mut lap);
            let (row_nnz, nnz, amd_nnz) = ops.joint_row_nnz(a, &jc, pool);
            js.stats.nnz = Some(nnz);
            js.stats.amd_nnz = Some(amd_nnz);
            pattern = Some((jc, row_nnz));
        }
        times[2] = split(&mut lap);
        if js.stats.nnz.is_some_and(|n| n > cfg.max_block_nnz) {
            js.failed = true;
            break 'stages "over the nnz budget";
        }
        if !affordable(&js.stats, Stage::Assemble) {
            break 'stages "rejected (cost)";
        }
        let keep = kept_coords(ops);
        let (jc, row_nnz) = pattern.as_ref().expect("set above");
        let csr = ops.assemble_joint(a, jc, row_nnz, &keep, pool)?;
        times[3] = split(&mut lap);
        ctx.check()?;
        // Small coordinates (dense-ish rows, coupled to almost everything) are eliminated last.
        let small_kept = keep.partition_point(|&k| (k as usize) < ops.st.m_small);
        let analyzed = SparseChol::analyze_last(&csr, small_kept, pool);
        drop(csr);
        times[4] = split(&mut lap);
        let Some((mut f, vals)) = analyzed else {
            js.failed = true;
            break 'stages "symbolic analysis failed";
        };
        js.stats.len_val = Some(f.len_val());
        js.stats.flops = Some(f.flops());
        if f.len_val() > MAX_JOINT_LEN {
            js.failed = true;
            break 'stages "over the fill budget";
        }
        if !affordable(&js.stats, Stage::Factor) {
            break 'stages "rejected (cost)";
        }
        ctx.check()?;
        let ok = f.factor(&vals, pool)?;
        times[5] = split(&mut lap);
        if !ok {
            js.failed = true;
            break 'stages "numeric factorization failed";
        }
        built = Some(Joint { keep, f: Box::new(f) });
        "built"
    };
    if js.failed {
        (agg, pattern) = (None, None);
    } else if built.is_some() {
        agg = None;
    }
    (js.agg, js.pattern) = (agg, pattern);
    if ctx.verbose >= 2 {
        let s = &js.stats;
        let opt = |v: Option<u64>| v.map_or("?".to_string(), |v| v.to_string());
        ctx.log(&format!(
            "fereg: joint factor ({why}): m={} work={} nnz={} (amd {}) L={} flops={} est {:.4}s vs pcg {:.4}s -> {what} in {:.4}s (agg {:.4} cells {:.4} nnz {:.4} asm {:.4} amd {:.4} num {:.4})",
            ops.st.m,
            opt(s.work),
            opt(s.nnz),
            opt(s.amd_nnz),
            opt(s.len_val.map(|v| v as u64)),
            s.flops.map_or("?".to_string(), |f| format!("{f:.3e}")),
            cost(s, Stage::Assemble),
            without,
            t.elapsed().as_secs_f64(),
            times[0],
            times[1],
            times[2],
            times[3],
            times[4],
            times[5]
        ));
    }
    Ok(built)
}

/// Seconds since `t`, restarting it.
fn split(t: &mut Instant) -> f64 {
    let e = t.elapsed().as_secs_f64();
    *t = Instant::now();
    e
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
            Some(f) => {
                let rebuild = C_ASM * agg.work as f64 + C_ASM_NNZ * nnz_total as f64 + C_FLOP * f.flops() + C_ENTRY * f.len_val() as f64;
                BlockPrecond::SparseChol { keep, f: Box::new(f), rebuild }
            }
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
        let fill = match &out {
            BlockPrecond::SparseChol { f, .. } => format!(" (L={} flops={:.3e})", f.len_val(), f.flops()),
            _ => String::new(),
        };
        ctx.log(&format!(
            "fereg: escalate block {b} (G={g}): E={} nnz={} -> {}{fill} in {:.4}s",
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
    scratch: &'a mut Scratch,
    w: usize,
    can_escalate: bool,
    large_fresh: bool,
    cfg: &'a SolverConfig,
    joint: &'a mut JointState,
    /// The joint factor was built during this solve.
    joint_built: bool,
    /// The dense small factor is deferred (every block small; see `FeSolver::defer_small`).
    small_deferred: bool,
    /// The deferred small factor was built during this solve (a restart, not an escalation).
    small_built: bool,
    /// `pre.small` matches the current weights (a stale factor is a preconditioner, not S^-1).
    small_fresh: bool,
    /// [`Dense::work`] of an earlier small factor of this solver, for the build-cost estimate.
    small_known: Option<SmallWork>,
}

impl CgSystem for System<'_> {
    fn matvec(&mut self, p: &[f64], q: &mut [f64], pool: &Pool) -> Result<(), RfError> {
        self.ops.apply(p, self.w, q, self.scratch, pool)
    }

    fn precond(&mut self, r: &[f64], z: &mut [f64], pool: &Pool) {
        self.pre.apply(self.ops.st, r, z, self.w, pool);
    }

    fn exact(&self) -> bool {
        let st = self.ops.st;
        // M = S on the whole reduced space: a current joint factor, every block in the dense
        // small super-block, or the only block is large with a current exact sparse factor.
        (self.large_fresh && self.pre.joint.is_some())
            || (self.pre.joint.is_none() && st.m_small == st.m && self.pre.small.is_some() && self.small_fresh)
            || (st.nb == 1 && self.large_fresh && matches!(self.pre.large.first(), Some(BlockPrecond::SparseChol { .. })))
    }

    fn probe(&mut self, p: &cg::Probe, pool: &Pool, ctx: &Ctx) -> Result<Option<bool>, RfError> {
        if self.small_deferred {
            // Build the dense small factor when the projected Jacobi iterations cost more; the
            // restart from the current iterate is then exact.
            let st = self.ops.st;
            let (iters, build) = (p.remaining * jacobi_iter_cost(st, self.w), small_build_cost(st, self.ops.ws, self.small_known));
            if ctx.verbose >= 2 {
                ctx.log(&format!(
                    "fereg: deferred small block at k={} (rho={:.3}, remaining={:.0}): pcg est {iters:.4}s vs factor est {build:.4}s",
                    p.k, p.rho_hat, p.remaining
                ));
            }
            if iters > build {
                self.pre.small = Some(small_factor(&self.ops, pool)?);
                self.small_deferred = false;
                self.small_built = true;
                self.small_fresh = true;
                return Ok(Some(true));
            }
            return Ok(Some(false));
        }
        // Every block small: the small factor already covers all of S (a stale one is still the
        // cheaper preconditioner), so there is no joint factor to build.
        if self.pre.joint.is_some() || !joint_allowed(self.cfg, self.joint) || !self.ops.st.has_large() {
            return Ok(None);
        }
        let st = self.ops.st;
        let without = p.remaining * (matvec_cost(st, self.w) + block_precond_cost(st, self.pre, self.w));
        let why = format!("k={} rho={:.3} remaining={:.0}", p.k, p.rho_hat, p.remaining);
        match build_joint(&self.ops, self.cfg, self.joint, without, self.w, &why, pool, ctx)? {
            Some(j) => {
                self.pre.joint = Some(j);
                self.large_fresh = true;
                self.joint_built = true;
                Ok(Some(true))
            }
            None => Ok(None),
        }
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

    fn set_weights(&mut self, w: Option<&[f64]>, pool: &Pool, ctx: &Ctx) -> Result<(), RfError> {
        if w.is_some_and(|v| v.len() != self.st.n || !par::all_positive(v, pool)) {
            return Err(RfError::InvalidInput("weights must be positive and finite".into()));
        }
        ctx.check()?;
        let mut t = Instant::now();
        if self.pre.small.is_some() {
            self.small_drift *= ratio_range(self.ws.w_rows.as_deref(), w, pool);
        }
        let w_rows = w.map(|v| par::copy_into(self.ws.w_rows.take(), v, pool)).transpose()?;
        self.ws = weight_state(&self.st, w_rows, pool)?;
        log_phase(ctx, &mut t, "set_weights");
        self.joint.agg = None;
        self.basic_fresh = false;
        self.small_fresh = false;
        if self.escalated || self.pre.joint.is_some() {
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
            return Err(RfError::InvalidInput("residualize: block size differs from n*R".into()));
        }
        if warm.is_some_and(|wc| wc.r != r || wc.p != m + g1) {
            return Err(RfError::InvalidInput("residualize: warm start has a different shape".into()));
        }
        if r == 0 {
            return Ok(SolveInfo { iterations: 1, escalated: false, coefs: o.want_coefs.then(|| FeCoefs { p: m + g1, r, data: Vec::new() }), exact: true });
        }
        ctx.check()?;
        let mut t = Instant::now();
        let mut joint_now = false;
        if self.need_refactor || self.stale_refresh_pays(r, ctx) {
            self.refresh_large(pool, ctx)?;
            self.need_refactor = false;
        } else if self.pre.joint.is_some() && !self.large_fresh {
            // Stale joint factor (weights changed): refresh it numerically when that costs less
            // than iterating with it (at least the window, else as many as the last stale solve).
            let (mv, sol) = (matvec_cost(&self.st, r), joint_solve_cost(&self.joint.stats, r));
            let stale = cg::WINDOW.max(self.joint.stale_iters.unwrap_or(0)) as f64 * (mv + sol);
            let fresh = joint_refresh_cost(&self.joint.stats) + 2.0 * (mv + sol);
            if ctx.verbose >= 2 {
                ctx.log(&format!("fereg: stale joint factor: refresh est {fresh:.4}s vs stale pcg {stale:.4}s"));
            }
            if fresh <= stale {
                self.refresh_joint(pool, ctx)?;
            }
        }
        // `Always`: build before the first iteration. `Auto` decides at the PCG probes, where the
        // observed contraction projects the remaining iterations (an up-front attempt would pay
        // the aggregation pass even on designs that converge in a few iterations).
        if m > 0 && self.cfg.joint == JointPolicy::Always && self.joint_allowed() {
            let ops = Ops { st: &self.st, ws: &self.ws };
            if let Some(j) = build_joint(&ops, &self.cfg, &mut self.joint, f64::INFINITY, r, "k=0", pool, ctx)? {
                self.pre.joint = Some(j);
                self.large_fresh = true;
                joint_now = true;
            }
        }
        if self.pre.joint.is_none() {
            self.refresh_small(r, ctx);
        }
        let defer = self.pre.joint.is_none() && self.defer_small(r);
        if self.pre.joint.is_none() {
            self.ensure_basic(defer, pool)?;
        }
        if defer && ctx.verbose >= 2 {
            ctx.log(&format!(
                "fereg: small block deferred: factor est {:.4}s vs pcg iteration est {:.5}s",
                small_build_cost(&self.st, &self.ws, self.small_work),
                jacobi_iter_cost(&self.st, r)
            ));
        }
        log_phase(ctx, &mut t, "preconditioner");
        let ops = Ops { st: &self.st, ws: &self.ws };
        let Rhs { rhs, b1, wsq, wsum } = ops.rhs(z, r, &mut self.scratch, pool)?;
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
        // Closed form without reduced blocks; else set from the PCG outcome.
        let mut exact = true;
        if m > 0 {
            let can_escalate = !self.escalated && self.st.has_large() && self.pre.joint.is_none();
            let small_fresh = self.small_fresh;
            let mut sys = System {
                ops: Ops { st: &self.st, ws: &self.ws },
                pre: &mut self.pre,
                scratch: &mut self.scratch,
                w: r,
                can_escalate,
                large_fresh: self.large_fresh,
                cfg: &self.cfg,
                joint: &mut self.joint,
                joint_built: false,
                small_deferred: defer,
                small_built: false,
                small_fresh,
                small_known: self.small_work,
            };
            let out = cg::pcg(&mut sys, &rhs, &mut x, r, &ss1, &ss0, o.tol, o.maxiter, pool, ctx)?;
            iterations = out.iterations.max(1);
            self.last_iters = Some(iterations);
            exact = out.exact || ss0.iter().all(|&s| s == 0.0);
            let joint_built = sys.joint_built;
            // Building a deferred small factor restarts PCG; it is not a large-block escalation
            // (deferral requires that there are no large blocks).
            let escalated_large = out.escalated && !sys.small_built;
            self.small_upfront |= sys.small_built;
            if sys.small_built {
                self.small_fresh = true;
                self.small_drift = 1.0;
                self.small_work = sys.pre.small.as_ref().map(Dense::work);
            } else if sys.pre.joint.is_none() && sys.pre.small.is_some() {
                // Iteration counts with the small factor, for the stale-factor decision (solves
                // that built it at a probe also ran Jacobi iterations and are not recorded).
                if small_fresh {
                    self.small_iters.fresh = Some(iterations);
                } else {
                    self.small_iters.record_stale(iterations);
                }
            }
            if joint_built {
                joint_now = true;
                self.large_fresh = true;
            } else if escalated_large || !sys.can_escalate && can_escalate {
                self.escalated = true;
                self.large_fresh = true;
                self.base_iters = None;
            }
            escalated_now = escalated_large || joint_now;
            if ctx.verbose >= 2 {
                ctx.log(&format!("fereg: pcg {} iterations (escalated: {}, joint: {})", out.iterations, escalated_now, self.pre.joint.is_some()));
            }
            if joint_now {
                // The joint factor replaces the block preconditioners: free their factors.
                self.pre.large.iter_mut().for_each(|bp| *bp = BlockPrecond::Jacobi);
                self.joint.stale_iters = None;
            } else if self.pre.joint.is_some() {
                if !self.large_fresh {
                    self.joint.stale_iters = Some(iterations);
                }
            } else if self.escalated {
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
        Ok(SolveInfo { iterations, escalated: escalated_now, coefs, exact })
    }

    fn fit_rows(&self, c: &FeCoefs, col: usize, pool: &Pool) -> Vec<f64> {
        let v = c.col(col);
        let m = self.st.m;
        Ops { st: &self.st, ws: &self.ws }.fit_rows(&v[..m], &v[m..], pool)
    }

    fn term_values(&self, c: &FeCoefs, col: usize) -> Vec<Vec<f64>> {
        let v = c.col(col);
        let st = &self.st;
        let (Some(elim), Some(norm)) = (st.fe1, self.norm_fe1) else { return Vec::new() };
        let nterms = st.blocks.len() + 1;
        let mut out: Vec<Vec<f64>> = vec![Vec::new(); nterms];
        // Level weights of every term: n1 for the eliminated one, n_r for the blocks.
        let mut lw: Vec<&[f64]> = vec![&[]; nterms];
        out[elim] = v[st.m..].to_vec();
        lw[elim] = &self.ws.n1;
        for blk in &st.blocks {
            out[blk.term] = v[blk.off..blk.off + blk.g].to_vec();
            lw[blk.term] = &self.ws.nr[blk.off..blk.off + blk.g];
        }
        // Normalization: every term but FE1 (`fe1_index`, not necessarily the eliminated term)
        // has weighted mean 0; FE1 absorbs the shifts.
        let mut shift = 0.0;
        for t in (0..nterms).filter(|&t| t != norm) {
            let mean: f64 = out[t].iter().zip(lw[t]).map(|(a, w)| a * w).sum::<f64>() / self.ws.sw;
            out[t].iter_mut().for_each(|a| *a -= mean);
            shift += mean;
        }
        out[norm].iter_mut().for_each(|a| *a += shift);
        out
    }
}
