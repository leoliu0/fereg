//! Wild bootstrap (`wb_*`) key parsing and extra matrix storage for the Stata plugin.

use fereg_core::wildboot::{BootMethod, BootType, BootWeights, Hypothesis, PvalType, WildBootResult, WildBootSpec};
use fereg_core::RfError;

use crate::args::Args;
use crate::{io, spi};

/// Parsed wild bootstrap request from `plugin call` keys (`wb_*`).
pub struct WbRequest {
    pub hyp: Hypothesis,
    pub spec: WildBootSpec,
    pub ci_mat: Option<String>,
    pub savet_mat: Option<String>,
}

fn bad(msg: impl Into<String>) -> RfError {
    RfError::InvalidInput(format!("fereg plugin: {}", msg.into()))
}

fn parse_csv_f64(s: &str, key: &str) -> Result<Vec<f64>, RfError> {
    if s.is_empty() {
        return Err(bad(format!("{key}= must not be empty")));
    }
    s.split(',')
        .map(|tok| {
            let v: f64 = tok.parse().map_err(|_| bad(format!("invalid float `{tok}` in {key}=")))?;
            if !v.is_finite() {
                return Err(bad(format!("non-finite float `{tok}` in {key}=")));
            }
            Ok(v)
        })
        .collect()
}

/// Parses `wb_*` keys; returns `Ok(None)` when `wb_q=` is absent so unknown `wb_*` keys are caught by `Args::finish`.
pub fn parse(a: &Args) -> Result<Option<WbRequest>, RfError> {
    let Some(q_str) = a.str_opt("wb_q") else {
        return Ok(None);
    };
    let q: usize = q_str.parse().map_err(|_| bad(format!("invalid value `{q_str}` for wb_q")))?;
    if q == 0 {
        return Err(bad("wb_q= must be >= 1"));
    }
    let r_mat = parse_csv_f64(a.str_req("wb_R")?, "wb_R")?;
    if !r_mat.len().is_multiple_of(q) {
        return Err(bad(format!("wb_R length {} is not divisible by wb_q={q}", r_mat.len())));
    }
    let k_cols = r_mat.len() / q;
    let r_vec = parse_csv_f64(a.str_req("wb_r")?, "wb_r")?;
    if r_vec.len() != q {
        return Err(bad(format!("wb_r length {} does not match wb_q={q}", r_vec.len())));
    }
    let hyp = Hypothesis::new(r_mat, r_vec, q, k_cols)?;

    let reps: usize = a.num_or("wb_reps", 999)?;
    let weights = match a.str_or("wb_weights", "rademacher") {
        "rademacher" => BootWeights::Rademacher,
        "webb" => BootWeights::Webb,
        "mammen" => BootWeights::Mammen,
        "normal" => BootWeights::Normal,
        v => return Err(bad(format!("invalid wb_weights `{v}`"))),
    };
    let boot_type = match a.str_or("wb_boottype", "11") {
        "11" => BootType::T11,
        "21" => BootType::T21,
        "31" => BootType::T31,
        "13" => BootType::T13,
        "33" => BootType::T33,
        v => return Err(bad(format!("invalid wb_boottype `{v}`"))),
    };
    let method = match a.str_or("wb_method", "auto") {
        "auto" => BootMethod::Auto,
        "wild" => BootMethod::WildCluster,
        "score" => BootMethod::Score,
        "wre" => BootMethod::Wre,
        "hetero" => BootMethod::Hetero,
        v => return Err(bad(format!("invalid wb_method `{v}`"))),
    };
    let impose_null = a.flag("wb_null", true)?;
    let pval_type = match a.str_or("wb_pvaltype", "twotailed") {
        "twotailed" => PvalType::TwoTailed,
        "equal" => PvalType::EqualTailed,
        "greater" => PvalType::Greater,
        "less" => PvalType::Less,
        v => return Err(bad(format!("invalid wb_pvaltype `{v}`"))),
    };
    let conf_int = a.flag("wb_ci", true)?;
    let level_pct: f64 = a.num_or("wb_level", 95.0)?;
    if !level_pct.is_finite() || level_pct <= 0.0 || level_pct >= 100.0 {
        return Err(bad("wb_level= must be in (0, 100)"));
    }
    let seed: u64 = a.num_or("wb_seed", 0)?;
    let tol: f64 = a.num_or("wb_tol", 1e-6)?;
    let maxiter: usize = a.num_or("wb_maxiter", 40)?;
    let ci_mat = a.str_opt("wb_ci_mat").map(str::to_string);
    let savet_mat = a.str_opt("wb_savet").map(str::to_string);

    let spec = WildBootSpec {
        reps,
        weights,
        boot_type,
        method,
        impose_null,
        pval_type,
        conf_int,
        level: level_pct / 100.0,
        seed,
        tol,
        maxiter,
    };
    spec.validate(q)?;
    Ok(Some(WbRequest { hyp, spec, ci_mat, savet_mat }))
}

/// Stores optional confidence interval matrix (`wb_ci_mat`) and bootstrap t-draw column vector (`wb_savet`).
pub fn store_extra(wb_res: &WildBootResult, req: &WbRequest) -> Result<(), RfError> {
    if let (Some(name), Some((lo, hi))) = (&req.ci_mat, wb_res.conf_int) {
        io::store_matrix(name, 1, 2, &[lo, hi])?;
    }
    if let Some(name) = &req.savet_mat {
        let (mr, mc) = spi::mat_dims(name);
        let need = wb_res.t_boot.len();
        if mc != 1 || mr < need as i32 {
            return Err(RfError::Code(
                503,
                format!("fereg: savet matrix {name} has wrong dimensions ({mr}x{mc}, need >={need}x1)"),
            ));
        }
        let mut padded = vec![f64::NAN; mr as usize];
        padded[..need].copy_from_slice(&wb_res.t_boot);
        spi::mat_store(name, mr as usize, 1, &padded)?;
    }
    Ok(())
}
