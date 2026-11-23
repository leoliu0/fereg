//! Deterministic wild bootstrap weight generation (Rademacher, Webb 6-point, Mammen, Normal).
//!
//! Every draw `b` uses `SplitMix64::stream(seed, b)` (or per `(b, chunk)` for unclustered row
//! streams) so outputs are bit-identical regardless of thread count.

use rayon::prelude::*;

use crate::error::RfError;
use crate::par::{self, Pool};
use crate::rng::SplitMix64;

/// Wild bootstrap auxiliary weight distribution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BootWeights {
    #[default]
    Rademacher,
    Webb,
    Mammen,
    Normal,
}

impl BootWeights {
    pub fn name(self) -> &'static str {
        match self {
            BootWeights::Rademacher => "rademacher",
            BootWeights::Webb => "webb",
            BootWeights::Mammen => "mammen",
            BootWeights::Normal => "normal",
        }
    }
}

/// 6-point Webb (2023) distribution values with equal probabilities `1/6`.
const WEBB_VALS: [f64; 6] = [-1.224_744_871_391_589, -1.0, -0.707_106_781_186_547_6, 0.707_106_781_186_547_6, 1.0, 1.224_744_871_391_589];

/// Resolves effective `(reps, full_enumeration)` for the requested `weights`, `reps`, and cluster count.
/// For clustered Rademacher (`G <= 20` and `2^G <= reps`) or Webb (`G <= 8` and `6^G <= reps`),
/// switches to exact full enumeration.
pub(crate) fn resolve_reps(weights: BootWeights, reps: usize, g: Option<usize>) -> (usize, bool) {
    let Some(g) = g else {
        return (reps, false);
    };
    if g == 0 {
        return (reps, false);
    }
    match weights {
        BootWeights::Rademacher if g <= 20 => {
            let total = 1usize << g;
            if total <= reps { (total, true) } else { (reps, false) }
        }
        BootWeights::Webb if g <= 8 => {
            if let Some(total) = 6usize.checked_pow(g as u32)
                && total <= reps
            {
                return (total, true);
            }
            (reps, false)
        }
        _ => (reps, false),
    }
}

/// Fills `out[0..len]` from `rng` using `weights`.
#[inline]
pub(crate) fn fill_from_rng(weights: BootWeights, rng: &mut SplitMix64, out: &mut [f64]) {
    match weights {
        BootWeights::Rademacher => {
            // Draw 64 bits at a time? One u64 gives 64 signs; still keep simple and fast.
            let mut bits = 0u64;
            let mut rem = 0u32;
            for v in out.iter_mut() {
                if rem == 0 {
                    bits = rng.next_u64();
                    rem = 64;
                }
                *v = if bits & 1 == 0 { 1.0 } else { -1.0 };
                bits >>= 1;
                rem -= 1;
            }
        }
        BootWeights::Webb => {
            for v in out.iter_mut() {
                // Mul-shift trick in [0, 6) from high 32 bits avoids bias.
                let idx = (((rng.next_u64() >> 32) * 6) >> 32) as usize;
                *v = WEBB_VALS[idx];
            }
        }
        BootWeights::Mammen => {
            let sqrt5 = 5.0_f64.sqrt();
            let v1 = -(sqrt5 - 1.0) / 2.0;
            let v2 = (sqrt5 + 1.0) / 2.0;
            let p1 = (sqrt5 + 1.0) / (2.0 * sqrt5);
            for v in out.iter_mut() {
                *v = if rng.next_f64() < p1 { v1 } else { v2 };
            }
        }
        BootWeights::Normal => {
            let mut i = 0;
            while i + 1 < out.len() {
                let u1 = (1.0 - rng.next_f64()).max(f64::MIN_POSITIVE);
                let u2 = rng.next_f64();
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = std::f64::consts::TAU * u2;
                out[i] = r * theta.cos();
                out[i + 1] = r * theta.sin();
                i += 2;
            }
            if i < out.len() {
                let u1 = (1.0 - rng.next_f64()).max(f64::MIN_POSITIVE);
                let u2 = rng.next_f64();
                out[i] = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            }
        }
    }
}

/// Fills `out[0..g]` for clustered draw `b` (supports full enumeration when `full_enum == true`).
pub(crate) fn fill_cluster_draw(weights: BootWeights, full_enum: bool, seed: u64, b: usize, out: &mut [f64]) {
    let g = out.len();
    if full_enum {
        match weights {
            BootWeights::Rademacher => {
                for (c, v) in out.iter_mut().enumerate() {
                    *v = if (b >> c) & 1 == 0 { 1.0 } else { -1.0 };
                }
                return;
            }
            BootWeights::Webb => {
                let mut x = b;
                for v in out.iter_mut() {
                    *v = WEBB_VALS[x % 6];
                    x /= 6;
                }
                return;
            }
            _ => {}
        }
    }
    let _ = g;
    let mut rng = SplitMix64::stream(seed, b as u64);
    fill_from_rng(weights, &mut rng, out);
}

/// Generates `reps x g` row-major weight matrix `V` (`v[b * g + c]` = draw `b`, cluster `c`).
pub(crate) fn cluster_matrix(weights: BootWeights, g: usize, reps: usize, full_enum: bool, seed: u64, pool: &Pool) -> Result<Vec<f64>, RfError> {
    let mut v = par::alloc_vec(reps * g, 0.0)?;
    if reps * g <= par::CHUNK || pool.threads() <= 1 {
        for (b, row) in v.chunks_exact_mut(g).enumerate() {
            fill_cluster_draw(weights, full_enum, seed, b, row);
        }
    } else {
        pool.install(|| {
            v.par_chunks_mut(g).enumerate().for_each(|(b, row)| {
                fill_cluster_draw(weights, full_enum, seed, b, row);
            });
        });
    }
    Ok(v)
}

/// Deterministic RNG for unclustered draw `b`, fixed row chunk `chunk_idx`.
#[inline]
pub(crate) fn hetero_chunk_rng(seed: u64, draw: usize, chunk_idx: usize) -> SplitMix64 {
    let mixed = seed ^ (draw as u64).wrapping_mul(0x517C_C1B7_2722_0A95);
    SplitMix64::stream(mixed, chunk_idx as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_enumeration_triggers_and_covers_space() {
        assert_eq!(resolve_reps(BootWeights::Rademacher, 999, Some(4)), (16, true));
        assert_eq!(resolve_reps(BootWeights::Rademacher, 999, Some(10)), (999, false));
        assert_eq!(resolve_reps(BootWeights::Webb, 999, Some(3)), (216, true));
        assert_eq!(resolve_reps(BootWeights::Webb, 999, Some(4)), (999, false));
        assert_eq!(resolve_reps(BootWeights::Mammen, 999, Some(3)), (999, false));
        assert_eq!(resolve_reps(BootWeights::Rademacher, 999, None), (999, false));
    }

    #[test]
    fn webb_constants_have_unit_variance_and_zero_mean() {
        let mean: f64 = WEBB_VALS.iter().sum::<f64>() / 6.0;
        let var: f64 = WEBB_VALS.iter().map(|x| x * x).sum::<f64>() / 6.0;
        assert!(mean.abs() < 1e-15);
        assert!((var - 1.0).abs() < 1e-15);
    }
}
