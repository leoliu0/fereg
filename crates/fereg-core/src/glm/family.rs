//! GLM families: inverse link, IRLS weight and working response, deviance and log-likelihood
//! pieces for Poisson (log link) and logit.

use std::sync::OnceLock;

use crate::api::Family;

/// Mata `epsilon(1)`.
pub(crate) const EPS1: f64 = f64::EPSILON;

/// Lower bound of mu (Mata `epsilon(100)`, ppmlhdfe `_vector_scalar_max(mu, epsilon(100))`);
/// logit also caps mu at 1 - `MU_MIN`.
pub(crate) const MU_MIN: f64 = 100.0 * f64::EPSILON;

// Elementary functions of the row passes: plain arithmetic without branches, tables or library
// calls, so the loops over the rows vectorize, and a row gets the same bits whichever
// instruction set runs its loop (nothing is contracted or reordered) and on every platform.

const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000);
const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76);
const INV_LN2: f64 = f64::from_bits(0x3ff7_1547_652b_82fe);
/// 1.5 * 2^52: adding and subtracting it rounds to the nearest integer, whose two's complement
/// is in the low bits of the sum.
const ROUND: f64 = 6_755_399_441_055_744.0;

/// e^x with an error below 1 ulp (fdlibm `__ieee754_exp`): 0 below about -745.13, inf above
/// about 709.78.
#[inline(always)]
pub(crate) fn exp(x: f64) -> f64 {
    const P: [f64; 5] =
        [1.666_666_666_666_660_190_37e-01, -2.777_777_777_701_559_338_42e-03, 6.613_756_321_437_934_361_17e-05, -1.653_390_220_546_525_153_90e-06, 4.138_136_797_057_238_460_39e-08];
    // Beyond these bounds the result is 0 or inf and k below stays within 11 bits; NaN passes.
    let x = if x > 750.0 {
        750.0
    } else if x < -750.0 {
        -750.0
    } else {
        x
    };
    // x = k ln 2 + r with |r| <= ln(2) / 2; e^r from r = hi - lo.
    let kd = x * INV_LN2 + ROUND;
    let k = kd - ROUND;
    let hi = x - k * LN2_HI;
    let lo = k * LN2_LO;
    let r = hi - lo;
    let t = r * r;
    let c = r - t * (P[0] + t * (P[1] + t * (P[2] + t * (P[3] + t * P[4]))));
    let y = 1.0 - ((lo - (r * c) / (2.0 - c)) - hi);
    // y * 2^k in two exact steps (2^k itself can be out of range): k + 2048, split in halves.
    let biased = (kd.to_bits() & ((1 << 52) - 1)).wrapping_sub(1 << 51).wrapping_add(2048);
    let half = biased >> 1;
    y * f64::from_bits(half.wrapping_sub(1) << 52) * f64::from_bits(biased.wrapping_sub(half).wrapping_sub(1) << 52)
}

/// ln(1 + x) for 0 <= x <= 1 with an error below 1 ulp (fdlibm `log1p`).
#[inline(always)]
fn ln_1p_unit(x: f64) -> f64 {
    const L: [f64; 7] = [
        6.666_666_666_666_735_130e-01,
        3.999_999_999_940_941_908e-01,
        2.857_142_874_366_239_149e-01,
        2.222_219_843_214_978_396e-01,
        1.818_357_216_161_805_012e-01,
        1.531_383_769_920_937_332e-01,
        1.479_819_860_511_658_591e-01,
    ];
    // 1 + x = 2^k (1 + f) with sqrt(2)/2 < 1 + f < sqrt(2): k = 0 and f = x below sqrt(2) - 1,
    // else k = 1, with the rounding error of 1 + x in c.
    let u = 1.0 + x;
    let high = x >= f64::from_bits(0x3fda_827a_0000_0000);
    let f = if high { 0.5 * u - 1.0 } else { x };
    let c = if high { (x - (u - 1.0)) / u } else { 0.0 };
    let k = if high { 1.0 } else { 0.0 };
    let hfsq = 0.5 * f * f;
    let s = f / (2.0 + f);
    let z = s * s;
    let r = z * (L[0] + z * (L[1] + z * (L[2] + z * (L[3] + z * (L[4] + z * (L[5] + z * L[6]))))));
    k * LN2_HI - ((hfsq - (s * (hfsq + r) + (k * LN2_LO + c))) - f)
}

/// Inverse logit and ln(1 + e^x), neither overflowing, from one exponential.
#[inline(always)]
pub(crate) fn invlogit_softplus(x: f64) -> (f64, f64) {
    let e = exp(-x.abs());
    let l = ln_1p_unit(e);
    ((if x >= 0.0 { 1.0 } else { e }) / (1.0 + e), if x > 0.0 { x + l } else { l })
}

/// Numerically stable inverse logit.
#[inline]
pub(crate) fn invlogit(x: f64) -> f64 {
    invlogit_softplus(x).0
}

/// ln(1 + e^x) without overflow.
#[inline]
pub(crate) fn softplus(x: f64) -> f64 {
    invlogit_softplus(x).1
}

/// mu from the linear predictor eta (offset included), clamped as ppmlhdfe does.
#[inline(always)]
pub(crate) fn mean(f: Family, eta: f64) -> f64 {
    match f {
        Family::Logit => invlogit(eta).clamp(MU_MIN, 1.0 - MU_MIN),
        _ => {
            // `max` without the library call (a NaN gives the bound too).
            let m = exp(eta);
            if m >= MU_MIN { m } else { MU_MIN }
        }
    }
}

/// Link function eta = g(mu).
#[inline]
pub(crate) fn link(f: Family, mu: f64) -> f64 {
    match f {
        Family::Logit => (mu / (1.0 - mu)).ln(),
        _ => mu.ln(),
    }
}

/// IRLS weight per unit of true weight: mu (Poisson) or mu(1 - mu) (logit).
#[inline]
pub(crate) fn irls_weight(f: Family, mu: f64) -> f64 {
    match f {
        Family::Logit => mu * (1.0 - mu),
        _ => mu,
    }
}

/// Working response without the offset: eta - 1 + y/mu (Poisson, ppmlhdfe) or
/// eta + (y - mu)/(mu(1 - mu)) (logit: eta + 1/mu for y = 1 and eta - 1/(1 - mu) for y = 0,
/// written with one division and no branch on y).
#[inline]
pub(crate) fn working(f: Family, eta: f64, y: f64, mu: f64) -> f64 {
    match f {
        Family::Logit => eta + 1.0 / if y > 0.0 { mu } else { mu - 1.0 },
        _ => eta - 1.0 + y / mu,
    }
}

/// Logit log-likelihood contribution y*eta - ln(1 + e^eta) (eta without clamping).
#[inline]
pub(crate) fn logit_ll(y: f64, eta: f64) -> f64 {
    y * eta - softplus(eta)
}

/// ln Gamma(x) for x > 0 (Lanczos, g = 7, n = 9; relative error ~= 1e-15).
pub(crate) fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Gamma(x)Gamma(1 - x) = pi / sin(pi x).
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).abs().ln() - ln_gamma(1.0 - x);
    }
    if x == 1.0 || x == 2.0 {
        return 0.0;
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// `ln_gamma(y + 1.0)` with an L1-resident table for integers `0 <= y <= 256` (exact bit match
/// with [`ln_gamma`]).
#[inline]
pub(crate) fn ln_gamma_plus1(y: f64) -> f64 {
    static TABLE: OnceLock<[f64; 257]> = OnceLock::new();
    if (0.0..=256.0).contains(&y) {
        let k = y as usize;
        if k as f64 == y {
            return TABLE.get_or_init(|| std::array::from_fn(|i| ln_gamma(i as f64 + 1.0)))[k];
        }
    }
    ln_gamma(y + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_matches_factorials() {
        let mut lf = 0.0f64;
        for k in 1..60u32 {
            lf += f64::from(k).ln();
            let got = ln_gamma(f64::from(k) + 1.0);
            assert!((got - lf).abs() <= 1e-13 * lf.max(1.0), "k={k}: {got} vs {lf}");
        }
        // Gamma(1/2) = sqrt pi
        assert!((ln_gamma(0.5) - 0.5 * std::f64::consts::PI.ln()).abs() < 1e-14);
    }

    /// Units in the last place between two finite values of one sign. The functions below and
    /// the library's are each within 1 ulp of the exact value, so they differ by at most 2.
    fn ulps(a: f64, b: f64) -> u64 {
        a.to_bits().abs_diff(b.to_bits())
    }

    #[test]
    fn exp_agrees_with_the_library() {
        let mut rng = crate::rng::SplitMix64::new(11);
        for i in 0..2_000_000 {
            let u = rng.next_f64();
            // The whole range, the usual one, and the subnormal results.
            let x = match i % 3 {
                0 => (u - 0.5) * 1418.0,
                1 => (u - 0.5) * 40.0,
                _ => -708.0 - 37.0 * u,
            };
            assert!(ulps(exp(x), x.exp()) <= 2, "exp({x:e}) = {:e}, library {:e}", exp(x), x.exp());
        }
        for x in [0.0, -0.0, 5e-324, -1e-300, 1e-20] {
            assert_eq!(exp(x), 1.0, "{x:e}");
        }
        for x in [709.782_712_893_385, 750.0, 1e300, f64::INFINITY] {
            assert_eq!(exp(x), f64::INFINITY, "{x:e}");
        }
        for x in [-745.133_219_101_941_2, -750.0, -1e300, f64::NEG_INFINITY] {
            assert_eq!(exp(x), 0.0, "{x:e}");
        }
        assert_eq!(exp(-745.133_219_101_941_1), 5e-324);
        assert!(exp(709.782_712_893_384).is_finite() && exp(f64::NAN).is_nan());
    }

    #[test]
    fn ln_1p_unit_agrees_with_the_library() {
        let mut rng = crate::rng::SplitMix64::new(12);
        let split = f64::from_bits(0x3fda_827a_0000_0000);
        for i in 0..2_000_000 {
            let u = rng.next_f64();
            // [0, 1], small values, both sides of the split, and e^-|eta|.
            let x = match i % 4 {
                0 => u,
                1 => 1e-3 * u,
                2 => split + 1e-9 * (u - 0.5),
                _ => (-745.0 * u).exp(),
            };
            assert!(ulps(ln_1p_unit(x), x.ln_1p()) <= 2, "ln_1p({x:e}) = {:e}, library {:e}", ln_1p_unit(x), x.ln_1p());
        }
        assert_eq!(ln_1p_unit(0.0), 0.0);
        assert_eq!(ln_1p_unit(5e-324), 5e-324);
        assert_eq!(ln_1p_unit(1.0), std::f64::consts::LN_2);
        assert!(ln_1p_unit(f64::NAN).is_nan());
    }

    #[test]
    fn logit_pieces() {
        assert!((invlogit(0.3) - 1.0 / (1.0 + (-0.3f64).exp())).abs() < 4e-16);
        assert_eq!(invlogit(0.0), 0.5);
        assert_eq!((invlogit(800.0), invlogit(-800.0)), (1.0, 0.0));
        assert!((softplus(800.0) - 800.0).abs() < 1e-12);
        assert!(softplus(-800.0) >= 0.0);
        assert!(invlogit(f64::NAN).is_nan() && softplus(f64::NAN).is_nan());
        let mu = 0.3;
        assert!((working(Family::Logit, 0.1, 1.0, mu) - (0.1 + (1.0 - mu) / (mu * (1.0 - mu)))).abs() < 1e-14);
        assert!((working(Family::Logit, 0.1, 0.0, mu) - (0.1 - mu / (mu * (1.0 - mu)))).abs() < 1e-14);
        // The same bits as the two-division form.
        assert_eq!(working(Family::Logit, 0.1, 0.0, mu), 0.1 - 1.0 / (1.0 - mu));
        assert_eq!(mean(Family::Poisson, f64::NAN), MU_MIN);
    }
}
