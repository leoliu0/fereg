//! GLM families: inverse link, IRLS weight and working response, deviance and log-likelihood
//! pieces for Poisson (log link) and logit.

use crate::api::Family;

/// Mata `epsilon(1)`.
pub(crate) const EPS1: f64 = f64::EPSILON;

/// Lower bound of mu (Mata `epsilon(100)`, ppmlhdfe `_vector_scalar_max(mu, epsilon(100))`);
/// logit also caps mu at 1 - `MU_MIN`.
pub(crate) const MU_MIN: f64 = 100.0 * f64::EPSILON;

/// Numerically stable inverse logit.
#[inline]
pub(crate) fn invlogit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// ln(1 + e^x) without overflow.
#[inline]
pub(crate) fn softplus(x: f64) -> f64 {
    if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() }
}

/// mu from the linear predictor eta (offset included), clamped as ppmlhdfe does.
#[inline]
pub(crate) fn mean(f: Family, eta: f64) -> f64 {
    match f {
        Family::Logit => invlogit(eta).clamp(MU_MIN, 1.0 - MU_MIN),
        _ => eta.exp().max(MU_MIN),
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
/// eta + (y - mu)/(mu(1 - mu)) (logit, written as eta + 1/mu for y = 1 and eta - 1/(1 - mu) for y = 0).
#[inline]
pub(crate) fn working(f: Family, eta: f64, y: f64, mu: f64) -> f64 {
    match f {
        Family::Logit => {
            if y > 0.0 {
                eta + 1.0 / mu
            } else {
                eta - 1.0 / (1.0 - mu)
            }
        }
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

    #[test]
    fn logit_pieces() {
        assert!((invlogit(0.3) - 1.0 / (1.0 + (-0.3f64).exp())).abs() < 1e-16);
        assert!((softplus(800.0) - 800.0).abs() < 1e-12);
        assert!(softplus(-800.0) >= 0.0);
        let mu = 0.3;
        assert!((working(Family::Logit, 0.1, 1.0, mu) - (0.1 + (1.0 - mu) / (mu * (1.0 - mu)))).abs() < 1e-14);
        assert!((working(Family::Logit, 0.1, 0.0, mu) - (0.1 - mu / (mu * (1.0 - mu)))).abs() < 1e-14);
    }
}
