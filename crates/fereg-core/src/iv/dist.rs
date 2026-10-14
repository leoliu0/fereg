//! Upper-tail probabilities of the chi^2 and F distributions (Stata `chi2tail()` / `chiprob()` and
//! `Ftail()`), via the regularized incomplete gamma and beta functions (Lentz continued
//! fractions, relative accuracy ~= 1e-14 including far tails).

const EPS: f64 = 1e-16;
const TINY: f64 = 1e-300;

/// ln Gamma(x) for x > 0 (Lanczos, g = 7, n = 9; relative error < 1e-15).
pub(crate) fn ln_gamma(x: f64) -> f64 {
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
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularized lower incomplete gamma P(a, x) by its series (x < a + 1).
fn gamma_p_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..100_000 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPS {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Regularized upper incomplete gamma Q(a, x) by its continued fraction (x >= a + 1).
fn gamma_q_cf(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..100_000 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Stata `chi2tail(df, x)`.
pub(crate) fn chi2tail(df: f64, x: f64) -> f64 {
    if !(df > 0.0) || x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    let (a, x) = (0.5 * df, 0.5 * x);
    if x < a + 1.0 { 1.0 - gamma_p_series(a, x) } else { gamma_q_cf(a, x) }
}

/// Continued fraction of the incomplete beta function (Numerical Recipes `betacf`).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..100_000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta I_x(a, b).
pub(crate) fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let lbt = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    if x < (a + 1.0) / (a + b + 2.0) {
        lbt.exp() * beta_cf(a, b, x) / a
    } else {
        1.0 - lbt.exp() * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Stata `Ftail(df1, df2, f)`.
pub(crate) fn ftail(df1: f64, df2: f64, f: f64) -> f64 {
    if !(df1 > 0.0) || !(df2 > 0.0) || f.is_nan() {
        return f64::NAN;
    }
    if f <= 0.0 {
        return 1.0;
    }
    // P(F > f) = I_{df2/(df2 + df1*f)}(df2/2, df1/2), evaluated on the side that avoids cancellation.
    let x = df2 / (df2 + df1 * f);
    if x < (0.5 * df2 + 1.0) / (0.5 * (df1 + df2) + 2.0) {
        beta_inc(0.5 * df2, 0.5 * df1, x)
    } else {
        1.0 - beta_inc(0.5 * df1, 0.5 * df2, 1.0 - x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs()
    }

    #[test]
    fn matches_stata_values() {
        // Values from Stata 19.5: di %21.17g chi2tail(...) / Ftail(...).
        assert!(rel(chi2tail(1.0, 12.959595654620562), 0.0003182859154985906) < 1e-10);
        assert!(rel(chi2tail(4.0, 172.78352741617391), 2.64243457887000e-36) < 1e-9);
        assert!(rel(ftail(2.0, 2201.0, 6.593911762259969), 0.0013958722217602) < 1e-9);
        assert!(rel(ftail(4.0, 8.0, 38.140193065404041), 0.0000295873853329) < 1e-9);
        assert!(rel(ftail(3.0, 8.0, 490.39618438045113), 2.10099380317007e-09) < 1e-9);
        assert_eq!(chi2tail(3.0, 0.0), 1.0);
    }
}
