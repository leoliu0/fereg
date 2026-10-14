//! Distribution functions for p-values and confidence intervals (hosts call these so Python and
//! R report identical inference without depending on SciPy). Accuracy target: 1e-14 relative
//! in the body of each distribution; +/-inf and NaN inputs follow R's `pnorm`/`qnorm`/`pt`/`qt`.
//!
//! The regularized incomplete gamma and beta functions use continued fractions (Lentz) or power
//! series with prefactors computed as R computes them (Stirling corrections, `bd0`, a stable log
//! beta), so the relative accuracy does not degrade with large degrees of freedom.

use std::f64::consts::PI;

/// ln sqrt(2*pi).
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_7;
const EPS: f64 = 1e-16;
const TINY: f64 = 1e-300;
const MAX_ITER: usize = 1_000_000;

/// Standard normal CDF, lower tail.
pub fn pnorm(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }
    // Phi(x) = 1/2*(1 +/- P(1/2, x^2/2)); the smaller tail is 1/2*Q(1/2, x^2/2).
    let (p, q) = gamma_pq(0.5, 0.5 * x * x);
    if x < 0.0 { 0.5 * q } else { 0.5 + 0.5 * p }
}

/// Standard normal quantile (Wichura's AS 241, as R's `qnorm`).
pub fn qnorm(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return q
            * (((((((r * 2_509.080_928_730_122_7 + 33_430.575_583_588_13) * r + 67_265.770_927_008_7) * r + 45_921.953_931_549_87) * r
                + 13_731.693_765_509_46)
                * r
                + 1_971.590_950_306_551_4)
                * r
                + 133.141_667_891_784_38)
                * r
                + 3.387_132_872_796_366_5)
            / (((((((r * 5_226.495_278_852_545 + 28_729.085_735_721_943) * r + 39_307.895_800_092_71) * r + 21_213.794_301_586_597) * r
                + 5_394.196_021_424_751)
                * r
                + 687.187_007_492_057_9)
                * r
                + 42.313_330_701_600_91)
                * r
                + 1.0);
    }
    let r = (-(if q < 0.0 { p } else { 1.0 - p }).ln()).sqrt();
    let val = if r <= 5.0 {
        let r = r - 1.6;
        (((((((r * 7.745_450_142_783_414e-4 + 0.022_723_844_989_269_184) * r + 0.241_780_725_177_450_6) * r + 1.270_458_252_452_368_4) * r
            + 3.647_848_324_763_204_5)
            * r
            + 5.769_497_221_460_691)
            * r
            + 4.630_337_846_156_545)
            * r
            + 1.423_437_110_749_683_5)
            / (((((((r * 1.050_750_071_644_416_9e-9 + 5.475_938_084_995_345e-4) * r + 0.015_198_666_563_616_457) * r + 0.148_103_976_427_480_08)
                * r
                + 0.689_767_334_985_1)
                * r
                + 1.676_384_830_183_803_8)
                * r
                + 2.053_191_626_637_759)
                * r
                + 1.0)
    } else {
        let r = r - 5.0;
        (((((((r * 2.010_334_399_292_288_1e-7 + 2.711_555_568_743_487_6e-5) * r + 0.001_242_660_947_388_078_4) * r + 0.026_532_189_526_576_124)
            * r
            + 0.296_560_571_828_504_9)
            * r
            + 1.784_826_539_917_291_3)
            * r
            + 5.463_784_911_164_114)
            * r
            + 6.657_904_643_501_103)
            / (((((((r * 2.044_263_103_389_939_7e-15 + 1.421_511_758_316_446e-7) * r + 1.846_318_317_510_054_8e-5) * r + 7.868_691_311_456_133e-4)
                * r
                + 0.014_875_361_290_850_615)
                * r
                + 0.136_929_880_922_735_8)
                * r
                + 0.599_832_206_555_887_9)
                * r
                + 1.0)
    };
    if q < 0.0 { -val } else { val }
}

/// Student t CDF, lower tail.
pub fn pt(t: f64, df: f64) -> f64 {
    if t.is_nan() || df.is_nan() || df <= 0.0 {
        return f64::NAN;
    }
    if t.is_infinite() {
        return if t > 0.0 { 1.0 } else { 0.0 };
    }
    if df.is_infinite() {
        return pnorm(t);
    }
    // P(|T| > |t|) = I_{df/(df+t^2)}(df/2, 1/2); both arguments of the beta are computed directly.
    let tt = t * t;
    let (x, y) = (df / (df + tt), tt / (df + tt));
    let (two_sided, inner) = beta_lower_upper(0.5 * df, 0.5, x, y);
    if t <= 0.0 { 0.5 * two_sided } else { 0.5 + 0.5 * inner }
}

/// Student t quantile: safeguarded Newton iterations on [`pt`] from a Cornish-Fisher start.
pub fn qt(p: f64, df: f64) -> f64 {
    if p.is_nan() || df.is_nan() || df <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }
    if df.is_infinite() {
        return qnorm(p);
    }
    // Solve in the lower tail (q < 0), where pt has full relative accuracy.
    let (pp, sign) = if p < 0.5 { (p, -1.0) } else { (1.0 - p, 1.0) };
    let z = qnorm(pp);
    let mut q = z + (z * z * z + z) / (4.0 * df) + (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / (96.0 * df * df);
    if !(q < 0.0) || !q.is_finite() {
        q = z;
    }
    // Bracket [lo, hi] with pt(lo) <= pp <= pt(hi), hi <= 0.
    let (mut lo, mut hi) = (q, 0.0f64);
    while pt(lo, df) > pp {
        hi = lo;
        lo *= 2.0;
        if !lo.is_finite() {
            return sign * f64::INFINITY;
        }
    }
    let ln_c = -0.5 * df.ln() - lbeta(0.5 * df, 0.5);
    for _ in 0..200 {
        let f = pt(q, df) - pp;
        if f == 0.0 {
            break;
        }
        if f > 0.0 {
            hi = q;
        } else {
            lo = q;
        }
        let dens = (ln_c - 0.5 * (df + 1.0) * (q * q / df).ln_1p()).exp();
        let mut next = q - f / dens;
        if !(next > lo && next < hi) {
            next = 0.5 * (lo + hi);
        }
        let done = (next - q).abs() <= 1e-15 * q.abs();
        q = next;
        if done || hi - lo <= 1e-15 * q.abs() {
            break;
        }
    }
    -sign * q
}

/// F distribution upper tail P(F > f).
pub fn pf_upper(f: f64, df1: f64, df2: f64) -> f64 {
    if f.is_nan() || df1.is_nan() || df2.is_nan() || df1 <= 0.0 || df2 <= 0.0 {
        return f64::NAN;
    }
    if f <= 0.0 {
        return 1.0;
    }
    if f.is_infinite() {
        return 0.0;
    }
    if df2.is_infinite() {
        return if df1.is_infinite() { if f < 1.0 { 1.0 } else { 0.0 } } else { pchisq_upper(f * df1, df1) };
    }
    if df1.is_infinite() {
        return 1.0 - pchisq_upper(df2 / f, df2);
    }
    let d = df2 + df1 * f;
    let (x, y) = (df2 / d, df1 * f / d);
    // As R: the beta's first parameter is not squeezed against 1.
    if df1 * f > df2 { beta_lower_upper(0.5 * df2, 0.5 * df1, x, y).0 } else { beta_lower_upper(0.5 * df1, 0.5 * df2, y, x).1 }
}

/// chi^2 upper tail P(X > x).
pub fn pchisq_upper(x: f64, df: f64) -> f64 {
    if x.is_nan() || df.is_nan() || df < 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if df == 0.0 || x.is_infinite() {
        return 0.0;
    }
    gamma_pq(0.5 * df, 0.5 * x).1
}

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
        return (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + 7.5;
    let a = C.iter().enumerate().skip(1).fold(C[0], |a, (i, c)| a + c / (x + i as f64));
    LN_SQRT_2PI + (x + 0.5) * t.ln() - t + a.ln()
}

/// Stirling correction ln Gamma(x) - ((x - 1/2)*ln x - x + ln sqrt(2*pi)) for x >= 10 (asymptotic series;
/// R's `lgammacor`, equal to `stirlerr`).
fn stirling_corr(x: f64) -> f64 {
    const S: [f64; 7] = [1.0 / 12.0, 1.0 / 360.0, 1.0 / 1260.0, 1.0 / 1680.0, 1.0 / 1188.0, 691.0 / 360_360.0, 1.0 / 156.0];
    let x2 = 1.0 / (x * x);
    let mut s = 0.0;
    for c in S.iter().rev() {
        s = c - s * x2;
    }
    s / x
}

/// R's `bd0(x, np) = x*ln(x/np) + np - x`, without cancellation when x ~= np.
fn bd0(x: f64, np: f64) -> f64 {
    if (x - np).abs() < 0.1 * (x + np) {
        let v = (x - np) / (x + np);
        let mut s = (x - np) * v;
        let mut ej = 2.0 * x * v;
        let v2 = v * v;
        for j in 1..1000 {
            ej *= v2;
            let s1 = s + ej / (2 * j + 1) as f64;
            if s1 == s {
                return s1;
            }
            s = s1;
        }
        s
    } else {
        x * (x / np).ln() + np - x
    }
}

/// ln B(a, b) (R's `lbeta`): Stirling corrections instead of differences of large ln Gamma values.
fn lbeta(a: f64, b: f64) -> f64 {
    let (p, q) = if a < b { (a, b) } else { (b, a) };
    if p >= 10.0 {
        let corr = stirling_corr(p) + stirling_corr(q) - stirling_corr(p + q);
        q.ln() * -0.5 + LN_SQRT_2PI + corr + (p - 0.5) * (p / (p + q)).ln() + q * (-p / (p + q)).ln_1p()
    } else if q >= 10.0 {
        let corr = stirling_corr(q) - stirling_corr(p + q);
        ln_gamma(p) + corr + p - p * (p + q).ln() + (q - 0.5) * (-p / (p + q)).ln_1p()
    } else {
        ln_gamma(p) + ln_gamma(q) - ln_gamma(p + q)
    }
}

/// x^a*e^{-x}/Gamma(a), the prefactor of the incomplete gamma function.
fn gamma_prefactor(a: f64, x: f64) -> f64 {
    if a == 0.5 {
        return x.sqrt() * (-x).exp() / PI.sqrt();
    }
    if a >= 10.0 {
        // a*x^a*e^{-x}/Gamma(a+1) with Gamma(a+1) by Stirling (R's dpois_raw).
        return a * (-stirling_corr(a) - bd0(a, x)).exp() / (2.0 * PI * a).sqrt();
    }
    (a * x.ln() - x - ln_gamma(a)).exp()
}

/// Regularized incomplete gamma (P(a, x), Q(a, x)); the one computed directly has full relative
/// accuracy (series for x < a + 1, continued fraction otherwise).
fn gamma_pq(a: f64, x: f64) -> (f64, f64) {
    if x <= 0.0 {
        return (0.0, 1.0);
    }
    let f = gamma_prefactor(a, x);
    if x < a + 1.0 {
        let (mut ap, mut del) = (a, 1.0 / a);
        let mut sum = del;
        for _ in 0..MAX_ITER {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * EPS {
                break;
            }
        }
        let p = sum * f;
        (p, 1.0 - p)
    } else {
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..MAX_ITER {
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
        let q = f * h;
        (1.0 - q, q)
    }
}

/// Continued fraction of the incomplete beta function (Numerical Recipes `betacf`, Lentz).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let clamp = |v: f64| if v.abs() < TINY { TINY } else { v };
    let mut c = 1.0;
    let mut d = 1.0 / clamp(1.0 - qab * x / qap);
    let mut h = d;
    for m in 1..MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 / clamp(1.0 + aa * d);
        c = clamp(1.0 + aa / c);
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 / clamp(1.0 + aa * d);
        c = clamp(1.0 + aa / c);
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta (I_x(a, b), 1 - I_x(a, b)) given x and y = 1 - x (both computed
/// by the caller without cancellation). A tail that is not close to 1 is computed directly, so
/// the smaller tail has full relative accuracy. Tails come from the positive series
/// I_z(p, q) = z^p*(1-z)^q / (p*B(p, q)) * _2F_1(p+q, 1; p+1; z), first on the side of the smaller
/// argument (fast convergence), then, if that tail is close to 1, on the other side; the
/// continued fraction is the fallback when the series would need too many terms (both parameters
/// large, argument near the mean). The continued fraction alone loses digits near its
/// switching point when one parameter is large, which the series avoids.
fn beta_lower_upper(a: f64, b: f64, x: f64, y: f64) -> (f64, f64) {
    if x <= 0.0 {
        return (0.0, 1.0);
    }
    if y <= 0.0 {
        return (1.0, 0.0);
    }
    let front = beta_prefactor(a, b, x, y);
    let lower_first = x <= y;
    let (z, p, q) = if lower_first { (x, a, b) } else { (y, b, a) };
    let first = beta_series(p, q, z, front);
    let from_first = |t: f64| if lower_first { (t, 1.0 - t) } else { (1.0 - t, t) };
    if let Some(t) = first.filter(|t| *t <= 0.9) {
        return from_first(t);
    }
    if let Some(t) = beta_series(q, p, 1.0 - z, front) {
        return if lower_first { (1.0 - t, t) } else { (t, 1.0 - t) };
    }
    // The other tail I_{1-z}(q, p) for a large q (e.g. `pt` and `pf` with a large df).
    if q >= 15.0 {
        if let Some(t) = bgrat_up(q, p, 1.0 - z, z) {
            return if lower_first { (1.0 - t, t) } else { (t, 1.0 - t) };
        }
    }
    // The complement costs at most 4 digits here, less than the continued fraction can.
    if let Some(t) = first.filter(|t| *t <= 0.9999) {
        return from_first(t);
    }
    if x < (a + 1.0) / (a + b + 2.0) {
        let lower = front * beta_cf(a, b, x) / a;
        (lower, 1.0 - lower)
    } else {
        let upper = front * beta_cf(b, a, y) / b;
        (1.0 - upper, upper)
    }
}

/// x^a*y^b / B(a, b). With n = a + b, s the Stirling correction and the large logarithms
/// cancelled exactly instead of in floating point (which costs ~= |b*ln y| ulps):
/// - both parameters >= 10: sqrt(ab/(2*pi n))*exp(s(n) - s(a) - s(b) - bd0(a, nx) - bd0(b, ny))
///   (R's `dbinom_raw`);
/// - only b >= 10: (nx)^a*e^{-nx}/Gamma(a) * sqrt(b/n)*exp(s(n) - s(b) - bd0(b, ny)), and symmetrically.
fn beta_prefactor(a: f64, b: f64, x: f64, y: f64) -> f64 {
    let n = a + b;
    if a >= 10.0 && b >= 10.0 {
        let corr = stirling_corr(n) - stirling_corr(a) - stirling_corr(b);
        return (a * b / (2.0 * PI * n)).sqrt() * (corr - bd0(a, n * x) - bd0(b, n * y)).exp();
    }
    if b >= 10.0 {
        return gamma_prefactor(a, n * x) * (b / n).sqrt() * (stirling_corr(n) - stirling_corr(b) - bd0(b, n * y)).exp();
    }
    if a >= 10.0 {
        return gamma_prefactor(b, n * y) * (a / n).sqrt() * (stirling_corr(n) - stirling_corr(a) - bd0(a, n * x)).exp();
    }
    let ln_x = if x > 0.5 { (-y).ln_1p() } else { x.ln() };
    let ln_y = if y > 0.5 { (-x).ln_1p() } else { y.ln() };
    (a * ln_x + b * ln_y - lbeta(a, b)).exp()
}

/// I_z(p, q) by its positive hypergeometric series given `front` = z^p*(1-z)^q / B(p, q);
/// `None` when the terms would take too long to peak and decay (~= 10^4 terms; rounding then
/// accumulates beyond the target accuracy).
fn beta_series(p: f64, q: f64, z: f64, front: f64) -> Option<f64> {
    let peak = (((p + q) * z - p - 1.0) / (1.0 - z)).max(0.0);
    if peak + 37.0 / -z.ln() > 1e4 {
        return None;
    }
    let (mut term, mut sum) = (1.0, 1.0);
    for n in 0..MAX_ITER {
        let n = n as f64;
        term *= (p + q + n) / (p + 1.0 + n) * z;
        sum += term;
        if term < sum * EPS && (p + q + n) * z < p + 1.0 + n {
            break;
        }
    }
    Some(front * sum / p)
}

/// I_x(a, b) for a >= 15 given x and y = 1 - x: [`bgrat`] at b_0 = b - n in (0, 1], plus the n
/// positive terms of the recurrence I_x(a, c + 1) = I_x(a, c) + x^a*y^c / (c*B(a, c)) (TOMS 708
/// reduces b the same way). `None` for more than 10^4 terms or when `bgrat` fails.
fn bgrat_up(a: f64, b: f64, x: f64, y: f64) -> Option<f64> {
    let n = (b.ceil() - 1.0).max(0.0);
    if n > 1e4 {
        return None;
    }
    let b0 = b - n;
    let mut sum = bgrat(a, b0, x, y)?;
    let ln_x = if y > 0.375 { x.ln() } else { (-y).ln_1p() };
    let mut term = (a * ln_x + b0 * y.ln() - lbeta(a, b0)).exp() / b0;
    for k in 0..n as usize {
        let c = b0 + k as f64;
        sum += term;
        term *= y * (a + c) / (c + 1.0);
    }
    Some(sum)
}

/// I_x(a, b) for a >= 15 and b <= 1 given x and y = 1 - x: the asymptotic expansion of
/// DiDonato & Morris (1992, sec. 9), as `bgrat` of TOMS 708 (R's `pbeta`). `None` when it cannot be
/// evaluated (underflow) or does not converge in 30 terms.
fn bgrat(a: f64, b: f64, x: f64, y: f64) -> Option<f64> {
    let bm1 = b - 1.0;
    let nu = a + 0.5 * bm1;
    let lnx = if y > 0.375 { x.ln() } else { (-y).ln_1p() };
    let z = -nu * lnx;
    if b * z == 0.0 {
        return None;
    }
    // r = e^{-z}*z^b/Gamma(b); u = r*Gamma(a+b)/(Gamma(a)*nu^b), with ln(Gamma(a)/Gamma(a+b)) by Stirling.
    let log_r = -ln_gamma(b) + b * z.ln() - z;
    let ln_ga_gab = -b * a.ln() - (a + b - 0.5) * (b / a).ln_1p() + b + stirling_corr(a) - stirling_corr(a + b);
    let log_u = log_r - (ln_ga_gab + b * nu.ln());
    if log_u == f64::NEG_INFINITY {
        return None;
    }
    let v = 0.25 / (nu * nu);
    let t2 = 0.25 * lnx * lnx;
    let mut j = gamma_pq(b, z).1 / log_r.exp();
    let mut sum = j;
    let (mut t, mut cn, mut n2) = (1.0, 1.0, 0.0);
    let mut c = [0.0f64; 30];
    let mut d = [0.0f64; 30];
    for n in 1..=30usize {
        let bp2n = b + n2;
        j = (bp2n * (bp2n + 1.0) * j + (z + bp2n + 1.0) * t) * v;
        n2 += 2.0;
        t *= t2;
        cn /= n2 * (n2 + 1.0);
        c[n - 1] = cn;
        let mut s = 0.0;
        let mut coef = b - n as f64;
        for i in 1..n {
            s += coef * c[i - 1] * d[n - 1 - i];
            coef += b;
        }
        d[n - 1] = bm1 * cn + s / n as f64;
        let dj = d[n - 1] * j;
        sum += dj;
        if sum <= 0.0 {
            return None;
        }
        if dj.abs() <= EPS * sum {
            return Some((log_u + sum.ln()).exp());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative error (absolute below 1e-300).
    fn rel(a: f64, b: f64) -> f64 {
        if a == b { 0.0 } else { (a - b).abs() / b.abs().max(1e-300) }
    }

    /// Target: 1e-14 relative in the body; far tails (< 1e-10) 1e-12.
    fn tol(p: f64) -> f64 {
        if p.abs() < 1e-10 { 1e-12 } else { 1e-14 }
    }

    // Reference values: R 4.6.1, `sprintf("%.17g", ...)` (kept verbatim, hence 17 digits).
    #[allow(clippy::excessive_precision)]
    #[test]
    fn matches_r() {
        let pn = [
            (-40.0, 0.0),
            (-10.0, 7.6198530241605269e-24),
            (-5.0, 2.8665157187919391e-07),
            (-2.5, 0.0062096653257761349),
            (-1.0, 0.15865525393145705),
            (-0.3, 0.38208857781104733),
            (0.0, 0.5),
            (1e-08, 0.50000000398942279),
            (0.7, 0.75803634777692697),
            (1.96, 0.97500210485177963),
            (3.0, 0.9986501019683699),
            (8.0, 0.99999999999999933),
        ];
        for (x, v) in pn {
            assert!(rel(pnorm(x), v) < 1e-14, "pnorm({x}) = {} vs {v}", pnorm(x));
        }
        let qn = [
            (1e-300, -37.047096299361201),
            (1e-20, -9.2623400897984052),
            (1e-10, -6.3613409024040557),
            (0.001, -3.0902323061678132),
            (0.025, -1.9599639845400538),
            (0.3, -0.52440051270804067),
            (0.5, 0.0),
            (0.7, 0.52440051270804067),
            (0.975, 1.9599639845400536),
            (0.999999, 4.7534243088170891),
        ];
        for (p, v) in qn {
            assert!(rel(qnorm(p), v) < 1e-14, "qnorm({p}) = {} vs {v}", qnorm(p));
        }
        let ptv = [
            (-3.0, 1.0, 0.10241638234956672),
            (-2.1, 2.0, 0.085274934594983726),
            (0.5, 3.0, 0.67427601757592448),
            (-1.3, 7.5, 0.11606338940780428),
            (2.2, 30.0, 0.98217578000158212),
            (-4.0, 100.0, 6.0761822150380825e-05),
            (-6.0, 1234.0, 1.2946919240269945e-09),
            (1.96, 50000.0, 0.97499933194328459),
            (-2.0, 300000.0, 0.022750581873612525),
            (-2.5, 1000000.0, 0.0062097447510816167),
            (-30.0, 5.0, 3.8593243102480234e-07),
            (10.0, 2.0, 0.99507377148833709),
            (-0.01, 10.0, 0.49610898749548571),
        ];
        for (t, df, v) in ptv {
            assert!(rel(pt(t, df), v) < 1e-14, "pt({t}, {df}) = {} vs {v}", pt(t, df));
        }
        let qtv = [
            (0.025, 1.0, -12.706204736174707),
            (0.975, 2.0, 4.3026527297494619),
            (0.05, 3.0, -2.3533634348018255),
            (0.995, 7.5, 3.4214226881240428),
            (1e-10, 30.0, -9.377489780407144),
            (0.3, 100.0, -0.5260762706003459),
            (0.975, 1234.0, 1.961888263458698),
            (0.025, 50000.0, -1.9600114310936798),
            (0.975, 1000000.0, 1.9599663568141066),
            (1e-06, 4.0, -41.577854150450989),
        ];
        for (p, df, v) in qtv {
            assert!(rel(qt(p, df), v) < 1e-14, "qt({p}, {df}) = {} vs {v}", qt(p, df));
        }
        let pfv = [
            // A large denominator df: Stirling prefactor (R 4.6.1).
            (1.3660856508831369, 10.0, 2712435.9985480644, 0.18903119857211093),
            (25.111, 3.0, 2.0, 0.038541359272674533),
            (1.5, 1.0, 10.0, 0.24873843369330934),
            (4.0, 5.0, 100.0, 0.0023809249672239706),
            (0.3, 10.0, 20.0, 0.97273070405819539),
            (100.0, 2.0, 1000000.0, 3.7574583519491371e-44),
            (2.5, 40.0, 5000000.0, 4.7922948812299631e-07),
            (0.001, 3.0, 7.0, 0.9999518633978941),
            (50.0, 1.0, 1.0, 0.0894385219503155),
        ];
        for (f, a, b, v) in pfv {
            assert!(rel(pf_upper(f, a, b), v) < tol(v), "pf_upper({f}, {a}, {b}) = {} vs {v}", pf_upper(f, a, b));
        }
        let pcv = [
            (3.84, 1.0, 0.050043521248705224),
            (0.1, 2.0, 0.95122942450071402),
            (10.0, 3.0, 0.018566135463043233),
            (200.0, 4.0, 3.7572767357810449e-42),
            (1000.0, 1000.0, 0.49405285382923969),
            (30.0, 50.0, 0.98883521972844968),
            (1e-05, 1.0, 0.99747687168319443),
        ];
        for (x, df, v) in pcv {
            assert!(rel(pchisq_upper(x, df), v) < tol(v), "pchisq_upper({x}, {df}) = {} vs {v}", pchisq_upper(x, df));
        }
        assert!(pnorm(f64::NAN).is_nan() && qt(0.5, -1.0).is_nan());
        assert_eq!((pt(f64::INFINITY, 3.0), qnorm(0.0), qt(1.0, 5.0)), (1.0, f64::NEG_INFINITY, f64::INFINITY));
    }
}
