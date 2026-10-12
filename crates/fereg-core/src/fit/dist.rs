//! Distribution functions for p-values and confidence intervals (hosts call these so Python and
//! R report identical inference without depending on SciPy). Accuracy target: 1e-14 relative
//! in the body of each distribution; ±inf and NaN inputs follow R's `pnorm`/`qnorm`/`pt`/`qt`.

/// Standard normal CDF, lower tail.
pub fn pnorm(x: f64) -> f64 {
    let _ = x;
    f64::NAN
}

/// Standard normal quantile.
pub fn qnorm(p: f64) -> f64 {
    let _ = p;
    f64::NAN
}

/// Student t CDF, lower tail.
pub fn pt(t: f64, df: f64) -> f64 {
    let _ = (t, df);
    f64::NAN
}

/// Student t quantile.
pub fn qt(p: f64, df: f64) -> f64 {
    let _ = (p, df);
    f64::NAN
}

/// F distribution upper tail P(F > f).
pub fn pf_upper(f: f64, df1: f64, df2: f64) -> f64 {
    let _ = (f, df1, df2);
    f64::NAN
}

/// χ² upper tail P(X > x).
pub fn pchisq_upper(x: f64, df: f64) -> f64 {
    let _ = (x, df);
    f64::NAN
}
