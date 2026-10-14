//! Conley spatial HAC meat with fixest 0.14.2 conventions: the preprocessing of
//! `vcov_conley_internal` (VCOV.R: coordinate shifts and range errors, deduced cutoff, pixel
//! binning, pooling of the scores by location) and the scan of `cpp_vcov_conley`
//! (vcov_related.cpp: uniform kernel, locations sorted by latitude with an early break, its
//! constants 111 km per degree, pi = 3.14159, and the spherical distance 12752 asin(.)).
//!
//! The meat is sum_i S_i S_i' + sum_{i<j, d_ij <= cutoff} (S_i S_j' + S_j S_i') over locations,
//! S the scores summed by location (rows at the same location pair with weight 1 through the
//! pooled sum). Deterministic for any thread count: scores are pooled by
//! [`hac::slot_sums`], the scan runs in fixed chunks of locations combined in order.

use crate::api::FactorValues;
use crate::error::RfError;
use crate::factor;
use crate::hac;
use crate::par::{self, Pool};

/// Conley options in kilometres (miles are converted by the front ends).
pub(crate) struct ConleyOpts {
    /// `None`: [`conley_cutoff_deduce`].
    pub cutoff_km: Option<f64>,
    /// Pixel side; 0 = exact coordinates.
    pub pixel_km: f64,
    /// fixest `distance = "spherical"` (else "triangular").
    pub spherical: bool,
}

/// The pi of `cpp_vcov_conley`'s `degree_to_radian` and `fabs_lon`, and its 2 pi: fixest's
/// rounded literals, kept for parity.
#[allow(clippy::approx_constant)]
const PI_FIXEST: f64 = 3.14159;
#[allow(clippy::approx_constant)]
const TWO_PI_FIXEST: f64 = 6.28318;
/// Locations per task of the scan.
const SCAN_CHUNK: usize = 512;

/// fixest Conley meat (m x m) of the rows' scores s_i a_i, with lat/lon in degrees (row order).
/// Returns the meat and the cutoff (km) actually used. The variance is bread*meat*bread times
/// fixest's n/(n-K) (no G adjustment), applied by the caller.
pub(crate) fn conley_meat(
    lat: &[f64],
    lon: &[f64],
    opts: &ConleyOpts,
    m: usize,
    pool: &Pool,
    row: impl Fn(usize, &mut [f64]) -> f64 + Sync,
) -> Result<(Vec<f64>, f64), RfError> {
    let n = lat.len();
    if lon.len() != n {
        return Err(RfError::Internal("conley_meat: lat and lon lengths differ".into()));
    }
    if !lat.iter().chain(lon).all(|v| v.is_finite()) {
        return Err(RfError::InvalidInput("In vcov.fixest, Conley VCOV cannot be computed: the latitude and longitude must be finite.".into()));
    }
    let lon = shift(lon, 180.0, "longitude", "[-180, 180], [0, 360] or [-360, 0]")?;
    let lat = shift(lat, 90.0, "latitude", "[-90, 90], [0, 180] or [-180, 0]")?;
    let cutoff = match opts.cutoff_km {
        Some(c) => c,
        None => conley_cutoff_deduce(&lat, &lon),
    };
    if !(cutoff >= 0.0 && cutoff.is_finite()) {
        return Err(RfError::InvalidInput(
            "In vcov.fixest, Conley VCOV cannot be computed: the argument 'cutoff' must be a numeric for kilometers, or a number in character form with the 'mi' suffix for miles (ex: 100 means 100km, '100mi' means 100 miles)."
                .into(),
        ));
    }
    let pixel = opts.pixel_km;
    if !(pixel >= 0.0 && pixel.is_finite()) {
        return Err(RfError::InvalidInput(
            "In vcov.fixest, Conley VCOV cannot be computed: the argument 'pixel' must be a numeric scalar greater than, or equal to, 0.".into(),
        ));
    }
    if n == 0 || m == 0 {
        return Ok((vec![0.0; m * m], cutoff));
    }
    let pi = std::f64::consts::PI;
    let (lat_cell, lon_cell) = if pixel > 0.0 {
        // VCOV.R: cells of pixel/111 degrees of latitude and pixel/(111 cos(lat)) of longitude.
        let pixel_lat = pixel / 111.0;
        let la = lat.iter().map(|&x| r_floor_div(x + 90.0, pixel_lat) * pixel_lat - 90.0).collect();
        let lo = lat
            .iter()
            .zip(&lon)
            .map(|(&la, &lo)| {
                let pixel_lon = pixel / (111.0 * (la * pi / 180.0).cos());
                r_floor_div(lo + 180.0, pixel_lon) * pixel_lon - 180.0
            })
            .collect();
        (la, lo)
    } else {
        (lat, lon)
    };
    // Locations: distinct (lat, lon) cells, numbered in (lat, lon) order (`to_integer(...,
    // sorted = TRUE)`), so the scan sees them sorted by latitude.
    let lat_codes = factor::factorize(&FactorValues::Num(&lat_cell), pool)?;
    let lon_codes = factor::factorize(&FactorValues::Num(&lon_cell), pool)?;
    let loc = factor::combine(&[&lat_codes, &lon_codes], pool)?;
    let nl = loc.n as usize;
    let mut loc_lat = par::alloc_vec(nl, 0.0)?;
    let mut loc_lon = par::alloc_vec(nl, 0.0)?;
    for ((&c, &la), &lo) in loc.codes.iter().zip(&lat_cell).zip(&lon_cell) {
        loc_lat[c as usize] = la / 180.0 * pi;
        loc_lon[c as usize] = lo / 180.0 * pi;
    }
    let scores = hac::slot_sums(&loc.codes, nl, m, pool, row)?;
    Ok((scan(&loc_lat, &loc_lon, &scores, m, cutoff, opts.spherical, pool), cutoff))
}

/// VCOV.R's shift of a coordinate: an error outside [-2h, 2h] or with a range wider than 2h,
/// then x - h when max > h, else x + h when min < -h (h = 180 for longitude, 90 for latitude).
fn shift(x: &[f64], h: f64, what: &str, ranges: &str) -> Result<Vec<f64>, RfError> {
    let lo = x.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if x.is_empty() {
        return Ok(Vec::new());
    }
    if lo < -2.0 * h || hi > 2.0 * h || hi - lo > 2.0 * h {
        return Err(RfError::InvalidInput(format!(
            "In vcov.fixest, Conley VCOV cannot be computed: the {what} is outside the {ranges} range (current range is [{}, {}]).",
            fsignif(lo),
            fsignif(hi)
        )));
    }
    Ok(if hi > h {
        x.iter().map(|v| v - h).collect()
    } else if lo < -h {
        x.iter().map(|v| v + h).collect()
    } else {
        x.to_vec()
    })
}

/// `cpp_vcov_conley` on locations sorted by latitude (radians) with pooled scores (nl x m).
fn scan(lat: &[f64], lon: &[f64], scores: &[f64], m: usize, cutoff: f64, spherical: bool, pool: &Pool) -> Vec<f64> {
    let nl = lat.len();
    let cos_lat: Vec<f64> = lat.iter().map(|x| x.cos()).collect();
    let deg2rad = |x: f64| x * PI_FIXEST / 180.0;
    let lat_cut = deg2rad(cutoff / 111.0);
    let lon_factor = deg2rad(cutoff / 111.0);
    let cut_sq = {
        let r = deg2rad(cutoff) / 111.0;
        r * r
    };
    let mut res = par::sum_chunks(pool, nl, SCAN_CHUNK, m * m, |range, acc| {
        let mut cum = vec![0.0; m];
        for i in range {
            let si = &scores[i * m..(i + 1) * m];
            for (c, s) in cum.iter_mut().zip(si) {
                *c = 0.5 * s;
            }
            for i2 in i + 1..nl {
                let dlat = (lat[i2] - lat[i]).abs();
                if dlat > lat_cut {
                    break;
                }
                let d = (lon[i2] - lon[i]).abs();
                let dlon = if d < PI_FIXEST { d } else { TWO_PI_FIXEST - d };
                // Mean latitude for symmetry.
                let cos_mean = ((lat[i] + lat[i2]) / 2.0).cos();
                if dlon > lon_factor / cos_mean {
                    continue;
                }
                let ok = if spherical {
                    let dlo = (lon[i2] - lon[i]) / 2.0;
                    let dla = (lat[i2] - lat[i]) / 2.0;
                    let (sa, so) = (dla.sin(), dlo.sin());
                    let a = sa * sa + cos_lat[i] * cos_lat[i2] * (so * so);
                    12752.0 * a.sqrt().min(1.0).asin() <= cutoff
                } else {
                    let x = cos_mean * dlon;
                    dlat * dlat + x * x <= cut_sq
                };
                if ok {
                    for (c, s) in cum.iter_mut().zip(&scores[i2 * m..(i2 + 1) * m]) {
                        *c += s;
                    }
                }
            }
            for (arow, &s) in acc.chunks_exact_mut(m).zip(si) {
                for (a, c) in arow.iter_mut().zip(&cum) {
                    *a += s * c;
                }
            }
        }
    });
    for k1 in 0..m {
        res[k1 * m + k1] *= 2.0;
        for k2 in k1 + 1..m {
            let t = res[k1 * m + k2];
            res[k1 * m + k2] += res[k2 * m + k1];
            res[k2 * m + k1] += t;
        }
    }
    res
}

/// fixest `cutoff_deduce` (km): over the distinct (lat, lon) points ordered by (lat, lon) and by
/// (lon, lat), the distance of each point to the nearest of the next three (great circle,
/// R = 6371 km; the last three points dropped); cutoff = sum of the two medians, rounded down
/// to 10^max(floor(log10 c) - 1, 1) (the max only when c > 20). Coordinates are taken as given
/// ([`conley_meat`] passes its shifted ones). NaN with fewer than 4 distinct points.
pub(crate) fn conley_cutoff_deduce(lat: &[f64], lon: &[f64]) -> f64 {
    let mut latlon: Vec<(f64, f64)> = lat.iter().zip(lon).map(|(&a, &b)| (a + 0.0, b + 0.0)).collect();
    latlon.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    latlon.dedup();
    let n = latlon.len();
    if n < 4 {
        return f64::NAN;
    }
    let mut lonlat = latlon.clone();
    lonlat.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.total_cmp(&b.0)));
    let nearest = |p: &[(f64, f64)]| -> Vec<f64> {
        (0..n - 3)
            .map(|i| {
                let d = |k: usize| great_circle(p[i].0, p[i].1, p[i + k].0, p[i + k].1);
                d(1).min(d(2)).min(d(3))
            })
            .collect()
    };
    let cutoff = median(nearest(&latlon)) + median(nearest(&lonlat));
    let mut m = cutoff.log10().floor() - 1.0;
    if cutoff > 20.0 {
        m = m.max(1.0);
    }
    let p = r_pow10(m);
    r_floor_div(cutoff, p) * p
}

/// fixest `great_circle_distance` (km, R = 6371), operation by operation.
fn great_circle(lat1: f64, long1: f64, lat2: f64, long2: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let deg2rad = |x: f64| x / 180.0 * pi;
    let (lat1, lat2, long1, long2) = (deg2rad(lat1), deg2rad(lat2), deg2rad(long1), deg2rad(long2));
    let (sa, so) = (((lat2 - lat1) / 2.0).sin(), ((long2 - long1) / 2.0).sin());
    let a = sa * sa + lat1.cos() * lat2.cos() * (so * so);
    6371.0 * (2.0 * a.sqrt().min(1.0).asin())
}

/// R `median` (mean of the two middle values for even n).
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

/// R `10^m` (`R_pow`: exact products for m = 0, 2, 3, 4, libm `pow` otherwise).
fn r_pow10(m: f64) -> f64 {
    match m {
        0.0 => 1.0,
        2.0 => 100.0,
        3.0 => 1000.0,
        4.0 => 10_000.0,
        _ => 10f64.powf(m),
    }
}

/// R `x1 %/% x2` (`myfloor`): floor(q) corrected by the sign of the remainder x1 - floor(q)*x2,
/// which R evaluates in long double; here it is exact by a fused multiply-add.
fn r_floor_div(x1: f64, x2: f64) -> f64 {
    let q = x1 / x2;
    if x2 == 0.0 || q.abs() * f64::EPSILON > 1.0 || !q.is_finite() {
        return q;
    }
    if q.abs() < 1.0 {
        return if q < 0.0 || (x1 < 0.0 && x2 > 0.0) || (x1 > 0.0 && x2 < 0.0) { -1.0 } else { 0.0 };
    }
    let fq = q.floor();
    let tmp = (-fq).mul_add(x2, x1);
    fq + if tmp / x2 < 0.0 { -1.0 } else { 0.0 }
}

/// dreamerr `fsignif(x)` (s = 2, r = 0, thousands commas), used in fixest's range errors.
fn fsignif(x: f64) -> String {
    let v = if x.abs() >= 10.0 { x.round_ties_even() } else { signif2(x) };
    let s = if !v.is_finite() || v.abs() < 1.0 {
        r_as_character(v)
    } else {
        let p = v.abs().log10().ceil();
        let r = if p < 2.0 { (2.0 - p) as usize } else { 0 };
        let xs = r_as_character(v.abs());
        let dec = match xs.find('.') {
            Some(at) if r > 0 => xs[at..].chars().take(1 + r).collect::<String>(),
            _ => String::new(),
        };
        let digits = format!("{:.0}", v.abs().floor());
        let mut grouped = String::new();
        for (i, ch) in digits.chars().enumerate() {
            if i > 0 && (digits.len() - i) % 3 == 0 {
                grouped.push(',');
            }
            grouped.push(ch);
        }
        format!("{}{grouped}{}", if v < 0.0 { "-" } else { "" }, if dec == "." { "" } else { &dec })
    };
    let b = s.as_bytes();
    if b.len() >= 2 && b[0] == b'0' && b.len() < 4 { format!("{s}{}", &"0000"[..4 - b.len()]) } else { s }
}

/// R `signif(x, 2)`.
fn signif2(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let e = 1.0 - x.abs().log10().floor();
    let p = 10f64.powf(e);
    (x * p).round_ties_even() / p
}

/// R `as.character` of a double: up to 15 significant digits, fixed notation unless the
/// scientific one is shorter.
fn r_as_character(x: f64) -> String {
    if !x.is_finite() {
        return if x.is_nan() { "NaN".into() } else if x > 0.0 { "Inf".into() } else { "-Inf".into() };
    }
    let v: f64 = format!("{x:.14e}").parse().unwrap_or(x);
    let fixed = format!("{v}");
    let sci = {
        let s = format!("{v:e}");
        match s.split_once('e') {
            Some((mant, exp)) => {
                let e: i32 = exp.parse().unwrap_or(0);
                format!("{mant}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
            }
            None => s,
        }
    };
    if fixed.len() <= sci.len() { fixed } else { sci }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hac::testdata::{Csv, Refs, num, ols, rel_err_v};
    use crate::rng::SplitMix64;
    use crate::vcov;

    fn pool_run<R: Send>(threads: usize, f: impl FnOnce(&Pool) -> R) -> R {
        par::with_pool(threads, f).unwrap()
    }

    #[test]
    fn fixest_conley_variances() {
        let refs = Refs::read("fixest_ref.csv");
        let geo = Csv::read("geo.csv");
        let n = geo.rows.len();
        let k = 3;
        let (x1, x2) = (geo.col("x1"), geo.col("x2"));
        let x: Vec<f64> = (0..n).flat_map(|i| [1.0, x1[i], x2[i]]).collect();
        let (beta, e, bread) = ols(&x, &geo.col("y"), None, k);
        let row = |i: usize, a: &mut [f64]| {
            a.copy_from_slice(&x[i * k..(i + 1) * k]);
            e[i]
        };
        let mut worst = 0.0f64;
        for (case, la, lo, cutoff, pixel, spherical) in [
            ("conley_100", "lat", "lon", Some(100.0), 0.0, false),
            ("conley_sph_150", "lat", "lon", Some(150.0), 0.0, true),
            ("conley_pix", "lat", "lon", Some(100.0), 30.0, false),
            ("conley_pix_sph", "lat", "lon", Some(200.0), 50.0, true),
            ("conley_def", "lat", "lon", None, 0.0, false),
            ("conley_shift", "lat90", "lon360", Some(100.0), 0.0, false),
        ] {
            let (_, _, b) = refs.mat(case, "b");
            for (p, q) in beta.iter().zip(&b) {
                assert!((p - q).abs() <= 1e-10 * (1.0 + q.abs()));
            }
            let opts = ConleyOpts { cutoff_km: cutoff, pixel_km: pixel, spherical };
            let (meat, used) = pool_run(4, |pool| conley_meat(&geo.col(la), &geo.col(lo), &opts, k, pool, row)).unwrap();
            assert_eq!(used, refs.scalar(case, "cutoff"), "{case}: cutoff");
            let (nf, kf) = (refs.scalar(case, "n"), refs.scalar(case, "K"));
            let v = vcov::sandwich(&bread, &meat, k, nf / (nf - kf));
            let (_, _, want) = refs.mat(case, "V");
            let err = rel_err_v(&v, &want, k);
            assert!(err <= 1e-10, "{case}: V rel err {err:e}\n{v:?}\n{want:?}");
            worst = worst.max(err);
        }
        eprintln!("fixest Conley: max rel diff {worst:e}");
    }

    #[test]
    fn fixest_cutoff_deduce() {
        let refs = Refs::read("fixest_ref.csv");
        let pts = Csv::read("fixest_cutoff_pts.csv");
        let mut cases: Vec<String> = pts.rows.iter().map(|r| r[0].clone()).collect();
        cases.dedup();
        assert_eq!(cases.len(), 5);
        for case in &cases {
            let rows: Vec<&Vec<String>> = pts.rows.iter().filter(|r| &r[0] == case).collect();
            let lat: Vec<f64> = rows.iter().map(|r| num(&r[1])).collect();
            let lon: Vec<f64> = rows.iter().map(|r| num(&r[2])).collect();
            assert_eq!(conley_cutoff_deduce(&lat, &lon), refs.scalar(case, "cutoff"), "{case}");
        }
    }

    /// Direct O(n^2) evaluation of fixest's rule on the rows (no pooling, no early break).
    fn brute(lat: &[f64], lon: &[f64], cutoff: f64, spherical: bool, m: usize, row: &impl Fn(usize, &mut [f64]) -> f64) -> Vec<f64> {
        let n = lat.len();
        let pi = std::f64::consts::PI;
        let x: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut a = vec![0.0; m];
                let s = row(i, &mut a);
                a.iter().map(|v| s * v).collect()
            })
            .collect();
        let rad = |v: f64| v / 180.0 * pi;
        let deg2rad = |v: f64| v * PI_FIXEST / 180.0;
        let mut meat = vec![0.0; m * m];
        for i in 0..n {
            for j in 0..n {
                let (la1, la2, lo1, lo2) = (rad(lat[i]), rad(lat[j]), rad(lon[i]), rad(lon[j]));
                let same = lat[i] == lat[j] && lon[i] == lon[j];
                let dlat = (la2 - la1).abs();
                let d = (lo2 - lo1).abs();
                let dlon = if d < PI_FIXEST { d } else { TWO_PI_FIXEST - d };
                let cm = ((la1 + la2) / 2.0).cos();
                let ok = same
                    || (dlat <= deg2rad(cutoff / 111.0)
                        && dlon <= deg2rad(cutoff / 111.0) / cm
                        && if spherical {
                            let a = ((la2 - la1) / 2.0).sin().powi(2) + la1.cos() * la2.cos() * ((lo2 - lo1) / 2.0).sin().powi(2);
                            12752.0 * a.sqrt().min(1.0).asin() <= cutoff
                        } else {
                            dlat * dlat + (cm * dlon).powi(2) <= (deg2rad(cutoff) / 111.0).powi(2)
                        });
                if ok {
                    for a in 0..m {
                        for b in 0..m {
                            meat[a * m + b] += x[i][a] * x[j][b];
                        }
                    }
                }
            }
        }
        meat
    }

    fn random_points(seed: u64, n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut g = SplitMix64::new(seed);
        // Coarse grid (repeated locations) around the antimeridian and a pole-ward band.
        (0..n)
            .map(|_| {
                let la = (40.0 + g.next_f64() * 30.0) * 10.0;
                let lo = (170.0 + g.next_f64() * 20.0) * 10.0;
                (la.round() / 10.0, if lo > 1800.0 { lo.round() / 10.0 - 360.0 } else { lo.round() / 10.0 })
            })
            .unzip()
    }

    fn rand_row(seed: u64) -> impl Fn(usize, &mut [f64]) -> f64 + Sync {
        move |i, a| {
            let mut g = SplitMix64::stream(seed, i as u64);
            for x in a.iter_mut() {
                *x = g.next_f64() - 0.5;
            }
            g.next_f64() + 0.1
        }
    }

    #[test]
    fn scan_matches_brute_force() {
        let (lat, lon) = random_points(3, 700);
        let row = rand_row(3);
        for spherical in [false, true] {
            for cutoff in [30.0, 250.0, 900.0] {
                let opts = ConleyOpts { cutoff_km: Some(cutoff), pixel_km: 0.0, spherical };
                let (meat, _) = pool_run(4, |pool| conley_meat(&lat, &lon, &opts, 2, pool, &row)).unwrap();
                // The longitude wraps through +-180, so the shift does not apply (range within [-180, 180]).
                let want = brute(&lat, &lon, cutoff, spherical, 2, &row);
                for (a, b) in meat.iter().zip(&want) {
                    assert!((a - b).abs() <= 1e-10 * (1.0 + b.abs()), "cutoff {cutoff} spherical {spherical}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn deterministic_across_threads() {
        let (lat, lon) = random_points(5, 30_000);
        let row = rand_row(5);
        for pixel in [0.0, 20.0] {
            let opts = ConleyOpts { cutoff_km: Some(150.0), pixel_km: pixel, spherical: pixel > 0.0 };
            let run = |t| pool_run(t, |pool| conley_meat(&lat, &lon, &opts, 3, pool, &row)).unwrap().0;
            let base: Vec<u64> = run(1).iter().map(|v| v.to_bits()).collect();
            for t in [4, 16] {
                assert_eq!(base, run(t).iter().map(|v| v.to_bits()).collect::<Vec<u64>>());
            }
        }
    }

    #[test]
    fn range_errors_and_formatting() {
        // dreamerr fsignif values checked in R 4.6.1.
        assert_eq!(fsignif(-400.123), "-400");
        assert_eq!(fsignif(361.5), "362");
        assert_eq!(fsignif(-0.000123456), "-0.00012");
        assert_eq!(fsignif(359.99999), "360");
        assert_eq!(fsignif(-190.25), "-190");
        assert_eq!(fsignif(1234567.8), "1,234,568");
        assert_eq!(fsignif(12.3456), "12");
        assert_eq!(fsignif(-360.5), "-360");
        assert_eq!(fsignif(400.0), "400");
        let opts = ConleyOpts { cutoff_km: Some(100.0), pixel_km: 0.0, spherical: false };
        let err = pool_run(1, |pool| conley_meat(&[1.0, 2.0], &[-400.123, 10.0], &opts, 1, pool, |_, a| {
            a[0] = 1.0;
            1.0
        }))
        .unwrap_err();
        assert_eq!(
            err,
            RfError::InvalidInput(
                "In vcov.fixest, Conley VCOV cannot be computed: the longitude is outside the [-180, 180], [0, 360] or [-360, 0] range (current range is [-400, 10]).".into()
            )
        );
        let err = pool_run(1, |pool| conley_meat(&[-100.0, 95.0], &[1.0, 2.0], &opts, 1, pool, |_, a| {
            a[0] = 1.0;
            1.0
        }))
        .unwrap_err();
        assert!(matches!(&err, RfError::InvalidInput(s) if s.ends_with("latitude is outside the [-90, 90], [0, 180] or [-180, 0] range (current range is [-100, 95]).")));
        assert_eq!(r_floor_div(7.0, 2.0), 3.0);
        assert_eq!(r_floor_div(-7.0, 2.0), -4.0);
        assert_eq!(r_floor_div(0.3, 0.1), 2.0); // 0.3 / 0.1 rounds to 3, the remainder is negative
    }
}
