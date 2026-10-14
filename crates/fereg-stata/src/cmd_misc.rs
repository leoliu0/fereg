//! `cmd=version` and `cmd=selftest` (ABI check, I/O stress test and read benchmarks).

use std::time::{Duration, Instant};

use fereg_core::{RfError, gather, par};

use crate::args::{Args, RowRead};
use crate::{io, spi, with_ctx};

/// Sets local `_rf_version`.
pub fn version(a: &Args) -> Result<(), RfError> {
    a.finish()?;
    spi::local_save("_rf_version", fereg_core::VERSION)
}

/// Varlist v1..vk (numeric), sample = `if`/`in`.
///
/// Keys: `serialio`, `threads`, `verbose`, `write=j` (store the observation number into v_j),
/// `sortby=j` + `rowread=direct|kept` (read rows in the order that sorts the sample by v_j, as a
/// session with that FE1 would: `direct` reads each row from its observation, `kept` reads
/// sequentially and permutes in memory), `spin=<seconds>` (poll Break for that long; r(1) on Break).
///
/// Results: `__rf_nobs`, `__rf_chk` = sum_rows sum_j j*v_j (missing -> 0; summed left to right in
/// observation order, or in row order with `sortby`), `__rf_tread` (read seconds).
pub fn selftest(a: &Args) -> Result<(), RfError> {
    let common = a.common(true)?;
    let write: usize = a.num_or("write", 0)?;
    let sortby: usize = a.num_or("sortby", 0)?;
    let mode = a.row_read()?;
    let spin: f64 = a.num_or("spin", 0.0)?;
    a.finish()?;
    let k = spi::nvars();
    if write > k || sortby > k {
        return Err(RfError::InvalidInput("fereg plugin: write/sortby position outside the varlist".into()));
    }
    let vars: Vec<i32> = (1..=k as i32).collect();
    let obs = io::sample_obs();
    let parallel = !common.serialio;

    with_ctx(&common, |ctx| {
        if spin > 0.0 {
            let end = Instant::now() + Duration::from_secs_f64(spin);
            while Instant::now() < end {
                ctx.check()?;
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        let (rows, tread) = if sortby == 0 {
            let t = Instant::now();
            let rows = io::read_rows(&vars, &obs, ctx.threads, parallel)?;
            (rows, t.elapsed().as_secs_f64())
        } else {
            let key = io::read_col(sortby as i32, &obs, ctx.threads, parallel)?;
            let mut order: Vec<u32> = (0..obs.len() as u32).collect();
            order.sort_by(|&x, &y| key[x as usize].total_cmp(&key[y as usize]));
            let t = Instant::now();
            let rows = match mode {
                RowRead::Direct | RowRead::Auto => {
                    let row_obs: Vec<i32> = order.iter().map(|&i| obs[i as usize]).collect();
                    io::read_rows(&vars, &row_obs, ctx.threads, parallel)?
                }
                RowRead::Kept => {
                    let kept = io::read_rows(&vars, &obs, ctx.threads, parallel)?;
                    par::with_pool(ctx.threads, |pool| gather::permute_rows(&order, &kept, k, pool))??
                }
            };
            (rows, t.elapsed().as_secs_f64())
        };
        let mut chk = 0.0f64;
        for row in rows.chunks_exact(k.max(1)) {
            for (j, v) in row.iter().enumerate() {
                if !v.is_nan() {
                    chk += (j + 1) as f64 * v;
                }
            }
        }
        if ctx.verbose >= 1 {
            let vals = (obs.len() * k) as f64;
            ctx.log(&format!(
                "{{txt}}selftest: read {} obs x {k} vars in {tread:.4} s ({:.3e} values/s, {})",
                obs.len(),
                vals / tread.max(1e-12),
                if parallel { "parallel" } else { "serial" }
            ));
        }
        if write > 0 {
            let v: Vec<f64> = obs.iter().map(|&o| o as f64).collect();
            spi::store_num(write as i32, &obs, &v, 1)?;
        }
        spi::scal_save("__rf_nobs", obs.len() as f64)?;
        spi::scal_save("__rf_chk", chk)?;
        spi::scal_save("__rf_tread", tread)
    })
}
