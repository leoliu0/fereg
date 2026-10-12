//! Safe wrappers over `c/glue.c`, the only code that calls the Stata plugin interface.
//!
//! Every function here must be called on Stata's thread, except [`read_rows`],
//! [`count_sample`] and [`read_sample`], which the parallel-read path (plan §4.3) calls
//! concurrently on disjoint observation ranges.
//! Indices are 1-based (Stata convention).

use std::ffi::{CString, c_char, c_int, c_void};
use std::mem::MaybeUninit;
use std::time::{Duration, Instant};
use std::sync::Mutex;

use fereg_core::RfError;

unsafe extern "C" {
    fn rf_set_plugin(p: *mut c_void);
    fn rf_nobs() -> c_int;
    fn rf_nvars() -> c_int;
    fn rf_in1() -> c_int;
    fn rf_in2() -> c_int;
    fn rf_ifobs(obs: c_int) -> c_int;
    fn rf_read_rows(vars: *const c_int, nv: c_int, obs: *const c_int, n: c_int, out: *mut f64, nmiss: *mut i64) -> c_int;
    fn rf_count_sample(tv: c_int, o1: c_int, o2: c_int) -> c_int;
    fn rf_read_sample(
        tv: c_int,
        vars: *const c_int,
        nv: c_int,
        o1: c_int,
        o2: c_int,
        cap: c_int,
        obs_out: *mut c_int,
        cols: *const *mut f64,
        strides: *const c_int,
        nmiss: *mut i64,
    ) -> c_int;
    fn rf_store_num(var: c_int, obs: *const c_int, n: c_int, v: *const f64, stride: c_int) -> c_int;
    fn rf_is_str(var: c_int) -> c_int;
    fn rf_is_strl(var: c_int) -> c_int;
    fn rf_read_str(var: c_int, obs: c_int, buf: *mut c_char) -> c_int;
    fn rf_strl_len(var: c_int, obs: c_int) -> c_int;
    fn rf_read_strl(var: c_int, obs: c_int, buf: *mut c_char, cap: c_int) -> c_int;
    fn rf_scal_save(name: *const c_char, v: f64) -> c_int;
    fn rf_mat_rows(name: *const c_char) -> c_int;
    fn rf_mat_cols(name: *const c_char) -> c_int;
    fn rf_mat_store_all(name: *const c_char, rows: c_int, cols: c_int, v: *const f64) -> c_int;
    fn rf_macro_save(name: *const c_char, value: *const c_char) -> c_int;
    fn rf_display(s: *const c_char);
    fn rf_error(s: *const c_char);
    fn rf_poll_break() -> c_int;
}

/// Buffer size for `str#` values (str2045 + terminator).
const STR_BUF: usize = 2046;

/// Largest count passed to one C call (the SPI uses `int`).
const MAX_CALL: usize = c_int::MAX as usize;

fn cstr(s: &str) -> CString {
    // Interior NULs cannot occur in names we build; replace them defensively in messages.
    CString::new(s.replace('\0', " ")).expect("no interior NUL")
}

fn rc_err(rc: c_int, what: &str) -> RfError {
    RfError::Code(rc, format!("fereg: Stata returned r({rc}) while {what}"))
}

/// Stores the plugin table handed to `pginit`.
///
/// # Safety
/// `p` must be the `ST_plugin*` Stata passes to `pginit`.
pub unsafe fn set_plugin(p: *mut c_void) {
    unsafe { rf_set_plugin(p) }
}

pub fn nobs() -> usize {
    unsafe { rf_nobs() }.max(0) as usize
}

/// Number of variables in the `plugin call` varlist.
pub fn nvars() -> usize {
    unsafe { rf_nvars() }.max(0) as usize
}

/// `in` range (1-based, inclusive).
pub fn in_range() -> (i32, i32) {
    unsafe { (rf_in1(), rf_in2()) }
}

/// Whether observation `obs` satisfies the `if`/`in` of the call.
pub fn ifobs(obs: i32) -> bool {
    unsafe { rf_ifobs(obs) != 0 }
}

/// Row-major read `out[k * vars.len() + j] = vars[j][obs[k]]` (NaN for missing) into memory
/// that may be uninitialized; every element is written on success. Returns the number of
/// missing values read.
pub fn read_rows(vars: &[i32], obs: &[i32], out: &mut [MaybeUninit<f64>]) -> Result<u64, RfError> {
    assert!(obs.len() <= MAX_CALL);
    assert_eq!(out.len(), obs.len() * vars.len());
    if out.is_empty() {
        return Ok(0);
    }
    let mut nmiss = 0i64;
    let rc = unsafe {
        rf_read_rows(vars.as_ptr(), vars.len() as c_int, obs.as_ptr(), obs.len() as c_int, out.as_mut_ptr().cast(), &mut nmiss)
    };
    if rc != 0 { Err(rc_err(rc, "reading variables")) } else { Ok(nmiss as u64) }
}

/// Number of observations in `o1..=o2` whose `touse` value is not 0 (`touse` 0: all of them).
pub fn count_sample(touse: i32, o1: i32, o2: i32) -> Result<usize, RfError> {
    let c = unsafe { rf_count_sample(touse, o1, o2) };
    if c < 0 { Err(rc_err(-c, "reading the estimation sample")) } else { Ok(c as usize) }
}

/// Sample scan of observations `o1..=o2`: the c-th observation whose `touse` value is not 0
/// (`touse` 0: every observation; touse is not read)
/// goes to `obs_out[c]`, its value of the leading `cols.len()` variables to `cols[j][c]` and
/// that of the remaining `w` variables to row c of the row-major `block` (`block[c·w + j]`);
/// NaN for missing. Returns the number of such observations and the number of missing values
/// among them; fails when there are more than `obs_out.len()`.
pub fn read_sample(
    touse: i32,
    vars: &[i32],
    o1: i32,
    o2: i32,
    obs_out: &mut [MaybeUninit<i32>],
    cols: &mut [&mut [MaybeUninit<f64>]],
    block: &mut [MaybeUninit<f64>],
) -> Result<(usize, u64), RfError> {
    let cap = obs_out.len();
    let w = vars.len() - cols.len();
    assert!(cap <= MAX_CALL && cols.iter().all(|c| c.len() == cap) && block.len() == cap * w);
    // Column j of the block starts at its j-th value (the C side writes only rows c < cap).
    let base: *mut f64 = block.as_mut_ptr().cast();
    let ptrs: Vec<*mut f64> = cols.iter_mut().map(|c| c.as_mut_ptr().cast()).chain((0..w).map(|j| base.wrapping_add(j))).collect();
    let strides: Vec<c_int> = cols.iter().map(|_| 1).chain((0..w).map(|_| w as c_int)).collect();
    let mut miss: i64 = 0;
    let c = unsafe {
        rf_read_sample(
            touse,
            vars.as_ptr(),
            vars.len() as c_int,
            o1,
            o2,
            cap as c_int,
            obs_out.as_mut_ptr().cast(),
            ptrs.as_ptr(),
            strides.as_ptr(),
            &mut miss,
        )
    };
    match c {
        c if c >= 0 => Ok((c as usize, miss as u64)),
        -9999 => Err(RfError::Internal("the estimation sample changed while it was read".into())),
        c => Err(rc_err(-c, "reading the estimation sample")),
    }
}

/// `var[obs[k]] = v[k * stride]` (NaN stored as missing).
pub fn store_num(var: i32, obs: &[i32], v: &[f64], stride: usize) -> Result<(), RfError> {
    assert!(stride >= 1 && obs.len() <= MAX_CALL);
    if obs.is_empty() {
        return Ok(());
    }
    assert!(v.len() > (obs.len() - 1) * stride);
    let rc = unsafe { rf_store_num(var, obs.as_ptr(), obs.len() as c_int, v.as_ptr(), stride as c_int) };
    if rc != 0 { Err(rc_err(rc, "storing a variable")) } else { Ok(()) }
}

pub fn is_str(var: i32) -> bool {
    unsafe { rf_is_str(var) != 0 }
}

pub fn is_strl(var: i32) -> bool {
    unsafe { rf_is_strl(var) != 0 }
}

/// Appends the string value of `var[obs]` (str# or strL) to `out`.
pub fn read_str_into(var: i32, obs: i32, strl: bool, out: &mut Vec<u8>, buf: &mut Vec<u8>) -> Result<(), RfError> {
    if strl {
        let len = unsafe { rf_strl_len(var, obs) };
        if len < 0 {
            return Err(RfError::InvalidInput("fereg: binary strL values cannot be used as factors".into()));
        }
        let len = len as usize;
        buf.clear();
        buf.resize(len + 1, 0);
        let rc = unsafe { rf_read_strl(var, obs, buf.as_mut_ptr().cast(), (len + 1) as c_int) };
        if rc < 0 {
            return Err(rc_err(-rc, "reading a strL variable"));
        }
        let end = buf[..len].iter().position(|&b| b == 0).unwrap_or(len);
        out.extend_from_slice(&buf[..end]);
    } else {
        buf.clear();
        buf.resize(STR_BUF, 0);
        let len = unsafe { rf_read_str(var, obs, buf.as_mut_ptr().cast()) };
        if len < 0 {
            return Err(rc_err(-len, "reading a string variable"));
        }
        out.extend_from_slice(&buf[..len as usize]);
    }
    Ok(())
}

/// Saves Stata scalar `name` (NaN → missing).
pub fn scal_save(name: &str, v: f64) -> Result<(), RfError> {
    let c = cstr(name);
    let rc = unsafe { rf_scal_save(c.as_ptr(), v) };
    if rc != 0 { Err(rc_err(rc, &format!("saving scalar {name}"))) } else { Ok(()) }
}

/// Dimensions of an existing Stata matrix (non-positive when it does not exist).
pub fn mat_dims(name: &str) -> (i32, i32) {
    let c = cstr(name);
    unsafe { (rf_mat_rows(c.as_ptr()), rf_mat_cols(c.as_ptr())) }
}

/// Fills an existing `rows × cols` Stata matrix from a row-major block (NaN → missing).
pub fn mat_store(name: &str, rows: usize, cols: usize, data: &[f64]) -> Result<(), RfError> {
    assert_eq!(data.len(), rows * cols);
    let c = cstr(name);
    let rc = unsafe { rf_mat_store_all(c.as_ptr(), rows as c_int, cols as c_int, data.as_ptr()) };
    if rc != 0 { Err(rc_err(rc, &format!("storing matrix {name}"))) } else { Ok(()) }
}

/// Sets local macro `name` in the program that ran `plugin call`.
pub fn local_save(name: &str, value: &str) -> Result<(), RfError> {
    let n = cstr(&format!("_{name}"));
    let v = cstr(value);
    let rc = unsafe { rf_macro_save(n.as_ptr(), v.as_ptr()) };
    if rc != 0 { Err(rc_err(rc, &format!("saving local {name}"))) } else { Ok(()) }
}

/// Prints SMCL text in the Results window (no newline added) and flushes.
pub fn display(s: &str) {
    // The SPI's output buffer is finite; print long text in pieces cut at char boundaries.
    const PIECE: usize = 2000;
    let mut rest = s;
    while !rest.is_empty() {
        let mut cut = rest.len().min(PIECE);
        while !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        let c = cstr(&rest[..cut]);
        unsafe { rf_display(c.as_ptr()) };
        rest = &rest[cut..];
    }
}

/// Prints an error message (red) in the Results window.
pub fn error(s: &str) {
    let c = cstr(s);
    unsafe { rf_error(c.as_ptr()) };
}

static LAST_POLL: Mutex<Option<Instant>> = Mutex::new(None);

/// Polls Stata for Break at most every 50 ms (the poll lets Stata process GUI events).
pub fn break_pressed() -> bool {
    let now = Instant::now();
    {
        let mut last = LAST_POLL.lock().unwrap_or_else(|e| e.into_inner());
        if last.is_some_and(|t| now.duration_since(t) < Duration::from_millis(50)) {
            return false;
        }
        *last = Some(now);
    }
    unsafe { rf_poll_break() != 0 }
}
