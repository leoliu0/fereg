//! Stata plugin (SPI 3.0) for the `fereg`, `feppml` and `felogit` commands (plan Phase 4).
//!
//! Only `pginit` and `stata_call` are exported. Every Stata API call goes through `c/glue.c`
//! (wrapped by [`spi`]) on Stata's thread; worker threads live only inside a call (plan D4).
//! The protocol between the ado files and this plugin is documented in `PROTOCOL.md`.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Mutex, Once};
use std::time::Instant;

use fereg_core::{Ctx, RfError};

mod args;
mod cmd_glm;
mod cmd_iv;
mod cmd_misc;
mod cmd_ols;
mod cmd_prepare;
mod est;
mod io;
mod spi;
mod store;

use args::{Args, Common};

/// `SD_PLUGINVER` = SF_MAKELONG(3, 0).
const SD_PLUGINVER: c_int = 3;

/// Default of the `serialio` key for `prepare`/`ols`/`iv`/`glm`: parallel numeric reads are on
/// (decided by `tests/stata/stress_io.do`, plan sec. 4.3).
const SERIALIO_DEFAULT: bool = false;

/// Message of the last panic (filled by the panic hook, consumed by `stata_call`).
static PANIC_MSG: Mutex<Option<String>> = Mutex::new(None);
static INIT: Once = Once::new();

/// Called by Stata when the plugin is loaded.
///
/// # Safety
/// `p` is the `ST_plugin*` table provided by Stata.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pginit(p: *mut c_void) -> c_int {
    unsafe { spi::set_plugin(p) };
    INIT.call_once(|| {
        // No faer call may start rayon's global pool (plan D4).
        faer::set_global_parallelism(faer::Par::Seq);
        panic::set_hook(Box::new(|info| {
            let msg = match info.payload().downcast_ref::<&str>() {
                Some(s) => (*s).to_string(),
                None => info.payload().downcast_ref::<String>().cloned().unwrap_or_else(|| "unknown panic".into()),
            };
            let loc = info.location().map(|l| format!(" at {}:{}", l.file(), l.line())).unwrap_or_default();
            *PANIC_MSG.lock().unwrap_or_else(|e| e.into_inner()) = Some(format!("{msg}{loc}"));
        }));
    });
    SD_PLUGINVER
}

/// Entry point of `plugin call`; `argv[0]` is the command.
///
/// # Safety
/// `argv` holds `argc` NUL-terminated strings (Stata guarantees this).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn stata_call(argc: c_int, argv: *mut *mut c_char) -> c_int {
    let argv: Vec<String> = (0..argc.max(0) as usize)
        .map(|i| unsafe { CStr::from_ptr(*argv.add(i)) }.to_string_lossy().into_owned())
        .collect();
    match panic::catch_unwind(AssertUnwindSafe(|| run(&argv))) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            spi::error(&format!("{e}\n"));
            e.rc()
        }
        Err(payload) => {
            store::clear();
            let msg = PANIC_MSG.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_else(|| {
                payload
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".into())
            });
            spi::error(&format!("fereg: internal error (panic: {msg}); please report it\n"));
            9999
        }
    }
}

fn run(argv: &[String]) -> Result<(), RfError> {
    let args = Args::parse(argv)?;
    match args.cmd.as_str() {
        "version" => cmd_misc::version(&args),
        "selftest" => cmd_misc::selftest(&args),
        "prepare" => cmd_prepare::prepare(&args),
        "ols" => cmd_ols::ols(&args),
        "iv" => cmd_iv::iv(&args),
        "glm" => cmd_glm::glm(&args),
        c => Err(RfError::InvalidInput(format!("fereg plugin: unknown command `{c}`"))),
    }
}

/// Runs `f` with a context whose interrupt polls Stata's Break flag and whose log prints to the
/// Results window, one line per call (the callback appends the newline). Both callbacks run only
/// on Stata's thread (`Ctx` callbacks are not `Sync`).
fn with_ctx<R>(c: &Common, f: impl FnOnce(&Ctx) -> R) -> R {
    let interrupt = spi::break_pressed;
    let log = |s: &str| spi::display(&format!("{s}\n"));
    let ctx = Ctx { threads: c.threads, verbose: c.verbose, interrupt: &interrupt, log: &log };
    f(&ctx)
}

/// `verbose(2)` phase timer.
struct Phases {
    on: bool,
    t: Instant,
}

impl Phases {
    fn new(verbose: u8) -> Self {
        Phases { on: verbose >= 2, t: Instant::now() }
    }

    /// Prints the time since the previous mark.
    fn mark(&mut self, what: &str) {
        if self.on {
            let now = Instant::now();
            spi::display(&format!("{{txt}}fereg: {what}: {:.4} s\n", (now - self.t).as_secs_f64()));
            self.t = now;
        }
    }
}

/// Fails when the varlist does not have `want` variables.
fn check_nvars(want: usize) -> Result<(), RfError> {
    let got = spi::nvars();
    if got != want {
        return Err(RfError::InvalidInput(format!("fereg plugin: expected {want} variables in the varlist, got {got}")));
    }
    Ok(())
}
