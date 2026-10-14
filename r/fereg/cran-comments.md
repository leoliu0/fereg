## Submission

First submission of fereg 0.4.0.

## Test environments

* Local: Arch Linux x86_64, R 4.6.1, rustc/cargo 1.99.0: `R CMD check --as-cran`.
* Local: same machine with rustc/cargo 1.85.0 (the minimum version declared in
  SystemRequirements): `R CMD check --as-cran`.

## R CMD check results

0 errors | 0 warnings | 2 notes

* checking CRAN incoming feasibility ... NOTE: New submission.
* checking compilation flags used ... NOTE: the flags (`-march=x86-64`,
  `-Wp,-D_FORTIFY_SOURCE=3`, ...) come from the Makeconf of the local R build (Arch Linux
  package); the package sets none of them.
* The installed size is about 20 MB (the compiled Rust library in `libs`); some check platforms
  report it in a NOTE. The library is statically linked Rust code (numerical kernels of faer and
  the fixed-effects solver).

## Rust

* The package follows "Using Rust in CRAN packages": all crates are vendored in
  `src/rust/vendor.tar.xz` (xz) and built offline (`cargo build --offline -j 2`), with
  `CARGO_HOME` inside the build directory, so nothing is downloaded or written to `~/.cargo`.
* `configure` / `configure.win` look for `cargo` and `rustc` on the PATH and in `~/.cargo/bin`,
  check rustc >= 1.85 and print installation instructions otherwise; the rustc and cargo
  versions are reported in the installation log before compilation.
* The authors and licenses of every bundled crate are listed in `inst/AUTHORS`; they are
  credited in Authors@R as contributors and copyright holders.
* Crates no supported platform compiles are shipped as manifest-only stubs (needed to
  resolve `Cargo.lock`); tests, benches and examples of the others are removed.
* Examples and tests use at most 2 threads: the default thread count is 2 when
  `_R_CHECK_LIMIT_CORES_` is set and never exceeds `OMP_THREAD_LIMIT`.
