# Calls into the Rust library (src/rust/src/lib.rs, extendr module `fereg`). Rust errors come
# back as `extendr_error` conditions (extendr feature `result_condition`) and are raised here.

rust_result <- function(x) {
  if (inherits(x, "extendr_error")) stop(x$value, call. = FALSE)
  x
}

fereg_fit <- function(input, spec, nthreads) {
  rust_result(.Call(wrap__fereg_fit, input, spec, as.integer(nthreads)))
}

fereg_vcov <- function(state, kind, ssc, factors, factor_names, cluster, vars, nthreads) {
  rust_result(.Call(wrap__fereg_vcov, state, kind, ssc, factors, factor_names, cluster, vars, as.integer(nthreads)))
}

fereg_core_version <- function() .Call(wrap__fereg_core_version)

fereg_fixef <- function(state, nthreads) rust_result(.Call(wrap__fereg_fixef, state, as.integer(nthreads)))
