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

fereg_fixef <- function(state, nthreads, min_norm = FALSE) {
  rust_result(.Call(wrap__fereg_fixef, state, as.integer(nthreads), isTRUE(min_norm)))
}

fereg_fixef_references <- function(state, order, nthreads) {
  rust_result(.Call(wrap__fereg_fixef_references, state, as.integer(order), as.integer(nthreads)))
}

fereg_bias_dims <- function(input, spec, bias, nthreads) {
  rust_result(.Call(wrap__fereg_bias_dims, input, spec, bias, as.integer(nthreads)))
}

fereg_bias_corr <- function(input, spec, bias, nthreads) {
  rust_result(.Call(wrap__fereg_bias_corr, input, spec, bias, as.integer(nthreads)))
}

fereg_apes <- function(input, spec, at, ape, nthreads) {
  rust_result(.Call(wrap__fereg_apes, input, spec, at, ape, as.integer(nthreads)))
}
