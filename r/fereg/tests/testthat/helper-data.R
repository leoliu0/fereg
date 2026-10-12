# Simulated panel shared by the tests: three crossed fixed effects (id, year, firm), weights,
# cluster variables, instruments, count and binary outcomes.
options(fereg.nthreads = 2L)

make_panel <- function(n = 1500, seed = 42) {
  set.seed(seed)
  id <- sample.int(150, n, TRUE)
  year <- sample(2001:2010, n, TRUE)
  firm <- sample(letters[1:20], n, TRUE)
  a <- rnorm(150)[id]
  b <- rnorm(10)[year - 2000]
  f <- rnorm(20)[match(firm, letters)]
  x1 <- rnorm(n) + 0.3 * a
  x2 <- rnorm(n) + 0.2 * f
  z1 <- rnorm(n)
  z2 <- rnorm(n)
  u <- rnorm(n)
  e1 <- 0.6 * z1 + 0.4 * z2 + 0.5 * u + a + rnorm(n)
  e2 <- 0.3 * z1 - 0.5 * z2 + 0.3 * u + f + rnorm(n)
  y <- 0.5 * x1 - 0.25 * x2 + 0.7 * e1 + a + b + f + u + rnorm(n)
  data.frame(
    id = id, year = year, firm = firm, region = (id %% 7) + 1L, grp = id %% 15,
    x1 = x1, x2 = x2, x3 = rnorm(n), z1 = z1, z2 = z2, e1 = e1, e2 = e2, y = y,
    w = runif(n, 0.5, 2),
    count = rpois(n, exp(0.3 * x1 - 0.2 * x2 + 0.3 * a)),
    binary = as.integer(0.5 * x1 - 0.3 * x2 + a + rlogis(n) > 0),
    h = sample(1:5, n, TRUE), k = sample(c("u", "v", "w"), n, TRUE),
    expo = runif(n, 0.5, 3),
    stringsAsFactors = FALSE
  )
}

panel <- make_panel()

fixest_ok <- function() requireNamespace("fixest", quietly = TRUE)

# fixest reference fits are run at tight tolerances (fixest's fixef.tol must exceed 2.2e-12), on
# one thread: CRAN's macOS fixest binary has no OpenMP and warns when asked for more.
fx_feols <- function(...) fixest::feols(..., fixef.tol = 3e-12, nthreads = 1, notes = FALSE)
fx_fepois <- function(...) fixest::fepois(..., fixef.tol = 3e-12, glm.tol = 1e-12, nthreads = 1, notes = FALSE)
fx_feglm <- function(...) fixest::feglm(..., fixef.tol = 3e-12, glm.tol = 1e-12, nthreads = 1, notes = FALSE)

fe_feols <- function(...) feols(..., fixef.tol = 1e-12, notes = FALSE)
fe_fepois <- function(...) fepois(..., fixef.tol = 1e-12, glm.tol = 1e-12, notes = FALSE)
fe_feglm <- function(...) feglm(..., fixef.tol = 1e-12, glm.tol = 1e-12, notes = FALSE)

plain <- function(V) {
  V <- unclass(V)
  attributes(V) <- list(dim = dim(V))
  V
}

# Coefficients, names, variance, degrees of freedom and sample size agree with fixest. GLMs use
# the normal distribution (no df.t).
expect_same_fit <- function(ours, theirs, tol = 1e-8) {
  testthat::expect_equal(names(coef(ours)), names(coef(theirs)))
  testthat::expect_equal(unname(coef(ours)), unname(coef(theirs)), tolerance = tol)
  testthat::expect_equal(plain(vcov(ours)), plain(stats::vcov(theirs)), tolerance = tol)
  testthat::expect_equal(nobs(ours), stats::nobs(theirs))
  testthat::expect_equal(ours$nparams, theirs$nparams)
  va <- stats::vcov(theirs, attr = TRUE)
  testthat::expect_equal(ours$vcov_info$df_k, unname(attr(va, "df.K")))
  if (identical(theirs$method_type, "feols")) testthat::expect_equal(ours$vcov_info$df_t, unname(attr(va, "df.t")))
}
