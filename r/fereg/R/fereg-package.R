#' fereg: fast fixed-effects estimation with fixest's interface
#'
#' OLS, IV/2SLS, Poisson and logit estimation with any number of fixed effects, interactions and
#' varying slopes, weights and iid / heteroskedasticity-robust / multiway-clustered variances.
#' The interface and the numbers are those of fixest ([feols()], [fepois()], [feglm()],
#' [etable()], [fixef()], ...); estimation runs in a parallel Rust core that eliminates the
#' largest fixed effect exactly and solves the rest by preconditioned conjugate gradient.
#'
#' Options: `fereg.nthreads` (threads, `0` = all cores) and `fereg.notes` (print notes on removed
#' observations).
#'
#' @keywords internal
#' @useDynLib fereg, .registration = TRUE
#' @importFrom stats AIC BIC coef confint deviance fitted logLik nobs predict residuals sigma vcov
"_PACKAGE"

#' Simulated panel for examples
#'
#' A small simulated panel: 600 observations of 60 units over 10 years.
#'
#' @format A data frame with columns `id`, `year`, `x1`, `x2`, `z` (instrument), `xe`
#'   (endogenous regressor), `y`, `count` (Poisson outcome) and `binary` (0/1 outcome).
#' @docType data
#' @name trade_toy
#' @usage data(trade_toy)
NULL
