# Incidental-parameter bias correction of fixed-effects logit estimates and average partial
# effects, with alpaca's interface (biasCorr(), getAPEs()). The arguments are checked here; the
# fereg core estimates the model again on its data, corrects the coefficients (analytically or by
# jackknife), solves the fixed effects again at the corrected coefficients and computes the
# average partial effects and their variance.

#' Bias correction of fixed-effects logit estimates
#'
#' Reduces the incidental parameter bias of a fixed-effects logit estimation with [feglm()], as
#' `alpaca::biasCorr()` (alpaca 0.3.5): the analytical correction of Fernández-Val and Weidner
#' (2016) for one- and two-way models, or of Hinz, Stammann and Wanner (2020) for networks with
#' two or three fixed effects, with a bandwidth `L` for weakly exogenous regressors (Hahn and
#' Kuersteiner, 2011). fereg adds the split-panel and delete-one jackknife corrections of Stata's
#' `logitfe` (Cruz-Gonzalez, Fernández-Val and Weidner, 2017) and the choice of the corrected
#' bias.
#'
#' The corrected estimation is a `fereg` object at the corrected coefficients, with the fixed
#' effects estimated again by maximum likelihood holding the corrected linear index fixed:
#' `coef()`, `vcov()`, `se()`, `summary()` (with any `vcov` of [feglm()] except CRV3),
#' `confint()`, [etable()], [fixef()], `predict()`, `fitted()` and `resid()` all refer to the
#' corrected point, and `print()` / `summary()` name the correction. [getAPEs()] gives the
#' corrected average partial effects.
#'
#' **Analytical correction.** `bias = "both"` corrects the bias of all fixed effects (alpaca);
#' `"individual"` only that of the first fixed effect (with the spectral term when `L > 0`);
#' `"time"` only that of the second (two-way models). In a classic panel the first fixed effect is
#' the individual and the second the time effect, in formula order; in a network the fixed
#' effects are exporter-time, importer-time and pair, in that order.
#'
#' The bandwidth `L` counts **rows**, as in alpaca: the lag `l` of an observation is the
#' observation `l` rows earlier in its group (first fixed effect in a classic panel, pair in a
#' network), with the rows in time order, and gaps in unbalanced panels are ignored. The time
#' order is that of the levels of the second fixed effect within each individual (two-way), the
#' order of the data rows (one-way), and the order of the (exporter-time, importer-time) levels
#' within each pair (network): numbers ascending, factors in level order, character strings in
#' C-locale order (alpaca sorts the data by the fixed-effect variables). A network model with two
#' fixed effects has no pair effect: `L` is then ignored, with a warning (alpaca ignores it
#' silently).
#'
#' **Jackknife.** The full-sample estimate `b` is combined with estimates on subpanels, estimated
#' again with the sample rules of [feglm()] (each with its own removal of fixed-effect groups with
#' a constant outcome). Individuals `1..N` are the levels of the first fixed effect and periods
#' `1..T` those of the second (or of the time variable of `panel.id` in a one-way model), counted
#' before any removal. Halves of `m` units: the first has the units `<= ceiling(m/2)`, the second
#' those `> m/2` (`m` even) or `>= ceiling(m/2)` (`m` odd: the middle unit is in both). `IH`:
#' halves of the individuals; `TH`: halves of the periods; `PH`: halves of the observations of
#' each individual, in time order; `Q`: the four quadrants `IH x TH`; `b(-i)`, `b(-t)`: estimates
#' without individual `i` / period `t`; `mean()`: average over the subpanels.
#' \tabular{llll}{
#'   variant \tab `bias = "both"` \tab `"individual"` \tab `"time"` \cr
#'   `ss2` \tab `3b - mean(IH) - mean(PH)` \tab `2b - mean(PH)` \tab `2b - mean(IH)` \cr
#'   `ss1` \tab `2b - mean(Q)` \tab `2b - mean(TH)` \tab `2b - mean(IH)` \cr
#'   `js` \tab `(N+1)b - (N-1)mean(b(-i)) - mean(TH)` \tab `2b - mean(TH)` \tab `Nb - (N-1)mean(b(-i))` \cr
#'   `sj` \tab `(T+1)b - (T-1)mean(b(-t)) - mean(IH)` \tab `Tb - (T-1)mean(b(-t))` \tab `2b - mean(IH)` \cr
#'   `jj` \tab `(N+T-1)b - (N-1)mean(b(-i)) - (T-1)mean(b(-t))` \tab `Tb - (T-1)mean(b(-t))` \tab `Nb - (N-1)mean(b(-i))` \cr
#'   `double` \tab `Jb - (J-1)mean(b(-c))` \tab `Tb - (T-1)mean(b(-t))` \tab `Nb - (N-1)mean(b(-i))` \cr
#' }
#' `double` is for dyadic data whose two fixed effects share their identifiers (exporter and
#' importer): `b(-c)` drops the rows whose first or second fixed effect is `c`, for the `J`
#' identifiers present in both. `multiple = R` (`ss1`, `ss2`) averages the correction over `R`
#' random partitions: in each, the individuals and/or periods (`split`) are randomly permuted
#' (with [sample.int()], so `set.seed()` makes it reproducible; per partition the individuals are
#' drawn before the periods) and the halves are taken in the permuted order. With `multiple`,
#' `ss2` correcting the individual-effect bias alone (`bias = "individual"` or a one-way model)
#' uses `2b - mean(TH)` per partition (logitfe). A jackknife on a model with one fixed effect
#' (individual effects) splits each individual's observations in time
#' and needs the time variable: `panel.id = ~id + time` here or in [feglm()]. The jackknife
#' average partial effects ([getAPEs()]) combine those of the subpanels in the same way.
#'
#' **Differences from the references.**
#' * alpaca's default `feglm()` does not return the maximum likelihood estimate: it rejects its
#'   last IRLS step, so its coefficients differ by up to about 1e-5 (relative) from those of
#'   fereg, fixest and `glm()`, and its corrections inherit the difference. With
#'   `feglmControl(dev.tol = 1e-20, center.tol = 1e-14, iter.max = 100)` alpaca and fereg (with
#'   `glm.tol = 1e-12, fixef.tol = 1e-12`) agree to about 1e-10.
#' * alpaca's re-estimation of the fixed effects after the correction stops early (fixed-effect
#'   scores up to 1e-7 even with the tight control above, 1e-4 with its defaults); fereg solves
#'   them exactly. The corrected coefficients agree; variances and average partial effects of the
#'   corrected estimation differ from alpaca's by about 1e-6 (tight control).
#' * Variances of the corrected estimation are evaluated at the corrected point (as alpaca), with
#'   fixest's variance types and small-sample corrections: alpaca's `vcov(type = "hessian")` is
#'   `vcov = "iid", ssc = ssc(K.adj = FALSE)`, `"sandwich"` is
#'   `vcov = "hetero", ssc = ssc(K.adj = FALSE, G.adj = FALSE)` and `"clustered"` is
#'   `vcov = ~cl, ssc = ssc(K.adj = FALSE, G.adj = FALSE)`. CRV3 is an error.
#' * The log-likelihood, deviance and fit statistics are those of the corrected point (alpaca
#'   keeps the uncorrected deviance, which is in `deviance.uncorr`).
#' * alpaca has no jackknife. The jackknife follows logitfe's definitions with two fixes: `js`
#'   and `sj` combine the half-panels of the other dimension (logitfe uses the same dimension
#'   twice), and `ss2` with a single bias works (logitfe stops). A subpanel estimation that loses
#'   a regressor or does not converge is an error naming the subpanel (logitfe sets the
#'   coefficient to zero).
#' * `biasCorr()` cannot be applied to a corrected estimation (alpaca corrects again).
#'
#' @param object A logit estimation of [feglm()] with fixed effects (intercept-only terms: one or
#'   two in a classic panel, two or three in a network), estimated with `lean = FALSE`. Its data
#'   must still be available (it is found again from the call, as for clustered variances after
#'   estimation): the correction estimates the model again.
#' @param L Bandwidth of the spectral term for weakly exogenous regressors (e.g. a lagged
#'   outcome); `0` (default) for strictly exogenous regressors. Fernández-Val and Weidner suggest
#'   1 to 4. Analytical correction only.
#' @param panel.structure `"classic"` (individuals observed over time; default) or `"network"`
#'   (exporter-time, importer-time and pair fixed effects).
#' @param method `"analytical"` (default) or `"jackknife"` (classic panels only).
#' @param jackknife Jackknife variant (see Details): `"ss2"` (default), `"ss1"`, `"js"`, `"sj"`,
#'   `"jj"` or `"double"`.
#' @param bias Bias to correct: `"both"` (default; all fixed effects), `"individual"` (first fixed
#'   effect) or `"time"` (second fixed effect; two-way classic models).
#' @param multiple Number of random partitions for `ss1` / `ss2`; `0` (default) splits in the
#'   data order.
#' @param split With `multiple`: permute the `"individuals"`, the periods (`"time"`) or both
#'   (`"both"`, default).
#' @param panel.id The time variable of a one-way model, needed by the jackknife: a formula
#'   `~id + time` or a string `"id,time"` whose first variable is the fixed effect. Defaults to
#'   the `panel.id` of the estimation.
#' @return A `fereg` object of class `c("fereg_biascorr", "fereg")` at the corrected coefficients,
#'   with in addition `coefficients.uncorr` (the uncorrected coefficients), `bias.term` (the
#'   uncorrected minus the corrected coefficients), `panel.structure`, `bandwidth` (`L`), `method`,
#'   `jackknife` (the variant, `NULL` for the analytical correction), `bias`, `multiple`, `split`
#'   (with `multiple`), `nobs.full` (observations before the removal of fixed-effect groups with
#'   a constant outcome) and `deviance.uncorr` (the deviance of the uncorrected estimation).
#' @references
#' Cruz-Gonzalez, M., I. Fernández-Val and M. Weidner (2017). "Bias corrections for probit and
#' logit models with two-way fixed effects". The Stata Journal, 17(3), 517-545.
#'
#' Fernández-Val, I. and M. Weidner (2016). "Individual and time effects in nonlinear panel
#' models with large N, T". Journal of Econometrics, 192(1), 291-312.
#'
#' Hahn, J. and G. Kuersteiner (2011). "Bias reduction for dynamic nonlinear panel models with
#' fixed effects". Econometric Theory, 27(6), 1152-1191.
#'
#' Hinz, J., A. Stammann and J. Wanner (2020). "State dependence and unobserved heterogeneity in
#' the extensive margin of trade". ArXiv e-prints.
#' @seealso [getAPEs()], [feglm()]
#' @examples
#' data(trade_toy, package = "fereg")
#' m <- feglm(binary ~ x1 + x2 | id + year, trade_toy, nthreads = 2)
#' bc <- biasCorr(m)
#' bc
#' summary(bc, vcov = ~id)
#' cbind(uncorrected = bc$coefficients.uncorr, corrected = coef(bc))
#' jk <- biasCorr(m, method = "jackknife", jackknife = "ss2")
#' coef(jk)
#' @export
biasCorr <- function(object = NULL, L = 0L, panel.structure = c("classic", "network"),
                     method = c("analytical", "jackknife"),
                     jackknife = c("ss2", "ss1", "js", "sj", "jj", "double"),
                     bias = c("both", "individual", "time"), multiple = 0L,
                     split = c("both", "individuals", "time"), panel.id = NULL) {
  bc_check_object(object, "biasCorr")
  if (inherits(object, "fereg_biascorr")) {
    stop("fereg: `object` is already bias-corrected; apply biasCorr() to the feglm() estimation", call. = FALSE)
  }
  panel.structure <- match.arg(panel.structure)
  method <- match.arg(method)
  jackknife_given <- !missing(jackknife)
  jackknife <- match.arg(jackknife)
  bias <- match.arg(bias)
  split_given <- !missing(split)
  split <- match.arg(split)
  L <- bc_count(L, "L")
  multiple <- bc_count(multiple, "multiple")
  k <- bc_check_fe(object, panel.structure, "biasCorr")
  if (identical(object$vcov_info$type, "crv3")) {
    stop("fereg: CRV3 is not available after bias correction; estimate the model with another `vcov`", call. = FALSE)
  }
  if (method == "analytical") {
    if (jackknife_given) stop("fereg: `jackknife` needs method = \"jackknife\"", call. = FALSE)
    if (multiple > 0) stop("fereg: `multiple` needs method = \"jackknife\"", call. = FALSE)
  } else {
    if (panel.structure == "network") stop("fereg: the jackknife corrections need panel.structure = \"classic\"", call. = FALSE)
    if (L > 0) stop("fereg: the bandwidth `L` is only used by the analytical correction (method = \"analytical\")", call. = FALSE)
    if (multiple > 0 && !jackknife %in% c("ss1", "ss2")) stop("fereg: `multiple` needs jackknife = \"ss1\" or \"ss2\"", call. = FALSE)
  }
  if (split_given && multiple == 0) stop("fereg: `split` is only used with `multiple`", call. = FALSE)
  if (panel.structure == "network" && bias != "both") {
    stop("fereg: panel.structure = \"network\" corrects the bias of all fixed effects; use bias = \"both\"", call. = FALSE)
  }
  if (bias == "time" && k == 1) stop("fereg: bias = \"time\" needs a two-way model (individual and time fixed effects)", call. = FALSE)
  if (multiple > 0) {
    if ((bias == "individual" || k == 1) && split == "individuals") {
      stop("fereg: split = \"individuals\" has no effect when only the individual-effect bias is corrected ",
        "(the panel is split in time); use split = \"time\" or \"both\"", call. = FALSE)
    }
    if (bias == "time" && split == "time") {
      stop("fereg: split = \"time\" has no effect when only the time-effect bias is corrected ",
        "(the panel is split across individuals); use split = \"individuals\" or \"both\"", call. = FALSE)
    }
  }
  needs_time <- method == "jackknife" && k == 1
  if (!is.null(panel.id) && !needs_time) {
    stop("fereg: `panel.id` is only used by the jackknife of a model with one fixed effect (to split the observations of each individual in time)", call. = FALSE)
  }
  common <- method == "jackknife" && jackknife == "double" && k == 2 && bias == "both"

  data <- fetch_data(object, "biasCorr() estimates the model again on it")
  time_var <- if (needs_time) bc_time_var(object, panel.id, data)
  s <- bias_setup(object, data, time_var, common)
  opts <- list(
    method = method, structure = panel.structure, which = bias, bandwidth = L, time = s$time,
    variant = if (method == "jackknife") jackknife, partitions = list()
  )
  nthreads <- getOption("fereg.nthreads", default_threads())
  if (multiple > 0) {
    dims <- fereg_bias_dims(s$input, s$spec, opts, nthreads)
    opts$partitions <- lapply(seq_len(multiple), function(r) {
      list(
        individual = if (split != "time") sample.int(dims$n_individuals),
        period = if (split != "individuals") sample.int(dims$n_periods)
      )
    })
  }
  out <- fereg_bias_corr(s$input, s$spec, opts, nthreads)
  res <- build_result(out$fit, s$info)
  rep <- out$report
  perm <- res$coef_perm
  nm <- names(res$coefficients)
  res$coefficients.uncorr <- stats::setNames(rep$coef_uncorr[perm], nm)
  res$bias.term <- stats::setNames(rep$bias_term[perm], nm)
  res$panel.structure <- panel.structure
  res$bandwidth <- L
  res$method <- method
  res$jackknife <- if (method == "jackknife") jackknife
  res$bias <- bias
  res$multiple <- multiple
  res$split <- if (multiple > 0) split
  res$nobs.full <- as.integer(rep$n_full)
  res$deviance.uncorr <- rep$deviance_uncorr
  # What getAPEs() needs: the jackknife-corrected average partial effects and the correction's
  # options (with the setup that gives the same core input).
  res$ape.jackknife <- if (!is.null(rep$ape_jackknife)) stats::setNames(rep$ape_jackknife[perm], nm)
  res$bias.options <- opts
  res$bias.setup <- list(time_var = time_var, common = common)
  class(res) <- c("fereg_biascorr", class(res))
  for (nt in rep$notes) warning(nt, call. = FALSE)
  res
}

#' Average partial effects of fixed-effects logit estimates
#'
#' Average partial effects (APEs) of the regressors of a fixed-effects logit estimation and their
#' variance, as `alpaca::getAPEs()` (alpaca 0.3.5): uncorrected for a [feglm()] estimation,
#' corrected for a [biasCorr()] one. The partial effect of a binary regressor (all values 0 or 1
#' in the estimation sample) is the change of the probability from 0 to 1, that of another
#' regressor the derivative of the probability. The averages run over all observations before the
#' removal of fixed-effect groups with a constant outcome, those removed counting as zero effects
#' (alpaca, logitfe), and are unweighted even with weights (alpaca).
#'
#' For an analytical correction, the APEs are evaluated at the corrected point and their bias is
#' removed (Fernández-Val and Weidner, 2016); `panel.structure` and the bandwidth are those of the
#' correction and `weak.exo` is `L > 0` (alpaca). For a jackknife correction, the APEs are the
#' jackknife combination of the APEs of the full sample and of the subpanels (each averaged over
#' the subpanel's own observations), with the variance at the corrected point.
#'
#' The variance is the delta-method variance (Fernández-Val and Weidner, 2016, as alpaca) plus,
#' when `n.pop` is given, a finite-population term for the sampling of the effects multiplied by
#' \eqn{(n.pop - N^*)/(n.pop - 1)}, \eqn{N^*} being the number of observations before the removal
#' of constant-outcome groups. When `n.pop` is smaller than \eqn{N^*} the factor is zero, with a
#' warning (alpaca). `sampling.fe` and `weak.exo` only change this term.
#'
#' **Difference from alpaca.** With `n.pop` and fixed-effect groups removed because of a constant
#' outcome, the sampling-of-effects term sums over all observations, with zero partial effects for
#' the removed ones, consistently with the point estimate; alpaca (and Stata's logitfe) sum over
#' the retained observations only, which overstates the variance. The numbers are identical when
#' no group is removed. After a correction, the APEs and their variance differ from alpaca's by
#' about 1e-6 because alpaca's fixed effects at the corrected coefficients are imprecise (see
#' [biasCorr()]). Giving `panel.structure` or `weak.exo` values that contradict the correction of
#' a [biasCorr()] object is an error (alpaca ignores them).
#'
#' @param object A logit estimation of [feglm()] (see [biasCorr()] for the requirements) or a
#'   [biasCorr()] result.
#' @param n.pop Size of the population, for the finite-population term of the variance (truncated
#'   to an integer, as alpaca); `NULL` (default) gives the delta-method variance only.
#' @param panel.structure `"classic"` (default) or `"network"`, as in [biasCorr()]. Taken from the
#'   correction for a [biasCorr()] object.
#' @param sampling.fe `"independence"` (default: the fixed effects are independent sequences) or
#'   `"unrestricted"` (no sampling term).
#' @param weak.exo `TRUE` if some regressors are weakly exogenous (adds their covariance with the
#'   sampling term). Set to `L > 0` for a [biasCorr()] object.
#' @return An object of class `"APEs"`: a list with `delta` (the APEs), `vcov` (their variance),
#'   `panel.structure`, `sampling.fe`, `weak.exo`, and for a [biasCorr()] object `bandwidth` and,
#'   after an analytical correction, `bias.term` (the APE bias removed from the APEs at the
#'   corrected point). Methods: `print()`, `summary()`, `coef()`, `vcov()`.
#' @references
#' Cruz-Gonzalez, M., I. Fernández-Val and M. Weidner (2017). "Bias corrections for probit and
#' logit models with two-way fixed effects". The Stata Journal, 17(3), 517-545.
#'
#' Fernández-Val, I. and M. Weidner (2016). "Individual and time effects in nonlinear panel
#' models with large N, T". Journal of Econometrics, 192(1), 291-312.
#' @seealso [biasCorr()], [feglm()]
#' @examples
#' data(trade_toy, package = "fereg")
#' m <- feglm(binary ~ x1 + x2 | id + year, trade_toy, nthreads = 2)
#' summary(getAPEs(m))
#' ape <- getAPEs(biasCorr(m), n.pop = 1000)
#' coef(ape)
#' vcov(ape)
#' @export
getAPEs <- function(object = NULL, n.pop = NULL, panel.structure = c("classic", "network"),
                    sampling.fe = c("independence", "unrestricted"), weak.exo = FALSE) {
  bc_check_object(object, "getAPEs")
  if (!is.logical(weak.exo) || length(weak.exo) != 1 || is.na(weak.exo)) {
    stop("fereg: `weak.exo` must be TRUE or FALSE", call. = FALSE)
  }
  corrected <- inherits(object, "fereg_biascorr")
  if (corrected) {
    if (!missing(panel.structure) && match.arg(panel.structure) != object$panel.structure) {
      stop("fereg: the bias correction used panel.structure = \"", object$panel.structure,
        "\", which getAPEs() takes from it; drop `panel.structure`", call. = FALSE)
    }
    panel.structure <- object$panel.structure
    exo <- object$bandwidth > 0
    if (!missing(weak.exo) && weak.exo != exo) {
      stop("fereg: after a bias correction, weak.exo is ", exo, " (L = ", object$bandwidth,
        "); drop `weak.exo`", call. = FALSE)
    }
    weak.exo <- exo
  } else {
    panel.structure <- match.arg(panel.structure)
  }
  sampling.fe <- match.arg(sampling.fe)
  bc_check_fe(object, panel.structure, "getAPEs")
  if (!is.null(n.pop)) {
    if (!is.numeric(n.pop) || length(n.pop) != 1 || is.na(n.pop) || n.pop < 1) {
      stop("fereg: `n.pop` must be a positive number", call. = FALSE)
    }
    n.pop <- trunc(as.double(n.pop))
  }

  data <- fetch_data(object, "getAPEs() estimates the model again on it")
  setup <- object$bias.setup
  s <- bias_setup(object, data, setup$time_var, isTRUE(setup$common), vcov = new_vcov("iid"))
  at <- NULL
  if (corrected) {
    core <- order(object$coef_perm)
    opts <- object$bias.options
    opts$time <- s$time
    at <- list(
      bias = opts, coef = unname(object$coefficients[core]),
      ape_jackknife = if (!is.null(object$ape.jackknife)) unname(object$ape.jackknife[core])
    )
  }
  ape <- list(n_pop = n.pop, structure = panel.structure, sampling = sampling.fe, weak_exo = weak.exo)
  out <- fereg_apes(s$input, s$spec, at, ape, getOption("fereg.nthreads", default_threads()))
  for (nt in out$notes) warning(nt, call. = FALSE)
  perm <- coef_perm(out$coef_source)
  nm <- source_names(out$coef_source, s$info)[perm]
  V <- out$vcov[perm, perm, drop = FALSE]
  dimnames(V) <- list(nm, nm)
  res <- list(
    delta = stats::setNames(out$delta[perm], nm), vcov = V, panel.structure = panel.structure,
    sampling.fe = sampling.fe, weak.exo = weak.exo
  )
  if (corrected) {
    if (!is.null(out$bias_term)) res$bias.term <- stats::setNames(out$bias_term[perm], nm)
    res$bandwidth <- object$bandwidth
  }
  structure(res, class = "APEs")
}

#' Methods for average partial effects
#'
#' Methods for the results of [getAPEs()], as in alpaca: `print()` shows the APEs, `summary()`
#' their table with standard errors and z statistics (normal distribution), `coef()` and
#' `vcov()` extract the APEs and their variance.
#'
#' @param x,object An `"APEs"` object (`print`, `summary`, `coef`, `vcov`) or its summary
#'   (`"summary.APEs"`: `print`, `coef`).
#' @param digits Significant digits.
#' @param ... Not used.
#' @return `summary()`: an object of class `"summary.APEs"` holding the coefficient matrix `cm`
#'   (columns `Estimate`, `Std. error`, `z value`, `Pr(> |z|)`), which `coef()` returns.
#'   `coef()`: the named APEs. `vcov()`: their variance matrix.
#' @examples
#' data(trade_toy, package = "fereg")
#' ape <- getAPEs(feglm(binary ~ x1 + x2 | id + year, trade_toy, nthreads = 2))
#' ape
#' summary(ape)
#' @name APEs-methods
NULL

#' @rdname APEs-methods
#' @export
print.APEs <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  print(x[["delta"]], digits = digits)
  invisible(x)
}

#' @rdname APEs-methods
#' @export
summary.APEs <- function(object, ...) {
  est <- object[["delta"]]
  se <- sqrt(diag(object[["vcov"]]))
  z <- est / se
  cm <- cbind(est, se, z, 2 * stats::pnorm(-abs(z)))
  dimnames(cm) <- list(names(est), c("Estimate", "Std. error", "z value", "Pr(> |z|)"))
  structure(list(cm = cm), class = "summary.APEs")
}

#' @rdname APEs-methods
#' @export
print.summary.APEs <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("Estimates:\n")
  stats::printCoefmat(x[["cm"]], P.values = TRUE, has.Pvalue = TRUE, digits = digits)
  invisible(x)
}

#' @rdname APEs-methods
#' @export
coef.APEs <- function(object, ...) object[["delta"]]

#' @rdname APEs-methods
#' @export
coef.summary.APEs <- function(object, ...) object[["cm"]]

#' @rdname APEs-methods
#' @export
vcov.APEs <- function(object, ...) object[["vcov"]]

# ---- checks and setup ------------------------------------------------------------------------------

# A single fereg logit estimation with its estimation state (alpaca's messages where it has them).
bc_check_object <- function(object, fun) {
  if (is.null(object)) stop("'object' has to be specified.", call. = FALSE)
  if (inherits(object, "fereg_multi")) {
    stop("fereg: ", fun, "() takes a single estimation; select one of the multiple estimation (e.g. `est[[1]]`)", call. = FALSE)
  }
  if (!inherits(object, "fereg") || !identical(object$method_type, "feglm")) {
    stop("'", fun, "' called on a non-'feglm' object.", call. = FALSE)
  }
  if (!identical(object$family$family, "binomial")) {
    stop("'", fun, "' currently only supports binary choice models.", call. = FALSE)
  }
  if (is.null(object$state)) {
    stop("fereg: the estimation state was not kept (lean = TRUE); re-estimate the model with lean = FALSE", call. = FALSE)
  }
  if (length(object$coefficients) == 0) stop("fereg: ", fun, "() needs at least one regressor", call. = FALSE)
  invisible()
}

# Number of fixed effects, checked against the panel structure (alpaca's messages).
bc_check_fe <- function(object, panel.structure, fun) {
  terms <- object$fixef_terms
  if (any(vapply(terms, function(t) !t$intercept || length(t$slopes) > 0, TRUE))) {
    stop("fereg: ", fun, "() needs fixed effects without varying slopes", call. = FALSE)
  }
  k <- length(terms)
  if (panel.structure == "classic" && !k %in% 1:2) {
    stop("panel.structure == 'classic' expects a one- or two-way fixed effects model.", call. = FALSE)
  }
  if (panel.structure == "network" && !k %in% 2:3) {
    stop("panel.structure == 'network' expects a two- or three-way fixed effects model.", call. = FALSE)
  }
  k
}

bc_count <- function(x, name) {
  if (!is.numeric(x) || length(x) != 1 || is.na(x) || x < 0 || x != round(x)) {
    stop("fereg: `", name, "` must be a non-negative integer", call. = FALSE)
  }
  as.integer(x)
}

# The time variable of a one-way model: second variable of `panel.id` (argument, else the
# estimation's), whose first variable must be the fixed effect.
bc_time_var <- function(object, panel.id, data) {
  pid <- if (!is.null(panel.id)) parse_panel_id(panel.id, data) else object$panel.id
  if (is.null(pid)) {
    stop("fereg: the jackknife of a model with one fixed effect splits the observations of each individual in time; ",
      "give the time variable with `panel.id = ~id + time` (in biasCorr() or feglm())", call. = FALSE)
  }
  fe <- object$fixef_terms[[1]]$label
  if (!identical(fe, pid[1])) {
    stop("fereg: the fixed effect (`", fe, "`) must be the panel variable of `panel.id` (`", pid[1], "`)", call. = FALSE)
  }
  pid[2]
}

# The core input of `object`'s model for the bias-correction bindings: the estimation set up
# again on `data` at the object's variance (or `vcov`), with the character variables of the fixed
# effects coded in sorted (C-locale) order, the order of the fixed-effect levels being the time
# order of alpaca (which sorts the data by the fixed-effect variables). `time_var`: a time
# variable registered as a factor (index in `$time`). `common`: the first two fixed effects coded
# on one set of identifiers (double jackknife; numbers are matched by value).
bias_setup <- function(object, data, time_var = NULL, common = FALSE, vcov = NULL) {
  vi <- object$vcov_info
  modify <- list(vcov = vcov %||% vi$request, ssc = if (is.null(vcov)) vi$ssc, lean = FALSE, notes = FALSE)
  s <- refit_setup(object, modify, data)
  reg <- s$info$reg
  s$time <- if (!is.null(time_var)) time_index(reg, as.name(time_var))
  s$input$factors <- reg$cols
  s$input$factor_names <- reg$names
  for (j in unique(unlist(lapply(s$input$fe, `[[`, "factors")))) {
    v <- reg$values[[j]]
    if (is.character(v)) s$input$factors[[j]] <- sorted_codes(v, v)
  }
  if (common) {
    i1 <- s$input$fe[[1]]$factors
    i2 <- s$input$fe[[2]]$factors
    if (length(i1) != 1 || length(i2) != 1) {
      stop("fereg: jackknife = \"double\" needs two fixed effects that are single variables", call. = FALSE)
    }
    v1 <- reg$values[[i1]]
    v2 <- reg$values[[i2]]
    if (!(is.numeric(v1) && is.numeric(v2))) {
      l1 <- as.character(v1)
      l2 <- as.character(v2)
      s$input$factors[[i1]] <- sorted_codes(l1, c(l1, l2))
      s$input$factors[[i2]] <- sorted_codes(l2, c(l1, l2))
    }
  }
  s
}

# Codes of `v` (1-based, `levels` attribute) on the sorted distinct values of `all`.
sorted_codes <- function(v, all) {
  u <- sort(unique(all[!is.na(all)]), method = "radix")
  structure(match(v, u), levels = u)
}

# The line print() adds for a corrected estimation.
bias_label <- function(x) {
  k <- length(x$fixef_terms)
  how <- if (x$method == "analytical") {
    paste0("analytical (L = ", x$bandwidth, if (x$panel.structure == "network") ", network", ")")
  } else {
    paste0("jackknife (", x$jackknife, if (x$multiple > 0) paste0(", ", x$multiple, " random partitions"), ")")
  }
  what <- if (x$panel.structure == "network") {
    "all fixed effects"
  } else if (x$bias == "time") {
    "time effects"
  } else if (x$bias == "individual" || k == 1) {
    "individual effects"
  } else {
    "individual and time effects"
  }
  paste0("Bias correction: ", how, ", ", what)
}
