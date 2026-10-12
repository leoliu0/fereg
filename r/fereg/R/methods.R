# S3 methods of `fereg` objects (fixest-compatible names and behaviour).

#' Methods for fereg objects
#'
#' Accessors, post-hoc variances, predictions and fixed effects of [feols()], [fepois()] and
#' [feglm()] results, as in fixest.
#'
#' @param object A `fereg` object.
#' @param vcov,ssc,cluster,se Post-hoc variance (same forms as in [feols()]); computed from the
#'   retained estimation state without re-estimating. `NULL` keeps the estimation's variance.
#' @param stage IV only: `1` first stage(s), `2` second stage (default), `1:2` both.
#' @param attr Add the attributes `type`, `df.K`, `df.t` (and `G`) to the matrix.
#' @param parm Coefficients (names or indices); all by default.
#' @param level Confidence level.
#' @param type `resid()`: `"response"`, `"deviance"`, `"pearson"` or `"working"`;
#'   `fitted()` / `predict()`: `"response"` or `"link"`.
#' @param na.rm `FALSE` returns one value per row of the estimation data (NA for removed rows).
#' @param newdata A data frame for predictions; fixed-effect levels absent at estimation give NA.
#' @param k AIC penalty per parameter.
#' @param ... Not used.
#' @name fereg-methods
NULL

#' @rdname fereg-methods
#' @export
summary.fereg <- function(object, vcov = NULL, ssc = NULL, cluster = NULL, se = NULL, stage = 2, ...) {
  if (isTRUE(object$is_iv) && object$iv_stage == 2 && !identical(as.integer(stage), 2L)) {
    stages <- list()
    if (1 %in% stage) stages <- c(stages, object$iv_first_stage)
    if (2 %in% stage) stages <- c(stages, list(update_vcov(object, vcov, ssc, cluster, se)))
    if (length(stages) == 1) {
      stages[[1]]$summary <- TRUE
      return(stages[[1]])
    }
    ids <- c(lapply(names(object$iv_first_stage), function(e) c(stage = "1", endo = e)), if (2 %in% stage) list(c(stage = "2")))
    return(new_multi(stages, ids[seq_along(stages)]))
  }
  res <- update_vcov(object, vcov, ssc, cluster, se)
  res$summary <- TRUE
  res
}

# Variance matrix for another vcov / ssc from the retained estimation state.
update_vcov <- function(object, vcov = NULL, ssc = NULL, cluster = NULL, se = NULL) {
  if (is.null(vcov) && is.null(ssc) && is.null(cluster) && is.null(se)) return(object)
  if (isTRUE(object$iv_stage == 1)) {
    stop("fereg: post-hoc variances are not available for IV first stages; re-estimate with `vcov`", call. = FALSE)
  }
  if (is.null(object$state)) {
    stop("fereg: the estimation state was not kept (lean = TRUE); re-estimate with `vcov`", call. = FALSE)
  }
  if (!is.null(ssc) && !inherits(ssc, "ssc.type")) stop("fereg: `ssc` must be created with ssc()", call. = FALSE)
  v <- if (is.null(vcov) && is.null(cluster) && is.null(se)) object$vcov_info[c("type", "dims")] else parse_vcov(vcov, cluster, se)
  v <- resolve_vcov(v, object$fixef_terms)
  ssc_used <- effective_ssc(ssc, v$type)
  nthreads <- getOption("fereg.nthreads", default_threads())
  if (v$type == "cluster") {
    data <- fetch_data(object)
    reg <- new_factors(data, environment(object$fml), new.env(parent = emptyenv()))
    idx <- lapply(v$dims, function(d) vapply(d, function(e) factor_index(reg, e), 0))
    out <- fereg_vcov(object$state, "cluster", ssc_core(ssc_used), reg$cols, reg$names, idx, nthreads)
  } else {
    out <- fereg_vcov(object$state, v$type, ssc_core(ssc_used), list(), character(0), list(), nthreads)
  }
  set_vcov(object, out, v, ssc_used)
}

# The estimation data, found again from the call (as fixest does).
fetch_data <- function(object) {
  data <- tryCatch(eval(object$call$data, object$call_env), error = function(e) NULL)
  if (is.null(data) || !is.list(data) || data_nrow(data) != object$nobs_origin) {
    stop("fereg: the estimation data (`", deparse1l(object$call$data), "`) cannot be found or has changed; ",
      "clustered variances after estimation need it", call. = FALSE)
  }
  data
}

#' @rdname fereg-methods
#' @export
coef.fereg <- function(object, ...) object$coefficients

#' @rdname fereg-methods
#' @export
vcov.fereg <- function(object, vcov = NULL, ssc = NULL, attr = FALSE, cluster = NULL, se = NULL, ...) {
  V <- update_vcov(object, vcov, ssc, cluster, se)$cov.scaled
  if (!attr) attributes(V) <- attributes(V)[c("dim", "dimnames")]
  V
}

#' Coefficient table accessors
#'
#' `se()`, `tstat()`, `pvalue()` and `coeftable()` as in fixest.
#'
#' @param object A `fereg` (or `fereg_multi`) object.
#' @param ... Passed to [summary()][fereg-methods] (`vcov`, `ssc`, `cluster`).
#' @return Named vectors, or the coefficient matrix for `coeftable()`.
#' @export
se <- function(object, ...) UseMethod("se")

#' @rdname se
#' @export
tstat <- function(object, ...) UseMethod("tstat")

#' @rdname se
#' @export
pvalue <- function(object, ...) UseMethod("pvalue")

#' @rdname se
#' @export
coeftable <- function(object, ...) UseMethod("coeftable")

#' @export
coeftable.fereg <- function(object, ...) summary.fereg(object, ...)$coeftable

# Column `j` of the coefficient table, with fixest's `vcov_type` attribute.
coeftable_column <- function(object, j, ...) {
  ct <- coeftable.fereg(object, ...)
  structure(stats::setNames(ct[, j], rownames(ct)), vcov_type = attr(ct, "vcov_type"))
}

#' @export
se.fereg <- function(object, ...) coeftable_column(object, 2, ...)

#' @export
tstat.fereg <- function(object, ...) coeftable_column(object, 3, ...)

#' @export
pvalue.fereg <- function(object, ...) coeftable_column(object, 4, ...)

#' @rdname fereg-methods
#' @export
confint.fereg <- function(object, parm, level = 0.95, vcov = NULL, ssc = NULL, cluster = NULL, se = NULL, ...) {
  s <- update_vcov(object, vcov, ssc, cluster, se)
  ct <- s$coeftable
  if (!missing(parm)) ct <- ct[parm, , drop = FALSE]
  a <- (1 - level) / 2
  q <- if (isTRUE(s$vcov_info$t_dist)) stats::qt(1 - a, s$vcov_info$df_t) else stats::qnorm(1 - a)
  out <- data.frame(ct[, 1] - q * ct[, 2], ct[, 1] + q * ct[, 2], row.names = rownames(ct))
  names(out) <- paste(format(100 * c(a, 1 - a), trim = TRUE, scientific = FALSE, digits = 3), "%")
  out
}

#' @rdname fereg-methods
#' @export
nobs.fereg <- function(object, ...) object$nobs

# Per-observation vector of the estimation rows, optionally expanded to all input rows.
obs_vector <- function(object, v, na.rm) {
  if (is.null(v)) stop("fereg: not available (lean = TRUE)", call. = FALSE)
  if (na.rm || is.null(object$obs_selection)) return(v)
  out <- rep(NA_real_, object$nobs_origin)
  out[object$obs_selection] <- v
  out
}

#' @rdname fereg-methods
#' @export
residuals.fereg <- function(object, type = c("response", "deviance", "pearson", "working"), na.rm = TRUE, ...) {
  type <- match.arg(type)
  r <- object$residuals
  if (is.null(r)) stop("fereg: residuals are not available (lean = TRUE)", call. = FALSE)
  w <- object$weights %||% 1
  if (type != "response") {
    if (!is_glm(object)) {
      if (type == "working") stop("fereg: working residuals are only defined for GLMs", call. = FALSE)
      r <- r * sqrt(w)
    } else {
      fam <- object$family
      mu <- object$fitted.values
      y <- r + mu
      r <- switch(type,
        deviance = sign(r) * sqrt(pmax(fam$dev.resids(y, mu, rep_len(w, length(y))), 0)),
        pearson = r * sqrt(w) / sqrt(fam$variance(mu)),
        working = r / fam$mu.eta(object$linear.predictors)
      )
    }
  }
  obs_vector(object, r, na.rm)
}

#' @rdname fereg-methods
#' @export
fitted.fereg <- function(object, type = c("response", "link"), na.rm = TRUE, ...) {
  type <- match.arg(type)
  v <- if (type == "link" && is_glm(object)) object$linear.predictors else object$fitted.values
  obs_vector(object, v, na.rm)
}

#' @rdname fereg-methods
#' @export
predict.fereg <- function(object, newdata, type = c("response", "link"), ...) {
  type <- match.arg(type)
  if (missing(newdata) || is.null(newdata)) return(fitted.fereg(object, type = type))
  if (!is.list(newdata)) stop("fereg: `newdata` must be a data frame", call. = FALSE)
  if (isTRUE(object$iv_stage == 1)) stop("fereg: predict() is not available for IV first stages", call. = FALSE)
  env <- environment(object$fml)
  n <- data_nrow(newdata)
  coef <- object$coefficients
  lin <- linear_design(NULL, newdata, env, length(object$fixef_terms) > 0, meta = object$design$lin_meta)
  cols <- stats::setNames(lin$cols, lin$names)
  if (isTRUE(object$is_iv)) {
    for (j in seq_along(object$design$endo_exprs)) {
      cols[[paste0("fit_", object$design$endo_names[j])]] <- as_num(eval_var(object$design$endo_exprs[[j]], newdata, env), object$design$endo_names[j])
    }
  }
  eta <- rep(0, n)
  for (nm in names(coef)) {
    eta <- eta + coef[[nm]] * if (nm == "(Intercept)") 1 else cols[[nm]]
  }
  eta <- eta + fixef_predict(object, newdata, env, n)
  if (isTRUE(object$design$has_offset)) {
    off <- object$design$offset
    if (!inherits(off, "formula")) stop("fereg: predict() with newdata needs the offset given as a formula (offset = ~x)", call. = FALSE)
    eta <- eta + as_num(eval(off[[length(off)]], newdata, environment(off)), "offset")
  }
  if (type == "response" && is_glm(object)) object$family$linkinv(eta) else eta
}

# Sum of the fixed effects (and varying slopes) at the rows of `newdata`.
fixef_predict <- function(object, newdata, env, n) {
  if (length(object$fixef_raw) == 0) return(0)
  fe <- fixef.fereg(object)
  out <- rep(0, n)
  k <- 0
  for (f in object$fixef_raw) {
    k <- k + 1
    t <- object$fixef_terms[[f$term]]
    labels <- fe_labels(lapply(t$vars, function(e) eval_var(e, newdata, env)))
    v <- unname(fe[[k]][match(labels, names(fe[[k]]))])
    if (!is.na(f$slope)) v <- v * as_num(eval_var(str2lang(object$slope_names[f$slope]), newdata, env), object$slope_names[f$slope])
    out <- out + v
  }
  out
}

#' Fixed effects of an estimation
#'
#' Fixed-effect coefficients with fixest's normalization: the first fixed effect is free, and in
#' every connected component one level of each other fixed effect is set to 0 (counted in the
#' `references` attribute). Varying slopes are listed as `fe[[x]]`.
#'
#' @param object A `fereg` object.
#' @param ... Not used.
#' @return A list (class `fereg.fixef`) of named vectors, one per fixed effect / slope.
#' @export
fixef <- function(object, ...) UseMethod("fixef")

#' @export
fixef.fereg <- function(object, ...) {
  raw <- object$fixef_raw
  if (length(raw) == 0) stop("fereg: the estimation has no fixed effects", call. = FALSE)
  out <- list()
  refs <- integer(0)
  for (k in seq_along(raw)) {
    f <- raw[[k]]
    t <- object$fixef_terms[[f$term]]
    vals <- object$fixef_level_values[[k]]
    ord <- fe_order(vals)
    v <- stats::setNames(f$values, fe_labels(vals))[ord]
    name <- if (is.na(f$slope)) t$label else paste0(t$label, "[[", object$slope_names[f$slope], "]]")
    out[[name]] <- v
    refs[name] <- as.integer(f$references)
  }
  structure(out, class = c("fereg.fixef", "list"), references = refs)
}

#' @export
print.fereg.fixef <- function(x, ...) {
  print(lapply(unclass(x), identity))
  cat("References:", paste0(names(attr(x, "references")), ": ", attr(x, "references"), collapse = ", "), "\n")
  invisible(x)
}

#' @export
summary.fereg.fixef <- function(object, ...) {
  tab <- rbind(
    "Number of fixed-effects" = lengths(object),
    "Number of references" = attr(object, "references"),
    "Mean" = vapply(object, mean, 0),
    "Standard-deviation" = vapply(object, stats::sd, 0)
  )
  cat("Fixed_effects coefficients\n")
  print(signif(tab, 3))
  invisible(tab)
}

#' @rdname fereg-methods
#' @export
logLik.fereg <- function(object, ...) {
  # OLS also estimates the residual variance (fixest counts it in `df`).
  df <- object$nparams + if (is_glm(object)) 0 else 1
  structure(object$loglik, df = df, nobs = object$nobs, class = "logLik")
}

#' @rdname fereg-methods
#' @export
AIC.fereg <- function(object, ..., k = 2) -2 * object$loglik + k * object$nparams

#' @rdname fereg-methods
#' @export
BIC.fereg <- function(object, ...) -2 * object$loglik + log(object$nobs) * object$nparams

#' @rdname fereg-methods
#' @export
deviance.fereg <- function(object, ...) if (is_glm(object)) object$deviance else object$ssr

#' @rdname fereg-methods
#' @export
sigma.fereg <- function(object, ...) sqrt(deviance.fereg(object) / (object$nobs - object$nparams))

# ---- broom-style tidiers (registered on generics::tidy / glance when available) ------------------

tidy.fereg <- function(x, conf.int = FALSE, conf.level = 0.95, ...) {
  ct <- summary.fereg(x, ...)$coeftable
  out <- data.frame(
    term = rownames(ct), estimate = unname(ct[, 1]), std.error = unname(ct[, 2]),
    statistic = unname(ct[, 3]), p.value = unname(ct[, 4]), stringsAsFactors = FALSE
  )
  if (conf.int) {
    ci <- confint.fereg(x, level = conf.level, ...)
    out$conf.low <- ci[[1]]
    out$conf.high <- ci[[2]]
  }
  out
}

glance.fereg <- function(x, ...) {
  r <- r2(x)
  data.frame(
    r.squared = r[["r2"]], adj.r.squared = r[["ar2"]], within.r.squared = r[["wr2"]],
    pseudo.r.squared = r[["pr2"]], sigma = sigma.fereg(x), nobs = x$nobs,
    AIC = AIC.fereg(x), BIC = BIC.fereg(x), logLik = x$loglik
  )
}
