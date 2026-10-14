# S3 methods of `fereg` objects (fixest-compatible names and behaviour).

#' Methods for fereg objects
#'
#' Accessors, post-hoc variances, predictions and fixed effects of [feols()], [fepois()] and
#' [feglm()] results, as in fixest.
#'
#' @param object A `fereg` object.
#' @param vcov,ssc,cluster,se Post-hoc variance (same forms as in [feols()], including the
#'   request objects of [vcov_requests]); computed from the retained estimation state without
#'   re-estimating, except CRV3, whose leave-one-cluster-out refits re-estimate the model on the
#'   estimation data (found again from the call). `NULL` keeps the estimation's variance.
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
#' @return `summary()`: the `fereg` object with the requested variance (coefficient table in
#'   `$coeftable`); for IV with `stage = 1` or `1:2`, the requested stages (a `fereg_multi` when
#'   there are several). `coef()`: named vector of coefficients. `vcov()`: variance matrix.
#'   `confint()`: data frame of lower and upper bounds, one row per coefficient. `nobs()`: number
#'   of observations. `resid()`, `fitted()`, `predict()`: numeric vectors. `logLik()`: object of
#'   class `logLik`. `AIC()`, `BIC()`, `deviance()`, `sigma()`: numbers.
#' @examples
#' data(trade_toy, package = "fereg")
#' m <- feols(y ~ x1 + x2 | id + year, trade_toy, nthreads = 2)
#' summary(m, vcov = "hetero")
#' coef(m)
#' vcov(m, vcov = ~id)
#' confint(m, level = 0.9)
#' head(predict(m, newdata = trade_toy))
#' c(AIC = AIC(m), BIC = BIC(m))
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
  if (!is.null(ssc) && !inherits(ssc, "ssc.type")) stop("fereg: `ssc` must be created with ssc()", call. = FALSE)
  v <- if (is.null(vcov) && is.null(cluster) && is.null(se)) object$vcov_info$request else parse_vcov(vcov, cluster, se)
  if (v$type == "crv3") return(refit_vcov(object, v, ssc))
  if (is.null(object$state)) {
    stop("fereg: the estimation state was not kept (lean = TRUE); re-estimate with `vcov`", call. = FALSE)
  }
  ssc_used <- effective_ssc(v$ssc %||% ssc, v$type)
  nthreads <- getOption("fereg.nthreads", default_threads())
  if (v$type %in% c("iid", "hetero", "hc2", "hc3")) {
    out <- fereg_vcov(object$state, core_kind(v), ssc_core(ssc_used), list(), character(0), list(), list(), nthreads)
    return(set_vcov(object, out, v, ssc_used))
  }
  data <- fetch_data(object)
  v <- resolve_vcov(v, object$fixef_terms, object$panel.id, names(data))
  reg <- new_factors(data, environment(object$fml), new.env(parent = emptyenv()))
  vc <- vcov_columns(v, reg)
  for (k in c("lat", "lon")) check_coord_na(object, vc[[k]], v$vars[[k]], c(lat = "latitude", lon = "longitude")[[k]])
  vars <- list(time = vc$time, panel = vc$panel, lat = vc$lat, lon = vc$lon)
  out <- fereg_vcov(object$state, core_kind(v), ssc_core(ssc_used), reg$cols, reg$names, vc$cluster, vars, nthreads)
  set_vcov(object, out, v, ssc_used)
}

# fixest's error for a coordinate with missing values on the estimation rows (post hoc).
check_coord_na <- function(object, values, expr, label) {
  if (is.null(values)) return(invisible())
  if (anyNA(obs_subset(values, object$obs_selection))) {
    stop("The variable '", deparse1l(expr), "' used to estimate the VCOV (employed as ", label,
      ") has NA values which would lead to a sample used to compute the VCOV different from the sample used to estimate the parameters. ",
      "This would lead to wrong inference.\nPossible solutions: i) ex ante prune them, ii) impute them, or iii) use the argument 'vcov' at estimation time.",
      call. = FALSE)
  }
  invisible()
}

# CRV3 after estimation: the jackknife refits need the data, so the model is estimated again on
# the same rows with vcov = CRV3 (only its variance is used).
refit_vcov <- function(object, v, ssc) {
  r <- object$refit
  data <- fetch_data(object)
  n <- data_nrow(data)
  args <- r$args
  args$data <- data
  args$vcov <- v
  args$ssc <- ssc
  args$lean <- TRUE
  args$notes <- FALSE
  w <- row_vector(args$weights, data, r$env, n, "weights")
  if (!is.null(w)) w <- as_num(w, "weights")
  off <- row_vector(args$offset, data, r$env, n, "offset")
  if (!is.null(off)) off <- as_num(off, "offset")
  new <- fit_one(r$m, args, data, r$env, n, w, off, r$drop, new.env(parent = emptyenv()))
  if (!identical(names(new$coefficients), names(object$coefficients))) {
    stop("fereg: the re-estimation for CRV3 does not reproduce the model; is the estimation data unchanged?", call. = FALSE)
  }
  apply_vcov(object, new$cov.scaled, new$vcov_info)
}

# The estimation data, found again from the call (as fixest does).
fetch_data <- function(object) {
  data <- tryCatch(eval(object$call$data, object$call_env), error = function(e) NULL)
  if (is.null(data) || !is.list(data) || data_nrow(data) != object$nobs_origin) {
    stop("fereg: the estimation data (`", deparse1l(object$call$data), "`) cannot be found or has changed; ",
      "clustered, Newey-West, Driscoll-Kraay, Conley and CRV3 variances after estimation need it", call. = FALSE)
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
#' @examples
#' data(trade_toy, package = "fereg")
#' m <- feols(y ~ x1 + x2 | id + year, trade_toy, nthreads = 2)
#' se(m)
#' pvalue(m, vcov = "hetero")
#' coeftable(m, vcov = ~id)
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
  if (length(object$fixef_terms) == 0) return(0)
  fe <- fixef.fereg(object)
  raw <- fixef_raw(object)
  out <- rep(0, n)
  for (k in seq_along(raw)) {
    f <- raw[[k]]
    t <- object$fixef_terms[[f$term]]
    labels <- fe_labels(lapply(t$vars, function(e) eval_var(e, newdata, env)))
    v <- unname(fe[[k]][match(labels, names(fe[[k]]))])
    if (!is.na(f$slope)) v <- v * as_num(eval_var(str2lang(object$slope_names[f$slope]), newdata, env), object$slope_names[f$slope])
    out <- out + v
  }
  out
}

# Fixed-effect values per term and slot (core order), computed from the estimation state on
# first use and cached on the object.
fixef_raw <- function(object) {
  cache <- object$fixef_cache
  if (is.null(cache$raw)) {
    if (is.null(object$state)) {
      stop("fereg: fixed-effect values need the estimation state, which was not kept (lean = TRUE)", call. = FALSE)
    }
    cache$raw <- fereg_fixef(object$state, getOption("fereg.nthreads", default_threads()))
  }
  cache$raw
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
#' @examples
#' data(trade_toy, package = "fereg")
#' m <- feols(y ~ x1 + x2 | id + year, trade_toy, nthreads = 2)
#' fe <- fixef(m)
#' summary(fe)
#' head(fe$year)
#' @export
fixef <- function(object, ...) UseMethod("fixef")

#' @export
fixef.fereg <- function(object, ...) {
  if (length(object$fixef_terms) == 0) stop("fereg: the estimation has no fixed effects", call. = FALSE)
  if (isTRUE(object$iv_stage == 1)) {
    stop("fereg: fixed-effect values are not available for first-stage models; use fixef() on the second stage", call. = FALSE)
  }
  cache <- object$fixef_cache
  if (!is.null(cache$fixef)) return(cache$fixef)
  raw <- fixef_raw(object)
  out <- list()
  refs <- integer(0)
  for (f in raw) {
    t <- object$fixef_terms[[f$term]]
    vals <- lapply(object$fe_columns[[f$term]], function(v) v[f$first_obs])
    v <- stats::setNames(f$values, fe_labels(vals))[fe_order(vals)]
    name <- if (is.na(f$slope)) t$label else paste0(t$label, "[[", object$slope_names[f$slope], "]]")
    out[[name]] <- v
    refs[name] <- as.integer(f$references)
  }
  cache$fixef <- structure(out, class = c("fereg.fixef", "list"), references = refs)
  cache$fixef
}

#' @export
print.fereg.fixef <- function(x, ...) {
  print(lapply(unclass(x), identity))
  cat("References:", paste0(names(attr(x, "references")), ": ", attr(x, "references"), collapse = ", "), "\n")
  invisible(x)
}

# Layout of fixest's summary.fixest.fixef: a table of counts (integers), means and standard
# deviations (3 significant digits) per fixed effect / slope (one line when there is a single
# one), then the first `n` coefficients of each (4 significant digits). Returns NULL invisibly.
#' @export
summary.fereg.fixef <- function(object, n = 5, ...) {
  q <- length(object)
  nm <- names(object)
  is_slope <- grepl("\\[", nm)
  fe <- gsub("\\[.+", "", nm)
  slope <- gsub(".+\\[|\\].+", "", nm)
  nm[is_slope] <- paste0(slope[is_slope], " (slopes: ", fe[is_slope], ")")
  kind <- if (all(is_slope)) "slope" else if (any(is_slope)) "both" else "fe"
  cat(switch(kind, both = "Fixed-effects/Slope", fe = "Fixed_effects", slope = "Slope"), " coefficients\n", sep = "")
  if (q == 1) {
    x1 <- object[[1]]
    if (is_slope) {
      cat("Number of slope coefficients for variable ", slope, " (slope: ", fe, ") is ", format_int(length(x1)), ".\n", sep = "")
    } else {
      cat("Number of fixed-effects for variable ", nm, " is ", format_int(length(x1)), ".\n", sep = "")
    }
    cat("\tMean = ", signif(mean(x1), 3), "\tVariance = ", signif(stats::var(x1), 3), "\n", sep = "")
  } else {
    res <- as.data.frame(rbind(
      lengths(object),
      attr(object, "references"),
      vapply(object, function(v) as.character(signif(mean(v), 3)), ""),
      vapply(object, function(v) as.character(signif(stats::sd(v), 3)), "")
    ))
    rownames(res) <- c(
      paste0("Number of ", switch(kind, both = "fixed-effects/slopes", fe = "fixed-effects", slope = "slopes")),
      "Number of references", "Mean", "Standard-deviation"
    )
    colnames(res) <- nm
    print(res)
  }
  cat("\nCOEFFICIENTS:\n")
  for (i in seq_len(q)) {
    m <- object[[i]][seq_len(min(n, length(object[[i]])))]
    m_char <- as.data.frame(t(as.data.frame(c("", as.character(signif(m, 4))))))
    names(m_char) <- c(paste0(nm[i], ":"), names(m))
    rownames(m_char) <- " "
    if (length(object[[i]]) > n) m_char[["   "]] <- paste0("... ", format_int(length(object[[i]]) - n), " remaining")
    print(m_char)
    if (i != q) cat("-----\n")
  }
  invisible(NULL)
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
