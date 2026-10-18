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
  if (inherits(object, "fereg_biascorr")) stop("CRV3 is not available after bias correction", call. = FALSE)
  s <- refit_setup(object, list(vcov = v, ssc = ssc, lean = TRUE, notes = FALSE))
  new <- build_result(fereg_fit(s$input, s$spec, s$info$args$nthreads), s$info)
  if (!identical(names(new$coefficients), names(object$coefficients))) {
    stop("fereg: the re-estimation for CRV3 does not reproduce the model; is the estimation data unchanged?", call. = FALSE)
  }
  apply_vcov(object, new$cov.scaled, new$vcov_info)
}

# The estimation of `object` set up again on its data (found again from the call): fit_setup()'s
# core input and spec, and build_result()'s info, with the arguments in `modify` replaced.
refit_setup <- function(object, modify = list(), data = fetch_data(object)) {
  r <- object$refit
  n <- data_nrow(data)
  args <- r$args
  args$data <- data
  args[names(modify)] <- modify
  w <- row_vector(args$weights, data, r$env, n, "weights")
  if (!is.null(w)) w <- as_num(w, "weights")
  off <- row_vector(args$offset, data, r$env, n, "offset")
  if (!is.null(off)) off <- as_num(off, "offset")
  fit_setup(r$m, args, data, r$env, n, w, off, r$drop, new.env(parent = emptyenv()))
}

# The estimation data, found again from the call (as fixest does). `why` names what needs it.
fetch_data <- function(object, why = NULL) {
  data <- tryCatch(eval(object$call$data, object$call_env), error = function(e) NULL)
  if (is.null(data) || !is.list(data) || data_nrow(data) != object$nobs_origin) {
    why <- why %||% "clustered, Newey-West, Driscoll-Kraay, Conley and CRV3 variances after estimation need it"
    stop("fereg: the estimation data (`", deparse1l(object$call$data), "`) cannot be found or has changed; ", why, call. = FALSE)
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
  fe <- fixef.fereg(object, notes = FALSE)
  raw <- fixef_raw(object)
  out <- rep(0, n)
  for (f in raw) {
    t <- object$fixef_terms[[f$term]]
    labels <- fe_labels(lapply(t$vars, function(e) eval_var(e, newdata, env)))
    vals <- fe[[fixef_name(object, f)]]
    v <- unname(vals[match(labels, names(vals))])
    if (!is.na(f$slope)) v <- v * as_num(eval_var(str2lang(object$slope_names[f$slope]), newdata, env), object$slope_names[f$slope])
    out <- out + v
  }
  out
}

# Fixed-effect values per term and slot (core order), computed from the estimation state on
# first use and cached on the object.
fixef_raw <- function(object) {
  cache <- object$fixef_cache
  if (is.null(cache$raw)) cache$raw <- fixef_state(object, min_norm = FALSE)
  cache$raw
}

fixef_state <- function(object, min_norm) {
  if (is.null(object$state)) {
    stop("fereg: fixed-effect values need the estimation state, which was not kept (lean = TRUE)", call. = FALSE)
  }
  fereg_fixef(object$state, getOption("fereg.nthreads", default_threads()), min_norm)
}

#' Fixed effects of an estimation
#'
#' Fixed-effect coefficients with fixest's normalization: the first fixed effect is free, and in
#' every connected component one level of each other fixed effect is set to 0 (counted in the
#' `references` attribute when there are several fixed effects / slopes). Varying slopes are
#' listed as `fe[[x]]`; with varying slopes, the terms are in fixest's order (by number of
#' coefficients, decreasing). As fixest, a note says when the fixed effects are not regular (more
#' references than one per fixed effect after the first), and when, with three fixed effects or
#' more, these references do not reproduce the estimated fixed effects: the values are then those
#' of another normalization and the number of references is only approximate.
#'
#' @param object A `fereg` object.
#' @param notes Print the note on irregular fixed effects (option `fereg.notes`, default `TRUE`).
#' @param ... Not used.
#' @return A list (class `fereg.fixef`) of named vectors, one per fixed effect / slope, with
#'   fixest's attributes `references` and `exponential` (`TRUE` for Poisson models).
#' @examples
#' data(trade_toy, package = "fereg")
#' m <- feols(y ~ x1 + x2 | id + year, trade_toy, nthreads = 2)
#' fe <- fixef(m)
#' summary(fe)
#' head(fe$year)
#' @export
fixef <- function(object, ...) UseMethod("fixef")

#' @rdname fixef
#' @export
fixef.fereg <- function(object, notes = getOption("fereg.notes", TRUE), ...) {
  if (length(object$fixef_terms) == 0) stop("fereg: the estimation has no fixed effects", call. = FALSE)
  if (isTRUE(object$iv_stage == 1)) {
    stop("fereg: fixed-effect values are not available for first-stage models; use fixef() on the second stage", call. = FALSE)
  }
  cache <- object$fixef_cache
  if (is.null(cache$fixef)) fixef_build(object, cache)
  if (isTRUE(notes) && !is.null(cache$fixef_note)) message(cache$fixef_note)
  cache$fixef
}

# fixest's fixef.fixest: the values, the `references` attribute (more than one vector) and the
# note on irregular fixed effects, cached in `cache` (`fixef`, `fixef_note`).
fixef_build <- function(object, cache) {
  raw <- fixef_raw(object)
  terms <- vapply(raw, function(f) as.integer(f$term), 0L)
  slope <- vapply(raw, function(f) !is.na(f$slope), TRUE)
  refs <- vapply(raw, function(f) as.integer(f$references), 0L)
  q <- length(unique(terms))
  if (any(slope)) {
    # With varying slopes, fixest orders the terms by number of coefficients (levels x vectors),
    # decreasing (not in multiple-outcome GLMs), and counts the references in that order.
    ncoef <- object$fixef_sizes[unique(terms)] * tabulate(terms, max(terms))[unique(terms)]
    multi_glm <- is_glm(object) && "lhs" %in% names(object$refit$m$id)
    if (!multi_glm && any(ncoef != sort(ncoef, decreasing = TRUE))) {
      new <- unique(terms)[order(ncoef, decreasing = TRUE)]
      idx <- unlist(lapply(new, function(k) which(terms == k)))
      raw <- raw[idx]
      terms <- terms[idx]
      slope <- slope[idx]
      refs <- refs[idx]
      by_term <- fereg_fixef_references(object$state, new, getOption("fereg.nthreads", default_threads()))
      refs[!slope] <- by_term[terms[!slope]]
    }
  }
  approx <- FALSE
  if (!any(slope) && q >= 3 && any(refs > 1)) {
    # With three fixed effects or more, fixest's references may not reproduce the fixed-effect
    # sum S of every observation; fixest then solves the values again and keeps the number of
    # references, as only approximate. Here S comes from the minimum-norm solution.
    exact <- fixef_state(object, min_norm = TRUE)
    sum_fe <- s <- 0
    for (k in seq_along(raw)) {
      code <- fe_level_codes(object$fe_columns[[terms[k]]], object$obs_selection, raw[[k]]$first_obs)
      sum_fe <- sum_fe + raw[[k]]$values[code]
      s <- s + exact[[k]]$values[code]
    }
    if (max(abs(sum_fe - s)) > 0.1) {
      approx <- TRUE
      raw <- exact
    }
  }
  out <- list()
  for (f in raw) {
    vals <- lapply(object$fe_columns[[f$term]], function(v) v[f$first_obs])
    out[[fixef_name(object, f)]] <- stats::setNames(f$values, fe_labels(vals))[fe_order(vals)]
  }
  class(out) <- c("fereg.fixef", "list")
  if (length(out) > 1) {
    names(refs) <- names(out)
    attr(out, "references") <- refs
    # fixest's test: more than q - 1 references (q: fixed-effect variables), counted over the
    # vectors selected by its per-variable "no slope" flags, recycled over all the vectors.
    no_slope <- vapply(split(slope, factor(terms, unique(terms))), function(s) !any(s), TRUE)
    if (sum(refs[rep_len(no_slope, length(refs))]) > q - 1) {
      cache$fixef_note <- paste0(
        "NOTE: The fixed-effects are not regular, they cannot be straightforwardly interpreted.",
        if (approx) " The number of references is only approximate."
      )
    }
  }
  attr(out, "exponential") <- identical(object$family$family, "poisson")
  cache$fixef <- out
}

fixef_name <- function(object, f) {
  label <- object$fixef_terms[[f$term]]$label
  if (is.na(f$slope)) label else paste0(label, "[[", object$slope_names[f$slope], "]]")
}

# Level (index into the values of `first_obs`) of every estimation row, for the FE variables
# `cols` (all rows) of one term: interactions are matched on a mixed-radix key of the codes of
# their variables (exact in doubles below 2^53 combinations, else on the labels).
fe_level_codes <- function(cols, rows, first_obs) {
  if (length(cols) == 1) return(match(obs_subset(cols[[1]], rows), cols[[1]][first_obs]))
  uniq <- lapply(cols, function(v) unique(v[first_obs]))
  if (prod(lengths(uniq)) >= 2^53) {
    return(match(fe_labels(lapply(cols, obs_subset, rows)), fe_labels(lapply(cols, function(v) v[first_obs]))))
  }
  key_rows <- key_first <- 0
  for (j in seq_along(cols)) {
    key_rows <- key_rows * length(uniq[[j]]) + match(obs_subset(cols[[j]], rows), uniq[[j]]) - 1
    key_first <- key_first * length(uniq[[j]]) + match(cols[[j]][first_obs], uniq[[j]]) - 1
  }
  match(key_rows, key_first)
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
