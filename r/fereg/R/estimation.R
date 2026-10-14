# Estimation functions: feols(), fepois(), feglm(). Formula parsing and design columns are
# built here; everything else (sample, fixed effects, estimation, variance, statistics) is done
# by the Rust core (fereg-core `fit`, fixest conventions).

#' Fixed-effects OLS and IV estimation
#'
#' Drop-in replacement for [fixest::feols()]: same formula syntax, arguments and defaults, same
#' numbers. Estimation runs in the fereg Rust core.
#'
#' @param fml A formula `y ~ x1 + x2 | fe1 + fe2 | endo ~ inst`. The fixed-effects part accepts
#'   interactions `fe1^fe2` and varying slopes `fe[x]` (fixed effect and slope) or `fe[[x]]`
#'   (slope only). The right-hand side accepts `i()`, `factor()`, `I()`, functions of variables,
#'   interactions and `-1`. Several dependent variables `c(y1, y2)` and the stepwise functions
#'   `sw()`, `sw0()`, `csw()`, `csw0()` (right-hand side and fixed-effects part) give multiple
#'   estimations.
#' @param data A data frame (or a list of equal-length vectors).
#' @param vcov Variance type, as in fixest: a string (`"iid"`, the default; `"hetero"` (HC1),
#'   `"hc2"`, `"hc3"`, `"cluster"` (first `panel.id` variable, else first fixed effect),
#'   `"twoway"`, `"threeway"`, `"fourway"`, `"NW"` / `"newey_west"`, `"DK"` /
#'   `"driscoll_kraay"`, `"conley"`, and fereg's extension `"crv3"`; case and unique
#'   abbreviations are accepted), a one-sided formula (`~cl`, `~cl1 + cl2`, up to four
#'   dimensions, `~cl1^cl2` clusters on the interaction), a formula `type ~ variables`
#'   (`NW ~ id + period`, `NW(2) ~ id + period`, `NW ~ period` (time series without
#'   `panel.id`), `DK(2) ~ period`, `conley(100) ~ lat + lon`, `conley("60mi") ~ lat + lon`,
#'   `crv3 ~ g1 + g2`; `+ ssc(...)` among the variables sets the small-sample correction) or a
#'   request object (see [vcov_requests]). Variables of Newey-West and Driscoll-Kraay variances
#'   default to `panel.id`, Conley coordinates are guessed from the column names. Rows with
#'   missing values in a variance variable are removed (counted as "vcov" in the NA note).
#' @param weights Observation weights: a one-sided formula (`~w`), a column name or a vector.
#' @param offset Offset on the link scale: a one-sided formula, a column name or a vector.
#' @param subset Rows to use: logical or integer vector, or a one-sided formula (`~year > 2000`).
#' @param split,fsplit A one-sided formula (or column name): one estimation per value of the
#'   variable; `fsplit` adds the full sample first.
#' @param cluster,se Former fixest ways of requesting a variance type (`cluster = ~g`,
#'   `se = "hetero"`).
#' @param ssc Small-sample correction, see [ssc()].
#' @param panel.id Panel identifiers, as in fixest: a formula `~unit + time` or a string
#'   `"unit,time"`. They are the default unit and time of Newey-West / Driscoll-Kraay variances
#'   and the default cluster of `vcov = "cluster"`. (fereg has no lag operators.)
#' @param fixef.rm `"perfect_fit"` (default; singletons, plus fixed-effect groups with a
#'   perfectly fitted outcome in GLMs), `"singleton"` or `"none"`.
#' @param fixef.tol,fixef.iter Tolerance and iteration cap of the fixed-effects solver.
#' @param collin.tol Collinearity tolerance (fixest rule).
#' @param nthreads Number of threads; `0` = all logical cores. Defaults to option
#'   `fereg.nthreads`, else all cores, capped at `OMP_THREAD_LIMIT` when set (2 under R CMD
#'   check, i.e. when `_R_CHECK_LIMIT_CORES_` is set).
#' @param lean If `TRUE`, residuals, fitted values and the estimation state are not kept
#'   (post-hoc variances then need a new estimation).
#' @param notes Print notes about removed observations (option `fereg.notes`, default `TRUE`).
#' @param ... Not used; unknown arguments are an error.
#' @return An object of class `fereg` (or `fereg_multi` for multiple estimations).
#' @examples
#' data(trade_toy, package = "fereg")
#' m <- feols(y ~ x1 + x2 | id + year, trade_toy, nthreads = 2)
#' m
#' summary(m, vcov = ~id)
#' @export
feols <- function(fml, data, vcov, weights, offset, subset, split, fsplit, cluster, se, ssc,
                  panel.id, fixef.rm = "perfect_fit", fixef.tol = 1e-8, fixef.iter = 16000,
                  collin.tol = 1e-9, nthreads = getOption("fereg.nthreads", default_threads()),
                  lean = FALSE, notes = getOption("fereg.notes", TRUE), ...) {
  check_dots(...)
  args <- estimation_args(
    method = "feols", family = NULL, data = data, env = parent.frame(), call = match.call(),
    vcov = arg_or_null(vcov), weights = arg_or_null(weights), offset = arg_or_null(offset),
    subset = arg_or_null(subset), split = arg_or_null(split), fsplit = arg_or_null(fsplit),
    cluster = arg_or_null(cluster), se = arg_or_null(se), ssc = arg_or_null(ssc), panel.id = arg_or_null(panel.id),
    fixef.rm = fixef.rm, fixef.tol = fixef.tol, fixef.iter = fixef.iter, collin.tol = collin.tol,
    glm.tol = 1e-8, glm.iter = 25, nthreads = nthreads, lean = lean, notes = notes
  )
  estimate(fml, args)
}

#' Fixed-effects Poisson and logit estimation
#'
#' Drop-in replacements for [fixest::fepois()] and [fixest::feglm()] (Poisson and logit
#' families).
#'
#' @inheritParams feols
#' @param family `"poisson"` / `poisson()` or `"logit"` / `"binomial"` / `binomial()`.
#' @param glm.tol,glm.iter IRLS convergence tolerance (relative deviance change) and iteration
#'   cap.
#' @return An object of class `fereg` (or `fereg_multi`).
#' @examples
#' data(trade_toy, package = "fereg")
#' p <- fepois(count ~ x1 | id + year, trade_toy, nthreads = 2)
#' p
#' @export
fepois <- function(fml, data, vcov, offset, weights, subset, split, fsplit, cluster, se, ssc,
                   panel.id, fixef.rm = "perfect_fit", fixef.tol = 1e-8, fixef.iter = 16000,
                   glm.iter = 25, glm.tol = 1e-8, collin.tol = 1e-9,
                   nthreads = getOption("fereg.nthreads", default_threads()), lean = FALSE,
                   notes = getOption("fereg.notes", TRUE), ...) {
  check_dots(...)
  args <- estimation_args(
    method = "fepois", family = "poisson", data = data, env = parent.frame(), call = match.call(),
    vcov = arg_or_null(vcov), weights = arg_or_null(weights), offset = arg_or_null(offset),
    subset = arg_or_null(subset), split = arg_or_null(split), fsplit = arg_or_null(fsplit),
    cluster = arg_or_null(cluster), se = arg_or_null(se), ssc = arg_or_null(ssc), panel.id = arg_or_null(panel.id),
    fixef.rm = fixef.rm, fixef.tol = fixef.tol, fixef.iter = fixef.iter, collin.tol = collin.tol,
    glm.tol = glm.tol, glm.iter = glm.iter, nthreads = nthreads, lean = lean, notes = notes
  )
  estimate(fml, args)
}

#' @rdname fepois
#' @export
feglm <- function(fml, data, family = "logit", vcov, offset, weights, subset, split, fsplit,
                  cluster, se, ssc, panel.id, fixef.rm = "perfect_fit", fixef.tol = 1e-8,
                  fixef.iter = 16000, glm.iter = 25, glm.tol = 1e-8, collin.tol = 1e-9,
                  nthreads = getOption("fereg.nthreads", default_threads()), lean = FALSE,
                  notes = getOption("fereg.notes", TRUE), ...) {
  check_dots(...)
  fam <- glm_family(family)
  args <- estimation_args(
    method = if (fam == "poisson") "fepois" else "feglm", family = fam, data = data,
    env = parent.frame(), call = match.call(),
    vcov = arg_or_null(vcov), weights = arg_or_null(weights), offset = arg_or_null(offset),
    subset = arg_or_null(subset), split = arg_or_null(split), fsplit = arg_or_null(fsplit),
    cluster = arg_or_null(cluster), se = arg_or_null(se), ssc = arg_or_null(ssc), panel.id = arg_or_null(panel.id),
    fixef.rm = fixef.rm, fixef.tol = fixef.tol, fixef.iter = fixef.iter, collin.tol = collin.tol,
    glm.tol = glm.tol, glm.iter = glm.iter, nthreads = nthreads, lean = lean, notes = notes
  )
  if (fam == "poisson") args$method <- "feglm"
  estimate(fml, args)
}

# ---- helpers ---------------------------------------------------------------------------------

# Default thread count: all cores (0), except 2 under R CMD check (`_R_CHECK_LIMIT_CORES_` set and
# not "false", as CRAN's policy asks), and at most OMP_THREAD_LIMIT when that is set.
default_threads <- function() {
  check <- Sys.getenv("_R_CHECK_LIMIT_CORES_")
  n <- if (nzchar(check) && !identical(tolower(check), "false")) 2L else 0L
  limit <- suppressWarnings(as.integer(Sys.getenv("OMP_THREAD_LIMIT")))
  if (!is.na(limit) && limit > 0L) n <- if (n == 0L) limit else min(n, limit)
  n
}

check_dots <- function(...) {
  if (...length()) {
    nm <- names(list(...))
    nm <- if (is.null(nm)) "" else nm[nzchar(nm)]
    stop("fereg: unknown argument", if (length(nm) != 1) "s", " ", paste0("`", nm, "`", collapse = ", "), call. = FALSE)
  }
}

# Value of a possibly missing argument of the calling estimation function.
arg_or_null <- function(x) if (missing(x)) NULL else x

glm_family <- function(family) {
  if (inherits(family, "family")) {
    if (family$family == "poisson" && family$link == "log") return("poisson")
    if (family$family == "binomial" && family$link == "logit") return("logit")
    stop("fereg: family ", family$family, "(link = \"", family$link, "\") is not supported; use poisson() or binomial()", call. = FALSE)
  }
  if (is.function(family)) return(glm_family(family()))
  if (is.character(family) && length(family) == 1) {
    fam <- pmatch(family, c("poisson", "logit", "binomial"))
    if (!is.na(fam)) return(c("poisson", "logit", "logit")[fam])
  }
  stop("fereg: feglm() supports the poisson and logit (binomial) families; use feols() for linear models", call. = FALSE)
}

estimation_args <- function(method, family, data, env, call, vcov, weights, offset, subset,
                            split, fsplit, cluster, se, ssc, panel.id, fixef.rm, fixef.tol, fixef.iter,
                            collin.tol, glm.tol, glm.iter, nthreads, lean, notes) {
  if (!is.list(data)) stop("fereg: `data` must be a data frame", call. = FALSE)
  if (!is.null(ssc) && !inherits(ssc, "ssc.type")) stop("fereg: `ssc` must be created with ssc()", call. = FALSE)
  fixef.rm <- match.arg(fixef.rm, c("perfect_fit", "singleton", "singletons", "none", "perfect"))
  fixef.rm <- switch(fixef.rm, singletons = "singleton", perfect = "perfect_fit", fixef.rm)
  num <- function(x, name, lower) {
    if (!is.numeric(x) || length(x) != 1 || is.na(x) || x < lower) stop("fereg: invalid `", name, "`", call. = FALSE)
    as.double(x)
  }
  if (!is.numeric(nthreads) || length(nthreads) != 1 || is.na(nthreads) || nthreads < 0) {
    stop("fereg: `nthreads` must be a non-negative integer (0 = all cores)", call. = FALSE)
  }
  list(
    method = method, family = family, data = data, env = env, call = call,
    vcov = parse_vcov(vcov, cluster, se), ssc = ssc, panel_id = parse_panel_id(panel.id, data),
    weights = weights, offset = offset, subset = subset, split = split, fsplit = fsplit,
    fixef.rm = fixef.rm, fixef.tol = num(fixef.tol, "fixef.tol", 0), fixef.iter = num(fixef.iter, "fixef.iter", 1),
    collin.tol = num(collin.tol, "collin.tol", 0), glm.tol = num(glm.tol, "glm.tol", 0),
    glm.iter = num(glm.iter, "glm.iter", 1), nthreads = as.integer(nthreads),
    lean = isTRUE(lean), notes = isTRUE(notes)
  )
}

data_nrow <- function(data) {
  if (is.data.frame(data)) return(nrow(data))
  n <- unique(lengths(data))
  if (length(n) != 1) stop("fereg: the columns of `data` must have equal lengths", call. = FALSE)
  n
}

# A per-row vector argument: one-sided formula, column name, or vector.
row_vector <- function(x, data, env, n, what) {
  if (is.null(x)) return(NULL)
  v <- if (inherits(x, "formula")) {
    eval(x[[length(x)]], data, environment(x))
  } else if (is.character(x) && length(x) == 1 && n != 1) {
    if (is.null(data[[x]])) stop("fereg: `", what, "` column '", x, "' not found", call. = FALSE)
    data[[x]]
  } else {
    x
  }
  if (length(v) != n) stop("fereg: `", what, "` must have one value per row of `data`", call. = FALSE)
  v
}

# Rows to drop because of `subset` (TRUE = drop) or NULL.
subset_drop <- function(subset, data, env, n) {
  if (is.null(subset)) return(NULL)
  if (inherits(subset, "formula")) subset <- eval(subset[[length(subset)]], data, environment(subset))
  if (is.logical(subset)) {
    if (length(subset) != n) stop("fereg: a logical `subset` must have one value per row", call. = FALSE)
    return(is.na(subset) | !subset)
  }
  if (is.numeric(subset)) {
    keep <- rep(FALSE, n)
    keep[subset] <- TRUE
    return(!keep)
  }
  stop("fereg: `subset` must be logical, integer or a one-sided formula", call. = FALSE)
}

# Split samples: list of list(label, drop).
split_samples <- function(args, n) {
  sp <- if (!is.null(args$fsplit)) args$fsplit else args$split
  if (is.null(sp)) return(list(list(label = NULL, drop = NULL)))
  var_name <- if (inherits(sp, "formula")) deparse1l(sp[[length(sp)]]) else if (is.character(sp) && length(sp) == 1) sp else "split"
  v <- row_vector(sp, args$data, args$env, n, "split")
  vals <- if (is.factor(v)) levels(droplevels(v[!is.na(v)])) else sort(unique(v[!is.na(v)]))
  out <- lapply(vals, function(val) list(label = list(var = var_name, value = as.character(val)), drop = is.na(v) | v != val))
  if (!is.null(args$fsplit)) out <- c(list(list(label = list(var = var_name, value = "Full sample"), drop = NULL)), out)
  out
}

combine_drop <- function(a, b) if (is.null(a)) b else if (is.null(b)) a else a | b

# ---- estimation driver ---------------------------------------------------------------------------

estimate <- function(fml, args) {
  models <- parse_fml(fml)
  data <- args$data
  n <- data_nrow(data)
  env <- environment(fml)
  w <- row_vector(args$weights, data, env, n, "weights")
  if (!is.null(w)) w <- as_num(w, "weights")
  off <- row_vector(args$offset, data, env, n, "offset")
  if (!is.null(off)) {
    if (args$method == "feols") stop("fereg: `offset` is only available for fepois() and feglm()", call. = FALSE)
    off <- as_num(off, "offset")
  }
  base_drop <- subset_drop(args$subset, data, env, n)
  samples <- split_samples(args, n)
  cache <- new.env(parent = emptyenv())

  results <- list()
  ids <- list()
  for (s in samples) {
    for (m in models) {
      drop <- combine_drop(base_drop, s$drop)
      res <- fit_one(m, args, data, env, n, w, off, drop, cache)
      res$model_info$sample <- s$label
      results[[length(results) + 1]] <- res
      sample_id <- if (!is.null(s$label)) c(sample.var = s$label$var, sample = s$label$value)
      ids[[length(ids) + 1]] <- c(sample_id, m$id)
    }
  }
  if (length(results) == 1) return(results[[1]])
  new_multi(results, ids)
}

# Evaluates `expr` once per estimation call (multiple estimations share columns).
cached <- function(cache, key, make) {
  if (!exists(key, envir = cache, inherits = FALSE)) assign(key, make(), envir = cache)
  get(key, envir = cache, inherits = FALSE)
}

fit_one <- function(m, args, data, env, n, w, off, drop, cache) {
  model <- if (is.null(args$family)) "ols" else args$family
  is_iv <- length(m$endo) > 0
  if (is_iv && model != "ols") stop("fereg: IV is only available in feols()", call. = FALSE)

  # Fixed effects and cluster factors (registry indices are per model).
  reg <- new_factors(data, env, cache)
  slope_reg <- new_slopes(data, env, cache)
  fe <- fe_design(m$fe_terms, reg, slope_reg)
  has_fe <- length(fe$terms) > 0
  v <- resolve_vcov(args$vcov, fe$terms, args$panel_id, names(data))
  vc <- vcov_columns(v, reg)

  y_name <- deparse1l(m$lhs)
  y <- cached(cache, paste0("y:", y_name), function() as_num(eval_var(m$lhs, data, env), y_name))
  lin <- linear_design(m$rhs_terms, data, env, has_fe)
  endo_names <- vapply(m$endo, deparse1l, "")
  endo <- lapply(seq_along(m$endo), function(j) {
    cached(cache, paste0("x:", endo_names[j]), function() as_num(eval_var(m$endo[[j]], data, env), endo_names[j]))
  })
  # Instruments use the right-hand side rules (i(), factor(), ...) with intercept coding, as the
  # first stage has an intercept or fixed effects.
  inst <- if (is_iv) linear_design(lapply(m$inst, function(e) list(expr = e, sign = 1)), data, env, TRUE)

  ssc_used <- effective_ssc(v$ssc %||% args$ssc, v$type)
  input <- list(
    n = n, y = y, x = lin$cols, endog = endo, inst = if (is_iv) inst$cols else list(),
    factors = reg$cols, factor_names = reg$names, fe = fe$core,
    slopes = slope_reg$cols, weights = w, offset = off, cluster = vc$cluster,
    time = vc$time, panel = vc$panel, lat = vc$lat, lon = vc$lon, drop = drop
  )
  spec <- list(
    model = model, intercept = lin$intercept, vcov = core_kind(v), ssc = ssc_core(ssc_used),
    fixef_rm = args$fixef.rm, collin_tol = args$collin.tol, fixef_tol = args$fixef.tol,
    fixef_maxiter = args$fixef.iter, glm_tol = args$glm.tol, glm_maxiter = args$glm.iter,
    keep_resid = !args$lean, keep_fixef = FALSE, keep_state = !args$lean
  )
  out <- fereg_fit(input, spec, args$nthreads)

  info <- list(
    model = model, args = args, m = m, n = n, env = env, y = y, y_name = y_name, lin = lin,
    x_names = lin$names, endo_names = endo_names, inst_names = if (is_iv) inst$names else character(0),
    fe = fe, reg = reg, slope_names = slope_reg$names,
    vcov = v, ssc = ssc_used, weights = w, offset = off, drop = drop
  )
  build_result(out, info)
}

# ---- result object -----------------------------------------------------------------------------

source_names <- function(src, info) {
  vapply(seq_along(src$type), function(k) {
    j <- src$index[k]
    switch(src$type[k],
      intercept = "(Intercept)",
      x = info$x_names[j],
      endog = paste0("fit_", info$endo_names[j]),
      inst = info$inst_names[j]
    )
  }, "")
}

# fixest coefficient order: intercept, endogenous regressors, exogenous regressors.
coef_perm <- function(src) {
  c(which(src$type == "intercept"), which(src$type == "endog"), which(src$type %in% c("x", "inst")))
}

nan_na <- function(x) {
  x[is.nan(x)] <- NA
  x
}

# Coefficient table (fixest layout: t or z statistics).
make_coeftable <- function(coef, V, vinfo) {
  se <- sqrt(pmax(diag(V), 0))
  stat <- coef / se
  if (isTRUE(vinfo$t_dist)) {
    p <- 2 * stats::pt(-abs(stat), vinfo$df_t)
    lab <- c("t value", "Pr(>|t|)")
  } else {
    p <- 2 * stats::pnorm(-abs(stat))
    lab <- c("z value", "Pr(>|z|)")
  }
  ct <- cbind(coef, se, stat, p)
  dimnames(ct) <- list(names(coef), c("Estimate", "Std. Error", lab))
  ct
}

vcov_matrix <- function(vout, perm, names) {
  V <- vout$vcov[perm, perm, drop = FALSE]
  dimnames(V) <- list(names, names)
  V
}

set_vcov <- function(res, vout, v, ssc_used) {
  V <- vcov_matrix(vout, res$coef_perm, names(res$coefficients))
  vinfo <- list(
    type = v$type, dims = v$dims, label = vcov_label(v, vout), ssc = ssc_used,
    df_k = vout$df_k, df_t = nan_na(vout$df_t), t_dist = vout$t_dist,
    n_clusters = stats::setNames(vout$n_clusters, vapply(v$dims, dim_label, "")),
    n_periods = nan_na(vout$n_periods), psd_fixed = vout$psd_fixed, request = v
  )
  attr(V, "type") <- vinfo$label
  attr(V, "df.K") <- vinfo$df_k
  attr(V, "df.t") <- vinfo$df_t
  # fixest's attribute G: clusters per dimension; Newey-West / Driscoll-Kraay: time periods.
  if (v$type %in% c("cluster", "crv3")) attr(V, "G") <- vinfo$n_clusters
  if (v$type %in% c("nw", "dk")) attr(V, "G") <- vinfo$n_periods
  apply_vcov(res, V, vinfo)
}

apply_vcov <- function(res, V, vinfo) {
  res$cov.scaled <- V
  res$vcov_info <- vinfo
  res$coeftable <- make_coeftable(res$coefficients, V, vinfo)
  attr(res$coeftable, "vcov_type") <- vinfo$label
  res$se <- res$coeftable[, 2]
  res
}

build_result <- function(out, info) {
  args <- info$args
  src <- out$coef_source
  perm <- coef_perm(src)
  coef <- stats::setNames(out$coef, source_names(src, info))[perm]
  collin <- source_names(out$collinear, info)

  fe_terms <- info$fe$terms
  fe_int <- vapply(fe_terms, `[[`, TRUE, "intercept")
  fe_labels_all <- vapply(fe_terms, `[[`, "", "label")
  fe_sizes <- stats::setNames(as.integer(out$fe_sizes), fe_labels_all)
  st <- lapply(out$stats, function(x) if (is.numeric(x)) nan_na(x) else x)

  res <- list(
    nobs = as.integer(out$nobs),
    nobs_origin = as.integer(out$nobs_origin),
    sumw = out$sumw,
    fml = info$m$fml,
    fml_all = list(linear = info$m$fml, fixef = info$m$fml_fixef, iv = info$m$fml_iv),
    call = args$call,
    call_env = args$env,
    method = args$method,
    method_type = if (info$model == "ols") "feols" else "feglm",
    family = if (info$model == "poisson") stats::poisson() else if (info$model == "logit") stats::binomial(),
    nparams = as.integer(out$nparams),
    coefficients = coef,
    coef_perm = perm,
    collin.var = if (length(collin)) collin,
    fixef_vars = if (any(fe_int)) fe_labels_all[fe_int],
    fixef_sizes = if (length(fe_terms)) fe_sizes,
    fixef_terms = fe_terms,
    slope_names = info$slope_names,
    obs_selection = out$obs,
    dropped = out$dropped,
    ssr = st$ssr, ssr_null = st$ssr_null, ssr_fe_only = st$ssr_fe_only, sigma2 = st$sigma2,
    rmse = st$rmse, r2_all = c(r2 = st$r2, ar2 = st$ar2, wr2 = st$wr2, war2 = st$war2),
    loglik = st$loglik, ll_null = st$ll_null, deviance = st$deviance,
    null_deviance = st$null_deviance, pseudo_r2 = st$pseudo_r2, apr2 = st$apr2,
    sq.cor = st$sq_cor, f_test = st$f_test,
    iterations = if (info$model == "ols") st$fe_iterations else st$irls_iterations,
    convStatus = isTRUE(st$converged),
    residuals = out$resid,
    fitted.values = out$fitted,
    linear.predictors = out$eta,
    irls_weights = out$irls_weights,
    weights = if (!is.null(info$weights)) obs_subset(info$weights, out$obs),
    # Fixed-effect values are computed on first use from `state` (fixef(), predict()) and
    # cached in `fixef_cache`; `fe_columns` holds the FE variables of every term to label them.
    fe_columns = lapply(fe_terms, function(t) {
      lapply(t$vars, function(e) info$reg$values[[match(deparse1l(e), info$reg$names)]])
    }),
    fixef_cache = new.env(parent = emptyenv()),
    design = list(
      lin_meta = info$lin$meta, y_name = info$y_name, endo_exprs = info$m$endo,
      endo_names = info$endo_names, inst_names = info$inst_names,
      offset = args$offset, has_offset = !is.null(info$offset)
    ),
    model_info = list(),
    panel.id = args$panel_id,
    # What a re-estimation needs (CRV3 after estimation): the model, the arguments without the
    # data (found again from the call), the formula environment and the dropped rows.
    refit = list(m = info$m, args = args[setdiff(names(args), "data")], env = info$env, drop = info$drop),
    state = out$state,
    lean = args$lean
  )
  if (!is.null(info$weights)) res$model_info$weights <- weights_label(args$weights)
  if (!is.null(info$offset)) res$model_info$offset <- weights_label(args$offset)
  class(res) <- "fereg"
  res <- set_vcov(res, out$vcov, info$vcov, info$ssc)

  if (!is.null(out$iv)) res <- add_iv(res, out$iv, info)
  if (args$notes) estimation_notes(res, out)
  for (nt in out$notes) warning(nt, call. = FALSE)
  res
}

obs_subset <- function(v, obs) if (is.null(obs)) v else v[obs]

weights_label <- function(w) {
  if (inherits(w, "formula")) deparse1l(w[[length(w)]]) else if (is.character(w) && length(w) == 1) w else "(vector)"
}

add_iv <- function(res, iv, info) {
  res$is_iv <- TRUE
  res$iv_stage <- 2L
  res$iv_endo_names <- info$endo_names
  res$iv_inst_names <- info$inst_names
  res$iv_wh <- nan_na(iv$wu_hausman)
  res$iv_sargan <- nan_na(iv$sargan)
  res$ssr_iv <- iv$ssr_iv
  res$iv_first_stage <- stats::setNames(lapply(iv$first_stage, function(fs) {
    perm <- coef_perm(fs$coef_source)
    coef <- stats::setNames(fs$coef, source_names(fs$coef_source, info))[perm]
    first <- list(
      nobs = res$nobs, nobs_origin = res$nobs_origin, method = "feols", method_type = "feols",
      is_iv = TRUE, iv_stage = 1L, coefficients = coef, coef_perm = perm,
      collin.var = if (length(fs$collinear$type)) source_names(fs$collinear, info),
      fixef_vars = res$fixef_vars, fixef_sizes = res$fixef_sizes, fixef_terms = res$fixef_terms,
      fml = stats::as.formula(call("~", info$m$endo[[fs$endog]], 1)),
      iv_endo_names = info$endo_names, iv_inst_names = info$inst_names,
      iv_dep_var = info$y_name, iv_f_test = nan_na(fs$f_test), iv_wald = nan_na(fs$wald),
      model_info = res$model_info
    )
    class(first) <- "fereg"
    set_vcov(first, fs$vcov, info$vcov, info$ssc)
  }), info$endo_names)
  res
}

estimation_notes <- function(res, out) {
  d <- out$dropped
  msgs <- character(0)
  if (d$na > 0) {
    by <- d$na_by
    lab <- c(y = "LHS", x = "RHS", endog = "Endo.", inst = "Instr.", fe = "Fixed-effects",
      weights = "Weights", offset = "Offset", cluster = "vcov", extra = "subset")
    keep <- by > 0 & names(by) != "extra"
    detail <- paste0(lab[names(by)[keep]], ": ", by[keep], collapse = ", ")
    msgs <- c(msgs, sprintf("%s removed because of NA values (%s).", plural(d$na, "observation"), detail))
  }
  if (d$zero_weight > 0) msgs <- c(msgs, sprintf("%s removed because of zero weights.", plural(d$zero_weight, "observation")))
  if (d$singletons > 0) msgs <- c(msgs, sprintf("%s removed because of fixed-effect singletons.", plural(d$singletons, "observation")))
  if (d$perfect_fit > 0) {
    msgs <- c(msgs, sprintf("%s removed because of only %s outcomes in their fixed-effect group.",
      plural(d$perfect_fit, "observation"), if (identical(res$family$family, "poisson")) "0" else "0 (or only 1)"))
  }
  if (length(res$collin.var)) {
    k <- length(res$collin.var)
    msgs <- c(msgs, sprintf("The variable%s %s ha%s been removed because of collinearity (see $collin.var).",
      if (k > 1) "s" else "", enum_quote(res$collin.var), if (k > 1) "ve" else "s"))
  }
  if (length(msgs)) {
    head <- if (length(msgs) > 1) "NOTES: " else "NOTE: "
    message(paste0(c(head, rep("       ", length(msgs) - 1)), msgs, collapse = "\n"))
  }
}

plural <- function(k, word) paste0(format_int(k), " ", word, if (k != 1) "s")

enum_quote <- function(x) {
  q <- paste0("'", x, "'")
  if (length(q) == 1) return(q)
  paste0(paste(q[-length(q)], collapse = ", "), " and ", q[length(q)])
}

format_int <- function(x) formatC(x, format = "d", big.mark = ",")
