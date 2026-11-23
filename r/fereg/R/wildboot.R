# Wild bootstrap inference (fwildclusterboot-compatible interface for feols / feglm / fepois).

#' Wild bootstrap inference
#'
#' Fast wild bootstrap tests, p-values and confidence intervals by CI inversion for estimations of
#' [feols()], [fepois()] and [feglm()]. Covers clustered OLS (WCR / WCU, including
#' MacKinnon--Nielsen--Webb `13`, `31`, `33` jackknife types and exact non-nested HDFE projection),
#' unclustered heteroskedastic OLS (`11`, `21`, `31`, `13`, `33`), IV/2SLS (Davidson--MacKinnon
#' WRE and score bootstrap) and Poisson / logit GLMs (wild score bootstrap).
#'
#' @param object A single `fereg` estimation returned by [feols()], [fepois()] or [feglm()] (with
#'   its estimation data still available). Bias-corrected models (`biasCorr()`) and IV first stages
#'   are not supported.
#' @param param Character vector naming the coefficient(s) to test (e.g. `"x1"` or
#'   `c("x1", "x2")`), mutually exclusive with `R`. For IV models, endogenous regressors are named
#'   `"fit_<var>"`, as in `coef(object)`.
#' @param B Number of bootstrap replications (positive integer, default `999L`). For clustered
#'   Rademacher (`G <= 20`, `2^G <= B`) or Webb (`G <= 8`, `6^G <= B`), full enumeration is used
#'   automatically.
#' @param clustid Optional clustering specification overriding `object`'s variance: a one-sided
#'   formula (`~firm`, `~firm + year`, `~firm^year`), character vector (`"firm"`, `c("firm",
#'   "year")`), or `character(0)` for unclustered inference. Defaults to `object`'s current
#'   clustering (after any post-hoc [summary()][fereg-methods] / [vcov()][fereg-methods] choice).
#' @param bootcluster Optional bootstrap subcluster dimension overriding the cluster dimension used
#'   to generate the wild bootstrap weights (same forms as `clustid`; e.g. `~firm` when errors are
#'   two-way clustered by `~firm + year`).
#' @param R Optional numeric constraint matrix (`q x k`) or vector (`k`) for the linear hypothesis
#'   `H0: R * beta = r`, columns ordered as `names(coef(object))`, mutually exclusive with `param`.
#' @param r Optional numeric right-hand side vector of length `q` (default zero vector).
#' @param type Auxiliary weight distribution: `"rademacher"` (default), `"webb"`, `"mammen"` or
#'   `"normal"`.
#' @param bootstrap_type Two-digit bootstrap variant (`"11"` default; MacKinnon--Nielsen--Webb
#'   `"13"`, `"31"`, `"33"`, or `"21"` for unclustered HC2). Also accepts `"fnw11"` as an alias
#'   for `"11"`.
#' @param method Algorithm: `"auto"` (default; selects `"wildcluster"` for clustered OLS,
#'   `"hetero"` for unclustered OLS, `"wre"` for IV with `bootstrap_type = "11"`, and `"score"`
#'   for GLMs or other IV variants), `"wildcluster"`, `"score"`, `"wre"`, or `"hetero"`.
#' @param impose_null Logical; if `TRUE` (default) imposes `H0` in the bootstrap DGP (restricted),
#'   else unrestricted.
#' @param p_val_type P-value type: `"two-tailed"` (default), `"equal-tailed"`, `">"` (`"greater"`),
#'   or `"<"` (`"less"`). Joint tests (`q > 1`) only support `"two-tailed"`.
#' @param conf_int Logical; compute confidence interval for scalar tests (`q == 1`, default `TRUE`).
#' @param sign_level Significance level for confidence intervals (default `0.05` for a 95% CI).
#' @param seed Optional non-negative integer seed for reproducible bootstrap draws.
#' @param ssc Optional small-sample correction built with [ssc()]; defaults to the model's `ssc()`.
#' @param tol Numerical tolerance for confidence interval inversion root-finding (default `1e-6`).
#' @param maxiter Maximum iterations per CI endpoint search (default `50L`).
#' @param nthreads Number of threads (default `getOption("fereg.nthreads", ...)`).
#' @param ... Unused; for S3 method compatibility.
#' @return An object of class `"fereg_boottest"` containing `statistic`, `stat_type`, `p_val`,
#'   `point_estimate`, `se`, `conf_int`, `t_boot`, `reps`, `reps_feasible`, `nobs`, `n_clusters`,
#'   `q`, `full_enumeration`, `method`, `bootstrap_type`, `type`, `impose_null`, `p_val_type`,
#'   `ssc`, `sign_level`, `param`, `clustid`, `bootcluster`, `seed`, `fml`, and `call`. Methods:
#'   `print()`, `summary()`, `confint()`, `pvalue()`, `tstat()`.
#' @examples
#' data(trade_toy, package = "fereg")
#' m <- feols(y ~ x1 + x2 | id + year, trade_toy, cluster = ~id, nthreads = 2)
#' bt <- boottest(m, param = "x1", B = 99L, seed = 1, nthreads = 2)
#' summary(bt)
#' confint(bt)
#' @export
boottest <- function(object, ...) UseMethod("boottest")

#' @rdname boottest
#' @export
boottest.fereg_multi <- function(object, ...) {
  stop("fereg: boottest() works on a single model; index the fereg_multi object first (e.g., object[[1]])", call. = FALSE)
}

#' @rdname boottest
#' @export
boottest.fereg <- function(
  object,
  param = NULL,
  B = 999L,
  clustid = NULL,
  bootcluster = NULL,
  R = NULL,
  r = NULL,
  type = c("rademacher", "webb", "mammen", "normal"),
  bootstrap_type = c("11", "13", "31", "33", "21"),
  method = c("auto", "wildcluster", "score", "wre", "hetero"),
  impose_null = TRUE,
  p_val_type = c("two-tailed", "equal-tailed", ">", "<"),
  conf_int = TRUE,
  sign_level = 0.05,
  seed = NULL,
  ssc = NULL,
  tol = 1e-6,
  maxiter = 50L,
  nthreads = NULL,
  ...
) {
  if (!inherits(object, "fereg")) stop("fereg: `object` must be a fitted fereg model", call. = FALSE)
  if (inherits(object, "fereg_biascorr")) stop("fereg: boottest() is not available after bias correction", call. = FALSE)
  if (isTRUE(object$iv_stage == 1)) stop("fereg: boottest() is not available for IV first stages", call. = FALSE)

  type <- match.arg(type)
  if (is.character(bootstrap_type) && length(bootstrap_type) == 1L && identical(bootstrap_type, "fnw11")) {
    bootstrap_type <- "11"
  }
  bootstrap_type <- match.arg(bootstrap_type)
  method <- match.arg(method)
  if (is.character(p_val_type) && length(p_val_type) == 1L && p_val_type %in% c("greater", "less")) {
    p_val_type <- c(greater = ">", less = "<")[[p_val_type]]
  }
  p_val_type <- match.arg(p_val_type)

  B <- check_wb_int(B, "B", min_val = 1L)
  impose_null <- check_wb_flag(impose_null, "impose_null")
  conf_int <- check_wb_flag(conf_int, "conf_int")
  if (!is.numeric(sign_level) || length(sign_level) != 1L || !is.finite(sign_level) || sign_level <= 0 || sign_level >= 1) {
    stop("fereg: `sign_level` must be a single number strictly between 0 and 1", call. = FALSE)
  }
  sign_level <- as.double(sign_level)
  if (!is.numeric(tol) || length(tol) != 1L || !is.finite(tol) || tol <= 0) {
    stop("fereg: `tol` must be a positive finite number", call. = FALSE)
  }
  tol <- as.double(tol)
  maxiter <- check_wb_int(maxiter, "maxiter", min_val = 1L)

  seed_num <- if (is.null(seed)) {
    as.double(sample.int(.Machine$integer.max, 1L))
  } else {
    if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed) || seed < 0 || seed != floor(seed) || seed > 9007199254740992) {
      stop("fereg: `seed` must be a single non-negative integer", call. = FALSE)
    }
    as.double(seed)
  }
  nthreads <- if (is.null(nthreads)) getOption("fereg.nthreads", default_threads()) else check_wb_int(nthreads, "nthreads", min_val = 0L)

  hyp <- build_wb_hypothesis(object, param, R, r)

  if (!is.null(clustid)) {
    if (is.character(clustid) && length(clustid) == 0L) {
      v_eff <- new_vcov("hetero")
    } else {
      v_eff <- parse_wb_cluster_arg(clustid, "clustid")
    }
  } else if (object$vcov_info$type %in% c("cluster", "crv3")) {
    v_eff <- object$vcov_info$request
    v_eff$type <- "cluster"
    v_eff$ssc <- NULL
  } else if (object$vcov_info$type %in% c("nw", "dk", "conley")) {
    stop(
      "fereg: boottest() is not defined for '", object$vcov_info$label,
      "' VCOV; pass `clustid` explicitly or use iid/hetero/clustered VCOV",
      call. = FALSE
    )
  } else {
    v_eff <- new_vcov("hetero")
  }

  ssc_base <- if (!is.null(ssc)) ssc else if (is.null(clustid)) object$vcov_info$ssc else object$refit$args$ssc
  ssc_eff <- effective_ssc(ssc_base, v_eff$type)

  data <- fetch_data(object, why = "boottest() needs the estimation data")
  s <- refit_setup(object, list(vcov = v_eff, ssc = ssc_eff, lean = TRUE, notes = FALSE), data = data)

  bc_idx <- NULL
  bc_labels <- NULL
  if (!is.null(bootcluster)) {
    if (is.character(bootcluster) && length(bootcluster) == 0L) {
      stop("fereg: `bootcluster` must specify at least one cluster variable", call. = FALSE)
    }
    bc_req <- resolve_vcov(parse_wb_cluster_arg(bootcluster, "bootcluster"), s$info$fe$terms, object$panel.id, names(data))
    bc_exprs <- unlist(bc_req$dims, recursive = FALSE)
    bc_idx <- unique(as.integer(vapply(bc_exprs, function(e) factor_index(s$info$reg, e), 0)))
    s$input$factors <- s$info$reg$cols
    s$input$factor_names <- s$info$reg$names
    bc_labels <- vapply(bc_req$dims, dim_label, "")
  }

  wb <- list(
    r_mat = as.double(t(hyp$R_core)),
    r_vec = as.double(hyp$r_vec),
    q = nrow(hyp$R_core),
    k = ncol(hyp$R_core),
    reps = B,
    weights = type,
    boot_type = bootstrap_type,
    method = method,
    impose_null = impose_null,
    pval_type = p_val_type,
    conf_int = conf_int,
    level = 1 - sign_level,
    seed = seed_num,
    tol = tol,
    maxiter = maxiter,
    bootcluster = bc_idx
  )

  out <- fereg_wildboot(s$input, s$spec, wb, nthreads)
  for (nt in out$notes) warning(nt, call. = FALSE)

  ret_names <- source_names(out$coef_source, s$info)[coef_perm(out$coef_source)]
  if (!identical(ret_names, names(object$coefficients))) {
    stop("fereg: wild bootstrap sample / collinearity changed estimable coefficients relative to the fitted model", call. = FALSE)
  }

  res <- list(
    statistic = out$stat,
    stat_type = out$stat_type,
    p_val = out$p_val,
    point_estimate = nan_na(out$point_estimate),
    se = nan_na(out$se),
    conf_int = out$conf_int,
    t_boot = out$t_boot,
    reps = as.integer(out$reps),
    reps_feasible = as.integer(out$reps_feasible),
    nobs = as.integer(out$nobs),
    n_clusters = if (is.null(out$n_clusters)) NULL else as.integer(out$n_clusters),
    q = as.integer(out$q),
    full_enumeration = isTRUE(out$full_enumeration),
    method = out$method,
    bootstrap_type = out$boot_type,
    type = out$weights,
    impose_null = isTRUE(out$impose_null),
    p_val_type = out$pval_type,
    ssc = out$ssc,
    sign_level = sign_level,
    param = hyp$label,
    clustid = if (v_eff$type == "cluster") vapply(v_eff$dims, dim_label, "") else NULL,
    bootcluster = bc_labels,
    seed = seed_num,
    fml = object$fml,
    call = match.call()
  )
  class(res) <- "fereg_boottest"
  res
}

check_wb_flag <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) stop("fereg: `", name, "` must be TRUE or FALSE", call. = FALSE)
  isTRUE(x)
}

check_wb_int <- function(x, name, min_val = 0L) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) || x != floor(x) || x < min_val) {
    stop("fereg: `", name, "` must be an integer >= ", min_val, call. = FALSE)
  }
  as.integer(x)
}

parse_wb_cluster_arg <- function(x, name) {
  if (inherits(x, "formula")) {
    if (length(x) != 2L) stop("fereg: `", name, "` must be a one-sided formula or character vector", call. = FALSE)
    return(parse_vcov(cluster = x))
  }
  if (!is.character(x) || anyNA(x) || any(!nzchar(trimws(x)))) {
    stop("fereg: `", name, "` must be variable names or a one-sided formula", call. = FALSE)
  }
  parse_vcov(cluster = x)
}

parse_wb_r <- function(r, q) {
  if (is.null(r)) return(numeric(q))
  if (!is.numeric(r) || any(!is.finite(r))) stop("fereg: `r` must be a finite numeric vector", call. = FALSE)
  r <- as.double(r)
  if (length(r) == 1L && q > 1L) stop("fereg: `r` must have length ", q, call. = FALSE)
  if (length(r) != q) stop("fereg: `r` must have length ", q, call. = FALSE)
  r
}

build_wb_hypothesis <- function(object, param, R, r) {
  if (!is.null(param) && !is.null(R)) stop("fereg: specify either `param` or `R`, not both", call. = FALSE)
  if (is.null(param) && is.null(R)) stop("fereg: specify either `param` or `R`", call. = FALSE)
  cnames <- names(object$coefficients)
  k <- length(cnames)
  if (k == 0L) stop("fereg: model has no estimable parameters", call. = FALSE)
  collin <- object$collin.var %||% character(0)

  if (!is.null(param)) {
    if (!is.character(param) || length(param) == 0L || anyNA(param) || any(!nzchar(param))) {
      stop("fereg: `param` must be a non-empty character vector of coefficient names", call. = FALSE)
    }
    if (anyDuplicated(param)) stop("fereg: `param` contains duplicate coefficient names", call. = FALSE)
    for (p in param) {
      if (p %in% collin) stop("fereg: parameter '", p, "' was removed due to collinearity", call. = FALSE)
      if (!p %in% cnames) stop("fereg: parameter '", p, "' not found in model coefficients", call. = FALSE)
    }
    idx <- match(param, cnames)
    q <- length(idx)
    R_user <- matrix(0, nrow = q, ncol = k)
    for (row in seq_len(q)) R_user[row, idx[row]] <- 1
    r_vec <- parse_wb_r(r, q)
    label <- paste(sprintf("%s = %g", param, r_vec), collapse = ", ")
  } else {
    if (!is.numeric(R)) stop("fereg: `R` must be numeric", call. = FALSE)
    if (is.null(dim(R))) {
      R_user <- matrix(as.double(R), nrow = 1L)
    } else if (length(dim(R)) == 2L) {
      R_user <- matrix(as.double(R), nrow = nrow(R), ncol = ncol(R))
    } else {
      stop("fereg: `R` must be a vector or 2D matrix", call. = FALSE)
    }
    if (nrow(R_user) == 0L || ncol(R_user) != k) {
      stop("fereg: `R` must have ", k, " columns matching coef(object)", call. = FALSE)
    }
    q <- nrow(R_user)
    r_vec <- parse_wb_r(r, q)
    label <- if (q > 1L) sprintf("R * beta = r (q=%d)", q) else "R * beta = r"
  }

  # R presents coefficients ordered by `object$coef_perm` (notably IV: [intercept?, endog..., x...]),
  # while `fereg_core::fit::wildboot` expects core order [intercept?, x..., endog...].
  R_core <- matrix(0, nrow = nrow(R_user), ncol = k)
  R_core[, object$coef_perm] <- R_user
  list(R_core = R_core, r_vec = r_vec, label = label)
}

#' @export
print.fereg_boottest <- function(x, digits = 4L, ...) {
  cl <- if (length(x$clustid)) paste(x$clustid, collapse = " & ") else "unclustered"
  bc <- if (length(x$bootcluster)) paste(x$bootcluster, collapse = " & ") else cl
  enum <- if (x$full_enumeration) " (full enumeration)" else ""
  cat("Wild bootstrap test: H0: ", x$param, "\n", sep = "")
  cat(sprintf(
    "  method = %s, type = %s, weights = %s, impose_null = %s, p_val_type = %s\n",
    x$method, x$bootstrap_type, x$type, if (x$impose_null) "TRUE" else "FALSE", x$p_val_type
  ))
  if (!is.null(x$n_clusters)) {
    cat(sprintf("  clusters = %s (bootcluster = %s, G = %d), nobs = %s\n", cl, bc, x$n_clusters, format_int(x$nobs)))
  } else {
    cat(sprintf("  clusters = %s, nobs = %s\n", cl, format_int(x$nobs)))
  }
  cat(sprintf("  B = %d%s (feasible = %d), seed = %.0f\n", x$reps, enum, x$reps_feasible, x$seed))
  parts <- c(
    sprintf("%s = %s", x$stat_type, format(round(x$statistic, digits), nsmall = min(digits, 4L))),
    sprintf("p = %s", format(signif(x$p_val, digits)))
  )
  if (!is.na(x$point_estimate)) parts <- c(parts, sprintf("estimate = %s", format(round(x$point_estimate, digits))))
  if (!is.na(x$se)) parts <- c(parts, sprintf("se = %s", format(round(x$se, digits))))
  if (!is.null(x$conf_int)) {
    parts <- c(parts, sprintf(
      "%g%% CI = [%s, %s]", (1 - x$sign_level) * 100,
      format(round(x$conf_int[1], digits)), format(round(x$conf_int[2], digits))
    ))
  }
  cat("  ", paste(parts, collapse = ", "), "\n", sep = "")
  invisible(x)
}

#' @export
summary.fereg_boottest <- function(object, digits = 4L, ...) {
  print.fereg_boottest(object, digits = digits, ...)
}

#' @export
confint.fereg_boottest <- function(object, parm, level = NULL, ...) {
  if (!is.null(level) && !isTRUE(all.equal(as.double(level), 1 - object$sign_level))) {
    stop("fereg: bootstrap confidence intervals depend on `sign_level`; re-run boottest(..., sign_level = ", 1 - level, ")", call. = FALSE)
  }
  if (is.null(object$conf_int)) {
    stop("fereg: confidence interval is not available (conf_int = FALSE or joint test q > 1)", call. = FALSE)
  }
  a <- object$sign_level / 2
  out <- data.frame(object$conf_int[1], object$conf_int[2], row.names = object$param)
  names(out) <- paste(format(100 * c(a, 1 - a), trim = TRUE, scientific = FALSE, digits = 3), "%")
  out
}

#' @export
pvalue.fereg_boottest <- function(object, ...) object$p_val

#' @export
tstat.fereg_boottest <- function(object, ...) object$statistic
