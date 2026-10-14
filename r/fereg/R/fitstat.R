# Fit statistics: fitstat() and r2() as in fixest.

fitstat_alias <- c(
  n = "Observations", ll = "Log-Likelihood", aic = "AIC", bic = "BIC", rmse = "RMSE",
  cor2 = "Squared Cor.", r2 = "R2", ar2 = "Adj. R2", pr2 = "Pseudo R2", apr2 = "Adj. Pseudo R2",
  wr2 = "Within R2", war2 = "Within Adj. R2", f = "F-test", ivf1 = "F-test (1st stage)",
  ivwald1 = "Wald (1st stage)", wh = "Wu-Hausman", sargan = "Sargan", my = "Dep. Var. mean",
  g = "G"
)
fitstat_type_alias <- c(sq.cor = "cor2", ivf = "ivf1", ivwald = "ivwald1", ll_null = "ll")

test_list <- function(t) {
  if (is.null(t) || all(is.na(t))) return(NA)
  list(stat = unname(t["stat"]), p = unname(t["p"]), df1 = unname(t["df1"]), df2 = unname(t["df2"]))
}

is_glm <- function(x) identical(x$method_type, "feglm")

#' Fit statistics
#'
#' Same types and names as fixest's `fitstat()`: `n`, `rmse`, `r2`, `ar2`, `wr2`, `war2`, `ll`,
#' `pr2`, `apr2`, `cor2` (`sq.cor`), `aic`, `bic`, `f`, `my`, `g` and, for IV, `ivf1`
#' (`ivf`), `ivwald1` (`ivwald`), `wh` and `sargan`.
#'
#' @param x A `fereg` object.
#' @param type A character vector of types, or a one-sided formula (`~ r2 + ivf1`).
#' @param simplify Return a plain vector when every statistic is a single number.
#' @param ... Not used.
#' @return An object of class `fereg_fitstat` (a named list), or a vector if `simplify`.
#' @examples
#' data(trade_toy, package = "fereg")
#' m <- feols(y ~ x1 + x2 | id + year, trade_toy, nthreads = 2)
#' fitstat(m, ~ n + r2 + wr2)
#' @export
fitstat <- function(x, type, simplify = FALSE, ...) {
  if (inherits(x, "fereg_multi")) return(lapply(x, fitstat, type = type, simplify = simplify))
  if (!inherits(x, "fereg")) stop("fereg: fitstat() needs a fereg object", call. = FALSE)
  if (inherits(type, "formula")) type <- attr(stats::terms(type), "term.labels")
  type <- ifelse(type %in% names(fitstat_type_alias), fitstat_type_alias[type], type)
  bad <- setdiff(type, names(fitstat_alias))
  if (length(bad)) stop("fereg: unknown fitstat type", if (length(bad) > 1) "s", ": ", paste(bad, collapse = ", "), call. = FALSE)
  out <- list()
  for (t in type) {
    if (t %in% c("ivf1", "ivwald1")) {
      fs <- x$iv_first_stage
      if (isTRUE(x$iv_stage == 1)) fs <- list(x)
      for (k in seq_along(fs)) {
        name <- if (length(fs) > 1 || isTRUE(x$iv_stage == 2)) paste0(t, "::", x$iv_endo_names[k]) else t
        out[[name]] <- test_list(if (t == "ivf1") fs[[k]]$iv_f_test else fs[[k]]$iv_wald)
      }
      if (length(fs) == 0) out[[t]] <- NA
      next
    }
    out[[t]] <- switch(t,
      n = x$nobs,
      rmse = x$rmse %||% NA,
      r2 = r2_value(x, "r2"), ar2 = r2_value(x, "ar2"), wr2 = r2_value(x, "wr2"),
      war2 = r2_value(x, "war2"), pr2 = r2_value(x, "pr2"), apr2 = r2_value(x, "apr2"),
      cor2 = x$sq.cor %||% NA,
      ll = x$loglik %||% NA,
      aic = if (is.null(x$loglik)) NA else -2 * x$loglik + 2 * x$nparams,
      bic = if (is.null(x$loglik)) NA else -2 * x$loglik + log(x$nobs) * x$nparams,
      f = test_list(x$f_test),
      wh = test_list(x$iv_wh),
      sargan = test_list(x$iv_sargan),
      my = dep_var_mean(x),
      g = attr(x$cov.scaled, "G") %||% NA
    )
  }
  if (simplify && all(lengths(out) == 1)) return(unlist(out))
  class(out) <- "fereg_fitstat"
  out
}

dep_var_mean <- function(x) {
  if (is.null(x$residuals) || is.null(x$fitted.values)) return(NA)
  y <- x$residuals + x$fitted.values
  if (is.null(x$weights)) mean(y) else stats::weighted.mean(y, x$weights)
}

r2_value <- function(x, type) {
  glm <- is_glm(x)
  switch(type,
    r2 = if (glm) NA else x$r2_all[["r2"]],
    ar2 = if (glm) NA else x$r2_all[["ar2"]],
    wr2 = if (glm || length(x$fixef_terms) == 0) NA else x$r2_all[["wr2"]],
    war2 = if (glm || length(x$fixef_terms) == 0) NA else x$r2_all[["war2"]],
    pr2 = if (glm) x$pseudo_r2 else ols_pseudo_r2(x, adj = FALSE),
    apr2 = if (glm) x$apr2 else ols_pseudo_r2(x, adj = TRUE),
    cor2 = x$sq.cor %||% NA
  )
}

# fixest's pseudo R2 of an OLS model from the Gaussian log-likelihoods (NA for a constant-only
# model): 1 - ll/ll_null, adjusted 1 - (ll - nparams + 1)/ll_null.
ols_pseudo_r2 <- function(x, adj) {
  if (is.null(x$loglik) || is.null(x$ll_null)) return(NA)
  if (x$nparams == 1 && identical(names(x$coefficients), "(Intercept)")) return(NA)
  if (adj) 1 - (x$loglik - x$nparams + 1) / x$ll_null else 1 - x$loglik / x$ll_null
}

#' R-squared variants
#'
#' @param x A `fereg` object.
#' @param type Types among `"r2"`, `"ar2"`, `"wr2"`, `"war2"`, `"pr2"`, `"apr2"`, `"cor2"`, or
#'   `"all"`.
#' @return A named numeric vector (`NA` where a type does not apply).
#' @examples
#' data(trade_toy, package = "fereg")
#' m <- feols(y ~ x1 + x2 | id + year, trade_toy, nthreads = 2)
#' r2(m)
#' @export
r2 <- function(x, type = "all") {
  if (inherits(x, "fereg_multi")) return(do.call(rbind, lapply(x, r2, type = type)))
  types <- c("r2", "ar2", "pr2", "apr2", "wr2", "war2", "cor2")
  if (identical(type, "all")) type <- types
  bad <- setdiff(type, types)
  if (length(bad)) stop("fereg: unknown r2 type: ", paste(bad, collapse = ", "), call. = FALSE)
  vapply(stats::setNames(type, type), function(t) as.double(r2_value(x, t)), 0)
}

#' @export
print.fereg_fitstat <- function(x, na.rm = FALSE, group.solo = FALSE, ...) {
  if (na.rm) x <- x[!vapply(x, function(z) identical(z, NA) || (length(z) == 1 && is.na(z)), TRUE)]
  if (length(x) == 0) return(invisible(NULL))
  alias <- function(t) {
    parts <- strsplit(t, "::", fixed = TRUE)[[1]]
    a <- fitstat_alias[[parts[1]]]
    if (length(parts) > 1) paste0(a, ", ", parts[2]) else a
  }
  solo <- lengths(x) == 1
  if (group.solo && sum(solo) > 1) {
    vals <- unlist(x[solo])
    labs <- vapply(names(x)[solo], alias, "")
    x <- x[!solo]
    nr <- ceiling(length(vals) / 2)
    m <- matrix(c(vals, NA)[1:(2 * nr)], nr, 2, byrow = TRUE)
    ml <- matrix(c(labs, "")[1:(2 * nr)], nr, 2, byrow = TRUE)
    if (nr > 1 && anyNA(m[nr, ])) {
      m[nr, ] <- m[nr, 2:1]
      ml[nr, ] <- ml[nr, 2:1]
    }
    cols <- lapply(1:2, function(i) {
      col <- m[, i]
      txt <- number_format(col)
      # fixest aligns the decimal points of |x| > 1 and of the other values separately.
      gt1 <- abs(col) > 1 & !is.na(col)
      if (any(gt1)) txt[gt1] <- str_fill(txt[gt1], anchor = ".")
      if (any(!gt1)) txt[!gt1] <- str_fill(txt[!gt1], anchor = ".")
      txt <- str_fill(txt, right = TRUE)
      txt <- paste0(str_fill(ml[, i]), ": ", txt)
      txt[is.na(col)] <- strrep(" ", nchar(txt[1]))
      txt
    })
    cat(paste(cols[[1]], cols[[2]], sep = "   "), sep = "\n")
  }
  if (length(x) == 0) return(invisible(NULL))
  # Test statistics share one number of decimals and are right-aligned; p-values are padded
  # to a common width (fixest layout).
  tests <- names(x)[vapply(x, is.list, TRUE)]
  stat_txt <- p_txt <- stats::setNames(character(length(tests)), tests)
  if (length(tests)) {
    stats_all <- vapply(x[tests], function(v) v$stat, 0)
    s5 <- vapply(signif(stats_all, 5), format, "", scientific = FALSE, trim = TRUE)
    decimals <- nchar(sub("^[^.]*\\.?", "", s5))
    stat_txt[] <- str_fill(formatC(stats_all, format = "f", digits = min(max(decimals, 1), 5)))
    pv <- vapply(x[tests], function(v) v$p, 0)
    low <- !is.na(pv) & pv < 2.2e-16
    p_txt[] <- str_fill(ifelse(low, "p < 2.2e-16", paste0("p = ", number_format(pv))), right = TRUE)
  }
  lines <- vapply(names(x), function(t) {
    v <- x[[t]]
    if (!is.list(v)) return(paste0(alias(t), "! ", number_format(v)))
    dof <- if (is.na(v$df2)) paste0("on ", number_format(v$df1), " DoF") else paste0("on ", number_format(v$df1), " and ", number_format(v$df2), " DoF")
    paste0(alias(t), "! stat = ", stat_txt[[t]], ", ", p_txt[[t]], ", ", dof, ".")
  }, "")
  cat(sub("\\s+$", "", gsub("!", ":", str_fill(lines, anchor = "!"), fixed = TRUE)), sep = "\n")
  invisible(x)
}
