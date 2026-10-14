# Regression tables in the layout of fixest's etable().

#' Table of estimation results
#'
#' Side-by-side table of several estimations in fixest's `etable()` layout: coefficients with
#' standard errors in parentheses and significance stars, fixed effects, variance type and fit
#' statistics.
#'
#' @param ... `fereg` or `fereg_multi` objects (named arguments set the column titles).
#' @param vcov,ssc Variance for every model (see [summary()][fereg-methods]); `vcov` may also be
#'   a list with one variance per model, as in fixest.
#' @param tex Return LaTeX code instead of a data frame.
#' @param digits Significant digits of coefficients and standard errors.
#' @param digits.stats Digits of the fit statistics, as in fixest: a number (significant digits,
#'   at least one decimal for large values), or `"rN"` (round to N decimals) / `"sN"`.
#' @param fitstat Fit statistics (see [fitstat()]). The default is fixest's: observations, R2 and
#'   within R2 (adjusted R2 without fixed effects) when every model is OLS, else observations,
#'   squared correlation, pseudo R2 and BIC.
#' @param family Show the row "Family" (OLS, Poisson, Logit); by default only when the models
#'   are of different families.
#' @param keep,drop Character vectors of regular expressions (Perl syntax) selecting the
#'   coefficient rows, as in fixest: `keep` shows the coefficients matching any of them, then
#'   `drop` removes those matching any of them. Names are matched as displayed (the intercept is
#'   `"Constant"`); a leading `"%"` matches the original names instead (`"(Intercept)"`), and a
#'   leading `"!"` negates the expression.
#' @return A data frame (class `fereg_etable`) or, with `tex = TRUE`, a character string (class
#'   `fereg_etable_tex`).
#' @examples
#' data(trade_toy, package = "fereg")
#' m1 <- feols(y ~ x1 + x2 | id + year, trade_toy, nthreads = 2)
#' m2 <- fepois(count ~ x1 + x2 | id + year, trade_toy, nthreads = 2)
#' etable(m1, m2, vcov = ~id)
#' etable(m1, tex = TRUE)
#' @export
etable <- function(..., vcov = NULL, ssc = NULL, tex = FALSE, digits = 4, digits.stats = 5, fitstat = NULL,
                   keep = NULL, drop = NULL, family = NULL) {
  dots <- list(...)
  arg_names <- names(dots) %||% rep("", length(dots))
  exprs <- as.list(substitute(list(...)))[-1]
  models <- list()
  titles <- character(0)
  for (k in seq_along(dots)) {
    d <- dots[[k]]
    if (inherits(d, "fereg")) {
      models <- c(models, list(d))
      titles <- c(titles, if (nzchar(arg_names[k])) arg_names[k] else deparse1l(exprs[[k]]))
    } else if (inherits(d, "fereg_multi")) {
      base <- if (nzchar(arg_names[k])) arg_names[k] else deparse1l(exprs[[k]])
      models <- c(models, as.list.fereg_multi(d))
      titles <- c(titles, paste0(base, ".", seq_along(d)))
    } else {
      stop("fereg: etable() takes fereg estimation results", call. = FALSE)
    }
  }
  if (length(models) == 0) stop("fereg: etable() needs at least one estimation", call. = FALSE)
  vcovs <- if (is.list(vcov) && !inherits(vcov, "fixest_vcov_request")) vcov else list(vcov)
  if (length(vcovs) == 1) vcovs <- rep(vcovs, length(models))
  if (length(vcovs) != length(models)) {
    stop("fereg: a list `vcov` must have one element per model (", length(models), "), or a single one", call. = FALSE)
  }
  models <- Map(function(m, v) summary.fereg(m, vcov = v, ssc = ssc), models, vcovs)
  titles <- make.unique(titles)

  # Coefficients in order of first appearance across the models (fixest).
  coef_names <- unique(unlist(lapply(models, function(m) names(m$coefficients))))
  if (length(keep) || length(drop)) {
    shown <- stats::setNames(ifelse(coef_names == "(Intercept)", "Constant", coef_names), coef_names)
    shown <- drop_apply(keep_apply(shown, keep), drop)
    if (length(shown) == 0) stop("fereg: no coefficient was selected, please reframe the keep/drop arguments", call. = FALSE)
    coef_names <- names(shown)
  }
  cells <- vapply(models, function(m) {
    ct <- m$coeftable
    vapply(coef_names, function(nm) {
      if (!nm %in% rownames(ct)) return("")
      paste0(fmt_coef(ct[nm, 1], digits), stars(ct[nm, 4]), " (", fmt_coef(ct[nm, 2], digits), ")")
    }, "")
  }, character(length(coef_names)))
  cells <- matrix(cells, nrow = length(coef_names))
  row_names <- ifelse(coef_names == "(Intercept)", "Constant", coef_names)

  rows <- list()
  add <- function(name, values) rows[[length(rows) + 1]] <<- c(name, values)
  if (any(vapply(models, function(m) !is.null(m$model_info$sample), TRUE))) {
    add("Sample", vapply(models, function(m) if (is.null(m$model_info$sample)) "Full sample" else m$model_info$sample$value, ""))
  }
  add("Dependent Var.:", vapply(models, function(m) deparse1l(m$fml[[2]]), ""))
  add("", rep("", length(models)))
  for (i in seq_along(coef_names)) add(row_names[i], cells[i, ])

  fe_labels_all <- unique(unlist(lapply(models, function(m) {
    vapply(Filter(function(t) t$intercept, m$fixef_terms), `[[`, "", "label")
  })))
  slope_labels <- unique(unlist(lapply(models, slope_texts)))
  if (length(fe_labels_all)) {
    add("Fixed-Effects:", rep("-", length(models)))
    # Combined fixed effects are shown as "a-b" (fixest's rename_fe).
    for (fe in fe_labels_all) add(gsub("^", "-", fe, fixed = TRUE), vapply(models, function(m) if (fe %in% m$fixef_vars) "Yes" else "No", ""))
  }
  if (length(slope_labels)) {
    add("Varying Slopes:", rep("-", length(models)))
    for (s in slope_labels) add(s, vapply(models, function(m) if (s %in% slope_texts(m)) "Yes" else "No", ""))
  }
  add("_", rep("_", length(models)))
  families <- vapply(models, model_family, "")
  if (isTRUE(family) || (is.null(family) && length(unique(families)) > 1)) add("Family", families)
  # fixest: the variance label shortened to the column width (coefficients and dependent
  # variable, at least 8); "S.E.: Clustered" with "by: ..." cells when every model is clustered.
  widths <- vapply(seq_along(models), function(j) {
    max(nchar(cells[, j]), nchar(deparse1l(models[[j]]$fml[[2]])), 8L)
  }, 1L)
  labels <- vapply(models, function(m) m$vcov_info$label, "")
  se_title <- if (all(grepl("Clustered", labels, fixed = TRUE))) "S.E.: Clustered" else "S.E. type"
  add(se_title, mapply(format_se_type, labels, widths, USE.NAMES = FALSE))

  if (is.null(fitstat)) {
    is_ols <- vapply(models, function(m) m$method_type == "feols", TRUE)
    fitstat <- if (!all(is_ols)) {
      c("cor2", "pr2", "bic")
    } else if (any(vapply(models, function(m) length(m$fixef_terms) > 0, TRUE))) {
      c("r2", "wr2")
    } else {
      c("r2", "ar2")
    }
    fitstat <- c("n", fitstat)
  }
  ds <- stats_digits(digits.stats)
  for (t in fitstat) {
    vals <- vapply(models, function(m) {
      v <- fitstat(m, t)[[1]]
      if (is.list(v)) v <- v$stat
      if (length(v) != 1 || is.na(v)) return("--")
      format_stat(as.double(v), ds$digits, ds$round)
    }, "")
    add(fitstat_alias[[t]], vals)
  }

  tab <- as.data.frame(do.call(rbind, rows), stringsAsFactors = FALSE)
  names(tab) <- c("", titles)
  if (tex) return(etable_tex(tab))
  class(tab) <- c("fereg_etable", "data.frame")
  tab
}

# fixest's "Family" row labels.
model_family <- function(m) {
  if (m$method_type == "feols") return("OLS")
  if (identical(m$family$family, "poisson")) "Poisson" else "Logit"
}

# digits.stats as fixest: a number, or "rN" (round to N decimals) / "sN" (N significant digits).
stats_digits <- function(d) {
  if (is.character(d) && length(d) == 1 && grepl("^[rs][0-9]$", d)) {
    return(list(digits = as.numeric(substr(d, 2, 2)), round = substr(d, 1, 1) == "r"))
  }
  if (!is.numeric(d) || length(d) != 1 || is.na(d) || d < 1) {
    stop("fereg: `digits.stats` must be a number >= 1 or a string like \"r3\" or \"s3\"", call. = FALSE)
  }
  list(digits = d, round = FALSE)
}

# One fit statistic as text, as fixest's etable (format_number): `digits` significant digits but
# one decimal from 10^(digits - 1) on, thousands separators, integers without decimals, and the
# short scientific form from 1e10 on and for |x| < 1e-4.
format_stat <- function(x, digits = 5, round = FALSE) {
  if (round) return(stat_commas(x, digits, TRUE))
  whole <- x %% 1 == 0
  x <- if (abs(x) >= 10^(digits - 1)) base::round(x, 1) else signif(x, digits)
  e <- if (x == 0) -16 else floor(log10(abs(x)))
  if (e >= 10 || e <= -5) {
    paste0(base::round(x * 10^-e, min(digits, 2)), "e", if (e > 0) "+" else "", e)
  } else if (e >= 0) {
    stat_commas(x, max(-(e + 1 - digits), 1), whole)
  } else if (abs(x) > 10^(-digits)) {
    sprintf("%.*f", digits, x)
  } else {
    sprintf("%.*f", abs(e), x)
  }
}

# `r` decimals (none for an integer when `whole`), halves rounded away from zero, with thousands
# separators; no "-0.0".
stat_commas <- function(x, r, whole) {
  if (whole && x %% 1 == 0) r <- 0
  x <- sign(x) * floor(abs(x) * 10^r + 0.5) / 10^r
  s <- formatC(x, format = "f", digits = r, big.mark = ",")
  if (as.numeric(gsub(",", "", s, fixed = TRUE)) == 0) s <- sub("^-", "", s)
  s
}

# fixest's keep/drop rules: regular expressions on the displayed names, or on the original names
# (names(x)) with a leading "%"; a leading "!" negates ("!%" and "%!" are the same).
keep_drop_target <- function(x, pattern) {
  if (grepl("^!%", pattern)) pattern <- sub("^!%", "%!", pattern)
  target <- unname(x)
  if (grepl("^%", pattern)) {
    pattern <- sub("^%", "", pattern)
    target <- names(x)
  }
  negate <- grepl("^!", pattern)
  if (negate) pattern <- substr(pattern, 2, nchar(pattern))
  list(hit = grepl(pattern, target, perl = TRUE), negate = negate)
}

keep_apply <- function(x, keep) {
  if (length(keep) == 0) return(x)
  check_patterns(keep, "keep")
  sel <- rep(FALSE, length(x))
  for (p in keep) {
    m <- keep_drop_target(x, p)
    sel <- sel | if (m$negate) !m$hit else m$hit
  }
  x[sel]
}

drop_apply <- function(x, drop) {
  if (length(drop) == 0) return(x)
  check_patterns(drop, "drop")
  for (p in drop) {
    m <- keep_drop_target(x, p)
    x <- x[if (m$negate) m$hit else !m$hit]
  }
  x
}

check_patterns <- function(p, name) {
  if (!is.character(p) || anyNA(p)) {
    stop("fereg: `", name, "` must be a character vector of regular expressions", call. = FALSE)
  }
}

slope_texts <- function(m) {
  unlist(lapply(m$fixef_terms, function(t) if (length(t$slopes)) paste0(vapply(t$slopes, deparse1l, ""), " (", t$label, ")")))
}

stars <- function(p) {
  if (is.na(p)) "" else if (p < 0.001) "***" else if (p < 0.01) "**" else if (p < 0.05) "*" else if (p < 0.1) "." else ""
}

# Coefficients and standard errors as fixest's etable (format_number with `digits`: rounded to
# `digits` significant digits first, then printed).
fmt_coef <- function(x, digits) {
  if (is.na(x)) return("NA")
  format_stat(x, digits)
}

#' @export
print.fereg_etable <- function(x, ...) {
  m <- as.matrix(as.data.frame(unclass(x), stringsAsFactors = FALSE))
  header <- c("", names(x)[-1])
  w <- pmax(nchar(header), apply(m, 2, function(col) max(nchar(col))))
  line <- function(row) paste(c(pad(row[1], w[1], left = TRUE), pad(row[-1], w[-1])), collapse = " ")
  rule <- function(first, ch) paste(c(pad(first, w[1], left = TRUE), strrep(ch, w[-1])), collapse = " ")
  cat(line(header), "\n", sep = "")
  for (i in seq_len(nrow(m))) {
    row <- m[i, ]
    out <- if (row[1] == "_") {
      paste(strrep("_", w), collapse = " ")
    } else if (all(row[-1] == "-")) {
      rule(row[1], "-")
    } else {
      line(row)
    }
    cat(out, "\n", sep = "")
  }
  cat("---\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")
  invisible(x)
}

etable_tex <- function(tab) {
  esc <- function(s) gsub("([&%$#_{}])", "\\\\\\1", s)
  m <- as.matrix(as.data.frame(unclass(tab), stringsAsFactors = FALSE))
  k <- ncol(m) - 1
  lines <- c(
    paste0("\\begin{tabular}{l", strrep("c", k), "}"),
    "\\hline",
    paste0(" & ", paste(esc(names(tab)[-1]), collapse = " & "), " \\\\"),
    "\\hline"
  )
  for (i in seq_len(nrow(m))) {
    row <- m[i, ]
    if (row[1] == "") next
    if (row[1] == "_") {
      lines <- c(lines, "\\hline")
      next
    }
    vals <- row[-1]
    if (all(vals == "-")) {
      lines <- c(lines, paste0("\\emph{", esc(row[1]), "}", strrep(" &", k), " \\\\"))
      next
    }
    vals <- gsub("(\\*+|\\.)? \\(", "$^{\\1}$ (", vals)
    vals <- gsub("\\$\\^\\{\\}\\$", "", vals)
    lines <- c(lines, paste0(esc(row[1]), " & ", paste(vals, collapse = " & "), " \\\\"))
  }
  lines <- c(lines, "\\hline", "\\end{tabular}")
  structure(paste(lines, collapse = "\n"), class = "fereg_etable_tex")
}

#' @export
print.fereg_etable_tex <- function(x, ...) {
  cat(x, "\n", sep = "")
  invisible(x)
}
