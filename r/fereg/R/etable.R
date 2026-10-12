# Regression tables in the layout of fixest's etable().

#' Table of estimation results
#'
#' Side-by-side table of several estimations in fixest's `etable()` layout: coefficients with
#' standard errors in parentheses and significance stars, fixed effects, variance type and fit
#' statistics.
#'
#' @param ... `fereg` or `fereg_multi` objects (named arguments set the column titles).
#' @param vcov,ssc Variance for every model (see [summary()][fereg-methods]).
#' @param tex Return LaTeX code instead of a data frame.
#' @param digits Significant digits of coefficients and standard errors.
#' @param digits.stats Decimals of the fit statistics.
#' @param fitstat Fit statistics (see [fitstat()]); by default observations, R2 (pseudo R2 and
#'   BIC for GLMs) and within R2 when there are fixed effects.
#' @return A data frame (class `fereg_etable`) or, with `tex = TRUE`, a character string (class
#'   `fereg_etable_tex`).
#' @export
etable <- function(..., vcov = NULL, ssc = NULL, tex = FALSE, digits = 4, digits.stats = 5, fitstat = NULL) {
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
  models <- lapply(models, function(m) summary.fereg(m, vcov = vcov, ssc = ssc))
  titles <- make.unique(titles)

  coef_names <- unique(unlist(lapply(models, function(m) names(m$coefficients))))
  coef_names <- c(setdiff(coef_names, "(Intercept)"), intersect(coef_names, "(Intercept)"))
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
    for (fe in fe_labels_all) add(fe, vapply(models, function(m) if (fe %in% m$fixef_vars) "Yes" else "No", ""))
  }
  if (length(slope_labels)) {
    add("Varying Slopes:", rep("-", length(models)))
    for (s in slope_labels) add(s, vapply(models, function(m) if (s %in% slope_texts(m)) "Yes" else "No", ""))
  }
  add("_", rep("_", length(models)))
  add("S.E. type", vapply(models, function(m) vcov_label_short(m$vcov_info), ""))

  any_fe <- length(fe_labels_all) > 0
  any_glm <- any(vapply(models, is_glm, TRUE))
  if (is.null(fitstat)) {
    fitstat <- c("n", if (any_glm) c("pr2", "bic") else "r2", if (any_fe && !all(vapply(models, is_glm, TRUE))) "wr2")
  }
  for (t in fitstat) {
    vals <- vapply(models, function(m) {
      v <- fitstat(m, t)[[1]]
      if (is.list(v)) v <- v$stat
      if (length(v) != 1 || is.na(v)) return("--")
      if (t == "n") format_int(v) else formatC(v, format = "f", digits = if (t %in% c("bic", "aic", "ll")) 1 else digits.stats)
    }, "")
    add(fitstat_alias[[t]], vals)
  }

  tab <- as.data.frame(do.call(rbind, rows), stringsAsFactors = FALSE)
  names(tab) <- c("", titles)
  if (tex) return(etable_tex(tab))
  class(tab) <- c("fereg_etable", "data.frame")
  tab
}

slope_texts <- function(m) {
  unlist(lapply(m$fixef_terms, function(t) if (length(t$slopes)) paste0(vapply(t$slopes, deparse1l, ""), " (", t$label, ")")))
}

stars <- function(p) {
  if (is.na(p)) "" else if (p < 0.001) "***" else if (p < 0.01) "**" else if (p < 0.05) "*" else if (p < 0.1) "." else ""
}

# `digits` significant digits for |x| >= 1, `digits` decimals below (fixest etable style).
fmt_coef <- function(x, digits) {
  if (is.na(x)) return("NA")
  a <- abs(x)
  d <- if (a >= 1) max(digits - floor(log10(a)) - 1, 0) else digits
  s <- formatC(x, format = "f", digits = d)
  if (x != 0 && as.numeric(s) == 0) s <- formatC(x, format = "e", digits = digits - 2)
  s
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
