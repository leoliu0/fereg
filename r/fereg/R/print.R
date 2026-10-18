# Printing in the layout of fixest's print.fixest.

`%||%` <- function(a, b) if (is.null(a)) b else a

# fixest decimalFormat: rounds a vector to a common number of decimals for display.
decimal_format <- function(x) {
  ok <- which(!is.na(x) & is.finite(x) & x != 0)
  if (length(ok) == 0) return(x)
  p <- log10(abs(x[ok]))
  digits <- if (min(p) > 0) max(1, 6 - ceiling(min(p))) else if (min(p) < -5) ceiling(abs(min(p))) + 2 else 6
  x[ok] <- round(x[ok], digits)
  x
}

# fixest numberFormatNormal: one number as text (thousands separators, short scientific form).
number_format <- function(x) {
  vapply(x, function(v) {
    if (is.na(v)) return(NA_character_)
    if (!is.finite(v)) return(as.character(v))
    if (v == 0) return("0")
    e <- floor(log10(abs(v)))
    if (-4 < e && e < 9) {
      if (e > 0) return(add_commas(v))
      return(as.character(decimal_format(v)))
    }
    paste0(round(v * 10^-e, 3), "e", if (e > 0) "+" else "", e)
  }, "")
}

# Numbers >= 10 (fixest): thousands separators and one decimal, none for integers.
add_commas <- function(v) {
  if (v == round(v)) return(formatC(v, format = "f", digits = 0, big.mark = ","))
  formatC(v, format = "f", digits = 1, big.mark = ",")
}

# Pads strings to a common width: right-aligned, or left-aligned when `right`. With `anchor`,
# strings are first shifted so that the anchor character lines up, then left-aligned.
str_fill <- function(x, right = FALSE, anchor = NULL) {
  x[is.na(x)] <- ""
  if (!is.null(anchor)) {
    pos <- regexpr(anchor, x, fixed = TRUE)
    pos[pos < 0] <- nchar(x[pos < 0]) + 1
    x <- paste0(strrep(" ", max(pos) - pos), x)
    right <- TRUE
  }
  pad(x, max(nchar(x)), left = right)
}

# Pads each string to width `w` (recycled), left-aligned when `left`.
pad <- function(x, w, left = FALSE) {
  fill <- strrep(" ", pmax(w - nchar(x), 0))
  if (left) paste0(x, fill) else paste0(fill, x)
}

print_coeftable <- function(ct, last_line = "") {
  df <- as.data.frame(ct)
  p <- df[[4]]
  stars <- as.character(cut(p, breaks = c(-1, 0.001, 0.01, 0.05, 0.1, 100), labels = c("***", "** ", "*  ", ".  ", "")))
  stars[is.na(stars)] <- ""
  low <- !is.na(p) & p < 2.2e-16
  for (i in 1:3) df[[i]] <- decimal_format(df[[i]])
  p_txt <- character(length(p))
  p_txt[!low] <- format(p[!low], digits = 5)
  p_txt[low] <- "< 2.2e-16"
  p_txt[is.na(p)] <- "NA"
  df[[4]] <- p_txt
  df[[5]] <- stars
  names(df)[5] <- ""
  print(df)
  if (nzchar(last_line)) cat(last_line)
  cat("---\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")
}

fe_header_lines <- function(x) {
  terms <- x$fixef_terms
  if (length(terms) == 0) return(character(0))
  sizes <- x$fixef_sizes
  out <- character(0)
  int <- vapply(terms, `[[`, TRUE, "intercept")
  if (any(int)) {
    labs <- vapply(terms[int], `[[`, "", "label")
    out <- c(out, paste0("Fixed-effects: ", paste0(labs, ": ", format_int(sizes[labs]), collapse = ",  ")))
  }
  slopes <- unlist(lapply(terms, function(t) {
    if (length(t$slopes) == 0) return(NULL)
    paste0(vapply(t$slopes, deparse1l, ""), " (", t$label, "): ", format_int(sizes[[t$label]]))
  }))
  if (length(slopes)) out <- c(out, paste0("Varying slopes: ", paste(slopes, collapse = ",  ")))
  out
}

estimation_title <- function(x) {
  if (x$method_type == "feols") return("OLS estimation")
  if (identical(x$method, "fepois")) return("Poisson estimation")
  paste0("GLM estimation, family = ", glm_family_label(x))
}

# fixest prints the family as given in the call when it has named arguments
# (`binomial(link = "logit")`, which is also what the string "logit" becomes), else the family
# name (`binomial()`, "binomial", a variable holding a family).
glm_family_label <- function(x) {
  fam_call <- x$call$family
  if (is.null(fam_call) || identical(fam_call, "logit")) {
    if (x$family$family == "binomial") return("binomial(link = \"logit\")")
  }
  if (is.call(fam_call) && !is.null(names(fam_call))) return(deparse1l(fam_call))
  x$family$family
}

#' @export
print.fereg <- function(x, n, ...) {
  if (isTRUE(x$is_iv)) {
    stage <- if (x$iv_stage == 1) "First" else "Second"
    cat("TSLS estimation: ", stage, " stage\n", sep = "")
    cat("|- D.V.   : ", if (x$iv_stage == 1) x$iv_dep_var else x$design$y_name, "\n", sep = "")
    cat("|- Endo.  : ", paste(x$iv_endo_names, collapse = ", "), "\n", sep = "")
    cat("|- Instr. : ", paste(x$iv_inst_names, collapse = ", "), "\n", sep = "")
    cat("Dep. Var.: ", deparse1l(x$fml[[2]]), "\n", sep = "")
  } else {
    title <- estimation_title(x)
    dep <- deparse1l(x$fml[[2]])
    cat(title, if (nchar(title) + nchar(dep) <= 60) ", " else "\n", "Dep. Var.: ", dep, "\n", sep = "")
  }
  cat("Observations: ", format_int(x$nobs), "\n", sep = "")
  mi <- x$model_info
  if (!is.null(mi$sample) && mi$sample$value != "Full sample") cat("Sample (", mi$sample$var, "): ", mi$sample$value, "\n", sep = "")
  if (!is.null(mi$offset)) cat("Offset: ", mi$offset, "\n", sep = "")
  if (!is.null(mi$weights)) cat("Weights: ", mi$weights, "\n", sep = "")
  for (l in fe_header_lines(x)) cat(l, "\n", sep = "")
  if (inherits(x, "fereg_biascorr")) cat(bias_label(x), "\n", sep = "")
  cat("Standard-errors:", x$vcov_info$label, "\n")

  ct <- x$coeftable
  k <- nrow(ct)
  remaining <- ""
  if (missing(n)) {
    n <- if (isTRUE(x$summary) || k <= 15) k else 8
    if (n < k) remaining <- sprintf("... %d coefficients remaining (display them with summary() or use argument n)\n", k - n)
  } else if (n < k) {
    remaining <- sprintf("... %d coefficients remaining\n", k - n)
  }
  collin <- ""
  if (length(x$collin.var)) {
    nc <- length(x$collin.var)
    shown <- x$collin.var[seq_len(min(3, nc))]
    collin <- sprintf("... %d variable%s removed because of collinearity (%s%s)\n", nc,
      if (nc > 1) "s were" else " was", paste(shown, collapse = ", "), if (nc > 3) " [full set in $collin.var]" else "")
  }
  if (k > 0) {
    print_coeftable(ct[seq_len(min(n, k)), , drop = FALSE], last_line = paste0(remaining, collin))
  } else if (nzchar(collin)) {
    cat(collin)
  }
  print(fitstat(x, default_fitstat(x)), na.rm = TRUE, group.solo = TRUE)
  if (!isTRUE(x$convStatus) && x$method_type == "feglm") {
    cat("# Evaluations:", x$iterations, "-- the IRLS algorithm did not converge\n")
  }
  invisible(x)
}

default_fitstat <- function(x) {
  if (x$method_type == "feols") {
    out <- c("rmse", "ar2")
    if (length(x$fixef_terms)) out <- c(out, "wr2")
    if (isTRUE(x$is_iv) && x$iv_stage == 2) out <- c(out, "ivf1", "wh", "sargan")
    if (isTRUE(x$is_iv) && x$iv_stage == 1) out <- c(out, "ivf1")
    out
  } else {
    c("ll", "apr2", "bic", "cor2")
  }
}
