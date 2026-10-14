# Multiple estimations (several dependent variables, stepwise functions, split samples).

new_multi <- function(results, ids) {
  names(results) <- vapply(ids, function(id) paste0(names(id), ": ", id, collapse = "; "), "")
  structure(results, class = "fereg_multi", ids = ids)
}

#' Multiple estimations
#'
#' A `fereg_multi` object is a list of `fereg` objects (one per estimation, fixest order:
#' sample, fixed effects, dependent variable, right-hand side). Methods apply to every
#' estimation.
#'
#' @param x,object A `fereg_multi` object.
#' @param i Indices or names of estimations.
#' @param ... Passed to the methods of each estimation.
#' @return `print()`: `x`, invisibly. `[` and `summary()`: a `fereg_multi` object. `as.list()`:
#'   a list of `fereg` objects. `coef()`: data frame with one row per estimation and one column
#'   per coefficient. `nobs()`: number of observations of each estimation.
#' @examples
#' data(trade_toy, package = "fereg")
#' ms <- feols(y ~ csw(x1, x2) | id + year, trade_toy, nthreads = 2)
#' ms
#' coef(ms)
#' nobs(ms)
#' summary(ms[2], vcov = ~id)
#' @name fereg_multi
NULL

#' @rdname fereg_multi
#' @export
print.fereg_multi <- function(x, ...) {
  print(etable(x, ...))
  invisible(x)
}

#' @rdname fereg_multi
#' @export
`[.fereg_multi` <- function(x, i) {
  ids <- attr(x, "ids")
  names(ids) <- names(x)
  sel <- unclass(x)[i]
  structure(sel, class = "fereg_multi", ids = unname(ids[names(sel)]))
}

#' @rdname fereg_multi
#' @export
as.list.fereg_multi <- function(x, ...) {
  out <- unclass(x)
  attr(out, "ids") <- NULL
  out
}

#' @rdname fereg_multi
#' @export
summary.fereg_multi <- function(object, ...) {
  out <- lapply(object, summary.fereg, ...)
  structure(out, class = "fereg_multi", ids = attr(object, "ids"))
}

# One row per estimation, one column per coefficient (fixest layout).
multi_table <- function(object, f) {
  vals <- lapply(object, f)
  coef_names <- unique(unlist(lapply(vals, names)))
  ids <- attr(object, "ids")
  id_cols <- unique(unlist(lapply(ids, names)))
  rows <- lapply(seq_along(vals), function(k) {
    idv <- stats::setNames(as.list(ids[[k]][id_cols]), id_cols)
    v <- stats::setNames(as.list(unname(vals[[k]][coef_names])), coef_names)
    c(list(id = k), idv, v)
  })
  out <- do.call(rbind, lapply(rows, function(r) as.data.frame(r, check.names = FALSE, stringsAsFactors = FALSE)))
  out
}

#' @rdname fereg_multi
#' @export
coef.fereg_multi <- function(object, ...) multi_table(object, coef.fereg)

#' @export
se.fereg_multi <- function(object, ...) multi_table(object, function(m) se.fereg(m, ...))

#' @export
tstat.fereg_multi <- function(object, ...) multi_table(object, function(m) tstat.fereg(m, ...))

#' @export
pvalue.fereg_multi <- function(object, ...) multi_table(object, function(m) pvalue.fereg(m, ...))

#' @export
coeftable.fereg_multi <- function(object, ...) lapply(object, coeftable.fereg, ...)

#' @rdname fereg_multi
#' @export
nobs.fereg_multi <- function(object, ...) vapply(object, nobs.fereg, 0L)
