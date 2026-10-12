# Variance-covariance requests: vcov = "iid" / "hetero" / "cluster" / "twoway" / ~cl1 + cl2,
# the legacy `cluster` and `se` arguments, and small-sample corrections (ssc()).

#' Small-sample correction
#'
#' Same arguments and defaults as fixest's `ssc()`.
#'
#' @param K.adj Multiply the variance by `(n - 1) / (n - K)` (iid and clustered) or
#'   `n / (n - K)` (heteroskedasticity-robust).
#' @param K.fixef How fixed effects count in `K`: `"nonnested"` (fixed effects nested in a
#'   cluster variable do not count), `"full"` or `"none"`.
#' @param K.exact Count the exact number of redundant fixed-effect coefficients.
#' @param G.adj Multiply clustered variances by `G / (G - 1)`.
#' @param G.df `"min"` (one factor with the smallest number of clusters) or `"conventional"`
#'   (each cluster dimension uses its own `G`).
#' @param t.df Degrees of freedom of the t distribution with clustered variances: `"min"`
#'   (`G - 1`), `"conventional"` (`n - K`) or a number.
#' @param adj,fixef.K,cluster.adj,cluster.df Former fixest names of `K.adj`, `K.fixef`, `G.adj`
#'   and `G.df`.
#' @return An object of class `ssc.type`.
#' @export
ssc <- function(K.adj = TRUE, K.fixef = "nonnested", K.exact = FALSE, G.adj = TRUE,
                G.df = "min", t.df = "min", adj, fixef.K, cluster.adj, cluster.df) {
  if (!missing(adj)) K.adj <- adj
  if (!missing(fixef.K)) K.fixef <- fixef.K
  if (!missing(cluster.adj)) G.adj <- cluster.adj
  if (!missing(cluster.df)) G.df <- cluster.df
  flag <- function(x, name) {
    if (!is.logical(x) || length(x) != 1 || is.na(x)) stop("fereg: ssc(", name, ") must be TRUE or FALSE", call. = FALSE)
    x
  }
  K.fixef <- match.arg(K.fixef, c("nonnested", "full", "none", "nested", "all"))
  K.fixef <- switch(K.fixef, nested = "nonnested", all = "full", K.fixef)
  G.df <- match.arg(G.df, c("min", "conventional"))
  if (is.character(t.df)) {
    t.df <- match.arg(t.df, c("min", "conventional"))
  } else if (!is.numeric(t.df) || length(t.df) != 1 || is.na(t.df) || t.df < 1) {
    stop("fereg: ssc(t.df) must be \"min\", \"conventional\" or a number >= 1", call. = FALSE)
  }
  structure(
    list(
      K.adj = flag(K.adj, "K.adj"), K.fixef = K.fixef, K.exact = flag(K.exact, "K.exact"),
      G.adj = flag(G.adj, "G.adj"), G.df = G.df, t.df = t.df
    ),
    class = "ssc.type"
  )
}

#' @export
print.ssc.type <- function(x, ...) {
  t_df <- if (is.numeric(x$t.df)) format(x$t.df) else paste0('"', x$t.df, '"')
  cat(sprintf(
    "Small sample correction: K.adj = %s, K.fixef = \"%s\", K.exact = %s, G.adj = %s, G.df = \"%s\", t.df = %s\n",
    x$K.adj, x$K.fixef, x$K.exact, x$G.adj, x$G.df, t_df
  ))
  invisible(x)
}

# The ssc applied to a vcov type: fixest uses ssc(K.fixef = "full") by default for iid and
# heteroskedasticity-robust variances, which cannot use "nonnested".
effective_ssc <- function(ssc_arg, type) {
  s <- if (is.null(ssc_arg)) ssc() else ssc_arg
  if (!inherits(s, "ssc.type")) stop("fereg: `ssc` must be created with ssc()", call. = FALSE)
  if (type != "cluster" && s$K.fixef == "nonnested") s$K.fixef <- "full"
  s
}

vcov_aliases <- c(
  iid = "iid", normal = "iid", standard = "iid", hetero = "hetero", white = "hetero",
  hc1 = "hetero", cluster = "cluster", twoway = "twoway", threeway = "threeway", fourway = "fourway"
)

# Parses vcov / cluster / se into list(type, dims) where `type` is "iid", "hetero" or "cluster"
# and `dims` is either a list of cluster dimensions (each a list of variable expressions,
# interacted when > 1) or the number of leading fixed effects to cluster on.
parse_vcov <- function(vcov = NULL, cluster = NULL, se = NULL) {
  if (is.null(vcov) && !is.null(se)) vcov <- se
  if (is.null(vcov) && !is.null(cluster)) vcov <- cluster
  if (is.null(vcov)) return(list(type = "iid", dims = list()))
  if (inherits(vcov, "formula")) return(list(type = "cluster", dims = cluster_dims(vcov[[length(vcov)]])))
  if (!is.character(vcov) || length(vcov) == 0) {
    stop("fereg: `vcov` must be \"iid\", \"hetero\", \"cluster\", \"twoway\" or a formula such as ~cl", call. = FALSE)
  }
  key <- vcov_aliases[tolower(vcov[1])]
  if (length(vcov) == 1 && !is.na(key)) {
    if (key == "cluster" && !is.null(cluster)) return(parse_vcov(cluster))
    return(switch(key,
      iid = list(type = "iid", dims = list()),
      hetero = list(type = "hetero", dims = list()),
      cluster = list(type = "cluster", dims = 1L),
      twoway = list(type = "cluster", dims = 2L),
      threeway = list(type = "cluster", dims = 3L),
      fourway = list(type = "cluster", dims = 4L)
    ))
  }
  # Character vector of cluster variable names (legacy `cluster = c("a", "b")`).
  list(type = "cluster", dims = lapply(vcov, function(v) split_hat(str2lang(v))))
}

cluster_dims <- function(e) {
  dims <- lapply(split_plus(e), function(t) split_hat(t$expr))
  if (length(dims) > 4) stop("fereg: at most four cluster dimensions are supported", call. = FALSE)
  dims
}

# Cluster dimensions for a model: clustering "on the first k fixed effects" uses its FE terms.
resolve_vcov <- function(v, fe_terms) {
  if (v$type != "cluster" || is.list(v$dims)) return(v)
  k <- v$dims
  fe_int <- Filter(function(t) t$intercept, fe_terms)
  if (length(fe_int) < k) {
    stop("fereg: vcov = \"", c("cluster", "twoway", "threeway", "fourway")[k], "\" needs ", k,
      " fixed effect", if (k > 1) "s", "; use a formula such as vcov = ~cl", call. = FALSE)
  }
  list(type = "cluster", dims = lapply(fe_int[seq_len(k)], `[[`, "vars"))
}

dim_label <- function(d) paste(vapply(d, deparse1l, ""), collapse = "^")

# fixest's "Standard-errors:" text.
vcov_label <- function(v) {
  switch(v$type,
    iid = "IID",
    hetero = "Heteroskedasticity-robust",
    cluster = {
      paste0("Clustered (", paste(vapply(v$dims, dim_label, ""), collapse = " & "), ")")
    }
  )
}

# Short form for etable.
vcov_label_short <- function(v) {
  switch(v$type,
    iid = "IID",
    hetero = "Heteroskedas.-rob.",
    cluster = paste0("by: ", paste(vapply(v$dims, dim_label, ""), collapse = " & "))
  )
}

ssc_core <- function(s) unclass(s)
