# Variance-covariance requests (fixest 0.14.2 syntax): vcov = "iid" / "hetero" / "hc2" / "hc3" /
# "cluster" / "twoway" / "NW" / "DK" / "conley" / "crv3" (fereg extension), formulas
# `type ~ vars` (`NW(2) ~ id + period`, `conley(100) ~ lat + lon`, ...), one-sided cluster
# formulas, fixest's request objects (NW(), vcov_NW(), ...), the legacy `cluster` and `se`
# arguments, and small-sample corrections (ssc()).

#' Small-sample correction
#'
#' Same arguments and defaults as fixest's `ssc()`. As in fixest, the correction depends on the
#' variance type: heteroskedasticity-robust, Newey-West, Driscoll-Kraay and Conley variances use
#' `K.fixef = "full"` (no nesting), Newey-West and Driscoll-Kraay apply `G.adj` with `G` = the
#' number of time periods, Conley uses `n / (n - K)`, and HC2 / HC3 use no correction at all.
#'
#' @param K.adj Multiply the variance by `(n - 1) / (n - K)` (iid, clustered, Newey-West,
#'   Driscoll-Kraay) or `n / (n - K)` (heteroskedasticity-robust, Conley).
#' @param K.fixef How fixed effects count in `K`: `"nonnested"` (fixed effects nested in a
#'   cluster variable do not count), `"full"` or `"none"`.
#' @param K.exact Count the exact number of redundant fixed-effect coefficients.
#' @param G.adj Multiply clustered variances by `G / (G - 1)` (Newey-West and Driscoll-Kraay:
#'   `T / (T - 1)` with `T` time periods).
#' @param G.df `"min"` (one factor with the smallest number of clusters) or `"conventional"`
#'   (each cluster dimension uses its own `G`).
#' @param t.df Degrees of freedom of the t distribution with clustered, Newey-West and
#'   Driscoll-Kraay variances: `"min"` (`G - 1`, or `T - 1`), `"conventional"` (`n - K`) or a
#'   number.
#' @param adj,fixef.K,cluster.adj,cluster.df Former fixest names of `K.adj`, `K.fixef`, `G.adj`
#'   and `G.df`.
#' @return An object of class `ssc.type`.
#' @examples
#' data(trade_toy, package = "fereg")
#' m <- feols(y ~ x1 + x2 | id + year, trade_toy, nthreads = 2)
#' summary(m, vcov = ~id, ssc = ssc(K.adj = FALSE, G.adj = FALSE))
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

# The ssc applied to a vcov type: fixest's default for every type except clustering is
# ssc(K.fixef = "full"), and only clustered variances can use "nonnested".
effective_ssc <- function(ssc_arg, type) {
  s <- if (is.null(ssc_arg)) ssc() else ssc_arg
  if (!inherits(s, "ssc.type")) stop("fereg: `ssc` must be created with ssc()", call. = FALSE)
  if (!type %in% c("cluster", "crv3") && s$K.fixef == "nonnested") s$K.fixef <- "full"
  s
}

# ---- request objects (fixest's fixest_vcov_request) ---------------------------------------------

#' Variance requests
#'
#' fixest's request objects and helpers for the `vcov` argument of [feols()], [fepois()],
#' [feglm()], [summary()][fereg-methods], [vcov()][fereg-methods], `confint()` and [etable()].
#' They return objects of class `fixest_vcov_request` (the same structure as fixest's, so both
#' packages' requests are accepted). Called with an estimation as first argument, the `vcov_*()`
#' functions return its variance matrix instead, as in fixest.
#'
#' * `NW(lag)` / `newey_west(lag)`: Newey-West (Bartlett kernel). With a unit and a time variable
#'   (`vcov = NW ~ id + period`, or `panel.id`), only pairs of observations of the same unit enter;
#'   with a time variable only (`NW ~ period` without `panel.id`) it is the time-series estimator.
#'   Time is the dense rank of the time variable over the estimation sample (gaps are collapsed).
#'   Default lag: `floor(T^(1/4))` for panels, fixest's `sandwich::bwNeweyWest()` rule for a time
#'   series. Label `Newey-West (L=<lag>)`.
#' * `DK(lag)` / `driscoll_kraay(lag)`: Driscoll-Kraay on the per-period score sums; default lag
#'   `floor(T^(1/4))`. Label `Driscoll-Kraay (L=<lag>)`.
#' * `conley(cutoff, pixel, distance)`: Conley spatial HAC with fixest's uniform kernel. `cutoff`
#'   is in kilometers (a number) or a string such as `"100km"` / `"60mi"`; `NULL` deduces it from
#'   the coordinates as fixest does. `pixel` (same unit as `cutoff`, default 0) aggregates the
#'   scores on a grid first; `distance` is `"triangular"` (default, fixest's flat-earth distance)
#'   or `"spherical"` (great-circle). Latitude and longitude come from the formula
#'   (`conley(100) ~ lat + lon`) or are guessed from the column names (`lat`, `latitude`, `lat_*`;
#'   `lon`, `lng`, `long`, `longitude`, `lon_*`, ...). Label `Conley (<cutoff><km|mi>)`.
#' * `vcov_hetero(type)`: `"hc1"` (heteroskedasticity-robust, default), `"hc2"` or `"hc3"`. HC2 and
#'   HC3 use the exact leverage of the full model (fixed effects included) and no small-sample
#'   correction; they are not defined for IV estimations.
#' * `vcov_cluster(cluster)`: clustering on up to four variables (names or a formula; `a^b`
#'   interacts).
#' * `vcov_crv3(cluster)`: **fereg extension** (fixest has no CRV3): leave-one-cluster-out
#'   jackknife centered at the full-sample coefficients (pyfixest's CRV3), with fixest's
#'   cluster small-sample correction and multiway inclusion-exclusion. Also requested with
#'   `vcov = "crv3"` (clusters guessed as for `"cluster"`) or `vcov = crv3 ~ g1 + g2`. CRV3 needs
#'   the leave-one-out refits: after estimation (`summary(m, vcov = "crv3")`) the model is
#'   re-estimated.
#'
#' @param lag Number of lags (`NULL` = the default rule).
#' @param cutoff,pixel,distance Conley options, see Details.
#' @param x An estimation (`fereg` object), or omitted to build a request. As in fixest, when the
#'   first argument is not an estimation it is taken as the next argument (`vcov_NW("id", "t")`).
#' @param unit,time Unit and time variables of Newey-West / Driscoll-Kraay (names or one-sided
#'   formulas); by default the variables of `panel.id`.
#' @param lat,lon Latitude and longitude variables (names or one-sided formulas, degrees).
#' @param type `vcov_hetero()`: `"hc1"`, `"hc2"` or `"hc3"`.
#' @param exact,boot.size `vcov_hetero()`: fereg always computes the exact leverage
#'   (`exact = FALSE` is not supported; `boot.size` is not used).
#' @param cluster Cluster variables: a character vector or a one-sided formula.
#' @param ssc A small-sample correction ([ssc()]); overrides the `ssc` argument of the caller.
#' @param vcov_fix Must be `TRUE` (fereg always applies fixest's positive-definiteness fix).
#' @return A `fixest_vcov_request` object, or the variance matrix of `x`.
#' @examples
#' data(trade_toy, package = "fereg")
#' m <- feols(y ~ x1 | id + year, trade_toy, panel.id = ~id + year, nthreads = 2)
#' summary(m, vcov = NW(1))
#' summary(m, vcov = DK ~ year)
#' summary(m, vcov = vcov_hetero(type = "hc2"))
#' @name vcov_requests
NULL

new_request <- function(vcov, extra_args = NULL, ssc = NULL, vcov_fix = TRUE) {
  structure(list(vcov = vcov, ssc = ssc, extra_args = extra_args, vcov_fix = vcov_fix), class = "fixest_vcov_request")
}

#' @rdname vcov_requests
#' @export
NW <- function(lag = NULL) new_request("NW", list(lag = lag))

#' @rdname vcov_requests
#' @export
newey_west <- function(lag = NULL) new_request("NW", list(lag = lag))

#' @rdname vcov_requests
#' @export
DK <- function(lag = NULL) new_request("DK", list(lag = lag))

#' @rdname vcov_requests
#' @export
driscoll_kraay <- function(lag = NULL) new_request("DK", list(lag = lag))

#' @rdname vcov_requests
#' @export
conley <- function(cutoff = NULL, pixel = NULL, distance = NULL) {
  new_request("conley", list(cutoff = cutoff, pixel = pixel, distance = distance))
}

#' @rdname vcov_requests
#' @export
vcov_hetero <- function(x, type = "hc1", exact = TRUE, boot.size = NULL, ssc = NULL, vcov_fix = TRUE) {
  if (missing(x)) x <- NULL
  slide_args(x, c("type", "exact", "boot.size", "ssc"))
  if (!isTRUE(exact)) stop("fereg: only the exact leverage is computed (exact = FALSE is not supported)", call. = FALSE)
  request_or_vcov(x, new_request(type, list(), ssc, vcov_fix))
}

#' @rdname vcov_requests
#' @export
vcov_cluster <- function(x, cluster = NULL, ssc = NULL, vcov_fix = TRUE) {
  if (missing(x)) x <- NULL
  slide_args(x, c("cluster", "ssc"))
  vcov <- if (is.null(cluster)) "cluster" else if (inherits(cluster, "formula")) cluster else request_formula("cluster", cluster, "cluster")
  request_or_vcov(x, new_request(vcov, NULL, ssc, vcov_fix))
}

#' @rdname vcov_requests
#' @export
vcov_crv3 <- function(x, cluster = NULL, ssc = NULL) {
  if (missing(x)) x <- NULL
  slide_args(x, c("cluster", "ssc"))
  vcov <- if (is.null(cluster)) "crv3" else request_formula("crv3", cluster, "cluster")
  request_or_vcov(x, new_request(vcov, NULL, ssc))
}

#' @rdname vcov_requests
#' @export
vcov_NW <- function(x, unit = NULL, time = NULL, lag = NULL, ssc = NULL, vcov_fix = TRUE) {
  if (missing(x)) x <- NULL
  slide_args(x, c("unit", "time", "lag", "ssc"))
  check_lag(lag)
  vars <- c(one_var(unit, "unit"), one_var(time, "time"))
  request_or_vcov(x, new_request(request_formula("nw", vars), list(lag = lag), ssc, vcov_fix))
}

#' @rdname vcov_requests
#' @export
vcov_DK <- function(x, time = NULL, lag = NULL, ssc = NULL, vcov_fix = TRUE) {
  if (missing(x)) x <- NULL
  slide_args(x, c("time", "lag", "ssc"))
  check_lag(lag)
  request_or_vcov(x, new_request(request_formula("dk", one_var(time, "time")), list(lag = lag), ssc, vcov_fix))
}

#' @rdname vcov_requests
#' @export
vcov_conley <- function(x, lat = NULL, lon = NULL, cutoff = NULL, pixel = 0, distance = "triangular",
                        ssc = NULL, vcov_fix = TRUE) {
  if (missing(x)) x <- NULL
  slide_args(x, c("lat", "lon", "cutoff", "pixel", "distance", "ssc"))
  vars <- c(one_var(lat, "lat"), one_var(lon, "lon"))
  extra <- list(cutoff = check_cutoff(cutoff), pixel = check_pixel(pixel), distance = match_distance(distance))
  request_or_vcov(x, new_request(request_formula("conley", vars), extra, ssc, vcov_fix))
}

# fixest's slide_args(): when the first argument is not an estimation, the positional arguments
# shift one slot to the right (`vcov_NW("id", "period")` = unit "id", time "period").
slide_args <- function(x, slots) {
  if (is.null(x) || inherits(x, "fereg")) return(invisible())
  env <- parent.frame()
  sc <- sys.call(-1)
  mc <- match.call(sys.function(-1), sc)
  named <- names(sc)[-1]
  named <- if (is.null(named)) character(0) else named[nzchar(named)]
  if ("x" %in% named) stop("fereg: `x` must be an estimation (a fereg object)", call. = FALSE)
  implicit <- setdiff(names(mc)[-1], c(named, ""))
  remaining <- setdiff(slots, named)
  values <- mget(implicit, envir = env)
  if (length(implicit) > length(remaining)) stop("fereg: too many arguments", call. = FALSE)
  for (i in seq_along(implicit)) assign(remaining[i], values[[i]], envir = env)
  assign("x", NULL, envir = env)
  invisible()
}

request_or_vcov <- function(x, request) {
  if (is.null(x)) return(request)
  vcov.fereg(x, vcov = request, attr = TRUE)
}

# One variable given as a name or a one-sided formula (fixest: "NULL character scalar | os formula").
one_var <- function(v, what) {
  if (is.null(v)) return(NULL)
  if (inherits(v, "formula")) {
    vars <- all.vars(v[[length(v)]])
    if (length(v) != 2 || length(vars) != 1) {
      stop("The argument '", what, "' must be composed of only one variable.", call. = FALSE)
    }
    return(vars)
  }
  if (!is.character(v) || length(v) != 1 || is.na(v)) {
    stop("fereg: `", what, "` must be a variable name or a one-sided formula", call. = FALSE)
  }
  v
}

# `lhs ~ v1 + v2` from variable names / one-sided formulas (fixest's .xpd()); `lhs ~ 1` when there
# are none.
request_formula <- function(lhs, vars, what = NULL) {
  if (inherits(vars, "formula")) vars <- deparse1l(vars[[length(vars)]])
  if (!is.null(what) && length(vars) > 0 && (!is.character(vars) || anyNA(vars))) {
    stop("fereg: `", what, "` must be variable names or a one-sided formula (vectors of values are not supported)", call. = FALSE)
  }
  rhs <- if (length(vars) == 0) 1 else Reduce(function(a, b) call("+", a, b), lapply(vars, str2lang))
  eval(call("~", as.name(lhs), rhs), baseenv())
}

check_lag <- function(lag) {
  if (is.null(lag)) return(invisible())
  if (!is.numeric(lag) || length(lag) != 1 || is.na(lag) || lag != round(lag) || lag < 1) {
    problem <- if (is.numeric(lag) && length(lag) == 1 && !is.na(lag) && lag < 1) sprintf("\nPROBLEM: it is strictly lower than 1 (%s).", format(lag)) else ""
    stop("The (nullable) value `lag` must be an integer scalar greater than, or equal to, 1.", problem, call. = FALSE)
  }
  invisible()
}

# fixest check_set_cutoff(): km as a number, or "100", "100km", "100mi" (miles x 1.60934). The
# result carries the unit of the label in attribute "metric".
check_cutoff <- function(cutoff) {
  if (is.null(cutoff)) return(NULL)
  if (!is.null(attr(cutoff, "metric"))) return(cutoff)
  ok_num <- is.numeric(cutoff) && length(cutoff) == 1 && !is.na(cutoff) && cutoff >= 0
  if (!ok_num && !(is.character(cutoff) && length(cutoff) == 1 && !is.na(cutoff))) {
    problem <- if (is.numeric(cutoff) && length(cutoff) == 1 && !is.na(cutoff)) {
      sprintf("\nPROBLEM: it is a scalar but a) it is strictly lower than 0 (%s), or b) it is not of type character (instead it is of type 'numeric').", format(cutoff))
    } else {
      ""
    }
    stop("In vcov.fixest, Conley VCOV cannot be computed: the argument 'cutoff' must be a numeric for kilometers, or a number in character form with the 'mi' suffix for miles (ex: 100 means 100km, '100mi' means 100 miles).", problem, call. = FALSE)
  }
  metric <- "km"
  if (is.character(cutoff)) {
    if (!grepl("^[[:digit:]]+ ?((m|mi|mil|mile|miles)|(k|km))?$", cutoff)) {
      stop("In vcov.fixest, Conley VCOV cannot be computed: the argument 'cutoff', equal to '", cutoff,
        "' is not valid. It must be either a numeric scalar or a number in character form with the suffix 'mi' (ex: 100 means 100km, '100mi' means 100 miles).", call. = FALSE)
    }
    if (grepl("k", cutoff)) {
      cutoff <- as.numeric(gsub("k.*", "", cutoff))
    } else if (grepl("m", cutoff)) {
      cutoff <- as.numeric(gsub("m.*", "", cutoff)) * 1.60934
      metric <- "mi"
    } else {
      cutoff <- as.numeric(cutoff)
    }
  }
  structure(as.double(cutoff), metric = metric)
}

check_pixel <- function(pixel) {
  if (is.null(pixel)) return(0)
  if (!is.numeric(pixel) || length(pixel) != 1 || is.na(pixel) || pixel < 0) {
    problem <- if (is.numeric(pixel) && length(pixel) == 1 && !is.na(pixel)) sprintf("\nPROBLEM: it is strictly lower than 0 (%s).", format(pixel)) else ""
    stop("In vcov.fixest, Conley VCOV cannot be computed: the argument 'pixel' must be a numeric scalar with positive values.", problem, call. = FALSE)
  }
  as.double(pixel)
}

match_distance <- function(distance) {
  if (is.null(distance)) return("triangular")
  m <- if (is.character(distance) && length(distance) == 1 && !is.na(distance)) match_choice(distance, c("triangular", "spherical")) else list(problem = "it is not a single character string")
  if (!is.null(m$problem)) {
    stop("Value `distance` must be a single character equal to 'triangular' or 'spherical'.\nPROBLEM: ", m$problem, ".", call. = FALSE)
  }
  m$value
}

# ---- parsing -------------------------------------------------------------------------------------

# fixest's built-in vcov names (dreamerr "match" choices, in fixest's order) plus fereg's "crv3".
vcov_choices <- c(
  "iid", "normal", "standard", "hetero", "white", "hc1", "hc2", "hc3", "cluster", "twoway",
  "threeway", "fourway", "NW", "newey_west", "DK", "driscoll_kraay", "conley", "crv3"
)
vcov_types <- c(
  iid = "iid", normal = "iid", standard = "iid", hetero = "hetero", white = "hetero", hc1 = "hetero",
  hc2 = "hc2", hc3 = "hc3", cluster = "cluster", twoway = "cluster", threeway = "cluster",
  fourway = "cluster", NW = "nw", newey_west = "nw", DK = "dk", driscoll_kraay = "dk",
  conley = "conley", crv3 = "crv3"
)

# dreamerr's "match": exact or unique partial match, then the same ignoring case (an exact match
# ignoring case wins among several partial ones). list(value) or list(problem).
match_choice <- function(x, choices) {
  k <- pmatch(x, choices, duplicates.ok = TRUE)
  if (!is.na(k)) return(list(value = choices[k]))
  if (!nzchar(x)) return(list(problem = "empty strings are not accepted"))
  ok <- startsWith(tolower(choices), tolower(x))
  if (!any(ok)) return(list(problem = sprintf("no match was found for '%s'", x)))
  if (sum(ok) == 1) return(list(value = choices[ok]))
  bis <- ok & nchar(choices) == nchar(x)
  if (sum(bis) == 1) return(list(value = choices[bis]))
  if (sum(bis) == 0) return(list(problem = sprintf("more than one value was matched for '%s': %s", x, enum_quote(choices[ok]))))
  ter <- choices == x
  if (sum(ter) == 1) return(list(value = choices[ter]))
  list(problem = sprintf("more than one value was matched for '%s': %s, please check the case", x, enum_quote(choices[bis])))
}

# Number of variables each type accepts in `type ~ vars` (fixest's `patterns`), and their text.
vcov_patterns <- list(
  cluster = c("", "cl1", "cl1 + cl2", "cl1 + cl2 + cl3", "c1 + cl2 + cl3 + cl4"),
  twoway = c("", "cl1 + cl2"), threeway = c("", "cl1 + cl2 + cl3"), fourway = c("", "cl1 + cl2 + cl3 + cl4"),
  nw = c("", "time", "unit + time"), dk = c("", "time"), conley = c("", "lat + lng")
)

vcov_label_base <- c(
  iid = "IID", hetero = "Heteroskedasticity-robust", hc2 = "HC2", hc3 = "HC3", cluster = "Clustered",
  crv3 = "CRV3", nw = "Newey-West", dk = "Driscoll-Kraay", conley = "Conley"
)

# Parses vcov / cluster / se (fixest's oldargs_to_vcov + vcov.fixest parsing) into a request:
#   type      iid, hetero, hc2, hc3, cluster, crv3, nw, dk or conley
#   dims      cluster / crv3: list of dimensions (each a list of expressions, interacted when
#             > 1), the number of leading fixed effects (twoway, ...), or NULL (guess: first
#             panel.id variable, else first fixed effect)
#   vars      nw: unit, time; dk: time; conley: lat, lon (expressions, NULL = guess)
#   lag, cutoff (km, attribute "metric"), pixel, distance, ssc (from the request), resolved
parse_vcov <- function(vcov = NULL, cluster = NULL, se = NULL) {
  if (!is.null(se) || !is.null(cluster)) {
    if (!is.null(vcov)) {
      used <- c("se", "cluster")[c(!is.null(se), !is.null(cluster))]
      stop("You cannot use the argument 'vcov' in combination with ", paste0("'", used, "'", collapse = " or "), ".", call. = FALSE)
    }
    se_name <- if (is.null(se)) "cluster" else match_vcov_name(se, "Argument 'se' (which has been replaced by arg. 'vcov')")
    if (is.null(cluster)) {
      vcov <- se_name
    } else if (!se_name %in% c("cluster", "twoway", "threeway", "fourway")) {
      stop("The VCOV requested with the argument se = '", se, "' is not compatible with the use of the argument 'cluster' (i.e. you have to choose!).", call. = FALSE)
    } else if (inherits(cluster, "formula")) {
      vcov <- cluster
    } else {
      vcov <- request_formula("cluster", cluster, "cluster")
    }
  }
  if (is.null(vcov)) return(new_vcov("iid"))
  req_ssc <- NULL
  extra <- list()
  if (inherits(vcov, "fixest_vcov_request")) {
    if (!is.null(vcov$vcov_vars)) {
      stop("fereg: clustering on vectors of values (vcov_cluster(cluster = <vectors>)) is not supported; use variable names", call. = FALSE)
    }
    if (isFALSE(vcov$vcov_fix)) {
      stop("fereg: vcov_fix = FALSE is not supported (fixest's positive-definiteness fix is always applied)", call. = FALSE)
    }
    req_ssc <- vcov$ssc
    extra <- vcov$extra_args %||% list()
    vcov <- vcov$vcov
  }
  if (inherits(vcov, "formula")) {
    out <- parse_vcov_formula(vcov)
  } else {
    if (!is.character(vcov) || length(vcov) != 1 || is.na(vcov)) {
      stop("Argument `vcov` must be either: i) a single character equal to ", vcov_choices_text(),
        ", ii) a formula, or iii) an object of class 'fixest_vcov_request'.", call. = FALSE)
    }
    out <- new_vcov(match_vcov_name(vcov))
  }
  if (!is.null(req_ssc)) out$ssc <- req_ssc
  apply_extra(out, extra)
}

vcov_choices_text <- function() {
  paste0(paste0("'", vcov_choices[1:6], "'", collapse = ", "), " or ", length(vcov_choices) - 6, " others")
}

match_vcov_name <- function(x, prefix = NULL) {
  m <- match_choice(x, vcov_choices)
  if (is.null(m$problem)) return(m$value)
  if (is.null(prefix)) {
    stop("Argument `vcov` must be either: i) a single character equal to ", vcov_choices_text(),
      ", ii) a formula, or iii) an object of class 'fixest_vcov_request'.\nPROBLEM: ", m$problem, ".", call. = FALSE)
  }
  stop(prefix, " must be a single character equal to ", vcov_choices_text(), ".\nPROBLEM: ", m$problem, ".", call. = FALSE)
}

# A request of the vcov name `name` (one of vcov_choices) without variables.
new_vcov <- function(name) {
  type <- unname(vcov_types[name])
  v <- list(type = type, name = name, dims = NULL, vars = list(), lag = NULL, cutoff = NULL, pixel = 0,
    distance = "triangular", ssc = NULL, resolved = FALSE)
  if (name %in% c("twoway", "threeway", "fourway")) v$dims <- match(name, c("cluster", "twoway", "threeway", "fourway"))
  if (type %in% c("iid", "hetero", "hc2", "hc3")) v$resolved <- TRUE
  v
}

# `type ~ vars` (type possibly with arguments: `NW(2) ~ id + period`, `conley(100) ~ lat + lon`),
# or a one-sided formula `~ cl1 + cl2` (clustering). `ssc(...)` among the variables sets the ssc.
parse_vcov_formula <- function(f) {
  extra <- list()
  if (length(f) == 2) {
    v <- new_vcov("cluster")
    rhs <- f[[2]]
  } else {
    lhs <- f[[2]]
    lhs_txt <- deparse1l(lhs)
    is_extra <- grepl("(", lhs_txt, fixed = TRUE)
    name <- if (is_extra) trimws(gsub("\\(.+", "", lhs_txt)) else lhs_txt
    m <- match_choice(name, vcov_choices)
    if (!is.null(m$problem)) {
      stop("If a formula, the arg. 'vcov' must be of the form 'vcov_type ~ vars'. The `vcov_type` must be a supported VCOV type.\nPROBLEM: ",
        m$problem, ".", call. = FALSE)
    }
    v <- new_vcov(m$value)
    if (is_extra) {
      req <- eval(lhs, request_env(environment(f)))
      if (!inherits(req, "fixest_vcov_request")) stop("fereg: `", lhs_txt, "` must be a variance request such as NW(2)", call. = FALSE)
      extra <- req$extra_args %||% list()
      if (!is.null(req$ssc)) v$ssc <- req$ssc
    }
    rhs <- f[[3]]
  }
  terms <- lapply(split_plus(rhs), `[[`, "expr")
  is_ssc <- vapply(terms, function(e) is_call_to(e, "ssc"), TRUE)
  for (e in terms[is_ssc]) v$ssc <- eval(e, request_env(environment(f)))
  terms <- terms[!is_ssc]
  terms <- Filter(function(e) !(is.numeric(e) && length(e) == 1), terms) # `hetero ~ 1`
  set_vcov_vars(v, terms, extra)
}

# Evaluation environment of request calls in formulas: fereg's request functions first.
request_env <- function(parent) {
  list2env(list(NW = NW, newey_west = newey_west, DK = DK, driscoll_kraay = driscoll_kraay, conley = conley, ssc = ssc),
    parent = if (is.null(parent)) globalenv() else parent)
}

# Puts the formula variables of request `v` in place, checking their number (fixest patterns).
set_vcov_vars <- function(v, terms, extra) {
  key <- if (v$name %in% c("twoway", "threeway", "fourway")) v$name else if (v$type == "crv3") "cluster" else v$type
  patterns <- vcov_patterns[[key]]
  if (!is.null(patterns) && length(terms) > 0) {
    n_pat <- lengths(strsplit(patterns, " ?\\+ ?"))
    if (!length(terms) %in% n_pat) {
      shown <- paste0("`~ ", patterns[n_pat != 0], "`")
      enum <- if (length(shown) == 1) shown else paste0(paste(shown[-length(shown)], collapse = ", "), " or ", shown[length(shown)])
      stop("In the argument 'vcov', the number of variables in the RHS of the formula (", length(terms),
        ") is not valid. The formula should correspond to ", enum, ".", call. = FALSE)
    }
  }
  if (length(terms) > 0) {
    switch(v$type,
      cluster = ,
      crv3 = {
        v$dims <- lapply(terms, split_hat)
      },
      nw = {
        if (length(terms) == 2) v$vars$unit <- terms[[1]]
        v$vars$time <- terms[[length(terms)]]
      },
      dk = {
        v$vars$time <- terms[[1]]
      },
      conley = {
        v$vars$lat <- terms[[1]]
        v$vars$lon <- terms[[2]]
      }
    )
  }
  apply_extra(v, extra)
}

# Request arguments (lag, cutoff, pixel, distance) of NW() / DK() / conley() objects.
apply_extra <- function(v, extra) {
  if (!is.null(extra$lag)) {
    lag <- extra$lag
    if (!is.numeric(lag) || length(lag) != 1 || is.na(lag) || lag < 0 || lag != round(lag)) {
      stop("fereg: the lag must be a non-negative integer", call. = FALSE)
    }
    v$lag <- lag
  }
  if (v$type == "conley") {
    if (!is.null(extra$cutoff)) v$cutoff <- check_cutoff(extra$cutoff)
    if (!is.null(extra$pixel)) v$pixel <- check_pixel(extra$pixel)
    if (!is.null(extra$distance)) v$distance <- match_distance(extra$distance)
  }
  v
}

# ---- resolution against a model ------------------------------------------------------------------

# Fills the variables a request leaves to fixest's guesses: clusters (first panel.id variable,
# else the first fixed effect; twoway... = the first k fixed effects), Newey-West unit and time
# (panel.id), Driscoll-Kraay time (panel.id), Conley latitude and longitude (column names).
#   fe_terms    fixed-effect terms of the model
#   panel_id    character(2) or NULL
#   data_names  column names of the estimation data (NULL: no check, no guess)
resolve_vcov <- function(v, fe_terms, panel_id = NULL, data_names = NULL) {
  if (isTRUE(v$resolved)) return(v)
  label <- vcov_label_base[[v$type]]
  if (v$type %in% c("cluster", "crv3")) {
    if (is.null(v$dims)) {
      if (!is.null(panel_id)) {
        v$dims <- list(list(str2lang(panel_id[1])))
      } else {
        fe_int <- Filter(function(t) t$intercept, fe_terms)
        if (length(fe_int) == 0) {
          stop("To compute the ", label, " VCOV, we need a variable for the clusters. Since you didn't provide it in the formula, ",
            "we typically deduce it from a) the 'panel.id' identifiers, or b) the fixed-effects. PROBLEM: a) no 'panel.id' was set in this estimation, ",
            "and b) there was no fixed-effect in the estimation. Please provide it in the formula.", call. = FALSE)
        }
        v$dims <- list(fe_int[[1]]$vars)
      }
    } else if (!is.list(v$dims)) {
      k <- v$dims
      fe_int <- Filter(function(t) t$intercept, fe_terms)
      if (length(fe_int) < k) {
        stop("fereg: vcov = \"", c("cluster", "twoway", "threeway", "fourway")[k], "\" needs ", k,
          " fixed effect", if (k > 1) "s", "; use a formula such as vcov = ~cl", call. = FALSE)
      }
      v$dims <- lapply(fe_int[seq_len(k)], `[[`, "vars")
    }
  }
  guess_panel <- function(i, var_label, optional) {
    if (is.null(panel_id)) {
      if (optional) return(NULL)
      stop("To compute the ", label, " VCOV, we need a variable for the ", var_label, ". Since you didn't provide it in the formula, ",
        "we typically deduce it from the 'panel.id' identifiers. PROBLEM: no 'panel.id' was set in this estimation. Please provide it in the formula.",
        call. = FALSE)
    }
    if (!is.null(data_names) && !panel_id[i] %in% data_names) {
      if (optional) return(NULL)
      stop("To compute the ", label, " VCOV, we need a variable for the ", var_label, ". Since you didn't provide it in the formula, ",
        "we typically deduce it from the 'panel.id' identifiers. PROBLEM: the variable '", panel_id[i], "' set in 'panel.id' is not in the data set. ",
        "Please provide it in the formula.", call. = FALSE)
    }
    as.name(panel_id[i])
  }
  if (v$type == "nw") {
    if (is.null(v$vars$unit)) v$vars["unit"] <- list(guess_panel(1, "panel ID", TRUE))
    if (is.null(v$vars$time)) v$vars$time <- guess_panel(2, "time", FALSE)
  } else if (v$type == "dk") {
    if (is.null(v$vars$time)) v$vars$time <- guess_panel(2, "time", FALSE)
  } else if (v$type == "conley") {
    if (is.null(v$vars$lat)) v$vars$lat <- guess_coord(data_names, "latitude", c("^lat(itude)?$", "^lat_.+"))
    if (is.null(v$vars$lon)) v$vars$lon <- guess_coord(data_names, "longitude", c("^(lon|lng|long|longitude)$", "^(lng|lon|long)_.+"))
  }
  if (v$type %in% c("nw", "dk", "conley") && !is.null(data_names)) {
    used <- unique(unlist(lapply(Filter(Negate(is.null), v$vars), all.vars)))
    missing_vars <- setdiff(used, data_names)
    if (length(missing_vars)) {
      k <- length(missing_vars)
      stop("The variable", if (k > 1) "s", " ", enum_quote(missing_vars), ", used to compute the VCOV, ", if (k > 1) "are" else "is",
        " not in the original data set. Only variables in the data set can be used.", call. = FALSE)
    }
  }
  v$resolved <- TRUE
  v
}

# fixest's guess of the Conley coordinates from the column names (`guess_from = regex`).
guess_coord <- function(data_names, label, patterns) {
  msg <- NULL
  for (p in patterns) {
    hit <- grep(p, data_names)
    if (length(hit) == 1) return(as.name(data_names[hit]))
    if (length(hit) == 0) {
      if (is.null(msg)) msg <- paste0("no match was found for ", label)
    } else {
      msg <- paste0("several matches were found: ", enum_roman(data_names[hit]))
    }
  }
  stop("To compute the Conley VCOV, we need a variable for the ", label, ". Since you didn't provide it in the formula, ",
    "we typically deduce it from the variable names of the data set. PROBLEM: ", msg, ". Please provide it in the formula.", call. = FALSE)
}

# dreamerr's "enum.i.q": "i) 'a', and ii) 'b'".
enum_roman <- function(x) {
  roman <- c("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x")
  items <- paste0(c(roman, rep("..", max(0, length(x) - 10)))[seq_along(x)], ") '", x, "'")
  if (length(items) == 1) return(items)
  paste0(paste(items[-length(items)], collapse = ", "), ", and ", items[length(items)])
}

# The columns of a resolved request, registered in the factor registry `reg`:
# list(cluster = factor indices per dimension, time, panel = factor indices or NULL,
#      lat, lon = double vectors or NULL).
vcov_columns <- function(v, reg) {
  out <- list(cluster = list(), time = NULL, panel = NULL, lat = NULL, lon = NULL)
  if (v$type %in% c("cluster", "crv3")) {
    out$cluster <- lapply(v$dims, function(d) vapply(d, function(e) factor_index(reg, e), 0))
  } else if (v$type %in% c("nw", "dk")) {
    if (!is.null(v$vars$unit)) out$panel <- factor_index(reg, v$vars$unit)
    out$time <- time_index(reg, v$vars$time)
  } else if (v$type == "conley") {
    out$lat <- coord_values(reg, v$vars$lat, "latitude")
    out$lon <- coord_values(reg, v$vars$lon, "longitude")
  }
  out
}

# Time variable as a factor column whose level order is the time order (fixest: to_integer(time,
# sorted = TRUE)): numbers and dates by value, character strings in C-locale order, factors by
# level order.
time_index <- function(reg, expr) {
  name <- deparse1l(expr)
  e <- cached(reg$cache, paste0("t:", name), function() {
    v <- eval_var(expr, reg$data, reg$env)
    col <- if (is.character(v)) {
      u <- sort(unique(v[!is.na(v)]), method = "radix")
      structure(match(v, u), levels = u)
    } else {
      factor_column(v, name)
    }
    list(value = v, col = col)
  })
  if (is.numeric(e$value) && !is.factor(e$value)) return(factor_index(reg, expr))
  k <- match(name, reg$time_names)
  if (!is.na(k)) return(reg$time_idx[k])
  reg$names <- c(reg$names, name)
  reg$values[[length(reg$values) + 1]] <- e$value
  reg$cols[[length(reg$cols) + 1]] <- e$col
  reg$time_names <- c(reg$time_names, name)
  reg$time_idx <- c(reg$time_idx, length(reg$names))
  length(reg$names)
}

coord_values <- function(reg, expr, label) {
  name <- deparse1l(expr)
  v <- eval_var(expr, reg$data, reg$env)
  if (!is.numeric(v)) {
    stop("To compute the VCOV, the ", label, " (here '", name, "') must be a numeric vector.\nPROBLEM: it is not numeric (instead it is of type '",
      typeof(v), "').", call. = FALSE)
  }
  as.double(v)
}

# The request for the core: list(type, lag, cutoff (km), pixel (km), spherical).
core_kind <- function(v) {
  km <- function(x) if (identical(attr(v$cutoff, "metric"), "mi")) x * 1.60934 else x
  list(
    type = v$type,
    lag = if (is.null(v$lag)) NA_real_ else as.double(v$lag),
    cutoff = if (is.null(v$cutoff)) NA_real_ else as.double(v$cutoff),
    pixel = as.double(km(v$pixel)),
    spherical = identical(v$distance, "spherical")
  )
}

# ---- labels --------------------------------------------------------------------------------------

dim_label <- function(d) paste(vapply(d, deparse1l, ""), collapse = "^")

# fixest's "Standard-errors:" text; `vout` (the core's output) gives the resolved lag and
# cutoff.
vcov_label <- function(v, vout = NULL) {
  dims <- function() paste(vapply(v$dims, dim_label, ""), collapse = " & ")
  switch(v$type,
    iid = "IID",
    hetero = "Heteroskedasticity-robust",
    hc2 = "HC2",
    hc3 = "HC3",
    cluster = paste0("Clustered (", dims(), ")"),
    crv3 = paste0("CRV3 (", dims(), ")"),
    nw = paste0("Newey-West (L=", format_num(if (is.null(v$lag)) vout$lag else v$lag), ")"),
    dk = paste0("Driscoll-Kraay (L=", format_num(if (is.null(v$lag)) vout$lag else v$lag), ")"),
    conley = {
      metric <- attr(v$cutoff, "metric") %||% "km"
      km <- if (is.null(v$cutoff)) vout$cutoff else as.double(v$cutoff)
      paste0("Conley (", format_num(if (metric == "mi") km / 1.60934 else km), metric, ")")
    }
  )
}

# R's paste0() text of a number (fixest builds its labels with paste0()).
format_num <- function(x) paste0(as.double(x))

# fixest's format_se_type(): the "S.E. type" cell of etable, shortened to `width` characters.
format_se_type <- function(x, width, by = TRUE) {
  if (!grepl("(", x, fixed = TRUE) || !grepl("Clustered", x, fixed = TRUE)) {
    if (nchar(x) <= width) return(x)
    x_split <- c("$", strsplit(x, "")[[1]])
    x_new <- x_split
    end_word <- length(x_split)
    non_letter <- grepl("[^[:alpha:]]", x_split) * seq_len(end_word)
    letter <- grepl("[[:alpha:]]", x_split) * seq_len(end_word)
    repeat {
      start_word <- which.max(non_letter[1:end_word]) + 1
      word_length <- end_word - start_word + 1
      slack <- length(x_new) - (width + 1)
      to_rm <- min(word_length - 4, slack)
      if (to_rm > 0) {
        i_max <- end_word - to_rm
        x_new <- x_new[-((i_max + 1):end_word)]
        x_new[i_max] <- "."
      }
      lf <- letter[1:(start_word - 1)]
      if (all(lf == 0)) break
      end_word <- which.max(lf)
    }
    return(paste(x_new[-1], collapse = ""))
  }
  fes <- gsub(" ", "", strsplit(gsub(".+\\((.+)\\)", "\\1", x), "&")[[1]])
  fes <- gsub("^", "-", fes, fixed = TRUE)
  n_fe <- length(fes)
  n_char <- nchar(fes)
  if (n_fe == 1 && !grepl("\\^", fes[1])) {
    out <- paste0(if (by) "by: " else "1-way: ", fes[1])
    if (nchar(out) > width) out <- paste0(substr(out, 1, width - 2), "..")
    return(out)
  }
  nb <- if (by) 3 else 6
  if (width < nb + sum(n_char) + (n_fe - 1) * 3) {
    for (i in which(n_char > 5)) fes[i] <- paste0(substr(fes[i], 1, 4), ".")
  }
  if (by) paste0("by: ", paste(fes, collapse = " & ")) else paste0(n_fe, "-way: ", paste(fes, collapse = " & "))
}

ssc_core <- function(s) unclass(s)

# ---- panel.id ------------------------------------------------------------------------------------

# panel.id = ~id + period, "id,period" or c("id", "period") -> its variable names, checked
# against the data (fixest's messages).
parse_panel_id <- function(panel.id, data) {
  if (is.null(panel.id)) return(NULL)
  vars <- if (inherits(panel.id, "formula")) {
    all.vars(panel.id[[length(panel.id)]])
  } else if (is.character(panel.id) && !anyNA(panel.id)) {
    trimws(unlist(strsplit(panel.id, ",", fixed = TRUE)))
  } else {
    stop("fereg: `panel.id` must be a formula such as ~id + period or a string such as \"id,period\"", call. = FALSE)
  }
  bad <- setdiff(vars, names(data))
  if (length(bad)) {
    k <- length(bad)
    stop("Problem in the argument 'panel.id': the variable", if (k > 1) "s", " ", enum_quote(bad), " ", if (k > 1) "are" else "is",
      " not in the data set.", call. = FALSE)
  }
  if (length(vars) < 2) {
    stop("fereg: `panel.id` must give the panel (unit) and the time variables, as in panel.id = ~id + period", call. = FALSE)
  }
  vars
}
