# Ground truth for crates/fereg-core/tests/fit_se.rs (fixest flavor): HC2, HC3, Newey-West (panel and
# time series), Driscoll-Kraay, Conley and CRV3 with fixest 0.14.2 on an unbalanced panel with gaps.
#
#   R_LIBS=~/fereg-dev/rlib Rscript crates/fereg-core/tests/data/gen_se_fixtures.R   (repository root)
#
# Writes tests/data/fit_se.csv (the data, also read by gen_se_fixtures.py), tests/data/fit_se_fixest.json
# and tests/data/fit_se_pyconley.json (the Pyfixest-flavor Conley cases). Each case holds the
# request (`vcov`: a string or {type, lag, cutoff, pixel, spherical}; `time`, `panel`, `lat`, `lon`
# variable names) and fixest's
# variance matrix with its df.K, df.t, resolved lag / cutoff and number of periods. fixest has no
# CRV3: the fixest-flavor CRV3 (fereg's extension) is a brute-force leave-one-cluster-out loop of
# fixest refits with fixest's cluster small-sample correction (G.df = "min", K.fixef =
# "nonnested", inclusion-exclusion over the dimensions) and fixest's PSD fix.
suppressPackageStartupMessages({
  library(fixest)
  library(jsonlite)
})
stopifnot(as.character(packageVersion("fixest")) == "0.14.2")
dir <- "crates/fereg-core/tests/data"
setFixest_nthreads(1)
TOL <- 3e-12

set.seed(20261013)
n_id <- 60
years <- 2001:2016
rows <- do.call(rbind, lapply(1:n_id, function(i) {
  ti <- sort(sample(years, sample(5:14, 1)))           # unbalanced, with calendar gaps
  data.frame(id = i, t = ti)
}))
n <- nrow(rows)
d <- rows
d$ts <- sample(seq_len(2 * n), n)                       # time series: unique times with gaps
d$g <- sample(1:30, n, TRUE)
d$h <- sample(1:8, n, TRUE)
d$k <- sample(1:4, n, TRUE)
d$c1 <- (d$g - 1) %/% 3
d$c3 <- sample(1:15, n, TRUE)
ue <- rnorm(n_id)
# AR(1) errors and regressors within the unit (HAC matters).
ar <- function(v, rho) { for (j in seq_along(v)[-1]) if (d$id[j] == d$id[j - 1]) v[j] <- rho * v[j - 1] + v[j]; v }
d$x1 <- ar(rnorm(n), 0.6) + 0.3 * ue[d$id]
d$x2 <- rnorm(n) + 0.3 * d$x1
d$s1 <- runif(n, 0, 3)
gm <- rnorm(30)
e <- ar(rnorm(n), 0.5)
d$y <- 1 + 0.5 * d$x1 - 0.8 * d$x2 + gm[d$g] + rnorm(8)[d$h] + rnorm(4)[d$k] + ue[d$id] + rnorm(30)[d$g] * d$s1 + e
d$w <- runif(n, 0.5, 2)
d$wi <- sample(1:3, n, TRUE)
d$z1 <- rnorm(n); d$z2 <- rnorm(n); u <- rnorm(n)
d$xe <- 0.6 * d$z1 + 0.4 * d$z2 + 0.5 * u + gm[d$g] + rnorm(n)
d$yiv <- 1 + d$xe + 0.5 * d$x1 + gm[d$g] + u + e
d$yp <- rpois(n, exp(0.2 + 0.3 * d$x1 - 0.2 * d$x2 + 0.3 * gm[d$g] + 0.2 * e))
d$yb <- rbinom(n, 1, plogis(0.5 * d$x1 - 0.5 * d$x2 + 0.5 * gm[d$g] + 0.3 * e))
# Coordinates per unit (rows of a unit share a location: pixel aggregation), a few moved rows.
lat_u <- runif(n_id, 32, 44); lon_u <- runif(n_id, -100, -80)
d$lat <- lat_u[d$id]; d$lon <- lon_u[d$id]
mv <- sample(n, 40)
d$lat[mv] <- d$lat[mv] + rnorm(40, 0, 0.5); d$lon[mv] <- d$lon[mv] + rnorm(40, 0, 0.5)
# Missing values in vcov variables (fixest drops these rows at estimation).
d$tna <- d$t; d$tna[c(4, 50, 51, 300)] <- NA
d$latna <- d$lat; d$latna[c(9, 120, 121)] <- NA
write.csv(d, file.path(dir, "fit_se.csv"), row.names = FALSE, na = "NA")

num <- function(v) unname(as.numeric(v))
term <- function(factors, slopes = character(), intercept = TRUE) list(factors = as.list(factors), slopes = as.list(slopes), intercept = intercept)
fe_terms <- function(fe) lapply(fe, function(t) if (is.list(t) && !is.null(t$factors)) t else term(unlist(t)))

# fixest's `vcov` argument of a request.
vcov_arg <- function(v, time, panel, lat, lon) {
  if (is.character(v)) return(v)
  lag <- if (is.null(v$lag)) "" else paste0("(", v$lag, ")")
  switch(v$type,
    NW = as.formula(paste0("NW", lag, " ~ ", if (is.null(panel)) time else paste(panel, "+", time))),
    DK = as.formula(paste0("DK", lag, " ~ ", time)),
    conley = {
      a <- c(if (!is.null(v$cutoff)) v$cutoff, if (!is.null(v$pixel)) paste0("pixel = ", v$pixel),
             if (isTRUE(v$spherical)) "distance = \"spherical\"")
      as.formula(paste0("conley(", paste(a, collapse = ", "), ") ~ ", lat, " + ", lon))
    })
}

vcov_fields <- function(V, v) {
  out <- list(vcov = num(V), se = num(sqrt(diag(V))), df_k = attr(V, "df.K"), df_t = attr(V, "df.t"))
  info <- attr(V, "vcov_type")                         # "Newey-West (L=2)", "Conley (250km)"
  if (is.list(v) && v$type %in% c("NW", "DK")) {
    out$lag <- as.numeric(sub(".*\\(L=(.*)\\)$", "\\1", info))
    out$n_periods <- attr(V, "G")
  }
  if (is.list(v) && v$type == "conley") out$cutoff <- as.numeric(sub(".*\\((.*)km\\)$", "\\1", info))
  out
}

fit_model <- function(model, fml, data, vcov, weights, offset = NULL) {
  args <- list(fml = fml, data = data, fixef.tol = TOL, fixef.iter = 100000, vcov = vcov, notes = FALSE)
  if (!is.null(weights)) args$weights <- as.formula(paste0("~", weights))
  if (!is.null(offset)) args$offset <- as.formula(paste0("~", offset))
  glm <- c(args, list(glm.tol = 1e-12, glm.iter = 100))
  switch(model, ols = do.call(feols, args), poisson = do.call(fepois, glm),
         logit = do.call(feglm, c(glm, list(family = "logit"))))
}

# fereg's fixest-flavor CRV3: leave-one-cluster-out refits over every non-empty subset of the
# dimensions (sign (-1)^(|S|+1)), centered at the full-sample coefficients, times fixest's
# cluster correction G_min/(G_min - 1) * (n - 1)/(n - K) with K of fixest's clustered vcov.
crv3_fixest <- function(m, model, fml, dat, weights, cluster) {
  sub <- dat[obs(m), , drop = FALSE]
  beta <- coef(m)
  ids <- lapply(cluster, function(dim) do.call(paste, c(sub[, unlist(dim), drop = FALSE], sep = "_")))
  q <- length(ids)
  meat <- matrix(0, length(beta), length(beta))
  for (size in 1:q) for (S in combn(q, size, simplify = FALSE)) {
    id <- do.call(paste, c(ids[S], sep = "|"))
    for (gv in unique(id)) {
      f <- fit_model(model, fml, sub[id != gv, , drop = FALSE], "iid", weights)
      b <- coef(f)[names(beta)]
      stopifnot(!anyNA(b))
      meat <- meat + (if (size %% 2 == 1) 1 else -1) * tcrossprod(b - beta)
    }
  }
  clf <- as.formula(paste0("~", paste(sapply(cluster, function(dd) paste(unlist(dd), collapse = "^")), collapse = "+")))
  V1 <- vcov(m, vcov = clf, attr = TRUE)
  G <- sapply(ids, function(x) length(unique(x)))
  K <- attr(V1, "df.K"); nn <- nobs(m)
  V <- meat * min(G) / (min(G) - 1) * (nn - 1) / (nn - K)
  if (any(eigen(V, symmetric = TRUE, only.values = TRUE)$values <= 0)) V <- fixest:::mat_posdef_fix(V)
  list(vcov = num(V), se = num(sqrt(diag(V))), df_k = K, df_t = max(min(G) - 1, 1), G = G)
}

# HC1/HC2/HC3 of the dense design (`dense`: the formula with explicit dummies, fitted by lm / glm
# to convergence) through sandwich::vcovHC, the regressors' block (FWL): fixest's own
# `hatvalues` fails with varying slopes (Cholesky of its dummy cross-product) or, for feglm,
# demeans the regressors by the intercept fixed effects only, and its feglm scores with varying
# slopes carry its demeaning tolerance (HC1 off by up to 7e-8). fereg uses the exact values there
# (its documented deviation). HC1 gets fixest's n/(n - K); HC2/HC3 none; fixest's PSD fix.
hc_exact <- function(m, model, type, dense, weights) {
  dd <- d[obs(m), , drop = FALSE]
  fit <- if (model == "ols") {
    lm(as.formula(dense), dd, weights = if (is.null(weights)) NULL else dd[[weights]])
  } else {
    stats::glm(as.formula(dense), if (model == "poisson") poisson() else binomial(), dd,
               control = glm.control(epsilon = 1e-15, maxit = 100))
  }
  Vh <- vcov(m, vcov = "hetero", attr = TRUE)
  x <- names(coef(m))
  V <- if (type == "hetero") {
    sandwich::vcovHC(fit, type = "HC0")[x, x] * nobs(m) / (nobs(m) - attr(Vh, "df.K"))
  } else {
    sandwich::vcovHC(fit, type = toupper(type))[x, x]
  }
  if (any(eigen(V, symmetric = TRUE, only.values = TRUE)$values <= 0)) V <- fixest:::mat_posdef_fix(V)
  list(vcov = num(V), se = num(sqrt(diag(V))), df_k = attr(Vh, "df.K"), df_t = attr(Vh, "df.t"))
}

# One case: the estimate with `vcov` at estimation time.
case <- function(name, fml, y, x, fe, vcov, model = "ols", weights = NULL, endog = character(), inst = character(),
                 time = NULL, panel = NULL, lat = NULL, lon = NULL, cluster = list(), intercept = TRUE, dense = NULL) {
  out <- list(name = name, data = "fit_se.csv", model = model, y = y, x = x, endog = endog, inst = inst, fe = fe_terms(fe),
              intercept = intercept, weights = weights, vcov = vcov, cluster = cluster, time = time, panel = panel,
              lat = lat, lon = lon,
              ssc = list(K.adj = TRUE, K.fixef = "nonnested", K.exact = FALSE, G.adj = TRUE, G.df = "min", t.df = "min"))
  crv3 <- identical(vcov, "crv3")
  exact <- !is.null(dense)
  va <- if (crv3 || exact) "iid" else vcov_arg(vcov, time, panel, lat, lon)
  m <- tryCatch(fit_model(model, fml, d, va, weights), error = function(e) e)
  if (inherits(m, "error")) {
    out$error <- conditionMessage(m)
    return(out)
  }
  V <- vcov(m, attr = TRUE)
  ex <- if (crv3) crv3_fixest(m, model, fml, d, weights, cluster) else if (exact) hc_exact(m, model, vcov, dense, weights) else vcov_fields(V, vcov)
  out$expect <- c(ex, list(coef_names = names(coef(m)), coef = num(coef(m)), nobs = nobs(m), collin = list()))
  if (length(endog) > 0 && !crv3) {
    out$expect$first <- lapply(endog, function(en) {
      f1 <- m$iv_first_stage[[en]]
      c(vcov_fields(vcov(f1, attr = TRUE), vcov), list(coef_names = names(coef(f1)), coef = num(coef(f1))))
    })
  }
  out
}

cases <- list()
add <- function(...) cases[[length(cases) + 1]] <<- case(...)
nw <- function(lag = NULL) list(type = "NW", lag = lag)
dk <- function(lag = NULL) list(type = "DK", lag = lag)
cy <- function(cutoff = NULL, pixel = NULL, spherical = FALSE) list(type = "conley", cutoff = cutoff, pixel = pixel, spherical = spherical)
X2 <- c("x1", "x2")
specs <- list(
  nofe = list(rhs = "x1 + x2", fe = list()),
  fe1 = list(rhs = "x1 + x2 | g", fe = list(list("g"))),
  fe2 = list(rhs = "x1 + x2 | g + h", fe = list(list("g"), list("h"))),
  fe3 = list(rhs = "x1 + x2 | g + h + k", fe = list(list("g"), list("h"), list("k"))),
  sl = list(rhs = "x1 + x2 | g[s1] + h", fe = list(term("g", "s1"), list("h")))
)
fml_of <- function(y, sp) as.formula(paste(y, "~", specs[[sp]]$rhs))
ols <- function(name, sp, vcov, ...) do.call(add, c(list(name = name, fml = fml_of("y", sp), y = "y", x = X2, fe = specs[[sp]]$fe, vcov = vcov), list(...)))

SLD <- "~ x1 + x2 + factor(g) + factor(g):s1 + factor(h)"
for (sp in names(specs)) {
  for (hc in c("hc2", "hc3")) ols(paste0(hc, "_", sp), sp, hc, dense = if (sp == "sl") paste("y", SLD))
  ols(paste0("nw_", sp), sp, nw(), time = "t", panel = "id")
  ols(paste0("dk_", sp), sp, dk(), time = "t")
  ols(paste0("conley_", sp), sp, cy(), lat = "lat", lon = "lon")
}
for (sp in c("nofe", "fe2")) {
  ols(paste0("hc2_w_", sp), sp, "hc2", weights = "w")
  ols(paste0("hc3_w_", sp), sp, "hc3", weights = "w")
  ols(paste0("nw_w_", sp), sp, nw(), time = "t", panel = "id", weights = "w")
  ols(paste0("dk_w_", sp), sp, dk(1), time = "t", weights = "w")
  ols(paste0("conley300_w_", sp), sp, cy(300), lat = "lat", lon = "lon", weights = "w")
  ols(paste0("nwts_", sp), sp, nw(), time = "ts")
  ols(paste0("nwts3_", sp), sp, nw(3), time = "ts")
  ols(paste0("crv3_c3_", sp), sp, "crv3", cluster = list(list("c3")))
}
ols("hc2_w_sl", "sl", "hc2", weights = "w", dense = paste("y", SLD))
ols("nw3_fe2", "fe2", nw(3), time = "t", panel = "id")
ols("nw40_fe1", "fe1", nw(40), time = "t", panel = "id")           # lag beyond T - 2
ols("dk2_fe3", "fe3", dk(2), time = "t")
ols("dk40_fe1", "fe1", dk(40), time = "t")
ols("conley_sph_px_fe1", "fe1", cy(300, 50, TRUE), lat = "lat", lon = "lon")
ols("conley_px_fe2", "fe2", cy(200, 30), lat = "lat", lon = "lon")
ols("crv3_c3_h_fe2", "fe2", "crv3", cluster = list(list("c3"), list("h")))
ols("crv3_g_fe1_w", "fe1", "crv3", cluster = list(list("g")), weights = "w")
ols("crv3_c1_sl", "sl", "crv3", cluster = list(list("c1")))
# Rows missing a vcov variable are dropped at estimation.
ols("nw_tna_fe2", "fe2", nw(), time = "tna", panel = "id")
ols("conley_latna_fe1", "fe1", cy(250), lat = "latna", lon = "lon")
# IV.
FIV <- yiv ~ x1 | g + h | xe ~ z1 + z2
iv <- function(name, vcov, ...) add(name, FIV, "yiv", "x1", specs$fe2$fe, vcov, endog = "xe", inst = c("z1", "z2"), ...)
iv("iv_nw_fe2", nw(), time = "t", panel = "id")
iv("iv_dk_fe2", dk(), time = "t")
iv("iv_conley_fe2", cy(), lat = "lat", lon = "lon")
iv("iv_crv3_c3_fe2", "crv3", cluster = list(list("c3")))
iv("iv_hc2_fe2", "hc2")
iv("iv_nwts_fe2", nw(2), time = "ts")
# Poisson and logit.
glmc <- function(name, model, sp, vcov, ...) {
  yv <- if (model == "poisson") "yp" else "yb"
  add(name, fml_of(yv, sp), yv, X2, specs[[sp]]$fe, vcov, model = model, ...)
}
for (model in c("poisson", "logit")) {
  pre <- substr(model, 1, 4)
  for (sp in c("nofe", "fe2")) {
    glmc(paste0(pre, "_hc2_", sp), model, sp, "hc2")
    glmc(paste0(pre, "_hc3_", sp), model, sp, "hc3")
    glmc(paste0(pre, "_nw_", sp), model, sp, nw(), time = "t", panel = "id")
    glmc(paste0(pre, "_dk_", sp), model, sp, dk(), time = "t")
    glmc(paste0(pre, "_conley_", sp), model, sp, cy(), lat = "lat", lon = "lon")
    glmc(paste0(pre, "_crv3_", sp), model, sp, "crv3", cluster = list(list("c3")))
  }
  glmc(paste0(pre, "_hc2_w_fe2"), model, "fe2", "hc2", weights = "w")
  glmc(paste0(pre, "_hc2_sl"), model, "sl", "hc2", dense = paste(if (model == "poisson") "yp" else "yb", SLD))
  glmc(paste0(pre, "_nwts2_fe1"), model, "fe1", nw(2), time = "ts")
  glmc(paste0(pre, "_hetero_sl"), model, "sl", "hetero", dense = paste(if (model == "poisson") "yp" else "yb", SLD))
}
writeLines(toJSON(cases, auto_unbox = TRUE, digits = NA, na = "null", null = "null"), file.path(dir, "fit_se_fixest.json"))
cat("fit_se_fixest.json", length(cases), "cases\n")

# Pyfixest flavor of Conley (fereg's extension, pyfixest has none): fixest's Conley meat with
# pyfixest's N/(N - df_k) (equal to fixest's n/(n - K) here) and no PSD fix; df_t = N - df_k.
py_conley <- function(name, model, sp, v, weights = NULL) {
  yv <- c(ols = "y", poisson = "yp", logit = "yb")[[model]]
  m <- fit_model(model, fml_of(yv, sp), d, "iid", weights)
  V <- vcov(m, vcov = vcov_arg(v, NULL, NULL, "lat", "lon"), vcov_fix = FALSE, attr = TRUE)
  ex <- vcov_fields(V, v)
  ex$df_t <- nobs(m) - ex$df_k
  list(name = name, data = "fit_se.csv", model = model, y = yv, x = X2, endog = character(), inst = character(),
       fe = fe_terms(specs[[sp]]$fe), intercept = TRUE, weights = weights, weights_type = "aweights", vcov = v,
       cluster = list(), lat = "lat", lon = "lon",
       ssc = list(K.adj = TRUE, K.fixef = "nonnested", K.exact = FALSE, G.adj = TRUE, G.df = "min", t.df = "min"),
       expect = c(ex, list(coef_names = names(coef(m)), coef = num(coef(m)), nobs = nobs(m), collin = list())))
}
cases <- list(
  py_conley("pyc_nofe", "ols", "nofe", cy()),
  py_conley("pyc_fe2", "ols", "fe2", cy(300)),
  py_conley("pyc_w_fe2", "ols", "fe2", cy(), weights = "w"),
  py_conley("pyc_sph_px_fe1", "ols", "fe1", cy(300, 50, TRUE)),
  py_conley("pyc_pois_fe2", "poisson", "fe2", cy()),
  py_conley("pyc_logit_nofe", "logit", "nofe", cy(200))
)
writeLines(toJSON(cases, auto_unbox = TRUE, digits = NA, na = "null", null = "null"), file.path(dir, "fit_se_pyconley.json"))
cat("fit_se_pyconley.json", length(cases), "cases\n")
