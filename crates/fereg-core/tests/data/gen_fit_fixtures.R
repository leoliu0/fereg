# Ground truth for crates/fereg-core/tests/fit_*.rs: fixest 0.14.2 on small generated data.
#
#   Rscript crates/fereg-core/tests/data/gen_fit_fixtures.R   (from the repository root)
#
# Writes tests/data/fit_d1.csv (the data) and tests/data/fit_fixest.json (one case per model and
# variance matrix). fixef.tol is fixest's minimum (> 2.2e-12); the Rust tests use 1e-12.
suppressPackageStartupMessages({
  library(fixest)
  library(jsonlite)
})
stopifnot(as.character(packageVersion("fixest")) == "0.14.2")
dir <- "crates/fereg-core/tests/data"
setFixest_nthreads(1)
TOL <- 3e-12

set.seed(20261012)
n <- 600
g <- sample(1:40, n, TRUE)
g[1:5] <- 41:45                       # singleton levels of g
h <- sample(1:12, n, TRUE)
k <- sample(1:4, n, TRUE)
c1 <- (g - 1) %/% 3                   # g is nested in c1 (by value)
c3 <- sample(1:15, n, TRUE)           # unrelated cluster
x1 <- rnorm(n)
x2 <- rnorm(n) + 0.3 * x1
x3 <- x1 + 2 * x2                     # collinear with x1, x2
gm <- rnorm(45)
xg <- gm[g]                           # constant within g: absorbed by the g fixed effect
w <- runif(n, 0.5, 2)
y <- 1 + 0.5 * x1 - 0.8 * x2 + gm[g] + rnorm(12)[h] + rnorm(4)[k] + rnorm(n)
cna <- c3
y[c(11, 57, 300)] <- NA
x1[c(23, 401)] <- NA
h[c(77, 78)] <- NA
cna[c(5, 90, 91, 92, 250)] <- NA       # NA in a cluster variable
wi <- sample(1:3, n, TRUE)            # frequency weights (pyfixest fixtures)
d <- data.frame(y, x1, x2, x3, xg, g, h, k, c1, c3, cna, w, wi)
write.csv(d, file.path(dir, "fit_d1.csv"), row.names = FALSE, na = "NA")
toy <- data.frame(g = rep(c(1, 2, 3), each = 2), x = c(1, 2, 1, 3, 2, 4), y = c(2, 3, 2, 5, 3, 7), t = c(1, 1, 1, 2, 2, 2))
write.csv(toy, file.path(dir, "fit_toy.csv"), row.names = FALSE)

num <- function(v) { v <- unname(as.numeric(v)); v }
ssc_list <- function(s) list(K.adj = s$K.adj, K.fixef = s$K.fixef, K.exact = s$K.exact, G.adj = s$G.adj, G.df = s$G.df, t.df = s$t.df)

case <- function(name, data, fml, y, x, fe, intercept, vcov, cluster = list(), ssc = fixest::ssc(), weights = NULL) {
  dat <- read.csv(file.path(dir, data))
  args <- list(fml = fml, data = dat, fixef.tol = TOL, fixef.iter = 100000, vcov = vcov, ssc = ssc, notes = FALSE)
  if (!is.null(weights)) args$weights <- as.formula(paste0("~", weights))
  m <- do.call(feols, args)
  V <- vcov(m, attr = TRUE)
  f <- fitstat(m, "f", simplify = FALSE)$f
  r2s <- r2(m, c("r2", "ar2", "wr2", "war2", "pr2", "apr2", "cor2"))
  list(
    name = name, data = data, y = y, x = x, fe = fe, intercept = intercept, weights = weights,
    vcov = if (is.character(vcov)) vcov else "cluster", cluster = cluster, ssc = ssc_list(ssc),
    expect = list(
      coef_names = names(coef(m)), coef = num(coef(m)), se = num(se(m)), vcov = num(V),
      nobs = nobs(m), nparams = m$nparams, df_k = attr(V, "df.K"), df_t = attr(V, "df.t"),
      G = lapply(cluster, function(dim) length(unique(do.call(paste, dat[obs(m), unlist(dim), drop = FALSE])))),
      collin = if (is.null(m$collin.var)) list() else m$collin.var,
      r2 = num(r2s["r2"]), ar2 = num(r2s["ar2"]), wr2 = num(r2s["wr2"]), war2 = num(r2s["war2"]),
      pr2 = num(r2s["pr2"]), apr2 = num(r2s["apr2"]), sq_cor = num(r2s["cor2"]),
      rmse = num(fitstat(m, "rmse")[[1]]), ll = num(logLik(m)), ll_null = m$ll_null, ssr = m$ssr,
      ssr_null = m$ssr_null, ssr_fe_only = if (is.null(m$ssr_fe_only)) NA else m$ssr_fe_only,
      sigma2 = m$sigma2, f_stat = num(f$stat), f_p = num(f$p), f_df1 = num(f$df1), f_df2 = num(f$df2),
      # Residuals and fitted values do not depend on the variance matrix: stored once per model.
      resid = if (identical(vcov, "iid")) num(resid(m)) else list(),
      fitted = if (identical(vcov, "iid")) num(fitted(m)) else list(), obs = num(obs(m))
    )
  )
}

cases <- list()
add <- function(...) cases[[length(cases) + 1]] <<- case(...)
F2 <- y ~ x1 + x2 | g + h
fe2 <- list(list("g"), list("h"))
for (v in list("iid", "hetero")) add(paste0("fe2_", v), "fit_d1.csv", F2, "y", c("x1", "x2"), fe2, TRUE, v)
cl <- list(g = list(list("g")), c1 = list(list("c1")), c3 = list(list("c3")), g_h = list(list("g"), list("h")),
           c3_h = list(list("c3"), list("h")), c1_c3 = list(list("c1"), list("c3")), c1_c3_h = list(list("c1"), list("c3"), list("h")))
for (nm in names(cl)) {
  f <- as.formula(paste0("~", paste(sapply(cl[[nm]], function(d) paste(unlist(d), collapse = "^")), collapse = "+")))
  add(paste0("fe2_cl_", nm), "fit_d1.csv", F2, "y", c("x1", "x2"), fe2, TRUE, f, cluster = cl[[nm]])
}
sscs <- list(none = ssc(K.fixef = "none"), full = ssc(K.fixef = "full"), noadj = ssc(K.adj = FALSE),
             conv = ssc(G.df = "conventional"), tconv = ssc(t.df = "conventional"), exact = ssc(K.exact = TRUE),
             nog = ssc(G.adj = FALSE))
for (nm in names(sscs)) {
  add(paste0("fe2_c1_c3_ssc_", nm), "fit_d1.csv", F2, "y", c("x1", "x2"), fe2, TRUE, ~c1 + c3, cluster = cl$c1_c3, ssc = sscs[[nm]])
  add(paste0("fe2_iid_ssc_", nm), "fit_d1.csv", F2, "y", c("x1", "x2"), fe2, TRUE, "iid", ssc = sscs[[nm]])
}
add("fe2_g_ssc_exact", "fit_d1.csv", F2, "y", c("x1", "x2"), fe2, TRUE, ~g, cluster = cl$g, ssc = ssc(K.exact = TRUE))
add("collin", "fit_d1.csv", y ~ x1 + x2 + x3 + xg | g + h, "y", c("x1", "x2", "x3", "xg"), fe2, TRUE, ~c3, cluster = cl$c3)
for (v in list("iid", "hetero")) add(paste0("nofe_", v), "fit_d1.csv", y ~ x1 + x2, "y", c("x1", "x2"), list(), TRUE, v)
add("nofe_cl", "fit_d1.csv", y ~ x1 + x2, "y", c("x1", "x2"), list(), TRUE, ~c3 + h, cluster = cl$c3_h)
add("nofe_collin", "fit_d1.csv", y ~ x1 + x2 + x3, "y", c("x1", "x2", "x3"), list(), TRUE, "iid")
add("noint_iid", "fit_d1.csv", y ~ x1 + x2 - 1, "y", c("x1", "x2"), list(), FALSE, "iid")
add("noint_hetero", "fit_d1.csv", y ~ x1 + x2 - 1, "y", c("x1", "x2"), list(), FALSE, "hetero")
for (v in list("iid", "hetero")) add(paste0("fe1_w_", v), "fit_d1.csv", y ~ x1 | g, "y", "x1", list(list("g")), TRUE, v, weights = "w")
add("fe1_w_cl", "fit_d1.csv", y ~ x1 | g, "y", "x1", list(list("g")), TRUE, ~c1, cluster = cl$c1, weights = "w")
add("nofe_w_iid", "fit_d1.csv", y ~ x1 + x2, "y", c("x1", "x2"), list(), TRUE, "iid", weights = "w")
add("inter_w", "fit_d1.csv", y ~ x1 + x2 | g + h^k, "y", c("x1", "x2"), list(list("g"), list("h", "k")), TRUE, ~c3, cluster = cl$c3, weights = "w")
add("inter_h_cl", "fit_d1.csv", y ~ x1 + x2 | g + h^k, "y", c("x1", "x2"), list(list("g"), list("h", "k")), TRUE, ~h, cluster = list(list("h")))
fe3 <- list(list("g"), list("h"), list("k"))
add("fe3_iid_exact", "fit_d1.csv", y ~ x1 + x2 | g + h + k, "y", c("x1", "x2"), fe3, TRUE, "iid", ssc = ssc(K.exact = TRUE))
add("fe3_tw", "fit_d1.csv", y ~ x1 + x2 | g + h + k, "y", c("x1", "x2"), fe3, TRUE, ~c3 + h, cluster = cl$c3_h)
add("fe3_tw_exact", "fit_d1.csv", y ~ x1 + x2 | g + h + k, "y", c("x1", "x2"), fe3, TRUE, ~c3 + h, cluster = cl$c3_h, ssc = ssc(K.exact = TRUE))
add("nacluster", "fit_d1.csv", y ~ x1 | g, "y", "x1", list(list("g")), TRUE, ~cna, cluster = list(list("cna")))
# g is nested in c1 by value, but every cluster variable (c1) is a fixed effect: no value test.
add("fe_g_c1_cl_c1", "fit_d1.csv", y ~ x1 + x2 | g + c1, "y", c("x1", "x2"), list(list("g"), list("c1")), TRUE, ~c1, cluster = list(list("c1")))
add("inter_g_c1_cl_c1", "fit_d1.csv", y ~ x1 + x2 | g^c1 + h, "y", c("x1", "x2"), list(list("g", "c1"), list("h")), TRUE, ~c1, cluster = list(list("c1")))
toyfe <- list(list("g"))
for (v in list("iid", "hetero")) add(paste0("toy_", v), "fit_toy.csv", y ~ x | g, "y", "x", toyfe, TRUE, v)
add("toy_cl_g", "fit_toy.csv", y ~ x | g, "y", "x", toyfe, TRUE, ~g, cluster = list(list("g")))
add("toy_cl_g_full", "fit_toy.csv", y ~ x | g, "y", "x", toyfe, TRUE, ~g, cluster = list(list("g")), ssc = ssc(K.fixef = "full"))
add("toy_cl_g_none", "fit_toy.csv", y ~ x | g, "y", "x", toyfe, TRUE, ~g, cluster = list(list("g")), ssc = ssc(K.fixef = "none"))
add("toy_tw", "fit_toy.csv", y ~ x | g, "y", "x", toyfe, TRUE, ~g + t, cluster = list(list("g"), list("t")))

writeLines(toJSON(cases, auto_unbox = TRUE, digits = NA, na = "null", null = "null"), file.path(dir, "fit_fixest.json"))
cat(length(cases), "cases\n")
