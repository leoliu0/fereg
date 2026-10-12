# Ground truth for crates/fereg-core/tests/fit_*.rs: fixest 0.14.2 on small generated data.
#
#   Rscript crates/fereg-core/tests/data/gen_fit_fixtures.R   (from the repository root)
#
# Writes tests/data/fit_d1.csv (the data) and, one case per model and variance matrix,
# tests/data/fit_fixest.json (OLS), fit_fixest_iv.json (IV), fit_fixest_fe.json (varying
# slopes, fixed-effect values, post-hoc variance matrices) and fit_fixest_glm.json (Poisson and
# logit). fixef.tol is fixest's minimum (> 2.2e-12); the Rust tests use 1e-12. GLMs run with
# glm.tol = 1e-12.
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
# IV: instruments z1..z3, endogenous xe, xe2 (u enters y and both), outcome yiv.
z1 <- rnorm(n); z2 <- rnorm(n); z3 <- rnorm(n); u <- rnorm(n)
xe <- 0.6 * z1 + 0.4 * z2 + 0.5 * u + gm[g] + rnorm(n)
xe2 <- 0.3 * z1 - 0.5 * z3 + 0.4 * u + rnorm(n)
yiv <- 1 + xe - 0.5 * xe2 + 0.5 * x1 + gm[g] + rnorm(12)[h] + u + rnorm(n)
# Varying slopes: s1 with a slope per g level; hb splits the g-h graph into two components.
s1 <- runif(n, 0, 3)
ys <- 1 + 0.5 * x1 + gm[g] + rnorm(45)[g] * s1 + rnorm(12)[h] + rnorm(n)
hb <- ifelse(g <= 20, h, h + 12)
# GLM: offset lo, count outcome yp (all zero in g levels 38-39), binary outcome yb (all one in
# g level 37, all zero in 36): fixest drops those perfect-fit levels.
lo <- runif(n, -0.5, 0.5)
yp <- rpois(n, exp(0.2 + 0.3 * x1 - 0.2 * x2 + 0.5 * gm[g] + 0.3 * rnorm(12)[h] + lo))
yp[g %in% c(38, 39)] <- 0
yb <- rbinom(n, 1, plogis(0.5 * x1 - 0.5 * x2 + gm[g] + rnorm(12)[h]))
yb[g == 37] <- 1
yb[g == 36] <- 0
d <- data.frame(y, x1, x2, x3, xg, g, h, k, c1, c3, cna, w, wi, z1, z2, z3, xe, xe2, yiv, s1, ys, hb, lo, yp, yb)
write.csv(d, file.path(dir, "fit_d1.csv"), row.names = FALSE, na = "NA")
toy <- data.frame(g = rep(c(1, 2, 3), each = 2), x = c(1, 2, 1, 3, 2, 4), y = c(2, 3, 2, 5, 3, 7), t = c(1, 1, 1, 2, 2, 2))
write.csv(toy, file.path(dir, "fit_toy.csv"), row.names = FALSE)

num <- function(v) { v <- unname(as.numeric(v)); v }
# A scalar that fixest may leave NULL / NA for some models.
opt <- function(v) if (is.null(v) || length(v) == 0) NA else num(v)
ssc_list <- function(s) list(K.adj = s$K.adj, K.fixef = s$K.fixef, K.exact = s$K.exact, G.adj = s$G.adj, G.df = s$G.df, t.df = s$t.df)

# A fixed-effect term: factors (`^` interaction when several), varying slopes, intercept.
term <- function(factors, slopes = character(), intercept = TRUE) list(factors = as.list(factors), slopes = as.list(slopes), intercept = intercept)
fe_terms <- function(fe) lapply(fe, function(t) if (is.list(t) && !is.null(t$factors)) t else term(unlist(t)))

vcov_attrs <- function(V, dat, m, cluster) list(
  vcov = num(V), se = num(sqrt(diag(V))), df_k = attr(V, "df.K"), df_t = attr(V, "df.t"),
  G = lapply(cluster, function(dim) length(unique(do.call(paste, dat[obs(m), unlist(dim), drop = FALSE]))))
)

# fixef(m) per slot: level labels (sorted), values and references.
fixef_list <- function(m) {
  # With varying slopes fixef() re-solves iteratively (default fixef.tol = 1e-5).
  fe <- fixef(m, notes = FALSE, fixef.tol = 1e-14, fixef.iter = 100000)
  refs <- attr(fe, "references")
  lapply(names(fe), function(nm) list(name = nm, levels = names(fe[[nm]]), values = num(fe[[nm]]), references = num(refs[nm])))
}

# `posthoc`: list of list(vcov, cluster, ssc) evaluated with summary(m, vcov =, ssc =).
# `model`: "ols" (feols), "poisson" (fepois) or "logit" (feglm, family = "logit").
case <- function(name, data, fml, y, x, fe, intercept, vcov, cluster = list(), ssc = fixest::ssc(), weights = NULL,
                 endog = character(), inst = character(), fixef = FALSE, posthoc = list(), model = "ols", offset = NULL) {
  dat <- read.csv(file.path(dir, data))
  args <- list(fml = fml, data = dat, fixef.tol = TOL, fixef.iter = 100000, vcov = vcov, ssc = ssc, notes = FALSE)
  if (!is.null(weights)) args$weights <- as.formula(paste0("~", weights))
  if (!is.null(offset)) args$offset <- as.formula(paste0("~", offset))
  glm <- model != "ols"
  if (glm) args <- c(args, list(glm.tol = 1e-12, glm.iter = 100))
  m <- switch(model, ols = do.call(feols, args), poisson = do.call(fepois, args),
              logit = do.call(feglm, c(args, list(family = "logit"))))
  V <- vcov(m, attr = TRUE)
  f <- if (glm) list(stat = NA, p = NA, df1 = NA, df2 = NA) else fitstat(m, "f", simplify = FALSE)$f
  r2s <- r2(m, c("r2", "ar2", "wr2", "war2", "pr2", "apr2", "cor2"))
  expect <- c(vcov_attrs(V, dat, m, cluster), list(
    coef_names = names(coef(m)), coef = num(coef(m)),
    nobs = nobs(m), nparams = m$nparams,
    collin = if (is.null(m$collin.var)) list() else m$collin.var,
    r2 = num(r2s["r2"]), ar2 = num(r2s["ar2"]), wr2 = num(r2s["wr2"]), war2 = num(r2s["war2"]),
    pr2 = num(r2s["pr2"]), apr2 = num(r2s["apr2"]), sq_cor = num(r2s["cor2"]),
    rmse = num(fitstat(m, "rmse")[[1]]), ll = num(logLik(m)), ll_null = m$ll_null, ssr = opt(m$ssr),
    ssr_null = opt(m$ssr_null), ssr_fe_only = opt(m$ssr_fe_only),
    sigma2 = opt(m$sigma2), f_stat = num(f$stat), f_p = num(f$p), f_df1 = num(f$df1), f_df2 = num(f$df2),
    # Residuals and fitted values do not depend on the variance matrix: stored once per model.
    resid = if (identical(vcov, "iid")) num(resid(m)) else list(),
    fitted = if (identical(vcov, "iid")) num(fitted(m)) else list(), obs = num(obs(m))
  ))
  if (glm) {
    expect$deviance <- m$deviance
    expect$null_deviance <- opt(m$null.deviance)
    expect$iterations <- m$iterations
    expect$converged <- m$convStatus
    expect$eta <- if (identical(vcov, "iid")) num(m$linear.predictors) else list()
    expect$irls_weights <- if (identical(vcov, "iid")) num(m$irls_weights) else list()
  }
  if (length(endog) > 0) {
    w <- if (is.null(weights)) 1 else dat[obs(m), weights]
    fs <- fitstat(m, c("ivf1", "ivwald1", "wh", "sargan"), simplify = FALSE)
    test <- function(t) if (is.null(t) || identical(t, NA)) list(stat = NA, p = NA, df1 = NA, df2 = NA)
      else list(stat = num(t$stat), p = num(t$p), df1 = num(if (is.null(t$df1)) t$df else t$df1), df2 = if (is.null(t$df2)) NA else num(t$df2))
    expect$iv <- list(
      ssr_iv = sum(w * m$iv_residuals^2),
      wh = test(fs$wh), sargan = test(fs$sargan),
      first = lapply(endog, function(e) {
        f1 <- m$iv_first_stage[[e]]
        V1 <- vcov(f1, attr = TRUE)
        c(vcov_attrs(V1, dat, f1, cluster), list(
          coef_names = names(coef(f1)), coef = num(coef(f1)), nparams = f1$nparams,
          collin = if (is.null(f1$collin.var)) list() else f1$collin.var,
          f = test(fs[[paste0("ivf1::", e)]]), wald = test(fs[[paste0("ivwald1::", e)]])
        ))
      })
    )
  }
  if (fixef) expect$fixef <- fixef_list(m)
  expect$posthoc <- lapply(posthoc, function(p) {
    s <- summary(m, vcov = p$vcov, ssc = if (is.null(p$ssc)) fixest::ssc() else p$ssc)
    vcov_attrs(vcov(s, attr = TRUE), dat, m, if (is.null(p$cluster)) list() else p$cluster)
  })
  list(
    name = name, data = data, model = model, y = y, x = x, endog = endog, inst = inst, fe = fe_terms(fe), intercept = intercept,
    offset = offset,
    weights = weights, vcov = if (is.character(vcov)) vcov else "cluster", cluster = cluster, ssc = ssc_list(ssc),
    posthoc = lapply(posthoc, function(p) list(vcov = if (is.character(p$vcov)) p$vcov else "cluster",
                                               cluster = if (is.null(p$cluster)) list() else p$cluster,
                                               ssc = ssc_list(if (is.null(p$ssc)) fixest::ssc() else p$ssc))),
    expect = expect
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

write_cases <- function(file) {
  writeLines(toJSON(cases, auto_unbox = TRUE, digits = NA, na = "null", null = "null"), file.path(dir, file))
  cat(file, length(cases), "cases\n")
  cases <<- list()
}
write_cases("fit_fixest.json")

# IV: over- and just-identified, two endogenous variables, no exogenous regressor, no FE,
# weights, every variance matrix.
FIV <- yiv ~ x1 | g + h | xe ~ z1 + z2
ivargs <- list(data = "fit_d1.csv", y = "yiv", x = "x1", fe = fe2, intercept = TRUE, endog = "xe", inst = c("z1", "z2"))
addiv <- function(name, fml, vcov, ...) {
  a <- ivargs
  extra <- list(...)
  a[names(extra)] <- extra  # not modifyList: it merges nested lists such as `fe`
  do.call(add, c(list(name = name, fml = fml, vcov = vcov), a))
}
for (v in list("iid", "hetero")) addiv(paste0("iv_fe2_", v), FIV, v)
addiv("iv_fe2_cl_c3", FIV, ~c3, cluster = cl$c3)
addiv("iv_fe2_cl_c3_h", FIV, ~c3 + h, cluster = cl$c3_h)
addiv("iv_fe2_cl_g", FIV, ~g, cluster = cl$g)
addiv("iv_fe2_w_hetero", FIV, "hetero", weights = "w")
addiv("iv_fe2_w_iid", FIV, "iid", weights = "w")
addiv("iv_fe1_just", yiv ~ x1 | g | xe ~ z1, "iid", fe = list(list("g")), inst = "z1")
addiv("iv_fe1_two_endog", yiv ~ x1 | g | xe + xe2 ~ z1 + z2 + z3, "iid", fe = list(list("g")), endog = c("xe", "xe2"), inst = c("z1", "z2", "z3"))
addiv("iv_fe1_two_endog_cl", yiv ~ x1 | g | xe + xe2 ~ z1 + z2 + z3, ~c3, fe = list(list("g")), endog = c("xe", "xe2"), inst = c("z1", "z2", "z3"), cluster = cl$c3)
addiv("iv_fe1_noexog", yiv ~ 1 | g | xe ~ z1 + z2, "iid", fe = list(list("g")), x = character())
addiv("iv_nofe_iid", yiv ~ x1 | xe ~ z1 + z2, "iid", fe = list())
addiv("iv_nofe_hetero_w", yiv ~ x1 | xe ~ z1 + z2, "hetero", fe = list(), weights = "w")
addiv("iv_nofe_noexog", yiv ~ 1 | xe ~ z1 + z2, "iid", fe = list(), x = character())
addiv("iv_fe2_fixef", FIV, "iid", fixef = TRUE, posthoc = list(list(vcov = "hetero"), list(vcov = ~c3, cluster = cl$c3)))
write_cases("fit_fixest_iv.json")

# Varying slopes (intercept + slope, slope only, slope only without any intercept term).
tg_s <- term("g", "s1")
add("sl_g_s_h_iid", "fit_d1.csv", ys ~ x1 | g[s1] + h, "ys", "x1", list(tg_s, list("h")), TRUE, "iid", fixef = TRUE)
add("sl_g_s_h_cl", "fit_d1.csv", ys ~ x1 | g[s1] + h, "ys", "x1", list(tg_s, list("h")), TRUE, ~c3, cluster = cl$c3)
add("sl_g_s_h_cl_g", "fit_d1.csv", ys ~ x1 | g[s1] + h, "ys", "x1", list(tg_s, list("h")), TRUE, ~g, cluster = cl$g)
add("sl_g_h_ss_w", "fit_d1.csv", ys ~ x1 | g + h[[s1]], "ys", "x1", list(list("g"), term("h", "s1", FALSE)), TRUE, "hetero", weights = "w")
add("sl_only", "fit_d1.csv", ys ~ x1 | g[[s1]], "ys", "x1", list(term("g", "s1", FALSE)), TRUE, "iid", fixef = TRUE)
add("sl_only_cl", "fit_d1.csv", ys ~ x1 | g[[s1]], "ys", "x1", list(term("g", "s1", FALSE)), TRUE, ~c3, cluster = cl$c3)
add("sl_two", "fit_d1.csv", ys ~ x1 | g[s1, x2] + h, "ys", "x1", list(term("g", c("s1", "x2")), list("h")), TRUE, "hetero")
# Fixed-effect values: one term, two connected terms, two components, three terms, interaction.
add("fx_fe1", "fit_d1.csv", y ~ x1 + x2 | g, "y", c("x1", "x2"), list(list("g")), TRUE, "iid", fixef = TRUE)
add("fx_fe2", "fit_d1.csv", F2, "y", c("x1", "x2"), fe2, TRUE, "iid", fixef = TRUE,
    posthoc = list(list(vcov = "hetero"), list(vcov = ~c1 + c3, cluster = cl$c1_c3), list(vcov = ~g, cluster = cl$g, ssc = ssc(K.fixef = "full")),
                   list(vcov = "iid", ssc = ssc(K.adj = FALSE)), list(vcov = ~h^k, cluster = list(list("h", "k")))))
add("fx_fe2_comp", "fit_d1.csv", y ~ x1 + x2 | g + hb, "y", c("x1", "x2"), list(list("g"), list("hb")), TRUE, "iid", fixef = TRUE)
add("fx_fe2_comp_w", "fit_d1.csv", y ~ x1 + x2 | hb + g, "y", c("x1", "x2"), list(list("hb"), list("g")), TRUE, "iid", weights = "w", fixef = TRUE)
add("fx_fe3", "fit_d1.csv", y ~ x1 + x2 | g + h + k, "y", c("x1", "x2"), fe3, TRUE, "iid", fixef = TRUE)
add("fx_fe3_comp", "fit_d1.csv", y ~ x1 + x2 | g + hb + k, "y", c("x1", "x2"), list(list("g"), list("hb"), list("k")), TRUE, "iid", fixef = TRUE)
add("fx_inter", "fit_d1.csv", y ~ x1 + x2 | g + h^k, "y", c("x1", "x2"), list(list("g"), list("h", "k")), TRUE, "iid", fixef = TRUE)
add("fx_nofe_posthoc", "fit_d1.csv", y ~ x1 + x2, "y", c("x1", "x2"), list(), TRUE, "iid",
    posthoc = list(list(vcov = "hetero"), list(vcov = ~c3 + k, cluster = list(list("c3"), list("k")))))
write_cases("fit_fixest_fe.json")

# Poisson and logit: 1-2 FE, perfect-fit levels, weights, offset, no FE, slopes, interaction,
# every variance matrix, fixed-effect values, post-hoc variance matrices.
FP <- yp ~ x1 + x2 | g + h
FB <- yb ~ x1 + x2 | g + h
for (v in list("iid", "hetero")) {
  add(paste0("pois_fe2_", v), "fit_d1.csv", FP, "yp", c("x1", "x2"), fe2, TRUE, v, model = "poisson")
  add(paste0("logit_fe2_", v), "fit_d1.csv", FB, "yb", c("x1", "x2"), fe2, TRUE, v, model = "logit")
}
add("pois_fe2_cl_c3_h", "fit_d1.csv", FP, "yp", c("x1", "x2"), fe2, TRUE, ~c3 + h, cluster = cl$c3_h, model = "poisson")
add("logit_fe2_cl_g", "fit_d1.csv", FB, "yb", c("x1", "x2"), fe2, TRUE, ~g, cluster = cl$g, model = "logit")
add("pois_fe2_off_w", "fit_d1.csv", FP, "yp", c("x1", "x2"), fe2, TRUE, "iid", weights = "w", offset = "lo", model = "poisson",
    fixef = TRUE, posthoc = list(list(vcov = "hetero"), list(vcov = ~c1, cluster = cl$c1)))
add("logit_fe1_off", "fit_d1.csv", yb ~ x1 + x2 | g, "yb", c("x1", "x2"), list(list("g")), TRUE, "iid", offset = "lo", model = "logit",
    fixef = TRUE)
add("logit_fe2_w", "fit_d1.csv", FB, "yb", c("x1", "x2"), fe2, TRUE, "hetero", weights = "w", model = "logit")
add("pois_nofe", "fit_d1.csv", yp ~ x1 + x2, "yp", c("x1", "x2"), list(), TRUE, "iid", model = "poisson",
    posthoc = list(list(vcov = "hetero"), list(vcov = ~c3, cluster = cl$c3)))
add("pois_nofe_off_w_cl", "fit_d1.csv", yp ~ x1 + x2, "yp", c("x1", "x2"), list(), TRUE, ~c3, cluster = cl$c3, weights = "w",
    offset = "lo", model = "poisson")
add("logit_nofe", "fit_d1.csv", yb ~ x1 + x2, "yb", c("x1", "x2"), list(), TRUE, "iid", model = "logit")
add("logit_nofe_off", "fit_d1.csv", yb ~ x1 + x2, "yb", c("x1", "x2"), list(), TRUE, "hetero", offset = "lo", model = "logit")
add("pois_collin", "fit_d1.csv", yp ~ x1 + x2 + x3 + xg | g + h, "yp", c("x1", "x2", "x3", "xg"), fe2, TRUE, "iid", model = "poisson")
add("pois_sl", "fit_d1.csv", yp ~ x1 | g[s1] + h, "yp", "x1", list(tg_s, list("h")), TRUE, "iid", model = "poisson", fixef = TRUE)
add("pois_sl_only", "fit_d1.csv", yp ~ x1 | g[[s1]], "yp", "x1", list(term("g", "s1", FALSE)), TRUE, "iid", model = "poisson")
add("pois_inter_fx", "fit_d1.csv", yp ~ x1 + x2 | g + h^k, "yp", c("x1", "x2"), list(list("g"), list("h", "k")), TRUE, "iid",
    model = "poisson", fixef = TRUE)
add("logit_fe3_tw", "fit_d1.csv", yb ~ x1 + x2 | g + h + k, "yb", c("x1", "x2"), fe3, TRUE, ~c3 + h, cluster = cl$c3_h, model = "logit")
write_cases("fit_fixest_glm.json")
