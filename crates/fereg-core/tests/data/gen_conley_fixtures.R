# Ground-truth Conley fixtures for crates/fereg-core/tests/vce.rs (fixest 0.14.2).
# Run from this directory:  R_LIBS=<lib with fixest> Rscript gen_conley_fixtures.R
# Writes vce_geo.csv and vce_conley_<spec>.json with b and V of fixest's Conley variance without
# small-sample factor and without the PSD fix (ssc(K.adj = FALSE, G.adj = FALSE),
# vcov_fix = FALSE), plus the cutoff in km. fereg's Stata layer multiplies the same meat by
# N/(N - df_m - df_a) (OLS, IV) or N/(N - 1) (GLM).
# GLMs: fixest builds scores and Hessian from the IRLS weights of its last step, i.e. at the
# previous iterate (feglm.fit: `res$irls_weights = w`), which at glm.tol = 1e-12 can still be
# ~1e-6 away from the converged mu. Each GLM is therefore refit from its converged linear
# predictor (`etastart`), so the weights fixest uses are those of the converged fit, as fereg's.
suppressMessages(library(fixest))
set.seed(20261015)
n <- 300
d <- data.frame(id = 1:n)
d$g <- rep(1:15, each = 20)
d$lat <- 30 + 20 * runif(n)
d$lon <- -120 + 50 * runif(n)
d$x1 <- rnorm(n) + 0.02 * d$lat
d$x2 <- rnorm(n)
d$z1 <- rnorm(n)
s <- sin(d$lat / 3) + cos(d$lon / 4)
d$en <- 0.6 * d$z1 + 0.3 * d$x1 + 0.5 * s + rnorm(n)
d$y <- 1 + 0.5 * d$x1 - 0.3 * d$x2 + 0.7 * d$en + 0.1 * d$g + s + rnorm(n)
d$yp <- rpois(n, exp(0.3 + 0.3 * d$x1 - 0.2 * d$x2 + 0.3 * s))
d$w <- 0.5 + runif(n)
d$yb <- rbinom(n, 1, plogis(-0.2 + 0.6 * d$x1 - 0.4 * d$x2 + 0.5 * s))

num <- function(x) ifelse(is.na(x), "null", sprintf("%.17g", x))
csv <- d[, c("id", "g", "lat", "lon", "x1", "x2", "z1", "en", "y", "yp", "w", "yb")]
lines <- c(paste(names(csv), collapse = ","), apply(csv, 1, function(r) paste(num(r), collapse = ",")))
writeLines(lines, "vce_geo.csv")

nossc <- ssc(K.adj = FALSE, G.adj = FALSE)
save_fix <- function(spec, est, cutoff, pixel, distance) {
  V <- vcov_conley(est, lat = ~lat, lon = ~lon, cutoff = cutoff, pixel = pixel, distance = distance, ssc = nossc, vcov_fix = FALSE)
  info <- attr(V, "vcov_type")
  km <- if (is.null(cutoff)) as.numeric(sub("^.*\\(([0-9.]+)km\\).*$", "\\1", info)) else cutoff
  b <- coef(est)
  js <- c("{",
    sprintf('  "name": "%s",', spec),
    sprintf('  "vcov_type": "%s",', info),
    sprintf('  "cutoff": %s,', num(km)),
    sprintf('  "colnames": "%s",', paste(names(b), collapse = " ")),
    sprintf('  "b": [%s],', paste(num(b), collapse = ", ")),
    sprintf('  "V": [%s]', paste(num(as.vector(t(V))), collapse = ", ")),
    "}")
  writeLines(js, sprintf("vce_conley_%s.json", spec))
  cat("saved", spec, info, "\n")
}

ols <- feols(y ~ x1 + x2 | g, d, fixef.tol = 1e-11)
save_fix("ols_tri", ols, 500, 0, "triangular")
save_fix("ols_sph", ols, 300, 0, "spherical")
save_fix("ols_pixel", ols, 500, 100, "triangular")
save_fix("ols_deduce", ols, NULL, 0, "triangular")
olsw <- feols(y ~ x1 + x2 | g, d, weights = ~w, fixef.tol = 1e-11)
save_fix("ols_aw", olsw, 400, 0, "spherical")
iv <- feols(y ~ x1 | g | en ~ z1, d, fixef.tol = 1e-11)
save_fix("iv_tri", iv, 500, 0, "triangular")
pois0 <- fepois(yp ~ x1 + x2 | g, d, fixef.tol = 1e-11, glm.tol = 1e-12)
pois <- fepois(yp ~ x1 + x2 | g, d, fixef.tol = 1e-11, glm.tol = 1e-12, etastart = predict(pois0, type = "link"))
save_fix("ppml_tri", pois, 500, 0, "triangular")
lgt0 <- feglm(yb ~ x1 + x2 | g, d, family = "logit", fixef.tol = 1e-11, glm.tol = 1e-12)
lgt <- feglm(yb ~ x1 + x2 | g, d, family = "logit", fixef.tol = 1e-11, glm.tol = 1e-12, etastart = predict(lgt0, type = "link"))
save_fix("logit_sph_pixel", lgt, 400, 50, "spherical")
save_fix("logit_tri", lgt, 500, 0, "triangular")
save_fix("ols_sph_pixel", ols, 400, 50, "spherical")
