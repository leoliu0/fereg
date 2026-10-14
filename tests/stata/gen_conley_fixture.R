# Conley reference numbers for tests/stata/test_conley.do (fixest 0.14.2).
# Run from tests/stata on a machine with fixest (uts: R_LIBS=~/fereg-dev/rlib Rscript gen_conley_fixture.R).
# Writes conley_geo.csv (the data) and conley_ref.csv (spec, row, col, value): b, the Conley V of
# fixest without small-sample factor and without the PSD fix (ssc(K.adj = FALSE, G.adj = FALSE),
# vcov_fix = FALSE), and the cutoff in km actually used (row "cutoff"). fereg's Stata commands
# multiply the same meat by N/(N - df_m - df_a) (fereg OLS, IV) or N/(N - 1) (feppml, felogit).
suppressMessages(library(fixest))
set.seed(20261013)
n <- 400
d <- data.frame(id = 1:n)
d$g <- rep(1:20, each = 20)
d$lat <- 25 + 25 * runif(n)
d$lon <- -125 + 55 * runif(n)
d$x1 <- rnorm(n) + 0.03 * d$lat
d$x2 <- rnorm(n)
d$z1 <- rnorm(n)
s <- sin(d$lat / 3) + cos(d$lon / 4)
d$en <- 0.6 * d$z1 + 0.3 * d$x1 + 0.5 * s + rnorm(n)
d$y <- 1 + 0.5 * d$x1 - 0.3 * d$x2 + 0.7 * d$en + 0.05 * d$g + s + rnorm(n)
d$yp <- rpois(n, exp(0.3 + 0.3 * d$x1 - 0.2 * d$x2 + 0.3 * s))
d$yb <- as.integer(runif(n) < plogis(-0.2 + 0.5 * d$x1 - 0.4 * d$x2 + 0.5 * s))
d$w <- 0.5 + runif(n)

num <- function(x) sprintf("%.17g", x)
csv <- d[, c("id", "g", "lat", "lon", "x1", "x2", "z1", "en", "y", "yp", "yb", "w")]
lines <- c(paste(names(csv), collapse = ","), apply(csv, 1, function(r) paste(num(r), collapse = ",")))
writeLines(lines, "conley_geo.csv")

nossc <- ssc(K.adj = FALSE, G.adj = FALSE)
out <- "spec,row,col,value"
save_fix <- function(spec, est, cutoff, pixel, distance) {
  V <- vcov_conley(est, lat = ~lat, lon = ~lon, cutoff = cutoff, pixel = pixel, distance = distance,
                   ssc = nossc, vcov_fix = FALSE)
  info <- attr(V, "vcov_type")
  km <- if (is.null(cutoff)) as.numeric(sub("^.*\\(([0-9.]+)km\\).*$", "\\1", info)) else cutoff
  b <- coef(est)
  nm <- sub("^fit_", "", names(b))
  nm[nm == "(Intercept)"] <- "_cons"
  rows <- sprintf("%s,cutoff,,%s", spec, num(km))
  rows <- c(rows, sprintf("%s,b,%s,%s", spec, nm, num(b)))
  for (i in seq_along(nm)) rows <- c(rows, sprintf("%s,%s,%s,%s", spec, nm[i], nm, num(V[i, ])))
  out <<- c(out, rows)
  cat("saved", spec, info, "\n")
}

ols <- feols(y ~ x1 + x2 | g, d, fixef.tol = 1e-11)
save_fix("ols_tri", ols, 500, 0, "triangular")
save_fix("ols_sph", ols, 300, 0, "spherical")
save_fix("ols_pixel", ols, 500, 100, "triangular")
save_fix("ols_deduce", ols, NULL, 0, "triangular")
olsw <- feols(y ~ x1 + x2 | g, d, weights = ~w, fixef.tol = 1e-11)
save_fix("ols_aw", olsw, 400, 0, "spherical")
ols0 <- feols(y ~ x1 + x2, d)
save_fix("ols_nofe", ols0, 600, 0, "triangular")
iv <- feols(y ~ x1 | g | en ~ z1, d, fixef.tol = 1e-11)
save_fix("iv_tri", iv, 500, 0, "triangular")
# fixest builds GLM scores and the Hessian from the IRLS weights of its last step, i.e. of the
# previous iterate; refitting from the converged linear predictor makes them the converged ones
# (fereg's), as crates/fereg-core/tests/data/gen_conley_fixtures.R does.
pois0 <- fepois(yp ~ x1 + x2 | g, d, fixef.tol = 1e-11, glm.tol = 1e-12)
pois <- fepois(yp ~ x1 + x2 | g, d, fixef.tol = 1e-11, glm.tol = 1e-12, etastart = predict(pois0, type = "link"))
save_fix("ppml_tri", pois, 500, 0, "triangular")
logit0 <- feglm(yb ~ x1 + x2 | g, d, family = "logit", fixef.tol = 1e-11, glm.tol = 1e-12)
logit <- feglm(yb ~ x1 + x2 | g, d, family = "logit", fixef.tol = 1e-11, glm.tol = 1e-12, etastart = predict(logit0, type = "link"))
save_fix("logit_sph", logit, 400, 50, "spherical")
writeLines(out, "conley_ref.csv")
