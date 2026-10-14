# Conley reference numbers for python/tests/test_parity_se.py, from fixest 0.14.2.
# Run from this directory: R_LIBS=<lib with fixest 0.14.2, jsonlite> Rscript gen_conley_fixest.R
# Writes conley.csv (the data) and conley_fixest.json (coefficients, standard errors and the
# cutoff fixest used, km). fixest's Conley correction n/(n - K) equals fereg's pyfixest-flavor
# N/(N - df_k) without weights.
library(fixest)
library(jsonlite)

set.seed(20261013)
n <- 600
centers <- cbind(runif(12, 32, 48), runif(12, -118, -82))
k <- sample.int(12, n, replace = TRUE)
lat <- centers[k, 1] + rnorm(n, sd = 0.8)
lon <- centers[k, 2] + rnorm(n, sd = 1.2)
g <- sample.int(15, n, replace = TRUE)
x1 <- rnorm(n) + 0.3 * sin(lat)
x2 <- rnorm(n) + 0.2 * cos(lon)
u <- rnorm(12)[k]
y <- 1 + 0.5 * x1 - 0.3 * x2 + 0.2 * g / 15 + u + rnorm(n)
yp <- rpois(n, exp(0.2 + 0.3 * x1 - 0.2 * x2 + 0.3 * u))
yb <- as.numeric(runif(n) < plogis(-0.2 + 0.5 * x1 - 0.3 * x2 + 0.4 * u))
df <- data.frame(lat = lat, lon = lon, g = g, x1 = x1, x2 = x2, y = y, yp = yp, yb = yb)
write.csv(df, "conley.csv", row.names = FALSE)

cases <- list(
  list(id = "ols_100km", model = "ols", fml = y ~ x1 + x2, cutoff = 100, pixel = 0, distance = "triangular"),
  list(id = "ols_fe_60mi", model = "ols", fml = y ~ x1 + x2 | g, cutoff = "60mi", pixel = 0, distance = "triangular"),
  list(id = "ols_fe_spherical", model = "ols", fml = y ~ x1 + x2 | g, cutoff = 150, pixel = 0, distance = "spherical"),
  list(id = "ols_pixel", model = "ols", fml = y ~ x1 + x2, cutoff = 200, pixel = 25, distance = "triangular"),
  list(id = "ols_deduced", model = "ols", fml = y ~ x1 + x2 | g, cutoff = NULL, pixel = 0, distance = "triangular"),
  list(id = "pois_fe", model = "poisson", fml = yp ~ x1 + x2 | g, cutoff = 150, pixel = 0, distance = "triangular"),
  list(id = "logit_fe", model = "logit", fml = yb ~ x1 + x2 | g, cutoff = 150, pixel = 0, distance = "triangular")
)

out <- list()
for (cs in cases) {
  m <- switch(cs$model,
    ols = feols(cs$fml, df, fixef.tol = 1e-10),
    poisson = fepois(cs$fml, df, glm.tol = 1e-12, fixef.tol = 1e-10),
    logit = feglm(cs$fml, df, family = "logit", glm.tol = 1e-12, fixef.tol = 1e-10)
  )
  req <- conley(cutoff = cs$cutoff, pixel = cs$pixel, distance = cs$distance)
  v <- vcov(m, vcov = req, attr = TRUE)
  # fixest's label is "Conley (<cutoff><unit>)": the cutoff used, in km here.
  label <- attr(v, "vcov_type")
  used <- as.numeric(sub("^Conley \\(([0-9.]+)(km|mi)\\)$", "\\1", label))
  if (grepl("mi\\)$", label)) used <- used * 1.60934
  out[[cs$id]] <- list(
    coef = unname(coef(m)), se = unname(sqrt(diag(v))), names = names(coef(m)),
    label = label, cutoff_km = used
  )
}
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), "conley_fixest.json")
