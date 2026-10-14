# Fixtures for crates/fereg-core/src/leverage.rs (tests `fixest_hatvalues_*`).
# fixest 0.14.2 exact leverage: hatvalues(feols(y ~ x | fe, weights = w), exact = TRUE)
# = P_fixef + P_slope + P_rhs (methods.R hatvalues.fixest). The Rust test subtracts the
# regressor part w_i x~_i^2 / sum(w x~^2) (x~ = x residualized on every FE column) and
# compares the rest with fe_hat.
# fixest 0.14.2 itself errors on slope-only terms (fe[[x]]: "object 'slope_df' not found")
# and on 2 intercept FEs + fe[x] (CHOLMOD "not positive definite"), so those structures are
# covered by the dense oracle tests only. hatvalues demeans the slope variables with demean()'s
# default tol = 1e-6, which limits its accuracy for fe[x] with 2+ FEs to ~1e-10.
#
# Run from this directory: R_LIBS=~/fereg-dev/rlib Rscript gen_fixest.R
suppressMessages(library(fixest))
stopifnot(as.character(packageVersion("fixest")) == "0.14.2")

dense <- function(f) match(f, sort(unique(f)))
fmt <- function(v) sprintf("%.17g", v)
out <- function(name, df, h) {
  o <- data.frame(f1 = df$f1, f2 = df$f2, f3 = df$f3, x = fmt(df$x), x2 = fmt(df$x2), w = fmt(df$w), h = fmt(h))
  write.csv(o, paste0("fixest_", name, ".csv"), row.names = FALSE, quote = FALSE)
}
# Exact full-model leverage (explicit dummies, eigen pseudo-inverse) to report fixest's own
# accuracy on each fixture.
exact <- function(M, w) {
  A <- crossprod(M * sqrt(w)); e <- eigen(A, symmetric = TRUE)
  k <- e$values > 1e-11 * max(e$values)
  w * rowSums((M %*% (e$vectors[, k] %*% (t(e$vectors[, k]) / e$values[k]))) * M)
}
dum <- function(f) model.matrix(~ factor(f) - 1)
report <- function(name, h, M, w) cat(sprintf("%-12s max |fixest - exact| = %.3e\n", name, max(abs(h - exact(M, w)))))
fit <- function(fml, df, weighted) {
  if (weighted) feols(fml, df, weights = ~w, fixef.tol = 2.3e-12, fixef.rm = "none") else feols(fml, df, fixef.tol = 2.3e-12, fixef.rm = "none")
}

set.seed(20261013)
n <- 300
df <- data.frame(f1 = sample(1:40, n, TRUE), f2 = sample(1:8, n, TRUE), f3 = sample(1:5, n, TRUE),
                 x = rnorm(n), x2 = rnorm(n) + 3, w = runif(n, 0.5, 2))
df$y <- df$x + rnorm(n)

# Two connected components (f1 <= 20 with f2 1..8, f1 > 20 with f2 9..16) and a singleton
# level of f1 (h = 1 for its row).
d1 <- df
d1$f2[d1$f1 > 20] <- d1$f2[d1$f1 > 20] + 8
d1$f1[1] <- 41
d1$f1 <- dense(d1$f1); d1$f2 <- dense(d1$f2); d1$f3 <- dense(d1$f3)
h <- hatvalues(fit(y ~ x | f1 + f2, d1, TRUE), exact = TRUE)
report("2fe_w_disc", h, cbind(d1$x, dum(d1$f1), dum(d1$f2)), d1$w)
out("2fe_w_disc", d1, h)

d2 <- df
d2$f1 <- dense(d2$f1); d2$f2 <- dense(d2$f2); d2$f3 <- dense(d2$f3)
h <- hatvalues(fit(y ~ x | f1 + f2 + f3, d2, TRUE), exact = TRUE)
report("3fe_w", h, cbind(d2$x, dum(d2$f1), dum(d2$f2), dum(d2$f3)), d2$w)
out("3fe_w", d2, h)
d3 <- d2
d3$w <- 1
h <- hatvalues(fit(y ~ x | f1[x2], d3, FALSE), exact = TRUE)
report("slope1", h, cbind(d3$x, dum(d3$f1), dum(d3$f1) * d3$x2), d3$w)
out("slope1", d3, h)

# f1 + f2[x2]: hatvalues demeans x2 by (f1, f2) with demean()'s default tol = 1e-6, and its
# collin.rm dummy matrix is sometimes rank deficient (CHOLMOD "not positive definite"): on
# random 300-1000 row designs fixest is off the exact leverage by 1e-13 to 8e-9 or errors.
# This dense design (about 8 observations per (f1, f2) cell, seed 3) is one where it is exact
# to rounding (reported below).
set.seed(3)
n4 <- 600
d4 <- data.frame(f1 = sample(1:12, n4, TRUE), f2 = sample(1:6, n4, TRUE), f3 = 1L,
                 x = rnorm(n4), x2 = rnorm(n4) + 3, w = runif(n4, 0.5, 2))
d4$y <- d4$x + rnorm(n4)
d4$f1 <- dense(d4$f1); d4$f2 <- dense(d4$f2)
h <- hatvalues(fit(y ~ x | f1 + f2[x2], d4, TRUE), exact = TRUE)
report("slope_w", h, cbind(d4$x, dum(d4$f1), dum(d4$f2), dum(d4$f2) * d4$x2), d4$w)
out("slope_w", d4, h)
