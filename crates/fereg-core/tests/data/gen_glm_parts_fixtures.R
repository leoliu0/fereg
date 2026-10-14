# Ground truth for the fixest-mode GLM parts (tests/glm.rs, section "fixest convention"):
# fixest 0.14.2 fepois / feglm(family = "logit") at glm.tol = 1e-12 and the tightest fixef.tol
# fixest accepts (it rejects values <= 2.2e-12).
#
# Run from this directory:  Rscript gen_glm_parts_fixtures.R
# Writes glm_parts.csv (simulated data) and glm_fx_<model>.json.

suppressPackageStartupMessages(library(fixest))
stopifnot(packageVersion("fixest") == "0.14.2")

set.seed(20261012)
n <- 2000
f1 <- sample(1:40, n, TRUE)
f2 <- sample(1:15, n, TRUE)
f3 <- sample(1:6, n, TRUE)
x1 <- rnorm(n)
x2 <- 0.5 * x1 + rnorm(n)
w <- runif(n, 0.5, 2)
off <- log(runif(n, 0.5, 2))
eta <- 0.3 * x1 - 0.2 * x2 + rnorm(40, 0, 0.5)[f1] + rnorm(15, 0, 0.3)[f2] + rnorm(6, 0, 0.3)[f3]
yp <- rpois(n, exp(eta + off))
yb <- rbinom(n, 1, plogis(eta + off))
# FE levels with a constant outcome: removed by fixef.rm = "perfect_fit".
yp[f1 == 40] <- 0
yb[f1 == 39] <- 1
yb[f1 == 40] <- 0
d <- data.frame(obs = seq_len(n), yp, yb, x1, x2, f1, f2, f3, w, off)
write.csv(d, "glm_parts.csv", row.names = FALSE, quote = FALSE)

num <- function(v) sprintf("%.17g", v)
arr <- function(v) paste0("[", paste(num(v), collapse = ", "), "]")
strs <- function(v) paste0("[", paste0('"', v, '"', collapse = ", "), "]")

models <- list(
  list(name = "fe1", fe = "f1"),
  list(name = "fe2", fe = c("f1", "f2")),
  list(name = "fe3", fe = c("f1", "f2", "f3")),
  list(name = "w", fe = c("f1", "f2"), w = TRUE),
  list(name = "off", fe = c("f1", "f2"), off = TRUE),
  list(name = "w_off", fe = c("f1", "f2", "f3"), w = TRUE, off = TRUE),
  list(name = "nofe", fe = character(0)),
  list(name = "nofe_w_off", fe = character(0), w = TRUE, off = TRUE)
)

for (fam in c("p", "l")) {
  for (m in models) {
    y <- if (fam == "p") "yp" else "yb"
    rhs <- "x1 + x2"
    if (length(m$fe)) rhs <- paste(rhs, "|", paste(m$fe, collapse = " + "))
    fml <- as.formula(paste(y, "~", rhs))
    wts <- if (isTRUE(m$w)) ~w else NULL
    ofs <- if (isTRUE(m$off)) ~off else NULL
    fit <- if (fam == "p") {
      fepois(fml, d, weights = wts, offset = ofs, glm.tol = 1e-12, fixef.tol = 2.5e-12, notes = FALSE)
    } else {
      suppressWarnings(feglm(fml, d, family = "logit", weights = wts, offset = ofs, glm.tol = 1e-12,
                             fixef.tol = 2.5e-12, notes = FALSE))
    }
    stopifnot(fit$convStatus)
    # fixest's cov.iid uses the IRLS weights at the start of its last iteration, accurate to about
    # sqrt(glm.tol) (~=3e-7 relative for these logits); cov_iid_conv is the inverse Hessian at the
    # converged estimate itself.
    s <- d[obs(fit), ]
    mu <- fitted(fit)
    wt <- if (isTRUE(m$w)) s$w else 1
    hw <- wt * if (fam == "p") mu else mu * (1 - mu)
    X <- as.matrix(s[, c("x1", "x2")])
    if (length(m$fe)) {
      X <- demean(X, s[, m$fe, drop = FALSE], weights = hw, tol = 1e-13, iter = 1e5, notes = FALSE)
    } else {
      X <- cbind(1, X)
    }
    cov_conv <- solve(crossprod(X, hw * X))
    out <- c(
      sprintf('"name": "%s_%s"', fam, m$name),
      sprintf('"family": "%s"', if (fam == "p") "poisson" else "logit"),
      sprintf('"y": "%s"', y),
      sprintf('"fe": %s', if (length(m$fe)) strs(m$fe) else "[]"),
      sprintf('"weights": %d', if (isTRUE(m$w)) 1 else 0),
      sprintf('"offset": %d', if (isTRUE(m$off)) 1 else 0),
      sprintf('"coefnames": %s', strs(names(coef(fit)))),
      sprintf('"coef": %s', arr(coef(fit))),
      sprintf('"cov_iid": %s', arr(t(fit$cov.iid))),
      sprintf('"vcov_iid": %s', arr(t(vcov(fit, vcov = "iid")))),
      sprintf('"cov_iid_conv": %s', arr(t(cov_conv))),
      sprintf('"nobs": %s', num(nobs(fit))),
      sprintf('"nparams": %s', num(fit$nparams)),
      sprintf('"deviance": %s', num(deviance(fit))),
      sprintf('"loglik": %s', num(as.numeric(logLik(fit)))),
      sprintf('"iterations": %s', num(fit$iterations)),
      sprintf('"obs": %s', arr(obs(fit))),
      sprintf('"sumFE": %s', if (length(m$fe)) arr(fit$sumFE) else "null")
    )
    writeLines(c("{", paste0("  ", out, c(rep(",", length(out) - 1), "")), "}"),
               sprintf("glm_fx_%s_%s.json", fam, m$name))
  }
}
