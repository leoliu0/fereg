# Release smoke test of an installed fereg R package: version and a tiny feols.
# Usage: Rscript scripts/smoke_r.R [expected-version]
args <- commandArgs(trailingOnly = TRUE)
library(fereg)
v <- as.character(packageVersion("fereg"))
cat("fereg", v, "from", find.package("fereg"), "on", R.version.string, "\n")
if (length(args) >= 1 && v != args[[1]]) stop("installed fereg ", v, " != expected ", args[[1]])

# Deterministic data with two fixed effects; the reference is lm() on explicit dummies.
n <- 2000
i <- 0:(n - 1)
d <- data.frame(f1 = i %% 37, f2 = (i * 7) %% 11, x = sin(0.37 * i) + 0.1 * (i %% 5))
d$y <- 1.5 * d$x + 0.3 * d$f1 - 0.2 * d$f2 + 0.05 * cos(1.3 * i)

ref <- unname(coef(lm(y ~ x + factor(f1) + factor(f2), data = d))["x"])
fit <- feols(y ~ x | f1 + f2, d, nthreads = 2)
got <- unname(coef(fit)["x"])
cat(sprintf("coef(x): fereg %.12f, lm %.12f\n", got, ref))
if (abs(got - ref) > 1e-7 * abs(ref)) stop("fereg::feols coefficient does not match lm() with dummies")
if (nobs(fit) != n) stop("nobs ", nobs(fit), " != ", n)
cat("R smoke test passed\n")
