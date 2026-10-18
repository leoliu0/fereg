# Fixture of test_logitfe.do (network section): a small directed-pair panel and alpaca 0.3.5's
# network bias corrections and APEs on it.
#   bc_network.csv      exp imp year exp_time imp_time pair y ylag x rta
#   bc_network_ref.csv  case quantity name value  (alpaca numbers, %.17g)
# alpaca runs with tight controls (dev.tol = 1e-20, center.tol = 1e-14), where its fit is the exact
# MLE and its corrected point agrees with an exact fixed-effect re-solve to ~1e-10. Groups with a
# constant outcome are removed beforehand (iteratively), so that no observation is dropped by the fit
# and N* = n: the sampling-of-effects term over all N* rows (fereg) then equals alpaca's.
# Run from tests/stata:  Rscript gen_bc_network.R   (alpaca 0.3.5 on .libPaths(); here /tmp/bc/rlib)
.libPaths(c("/tmp/bc/rlib", .libPaths()))
suppressPackageStartupMessages({library(alpaca); library(data.table)})
stopifnot(packageVersion("alpaca") == "0.3.5")

set.seed(20261018L)
C <- 12L; Y <- 9L
nw <- CJ(exp = seq_len(C), imp = seq_len(C), year = seq_len(Y))[exp != imp]
a_et <- matrix(rnorm(C * Y, sd = 0.6), C, Y); b_jt <- matrix(rnorm(C * Y, sd = 0.6), C, Y)
c_ij <- matrix(rnorm(C * C, sd = 0.8), C, C)
nw[, x := rnorm(.N) + 0.3 * c_ij[cbind(exp, imp)]]
nw[, rta := as.integer(runif(.N) < 0.3 + 0.1 * (year > Y / 2))]
setkey(nw, exp, imp, year)
yy <- integer(nrow(nw)); yl <- 0L
for (r in seq_len(nrow(nw))) {
  if (nw$year[r] == 1L) yl <- as.integer(runif(1) < plogis(c_ij[nw$exp[r], nw$imp[r]]))
  e <- 0.7 * yl + 0.5 * nw$x[r] + 0.4 * nw$rta[r] + a_et[nw$exp[r], nw$year[r]] +
    b_jt[nw$imp[r], nw$year[r]] + c_ij[nw$exp[r], nw$imp[r]]
  yl <- as.integer(rlogis(1) < e); yy[r] <- yl
}
nw[, y := yy]
nw[, ylag := shift(y), by = .(exp, imp)]
nw <- nw[year > 1L]
nw[, year := year - 1L]
nw[, exp_time := exp * 100L + year]
nw[, imp_time := imp * 100L + year]
nw[, pair := exp * 100L + imp]
# iterated removal of groups with a constant outcome
repeat {
  n0 <- nrow(nw)
  for (g in c("exp_time", "imp_time", "pair")) nw <- nw[, if (min(y) < max(y)) .SD, by = g]
  if (nrow(nw) == n0) break
}
setcolorder(nw, c("exp", "imp", "year", "exp_time", "imp_time", "pair", "y", "ylag", "x", "rta"))
setkey(nw, exp, imp, year)
fwrite(nw, "bc_network.csv")

CTL <- list(dev.tol = 1e-20, center.tol = 1e-14, iter.max = 100L)
fml <- y ~ ylag + x + rta | exp_time + imp_time + pair
m <- feglm(fml, copy(nw), binomial(), control = CTL)
stopifnot(m$nobs[["nobs"]] == nrow(nw))
M <- 3 * nrow(nw)
f17 <- function(x) sprintf("%.17g", x)
rows <- list()
rec <- function(case, quantity, x) {
  if (is.matrix(x)) {
    g <- expand.grid(r = seq_len(nrow(x)), c = seq_len(ncol(x)))
    nm <- paste(rownames(x)[g$r], colnames(x)[g$c], sep = ":")
    v <- x[cbind(g$r, g$c)]
  } else {
    nm <- names(x); v <- x
  }
  rows[[length(rows) + 1L]] <<- data.table(case = case, quantity = quantity, name = nm, value = f17(v))
}
rec("noc", "b", coef(m))
for (we in c(FALSE, TRUE)) {
  a <- getAPEs(m, n.pop = M, panel.structure = "network", weak.exo = we)
  if (!we) rec("noc", "b2", a$delta)
  rec("noc", if (we) "V2_weakexo" else "V2", a$vcov)
}
for (L in c(0L, 2L)) {
  b <- biasCorr(m, L = L, panel.structure = "network")
  rec(sprintf("an%d", L), "b", coef(b))
  rec(sprintf("an%d", L), "b2", getAPEs(b)$delta)
}
rec("all", "N", c(N = nrow(nw)))
rec("all", "M", c(M = M))
fwrite(rbindlist(rows), "bc_network_ref.csv")
