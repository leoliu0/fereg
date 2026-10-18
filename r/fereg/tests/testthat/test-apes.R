# getAPEs(): average partial effects and their variance against alpaca 0.3.5 (exact control).
# With rows removed for a constant outcome, the finite-population term and the weak-exogeneity
# covariance sum over all N* rows (design 6.3, core API 2.2) and are checked against that
# computation on alpaca's ingredients (ape_vcov_all_rows()), which equals alpaca's when nothing
# is removed. Uncorrected estimations at 1e-8; corrected ones at 1e-6 (alpaca solves the fixed
# effects at the corrected coefficients imprecisely).

# Every variant of getAPEs() on `obj` (fereg) and `a` (alpaca), `full` being the N* rows and `fe`
# the fixed-effect variables: without n.pop the defaults (weak.exo = TRUE and "unrestricted" have
# no effect then and are errors), with n.pop both sampling.fe and weak.exo.
check_apes <- function(obj, a, full, fe, group, tol, ps = "classic") {
  corrected <- inherits(obj, "fereg_biascorr")
  drops <- a$nobs[["nobs.pc"]] > 0
  n.pop <- 3 * a$nobs[["nobs.full"]]
  for (np in list(NULL, n.pop)) {
    for (sfe in c("independence", "unrestricted")) {
      for (we in c(FALSE, TRUE)) {
        if (is.null(np) && (sfe == "unrestricted" || we)) next
        ours <- if (corrected) {
          getAPEs(obj, n.pop = np, sampling.fe = sfe)
        } else {
          getAPEs(obj, n.pop = np, panel.structure = ps, sampling.fe = sfe, weak.exo = we)
        }
        theirs <- alpaca::getAPEs(a, n.pop = np, panel.structure = ps, sampling.fe = sfe, weak.exo = we)
        g <- sprintf("%s npop=%s %s weak.exo=%s", group, if (is.null(np)) "NULL" else "3N*", sfe, we)
        expect_s3_class(ours, "APEs")
        expect_identical(names(ours$delta), names(theirs$delta))
        expect_identical(dimnames(ours$vcov), dimnames(theirs$vcov))
        expect_identical(ours$weak.exo, theirs$weak.exo)
        expect_identical(ours$panel.structure, theirs$panel.structure)
        expect_identical(ours$sampling.fe, sfe)
        expect_close(ours$delta, theirs$delta, tol, paste(g, "delta"))
        all_rows <- !is.null(np) && drops
        ref <- if (all_rows) {
          ape_vcov_all_rows(a, full, obj$obs_selection %||% seq_len(nrow(full)), fe, np, theirs$panel.structure, sfe, theirs$weak.exo)
        } else {
          theirs$vcov
        }
        expect_close(ours$vcov, ref, tol, paste(g, if (all_rows) "vcov (all rows)" else "vcov"))
        if (corrected) {
          expect_close(ours$bias.term, theirs$bias.term, tol, paste(g, "bias.term"))
          expect_equal(ours$bandwidth, as.integer(theirs$bandwidth))
        } else {
          expect_null(ours$bias.term)
        }
        if (corrected) break
      }
    }
  }
}

check_case <- function(fml, d, fe, group, ps = "classic", L = c(0L, 2L)) {
  m <- fe_logit(fml, d)
  a <- al_feglm(fml, d)
  check_apes(m, a, d, fe, paste(group, "feglm"), 1e-8, ps)
  for (l in L) {
    bc <- suppressWarnings(biasCorr(m, L = l, panel.structure = ps))
    check_apes(bc, alpaca::biasCorr(a, L = l, panel.structure = ps), d, fe, sprintf("%s biasCorr L=%d", group, l), 1e-6, ps)
  }
  invisible(a)
}

test_that("the all-rows variance reference is alpaca's when no row is removed", {
  skip_if_not(alpaca_ok())
  d <- sim_classic(60, 12, 1)
  a <- al_feglm(y ~ x1 + d | i + t, d)
  expect_identical(a$nobs[["nobs.pc"]], 0L)
  for (we in c(FALSE, TRUE)) {
    for (sfe in c("independence", "unrestricted")) {
      expect_close(ape_vcov_all_rows(a, d, seq_len(nrow(d)), c("i", "t"), 2000, sampling.fe = sfe, weak.exo = we),
        alpaca::getAPEs(a, n.pop = 2000, sampling.fe = sfe, weak.exo = we)$vcov, 1e-13, "all-rows reference vs alpaca, no drops")
    }
  }
  abc <- alpaca::biasCorr(a, L = 2)
  expect_close(ape_vcov_all_rows(a, d, seq_len(nrow(d)), c("i", "t"), NULL), alpaca::getAPEs(a)$vcov, 1e-13,
    "all-rows reference vs alpaca, no drops")
  expect_close(ape_vcov_all_rows(abc, d, seq_len(nrow(d)), c("i", "t"), 2000, weak.exo = TRUE),
    alpaca::getAPEs(abc, n.pop = 2000)$vcov, 1e-13, "all-rows reference vs alpaca, no drops")
  # Network: the rows alpaca keeps, estimated again, have nothing to remove.
  nw <- sim_network(10, 5, 4)
  nw <- nw[fe_logit(y ~ x + rta | exp_time + imp_time + pair, nw)$obs_selection, ]
  an <- al_feglm(y ~ x + rta | exp_time + imp_time + pair, nw)
  expect_identical(an$nobs[["nobs.pc"]], 0L)
  for (o in list(an, alpaca::biasCorr(an, L = 2, panel.structure = "network"))) {
    expect_close(ape_vcov_all_rows(o, nw, seq_len(nrow(nw)), c("exp_time", "imp_time", "pair"), 5000, "network", weak.exo = TRUE),
      alpaca::getAPEs(o, n.pop = 5000, panel.structure = "network", weak.exo = TRUE)$vcov, 1e-13, "all-rows reference vs alpaca, no drops")
  }
})

test_that("APEs match alpaca: classic two-way, balanced and with removed rows", {
  skip_if_not(alpaca_ok())
  check_case(y ~ x1 + d | i + t, sim_classic(60, 12, 1), c("i", "t"), "classic balanced")
  d <- sim_classic(80, 10, 2, sd_a = 2, drop = 0.15)
  a <- check_case(y ~ x1 + d | i + t, d, c("i", "t"), "classic drops")
  expect_gt(a$nobs[["nobs.pc"]], 0)
  # A period with a constant outcome: retained individuals lose a row, so the weak-exogeneity
  # covariance over all rows differs from alpaca's (shown without the sampling term).
  d$y[d$t == 4] <- 1L
  a <- check_case(y ~ x1 + d | i + t, d, c("i", "t"), "classic period dropped")
  expect_false(4 %in% a$data$t)
  m <- fe_logit(y ~ x1 + d | i + t, d)
  ours <- getAPEs(m, n.pop = 5000, sampling.fe = "unrestricted", weak.exo = TRUE)$vcov
  expect_gt(rel_diff(ours, alpaca::getAPEs(a, n.pop = 5000, sampling.fe = "unrestricted", weak.exo = TRUE)$vcov), 1e-6)
})

test_that("APEs match alpaca: dynamic panel (binary lagged outcome), one-way and weights", {
  skip_if_not(alpaca_ok())
  d <- sim_classic(60, 10, 3, dynamic = TRUE, drop = 0.1)
  check_case(y ~ ylag + x1 + d | i + t, d, c("i", "t"), "dynamic")
  check_case(y ~ ylag + x1 + d | i, d, "i", "one-way")
  # Weighted scores, unweighted averages (alpaca).
  set.seed(11)
  d$w <- runif(nrow(d), 0.5, 2)
  m <- fe_logit(y ~ ylag + x1 + d | i + t, d, weights = ~w)
  a <- al_feglm(y ~ ylag + x1 + d | i + t, d, weights = "w")
  check_apes(m, a, d, c("i", "t"), "weights feglm", 1e-8)
  check_apes(biasCorr(m, L = 1), alpaca::biasCorr(a, L = 1), d, c("i", "t"), "weights biasCorr L=1", 1e-6)
})

test_that("APEs match alpaca: network with two and three fixed effects", {
  skip_if_not(alpaca_ok())
  d <- sim_network(10, 5, 4)
  a <- check_case(y ~ x + rta | exp_time + imp_time + pair, d, c("exp_time", "imp_time", "pair"), "network 3-way", "network")
  expect_gt(a$nobs[["nobs.pc"]], 0)
  check_case(y ~ x + rta | exp_time + imp_time, d, c("exp_time", "imp_time"), "network 2-way", "network")
})

test_that("n.pop: truncation, population smaller than the sample, and alpaca's print layout", {
  skip_if_not(alpaca_ok())
  d <- sim_classic(40, 8, 7)
  m <- fe_logit(y ~ x1 + d | i + t, d)
  a <- al_feglm(y ~ x1 + d | i + t, d)
  base <- getAPEs(m)
  expect_equal(getAPEs(m, n.pop = 2000.7)$vcov, getAPEs(m, n.pop = 2000)$vcov)
  expect_equal(getAPEs(m, n.pop = nrow(d))$vcov, base$vcov)
  msg <- "Size of the entire population is lower than the full sample size. Correction factor set to zero."
  expect_warning(small <- getAPEs(m, n.pop = 10), msg, fixed = TRUE)
  expect_warning(alpaca::getAPEs(a, n.pop = 10), msg, fixed = TRUE)
  expect_equal(small$vcov, base$vcov)
  theirs <- alpaca::getAPEs(a)
  expect_identical(capture.output(print.APEs(base)), capture.output(alpaca:::print.APEs(theirs)))
  expect_identical(capture.output(print.summary.APEs(summary.APEs(base))),
    capture.output(alpaca:::print.summary.APEs(alpaca:::summary.APEs(theirs))))
  expect_identical(coef.APEs(base), base$delta)
  expect_identical(vcov.APEs(base), base$vcov)
  cm <- coef.summary.APEs(summary.APEs(base))
  expect_identical(colnames(cm), c("Estimate", "Std. error", "z value", "Pr(> |z|)"))
  expect_equal(cm[, 2], sqrt(diag(base$vcov)))
})

test_that("getAPEs() on a corrected estimation takes the structure and weak exogeneity from it", {
  d <- sim_classic(40, 8, 7)
  m <- fe_logit(y ~ x1 + d | i + t, d)
  bc0 <- biasCorr(m)
  bc2 <- biasCorr(m, L = 2)
  expect_false(getAPEs(bc0)$weak.exo)
  expect_true(getAPEs(bc2)$weak.exo)
  expect_identical(getAPEs(bc2)$bandwidth, 2L)
  expect_equal(getAPEs(bc2, n.pop = 5000, weak.exo = TRUE, panel.structure = "classic")$vcov, getAPEs(bc2, n.pop = 5000)$vcov)
  expect_error(getAPEs(bc2, n.pop = 5000, weak.exo = FALSE), "after an analytical bias correction, weak.exo is TRUE (L = 2); drop `weak.exo`", fixed = TRUE)
  expect_error(getAPEs(bc0, n.pop = 5000, weak.exo = TRUE), "weak.exo is FALSE (L = 0)", fixed = TRUE)
  expect_error(getAPEs(bc2, weak.exo = TRUE), "weak.exo = TRUE has no effect without n.pop", fixed = TRUE)
  expect_error(getAPEs(bc0, panel.structure = "network"), "the bias correction used panel.structure = \"classic\"", fixed = TRUE)
  ape <- getAPEs(bc0)
  expect_length(ape$bias.term, 2)
  expect_false(isTRUE(all.equal(ape$delta, getAPEs(m)$delta)))
})

test_that("getAPEs() rejects unsupported objects and arguments before estimating", {
  d <- sim_classic(30, 6, 8)
  d$k <- rep(1:3, length.out = nrow(d))
  m <- fe_logit(y ~ x1 + d | i + t, d)
  expect_error(getAPEs(), "'object' has to be specified.", fixed = TRUE)
  expect_error(getAPEs(stats::lm(y ~ x1, d)), "'getAPEs' called on a non-'feglm' object.", fixed = TRUE)
  expect_error(getAPEs(feols(y ~ x1 | i, d, notes = FALSE)), "'getAPEs' called on a non-'feglm' object.", fixed = TRUE)
  expect_error(getAPEs(fepois(y ~ x1 | i, d, notes = FALSE)), "'getAPEs' currently only supports binary choice models.", fixed = TRUE)
  expect_error(getAPEs(fe_logit(c(y, d) ~ x1 | i, d)), "takes a single estimation", fixed = TRUE)
  expect_error(getAPEs(fe_logit(y ~ 1 | i + t, d)), "getAPEs() needs at least one regressor", fixed = TRUE)
  expect_error(getAPEs(fe_logit(y ~ d | t[x1], d)), "bias correction and APEs need intercept-only fixed effects (no varying slopes)", fixed = TRUE)
  expect_error(getAPEs(fe_logit(y ~ x1 + d | i + t + k, d)), "panel.structure == 'classic' expects a one- or two-way fixed effects model.", fixed = TRUE)
  expect_error(getAPEs(fe_logit(y ~ x1 + d, d)), "panel.structure == 'classic' expects a one- or two-way fixed effects model.", fixed = TRUE)
  expect_error(getAPEs(fe_logit(y ~ x1 + d | i, d), panel.structure = "network"),
    "panel.structure == 'network' expects a two- or three-way fixed effects model.", fixed = TRUE)
  expect_error(getAPEs(m, sampling.fe = "dependent"), "should be one of")
  expect_error(getAPEs(m, panel.structure = "grid"), "should be one of")
  for (bad in list(-1, 0, NA, "1000", c(1000, 2000))) expect_error(getAPEs(m, n.pop = bad), "`n.pop` must be a positive number", fixed = TRUE)
  for (bad in list(NA, "yes", c(TRUE, FALSE), 1)) expect_error(getAPEs(m, weak.exo = bad), "`weak.exo` must be TRUE or FALSE", fixed = TRUE)
  local({
    dd <- d
    mm <- fe_logit(y ~ x1 + d | i + t, dd)
    rm(dd)
    expect_error(getAPEs(mm), "getAPEs() estimates the model again on it", fixed = TRUE)
  })
})
