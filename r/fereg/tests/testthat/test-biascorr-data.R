# biasCorr() / getAPEs() with a user offset (against an independent R implementation on fixest's
# offset fit, and fixest refits of the jackknife subpanels), rows with missing values, lean
# estimations, thread-count invariance, and the options without effect.

add_offset <- function(d, seed) {
  set.seed(seed)
  d$o <- rnorm(nrow(d), 0, 0.4)
  d
}

fx_logit <- function(fml, d, ...) {
  fixest::feglm(fml, d, family = "logit", glm.tol = 1e-12, fixef.tol = 3e-12, nthreads = 1, notes = FALSE, ...)
}

test_that("analytical correction and APEs with an offset match an independent implementation on fixest's fit", {
  skip_if_not(fixest_ok())
  for (case in list(
    list(name = "two-way", d = add_offset(sim_classic(60, 10, 41, sd_a = 1.5), 1), fe = c("i", "t"), fml = y ~ x1 + d | i + t),
    list(name = "dynamic", d = add_offset(sim_classic(60, 10, 42, dynamic = TRUE), 2), fe = c("i", "t"), fml = y ~ ylag + x1 + d | i + t),
    list(name = "one-way", d = add_offset(sim_classic(60, 10, 43, dynamic = TRUE), 3), fe = "i", fml = y ~ ylag + x1 + d | i)
  )) {
    d <- case$d
    m <- fe_logit(case$fml, d, offset = ~o)
    fx <- fx_logit(case$fml, d, offset = ~o)
    expect_identical(nobs(m), nobs(fx))
    ape <- fx_ape_ref(fx, d, case$fe, Nf = nrow(d), binary = c("d", "ylag"))
    ours <- getAPEs(m)
    expect_close(ours$delta, ape$delta[names(ours$delta)], 1e-8, paste("offset", case$name, "feglm APE delta"))
    expect_close(ours$vcov, ape$vcov, 1e-8, paste("offset", case$name, "feglm APE vcov"))
    for (L in 0:2) {
      bc <- biasCorr(m, L = L)
      ref <- fx_bias_ref(fx, d, case$fe, L)
      g <- sprintf("offset %s L=%d", case$name, L)
      expect_close(coef(bc), ref$coef[names(coef(bc))], 1e-8, paste(g, "coef"))
      expect_close(bc$bias.term, ref$bias.term[names(coef(bc))], 1e-8, paste(g, "bias.term"))
      pt <- fx_corrected_point(fx, d, case$fe, ref$coef, offset = ~o)
      expect_close(bc$linear.predictors, pt$eta, 1e-8, paste(g, "eta"))
      expect_close(vcov(bc, vcov = "iid", ssc = ssc(K.adj = FALSE)), pt$vcov, 1e-8, paste(g, "vcov"))
      # predict() adds the offset given as a formula.
      e <- d[bc$obs_selection %||% seq_len(nrow(d)), ]
      expect_equal(unname(predict(bc, newdata = e, type = "link")), bc$linear.predictors, tolerance = 1e-10)
    }
  }
})

test_that("jackknife with an offset matches fixest refits of the subpanels", {
  skip_if_not(fixest_ok())
  # (A seed whose subpanel fits all converge.)
  d <- add_offset(sim_classic(31, 8, 50, sd_a = 1), 4)
  fml <- y ~ x1 + d | i + t
  m <- fe_logit(fml, d, offset = ~o)
  cache <- new.env()
  for (v in c("ss2", "ss1", "sj", "jj")) {
    jk <- biasCorr(m, method = "jackknife", jackknife = v)
    ref <- jk_reference(fml, d, "i", "t", v, cache = cache, offset = ~o)
    expect_close(coef(jk), ref$coef[names(coef(jk))], 1e-8, sprintf("jackknife offset %s coef", v))
    expect_close(getAPEs(jk)$delta, ref$ape, 1e-8, sprintf("jackknife offset %s APE", v))
  }
})

test_that("rows with missing values (outcome, regressor, fixed effect, cluster) are left out as by the complete cases", {
  d <- sim_classic(50, 8, 45, sd_a = 1.2, dynamic = TRUE)
  set.seed(46)
  d$cl <- d$i %% 7
  d$y[sample(nrow(d), 8)] <- NA
  d$x1[sample(nrow(d), 8)] <- NA
  d$i[sample(nrow(d), 5)] <- NA
  d$cl[sample(nrow(d), 6)] <- NA
  full <- d[stats::complete.cases(d), ]
  fml <- y ~ ylag + x1 + d | i + t
  m <- fe_logit(fml, d, vcov = ~cl)
  m0 <- fe_logit(fml, full, vcov = ~cl)
  expect_gt(m$dropped$na, 0)
  expect_identical(nobs(m), nobs(m0))
  same <- function(a, b, what) expect_equal(a, b, tolerance = 1e-12, label = what)
  for (L in c(0L, 1L)) {
    bc <- biasCorr(m, L = L)
    bc0 <- biasCorr(m0, L = L)
    expect_identical(bc$nobs.full, nrow(full))
    same(coef(bc), coef(bc0), "coef")
    same(vcov(bc), vcov(bc0), "vcov ~cl")
    same(bc$linear.predictors, bc0$linear.predictors, "eta")
    a <- getAPEs(bc, n.pop = 2000)
    a0 <- getAPEs(bc0, n.pop = 2000)
    same(a$delta, a0$delta, "APE")
    same(a$vcov, a0$vcov, "APE vcov")
  }
  a <- getAPEs(m, n.pop = 2000, weak.exo = TRUE)
  a0 <- getAPEs(m0, n.pop = 2000, weak.exo = TRUE)
  same(a$vcov, a0$vcov, "APE vcov, weak.exo")
  for (v in c("ss2", "sj", "double")) {
    jk <- biasCorr(m, method = "jackknife", jackknife = v)
    jk0 <- biasCorr(m0, method = "jackknife", jackknife = v)
    same(coef(jk), coef(jk0), paste("jackknife", v))
    same(getAPEs(jk)$delta, getAPEs(jk0)$delta, paste("jackknife APE", v))
  }
  # The time variable of a one-way jackknife must be present on the estimation rows (core).
  d1 <- full
  d1$tt <- d1$t
  d1$tt[3] <- NA
  m1 <- fe_logit(y ~ x1 + d | i, d1)
  expect_error(biasCorr(m1, method = "jackknife", panel.id = ~ i + tt), "has missing values in the estimation sample", fixed = TRUE)
})

test_that("lean estimations are corrected from their data, and the result is lean", {
  d <- sim_classic(40, 8, 47, dynamic = TRUE)
  fml <- y ~ ylag + x1 + d | i + t
  m <- fe_logit(fml, d)
  ml <- fe_logit(fml, d, lean = TRUE)
  expect_null(ml$state)
  for (args in list(list(L = 1), list(method = "jackknife", jackknife = "sj"))) {
    bc <- do.call(biasCorr, c(list(m), args))
    bcl <- do.call(biasCorr, c(list(ml), args))
    expect_identical(coef(bcl), coef(bc))
    expect_identical(vcov(bcl), vcov(bc))
    expect_true(bcl$lean)
    expect_null(bcl$state)
    expect_null(bcl$residuals)
    expect_false(isTRUE(bc$lean))
    expect_error(vcov(bcl, vcov = "hetero"), "lean = TRUE", fixed = TRUE)
    expect_identical(getAPEs(bcl, n.pop = 3000), getAPEs(bc, n.pop = 3000))
  }
  expect_identical(getAPEs(ml, n.pop = 3000, weak.exo = TRUE), getAPEs(m, n.pop = 3000, weak.exo = TRUE))
})

test_that("results are bit-identical with 1 and 16 threads", {
  d <- sim_classic(4000, 10, 48, sd_a = 1.5)
  expect_gte(nrow(d), 40000)
  run <- function(k) {
    old <- options(fereg.nthreads = k)
    on.exit(options(old))
    m <- fe_logit(y ~ x1 + d | i + t, d, nthreads = k)
    bc <- biasCorr(m, L = 1)
    jk <- biasCorr(m, method = "jackknife")
    list(
      coef = coef(m), bc = coef(bc), bc_vcov = vcov(bc, vcov = ~i), eta = bc$linear.predictors,
      fe = fixef(bc, notes = FALSE)$i,
      ape = getAPEs(m, n.pop = 1e6, weak.exo = TRUE), ape_bc = getAPEs(bc, n.pop = 1e6),
      jk = coef(jk), ape_jk = getAPEs(jk, n.pop = 1e6, weak.exo = TRUE)
    )
  }
  r1 <- run(1)
  r16 <- run(16)
  for (nm in names(r1)) expect_identical(r16[[nm]], r1[[nm]], label = paste("16 threads:", nm))
})

test_that("options without effect and bandwidths too large are errors", {
  d <- sim_classic(30, 6, 8)
  m <- fe_logit(y ~ x1 + d | i + t, d)
  expect_error(biasCorr(m, L = 6), "bandwidth L = 6 is too large: the longest group has 6 observations", fixed = TRUE)
  expect_s3_class(biasCorr(m, L = 5), "fereg_biascorr")
  # Without n.pop the sampling term and the weak-exogeneity covariance are multiplied by 0.
  expect_error(getAPEs(m, weak.exo = TRUE), "weak.exo = TRUE has no effect without n.pop", fixed = TRUE)
  expect_error(getAPEs(m, sampling.fe = "unrestricted"), "sampling.fe = \"unrestricted\" has no effect without n.pop", fixed = TRUE)
  expect_error(getAPEs(biasCorr(m), sampling.fe = "unr"), "sampling.fe = \"unrestricted\" has no effect without n.pop", fixed = TRUE)
  base <- getAPEs(m)
  expect_identical(getAPEs(m, weak.exo = FALSE, sampling.fe = "independence"), base)
  expect_false(identical(getAPEs(m, n.pop = 1000, sampling.fe = "unrestricted", weak.exo = TRUE)$vcov, base$vcov))
  # An infinite population: factor 1.
  inf <- getAPEs(m, n.pop = Inf)
  big <- getAPEs(m, n.pop = 1e15)
  expect_equal(inf$vcov, big$vcov, tolerance = 1e-12)
  expect_false(isTRUE(all.equal(inf$vcov, base$vcov)))
})
