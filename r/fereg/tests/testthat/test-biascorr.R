# biasCorr(): the analytical correction against alpaca 0.3.5 (exact control), the corrected
# estimation's methods, and every rejected combination.

# Corrected coefficients (from the uncorrected fit: 1e-8), and what depends on the fixed effects
# solved again at them (alpaca's solve is imprecise: 1e-6): variances and linear predictor.
check_bc <- function(bc, abc, d, fe, group) {
  expect_close(coef(bc), coef(abc), 1e-8, paste(group, "coef"))
  expect_close(bc$bias.term, abc$bias.term, 1e-8, paste(group, "bias.term"))
  expect_close(bc$coefficients.uncorr, abc$coefficients.uncorr, 1e-8, paste(group, "coef.uncorr"))
  expect_equal(coef(bc), bc$coefficients.uncorr - bc$bias.term, tolerance = 1e-14)
  expect_close(vcov(bc, vcov = "iid", ssc = ssc(K.adj = FALSE)), vcov(abc), 1e-6, paste(group, "vcov hessian"))
  expect_close(vcov(bc, vcov = "hetero", ssc = ssc(K.adj = FALSE, G.adj = FALSE)), vcov(abc, type = "sandwich"),
    1e-6, paste(group, "vcov sandwich"))
  cl <- stats::as.formula(paste("~", fe[1]))
  expect_close(vcov(bc, vcov = cl, ssc = ssc(K.adj = FALSE, G.adj = FALSE)), vcov(abc, type = "clustered", cluster = cl),
    1e-6, paste(group, "vcov clustered"))
  # alpaca's rows: the estimation rows sorted (stably) by the fixed-effect variables.
  e <- d[bc$obs_selection %||% seq_len(nrow(d)), , drop = FALSE]
  ord <- do.call(order, c(lapply(fe, function(v) e[[v]]), method = "radix"))
  expect_close(bc$linear.predictors[ord], abc$eta, 1e-6, paste(group, "eta"))
}

test_that("analytical correction matches alpaca: classic two-way, balanced and unbalanced, L = 0, 1, 2", {
  skip_if_not(alpaca_ok())
  for (case in list(
    list(name = "balanced", d = sim_classic(60, 12, 1)),
    list(name = "unbalanced", d = sim_classic(80, 10, 2, sd_a = 2, drop = 0.15)),
    list(name = "dynamic", d = sim_classic(60, 10, 3, dynamic = TRUE, drop = 0.1))
  )) {
    d <- case$d
    fml <- if (is.null(d$ylag)) y ~ x1 + d | i + t else y ~ ylag + x1 + d | i + t
    m <- fe_logit(fml, d)
    a <- al_feglm(fml, d)
    expect_equal(nobs(m), a$nobs[["nobs"]])
    for (L in 0:2) {
      bc <- biasCorr(m, L = L)
      expect_s3_class(bc, c("fereg_biascorr", "fereg"), exact = TRUE)
      expect_identical(bc$bandwidth, L)
      check_bc(bc, alpaca::biasCorr(a, L = L), d, c("i", "t"), sprintf("classic %s L=%d", case$name, L))
    }
  }
})

test_that("analytical correction matches alpaca: one-way, weights, character and factor identifiers", {
  skip_if_not(alpaca_ok())
  d <- sim_classic(60, 10, 3, dynamic = TRUE, drop = 0.1)
  m <- fe_logit(y ~ ylag + x1 + d | i, d)
  a <- al_feglm(y ~ ylag + x1 + d | i, d)
  for (L in 0:2) check_bc(biasCorr(m, L = L), alpaca::biasCorr(a, L = L), d, "i", sprintf("one-way L=%d", L))
  # The individual effect is the only bias of a one-way model.
  bc1 <- biasCorr(m, L = 1)
  expect_identical(bc1$bias, "individual")
  expect_equal(coef(biasCorr(m, L = 1, bias = "individual")), coef(bc1), tolerance = 1e-14)

  set.seed(11)
  d$w <- runif(nrow(d), 0.5, 2)
  m <- fe_logit(y ~ ylag + x1 + d | i + t, d, weights = ~w)
  a <- al_feglm(y ~ ylag + x1 + d | i + t, d, weights = "w")
  check_bc(biasCorr(m, L = 2), alpaca::biasCorr(a, L = 2), d, c("i", "t"), "weights L=2")

  # Time order of the spectral term: levels of the second fixed effect within each individual,
  # character strings in C-locale order and factors in level order (alpaca sorts the data).
  d <- sim_classic(50, 9, 21, dynamic = TRUE)
  d <- d[sample(nrow(d)), ]
  d$ic <- sprintf("u%d", d$i)
  d$tc <- c("B", "a", "C", "b", "D", "c", "E", "d", "F")[d$t]
  d$tf <- factor(d$t, levels = c(9, 1:8))
  for (fe in list(c("ic", "tc"), c("tf", "ic"), c("ic", "tf"))) {
    fml <- stats::as.formula(paste("y ~ ylag + x1 + d |", paste(fe, collapse = " + ")))
    m <- fe_logit(fml, d)
    a <- al_feglm(fml, d)
    check_bc(biasCorr(m, L = 2), alpaca::biasCorr(a, L = 2), d, fe, paste("identifiers", paste(fe, collapse = "+"), "L=2"))
  }
})

test_that("analytical correction matches alpaca: network with two and three fixed effects", {
  skip_if_not(alpaca_ok())
  d <- sim_network(10, 5, 4)
  fe3 <- c("exp_time", "imp_time", "pair")
  m <- fe_logit(y ~ x + rta | exp_time + imp_time + pair, d)
  a <- al_feglm(y ~ x + rta | exp_time + imp_time + pair, d)
  for (L in c(0L, 2L)) {
    bc <- biasCorr(m, L = L, panel.structure = "network")
    check_bc(bc, alpaca::biasCorr(a, L = L, panel.structure = "network"), d, fe3, sprintf("network 3-way L=%d", L))
  }
  m2 <- fe_logit(y ~ x + rta | exp_time + imp_time, d)
  a2 <- al_feglm(y ~ x + rta | exp_time + imp_time, d)
  bc0 <- biasCorr(m2, panel.structure = "network")
  check_bc(bc0, alpaca::biasCorr(a2, panel.structure = "network"), d, fe3[1:2], "network 2-way L=0")
  # No pair effect: the bandwidth has nothing to act on (alpaca ignores it silently).
  expect_warning(bc2 <- biasCorr(m2, L = 2, panel.structure = "network"), "L ignored")
  expect_equal(coef(bc2), coef(bc0), tolerance = 1e-14)
  check_bc(bc2, alpaca::biasCorr(a2, L = 2, panel.structure = "network"), d, fe3[1:2], "network 2-way L=2")
  # Two fixed effects: classic and network corrections coincide (alpaca).
  expect_equal(coef(biasCorr(m2)), coef(bc0), tolerance = 1e-12)
})

test_that("bias = 'individual' / 'time' correct one bias (alpaca's terms of each fixed effect)", {
  skip_if_not(alpaca_ok())
  d <- sim_classic(60, 10, 3, dynamic = TRUE, drop = 0.1)
  m <- fe_logit(y ~ ylag + x1 + d | i + t, d)
  a <- al_feglm(y ~ ylag + x1 + d | i + t, d)
  for (L in c(0L, 2L)) {
    terms <- list()
    for (b in c("both", "individual", "time")) {
      if (b == "time" && L > 0) next
      bc <- biasCorr(m, L = L, bias = b)
      ref <- alpaca_bias_term(a, b, L)
      expect_close(bc$bias.term, ref, 1e-8, sprintf("bias=%s L=%d bias.term", b, L))
      expect_close(coef(bc), stats::coef(a) - ref, 1e-8, sprintf("bias=%s L=%d coef", b, L))
      expect_identical(bc$bias, b)
      terms[[b]] <- bc$bias.term
    }
    if (L == 0) expect_equal(terms$both, terms$individual + terms$time, tolerance = 1e-12)
  }
  expect_close(alpaca_bias_term(a, "both", 2), alpaca::biasCorr(a, L = 2)$bias.term, 1e-14, "bias reference vs alpaca")
})

test_that("the corrected estimation works with fereg's methods at the corrected point", {
  d <- sim_classic(40, 8, 7)
  m <- fe_logit(y ~ x1 + d | i + t, d)
  bc <- biasCorr(m)
  expect_identical(bc$method, "analytical")
  expect_null(bc$jackknife)
  expect_identical(bc$panel.structure, "classic")
  expect_identical(bc$nobs.full, nrow(d))
  expect_output(print(bc), "Bias correction: analytical (L = 0), individual and time effects", fixed = TRUE)
  expect_output(print(summary(bc, vcov = ~i)), "Bias correction: analytical (L = 0)", fixed = TRUE)
  expect_output(print(summary(bc, vcov = ~i)), "Clustered (i)", fixed = TRUE)
  s <- summary(bc, vcov = "hetero")
  expect_equal(as.vector(se(s)), unname(sqrt(diag(vcov(bc, vcov = "hetero")))), tolerance = 1e-14)
  expect_equal(as.vector(se(bc)), unname(sqrt(diag(vcov(bc)))), tolerance = 1e-14)
  expect_equal(coef(s), coef(bc))
  expect_false(isTRUE(all.equal(unname(coef(bc)), unname(coef(m)))))
  # Fixed effects, predictions and fitted values at the corrected point (estimation rows `e`).
  e <- d[bc$obs_selection %||% seq_len(nrow(d)), ]
  expect_equal(unname(predict(bc, newdata = e, type = "link")), bc$linear.predictors, tolerance = 1e-10)
  expect_equal(fitted(bc), stats::plogis(bc$linear.predictors), tolerance = 1e-14)
  fe <- fixef(bc)
  expect_equal(unname(fe$i[as.character(e$i)] + fe$t[as.character(e$t)] + as.matrix(e[, c("x1", "d")]) %*% coef(bc))[, 1],
    bc$linear.predictors, tolerance = 1e-10)
  expect_false(isTRUE(all.equal(fe$i, fixef(m)$i)))
  # The fixed effects are the maximum likelihood ones given the corrected index: zero scores.
  r <- e$y - fitted(bc)
  expect_lt(max(abs(tapply(r, e$i, sum)), abs(tapply(r, e$t, sum))), 1e-8)
  et <- etable(m, bc)
  expect_s3_class(et, "fereg_etable")
  expect_true(any(grepl(fmt_coef(coef(bc)[["x1"]], 4), et[[3]], fixed = TRUE)))
  expect_false(any(grepl(fmt_coef(coef(bc)[["x1"]], 4), et[[2]], fixed = TRUE)))
  expect_equal(confint(bc)[[1]], as.vector(coef(bc) - stats::qnorm(0.975) * se(bc)), tolerance = 1e-12)
  expect_error(vcov(bc, vcov = "crv3"), "CRV3 is not available after bias correction", fixed = TRUE)
  expect_error(summary(bc, vcov = crv3 ~ i), "CRV3 is not available after bias correction", fixed = TRUE)
  expect_error(biasCorr(bc), "already bias-corrected", fixed = TRUE)
  # The estimation's variance type carries over.
  mc <- fe_logit(y ~ x1 + d | i + t, d, vcov = ~i)
  expect_identical(biasCorr(mc)$vcov_info$label, "Clustered (i)")
  expect_equal(vcov(biasCorr(mc)), vcov(bc, vcov = ~i), tolerance = 1e-12)
})

test_that("biasCorr() rejects unsupported objects and combinations (R checks, then the core's texts)", {
  d <- sim_classic(30, 6, 8)
  d$k <- rep(1:3, length.out = nrow(d))
  m2 <- fe_logit(y ~ x1 + d | i + t, d)
  m1 <- fe_logit(y ~ x1 + d | i, d)
  m3 <- fe_logit(y ~ x1 + d | i + t + k, d)
  m0 <- fe_logit(y ~ x1 + d, d)
  expect_error(biasCorr(), "'object' has to be specified.", fixed = TRUE)
  expect_error(biasCorr(stats::lm(y ~ x1, d)), "'biasCorr' called on a non-'feglm' object.", fixed = TRUE)
  expect_error(biasCorr(feols(y ~ x1 | i, d, notes = FALSE)), "'biasCorr' called on a non-'feglm' object.", fixed = TRUE)
  expect_error(biasCorr(fepois(y ~ x1 | i, d, notes = FALSE)), "'biasCorr' currently only supports binary choice models.", fixed = TRUE)
  expect_error(biasCorr(fe_logit(c(y, d) ~ x1 | i, d)), "takes a single estimation", fixed = TRUE)
  expect_error(biasCorr(fe_logit(y ~ 1 | i + t, d)), "biasCorr() needs at least one regressor", fixed = TRUE)
  slopes <- "bias correction and APEs need intercept-only fixed effects (no varying slopes)"
  expect_error(biasCorr(fe_logit(y ~ d | t[x1], d)), slopes, fixed = TRUE)
  expect_error(biasCorr(fe_logit(y ~ d | i + t[x1], d)), slopes, fixed = TRUE)
  classic <- "panel.structure == 'classic' expects a one- or two-way fixed effects model."
  network <- "panel.structure == 'network' expects a two- or three-way fixed effects model."
  expect_error(biasCorr(m3), classic, fixed = TRUE)
  expect_error(biasCorr(m0), classic, fixed = TRUE)
  expect_error(biasCorr(m1, panel.structure = "network"), network, fixed = TRUE)
  expect_error(biasCorr(m2, panel.structure = "netw0rk"), "should be one of")
  expect_error(biasCorr(m2, method = "bootstrap"), "should be one of")
  expect_error(biasCorr(m2, jackknife = "ss3", method = "jackknife"), "should be one of")
  expect_error(biasCorr(m2, bias = "pair"), "should be one of")
  for (bad in list(-1, 1.5, NA, "1", c(1, 2))) {
    expect_error(biasCorr(m2, L = bad), "`L` must be a non-negative integer", fixed = TRUE)
    expect_error(biasCorr(m2, method = "jackknife", multiple = bad), "`multiple` must be a non-negative integer", fixed = TRUE)
  }
  expect_error(biasCorr(fe_logit(y ~ x1 + d | i + t, d, vcov = crv3 ~ i)), "CRV3 is not available after bias correction", fixed = TRUE)
  # Analytical: no jackknife options.
  expect_error(biasCorr(m2, jackknife = "ss1"), "`jackknife` needs method = \"jackknife\"", fixed = TRUE)
  expect_error(biasCorr(m2, jackknife = "ss2"), "`jackknife` needs method = \"jackknife\"", fixed = TRUE)
  expect_error(biasCorr(m2, multiple = 2), "`multiple` needs method = \"jackknife\"", fixed = TRUE)
  expect_error(biasCorr(m2, split = "time"), "`split` is only used with `multiple`", fixed = TRUE)
  # Jackknife: classic panels, no bandwidth, multiple only for ss1 / ss2.
  expect_error(biasCorr(m3, method = "jackknife", panel.structure = "network"), "the jackknife is not available for the network structure", fixed = TRUE)
  expect_error(biasCorr(m2, method = "jackknife", L = 1), "bandwidth L = 1 has no effect: it only applies to the analytical correction", fixed = TRUE)
  for (v in c("js", "sj", "jj", "double")) {
    expect_error(biasCorr(m2, method = "jackknife", jackknife = v, multiple = 2),
      "jackknife: multiple partitions are only available with the ss1 and ss2 variants", fixed = TRUE)
  }
  expect_error(biasCorr(m2, method = "jackknife", split = "time"), "`split` is only used with `multiple`", fixed = TRUE)
  # Which bias.
  both <- "network structure: only the correction of both biases is available"
  expect_error(biasCorr(m3, panel.structure = "network", bias = "individual"), both, fixed = TRUE)
  expect_error(biasCorr(m2, panel.structure = "network", bias = "time"), both, fixed = TRUE)
  time_fe <- "the time-effect bias needs a time fixed effect (a two-way model)"
  expect_error(biasCorr(m1, bias = "time"), time_fe, fixed = TRUE)
  expect_error(biasCorr(m1, method = "jackknife", bias = "time", panel.id = ~ i + t), time_fe, fixed = TRUE)
  expect_error(biasCorr(m2, L = 2, bias = "time"), "bandwidth L = 2 has no effect: it only applies to the individual-effect bias", fixed = TRUE)
  # Random partitions that would not change the split dimension.
  no_time <- "multiple partitions have no effect: the subpanels of this bias split only the time dimension, which no partition permutes"
  expect_error(biasCorr(m2, method = "jackknife", bias = "individual", multiple = 2, split = "individuals"), no_time, fixed = TRUE)
  expect_error(biasCorr(m1, method = "jackknife", multiple = 2, split = "individuals", panel.id = ~ i + t), no_time, fixed = TRUE)
  expect_error(biasCorr(m2, method = "jackknife", jackknife = "ss1", bias = "time", multiple = 2, split = "time"),
    "multiple partitions have no effect: the subpanels of this bias split only the individual dimension, which no partition permutes", fixed = TRUE)
  # panel.id: only for the one-way jackknife, whose fixed effect must be its panel variable.
  expect_error(biasCorr(m1, panel.id = ~ i + t), "`panel.id` is only used by the jackknife of a model with one fixed effect", fixed = TRUE)
  expect_error(biasCorr(m2, method = "jackknife", panel.id = ~ i + t), "`panel.id` is only used by the jackknife", fixed = TRUE)
  expect_error(biasCorr(m1, method = "jackknife"), "give the time variable with `panel.id = ~id + time`", fixed = TRUE)
  expect_error(biasCorr(m1, method = "jackknife", panel.id = ~ t + i), "the fixed effect (`i`) must be the panel variable of `panel.id` (`t`)", fixed = TRUE)
  expect_error(biasCorr(m1, method = "jackknife", panel.id = ~ i + period), "the variable 'period' is not in the data set", fixed = TRUE)
  expect_error(biasCorr(m1, method = "jackknife", panel.id = ~i), "`panel.id` must give the panel (unit) and the time variables", fixed = TRUE)
  mp <- fe_logit(y ~ x1 + d | t, d, panel.id = ~ i + t)
  expect_error(biasCorr(mp, method = "jackknife"), "the fixed effect (`t`) must be the panel variable of `panel.id` (`i`)", fixed = TRUE)
  # Double jackknife: two single-variable fixed effects.
  mi <- fe_logit(y ~ x1 + d | i^k + t, d)
  expect_error(biasCorr(mi, method = "jackknife", jackknife = "double"), "the double jackknife needs two single-variable fixed effects", fixed = TRUE)
  # The data must still be there.
  local({
    dd <- d
    mm <- fe_logit(y ~ x1 + d | i + t, dd)
    rm(dd)
    expect_error(biasCorr(mm), "biasCorr() estimates the model again on it", fixed = TRUE)
  })
})
