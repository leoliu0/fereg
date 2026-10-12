skip_if_not(fixest_ok())

test_that("Poisson matches fixest (fixed effects, offset, weights, variances)", {
  d <- panel
  d$count[d$id %in% 1:4] <- 0 # groups with only zeros are removed (perfect fit)
  for (args in list(
    list(count ~ x1 + x2 | id + year),
    list(count ~ x1 + x2, vcov = "hetero"),
    list(count ~ x1 + x2 | id, offset = ~ log(expo)),
    list(count ~ x1 + x2 | id + firm, weights = ~w, vcov = ~id),
    list(count ~ x1 | id + year, vcov = ~ id + firm)
  )) {
    ours <- do.call(fe_fepois, c(args, list(data = d)))
    theirs <- do.call(fx_fepois, c(args, list(data = d)))
    expect_same_fit(ours, theirs, tol = 1e-6)
    expect_equal(ours$loglik, theirs$loglik, tolerance = 1e-6)
    expect_equal(deviance(ours), deviance(theirs), tolerance = 1e-6)
    expect_equal(fitted(ours), unname(fitted(theirs)), tolerance = 1e-6)
  }
})

test_that("logit matches fixest", {
  for (args in list(
    list(binary ~ x1 + x2 | id + year),
    list(binary ~ x1 + x2, vcov = "hetero"),
    list(binary ~ x1 | id, vcov = ~id),
    list(binary ~ x1 + x2 | id, weights = ~w)
  )) {
    ours <- do.call(fe_feglm, c(args, list(data = panel, family = "logit")))
    theirs <- do.call(fx_feglm, c(args, list(data = panel, family = "logit")))
    expect_same_fit(ours, theirs, tol = 1e-6)
    expect_equal(ours$loglik, theirs$loglik, tolerance = 1e-6)
    expect_equal(unname(r2(ours, c("pr2", "apr2"))), unname(fixest::r2(theirs, c("pr2", "apr2"))), tolerance = 1e-6)
  }
  b <- fe_feglm(binary ~ x1 | id, panel, family = binomial())
  expect_output(print(b), "GLM estimation, family = binomial")
})

test_that("GLM residual types, predictions and post-hoc variances match fixest", {
  ours <- fe_fepois(count ~ x1 + x2 | id, panel, offset = ~ log(expo))
  # Called directly: fixest's predict() re-evaluates the offset argument from the call.
  theirs <- fixest::fepois(count ~ x1 + x2 | id, panel, offset = ~ log(expo), fixef.tol = 3e-12, glm.tol = 1e-12, nthreads = 1, notes = FALSE)
  for (t in c("response", "deviance", "pearson", "working")) {
    expect_equal(resid(ours, type = t), unname(resid(theirs, type = t)), tolerance = 1e-6, label = t)
  }
  nd <- panel[1:40, ]
  expect_equal(predict(ours, nd), unname(predict(theirs, nd)), tolerance = 1e-6)
  expect_equal(predict(ours, nd, type = "link"), unname(predict(theirs, nd, type = "link")), tolerance = 1e-6)
  expect_equal(plain(vcov(ours, vcov = ~firm)), plain(stats::vcov(theirs, vcov = ~firm)), tolerance = 1e-6)
  expect_output(print(ours), "Poisson estimation, Dep. Var.: count")
})
