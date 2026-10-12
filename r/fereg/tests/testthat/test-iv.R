skip_if_not(fixest_ok())

test_that("IV with one and two endogenous variables matches fixest", {
  for (fml in list(
    y ~ x1 | id + year | e1 ~ z1 + z2,
    y ~ x1 + x2 | e1 ~ z1 + z2,
    y ~ x1 | id | e1 + e2 ~ z1 + z2 + x3,
    y ~ x1 | e1 + e2 ~ z1 + z2 + x3
  )) {
    for (v in list("iid", "hetero", ~id)) {
      ours <- fe_feols(fml, panel, vcov = v)
      theirs <- fx_feols(fml, panel, vcov = v)
      expect_same_fit(ours, theirs)
    }
  }
})

test_that("IV names, residuals and first stages match fixest", {
  ours <- fe_feols(y ~ x1 | id | e1 ~ z1 + z2, panel)
  theirs <- fx_feols(y ~ x1 | id | e1 ~ z1 + z2, panel)
  expect_identical(names(coef(ours)), c("fit_e1", "x1"))
  expect_equal(resid(ours), unname(resid(theirs)), tolerance = 1e-8)
  fs <- summary(ours, stage = 1)
  fst <- summary(theirs, stage = 1)
  expect_identical(names(coef(fs)), names(coef(fst)))
  expect_equal(unname(coef(fs)), unname(coef(fst)), tolerance = 1e-8)
  expect_equal(unname(se(fs)), unname(fixest::se(fst)), tolerance = 1e-8)
  for (t in c("ivf1", "wh")) {
    a <- fitstat(ours, t)[[1]]
    b <- fixest::fitstat(theirs, t)[[1]]
    expect_equal(a$stat, unname(b$stat), tolerance = 1e-8, label = t)
    expect_equal(a$p, unname(b$p), tolerance = 1e-6, label = t)
  }
  over <- fe_feols(y ~ x1 | id | e1 ~ z1 + z2, panel)
  expect_equal(fitstat(over, "sargan")[[1]]$stat, unname(fixest::fitstat(theirs, "sargan")[[1]]$stat), tolerance = 1e-8)
  expect_output(print(ours), "TSLS estimation: Second stage")
  expect_equal(unname(fitstat(ours, "ivwald1")[[1]]$stat), unname(fixest::fitstat(theirs, "ivwald1")[[1]]$stat), tolerance = 1e-8)
})

test_that("IV predictions use the endogenous variables", {
  ours <- fe_feols(y ~ x1 | id | e1 ~ z1 + z2, panel)
  theirs <- fx_feols(y ~ x1 | id | e1 ~ z1 + z2, panel)
  expect_equal(predict(ours, panel[1:50, ]), unname(predict(theirs, panel[1:50, ])), tolerance = 1e-8)
})
