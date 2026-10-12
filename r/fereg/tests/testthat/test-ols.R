skip_if_not(fixest_ok())

test_that("OLS with 0 to 3 fixed effects matches fixest", {
  for (fml in list(
    y ~ x1 + x2,
    y ~ x1 + x2 | id,
    y ~ x1 + x2 | id + year,
    y ~ x1 + x2 | id + year + firm,
    y ~ x1 + x2 - 1
  )) {
    expect_same_fit(fe_feols(fml, panel), fx_feols(fml, panel))
  }
})

test_that("interacted fixed effects and varying slopes match fixest", {
  for (fml in list(
    y ~ x1 + x2 | id + year^firm,
    y ~ x1 | id + firm[x2],
    y ~ x1 | id[[x2]] + year,
    y ~ x1 | id[x2, x3] + year
  )) {
    expect_same_fit(fe_feols(fml, panel), fx_feols(fml, panel))
  }
})

test_that("weights match fixest", {
  expect_same_fit(fe_feols(y ~ x1 + x2 | id + year, panel, weights = ~w), fx_feols(y ~ x1 + x2 | id + year, panel, weights = ~w))
  expect_same_fit(fe_feols(y ~ x1 + x2, panel, weights = ~w, vcov = "hetero"), fx_feols(y ~ x1 + x2, panel, weights = ~w, vcov = "hetero"))
})

test_that("variance types match fixest", {
  fml <- y ~ x1 + x2 | id + year
  for (v in list("iid", "hetero", ~id, ~region, ~grp, ~ id + year, ~ firm + region, "twoway", ~ id^year)) {
    # Multiway variances may need fixest's eigenvalue fix, which both packages report.
    psd <- function(w) if (grepl("not positive definite", conditionMessage(w))) invokeRestart("muffleWarning")
    ours <- withCallingHandlers(fe_feols(fml, panel, vcov = v), warning = psd)
    expect_same_fit(ours, withCallingHandlers(fx_feols(fml, panel, vcov = v), warning = psd))
  }
  ssc_args <- list(list(K.fixef = "full"), list(K.fixef = "none"), list(G.df = "conventional"), list(K.adj = FALSE, t.df = "conventional"))
  for (a in ssc_args) {
    ours <- fe_feols(fml, panel, vcov = ~id, ssc = do.call(ssc, a))
    expect_same_fit(ours, fx_feols(fml, panel, vcov = ~id, ssc = do.call(fixest::ssc, a)))
  }
})

test_that("post-hoc variances match fixest", {
  ours <- fe_feols(y ~ x1 + x2 | id + year, panel)
  theirs <- fx_feols(y ~ x1 + x2 | id + year, panel)
  for (v in list("hetero", ~id, ~ id + firm)) {
    expect_equal(plain(vcov(ours, vcov = v)), plain(stats::vcov(theirs, vcov = v)), tolerance = 1e-8)
    expect_equal(unname(se(summary(ours, vcov = v))), unname(fixest::se(summary(theirs, vcov = v))), tolerance = 1e-8)
  }
  expect_equal(
    plain(vcov(ours, vcov = ~id, ssc = ssc(K.fixef = "full"))),
    plain(stats::vcov(theirs, vcov = ~id, ssc = fixest::ssc(K.fixef = "full"))),
    tolerance = 1e-8
  )
})

test_that("missing values, singletons and missing cluster values match fixest", {
  d <- panel
  d$y[c(3, 50)] <- NA
  d$x1[7] <- NA
  d$id[11] <- NA
  d$region[c(20, 21)] <- NA
  d$id[c(30, 31, 32)] <- 1000:1002 # three singleton levels
  for (v in list("iid", ~region)) {
    ours <- fe_feols(y ~ x1 + x2 | id + year, d, vcov = v)
    theirs <- fx_feols(y ~ x1 + x2 | id + year, d, vcov = v)
    expect_same_fit(ours, theirs)
    expect_identical(ours$obs_selection, as.integer(fixest::obs(theirs)))
  }
  ours <- fe_feols(y ~ x1 + x2 | id + year, d)
  expect_error(summary(ours, vcov = ~region), "missing|NA")
})

test_that("fit statistics match fixest", {
  for (fml in list(y ~ x1 + x2 | id + year, y ~ x1 + x2)) {
    ours <- fe_feols(fml, panel, weights = ~w)
    theirs <- fx_feols(fml, panel, weights = ~w)
    for (t in c("rmse", "r2", "ar2", "ll", "aic", "bic")) {
      expect_equal(as.numeric(fitstat(ours, t)[[1]]), as.numeric(fixest::fitstat(theirs, t)[[1]]), tolerance = 1e-8, label = t)
    }
    expect_equal(sigma(ours), sigma(theirs), tolerance = 1e-8)
    expect_equal(deviance(ours), deviance(theirs), tolerance = 1e-8)
    expect_equal(as.numeric(logLik(ours)), as.numeric(logLik(theirs)), tolerance = 1e-8)
  }
  ours <- fe_feols(y ~ x1 + x2 | id + year, panel)
  theirs <- fx_feols(y ~ x1 + x2 | id + year, panel)
  expect_equal(unname(r2(ours, c("wr2", "war2"))), unname(fixest::r2(theirs, c("wr2", "war2"))), tolerance = 1e-8)
})

test_that("residuals, fitted values and fixed effects match fixest", {
  ours <- fe_feols(y ~ x1 + x2 | id + year + firm, panel)
  theirs <- fx_feols(y ~ x1 + x2 | id + year + firm, panel)
  expect_equal(resid(ours), unname(resid(theirs)), tolerance = 1e-8)
  expect_equal(fitted(ours), unname(fitted(theirs)), tolerance = 1e-8)
  fo <- fixef(ours)
  ft <- fixest::fixef(theirs)
  expect_identical(names(fo), names(ft))
  for (k in names(ft)) {
    expect_identical(names(fo[[k]]), names(ft[[k]]))
    expect_equal(unname(fo[[k]]), unname(ft[[k]]), tolerance = 1e-8)
  }
  expect_identical(unname(attr(fo, "references")), unname(attr(ft, "references")))

  so <- fixef(fe_feols(y ~ x1 | id + firm[x2], panel))
  st <- fixest::fixef(fx_feols(y ~ x1 | id + firm[x2], panel))
  expect_identical(names(so), names(st))
  # fixest recovers varying slopes iteratively (about 1e-6 relative here); check against the
  # exact dummy-variable solution, and loosely against fixest.
  exact <- stats::coef(stats::lm(y ~ x1 + factor(id) + factor(firm) + factor(firm):x2 - 1, panel))
  expect_equal(unname(so[["firm[[x2]]"]]), unname(exact[grep(":x2$", names(exact))]), tolerance = 1e-8)
  expect_equal(unname(so[["firm[[x2]]"]]), unname(st[["firm[[x2]]"]]), tolerance = 1e-5)
})

test_that("predictions on new data match fixest, unseen levels are NA", {
  ours <- fe_feols(y ~ x1 + x2 + i(h) | id + year, panel)
  theirs <- fx_feols(y ~ x1 + x2 + i(h) | id + year, panel)
  nd <- panel[1:30, ]
  nd$id[1:3] <- 9999L
  p <- predict(ours, nd)
  expect_true(all(is.na(p[1:3])))
  expect_equal(p[-(1:3)], unname(predict(theirs, nd)[-(1:3)]), tolerance = 1e-8)
  expect_equal(predict(ours), unname(predict(theirs)), tolerance = 1e-8)
})

test_that("factor() and i() regressors match fixest", {
  for (fml in list(y ~ factor(h) + x1 | id, y ~ i(h, ref = 3) + x1 | id + year, y ~ i(k, x1) + x2, y ~ k + x1 | year)) {
    expect_same_fit(fe_feols(fml, panel), fx_feols(fml, panel))
  }
})

test_that("collinear regressors are removed as in fixest", {
  d <- panel
  d$x4 <- d$x1 + d$x2
  d$xid <- d$id * 2
  ours <- fe_feols(y ~ x1 + x2 + x4 + xid | id, d)
  theirs <- fx_feols(y ~ x1 + x2 + x4 + xid | id, d)
  expect_same_fit(ours, theirs)
  expect_identical(ours$collin.var, theirs$collin.var)
})

test_that("printing works", {
  m <- fe_feols(y ~ x1 + x2 | id + year, panel, vcov = ~id)
  expect_output(print(m), "OLS estimation, Dep. Var.: y")
  expect_output(print(m), "Clustered \\(id\\)")
  expect_output(print(m), "Within R2")
  expect_output(print(etable(m, fe_feols(y ~ x1, panel))), "Constant")
  expect_match(as.character(etable(m, tex = TRUE)), "tabular")
})

test_that("fixed effects are computed on demand and cached; lean drops them", {
  m <- fe_feols(y ~ x1 | id + year, panel)
  expect_null(m$fixef_cache$fixef)
  f1 <- fixef(m)
  expect_identical(fixef(summary(m, vcov = "hetero")), f1)
  expect_false(is.null(m$fixef_cache$fixef))
  lean <- fe_feols(y ~ x1 | id + year, panel, lean = TRUE)
  expect_error(fixef(lean), "lean")
  expect_null(lean$residuals)
})
