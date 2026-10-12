skip_if_not(fixest_ok())

test_that("multiple estimation matches fixest model by model", {
  ours <- fe_feols(c(y, x3) ~ x1 + sw(x2, e1) | csw0(id, year), panel)
  theirs <- fx_feols(c(y, x3) ~ x1 + sw(x2, e1) | csw0(id, year), panel)
  expect_s3_class(ours, "fereg_multi")
  expect_identical(names(ours), names(theirs))
  for (k in seq_along(theirs)) expect_same_fit(ours[[k]], theirs[[k]])
  expect_output(print(ours), "Dependent Var.")
  cf <- coef(ours)
  expect_identical(nrow(cf), length(theirs))
})

test_that("split samples match fixest", {
  ours <- fe_feols(y ~ x1 | id, panel, fsplit = ~region)
  theirs <- fx_feols(y ~ x1 | id, panel, fsplit = ~region)
  expect_identical(names(ours), names(theirs))
  for (k in seq_along(theirs)) expect_same_fit(ours[[k]], theirs[[k]])
})

test_that("subset matches fixest", {
  ours <- fe_feols(y ~ x1 | id, panel, subset = ~ year > 2003)
  theirs <- fx_feols(y ~ x1 | id, panel, subset = ~ year > 2003)
  expect_same_fit(ours, theirs)
})

test_that("tidiers", {
  m <- fe_feols(y ~ x1 + x2 | id, panel)
  td <- fereg:::tidy.fereg(m, conf.int = TRUE)
  expect_identical(td$term, c("x1", "x2"))
  expect_equal(td$conf.low, confint(m)[[1]])
  gl <- fereg:::glance.fereg(m)
  expect_equal(gl$nobs, nobs(m))
  skip_if_not_installed("generics")
  expect_identical(generics::tidy(m)$term, c("x1", "x2"))
})
