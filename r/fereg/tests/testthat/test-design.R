design_names <- function(fml, data, has_fe = FALSE) {
  m <- fereg:::parse_fml(fml)[[1]]
  fereg:::linear_design(m$rhs_terms, data, environment(fml), has_fe)$names
}

test_that("plain numeric columns are passed without copies", {
  d <- panel
  m <- fereg:::parse_fml(y ~ x1 + x2)[[1]]
  lin <- fereg:::linear_design(m$rhs_terms, d, environment(), FALSE)
  expect_identical(lin$names, c("x1", "x2"))
  expect_true(lin$intercept)
  expect_identical(lin$cols[[1]], d$x1)
})

test_that("model.matrix coding and names", {
  d <- panel
  d$fk <- factor(d$k)
  d$l <- d$x3 > 0
  cases <- list(
    y ~ x1:x2 + x1,
    y ~ x1 * x2,
    y ~ factor(h) + x1,
    y ~ k + x1 - 1,
    y ~ fk + k - 1,
    y ~ x1 + x2:fk,
    y ~ log(expo) + l,
    y ~ poly(x1, 2) + x2
  )
  for (fml in cases) {
    mm <- stats::model.matrix(fml, d)
    expected <- setdiff(colnames(mm), "(Intercept)")
    expect_identical(design_names(fml, d), expected, label = deparse(fml))
    lin <- fereg:::linear_design(fereg:::parse_fml(fml)[[1]]$rhs_terms, d, environment(), FALSE)
    expect_equal(unname(do.call(cbind, lin$cols)), unname(mm[, expected, drop = FALSE]), label = deparse(fml))
  }
})

test_that("i() names follow fixest", {
  d <- panel
  expect_identical(design_names(y ~ i(h) + x1, d), c("h::2", "h::3", "h::4", "h::5", "x1"))
  expect_identical(design_names(y ~ i(h, x1), d), paste0("h::", 1:5, ":x1"))
  expect_identical(design_names(y ~ i(h, ref = 2), d, TRUE), paste0("h::", c(1, 3, 4, 5)))
  expect_identical(design_names(y ~ i(h, keep = 2:3), d), c("h::2", "h::3"))
  expect_identical(design_names(y ~ i(h) - 1, d), paste0("h::", 1:5))
  ff <- design_names(y ~ i(k, i.h), d)
  expect_identical(ff[1:3], c("k::u:h::2", "k::u:h::3", "k::u:h::4"))
  expect_length(ff, 14)
  expect_identical(design_names(y ~ i(h, 3), d), paste0("h::", c(1, 2, 4, 5)))
})

test_that("i() columns", {
  d <- panel[1:20, ]
  d$h[3] <- NA
  lin <- fereg:::linear_design(fereg:::parse_fml(y ~ i(h, x1))[[1]]$rhs_terms, d, environment(), FALSE)
  m <- do.call(cbind, lin$cols)
  expect_true(all(is.na(m[3, ])))
  expect_equal(rowSums(m[-3, ]), d$x1[-3])
})

test_that("names match fixest coefficient names", {
  skip_if_not(fixest_ok())
  d <- panel
  for (fml in list(y ~ factor(h) + x1, y ~ i(h, ref = 2) + x1, y ~ i(k, x1) + x2, y ~ k + x1, y ~ x1 * x2)) {
    fx <- fixest::feols(fml, d, notes = FALSE)
    expect_identical(c("(Intercept)", design_names(fml, d)), names(coef(fx)), label = deparse(fml))
  }
  for (fml in list(y ~ factor(h) + x1 | id, y ~ i(h) + x1 | id)) {
    fx <- fixest::feols(fml, d, notes = FALSE)
    expect_identical(design_names(fml, d, TRUE), names(coef(fx)), label = deparse(fml))
  }
})

test_that("factor columns for the core", {
  expect_identical(fereg:::factor_column(c(2.5, 1), "a"), c(2.5, 1))
  expect_identical(fereg:::factor_column(c(3L, 1L), "a"), c(3L, 1L))
  ch <- fereg:::factor_column(c("b", NA, "a", "b"), "a")
  expect_identical(as.vector(ch), c(1L, NA, 2L, 1L))
  expect_identical(attr(ch, "levels"), c("b", "a"))
  f <- factor(c("x", "y"))
  expect_identical(fereg:::factor_column(f, "f"), f)
})
