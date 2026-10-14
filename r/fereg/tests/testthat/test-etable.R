skip_if_not(fixest_ok())

fit_both <- function(d) {
  calls <- list(
    m = quote(feols(mpg ~ wt + hp | cyl + gear, d)),
    m0 = quote(feols(mpg ~ wt + hp, d)),
    mc = quote(feols(mpg ~ wt | cyl^gear, d)),
    p = quote(fepois(carb ~ wt + hp | cyl + gear, d)),
    p0 = quote(fepois(carb ~ wt + hp, d)),
    l = quote(feglm(vs ~ mpg | gear, d, family = "logit")),
    l0 = quote(feglm(vs ~ mpg, d, family = binomial()))
  )
  ns <- function(pkg) lapply(calls, function(cl) {
    cl[[1]] <- call("::", as.name(pkg), cl[[1]])
    suppressWarnings(eval(cl, list(d = d)))
  })
  list(ours = ns("fereg"), theirs = ns("fixest"))
}

test_that("etable matches fixest for OLS-only, GLM-only and mixed sets", {
  fits <- fit_both(mtcars)
  sets <- list(c("m", "p"), c("m", "p", "l"), "m", c("m", "m0"), "mc", "p", c("p", "l"), "m0", "p0",
               "l0", c("m0", "p0"), c("m0", "l0"), c("p", "p0"), c("l", "l0", "m"), c("mc", "p0"))
  for (s in sets) {
    o <- capture.output(print(do.call(fereg::etable, fits$ours[s])))
    t <- capture.output(print(do.call(fixest::etable, fits$theirs[s])))
    expect_identical(o, t, label = paste("etable of", paste(s, collapse = ", ")))
  }
  # The Family row can be forced on or off.
  o <- capture.output(print(fereg::etable(m = fits$ours$m, family = TRUE)))
  t <- capture.output(print(fixest::etable(m = fits$theirs$m, family = TRUE)))
  expect_identical(o, t)
  o <- capture.output(print(fereg::etable(m = fits$ours$m, p = fits$ours$p, family = FALSE)))
  t <- capture.output(print(fixest::etable(m = fits$theirs$m, p = fits$theirs$p, family = FALSE)))
  expect_identical(o, t)
})

test_that("etable fit statistics are formatted as fixest's format_number", {
  set.seed(3)
  xs <- c(0, 1, 32, 6656858, 49671918.83, 1522665.8, -0.04, 99999.99, 1e10 + 0.5, -2.5e-6, 0.5, -0.5,
          2526.5, signif(rnorm(400) * 10^sample(-7:9, 400, TRUE), sample(1:9, 400, TRUE)),
          round(runif(50) * 1e7), round(rnorm(200) * 1e4, 4))
  # fixest's C++ comma formatter overflows (prints "0") from 2^31 to 1e10; fereg prints the number.
  xs <- xs[abs(xs) < 2^31 - 1]
  for (d in list(5, 3, 7, "r3", "r0", "s4", 1)) {
    ds <- fereg:::stats_digits(d)
    ours <- vapply(xs, fereg:::format_stat, "", digits = ds$digits, round = ds$round)
    theirs <- vapply(xs, function(x) unname(fixest:::format_number(x, ds$digits, ds$round)), "")
    expect_identical(ours, theirs, label = paste("digits.stats =", d))
  }
  fits <- fit_both(mtcars)
  for (d in list(3, "r2")) {
    o <- capture.output(print(fereg::etable(m = fits$ours$m, p = fits$ours$p, digits.stats = d)))
    t <- capture.output(print(fixest::etable(m = fits$theirs$m, p = fits$theirs$p, digits.stats = d)))
    expect_identical(o, t)
  }
})

test_that("pseudo R2 of OLS models matches fixest", {
  for (fml in list(mpg ~ wt + hp | cyl + gear, mpg ~ wt + hp, mpg ~ wt - 1)) {
    expect_equal(fereg::r2(fereg::feols(fml, mtcars), c("pr2", "apr2")),
                 fixest::r2(fixest::feols(fml, mtcars), c("pr2", "apr2")), tolerance = 1e-10)
  }
  expect_true(is.na(fereg::r2(fereg::feols(mpg ~ 1, mtcars), "pr2")))
})

test_that("print() and summary() fit-statistic lines match fixest", {
  d <- mtcars
  for (cl in list(
    quote(feols(mpg ~ wt + hp | cyl + gear, d)),
    quote(feols(mpg ~ wt + hp, d)),
    quote(feols(mpg ~ wt + hp | cyl, d, weights = ~qsec, vcov = "hetero")),
    quote(feols(mpg ~ hp | cyl | wt ~ qsec, d)),
    quote(fepois(carb ~ wt + hp | cyl + gear, d)),
    quote(feglm(vs ~ mpg | gear, d, family = "logit")),
    quote(feglm(vs ~ mpg + wt, d, family = binomial())),
    quote(feglm(vs ~ mpg + wt, d, family = binomial(link = "logit"))),
    quote(feglm(carb ~ wt, d, family = poisson()))
  )) {
    run <- function(pkg) {
      cl[[1]] <- call("::", as.name(pkg), cl[[1]])
      m <- suppressWarnings(eval(cl))
      c(capture.output(print(m)), capture.output(print(summary(m, vcov = "hetero"))))
    }
    o <- run("fereg")
    t <- run("fixest")
    # Notes printed by fixest about removed observations are not part of the comparison.
    t <- t[!grepl("^NOTE", t)]
    expect_identical(o, t, label = deparse1(cl))
  }
})
