# biasCorr(method = "jackknife"): every variant x bias of design 7 against a plain R loop over
# fixest refits of the subpanels (helper-biascorr.R: jk_reference()), coefficients and average
# partial effects at 1e-8; multiple(R) partitions reproduced from the same R random numbers.

check_jk <- function(jk, ref, group) {
  expect_s3_class(jk, "fereg_biascorr")
  expect_identical(jk$method, "jackknife")
  expect_close(coef(jk), ref$coef[names(coef(jk))], 1e-8, paste(group, "coef"))
  expect_equal(jk$bias.term, jk$coefficients.uncorr - coef(jk), tolerance = 1e-14)
  ape <- getAPEs(jk)
  expect_close(ape$delta, ref$ape[names(ape$delta)], 1e-8, paste(group, "APE"))
  expect_null(ape$bias.term)
  expect_false(ape$weak.exo)
}

variants <- c("ss2", "ss1", "js", "sj", "jj", "double")

test_that("jackknife variants x biases match fixest refits of the subpanels (two-way)", {
  skip_if_not(fixest_ok())
  # Odd numbers of individuals and periods (the middle unit is in both halves), individuals with
  # a constant outcome (counted in N and T), character identifiers whose sorted order differs
  # from their order of appearance.
  d <- sim_classic(41, 9, 5, sd_a = 1.2)
  d <- d[sample(nrow(d)), ]
  d$id <- paste0("u", d$i)
  fml <- y ~ x1 + d | id + t
  m <- fe_logit(fml, d)
  expect_gt(m$dropped$perfect_fit, 0)
  cache <- new.env()
  for (v in setdiff(variants, "double")) {
    for (b in c("both", "individual", "time")) {
      jk <- biasCorr(m, method = "jackknife", jackknife = v, bias = b)
      expect_identical(jk$jackknife, v)
      expect_identical(jk$bias, b)
      ref <- jk_reference(fml, d, "id", "t", v, b, cache = cache)
      check_jk(jk, ref, sprintf("jackknife two-way %s bias=%s", v, b))
    }
  }
  expect_output(print(biasCorr(m, method = "jackknife", jackknife = "sj")), "Bias correction: jackknife (sj), individual and time effects", fixed = TRUE)
  expect_output(print(biasCorr(m, method = "jackknife", bias = "time")), "Bias correction: jackknife (ss2), time effects", fixed = TRUE)
  # After a jackknife, weak.exo applies as for an uncorrected estimation (it adds a term with n.pop).
  jk <- biasCorr(m, method = "jackknife")
  a0 <- getAPEs(jk, n.pop = 5000)
  a1 <- getAPEs(jk, n.pop = 5000, weak.exo = TRUE)
  expect_false(a0$weak.exo)
  expect_true(a1$weak.exo)
  expect_identical(a1$delta, a0$delta)
  expect_false(isTRUE(all.equal(a1$vcov, a0$vcov)))
  expect_equal(getAPEs(jk, weak.exo = TRUE)$vcov, getAPEs(jk)$vcov)
})

test_that("jackknife variants match fixest refits (one-way: time variable from panel.id)", {
  skip_if_not(fixest_ok())
  # Even numbers of individuals and periods; unbalanced, so individuals have odd and even counts.
  d <- sim_classic(30, 8, 6, sd_a = 1, drop = 0.1)
  fml <- y ~ x1 + d | i
  m <- fe_logit(fml, d)
  mp <- fe_logit(fml, d, panel.id = ~ i + t)
  cache <- new.env()
  for (v in variants) {
    jk <- biasCorr(m, method = "jackknife", jackknife = v, panel.id = ~ i + t)
    ref <- jk_reference(fml, d, "i", "t", v, two_way = FALSE, cache = cache)
    check_jk(jk, ref, sprintf("jackknife one-way %s", v))
    expect_equal(coef(biasCorr(mp, method = "jackknife", jackknife = v)), coef(jk), tolerance = 1e-14)
    expect_equal(coef(biasCorr(m, method = "jackknife", jackknife = v, bias = "individual", panel.id = "i,t")), coef(jk), tolerance = 1e-14)
  }
  expect_output(print(biasCorr(mp, method = "jackknife")), "Bias correction: jackknife (ss2), individual effects", fixed = TRUE)
})

test_that("double jackknife matches fixest refits (dyadic data, shared identifiers)", {
  skip_if_not(fixest_ok())
  d <- sim_dyadic(16, 9)
  fml <- y ~ x1 + d | id + jd
  m <- fe_logit(fml, d)
  cache <- new.env()
  for (b in c("both", "individual", "time")) {
    jk <- biasCorr(m, method = "jackknife", jackknife = "double", bias = b)
    ref <- jk_reference(fml, d, "id", "jd", "double", b, double = c("id", "jd"), cache = cache)
    check_jk(jk, ref, sprintf("jackknife double bias=%s", b))
  }
  # Numeric identifiers are matched by value.
  d$idn <- as.integer(sub("c", "", d$id))
  d$jdn <- as.numeric(sub("c", "", d$jd))
  mn <- fe_logit(y ~ x1 + d | idn + jdn, d)
  expect_equal(coef(biasCorr(mn, method = "jackknife", jackknife = "double")), coef(biasCorr(m, method = "jackknife", jackknife = "double")),
    tolerance = 1e-12)
})

test_that("multiple(R) random partitions match the refits and are reproducible with set.seed()", {
  skip_if_not(fixest_ok())
  d <- sim_classic(41, 9, 5, sd_a = 1.2)
  fml <- y ~ x1 + d | i + t
  m <- fe_logit(fml, d)
  N <- length(unique(d$i))
  T <- length(unique(d$t))
  cache <- new.env()
  draw <- function(R, split, N, T) {
    lapply(seq_len(R), function(r) list(
      individual = if (split != "time") sample.int(N),
      period = if (split != "individuals") sample.int(T)
    ))
  }
  for (cfg in list(
    list(v = "ss1", b = "both", s = "both"), list(v = "ss2", b = "both", s = "both"),
    list(v = "ss2", b = "both", s = "individuals"), list(v = "ss1", b = "both", s = "time"),
    list(v = "ss2", b = "individual", s = "time"), list(v = "ss1", b = "time", s = "individuals")
  )) {
    set.seed(123)
    jk <- biasCorr(m, method = "jackknife", jackknife = cfg$v, bias = cfg$b, multiple = 3, split = cfg$s)
    expect_identical(jk$multiple, 3L)
    expect_identical(jk$split, cfg$s)
    set.seed(123)
    ref <- jk_reference(fml, d, "i", "t", cfg$v, cfg$b, perms = draw(3, cfg$s, N, T), cache = cache)
    check_jk(jk, ref, sprintf("jackknife multiple %s bias=%s split=%s", cfg$v, cfg$b, cfg$s))
  }
  set.seed(1)
  a <- coef(biasCorr(m, method = "jackknife", multiple = 4))
  set.seed(1)
  expect_identical(coef(biasCorr(m, method = "jackknife", multiple = 4)), a)
  set.seed(2)
  expect_false(isTRUE(all.equal(coef(biasCorr(m, method = "jackknife", multiple = 4)), a)))
  expect_output(print(biasCorr(m, method = "jackknife", jackknife = "ss1", multiple = 2)),
    "Bias correction: jackknife (ss1, 2 random partitions), individual and time effects", fixed = TRUE)

  # One-way: the periods come from panel.id.
  d1 <- sim_classic(30, 8, 6, sd_a = 1, drop = 0.1)
  m1 <- fe_logit(y ~ x1 + d | i, d1)
  set.seed(7)
  jk <- biasCorr(m1, method = "jackknife", multiple = 2, split = "time", panel.id = ~ i + t)
  set.seed(7)
  ref <- jk_reference(y ~ x1 + d | i, d1, "i", "t", "ss2", two_way = FALSE, perms = draw(2, "time", 30, 8))
  check_jk(jk, ref, "jackknife multiple one-way ss2")
})

test_that("a subpanel that loses a regressor is an error naming the subpanel", {
  d <- sim_classic(20, 6, 12)
  d$z <- ifelse(d$i <= 10, d$x1 + rnorm(nrow(d)), 0)
  m <- fe_logit(y ~ x1 + z | i + t, d)
  expect_error(biasCorr(m, method = "jackknife", jackknife = "ss1", bias = "time"), "jackknife, subpanel", fixed = TRUE)
})
