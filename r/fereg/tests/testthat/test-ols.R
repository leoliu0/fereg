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

test_that("combined fixed effects are ordered as fixest (labels as character strings)", {
  for (fml in list(y ~ x1 | id^year + firm, y ~ x1 | firm + year^id, y ~ x1 | firm^h + id)) {
    fo <- fixef(fe_feols(fml, panel))
    ft <- fixest::fixef(fx_feols(fml, panel))
    expect_identical(names(fo), names(ft))
    for (k in names(ft)) {
      expect_identical(names(fo[[k]]), names(ft[[k]]))
      expect_equal(unname(fo[[k]]), unname(ft[[k]]), tolerance = 1e-8)
    }
  }
})

test_that("summary of fixef() prints as fixest's", {
  for (fml in list(y ~ x1 + x2 | id + year + firm, y ~ x1 | year, y ~ x1 | id^year + firm)) {
    fo <- fixef(fe_feols(fml, panel))
    ft <- fixest::fixef(fx_feols(fml, panel))
    expect_identical(capture.output(s <- summary(fo)), capture.output(s <- summary(ft)))
  }
  capture.output(out <- withVisible(summary(fixef(fe_feols(y ~ x1 | id + year, panel)))))
  expect_null(out$value)
  expect_false(out$visible)
})

# Text of the messages of an expression, and its value.
with_messages <- function(expr) {
  msgs <- character(0)
  value <- withCallingHandlers(expr, message = function(e) {
    msgs <<- c(msgs, conditionMessage(e))
    invokeRestart("muffleMessage")
  })
  list(value = value, messages = msgs)
}

test_that("fixef() notes irregular fixed effects as fixest; attributes, print and summary match", {
  set.seed(1)
  n <- 60
  d <- data.frame(a = sample(8, n, TRUE), b = sample(8, n, TRUE), c = sample(6, n, TRUE), x = rnorm(n), z = rnorm(n))
  # a2 and b2: two connected components.
  d$a2 <- c(sample(4, 30, TRUE), sample(5:8, 30, TRUE))
  d$b2 <- c(sample(4, 30, TRUE), sample(5:8, 30, TRUE))
  d$y <- d$x + rnorm(n)
  d$cnt <- rpois(n, exp(0.2 * d$x))
  # Regular: one fixed effect, two or three connected ones. Not regular: a2 and b2 with others.
  # Varying slopes: fixest orders the terms by number of coefficients (y ~ x1 | year + firm +
  # id[x2]: id, id[[x2]], firm, year), counts the references in that order and tests them in
  # its own way (a2[z] + b2 has 2 references, no note).
  plain_fe <- list(y ~ x | a, y ~ x | a + b, y ~ x | a2 + b2, y ~ x | a + b + c, y ~ x | a2 + b2 + c, y ~ x | a2^c + b2)
  slopes <- list(y ~ x | a[z] + b, y ~ x | a2 + b2[z], y ~ x | a2[z] + b2, y ~ x | a2[[z]] + b2 + c)
  cases <- c(
    lapply(c(plain_fe, slopes), function(f) list(fml = f, data = d, pois = FALSE)),
    lapply(list(y ~ x1 | year + firm + id[x2], y ~ x1 | year[x2] + firm + id), function(f) list(fml = f, data = panel, pois = FALSE)),
    list(list(fml = cnt ~ x | a2 + b2, data = d, pois = TRUE))
  )
  note <- "NOTE: The fixed-effects are not regular, they cannot be straightforwardly interpreted.\n"
  n_notes <- 0
  for (cs in cases) {
    lab <- deparse(cs$fml)
    if (cs$pois) {
      ours <- with_messages(fixef(fe_fepois(cs$fml, cs$data)))
      theirs <- with_messages(fixest::fixef(fx_fepois(cs$fml, cs$data)))
    } else {
      ours <- with_messages(fixef(fe_feols(cs$fml, cs$data)))
      theirs <- with_messages(fixest::fixef(fx_feols(cs$fml, cs$data)))
    }
    expect_identical(ours$messages, theirs$messages, label = lab)
    n_notes <- n_notes + identical(theirs$messages, note)
    fo <- ours$value
    ft <- theirs$value
    expect_identical(names(fo), names(ft), label = lab)
    expect_identical(attr(fo, "references"), attr(ft, "references"), label = lab)
    expect_identical(attr(fo, "exponential"), attr(ft, "exponential"), label = lab)
    expect_identical(setdiff(names(attributes(fo)), "class"), setdiff(names(attributes(ft)), "class"), label = lab)
    # fixest recovers the fixed effects of models with varying slopes iteratively, in its own
    # normalization; the other values agree, printed as fixest's (no print method: only the class
    # differs).
    if (!any(grepl("[", names(ft), fixed = TRUE))) {
      for (k in names(ft)) expect_equal(fo[[k]], ft[[k]], tolerance = 1e-8)
      printed <- fo
      class(printed) <- class(ft)
      expect_identical(capture.output(print(printed)), capture.output(print(ft)), label = lab)
      expect_identical(capture.output(s <- summary(fo)), capture.output(s <- summary(ft)))
    }
  }
  expect_identical(n_notes, 4)

  # The note is a message, printed at every call; notes = FALSE or option fereg.notes = FALSE
  # (fixest: notes = FALSE or setFixest_notes(FALSE)) switch it off. fixest's fixef() (nlme's
  # generic) also dispatches to fereg's method.
  m <- fe_feols(y ~ x | a2 + b2, d)
  expect_message(fixef(m), note, fixed = TRUE)
  expect_message(fixef(m), note, fixed = TRUE)
  expect_message(fixest::fixef(m), note, fixed = TRUE)
  expect_silent(fixef(m, notes = FALSE))
  old <- options(fereg.notes = FALSE)
  expect_silent(fixef(m))
  options(old)
  expect_silent(predict(m, d))
  expect_identical(fixest::fixef(m, notes = FALSE), fixef(m, notes = FALSE))
})

test_that("fixef() with only approximate references: fixest's note, values reproducing the fit", {
  set.seed(1)
  d <- data.frame(a = sample(10, 40, TRUE), b = sample(10, 40, TRUE), c = sample(10, 40, TRUE), x = rnorm(40))
  d$y <- d$x + rnorm(40)
  d$h <- rep(1:2, 20)
  # Fixed-effect sum of every observation from the values of fixef() (a^h: labels "a_h").
  fe_sum <- function(f) {
    unname(Reduce(`+`, lapply(names(f), function(k) {
      f[[k]][do.call(paste, c(unname(d[strsplit(k, "^", fixed = TRUE)[[1]]]), sep = "_"))]
    })))
  }
  for (fml in list(y ~ x | a + b + c, y ~ x | a^h + b + c, y ~ x | b + c + a^h)) {
    ours <- fe_feols(fml, d, fixef.rm = "none")
    theirs <- fx_feols(fml, d, fixef.rm = "none")
    fo <- with_messages(fixef(ours))
    ft <- with_messages(fixest::fixef(theirs))
    expect_identical(fo$messages, ft$messages)
    expect_match(fo$messages, "The number of references is only approximate.", fixed = TRUE)
    expect_identical(attr(fo$value, "references"), attr(ft$value, "references"))
    # fixest solves the values again (by demeaning); both reproduce the fixed-effect sum.
    s <- unname(fitted(theirs) - coef(theirs)[["x"]] * d$x)
    expect_equal(fe_sum(fo$value), s, tolerance = 1e-8)
    expect_equal(fe_sum(ft$value), s, tolerance = 1e-4)
    expect_equal(predict(ours, d), unname(fitted(theirs)), tolerance = 1e-8)
  }
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

test_that("etable keep/drop select coefficient rows as fixest", {
  coef_rows <- function(labels) {
    from <- which(labels == "Dependent Var.:") + 2
    to <- which(labels == "Fixed-Effects:" | grepl("^_+$", labels))[1] - 1
    if (to < from) character(0) else labels[from:to]
  }
  fmls <- list(y ~ x1 + x2 + x3 | id, y ~ x1 + x2)
  ours <- stats::setNames(lapply(fmls, fe_feols, data = panel), c("m1", "m2"))
  theirs <- stats::setNames(lapply(fmls, fx_feols, data = panel), c("m1", "m2"))
  full <- do.call(etable, ours)
  for (sel in list(
    list(keep = c("x1", "x3")), list(keep = c("x1", "Constant")), list(keep = "%Intercept", drop = "x3"),
    list(keep = "!x2"), list(drop = "^x[12]$"), list(drop = "!x"), list(keep = "x", drop = "%!x3")
  )) {
    o <- do.call(etable, c(ours, sel))
    t <- do.call(fixest::etable, c(theirs, sel))
    expect_identical(coef_rows(o[[1]]), coef_rows(t[[1]]))
    expect_identical(o[match(coef_rows(o[[1]]), o[[1]]), ], full[match(coef_rows(o[[1]]), full[[1]]), ], ignore_attr = TRUE)
  }
  expect_error(etable(ours[[1]], keep = "nothing"), "no coefficient was selected")
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
  iv <- fe_feols(y ~ x1 | id | e1 ~ z1 + z2, panel)
  expect_error(fixef(summary(iv, stage = 1)), "not available for first-stage models")
  expect_named(fixef(iv), "id")
})
