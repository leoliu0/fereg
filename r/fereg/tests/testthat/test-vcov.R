# HC2, HC3, Newey-West, Driscoll-Kraay, Conley and CRV3 variances against fixest 0.14.2 (CRV3,
# a fereg extension, against a leave-one-cluster-out loop of fixest refits).
skip_if_not(fixest_ok())

# Unbalanced panel with calendar gaps: 60 units observed in 5 to 14 of 16 years, AR(1) errors and
# regressors within the unit, coordinates per unit (a few moved rows), a unique time variable
# `ts` (time series), missing values in `tna` and `latna`.
make_se_panel <- function(seed = 20261013) {
  set.seed(seed)
  rows <- do.call(rbind, lapply(1:60, function(i) data.frame(id = i, t = sort(sample(2001:2016, sample(5:14, 1))))))
  d <- rows
  n <- nrow(d)
  ar <- function(v, rho) {
    for (j in seq_along(v)[-1]) if (d$id[j] == d$id[j - 1]) v[j] <- rho * v[j - 1] + v[j]
    v
  }
  ue <- rnorm(60)
  gm <- rnorm(30)
  d$ts <- sample(seq_len(2 * n), n)
  d$g <- sample(1:30, n, TRUE)
  d$h <- sample(1:8, n, TRUE)
  d$c3 <- sample(1:15, n, TRUE)
  d$x1 <- ar(rnorm(n), 0.6) + 0.3 * ue[d$id]
  d$x2 <- rnorm(n) + 0.3 * d$x1
  d$s1 <- runif(n, 0, 3)
  e <- ar(rnorm(n), 0.5)
  d$y <- 1 + 0.5 * d$x1 - 0.8 * d$x2 + gm[d$g] + rnorm(8)[d$h] + ue[d$id] + rnorm(30)[d$g] * d$s1 + e
  d$w <- runif(n, 0.5, 2)
  d$z1 <- rnorm(n)
  d$z2 <- rnorm(n)
  u <- rnorm(n)
  d$xe <- 0.6 * d$z1 + 0.4 * d$z2 + 0.5 * u + gm[d$g] + rnorm(n)
  d$yiv <- 1 + d$xe + 0.5 * d$x1 + gm[d$g] + u + e
  d$yp <- rpois(n, exp(0.2 + 0.3 * d$x1 - 0.2 * d$x2 + 0.3 * gm[d$g] + 0.2 * e))
  d$yb <- rbinom(n, 1, stats::plogis(0.5 * d$x1 - 0.5 * d$x2 + 0.5 * gm[d$g] + 0.3 * e))
  lat_u <- runif(60, 32, 44)
  lon_u <- runif(60, -100, -80)
  d$lat <- lat_u[d$id]
  d$lon <- lon_u[d$id]
  mv <- sample(n, 40)
  d$lat[mv] <- d$lat[mv] + rnorm(40, 0, 0.5)
  d$lon[mv] <- d$lon[mv] + rnorm(40, 0, 0.5)
  d$tna <- d$t
  d$tna[c(4, 50, 51, 300)] <- NA
  d$latna <- d$lat
  d$latna[c(9, 120, 121)] <- NA
  d$tc <- sprintf("y%d", d$t)
  d
}

sep <- make_se_panel()
# fixest's message text without its call prefix and line wrapping.
norm_msg <- function(m) gsub("\\s+", " ", trimws(m))

expect_same_message <- function(ours, theirs) {
  testthat::expect_true(is.character(ours) && is.character(theirs))
  testthat::expect_true(grepl(norm_msg(ours), norm_msg(theirs), fixed = TRUE),
    label = paste0("fereg: ", norm_msg(ours), "\nfixest: ", norm_msg(theirs)))
}


# fixest's eigenvalue fix warns in both packages; the matrices are compared.
quiet_psd <- function(expr) {
  withCallingHandlers(expr, warning = function(w) if (grepl("not positive definite", conditionMessage(w))) invokeRestart("muffleWarning"))
}

# Fits with each package. `model`: "ols", "poisson" or "logit".
fit_pair <- function(model, ...) {
  switch(model,
    ols = list(ours = quiet_psd(fe_feols(...)), theirs = quiet_psd(fx_feols(...))),
    poisson = list(ours = quiet_psd(fe_fepois(...)), theirs = quiet_psd(fx_fepois(...))),
    logit = list(ours = quiet_psd(fe_feglm(..., family = "logit")), theirs = quiet_psd(fx_feglm(..., family = "logit")))
  )
}

# Variance, df.K, df.t (t-distributed models) and label of a fit / summary against fixest.
expect_same_vcov <- function(ours, theirs, tol, label) {
  va <- stats::vcov(theirs, attr = TRUE)
  testthat::expect_equal(unname(coef(ours)), unname(coef(theirs)), tolerance = tol, label = label)
  testthat::expect_equal(plain(vcov(ours)), plain(va), tolerance = tol, label = label)
  testthat::expect_identical(ours$vcov_info$label, attr(va, "vcov_type"), label = label)
  testthat::expect_identical(attr(ours$coeftable, "vcov_type"), attr(fixest::coeftable(theirs), "vcov_type"), label = label)
  testthat::expect_equal(ours$vcov_info$df_k, unname(attr(va, "df.K")), label = label)
  if (identical(theirs$method_type, "feols")) testthat::expect_equal(ours$vcov_info$df_t, unname(attr(va, "df.t")), label = label)
  testthat::expect_equal(unname(se(ours)), unname(fixest::se(theirs)), tolerance = tol, label = label)
}

# The variance requests of every type (the same expression for both packages: fereg's request
# functions mirror fixest's). `panel` = the model has panel.id = ~id + t.
se_requests <- function(panel) {
  out <- list(
    hc2 = "hc2", HC3 = "HC3",
    nw = NW ~ id + t, nw2 = NW(2) ~ id + t, nw_obj = NW(3), dk = DK ~ t, dk2 = DK(2) ~ t, dk_obj = DK(),
    conley = conley(250) ~ lat + lon, conley_sph = conley(250, distance = "spherical") ~ lat + lon,
    conley_pixel = conley(250, pixel = 60) ~ lat + lon, conley_deduced = "conley", conley_mi = conley("150mi"),
    conley_mi_pixel = conley("150 mi", pixel = 30)
  )
  if (!panel) out[c("nw_obj", "dk_obj")] <- NULL
  out
}

# fereg's and fixest's request functions build identical requests; fixest's are used for fixest.
as_fixest_request <- function(v) {
  if (!inherits(v, "fixest_vcov_request")) return(v)
  r <- v$extra_args
  switch(v$vcov,
    NW = fixest::NW(r$lag),
    DK = fixest::DK(r$lag),
    conley = fixest::conley(r$cutoff, r$pixel, r$distance)
  )
}

with_fixest_env <- function(v) {
  if (inherits(v, "formula")) environment(v) <- asNamespace("fixest")
  v
}

check_types <- function(model, fml, tol, data = sep, weights = NULL, panel = TRUE, skip = character(0)) {
  args <- list(fml = fml, data = data)
  if (!is.null(weights)) args$weights <- weights
  if (panel) args$panel.id <- ~ id + t
  base <- do.call(fit_pair, c(list(model), args))
  reqs <- se_requests(panel)
  reqs <- reqs[setdiff(names(reqs), skip)]
  for (nm in names(reqs)) {
    v <- reqs[[nm]]
    lab <- paste(deparse1(fml), nm)
    # Estimation time.
    ours <- quiet_psd(do.call(switch(model, ols = fe_feols, poisson = fe_fepois, logit = fe_feglm),
      c(args, list(vcov = v), if (model == "logit") list(family = "logit"))))
    theirs <- quiet_psd(do.call(switch(model, ols = fx_feols, poisson = fx_fepois, logit = fx_feglm),
      c(args, list(vcov = with_fixest_env(as_fixest_request(v))), if (model == "logit") list(family = "logit"))))
    expect_same_vcov(ours, theirs, tol, paste(lab, "(estimation)"))
    # After estimation.
    s_ours <- quiet_psd(summary(base$ours, vcov = v))
    s_theirs <- quiet_psd(summary(base$theirs, vcov = with_fixest_env(as_fixest_request(v))))
    expect_same_vcov(s_ours, s_theirs, tol, paste(lab, "(summary)"))
  }
  invisible(base)
}

test_that("a two-sided NW formula is Newey-West, not clustering on its variables", {
  v <- fereg:::parse_vcov(NW ~ id + t)
  expect_identical(v$type, "nw")
  expect_identical(v$vars, list(unit = quote(id), time = quote(t)))
  ours <- fe_feols(y ~ x1 + x2 | id, sep, vcov = NW ~ id + t)
  clustered <- fe_feols(y ~ x1 + x2 | id, sep, vcov = ~ id + t)
  expect_identical(ours$vcov_info$label, attr(stats::vcov(fx_feols(y ~ x1 + x2 | id, sep, vcov = NW ~ id + t), attr = TRUE), "vcov_type"))
  expect_false(isTRUE(all.equal(plain(vcov(ours)), plain(vcov(clustered)))))
  expect_equal(plain(vcov(ours)), plain(stats::vcov(fx_feols(y ~ x1 + x2 | id, sep, vcov = NW ~ id + t))), tolerance = 1e-8)
  for (f in list(DK(2) ~ t, conley(100) ~ lat + lon, crv3 ~ c3)) {
    expect_false(fereg:::parse_vcov(f)$type == "cluster", label = deparse1(f))
  }
})

test_that("OLS variances of every type match fixest (1 and 2 fixed effects, weights, slopes)", {
  check_types("ols", y ~ x1 + x2 | id, 1e-8)
  check_types("ols", y ~ x1 + x2 | id + g, 1e-8, weights = ~w)
  check_types("ols", y ~ x1 + x2, 1e-8, panel = FALSE)
  # fixest's hatvalues is not exact with varying slopes: HC2 / HC3 are left out there.
  check_types("ols", y ~ x1 + x2 | id + g[s1], 1e-8, skip = c("hc2", "HC3"))
})

test_that("IV variances of every type match fixest; HC2 / HC3 are errors as in fixest", {
  check_types("ols", yiv ~ x1 | id | xe ~ z1 + z2, 1e-8, skip = c("hc2", "HC3"))
  check_types("ols", yiv ~ x1 | id + g | xe ~ z1 + z2, 1e-8, weights = ~w, skip = c("hc2", "HC3"))
  for (v in c("hc2", "hc3")) {
    ours <- tryCatch(fe_feols(yiv ~ x1 | id | xe ~ z1 + z2, sep, vcov = v), error = conditionMessage)
    theirs <- tryCatch(fx_feols(yiv ~ x1 | id | xe ~ z1 + z2, sep, vcov = v), error = conditionMessage)
    expect_same_message(ours, theirs)
    m <- fe_feols(yiv ~ x1 | id | xe ~ z1 + z2, sep)
    expect_same_message(tryCatch(summary(m, vcov = v), error = conditionMessage), theirs)
  }
})

test_that("Poisson variances of every type match fixest", {
  check_types("poisson", yp ~ x1 + x2 | id, 1e-8)
  check_types("poisson", yp ~ x1 + x2 | id + g, 1e-8, weights = ~w)
})

test_that("logit variances of every type match fixest", {
  check_types("logit", yb ~ x1 + x2 | g, 1e-8)
  check_types("logit", yb ~ x1 + x2 | g + h, 1e-8, weights = ~w)
})

# fereg's request objects have fixest's structure: the same object is given to both packages.
test_that("time-series Newey-West matches fixest (default and explicit lag)", {
  for (v in list(NW ~ ts, NW(3) ~ ts, vcov_NW(time = "ts", lag = 2))) {
    if (is.null(fereg:::parse_vcov(v)$lag)) skip_if_not_installed("sandwich") # fixest's default lag calls sandwich
    lab <- if (inherits(v, "formula")) deparse1(v) else "vcov_NW(time = \"ts\", lag = 2)"
    expect_same_vcov(fe_feols(y ~ x1 + x2, sep, vcov = v), fx_feols(y ~ x1 + x2, sep, vcov = with_fixest_env(v)), 1e-8, lab)
    m <- fe_feols(y ~ x1 + x2 | g, sep)
    t <- fx_feols(y ~ x1 + x2 | g, sep)
    expect_same_vcov(summary(m, vcov = v), summary(t, vcov = with_fixest_env(v)), 1e-8, lab)
  }
})

test_that("panel.id gives the default unit, time and cluster; strings, factors and vcov_*() requests", {
  ours <- fe_feols(y ~ x1 + x2 | g, sep, panel.id = "id,t")
  theirs <- fx_feols(y ~ x1 + x2 | g, sep, panel.id = "id,t")
  expect_identical(ours$panel.id, c("id", "t"))
  reqs <- list("NW", "newey_west", "nw", "DK", "driscoll_kraay", "cluster", NW(2), DK(1),
    vcov_NW(lag = 2), vcov_NW("id", "t"), vcov_NW(unit = ~id, time = ~t), vcov_DK(time = "t", lag = 2),
    vcov_conley(lat = "lat", lon = "lon", cutoff = 300), vcov_conley(cutoff = "100mi", pixel = 5),
    vcov_hetero(type = "hc2"), vcov_hetero(), vcov_cluster("id"), vcov_cluster(~ id + g), NW ~ t)
  for (k in seq_along(reqs)) {
    v <- reqs[[k]]
    expect_same_vcov(quiet_psd(summary(ours, vcov = v)), quiet_psd(summary(theirs, vcov = with_fixest_env(v))), 1e-8, paste("request", k))
  }
  # Character and factor time variables: fixest's time order (sorted strings, factor levels).
  sep2 <- sep
  sep2$tf <- factor(sep2$t, levels = sample(unique(sep2$t)))
  for (v in list(NW ~ id + tc, DK ~ tc, NW ~ id + tf, DK ~ tf)) {
    expect_same_vcov(fe_feols(y ~ x1 | g, sep2, vcov = v), fx_feols(y ~ x1 | g, sep2, vcov = v), 1e-8, deparse1(v))
  }
  # Called with an estimation, the vcov_*() functions return its variance matrix.
  expect_equal(plain(vcov_NW(ours, lag = 2)), plain(fixest::vcov_NW(theirs, lag = 2)), tolerance = 1e-8)
  expect_equal(plain(vcov_hetero(ours, "hc3")), plain(fixest::vcov_hetero(theirs, "hc3")), tolerance = 1e-8)
  expect_equal(fitstat(summary(ours, vcov = "NW"), "g")[[1]], unname(fixest::fitstat(summary(theirs, vcov = "NW"), "g")[[1]]))
})

test_that("vcov(), confint() and ssc() after estimation match fixest for every type", {
  ours <- fe_feols(y ~ x1 + x2 | id + g, sep, panel.id = ~ id + t)
  theirs <- fx_feols(y ~ x1 + x2 | id + g, sep, panel.id = ~ id + t)
  for (v in list("hc2", "NW", DK(2) ~ t, conley(200) ~ lat + lon)) {
    for (s in list(NULL, ssc(K.adj = FALSE), ssc(G.adj = FALSE), ssc(K.fixef = "none"), ssc(t.df = "conventional"))) {
      fs <- if (is.null(s)) NULL else do.call(fixest::ssc, unclass(s))
      va <- quiet_psd(stats::vcov(theirs, vcov = v, ssc = fs, attr = TRUE))
      vo <- quiet_psd(vcov(ours, vcov = v, ssc = s, attr = TRUE))
      expect_equal(plain(vo), plain(va), tolerance = 1e-8)
      expect_identical(attr(vo, "type"), attr(va, "vcov_type"))
      expect_equal(attr(vo, "df.t"), attr(va, "df.t"))
      expect_equal(as.matrix(quiet_psd(confint(ours, vcov = v, ssc = s))), as.matrix(quiet_psd(confint(theirs, vcov = v, ssc = fs))), tolerance = 1e-8)
    }
  }
  # ssc() inside the formula.
  expect_equal(plain(vcov(ours, vcov = NW ~ id + t + ssc(K.adj = FALSE))),
    plain(stats::vcov(theirs, vcov = NW ~ id + t + ssc(K.adj = FALSE))), tolerance = 1e-8)
})

test_that("missing values in variance variables: dropped at estimation, errors after estimation (fixest)", {
  for (v in list(NW ~ id + tna, DK ~ tna, conley(200) ~ latna + lon)) {
    ours <- fe_feols(y ~ x1 + x2 | id, sep, vcov = v)
    theirs <- fx_feols(y ~ x1 + x2 | id, sep, vcov = v)
    expect_same_vcov(ours, theirs, 1e-8, deparse1(v))
    expect_identical(ours$obs_selection, as.integer(fixest::obs(theirs)))
    msg_ours <- capture_messages(feols(y ~ x1 + x2 | id, sep, vcov = v, notes = TRUE))
    msg_theirs <- capture_messages(fixest::feols(y ~ x1 + x2 | id, sep, vcov = v, notes = TRUE, nthreads = 1))
    expect_identical(trimws(msg_ours), trimws(msg_theirs))
    m <- fe_feols(y ~ x1 + x2 | id, sep)
    t <- fx_feols(y ~ x1 + x2 | id, sep)
    expect_same_message(tryCatch(summary(m, vcov = v), error = conditionMessage), tryCatch(summary(t, vcov = v), error = conditionMessage))
  }
})

test_that("labels and the etable S.E. row match fixest", {
  ours <- fe_feols(y ~ x1 + x2 | id + g, sep, panel.id = ~ id + t)
  theirs <- fx_feols(y ~ x1 + x2 | id + g, sep, panel.id = ~ id + t)
  # One model per table (fixest wraps wide tables to the console width), and mixed pairs.
  vs <- list("hc2", "HC3", NW(2) ~ id + t, "NW", "DK", DK(3) ~ t, conley(100) ~ lat + lon, conley("60mi"), "conley",
    "hetero", ~id, ~ id + g, "twoway", list("hc2", ~id), list(~id, ~ id + t), list("NW", conley(100) ~ lat + lon))
  for (k in seq_along(vs)) {
    v <- vs[[k]]
    if (is.list(v) && !inherits(v, "fixest_vcov_request")) {
      o <- capture.output(print(quiet_psd(etable(ours, ours, vcov = v))))
      t <- capture.output(print(quiet_psd(fixest::etable(theirs, theirs, vcov = lapply(v, with_fixest_env)))))
    } else {
      o <- capture.output(print(quiet_psd(etable(ours, vcov = v))))
      t <- capture.output(print(quiet_psd(fixest::etable(theirs, vcov = with_fixest_env(v)))))
    }
    expect_identical(o[-1], t[-1], label = paste("etable", k)) # the header line holds the model names
  }
  for (v in list("NW", DK(2) ~ t, "hc2", conley("150mi"))) {
    so <- capture.output(print(quiet_psd(summary(ours, vcov = v))))
    st <- capture.output(print(quiet_psd(summary(theirs, vcov = v))))
    expect_identical(grep("Standard-errors", so, value = TRUE), grep("Standard-errors", st, value = TRUE))
  }
  for (x in c("Driscoll-Kraay (L=1)", "Heteroskedasticity-robust", "Newey-West (L=12)", "Conley (62.1371mi)", "Clustered (id & g)",
              "Clustered (a_long_name & another_one)", "Clustered (year^firm)", "CRV3 (c3)")) {
    for (width in c(8, 10, 12, 16, 17, 20, 30)) {
      expect_identical(fereg:::format_se_type(x, width), fixest:::format_se_type(x, width, by = TRUE), label = paste(x, width))
    }
  }
})

test_that("invalid requests give fixest's errors", {
  m <- fe_feols(y ~ x1 + x2 | id, sep)
  t <- fx_feols(y ~ x1 + x2 | id, sep)
  mp <- fe_feols(y ~ x1 + x2 | id, sep[, c("id", "t", "x1", "x2", "y")])
  tp <- fx_feols(y ~ x1 + x2 | id, sep[, c("id", "t", "x1", "x2", "y")])
  err <- function(expr) tryCatch({
    expr
    "no error"
  }, error = conditionMessage)
  cases <- list(
    quote(summary(M, vcov = NW ~ a + b + c)),
    quote(summary(M, vcov = DK ~ id + t)),
    quote(summary(M, vcov = conley ~ lat)),
    quote(summary(M, vcov = bogus ~ id)),
    quote(summary(M, vcov = "NW")),
    quote(summary(M, vcov = "DK")),
    quote(summary(M, vcov = NW ~ id + zz)),
    quote(summary(M, vcov = NW ~ t)),
    quote(summary(M, vcov = conley("abc"))),
    quote(summary(M, vcov = conley(-1))),
    quote(summary(M, vcov = conley(100, pixel = -1))),
    quote(summary(M, vcov = conley(100, distance = "foo"))),
    quote(summary(MP, vcov = "conley")),
    quote(vcov_NW(lag = 0)),
    quote(summary(M, vcov = ~ id + t + g + h + c3))
  )
  for (cl in cases) {
    ours <- err(eval(do.call(substitute, list(cl, list(M = quote(m), MP = quote(mp))))))
    fx <- do.call(substitute, list(cl, list(M = quote(t), MP = quote(tp))))
    theirs <- err(eval(fx, list(vcov_NW = fixest::vcov_NW, conley = fixest::conley)))
    expect_same_message(ours, theirs)
  }
  ours <- err(summary(m, vcov = "bogus"))
  expect_true(grepl("no match was found for 'bogus'", ours, fixed = TRUE))
  expect_true(grepl("no match was found for 'bogus'", err(summary(t, vcov = "bogus")), fixed = TRUE))
  # Several candidate latitude columns.
  d3 <- sep
  d3$lat_a <- d3$lat
  d3$lat_b <- d3$lat
  d3$lat <- NULL
  expect_same_message(err(summary(fe_feols(y ~ x1 | id, d3), vcov = "conley")), err(summary(fx_feols(y ~ x1 | id, d3), vcov = "conley")))
  # panel.id variables must be in the data.
  for (p in list(~ id + zz, ~ aa + zz)) {
    expect_same_message(err(fe_feols(y ~ x1 | id, sep, panel.id = p)), err(fx_feols(y ~ x1 | id, sep, panel.id = p)))
  }
  # Unique abbreviations and case are accepted as in fixest; fereg adds "crv3" to the names.
  mpan <- fe_feols(y ~ x1 + x2 | id, sep, panel.id = ~ id + t)
  tpan <- fx_feols(y ~ x1 + x2 | id, sep, panel.id = ~ id + t)
  for (v in c("het", "white", "Hc3", "dr", "ne", "co", "s")) {
    expect_identical(summary(mpan, vcov = v)$vcov_info$label, attr(stats::vcov(tpan, vcov = v, attr = TRUE), "vcov_type"), label = v)
  }
  expect_error(summary(m, vcov = "c"), "more than one value was matched for 'c': 'cluster', 'conley' and 'crv3'", fixed = TRUE)
  expect_error(summary(m, vcov = "vcov_fix"), "no match was found for 'vcov_fix'", fixed = TRUE)
})

# fereg's CRV3 (fixest flavor): leave-one-cluster-out refits over every non-empty subset of the
# cluster dimensions (sign (-1)^(|S|+1)), centered at the full-sample coefficients, times fixest's
# cluster correction G_min / (G_min - 1) * (n - 1) / (n - K) with K of fixest's clustered vcov,
# and fixest's positive-definiteness fix.
crv3_reference <- function(theirs, refit, data, cluster) {
  sub <- data[fixest::obs(theirs), , drop = FALSE]
  beta <- coef(theirs)
  ids <- lapply(cluster, function(v) as.character(sub[[v]]))
  meat <- matrix(0, length(beta), length(beta))
  q <- length(ids)
  for (size in seq_len(q)) {
    for (S in utils::combn(q, size, simplify = FALSE)) {
      id <- do.call(paste, c(ids[S], sep = "|"))
      for (gv in unique(id)) {
        b <- coef(refit(sub[id != gv, , drop = FALSE]))[names(beta)]
        stopifnot(!anyNA(b))
        meat <- meat + (if (size %% 2 == 1) 1 else -1) * tcrossprod(b - beta)
      }
    }
  }
  V1 <- stats::vcov(theirs, vcov = stats::reformulate(cluster), attr = TRUE)
  G <- vapply(ids, function(x) length(unique(x)), 1)
  K <- attr(V1, "df.K")
  n <- stats::nobs(theirs)
  V <- meat * min(G) / (min(G) - 1) * (n - 1) / (n - K)
  if (any(eigen(V, symmetric = TRUE, only.values = TRUE)$values <= 0)) V <- fixest:::mat_posdef_fix(V)
  list(V = V, K = K, df_t = max(min(G) - 1, 1))
}

test_that("CRV3 (fereg extension) matches a leave-one-cluster-out loop of fixest refits", {
  cases <- list(
    list(model = "ols", fml = y ~ x1 + x2 | id, cluster = "c3"),
    list(model = "ols", fml = y ~ x1 + x2 | id + g, cluster = "id", weights = ~w),
    list(model = "ols", fml = y ~ x1 + x2 | g, cluster = c("c3", "h")),
    list(model = "ols", fml = y ~ x1 + x2, cluster = "c3"),
    list(model = "ols", fml = yiv ~ x1 | id | xe ~ z1 + z2, cluster = "c3"),
    list(model = "poisson", fml = yp ~ x1 + x2 | id, cluster = "c3"),
    list(model = "logit", fml = yb ~ x1 + x2 | h, cluster = "c3")
  )
  for (cs in cases) {
    tol <- 1e-8
    lab <- paste(deparse1(cs$fml), paste(cs$cluster, collapse = "+"))
    w <- cs$weights
    fx <- function(data, vcov = "iid") {
      a <- list(fml = cs$fml, data = data, vcov = vcov)
      if (!is.null(w)) a$weights <- w
      switch(cs$model, ols = do.call(fx_feols, a), poisson = do.call(fx_fepois, a), logit = do.call(fx_feglm, c(a, list(family = "logit"))))
    }
    theirs <- fx(sep)
    ref <- crv3_reference(theirs, fx, sep, cs$cluster)
    vf <- stats::as.formula(paste("crv3 ~", paste(cs$cluster, collapse = " + ")))
    a <- list(fml = cs$fml, data = sep, vcov = vf)
    if (!is.null(w)) a$weights <- w
    ours <- quiet_psd(switch(cs$model, ols = do.call(fe_feols, a), poisson = do.call(fe_fepois, a), logit = do.call(fe_feglm, c(a, list(family = "logit")))))
    expect_equal(plain(vcov(ours)), unname(ref$V), tolerance = tol, label = lab)
    expect_identical(ours$vcov_info$label, paste0("CRV3 (", paste(cs$cluster, collapse = " & "), ")"), label = lab)
    expect_equal(ours$vcov_info$df_k, ref$K, label = lab)
    if (cs$model == "ols") expect_equal(ours$vcov_info$df_t, ref$df_t, label = lab)
    # After estimation (re-estimates the model), with a request object.
    a$vcov <- NULL
    base <- switch(cs$model, ols = do.call(fe_feols, a), poisson = do.call(fe_fepois, a), logit = do.call(fe_feglm, c(a, list(family = "logit"))))
    s <- quiet_psd(summary(base, vcov = vcov_crv3(cluster = cs$cluster)))
    expect_equal(plain(vcov(s)), plain(vcov(ours)), tolerance = 1e-10, label = lab)
    expect_identical(s$vcov_info$label, ours$vcov_info$label)
  }
  # "crv3" guesses the cluster as "cluster" does (panel.id, else the first fixed effect).
  m <- fe_feols(y ~ x1 + x2 | id + g, sep)
  expect_identical(summary(m, vcov = "crv3")$vcov_info$label, "CRV3 (id)")
  expect_identical(summary(fe_feols(y ~ x1 + x2 | g, sep, panel.id = ~ c3 + t), vcov = "CRV3")$vcov_info$label, "CRV3 (c3)")
  expect_output(print(summary(m, vcov = crv3 ~ c3)), "Standard-errors: CRV3 (c3)", fixed = TRUE)
})
