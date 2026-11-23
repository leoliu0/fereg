# Tests for wild bootstrap inference in fereg (R)

make_wb_data <- function(n = 240L, seed = 42L) {
  set.seed(seed)
  i <- seq_len(n) - 1L
  c1 <- i %% 12L
  c2 <- i %% 8L
  fe1 <- i %% 15L
  fe2 <- (i * 3L) %% 6L
  z1 <- rnorm(n)
  z2 <- rnorm(n)
  x1 <- rnorm(n) + 0.3 * sin(c1)
  x2 <- rnorm(n) + 0.25 * cos(fe1)
  u <- rnorm(n)
  endog <- 0.7 * z1 - 0.4 * z2 + 0.5 * x1 + 0.6 * u + rnorm(n, sd = 0.5)
  fe_eff <- 0.4 * sin(fe1) + 0.3 * cos(fe2) + 0.35 * sin(c1)
  y <- 0.8 + 1.2 * x1 - 0.7 * x2 + 0.9 * endog + fe_eff + u
  eta_p <- -0.3 + 0.5 * x1 - 0.3 * x2 + 0.2 * sin(fe1)
  y_pois <- rpois(n, exp(pmin(pmax(eta_p, -3), 3)))
  eta_l <- -0.2 + 0.8 * x1 - 0.5 * x2 + 0.25 * sin(fe2)
  y_bin <- as.numeric(runif(n) < stats::plogis(eta_l))
  w <- 0.5 + (i %% 4L) * 0.4
  data.frame(
    y = y, x1 = x1, x2 = x2, endog = endog, z1 = z1, z2 = z2,
    c1 = c1, c2 = c2, fe1 = fe1, fe2 = fe2,
    y_pois = y_pois, y_bin = y_bin, w = w
  )
}

explicit_dummy_wls_oracle <- function(df, R_mat, r_vec, impose_null = TRUE) {
  n <- nrow(df)
  D1 <- stats::model.matrix(~ 0 + factor(fe1), df)
  D2 <- stats::model.matrix(~ 0 + factor(fe2), df)[, -1L, drop = FALSE]
  D <- cbind(df$x1, df$x2, D1, D2)
  y <- df$y
  w <- df$w
  sw <- sqrt(w)
  Dw <- D * sw
  yw <- y * sw
  k <- 2L
  p_tot <- ncol(D)
  qr_fit <- qr(Dw)
  beta_full <- qr.coef(qr_fit, yw)
  beta_full[is.na(beta_full)] <- 0
  XtX_inv <- MASS::ginv(crossprod(Dw))
  u_unr <- as.numeric(y - D %*% beta_full)

  R_full <- matrix(0, nrow = nrow(R_mat), ncol = p_tot)
  R_full[, seq_len(k)] <- R_mat
  if (impose_null) {
    RBR <- R_full %*% XtX_inv %*% t(R_full)
    diff <- as.numeric(R_full %*% beta_full - r_vec)
    beta_dgp <- as.numeric(beta_full - XtX_inv %*% t(R_full) %*% solve(RBR, diff))
    u_dgp <- as.numeric(y - D %*% beta_dgp)
  } else {
    beta_dgp <- beta_full
    u_dgp <- u_unr
  }

  cl <- df$cl
  uniq_cl <- sort(unique(cl))
  G <- length(uniq_cl)
  fe_k <- length(unique(df$fe1)) + length(unique(df$fe2)) - 1L
  df_k <- k + fe_k
  ssc_fac <- ((n - 1) / (n - df_k)) * (G / (G - 1))

  scores_obs <- matrix(0, nrow = G, ncol = p_tot)
  for (gi in seq_len(G)) {
    msk <- cl == uniq_cl[gi]
    scores_obs[gi, ] <- colSums(Dw[msk, , drop = FALSE] * (sw[msk] * u_unr[msk]))
  }
  V_obs <- XtX_inv[seq_len(k), , drop = FALSE] %*% (ssc_fac * crossprod(scores_obs)) %*% XtX_inv[, seq_len(k), drop = FALSE]
  diff_obs <- as.numeric(R_mat %*% beta_full[seq_len(k)] - r_vec)
  RV <- R_mat %*% V_obs %*% t(R_mat)
  stat_obs <- if (nrow(R_mat) == 1L) diff_obs[1] / sqrt(RV[1, 1]) else as.numeric(crossprod(diff_obs, solve(RV, diff_obs))) / nrow(R_mat)

  grids <- as.matrix(do.call(expand.grid, replicate(G, c(-1, 1), simplify = FALSE)))
  cl_masks <- lapply(uniq_cl, function(gval) cl == gval)
  t_boot <- numeric(nrow(grids))
  for (b in seq_len(nrow(grids))) {
    s_vec <- numeric(n)
    for (gi in seq_len(G)) s_vec[cl_masks[[gi]]] <- grids[b, gi]
    u_b <- u_dgp * s_vec
    y_b <- as.numeric(D %*% beta_dgp) + u_b
    beta_b <- qr.coef(qr_fit, y_b * sw)
    beta_b[is.na(beta_b)] <- 0
    e_b <- as.numeric(y_b - D %*% beta_b)
    sc <- matrix(0, nrow = G, ncol = p_tot)
    for (gi in seq_len(G)) {
      msk <- cl_masks[[gi]]
      sc[gi, ] <- colSums(Dw[msk, , drop = FALSE] * (sw[msk] * e_b[msk]))
    }
    V_b <- XtX_inv[seq_len(k), , drop = FALSE] %*% (ssc_fac * crossprod(sc)) %*% XtX_inv[, seq_len(k), drop = FALSE]
    diff_b <- as.numeric(R_mat %*% beta_b[seq_len(k)] - R_mat %*% beta_dgp[seq_len(k)])
    RV_b <- R_mat %*% V_b %*% t(R_mat)
    t_boot[b] <- if (nrow(R_mat) == 1L) diff_b[1] / sqrt(RV_b[1, 1]) else as.numeric(crossprod(diff_b, solve(RV_b, diff_b))) / nrow(R_mat)
  }
  eps <- 1e-12 * (1 + abs(stat_obs))
  pval <- if (nrow(R_mat) == 1L) mean(abs(t_boot) > abs(stat_obs) + eps) else mean(t_boot > stat_obs + eps)
  list(stat = stat_obs, p_val = pval, t_boot = sort(t_boot))
}

test_that("explicit-dummy WLS brute-force oracle matches WCR11 and joint WCU11", {
  skip_if_not_installed("MASS")
  df <- make_wb_data(96L, seed = 7L)
  df$cl <- (seq_len(nrow(df)) - 1L) %% 4L
  m <- feols(y ~ x1 + x2 | fe1 + fe2, df, weights = ~w, cluster = ~cl, nthreads = 1)

  R1 <- matrix(c(1, 2), nrow = 1L)
  r1 <- 0.2
  o1 <- explicit_dummy_wls_oracle(df, R1, r1, impose_null = TRUE)
  bt1 <- boottest(m, R = R1, r = r1, B = 16L, type = "rademacher", bootstrap_type = "11", impose_null = TRUE, seed = 1, nthreads = 1)
  expect_true(bt1$full_enumeration)
  expect_equal(bt1$reps, 16L)
  expect_equal(bt1$statistic, o1$stat, tolerance = 1e-10)
  expect_equal(bt1$p_val, o1$p_val, tolerance = 1e-12)
  expect_equal(sort(bt1$t_boot), o1$t_boot, tolerance = 1e-9)

  R2 <- diag(2)
  r2 <- c(0.1, -0.2)
  o2 <- explicit_dummy_wls_oracle(df, R2, r2, impose_null = FALSE)
  bt2 <- boottest(m, R = R2, r = r2, B = 16L, type = "rademacher", bootstrap_type = "11", impose_null = FALSE, seed = 1, nthreads = 1)
  expect_equal(bt2$q, 2L)
  expect_equal(bt2$stat_type, "F")
  expect_equal(bt2$statistic, o2$stat, tolerance = 1e-10)
  expect_equal(bt2$p_val, o2$p_val, tolerance = 1e-12)
  expect_equal(sort(bt2$t_boot), o2$t_boot, tolerance = 1e-9)
})

test_that("OLS MNW types, hetero, weights, p_val_type, full enum, multiway, posthoc summary", {
  df <- make_wb_data(240L, seed = 11L)
  m <- feols(y ~ x1 + x2 | fe1 + fe2, df, vcov = ~c1, nthreads = 2)

  for (bt in c("11", "13", "31", "33")) {
    r <- boottest(m, param = "x1", B = 99L, bootstrap_type = bt, seed = 123, nthreads = 2)
    expect_equal(r$method, "wildcluster")
    expect_equal(r$bootstrap_type, bt)
    expect_true(r$p_val >= 0 && r$p_val <= 1)
  }

  for (wt in c("rademacher", "webb", "mammen", "normal")) {
    rw <- boottest(m, param = "x1", B = 49L, type = wt, seed = 5, nthreads = 2)
    expect_equal(rw$type, wt)
    expect_length(rw$t_boot, 49L)
  }

  p_gt <- pvalue(boottest(m, param = "x1", B = 99L, p_val_type = ">", conf_int = FALSE, seed = 7, nthreads = 2))
  p_lt <- pvalue(boottest(m, param = "x1", B = 99L, p_val_type = "<", conf_int = FALSE, seed = 7, nthreads = 2))
  p_eq <- pvalue(boottest(m, param = "x1", B = 99L, p_val_type = "equal-tailed", conf_int = FALSE, seed = 7, nthreads = 2))
  p_gt_a <- pvalue(boottest(m, param = "x1", B = 99L, p_val_type = "greater", conf_int = FALSE, seed = 7, nthreads = 2))
  expect_equal(p_gt, p_gt_a)
  expect_equal(p_gt + p_lt, 1.0, tolerance = 1e-12)
  expect_equal(p_eq, 2 * min(p_gt, p_lt), tolerance = 1e-12)

  # Unclustered hetero variants
  mh <- feols(y ~ x1 + x2 | fe1, df, vcov = "hetero", nthreads = 2)
  for (bt in c("11", "21", "31", "13", "33")) {
    rh <- boottest(mh, param = "x1", B = 49L, bootstrap_type = bt, seed = 9, nthreads = 2)
    expect_equal(rh$method, "hetero")
    expect_null(rh$n_clusters)
  }

  # 2-way cluster + bootcluster overrides + formula/character forms
  m2 <- feols(y ~ x1 + x2 | fe1, df, vcov = ~c1 + c2, nthreads = 2)
  r2w <- boottest(m2, param = "x1", B = 49L, seed = 17, nthreads = 2)
  expect_equal(r2w$n_clusters, 24L)
  rbc <- boottest(m2, param = "x1", B = 49L, bootcluster = ~c1, seed = 17, nthreads = 2)
  expect_equal(rbc$n_clusters, 12L)
  expect_equal(rbc$bootcluster, "c1")

  # Respects post-hoc summary(..., vcov = ~c1)
  m_iid <- feols(y ~ x1 + x2 | fe1, df, vcov = "iid", nthreads = 2)
  m_post <- summary(m_iid, vcov = ~c1)
  r_post <- boottest(m_post, param = "x1", B = 49L, seed = 17, nthreads = 2)
  r_ov <- boottest(m_iid, param = "x1", B = 49L, clustid = c("c1"), seed = 17, nthreads = 2)
  expect_equal(r_post$method, "wildcluster")
  expect_equal(r_post$statistic, r_ov$statistic, tolerance = 1e-12)
  expect_equal(r_post$p_val, r_ov$p_val)
})

test_that("IV coef_perm alignment, GLM score bootstrap, CI inversion, threads, S3 methods & errors", {
  df <- make_wb_data(240L, seed = 41L)

  # IV/2SLS: coef(iv) puts endog first (`fit_endog`, `x1`, `x2`) while core puts [x1, x2, endog]
  iv <- feols(y ~ x1 + x2 | fe1 | endog ~ z1 + z2, df, vcov = ~c1, nthreads = 2)
  b_iv <- coef(iv)
  expect_identical(names(b_iv)[1L], "fit_endog")

  r_en <- boottest(iv, param = "fit_endog", B = 99L, method = "wre", seed = 5, nthreads = 2)
  expect_equal(r_en$method, "wre")
  expect_equal(r_en$point_estimate, unname(b_iv["fit_endog"]), tolerance = 1e-12)

  r_ex <- boottest(iv, param = "x1", B = 99L, method = "wre", seed = 5, nthreads = 2)
  expect_equal(r_ex$point_estimate, unname(b_iv["x1"]), tolerance = 1e-12)

  r_j_iv <- boottest(iv, param = c("fit_endog", "x2"), B = 99L, method = "score", seed = 5, nthreads = 2)
  expect_equal(r_j_iv$q, 2L)
  expect_equal(r_j_iv$stat_type, "score F")

  R_iv <- matrix(c(2, -1, 3), nrow = 1L)
  r_mat_iv <- boottest(iv, R = R_iv, r = 0.4, B = 99L, method = "wre", seed = 5, nthreads = 2)
  expect_equal(r_mat_iv$point_estimate, as.numeric(R_iv %*% b_iv), tolerance = 1e-12)

  # Poisson & Logit
  p <- fepois(y_pois ~ x1 + x2 | fe1, df, vcov = ~c1, nthreads = 2)
  rp <- boottest(p, param = "x1", B = 99L, seed = 8, nthreads = 2)
  expect_equal(rp$method, "score")
  expect_equal(rp$stat_type, "score t")

  lg <- feglm(y_bin ~ x1 + x2 | fe2, df, family = "logit", vcov = ~c1, nthreads = 2)
  rl <- boottest(lg, param = "x2", B = 99L, seed = 8, nthreads = 2)
  expect_equal(rl$method, "score")
  expect_equal(rl$stat_type, "score t")

  # CI inversion check
  m <- feols(y ~ x1 + x2 | fe1, df, vcov = ~c1, nthreads = 2)
  r_ci <- boottest(m, param = "x1", B = 199L, sign_level = 0.05, seed = 11, tol = 1e-6, nthreads = 2)
  ci <- confint(r_ci)
  expect_true(is.data.frame(ci) && nrow(ci) == 1L)
  expect_lt(ci[1, 1], r_ci$point_estimate)
  expect_gt(ci[1, 2], r_ci$point_estimate)
  p_lo <- boottest(m, param = "x1", r = ci[1, 1], B = 199L, conf_int = FALSE, seed = 11, nthreads = 2)$p_val
  p_hi <- boottest(m, param = "x1", r = ci[1, 2], B = 199L, conf_int = FALSE, seed = 11, nthreads = 2)$p_val
  expect_true(abs(p_lo - 0.05) <= 0.02)
  expect_true(abs(p_hi - 0.05) <= 0.02)

  # Thread bit-exactness
  r_t1 <- boottest(m, param = "x1", B = 99L, bootstrap_type = "33", seed = 99, nthreads = 1)
  r_t2 <- boottest(m, param = "x1", B = 99L, bootstrap_type = "33", seed = 99, nthreads = 2)
  expect_identical(r_t1$statistic, r_t2$statistic)
  expect_identical(r_t1$p_val, r_t2$p_val)
  expect_identical(r_t1$t_boot, r_t2$t_boot)
  expect_equal(tstat(r_t1), r_t1$statistic)
  expect_output(print(r_ci), "Wild bootstrap test:")
  expect_output(summary(r_ci), "Wild bootstrap test:")

  # Errors
  df$x_collin <- 2 * df$x1
  mc <- feols(y ~ x1 + x_collin + x2 | fe1, df, vcov = ~c1, nthreads = 2)
  expect_error(boottest(mc, param = "x_collin", B = 19L), "collinearity")
  expect_error(boottest(mc, param = "missing", B = 19L), "not found")
  expect_error(boottest(m, param = "x1", R = diag(2), B = 19L), "not both")
  expect_error(boottest(m, B = 19L), "either")
  expect_error(boottest(m, R = matrix(c(1, 1, 2, 2), nrow = 2L, byrow = TRUE), B = 19L), "full row rank")

  df$t <- seq_len(nrow(df))
  mnw <- feols(y ~ x1 + x2 | fe1, df, vcov = NW ~ t, nthreads = 2)
  expect_error(boottest(mnw, param = "x1", B = 19L), "clustid")

  multi <- feols(c(y, y_pois) ~ x1 | fe1, df, nthreads = 2)
  expect_error(boottest(multi, param = "x1", B = 19L), "fereg_multi")

  bc <- suppressWarnings(biasCorr(lg))
  expect_error(boottest(bc, param = "x1", B = 19L), "bias correction")
})
