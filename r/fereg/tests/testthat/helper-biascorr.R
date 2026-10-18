# Data and references for the bias-correction and APE tests (test-biascorr.R, test-apes.R,
# test-jackknife.R): small simulated logit panels, alpaca at its exact control, and plain-R
# references for what alpaca does not have (jackknife; sampling term over all observations).

alpaca_ok <- function() requireNamespace("alpaca", quietly = TRUE) && requireNamespace("data.table", quietly = TRUE)

# fereg at tight tolerances, evaluated in the caller's frame so that biasCorr() / getAPEs() find
# the data again from the call; alpaca at the control that gives the maximum likelihood estimate
# (its default control rejects the last IRLS step).
fe_logit <- function(fml, data, ...) {
  eval.parent(substitute(feglm(fml, data, family = "logit", glm.tol = 1e-12, fixef.tol = 1e-12, notes = FALSE, ...)))
}
al_feglm <- function(fml, data, ...) {
  alpaca::feglm(fml, data.table::copy(data.table::as.data.table(data)), stats::binomial(), ...,
    control = alpaca::feglmControl(dev.tol = 1e-20, center.tol = 1e-14, iter.max = 100L))
}

# Relative difference scaled by the largest reference entry (vectors and matrices).
rel_diff <- function(ours, ref) {
  ours <- as.vector(unclass(ours))
  ref <- as.vector(unclass(ref))
  stopifnot(length(ours) == length(ref))
  max(abs(ours - ref)) / max(abs(ref), .Machine$double.xmin)
}

# Checks `ours` against `ref` at `tol` and records the difference under `group` (written to the
# CSV file in the environment variable FEREG_BC_REPORT when set, for the test report).
expect_close <- function(ours, ref, tol, group) {
  d <- rel_diff(ours, ref)
  out <- Sys.getenv("FEREG_BC_REPORT")
  if (nzchar(out)) cat(sprintf("%s,%.3e\n", group, d), file = out, append = TRUE)
  testthat::expect_lte(d, tol, label = paste0(group, ": relative difference"))
}

# ---- simulated panels ------------------------------------------------------------------------------

# Classic panel of N individuals over T periods: continuous x1, binary d, outcome y; `sd_a` the
# spread of the individual effects (large values give individuals with a constant outcome);
# `drop` the share of rows removed at random (unbalanced panel); `dynamic` adds the lagged
# outcome ylag (first period removed).
sim_classic <- function(N, T, seed, sd_a = 0.5, drop = 0, dynamic = FALSE) {
  set.seed(seed)
  a <- rnorm(N, 0, sd_a)
  g <- rnorm(T + dynamic, 0, 0.5)
  d <- expand.grid(t = seq_len(T + dynamic), i = seq_len(N))[, c("i", "t")]
  d$x1 <- rnorm(nrow(d)) + 0.3 * a[d$i]
  d$d <- as.integer(runif(nrow(d)) < 0.4)
  y <- integer(nrow(d))
  for (r in seq_len(nrow(d))) {
    lag <- if (dynamic && d$t[r] > 1) y[r - 1] else 0L
    y[r] <- as.integer(rlogis(1) < 0.5 * lag + d$x1[r] - 0.5 * d$d[r] + a[d$i[r]] + g[d$t[r]])
  }
  d$y <- y
  if (dynamic) {
    d$ylag <- c(NA, y[-length(y)])
    d <- d[d$t > 1, ]
    d$t <- d$t - 1L
  }
  if (drop > 0) d <- d[runif(nrow(d)) > drop, ]
  rownames(d) <- NULL
  d
}

# Network panel: all ordered pairs of `C` countries over `Y` years; exporter-time, importer-time
# and pair identifiers as numbers that sort in time order (alpaca's convention).
sim_network <- function(C, Y, seed) {
  set.seed(seed)
  d <- expand.grid(year = seq_len(Y), imp = seq_len(C), exp = seq_len(C))[, c("exp", "imp", "year")]
  d <- d[d$exp != d$imp, ]
  a <- matrix(rnorm(C * Y, sd = 0.5), C, Y)
  b <- matrix(rnorm(C * Y, sd = 0.5), C, Y)
  p <- matrix(rnorm(C * C, sd = 0.7), C, C)
  d$x <- rnorm(nrow(d))
  d$rta <- as.integer(runif(nrow(d)) < 0.3)
  d$y <- as.integer(rlogis(nrow(d)) < 0.8 * d$x + 0.5 * d$rta + a[cbind(d$exp, d$year)] +
    b[cbind(d$imp, d$year)] + p[cbind(d$exp, d$imp)])
  d$exp_time <- d$exp * 100 + d$year
  d$imp_time <- d$imp * 100 + d$year
  d$pair <- d$exp * 100 + d$imp
  rownames(d) <- NULL
  d
}

# Dyadic cross-section for the double jackknife: all ordered pairs of `C` countries.
sim_dyadic <- function(C, seed) {
  set.seed(seed)
  d <- expand.grid(jd = seq_len(C), id = seq_len(C))[, c("id", "jd")]
  d <- d[d$id != d$jd, ]
  a <- rnorm(C, sd = 0.5)
  b <- rnorm(C, sd = 0.5)
  d$x1 <- rnorm(nrow(d))
  d$d <- as.integer(runif(nrow(d)) < 0.4)
  d$y <- as.integer(rlogis(nrow(d)) < d$x1 - 0.5 * d$d + a[d$id] + b[d$jd])
  d$id <- sprintf("c%02d", d$id)
  d$jd <- sprintf("c%02d", d$jd)
  rownames(d) <- NULL
  d
}

# ---- alpaca-based references --------------------------------------------------------------------------

# alpaca's bias term (biasCorr.R:117-167) for one or both biases: `which` = "both", "individual"
# (first fixed effect, with the spectral term) or "time" (second fixed effect). Classic only.
alpaca_bias_term <- function(a, which = "both", L = 0L) {
  fam <- a$family
  eta <- a$eta
  y <- a$data[[1L]]
  w <- fam$mu.eta(eta)
  v <- y - fam$linkinv(eta)
  z <- alpaca:::partialMuEta(eta, fam, 2L)
  n <- a$nobs[["nobs"]]
  kl <- alpaca:::getIndexList(names(a$lvls.k), a$data)
  MX <- a$MX
  b <- 0
  if (which != "time") {
    b <- b + as.vector(alpaca:::groupSums(MX * z, w, kl[[1L]])) / 2 / n
    if (L > 0) b <- b + as.vector(alpaca:::groupSumsSpectral(MX * w, v, w, L, kl[[1L]])) / n
  }
  if (which != "individual") b <- b + as.vector(alpaca:::groupSums(MX * z, w, kl[[2L]])) / 2 / n
  stats::setNames(solve(a$Hessian / n, -b), names(stats::coef(a)))
}

# alpaca's APE ingredients at its estimation point (getAPEs.R:168-288, weights included): partial
# effects `Delta` of the retained rows (alpaca's sorted order), their average `delta` over the N*
# rows, and the delta-method rows `Gamma`.
alpaca_ape_parts <- function(a) {
  fam <- a$family
  beta <- stats::coef(a)
  Nf <- a$nobs[["nobs.full"]]
  X <- stats::model.matrix(a$formula, a$data, rhs = 1L)[, -1L, drop = FALSE]
  attr(X, "dimnames") <- NULL
  y <- a$data[[1L]]
  wt <- a$weights
  if (is.null(wt)) wt <- 1
  eta <- a$eta
  mu.eta <- fam$mu.eta(eta)
  v <- wt * (y - fam$linkinv(eta))
  w <- wt * mu.eta
  MX <- a$MX
  PX <- X - MX
  p <- ncol(X)
  Delta <- Delta1 <- matrix(NA_real_, nrow(X), p)
  J <- matrix(NA_real_, p, p)
  for (j in seq_len(p)) {
    if (all(X[, j] %in% c(0, 1))) {
      eta0 <- eta - X[, j] * beta[[j]]
      f1 <- fam$mu.eta(eta0 + beta[[j]])
      Delta[, j] <- fam$linkinv(eta0 + beta[[j]]) - fam$linkinv(eta0)
      Delta1[, j] <- f1 - fam$mu.eta(eta0)
      J[, j] <- -colSums(PX * Delta1[, j]) / Nf
      J[j, j] <- sum(f1) / Nf + J[j, j]
      J[-j, j] <- colSums(X[, -j, drop = FALSE] * Delta1[, j]) / Nf + J[-j, j]
    } else {
      Delta[, j] <- beta[[j]] * mu.eta
      Delta1[, j] <- beta[[j]] * alpaca:::partialMuEta(eta, fam, 2L)
      J[, j] <- colSums(MX * Delta1[, j]) / Nf
      J[j, j] <- sum(mu.eta) / Nf + J[j, j]
    }
  }
  kl <- alpaca:::getIndexList(names(a$lvls.k), a$data)
  Psi <- -Delta1 / w
  PPsi <- Psi - alpaca:::centerVariables(Psi, w, kl, a$control$center.tol)
  Gamma <- (MX %*% solve(a$Hessian / Nf, J) - PPsi) * v / Nf
  list(Delta = Delta, delta = colSums(Delta) / Nf, Gamma = Gamma, Nf = Nf)
}

# The APE variance of fereg (design 6, core API 2.2) from alpaca's ingredients: delta-method part
# Gamma'Gamma, plus with `n.pop` the sampling-of-effects term and the weak-exogeneity covariance
# summed over all N* rows (rows removed for a constant outcome: Delta = 0, Gamma = 0,
# D = -delta / N*; grouped by their own fixed effects; C in time order within the individual
# (classic) / pair (network, three fixed effects), removed rows included): the include-all lines
# of RefAlpaca/mc_ape/mc_ape.R, generalized to one to three fixed effects and to C. `full`: the
# N* rows in input order, `kept`: the rows of `full` in fereg's estimation sample, `fe`: the
# fixed-effect variables in formula order (numbers or C-locale-sorted strings: alpaca's order).
ape_vcov_all_rows <- function(a, full, kept, fe, n.pop, panel.structure = "classic", sampling.fe = "independence",
                              weak.exo = FALSE) {
  P <- alpaca_ape_parts(a)
  V <- crossprod(P$Gamma)
  adj <- if (is.null(n.pop)) 0 else max((n.pop - P$Nf) / (n.pop - 1), 0)
  if (adj == 0) return(V)
  # alpaca's rows: the estimation rows sorted (stably) by the fixed-effect variables.
  retained <- full[kept, , drop = FALSE]
  pos <- kept[do.call(order, c(lapply(fe, function(v) retained[[v]]), method = "radix"))]
  stopifnot(nrow(full) == P$Nf, length(pos) == nrow(P$Delta))
  D <- matrix(rep(-P$delta / P$Nf, each = nrow(full)), ncol = length(P$delta))
  D[pos, ] <- sweep(P$Delta, 2, P$delta) / P$Nf
  G <- matrix(0, nrow(full), ncol(D))
  G[pos, ] <- P$Gamma
  if (sampling.fe == "independence") {
    S <- function(g) crossprod(rowsum(D, as.character(full[[g]])))
    V <- V + adj * S(fe[1])
    if (length(fe) > 1) V <- V + adj * (S(fe[2]) - crossprod(D))
    if (panel.structure == "network" && length(fe) > 2) V <- V + adj * (S(fe[3]) - crossprod(D))
  }
  if (weak.exo && (panel.structure == "classic" || length(fe) > 2)) {
    grp <- if (panel.structure == "classic") fe[1] else fe[3]
    keys <- if (panel.structure == "classic") fe[-1] else fe[1:2]
    o <- do.call(order, c(lapply(c(grp, keys), function(v) full[[v]]), list(seq_len(nrow(full))), method = "radix"))
    g <- as.character(full[[grp]])[o]
    Ds <- D[o, , drop = FALSE]
    before <- apply(Ds, 2, function(col) ave(col, g, FUN = cumsum)) - Ds
    C <- crossprod(matrix(before, nrow(Ds)), G[o, , drop = FALSE])
    V <- V + adj * (C + t(C))
  }
  V
}

# ---- jackknife reference (design 7): fixest refits on the subpanels ------------------------------

# Coefficients and APEs of a logit fit on the rows `keep` of `d` (fixest), the APEs averaged over
# the subpanel's own N* = sum(keep) rows; `binary`: names of the binary regressors.
jk_fit <- function(fml, d, keep, binary) {
  m <- fixest::feglm(fml, d[keep, , drop = FALSE], family = "logit", glm.tol = 1e-12, fixef.tol = 3e-12,
    nthreads = 1, notes = FALSE)
  beta <- stats::coef(m)
  X <- stats::model.matrix(m, type = "rhs")
  eta <- m$linear.predictors
  Delta <- vapply(names(beta), function(j) {
    if (j %in% binary) {
      eta0 <- eta - X[, j] * beta[[j]]
      stats::plogis(eta0 + beta[[j]]) - stats::plogis(eta0)
    } else {
      beta[[j]] * stats::dlogis(eta)
    }
  }, numeric(length(eta)))
  list(coef = beta, ape = colSums(Delta) / sum(keep))
}

# Halves of units 1..m (logitfe): the middle unit of an odd m is in both.
half <- function(idx, m, h) if (h == 1) idx <= ceiling(m / 2) else if (m %% 2 == 0) idx > m / 2 else idx >= ceiling(m / 2)

# Jackknife-corrected coefficients and APEs. `d`: the N* rows; `ind`, `time`: individual and
# time variables (the first / second fixed effect, or the panel.id time variable of a one-way
# model); `two_way`: whether the model has time effects (else only the individual-effect bias
# exists); `perms`: list of list(individual, period) permutations (1-based new indices) for
# `multiple`; `double`: the two fixed-effect variables sharing identifiers; `cache`: an
# environment keeping the subpanel fits between calls on the same data and model.
jk_reference <- function(fml, d, ind, time, variant, bias = "both", two_way = TRUE, perms = NULL,
                         binary = "d", double = NULL, cache = new.env()) {
  idx <- function(v) match(v, sort(unique(v), method = "radix"))
  ii <- idx(d[[ind]])
  tt <- idx(d[[time]])
  N <- max(ii)
  T <- max(tt)
  kind <- if (!two_way) "individual" else bias
  fit <- function(keep) {
    key <- paste(which(keep), collapse = ",")
    if (is.null(cache[[key]])) cache[[key]] <- jk_fit(fml, d, keep, binary)
    cache[[key]]
  }
  mean_of <- function(masks) {
    fits <- lapply(masks, fit)
    list(coef = Reduce(`+`, lapply(fits, `[[`, "coef")) / length(fits), ape = Reduce(`+`, lapply(fits, `[[`, "ape")) / length(fits))
  }
  full <- fit(rep(TRUE, nrow(d)))
  # Per-individual time halves: position of each row within its individual in time order.
  ph <- function(tt) {
    pos <- ave(seq_along(ii), ii, FUN = function(r) rank(tt[r] + r / (length(ii) + 1)))
    Ti <- ave(seq_along(ii), ii, FUN = length)
    list(pos <= ceiling(Ti / 2), ifelse(Ti %% 2 == 0, pos > Ti / 2, pos >= ceiling(Ti / 2)))
  }
  combine <- function(a, coef_full, terms) {
    # terms: list(list(weight, mean)) ; result a * full - sum(weight * mean)
    out <- list(coef = a * coef_full$coef, ape = a * coef_full$ape)
    for (t in terms) {
      out$coef <- out$coef - t[[1]] * t[[2]]$coef
      out$ape <- out$ape - t[[1]] * t[[2]]$ape
    }
    out
  }
  one <- function(ii, tt) {
    IH <- mean_of(list(half(ii, N, 1), half(ii, N, 2)))
    TH <- mean_of(list(half(tt, T, 1), half(tt, T, 2)))
    if (variant %in% c("js", "jj") || (variant == "double" && kind == "time")) {
      DI <- mean_of(lapply(seq_len(N), function(i) ii != i))
    }
    if (variant %in% c("sj", "jj") || (variant == "double" && kind == "individual")) {
      DT <- mean_of(lapply(seq_len(T), function(t) tt != t))
    }
    switch(variant,
      ss2 = switch(kind,
        both = combine(3, full, list(list(1, IH), list(1, mean_of(ph(tt))))),
        individual = combine(2, full, list(list(1, if (is.null(perms)) mean_of(ph(tt)) else TH))),
        time = combine(2, full, list(list(1, IH)))
      ),
      ss1 = switch(kind,
        both = combine(2, full, list(list(1, mean_of(list(
          half(ii, N, 1) & half(tt, T, 1), half(ii, N, 1) & half(tt, T, 2),
          half(ii, N, 2) & half(tt, T, 1), half(ii, N, 2) & half(tt, T, 2)
        ))))),
        individual = combine(2, full, list(list(1, TH))),
        time = combine(2, full, list(list(1, IH)))
      ),
      js = switch(kind,
        both = combine(N + 1, full, list(list(N - 1, DI), list(1, TH))),
        individual = combine(2, full, list(list(1, TH))),
        time = combine(N, full, list(list(N - 1, DI)))
      ),
      sj = switch(kind,
        both = combine(T + 1, full, list(list(T - 1, DT), list(1, IH))),
        individual = combine(T, full, list(list(T - 1, DT))),
        time = combine(2, full, list(list(1, IH)))
      ),
      jj = switch(kind,
        both = combine(N + T - 1, full, list(list(N - 1, DI), list(T - 1, DT))),
        individual = combine(T, full, list(list(T - 1, DT))),
        time = combine(N, full, list(list(N - 1, DI)))
      ),
      double = switch(kind,
        both = {
          ids <- intersect(as.character(d[[double[1]]]), as.character(d[[double[2]]]))
          J <- length(ids)
          DC <- mean_of(lapply(ids, function(c) !(as.character(d[[double[1]]]) == c | as.character(d[[double[2]]]) == c)))
          combine(J, full, list(list(J - 1, DC)))
        },
        individual = combine(T, full, list(list(T - 1, DT))),
        time = combine(N, full, list(list(N - 1, DI)))
      )
    )
  }
  if (is.null(perms)) return(one(ii, tt))
  res <- lapply(perms, function(p) {
    one(if (is.null(p$individual)) ii else p$individual[ii], if (is.null(p$period)) tt else p$period[tt])
  })
  list(coef = Reduce(`+`, lapply(res, `[[`, "coef")) / length(res), ape = Reduce(`+`, lapply(res, `[[`, "ape")) / length(res))
}
