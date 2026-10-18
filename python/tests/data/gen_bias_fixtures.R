# Reference numbers for python/tests/test_biascorr.py and test_apes.py (fixed-effects logit:
# analytical bias correction, average partial effects, jackknife corrections).
#
# Sources:
#  - alpaca 0.3.5 biasCorr() / getAPEs() / vcov() with alpaca's exact control
#    feglmControl(dev.tol = 1e-20, center.tol = 1e-14, iter.max = 100) (its default control stops
#    one IRLS step early, so its default coefficients are not the MLE);
#  - plain-R evaluations of the same formulas (alpaca's source, getAPEs.R / biasCorr.R /
#    GroupSums.cpp), asserted equal to alpaca wherever alpaca defines the quantity, for what alpaca
#    has no option for:
#      * bias = "individual" / "time" (alpaca always corrects both);
#      * quantities at the corrected point after an exact fixed-effect re-solve ("exact"; alpaca's
#        feglmOffset() stops early, so its own numbers, kept as "alpaca", differ at ~1e-7);
#      * the APE sampling-of-effects term over all N* observations, with zero partial effects for
#        the observations dropped for perfect classification (fereg design 6.3; alpaca sums over the
#        retained observations only, which differs when groups are dropped);
#  - jackknife corrections (fereg design 7, logitfe's definitions with the dimension fixes), by
#    refitting fixest::feglm (0.14.2) on every subpanel.
#
# The data are generated here and written as CSV files next to the JSON; every fit reads them back.
# Run from this directory:
#   R_LIBS=<lib with alpaca 0.3.5>:<lib with fixest 0.14.2> Rscript gen_bias_fixtures.R
# Writes bias_<name>.csv and bias_alpaca.json.
suppressPackageStartupMessages({
  library(alpaca)
  library(data.table)
  library(jsonlite)
})
stopifnot(packageVersion("alpaca") == "0.3.5", packageVersion("fixest") == "0.14.2")
CTL <- feglmControl(dev.tol = 1e-20, center.tol = 1e-14, iter.max = 100L)
CENTER_TOL <- 1e-14

# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------
set.seed(20261018)

# Redraws y on the rows of fixed-effect groups with a constant outcome (among the rows `ok`)
# until none is left, so that nothing is dropped and every interface keeps the same sample.
redraw <- function(dt, fes, ok = rep(TRUE, nrow(dt))) {
  for (it in 1:10000) {
    bad <- rep(FALSE, nrow(dt))
    for (f in fes) {
      m <- ave(ifelse(ok, dt$y, NA), dt[[f]], FUN = function(v) mean(v, na.rm = TRUE))
      bad <- bad | (ok & (m == 0 | m == 1))
    }
    if (!any(bad)) return(dt)
    dt$y[bad] <- as.numeric(runif(sum(bad)) < plogis(dt$eta0[bad]))
  }
  stop("constant groups remain")
}

mk_panel <- function(N, TT, sd_a) {
  a <- rnorm(N, 0, sd_a)
  g <- rnorm(TT, 0, 0.5)
  dt <- data.table(i = rep(seq_len(N), each = TT), t = rep(seq_len(TT), N))
  dt[, x1 := rnorm(.N) + 0.5 * a[i]]
  dt[, x2 := rnorm(.N) + 0.3 * g[t]]
  dt[, d := as.numeric(runif(.N) < 0.4)]
  dt[, eta0 := 0.8 * x1 - 0.6 * x2 + 0.5 * d + a[i] + g[t]]
  dt[, y := as.numeric(runif(.N) < plogis(eta0))]
  dt
}

finish <- function(dt, cols, name) {
  dt <- dt[, cols, with = FALSE]
  # multiples of 2^-12: exactly representable, written exactly, read back exactly by any parser
  for (v in names(dt)) if (is.double(dt[[v]])) set(dt, j = v, value = round(dt[[v]] * 4096) / 4096)
  dt[, rid := .I]
  file <- sprintf("bias_%s.csv", name)
  fwrite(dt, file)
  back <- fread(file)
  same <- function(a, b) if (is.character(a)) identical(a, b) else identical(as.numeric(a), as.numeric(b))
  ok <- mapply(same, back, dt)
  if (!identical(names(back), names(dt)) || !all(ok)) stop(sprintf("%s: CSV round trip differs in %s", file, paste(names(dt)[!ok], collapse = ", ")))
  back
}

# classic: balanced two-way panel, no constant groups (pyfixest's and alpaca's samples agree)
classic <- redraw(mk_panel(50, 8, 0.8), c("i", "t"))
classic <- finish(classic, c("i", "t", "y", "x1", "x2", "d"), "classic")

# unbal: two-way panel with 15% of the rows removed (gaps), missing x2 on 6 rows, rows shuffled
# (time order differs from input order)
unbal <- mk_panel(50, 10, 0.8)
unbal <- unbal[sort(sample.int(.N, round(0.85 * .N)))]
unbal[sample.int(.N, 6), x2 := NA]
unbal <- redraw(unbal, c("i", "t"), ok = !is.na(unbal$x2))
unbal <- unbal[sample.int(.N)]
unbal <- finish(unbal, c("i", "t", "y", "x1", "x2", "d"), "unbal")

# network: 10 countries, ordered pairs, 6 years, 5% of the rows removed; string identifiers
# (exporter-time "e03y2" sorts by year within an exporter, as alpaca's time order requires)
mk_network <- function(K, TT) {
  dt <- CJ(exp = seq_len(K), imp = seq_len(K), year = seq_len(TT))[exp != imp]
  ae <- matrix(rnorm(K * TT, 0, 0.5), K)
  bi <- matrix(rnorm(K * TT, 0, 0.5), K)
  cp <- matrix(rnorm(K * K, 0, 0.8), K)
  dt[, x := rnorm(.N) + 0.3 * cp[cbind(exp, imp)]]
  start <- matrix(sample.int(TT + 3, K * K, replace = TRUE), K)
  dt[, rta := as.numeric(year >= start[cbind(exp, imp)])]
  dt[, eta0 := 0.5 * x + 0.4 * rta + ae[cbind(exp, year)] + bi[cbind(imp, year)] + cp[cbind(exp, imp)]]
  dt[, y := as.numeric(runif(.N) < plogis(eta0))]
  dt[, exp_time := sprintf("e%02dy%d", exp, year)]
  dt[, imp_time := sprintf("i%02dy%d", imp, year)]
  dt[, pair := sprintf("p%02d_%02d", exp, imp)]
  dt
}
network <- mk_network(10, 6)
network <- network[sort(sample.int(.N, round(0.95 * .N)))]
network <- redraw(network, c("exp_time", "imp_time", "pair"))
network <- finish(network, c("exp", "imp", "year", "exp_time", "imp_time", "pair", "y", "x", "rta"), "network")

# drops: two-way panel with an all-one period and all-zero / all-one individuals: alpaca and fixest
# drop them iteratively (individuals 1-2, then period 1, then the individuals constant on periods
# 2-8); pyfixest drops the all-zero individuals once and keeps the other constant groups' rows at
# the bound of the mean. The all-one period is the first one and only the odd individuals have it,
# so some kept individuals' dropped rows precede their retained rows in time order and enter the
# all-rows weak-exogeneity covariance C (if every kept individual lost its first row, their
# contributions would cancel: the scores sum to zero at the MLE).
drops <- mk_panel(60, 8, 1.3)
drops[t == 1, y := 1]
drops <- drops[!(t == 1 & i %% 2 == 0)]
drops[i %in% 1:2, y := 0]
drops <- finish(drops, c("i", "t", "y", "x1", "x2", "d"), "drops")
local({
  m <- drops[t > 1 & i > 2, mean(y), by = i]$V1
  stopifnot(sum(m == 1) >= 2, sum(m == 0) >= 2)
})

# dyad: directed pairs of 12 countries, exporter and importer effects in one id space (the
# double jackknife); string copies of the ids in exps / imps
dyad <- CJ(exp = 1:12, imp = 1:12)[exp != imp]
local({
  a <- rnorm(12, 0, 0.7); b <- rnorm(12, 0, 0.7)
  dyad[, x1 := rnorm(.N) + 0.4 * a[exp]]
  dyad[, x2 := rnorm(.N)]
  dyad[, eta0 := 0.7 * x1 - 0.5 * x2 + a[exp] + b[imp]]
  dyad[, y := as.numeric(runif(.N) < plogis(eta0))]
})
dyad <- redraw(dyad, c("exp", "imp"))
dyad[, exps := sprintf("c%02d", exp)]
dyad[, imps := sprintf("c%02d", imp)]
dyad <- finish(dyad, c("exp", "imp", "exps", "imps", "y", "x1", "x2"), "dyad")

# prior weights w (alpaca weights; fereg aweights) and a user offset off for the classic data
# (drawn last, so the other data sets do not depend on them)
classic[, w := runif(.N, 0.5, 2)]
classic[, off := 0.5 * rnorm(.N)]
classic <- finish(classic, c("i", "t", "y", "x1", "x2", "d", "w", "off"), "classic")

DATA <- list(classic = classic, unbal = unbal, network = network, drops = drops, dyad = dyad)

# ------------------------------------------------------------------------------------------------
# Formulas (plain R, alpaca's definitions)
# ------------------------------------------------------------------------------------------------
mu.eta <- function(e) { m <- plogis(e); m * (1 - m) }
d2 <- function(e) mu.eta(e) * (1 - 2 * plogis(e))
d3 <- function(e) { f <- mu.eta(e); f * ((1 - 2 * plogis(e))^2 - 2 * f) }
klist <- function(fes) lapply(fes, function(f) split(seq_along(f) - 1L, f))
demean <- function(X, w, fes) alpaca:::centerVariables(as.matrix(X), w, klist(fes), CENTER_TOL)
gsum <- function(M, w, f) colSums(rowsum(M, f) / as.vector(rowsum(w, f)))
gspec <- function(M, v, w, L, f) {
  out <- numeric(ncol(M))
  for (g in split(seq_along(f), f)) {
    Tg <- length(g); num <- numeric(ncol(M))
    for (l in seq_len(L)) if (Tg > l) for (t in (l + 1):Tg) num <- num + M[g[t], ] * v[g[t - l]] * Tg / (Tg - l)
    out <- out + num / sum(w[g])
  }
  out
}
gvar <- function(M, f) crossprod(rowsum(M, f))
# sum_g sum_{t<s} M_{g[t]} N_{g[s]}' with the rows of each group in the order `ord`
gcov <- function(M, N, f, ord = seq_along(f)) {
  C <- matrix(0, ncol(M), ncol(M))
  for (g in split(ord, f[ord])) if (length(g) > 1) {
    suf <- apply(N[g, , drop = FALSE], 2, function(col) rev(cumsum(rev(col))))
    suf <- matrix(suf, length(g))
    C <- C + crossprod(M[g[-length(g)], , drop = FALSE], suf[-1L, , drop = FALSE])
  }
  C
}
relmax <- function(a, b) max(abs(as.vector(a) - as.vector(b)) / pmax(abs(as.vector(b)), 1e-300))

# Terms of the static bias for structure / bias choice; spectral term group (or NULL).
bias_terms <- function(structure, which, k) {
  if (structure == "network") return(seq_len(k))
  switch(which, both = seq_len(k), individual = 1L, time = 2L)
}
spec_group <- function(structure, which, k, L) {
  if (L == 0) return(NULL)
  if (structure == "classic") return(if (which == "time") NULL else 1L)
  if (k == 3) 3L else NULL
}

# Exact fixed effects at fixed coefficients: Newton on the FE-only problem from `eta_start`
# (prior weights `wt`, offset `off` = X beta + user offset).
fe_resolve <- function(y, off, fes, eta_start, wt = 1) {
  eta <- eta_start
  for (it in 1:200) {
    mu <- plogis(eta); w <- wt * mu * (1 - mu)
    zz <- (y - mu) / (mu * (1 - mu)) + eta - off
    eta_new <- off + zz - as.vector(demean(zz, w, fes))
    done <- max(abs(eta_new - eta)) < 1e-12
    eta <- eta_new
    if (done) return(eta)
  }
  stop("fe_resolve did not converge")
}

# A fitted model in alpaca's sorted sample `d` plus the pre-drop rows `P` (N* rows).
# `weights`: prior-weight column (alpaca's weights). `offset`: user-offset column; alpaca has no
# offset, so the model is then fitted by fixest (data in input order, which must be sorted by FE1
# and FE2, i.e. alpaca's order) and polished to the exact MLE by full Newton steps.
setup <- function(dat, fml_alpaca, xnames, fevars, structure, weights = NULL, offset = NULL) {
  vars <- unique(c("y", xnames, fevars, weights, offset, "rid"))
  P <- na.omit(dat[, vars, with = FALSE])
  if (is.null(offset)) {
    m <- feglm(fml_alpaca, copy(dat), binomial(), weights = weights, control = CTL)
    d <- m$data
    stopifnot(m$nobs[["nobs.full"]] == nrow(P), m$conv)
    coef0 <- unname(coef(m)); eta0 <- m$eta
  } else {
    stopifnot(is.null(weights))
    fml <- as.formula(sub(" \\| rid$", "", deparse1(fml_alpaca)))
    m <- fixest::feglm(fml, P, family = "logit", offset = as.formula(paste("~", offset)), glm.tol = 1e-12,
                       fixef.tol = 3e-12, glm.iter = 200, notes = FALSE)
    d <- P[fixest::obs(m)]
    stopifnot(identical(do.call(order, c(unname(as.list(d[, fevars, with = FALSE])), list(d$rid))), seq_len(nrow(d))))
    coef0 <- unname(coef(m)); eta0 <- m$linear.predictors
  }
  X <- as.matrix(d[, xnames, with = FALSE]); attr(X, "dimnames") <- NULL
  fes <- lapply(fevars, function(v) as.integer(factor(d[[v]])))
  # FE codes over the pre-drop rows (sorted levels) and the retained rows' positions in P
  fes_full <- lapply(fevars, function(v) match(P[[v]], sort(unique(P[[v]]))))
  pos <- match(d$rid, P$rid)
  # time order over P (fereg design 3.1 / 6.3): classic two-way FE2 within FE1, one-way input
  # order, network (FE1, FE2) within FE3; ties in input order
  k <- length(fevars)
  ord <- if (structure == "network") order(fes_full[[k]], fes_full[[1]], fes_full[[2]], P$rid)
         else if (k == 2) order(fes_full[[1]], fes_full[[2]], P$rid) else order(fes_full[[1]], P$rid)
  S <- list(m = m, d = d, P = P, X = X, y = d$y, fes = fes, fes_full = fes_full, pos = pos, ord = ord,
            n = nrow(d), Nf = nrow(P), k = k, structure = structure, xnames = xnames, fevars = fevars,
            wt = if (is.null(weights)) rep(1, nrow(d)) else d[[weights]],
            off = if (is.null(offset)) rep(0, nrow(d)) else d[[offset]])
  if (!is.null(offset)) {
    for (it in 1:5) {  # Newton polish: beta += H^-1 MX'v, then the exact FEs at beta
      q <- point(S, coef0, eta0)
      coef1 <- coef0 + as.vector(solve(q$H, crossprod(q$MX, q$v)))
      eta0 <- fe_resolve(S$y, as.vector(X %*% coef1) + S$off, fes, eta0 + as.vector(X %*% (coef1 - coef0)))
      coef0 <- coef1
    }
  }
  S$coef0 <- coef0; S$eta0 <- eta0
  S
}

# Ingredients at (beta, eta): v, w, z carry the prior weights (alpaca).
point <- function(S, beta, eta) {
  mu <- plogis(eta); w <- S$wt * mu.eta(eta); v <- S$wt * (S$y - mu); z <- S$wt * d2(eta)
  MX <- demean(S$X, w, S$fes)
  H <- crossprod(MX * sqrt(w))
  list(beta = beta, eta = eta, w = w, v = v, z = z, MX = MX, H = H,
       deviance = -2 * sum(S$wt * (S$y * log(mu) + (1 - S$y) * log(1 - mu))))
}

vcovs <- function(S, q) {
  G <- q$MX * q$v
  Hi <- solve(q$H)
  cl <- S$fes[[1]]
  list(iid = Hi, hetero = Hi %*% crossprod(G) %*% Hi, cluster = Hi %*% gvar(G, cl) %*% Hi)
}

# Analytical bias correction of the coefficients at the uncorrected point q.
correct <- function(S, q, which, L) {
  b <- Reduce(`+`, lapply(bias_terms(S$structure, which, S$k), function(kk) gsum(q$MX * q$z, q$w, S$fes[[kk]]))) / 2 / S$n
  sg <- spec_group(S$structure, which, S$k, L)
  if (!is.null(sg)) b <- b + gspec(q$MX * q$w, q$v, q$w, L, S$fes[[sg]]) / S$n
  bias <- solve(q$H / S$n, -b)
  list(bias_term = bias, coef = q$beta - bias)
}

# APEs at q: delta (corrected when `corr` = list(which, L)), the delta-method part and the
# pieces of the sampling term, for both row sets ("all" = fereg, "retained" = alpaca). The partial
# effects are unweighted (alpaca), the scores weighted.
apes <- function(S, q, corr = NULL) {
  X <- S$X; beta <- q$beta; eta <- q$eta; w <- q$w; v <- q$v; z <- q$z; MX <- q$MX
  n <- S$n; Nf <- S$Nf; p <- ncol(X)
  f <- mu.eta(eta)
  binary <- apply(X, 2, function(x) all(x %in% c(0, 1)))
  Delta <- Delta1 <- Delta2 <- matrix(NA_real_, n, p); J <- matrix(NA_real_, p, p)
  PX <- X - MX
  for (j in seq_len(p)) {
    if (binary[j]) {
      e0 <- eta - X[, j] * beta[j]; e1 <- e0 + beta[j]
      Delta[, j] <- plogis(e1) - plogis(e0); Delta1[, j] <- mu.eta(e1) - mu.eta(e0)
      Delta2[, j] <- d2(e1) - d2(e0)
      J[, j] <- colSums(MX * Delta1[, j]) / Nf
      J[j, j] <- (sum(mu.eta(e1)) - sum(PX[, j] * Delta1[, j])) / Nf
    } else {
      Delta[, j] <- beta[j] * f; Delta1[, j] <- beta[j] * d2(eta); Delta2[, j] <- beta[j] * d3(eta)
      J[, j] <- colSums(MX * Delta1[, j]) / Nf; J[j, j] <- J[j, j] + sum(f) / Nf
    }
  }
  delta <- colSums(Delta) / Nf
  Psi <- -Delta1 / w; MPsi <- demean(Psi, w, S$fes); PPsi <- Psi - MPsi
  bterm <- NULL
  if (!is.null(corr)) {
    bterm <- Reduce(`+`, lapply(bias_terms(S$structure, corr$which, S$k), function(kk) gsum(Delta2 + PPsi * z, w, S$fes[[kk]]))) / 2 / n
    sg <- spec_group(S$structure, corr$which, S$k, corr$L)
    if (!is.null(sg)) bterm <- bterm - gspec(MPsi * w, v, w, corr$L, S$fes[[sg]]) / n
  }
  Gam <- (MX %*% solve(q$H / Nf, J) - PPsi) * v / Nf
  D <- sweep(Delta, 2, delta) / Nf
  Dall <- matrix(rep(-delta / Nf, each = Nf), Nf); Dall[S$pos, ] <- D
  Gall <- matrix(0, Nf, p); Gall[S$pos, ] <- Gam
  list(delta = if (is.null(bterm)) delta else delta - bterm, bias_term = bterm, V0 = crossprod(Gam),
       D = D, Gam = Gam, Dall = Dall, Gall = Gall)
}

# APE variance for n_pop / sampling / weak exogeneity on a row set.
ape_vcov <- function(S, a, npop, sampling, weak, rows = c("all", "retained")) {
  rows <- match.arg(rows)
  adj <- if (is.null(npop)) 0 else (npop - S$Nf) / (npop - 1)
  if (adj < 0) adj <- 0
  V <- a$V0
  if (adj > 0) {
    if (rows == "all") { D <- a$Dall; G <- a$Gall; fes <- S$fes_full; ord <- S$ord }
    else { D <- a$D; G <- a$Gam; fes <- S$fes; ord <- seq_len(S$n) }
    if (sampling == "independence") {
      V <- V + adj * gvar(D, fes[[1]])
      if (S$k > 1) V <- V + adj * (gvar(D, fes[[2]]) - crossprod(D))
      if (S$structure == "network" && S$k > 2) V <- V + adj * (gvar(D, fes[[3]]) - crossprod(D))
    }
    if (weak) {
      grp <- if (S$structure == "classic") 1L else if (S$k > 2) 3L else NULL
      if (!is.null(grp)) { C <- gcov(D, G, fes[[grp]], ord); V <- V + adj * (C + t(C)) }
    }
  }
  V
}

mat <- function(M) unname(lapply(seq_len(nrow(M)), function(r) unname(M[r, ])))
TOL_ALPACA <- 1e-9  # plain-R formulas vs alpaca at alpaca's own point

check <- function(what, a, b, tol = TOL_ALPACA) {
  r <- relmax(a, b)
  if (!(r < tol)) stop(sprintf("%s: rel diff %.3g vs alpaca", what, r))
  r
}

# APE entries of one object (feglm: all 8 combinations; corrected: weak_exo = L > 0, alpaca Q7).
ape_entries <- function(S, q, corr, obj_alpaca = NULL, q_alpaca = NULL) {
  a <- apes(S, q, corr)
  aa <- if (!is.null(q_alpaca)) apes(S, q_alpaca, corr) else NULL
  drops <- S$Nf > S$n
  out <- list()
  weak_set <- if (is.null(corr)) c(FALSE, TRUE) else (corr$L > 0)
  for (npop in list(NULL, 2L * S$Nf)) for (sf in c("independence", "unrestricted")) for (we in weak_set) {
    V <- ape_vcov(S, a, npop, sf, we, "all")
    e <- list(n_pop = npop, sampling_fe = sf, weak_exo = we, delta = a$delta, vcov = mat(V))
    if (!is.null(corr)) e$bias_term <- a$bias_term
    if (!is.null(obj_alpaca)) {
      ga <- getAPEs(obj_alpaca, n.pop = npop, panel.structure = S$structure, sampling.fe = sf, weak.exo = we)
      # the plain-R formulas reproduce alpaca on its own row set at its own point ...
      check("ape delta", aa$delta, ga$delta)
      check("ape vcov", ape_vcov(S, aa, npop, sf, we, "retained"), ga$vcov)
      if (!is.null(corr)) check("ape bias term", aa$bias_term, ga$bias.term)
      # ... and on all rows whenever nothing is dropped
      if (!drops) check("ape vcov all rows", ape_vcov(S, aa, npop, sf, we, "all"), ga$vcov)
      e$alpaca <- list(delta = unname(ga$delta), vcov = mat(ga$vcov))
      if (!is.null(corr)) e$alpaca$bias_term <- unname(ga$bias.term)
    }
    out[[length(out) + 1L]] <- e
  }
  out
}

vcov_entry <- function(V) lapply(V, mat)

# ------------------------------------------------------------------------------------------------
# Analytical cases
# ------------------------------------------------------------------------------------------------
model_case <- function(id, data, xnames, fevars, structure, corrections, weights = NULL, offset = NULL) {
  fml_py <- sprintf("y ~ %s | %s", paste(xnames, collapse = " + "), paste(fevars, collapse = " + "))
  fml_alpaca <- as.formula(sprintf("%s | rid", fml_py))
  S <- setup(DATA[[data]], fml_alpaca, xnames, fevars, structure, weights = weights, offset = offset)
  m <- S$m
  q <- point(S, S$coef0, S$eta0)
  alpaca_fit <- is.null(offset)
  V <- vcovs(S, q)
  if (alpaca_fit) {
    check("Hessian", q$H, m$Hessian)
    check("vcov iid", V$iid, vcov(m, type = "hessian"))
    check("vcov hetero", V$hetero, vcov(m, type = "sandwich"))
    check("vcov cluster", V$cluster, vcov(m, type = "clustered", cluster = as.formula(paste("~", fevars[1]))))
  } else {
    # the polished MLE: score of the coefficients at zero
    stopifnot(max(abs(crossprod(q$MX, q$v))) < 1e-10)
  }
  out <- list(id = id, data = data, fml = fml_py, fe = fevars, structure = structure, weights = weights,
              offset = offset, n = S$n, n_full = S$Nf, coef_names = xnames, coef = S$coef0,
              deviance = q$deviance, cluster = fevars[1], vcov = vcov_entry(V),
              apes = ape_entries(S, q, NULL, if (alpaca_fit) m, q), corrections = list())
  for (cr in corrections) {
    L <- cr$L; which <- cr$bias
    c0 <- correct(S, q, which, L)
    eta_c <- fe_resolve(S$y, as.vector(S$X %*% c0$coef) + S$off, S$fes,
                        q$eta + as.vector(S$X %*% (c0$coef - q$beta)), S$wt)
    qc <- point(S, c0$coef, eta_c)
    Vc <- vcovs(S, qc)
    corr <- list(which = which, L = L)
    e <- list(L = L, bias = which, coef = c0$coef, bias_term = unname(c0$bias_term), deviance = qc$deviance,
              vcov = vcov_entry(Vc), apes = ape_entries(S, qc, corr))
    if (which == "both" && alpaca_fit) {
      bc <- biasCorr(m, L = L, panel.structure = structure)
      check("biasCorr coefficients", c0$coef, coef(bc))
      check("biasCorr bias.term", c0$bias_term, bc$bias.term)
      qa <- point(S, unname(coef(bc)), bc$eta)
      check("biasCorr Hessian", qa$H, bc$Hessian)
      Va <- vcovs(S, qa)
      check("biasCorr vcov iid", Va$iid, vcov(bc, type = "hessian"))
      check("biasCorr vcov hetero", Va$hetero, vcov(bc, type = "sandwich"))
      check("biasCorr vcov cluster", Va$cluster, vcov(bc, type = "clustered", cluster = as.formula(paste("~", fevars[1]))))
      ea <- ape_entries(S, qa, corr, bc, qa)
      e$alpaca <- list(coef = unname(coef(bc)), bias_term = unname(bc$bias.term), vcov = vcov_entry(Va),
                       apes = lapply(ea, function(x) x$alpaca))
      cat(sprintf("  %s L=%d: alpaca re-solve vs exact: coef-space eta %.2g, vcov %.2g, APE %.2g\n", id, L,
                  max(abs(bc$eta - eta_c)), relmax(Va$iid, Vc$iid), relmax(ea[[1]]$alpaca$delta, e$apes[[1]]$delta)))
    }
    out$corrections[[length(out$corrections) + 1L]] <- e
  }
  cat(sprintf("%s: n = %d, N* = %d\n", id, S$n, S$Nf))
  out
}

C <- function(L, bias = "both") list(L = L, bias = bias)
X3 <- c("x1", "x2", "d")
MODELS <- list(
  model_case("classic", "classic", X3, c("i", "t"), "classic",
             list(C(0), C(1), C(2), C(0, "individual"), C(0, "time"), C(2, "individual"))),
  model_case("oneway", "classic", X3, "i", "classic", list(C(0), C(2))),
  model_case("unbal", "unbal", X3, c("i", "t"), "classic", list(C(0), C(1), C(2), C(0, "time"), C(1, "individual"))),
  model_case("oneway_unbal", "unbal", X3, "i", "classic", list(C(1))),
  model_case("network3", "network", c("x", "rta"), c("exp_time", "imp_time", "pair"), "network", list(C(0), C(2))),
  model_case("network2", "network", c("x", "rta"), c("exp_time", "imp_time"), "network", list(C(0), C(2))),
  model_case("drops", "drops", X3, c("i", "t"), "classic", list(C(0), C(1), C(0, "individual"))),
  model_case("weights", "classic", X3, c("i", "t"), "classic", list(C(0), C(1), C(0, "individual")), weights = "w"),
  model_case("offset", "classic", X3, c("i", "t"), "classic", list(C(0), C(2), C(0, "time")), offset = "off")
)

# ------------------------------------------------------------------------------------------------
# Jackknife cases (fereg design 7)
# ------------------------------------------------------------------------------------------------
jk_case <- function(id, data, xnames, fevars, variant, which, time_id = NULL) {
  dat <- DATA[[data]]
  fml_py <- sprintf("y ~ %s | %s", paste(xnames, collapse = " + "), paste(fevars, collapse = " + "))
  fml <- as.formula(fml_py)
  vars <- unique(c("y", xnames, fevars, time_id, "rid"))
  P <- na.omit(dat[, vars, with = FALSE])
  p <- length(xnames)
  oneway <- length(fevars) == 1L
  if (oneway) which_eff <- "individual" else which_eff <- which
  ind <- match(P[[fevars[1]]], sort(unique(P[[fevars[1]]])))
  tv <- if (oneway) time_id else fevars[2]
  per <- match(P[[tv]], sort(unique(P[[tv]])))
  N <- max(ind); TT <- max(per)
  first <- function(idx, m) idx <= ceiling(m / 2)
  second <- function(idx, m) if (m %% 2 == 0) idx > m / 2 else idx >= ceiling(m / 2)
  o <- order(ind, per, P$rid)
  pos <- integer(nrow(P)); pos[o] <- ave(seq_along(o), ind[o], FUN = seq_along)
  Ti <- tabulate(ind, N)[ind]
  PH1 <- pos <= ceiling(Ti / 2)
  PH2 <- ifelse(Ti %% 2 == 0, pos > Ti / 2, pos >= ceiling(Ti / 2))
  IH1 <- first(ind, N); IH2 <- second(ind, N); TH1 <- first(per, TT); TH2 <- second(per, TT)

  nfits <- 0L
  fit <- function(keep, label) {
    S <- P[keep]
    m <- fixest::feglm(fml, S, family = "logit", glm.tol = 1e-12, fixef.tol = 3e-12, glm.iter = 200,
                       notes = FALSE, warn = FALSE)
    b <- coef(m)
    if (length(b) != p || !all(is.finite(b)) || !m$convStatus) stop(sprintf("%s: subpanel %s lost a coefficient", id, label))
    eta <- m$linear.predictors
    X <- as.matrix(S[fixest::obs(m), xnames, with = FALSE])
    Delta <- matrix(NA_real_, nrow(X), p)
    for (j in seq_len(p)) {
      if (all(X[, j] %in% c(0, 1))) {
        e0 <- eta - X[, j] * b[j]
        Delta[, j] <- plogis(e0 + b[j]) - plogis(e0)
      } else Delta[, j] <- b[j] * mu.eta(eta)
    }
    list(coef = unname(b), ape = colSums(Delta) / nrow(S))
  }
  sub <- function(keep, label) { nfits <<- nfits + 1L; fit(keep, label) }
  avg <- function(fits) list(coef = Reduce(`+`, lapply(fits, `[[`, "coef")) / length(fits),
                             ape = Reduce(`+`, lapply(fits, `[[`, "ape")) / length(fits))
  full <- fit(rep(TRUE, nrow(P)), "full")
  halves <- function(a, b) avg(list(sub(a, "h1"), sub(b, "h2")))
  del_ind <- function() avg(lapply(seq_len(N), function(i) sub(ind != i, sprintf("i%d", i))))
  del_per <- function() avg(lapply(seq_len(TT), function(t) sub(per != t, sprintf("t%d", t))))
  comb <- function(a, terms) {  # a * full - sum_k c_k * terms_k
    list(coef = a * full$coef - Reduce(`+`, lapply(terms, function(tc) tc[[1]] * tc[[2]]$coef)),
         ape = a * full$ape - Reduce(`+`, lapply(terms, function(tc) tc[[1]] * tc[[2]]$ape)))
  }
  J <- NA_integer_
  res <- switch(paste(variant, which_eff),
    "ss2 both" = comb(3, list(list(1, halves(IH1, IH2)), list(1, halves(PH1, PH2)))),
    "ss2 individual" = comb(2, list(list(1, halves(PH1, PH2)))),
    "ss2 time" = comb(2, list(list(1, halves(IH1, IH2)))),
    "ss1 both" = comb(2, list(list(1, avg(list(sub(IH1 & TH1, "q11"), sub(IH1 & TH2, "q12"),
                                                sub(IH2 & TH1, "q21"), sub(IH2 & TH2, "q22")))))),
    "ss1 individual" = comb(2, list(list(1, halves(TH1, TH2)))),
    "ss1 time" = comb(2, list(list(1, halves(IH1, IH2)))),
    "js both" = comb(N + 1, list(list(N - 1, del_ind()), list(1, halves(TH1, TH2)))),
    "js individual" = comb(2, list(list(1, halves(TH1, TH2)))),
    "js time" = comb(N, list(list(N - 1, del_ind()))),
    "sj both" = comb(TT + 1, list(list(TT - 1, del_per()), list(1, halves(IH1, IH2)))),
    "sj individual" = comb(TT, list(list(TT - 1, del_per()))),
    "sj time" = comb(2, list(list(1, halves(IH1, IH2)))),
    "jj both" = comb(N + TT - 1, list(list(N - 1, del_ind()), list(TT - 1, del_per()))),
    "jj individual" = comb(TT, list(list(TT - 1, del_per()))),
    "jj time" = comb(N, list(list(N - 1, del_ind()))),
    "double both" = {
      ids <- sort(intersect(unique(P[[fevars[1]]]), unique(P[[fevars[2]]])))
      J <- length(ids)
      dd <- avg(lapply(ids, function(c) sub(P[[fevars[1]]] != c & P[[fevars[2]]] != c, sprintf("c%s", c))))
      comb(J, list(list(J - 1, dd)))
    },
    "double individual" = comb(TT, list(list(TT - 1, del_per()))),
    "double time" = comb(N, list(list(N - 1, del_ind()))),
    stop("unknown variant")
  )
  # variance at (beta_J, eta_J): exact fixed effects re-solved on the full-sample fit
  S <- setup(dat, as.formula(sprintf("%s | rid", fml_py)), xnames, fevars, "classic")
  stopifnot(relmax(coef(S$m), full$coef) < 1e-8)
  q0 <- point(S, unname(coef(S$m)), S$m$eta)
  off <- as.vector(S$X %*% res$coef)
  eta_j <- fe_resolve(S$y, off, S$fes, q0$eta - as.vector(S$X %*% q0$beta) + off)
  qj <- point(S, res$coef, eta_j)
  aj <- apes(S, qj, NULL)
  cat(sprintf("jackknife %s %s %s: %d fits\n", id, variant, which, nfits))
  list(id = id, data = data, fml = fml_py, time_id = time_id, variant = variant, bias = which,
       n_individuals = N, n_periods = TT, n_double = J, n_full = nrow(P), n_fits = nfits,
       coef_names = xnames, coef = res$coef, ape = res$ape, coef_uncorr = full$coef, ape_uncorr = full$ape,
       deviance = qj$deviance, vcov = vcov_entry(vcovs(S, qj)), ape_vcov = mat(aj$V0))
}

JK <- list()
for (v in c("ss2", "ss1", "js", "sj", "jj")) for (b in c("both", "individual", "time")) {
  JK[[length(JK) + 1L]] <- jk_case(sprintf("classic_%s_%s", v, b), "classic", X3, c("i", "t"), v, b)
  JK[[length(JK) + 1L]] <- jk_case(sprintf("drops_%s_%s", v, b), "drops", X3, c("i", "t"), v, b)
}
for (b in c("both", "individual", "time"))
  JK[[length(JK) + 1L]] <- jk_case(sprintf("dyad_double_%s", b), "dyad", c("x1", "x2"), c("exp", "imp"), "double", b)
for (v in c("ss2", "ss1", "js", "sj", "jj", "double"))
  JK[[length(JK) + 1L]] <- jk_case(sprintf("oneway_%s", v), "unbal", X3, "i", v, "individual", time_id = "t")

out <- list(
  meta = list(alpaca = as.character(packageVersion("alpaca")), fixest = as.character(packageVersion("fixest")),
              control = list(dev.tol = 1e-20, center.tol = 1e-14, iter.max = 100L), seed = 20261018L,
              data = lapply(DATA, function(d) nrow(d))),
  models = MODELS,
  jackknife = JK
)
writeLines(toJSON(out, digits = I(17), auto_unbox = TRUE, null = "null", pretty = FALSE), "bias_alpaca.json")
cat("wrote bias_alpaca.json\n")
