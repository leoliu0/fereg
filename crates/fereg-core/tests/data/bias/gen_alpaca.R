# Reference fixtures for the fixed-effects-logit bias correction and APEs (alpaca normalization).
# Run from the repository root:  Rscript crates/fereg-core/tests/data/bias/gen_alpaca.R
# Writes (deterministic, fixed seeds) into crates/fereg-core/tests/data/bias/:
#   a_classic.csv        i,t,y,x1,x2,d,w,off    N=200 x T=8 (d binary); period 8 is all-one among the
#                                               individuals that are not all-zero, so after the
#                                               constant-outcome individuals are dropped period 8 is
#                                               dropped too (dropped rows inside kept individuals).
#                                               w = prior weight round(U(0.5, 2), 3) (own seed),
#                                               off = 0.3 sin(i t) + 0.1 t / 8 (offset; an additive
#                                               f(i) + g(t) would be absorbed by the i + t FEs and
#                                               leave beta unchanged); only the classic_w /
#                                               classic_off cases use w / off.
#   a_dynamic.csv        i,t,y,ylag,x1,d        N=200 x T=10, ylag = y of the previous period (NA at t=1).
#   a_dynamic_unbal.csv  same columns           a_dynamic with 15% of the rows deleted at random; ylag is
#                                               NOT recomputed: it stays the generated value of the
#                                               previous calendar period, also when that row was deleted.
#   a_network.csv        exp,imp,year,exp_time,imp_time,pair,y,ylag,x,rta
#                                               12 countries, 132 ordered pairs x 6 years,
#                                               exp_time = exp*100+year, imp_time = imp*100+year,
#                                               pair = exp*100+imp, ylag NA in year 1, rta binary.
#   alpaca.json          expectations, see below.
# CSV: header row, comma separated, numeric only, NA written as an EMPTY field. Doubles are written
# with the shortest of %.15g/%.16g/%.17g that round-trips; the script re-reads the CSVs and estimates
# on the re-read data, so the fixtures describe exactly the written numbers.
#
# alpaca.json = {"cases": [case, ...]}; each case:
#   name, data (relative to tests/data), structure ("classic"|"network"), y, x (coefficient order),
#   fe (FE columns in formula order), n (estimation rows), n_full (N*: rows after NA removal, before
#   the constant-outcome drop), n_pop (= 2 N*, the finite-population size used in "apes"),
#   binary (per x: all values in {0,1} on the estimation sample),
#   weights / offset (only classic_w / classic_off): name of the prior-weight / offset column,
#   feglm      {coef, hessian, eta_sum, deviance}: alpaca::feglm with exact control
#              feglmControl(dev.tol = 1e-20, center.tol = 1e-14, iter.max = 100); eta includes the
#              offset; deviance = -2 sum wt log-likelihood.
#              classic_off (alpaca has no offset): EVERYTHING in the case is this script's
#              reimplementation (source "reimpl"): exact joint Newton (beta + FEs) from fixest's fit to
#              max|d eta| < 1e-13, biasCorr formulas, re-solve with offset X beta_c + off.
#              Weights (classic_w): v = wt (y - mu), w = wt mu', z = wt mu'' everywhere (bias, MX,
#              Hessian, scores, re-solve); APE Delta/delta and J are UNweighted (alpaca Q13).
#   fixest_coef: fixest::feglm(family = "logit", glm.tol = 1e-12, fixef.tol = 1e-11) (cross-check;
#              fixest 0.14.2 rejects fixef.tol <= 2.2e-12)
#   bc         [{L, which:"both", coef, bias_term, coef_uncorr, source:"alpaca",
#                hessian_c, vcov_hessian_c, vcov_sandwich_c, eta_c_sum, fe_score_max}]
#              coef/bias_term/coef_uncorr = alpaca::biasCorr (bias_term = solve(H/n, -b) = beta_hat -
#              beta_c); the *_c quantities come from this script's independent reimplementation at the
#              EXACT FE re-solve (Newton on the FEs with offset X beta_c, max|d eta| < 1e-13):
#              hessian_c = MX_c' W_c MX_c, vcov_hessian_c = solve(hessian_c),
#              vcov_sandwich_c = H^-1 (G'G) H^-1 with G = MX_c * wt (y - mu_c), eta_c_sum = sum(eta_c),
#              fe_score_max = max over all FE groups of |sum_g wt (y - mu_c)|.
#   bc_which   (classic, oneway, dynamic_unbal) [{L, which, coef, bias_term, source:"reimpl"}]
#              individual = GS over FE1 (+ spectral over FE1 if L > 0); time = GS over FE2 only.
#   apes       [{object:"feglm"|"bc", L, n_pop (null|2N*), sampling ("independence"|"unrestricted"),
#                weak_exo (requested), weak_exo_used, rows ("retained"|"all"), delta, vcov (p x p),
#                bias_term (APE bias b, null for feglm), source ("alpaca"|"reimpl"),
#                [alpaca_delta, alpaca_vcov: alpaca getAPEs on its own biasCorr object, for
#                 information only; bc + rows "retained"]}]
#              rows "retained": alpaca's rule (D and S(.) over the n estimation rows).
#              rows "all": D and S(.) over all N* rows; dropped rows have Delta = 0, D = -delta/N*, and
#              are grouped by their own FE levels; weak-exogeneity C runs over all rows of an FE1
#              (classic) / FE3 (network, 3 terms) group in time order with Gamma = 0 on dropped rows.
#              Time order (alpaca): rows sorted stably by (FE1, FE2[, FE3]) values.
#              bc objects: alpaca forces weak.exo = (L > 0) (weak_exo_used).
# Matrices are flat row-major arrays (p x p), coefficient order = order of "x".
.libPaths(c("/tmp/bc/rlib", "/home/leo/Documents/fereg-jss/tools/rlib", .libPaths()))
suppressPackageStartupMessages({library(alpaca); library(data.table)})
stopifnot(packageVersion("alpaca") == "0.3.5", packageVersion("fixest") == "0.14.2",
          packageVersion("data.table") == "1.18.6.1")
fixest::setFixest_notes(FALSE)
dir <- "crates/fereg-core/tests/data/bias"
stopifnot(dir.exists(dirname(dir)))
dir.create(dir, showWarnings = FALSE)
CTL <- feglmControl(dev.tol = 1e-20, center.tol = 1e-14, iter.max = 100L)

# ---------------------------------------------------------------- output helpers
fmt <- function(x) {                       # shortest round-trip decimal of finite doubles
  x <- as.double(x); out <- sprintf("%.15g", x)
  for (d in c(16L, 17L)) {
    bad <- as.numeric(out) != x
    if (any(bad)) out[bad] <- sprintf(paste0("%.", d, "g"), x[bad])
  }
  stopifnot(all(as.numeric(out) == x))
  out
}
wcsv <- function(d, f) {
  d <- as.data.frame(d)
  cols <- lapply(d, function(v) {
    s <- rep("", length(v)); ok <- !is.na(v)
    s[ok] <- if (is.double(v)) fmt(v[ok]) else as.character(v[ok]); s
  })
  writeLines(c(paste(names(d), collapse = ","), do.call(paste, c(cols, sep = ","))), file.path(dir, f))
  fread(file.path(dir, f))                 # estimate on exactly what was written
}
rm_ <- function(M) I(as.vector(t(M)))       # p x p matrix -> flat row-major array
vec <- function(v) I(unname(as.vector(v)))
to_json <- function(v, ind = "") {
  if (is.null(v)) return("null")
  if (is.list(v)) {
    ind2 <- paste0(ind, " ")
    if (!is.null(names(v))) {
      items <- vapply(names(v), function(k) paste0(ind2, '"', k, '": ', to_json(v[[k]], ind2)), "")
      return(paste0("{\n", paste(items, collapse = ",\n"), "\n", ind, "}"))
    }
    items <- vapply(v, function(e) paste0(ind2, to_json(e, ind2)), "")
    return(paste0("[\n", paste(items, collapse = ",\n"), "\n", ind, "]"))
  }
  asis <- inherits(v, "AsIs"); v <- unclass(v)
  s <- if (is.character(v)) ifelse(is.na(v), "null", paste0('"', v, '"'))
       else if (is.logical(v)) ifelse(is.na(v), "null", ifelse(v, "true", "false"))
       else { s <- rep("null", length(v)); ok <- is.finite(v); s[ok] <- fmt(v[ok]); s }
  if (length(v) == 1L && !asis) s else paste0("[", paste(s, collapse = ", "), "]")
}

# internal check maxima
CHK <- list()
chk <- function(name, val, tol) {
  CHK[[name]] <<- max(CHK[[name]], val)
  if (!(val <= tol)) stop(sprintf("check %s failed: %.3g > %.3g", name, val, tol))
}
rel <- function(a, b) max(abs(as.vector(a) - as.vector(b))) / max(abs(as.vector(b)), 1e-300)

# ---------------------------------------------------------------- data
# 1) classic two-way: strong individual effects; period 8 forced all-one except all-zero individuals
set.seed(20261018L)
N <- 200L; TT <- 8L
alpha <- rnorm(N, sd = 2.5); gam <- rnorm(TT, sd = 0.5)
dc <- data.table(i = rep(seq_len(N), each = TT), t = rep(seq_len(TT), N))
dc[, x1 := round(rnorm(.N) + 0.3 * alpha[i], 6)]
dc[, x2 := round(rnorm(.N), 6)]
dc[, d := as.integer(runif(.N) < 0.4)]
dc[, y := as.integer(rlogis(.N) < 1.0 * x1 - 1.0 * x2 + 0.5 * d + alpha[i] + gam[t])]
zero17 <- dc[t < TT, .(s = sum(y)), by = i][s == 0L, i]
stopifnot(length(zero17) > 0L)
dc[t == TT, y := as.integer(!(i %in% zero17))]
setcolorder(dc, c("i", "t", "y", "x1", "x2", "d"))
set.seed(20261021L)
dc[, w := round(runif(.N, 0.5, 2), 3)]
dc[, off := 0.3 * sin(i * t) + 0.1 * t / 8]
dc <- wcsv(dc, "a_classic.csv")

# 2) dynamic classic panel, ylag NA in the first period
set.seed(20261019L)
N <- 200L; TT <- 10L
alpha <- rnorm(N, sd = 1.5); gam <- rnorm(TT, sd = 0.5)
dd <- data.table(i = rep(seq_len(N), each = TT), t = rep(seq_len(TT), N))
dd[, x1 := round(rnorm(.N) + 0.5 * alpha[i], 6)]
dd[, d := as.integer(runif(.N) < 0.4)]
yy <- integer(nrow(dd))
for (ii in seq_len(N)) {
  yl <- as.integer(runif(1) < plogis(alpha[ii]))
  for (r in which(dd$i == ii)) {
    yl <- as.integer(rlogis(1) < 0.5 * yl + dd$x1[r] - 0.5 * dd$d[r] + alpha[ii] + gam[dd$t[r]])
    yy[r] <- yl
  }
}
dd[, y := yy]
dd[, ylag := shift(y), by = i]
setcolorder(dd, c("i", "t", "y", "ylag", "x1", "d"))
set.seed(7L)
du <- dd[runif(.N) > 0.15]                 # gaps; ylag keeps the generated previous-period value
dd <- wcsv(dd, "a_dynamic.csv")
du <- wcsv(du, "a_dynamic_unbal.csv")

# 3) network: 12 countries, ordered pairs, 6 years, ylag NA in year 1
set.seed(4041L)
C <- 12L; Y <- 6L
nw <- CJ(exp = seq_len(C), imp = seq_len(C), year = seq_len(Y))[exp != imp]
a_et <- matrix(rnorm(C * Y, sd = 0.5), C, Y); b_jt <- matrix(rnorm(C * Y, sd = 0.5), C, Y)
c_ij <- matrix(rnorm(C * C, sd = 0.5), C, C)
nw[, x := round(rnorm(.N) + 0.3 * c_ij[cbind(exp, imp)], 6)]
sw <- matrix(sample(c(Y + 1L, 2:Y), C * C, replace = TRUE, prob = c(0.5, rep(0.5 / (Y - 1), Y - 1))), C, C)
nw[, rta := as.integer(year >= sw[cbind(exp, imp)])]
setkey(nw, exp, imp, year)
yy <- integer(nrow(nw))
for (r in seq_len(nrow(nw))) {
  if (nw$year[r] == 1L) yl <- as.integer(runif(1) < plogis(c_ij[nw$exp[r], nw$imp[r]]))
  yl <- as.integer(rlogis(1) < 0.8 * yl + 0.5 * nw$x[r] + 0.4 * nw$rta[r] + a_et[nw$exp[r], nw$year[r]] +
                     b_jt[nw$imp[r], nw$year[r]] + c_ij[nw$exp[r], nw$imp[r]] - 0.4)
  yy[r] <- yl
}
nw[, y := yy]
nw[, ylag := shift(y), by = .(exp, imp)]
nw[, `:=`(exp_time = exp * 100L + year, imp_time = imp * 100L + year, pair = exp * 100L + imp)]
setcolorder(nw, c("exp", "imp", "year", "exp_time", "imp_time", "pair", "y", "ylag", "x", "rta"))
setkey(nw, NULL)
nw <- wcsv(nw, "a_network.csv")

# ---------------------------------------------------------------- independent reimplementation
mu.eta <- function(e) { m <- plogis(e); m * (1 - m) }
d2 <- function(e) mu.eta(e) * (1 - 2 * plogis(e))
d3 <- function(e) { f <- mu.eta(e); f * ((1 - 2 * plogis(e))^2 - 2 * f) }
dummies <- function(fes) do.call(cbind, lapply(fes, function(f) {
  f <- factor(f); m <- matrix(0, length(f), nlevels(f)); m[cbind(seq_along(f), as.integer(f))] <- 1; m
}))
# exact weighted residual on all FE dummies (dense QR, no iteration tolerance)
demean <- function(A, w, Dm) { sw <- sqrt(w); qr.resid(qr(Dm * sw), as.matrix(A) * sw) / sw }
gsum <- function(M, w, f) colSums(rowsum(M, f) / as.vector(rowsum(w, f)))
gspec <- function(M, v, w, L, f) {                  # rows of each group in stored (time) order
  out <- numeric(ncol(M))
  for (g in split(seq_along(f), f)) {
    Tg <- length(g); num <- numeric(ncol(M))
    for (l in seq_len(L)) if (Tg > l) for (t in (l + 1):Tg) num <- num + M[g[t], ] * v[g[t - l]] * Tg / (Tg - l)
    out <- out + num / sum(w[g])
  }
  out
}
gvar <- function(M, f) crossprod(rowsum(M, f))
gcov <- function(M, N, f) {                         # sum_g sum_{t<s} M_t N_s', suffix sums
  C <- matrix(0, ncol(M), ncol(M))
  for (g in split(seq_along(f), f)) if (length(g) > 1L) {
    Ng <- N[g, , drop = FALSE]
    suf <- apply(Ng, 2, function(c) rev(cumsum(rev(c))))
    if (!is.matrix(suf)) suf <- matrix(suf, nrow = length(g))
    suf <- rbind(suf[-1L, , drop = FALSE], 0)       # strictly later rows
    C <- C + crossprod(M[g, , drop = FALSE], suf)
  }
  C
}
fe_resolve <- function(y, off, Dm, eta0, wt) {      # Newton on the FEs given the offset
  eta <- eta0                                       # eta0 - off must lie in the FE span (warm start)
  for (it in 1:100) {
    mu <- plogis(eta); me <- mu * (1 - mu); w <- wt * me
    u <- (y - mu) / me                              # Newton step = P_w u (incremental form: the
    step <- u - demean(u, w, Dm)[, 1]               # rounding error shrinks with the step)
    eta <- eta + step
    dlt <- max(abs(step))
    if (dlt < 1e-13) break
  }
  if (!(dlt < 1e-13)) stop("FE re-solve did not converge: ", dlt)
  eta
}
fit_exact <- function(S, beta, eta) {               # joint Newton on (beta, FEs), offset in eta
  X <- S$X; y <- S$y
  for (it in 1:100) {
    mu <- plogis(eta); me <- mu * (1 - mu); w <- S$wt * me
    u <- (y - mu) / me
    MX <- demean(X, w, S$Dm); Mu <- demean(u, w, S$Dm)[, 1]
    db <- solve(crossprod(MX * sqrt(w)), crossprod(MX, w * Mu))[, 1]
    step <- u - Mu + as.vector(MX %*% db)           # fitted values of the WLS of u on [X, FE]
    beta <- beta + db; eta <- eta + step
    dlt <- max(abs(step))
    if (dlt < 1e-13) break
  }
  if (!(dlt < 1e-13)) stop("exact fit did not converge: ", dlt)
  list(beta = beta, eta = eta)
}
deviance_ <- function(y, eta, wt) -2 * sum(wt * ifelse(y == 1, plogis(eta, log.p = TRUE), plogis(-eta, log.p = TRUE)))

# sample: NA removal, stable sort by FE values (alpaca setkeyv), iterated constant-outcome drop
build_sample <- function(dat, cd) {
  vars <- unique(c(cd$y, cd$x, cd$fe, cd$extra, cd$weights, cd$offset))
  cc <- which(complete.cases(dat[, ..vars]))
  full <- dat[cc]
  ord <- do.call(order, c(unname(as.list(full[, cd$fe, with = FALSE])), method = "radix"))
  full <- full[ord]; rid <- cc[ord]
  yv <- full[[cd$y]]; keep <- rep(TRUE, nrow(full))
  repeat {
    n0 <- sum(keep)
    for (f in cd$fe) {
      idx <- which(keep); m <- ave(as.numeric(yv[idx]), full[[f]][idx])
      keep[idx[m <= 0 | m >= 1]] <- FALSE
    }
    if (sum(keep) == n0) break
  }
  fe_all <- lapply(cd$fe, function(f) full[[f]])
  X_all <- as.matrix(full[, cd$x, with = FALSE]); storage.mode(X_all) <- "double"
  fes <- lapply(fe_all, function(f) f[keep])
  wt <- if (is.null(cd$weights)) rep(1, nrow(full)) else as.numeric(full[[cd$weights]])
  off <- if (is.null(cd$offset)) rep(0, nrow(full)) else as.numeric(full[[cd$offset]])
  list(full = full, keep = keep, y = as.numeric(yv[keep]), X = X_all[keep, , drop = FALSE],
       fes = fes, fe_all = fe_all, Dm = dummies(fes), n = sum(keep), Nf = nrow(full),
       k = length(cd$fe), st = cd$st, rid = rid[keep], wt = wt[keep], off = off[keep])
}

bias_coef <- function(S, MX, w, v, z, L, which = "both") {
  n <- S$n; fes <- S$fes; k <- S$k; b <- numeric(ncol(MX))
  if (S$st == "classic") {
    if (which %in% c("both", "individual")) {
      b <- b + gsum(MX * z, w, fes[[1]]) / 2 / n
      if (L > 0) b <- b + gspec(MX * w, v, w, L, fes[[1]]) / n
    }
    if (k == 2L && which %in% c("both", "time")) b <- b + gsum(MX * z, w, fes[[2]]) / 2 / n
  } else {
    for (kk in seq_len(k)) b <- b + gsum(MX * z, w, fes[[kk]]) / 2 / n
    if (k == 3L && L > 0) b <- b + gspec(MX * w, v, w, L, fes[[3]]) / n
  }
  b
}

# APE ingredients at (beta, eta) on the retained rows (alpaca getAPEs.R:168-288). Prior weights enter
# v, w, z only; Delta, Delta1, Delta2 and J use the unweighted derivatives (alpaca Q13).
ape_point <- function(S, beta, eta, bc, L) {
  y <- S$y; X <- S$X; p <- ncol(X); n <- S$n; Nf <- S$Nf; fes <- S$fes; wt <- S$wt
  mu <- plogis(eta); me <- mu.eta(eta); m2 <- d2(eta)
  w <- wt * me; v <- wt * (y - mu); z <- wt * m2
  MX <- demean(X, w, S$Dm); H <- crossprod(MX * sqrt(w)); PX <- X - MX
  binary <- apply(X, 2, function(x) all(x %in% c(0, 1)))
  Delta <- Delta1 <- Delta2 <- matrix(NA_real_, n, p); J <- matrix(NA_real_, p, p)
  for (j in seq_len(p)) {
    if (binary[j]) {
      e0 <- eta - X[, j] * beta[j]; e1 <- e0 + beta[j]
      Delta[, j] <- plogis(e1) - plogis(e0); Delta1[, j] <- mu.eta(e1) - mu.eta(e0)
      Delta2[, j] <- d2(e1) - d2(e0)
      J[, j] <- colSums(MX * Delta1[, j]) / Nf
      J[j, j] <- (sum(mu.eta(e1)) - sum(PX[, j] * Delta1[, j])) / Nf
    } else {
      Delta[, j] <- beta[j] * me; Delta1[, j] <- beta[j] * m2; Delta2[, j] <- beta[j] * d3(eta)
      J[, j] <- colSums(MX * Delta1[, j]) / Nf; J[j, j] <- J[j, j] + sum(me) / Nf
    }
  }
  delta_unc <- colSums(Delta) / Nf
  delta <- delta_unc
  D <- sweep(Delta, 2, delta_unc) / Nf
  Psi <- -Delta1 / w; MPsi <- demean(Psi, w, S$Dm); PPsi <- Psi - MPsi
  b <- NULL
  if (bc) {
    ks <- seq_len(S$k)
    b <- Reduce(`+`, lapply(ks, function(kk) gsum(Delta2 + PPsi * z, w, fes[[kk]]))) / 2 / n
    if (L > 0) {
      if (S$st == "classic") b <- b - gspec(MPsi * w, v, w, L, fes[[1]]) / n
      if (S$st == "network" && S$k == 3L) b <- b - gspec(MPsi * w, v, w, L, fes[[3]]) / n
    }
    delta <- delta - b
  }
  Gam <- (MX %*% solve(H / Nf, J) - PPsi) * v / Nf
  list(delta = delta, delta_unc = delta_unc, b = b, D = D, Gam = Gam, H = H, MX = MX, mu = mu, w = w,
       v = v, z = z)
}
ape_vcov <- function(S, P, adj, sf, we, rows) {
  V <- crossprod(P$Gam)
  if (adj > 0) {
    if (rows == "retained") { D <- P$D; G <- P$Gam; fes <- S$fes }
    else {
      D <- matrix(rep(-P$delta_unc / S$Nf, each = S$Nf), S$Nf); D[S$keep, ] <- P$D
      G <- matrix(0, S$Nf, ncol(D)); G[S$keep, ] <- P$Gam; fes <- S$fe_all
    }
    if (sf == "independence") {
      V <- V + adj * gvar(D, fes[[1]])
      for (kk in seq_len(S$k)[-1]) V <- V + adj * (gvar(D, fes[[kk]]) - crossprod(D))
    }
    if (we) {
      gW <- if (S$st == "classic") fes[[1]] else if (S$k == 3L) fes[[3]] else NULL
      if (!is.null(gW)) { Cm <- gcov(D, G, gW); V <- V + adj * (Cm + t(Cm)) }
    }
  }
  V
}

# ---------------------------------------------------------------- one case
NPOP_GRID <- function(Nf) list(NULL, 2L * Nf)
run_case <- function(cd, dat) {
  S <- build_sample(dat, cd)
  y <- S$y; X <- S$X; n <- S$n; Nf <- S$Nf; p <- ncol(X)
  binary <- apply(X, 2, function(x) all(x %in% c(0, 1)))
  # fixest (cross-check and warm start of the exact joint Newton fit)
  ffml <- as.formula(paste(cd$y, "~", paste(cd$x, collapse = " + "), "|", paste(cd$fe, collapse = " + ")))
  fw <- if (is.null(cd$weights)) NULL else as.formula(paste("~", cd$weights))
  fo <- if (is.null(cd$offset)) NULL else as.formula(paste("~", cd$offset))
  fx <- fixest::feglm(ffml, data = as.data.frame(dat), family = "logit", weights = fw, offset = fo,
                      glm.tol = 1e-12, fixef.tol = 1e-11)
  stopifnot(fx$nobs == n)
  lp <- rep(NA_real_, nrow(dat)); lp[fixest::obs(fx)] <- fx$linear.predictors
  b0 <- coef(fx)[cd$x]; base <- S$off + as.vector(X %*% b0)
  own <- fit_exact(S, b0, base + qr.fitted(qr(S$Dm), lp[S$rid] - base))
  alp <- is.null(cd$offset)                          # alpaca has no offset
  if (alp) {
    rhs <- paste(paste(cd$x, collapse = " + "), "|", paste(cd$fe, collapse = " + "))
    if (!is.null(cd$extra)) rhs <- paste(rhs, "|", paste(cd$extra, collapse = " + "))
    m <- feglm(as.formula(paste(cd$y, "~", rhs)), copy(dat), binomial(), weights = cd$weights, control = CTL)
    stopifnot(m$conv)
    # same sample and row order as alpaca
    stopifnot(S$n == m$nobs[["nobs"]], S$Nf == m$nobs[["nobs.full"]])
    Xa <- model.matrix(m$formula, m$data, rhs = 1L)[, -1L, drop = FALSE]
    stopifnot(all(Xa == S$X), all(m$data[[cd$y]] == S$y), all(m$weights == S$wt),
              all(mapply(function(f, g) all(as.numeric(as.character(m$data[[f]])) == g), cd$fe, S$fes)))
    beta <- coef(m); eta <- m$eta
    # alpaca's exact-control fit is not always the MLE to 1e-10 (oneway: score 2.5e-7, coef 3.6e-9 off)
    a_score <- max(abs(crossprod(X, S$wt * (y - plogis(eta)))))
    cat(sprintf("  %s: alpaca beta-score max %.2g, rel diff own exact MLE vs alpaca %.2g\n", cd$name, a_score,
                rel(own$beta, beta)))
    chk("own_exact_fit_coef_vs_alpaca", rel(own$beta, beta), 1e-8)
    chk("own_deviance_vs_alpaca", rel(deviance_(y, eta, S$wt), m$deviance), 1e-12)
  } else {
    m <- NULL; beta <- own$beta; eta <- own$eta
  }
  chk("fixest_coef_vs_own_exact_mle", rel(coef(fx)[cd$x], own$beta), 1e-8)

  # uncorrected fit: my MX vs alpaca's
  q <- ape_point(S, beta, eta, FALSE, 0L)
  if (alp) {
    chk("MX_vs_alpaca", rel(q$MX, m$MX), 1e-9)
    chk("H_vs_alpaca", rel(q$H, m$Hessian), 1e-10)
    fit <- list(coef = vec(beta), hessian = rm_(m$Hessian), eta_sum = sum(m$eta), deviance = m$deviance)
  } else {
    chk("own_fit_score_max", max(abs(crossprod(X, q$v)), sapply(S$fes, function(f) max(abs(rowsum(q$v, f))))), 1e-10)
    fit <- list(coef = vec(beta), hessian = rm_(q$H), eta_sum = sum(eta), deviance = deviance_(y, eta, S$wt))
  }

  out <- list(name = cd$name, data = paste0("bias/", cd$file), structure = cd$st, y = cd$y,
              x = I(cd$x), fe = I(cd$fe), n = n, n_full = Nf, n_pop = 2L * Nf, binary = I(unname(binary)),
              feglm = fit, fixest_coef = vec(coef(fx)[cd$x]))
  if (!is.null(cd$weights)) out$weights <- cd$weights
  if (!is.null(cd$offset)) out$offset <- cd$offset

  apes <- list()
  add_apes <- function(obj, L, P, alp_obj) {
    for (np in NPOP_GRID(Nf)) for (sf in c("independence", "unrestricted")) for (we in c(FALSE, TRUE)) {
      adj <- if (is.null(np)) 0 else (np - Nf) / (np - 1)
      we_used <- if (obj == "bc") L > 0 else we
      a <- NULL
      if (!is.null(alp_obj)) {
        a <- getAPEs(alp_obj, n.pop = np, panel.structure = cd$st, sampling.fe = sf, weak.exo = we)
        stopifnot(isTRUE(a$weak.exo == we_used))
      }
      for (rows in c("retained", "all")) {
        V <- ape_vcov(S, P, adj, sf, we_used, rows)
        e <- list(object = obj, L = L, n_pop = if (is.null(np)) NA else np, sampling = sf,
                  weak_exo = we, weak_exo_used = we_used, rows = rows)
        if (obj == "feglm" && rows == "retained" && !is.null(a)) {
          chk("ape_feglm_delta_reimpl_vs_alpaca", rel(P$delta, a$delta), 1e-10)
          chk("ape_feglm_vcov_reimpl_vs_alpaca", rel(V, a$vcov), 1e-10)
          e <- c(e, list(delta = vec(a$delta), vcov = rm_(a$vcov), bias_term = NULL, source = "alpaca"))
        } else {
          e <- c(e, list(delta = vec(P$delta), vcov = rm_(V),
                         bias_term = if (is.null(P$b)) NULL else vec(P$b), source = "reimpl"))
          if (obj == "bc" && rows == "retained" && !is.null(a)) {
            chk("ape_bc_alpaca_vs_exact_resolve_delta", rel(P$delta, a$delta), 1e-4)
            e <- c(e, list(alpaca_delta = vec(a$delta), alpaca_vcov = rm_(a$vcov)))
          }
        }
        apes[[length(apes) + 1L]] <<- e
      }
    }
  }
  add_apes("feglm", 0L, q, m)

  bcl <- list()
  for (L in cd$Ls) {
    b <- bias_coef(S, q$MX, q$w, q$v, q$z, L)
    bt <- solve(q$H / n, -b); bcoef <- beta - bt
    if (alp) {
      bc <- biasCorr(m, L = L, panel.structure = cd$st)
      chk("bias_term_reimpl_vs_alpaca", rel(bt, bc$bias.term), 1e-10)
      chk("coef_bc_reimpl_vs_alpaca", rel(bcoef, coef(bc)), 1e-10)
      beta_c <- coef(bc); bt_out <- bc$bias.term; cu <- bc$coefficients.uncorr; src <- "alpaca"
    } else {
      bc <- NULL; beta_c <- bcoef; bt_out <- bt; cu <- beta; src <- "reimpl"
    }
    xb <- as.vector(X %*% beta_c)
    eta_c <- fe_resolve(y, S$off + xb, S$Dm, eta - as.vector(X %*% beta) + xb, S$wt)
    P <- ape_point(S, beta_c, eta_c, TRUE, L)
    fe_score <- max(sapply(S$fes, function(f) max(abs(rowsum(P$v, f)))))
    chk("fe_score_max_after_resolve", fe_score, 1e-10)
    if (alp) chk("H_c_alpaca_vs_exact_resolve", rel(P$H, bc$Hessian), 1e-4)
    Hi <- solve(P$H); G <- P$MX * P$v
    bcl[[length(bcl) + 1L]] <- list(L = L, which = "both", coef = vec(beta_c), bias_term = vec(bt_out),
      coef_uncorr = vec(cu), source = src, hessian_c = rm_(P$H),
      vcov_hessian_c = rm_(Hi), vcov_sandwich_c = rm_(Hi %*% crossprod(G) %*% Hi),
      eta_c_sum = sum(eta_c), fe_score_max = fe_score)
    add_apes("bc", L, P, bc)
  }
  out$bc <- bcl
  if (isTRUE(cd$which)) {
    bw <- list()
    for (wh in c("individual", if (S$k == 2L) "time")) for (L in c(0L, 2L)) {
      bt <- solve(q$H / n, -bias_coef(S, q$MX, q$w, q$v, q$z, L, wh))
      bw[[length(bw) + 1L]] <- list(L = L, which = wh, coef = vec(beta - bt), bias_term = vec(bt), source = "reimpl")
    }
    out$bc_which <- bw
  }
  out$apes <- apes
  list(json = out, S = S)
}

# ---------------------------------------------------------------- cases
CASES <- list(
  list(name = "classic", file = "a_classic.csv", dat = dc, y = "y", x = c("x1", "x2", "d"), fe = c("i", "t"),
       st = "classic", Ls = 0:2, which = TRUE),
  list(name = "classic_rev", file = "a_classic.csv", dat = dc, y = "y", x = c("x1", "x2", "d"), fe = c("t", "i"),
       st = "classic", Ls = 2L),
  list(name = "oneway", file = "a_classic.csv", dat = dc, y = "y", x = c("x1", "x2", "d"), fe = "i", extra = "t",
       st = "classic", Ls = c(0L, 2L), which = TRUE),
  list(name = "dynamic", file = "a_dynamic.csv", dat = dd, y = "y", x = c("ylag", "x1", "d"), fe = c("i", "t"),
       st = "classic", Ls = 0:2),
  list(name = "dynamic_unbal", file = "a_dynamic_unbal.csv", dat = du, y = "y", x = c("ylag", "x1", "d"),
       fe = c("i", "t"), st = "classic", Ls = 0:2, which = TRUE),
  list(name = "network3", file = "a_network.csv", dat = nw, y = "y", x = c("ylag", "x", "rta"),
       fe = c("exp_time", "imp_time", "pair"), st = "network", Ls = c(0L, 2L)),
  list(name = "network2", file = "a_network.csv", dat = nw, y = "y", x = c("x", "rta"),
       fe = c("exp_time", "imp_time"), st = "network", Ls = c(0L, 2L)),
  list(name = "classic_w", file = "a_classic.csv", dat = dc, y = "y", x = c("x1", "x2", "d"), fe = c("i", "t"),
       weights = "w", st = "classic", Ls = 0:1),
  list(name = "classic_off", file = "a_classic.csv", dat = dc, y = "y", x = c("x1", "x2", "d"), fe = c("i", "t"),
       offset = "off", st = "classic", Ls = 0:1))

t0 <- proc.time()[["elapsed"]]
res <- lapply(CASES, function(cd) { cat("case", cd$name, "\n"); run_case(cd, cd$dat) })
names(res) <- vapply(CASES, `[[`, "", "name")

# data design checks: drops exist; classic has dropped rows inside kept individuals
for (nm in names(res)) {
  S <- res[[nm]]$S
  cat(sprintf("%-14s n=%5d N*=%5d dropped=%4d\n", nm, S$n, S$Nf, S$Nf - S$n))
  if (nm != "network2") stopifnot(S$n < S$Nf)
}
S <- res$classic$S
stopifnot(any(!S$keep & S$fe_all[[1]] %in% S$fes[[1]]))
cat("classic: dropped rows inside kept individuals:", sum(!S$keep & S$fe_all[[1]] %in% S$fes[[1]]), "\n")
# network2: L = 2 equals L = 0 in alpaca
nb <- res$network2$json$bc
chk("network2_L2_eq_L0", rel(nb[[2]]$coef, nb[[1]]$coef), 0)

# no-drop control: on the retained classic sample rows = "all" equals rows = "retained"
nd <- S$full[S$keep, .(i, t, y, x1, x2, d)]
cnd <- modifyList(CASES[[1]], list(name = "nodrop", Ls = 1L, which = FALSE))
rnd <- run_case(cnd, nd)
stopifnot(rnd$S$n == rnd$S$Nf)
A <- rnd$json$apes
for (j in seq(1L, length(A), by = 2L)) {
  stopifnot(A[[j]]$rows == "retained", A[[j + 1L]]$rows == "all")
  chk("nodrop_all_vs_retained", rel(A[[j + 1L]]$vcov, A[[j]]$vcov), 1e-12)
}

json <- to_json(list(cases = unname(lapply(res, function(r) r$json))))
writeLines(json, file.path(dir, "alpaca.json"))
back <- jsonlite::fromJSON(file.path(dir, "alpaca.json"), simplifyVector = FALSE)$cases
stopifnot(is.null(names(back)), identical(vapply(back, `[[`, "", "name"), names(res)),
          identical(unlist(back[[1]]$feglm$coef), as.vector(res$classic$json$feglm$coef)))

cat("\ninternal check maxima:\n")
for (k in names(CHK)) cat(sprintf("  %-40s %.3g\n", k, CHK[[k]]))
files <- c("a_classic.csv", "a_dynamic.csv", "a_dynamic_unbal.csv", "a_network.csv", "alpaca.json")
sz <- file.size(file.path(dir, files)); print(data.frame(file = files, bytes = sz)); cat("total", sum(sz), "\n")
cat("elapsed", proc.time()[["elapsed"]] - t0, "s\n")
