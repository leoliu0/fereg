# Reference fixtures for the fixed-effects logit bias correction / APEs / jackknife tests of
# crates/fereg-core (part 2 of 2): independent computations + assembly of logitfe.json.
#
# Run from this directory (crates/fereg-core/tests/data/bias), after gen_logitfe.do:
#     FEREG_REF_RLIB=/path/to/rlib Rscript gen_logitfe_indep.R
# FEREG_REF_RLIB (optional, ':'-separated) is prepended to .libPaths(). Needs fixest 0.14.2 and
# data.table (fixest is only called as fixest::).
#
# Reads l_static.csv, l_dynamic.csv, l_unbal.csv, l_trade.csv (header row, comma separated,
# numeric only, NA = empty field) and logitfe_stata.csv (live logitfe 3.0.3 numbers written by
# gen_logitfe.do); writes logitfe.json.
#
# logitfe.json = {"generator", "cases": [...], "validation": {...}}. Each case: name, data (path
# relative to tests/data), y, x (coefficient order), fe (absorbed FE columns), panel, time, method
# ("none" | "analytical" | "jackknife"), lags, ibias, tbias, variant (jackknife variant or null),
# normalization ("logitfe" for analytical cases), n (estimation rows), n_full (N*: rows after NA
# removal, before the FE drops), binary (per x), expect {coef, ape, coef_uncorr, ape_uncorr},
# source ("logitfe" = live logitfe 3.0.3 output | "indep" = computed here), and optionally
# logitfe_coef / logitfe_ape (logitfe's own, buggy, numbers for an indep case), ape_vcov (array of
# {adj, population, weak_exo, rows, vcov}; vcov row-major p x p), multiple / split / partitions.
#
# Definitions (design bc-design.md):
# * Sample: rows with no NA in y, x, FE columns = N* (n_full); then FE groups with constant y are
#   dropped iteratively (fereg). logitfe drops once (individuals, then periods); every sample used
#   here is asserted to be identical under both rules.
# * Fits: fixest::feglm.fit logit (glm.tol 1e-12, fixef.tol 1e-11) followed by two exact profile
#   Newton steps (weighted alternating projections at 1e-15, exact FE re-solve) -> the MLE.
# * APEs (design s5): continuous beta_j mu'(eta), binary Lambda(eta1) - Lambda(eta0); sum over the
#   estimation rows divided by N* (binary iff all values in {0,1} on the estimation sample).
# * Analytical correction, logitfe normalization (design s3.2, logitfe spec s4a/s4c) with the
#   B1 fix (psi_{i,t-j}); T_i, TN_ij = #{t: (i,t), (i,t-j) in S} (calendar lags), casewise mean
#   over individuals with TN_ij > 0 for all j <= L, D / ng; FEs re-solved exactly at beta_c.
#   With b1bug = TRUE the same code reproduces logitfe's psi of the previous data row (validation).
# * Jackknife (design s7): subpanels are masks over the pre-drop sample (N* rows); individuals
#   1..N and periods 1..T indexed over that sample in ascending id / t order; halves h(m): first =
#   index <= ceil(m/2), second = index > m/2 (m even) or >= ceil(m/2) (m odd); PH = per-individual
#   time halves by position (rows of i in period order); every subpanel refitted with the full
#   sample rules, its APE divided by its own pre-drop row count.
# * multiple(R): partitions[r] = {individual: new 0-based index of each 0-based old individual
#   index, period: same for periods or null (individuals only)}; halves are taken on the new
#   1-based index (new0 + 1); PH positions follow the new period order; per-r combination =
#   the non-multiple both-bias formula (ss2: 3b - mean(IH) - mean(PH); ss1: 2b - mean(Q)); final
#   = mean over r.
# * APE variance (design s6, alpaca's formulas) at the uncorrected fit; rows "all": sums over all
#   N* rows (dropped rows: Delta = 0, D = -delta/N*, own FE levels); weak exogeneity
#   C = sum_i sum_{t<s} D_{i[t]} Gamma_{i[s]}' over individuals, rows in ascending t (incl.
#   dropped rows, Gamma = 0 there); V = Gamma'Gamma + adj [S(id) + S(t) - D'D] (+ adj (C + C')).
#   rows "retained" = alpaca's rule (same formulas over the estimation rows only); this machinery
#   reproduces alpaca 0.3.5 getAPEs() on these data to ~1e-11 (checked when writing this script).
#   The adj = 1 default and population(M) -> adj = (M - N*)/(M - 1) are Stata's conventions.

lp <- Sys.getenv("FEREG_REF_RLIB")
if (nzchar(lp)) .libPaths(c(strsplit(lp, ":")[[1]], .libPaths()))
suppressPackageStartupMessages(library(data.table))
stopifnot(packageVersion("fixest") == "0.14.2")
setDTthreads(1L)
t_start <- proc.time()[["elapsed"]]

rd <- function(f) fread(f, na.strings = c("", "NA"))
DATA <- list(static = rd("l_static.csv"), dynamic = rd("l_dynamic.csv"),
             unbal = rd("l_unbal.csv"), trade = rd("l_trade.csv"))
for (k in names(DATA)) DATA[[k]][, rown := .I]          # row number in the data file
LF <- fread("logitfe_stata.csv")

# ---------------------------------------------------------------------------------------------
# basic pieces
mu.eta <- function(e) { m <- plogis(e); m * (1 - m) }
d2 <- function(e) mu.eta(e) * (1 - 2 * plogis(e))
d3 <- function(e) { f <- mu.eta(e); f * ((1 - 2 * plogis(e))^2 - 2 * f) }
codes <- function(v) match(v, sort(unique(v)))

demean <- function(X, w, fes, tol = 1e-15) {               # M_w X (weighted alternating projections)
  X <- as.matrix(X)
  if (length(fes) == 0L) return(X)
  sw <- lapply(fes, function(f) as.vector(rowsum(w, f, reorder = TRUE)))
  for (it in 1:100000) {
    X0 <- X
    for (k in seq_along(fes)) X <- X - (rowsum(w * X, fes[[k]], reorder = TRUE) / sw[[k]])[fes[[k]], , drop = FALSE]
    if (length(fes) == 1L || max(abs(X - X0)) < tol * (1 + max(abs(X)))) break
  }
  X
}

# FE-only ML given an offset (IRLS on the FE, warm start fe0); returns eta
fe_resolve <- function(y, off, fes, fe0) {
  fe <- fe0
  for (it in 1:200) {
    eta <- off + fe; mu <- plogis(eta); w <- mu * (1 - mu)
    zz <- (y - mu) / w + fe
    fe.new <- zz - demean(zz, w, fes)[, 1]
    if (max(abs(fe.new - fe)) < 1e-13) return(off + fe.new)
    fe <- fe.new
  }
  stop("fe_resolve did not converge")
}

# constant-y drops: iterated (fereg) and logitfe's one pass (individual FE first, then time FE)
drop_rows <- function(y, fes, iterate) {
  keep <- rep(TRUE, length(y))
  repeat {
    changed <- FALSE
    for (f in fes) {
      s <- ave(y[keep], f[keep], FUN = function(v) length(unique(v)) > 1)
      bad <- which(keep)[s == 0]
      if (length(bad)) { keep[bad] <- FALSE; changed <- TRUE }
    }
    if (!iterate || !changed) break
  }
  keep
}

# exact logit FE fit on the retained rows: fixest + one profile Newton step
fit_exact <- function(y, X, fes) {
  fdf <- as.data.frame(lapply(fes, identity)); names(fdf) <- paste0("f", seq_along(fes))
  m <- fixest::feglm.fit(y, X, fdf, family = "logit", glm.tol = 1e-12, fixef.tol = 1e-11,
                         glm.iter = 100, nthreads = 1, notes = FALSE, warn = FALSE)
  stopifnot(isTRUE(m$convStatus), m$nobs == length(y), !anyNA(coef(m)), length(coef(m)) == ncol(X))
  b <- unname(coef(m)); eta <- unname(m$linear.predictors)
  for (step in 1:2) {
    w <- mu.eta(eta); MX <- demean(X, w, fes)
    g <- colSums(MX * (y - plogis(eta)))
    db <- solve(crossprod(MX * sqrt(w)), g)
    b <- b + db
    eta <- fe_resolve(y, as.vector(X %*% b), fes, eta - as.vector(X %*% (b - db)))
  }
  list(b = b, eta = eta)
}

is_binary <- function(X) apply(X, 2, function(x) all(x %in% c(0, 1)))

ape_point <- function(X, b, eta, Nstar) {                   # design s5
  bin <- is_binary(X); p <- ncol(X); D <- matrix(0, nrow(X), p)
  for (j in seq_len(p)) {
    if (bin[j]) { e0 <- eta - X[, j] * b[j]; D[, j] <- plogis(e0 + b[j]) - plogis(e0) }
    else D[, j] <- b[j] * mu.eta(eta)
  }
  list(delta = colSums(D) / Nstar, Delta = D, binary = bin)
}

# ---------------------------------------------------------------------------------------------
# model setup: pre-drop sample P, estimation sample S
setup <- function(dname, y, xs, fes, time) {
  dat <- DATA[[dname]]
  ok <- complete.cases(dat[, c(y, xs, fes, time), with = FALSE])
  P <- dat[ok]
  setorderv(P, c("id", time))
  P[, `:=`(ii = codes(id), tt = codes(get(time)))]
  list(dname = dname, y = y, xs = xs, fes = fes, time = time, P = P)
}

# fit on a 0/1 mask over P: returns everything needed downstream
fit_mask <- function(M, mask = rep(TRUE, nrow(M$P))) {
  P <- M$P[mask]
  fP <- lapply(M$fes, function(f) codes(P[[f]]))
  keep <- drop_rows(P[[M$y]], fP, iterate = TRUE)
  ord <- c(intersect("id", M$fes), setdiff(M$fes, "id"))   # logitfe: individuals first
  keep1 <- drop_rows(P[[M$y]], lapply(ord, function(f) codes(P[[f]])), iterate = FALSE)
  if (!identical(keep, keep1)) stop("one-pass and iterated drops differ in ", M$dname)
  S <- P[keep]
  fes <- lapply(M$fes, function(f) codes(S[[f]]))
  X <- as.matrix(S[, M$xs, with = FALSE]); y <- S[[M$y]]
  f <- fit_exact(y, X, fes)
  a <- ape_point(X, f$b, f$eta, nrow(P))
  list(P = P, S = S, keep = keep, X = X, y = y, fes = fes, b = f$b, eta = f$eta, ape = a$delta,
       Delta = a$Delta, binary = a$binary, n = nrow(S), Nstar = nrow(P))
}

# ---------------------------------------------------------------------------------------------
# analytical correction, logitfe normalization (B1 fixed unless b1bug)
an_logitfe <- function(M, F, L, ib, tb, b1bug = FALSE) {
  S <- F$S; X <- F$X; y <- F$y; fes <- F$fes; n <- F$n
  has_i <- "id" %in% M$fes; has_t <- M$time %in% M$fes
  febias <- ib && has_i; tebias <- tb && has_t
  if (!febias) L <- 0L
  ii <- codes(S$id); tt <- codes(S[[M$time]]); ng <- max(ii)
  tval <- S[[M$time]]
  key <- paste(S$id, tval)
  lt <- lapply(seq_len(L), function(j) paste(S$id, tval - j) %in% key)
  TN <- lapply(lt, function(l) as.vector(rowsum(as.numeric(l), ii, reorder = TRUE)))
  Ti <- as.vector(table(ii))
  lagpsi <- function(psi) {
    if (b1bug) {                       # logitfe: psi of the previous data row (if in S), for every j
      pos <- match(S$rown - 1L, S$rown)
      prev <- ifelse(is.na(pos), 0, psi[pos])
      lapply(lt, function(l) ifelse(l, prev, 0))
    } else lapply(seq_len(L), function(j) {
      pos <- match(paste(S$id, tval - j), key)
      ifelse(is.na(pos), 0, psi[pos])
    })
  }
  gs <- function(A, g) rowsum(as.matrix(A), g, reorder = TRUE)
  # coefficients
  eta <- F$eta; w <- mu.eta(eta); mu <- plogis(eta)
  MX <- demean(X, w, fes)
  B <- D <- rep(0, ncol(X))
  if (febias) {
    swi <- as.vector(gs(w, ii))
    Bi <- gs(mu * w * MX, ii) / swi / Ti
    if (L > 0) {
      lp <- lagpsi(y - mu)
      for (j in seq_len(L)) Bi <- Bi - gs(lp[[j]] * lt[[j]] * w * MX, ii) / swi / TN[[j]]
    }
    B <- colMeans(Bi[complete.cases(Bi), , drop = FALSE])
  }
  if (tebias) D <- colMeans(gs(mu * w * MX, tt) / as.vector(gs(w, tt)))
  W <- crossprod(MX * sqrt(w)) / n
  bc <- F$b - solve(W, B + D / ng)
  # FEs at bc
  off <- as.vector(X %*% bc)
  etac <- fe_resolve(y, off, fes, eta - as.vector(X %*% F$b))
  wc <- mu.eta(etac); muc <- plogis(etac); dl <- d2(etac)
  swi <- as.vector(gs(wc, ii)); swt <- as.vector(gs(wc, tt))
  lpc <- if (L > 0) lagpsi(y - muc) else NULL
  apebias <- function(dd, ddd) {
    psi <- dd / wc; Ppsi <- psi - demean(psi, wc, fes)[, 1]; r <- psi - Ppsi
    Bv <- Dv <- 0
    if (febias) {
      Bi <- as.vector(gs(-dl * Ppsi + ddd, ii)) / swi / Ti
      for (j in seq_len(L)) Bi <- Bi + as.vector(gs(2 * lpc[[j]] * lt[[j]] * wc * r, ii)) / swi / TN[[j]]
      Bv <- 0.5 * mean(Bi[is.finite(Bi)])
    }
    if (tebias) Dv <- 0.5 * mean(as.vector(gs(-dl * Ppsi + ddd, tt)) / swt)
    Bv + Dv / ng
  }
  bin <- F$binary
  ape <- numeric(ncol(X))
  cont_scalar <- sum(wc) / F$Nstar - apebias(dl, d3(etac))
  for (j in seq_len(ncol(X))) {
    if (bin[j]) {
      z0 <- etac - bc[j] * X[, j]; z1 <- z0 + bc[j]
      ape[j] <- sum(plogis(z1) - plogis(z0)) / F$Nstar - apebias(mu.eta(z1) - mu.eta(z0), d2(z1) - d2(z0))
    } else ape[j] <- bc[j] * cont_scalar
  }
  list(coef = bc, ape = ape)
}

# ---------------------------------------------------------------------------------------------
# APE variance (alpaca formulas), uncorrected fit, two-way classic; rows = "all" (N* rows, the
# production rule) or "retained" (alpaca's rule: estimation rows only, same D centring)
ape_vcov <- function(M, F, adjs, weak, rows = "all") {
  X <- F$X; y <- F$y; fes <- F$fes; Nf <- F$Nstar; b <- F$b; eta <- F$eta; p <- ncol(X)
  w <- mu.eta(eta); v <- y - plogis(eta); z <- d2(eta)
  MX <- demean(X, w, fes); PX <- X - MX; H <- crossprod(MX * sqrt(w))
  Delta <- F$Delta; Delta1 <- matrix(0, nrow(X), p); J <- matrix(0, p, p)
  for (j in seq_len(p)) {
    if (F$binary[j]) {
      e0 <- eta - X[, j] * b[j]; e1 <- e0 + b[j]
      Delta1[, j] <- mu.eta(e1) - mu.eta(e0)
      J[, j] <- colSums(MX * Delta1[, j]) / Nf
      J[j, j] <- (sum(mu.eta(e1)) - sum(PX[, j] * Delta1[, j])) / Nf
    } else {
      Delta1[, j] <- b[j] * z
      J[, j] <- colSums(MX * Delta1[, j]) / Nf; J[j, j] <- J[j, j] + sum(w) / Nf
    }
  }
  delta <- colSums(Delta) / Nf
  Psi <- -Delta1 / w; PPsi <- Psi - demean(Psi, w, fes)
  Gam <- (MX %*% solve(H / Nf, J) - PPsi) * v / Nf
  V0 <- crossprod(Gam)
  # all N* rows: P in (id, t) order; retained rows carry their D and Gamma
  P <- F$P; keep <- F$keep
  if (rows == "retained") { P <- F$S; keep <- rep(TRUE, F$n) }
  NP <- nrow(P)
  Dall <- matrix(rep(-delta / Nf, each = NP), NP, p)
  Dall[keep, ] <- sweep(Delta, 2, delta) / Nf
  Gall <- matrix(0, NP, p); Gall[keep, ] <- Gam
  Sg <- function(A, g) crossprod(rowsum(A, g))
  gi <- codes(P$id); gt <- codes(P[[M$time]])
  samp <- Sg(Dall, gi) + Sg(Dall, gt) - crossprod(Dall)
  C <- matrix(0, p, p)
  if (weak) {                                              # P is sorted by (id, t)
    for (g in split(seq_len(NP), gi)) if (length(g) > 1)
      for (k in 1:(length(g) - 1)) C <- C + tcrossprod(Dall[g[k], ], colSums(Gall[g[(k + 1):length(g)], , drop = FALSE]))
  }
  lapply(adjs, function(a) {
    V <- V0 + a$adj * samp
    if (weak) V <- V + a$adj * (C + t(C))
    list(adj = a$adj, population = a$population, weak_exo = weak, rows = rows, vcov = arr(as.vector(t(V))))
  })
}

# ---------------------------------------------------------------------------------------------
# jackknife engine
halves <- function(idx, m) {
  h <- ceiling(m / 2)
  list(idx <= h, if (m %% 2 == 0) idx > m / 2 else idx >= h)
}
halves_v <- function(idx, m) {                             # vectorised over rows (per-row m)
  h <- ceiling(m / 2)
  list(idx <= h, ifelse(m %% 2 == 0, idx > m / 2, idx >= h))
}
JK <- function(M, F0) {
  P <- M$P; N <- max(P$ii); TT <- max(P$tt)
  cache <- new.env()
  sub <- function(tag, mask) {
    if (is.null(cache[[tag]])) { f <- fit_mask(M, mask); cache[[tag]] <- c(f$b, f$ape) }
    cache[[tag]]
  }
  mean_of <- function(L) Reduce(`+`, L) / length(L)
  IH <- function(pi = NULL) {
    ni <- if (is.null(pi)) P$ii else pi[P$ii]
    h <- halves(ni, N); tg <- if (is.null(pi)) "" else paste(pi, collapse = ".")
    list(sub(paste0("IH1", tg), h[[1]]), sub(paste0("IH2", tg), h[[2]]))
  }
  TH <- function(pt = NULL) {
    nt <- if (is.null(pt)) P$tt else pt[P$tt]
    h <- halves(nt, TT); tg <- if (is.null(pt)) "" else paste(pt, collapse = ".")
    list(sub(paste0("TH1", tg), h[[1]]), sub(paste0("TH2", tg), h[[2]]))
  }
  PH <- function(pt = NULL) {
    nt <- if (is.null(pt)) P$tt else pt[P$tt]
    pos <- ave(nt, P$ii, FUN = rank); Ti <- ave(nt, P$ii, FUN = length)
    h <- halves_v(pos, Ti); tg <- if (is.null(pt)) "" else paste(pt, collapse = ".")
    list(sub(paste0("PH1", tg), h[[1]]), sub(paste0("PH2", tg), h[[2]]))
  }
  Q <- function(pi = NULL, pt = NULL) {
    ni <- if (is.null(pi)) P$ii else pi[P$ii]; nt <- if (is.null(pt)) P$tt else pt[P$tt]
    hi <- halves(ni, N); ht <- halves(nt, TT)
    tg <- paste(paste(pi, collapse = "."), paste(pt, collapse = "."))
    list(sub(paste0("Q11", tg), hi[[1]] & ht[[1]]), sub(paste0("Q12", tg), hi[[1]] & ht[[2]]),
         sub(paste0("Q21", tg), hi[[2]] & ht[[1]]), sub(paste0("Q22", tg), hi[[2]] & ht[[2]]))
  }
  DI <- function() lapply(seq_len(N), function(i) sub(paste0("-i", i), P$ii != i))
  DT <- function() lapply(seq_len(TT), function(t) sub(paste0("-t", t), P$tt != t))
  DD <- function() {                                       # double: ids present as both id and time
    J <- intersect(unique(P$id), unique(P[[M$time]]))
    lapply(sort(J), function(c) sub(paste0("-c", c), P$id != c & P[[M$time]] != c))
  }
  b0 <- c(F0$b, F0$ape)
  list(N = N, TT = TT, b0 = b0, IH = IH, TH = TH, PH = PH, Q = Q, DI = DI, DT = DT, DD = DD,
       mean_of = mean_of, nfits = function() length(ls(cache)))
}
# design s7 combinations (and logitfe's own, for validation)
jk_combo <- function(E, variant, bias, form = c("design", "logitfe")) {
  form <- match.arg(form); b <- E$b0; m <- E$mean_of; N <- E$N; TT <- E$TT
  di <- function() m(E$DI()); dt <- function() m(E$DT())
  key <- paste(variant, bias)
  if (form == "logitfe") key <- switch(key,
    "js both" = return((N + 1) * b - (N - 1) * di() - m(E$IH())),
    "js time" = "ss1 time", "js individual" = "jj time",
    "sj both" = return((TT + 1) * b - (TT - 1) * dt() - m(E$TH())),
    "sj time" = "jj individual", "sj individual" = "ss1 individual",
    "jj time" = return(TT * b - (TT - 1) * dt()), "jj individual" = return(N * b - (N - 1) * di()),
    "double time" = return(TT * b - (TT - 1) * dt()), "double individual" = return(N * b - (N - 1) * di()),
    key)
  switch(key,
    "ss2 both" = 3 * b - m(E$IH()) - m(E$PH()),
    "ss2 individual" = 2 * b - m(E$PH()),
    "ss2 time" = 2 * b - m(E$IH()),
    "ss1 both" = 2 * b - m(E$Q()),
    "ss1 individual" = 2 * b - m(E$TH()),
    "ss1 time" = 2 * b - m(E$IH()),
    "js both" = (N + 1) * b - (N - 1) * di() - m(E$TH()),
    "js individual" = 2 * b - m(E$TH()),
    "js time" = N * b - (N - 1) * di(),
    "sj both" = (TT + 1) * b - (TT - 1) * dt() - m(E$IH()),
    "sj individual" = TT * b - (TT - 1) * dt(),
    "sj time" = 2 * b - m(E$IH()),
    "jj both" = (N + TT - 1) * b - (N - 1) * di() - (TT - 1) * dt(),
    "jj individual" = TT * b - (TT - 1) * dt(),
    "jj time" = N * b - (N - 1) * di(),
    "double both" = { L <- E$DD(); J <- length(L); J * b - (J - 1) * m(L) },
    "double individual" = TT * b - (TT - 1) * dt(),
    "double time" = N * b - (N - 1) * di(),
    stop("unknown jackknife ", key))
}
jk_multiple <- function(E, variant, parts) {
  b <- E$b0; m <- E$mean_of
  per_r <- lapply(parts, function(pr) {
    pi <- pr$individual + 1L; pt <- if (is.null(pr$period)) NULL else pr$period + 1L
    if (variant == "ss2") 3 * b - m(E$IH(pi)) - m(E$PH(pt)) else 2 * b - m(E$Q(pi, pt))
  })
  m(per_r)
}

# ---------------------------------------------------------------------------------------------
# JSON writer (numbers %.17g, NA -> null, arr() forces an array)
arr <- function(x) structure(x, class = c("jarr", class(x)))
jnum <- function(x) ifelse(is.na(x), "null", sprintf("%.17g", as.double(x)))
tojson <- function(x, ind = "") {
  ind2 <- paste0(ind, "  ")
  if (is.null(x)) return("null")
  if (is.list(x) && !inherits(x, "jarr")) {
    if (!is.null(names(x))) {
      items <- vapply(names(x), function(k) paste0(ind2, '"', k, '": ', tojson(x[[k]], ind2)), "")
      return(paste0("{\n", paste(items, collapse = ",\n"), "\n", ind, "}"))
    }
    items <- vapply(x, function(e) paste0(ind2, tojson(e, ind2)), "")
    return(paste0("[\n", paste(items, collapse = ",\n"), "\n", ind, "]"))
  }
  is_arr <- inherits(x, "jarr"); x <- unclass(x)
  s <- if (is.character(x)) paste0('"', x, '"') else if (is.logical(x)) ifelse(is.na(x), "null", ifelse(x, "true", "false")) else jnum(x)
  if (is_arr || length(x) != 1L) paste0("[", paste(s, collapse = ", "), "]") else s
}

# ---------------------------------------------------------------------------------------------
# models
MOD <- list(
  s  = list(d = "static",  y = "y", x = c("x1", "x2", "x3", "d"), time = "t"),
  d  = list(d = "dynamic", y = "y", x = c("ylag", "x1", "d"), time = "t"),
  u  = list(d = "unbal",   y = "y", x = c("x1", "x2", "x3", "d"), time = "t"),
  ud = list(d = "unbal",   y = "y", x = c("ylag", "x1", "x2", "x3", "d"), time = "t"),
  tr = list(d = "trade",   y = "y", x = c("x", "b1", "b2"), time = "jd"))
FEW <- list(two = function(m) c("id", m$time), ionly = function(m) "id", tonly = function(m) m$time)
models <- list(); fits <- list()
getfit <- function(mk, fw) {
  key <- paste(mk, fw)
  if (is.null(fits[[key]])) {
    m <- MOD[[mk]]
    models[[key]] <<- setup(m$d, m$y, m$x, FEW[[fw]](m), m$time)
    fits[[key]] <<- fit_mask(models[[key]])
  }
  list(M = models[[key]], F = fits[[key]])
}
lf <- function(run, what) {                                 # logitfe number (coef order = x)
  r <- LF[which(LF$tag == run)]
  if (!nrow(r)) stop("missing logitfe run ", run)
  r[[what]]
}
lfN <- function(run) lf(run, "N")[1]
maxrel <- function(a, b) { stopifnot(length(a) == length(b)); max(abs(a - b) / abs(b)) }

cases <- list(); VAL <- list()
add_case <- function(name, mk, fw, method, lags = 0L, ib = FALSE, tb = FALSE, variant = NULL,
                     expect, source, extra = list()) {
  g <- getfit(mk, fw); m <- MOD[[mk]]; F <- g$F
  base <- list(name = name, data = paste0("bias/l_", m$d, ".csv"), y = m$y, x = arr(m$x),
               fe = arr(FEW[[fw]](m)), panel = "id", time = m$time, method = method,
               lags = as.integer(lags), ibias = ib, tbias = tb, variant = variant,
               normalization = if (method == "analytical") "logitfe" else NULL,
               n = F$n, n_full = F$Nstar, binary = arr(F$binary),
               expect = lapply(expect, arr), source = source)
  base <- base[!vapply(base, is.null, TRUE) | names(base) == "variant"]
  cases[[length(cases) + 1L]] <<- c(base, extra)
}
uncorr_lf <- function(tag) list(coef_uncorr = lf(tag, "b"), ape_uncorr = lf(tag, "ape"))
uncorr_r <- function(F) list(coef_uncorr = F$b, ape_uncorr = F$ape)

# --- live logitfe: uncorrected and analytical ------------------------------------------------
noc_of <- c(two = "noc", ionly = "noc_ionly", tonly = "noc_tonly")
LIVE_AN <- list(
  list("s_noc", "s", "two", "none"), list("s_noc_ionly", "s", "ionly", "none"),
  list("s_noc_tonly", "s", "tonly", "none"),
  list("s_an_l0", "s", "two", "analytical", 0, TRUE, TRUE), list("s_an_l1", "s", "two", "analytical", 1, TRUE, TRUE),
  list("s_an_ibno", "s", "two", "analytical", 0, FALSE, TRUE), list("s_an_tbno", "s", "two", "analytical", 0, TRUE, FALSE),
  list("s_an_ionly", "s", "ionly", "analytical", 0, TRUE, FALSE), list("s_an_ionly_l1", "s", "ionly", "analytical", 1, TRUE, FALSE),
  list("s_an_tonly", "s", "tonly", "analytical", 0, FALSE, TRUE),
  list("d_noc", "d", "two", "none"), list("d_an_l1", "d", "two", "analytical", 1, TRUE, TRUE),
  list("d_an_l1_ibno", "d", "two", "analytical", 1, FALSE, TRUE), list("d_an_l1_tbno", "d", "two", "analytical", 1, TRUE, FALSE),
  list("u_noc", "u", "two", "none"), list("u_noc_ionly", "u", "ionly", "none"),
  list("u_an_l0", "u", "two", "analytical", 0, TRUE, TRUE), list("u_an_l1", "u", "two", "analytical", 1, TRUE, TRUE),
  list("u_an_ibno", "u", "two", "analytical", 0, FALSE, TRUE), list("u_an_tbno", "u", "two", "analytical", 0, TRUE, FALSE),
  list("u_an_ionly", "u", "ionly", "analytical", 0, TRUE, FALSE),
  list("ud_noc", "ud", "two", "none"), list("ud_an_l1", "ud", "two", "analytical", 1, TRUE, TRUE),
  list("tr_noc", "tr", "two", "none"), list("tr_an", "tr", "two", "analytical", 0, TRUE, TRUE))
val_an <- list()
for (a in LIVE_AN) {
  tag <- a[[1]]; mk <- a[[2]]; fw <- a[[3]]; meth <- a[[4]]
  L <- if (length(a) > 4) a[[5]] else 0; ib <- if (length(a) > 5) a[[6]] else FALSE; tb <- if (length(a) > 6) a[[7]] else FALSE
  g <- getfit(mk, fw)
  stopifnot(g$F$n == lfN(tag))
  utag <- paste0(mk, "_", noc_of[[fw]])
  ex <- c(list(coef = lf(tag, "b"), ape = lf(tag, "ape")), uncorr_lf(utag))
  extra <- list()
  if (tag %in% c("s_noc", "s_an_l0", "u_noc", "u_an_l0")) {
    M2 <- 2 * g$F$Nstar
    pop <- list(adj = (M2 - g$F$Nstar) / (M2 - 1), population = M2)
    extra$ape_vcov <- c(ape_vcov(g$M, g$F, list(list(adj = 1, population = NULL), pop,
                                                 list(adj = 0, population = NULL)), weak = FALSE),
                        ape_vcov(g$M, g$F, list(pop), weak = FALSE, rows = "retained"))
  }
  if (tag == "d_an_l1")
    extra$ape_vcov <- c(ape_vcov(g$M, g$F, list(list(adj = 1, population = NULL)), weak = TRUE),
                        ape_vcov(g$M, g$F, list(list(adj = 1, population = NULL)), weak = FALSE),
                        ape_vcov(g$M, g$F, list(list(adj = 1, population = NULL)), weak = TRUE, rows = "retained"))
  add_case(tag, mk, fw, meth, L, ib, tb, NULL, ex, "logitfe", extra)
  # validation: exact uncorrected fit vs logitfe noc, own logitfe-normalization vs logitfe an
  if (meth == "none") val_an[[tag]] <- list(coef = maxrel(g$F$b, lf(tag, "b")), ape = maxrel(g$F$ape, lf(tag, "ape")))
  else {
    r <- an_logitfe(g$M, g$F, L, ib, tb)
    val_an[[tag]] <- list(coef = maxrel(r$coef, lf(tag, "b")), ape = maxrel(r$ape, lf(tag, "ape")))
  }
}

# --- indep: analytical lags(2) with the B1 fix -----------------------------------------------
for (mk in c("s", "u")) {
  tag <- paste0(mk, "_an_l2"); g <- getfit(mk, "two")
  r <- an_logitfe(g$M, g$F, 2L, TRUE, TRUE)
  rb <- an_logitfe(g$M, g$F, 2L, TRUE, TRUE, b1bug = TRUE)
  val_an[[paste0(tag, "_b1bug")]] <- list(coef = maxrel(rb$coef, lf(tag, "b")), ape = maxrel(rb$ape, lf(tag, "ape")))
  add_case(tag, mk, "two", "analytical", 2L, TRUE, TRUE, NULL,
           c(list(coef = r$coef, ape = r$ape), uncorr_r(g$F)), "indep",
           list(logitfe_coef = arr(lf(tag, "b")), logitfe_ape = arr(lf(tag, "ape"))))
}

# --- jackknife -------------------------------------------------------------------------------
bias_name <- function(ib, tb) if (ib && tb) "both" else if (ib) "individual" else "time"
ENG <- list()
eng <- function(mk) {
  if (is.null(ENG[[mk]])) { g <- getfit(mk, "two"); ENG[[mk]] <<- JK(g$M, g$F) }
  ENG[[mk]]
}
p_of <- function(mk) length(MOD[[mk]]$x)
split_cp <- function(v, p) list(coef = v[seq_len(p)], ape = v[p + seq_len(p)])
val_jk <- list()
JKC <- list(  # name, model, variant, ibias, tbias, source, logitfe tag (if any)
  list("s_jk_ss2", "s", "ss2", TRUE, TRUE, "logitfe", "s_jk_ss2"),
  list("s_jk_ss2_ibno", "s", "ss2", FALSE, TRUE, "logitfe", "s_jk_ss2_ibno"),
  list("s_jk_ss2_tbno", "s", "ss2", TRUE, FALSE, "indep", NA),
  list("s_jk_ss1", "s", "ss1", TRUE, TRUE, "logitfe", "s_jk_ss1"),
  list("s_jk_ss1_ibno", "s", "ss1", FALSE, TRUE, "logitfe", "s_jk_ss1_ibno"),
  list("s_jk_ss1_tbno", "s", "ss1", TRUE, FALSE, "logitfe", "s_jk_ss1_tbno"),
  list("s_jk_js", "s", "js", TRUE, TRUE, "indep", "s_jk_js"),
  list("s_jk_js_ibno", "s", "js", FALSE, TRUE, "indep", "s_jk_js_ibno"),
  list("s_jk_js_tbno", "s", "js", TRUE, FALSE, "indep", "s_jk_jj_tbno"),
  list("s_jk_sj", "s", "sj", TRUE, TRUE, "indep", "s_jk_sj"),
  list("s_jk_sj_ibno", "s", "sj", FALSE, TRUE, "indep", "s_jk_sj_ibno"),
  list("s_jk_sj_tbno", "s", "sj", TRUE, FALSE, "indep", "s_jk_sj_tbno"),
  list("s_jk_jj", "s", "jj", TRUE, TRUE, "logitfe", "s_jk_jj"),
  list("s_jk_jj_ibno", "s", "jj", FALSE, TRUE, "indep", "s_jk_jj_ibno"),
  list("s_jk_jj_tbno", "s", "jj", TRUE, FALSE, "indep", "s_jk_jj_tbno"),
  list("u_jk_ss2", "u", "ss2", TRUE, TRUE, "logitfe", "u_jk_ss2"),
  list("u_jk_ss1", "u", "ss1", TRUE, TRUE, "logitfe", "u_jk_ss1"),
  list("tr_jk_double", "tr", "double", TRUE, TRUE, "logitfe", "tr_jk_double"),
  list("tr_jk_double_ibno", "tr", "double", FALSE, TRUE, "indep", "tr_jk_double_ibno"),
  list("tr_jk_double_tbno", "tr", "double", TRUE, FALSE, "indep", "tr_jk_double_tbno"))
for (a in JKC) {
  name <- a[[1]]; mk <- a[[2]]; variant <- a[[3]]; ib <- a[[4]]; tb <- a[[5]]; src <- a[[6]]; ltag <- a[[7]]
  E <- eng(mk); g <- getfit(mk, "two"); p <- p_of(mk); bn <- bias_name(ib, tb)
  mine <- split_cp(jk_combo(E, variant, bn, "design"), p)
  if (!is.na(ltag)) {
    stopifnot(g$F$n == lfN(ltag))
    # logitfe's own formula on our subpanel fits vs logitfe (js tbias(no) has no own logitfe run:
    # logitfe's js tbias(no) is code-identical to jj tbias(no))
    lv <- split_cp(jk_combo(E, variant, bn, "logitfe"), p)
    val_jk[[name]] <- list(logitfe_tag = ltag, coef = maxrel(lv$coef, lf(ltag, "b")), ape = maxrel(lv$ape, lf(ltag, "ape")))
  }
  ex <- if (src == "logitfe") c(list(coef = lf(ltag, "b"), ape = lf(ltag, "ape")), uncorr_lf(paste0(mk, "_noc")))
        else c(mine, uncorr_r(g$F))
  extra <- if (src == "indep" && !is.na(ltag)) list(logitfe_coef = arr(lf(ltag, "b")), logitfe_ape = arr(lf(ltag, "ape"))) else list()
  add_case(name, mk, "two", "jackknife", 0L, ib, tb, variant, ex, src, extra)
}

# --- jackknife multiple(3), own partitions -----------------------------------------------------
E <- eng("s"); g <- getfit("s", "two"); p <- p_of("s")
set.seed(20261018)
parts_both <- lapply(1:3, function(r) list(individual = arr(sample.int(E$N) - 1L), period = arr(sample.int(E$TT) - 1L)))
parts_ind <- lapply(1:3, function(r) list(individual = arr(sample.int(E$N) - 1L), period = NULL))
MULT <- list(list("s_jk_ss2_mul3", "ss2", parts_both, "both"), list("s_jk_ss1_mul3", "ss1", parts_both, "both"),
             list("s_jk_ss2_mul3_i", "ss2", parts_ind, "individuals"))
for (a in MULT) {
  v <- split_cp(jk_multiple(E, a[[2]], a[[3]]), p)
  add_case(a[[1]], "s", "two", "jackknife", 0L, TRUE, TRUE, a[[2]], c(v, uncorr_r(g$F)), "indep",
           list(multiple = 3L, split = a[[4]], partitions = a[[3]]))
}
VAL$note <- paste("max relative differences vs logitfe 3.0.3: 'analytical' = exact uncorrected fit (none) or",
  "this script's logitfe-normalization correction (an; *_b1bug = with logitfe's lags(2) bug B1);",
  "'jackknife' = logitfe's own combination formula on this script's subpanel fits")
VAL$analytical <- val_an
VAL$jackknife <- val_jk
req <- c("s_jk_ss2", "s_jk_ss1", "s_jk_jj", "s_jk_sj_ibno")
VAL$required_max <- list(cases = arr(req), coef = max(vapply(val_jk[req], `[[`, 0, "coef")),
                         ape = max(vapply(val_jk[req], `[[`, 0, "ape")))
VAL$all_max <- list(coef = max(vapply(c(val_an, val_jk), `[[`, 0, "coef")), ape = max(vapply(c(val_an, val_jk), `[[`, 0, "ape")))
VAL$subpanel_fits <- sum(vapply(ENG, function(e) e$nfits(), 0))

out <- list(
  generator = "gen_logitfe.do (logitfe 3.0.3, Stata) + gen_logitfe_indep.R (fixest 0.14.2)",
  cases = cases, validation = VAL)
writeLines(tojson(out), "logitfe.json")
cat("cases:", length(cases), " subpanel fits:", VAL$subpanel_fits,
    " seconds:", round(proc.time()[["elapsed"]] - t_start, 1), "\n")
for (k in names(val_an)) cat(sprintf("an  %-18s coef %.2e  ape %.2e\n", k, val_an[[k]]$coef, val_an[[k]]$ape))
for (k in names(val_jk)) cat(sprintf("jk  %-18s coef %.2e  ape %.2e\n", k, val_jk[[k]]$coef, val_jk[[k]]$ape))
cat("required max:", VAL$required_max$coef, VAL$required_max$ape, " all max:", VAL$all_max$coef, VAL$all_max$ape, "\n")
