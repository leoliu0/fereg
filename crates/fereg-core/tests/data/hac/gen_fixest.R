# fixest 0.14.2 / sandwich 3.1.3 references for hac.rs and conley.rs.
# Run (uts): R_LIBS=~/fereg-dev/rlib Rscript gen_fixest.R   (after gen_data.py)
# Writes fixest_ref.csv (case,name,i,j,value), fixest_bw_scores.csv, fixest_cutoff_pts.csv.
suppressMessages(library(fixest))
stopifnot(packageVersion("fixest") == "0.14.2", packageVersion("sandwich") == "3.1.3")
setFixest_nthreads(1)
options(digits = 17)

here = dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
setwd(here)

out = list()
put = function(case, name, value, i = 0, j = 0) {
  out[[length(out) + 1]] <<- data.frame(case = case, name = name, i = i, j = j,
                                         value = sprintf("%.17g", value))
}
put_mat = function(case, name, m) {
  for (i in seq_len(nrow(m))) for (j in seq_len(ncol(m))) put(case, name, m[i, j], i - 1, j - 1)
}
lag_of = function(v) {
  info = attr(v, "vcov_type")
  as.numeric(sub(".*L=([0-9]+).*", "\\1", info))
}

# write.csv keeps 15 significant digits; the Rust tests need the exact doubles.
write_full = function(d, file) {
  for (c in names(d)) if (is.double(d[[c]])) d[[c]] = ifelse(is.na(d[[c]]), "", sprintf("%.17g", d[[c]]))
  write.csv(d, file, row.names = FALSE, quote = FALSE)
}
panel = read.csv("panel.csv")
ts = read.csv("ts.csv")
geo = read.csv("geo.csv")

run = function(case, data, vcov, weights = NULL, kind = "lag") {
  est = if (is.null(weights)) feols(y ~ x1 + x2, data, vcov = "iid") else feols(y ~ x1 + x2, data, weights = weights, vcov = "iid")
  v = vcov(est, vcov = vcov, vcov_fix = FALSE)
  put_mat(case, "V", unclass(v)[, , drop = FALSE])
  put_mat(case, "b", matrix(coef(est), 1))
  put(case, "n", nobs(est))
  put(case, "K", est$nparams)
  if (kind == "lag") {
    put(case, "lag", lag_of(v))
    put(case, "T", attr(v, "G"))
  }
  if (kind == "conley") {
    info = attr(v, "vcov_type")
    put(case, "cutoff", as.numeric(sub(".*\\(([0-9.]+)km\\).*", "\\1", info)))
  }
}

run("nw_panel_L2", panel, NW(2) ~ id + t)
run("nw_panel_def", panel, NW ~ id + t)
run("nw_panel_L20", panel, NW(20) ~ id + t)
run("nw_panel_w_L3", panel, NW(3) ~ id + t, weights = ~w)
run("dk_L2", panel, DK(2) ~ t)
run("dk_def", panel, DK ~ t)
run("dk_L11", panel, DK(11) ~ t)
run("nw_ts_def", ts, NW ~ t)
run("nw_ts_L4", ts, NW(4) ~ t)
run("nw_ts_w_def", ts, NW ~ t, weights = ~w)
run("conley_100", geo, conley(100) ~ lat + lon, kind = "conley")
run("conley_sph_150", geo, conley(150, distance = "spherical") ~ lat + lon, kind = "conley")
run("conley_pix", geo, conley(100, pixel = 30) ~ lat + lon, kind = "conley")
run("conley_pix_sph", geo, conley(200, pixel = 50, distance = "spherical") ~ lat + lon, kind = "conley")
run("conley_def", geo, conley ~ lat + lon, kind = "conley")
run("conley_shift", geo, conley(100) ~ lat90 + lon360, kind = "conley")

# sandwich::bwNeweyWest exactly as vcov_newey_west_internal calls it (time series, no lag given).
set.seed(7)
bw_scores = list()
specs = list(c(40, 1), c(80, 2), c(150, 3), c(300, 2), c(600, 4), c(1200, 1), c(97, 5), c(2500, 3),
             c(500, 2), c(64, 3), c(1000, 6), c(333, 1))
for (s in seq_along(specs)) {
  n = specs[[s]][1]; k = specs[[s]][2]
  rho = runif(k, -0.6, 0.9)
  S = matrix(0, n, k)
  e = matrix(rnorm(n * k), n, k) %*% matrix(runif(k * k, -0.5, 1), k, k)
  S[1, ] = e[1, ]
  for (t in 2:n) S[t, ] = rho * S[t - 1, ] + e[t, ]
  S = S * runif(1, 0.01, 100)
  colnames(S) = paste0("x", seq_len(k))
  raw = sandwich::bwNeweyWest(structure(list(scores = S), class = "fixest"), weights = 1)
  case = sprintf("bw_%02d", s)
  put(case, "raw", raw)
  put(case, "lag", floor(raw))
  bw_scores[[s]] = data.frame(case = case, row = seq_len(n) - 1, S)
}
all_cols = unique(unlist(lapply(bw_scores, names)))
bw_scores = do.call(rbind, lapply(bw_scores, function(d) { for (c in setdiff(all_cols, names(d))) d[[c]] = NA; d[all_cols] }))
write_full(bw_scores, "fixest_bw_scores.csv")

# cutoff_deduce on the shifted coordinates of the Conley cases and on synthetic point sets.
pts = list(
  data.frame(case = "cut_geo", lat = geo$lat, lon = geo$lon),
  data.frame(case = "cut_world", lat = runif(300, -60, 70), lon = runif(300, -180, 180)),
  data.frame(case = "cut_city", lat = 48.85 + runif(200, 0, 0.05), lon = 2.35 + runif(200, 0, 0.05)),
  data.frame(case = "cut_region", lat = 40 + runif(150, 0, 2), lon = -3 + runif(150, 0, 2)),
  data.frame(case = "cut_dups", lat = round(40 + runif(120, 0, 3), 1), lon = round(10 + runif(120, 0, 3), 1))
)
for (p in pts) put(p$case[1], "cutoff", fixest:::cutoff_deduce(p$lat, p$lon))
write_full(do.call(rbind, pts), "fixest_cutoff_pts.csv")

res = do.call(rbind, out)
write.csv(res, "fixest_ref.csv", row.names = FALSE, quote = FALSE)
cat("fixest_ref.csv:", nrow(res), "rows\n")
