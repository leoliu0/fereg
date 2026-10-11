#!/usr/bin/env Rscript
# fixest benchmark runner (plan Phase 8.3).
# Usage: Rscript run_fixest.R data=<csv> fml=<formula> method=<feols|fepois|feglm_logit>
#        vcov=<iid|hetero|cluster:<var>> threads=<T> reps=<timed runs> [fixef_iter=<n>]
# Prints one `RESULT {json}` line per run (burn-in first). Only the estimation
# call is timed, with system.time()[["elapsed"]], on data already in memory.
suppressPackageStartupMessages({
  library(data.table)
  library(fixest)
})

kv <- commandArgs(trailingOnly = TRUE)
args <- setNames(sub("^[^=]*=", "", kv), sub("=.*$", "", kv))
threads <- as.integer(args[["threads"]])
reps <- as.integer(args[["reps"]])
setFixest_nthreads(threads)
setDTthreads(threads)

vcov_arg <- args[["vcov"]]
if (startsWith(vcov_arg, "cluster:")) {
  vcov_arg <- as.formula(paste0("~", sub("^cluster:", "", vcov_arg)))
}
fml <- as.formula(args[["fml"]])
method <- args[["method"]]
extra <- list()
if (!is.na(args["fixef_iter"])) extra$fixef.iter <- as.integer(args[["fixef_iter"]])

df <- fread(args[["data"]], nThread = threads, showProgress = FALSE)

estimate <- function() {
  common <- c(list(fml = fml, data = df, vcov = vcov_arg, nthreads = threads), extra)
  switch(method,
    feols = do.call(feols, common),
    fepois = do.call(fepois, common),
    feglm_logit = do.call(feglm, c(common, list(family = "logit"))),
    stop("unknown method ", method)
  )
}

num <- function(x) if (is.na(x) || !is.finite(x)) "null" else sprintf("%.17g", x)
json_coefs <- function(b) {
  b <- b[!is.na(b)]
  paste0("{", paste0('"', sub("^fit_", "", names(b)), '":', vapply(b, num, ""), collapse = ","), "}")
}
esc <- function(s) gsub('(["\\\\])', "\\\\\\1", gsub("[\r\n\t]", " ", s))

nrun <- 1 + reps
i <- 0
while (i < nrun) {
  phase <- if (i == 0) "burnin" else "timed"
  fit <- NULL
  err <- NULL
  t <- tryCatch(
    unname(system.time(suppressMessages(fit <- estimate()))[["elapsed"]]),
    error = function(e) { err <<- conditionMessage(e); NA_real_ }
  )
  if (!is.null(err)) {
    cat(sprintf('RESULT {"run":"%s","idx":%d,"ok":false,"error":"%s"}\n', phase, i, esc(err)))
    break
  }
  cat(sprintf('RESULT {"run":"%s","idx":%d,"ok":true,"time":%s,"n_obs":%d,"coefs":%s}\n',
              phase, i, num(t), nobs(fit), json_coefs(coef(fit))))
  if (i == 0 && t > 60) nrun <- 2
  i <- i + 1
}
