% Generated manually matching R/wildboot.R
\name{boottest}
\alias{boottest}
\alias{boottest.fereg}
\alias{boottest.fereg_multi}
\alias{print.fereg_boottest}
\alias{summary.fereg_boottest}
\alias{confint.fereg_boottest}
\alias{pvalue.fereg_boottest}
\alias{tstat.fereg_boottest}
\title{Wild bootstrap inference}
\usage{
boottest(object, ...)

\method{boottest}{fereg_multi}(object, ...)

\method{boottest}{fereg}(
  object,
  param = NULL,
  B = 999L,
  clustid = NULL,
  bootcluster = NULL,
  R = NULL,
  r = NULL,
  type = c("rademacher", "webb", "mammen", "normal"),
  bootstrap_type = c("11", "13", "31", "33", "21"),
  method = c("auto", "wildcluster", "score", "wre", "hetero"),
  impose_null = TRUE,
  p_val_type = c("two-tailed", "equal-tailed", ">", "<"),
  conf_int = TRUE,
  sign_level = 0.05,
  seed = NULL,
  ssc = NULL,
  tol = 1e-6,
  maxiter = 50L,
  nthreads = NULL,
  ...
)

\method{print}{fereg_boottest}(x, digits = 4L, ...)

\method{summary}{fereg_boottest}(object, digits = 4L, ...)

\method{confint}{fereg_boottest}(object, parm, level = NULL, ...)

\method{pvalue}{fereg_boottest}(object, ...)

\method{tstat}{fereg_boottest}(object, ...)
}
\arguments{
\item{object}{For \code{boottest()}, a single \code{fereg} estimation returned by \code{\link[=feols]{feols()}}, \code{\link[=fepois]{fepois()}} or \code{\link[=feglm]{feglm()}} (with its estimation data still available). For \code{summary()}, \code{confint()}, \code{pvalue()} and \code{tstat()}, an object of class \code{"fereg_boottest"}.}

\item{x}{An object of class \code{"fereg_boottest"}.}

\item{param}{Character vector naming the coefficient(s) to test (e.g. \code{"x1"} or \code{c("x1", "x2")}), mutually exclusive with \code{R}. For IV models, endogenous regressors are named \code{"fit_<var>"}, as in \code{coef(object)}.}

\item{B}{Number of bootstrap replications (positive integer, default \code{999L}). For clustered Rademacher (\code{G <= 20}, \code{2^G <= B}) or Webb (\code{G <= 8}, \code{6^G <= B}), full enumeration is used automatically.}

\item{clustid}{Optional clustering specification overriding \code{object}'s variance: a one-sided formula (\code{~firm}, \code{~firm + year}, \code{~firm^year}), character vector (\code{"firm"}, \code{c("firm", "year")}), or \code{character(0)} for unclustered inference. Defaults to \code{object}'s current clustering.}

\item{bootcluster}{Optional bootstrap subcluster dimension overriding the cluster dimension used to generate the wild bootstrap weights (same forms as \code{clustid}).}

\item{R}{Optional numeric constraint matrix (\code{q x k}) or vector (\code{k}) for the linear hypothesis \code{H0: R * beta = r}, columns ordered as \code{names(coef(object))}, mutually exclusive with \code{param}.}

\item{r}{Optional numeric right-hand side vector of length \code{q} (default zero vector).}

\item{type}{Auxiliary weight distribution: \code{"rademacher"} (default), \code{"webb"}, \code{"mammen"} or \code{"normal"}.}

\item{bootstrap_type}{Two-digit bootstrap variant (\code{"11"} default; MacKinnon--Nielsen--Webb \code{"13"}, \code{"31"}, \code{"33"}, or \code{"21"} for unclustered HC2). Also accepts \code{"fnw11"} as an alias for \code{"11"}.}

\item{method}{Algorithm: \code{"auto"} (default; selects \code{"wildcluster"} for clustered OLS, \code{"hetero"} for unclustered OLS, \code{"wre"} for IV with \code{bootstrap_type = "11"}, and \code{"score"} for GLMs or other IV variants), \code{"wildcluster"}, \code{"score"}, \code{"wre"}, or \code{"hetero"}.}

\item{impose_null}{Logical; if \code{TRUE} (default) imposes \code{H0} in the bootstrap DGP, else unrestricted.}

\item{p_val_type}{P-value type: \code{"two-tailed"} (default), \code{"equal-tailed"}, \code{">"} (\code{"greater"}), or \code{"<"} (\code{"less"}). Joint tests (\code{q > 1}) only support \code{"two-tailed"}.}

\item{conf_int}{Logical; compute confidence interval for scalar tests (\code{q == 1}, default \code{TRUE}).}

\item{sign_level}{Significance level for confidence intervals (default \code{0.05} for a 95\% CI).}

\item{seed}{Optional non-negative integer seed for reproducible bootstrap draws.}

\item{ssc}{Optional small-sample correction built with \code{\link[=ssc]{ssc()}}; defaults to the model's \code{ssc()}.}

\item{tol}{Numerical tolerance for confidence interval inversion root-finding (default \code{1e-6}).}

\item{maxiter}{Maximum iterations per CI endpoint search (default \code{50L}).}

\item{nthreads}{Number of threads (default \code{getOption("fereg.nthreads", ...)}).}

\item{digits}{Number of digits to display in \code{print()} / \code{summary()}.}

\item{parm,level}{Arguments for \code{confint()}; \code{level} must match \code{1 - object$sign_level} if supplied.}

\item{...}{Additional arguments (for S3 method compatibility).}
}
\value{
An object of class \code{"fereg_boottest"} containing \code{statistic}, \code{stat_type}, \code{p_val}, \code{point_estimate}, \code{se}, \code{conf_int}, \code{t_boot}, \code{reps}, \code{reps_feasible}, \code{nobs}, \code{n_clusters}, \code{q}, \code{full_enumeration}, \code{method}, \code{bootstrap_type}, \code{type}, \code{impose_null}, \code{p_val_type}, \code{ssc}, \code{sign_level}, \code{param}, \code{clustid}, \code{bootcluster}, \code{seed}, \code{fml}, and \code{call}. Methods: \code{print()}, \code{summary()}, \code{confint()}, \code{pvalue()}, \code{tstat()}.
}
\description{
Fast wild bootstrap tests, p-values and confidence intervals by CI inversion for estimations of \code{\link[=feols]{feols()}}, \code{\link[=fepois]{fepois()}} and \code{\link[=feglm]{feglm()}}. Covers clustered OLS (WCR / WCU, including MacKinnon--Nielsen--Webb \code{13}, \code{31}, \code{33} jackknife types and exact non-nested HDFE projection), unclustered heteroskedastic OLS (\code{11}, \code{21}, \code{31}, \code{13}, \code{33}), IV/2SLS (Davidson--MacKinnon WRE and score bootstrap) and Poisson / logit GLMs (wild score bootstrap).
}
\examples{
data(trade_toy, package = "fereg")
m <- feols(y ~ x1 + x2 | id + year, trade_toy, cluster = ~id, nthreads = 2)
bt <- boottest(m, param = "x1", B = 99L, seed = 1, nthreads = 2)
summary(bt)
confint(bt)
}
