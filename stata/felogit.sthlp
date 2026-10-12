{smcl}
{* *! version 0.2.0}{...}
{vieweralsosee "[R] logit" "help logit"}{...}
{vieweralsosee "fereg" "help fereg"}{...}
{vieweralsosee "feppml" "help feppml"}{...}
{viewerjumpto "Syntax" "felogit##syntax"}{...}
{viewerjumpto "Description" "felogit##description"}{...}
{viewerjumpto "Options" "felogit##options"}{...}
{viewerjumpto "Accepted but ignored, and unsupported options" "felogit##ignored"}{...}
{viewerjumpto "Differences from logit with dummies" "felogit##differences"}{...}
{viewerjumpto "Postestimation" "felogit##postestimation"}{...}
{viewerjumpto "Examples" "felogit##examples"}{...}
{viewerjumpto "Stored results" "felogit##results"}{...}
{viewerjumpto "Author" "felogit##author"}{...}
{title:Title}

{p2colset 5 18 20 2}{...}
{p2col :{cmd:felogit} {hline 2}}Logit regression with multiple high-dimensional fixed effects (compiled plugin){p_end}
{p2colreset}{...}


{marker syntax}{...}
{title:Syntax}

{p 8 15 2}{cmd:felogit}
{depvar} [{indepvars}]
{ifin} {weight}
[{cmd:,} {opth a:bsorb(fereg##absorb:absvars)} {help felogit##options_table:options}]
{p_end}

{pstd}
{cmd:felogit} {it:y x}{cmd:, absorb(}{it:fe1 fe2}{cmd:)} gives the same slope coefficients, standard errors and
log likelihood as {cmd:logit} {it:y x} {cmd:i.}{it:fe1} {cmd:i.}{it:fe2}, without creating the dummies.
As with {cmd:logit}, a nonzero, nonmissing {it:depvar} counts as a success.
{it:indepvars} may contain factor variables and time-series operators; see {help fvvarlist} and {help tsvarlist}.
{cmd:fweight}s and {cmd:pweight}s are allowed; see {help weight}. The weight must be a variable.

{marker options_table}{...}
{synoptset 26 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Fixed effects}
{synopt :{opth a:bsorb(fereg##absorb:absvars)}}categorical variables (and interactions) to absorb; same syntax as {help fereg##absorb:fereg}{p_end}
{synopt :{cmdab:a:bsorb(}{it:...}{cmd:,} {cmdab:save:fe)}}save every fixed effect as {it:__hdfe#__}{p_end}
{synopt :{opt noa:bsorb}}absorb only the constant (same as omitting {opt absorb()}){p_end}
{synopt :{opth d(newvar)}}save the sum of the fixed effects; required by {help felogit##postestimation:predict} {opt pr}, {opt xbd}, {opt d} and {opt score}{p_end}
{synopt :{opt d}}save the sum of the fixed effects in {it:_felogit_d} (overwritten if it exists){p_end}

{syntab:Model}
{synopt :{opth vce:(felogit##vce:vcetype)}}{opt oim} (default), {opt r:obust}, or {opt cl:uster} {it:clustervars} (multi-way allowed){p_end}
{synopt :{opth cl:uster(varlist)}}synonym of {cmd:vce(cluster} {it:varlist}{cmd:)}{p_end}
{synopt :{opth off:set(varname)}}include {it:varname} in the model with coefficient constrained to 1{p_end}
{synopt :{opt keepsin:gletons}}do not drop singleton groups{p_end}
{synopt :{opt sep:aration(list)}}{cmd:fe} (default), {cmd:fe mu}, {cmd:mu} or {cmd:none}{p_end}

{syntab:Optimization and performance}
{synopt :{opt tol:erance(#)}}IRLS convergence tolerance; default {cmd:1e-8}{p_end}
{synopt :{opt itol:erance(#)}}tolerance of the fixed-effects solver at the last IRLS iterations; default {cmd:1e-9}{p_end}
{synopt :{opt maxit:erations(#)}}maximum number of IRLS iterations; default {cmd:10000}{p_end}
{synopt :{opt thr:eads(#)}}number of threads; default {cmd:0} = all logical CPUs{p_end}
{synopt :{opt serial:io}}read the data from Stata with a single thread{p_end}
{synopt :{opt v:erbose(#)}}0 (default), 1 or 2 (per-phase timings); {cmd:-1} suppresses notes{p_end}
{synopt :{opt nolog}}suppress the iteration log{p_end}

{syntab:Reporting}
{synopt :{opt or}}report odds ratios{p_end}
{synopt :{opt l:evel(#)}}set confidence level; default is {cmd:level(95)}{p_end}
{synopt :{opt nohe:ader}}suppress the header{p_end}
{synopt :{opt notab:le}}suppress the coefficient table{p_end}
{synopt :{opt nofoot:note}}suppress the table of absorbed degrees of freedom{p_end}
{synopt :{it:display_options}}see {help estimation options##display_options:[R] estimation options}{p_end}
{synoptline}
{p2colreset}{...}


{marker description}{...}
{title:Description}

{pstd}
{cmd:felogit} fits logit models by maximum likelihood with any number of absorbed fixed effects. The estimator
is the plain (unconditional) MLE of the logit model with one dummy per fixed-effect category, without any
incidental-parameter bias correction: its slope coefficients equal those of {cmd:logit} with explicit dummies
(and of {cmd:fixest::feglm(family = "logit")}).

{pstd}
The model is fitted by iteratively reweighted least squares, with step halving when the deviance increases. Each
step partials out the fixed effects with the compiled solver of {help fereg} (warm-started across iterations),
following the algorithm of {cmd:ppmlhdfe} (Correia, Guimarães and Zylkin, 2020). Fixed-effect groups whose outcome
is always 0 or always 1 are dropped first, since their fixed effect is infinite. Results are bit-for-bit
identical for any number of threads.


{marker options}{...}
{title:Options}

{dlgtab:Fixed effects}

{phang}
{opt absorb(absvars)} lists the fixed effects; the syntax is that of {help fereg##absorb:fereg}.
Without {opt absorb()} (or with {opt noabsorb}) {cmd:felogit} is a plain logit.

{phang}
{opt d(newvar)} saves the sum of the fixed effects on the logit (index) scale, with weighted mean zero (weights:
the final IRLS weights w*p*(1-p)); the constant {cmd:_cons} carries the mean. Hence the predicted probability is
invlogit(xb + d), where xb includes {cmd:_cons} and the offset. {opt d} without a name uses {it:_felogit_d}.
Saved fixed effects ({it:target}{cmd:=} or {cmd:savefe}) add up to d.

{marker vce}{...}
{dlgtab:Model}

{phang}
{opt vce(vcetype)}: {opt oim} (the inverse of the observed information matrix; also {opt unadjusted} or
{opt ols}) is the default; {opt r:obust} gives Huber/White errors with small-sample factor N/(N-1); {opt cl:uster}
{it:clustervars} gives cluster-robust errors with factor G/(G-1). Several cluster variables give multi-way
clustering. {cmd:pweight}s imply {cmd:vce(robust)}. These are the conventions of {cmd:logit}.

{phang}
{opt offset(varname)} adds {it:varname} to the index with coefficient 1.

{phang}
{opt keepsingletons} keeps singleton groups.

{phang}
{opt separation(list)}: {cmd:fe} (the default; also {cmd:default}) drops, iteratively together with singletons,
fixed-effect groups whose outcome is always 0 or always 1. {cmd:mu} additionally drops observations whose fitted
probability reaches 0 (with y = 0) or 1 (with y = 1) during IRLS. {cmd:all} means {cmd:fe mu}; {cmd:none} disables
both. The {cmd:relu} and {cmd:simplex} methods of {help feppml} are Poisson-only and not allowed here.

{dlgtab:Optimization and performance}

{phang}
{opt tolerance(#)}: IRLS stops when the relative change in deviance,
|dev - dev_old| / max(min(dev, dev_old), 0.1), is below {it:#} and the fixed-effects solver has reached its
final tolerance {opt itolerance()}. {opt maxiterations(#)} caps the IRLS iterations (error 430 when reached).

{phang}
{opt threads(#)}, {opt serialio} and {opt verbose(#)} are as in {help fereg}.

{dlgtab:Reporting}

{phang}
{opt or} reports odds ratios, exp(b). Typing {cmd:felogit} without arguments replays the results and accepts
the reporting options.


{marker ignored}{...}
{title:Accepted but ignored, and unsupported options}

{pstd}
Solver-tuning options are accepted and ignored with a note: {opt technique()}, {opt transform()},
{opt acceleration()} and {opt poolsize()}. The {cmd:feppml} options {opt exposure()}, {opt eform}, {opt irr}
and {opt guess()} are not allowed. {cmd:logit}'s {opt asis} is not available: perfect predictors are always
dropped by the {cmd:fe} rule. {cmd:aweight}s and {cmd:iweight}s are not allowed.


{marker differences}{...}
{title:Differences from logit with dummies}

{phang2}o {cmd:_cons} is reported as in {cmd:ppmlhdfe}: it is the weighted mean of the fixed effects (weights: the
final IRLS weights), not {cmd:logit}'s intercept, which depends on the base categories of the dummies. The slope
coefficients, their variance matrix, the log likelihood and the predicted probabilities are identical.{p_end}
{phang2}o The {cmd:fe} separation rule is applied iteratively with the singleton check; in cascading cases (a group
that becomes all-0 or all-1 only after another group is dropped) this can drop more observations than {cmd:logit},
which drops perfect predictors once. Observations dropped by the {cmd:fe} and {cmd:mu} rules are counted in
{cmd:e(num_separated)}; singletons in {cmd:e(num_singletons)}. Singleton groups are dropped by default (a
singleton's dummy predicts its outcome perfectly, so {cmd:logit} drops it too).{p_end}
{phang2}o A regressor collinear with the fixed effects (for example a variable constant within {it:fe1}) is reported
as omitted ({cmd:o.}{it:x}); {cmd:logit} omits it too when the dummies precede it in the varlist, and omits a dummy
instead otherwise.{p_end}
{phang2}o {cmd:e(chi2)} is a Wald test of the regressors (not the dummies), {cmd:e(chi2type)} is {cmd:Wald},
and {cmd:e(ll_0)} is the log likelihood of the constant-only model on the estimation sample.{p_end}
{phang2}o {cmd:e(ic)} counts IRLS iterations, not Newton-Raphson iterations.{p_end}
{phang2}o Saved fixed effects are normalized: every intercept fixed effect other than the first (the one with the
most categories) has a weighted mean of zero. Only their sum (d) is unique.{p_end}
{phang2}o {cmd:e(estat_cmd)} is {cmd:fereg_estat}, which provides {cmd:estat summarize}, {cmd:estat vce} and
{cmd:estat ic} as after {cmd:ppmlhdfe}; {cmd:estat ic} counts e(df_m) + 1 parameters (the regressors and
{cmd:_cons}, not the fixed effects), so its AIC and BIC differ from those after {cmd:logit} with dummies.
{cmd:logit}'s {cmd:estat} subcommands {cmd:classification}, {cmd:gof}, {cmd:lroc} and {cmd:lsens} are not
available.{p_end}


{marker postestimation}{...}
{title:Postestimation}

{p 8 16 2}{cmd:predict} {newvar} {ifin} [{cmd:,} {it:statistic}]{p_end}

{synoptset 16}{...}
{synopthdr:statistic}
{synoptline}
{synopt :{opt pr}}probability of a positive outcome, invlogit(xb + d); the default{p_end}
{synopt :{opt xb}}linear prediction without the fixed effects (includes {cmd:_cons} and the offset){p_end}
{synopt :{opt xbd}}xb + d{p_end}
{synopt :{opt d}}sum of the fixed effects{p_end}
{synopt :{opt stdp}}standard error of xb{p_end}
{synopt :{opt sc:ore}}y - pr{p_end}
{synoptline}

{pstd}
{opt pr}, {opt xbd}, {opt d} and {opt score} require {opt d()} when fixed effects are absorbed.
{help test}, {help lincom}, {help nlcom}, {help estimates} and {help margins} work as after {cmd:logit}.
{cmd:estat summarize}, {cmd:estat vce} and {cmd:estat ic} are available (see {help felogit##differences:differences}).


{marker examples}{...}
{title:Examples}

{phang2}{cmd:. webuse nlswork, clear}{p_end}
{phang2}{cmd:. felogit union age ttl_exp, absorb(idcode year)}{p_end}
{phang2}{cmd:. felogit union age ttl_exp, absorb(idcode year) vce(cluster idcode) or}{p_end}
{phang2}{cmd:. felogit union age ttl_exp, absorb(idcode year) d(dsum)}{p_end}
{phang2}{cmd:. predict double phat, pr}{p_end}


{marker results}{...}
{title:Stored results}

{pstd}
{cmd:felogit} stores the following in {cmd:e()}:

{synoptset 24 tabbed}{...}
{p2col 5 24 28 2: Scalars}{p_end}
{synopt:{cmd:e(N)}}number of observations{p_end}
{synopt:{cmd:e(N_full)}}number of observations, including dropped singleton and separated observations{p_end}
{synopt:{cmd:e(num_singletons)}}number of dropped singleton observations{p_end}
{synopt:{cmd:e(num_separated)}}number of observations dropped by the separation rules{p_end}
{synopt:{cmd:e(separation_complete)}}1 if every requested separation check finished; 0 if a check stopped before
certifying its result (a note is printed){p_end}
{synopt:{cmd:e(drop_singletons)}}1 if singletons were dropped{p_end}
{synopt:{cmd:e(rank)}}rank of e(V){p_end}
{synopt:{cmd:e(df_m)}}model degrees of freedom{p_end}
{synopt:{cmd:e(df_a)}}degrees of freedom absorbed by the fixed effects{p_end}
{synopt:{cmd:e(df_a_initial)}}number of fixed-effect categories{p_end}
{synopt:{cmd:e(df_a_redundant)}}number of redundant fixed-effect parameters{p_end}
{synopt:{cmd:e(df_a_nested)}}number of fixed-effect parameters nested within clusters{p_end}
{synopt:{cmd:e(N_hdfe)}}number of absorbed terms{p_end}
{synopt:{cmd:e(N_hdfe_extended)}}number of absorbed coefficient slots{p_end}
{synopt:{cmd:e(chi2)}}Wald chi-squared statistic{p_end}
{synopt:{cmd:e(p)}}p-value of the Wald test{p_end}
{synopt:{cmd:e(r2_p)}}pseudo R-squared, 1 - ll/ll_0{p_end}
{synopt:{cmd:e(ll)}}log likelihood{p_end}
{synopt:{cmd:e(ll_0)}}log likelihood of the constant-only model{p_end}
{synopt:{cmd:e(deviance)}}deviance, -2 ll{p_end}
{synopt:{cmd:e(ic)}}number of IRLS iterations{p_end}
{synopt:{cmd:e(ic2)}}number of fixed-effects solver iterations{p_end}
{synopt:{cmd:e(converged)}}1 if converged{p_end}
{synopt:{cmd:e(N_clust)}}number of clusters (minimum across cluster variables){p_end}
{synopt:{cmd:e(N_clust#)}}number of clusters of the #th cluster variable{p_end}
{synopt:{cmd:e(N_clustervars)}}number of cluster variables{p_end}

{synoptset 24 tabbed}{...}
{p2col 5 24 28 2: Macros}{p_end}
{synopt:{cmd:e(cmd)}}{cmd:felogit}{p_end}
{synopt:{cmd:e(cmdline)}}command as typed{p_end}
{synopt:{cmd:e(depvar)}}name of the dependent variable{p_end}
{synopt:{cmd:e(indepvars)}}names of the estimated regressors, plus {cmd:_cons}{p_end}
{synopt:{cmd:e(absvars)}}absorbed terms as typed{p_end}
{synopt:{cmd:e(extended_absvars)}}absorbed coefficient slots{p_end}
{synopt:{cmd:e(separation)}}separation methods used{p_end}
{synopt:{cmd:e(dofmethod)}}degrees-of-freedom adjustments used{p_end}
{synopt:{cmd:e(title)}}{cmd:HDFE Logistic regression} ({cmd:Logistic regression} without fixed effects){p_end}
{synopt:{cmd:e(title2)}}{cmd:Absorbing # HDFE group(s)}{p_end}
{synopt:{cmd:e(title3)}}{cmd:Statistics robust to heteroskedasticity} (clustered errors){p_end}
{synopt:{cmd:e(vce)}}{cmd:oim}, {cmd:robust} or {cmd:cluster}{p_end}
{synopt:{cmd:e(vcetype)}}{cmd:Robust} for robust and clustered errors{p_end}
{synopt:{cmd:e(clustvar)}}cluster variables{p_end}
{synopt:{cmd:e(clustvar#)}}#th cluster variable{p_end}
{synopt:{cmd:e(chi2type)}}{cmd:Wald}{p_end}
{synopt:{cmd:e(offset)}}offset variable{p_end}
{synopt:{cmd:e(d)}}name of the variable saved by {opt d()}{p_end}
{synopt:{cmd:e(wtype)}}weight type{p_end}
{synopt:{cmd:e(wexp)}}weight expression{p_end}
{synopt:{cmd:e(predict)}}{cmd:felogit_p}{p_end}
{synopt:{cmd:e(estat_cmd)}}{cmd:fereg_estat}{p_end}
{synopt:{cmd:e(footnote)}}{cmd:_rf_footnote}{p_end}
{synopt:{cmd:e(marginsok)}}predictions allowed by {cmd:margins}{p_end}
{synopt:{cmd:e(marginsnotok)}}predictions not allowed by {cmd:margins}{p_end}
{synopt:{cmd:e(properties)}}{cmd:b V}{p_end}

{synoptset 24 tabbed}{...}
{p2col 5 24 28 2: Matrices}{p_end}
{synopt:{cmd:e(b)}}coefficient vector{p_end}
{synopt:{cmd:e(V)}}variance-covariance matrix of the estimators{p_end}
{synopt:{cmd:e(dof_table)}}absorbed degrees of freedom: categories, redundant, coefficients, inexact, nested{p_end}

{synoptset 24 tabbed}{...}
{p2col 5 24 28 2: Functions}{p_end}
{synopt:{cmd:e(sample)}}marks the estimation sample{p_end}


{marker author}{...}
{title:Author}

{pstd}Leo Liu{break}
{browse "https://github.com/leoliu0/fereg":github.com/leoliu0/fereg}{p_end}

{pstd}
The IRLS algorithm, separation checks and displays of {cmd:felogit} follow {cmd:ppmlhdfe} by Sergio Correia,
Paulo Guimarães and Thomas Zylkin (MIT license):{p_end}

{phang}Correia, S., P. Guimarães, and T. Zylkin. 2020. Fast Poisson estimation with high-dimensional fixed effects.
{it:Stata Journal} 20(1): 95-115.{p_end}
