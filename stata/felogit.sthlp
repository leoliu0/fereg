{smcl}
{* *! version 0.4.1}{...}
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
{viewerjumpto "Differences from logitfe" "felogit##logitfe"}{...}
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
[{cmd:,} {opth a:bsorb(fereg##absorb:absvars)} {help felogit##options_table:options} {help felogit##bias:bias_options}]
{p_end}

{pstd}
{cmd:felogit} {it:y x}{cmd:, absorb(}{it:fe1 fe2}{cmd:)} gives the same slope coefficients, standard errors and
log likelihood as {cmd:logit} {it:y x} {cmd:i.}{it:fe1} {cmd:i.}{it:fe2}, without creating the dummies.
As with {cmd:logit}, a nonzero, nonmissing {it:depvar} counts as a success.
{it:indepvars} may contain factor variables and time-series operators; see {help fvvarlist} and {help tsvarlist}.
{cmd:fweight}s and {cmd:pweight}s are allowed (not with the bias options); see {help weight}. The weight must be a variable.

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
{synopt :{opth vce:(felogit##vce:vcetype)}}{opt oim} (default), {opt r:obust}, {opt cl:uster} {it:clustervars} (multi-way allowed), {opt dk:raay} [{it:#}], {opt hc2}, {opt hc3}, {opt crv3} {it:clustervars}, or {opt conley} {it:latvar lonvar}[{cmd:,} {it:conley_options}]{p_end}
{synopt :{opth cl:uster(varlist)}}synonym of {cmd:vce(cluster} {it:varlist}{cmd:)}{p_end}
{synopt :{opt bw()} {opt kernel()} {opt dkraay()} {opt kiefer} {opt psd0} {opt psda}}kernel (HAC) variances, as in {help fereg##variance:fereg}{p_end}
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

{synoptset 26 tabbed}{...}
{synopthdr:bias_options}
{synoptline}
{syntab:Bias correction and APEs ({help felogit##bias:details})}
{synopt :{opt ape}}average partial effects of the uncorrected estimator{p_end}
{synopt :{opt an:alytical}}analytical bias correction of the coefficients and APEs{p_end}
{synopt :{opt lags(#)}}bandwidth for predetermined regressors; default {cmd:0}{p_end}
{synopt :{opt jack:knife}}jackknife bias correction: {opt ss2} (default), {opt ss1}, {opt js}, {opt sj}, {opt jj} or {opt double}{p_end}
{synopt :{opt mul:tiple(#)}}average over # random partitions ({opt ss1}, {opt ss2}){p_end}
{synopt :{opt i:ndividuals} {opt t:ime}}randomize only the individuals or only the periods{p_end}
{synopt :{opt ibias(yes|no)}}correct the individual-effect bias; default {cmd:yes}{p_end}
{synopt :{opt tbias(yes|no)}}correct the time-effect bias; default {cmd:yes}{p_end}
{synopt :{opt pop:ulation(#)}}population size for the APE variance{p_end}
{synopt :{opt net:work}}network structure: exporter-time, importer-time [and pair] effects{p_end}
{synoptline}
{p2colreset}{...}


{marker description}{...}
{title:Description}

{pstd}
{cmd:felogit} fits logit models by maximum likelihood with any number of absorbed fixed effects. The estimator
is the plain (unconditional) MLE of the logit model with one dummy per fixed-effect category: its slope
coefficients equal those of {cmd:logit} with explicit dummies (and of {cmd:fixest::feglm(family = "logit")}).
With panel fixed effects this MLE has an incidental-parameter bias; the {help felogit##bias:bias options} correct
it analytically or by jackknife and report average partial effects, as {cmd:logitfe} does.

{pstd}
The model is fitted by iteratively reweighted least squares, with step halving when the deviance increases. Each
step partials out the fixed effects with the compiled solver of {help fereg} (warm-started across iterations),
following the algorithm of {cmd:ppmlhdfe} (Correia, Guimar{c a~}es and Zylkin, 2020). Fixed-effect groups whose outcome
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
The other variance estimators of {help fereg##variance:fereg} are available as {it:fereg extensions}
({cmd:logit} has none of them for this model), with the same syntax, applied to the scores of the final IRLS
iteration: {opt vce(dkraay [#])} ({cmd:reghdfe}'s Driscoll-Kraay formula with the asymptotic factor T/(T-1));
the {cmd:ivreghdfe} kernel options {opt bw()}, {opt kernel()}, {opt dkraay(#)}, {opt kiefer}, {opt psd0} and
{opt psda} (HAC with {opt vce(robust)}, AC with the default {opt oim}; factor N/(N-1), or G/(G-1) with
clustering); {opt vce(hc2)} and {opt vce(hc3)} (leverage of the IRLS-weighted design including the fixed
effects, no factor); {opt vce(crv3 clustervars)} (leave-one-cluster-out logit refits, factor G/(G-1); the
jackknifed {cmd:_cons} is the linearized one); {opt vce(conley ...)} ({cmd:fixest}'s Conley variance, factor
N/(N-1)). The header adds the lines of {cmd:fereg} (DK bandwidth, Conley cutoff, kernel).

{phang}
{opt offset(varname)} adds {it:varname} to the index with coefficient 1.

{phang}
{opt keepsingletons} keeps singleton groups.

{phang}
{opt separation(list)}: {cmd:fe} (the default; also {cmd:default}) drops, iteratively together with singletons,
fixed-effect groups whose outcome is always 0 or always 1. {cmd:mu} additionally drops observations whose fitted
probability reaches 0 (with y = 0) or 1 (with y = 1) during IRLS. {cmd:all} means {cmd:fe mu}; {cmd:none} disables
both. The {cmd:relu} and {cmd:simplex} methods of {help feppml} are Poisson-only and not allowed here.

{marker bias}{...}
{dlgtab:Bias correction and average partial effects}

{pstd}
These options have the names of {cmd:logitfe} (Cruz-Gonzalez, Fern{c a'}ndez-Val and Weidner 2017), whose results
they reproduce except for the {help felogit##logitfe:differences} listed below. They need the fixed effects of the
panel, logit with no weights, at least one regressor and {cmd:separation(fe)} (the default). {opt vce()} applies
to {cmd:e(V)}, which is always the variance of the uncorrected estimator, as in {cmd:logitfe}.

{pmore}
{it:Classic structure} (the default): the data must be {cmd:xtset} {it:panelvar timevar}, and {opt absorb()} holds
the panel variable (individual effects), the time variable (time effects) or both, in any order. With individual
(time) effects only, only the individual-effect (time-effect) bias exists.
{it:Network structure} ({opt network}): {opt absorb()} holds exporter-time, importer-time and, optionally, pair
effects, in that order (Hinz, Stammann and Wanner 2020; the normalization of {cmd:alpaca}).

{phang}
{opt ape} reports the average partial effects (APEs) of the uncorrected estimator ({cmd:logitfe}'s
{opt nocorrection}). The APE of a regressor is the average over the observations before the drop of groups with a
constant outcome (whose partial effect is taken as 0) of beta*p(1-p) for a continuous regressor and of
Pr(y=1 | x=1) - Pr(y=1 | x=0) for a binary one (values exactly 0 and 1). The table of APEs follows the
coefficient table; they are stored in {cmd:e(b2)} and {cmd:e(V2)}.

{phang}
{opt analytical} corrects the coefficients and the APEs for the incidental-parameter bias by the analytical
formulas of Fern{c a'}ndez-Val and Weidner (2016) as implemented by {cmd:logitfe} (network: by {cmd:alpaca});
the fixed effects are then re-estimated by maximum likelihood at the corrected coefficients.
{opt lags(#)} (default 0) is the bandwidth for predetermined (weakly exogenous) regressors, such as a lagged
dependent variable: lags are calendar periods of the {cmd:xtset} time variable (gaps respected); # must not exceed
the number of observations of the shortest panel minus 1. {opt lags()} requires the individual-effect bias.

{phang}
{opt jackknife} corrects by the split-panel jackknife of Dhaene and Jochmans (2015) or delete-one jackknives
({help felogit##logitfe:table}): {opt ss2} (the default: halves of the individuals, and each individual's own
halves of its periods), {opt ss1} (four subpanels: halves of the individuals by halves of the periods), {opt js}
(delete-one individual and time halves), {opt sj} (individual halves and delete-one period), {opt jj}
(delete-one individual and delete-one period) or {opt double} (delete-one id, for panels whose panel and time
variables share their ids, such as importer-exporter data). Every subpanel is a subset of the observations
before the drop of groups with a constant outcome and is refitted with {cmd:felogit}'s own sample rules.
{opt multiple(R)} ({opt ss1} and {opt ss2}) averages the corrections of R random partitions; {opt individuals}
or {opt time} randomizes only the order of the individuals or of the periods (default both). The partitions are
{cmd:logitfe}'s and need {cmd:moremata} ({cmd:ssc install moremata}).

{phang}
{opt ibias(yes|no)} and {opt tbias(yes|no)} (with {opt analytical} or {opt jackknife}; default yes/yes) select the
individual-effect bias, of order 1/T, and the time-effect bias, of order 1/N.

{phang}
{opt population(#)} is the size of the population for the finite-population factor
(M - N*)/(M - 1) of the sampling-of-effects term of the APE variance (M = #, N* = e(N_full)); M must be at least
N*. Without it the factor is 1 (infinite population), as in {cmd:logitfe}.

{phang}
{opt network} selects the network structure (with {opt analytical} or {opt ape}; the jackknife is for the classic
structure). {opt lags()} then orders each pair's observations by the {cmd:xtset} time variable.

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


{marker logitfe}{...}
{title:Differences from logitfe}

{pstd}
The options {opt analytical}, {opt jackknife}, {opt ape} and their suboptions have the names and the meaning of
{cmd:logitfe} 3.0.3 (Cruz-Gonzalez, Fern{c a'}ndez-Val and Weidner 2017), and their results equal {cmd:logitfe}'s
(to {cmd:logitfe}'s optimization tolerance) except for the following deliberate differences, where {cmd:logitfe}
has a bug or {cmd:felogit} follows {cmd:fereg}'s rules:

{phang2}o {it:Lags of 2 or more} ({opt lags(L)}, L >= 2): {cmd:logitfe} uses the score of period t-1 for every
lag j (its ado-file, lines 3409 and 3514); {cmd:felogit} uses period t-j, the estimator of Fern{c a'}ndez-Val and
Weidner (2016). {opt lags(0)} and {opt lags(1)} are identical.{p_end}
{phang2}o {it:APE standard errors} follow Fern{c a'}ndez-Val and Weidner (2016) as implemented in {cmd:alpaca},
not {cmd:logitfe}'s, which are inflated when groups are dropped (the sampling term is centered at an APE divided
by the number of observations before the drop), use the derivative of the normal density in the delta method,
and have off-diagonal elements that are not covariances. {cmd:e(V2)} is the delta-method variance (including the
estimation noise of the fixed effects) plus e(fpc) times the sampling-of-effects term with dependence within
individuals and within periods, plus the weak-exogeneity term with {opt lags()}. With groups dropped for a
constant outcome, the sampling-of-effects term sums over all observations, with zero partial effects for the
dropped ones, consistently with the point estimate; {cmd:alpaca} and {cmd:logitfe} sum over the retained
observations only, which overstates the variance. The numbers are identical when no group is dropped.{p_end}
{phang2}o {it:Jackknife dimensions}: the individual-effect bias, of order 1/T, is removed by subpanels in the time
dimension, the time-effect bias, of order 1/N, by subpanels of individuals. {cmd:logitfe} swaps them in
{opt js} and {opt sj} (both biases: {opt js} corrects the 1/N bias twice and the 1/T bias not at all, {opt sj} the
reverse) and in the single-bias versions of {opt js}, {opt sj}, {opt jj} and {opt double}. {cmd:felogit} computes
(b = full-sample estimate, N individuals, T periods, b(-i) and b(-t) delete-one estimates; IH, TH, PH: individual
halves, global time halves, each individual's own time halves; Q: the four quadrants IH x TH):{p_end}

{p2colset 9 22 24 2}{...}
{p2col :variant}both biases / {opt tbias(no)} (individual effects) / {opt ibias(no)} (time effects){p_end}
{p2line}
{p2col :{opt ss2}}3b - mean(IH) - mean(PH) / 2b - mean(PH) / 2b - mean(IH){p_end}
{p2col :{opt ss1}}2b - mean(Q) / 2b - mean(TH) / 2b - mean(IH){p_end}
{p2col :{opt js}}(N+1)b - (N-1)mean b(-i) - mean(TH) / 2b - mean(TH) / Nb - (N-1)mean b(-i){p_end}
{p2col :{opt sj}}(T+1)b - (T-1)mean b(-t) - mean(IH) / Tb - (T-1)mean b(-t) / 2b - mean(IH){p_end}
{p2col :{opt jj}}(N+T-1)b - (N-1)mean b(-i) - (T-1)mean b(-t) / Tb - (T-1)mean b(-t) / Nb - (N-1)mean b(-i){p_end}
{p2col :{opt double}}Jb - (J-1)mean b(-c) / Tb - (T-1)mean b(-t) / Nb - (N-1)mean b(-i){p_end}
{p2line}
{p2colreset}{...}

{pmore}
With individual (time) effects only, the individual-effect (time-effect) column applies. N and T count every
individual and period before the drop of groups with a constant outcome, as in {cmd:logitfe}; odd halves share the
middle unit; {opt double}'s subpanel c drops the observations whose panel or time value equals c, over the J values
that occur as both. The same combinations apply to the APEs, each subpanel APE divided by the subpanel's own
number of observations.{p_end}
{phang2}o {opt jackknife ss2 tbias(no)} works ({cmd:logitfe} stops with error 110).{p_end}
{phang2}o A subpanel fit that loses a regressor (collinear or a perfect predictor in the subpanel) or does not
converge is an error naming the subpanel; {cmd:logitfe} enters a coefficient and APE of 0 in the average.{p_end}
{phang2}o {it:Sample}: {cmd:logitfe} drops perfect predictors with {cmd:_rmcoll, logit}, then individuals with a
constant outcome and then periods with a constant outcome, once; {cmd:felogit} drops groups with a constant
outcome and singletons iteratively (see {help felogit##differences:above}), in the full sample and in every
jackknife subpanel. The samples differ only in cascading cases.{p_end}
{phang2}o {opt population(M)}: the factor is (M - N*)/(M - 1) with N* the number of observations before the drop of
groups with a constant outcome (e(N_full)), as {cmd:logitfe}'s help defines it; {cmd:logitfe}'s code uses the
number after the drop.{p_end}
{phang2}o {opt multiple(R)} draws {cmd:logitfe}'s partitions ({cmd:set seed 13579} and {cmd:moremata}'s
{cmd:mm_sample()}, in {cmd:logitfe}'s order), but leaves the random-number state of the session unchanged;
{cmd:logitfe} leaves the seed at 13579 plus its draws. Time halves of a partition split the estimation sample's
periods by their permuted order; {cmd:logitfe} splits the permuted ids of all periods of the dataset, which is
the same when the sample contains every period.{p_end}
{phang2}o {it:Options}: no option means no correction ({cmd:logitfe}'s default is {opt analytical}); {opt ape} alone
is {cmd:logitfe}'s {opt nocorrection}. The fixed effects are those of {opt absorb()} ({cmd:logitfe}'s
{opt ieffects()} and {opt teffects()}). Combinations {cmd:logitfe} ignores or resets silently are errors:
{opt lags()} with {opt ibias(no)} or time effects only, {opt individuals} or {opt time} without {opt multiple()},
{opt ibias()}/{opt tbias()} without a correction.{p_end}
{phang2}o {cmd:e(b)} also holds {cmd:_cons} ({help felogit##differences:above}), re-evaluated at the corrected point.
{opt vce()} is allowed and {cmd:e(V)} is the uncorrected variance under it ({cmd:logitfe}: always the inverse
Hessian). {cmd:e(N_drop)}, {cmd:e(N_group_drop)}, {cmd:e(N_time_drop)}, {cmd:e(rankV)} and {cmd:e(rankV2)} are not
stored; see {cmd:e(N_full)} and {cmd:e(num_separated)}. {cmd:e(chi2)} is {cmd:felogit}'s Wald test, not
{cmd:logitfe}'s likelihood-ratio test including the dummies.{p_end}
{phang2}o The network correction ({opt network}) has no {cmd:logitfe} counterpart; it follows {cmd:alpaca}.{p_end}
{phang2}o Lags count calendar periods of the {cmd:xtset} time variable (gaps respected), as in {cmd:logitfe};
{cmd:fereg}'s R and Python versions follow {cmd:alpaca}, which counts observations.{p_end}


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

{pstd}
After {opt analytical} or {opt jackknife}, {cmd:predict} and {opt d()} use the corrected coefficients and the fixed
effects re-estimated at them; {cmd:test}, {cmd:lincom}, {cmd:nlcom} and {cmd:margins} combine the corrected
coefficients with the uncorrected variance in {cmd:e(V)}. {cmd:estat ic} uses the log likelihood of the
uncorrected fit. Replay ({cmd:felogit} alone) shows the table of average partial effects again and returns it in
{cmd:r(table_ape)}.


{marker examples}{...}
{title:Examples}

{phang2}{cmd:. webuse nlswork, clear}{p_end}
{phang2}{cmd:. felogit union age ttl_exp, absorb(idcode year)}{p_end}
{phang2}{cmd:. felogit union age ttl_exp, absorb(idcode year) vce(cluster idcode) or}{p_end}
{phang2}{cmd:. felogit union age ttl_exp, absorb(idcode year) d(dsum)}{p_end}
{phang2}{cmd:. predict double phat, pr}{p_end}

{pstd}Bias corrections and average partial effects ({cmd:logitfe}'s example data: {cmd:net get logitfe}):{p_end}

{phang2}{cmd:. use lfp_psid, clear}{p_end}
{phang2}{cmd:. xtset ID1979 year}{p_end}
{phang2}{cmd:. felogit lfp kids0_2 kids3_5 kids6_17 loghusbandincome age age2, absorb(ID1979 year) ape}{p_end}
{phang2}{cmd:. felogit lfp L.lfp kids0_2 kids3_5 kids6_17 loghusbandincome age age2, absorb(ID1979 year) analytical lags(1)}{p_end}
{phang2}{cmd:. felogit lfp kids0_2 kids3_5 kids6_17 loghusbandincome age age2, absorb(ID1979 year) jackknife ss2 d(dc)}{p_end}
{phang2}{cmd:. felogit lfp kids0_2 kids3_5 kids6_17 loghusbandincome age age2, absorb(ID1979 year) jackknife ss1 multiple(10)}{p_end}
{phang2}{cmd:. felogit lfp kids0_2 kids3_5 kids6_17 loghusbandincome age age2, absorb(ID1979) analytical population(20000)}{p_end}
{phang2}{cmd:. matrix list e(b2)}{p_end}


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
{synopt:{cmd:e(dkraay_bw)}, {cmd:e(bw)}, {cmd:e(conley_cutoff)}, {cmd:e(conley_pixel)}}as {help fereg##results:fereg}{p_end}

{pstd}
With {opt analytical}, {opt jackknife} or {opt ape} also:

{synoptset 24 tabbed}{...}
{p2col 5 24 28 2: Scalars}{p_end}
{synopt:{cmd:e(fpc)}}factor of the sampling term of the APE variance: 1, or (M - N*)/(M - 1) with {opt population(M)}{p_end}
{synopt:{cmd:e(N_group)}}number of individuals (classic structure){p_end}
{synopt:{cmd:e(T_min)}, {cmd:e(T_avg)}, {cmd:e(T_max)}}observations per individual (classic structure){p_end}
{synopt:{cmd:e(multiple)}}number of random partitions of {opt multiple()}{p_end}
{synopt:{cmd:e(N_subpanels)}}number of jackknife subpanels{p_end}

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
{synopt:{cmd:e(title3)}}{cmd:Statistics robust to heteroskedasticity} (clustered errors, {cmd:dkraay}, {cmd:conley}, robust kernels; {cmd:Statistics robust to autocorrelation} with an AC kernel){p_end}
{synopt:{cmd:e(title4)}}second line for {cmd:dkraay} and {cmd:conley}{p_end}
{synopt:{cmd:e(vce)}}{cmd:oim}, {cmd:robust}, {cmd:cluster}, {cmd:dkraay}, {cmd:hc2}, {cmd:hc3}, {cmd:crv3}, {cmd:conley}, or {cmd:ivreghdfe}'s description with a kernel{p_end}
{synopt:{cmd:e(vcetype)}}{cmd:Robust} for robust, clustered, Driscoll-Kraay and HAC errors; {cmd:Robust HC2}, {cmd:Robust HC3}, {cmd:CRV3}, {cmd:Conley}{p_end}
{synopt:{cmd:e(kernel)}, {cmd:e(tvar)}, {cmd:e(ivar)}, {cmd:e(bwchoice)}, {cmd:e(hacsubtitleV)}, {cmd:e(hacsubtitleV2)}}kernel variances, as {help fereg##results:fereg}{p_end}
{synopt:{cmd:e(conley_lat)}, {cmd:e(conley_lon)}, {cmd:e(conley_distance)}}Conley coordinates and distance{p_end}
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
{synopt:{cmd:e(bias)}}{cmd:analytical}, {cmd:jackknife} or {cmd:none} ({opt ape} alone){p_end}
{synopt:{cmd:e(jackknife)}}jackknife variant: {cmd:ss1}, {cmd:ss2}, {cmd:js}, {cmd:sj}, {cmd:jj} or {cmd:double}{p_end}
{synopt:{cmd:e(lags)}}bandwidth of {opt analytical} ({opt lags()}){p_end}
{synopt:{cmd:e(ibias)}, {cmd:e(tbias)}}{cmd:yes} or {cmd:no}: individual-effect and time-effect bias corrected{p_end}
{synopt:{cmd:e(structure)}}{cmd:classic} or {cmd:network}{p_end}
{synopt:{cmd:e(split)}}{cmd:both}, {cmd:individuals} or {cmd:time}: dimensions randomized by {opt multiple()}{p_end}
{synopt:{cmd:e(id)}, {cmd:e(time)}}the {cmd:xtset} panel and time variables{p_end}
{synopt:{cmd:e(bctitle)}}{cmd:logitfe}'s title, e.g. {cmd:Analytical bias-correction}{p_end}
{synopt:{cmd:e(bctitle1)}}{cmd:Type of included effects: ...}{p_end}
{synopt:{cmd:e(bctitle2)}}{cmd:Type of bias correction: ...}{p_end}
{synopt:{cmd:e(bctitle3)}}{cmd:Triming parameter = }{it:#} ({opt analytical}; {cmd:logitfe}'s spelling) or
{it:R} {cmd:multiple partitions in ...}{p_end}

{synoptset 24 tabbed}{...}
{p2col 5 24 28 2: Matrices}{p_end}
{synopt:{cmd:e(b)}}coefficient vector; the corrected coefficients with {opt analytical} or {opt jackknife}{p_end}
{synopt:{cmd:e(V)}}variance-covariance matrix of the estimators (of the uncorrected estimator, as {cmd:logitfe}){p_end}
{synopt:{cmd:e(dof_table)}}absorbed degrees of freedom: categories, redundant, coefficients, inexact, nested{p_end}
{synopt:{cmd:e(b0)}}uncorrected coefficients (with a bias correction or APEs){p_end}
{synopt:{cmd:e(b2)}}average partial effects (APEs), one per regressor; 0 for base and omitted levels{p_end}
{synopt:{cmd:e(V2)}}variance-covariance matrix of the APEs{p_end}

{synoptset 24 tabbed}{...}
{p2col 5 24 28 2: Functions}{p_end}
{synopt:{cmd:e(sample)}}marks the estimation sample{p_end}


{marker author}{...}
{title:Author}

{pstd}Leo Liu{break}
UTS Business School, University of Technology Sydney{break}
Sydney, Australia{break}
{browse "mailto:leo.liu@uts.edu.au":leo.liu@uts.edu.au}{break}
{browse "https://github.com/leoliu0/fereg":github.com/leoliu0/fereg}{p_end}

{pstd}
The IRLS algorithm, separation checks and displays of {cmd:felogit} follow {cmd:ppmlhdfe} by Sergio Correia,
Paulo Guimar{c a~}es and Thomas Zylkin (MIT license):{p_end}

{phang}Correia, S., P. Guimar{c a~}es, and T. Zylkin. 2020. Fast Poisson estimation with high-dimensional fixed effects.
{it:Stata Journal} 20(1): 95-115.{p_end}

{pstd}
The bias corrections, average partial effects and jackknives follow {cmd:logitfe} 3.0.3 by Mario Cruz-Gonzalez,
Iv{c a'}n Fern{c a'}ndez-Val and Martin Weidner, with the corrections listed under
{help felogit##logitfe:differences from logitfe}; the variance of the average partial effects and the network
correction follow {cmd:alpaca} 0.3.5 by Amrei Stammann:{p_end}

{phang}Cruz-Gonzalez, M., I. Fern{c a'}ndez-Val, and M. Weidner. 2017. Bias corrections for probit and logit models with
two-way fixed effects. {it:Stata Journal} 17(3): 517-545.{p_end}

{phang}Dhaene, G., and K. Jochmans. 2015. Split-panel jackknife estimation of fixed-effect models. {it:Review of
Economic Studies} 82(3): 991-1030.{p_end}

{phang}Fern{c a'}ndez-Val, I., and M. Weidner. 2016. Individual and time effects in nonlinear panel models with large
N, T. {it:Journal of Econometrics} 192(1): 291-312.{p_end}

{phang}Hinz, J., A. Stammann, and J. Wanner. 2020. State dependence and unobserved heterogeneity in the extensive
margin of trade. arXiv:2004.12655.{p_end}
