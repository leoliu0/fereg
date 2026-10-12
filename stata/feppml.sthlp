{smcl}
{* *! version 0.3.0}{...}
{vieweralsosee "ppmlhdfe" "help ppmlhdfe"}{...}
{vieweralsosee "fereg" "help fereg"}{...}
{vieweralsosee "felogit" "help felogit"}{...}
{vieweralsosee "[R] poisson" "help poisson"}{...}
{viewerjumpto "Syntax" "feppml##syntax"}{...}
{viewerjumpto "Description" "feppml##description"}{...}
{viewerjumpto "Options" "feppml##options"}{...}
{viewerjumpto "Accepted but ignored, and unsupported options" "feppml##ignored"}{...}
{viewerjumpto "Differences from ppmlhdfe" "feppml##differences"}{...}
{viewerjumpto "Postestimation" "feppml##postestimation"}{...}
{viewerjumpto "Examples" "feppml##examples"}{...}
{viewerjumpto "Stored results" "feppml##results"}{...}
{viewerjumpto "Author" "feppml##author"}{...}
{title:Title}

{p2colset 5 17 19 2}{...}
{p2col :{cmd:feppml} {hline 2}}Poisson pseudo-maximum likelihood with multiple high-dimensional fixed effects (compiled plugin){p_end}
{p2colreset}{...}


{marker syntax}{...}
{title:Syntax}

{p 8 15 2}{cmd:feppml}
{depvar} [{indepvars}]
{ifin} {weight}
[{cmd:,} {opth a:bsorb(fereg##absorb:absvars)} {help feppml##options_table:options}]
{p_end}

{pstd}
{cmd:feppml} is a drop-in replacement for {help ppmlhdfe} 2.3.3: same syntax, same defaults, same
separation checks, same output and the same stored results, computed by a parallel compiled solver.
{it:depvar} must be nonnegative; it need not be an integer.
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
{synopt :{opth d(newvar)}}save the sum of the fixed effects; required by most {help feppml##postestimation:predict} options{p_end}
{synopt :{opt d}}save the sum of the fixed effects in {it:_ppmlhdfe_d} (overwritten if it exists){p_end}

{syntab:Model}
{synopt :{opth vce:(feppml##vce:vcetype)}}{opt r:obust} (default) or {opt cl:uster} {it:clustervars} (multi-way allowed){p_end}
{synopt :{opth cl:uster(varlist)}}synonym of {cmd:vce(cluster} {it:varlist}{cmd:)}{p_end}
{synopt :{opth exp:osure(varname)}}include ln({it:varname}) in the model with coefficient constrained to 1{p_end}
{synopt :{opth off:set(varname)}}include {it:varname} in the model with coefficient constrained to 1{p_end}
{synopt :{opt keepsin:gletons}}do not drop singleton groups (not recommended){p_end}

{syntab:Separation}
{synopt :{opt sep:aration(list)}}methods used to detect and drop separated observations; default {cmd:fe simplex relu}{p_end}
{synopt :{opth tagsep(newvar)}}only identify the separated observations (ReLU) and tag them in {it:newvar}; nothing is estimated{p_end}
{synopt :{opth zvar:name(newvar)}}with {opt tagsep()}, also save the certificate of separation{p_end}

{syntab:Optimization and performance}
{synopt :{opt tol:erance(#)}}IRLS convergence tolerance; default {cmd:1e-8}{p_end}
{synopt :{opt itol:erance(#)}}tolerance of the fixed-effects solver at the last IRLS iterations; default {cmd:1e-9}{p_end}
{synopt :{opt maxit:erations(#)}}maximum number of IRLS iterations; default {cmd:10000}{p_end}
{synopt :{opt guess(string)}}initial values: {opt simple} (default) or {opt ols}{p_end}
{synopt :{opt thr:eads(#)}}number of threads; default {cmd:0} = all logical CPUs{p_end}
{synopt :{opt serial:io}}read the data from Stata with a single thread{p_end}
{synopt :{opt v:erbose(#)}}0 (default), 1 or 2 (per-phase timings); {cmd:-1} suppresses notes{p_end}
{synopt :{opt nolog}}suppress the iteration log{p_end}

{syntab:Reporting}
{synopt :{opt ef:orm}}report exponentiated coefficients (incidence-rate ratios){p_end}
{synopt :{opt ir:r}}synonym of {opt eform}{p_end}
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
{cmd:feppml} fits Poisson regressions by pseudo-maximum likelihood (PPML) with any number of absorbed
fixed effects, following {cmd:ppmlhdfe} 2.3.3 (Correia, Guimarães and Zylkin, 2020): iteratively reweighted
least squares with an inner tolerance that tightens as IRLS converges, singletons and observations separated
by a fixed effect dropped iteratively, the ReLU separation check, the same collinearity rules and messages,
robust or multi-way clustered standard errors, and the same {cmd:e()} results, so that {cmd:esttab},
{cmd:outreg2}, {cmd:margins}, {cmd:test} and {cmd:lincom} work unchanged.

{pstd}
Each IRLS step is a weighted least-squares problem; the fixed effects are partialled out by the same compiled
solver as {help fereg}, which reuses its data structures and warm-starts across IRLS iterations. Results are
bit-for-bit identical for any number of threads.


{marker options}{...}
{title:Options}

{dlgtab:Fixed effects}

{phang}
{opt absorb(absvars)} lists the fixed effects; the syntax is that of {help fereg##absorb:fereg}
({it:x}, {it:x}{cmd:#}{it:y}, string variables, {it:target}{cmd:=}{it:term}, and the suboption {cmd:savefe}).
Without {opt absorb()} (or with {opt noabsorb}) only the constant is absorbed.

{phang}
{opt d(newvar)} saves the sum of the fixed effects, on the scale of the linear predictor and without the
offset, with weighted mean zero (weights: the final IRLS weights mu*w); the constant {cmd:_cons} carries the
mean. Hence ln(mu) = xb + d, where xb includes {cmd:_cons} and the offset. {opt d} without a name uses
{it:_ppmlhdfe_d}. Saved fixed effects ({it:target}{cmd:=} or {cmd:savefe}) add up to d.

{marker vce}{...}
{dlgtab:Model}

{phang}
{opt vce(vcetype)}: {opt r:obust} (Huber/White, with small-sample factor N/(N-1)) is the default;
{cmd:vce(unadjusted)} is treated as {cmd:robust}, as in {cmd:ppmlhdfe}. {opt cl:uster} {it:clustervars} gives
cluster-robust errors (factor G/(G-1), G = smallest number of clusters); several cluster variables give
multi-way clustering with the Cameron-Gelbach-Miller adjustment when the variance matrix is not positive
semidefinite. Cluster variables may be interactions ({it:exporter#importer}) and string variables.

{phang}
{opt exposure(varname)} and {opt offset(varname)} add ln({it:varname}) or {it:varname} to the linear
predictor with coefficient 1. The exposure must be positive. Only one of them may be specified.

{phang}
{opt keepsingletons} keeps singleton groups (and, as in {cmd:ppmlhdfe}, disables the {cmd:fe} separation rule);
a warning is printed.

{marker separation}{...}
{dlgtab:Separation}

{phang}
{opt separation(list)} chooses the methods used to detect observations whose mean is zero at the
pseudo-maximum (separated observations), which are dropped before estimation (Correia, Guimarães and Zylkin, 2019):

{phang2}{cmd:fe}: drop fixed-effect groups whose outcome is always zero (iterated with the singleton check);{p_end}
{phang2}{cmd:relu}: the ReLU algorithm of {cmd:ppmlhdfe} (alias {cmd:ir});{p_end}
{phang2}{cmd:simplex}: accepted as an alias of {cmd:relu} (see {help feppml##differences:differences});{p_end}
{phang2}{cmd:mu}: drop observations with y = 0 whose fitted mean falls below a tolerance during IRLS.{p_end}

{pmore}
{cmd:default} (also {cmd:def}, {cmd:standard}, {cmd:auto}, {cmd:on}) means {cmd:fe simplex relu};
{cmd:all} (or {cmd:full}) means {cmd:fe relu simplex mu}; {cmd:none} (or {cmd:no}, {cmd:off}) disables the checks.
The number of separated observations is reported in the output and in {cmd:e(num_separated)}.

{phang}
{opt tagsep(newvar)} runs the separation check instead of the regression, as {cmd:ppmlhdfe}'s
{opt tagsep()} does, and ignores {cmd:separation()}. After singletons are dropped, observations in
fixed-effect groups whose outcome is always zero and the observations found by ReLU are removed, and the
check is repeated on the rest until ReLU finds nothing more. {it:newvar} (byte) is 1 for separated
observations, 0 for the other observations of the sample after the initial singleton drop, and missing
elsewhere, including those singletons. Observations that become singletons once separated observations are
removed are coded by whether they are separated: 1 if the outcome is zero (a zero-outcome singleton is
separated by its own fixed effect), 0 otherwise; the regression would drop both. Nothing is estimated and
no {cmd:e()} results are posted. An existing variable
named {it:newvar} is replaced (as {cmd:ppmlhdfe} does), unless the model uses it. A warning is printed if the
check stopped before certifying its result.

{phang}
{opt zvarname(newvar)} (requires {opt tagsep()}) also saves the certificate of separation z (double): z > 0
on separated observations and 0 elsewhere in the sample. Observations in fixed-effect groups whose outcome is
always zero get z = 1 (the group indicator); the other separated observations get the ReLU certificate, a
combination of the regressors and fixed effects that is 0 wherever y > 0 on the sample that remains. As in
{cmd:ppmlhdfe}, z is rescaled so that its most frequent positive value is 1 when the rescaling factor is
between 0.01 and 100. Certificates are not unique, so z can differ from {cmd:ppmlhdfe}'s; the tagged
observations are what matters.

{dlgtab:Optimization and performance}

{phang}
{opt tolerance(#)}: IRLS stops when the relative change in deviance,
|dev - dev_old| / max(min(dev, dev_old), 0.1), is below {it:#} and the fixed-effects solver has reached its
final tolerance.

{phang}
{opt itolerance(#)} sets the final tolerance of the fixed-effects solver. The solver starts at 1e-4 and tightens
the tolerance as IRLS converges, as {cmd:ppmlhdfe} does.

{phang}
{opt maxiterations(#)} caps the number of IRLS iterations; reaching it stops with error 430.

{phang}
{opt guess(simple|ols)} sets the initial values: {opt simple} uses mu = (y + mean(y))/2; {opt ols} fits a linear
regression of ln(y + mean(y)/100) with the fixed effects.

{phang}
{opt threads(#)}, {opt serialio} and {opt verbose(#)} are as in {help fereg}. The results do not depend on
the number of threads.

{dlgtab:Reporting}

{phang}
{opt eform} and {opt irr} report exp(b) (incidence-rate ratios). Typing {cmd:feppml} without arguments
replays the results and accepts the reporting options.


{marker ignored}{...}
{title:Accepted but ignored, and unsupported options}

{pstd}
Solver-tuning options are accepted and ignored with a note: {opt technique()}, {opt transform()},
{opt acceleration()} and {opt poolsize()}.

{pstd}
The following {cmd:ppmlhdfe} options are not available and stop with an error: {cmd:guess(variable} {it:varname}{cmd:)},
{opt r2}, {opt timeit}, and the undocumented advanced options of the
{cmd:ppmlhdfe} Mata class (e.g. {opt simplex_tol()}, {opt use_exact_solver}). {cmd:aweight}s are not allowed
(as in {cmd:ppmlhdfe}).


{marker differences}{...}
{title:Differences from ppmlhdfe}

{phang2}o {cmd:simplex} separation is an alias of {cmd:relu}: {cmd:feppml} does not implement
{cmd:ppmlhdfe}'s simplex method (which needs the {cmd:moremata} package). The ReLU method alone detects all
separated observations; the default {cmd:fe simplex relu} therefore runs {cmd:fe} and {cmd:relu}.{p_end}
{phang2}o {cmd:feppml}'s ReLU check can find separated observations that {cmd:ppmlhdfe} 2.3.x does not detect.
{cmd:feppml} then drops them, so {cmd:e(N)}, {cmd:e(num_separated)}, {cmd:e(df)} and {cmd:e(ll)} differ from
{cmd:ppmlhdfe}'s (observations that become singletons after the drop are also dropped), and the coefficients
differ slightly. For example, on a firm panel with firm and industry-by-year fixed effects {cmd:feppml} dropped
20 observations that {cmd:ppmlhdfe} 2.3.0 kept, also with {cmd:separation(fe simplex relu mu)}; an exact
linear program confirmed that these 20 observations are separated.{p_end}
{phang2}o {cmd:e(ic)} (IRLS iterations) can differ by a few iterations: {cmd:ppmlhdfe}'s stopping rule also
looks at the realized accuracy of its iterative solver, whereas {cmd:feppml} uses the requested inner
tolerance. {cmd:e(ic2)} counts the iterations of {cmd:feppml}'s solver, which differ from {cmd:ppmlhdfe}'s.{p_end}
{phang2}o {opt d()}: {cmd:feppml}'s d is the exact sum of fixed effects of the final IRLS step: it is constant
within fixed-effect groups and the fitted means satisfy the first-order conditions sum(y - mu) = 0 within every
group. {cmd:ppmlhdfe} computes d from the working variable of the previous IRLS iteration, so its d (and the
{cmd:predict} statistics built on it) can differ from {cmd:feppml}'s by about the square root of
{opt tolerance()}, more at observations with tiny fitted means, and by much more in badly fitting models with
a large {opt offset()}. The coefficients are not affected.{p_end}
{phang2}o Saved fixed effects ({it:target}{cmd:=} and {cmd:savefe}) are normalized: every intercept fixed effect
other than the first (the one with the most categories) has a weighted mean of zero, and the first absorbs the
shift. Only their sum (d) is unique.{p_end}
{phang2}o {cmd:e(title)} links to this help file; {cmd:e(footnote)} is {cmd:_rf_footnote};
{cmd:e(estat_cmd)} is {cmd:fereg_estat} ({cmd:ppmlhdfe}'s is {cmd:reghdfe5_estat}); both provide
{cmd:estat summarize}, {cmd:estat vce} and {cmd:estat ic} (with e(df_m) + 1 parameters).{p_end}
{phang2}o String variables are allowed inside fixed-effect and cluster interactions.{p_end}
{phang2}o {opt zvarname()} without {opt tagsep()} is an error ({cmd:ppmlhdfe} ignores it). With {opt tagsep()},
the variables are labeled {cmd:[feppml: ReLU]} ...; {opt r2} (the verification regression of the certificate) is
not available.{p_end}


{marker postestimation}{...}
{title:Postestimation}

{p 8 16 2}{cmd:predict} {newvar} {ifin} [{cmd:,} {it:statistic}]{p_end}

{synoptset 16}{...}
{synopthdr:statistic}
{synoptline}
{synopt :{opt mu}}predicted mean, exp(xb + d); the default{p_end}
{synopt :{opt xb}}linear prediction without the fixed effects (includes {cmd:_cons} and the offset){p_end}
{synopt :{opt xbd}}xb + d{p_end}
{synopt :{opt eta}}same as {opt xbd}{p_end}
{synopt :{opt d}}sum of the fixed effects{p_end}
{synopt :{opt stdp}}standard error of xb{p_end}
{synopt :{opt dev:iance}}deviance residual, 2[y ln(y/mu) - (y - mu)]{p_end}
{synopt :{opt r:esponse}}y - mu{p_end}
{synopt :{opt p:earson}}(y - mu)/sqrt(mu){p_end}
{synopt :{opt a:nscombe}}Anscombe residual, 1.5(y^(2/3) - mu^(2/3))/mu^(1/6){p_end}
{synopt :{opt sc:ores}}score, y - mu{p_end}
{synopt :{opt w:orking}}working residual, (y - mu)/mu{p_end}
{synoptline}

{pstd}
All statistics except {opt xb} and {opt stdp} require {opt d()} when fixed effects are absorbed.
{opt cooksd}, {opt hat} and {opt likelihood} are not implemented (error 198), as in {cmd:ppmlhdfe}.
{help test}, {help lincom}, {help nlcom}, {help estimates} and {help margins} work as after {cmd:poisson}.
{cmd:estat summarize}, {cmd:estat vce} and {cmd:estat ic} work as after {cmd:ppmlhdfe}.


{marker examples}{...}
{title:Examples}

{phang2}{cmd:. sysuse auto, clear}{p_end}
{phang2}{cmd:. feppml price weight gear length, absorb(turn trunk)}{p_end}
{phang2}{cmd:. feppml price weight gear length, absorb(turn) exposure(mpg) irr d(dhat)}{p_end}
{phang2}{cmd:. predict double muhat, mu}{p_end}
{phang2}{cmd:. webuse nlswork, clear}{p_end}
{phang2}{cmd:. feppml hours age ttl_exp tenure, absorb(idcode year occ_code) vce(cluster idcode)}{p_end}
{phang2}{cmd:. feppml hours age ttl_exp, absorb(fe1=idcode fe2=year) d(dsum) threads(8)}{p_end}


{marker results}{...}
{title:Stored results}

{pstd}
{cmd:feppml} stores the following in {cmd:e()}:

{synoptset 24 tabbed}{...}
{p2col 5 24 28 2: Scalars}{p_end}
{synopt:{cmd:e(N)}}number of observations{p_end}
{synopt:{cmd:e(N_full)}}number of observations, including dropped singleton and separated observations{p_end}
{synopt:{cmd:e(num_singletons)}}number of dropped singleton observations (including those separated by the {cmd:fe} rule){p_end}
{synopt:{cmd:e(num_separated)}}number of observations dropped by the {cmd:relu}/{cmd:mu} separation checks{p_end}
{synopt:{cmd:e(separation_complete)}}1 if every requested separation check finished; 0 if a check (e.g. ReLU at
its iteration limit) stopped before certifying its result (a note is printed), so separated observations may
remain in the sample{p_end}
{synopt:{cmd:e(drop_singletons)}}1 if singletons were dropped{p_end}
{synopt:{cmd:e(rank)}}rank of e(V){p_end}
{synopt:{cmd:e(df)}}residual degrees of freedom{p_end}
{synopt:{cmd:e(df_m)}}model degrees of freedom{p_end}
{synopt:{cmd:e(df_a)}}degrees of freedom absorbed by the fixed effects{p_end}
{synopt:{cmd:e(df_a_initial)}}number of fixed-effect categories{p_end}
{synopt:{cmd:e(df_a_redundant)}}number of redundant fixed-effect parameters{p_end}
{synopt:{cmd:e(df_a_nested)}}number of fixed-effect parameters nested within clusters{p_end}
{synopt:{cmd:e(N_hdfe)}}number of absorbed terms{p_end}
{synopt:{cmd:e(N_hdfe_extended)}}number of absorbed coefficient slots (intercepts and slopes){p_end}
{synopt:{cmd:e(rss)}}residual sum of squares of the final weighted least-squares step{p_end}
{synopt:{cmd:e(rmse)}}root mean squared error of the final weighted least-squares step{p_end}
{synopt:{cmd:e(chi2)}}Wald chi-squared statistic{p_end}
{synopt:{cmd:e(r2_p)}}pseudo R-squared, 1 - ll/ll_0{p_end}
{synopt:{cmd:e(ll)}}log pseudo-likelihood{p_end}
{synopt:{cmd:e(ll_0)}}log pseudo-likelihood of the constant-only model{p_end}
{synopt:{cmd:e(deviance)}}deviance{p_end}
{synopt:{cmd:e(ic)}}number of IRLS iterations{p_end}
{synopt:{cmd:e(ic2)}}number of fixed-effects solver iterations{p_end}
{synopt:{cmd:e(converged)}}1 if converged{p_end}
{synopt:{cmd:e(N_clust)}}number of clusters (minimum across cluster variables){p_end}
{synopt:{cmd:e(N_clust#)}}number of clusters of the #th cluster variable{p_end}
{synopt:{cmd:e(N_clustervars)}}number of cluster variables{p_end}

{synoptset 24 tabbed}{...}
{p2col 5 24 28 2: Macros}{p_end}
{synopt:{cmd:e(cmd)}}{cmd:feppml}{p_end}
{synopt:{cmd:e(cmdline)}}command as typed{p_end}
{synopt:{cmd:e(depvar)}}name of the dependent variable{p_end}
{synopt:{cmd:e(indepvars)}}names of the estimated regressors, plus {cmd:_cons}{p_end}
{synopt:{cmd:e(absvars)}}absorbed terms as typed{p_end}
{synopt:{cmd:e(extended_absvars)}}absorbed coefficient slots{p_end}
{synopt:{cmd:e(separation)}}separation methods used{p_end}
{synopt:{cmd:e(dofmethod)}}degrees-of-freedom adjustments used{p_end}
{synopt:{cmd:e(title)}}{cmd:HDFE PPML regression} ({cmd:PPML regression} without fixed effects){p_end}
{synopt:{cmd:e(title2)}}{cmd:Absorbing # HDFE group(s)}{p_end}
{synopt:{cmd:e(title3)}}{cmd:Statistics robust to heteroskedasticity} (clustered errors){p_end}
{synopt:{cmd:e(vce)}}{cmd:robust} or {cmd:cluster}{p_end}
{synopt:{cmd:e(vcetype)}}{cmd:Robust}{p_end}
{synopt:{cmd:e(clustvar)}}cluster variables{p_end}
{synopt:{cmd:e(clustvar#)}}#th cluster variable{p_end}
{synopt:{cmd:e(chi2type)}}{cmd:Wald}{p_end}
{synopt:{cmd:e(offset)}}offset variable, or ln({it:exposure}){p_end}
{synopt:{cmd:e(d)}}name of the variable saved by {opt d()}{p_end}
{synopt:{cmd:e(wtype)}}weight type{p_end}
{synopt:{cmd:e(wexp)}}weight expression{p_end}
{synopt:{cmd:e(predict)}}{cmd:feppml_p}{p_end}
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
{cmd:feppml} reproduces the conventions of {cmd:ppmlhdfe} by Sergio Correia, Paulo Guimarães and Thomas Zylkin;
its predict program and displays are adapted from {cmd:ppmlhdfe} (MIT license). Please cite ppmlhdfe when
reporting results:{p_end}

{phang}Correia, S., P. Guimarães, and T. Zylkin. 2020. Fast Poisson estimation with high-dimensional fixed effects.
{it:Stata Journal} 20(1): 95-115.{p_end}

{phang}Correia, S., P. Guimarães, and T. Zylkin. 2019. Verifying the existence of maximum likelihood estimates for
generalized linear models. arXiv:1903.01633.{p_end}
