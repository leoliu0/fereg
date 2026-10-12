{smcl}
{* *! version 0.2.0}{...}
{vieweralsosee "reghdfe" "help reghdfe"}{...}
{vieweralsosee "feppml" "help feppml"}{...}
{vieweralsosee "felogit" "help felogit"}{...}
{vieweralsosee "[R] areg" "help areg"}{...}
{viewerjumpto "Syntax" "fereg##syntax"}{...}
{viewerjumpto "Absorb syntax" "fereg##absorb"}{...}
{viewerjumpto "Description" "fereg##description"}{...}
{viewerjumpto "Options" "fereg##options"}{...}
{viewerjumpto "Accepted but ignored, and unsupported options" "fereg##ignored"}{...}
{viewerjumpto "Differences from reghdfe" "fereg##differences"}{...}
{viewerjumpto "Postestimation" "fereg##postestimation"}{...}
{viewerjumpto "Examples" "fereg##examples"}{...}
{viewerjumpto "Stored results" "fereg##results"}{...}
{viewerjumpto "Author" "fereg##author"}{...}
{title:Title}

{p2colset 5 16 18 2}{...}
{p2col :{cmd:fereg} {hline 2}}Fast linear regression with multiple high-dimensional fixed effects (compiled plugin){p_end}
{p2colreset}{...}


{marker syntax}{...}
{title:Syntax}

{p 8 15 2}{cmd:fereg}
{depvar} [{indepvars}]
{ifin} {weight}
[{cmd:,} {opth a:bsorb(fereg##absorb:absvars)} {help fereg##options_table:options}]
{p_end}

{pstd}
{cmd:fereg} is a drop-in replacement for {help reghdfe} 6.13.1 for linear models: same syntax, same
defaults, same output and the same stored results, computed by a parallel compiled solver.
{it:indepvars} may contain factor variables and time-series operators; see {help fvvarlist} and {help tsvarlist}.
{cmd:fweight}s, {cmd:aweight}s and {cmd:pweight}s are allowed; see {help weight}. The weight must be a variable.

{marker options_table}{...}
{synoptset 26 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Fixed effects}
{synopt :{opth a:bsorb(fereg##absorb:absvars)}}categorical variables (and interactions/slopes) to absorb{p_end}
{synopt :{cmdab:a:bsorb(}{it:...}{cmd:,} {cmdab:save:fe)}}save every fixed effect as {it:__hdfe#__} (also {cmd:generate}){p_end}
{synopt :{cmdab:a:bsorb(}{it:...}{cmd:,} {cmd:replace)}}overwrite existing target variables{p_end}
{synopt :{opt noa:bsorb}}absorb only the constant (same as omitting {opt absorb()}){p_end}

{syntab:Model}
{synopt :{opth vce:(fereg##vce:vcetype)}}{opt un:adjusted} (default), {opt r:obust}, or {opt cl:uster} {it:clustervars} (multi-way allowed){p_end}
{synopt :{opth cl:uster(varlist)}}synonym of {cmd:vce(cluster} {it:varlist}{cmd:)}{p_end}
{synopt :{opth res:iduals(newvar)}}save the regression residuals; required by several {help fereg##postestimation:predict} options{p_end}
{synopt :{opt res:iduals}}save the residuals in {it:_reghdfe_resid} (overwritten if it exists){p_end}
{synopt :{opt nocons:tant}}do not report the constant{p_end}
{synopt :{opt keepsin:gletons}}do not drop singleton groups (not recommended){p_end}

{syntab:Degrees of freedom}
{synopt :{opth dof:adjustments(fereg##dof:list)}}{opt all} (default), {opt none}, or any of {opt first:pair}|{opt pair:wise}, {opt cl:usters}, {opt cont:inuous}{p_end}
{synopt :{opth groupv:ar(newvar)}}save the identifier of the first mobility group{p_end}

{syntab:Optimization and performance}
{synopt :{opt tol:erance(#)}}convergence tolerance of the fixed-effects solver; default {cmd:1e-8}{p_end}
{synopt :{opt iter:ate(#)}}maximum number of solver iterations; default {cmd:16000}{p_end}
{synopt :{opt thr:eads(#)}}number of threads; default {cmd:0} = all logical CPUs{p_end}
{synopt :{opt serial:io}}read the data from Stata with a single thread{p_end}
{synopt :{opt v:erbose(#)}}0 (default), 1 or 2 (per-phase timings); {cmd:-1} suppresses the notes{p_end}
{synopt :{opt nowarn}}do not warn about {opt keepsingletons}{p_end}

{syntab:Reporting}
{synopt :{opt l:evel(#)}}set confidence level; default is {cmd:level(95)}{p_end}
{synopt :{opt nohe:ader}}suppress the header{p_end}
{synopt :{opt notab:le}}suppress the coefficient table{p_end}
{synopt :{opt nofoot:note}}suppress the table of absorbed degrees of freedom{p_end}
{synopt :{it:display_options}}see {help estimation options##display_options:[R] estimation options}{p_end}
{synoptline}
{p2colreset}{...}


{marker absorb}{...}
{title:Absorb syntax}

{p 8 15 2}{opt a:bsorb(absvar [absvar ...] [, savefe replace])}{p_end}

{pstd}
Each {it:absvar} is {cmd:[}{it:target}{cmd:=]}{it:term}, where {it:term} is one of

{p2colset 9 32 34 2}{...}
{p2col:{it:x} or {cmd:i.}{it:x}}one fixed effect per category of {it:x}{p_end}
{p2col:{it:x}{cmd:#}{it:y}[{cmd:#}{it:z}...]}one fixed effect per combination of {it:x} and {it:y} ({cmd:i.} prefixes are optional){p_end}
{p2col:{it:x}{cmd:##}{it:y}[{cmd:##}{it:z}...]}same as {it:x}{cmd:#}{it:y}[{cmd:#}{it:z}...]: the main effects are nested in the interaction (as in {cmd:reghdfe}; {cmd:e(absvars)} keeps the term as typed){p_end}
{p2col:{it:x}{cmd:#c.}{it:z}}a slope on {it:z} per category of {it:x} (no intercept){p_end}
{p2col:{it:x}{cmd:##c.}{it:z}}an intercept and a slope on {it:z} per category of {it:x}{p_end}
{p2col:{it:x}{cmd:##c.(}{it:z w}{cmd:)}}an intercept and slopes on {it:z} and {it:w} per category of {it:x}{p_end}
{p2colreset}{...}

{pstd}
Categorical variables may be numeric or string; string variables may also appear inside interactions.
Slope variables must be numeric. Time-series operators are allowed ({cmd:L.}{it:x}, {cmd:c.L.}{it:z}).
Variable abbreviations are expanded.

{pstd}
{it:target}{cmd:=} saves the estimated fixed effect in the new variable {it:target}; a term with slopes also
creates {it:target}{cmd:Slope1}, {it:target}{cmd:Slope2}, .... The suboption {cmd:savefe} saves every
term ({it:__hdfe1__}, {it:__hdfe2__Slope1}, ...). Saved fixed effects are labeled {cmd:[FE]} {it:term}.
As in {cmd:reghdfe}, variables named {it:__hdfe*} are dropped at the start of every {cmd:fereg} call.
{cmd:replace} overwrites existing target variables.


{marker description}{...}
{title:Description}

{pstd}
{cmd:fereg} estimates linear regressions with any number of absorbed fixed effects, heterogeneous slopes,
and unadjusted, robust or multi-way clustered standard errors, following the conventions of {cmd:reghdfe}
6.13.1: the same sample (singletons are dropped iteratively), the same degrees-of-freedom corrections
({it:absorbed degrees of freedom} table), the same collinearity rules and messages, the same small-sample
corrections, and the same {cmd:e()} results, so that {cmd:esttab}, {cmd:outreg2}, {cmd:margins},
{cmd:test} and {cmd:lincom} work unchanged.

{pstd}
The fixed effects are partialled out by a compiled solver: the largest fixed effect is eliminated exactly,
and the remaining ones are solved by preconditioned conjugate gradient on cell-compressed sufficient
statistics, with all kernels parallel over observations. Results are bit-for-bit identical for any number
of threads.


{marker options}{...}
{title:Options}

{dlgtab:Fixed effects}

{phang}
{opt absorb(absvars)} lists the fixed effects; see {help fereg##absorb:absorb syntax}.
Without {opt absorb()} (or with {opt noabsorb}) only the constant is absorbed and {cmd:fereg} is a
plain least-squares regression.

{marker vce}{...}
{dlgtab:Model}

{phang}
{opt vce(vcetype)}: {opt un:adjusted} (also {opt ols}) is the default; {opt r:obust} gives
heteroskedasticity-robust (Huber/White) errors; {opt cl:uster} {it:clustervars} gives cluster-robust errors.
Several cluster variables give multi-way clustering (Cameron, Gelbach and Miller, 2011), including the
adjustment applied when the variance matrix is not positive semidefinite. A cluster variable may be an
interaction such as {it:firm#year}, and may be a string variable. {cmd:pweight}s imply {cmd:vce(robust)} and
cannot be combined with {cmd:vce(unadjusted)}.

{phang}
{opt residuals(newvar)} saves the residuals; {opt residuals} without a name uses {it:_reghdfe_resid}.

{phang}
{opt noconstant} suppresses the reported constant. By default, as in {cmd:reghdfe}, {cmd:fereg} reports a
constant (the mean of the fixed effects) whenever a fixed effect has an intercept, so that {cmd:margins} works.

{phang}
{opt keepsingletons} keeps singleton groups; they do not change the estimates but bias the standard errors.

{marker dof}{...}
{dlgtab:Degrees of freedom}

{phang}
{opt dofadjustments(list)} selects how redundant fixed-effect parameters are counted, exactly as in
{cmd:reghdfe}: {opt pairwise} counts the connected components between every pair of intercept fixed effects
({opt firstpair} only between the first two), {opt clusters} treats fixed effects nested within a cluster
variable as fully redundant, and {opt continuous} counts categories whose slope variable is constant.
{opt all} (default) is {cmd:pairwise clusters continuous}; {opt none} disables the three adjustments.

{phang}
{opt groupvar(newvar)} saves the identifier of the mobility group (connected component) between the first two
intercept fixed effects. The variable is not created when there is no such pair.

{dlgtab:Optimization and performance}

{phang}
{opt tolerance(#)} sets the convergence criterion of the solver: iterations stop when the estimated relative
error of every partialled-out variable (in the weighted sum-of-squares norm) is below {it:#}.
{opt iterate(#)} caps the number of iterations; reaching it stops with error 430.

{phang}
{opt threads(#)} sets the number of threads used by the plugin; 0 (default) uses all logical CPUs.
The results do not depend on the number of threads.

{phang}
{opt serialio} reads the data from Stata with a single thread.

{phang}
{opt verbose(#)}: 1 prints solver diagnostics; 2 also prints per-phase timings; -1 suppresses notes and warnings.


{marker ignored}{...}
{title:Accepted but ignored, and unsupported options}

{pstd}
Solver-tuning options of {cmd:reghdfe} are accepted and ignored with a note, because {cmd:fereg} uses its own
solver: {opt technique()}, {opt transform()}, {opt acceleration()}, {opt preconditioner()}, {opt poolsize()} and
{opt compact}.

{pstd}
The following {cmd:reghdfe} features are not supported and stop with error 198:
{opt individual()}, {opt group()} and {opt aggregation()} (group fixed effects), and {cmd:vce(dkraay)}
(Driscoll-Kraay). Instrumental-variable syntax ({it:depvar} [{it:exog}] {cmd:(}{it:endog} {cmd:=} {it:instruments}{cmd:)})
and the options {opt first} and {opt ffirst} are not available in this version.


{marker differences}{...}
{title:Differences from reghdfe}

{phang2}o Saved fixed effects ({it:target}{cmd:=} and {cmd:savefe}) are normalized: every intercept fixed effect
other than the first (the one with the most categories) has a weighted mean of zero, and the first absorbs the
shift. Only their sum per observation (the {cmd:d} prediction) is unique; how the sum splits across fixed effects
depends on the algorithm and differs from {cmd:reghdfe}'s split.{p_end}
{phang2}o {cmd:e(ic)} reports the iterations of {cmd:fereg}'s solver, which differ from {cmd:reghdfe}'s.
Estimation messages read "(fereg converged in # iterations)".{p_end}
{phang2}o String variables are allowed inside fixed-effect and cluster interactions ({cmd:reghdfe} rejects them).{p_end}
{phang2}o {cmd:e(estat_cmd)} is {cmd:fereg_estat} (OLS), which provides the subcommands of {cmd:reghdfe_estat}
({cmd:estat summarize}, {cmd:estat vce}, {cmd:estat ic}); unlike {cmd:reghdfe_estat}, {cmd:estat summarize}
passes its variable list and options on to {help estat summarize}. After IV estimation it is not set, as after
{cmd:ivreghdfe}.{p_end}
{phang2}o When the model fits the data exactly (residual sum of squares at most 1e-20 times the total sum of
squares), {cmd:fereg} prints a note after the table that the standard errors are not meaningful
({cmd:reghdfe} prints nothing). The stored results are not affected.{p_end}
{phang2}o The mobility-group identifiers saved by {opt groupvar()} describe the same groups as {cmd:reghdfe}'s but
may be numbered differently.{p_end}
{phang2}o Notes that the plugin prints (e.g. the multi-way clustering adjustment) may appear before the
collinearity notes rather than after them.{p_end}


{marker postestimation}{...}
{title:Postestimation}

{p 8 16 2}{cmd:predict} {newvar} {ifin} [{cmd:,} {it:statistic}]{p_end}

{synoptset 16}{...}
{synopthdr:statistic}
{synoptline}
{synopt :{opt xb}}fitted values without the fixed effects; the default{p_end}
{synopt :{opt stdp}}standard error of the prediction (of the xb component){p_end}
{synopt :{opt r:esiduals}}residuals (requires {opt residuals()}){p_end}
{synopt :{opt d}}sum of the fixed effects, {it:y} - xb - residual (requires {opt residuals()}){p_end}
{synopt :{opt xbd}}xb + d = {it:y} - residual (requires {opt residuals()}){p_end}
{synopt :{opt dr:esiduals}}{it:y} - xb (requires {opt residuals()}){p_end}
{synopt :{opt sc:ore}}same as {opt residuals}{p_end}
{synoptline}

{pstd}
{help test}, {help lincom}, {help nlcom}, {help estimates} and {help margins} work as after {cmd:regress}
({cmd:margins} uses the xb component; {cmd:residuals} and {cmd:score} are not allowed in {cmd:margins}).
{cmd:estat summarize}, {cmd:estat vce} and {cmd:estat ic} work as after {cmd:reghdfe} ({cmd:estat ic} counts
e(df_m) + 1 parameters). Typing {cmd:fereg} without arguments replays the results.


{marker examples}{...}
{title:Examples}

{phang2}{cmd:. webuse nlswork, clear}{p_end}
{phang2}{cmd:. fereg ln_wage age ttl_exp tenure, absorb(idcode year occ_code) vce(cluster idcode)}{p_end}
{phang2}{cmd:. fereg ln_wage age i.race c.age#c.age, absorb(idcode year) vce(cluster idcode year)}{p_end}
{phang2}{cmd:. fereg ln_wage tenure, absorb(fe1=idcode fe2=year##c.age) residuals(res)}{p_end}
{phang2}{cmd:. predict double dhat, d}{p_end}
{phang2}{cmd:. fereg ln_wage tenure [aw=hours], absorb(idcode ind_code#year) threads(8)}{p_end}


{marker results}{...}
{title:Stored results}

{pstd}
{cmd:fereg} stores the following in {cmd:e()}:

{synoptset 24 tabbed}{...}
{p2col 5 24 28 2: Scalars}{p_end}
{synopt:{cmd:e(N)}}number of observations{p_end}
{synopt:{cmd:e(N_full)}}number of observations before dropping singletons{p_end}
{synopt:{cmd:e(num_singletons)}}number of singleton observations dropped{p_end}
{synopt:{cmd:e(drop_singletons)}}1 if singletons were dropped{p_end}
{synopt:{cmd:e(N_hdfe)}}number of absorbed terms ({cmd:1} without {opt absorb()}){p_end}
{synopt:{cmd:e(N_hdfe_extended)}}number of absorbed coefficient slots (intercepts and slopes){p_end}
{synopt:{cmd:e(df_a)}}degrees of freedom absorbed by the fixed effects{p_end}
{synopt:{cmd:e(df_a_initial)}}number of fixed-effect categories{p_end}
{synopt:{cmd:e(df_a_redundant)}}number of redundant fixed-effect parameters{p_end}
{synopt:{cmd:e(df_a_nested)}}number of fixed-effect parameters nested within clusters{p_end}
{synopt:{cmd:e(rank)}}rank of e(V) excluding the constant{p_end}
{synopt:{cmd:e(df_m)}}model degrees of freedom{p_end}
{synopt:{cmd:e(df_r)}}residual degrees of freedom{p_end}
{synopt:{cmd:e(tss)}}total sum of squares{p_end}
{synopt:{cmd:e(tss_within)}}sum of squares of the partialled-out dependent variable{p_end}
{synopt:{cmd:e(rss)}}residual sum of squares{p_end}
{synopt:{cmd:e(mss)}}model sum of squares (tss - rss){p_end}
{synopt:{cmd:e(r2)}}R-squared{p_end}
{synopt:{cmd:e(r2_a)}}adjusted R-squared{p_end}
{synopt:{cmd:e(r2_within)}}within R-squared{p_end}
{synopt:{cmd:e(r2_a_within)}}adjusted within R-squared{p_end}
{synopt:{cmd:e(rmse)}}root mean squared error{p_end}
{synopt:{cmd:e(F)}}F statistic{p_end}
{synopt:{cmd:e(ll)}}log likelihood{p_end}
{synopt:{cmd:e(ll_0)}}log likelihood of the fixed-effects-only model{p_end}
{synopt:{cmd:e(sumweights)}}sum of the weights{p_end}
{synopt:{cmd:e(ic)}}number of solver iterations{p_end}
{synopt:{cmd:e(converged)}}1 if the solver converged{p_end}
{synopt:{cmd:e(report_constant)}}1 if the constant is reported{p_end}
{synopt:{cmd:e(N_clust)}}number of clusters (minimum across cluster variables){p_end}
{synopt:{cmd:e(N_clust#)}}number of clusters of the #th cluster variable{p_end}
{synopt:{cmd:e(N_clustervars)}}number of cluster variables{p_end}

{synoptset 24 tabbed}{...}
{p2col 5 24 28 2: Macros}{p_end}
{synopt:{cmd:e(cmd)}}{cmd:fereg}{p_end}
{synopt:{cmd:e(cmdline)}}command as typed{p_end}
{synopt:{cmd:e(model)}}{cmd:ols}{p_end}
{synopt:{cmd:e(depvar)}}name of the dependent variable{p_end}
{synopt:{cmd:e(indepvars)}}names of the regressors sent to the solver, plus {cmd:_cons}{p_end}
{synopt:{cmd:e(absvars)}}absorbed terms as typed{p_end}
{synopt:{cmd:e(extended_absvars)}}absorbed coefficient slots{p_end}
{synopt:{cmd:e(dofmethod)}}degrees-of-freedom adjustments used{p_end}
{synopt:{cmd:e(title)}}{cmd:HDFE Linear regression}{p_end}
{synopt:{cmd:e(title2)}}{cmd:Absorbing # HDFE group(s)}{p_end}
{synopt:{cmd:e(title3)}}{cmd:Statistics robust to heteroskedasticity} (clustered errors){p_end}
{synopt:{cmd:e(vce)}}{cmd:ols}, {cmd:robust} or {cmd:cluster}{p_end}
{synopt:{cmd:e(vcetype)}}{cmd:Robust} for robust and clustered errors{p_end}
{synopt:{cmd:e(clustvar)}}cluster variables{p_end}
{synopt:{cmd:e(clustvar#)}}#th cluster variable{p_end}
{synopt:{cmd:e(wtype)}}weight type{p_end}
{synopt:{cmd:e(wexp)}}weight expression{p_end}
{synopt:{cmd:e(resid)}}name of the residuals variable{p_end}
{synopt:{cmd:e(predict)}}{cmd:fereg_p}{p_end}
{synopt:{cmd:e(estat_cmd)}}{cmd:fereg_estat} (OLS){p_end}
{synopt:{cmd:e(footnote)}}{cmd:_rf_footnote}{p_end}
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
{cmd:fereg} reproduces the conventions of {cmd:reghdfe} by Sergio Correia; its header and footnote displays are
adapted from {cmd:reghdfe} (MIT license). Please cite reghdfe when reporting results:{p_end}

{phang}Correia, S. 2016. Linear Models with High-Dimensional Fixed Effects: An Efficient and Feasible Estimator.
Working paper.{p_end}
