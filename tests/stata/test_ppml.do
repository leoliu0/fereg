* test_ppml.do -- feppml vs ppmlhdfe 2.3.3 (plan Phase 6.7)
*
* Datasets: the ppmlhdfe test-suite data ported in crates/fereg-core/tests/data/glm_*.csv
*   (MIT License, Copyright (c) 2019 Sergio Correia, Paulo Guimarães, and Thomas Zylkin; see
*   gen_glm_fixtures.do) and inline (test/slopes1-4.do), auto, nlswork and a simulated gravity
*   panel.
* Every ppmlhdfe run uses tol(1e-10) or tighter and sep(fe relu) (fereg treats `simplex` as an
* alias of `relu`; ppmlhdfe's simplex presolve needs moremata).
* Assertions (rf_compare_e): b rtol 1e-7, V rtol 1e-6 (correlation scale), integer e() exact
* (N, num_separated, num_singletons, df, ...), other e() scalars rtol 1e-6 (ll, deviance, r2_p,
* chi2, ...), all e() macros equal, e(dof_table) identical, e(sample) identical; d and the sum of
* saved FEs rtol 1e-6 (scaled by max|d|); every predict option.
*
* Run from tests/stata (needs fereg.plugin on the adopath and ppmlhdfe installed):
*   stata-mp -b do test_ppml.do
version 16
clear all
set more off
capture confirm file "../../stata/feppml.ado"
if (!_rc) adopath ++ "`c(pwd)'/../../stata"
which feppml
which ppmlhdfe
findfile fereg.plugin
global RF_DATA "`c(pwd)'/../../crates/fereg-core/tests/data"

* e() items that legitimately differ: ic2 counts solver sub-iterations; ic can differ by a few
* iterations because ppmlhdfe's stopping rule also looks at the realized accuracy of its MAP
* solver (HDFE.accuracy), which the fereg absorber does not report (it uses the requested inner
* tolerance, an upper bound); e(title) links to the feppml help file.
global PX ic ic2
global PM title

* check <arguments common to ppmlhdfe and feppml> [|| xs(<more excluded scalars>) <rf_compare_e options>]
capture program drop check
program check
	SplitSpec `0'
	quietly ppmlhdfe `spec'
	estimates store R
	feppml `spec'
	estimates store F
	rf_compare_e R F, xscalars($PX `xs') xmacros($PM) `cmp'
end

capture program drop SplitSpec
program SplitSpec
	gettoken spec rest : 0, parse("|")
	local cmp
	if (`"`rest'"' != "") {
		gettoken bar rest : rest, parse("|")
		gettoken bar cmp : rest, parse("|")
	}
	local 0 , `cmp'
	syntax [, xs(string) dtol(real 1e-6) *]
	local spec `"`spec'"'
	if (!strpos(`"`spec'"', ",")) local spec `"`spec',"'
	if (!strpos(`"`spec'"', "sep(")) local spec `"`spec' sep(fe relu)"'
	c_local spec `"`spec'"'
	c_local xs `xs'
	c_local dtol `dtol'
	c_local cmp `options'
end

* dcheck <arguments> [|| dtol(#) ...] : like check, plus d() and every predict option (rtol dtol)
capture program drop dcheck
program dcheck
	SplitSpec `0'
	capture drop __dR __dF __pR* __pF*
	quietly ppmlhdfe `spec' d(__dR)
	estimates store R
	foreach o in mu xb xbd eta deviance response pearson anscombe scores working {
		quietly predict double __pR_`o', `o'
	}
	feppml `spec' d(__dF)
	estimates store F
	rf_compare_e R F, xscalars($PX `xs') xmacros($PM d) `cmp'
	cmpvar d __dF __dR `dtol'
	foreach o in mu xb xbd eta deviance response pearson anscombe scores working {
		quietly predict double __pF_`o', `o'
		cmpvar "predict `o'" __pF_`o' __pR_`o' `dtol'
	}
	drop __dR __dF __pR* __pF*
end

* cmpvar name a b rtol : max|a-b| <= rtol * max(max|b|, 1) over observations where b is nonmissing,
* and a is missing exactly where b is
capture program drop cmpvar
program cmpvar
	args name a b rtol
	qui count if missing(`a') != missing(`b')
	if (r(N)) {
		di as err "FAIL `name': missing pattern differs in `r(N)' observations"
		exit 9
	}
	tempvar diff absb
	qui gen double `diff' = abs(`a' - `b')
	qui gen double `absb' = abs(`b')
	qui summarize `absb', meanonly
	local scale = max(r(max), 1)
	qui summarize `diff', meanonly
	if (r(N) & r(max) > `rtol' * `scale') {
		di as err "FAIL `name': max abs diff " %9.2e r(max) " (scale " %9.2e `scale' ", rtol `rtol')"
		exit 9
	}
end

capture program drop loaddata
program loaddata
	args name
	import delimited using "$RF_DATA/glm_`name'.csv", clear asdouble varnames(1)
	sort obs
end

* ---------------------------------------------------------------------------
* ppmlhdfe test-suite datasets
* ---------------------------------------------------------------------------
loaddata problem1
check y x1 x2, absorb(i j) tol(1e-10) keepsing sep(mu)
check y x1 x2 if !inlist(_n, 5, 7, 8, 14, 15), absorb(i j) tol(1e-10)
loaddata problem2
check y x1 x2, absorb(i j) tol(1e-10)
loaddata problem3
check y x1 x2 x3, noa tol(1e-12)
loaddata problem4
check y x1 x2 x3, noa tol(1e-12) maxiter(200)
loaddata problem5
check y x1 x2, absorb(i j) tol(1e-12)
* problem6: one observation with x = -716 makes ppmlhdfe's e(ll)/e(deviance) inexact (4e-3);
* tests/glm.rs checks the exact values recomputed from ppmlhdfe's own b.
loaddata problem6
check y x, noa tol(1e-10) || xs(ll deviance r2_p)
loaddata hard1
check y, absorb(id1 id2) tol(1e-10)
* hard2 (keepsingletons) fits the data exactly (deviance 0): _cons is 0 up to rounding, and
* ppmlhdfe's exact 0 makes _ms_findomitted rename it o._cons while fereg gets -2.2e-16; the
* values are checked in tests/glm.rs (p_hard2_keepsing).
loaddata hard3
check y, absorb(id1 id2) tol(1e-10)
loaddata hard5
check y x1 x2 x3 x4, noa tol(1e-10)
check y x2 x3 x4, noa tol(1e-10)
check y x2 x3 x4 if _n > 2, noa tol(1e-10)
loaddata collinear1
check y x z, absorb(im ex) tol(1e-10)
loaddata collinear2
check y x1 x2, noa tol(1e-10)
check y x1, noa tol(1e-10)
loaddata collinear3
check y x1 x2, noa tol(1e-10)
loaddata collinear4
check y x1 x2, noa tol(1e-10)
loaddata collinear5
check y x1 x2, noa tol(1e-10)
* collinear6: x2 = x1 + e/18000; V is accurate only to ~1e-6 in any implementation and
* ppmlhdfe reports a missing chi2 (rounding noise in its sweep)
loaddata collinear6
check y x1 x2, noa tol(1e-12) || vtol(1e-5) xs(chi2)
di as result "ppmlhdfe test-suite datasets passed"

* ---------------------------------------------------------------------------
* auto: two-way FEs, VCE, weights, exposure/offset, no regressors, savefe, predict
* ---------------------------------------------------------------------------
sysuse auto, clear
gen double W = foreign + 1
gen double m10 = mpg / 10
check price weight gear length, absorb(turn trunk) vce(robust) keepsingletons tol(1e-12)
check price weight gear length, absorb(turn trunk) vce(cluster turn trunk) keepsingletons tol(1e-12)
check price weight gear length, absorb(turn trunk) vce(cluster turn) tol(1e-12)
check price weight gear length [fw=displacement], noabsorb tol(1e-10)
check price weight gear length [fw=displacement], noabsorb vce(cluster turn) tol(1e-10)
check price weight gear length [fw=turn], noabsorb vce(cluster trunk) tol(1e-10)
check price weight gear length [fw=displacement], absorb(turn) keepsing tol(1e-10)
check price weight gear length [pw=displacement], noabsorb tol(1e-10)
check price weight gear length [pw=displacement], noabsorb vce(cluster turn) tol(1e-10)
check price weight gear length [pw=displacement], absorb(turn) keepsing tol(1e-10)
check price weight gear length, noa exposure(headroom) tol(1e-12)
check price weight gear length, absorb(turn) exposure(headroom) tol(1e-10) keepsingletons
check price weight gear length, offset(mpg) tol(1e-12) noa
check price weight gear length [fw=W], offset(mpg) tol(1e-10)
check price weight gear length, absorb(turn) offset(mpg) tol(1e-10) keepsingletons
check price, noabsorb tol(1e-10)
check price, absorb(turn) tol(1e-12) keepsingletons
check price weight gear length, absorb(turn) guess(ols) tol(1e-10)
check price weight i.foreign, absorb(turn) tol(1e-10)
check price weight gear length, absorb(turn trunk) vce(robust) tol(1e-10) irr
dcheck price weight gear length, absorb(turn trunk) tol(1e-12)
* d() with an offset: ppmlhdfe computes d = z - resid - xb with the working variable z of the
* previous IRLS iteration, so its d is constant within FE groups only up to the last IRLS step
* (O(sqrt(tol)) in eta). That error is ~1e-4 with a single FE (exact partial-out, ppmlhdfe stops
* after few iterations) and explodes on degenerate fits such as offset(mpg) on price, where
* ppmlhdfe's mu is ~0 for some observations and its d varies by 1e4 within a turn group (its
* group first-order conditions fail; feppml's d is exactly constant within groups). The cases
* below are well-conditioned two-FE models where ppmlhdfe's d is accurate to ~1e-7.
dcheck price weight gear length, absorb(turn trunk) offset(m10) tol(1e-12)
dcheck price weight gear length, absorb(turn trunk) exposure(mpg) tol(1e-12)
dcheck price weight gear length, noa tol(1e-10)

* savefe: the sum of the saved FEs equals d
capture drop FE*
feppml price weight gear, absorb(FE1=turn FE2=foreign) vce(robust) tol(1e-10) keepsing d(dfe)
gen double FEsum = FE1 + FE2
cmpvar "saved FEs" FEsum dfe 1e-9
qui ppmlhdfe price weight gear, absorb(FX1=turn FX2=foreign) vce(robust) tol(1e-10) keepsing d(dfx) sep(fe relu)
cmpvar "d vs ppmlhdfe" dfe dfx 1e-6
drop FE1 FE2 FEsum dfe FX1 FX2 dfx
feppml price weight gear, absorb(turn foreign, savefe) tol(1e-10) keepsing
confirm double variable __hdfe1__ __hdfe2__
drop __hdfe*
di as result "auto passed"

* ---------------------------------------------------------------------------
* nlswork: 2-3 FEs on the fast absorber, robust / cluster / two-way cluster
* ---------------------------------------------------------------------------
webuse nlswork, clear
check hours age ttl_exp tenure, absorb(idcode year) tol(1e-10)
check hours age ttl_exp tenure, absorb(idcode year occ_code) vce(cluster idcode) tol(1e-10)
check hours age ttl_exp tenure, absorb(idcode year) vce(cluster idcode year) tol(1e-10)
check union age ttl_exp, absorb(idcode year) vce(cluster idcode) tol(1e-10)
dcheck hours age ttl_exp tenure, absorb(idcode year) tol(1e-10)
di as result "nlswork passed"

* ---------------------------------------------------------------------------
* simulated gravity panel (exporter#year, importer#year, pair FE; zeros separated by the pair FE)
* ---------------------------------------------------------------------------
loaddata gravity
check trade rta x2, absorb(exp#year imp#year exp#imp) vce(cluster exp#imp) tol(1e-10)
* ppmlhdfe's d here satisfies the FE first-order conditions sum_g(y - mu) = 0 only to ~2e-6 (its
* d uses the previous IRLS iteration's working variable and the MAP inner tolerance; the error
* does not shrink with tol()/itol()); it reaches 1.3e-5 at an observation with mu = 0.07.
* feppml's d satisfies them to ~1e-12, asserted below. predict working = (y - mu)/mu amplifies
* the difference at small mu to 7e-6 (relative to its max).
dcheck trade rta x2, absorb(exp#year imp#year exp#imp) tol(1e-10) || dtol(1e-5)
qui feppml trade rta x2, absorb(exp#year imp#year exp#imp) tol(1e-10) d(dG)
qui predict double muG, mu
gen double rG = trade - muG
foreach g in "exp year" "imp year" "exp imp" {
	capture drop focG
	qui bys `g': egen double focG = total(rG)
	qui su focG
	if (max(abs(r(min)), abs(r(max))) > 1e-9) {
		di as err "FAIL first-order conditions of d by `g': " %9.2e max(abs(r(min)), abs(r(max)))
		exit 9
	}
}
drop dG muG rG focG
di as result "gravity passed"

* ---------------------------------------------------------------------------
* heterogeneous slopes: ports of ppmlhdfe test/slopes1-4.do (regressors explained by slope FEs,
* separation through a slope term) and nlswork
* ---------------------------------------------------------------------------
* slopes1: fta is an exact combination of exporter##c.trend (omitted), perfect fit
clear
input double trade byte(fta exporter trend c)
35412.88183544995 0  8 15 1
619124.7429669127 0  8 16 1
 645947.355868591 0 35  0 1
2113023.937040806 1 35 12 1
end
check trade fta, absorb(exporter##c.trend c) maxiter(100) tol(1e-10)
* slopes2: x is an exact combination of id1##c.z
clear
input double(id1 y) byte(x z c)
1   67793.6092504498 0 -1 1
1              68774 0  0 1
2 29460066.818006903 1 -1 1
2  40370377.65078497 1 10 1
3  4215.552981376648 0  7 1
3  4288.904005765915 0 15 1
4  483882.9337809086 0  7 1
4 306164.57000368834 0  8 1
5 29557.571687340736 0  6 1
5  170794.3896008134 0 12 1
6 1963894.9483253956 0  6 1
6 3364687.2930059433 0 12 1
end
check y x, absorb(id1##c.z c) maxiter(100) tol(1e-12)
* slopes3: within id2 == 3 the intercept and z separate obs 7; obs 8 is then a singleton.
* ppmlhdfe's V is nonsymmetric by rounding here (residual df 1, near-perfect fit), so Stata zeroes
* it when posting and its chi2 is missing; feppml posts its symmetric V (about 1e-14).
clear
input byte(index id1 id2 x z) double y
1 1 1 0 -1  3369071263
2 1 1 0  2  2535746605
3 1 1 1 15 15241548895
4 1 2 0 11        1040
5 1 2 0 13         166
6 1 2 0 15    21682500
7 1 3 0 -1           0
8 1 3 0  7      764800
end
check y x, absorb(id1 id2##c.z) maxiter(100) tol(1e-10) keepsing || nov xs(chi2)
check y x, absorb(id1 id2##c.z) maxiter(100) tol(1e-10) || nov xs(chi2)
* slopes4: every y == 0 observation is separated, through combinations of i, j and j#c.z.
* ppmlhdfe 2.3.3 finds 3 of the 4 (its own test of this dataset is disabled), so feppml is
* checked for all four, and against ppmlhdfe on the y > 0 sample.
clear
input long y byte(x z i j)
      0 0 -1  1  1
   5211 0 -1  1  4
  10498 0  2  2  4
   1310 0  2  2  5
    189 0  3  3  2
      0 0  3  3  6
   1468 0  6  4  5
     34 0  6  4  7
  12000 0  7  5  3
   1036 0  7  5  5
      0 0  8  6  3
   1247 0  8  6  7
    460 0  9  7  1
     70 0  9  7  7
     28 0 14  8  2
      4 0 14  8  3
     17 0 14  8  6
      0 0 15  9  1
      8 0 15  9  6
 203886 0  3 10  8
3840369 1  3 10  9
3961747 1  8 11  9
 710771 0  8 11 10
 810565 0 13 12  8
 249014 0 13 12 10
 819845 0 16 13  8
5589636 1 16 13  9
end
quietly feppml y x, absorb(i j##c.z) maxiter(400) tol(1e-10) keepsing
assert e(num_separated) == 4
qui count if e(sample) != (y > 0)
assert r(N) == 0
check y x if y > 0, absorb(i j##c.z) maxiter(400) tol(1e-10) keepsing
* (after singletons, 8 observations that the model fits exactly: ppmlhdfe's MAP solves this tiny
* system to machine precision, so it accepts eps < tol at deviance 2e-8 before tightening the
* inner tolerance; feppml runs two more iterations to deviance 0, which moves _cons by 2e-6)
check y x if y > 0, absorb(i j##c.z) maxiter(400) tol(1e-10) || btol(1e-5)
matrix b1 = e(b)
local N1 = e(N)
quietly feppml y x, absorb(i j##c.z) maxiter(400) tol(1e-10)
assert e(num_separated) == 4 & e(N) == `N1'
assert mreldif(e(b), b1) < 1e-8
* nlswork: slope terms with and without intercepts, robust / cluster, d() and predict
webuse nlswork, clear
check hours age ttl_exp, absorb(idcode year##c.tenure) tol(1e-10)
check hours ttl_exp tenure, absorb(idcode##c.age year) vce(cluster idcode) tol(1e-10)
check hours age ttl_exp, absorb(idcode ind_code#c.tenure year) vce(robust) tol(1e-10)
dcheck hours age ttl_exp, absorb(idcode year##c.tenure) tol(1e-10)
di as result "slopes passed"

* ---------------------------------------------------------------------------
* errors
* ---------------------------------------------------------------------------
sysuse auto, clear
gen double yneg = price - 5000
rcof "noi feppml yneg weight, absorb(turn)" == 459
rcof "noi feppml price weight, absorb(turn) exposure(mpg) offset(mpg)" == 198
rcof "noi feppml price weight, absorb(turn) sep(bogus)" == 198
qui feppml price weight, absorb(turn) d(dH)
rcof "noi predict double h, hat" == 198
rcof "noi predict double h, cooksd" == 198
rcof "noi predict double h, likelihood" == 198
qui feppml price weight, absorb(turn)
rcof "noi predict double h, mu" != 0
predict double h, xb

di as result "test_ppml passed"
