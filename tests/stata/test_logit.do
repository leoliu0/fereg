* test_logit.do -- logitfast vs Stata's logit with explicit FE dummies (plan Phase 6.7)
*
* logitfast y x, absorb(fe1 fe2) is plain MLE, so its slope coefficients equal those of
* logit y x i.fe1 i.fe2. Compared: the slope block of b (rtol 1e-7) and V (rtol 1e-6, correlation
* scale), N, ll, ll_0, r2_p, e(sample) and the predicted probabilities (predict pr with d()).
* Data: the simulated separation-free panel of crates/regfast-core/tests/data/glm_logit.csv
* (n = 3000, plus a variant with two all-zero fe1 groups) and nlswork (idcode <= 300).
*
* Run from tests/stata (needs regfast.plugin on the adopath):
*   stata-mp -b do test_logit.do
version 16
clear all
set more off
capture confirm file "../../stata/logitfast.ado"
if (!_rc) adopath ++ "`c(pwd)'/../../stata"
which logitfast
findfile regfast.plugin
global RF_DATA "`c(pwd)'/../../crates/regfast-core/tests/data"
* rf_compare_bv() (Mata) is defined in rf_compare_e.ado
quietly run rf_compare_e.ado

* lcheck "<slope names>" "<logit arguments>" "<logitfast arguments>" ["<omitted names>"]
* (omitted names: coefficients both commands must report as omitted, o.<name>)
capture program drop lcheck
program lcheck
	args names lspec fspec omitted
	tempname bR VR bF VF
	tempvar sR pR pF dF
	quietly logit `lspec' tol(1e-12) ltol(1e-14) nrtol(1e-12)
	mata: rf_pick("`bR'", "`VR'", "`names'")
	CheckOmitted logit "`omitted'"
	local N = e(N)
	local ll = e(ll)
	local ll_0 = e(ll_0)
	local r2_p = e(r2_p)
	qui gen byte `sR' = e(sample)
	qui predict double `pR' if e(sample), pr
	logitfast `fspec' d(`dF') tol(1e-12)
	mata: rf_pick("`bF'", "`VF'", "`names'")
	CheckOmitted logitfast "`omitted'"
	rf_assert_close e(N) e(N) `N' 0
	rf_assert_close e(ll) e(ll) `ll' 1e-9
	rf_assert_close e(ll_0) e(ll_0) `ll_0' 1e-9
	rf_assert_close e(r2_p) e(r2_p) `r2_p' 1e-7
	qui count if `sR' != e(sample)
	if (r(N)) {
		di as err "FAIL e(sample): `r(N)' observations differ"
		exit 9
	}
	mata: rf_compare_bv("`bF'", "`VF'", "`bR'", "`VR'", 1e-7, 1e-6)
	if (`rf_fail') exit 9
	qui predict double `pF', pr
	qui gen double `dF'_diff = abs(`pF' - `pR')
	qui summarize `dF'_diff, meanonly
	if (r(max) > 1e-7) {
		di as err "FAIL predict pr: max abs diff " %9.2e r(max)
		exit 9
	}
	qui count if missing(`pF') != missing(`pR')
	if (r(N)) {
		di as err "FAIL predict pr: missing pattern differs in `r(N)' observations"
		exit 9
	}
	drop `dF'_diff
	di as text "lcheck OK: `fspec'"
end

capture program drop CheckOmitted
program CheckOmitted
	args cmd omitted
	local cn : colnames e(b)
	foreach v of local omitted {
		if (!strpos(" `cn' ", " o.`v' ")) {
			di as err "FAIL `cmd': `v' is not reported as omitted (o.`v') in e(b): `cn'"
			exit 9
		}
	}
end

capture mata: mata drop rf_pick()
mata:
// b and V of the coefficients named in `names' (equation names ignored)
void rf_pick(string scalar bname, string scalar Vname, string scalar names)
{
	string matrix stripe
	string rowvector want
	real rowvector idx
	real scalar i, j
	stripe = st_matrixcolstripe("e(b)")
	want = tokens(names)
	idx = J(1, cols(want), .)
	for (i = 1; i <= cols(want); i++) {
		for (j = 1; j <= rows(stripe); j++) {
			if (stripe[j, 2] == want[i]) idx[i] = j
		}
		if (missing(idx[i])) _error(3498, "coefficient " + want[i] + " not found")
	}
	st_matrix(bname, st_matrix("e(b)")[idx])
	st_matrix(Vname, st_matrix("e(V)")[idx, idx])
}
end

import delimited using "$RF_DATA/glm_logit.csv", clear asdouble varnames(1)
sort obs
lcheck "x1 x2" "y x1 x2 i.fe1 i.fe2," "y x1 x2, absorb(fe1 fe2)"
lcheck "x1 x2" "y x1 x2 i.fe1 i.fe2, vce(robust)" "y x1 x2, absorb(fe1 fe2) vce(robust)"
lcheck "x1 x2" "y x1 x2 i.fe1 i.fe2, vce(cluster c)" "y x1 x2, absorb(fe1 fe2) vce(cluster c)"
lcheck "x1 x2" "y x1 x2 i.fe1 i.fe2 [fw=fw]," "y x1 x2 [fw=fw], absorb(fe1 fe2)"
lcheck "x1 x2" "y x1 x2 i.fe1 i.fe2 [pw=pw]," "y x1 x2 [pw=pw], absorb(fe1 fe2)"
lcheck "x1 x2" "y x1 x2 i.fe1," "y x1 x2, absorb(fe1)"
lcheck "x1 x2 _cons" "y x1 x2," "y x1 x2, noabsorb"
lcheck "x1 x2 _cons" "y x1 x2, vce(robust)" "y x1 x2, noabsorb vce(robust)"
* two all-zero fe1 groups: logit drops them ("predicts failure perfectly"), logitfast's fe rule too
lcheck "x1 x2" "y2 x1 x2 i.fe1 i.fe2," "y2 x1 x2, absorb(fe1 fe2)"
rf_assert_close e(num_separated) e(num_separated) 100 0
* factor-variable regressors
lcheck "x1 2.c" "y x1 i.c i.fe2 if c <= 2," "y x1 i.c if c <= 2, absorb(fe2)"
* odds ratios only change the display
logitfast y x1 x2, absorb(fe1 fe2) or
di as result "simulated panel passed"

* ---------------------------------------------------------------------------
* nlswork (plan V-3): union on age (and ttl_exp) and grade, idcode and year FEs. grade is
* constant within idcode, so logitfast omits it (o.grade), as does logit when grade follows the
* FE dummies (_rmcoll keeps the earlier dummies); only the estimable slopes are compared.
* ---------------------------------------------------------------------------
webuse nlswork, clear
keep if idcode <= 300
lcheck "age" "union age i.idcode i.year grade," "union age grade, absorb(idcode year)" "grade"
lcheck "age" "union age i.idcode i.year grade, vce(cluster idcode)" "union age grade, absorb(idcode year) vce(cluster idcode)" "grade"
lcheck "age ttl_exp" "union age ttl_exp i.idcode i.year grade, vce(robust)" "union age ttl_exp grade, absorb(idcode year) vce(robust)" "grade"
* heterogeneous slopes: absorb(year##c.tenure) is i.year i.year#c.tenure in logit
lcheck "age ttl_exp" "union age ttl_exp i.idcode i.year i.year#c.tenure," "union age ttl_exp, absorb(idcode year##c.tenure)"
lcheck "age ttl_exp" "union age ttl_exp i.idcode i.ind_code#c.tenure, vce(cluster idcode)" "union age ttl_exp, absorb(idcode ind_code#c.tenure) vce(cluster idcode)"
di as result "nlswork passed"

* ---------------------------------------------------------------------------
* errors and predict
* ---------------------------------------------------------------------------
rcof "noi logitfast union age grade, absorb(idcode) sep(relu)" == 198
qui logitfast union age grade, absorb(idcode year)
rcof "noi predict double p0, pr" != 0
predict double xb0, xb
predict double s0, stdp
di as result "test_logit passed"
