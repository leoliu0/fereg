* test_iv.do -- regfast (IV/2SLS) vs ivreghdfe 1.1.4 (plan Phase 5.6)
*
* Every ivreghdfe run uses tol(1e-12). Assertions (rf_compare_e): b rtol 1e-7, V rtol 1e-6 (correlation
* scale), integer e() exact, other e() scalars rtol 1e-6 (idstat, cd, cdf, rkf, widstat, j, AR, ...),
* e() macros equal except command/version bookkeeping, e(dof_table) identical, e(sample) identical;
* with first/ffirst also e(first) (rtol 1e-6) and every stored first-stage regression.
*
* Run from tests/stata (needs regfast.plugin on the adopath and ivreghdfe/ranktest installed):
*   stata-mp -b do test_iv.do
version 16
clear all
set more off
capture confirm file "../../stata/regfast.ado"
if (!_rc) adopath ++ "`c(pwd)'/../../stata"
which regfast
which ivreghdfe
findfile regfast.plugin
di as text "plugin: `r(fn)'"

* Macros posted only by ivreghdfe/ivreg2 (command bookkeeping, options regfast does not have) or
* whose content legitimately differs (estimation-store names of the first stages)
global IV_XMACROS version ivreg2cmd firsteqs

* checkfirst R F : e(first) of two stored results agree (rtol 1e-6)
capture program drop checkfirst
program checkfirst
	args ref test
	tempname a b
	qui estimates restore `ref'
	matrix `a' = e(first)
	qui estimates restore `test'
	matrix `b' = e(first)
	if (rowsof(`a') != rowsof(`b') | colsof(`a') != colsof(`b')) {
		di as err "FAIL e(first) dimensions"
		exit 9
	}
	forval i = 1/`=rowsof(`a')' {
		forval j = 1/`=colsof(`a')' {
			rf_assert_close e(first)[`i',`j'] `=`b'[`i',`j']' `=`a'[`i',`j']' 1e-6
		}
	}
	local ra : rownames `a'
	local rb : rownames `b'
	local ca : colnames `a'
	local cb : colnames `b'
	if (`"`ra'"' != `"`rb'"' | `"`ca'"' != `"`cb'"') {
		di as err `"FAIL e(first) names: <`rb'|`cb'> vs <`ra'|`ca'>"'
		exit 9
	}
end

* check <arguments common to ivreghdfe and regfast> : estimate both and compare everything
capture program drop check
program check
	local spec `"`0'"'
	if (!strpos(`"`spec'"', ",")) local spec `"`spec',"'
	local first = strpos(`"`spec'"', " first") > 0
	local ffirst = strpos(`"`spec'"', " ffirst") > 0
	capture estimates drop _ivreg2_*
	* (savefirst alone would make ivreghdfe compute the first-stage statistics too)
	quietly ivreghdfe `spec' tol(1e-12) `=cond(`first', "savefirst", "")'
	estimates store R
	local reqs `e(firsteqs)'
	regfast `spec'
	estimates store F
	local feqs `e(firsteqs)'
	rf_compare_e R F, xscalars(ic) xmacros($IV_XMACROS)
	if (`first' | `ffirst') checkfirst R F
	if (`first') {
		local n1 : word count `reqs'
		local n2 : word count `feqs'
		if (`n1' != `n2') {
			di as err "FAIL number of first-stage regressions: `n2' vs `n1'"
			exit 9
		}
		forval i = 1/`n1' {
			rf_compare_e `: word `i' of `reqs'' `: word `i' of `feqs'', ///
				xmacros(cmd title vce vcetype hacsubtitleV properties firsteqs ivreg2cmd version predict estat_cmd marginsok marginsnotok) ///
				nodoftable nosample
		}
	}
end

* ---------------------------------------------------------------------------
* nlswork: ln_wage age (tenure = union south [msp]), absorb(idcode year)
* ---------------------------------------------------------------------------
webuse nlswork, clear
keep if idcode <= 1500
keep idcode year ln_wage age tenure union south msp hours ind_code
foreach v of varlist _all {
	qui drop if missing(`v')
}
gen double w = 0.5 + mod(idcode * 7 + year, 10) / 10
gen int fw = 1 + mod(idcode + year, 3)

foreach inst in "union south" "union south msp" {
	foreach vce in "" "vce(robust)" "vce(cluster idcode)" "vce(cluster idcode year)" {
		foreach wt in "" "[aw=w]" "[fw=fw]" "[pw=w]" {
			check ln_wage age (tenure = `inst') `wt', absorb(idcode year) `vce'
		}
	}
}
di as result "nlswork grid passed"

* first / ffirst
check ln_wage age (tenure = union south msp), absorb(idcode year) ffirst
check ln_wage age (tenure = union south msp), absorb(idcode year) vce(robust) first
check ln_wage age (tenure = union south msp) [aw=w], absorb(idcode year) vce(cluster idcode) first ffirst
check ln_wage age (tenure = union south msp), absorb(idcode year) vce(cluster idcode year) ffirst
* two endogenous regressors
check ln_wage age (tenure hours = union south msp), absorb(idcode year) first
check ln_wage age (tenure hours = union south msp) [pw=w], absorb(idcode year) vce(cluster idcode) ffirst
* factor-variable exogenous regressors, three absorbed FEs, interaction absvar
check ln_wage age i.msp (tenure = union south), absorb(idcode year)
check ln_wage age (tenure = union south msp), absorb(idcode year ind_code) vce(robust)
check ln_wage age (tenure = union south msp), absorb(idcode ind_code#year) vce(cluster idcode)
* no exogenous regressors
check ln_wage (tenure = union south), absorb(idcode year) vce(robust)
di as result "nlswork first/ffirst passed"

* ---------------------------------------------------------------------------
* Residuals and predict (xb, residuals, d, xbd, dresiduals) vs ivreghdfe/reghdfe_p
* ---------------------------------------------------------------------------
foreach vce in "vce(robust)" "vce(cluster idcode year)" {
	capture drop rR* rF* pR_* pF_*
	quietly ivreghdfe ln_wage age (tenure = union south msp) [aw=w], absorb(idcode year) `vce' tol(1e-12) residuals(rR)
	estimates store R
	foreach p in xb residuals d xbd dresiduals {
		quietly predict double pR_`p', `p'
	}
	regfast ln_wage age (tenure = union south msp) [aw=w], absorb(idcode year) `vce' residuals(rF)
	estimates store F
	rf_compare_e R F, xscalars(ic) xmacros($IV_XMACROS resid)
	foreach p in xb residuals d xbd dresiduals {
		quietly predict double pF_`p', `p'
	}
	foreach v in rR pR_xb pR_residuals pR_d pR_xbd pR_dresiduals {
		local f = subinstr("`v'", "R", "F", 1)
		qui count if missing(`v') != missing(`f')
		assert r(N) == 0
		qui gen double diff = abs(`v' - `f') / (1 + abs(`v'))
		qui summarize diff
		drop diff
		if (r(N) > 0) rf_assert_close max|`v'-`f'| `r(max)' 0 1e-7
	}
}
di as result "residuals/predict passed"

* ---------------------------------------------------------------------------
* Error paths
* ---------------------------------------------------------------------------
foreach o in liml gmm2s cue "fuller(1)" "kclass(0.5)" {
	capture regfast ln_wage age (tenure = union south), absorb(idcode year) `o'
	assert _rc == 198
}
capture regfast ln_wage age (tenure = union south), absorb(idcode year) vce(cluster idcode year ind_code)
assert _rc == 198
capture regfast ln_wage age (tenure hours = union), absorb(idcode year)
assert _rc == 481
capture regfast ln_wage age, absorb(idcode year) first
assert _rc == 198
di as result "IV error paths passed"

di as result "test_iv passed"
