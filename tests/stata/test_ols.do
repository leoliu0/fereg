* test_ols.do -- regfast (OLS) vs reghdfe 6.13.1 (plan Phase 4.7)
*
* Every reghdfe run uses tol(1e-12). Assertions (rf_compare_e): b rtol 1e-7, V rtol 1e-6 (correlation
* scale), integer e() exact, other e() scalars rtol 1e-6, all e() macros equal (except cmd, cmdline,
* predict, footnote, estat_cmd), e(dof_table) identical, e(sample) identical, omitted names identical.
*
* Run from tests/stata (needs regfast.plugin on the adopath and reghdfe installed):
*   stata-mp -b do test_ols.do
* Dry run against the mock plugin (checks this file and the ado, not the numbers):
*   mock/build.sh, then in Stata: global RF_MOCK 1, adopath ++ mock, do test_ols.do
version 16
clear all
set more off
capture confirm file "../../stata/regfast.ado"
if (!_rc) adopath ++ "`c(pwd)'/../../stata"
if ("$RF_MOCK" != "") adopath ++ "`c(pwd)'/mock"
which regfast
which reghdfe
findfile regfast.plugin
di as text "plugin: `r(fn)'"

* check <arguments common to reghdfe and regfast> : estimate both and compare everything
capture program drop check
program check
	local spec `"`0'"'
	if (!strpos(`"`spec'"', ",")) local spec `"`spec',"'
	quietly reghdfe `spec' tol(1e-12)
	estimates store R
	if ("$RF_MOCK" != "") {
		syntax varlist(fv ts) [if] [in] [fw aw pw] [, *]
		gettoken depvar indepvars : varlist, bind
		if ("`indepvars'" == "") local indepvars _none
		rf_stage_mock, indepvars(`indepvars')
	}
	regfast `spec'
	estimates store F
	* (the mock plugin does not drop singletons, so e(sample) is only compared with the real plugin)
	rf_compare_e R F, xscalars(ic) `=cond("$RF_MOCK" != "", "nosample", "")'
end

* ---------------------------------------------------------------------------
* auto: absorb(rep78), cluster rep78, weights
* ---------------------------------------------------------------------------
sysuse auto, clear
gen int fwv = 1 + mod(_n, 4)
check price weight length, absorb(rep78)
check price weight length, absorb(rep78) vce(robust)
check price weight length, absorb(rep78) vce(cluster rep78)
foreach w in "aw=mpg" "fw=fwv" "pw=mpg" {
	check price weight length [`w'], absorb(rep78)
	check price weight length [`w'], absorb(rep78) vce(robust)
	check price weight length [`w'], absorb(rep78) vce(cluster rep78)
}
check price weight length, absorb(rep78 foreign)
check price weight length, absorb(rep78 foreign) vce(cluster foreign rep78)
* no absorbed FEs
check price weight length
check price weight length, noabsorb vce(robust)
check price weight length [aw=mpg], noabsorb vce(cluster rep78)
di as result "auto passed"

* ---------------------------------------------------------------------------
* nlswork: 3 FEs, robust / cluster / two-way cluster, weights
* ---------------------------------------------------------------------------
webuse nlswork, clear
gen double w = 1 + mod(_n, 5)
gen int fw = 1 + mod(_n, 3)
check ln_wage age ttl_exp tenure, absorb(idcode year occ_code)
check ln_wage age ttl_exp tenure, absorb(idcode year occ_code) vce(robust)
check ln_wage age ttl_exp tenure, absorb(idcode year occ_code) vce(cluster idcode)
check ln_wage age ttl_exp tenure, absorb(idcode year occ_code) vce(cluster idcode year)
check ln_wage age ttl_exp tenure, absorb(idcode year occ_code) cluster(idcode ind_code)
check ln_wage age ttl_exp tenure [aw=w], absorb(idcode year occ_code) vce(cluster idcode)
check ln_wage age ttl_exp tenure [fw=fw], absorb(idcode year occ_code) vce(robust)
check ln_wage age ttl_exp tenure [pw=w], absorb(idcode year occ_code)
* cluster variable nested in / equal to an absorbed FE; dof() variants
check ln_wage age ttl_exp tenure, absorb(idcode year) vce(cluster idcode)
check ln_wage age ttl_exp tenure, absorb(idcode year) vce(cluster idcode) dof(none)
check ln_wage age ttl_exp tenure, absorb(idcode year occ_code) dof(firstpair)
check ln_wage age ttl_exp tenure, absorb(idcode year occ_code) vce(cluster idcode) dof(pairwise clusters)
check ln_wage age ttl_exp tenure, absorb(idcode year occ_code) dof(clusters continuous)
di as result "nlswork vce/weights passed"

* factor variables in the regressors
check ln_wage age i.race c.age#c.age, absorb(idcode year occ_code) vce(cluster idcode year)
check ln_wage i.race##c.tenure ib3.grade, absorb(year occ_code)
* string absvar and string cluster variable
gen sid = string(idcode)
check ln_wage age ttl_exp tenure, absorb(sid year) vce(cluster sid)
* interaction absvar
check ln_wage age ttl_exp tenure, absorb(idcode ind_code#year)
check ln_wage age ttl_exp tenure, absorb(idcode i.ind_code#i.year) vce(cluster ind_code#year)
* regressor absorbed by the FEs; collinear pair
gen double x_abs = idcode
gen double tenure2 = 2 * tenure + 1
check ln_wage age x_abs tenure, absorb(idcode year)
check ln_wage age tenure tenure2 ttl_exp, absorb(idcode year)
* keepsingletons, noconstant
check ln_wage age ttl_exp tenure, absorb(idcode year) keepsingletons
check ln_wage age ttl_exp tenure, absorb(idcode year) noconstant
check ln_wage age ttl_exp tenure, absorb(idcode year) vce(cluster idcode) keepsingletons noconstant
* if / in, missing values
check ln_wage age ttl_exp tenure if union == 1, absorb(idcode year)
check ln_wage age ttl_exp tenure in 1/15000, absorb(idcode year) vce(robust)
di as result "nlswork specifications passed"

* ---------------------------------------------------------------------------
* groupvar(), residuals, savefe, predict
* ---------------------------------------------------------------------------
webuse nlswork, clear
quietly reghdfe ln_wage age ttl_exp tenure, absorb(feR1=idcode feR2=year feR3=occ_code) resid(rR) groupvar(gR) tol(1e-12)
estimates store R
foreach o in xb stdp residuals d xbd dresiduals score {
	quietly predict double pR_`o', `o'
}
if ("$RF_MOCK" != "") rf_stage_mock, indepvars(age ttl_exp tenure)
regfast ln_wage age ttl_exp tenure, absorb(feF1=idcode feF2=year feF3=occ_code) resid(rF) groupvar(gF)
estimates store F
rf_compare_e R F, xscalars(ic) xmacros(resid) `=cond("$RF_MOCK" != "", "nosample", "")'
assert "`e(resid)'" == "rF"
foreach o in xb stdp residuals d xbd dresiduals score {
	quietly predict double pF_`o', `o'
}
assert "`: type gF'" == "long"
assert `"`: variable label gF'"' == `"`: variable label gR'"'
assert `"`: variable label rF'"' == "Residuals"
forval k = 1/3 {
	assert `"`: variable label feF`k''"' == `"`: variable label feR`k''"'
}
if ("$RF_MOCK" == "") {
	* residuals and every predict option
	foreach o in xb stdp residuals d xbd dresiduals score {
		qui count if reldif(pF_`o', pR_`o') > 1e-6 | missing(pF_`o') != missing(pR_`o')
		if (r(N)) {
			di as err "FAIL predict `o': `r(N)' observations differ"
			exit 9
		}
	}
	* saved FEs: their sum equals d (FE split across terms is normalization-dependent, the sum is not)
	gen double dF = feF1 + feF2 + feF3
	gen double dR = feR1 + feR2 + feR3
	qui count if e(sample) & (reldif(dF, pF_d) > 1e-6 | reldif(dF, dR) > 1e-6)
	assert r(N) == 0
	qui count if !e(sample) & !missing(dF, rF)
	assert r(N) == 0
	* mobility groups: same partition
	qui count if missing(gF) != missing(gR)
	assert r(N) == 0
	qui tab gR
	local nR = r(r)
	qui tab gF
	assert r(r) == `nR'
	tempvar pair
	egen `pair' = group(gR gF)
	qui summarize `pair'
	assert r(max) == `nR'
}
* residuals2 (default name), savefe without names, replace
if ("$RF_MOCK" != "") {
	quietly reghdfe ln_wage age ttl_exp tenure, absorb(idcode year) tol(1e-12)
	rf_stage_mock, indepvars(age ttl_exp tenure)
}
regfast ln_wage age ttl_exp tenure, absorb(idcode year, savefe) residuals
confirm variable _reghdfe_resid __hdfe1__ __hdfe2__
assert "`e(resid)'" == "_reghdfe_resid"
assert `"`: variable label __hdfe2__'"' == "[FE] 1.year"
regfast ln_wage age ttl_exp tenure, absorb(idcode year, savefe) residuals
predict double xbd_check, xbd
if ("$RF_MOCK" == "") {
	qui count if e(sample) & reldif(xbd_check, ln_wage - _reghdfe_resid) > 1e-10
	assert r(N) == 0
}
capture regfast ln_wage age, absorb(fe1=idcode fe1=year)
assert _rc == 198
di as result "outputs passed"

di as result "TEST_OLS PASSED"
