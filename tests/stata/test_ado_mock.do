* test_ado_mock.do -- ado-level dry run of regfast against reghdfe, using the fake plugin in tests/stata/mock
*
* The mock plugin echoes reghdfe's numbers (staged by rf_stage_mock), so these tests check everything the
* ado does: option parsing, sample marking, factor-variable expansion, plugin keys, output variables,
* posting of e(), the display (header, coefficient table, absorbed-dof footnote) and predict.
* Numbers are checked against the real plugin in test_ols.do.
*
* Run from tests/stata after building the mock:  mock/build.sh && stata-mp -b do test_ado_mock.do
* (do not run it in the same session as test_ols.do: the mock shadows the real regfast.plugin)
version 16
clear all
set more off
capture confirm file "mock/regfast.plugin"
if (_rc) {
	di as err "build the mock first: tests/stata/mock/build.sh"
	exit 601
}
adopath ++ "`c(pwd)'/../../stata"
adopath ++ "`c(pwd)'/mock"
findfile regfast.plugin
assert strpos("`r(fn)'", "/mock/")

capture program drop expect_key
program expect_key
	* expect_key prepare|ols "key=value" : the last plugin call of that kind received exactly this argument
	args which kv
	local args " ${rfmock_`which'}"
	if (!strpos(`"`args'"', " `kv' ")) {
		di as err `"FAIL: `which' did not receive <`kv'>; got <`args'>"'
		exit 9
	}
end

* ---------------------------------------------------------------------------
* A) factor variables, base and FE-collinear levels, 2-way clustering, CGM fix
* ---------------------------------------------------------------------------
webuse nlswork, clear
reghdfe ln_wage age i.race c.age#c.age, absorb(idcode year occ_code) vce(cluster idcode year) tol(1e-12)
estimates store R
keep if e(sample)
rf_stage_mock, indepvars(age i.race c.age#c.age) status(0 2 2 0)
regfast ln_wage age i.race c.age#c.age, absorb(idcode year occ_code) vce(cluster idcode year)
estimates store F
rf_compare_e R F
rf_display_diff R F
expect_key prepare "absorb=1|1|-|0;2|1|-|0;3|1|-|0"
expect_key prepare "cluster=1;2"
expect_key prepare "factype=nnn"
expect_key prepare "vce=cluster"
expect_key prepare "singletons=1"
expect_key ols "nx=4"
expect_key ols "noconstant=0"
expect_key ols "dof=pairwise,clusters,continuous"
expect_key ols "tol=1.00000000000000e-08"
expect_key ols "maxiter=16000"
expect_key ols "fe=0"
assert "`e(indepvars)'" == "age 2bn.race 3bn.race c.age#c.age _cons"
di as result "A passed"

* ---------------------------------------------------------------------------
* B) robust, cluster variable also absorbed, dofadjustments(), tolerance(), iterate()
* ---------------------------------------------------------------------------
webuse nlswork, clear
reghdfe ln_wage age ttl_exp tenure, absorb(idcode year) vce(robust) tol(1e-12)
estimates store R
keep if e(sample)
rf_stage_mock, indepvars(age ttl_exp tenure) status(0 0 0)
regfast ln_wage age ttl_exp tenure, absorb(idcode year) vce(robust) tol(1e-12) iterate(500)
estimates store F
rf_compare_e R F
rf_display_diff R F
expect_key ols "tol=1.00000000000000e-12"
expect_key ols "maxiter=500"

reghdfe ln_wage age ttl_exp tenure, absorb(idcode year) cluster(idcode) dof(pairwise) tol(1e-12)
estimates store R
rf_stage_mock, indepvars(age ttl_exp tenure) status(0 0 0)
regfast ln_wage age ttl_exp tenure, absorb(idcode year) cluster(idcode) dof(pairwise)
estimates store F
rf_compare_e R F
rf_display_diff R F
expect_key prepare "cluster=1"
expect_key prepare "nfac=2"
expect_key ols "dof=pairwise"

reghdfe ln_wage age, absorb(idcode year) dof(none) vce(cluster ind_code) tol(1e-12)
estimates store R
keep if e(sample)
rf_stage_mock, indepvars(age) status(0)
regfast ln_wage age, absorb(idcode year) dof(none) vce(cluster ind_code)
estimates store F
rf_compare_e R F
rf_display_diff R F
expect_key prepare "nfac=3"
expect_key prepare "cluster=3"
expect_key ols "dof=none"
di as result "B passed"

* ---------------------------------------------------------------------------
* C) weights (aw/fw/pw) on auto
* ---------------------------------------------------------------------------
sysuse auto, clear
gen int fwv = 1 + mod(_n, 3)
foreach w in "aw=mpg" "fw=fwv" "pw=mpg" {
	reghdfe price weight length [`w'], absorb(rep78) vce(cluster rep78) tol(1e-12)
	estimates store R
	rf_stage_mock, indepvars(weight length) status(0 0)
	regfast price weight length [`w'], absorb(rep78) vce(cluster rep78)
	estimates store F
	rf_compare_e R F
	rf_display_diff R F
	local wt = substr("`w'", 1, 2)
	expect_key prepare "wtype=`wt'"
}
reghdfe price weight length [pw=mpg], absorb(rep78) tol(1e-12)
estimates store R
rf_stage_mock, indepvars(weight length) status(0 0)
regfast price weight length [pw=mpg], absorb(rep78)
estimates store F
rf_compare_e R F
rf_display_diff R F
expect_key prepare "vce=robust"
di as result "C passed"

* ---------------------------------------------------------------------------
* D) no absorbed FEs (absorb omitted, and noabsorb)
* ---------------------------------------------------------------------------
sysuse auto, clear
reghdfe price weight length, tol(1e-12)
estimates store R
rf_stage_mock, indepvars(weight length) status(0 0)
regfast price weight length
estimates store F
rf_compare_e R F
rf_display_diff R F
expect_key prepare "absorb=none"
expect_key prepare "nfac=0"
regfast price weight length, noabsorb
estimates store F
rf_compare_e R F
rf_display_diff R F
di as result "D passed"

* ---------------------------------------------------------------------------
* E) savefe with slopes, residuals(), groupvar(), FE-collinear regressor; predict
* ---------------------------------------------------------------------------
webuse nlswork, clear
reghdfe ln_wage age tenure, absorb(fe1=idcode year##c.age, savefe) resid(res) groupvar(gg) tol(1e-12)
estimates store R
keep if e(sample)
local labels
foreach v in res gg fe1 __hdfe2__ __hdfe2__Slope1 {
	local labels `"`labels' "`: variable label `v''""'
	local type_`v' : type `v'
}
drop res gg fe1 __hdfe2__ __hdfe2__Slope1
rf_stage_mock, indepvars(age tenure) status(2 0)
regfast ln_wage age tenure, absorb(fe1=idcode year##c.age, savefe) resid(res) groupvar(gg)
estimates store F
rf_compare_e R F
rf_display_diff R F
expect_key prepare "absorb=1|1|-|1;2|1|1|1"
expect_key prepare "nslope=1"
expect_key ols "nout=5"
expect_key ols "resid=1"
expect_key ols "groupvar=2"
expect_key ols "fe=3,4,5"
expect_key ols "d=0"
local i 0
foreach v in res gg fe1 __hdfe2__ __hdfe2__Slope1 {
	local ++i
	local lab : word `i' of `labels'
	assert `"`: variable label `v''"' == `"`lab'"'
	assert "`: type `v''" == "`type_`v''"
	* the mock writes 1000*k + obs into the k-th output variable
	assert `v' == 1000 * `i' + _n
}
assert "`e(resid)'" == "res"

* predict (identities with the mock residuals)
predict double xb1
predict double xb2, xb
_predict double xb3, xb
assert xb1 == xb2 & xb1 == xb3
predict double r1, resid
assert r1 == res
predict double sc1, score
assert sc1 == res
predict double d1, d
assert reldif(d1, ln_wage - xb1 - res) < 1e-12
predict double xbd1, xbd
assert reldif(xbd1, ln_wage - res) < 1e-12
predict double dr1, dresiduals
assert reldif(dr1, ln_wage - xb1) < 1e-12
predict double s1, stdp
_predict double s2, stdp
assert s1 == s2
assert "`: variable label d1'" == "d[idcode year##c.age]"
assert "`: variable label xbd1'" == "Xb + d[idcode year##c.age]"
assert "`: variable label r1'" == "Residuals"

* savefe without residuals: e(resid) not set, predict d refuses
drop fe1 __hdfe*
quietly reghdfe ln_wage age tenure, absorb(idcode year) tol(1e-12)
rf_stage_mock, indepvars(age tenure) status(0 0)
regfast ln_wage age tenure, absorb(idcode year, savefe)
assert "`e(resid)'" == ""
confirm variable __hdfe1__ __hdfe2__
capture predict double d2, d
assert _rc == 198
* residuals (no name) -> _reghdfe_resid, replaced on rerun
regfast ln_wage age tenure, absorb(idcode year) residuals
confirm variable _reghdfe_resid
regfast ln_wage age tenure, absorb(idcode year) residuals
assert "`e(resid)'" == "_reghdfe_resid"
* groupvar is dropped when there is no pair of intercept FEs (reghdfe does not create it)
quietly reghdfe ln_wage age tenure, absorb(idcode) tol(1e-12)
rf_stage_mock, indepvars(age tenure) status(0 0)
global rfmock_nogroup 1
regfast ln_wage age tenure, absorb(idcode) groupvar(gg2)
global rfmock_nogroup
capture confirm variable gg2
assert _rc
di as result "E passed"

* ---------------------------------------------------------------------------
* F) sample marking: missing values everywhere, if/in, zero/missing weights, string FE (keepsingletons)
* ---------------------------------------------------------------------------
webuse nlswork, clear
gen sid = string(idcode)
replace sid = "" if mod(_n, 97) == 0
gen double w = 1 + mod(_n, 5)
replace w = 0 if mod(_n, 89) == 0
replace w = . if mod(_n, 83) == 0
replace tenure = . if mod(_n, 79) == 0
replace ln_wage = . if mod(_n, 73) == 0
replace ind_code = . if mod(_n, 71) == 0
gen double z = age
replace z = . if mod(_n, 67) == 0
reghdfe ln_wage tenure [aw=w] if age > 20 in 1/20000, absorb(sid year##c.z) vce(cluster ind_code) keepsingletons tol(1e-12)
estimates store R
rf_stage_mock, indepvars(tenure) status(0)
regfast ln_wage tenure [aw=w] if age > 20 in 1/20000, absorb(sid year##c.z) vce(cluster ind_code) keepsingletons
estimates store F
rf_compare_e R F
rf_display_diff R F
expect_key prepare "factype=snn"
expect_key prepare "singletons=0"
expect_key prepare "cluster=3"
di as result "F passed"

* ---------------------------------------------------------------------------
* G) collinear regressors, noconstant, slopes-only absorb, string absvar, interactions
* ---------------------------------------------------------------------------
webuse nlswork, clear
gen double tenure2 = 2 * tenure
reghdfe ln_wage tenure tenure2 age, absorb(idcode year) tol(1e-12)
estimates store R
keep if e(sample)
rf_stage_mock, indepvars(tenure tenure2 age) status(0 3 0)
regfast ln_wage tenure tenure2 age, absorb(idcode year)
estimates store F
rf_compare_e R F
rf_display_diff R F

webuse nlswork, clear
reghdfe ln_wage age tenure, absorb(idcode year) noconstant tol(1e-12)
estimates store R
keep if e(sample)
rf_stage_mock, indepvars(age tenure) status(0 0)
regfast ln_wage age tenure, absorb(idcode year) noconstant
estimates store F
rf_compare_e R F
rf_display_diff R F
expect_key ols "noconstant=1"

webuse nlswork, clear
reghdfe ln_wage tenure, absorb(ind_code#c.age) tol(1e-12)
estimates store R
keep if e(sample)
rf_stage_mock, indepvars(tenure) status(0)
regfast ln_wage tenure, absorb(ind_code#c.age)
estimates store F
rf_compare_e R F
rf_display_diff R F
expect_key prepare "absorb=1|0|1|0"

webuse nlswork, clear
gen sid = string(idcode)
reghdfe ln_wage tenure i.race#c.age, absorb(sid ind_code#year) tol(1e-12)
estimates store R
keep if e(sample)
rf_stage_mock, indepvars(tenure i.race#c.age) status(0 0 0 0)
regfast ln_wage tenure i.race#c.age, absorb(sid ind_code#year)
estimates store F
rf_compare_e R F
rf_display_diff R F
expect_key prepare "absorb=1|1|-|0;2*3|1|-|0"
expect_key prepare "factype=snn"
di as result "G passed"

* ---------------------------------------------------------------------------
* H) replay and display options; option errors
* ---------------------------------------------------------------------------
rf_display_diff R F, options(level(90))
rf_display_diff R F, options(noheader)
rf_display_diff R F, options(nofootnote)
rf_display_diff R F, options(notable cformat(%9.3f))
estimates restore F
regfast
regfast, level(80) noomitted

webuse nlswork, clear
foreach bad in "ln_wage (tenure = union), absorb(idcode)" "ln_wage tenure, absorb(idcode) noabsorb" ///
	"ln_wage tenure, absorb(idcode) individual(idcode)" "ln_wage tenure, absorb(idcode) group(idcode)" ///
	"ln_wage tenure, absorb(idcode) aggregation(sum)" "ln_wage tenure, absorb(idcode) vce(dkraay)" ///
	"ln_wage tenure, absorb(idcode) residuals(r) residuals" "ln_wage tenure, absorb(idcode) first" ///
	"ln_wage tenure, absorb(idcode) bogus" "ln_wage tenure, absorb(idcode) vce(robust) cluster(idcode)" ///
	"ln_wage tenure, absorb(idcode, nosuchoption)" "ln_wage tenure [pw=age], absorb(idcode) vce(unadjusted)" {
	capture noisily regfast `bad'
	if (_rc != 198) {
		di as err `"FAIL: regfast `bad' gave rc `=_rc' (expected 198)"'
		exit 9
	}
}
quietly reghdfe ln_wage tenure, absorb(idcode) tol(1e-12)
rf_stage_mock, indepvars(tenure) status(0)
regfast ln_wage tenure, absorb(idcode) technique(lsmr) transform(cim) acceleration(sd) poolsize(5) compact
di as result "H passed"

di as result "TEST_ADO_MOCK PASSED"
