* test_ado_mock.do -- ado-level dry run of fereg against reghdfe, using the fake plugin in tests/stata/mock
*
* The mock plugin echoes reghdfe's numbers (staged by rf_stage_mock), so these tests check everything the
* ado does: option parsing, sample marking, factor-variable expansion, plugin keys, output variables,
* posting of e(), the display (header, coefficient table, absorbed-dof footnote) and predict.
* Numbers are checked against the real plugin in test_ols.do.
*
* Run from tests/stata after building the mock:  mock/build.sh && stata-mp -b do test_ado_mock.do
* (do not run it in the same session as test_ols.do: the mock shadows the real fereg.plugin)
version 16
clear all
set more off
capture confirm file "mock/fereg.plugin"
if (_rc) {
	di as err "build the mock first: tests/stata/mock/build.sh"
	exit 601
}
adopath ++ "`c(pwd)'/../../stata"
adopath ++ "`c(pwd)'/mock"
findfile fereg.plugin
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
fereg ln_wage age i.race c.age#c.age, absorb(idcode year occ_code) vce(cluster idcode year)
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
fereg ln_wage age ttl_exp tenure, absorb(idcode year) vce(robust) tol(1e-12) iterate(500)
estimates store F
rf_compare_e R F
rf_display_diff R F
expect_key ols "tol=1.00000000000000e-12"
expect_key ols "maxiter=500"

reghdfe ln_wage age ttl_exp tenure, absorb(idcode year) cluster(idcode) dof(pairwise) tol(1e-12)
estimates store R
rf_stage_mock, indepvars(age ttl_exp tenure) status(0 0 0)
fereg ln_wage age ttl_exp tenure, absorb(idcode year) cluster(idcode) dof(pairwise)
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
fereg ln_wage age, absorb(idcode year) dof(none) vce(cluster ind_code)
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
	fereg price weight length [`w'], absorb(rep78) vce(cluster rep78)
	estimates store F
	rf_compare_e R F
	rf_display_diff R F
	local wt = substr("`w'", 1, 2)
	expect_key prepare "wtype=`wt'"
}
reghdfe price weight length [pw=mpg], absorb(rep78) tol(1e-12)
estimates store R
rf_stage_mock, indepvars(weight length) status(0 0)
fereg price weight length [pw=mpg], absorb(rep78)
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
fereg price weight length
estimates store F
rf_compare_e R F
rf_display_diff R F
expect_key prepare "absorb=none"
expect_key prepare "nfac=0"
fereg price weight length, noabsorb
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
fereg ln_wage age tenure, absorb(fe1=idcode year##c.age, savefe) resid(res) groupvar(gg)
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
fereg ln_wage age tenure, absorb(idcode year, savefe)
assert "`e(resid)'" == ""
confirm variable __hdfe1__ __hdfe2__
capture predict double d2, d
assert _rc == 198
* residuals (no name) -> _reghdfe_resid, replaced on rerun
fereg ln_wage age tenure, absorb(idcode year) residuals
confirm variable _reghdfe_resid
fereg ln_wage age tenure, absorb(idcode year) residuals
assert "`e(resid)'" == "_reghdfe_resid"
* groupvar is dropped when there is no pair of intercept FEs (reghdfe does not create it)
quietly reghdfe ln_wage age tenure, absorb(idcode) tol(1e-12)
rf_stage_mock, indepvars(age tenure) status(0 0)
global rfmock_nogroup 1
fereg ln_wage age tenure, absorb(idcode) groupvar(gg2)
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
fereg ln_wage tenure [aw=w] if age > 20 in 1/20000, absorb(sid year##c.z) vce(cluster ind_code) keepsingletons
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
fereg ln_wage tenure tenure2 age, absorb(idcode year)
estimates store F
rf_compare_e R F
rf_display_diff R F

webuse nlswork, clear
reghdfe ln_wage age tenure, absorb(idcode year) noconstant tol(1e-12)
estimates store R
keep if e(sample)
rf_stage_mock, indepvars(age tenure) status(0 0)
fereg ln_wage age tenure, absorb(idcode year) noconstant
estimates store F
rf_compare_e R F
rf_display_diff R F
expect_key ols "noconstant=1"

webuse nlswork, clear
reghdfe ln_wage tenure, absorb(ind_code#c.age) tol(1e-12)
estimates store R
keep if e(sample)
rf_stage_mock, indepvars(tenure) status(0)
fereg ln_wage tenure, absorb(ind_code#c.age)
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
fereg ln_wage tenure i.race#c.age, absorb(sid ind_code#year)
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
fereg
fereg, level(80) noomitted

webuse nlswork, clear
foreach bad in "ln_wage (tenure = union), absorb(idcode)" "ln_wage tenure, absorb(idcode) noabsorb" ///
	"ln_wage tenure, absorb(idcode) individual(idcode)" "ln_wage tenure, absorb(idcode) group(idcode)" ///
	"ln_wage tenure, absorb(idcode) aggregation(sum)" "ln_wage tenure, absorb(idcode) vce(hc2 idcode)" ///
	"ln_wage tenure, absorb(idcode) residuals(r) residuals" "ln_wage tenure, absorb(idcode) first" ///
	"ln_wage tenure, absorb(idcode) bogus" "ln_wage tenure, absorb(idcode) vce(robust) cluster(idcode)" ///
	"ln_wage tenure, absorb(idcode, nosuchoption)" "ln_wage tenure [pw=age], absorb(idcode) vce(unadjusted)" ///
	"ln_wage tenure, absorb(idcode) bw(2) sw" "ln_wage tenure, absorb(idcode) bw(2) center" ///
	"ln_wage tenure, absorb(idcode) bw(2) dofminus(1)" "ln_wage tenure, absorb(idcode) bw(2) sdofminus(1)" ///
	"ln_wage (tenure = union), absorb(idcode) vce(hc3)" "ln_wage tenure, absorb(idcode) vce(crv3)" {
	capture noisily fereg `bad'
	if (_rc != 198) {
		di as err `"FAIL: fereg `bad' gave rc `=_rc' (expected 198)"'
		exit 9
	}
}
quietly reghdfe ln_wage tenure, absorb(idcode) tol(1e-12)
rf_stage_mock, indepvars(tenure) status(0)
fereg ln_wage tenure, absorb(idcode) technique(lsmr) transform(cim) acceleration(sd) poolsize(5) compact
di as result "H passed"

* ---------------------------------------------------------------------------
* I) variance estimators: vce(dkraay) vs reghdfe (e() and display), keys of the kernel options,
*    hc2/hc3, crv3 and conley
* ---------------------------------------------------------------------------
webuse nlswork, clear
xtset idcode year
reghdfe ln_wage age tenure, absorb(idcode) vce(dkraay 2) tol(1e-12)
estimates store R
keep if e(sample)
rf_stage_mock, indepvars(age tenure) status(0 0)
fereg ln_wage age tenure, absorb(idcode) vce(dkraay 2)
estimates store F
rf_compare_e R F
rf_display_diff R F
expect_key prepare "vce=dkraay"
expect_key prepare "dkbw=2"
expect_key prepare "tvar=2"
expect_key prepare "tdelta=1"
expect_key prepare "nfac=2"
assert !strpos(" $rfmock_prepare ", " ivar=")
quietly reghdfe ln_wage age tenure, absorb(year) vce(dkraay) tol(1e-12)
estimates store R
rf_stage_mock, indepvars(age tenure) status(0 0)
fereg ln_wage age tenure, absorb(year) vce(dkraay)
estimates store F
rf_compare_e R F
rf_display_diff R F
expect_key prepare "tvar=1"
expect_key prepare "nfac=1"
assert !strpos(" $rfmock_prepare ", " dkbw=")

* kernel options (numbers are the staged reghdfe ones; the e() strings and keys are checked)
capture program drop stage_hac
program stage_hac
	args bw
	quietly reghdfe ln_wage age tenure, absorb(idcode) tol(1e-12)
	rf_stage_mock, indepvars(age tenure) status(0 0)
	scalar __mock_bw = `bw'
end
stage_hac 3
fereg ln_wage age tenure, absorb(idcode) bw(3) kernel(par)
expect_key prepare "vce=unadjusted"
expect_key prepare "kernel=parzen"
expect_key prepare "bw=3"
expect_key prepare "tvar=2"
expect_key prepare "ivar=1"
expect_key prepare "psd=none"
assert "`e(vce)'" == "ac parzen bw=3" & "`e(vcetype)'" == "" & "`e(kernel)'" == "Parzen" & e(bw) == 3
assert "`e(tvar)'" == "year" & "`e(ivar)'" == "idcode" & "`e(title3)'" == "Statistics robust to autocorrelation"
assert "`e(hacsubtitleV)'" == "Statistics robust to autocorrelation"
stage_hac 2.5
fereg ln_wage age tenure, absorb(idcode) bw(2.5) kernel(quadratic spectral) vce(robust) psda
expect_key prepare "vce=robust"
expect_key prepare "kernel=quadratic-spectral"
expect_key prepare "bw=2.5"
expect_key prepare "psd=psda"
assert "`e(vce)'" == "robust hac quadratic spectral bw=2.5 psda" & "`e(vcetype)'" == "Robust"
assert "`e(hacsubtitleV)'" == "Statistics robust to heteroskedasticity and autocorrelation"
stage_hac 2
fereg ln_wage age tenure, absorb(idcode) dkraay(2)
expect_key prepare "vce=cluster"
expect_key prepare "cluster=2"
expect_key prepare "kernel=bartlett"
assert "`e(vce)'" == "cluster ac bartlett bw=2" & "`e(vcetype)'" == "Robust" & "`e(clustvar)'" == "year"
assert "`e(hacsubtitleV)'" == "Statistics robust to heteroskedasticity and clustering on year"
assert "`e(hacsubtitleV2)'" == "and kernel-robust to common correlated disturbances (Driscoll-Kraay)"
stage_hac 2
fereg ln_wage age tenure, absorb(idcode) vce(cluster year idcode) bw(2) kernel(tru)
expect_key prepare "cluster=1;2"
assert "`e(vce)'" == "robust two-way cluster hac truncated bw=2" & "`e(clustvar1)'" == "idcode" & "`e(clustvar2)'" == "year"
assert "`e(hacsubtitleV)'" == "Statistics robust to heteroskedasticity and clustering on idcode and year"
stage_hac 15
fereg ln_wage age tenure, absorb(idcode) kiefer
expect_key prepare "kiefer=1"
expect_key prepare "kernel=truncated"
assert !strpos(" $rfmock_prepare ", " bw=")
assert "`e(vce)'" == "ac truncated bw=15" & "`e(hacsubtitleV)'" == "Statistics robust to within-cluster autocorrelation (Kiefer)"
di as result "I kernels passed"

* hc2/hc3, crv3, conley: keys and e() labels
quietly reghdfe ln_wage age tenure, absorb(idcode) tol(1e-12)
rf_stage_mock, indepvars(age tenure) status(0 0)
fereg ln_wage age tenure, absorb(idcode) vce(hc3)
expect_key prepare "vce=hc3"
assert "`e(vce)'" == "hc3" & "`e(vcetype)'" == "Robust HC3"
quietly reghdfe ln_wage age tenure, absorb(idcode) vce(cluster idcode ind_code) tol(1e-12)
rf_stage_mock, indepvars(age tenure) status(0 0)
fereg ln_wage age tenure, absorb(idcode) vce(crv3 idcode ind_code)
expect_key prepare "vce=crv3"
expect_key prepare "cluster=1;2"
assert "`e(vce)'" == "crv3" & "`e(vcetype)'" == "CRV3" & "`e(clustvar)'" == "idcode ind_code" & e(N_clustervars) == 2
gen double lat = 30 + mod(idcode, 17)
gen double lon = -100 + mod(idcode, 23)
quietly reghdfe ln_wage age tenure, absorb(idcode) tol(1e-12)
rf_stage_mock, indepvars(age tenure) status(0 0)
scalar __mock_cutoff = 250
fereg ln_wage age tenure, absorb(idcode) vce(conley lat lon, cutoff(250) distance(spherical))
expect_key prepare "vce=conley"
expect_key prepare "ncoord=2"
expect_key prepare "cutoff=250"
expect_key prepare "pixel=0"
expect_key prepare "distance=spherical"
assert "`e(vce)'" == "conley" & "`e(vcetype)'" == "Conley" & e(conley_cutoff) == 250 & "`e(conley_lat)'" == "lat"
di as result "I passed"

* ---------------------------------------------------------------------------
* J) felogit bias correction and APEs (logitfe options): keys and bias-correction variables sent
*    to cmd=glm, posting of e(b) e(b0) e(b2) e(V2) and the macros, logitfe's titles and APE table,
*    replay, predict at the corrected point. Numbers are staged (checked against logitfe with the
*    real plugin in test_logitfe.do).
* ---------------------------------------------------------------------------
capture program drop stage_bc
program stage_bc
	* stage_bc <nx> <number of absorbed slots>: b = (1..nx, 0.5), b0 = 10*b, b2 = b/100, V2
	args nx nslots
	matrix __mock_b = J(1, `nx' + 1, 0)
	matrix __mock_b0 = J(1, `nx' + 1, 0)
	matrix __mock_b2 = J(1, `nx', 0)
	forval j = 1/`nx' {
		matrix __mock_b[1, `j'] = `j'
		matrix __mock_b0[1, `j'] = 10 * `j'
		matrix __mock_b2[1, `j'] = `j' / 100
	}
	matrix __mock_b[1, `nx' + 1] = 0.5
	matrix __mock_b0[1, `nx' + 1] = 5
	matrix __mock_V = 0.01 * I(`nx' + 1) + J(`nx' + 1, `nx' + 1, 0.001)
	matrix __mock_V2 = 0.0001 * I(`nx') + J(`nx', `nx', 0.00001)
	matrix __mock_doft = J(`nslots', 5, 1)
	foreach s in N ll ll_0 chi2 df_m r2_p num_separated converged N_full bc_nfits {
		capture scalar drop __mock_`s'
	}
	scalar __mock_ll = -250
	scalar __mock_ll_0 = -270
	scalar __mock_chi2 = 12
	scalar __mock_df_m = `nx'
	scalar __mock_r2_p = .07
	scalar __mock_num_separated = 0
	scalar __mock_converged = 1
	global rfmock_status
	forval j = 1/`nx' {
		global rfmock_status $rfmock_status 0
	}
end
capture program drop expect_glm
program expect_glm
	* expect_glm "key=value" ... : the last glm call received exactly these arguments
	foreach kv of local 0 {
		local args " $rfmock_glm "
		if (!strpos(`"`args'"', " `kv' ")) {
			di as err `"FAIL: glm did not receive <`kv'>; got <`args'>"'
			exit 9
		}
	}
end

clear
set seed 20261018
set obs 400
gen long id = ceil(_n / 8)
bysort id: gen int year = 2000 + 2 * _n
gen double x1 = rnormal()
gen byte x2 = runiform() > 0.5
gen byte y = runiform() < invlogit(0.5 * x1 + x2 - 0.5)
replace x1 = . in 3
xtset id year, delta(2)

* analytical, lags, population(), d(): keys, varlist (xtset variables), e(), display
stage_bc 2 2
tempfile lg
log using "`lg'", text name(bc) replace
felogit y x1 x2, absorb(id year) analytical lags(1) d(dd) population(1000)
log close bc
expect_glm "bc=analytical" "structure=classic" "which=both" "lags=1" "nbc=2" "effects=it" "tdelta=2" "nx=2" "d=1"
expect_glm "adj=`=strtrim(string((1000 - 399) / 999, "%21.17g"))'"
assert !strpos(" $rfmock_glm ", " jk=") & !strpos(" $rfmock_glm ", " multiple=")
assert "$rfmock_glm_nvars" == "7"
assert !strpos(" $rfmock_prepare ", " sample=all ")
assert e(N_full) == 399 & reldif(e(fpc), 601 / 999) < 1e-15
assert e(N_group) == 50 & e(T_min) == 7 & e(T_max) == 8 & reldif(e(T_avg), 399 / 50) < 1e-15
assert mreldif(e(b), __mock_b) == 0 & mreldif(e(b0), __mock_b0) == 0 & mreldif(e(V), __mock_V) == 0
assert mreldif(e(b2), __mock_b2) == 0 & mreldif(e(V2), __mock_V2) == 0
assert "`: colfullnames e(b0)'" == "x1 x2 _cons" & "`: colfullnames e(b2)'" == "x1 x2"
assert "`: rowfullnames e(V2)'" == "x1 x2"
assert "`e(bias)'" == "analytical" & "`e(lags)'" == "1" & "`e(ibias)'" == "yes" & "`e(tbias)'" == "yes"
assert "`e(jackknife)'" == "" & "`e(id)'" == "id" & "`e(time)'" == "year" & "`e(structure)'" == "classic"
assert "`e(bctitle)'" == "Analytical bias-correction" & "`e(bctitle3)'" == "Triming parameter = 1"
assert "`e(title)'" == "HDFE Logistic regression" & "`e(d)'" == "dd"
foreach t in "Analytical bias-correction" "Type of included effects: individual and time effects" ///
	"Type of bias correction: individual and time effects" "Triming parameter = 1" "ID variable    = id" ///
	"Time variable  = year" "Average Partial Effects" "Variance adjusted by the finite population parameter   0.6016" ///
	"HDFE Logistic regression" {
	rf_log_has "`lg'" "`t'"
	if (!r(found)) {
		di as err `"FAIL: display lacks <`t'>"'
		exit 9
	}
}
* replay, quietly (r(table) of the coefficients), noheader/notable
log using "`lg'", text name(bc) replace
felogit, level(90)
tempname T TA
matrix `T' = r(table)
matrix `TA' = r(table_ape)
log close bc
rf_log_has "`lg'" "Average Partial Effects"
assert r(found) == 1
assert colsof(`T') == 3 & colsof(`TA') == 2 & el(`TA', 1, 2) == 0.02 & reldif(el(`TA', 8, 1), invnormal(0.95)) < 1e-12
quietly felogit
assert colsof(r(table)) == 3
log using "`lg'", text name(bc) replace
felogit, notable noheader
log close bc
rf_log_has "`lg'" "Average Partial Effects"
assert r(found) == 0
rf_log_has "`lg'" "Triming parameter"
assert r(found) == 0
* predict at the posted (corrected) point: the mock writes 1000 + obs into d
predict double p1, pr
assert reldif(p1, invlogit(x1 + 2 * x2 + 0.5 + dd)) < 1e-12 if e(sample)
predict double xb1, xb
assert reldif(xb1, x1 + 2 * x2 + 0.5) < 1e-12 if e(sample)
estat ic
assert el(r(S), 1, 3) == -250
drop p1 xb1 dd

* no option: no bias keys, no extra variables, no e(b2)
stage_bc 2 2
felogit y x1 x2, absorb(id year)
assert !strpos(" $rfmock_glm ", " bc=") & "$rfmock_glm_nvars" == "4"
expect_key prepare "sample=all"
capture confirm matrix e(b2)
assert _rc
assert "`e(bias)'" == "" & "`e(bctitle)'" == ""

* ape alone, one absorbed term (time), factor variables (base level posted as 0 in e(b2))
gen byte c = mod(id, 3)
stage_bc 3 1
felogit y x1 i.c, absorb(year) ape
expect_glm "bc=ape" "structure=classic" "which=time" "effects=t" "nbc=2" "adj=1"
assert "`: colfullnames e(b2)'" == "x1 0b.c 1.c 2.c" & e(b2)[1, 2] == 0 & e(V2)[2, 2] == 0
assert e(b2)[1, 3] == 0.02 & e(b2)[1, 4] == 0.03 & e(fpc) == 1
assert "`e(bias)'" == "none" & "`e(ibias)'" == "" & "`e(lags)'" == ""
assert "`e(bctitle)'" == "Uncorrected fixed-effects estimates" & "`e(bctitle2)'" == "Type of bias correction: none"

* jackknife variants; multiple(): logitfe's permutation variables (one per partition and dimension)
stage_bc 2 2
felogit y x1 x2, absorb(year id) jackknife
expect_glm "bc=jackknife" "jk=ss2" "effects=ti" "nbc=2" "xnames=x1,x2"
assert "`e(jackknife)'" == "ss2" & "`e(bctitle)'" == "Split-panel jackknife in both dimensions"
stage_bc 2 2
scalar __mock_bc_nfits = 12
set seed 99
local rs = c(rngstate)
felogit y x1 x2, absorb(id year) jackknife ss1 multiple(3) ibias(no)
assert c(rngstate) == "`rs'"
expect_glm "jk=ss1" "multiple=3" "perm=it" "which=time" "nbc=8"
assert "$rfmock_glm_nvars" == "12"
assert e(multiple) == 3 & "`e(split)'" == "both" & e(N_subpanels) == 12 & "`e(ibias)'" == "no"
assert "`e(bctitle3)'" == "3 multiple partitions in both the cross-section and the time dimension"
stage_bc 2 2
felogit y x1 x2, absorb(id year) jackknife ss2 multiple(2) i
expect_glm "multiple=2" "perm=i" "nbc=4"
assert "`e(split)'" == "individuals"
* partitions with no effect (logitfe: a note and the plain estimator): an error
stage_bc 2 2
log using "`lg'", text name(bc) replace
capture noisily felogit y x1 x2, absorb(id year) jackknife ss2 multiple(2) individuals tbias(no)
assert _rc == 198
log close bc
rf_log_has "`lg'" "multiple partitions have no effect: the subpanels"
assert r(found) == 1

* network: group codes per absorbed term (missing absorbed values: dropped by prepare, so the
* codes are nonmissing on the sample), the time variable with lags()
gen long imp = mod(id * 7, 11)
replace imp = . in 5
egen long it = group(imp year)
egen long et = group(id year)
egen long pair = group(id imp)
stage_bc 2 3
felogit y x1 x2, absorb(et it pair) analytical network lags(2)
expect_glm "bc=analytical" "structure=network" "which=both" "lags=2" "nbc=4"
assert !strpos(" $rfmock_glm ", " effects=") & "`e(structure)'" == "network" & "`e(id)'" == ""
assert "`e(bctitle1)'" == "Type of included effects: exporter-time, importer-time and pair effects"
stage_bc 2 2
felogit y x1 x2, absorb(id#year imp#year) ape network
expect_glm "bc=ape" "structure=network" "nbc=2"
assert e(N_group) == .

* sample-dependent errors: population() below N*, lags() beyond the shortest panel (logitfe)
stage_bc 2 2
rcof "noi felogit y x1 x2, absorb(id year) analytical population(398)" == 198
rcof "noi felogit y x1 x2, absorb(id year) ape population(0)" == 198
rcof "noi felogit y x1 x2, absorb(id year) analytical ape" == 198
rcof "noi felogit y x1 x2, absorb(id year) analytical lags(7)" == 198
stage_bc 2 2
felogit y x1 x2, absorb(id year) analytical lags(6)
expect_glm "lags=6"
* errors from the parser reach felogit (complete list in test_parse.do)
rcof "noi felogit y x1 x2 [fw=c], absorb(id year) analytical" == 101
rcof "noi felogit y x1 x2, absorb(id year) analytical jackknife" == 198
rcof "noi felogit y x1 x2, absorb(id year) separation(mu) ape" == 198
rcof "noi feppml y x1 x2, absorb(id year) analytical" == 198
xtset, clear
rcof "noi felogit y x1 x2, absorb(id year) analytical" == 5
di as result "J passed"

di as result "TEST_ADO_MOCK PASSED"
