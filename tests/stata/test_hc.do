* test_hc.do -- fereg vce(hc2|hc3) vs Stata regress with explicit dummies (regress y x i.fe, vce(hc2|hc3))
*
* regress's eq. (10)/(12): leverage of the full design incl. the dummies, no N/(N-k) factor, fweights
* tau_i = w_i. The data have no singleton groups (leverage 1, where regress returns arbitrary numbers).
* Assertions: b and V of the regressors within 1e-8 (V on the correlation scale), e(N), e(df_r),
* e(vce) and e(vcetype) equal to regress's. feppml/felogit (borrowed, IRLS leverage) run.
*
* Run from tests/stata (needs fereg.plugin on the adopath): stata-mp -b do test_hc.do
version 16
clear all
set more off
capture confirm file "../../stata/fereg.ado"
if (!_rc) adopath ++ "`c(pwd)'/../../stata"
which fereg
findfile fereg.plugin
di as text "plugin: `r(fn)'"

* check "<regress arguments>" "<fereg arguments>" "<regressor names>" [nodf]
*   (nodf: three absorbed FEs, where reghdfe's df_a may count fewer redundant levels than regress's rank)
capture program drop check
program check
	args rspec fspec names nodf
	quietly regress `rspec'
	estimates store R
	local R_N = e(N)
	local R_df_r = e(df_r)
	local R_vce `e(vce)'
	local R_vcetype `e(vcetype)'
	fereg `fspec' tol(1e-12)
	estimates store F
	rf_assert_close e(N) `=e(N)' `R_N' 0
	if ("`nodf'" == "") rf_assert_close e(df_r) `=e(df_r)' `R_df_r' 0
	assert "`e(vce)'" == "`R_vce'" & "`e(vcetype)'" == "`R_vcetype'"
	tempname bR VR bF VF
	qui estimates restore R
	matrix `bR' = e(b)
	matrix `VR' = e(V)
	qui estimates restore F
	matrix `bF' = e(b)
	matrix `VF' = e(V)
	mata: rf_cmp_named("`bF'", "`VF'", "`bR'", "`VR'", "`names'", 1e-8)
end

capture mata: mata drop rf_cmp_named()
mata:
void rf_cmp_named(string scalar bT, string scalar VT, string scalar bR, string scalar VR,
	string scalar names, real scalar tol)
{
	real rowvector it, ir, s
	real matrix B1, B2, V1, V2
	string rowvector nm, cT, cR
	real scalar j, k
	nm = tokens(names)
	cT = st_matrixcolstripe(bT)[., 2]'
	cR = st_matrixcolstripe(bR)[., 2]'
	it = J(1, cols(nm), .)
	ir = J(1, cols(nm), .)
	for (j = 1; j <= cols(nm); j++) {
		for (k = 1; k <= cols(cT); k++) if (cT[k] == nm[j]) it[j] = k
		for (k = 1; k <= cols(cR); k++) if (cR[k] == nm[j]) ir[j] = k
	}
	if (missing(it) | missing(ir)) {
		printf("{err}FAIL coefficient names: %s\n", names)
		exit(9)
	}
	B1 = st_matrix(bT)[it]; B2 = st_matrix(bR)[ir]
	V1 = st_matrix(VT)[it, it]; V2 = st_matrix(VR)[ir, ir]
	s = sqrt(diagonal(V2))'
	if (any(abs(B1 - B2) :> tol :* (abs(B2) + s))) {
		B1 \ B2
		printf("{err}FAIL b\n")
		exit(9)
	}
	if (any(abs(V1 - V2) :> tol :* (s' * s))) {
		V1, V2
		printf("{err}FAIL V\n")
		exit(9)
	}
	printf("{txt}b and V of %s agree (tol %g)\n", names, tol)
}
end

* ---------------------------------------------------------------------------
* nlswork, idcode <= 400, no singleton idcode/year/ind_code groups
* ---------------------------------------------------------------------------
webuse nlswork, clear
keep if idcode <= 400
keep idcode year ln_wage age tenure ttl_exp hours ind_code union
foreach v of varlist _all {
	qui drop if missing(`v')
}
forval i = 1/3 {
	foreach v in idcode year ind_code {
		qui bys `v': drop if _N == 1
	}
}
gen double w = 0.5 + mod(idcode * 7 + year, 10) / 10
gen int fw = 1 + mod(idcode + year, 3)

foreach hc in hc2 hc3 {
	check "ln_wage age tenure i.idcode, vce(`hc')" "ln_wage age tenure, absorb(idcode) vce(`hc')" "age tenure"
	check "ln_wage age tenure ttl_exp i.idcode i.year, vce(`hc')" "ln_wage age tenure ttl_exp, absorb(idcode year) vce(`hc')" "age tenure ttl_exp"
	check "ln_wage age tenure i.idcode i.year i.ind_code, vce(`hc')" "ln_wage age tenure, absorb(idcode year ind_code) vce(`hc')" "age tenure" nodf
	check "ln_wage age tenure i.idcode [aw=w], vce(`hc')" "ln_wage age tenure [aw=w], absorb(idcode) vce(`hc')" "age tenure"
	check "ln_wage age tenure i.idcode i.year [fw=fw], vce(`hc')" "ln_wage age tenure [fw=fw], absorb(idcode year) vce(`hc')" "age tenure"
	check "ln_wage age tenure i.idcode i.ind_code c.hours#i.ind_code, vce(`hc')" "ln_wage age tenure, absorb(idcode ind_code##c.hours) vce(`hc')" "age tenure" nodf
	check "ln_wage age c.age#c.age tenure, vce(`hc')" "ln_wage age c.age#c.age tenure, vce(`hc')" "age c.age#c.age tenure _cons"
}
* pweights: allowed with hc2 (as aweights), not with hc3 (regress's message)
check "ln_wage age tenure i.idcode [pw=w], vce(hc2)" "ln_wage age tenure [pw=w], absorb(idcode) vce(hc2)" "age tenure"
capture regress ln_wage age tenure [pw=w], vce(hc3)
local rc_r = _rc
capture fereg ln_wage age tenure [pw=w], absorb(idcode) vce(hc3)
assert _rc == `rc_r' & _rc == 198
* clustered forms (Bell-McCaffrey) are not supported; IV is rejected (ivreghdfe's message)
capture fereg ln_wage age tenure, absorb(idcode) vce(hc2 idcode)
assert _rc == 198
capture fereg ln_wage age (tenure = union), absorb(idcode) vce(hc2)
assert _rc == 198
di as result "regress parity passed"

* ---------------------------------------------------------------------------
* feppml / felogit (borrowed: IRLS leverage, no factor)
* ---------------------------------------------------------------------------
gen int wks = round(hours / 4)
foreach cmd in feppml felogit {
	local y = cond("`cmd'" == "feppml", "wks", "union")
	foreach hc in hc2 hc3 {
		quietly `cmd' `y' age tenure, absorb(idcode) vce(robust)
		tempname Vr
		matrix `Vr' = e(V)
		`cmd' `y' age tenure, absorb(idcode) vce(`hc')
		assert "`e(vce)'" == "`hc'" & "`e(vcetype)'" == "Robust " + upper("`hc'")
		* the leverage inflates every observation's weight: diagonal at least the robust one
		* rescaled by (N-1)/N (robust's factor)
		mata: assert(all(diagonal(st_matrix("e(V)")) :>= diagonal(st_matrix("`Vr'")) :* (st_numscalar("e(N)") - 1) / st_numscalar("e(N)") :* (1 - 1e-12)))
	}
}
di as result "feppml/felogit passed"

di as result "test_hc passed"
