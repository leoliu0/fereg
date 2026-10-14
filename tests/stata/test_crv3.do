* test_crv3.do -- fereg/feppml/felogit vce(crv3 clustvars) vs a do-file jackknife loop
*
* Reference: leave one cluster out and refit with the reference command (reghdfe, ivreghdfe,
* ppmlhdfe, logit with dummies), V = factor * sum_S sign_S sum_g (b_(-g) - b)(b_(-g) - b)' over the
* non-empty subsets S of the cluster dimensions (intersections by egen group), with the cluster
* factor of the reference: reghdfe (N-1)/(N-nested_adj-df_m-df_a)*G/(G-1), ivreghdfe
* (N-1)/(N-K-sdofminus)*G/(G-1), GLM G/(G-1); G = the smallest number of clusters.
* The refits keep singletons (they never move b; fereg's jackknifed _cons averages over every
* observation outside the cluster). feppml/felogit: slopes only (their _cons is linearized).
* Assertions: b and V within 1e-8 (V on the correlation scale), e(vce) crv3, e(N_clust), e(df_r).
*
* Run from tests/stata (needs fereg.plugin, reghdfe, ivreghdfe, ppmlhdfe): stata-mp -b do test_crv3.do
version 16
clear all
set more off
capture confirm file "../../stata/fereg.ado"
if (!_rc) adopath ++ "`c(pwd)'/../../stata"
which fereg
findfile fereg.plugin
di as text "plugin: `r(fn)'"

* jk <command and variables> , cl(varlist) names(coefficients) [wgt([..]) opts(string)]
*   jackknife meat of the coefficients names() over the clusters of each non-empty subset of cl(),
*   inclusion-exclusion signs; the sample is the current e(sample) (saved in variable _jk_s)
*   returns matrix jk_M, scalar jk_G (smallest number of clusters)
capture program drop jk
program jk
	syntax anything(equalok), cl(varlist) names(string) [wgt(string) opts(string)]
	tempname b0 bg M
	tempvar grp
	capture drop _jk_s
	qui gen byte _jk_s = e(sample)
	qui `anything' `wgt' if _jk_s, `opts'
	matrix `b0' = e(b)
	local k : word count `names'
	matrix `M' = J(`k', `k', 0)
	local nd : word count `cl'
	local Gmin .
	forval mask = 1/`=2^`nd' - 1' {
		local sub
		local size 0
		forval j = 1/`nd' {
			if (mod(floor(`mask' / 2^(`j' - 1)), 2)) {
				local sub `sub' `: word `j' of `cl''
				local ++size
			}
		}
		local sign = cond(mod(`size', 2), 1, -1)
		capture drop `grp'
		qui egen long `grp' = group(`sub') if _jk_s
		qui summarize `grp', meanonly
		local G = r(max)
		if (`size' == 1) local Gmin = min(`Gmin', `G')
		forval g = 1/`G' {
			qui `anything' `wgt' if _jk_s & `grp' != `g', `opts'
			matrix `bg' = e(b)
			mata: rf_jk_add("`M'", "`bg'", "`b0'", "`names'", `sign')
		}
	}
	matrix jk_M = `M'
	scalar jk_G = `Gmin'
end

capture mata: mata drop rf_jk_add() rf_sel() rf_cmp_V()
mata:
real rowvector rf_sel(string scalar b, string scalar names)
{
	string rowvector nm, c
	real rowvector idx
	real scalar j, k
	nm = tokens(names)
	c = st_matrixcolstripe(b)[., 2]'
	idx = J(1, cols(nm), .)
	for (j = 1; j <= cols(nm); j++) for (k = 1; k <= cols(c); k++) if (c[k] == nm[j]) idx[j] = k
	if (missing(idx)) {
		printf("{err}coefficients %s not all in %s\n", names, invtokens(c))
		exit(9)
	}
	return(st_matrix(b)[idx])
}
void rf_jk_add(string scalar M, string scalar bg, string scalar b0, string scalar names, real scalar sign)
{
	real rowvector d
	d = rf_sel(bg, names) - rf_sel(b0, names)
	st_matrix(M, st_matrix(M) + sign :* (d' * d))
}
// fereg's V (coefficients names) vs factor * jk_M; also asserts the reference is PSD
void rf_cmp_V(string scalar names, real scalar factor, real scalar tol)
{
	real matrix VR, VT
	real rowvector s, idx
	string rowvector nm, c
	real scalar j, k
	VR = factor :* st_matrix("jk_M")
	nm = tokens(names)
	c = st_matrixcolstripe("e(V)")[., 2]'
	idx = J(1, cols(nm), .)
	for (j = 1; j <= cols(nm); j++) for (k = 1; k <= cols(c); k++) if (c[k] == nm[j]) idx[j] = k
	VT = st_matrix("e(V)")[idx, idx]
	if (min(symeigenvalues(VR)) < 0) {
		printf("{err}reference not PSD (the PSD fix would apply): choose other data\n")
		exit(9)
	}
	s = sqrt(diagonal(VR))'
	if (any(abs(VT - VR) :> tol :* (s' * s))) {
		VT, VR
		printf("{err}FAIL V (factor %g)\n", factor)
		exit(9)
	}
	printf("{txt}crv3 V of %s agrees with the jackknife (factor %g, tol %g)\n", names, factor, tol)
}
end

* ---------------------------------------------------------------------------
* OLS vs reghdfe refits; factor from the full-sample reghdfe cluster dof (fereg's own e())
* ---------------------------------------------------------------------------
webuse nlswork, clear
keep if idcode <= 600
keep idcode year ln_wage age tenure ttl_exp union south hours ind_code occ_code
foreach v of varlist _all {
	qui drop if missing(`v')
}
gen double w = 0.5 + mod(idcode * 7 + year, 10) / 10

capture program drop checkols
program checkols
	* checkols "<varlist>" "<absorb(...) or noabsorb>" "<crv3 vars>" ["[aw=w]"]
	args vars abs cl wgt
	fereg `vars' `wgt', `abs' vce(crv3 `cl') tol(1e-12)
	assert "`e(vce)'" == "crv3" & "`e(vcetype)'" == "CRV3" & "`e(clustvar)'" == "`cl'"
	local names : colnames e(b)
	local N = e(N)
	local adj = (e(df_a_nested) > 0)
	local q = (`N' - 1) / (`N' - `adj' - e(df_m) - e(df_a))
	estimates store F
	local opts = cond("`abs'" == "noabsorb", "noabsorb", "`abs' keepsingletons") + " tol(1e-12)"
	jk reghdfe `vars', cl(`cl') names(`names') wgt(`wgt') opts(`opts')
	qui estimates restore F
	rf_assert_close e(N_clust) `=e(N_clust)' `=jk_G' 0
	assert e(df_r) == min(`N' - e(df_m) - e(df_a), jk_G - 1)
	mata: rf_cmp_V("`names'", `q' * st_numscalar("jk_G") / (st_numscalar("jk_G") - 1), 1e-8)
end

checkols "ln_wage age tenure ttl_exp" "absorb(idcode)" "idcode"
checkols "ln_wage age tenure ttl_exp" "absorb(idcode year)" "idcode"
checkols "ln_wage age tenure ttl_exp" "absorb(idcode year)" "ind_code"
checkols "ln_wage age tenure ttl_exp" "absorb(idcode year)" "ind_code" "[aw=w]"
checkols "ln_wage age tenure ttl_exp" "absorb(year)" "ind_code occ_code"
checkols "ln_wage age tenure" "noabsorb" "ind_code"
di as result "OLS passed"

* ---------------------------------------------------------------------------
* IV vs ivreghdfe refits (slopes; ivreghdfe has no constant with absorb)
* ---------------------------------------------------------------------------
foreach cl in idcode ind_code {
	quietly ivreghdfe ln_wage age (tenure = union south), absorb(idcode year) cluster(`cl') tol(1e-12)
	local q = (e(N) - 1) / (e(N) - e(rankxx) - e(sdofminus))
	fereg ln_wage age (tenure = union south), absorb(idcode year) vce(crv3 `cl') tol(1e-12)
	assert "`e(vce)'" == "crv3"
	local names : colnames e(b)
	estimates store F
	jk ivreghdfe ln_wage age (tenure = union south), cl(`cl') names(`names') opts(absorb(idcode year) tol(1e-12))
	qui estimates restore F
	mata: rf_cmp_V("`names'", `q' * st_numscalar("jk_G") / (st_numscalar("jk_G") - 1), 1e-8)
}
di as result "IV passed"

* ---------------------------------------------------------------------------
* feppml vs ppmlhdfe refits, felogit vs logit-with-dummies refits (slopes; factor G/(G-1))
* ---------------------------------------------------------------------------
keep if idcode <= 250
gen int wks = round(hours / 4)
feppml wks age tenure, absorb(idcode) vce(crv3 ind_code) tol(1e-12)
assert "`e(vce)'" == "crv3"
estimates store F
jk ppmlhdfe wks age tenure, cl(ind_code) names(age tenure) opts(absorb(idcode) tol(1e-12))
qui estimates restore F
mata: rf_cmp_V("age tenure", st_numscalar("jk_G") / (st_numscalar("jk_G") - 1), 1e-7)

felogit union age tenure, absorb(year) vce(crv3 idcode) tol(1e-12)
assert "`e(vce)'" == "crv3"
estimates store F
jk logit union age tenure i.year, cl(idcode) names(age tenure) opts(tolerance(1e-12) nolog)
qui estimates restore F
mata: rf_cmp_V("age tenure", st_numscalar("jk_G") / (st_numscalar("jk_G") - 1), 1e-7)
di as result "feppml/felogit passed"

* errors
capture fereg ln_wage age, absorb(idcode) vce(crv3)
assert _rc == 198
capture fereg ln_wage age (tenure = union), absorb(idcode) vce(crv3 idcode year ind_code)
assert _rc == 198
di as result "test_crv3 passed"
