* test_hac.do -- fereg's ivreghdfe kernel options (bw, kernel, kiefer, dkraay(#), psd0/psda) vs
* ivreghdfe 1.1.4 on IV and on OLS (ivreghdfe without endogenous regressors)
*
* IV: rf_compare_e of everything ivreghdfe posts (b and V within 1e-8, integer e() exact, other e()
* scalars rtol 1e-6, every e() macro incl. e(vce) e(kernel) e(tvar) e(ivar) e(bwchoice)
* e(hacsubtitleV) e(hacsubtitleV2), e(sample)) and the header lines (rf_display_diff of the part
* above the coefficient table). OLS: fereg posts reghdfe's e(); b and V of the regressors within
* 1e-8, and N, df_r, F, N_clust, e(bw), e(kernel), e(tvar), e(ivar), e(bwchoice), e(vce),
* e(vcetype), e(hacsubtitleV), e(hacsubtitleV2) equal to ivreghdfe's.
* feppml/felogit (borrowed): a kernel with bw(1) (no lags) equals robust / clustering on time.
*
* Run from tests/stata (needs fereg.plugin on the adopath and ivreghdfe/ivreg2/ranktest installed):
*   stata-mp -b do test_hac.do
version 16
clear all
set more off
capture confirm file "../../stata/fereg.ado"
if (!_rc) adopath ++ "`c(pwd)'/../../stata"
which fereg
which ivreghdfe
findfile fereg.plugin
di as text "plugin: `r(fn)'"

global IV_XMACROS version ivreg2cmd firsteqs

* header R F : the lines of the two IV displays above the coefficient table are identical
capture program drop header
program header
	args ref test
	tempfile fR fT
	foreach side in R T {
		local est = cond("`side'" == "R", "`ref'", "`test'")
		local cmd = cond("`side'" == "R", "ivreghdfe", "fereg")
		qui estimates restore `est'
		qui log using "`f`side''", text replace name(rf_hdr)
		`cmd'
		qui log close rf_hdr
	}
	mata: rf_header_cmp("`fR'", "`fT'")
end

* check <ivreghdfe/fereg arguments> : IV, compare everything
capture program drop check
program check
	local spec `"`0'"'
	if (!strpos(`"`spec'"', ",")) local spec `"`spec',"'
	quietly ivreghdfe `spec' tol(1e-12)
	estimates store R
	fereg `spec' tol(1e-12)
	estimates store F
	rf_compare_e R F, xscalars(ic) xmacros($IV_XMACROS) btol(1e-8) vtol(1e-8)
	header R F
end

* checkols <arguments> : OLS through ivreghdfe (no endogenous regressors) vs fereg
capture program drop checkols
program checkols
	local spec `"`0'"'
	if (!strpos(`"`spec'"', ",")) local spec `"`spec',"'
	quietly ivreghdfe `spec' tol(1e-12)
	estimates store R
	local names : colnames e(b)
	foreach m in vce vcetype kernel tvar ivar bwchoice hacsubtitleV hacsubtitleV2 clustvar1 clustvar2 {
		local R_`m' `"`e(`m')'"'
	}
	* (fereg's reghdfe-style e(clustvar1) is also set with one cluster)
	local xm = cond(`"`R_clustvar2'"' == "", "clustvar1", "")
	foreach s in N df_r F N_clust bw {
		local R_`s' = e(`s')
	}
	fereg `spec' tol(1e-12)
	estimates store F
	local ms vce vcetype kernel tvar ivar bwchoice hacsubtitleV hacsubtitleV2 clustvar1 clustvar2
	local ms : list ms - xm
	foreach m of local ms {
		if (`"`e(`m')'"' != `"`R_`m''"') {
			di as err `"FAIL e(`m'): <`e(`m')'> vs <`R_`m''>"'
			exit 9
		}
	}
	foreach s in N df_r N_clust bw {
		rf_assert_close e(`s') `=e(`s')' `R_`s'' 0
	}
	rf_assert_close e(F) `=e(F)' `R_F' 1e-6
	tempname bR VR bF VF
	qui estimates restore R
	matrix `bR' = e(b)
	matrix `VR' = e(V)
	qui estimates restore F
	matrix `bF' = e(b)
	matrix `VF' = e(V)
	mata: rf_cmp_named("`bF'", "`VF'", "`bR'", "`VR'", "`names'", 1e-8)
end

capture mata: mata drop rf_cmp_named() rf_header_cmp()
mata:
// b and V of the coefficients `names' (all of REF's) in TEST and REF agree (V on the correlation scale)
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
		B1, B2
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

// The lines before the first line of the coefficient table ("----...") are identical
void rf_header_cmp(string scalar fR, string scalar fT)
{
	string colvector a, b
	real scalar i, n
	a = rf_head_lines(fR)
	b = rf_head_lines(fT)
	n = max((rows(a), rows(b)))
	if (rows(a) != rows(b)) printf("{err}header lengths differ: %g vs %g\n", rows(b), rows(a))
	for (i = 1; i <= n; i++) {
		if ((i > rows(a) ? "" : a[i]) != (i > rows(b) ? "" : b[i])) {
			printf("{err}REF : %s\n", i > rows(a) ? "" : a[i])
			printf("{err}TEST: %s\n", i > rows(b) ? "" : b[i])
			exit(9)
		}
	}
	printf("{txt}headers identical (%g lines)\n", rows(a))
}

string colvector rf_head_lines(string scalar f)
{
	string colvector L, out
	real scalar fh
	string scalar l
	fh = fopen(f, "r")
	out = J(0, 1, "")
	while ((l = fget(fh)) != J(0, 0, "")) {
		if (regexm(strtrim(l), "^(name|log|log type|opened on|closed on):")) continue
		// the coefficient table starts with a full-width rule (the title underline is shorter)
		if (substr(strtrim(l), 1, 4) == "----" & strlen(strtrim(l)) >= 70) break
		out = out \ l
	}
	fclose(fh)
	return(out)
}
end

* ---------------------------------------------------------------------------
* Panel data (xtset idcode year; the years have gaps: lags are calendar lags within idcode)
* ---------------------------------------------------------------------------
webuse nlswork, clear
keep if idcode <= 1500
keep idcode year ln_wage age tenure union south msp hours ind_code ttl_exp
foreach v of varlist _all {
	qui drop if missing(`v')
}
gen double w = 0.5 + mod(idcode * 7 + year, 10) / 10
xtset idcode year

local iv ln_wage age (tenure = union south msp)
local ols ln_wage age ttl_exp tenure

* all eight kernels, AC and HAC, integer and non-integer bandwidths
foreach k in bartlett parzen truncated thann thamm qs danielle tent {
	foreach bw in 2 3.5 {
		check `iv', absorb(idcode) kernel(`k') bw(`bw')
		check `iv', absorb(idcode) kernel(`k') bw(`bw') vce(robust)
		checkols `ols', absorb(idcode) kernel(`k') bw(`bw')
		checkols `ols', absorb(idcode) kernel(`k') bw(`bw') vce(robust)
	}
}
* bw = 1 (lag kernels: no lags and the s_vkernel note), default kernel, abbreviations
check `iv', absorb(idcode) bw(1)
check `iv', absorb(idcode) bw(1) kernel(qua) vce(robust)
check `iv', absorb(idcode year) bw(3)
checkols `ols', absorb(idcode year) bw(5) kernel(Parzen) vce(robust)
di as result "kernels passed"

* kiefer, dkraay(#), cluster + kernel (Driscoll-Kraay, Thompson), psd0/psda
check `iv', absorb(idcode) kiefer
checkols `ols', absorb(idcode) kiefer
check `iv', absorb(idcode) dkraay(2)
check `iv', absorb(idcode) dkraay(3) vce(robust)
check `iv', absorb(idcode) dkraay(2) vce(cluster year)
checkols `ols', absorb(idcode) dkraay(3)
check `iv', absorb(idcode) vce(cluster year) bw(3)
check `iv', absorb(idcode) vce(cluster year) bw(2.5) kernel(qs)
checkols `ols', absorb(idcode) vce(cluster year) bw(4) kernel(parzen)
check `iv', absorb(idcode) vce(cluster idcode year) bw(2) kernel(tru)
check `iv', absorb(idcode) vce(cluster year idcode) bw(3)
checkols `ols', absorb(year) vce(cluster idcode year) bw(2)
check `iv', absorb(idcode) vce(robust) bw(4) kernel(qs) psd0
check `iv', absorb(idcode) vce(robust) bw(4) kernel(dan) psda
checkols `ols', absorb(idcode) bw(3) kernel(tent) psda
di as result "kiefer/dkraay/cluster/psd passed"

* weights: aw (HAC and AC), pw (forces robust, also with kiefer)
check `iv' [aw=w], absorb(idcode) bw(3)
check `iv' [aw=w], absorb(idcode) bw(3) vce(robust)
check `iv' [pw=w], absorb(idcode) bw(3) kernel(par)
check `iv' [pw=w], absorb(idcode) kiefer
checkols `ols' [aw=w], absorb(idcode) vce(cluster year) bw(2)
checkols `ols' [pw=w], absorb(idcode) bw(2)
di as result "weights passed"

* first stages with kernels
check `iv', absorb(idcode) bw(3) vce(robust) ffirst
check `iv', absorb(idcode) dkraay(2) ffirst
di as result "ffirst passed"

* tsset delta: time in units of 2 (calendar lags of one delta)
gen int year2 = 2 * year
xtset idcode year2, delta(2)
check `iv', absorb(idcode) bw(3) vce(robust)
checkols `ols', absorb(idcode) bw(2)
xtset idcode year
di as result "delta passed"

* ---------------------------------------------------------------------------
* Time series (tsset, no panel): bw(auto) (Newey-West 1994), time gaps
* ---------------------------------------------------------------------------
preserve
clear
set seed 20261013
set obs 400
gen int t = _n + floor(_n / 37)
gen byte g = mod(t, 4)
gen double z1 = rnormal()
gen double z2 = rnormal()
gen double u = rnormal()
replace u = 0.6 * u[_n-1] + u in 2/l
gen double x = 0.5 * rnormal() + 0.3 * g
replace x = 0.5 * x[_n-1] + x in 2/l
gen double en = 0.7 * z1 + 0.4 * z2 + 0.5 * u + rnormal()
gen double y = 1 + 0.5 * x + 0.8 * en + 0.2 * g + u
gen double w = 0.5 + runiform()
tsset t
foreach k in bartlett parzen qs thann {
	check y x (en = z1 z2), absorb(g) bw(auto) kernel(`k')
	check y x (en = z1 z2), absorb(g) bw(auto) kernel(`k') vce(robust)
	checkols y x, absorb(g) bw(auto) kernel(`k') vce(robust)
}
check y x (en = z1 z2) [aw=w], absorb(g) bw(auto) vce(robust)
check y x (en = z1 z2), absorb(g) bw(7.5) kernel(tent) vce(robust)
checkols y x, absorb(g) bw(6)
restore
di as result "time series passed"

* ---------------------------------------------------------------------------
* Errors: texts and return codes of ivreghdfe/ivparse/s_vkernel
* ---------------------------------------------------------------------------
capture program drop errcheck
program errcheck
	* errcheck "<arguments>" : fereg and ivreghdfe fail with the same rc and the same last line
	args spec
	tempfile fR fT
	foreach side in R T {
		local cmd = cond("`side'" == "R", "ivreghdfe", "fereg")
		qui log using "`f`side''", text replace name(rf_err)
		capture noisily `cmd' `spec'
		local rc`side' = _rc
		qui log close rf_err
	}
	if (`rcR' == 0 | `rcR' != `rcT') {
		di as err `"FAIL `spec': rc `rcT' vs `rcR'"'
		exit 9
	}
	mata: rf_lastline_cmp("`fR'", "`fT'")
	di as text `"ok  `spec' -> r(`rcR')"'
end
capture mata: mata drop rf_lastline_cmp()
mata:
void rf_lastline_cmp(string scalar fR, string scalar fT)
{
	string scalar a, b, l
	real scalar fh
	a = b = ""
	fh = fopen(fR, "r")
	while ((l = fget(fh)) != J(0, 0, "")) if (strtrim(l) != "" & !regexm(strtrim(l), "^(name|log|log type|opened on|closed on):")) a = l
	fclose(fh)
	fh = fopen(fT, "r")
	while ((l = fget(fh)) != J(0, 0, "")) if (strtrim(l) != "" & !regexm(strtrim(l), "^(name|log|log type|opened on|closed on):")) b = l
	fclose(fh)
	if (a != b) {
		printf("{err}REF : %s\nTEST: %s\n", a, b)
		exit(9)
	}
}
end

gen int fw = 1 + mod(idcode, 3)
foreach spec in "bw(0)" "bw(-1)" "bw(abc)" "kernel(bar)" "kernel(bogus) bw(2)" "kernel(newey) bw(2)" ///
	"bw(auto)" "dkraay(2) bw(3)" "dkraay(2) cluster(idcode)" "kiefer vce(robust)" ///
	"kiefer cluster(idcode)" "kiefer kernel(par)" "kiefer bw(2)" "bw(2) psd0 psda" ///
	"bw(30)" "bw(20.5)" "bw(2) cluster(idcode)" "bw(2) cluster(idcode ind_code)" {
	errcheck `"`iv', absorb(idcode) `spec'"'
}
errcheck `"`iv' [fw=fw], absorb(idcode) bw(2)"'
xtset, clear
foreach spec in "bw(2)" "kiefer" "dkraay(2)" {
	errcheck `"`iv', absorb(idcode) `spec'"'
}
xtset idcode year
* fereg-only rules (no reference): kernels with hc2/hc3/crv3/dkraay/conley, psd without a kernel
gen double lat = 30 + mod(idcode, 17)
gen double lon = -100 + mod(idcode, 23)
foreach spec in "vce(hc2) bw(2)" "vce(crv3 idcode) bw(2)" "vce(dkraay) bw(2)" "vce(conley lat lon) bw(2)" "psd0" ///
	"bw(2) sw" "bw(2) center" "bw(2) dofminus(1)" "bw(2) sdofminus(1)" {
	capture fereg `ols', absorb(idcode) `spec'
	assert _rc == 198
}
capture fereg `iv', absorb(idcode) vce(hc3)
assert _rc == 198
di as result "error paths passed"

* ---------------------------------------------------------------------------
* feppml / felogit (borrowed): no lags = robust / clustering on time; every kernel runs
* ---------------------------------------------------------------------------
gen int wks = round(hours / 4)
foreach cmd in feppml felogit {
	local y = cond("`cmd'" == "feppml", "wks", "union")
	quietly `cmd' `y' age tenure, absorb(idcode) vce(robust)
	estimates store A
	`cmd' `y' age tenure, absorb(idcode) vce(robust) bw(1)
	estimates store B
	assert "`e(kernel)'" == "Bartlett" & e(bw) == 1 & "`e(tvar)'" == "year" & "`e(ivar)'" == "idcode"
	assert "`e(vce)'" == "robust hac bartlett bw=1"
	rf_compare_e A B, btol(1e-10) vtol(1e-8) xmacros(vce title3)
	quietly `cmd' `y' age tenure, absorb(idcode) vce(cluster year)
	estimates store A
	`cmd' `y' age tenure, absorb(idcode) vce(cluster year) bw(1)
	estimates store B
	rf_compare_e A B, btol(1e-10) vtol(1e-8) xmacros(vce)
	foreach k in bartlett parzen truncated thann thamm qs danielle tent {
		`cmd' `y' age tenure, absorb(idcode) vce(robust) kernel(`k') bw(2.5)
		mata: assert(all(diagonal(st_matrix("e(V)")) :> 0))
	}
	`cmd' `y' age tenure, absorb(idcode) dkraay(3)
	`cmd' `y' age tenure, absorb(idcode) vce(cluster idcode year) bw(2) kernel(tru)
}
felogit union age tenure, absorb(idcode) bw(3)
assert "`e(vce)'" == "ac bartlett bw=3" & "`e(vcetype)'" == ""
felogit union age tenure, absorb(idcode) kiefer
assert "`e(vce)'" == "ac truncated bw=" + string(e(bw))
capture feppml wks age tenure, absorb(idcode) kiefer
assert _rc == 198
di as result "feppml/felogit passed"

di as result "test_hac passed"
