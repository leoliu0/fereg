* test_conley.do -- fereg/feppml/felogit vce(conley latvar lonvar, cutoff() pixel() distance()) vs fixest
*
* Reference: conley_ref.csv, written by gen_conley_fixture.R with fixest 0.14.2 on the data in
* conley_geo.csv (b, fixest's Conley V without small-sample factor, the cutoff used). fereg's convention
* (a fereg extension, help fereg): the same meat times N/(N - df_m - df_a) (fereg OLS and IV, df_r =
* N - df_m - df_a) or N/(N - 1) (feppml, felogit). Assertions: b and V within 1e-8 (V on the correlation
* scale), e(conley_cutoff) equal to fixest's (also the deduced one), e(vce) conley.
*
* Run from tests/stata (needs fereg.plugin on the adopath): stata-mp -b do test_conley.do
version 16
clear all
set more off
capture confirm file "../../stata/fereg.ado"
if (!_rc) adopath ++ "`c(pwd)'/../../stata"
which fereg
findfile fereg.plugin
di as text "plugin: `r(fn)'"

capture program drop check
program check
	* check <spec> <command line> ; factor: ols (N/(N-df_m-df_a)) or glm (N/(N-1))
	gettoken spec 0 : 0
	gettoken kind 0 : 0
	`0'
	assert "`e(vce)'" == "conley" & "`e(vcetype)'" == "Conley"
	rf_assert_close e(conley_cutoff) `=e(conley_cutoff)' `=scalar(conley_`spec'_cut)' 1e-12
	if ("`kind'" == "ols") {
		local f = e(N) / (e(N) - e(df_m) - e(df_a))
		if ("`e(model)'" == "iv") local f = e(N) / (e(N) - e(rankxx) - e(sdofminus))
	}
	else local f = e(N) / (e(N) - 1)
	local tol = 1e-8
	mata: rf_cmp_ref("`spec'", `f', `tol')
end

capture mata: mata drop rf_load_ref() rf_cmp_ref()
mata:
void rf_load_ref()
{
	string matrix D
	string rowvector specs, nm
	string scalar s
	real scalar i, j, k, n
	real matrix V
	real rowvector b
	D = st_sdata(., ("spec", "row", "col", "value"))
	specs = uniqrows(D[., 1])'
	for (k = 1; k <= cols(specs); k++) {
		s = specs[k]
		nm = J(1, 0, "")
		for (i = 1; i <= rows(D); i++) if (D[i, 1] == s & D[i, 2] == "b") nm = nm, D[i, 3]
		n = cols(nm)
		b = J(1, n, .)
		V = J(n, n, .)
		for (i = 1; i <= rows(D); i++) {
			if (D[i, 1] != s) continue
			if (D[i, 2] == "cutoff") st_numscalar("conley_" + s + "_cut", strtoreal(D[i, 4]))
			else if (D[i, 2] == "b") b[selectindex(nm :== D[i, 3])] = strtoreal(D[i, 4])
			else V[selectindex(nm :== D[i, 2]), selectindex(nm :== D[i, 3])] = strtoreal(D[i, 4])
		}
		st_matrix("conley_" + s + "_b", b)
		st_matrixcolstripe("conley_" + s + "_b", (J(n, 1, ""), nm'))
		st_matrix("conley_" + s + "_V", V)
	}
}
void rf_cmp_ref(string scalar spec, real scalar f, real scalar tol)
{
	string rowvector nm, c
	real rowvector idx, bR, s
	real matrix VR, VT
	real scalar j, k
	nm = st_matrixcolstripe("conley_" + spec + "_b")[., 2]'
	bR = st_matrix("conley_" + spec + "_b")
	VR = f :* st_matrix("conley_" + spec + "_V")
	c = st_matrixcolstripe("e(b)")[., 2]'
	idx = J(1, cols(nm), .)
	for (j = 1; j <= cols(nm); j++) for (k = 1; k <= cols(c); k++) if (c[k] == nm[j]) idx[j] = k
	if (missing(idx)) {
		printf("{err}FAIL names %s vs %s\n", invtokens(nm), invtokens(c))
		exit(9)
	}
	VT = st_matrix("e(V)")[idx, idx]
	s = sqrt(diagonal(VR))'
	if (any(abs(st_matrix("e(b)")[idx] - bR) :> tol :* (abs(bR) + s))) {
		st_matrix("e(b)")[idx] \ bR
		printf("{err}FAIL b %s\n", spec)
		exit(9)
	}
	if (any(abs(VT - VR) :> tol :* (s' * s))) {
		VT, VR
		printf("{err}FAIL V %s (factor %g)\n", spec, f)
		exit(9)
	}
	printf("{txt}%s: b and V agree with fixest (factor %g, tol %g)\n", spec, f, tol)
}
end

* reference matrices: conley_<spec>_b, conley_<spec>_V, scalar conley_<spec>_cut
import delimited using conley_ref.csv, clear stringcols(_all) case(preserve)
mata: rf_load_ref()

import delimited using conley_geo.csv, clear case(preserve) asdouble
check ols_tri ols fereg y x1 x2, absorb(g) vce(conley lat lon, cutoff(500)) tol(1e-12)
assert e(df_r) == e(N) - e(df_m) - e(df_a) & "`e(conley_distance)'" == "triangular"
check ols_sph ols fereg y x1 x2, absorb(g) vce(conley lat lon, cutoff(300) distance(spherical)) tol(1e-12)
check ols_pixel ols fereg y x1 x2, absorb(g) vce(conley lat lon, cutoff(500) pixel(100)) tol(1e-12)
assert e(conley_pixel) == 100
check ols_deduce ols fereg y x1 x2, absorb(g) vce(conley lat lon) tol(1e-12)
check ols_aw ols fereg y x1 x2 [aw=w], absorb(g) vce(conley lat lon, cut(400) dist(spherical)) tol(1e-12)
check ols_nofe ols fereg y x1 x2, vce(conley lat lon, cutoff(600)) tol(1e-12)
check iv_tri ols fereg y x1 (en = z1), absorb(g) vce(conley lat lon, cutoff(500)) tol(1e-12)
check ppml_tri glm feppml yp x1 x2, absorb(g) vce(conley lat lon, cutoff(500)) tol(1e-12)
check logit_sph glm felogit yb x1 x2, absorb(g) vce(conley lat lon, cutoff(400) pixel(50) distance(spherical)) tol(1e-12)
di as result "fixest parity passed"

* missing coordinates drop the observation (as a cluster variable would)
replace lat = . in 1/5
fereg y x1 x2, absorb(g) vce(conley lat lon, cutoff(500))
assert e(N) == 395
* errors (fereg's texts)
foreach o in "conley lat" "conley lat lon x1" "conley lat lon, cutoff(-1)" "conley lat lon, cutoff(abc)" ///
	"conley lat lon, pixel(-2)" "conley lat lon, distance(manhattan)" "conley lat lon, bogus(1)" {
	capture fereg y x1 x2, absorb(g) vce(`o')
	assert _rc == 198
}
di as result "test_conley passed"
