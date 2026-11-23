* test_wildboot.do -- Wild bootstrap tests for fereg (OLS/IV), feppml, and felogit
*
* Covers:
*   - Independent Mata explicit-dummy WLS brute-force oracle (WCR11, WCU11, joint F, non-nested FE)
*   - MNW (11, 13, 31, 33), Hetero (11, 21, 31, 13, 33), 2-way cluster, bootcluster() (string & #)
*   - Joint q=2 via (eq1) (eq2), linear combinations, rmat()/rvec(), savet() (incl. full-enum trim)
*   - noabsorb _cons test, weights (aw/pw/fw), pvaltype(equal/greater/less), noci, level(90), threads
*   - IV WRE and Score, feppml (offset/exposure), felogit, replay, error handling
*
* Run from tests/stata: stata-mp -b do test_wildboot.do
version 16
clear all
set more off
capture confirm file "../../stata/fereg.ado"
if (!_rc) adopath ++ "`c(pwd)'/../../stata"
which fereg
findfile fereg.plugin
di as text "plugin: `r(fn)'"

capture program drop assert_close
program assert_close
	args label got want tol
	if (missing(`got') | missing(`want')) {
		di as err "FAIL `label': missing value (got=`got', want=`want')"
		exit 9
	}
	local diff = abs(`got' - `want')
	if (`diff' > `tol') {
		di as err "FAIL `label': got `got', want `want', diff `diff' > `tol'"
		exit 9
	}
end

* Generate deterministic synthetic dataset covering OLS, IV, PPML, Logit, weights, string/nested/nn FE & clusters
set obs 200
set seed 20261123
gen long obs_id = _n
gen int fe8 = mod((_n - 1) * 3 + floor((_n - 1) / 7), 8) + 1
gen int c4 = floor((fe8 - 1) / 2) + 1
gen int fe_nn = mod((_n - 1) * 5 + floor((_n - 1) / 5) * 2, 7) + 1
gen int c10 = mod((_n - 1) + floor((_n - 1) / 11) * 3, 10) + 1
gen int c6 = mod((_n - 1) * 7 + floor((_n - 1) / 3), 6) + 1
gen str6 c4_str = "g" + string(c4)
gen int fac3 = mod(_n - 1, 3) + 1

gen double u_fe = 0.35 * sin(fe8 * 1.3) - 0.25 * cos(fe_nn * 0.9)
gen double u_cl = 0.35 * sin(c10 * 0.7) + 0.30 * cos(c6 * 1.1) + 0.25 * sin(c4 * 1.7)

gen double x1 = rnormal() + 0.25 * sin(fe_nn)
gen double x2 = rnormal() + 0.20 * cos(c10)
gen double z1 = rnormal()
gen double z2 = rnormal()
gen double z3 = rnormal()
gen double u_iv = rnormal() + 0.25 * u_cl
gen double e1 = 0.65 * z1 - 0.40 * z2 + 0.20 * x1 + 0.50 * u_iv + 0.15 * sin(fe_nn) + 0.35 * rnormal()
gen double e2 = -0.35 * z2 + 0.70 * z3 + 0.15 * x2 + 0.40 * u_iv + 0.35 * rnormal()

gen double y = 0.75 * x1 - 0.50 * x2 + 0.40 + u_fe + 0.40 * u_cl + rnormal()
gen double y_iv = 0.90 * e1 + 0.50 * x1 - 0.30 * x2 + 0.30 + u_fe + u_iv

gen double aw = 0.5 + 1.5 * runiform()
gen int fw = 1 + floor(runiform() * 3)
gen double offv = 0.15 * rnormal()
gen double expv = exp(offv)

* Ensure positive/varying counts & binary outcomes across fe_nn and fe8
gen double lam_p = exp(min(2, max(-2, 0.3 + 0.35 * x1 - 0.20 * x2 + 0.25 * u_fe + offv)))
gen int y_pois = rpoisson(lam_p)
bysort fe8: egen double s_p1 = total(y_pois)
bysort fe_nn: egen double s_p2 = total(y_pois)
bysort fe8: replace y_pois = y_pois + 1 if _n == 1 & (s_p1 == 0 | s_p2 == 0)
drop s_p1 s_p2

gen double pr_l = invlogit(0.15 + 0.60 * x1 - 0.45 * x2 + 0.20 * u_fe)
gen byte y_logit = runiform() < pr_l
bysort fe8 (obs_id): replace y_logit = 1 - y_logit if _n == 1
bysort fe_nn (obs_id): replace y_logit = 1 - y_logit if _n == 1
sort obs_id

* -----------------------------------------------------------------------------
* 1. Independent Mata explicit-dummy WLS brute-force oracle vs fereg wildboot
* -----------------------------------------------------------------------------
mata:
void _test_wb_oracle(
	string scalar yname,
	string scalar xnames_str,
	string scalar fename,
	string scalar clname,
	string scalar wname,
	real scalar nested,
	real matrix R_in,
	real colvector r_in,
	real scalar impose_null,
	string scalar savet_mat
)
{
	real colvector y, fe, cl, wv, idx, eigvals, b_u, e_u, b_r, e_r, yhat_star, e_star
	real colvector b_star, r_target, tb_vec, tb_fereg, yb, b_b, e_b, gap, se_b, lam
	real matrix X, D, Q, eigvecs, Z, A, Ainv, meat, V, R_full, M_r
	real scalar n, k, g_fe, p, G, bigN, nested_adj, df_a, ssc, stat0, pt0, se0, bdraw, g, i, p_oracle, eps

	eigvals = .
	eigvecs = .

	y = st_data(., yname)
	X = st_data(., tokens(xnames_str))
	fe = st_data(., fename)
	cl = st_data(., clname)
	n = rows(y)
	k = cols(X)
	if (wname != "") {
		wv = st_data(., wname)
		wv = wv :* (n / sum(wv))
	}
	else {
		wv = J(n, 1, 1)
	}
	bigN = n

	g_fe = max(fe)
	D = J(n, g_fe, 0)
	for (i = 1; i <= n; i++) D[i, fe[i]] = 1
	symeigensystem(D' * (wv :* D), eigvecs, eigvals)
	idx = selectindex(eigvals :> 1e-10 * max(eigvals))
	Q = D * eigvecs[., idx]
	Z = X, Q
	p = cols(Z)

	A = cross(Z, wv, Z)
	Ainv = invsym(A)
	b_u = Ainv * cross(Z, wv, y)
	e_u = y - Z * b_u

	R_full = R_in, J(rows(R_in), p - k, 0)
	M_r = Ainv * R_full'
	lam = invsym(R_full * M_r) * (R_full * b_u - r_in)
	b_r = b_u - M_r * lam
	e_r = y - Z * b_r

	if (impose_null) {
		b_star = b_r
		e_star = e_r
	}
	else {
		b_star = b_u
		e_star = e_u
	}
	yhat_star = Z * b_star
	r_target = R_in * b_star[1::k]

	G = max(cl)
	nested_adj = (nested ? 1 : 0)
	df_a = (nested ? 0 : cols(idx))
	ssc = ((bigN - 1) / (bigN - nested_adj - k - df_a)) * (G / (G - 1))

	// Original CRV1 variance & stat
	meat = J(p, p, 0)
	for (g = 1; g <= G; g++) {
		idx = selectindex(cl :== g)
		real colvector sg
		sg = cross(Z[idx, .], wv[idx], e_u[idx])
		meat = meat + sg * sg'
	}
	V = ssc * (Ainv * meat * Ainv')[1::k, 1::k]
	if (rows(R_in) == 1) {
		pt0 = (R_in * b_u[1::k])[1, 1]
		se0 = sqrt((R_in * V * R_in')[1, 1])
		stat0 = (pt0 - r_in[1]) / se0
	}
	else {
		gap = R_in * b_u[1::k] - r_in
		stat0 = (gap' * invsym(R_in * V * R_in') * gap)[1, 1] / rows(R_in)
		pt0 = .
		se0 = .
	}

	// Full Rademacher enumeration: 2^G draws
	real scalar n_draws
	n_draws = 2^G
	tb_vec = J(n_draws, 1, 0)
	for (bdraw = 0; bdraw < n_draws; bdraw++) {
		real colvector vg
		vg = J(G, 1, 1)
		for (g = 1; g <= G; g++) {
			if (mod(floor(bdraw / 2^(g - 1)), 2) == 1) vg[g] = -1
		}
		yb = yhat_star + vg[cl] :* e_star
		b_b = Ainv * cross(Z, wv, yb)
		e_b = yb - Z * b_b
		meat = J(p, p, 0)
		for (g = 1; g <= G; g++) {
			idx = selectindex(cl :== g)
			real colvector sgb
			sgb = cross(Z[idx, .], wv[idx], e_b[idx])
			meat = meat + sgb * sgb'
		}
		V = ssc * (Ainv * meat * Ainv')[1::k, 1::k]
		if (rows(R_in) == 1) {
			se_b = sqrt((R_in * V * R_in')[1, 1])
			tb_vec[bdraw + 1] = ((R_in * b_b[1::k])[1, 1] - r_target[1]) / se_b
		}
		else {
			gap = R_in * b_b[1::k] - r_target
			tb_vec[bdraw + 1] = (gap' * invsym(R_in * V * R_in') * gap)[1, 1] / rows(R_in)
		}
	}

	eps = 1e-12 * (1 + abs(stat0))
	if (rows(R_in) == 1) {
		p_oracle = mean(abs(tb_vec) :> abs(stat0) + eps)
	}
	else {
		p_oracle = mean(tb_vec :> stat0 + eps)
	}

	assert(abs(st_numscalar("e(wb_stat)") - stat0) < 1e-9)
	assert(abs(st_numscalar("e(wb_p)") - p_oracle) < 1e-12)
	if (rows(R_in) == 1) {
		assert(abs(st_numscalar("e(wb_point)") - pt0) < 1e-9)
		assert(abs(st_numscalar("e(wb_se)") - se0) < 1e-9)
	}
	tb_fereg = st_matrix(savet_mat)
	assert(rows(tb_fereg) == n_draws)
	assert(max(abs(sort(tb_fereg, 1) - sort(tb_vec, 1))) < 1e-9)
}
end

* Verify Mata oracle on WCR11 (nested FE, linear comb) + full-enum trimming (reps(100) > 16 = 2^4)
matrix drop _all
matrix t_lin = .
fereg y x1 x2, absorb(fe8) vce(cluster c4) wildboot(2*x1 - x2 = 0.5, reps(100) savet(t_lin) seed(11))
assert e(wb_reps) == 16
assert e(wb_full_enum) == 1
assert e(wb_impose_null) == 1
assert rowsof(t_lin) == 16
mata: _test_wb_oracle("y", "x1 x2", "fe8", "c4", "", 1, (2, -1), (0.5), 1, "t_lin")

* Verify Mata oracle on WCU11 (non-nested FE, aw, joint q=2)
fereg y x1 x2 [aw=aw], absorb(fe_nn) vce(cluster c4) wildboot((x1 = 0.5) (x2 = -0.3), reps(50) noimposenull savet(t_wcu) seed(22))
assert e(wb_q) == 2
assert e(wb_reps) == 16
assert e(wb_impose_null) == 0
capture confirm matrix e(wb_ci)
assert _rc != 0
mata: _test_wb_oracle("y", "x1 x2", "fe_nn", "c4", "aw", 0, (1, 0 \ 0, 1), (0.5 \ -0.3), 0, "t_wcu")

* -----------------------------------------------------------------------------
* 2. OLS variants: MNW (13, 31, 33), Hetero (11, 21, 31, 13, 33), 2-way, bootcluster
* -----------------------------------------------------------------------------
foreach bt in 13 31 33 {
	fereg y x1 x2, absorb(fe_nn) vce(cluster c4) wildboot(x1 = 0.1, boottype(`bt') reps(19) seed(33))
	assert "`e(wb_boottype)'" == "`bt'"
	assert "`e(wb_method)'" == "wildcluster"
	assert !missing(e(wb_stat)) & !missing(e(wb_p)) & !missing(e(wb_ci_lo)) & !missing(e(wb_ci_hi))
	matrix list e(wb_ci)
}

foreach bt in 11 21 31 13 33 {
	fereg y x1 x2, absorb(fe8) vce(robust) wildboot(x1 = 0, boottype(`bt') reps(29) weights(mammen) seed(44))
	assert "`e(wb_method)'" == "hetero"
	assert "`e(wb_boottype)'" == "`bt'"
	assert "`e(wb_weights)'" == "mammen"
	assert !missing(e(wb_stat)) & !missing(e(wb_p))
}

* Two-way cluster, string bootcluster, interaction bootcluster, noci, level(90), pvaltypes
fereg y x1 x2, absorb(fe8 fe_nn) vce(cluster c4 c6) wildboot(x1, reps(25) weights(webb) level(90) seed(55))
assert e(wb_level) == 90
assert "`e(wb_weights)'" == "webb"
assert !missing(e(wb_stat)) & !missing(e(wb_p)) & !missing(e(wb_ci_lo))

fereg y x1 x2, absorb(fe8) vce(cluster c10) wildboot(x1 = 0, bootcluster(c4_str) noci pvaltype(equal) reps(16) seed(66))
assert "`e(wb_bootcluster)'" == "c4_str"
assert "`e(wb_pvaltype)'" == "equal-tailed"
assert e(wb_n_clust) == 4
capture confirm matrix e(wb_ci)
assert _rc != 0

fereg y x1 x2, absorb(fe8) vce(robust) wildboot(x1 = 0, bootcluster(c4#c6) reps(25) seed(67))
assert "`e(wb_bootcluster)'" == "c4#c6"
assert !missing(e(wb_n_clust))

fereg y x1 x2, absorb(fe8) vce(cluster c10) wildboot(x1, pvaltype(greater) reps(25) weights(normal) seed(68))
assert "`e(wb_pvaltype)'" == ">"
assert "`e(wb_weights)'" == "normal"

fereg y x1 x2, absorb(fe8) vce(cluster c10) wildboot(x1, pvaltype(less) reps(25) weights(normal) seed(68))
assert "`e(wb_pvaltype)'" == "<"

* noabsorb + testing _cons, factor variables + omitted level in rmat(), rmat()/rvec(), pw/fw, thread invariance
fereg y x1 x2, noabsorb vce(cluster c10) wildboot(_cons = 0.4, reps(31) seed(77))
assert !missing(e(wb_stat)) & !missing(e(wb_point)) & !missing(e(wb_se))

fereg y x1 i.fac3, absorb(fe8) vce(cluster c10) wildboot(2.fac3 - 3.fac3 = 0, reps(29) seed(78))
assert !missing(e(wb_stat))
* Check omitted column zero in rmat() matching e(b) full width
matrix R_full = (1, 0, 0, 0, 0 \ 0, 0, 1, 0, 0)
matrix r_full = (0 \ 0)
fereg y x1 i.fac3, absorb(fe8) vce(cluster c10) wildboot(, rmat(R_full) rvec(r_full) reps(29) seed(79))
assert e(wb_q) == 2

fereg y x1 x2 [pw=aw], absorb(fe8) vce(cluster c10) wildboot(x1 = 0, reps(29) seed(80) savet(t_th1)) threads(1)
local st1 = e(wb_stat)
local pv1 = e(wb_p)
local lo1 = e(wb_ci_lo)
fereg y x1 x2 [pw=aw], absorb(fe8) vce(cluster c10) wildboot(x1 = 0, reps(29) seed(80) savet(t_th4)) threads(4)
assert_close "threads stat" e(wb_stat) `st1' 1e-14
assert_close "threads p" e(wb_p) `pv1' 1e-14
assert_close "threads ci_lo" e(wb_ci_lo) `lo1' 1e-14
assert mreldif(t_th1, t_th4) < 1e-14

fereg y x1 x2 [fw=fw], absorb(fe8) vce(robust) wildboot(x1, boottype(31) reps(23) seed(81))
assert "`e(wb_method)'" == "hetero" & !missing(e(wb_p))

* -----------------------------------------------------------------------------
* 3. IV/2SLS (WRE and Score), with and without FE, clustered and unclustered
* -----------------------------------------------------------------------------
fereg y_iv x1 x2 (e1 = z1 z2), absorb(fe8) vce(cluster c10) wildboot(e1 = 0.8, reps(29) seed(90))
assert "`e(wb_method)'" == "wre"
assert "`e(wb_stattype)'" == "t"
assert !missing(e(wb_stat)) & !missing(e(wb_ci_lo))

fereg y_iv x1 (e1 e2 = z1 z2 z3), noabsorb vce(cluster c10) wildboot((e1 = 0) (e2 = 0), method(score) reps(29) seed(91))
assert "`e(wb_method)'" == "score"
assert "`e(wb_stattype)'" == "score F"
assert e(wb_q) == 2

fereg y_iv x1 (e1 = z1 z2), absorb(fe_nn) vce(robust) wildboot(e1 = 0.5, reps(25) seed(92))
assert "`e(wb_method)'" == "wre"
assert !missing(e(wb_stat))

* -----------------------------------------------------------------------------
* 4. feppml and felogit (score & hetero, exposure/offset, replay)
* -----------------------------------------------------------------------------
feppml y_pois x1 x2, absorb(fe8) vce(cluster c10) exposure(expv) wildboot(x1 = 0, reps(25) seed(101) savet(t_ppml))
assert "`e(wb_method)'" == "score"
assert !missing(e(wb_stat)) & !missing(e(wb_ci_lo))
local ppml_p = e(wb_p)
feppml
assert "`e(wb_method)'" == "score"
assert_close "replay feppml" e(wb_p) `ppml_p' 1e-15

feppml y_pois x1 x2, noabsorb vce(robust) offset(offv) wildboot((x1 = 0) (x2 = 0), noimposenull reps(25) seed(102))
assert "`e(wb_method)'" == "score"
assert e(wb_q) == 2

felogit y_logit x1 x2, absorb(fe8) vce(cluster c10) wildboot(x1 = 0, reps(25) seed(103))
assert "`e(wb_method)'" == "score"
assert !missing(e(wb_stat)) & !missing(e(wb_ci_lo))
local logit_p = e(wb_p)
felogit
assert_close "replay felogit" e(wb_p) `logit_p' 1e-15

* Replay fereg
qui fereg y x1 x2, absorb(fe8) vce(cluster c10) wildboot(x1 = 0, reps(25) seed(104))
local ols_p = e(wb_p)
fereg
assert "`e(wb_method)'" == "wildcluster"
assert_close "replay fereg" e(wb_p) `ols_p' 1e-15
assert !missing(r(level))

* -----------------------------------------------------------------------------
* 5. Error handling
* -----------------------------------------------------------------------------
xtset fe8 obs_id
capture fereg y x1 x2, absorb(fe8) vce(dkraay 2) wildboot(x1 = 0)
assert _rc == 198
capture fereg y x1 x2, absorb(fe8) vce(robust) bw(2) wildboot(x1 = 0)
assert _rc == 198
capture fereg y x1 x2, absorb(fe8) vce(cluster c10) wildboot(_cons = 0)
assert _rc == 198
capture fereg y x1 i.fac3, absorb(fe8) vce(cluster c10) wildboot(1b.fac3 = 0)
assert _rc == 198
capture fereg y x1 x2, absorb(fe8) vce(cluster c10) wildboot(x1 = 0, imposenull noimposenull)
assert _rc == 198
capture felogit y_logit x1 x2, absorb(fe8 fe_nn) ape wildboot(x1 = 0)
assert _rc == 198
capture feppml y_pois x1 x2, absorb(fe8) tagsep(tsep) wildboot(x1 = 0)
assert _rc == 198
matrix R_bad = (0, 1, 0, 0, 0)
capture fereg y x1 i.fac3, absorb(fe8) vce(cluster c10) wildboot(, rmat(R_bad) reps(19))
assert _rc == 198
matrix R_dim = (1, 0)
capture fereg y x1 i.fac3, absorb(fe8) vce(cluster c10) wildboot(, rmat(R_dim) reps(19))
assert _rc == 503

di as result "test_wildboot.do: ALL PASSED"
