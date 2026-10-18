* test_logitfe.do -- felogit's bias corrections and average partial effects vs logitfe 3.0.3 (live)
*
* felogit's analytical, jackknife and ape options follow logitfe (Cruz-Gonzalez, Fernandez-Val and
* Weidner 2017) where logitfe is correct, and the correct formula where it is not (felogit.sthlp,
* "Differences from logitfe"). Compared, on panels generated here (small enough that logitfe's
* delete-one jackknives take seconds):
*   - against logitfe: e(b) (relative 1e-6; jackknife combinations 1e-4, logitfe's Newton-Raphson
*     stops at Stata's default tolerances and the combinations amplify it), e(b2) (APE points, same
*     tolerances), e(V) (uncorrected, 1e-6), e(N), e(N_group), e(T_min), e(T_max);
*   - the documented logitfe bugs against the correct formula:
*     B1 lags >= 2: an independent Mata computation of logitfe's analytical correction with
*        psi(i,t-j) (rf_ref_an, rf_ref_ape_bc; both reproduce logitfe for lags 0 and 1 first);
*     B2 APE variance: an independent Mata computation of Fernandez-Val and Weidner's variance as in
*        alpaca (rf_ref_ape: delta method plus the sampling term over all N* observations, logitfe's
*        default adj = 1, population(), weak exogeneity with lags());
*     B3 jackknife dimensions (js, sj, and the single-bias jj and double): combinations of logitfe's
*        own correctly-dimensioned runs (e.g. js = jj tbias(no) + ss1 tbias(no) - noc);
*     B4 ss2 tbias(no) (r(110) in logitfe): on a balanced panel the per-individual time halves are
*        the global time halves, so it equals logitfe's ss1 tbias(no);
*   - multiple(): _rf_bc_perm's draws equal logitfe's own multiplepvar()/multipletvar() exactly, and
*     the estimates equal logitfe's;
*   - the fixed effects saved by d() at the corrected point satisfy the likelihood equations of
*     every absorbed group (FE re-solve; also with an offset); felogit leaves the RNG state
*     unchanged; results are bit-identical for 1 and 4 threads and do not depend on the order of
*     the absorb() terms; vce() changes e(V) only;
*   - the network structure against alpaca 0.3.5 (gen_bc_network.R: bc_network.csv and
*     bc_network_ref.csv): uncorrected APEs and e(V2), analytical L = 0 and L = 2.
*
* Run from tests/stata (needs fereg.plugin, logitfe 3.0.3 and moremata on the adopath; run_all.do
* checks the versions):
*   stata-mp -b do test_logitfe.do
version 16
clear all
set more off
capture confirm file "../../stata/felogit.ado"
if (!_rc) adopath ++ "`c(pwd)'/../../stata"
which felogit
findfile fereg.plugin
which logitfe
capture mata: mata which mm_sample()
if (_rc) {
	quietly mata: mata mlib index
	capture mata: mata which mm_sample()
	if (_rc) {
		di as err "test_logitfe needs moremata (mm_sample()) on the adopath"
		exit 9
	}
}
* logitfe.ado runs -mata: mata clear- when it is loaded: load it here, before any other Mata code
* (as a do-file, so that its Mata functions multiplepvar() and multipletvar() are callable below)
findfile logitfe.ado
quietly run "`r(fn)'"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
* lfe <tag> <varlist> , <logitfe options>: run logitfe quietly and keep its results in matrices
* L_<tag>_b, L_<tag>_b2, L_<tag>_V and scalars L_<tag>_N, L_<tag>_Ng, L_<tag>_Tmin, L_<tag>_Tmax
capture program drop lfe
program lfe
	gettoken tag 0 : 0
	quietly logitfe `0'
	matrix L_`tag'_b = e(b)
	matrix L_`tag'_b2 = e(b2)
	matrix L_`tag'_V = e(V)
	scalar L_`tag'_N = e(N)
	scalar L_`tag'_Ng = e(N_group)
	scalar L_`tag'_Tmin = e(T_min)
	scalar L_`tag'_Tmax = e(T_max)
	di as text "logitfe `tag': " `"`0'"'
end

* lcomb <tag> = <a1> <tag1> [<a2> <tag2> ...]: L_<tag>_b/b2 = sum of a_k * L_<tag_k>_b/b2 (the
* correctly-dimensioned jackknife combinations of logitfe's own runs)
capture program drop lcomb
program lcomb
	gettoken tag 0 : 0
	gettoken eq 0 : 0
	local first 1
	while ("`0'" != "") {
		gettoken a 0 : 0
		gettoken t 0 : 0
		if (`first') {
			matrix L_`tag'_b = `a' * L_`t'_b
			matrix L_`tag'_b2 = `a' * L_`t'_b2
		}
		else {
			matrix L_`tag'_b = L_`tag'_b + `a' * L_`t'_b
			matrix L_`tag'_b2 = L_`tag'_b2 + `a' * L_`t'_b2
		}
		local first 0
	}
end

* fcmp <reference tag> <tol b> <tol b2> <checkV 0|1> : <felogit command>
* Runs felogit (displayed) and compares e(b) and e(b2) (by name, relative), e(V) with the
* reference's (when checkV), e(N), e(N_group), e(T_min), e(T_max) (when the reference has them).
capture program drop fcmp
program fcmp
	gettoken tag 0 : 0
	gettoken tolb 0 : 0
	gettoken tola 0 : 0
	gettoken chkV 0 : 0
	gettoken colon 0 : 0
	if ("`colon'" != ":") error 198
	`0'
	mata: rf_cmp("e(b)", "L_`tag'_b", `tolb', "b `tag'")
	mata: rf_cmp("e(b2)", "L_`tag'_b2", `tola', "b2 `tag'")
	if (`chkV') mata: rf_cmp("e(V)", "L_`tag'_V", 1e-6, "V `tag'")
	capture confirm scalar L_`tag'_N
	if (!_rc) {
		rf_assert_close "e(N) `tag'" e(N) L_`tag'_N 0
		rf_assert_close "e(N_group) `tag'" e(N_group) L_`tag'_Ng 0
		rf_assert_close "e(T_min) `tag'" e(T_min) L_`tag'_Tmin 0
		rf_assert_close "e(T_max) `tag'" e(T_max) L_`tag'_Tmax 0
	}
	di as result "ok  `tag': " `"`0'"'
end

* fecheck: the fixed effects saved in e(d) solve the likelihood equations at the posted (corrected)
* coefficients: sum(y - Pr) = 0 within every group of each listed variable
capture program drop fecheck
program fecheck, sortpreserve
	args groups
	tempvar p r s
	quietly predict double `p' if e(sample), pr
	quietly gen double `r' = (`e(depvar)' != 0) - `p' if e(sample)
	foreach g of local groups {
		quietly bysort `g': egen double `s' = total(`r') if e(sample)
		quietly summarize `s', meanonly
		if (max(abs(r(min)), abs(r(max))) > 1e-8) {
			di as err "FAIL fixed effects at the corrected point: max |sum(y - p)| over `g' = " max(abs(r(min)), abs(r(max)))
			exit 9
		}
		drop `s'
	}
end

* refit <tag> <depvar> <xvars> <dummies> <touse>: the exact uncorrected fit by logit with explicit
* dummies on felogit's sample; R_<tag>_b (regressors) and variable eta_<tag> (index incl. dummies)
capture program drop refit
program refit
	args tag y xvars dummies touse
	quietly logit `y' `xvars' `dummies' if `touse', tol(1e-12) ltol(1e-14) nrtol(1e-12) iterate(200)
	tempname b
	matrix `b' = J(1, `: word count `xvars'', .)
	local j 0
	foreach x of local xvars {
		matrix `b'[1, `++j'] = _b[`x']
	}
	matrix R_`tag'_b = `b'
	capture drop eta_`tag'
	quietly predict double eta_`tag' if `touse', xb
end

mata:
// max relative difference over the coefficients of reference r, matched by name in a: vectors
// |a - r| / |r| (absolute where r = 0); square matrices on the correlation scale
// |a - r| / sqrt(r_ii r_kk); fails if above tol
void rf_cmp(string scalar a, string scalar r, real scalar tol, string scalar what)
{
	real matrix A, R
	string matrix sa, sr
	real scalar i, j, k, l, m, d, sc
	A = st_matrix(a)
	R = st_matrix(r)
	sa = st_matrixcolstripe(a)
	sr = st_matrixcolstripe(r)
	m = 0
	for (i = 1; i <= cols(R); i++) {
		j = rf_pos(sa, sr[i, 2])
		if (rows(R) == 1) {
			d = abs(A[1, j] - R[1, i]) / (R[1, i] == 0 ? 1 : abs(R[1, i]))
			if (d > m | missing(d)) m = d
			continue
		}
		for (k = 1; k <= cols(R); k++) {
			l = rf_pos(sa, sr[k, 2])
			sc = sqrt(abs(R[i, i] * R[k, k]))
			d = abs(A[j, l] - R[i, k]) / (sc == 0 ? 1 : sc)
			if (d > m | missing(d)) m = d
		}
	}
	printf("{txt}  %s: max rel. diff %9.2e (tol %g)\n", what, m, tol)
	if (m > tol | missing(m)) {
		printf("{err}FAIL %s: max rel. diff %g > %g\n", what, m, tol)
		exit(9)
	}
}

real scalar rf_pos(string matrix stripe, string scalar name)
{
	real scalar i
	for (i = 1; i <= rows(stripe); i++) {
		if (stripe[i, 2] == name) return(i)
	}
	_error(3498, "coefficient " + name + " not found")
}

// dummies of codes g (1..G)
real matrix rf_dum(real colvector g)
{
	real matrix D
	real scalar i
	D = J(rows(g), max(g), 0)
	for (i = 1; i <= rows(g); i++) D[i, g[i]] = 1
	return(D)
}

// columns of A residualized on the columns of D, weights w
real matrix rf_within(real matrix A, real matrix D, real colvector w)
{
	return(A - D * invsym(cross(D, w, D)) * cross(D, w, A))
}

// Model dummies: individual (fe) and/or period (te) codes
real matrix rf_fedum(real colvector id, real colvector per, real scalar fe, real scalar te)
{
	real matrix D
	D = J(rows(id), 0, .)
	if (fe) D = D, rf_dum(id)
	if (te) D = D, rf_dum(per :+ 1)
	return(D)
}

// Row of individual i at calendar period t, or 0 (rows of individual i in r)
real scalar rf_find(real colvector per, real colvector r, real scalar t)
{
	real scalar s
	for (s = 1; s <= rows(r); s++) {
		if (per[r[s]] == t) return(r[s])
	}
	return(0)
}

// logitfe's analytical correction of the coefficients (logitfe spec 4a; ado lines 3392-3446) with
// psi(i,t-j) for lag j (B1 fixed): individual bias B (casewise mean over individuals with lagged
// observations for every j <= L), time bias D (mean over periods), W = sum(H x~ x~')/n,
// bc = b - (ib B + tb D/ng) W^-1
real rowvector rf_ref_an(real matrix X, real colvector y, real colvector eta, real colvector id,
	real colvector per, real scalar fe, real scalar te, real scalar ib, real scalar tb,
	real scalar L, real rowvector b)
{
	real colvector lam, H, psi, r, ids, pers
	real matrix Xt, Bs
	real rowvector Bi, acc, Bbar, Dbar
	real scalar n, k, i, j, t, s, sH, TN, ok, ng
	n = rows(y)
	k = cols(X)
	lam = invlogit(eta)
	H = lam :* (1 :- lam)
	psi = y - lam
	Xt = rf_within(X, rf_fedum(id, per, fe, te), H)
	ids = uniqrows(id)
	ng = rows(ids)
	Bbar = J(1, k, 0)
	if (fe & ib) {
		Bs = J(0, k, .)
		for (i = 1; i <= ng; i++) {
			r = selectindex(id :== ids[i])
			sH = sum(H[r])
			Bi = colsum(lam[r] :* H[r] :* Xt[r, .]) / sH / rows(r)
			ok = 1
			for (j = 1; j <= L; j++) {
				acc = J(1, k, 0)
				TN = 0
				for (t = 1; t <= rows(r); t++) {
					s = rf_find(per, r, per[r[t]] - j)
					if (s) {
						acc = acc + psi[s] * H[r[t]] * Xt[r[t], .]
						TN++
					}
				}
				if (TN == 0) ok = 0
				else Bi = Bi - acc / sH / TN
			}
			if (ok) Bs = Bs \ Bi
		}
		Bbar = mean(Bs)
	}
	Dbar = J(1, k, 0)
	if (te & tb) {
		pers = uniqrows(per)
		for (t = 1; t <= rows(pers); t++) {
			r = selectindex(per :== pers[t])
			Dbar = Dbar + colsum(lam[r] :* H[r] :* Xt[r, .]) / sum(H[r])
		}
		Dbar = Dbar / rows(pers)
	}
	return(b - (Bbar + Dbar / ng) * invsym(cross(Xt, H, Xt) / n))
}

// logitfe's corrected APEs (logitfe spec 4c; ado lines 3490-3587) with psi(i,t-j) (B1 fixed), at the
// corrected point: bc (coefficients), eta (index with the fixed effects re-estimated at bc).
// binary: 1 = binary regressor; Nstar = observations before the constant-outcome drop.
real rowvector rf_ref_ape_bc(real matrix X, real colvector y, real colvector eta, real colvector id,
	real colvector per, real scalar fe, real scalar te, real scalar ib, real scalar tb,
	real scalar L, real rowvector bc, real rowvector binary, real scalar Nstar)
{
	real colvector lam, H, lp, lpp, psi, dD, d2D, z1, z0, l1, l0
	real rowvector ape
	real scalar k, j, ng, bias
	k = cols(X)
	lam = invlogit(eta)
	H = lam :* (1 :- lam)
	lp = H :* (1 :- 2 :* lam)
	lpp = lp :* (1 :- 2 :* lam) - 2 :* H :^ 2
	psi = y - lam
	ng = rows(uniqrows(id))
	ape = J(1, k, .)
	for (j = 1; j <= k; j++) {
		if (binary[j]) {
			z1 = eta + bc[j] :* (1 :- X[., j])
			z0 = eta - bc[j] :* X[., j]
			l1 = invlogit(z1)
			l0 = invlogit(z0)
			dD = l1 :* (1 :- l1) - l0 :* (1 :- l0)
			d2D = l1 :* (1 :- l1) :* (1 :- 2 :* l1) - l0 :* (1 :- l0) :* (1 :- 2 :* l0)
			bias = rf_ape_bias(dD, d2D, H, lp, psi, id, per, fe, te, ib, tb, L, ng)
			ape[j] = sum(l1 - l0) / Nstar - bias
		}
		else {
			bias = rf_ape_bias(lp, lpp, H, lp, psi, id, per, fe, te, ib, tb, L, ng)
			ape[j] = bc[j] * (sum(H) / Nstar - bias)
		}
	}
	return(ape)
}

real scalar rf_ape_bias(real colvector dD, real colvector d2D, real colvector H, real colvector lp,
	real colvector psi, real colvector id, real colvector per, real scalar fe, real scalar te,
	real scalar ib, real scalar tb, real scalar L, real scalar ng)
{
	real colvector Psi, rr, num, r, ids, pers, Bs
	real scalar i, j, t, s, sH, Bi, acc, TN, ok, Bb, Db
	Psi = dD :/ H
	Psi = Psi - rf_within(Psi, rf_fedum(id, per, fe, te), H)
	rr = dD :/ H - Psi
	num = -lp :* Psi + d2D
	Bb = 0
	if (fe & ib) {
		ids = uniqrows(id)
		Bs = J(0, 1, .)
		for (i = 1; i <= rows(ids); i++) {
			r = selectindex(id :== ids[i])
			sH = sum(H[r])
			Bi = sum(num[r]) / sH / rows(r)
			ok = 1
			for (j = 1; j <= L; j++) {
				acc = 0
				TN = 0
				for (t = 1; t <= rows(r); t++) {
					s = rf_find(per, r, per[r[t]] - j)
					if (s) {
						acc = acc + 2 * psi[s] * H[r[t]] * rr[r[t]]
						TN++
					}
				}
				if (TN == 0) ok = 0
				else Bi = Bi + acc / sH / TN
			}
			if (ok) Bs = Bs \ Bi
		}
		Bb = 0.5 * mean(Bs)
	}
	Db = 0
	if (te & tb) {
		pers = uniqrows(per)
		for (t = 1; t <= rows(pers); t++) {
			r = selectindex(per :== pers[t])
			Db = Db + sum(num[r]) / sum(H[r])
		}
		Db = 0.5 * Db / rows(pers)
	}
	return(Bb + Db / ng)
}

// APE points and variance of Fernandez-Val and Weidner (2016) as alpaca's getAPEs computes them
// (alpaca spec 5.3-5.5), with felogit's rules: evaluated at the uncorrected fit (b, eta); divisor
// N*; sampling-of-effects term over all N* observations (dropped ones: zero partial effect, own
// fixed-effect groups) times adj; weak exogeneity (L > 0): + adj (C + C'), C over individuals in
// time order. id/per of the estimation rows; idd/perd of the dropped rows.
void rf_ref_ape(real matrix X, real colvector y, real colvector eta, real colvector id,
	real colvector per, real colvector idd, real colvector perd, real scalar fe, real scalar te,
	real rowvector b, real rowvector binary, real scalar adj, real scalar L, string scalar dname,
	string scalar vname)
{
	real colvector mu, w, v, m2, e0, e1, m0, m1, gid, gper, r, o
	real matrix D, MX, PX, H, Dl, Dl1, Jm, Psi, PPsi, Gam, V, Dr, Dall, C, cum
	real rowvector delta
	real scalar n, p, j, Ns, g, s, G1
	n = rows(y)
	p = cols(X)
	Ns = n + rows(idd)
	mu = invlogit(eta)
	w = mu :* (1 :- mu)
	v = y - mu
	m2 = w :* (1 :- 2 :* mu)
	D = rf_fedum(id, per, fe, te)
	MX = rf_within(X, D, w)
	PX = X - MX
	H = cross(MX, w, MX)
	Dl = J(n, p, .)
	Dl1 = J(n, p, .)
	Jm = J(p, p, 0)
	for (j = 1; j <= p; j++) {
		if (binary[j]) {
			e0 = eta - X[., j] :* b[j]
			e1 = e0 :+ b[j]
			m1 = invlogit(e1) :* (1 :- invlogit(e1))
			m0 = invlogit(e0) :* (1 :- invlogit(e0))
			Dl[., j] = invlogit(e1) - invlogit(e0)
			Dl1[., j] = m1 - m0
			Jm[., j] = cross(MX, Dl1[., j]) / Ns
			Jm[j, j] = (sum(m1) - cross(PX[., j], Dl1[., j])) / Ns
		}
		else {
			Dl[., j] = b[j] :* w
			Dl1[., j] = b[j] :* m2
			Jm[., j] = cross(MX, Dl1[., j]) / Ns
			Jm[j, j] = Jm[j, j] + sum(w) / Ns
		}
	}
	delta = colsum(Dl) / Ns
	Psi = -Dl1 :/ w
	PPsi = Psi - rf_within(Psi, D, w)
	Gam = (MX * (invsym(H / Ns) * Jm) - PPsi) :* v / Ns
	V = cross(Gam, Gam)
	if (adj > 0) {
		Dr = (Dl :- delta) / Ns
		Dall = Dr \ J(rows(idd), 1, -delta / Ns)
		gid = id \ idd
		gper = per \ perd
		V = V + adj * rf_gsum(Dall, fe ? gid : gper)
		if (fe & te) V = V + adj * (rf_gsum(Dall, gper) - cross(Dall, Dall))
		if (L > 0 & fe) {
			C = J(p, p, 0)
			G1 = max(id)
			for (g = 1; g <= G1; g++) {
				r = selectindex(id :== g)
				if (!rows(r)) continue
				o = order(per[r], 1)
				r = r[o]
				cum = J(1, p, 0)
				for (s = 1; s <= rows(r); s++) {
					C = C + cum' * Gam[r[s], .]
					cum = cum + Dr[r[s], .]
				}
			}
			V = V + adj * (C + C')
		}
	}
	st_matrix(dname, delta)
	st_matrix(vname, V)
}

// sum over groups of (sum of the rows of D in the group)'(sum of the rows of D in the group)
real matrix rf_gsum(real matrix D, real colvector g)
{
	real matrix S
	real colvector u, r
	real scalar i
	u = uniqrows(g)
	S = J(cols(D), cols(D), 0)
	for (i = 1; i <= rows(u); i++) {
		r = selectindex(g :== u[i])
		S = S + colsum(D[r, .])' * colsum(D[r, .])
	}
	return(S)
}

// binary regressor flags (exactly {0, 1} with both present on the sample), as logitfe
real rowvector rf_binary(real matrix X)
{
	real rowvector f
	real scalar j
	f = J(1, cols(X), 0)
	for (j = 1; j <= cols(X); j++) {
		f[j] = all(X[., j] :== 0 :| X[., j] :== 1) & any(X[., j] :== 0) & any(X[., j] :== 1)
	}
	return(f)
}
end

* refs <tag> "<xvars>" <y> <touse> <nstar> <fe 0|1> <te 0|1> <ib> <tb> <L> [adj]
* Mata references at felogit's sample: R_<tag>_bc (analytical, B1-fixed), R_<tag>_apebc (corrected
* APEs at the FE re-solve of bc), R_<tag>_ape and R_<tag>_V2 (APEs and their variance at the
* uncorrected fit, sampling term times adj, default 1). Needs variables idc (individual codes),
* per (calendar period index). e() is left as it was.
capture program drop refs
program refs
	args tag xvars y touse nstar fe te ib tb L adj
	if ("`adj'" == "") local adj 1
	tempname hold
	_estimates hold `hold', restore
	local dummies
	if (`fe') local dummies `dummies' i.idc
	if (`te') local dummies `dummies' i.per
	local xs
	foreach x of local xvars {
		tsrevar `x'
		local xs `xs' `r(varlist)'
	}
	refit `tag' `y' "`xs'" "`dummies'" `touse'
	tempvar dropped
	quietly gen byte `dropped' = `nstar' & !`touse'
	mata: _rf_X = st_data(., "`xs'", "`touse'")
	mata: _rf_y = st_data(., "`y'", "`touse'") :!= 0
	mata: _rf_e = st_data(., "eta_`tag'", "`touse'")
	mata: _rf_id = st_data(., "idc", "`touse'")
	mata: _rf_pr = st_data(., "per", "`touse'")
	mata: _rf_b = st_matrix("R_`tag'_b")
	mata: _rf_bin = rf_binary(_rf_X)
	quietly count if `nstar'
	local Ns = r(N)
	mata: st_matrix("R_`tag'_bc", rf_ref_an(_rf_X, _rf_y, _rf_e, _rf_id, _rf_pr, `fe', `te', `ib', `tb', `L', _rf_b))
	mata: rf_ref_ape(_rf_X, _rf_y, _rf_e, _rf_id, _rf_pr, st_data(., "idc", "`dropped'"), st_data(., "per", "`dropped'"), ///
		`fe', `te', _rf_b, _rf_bin, `adj', `L', "R_`tag'_ape", "R_`tag'_V2")
	* fixed effects re-estimated at bc (offset), then logitfe's corrected APEs there
	tempvar off etac
	tempname bc
	matrix `bc' = R_`tag'_bc
	quietly gen double `off' = 0 if `touse'
	local j 0
	foreach x of local xs {
		quietly replace `off' = `off' + `bc'[1, `++j'] * `x' if `touse'
	}
	quietly logit `y' `dummies' if `touse', offset(`off') tol(1e-12) ltol(1e-14) nrtol(1e-12) iterate(200)
	quietly predict double `etac' if `touse', xb
	mata: st_matrix("R_`tag'_apebc", rf_ref_ape_bc(_rf_X, _rf_y, st_data(., "`etac'", "`touse'"), _rf_id, _rf_pr, ///
		`fe', `te', `ib', `tb', `L', st_matrix("R_`tag'_bc"), _rf_bin, `Ns'))
	foreach m in b bc ape apebc {
		matrix colnames R_`tag'_`m' = `xvars'
	}
	matrix colnames R_`tag'_V2 = `xvars'
	matrix rownames R_`tag'_V2 = `xvars'
	mata: mata drop _rf_X _rf_y _rf_e _rf_id _rf_pr _rf_b _rf_bin
end

* ---------------------------------------------------------------------------
* Data. D1: balanced panel, 60 individuals x 8 years (individuals with a constant outcome are
* dropped, no period is); x1 continuous, x2 binary. D2: D1 with 12% of the observations removed
* (unbalanced, gaps). D3: 15 x 15 directed pairs (tsset id jd), for -double-.
* ---------------------------------------------------------------------------
capture program drop mkdata
program mkdata
	args which
	clear
	set seed 20261018
	if ("`which'" == "D3") {
		set obs 225
		gen int id = ceil(_n / 15)
		gen int jd = _n - 15 * (id - 1)
		drop if id == jd
		gen double a = rnormal()
		bysort id (jd): replace a = a[1]
		gen double g = rnormal()
		bysort jd (id): replace g = g[1]
		gen double x1 = rnormal() + 0.5 * a
		gen byte x2 = runiform() < invlogit(g)
		gen byte y = runiform() < invlogit(0.8 * x1 - 0.7 * x2 + a + g)
		sort id jd
		tsset id jd
		gen int idc = id
		gen int per = jd - 1
		exit
	}
	set obs 480
	gen int id = 100 + 3 * ceil(_n / 8)
	bysort id: gen int year = 1990 + _n
	gen double a = rnormal() if year == 1991
	bysort id (year): replace a = a[1]
	gen double g = 0.4 * sin(year)
	gen double x1 = rnormal() + 0.5 * a
	gen byte x2 = runiform() < 0.4
	gen byte y = 0
	bysort id (year): replace y = runiform() < invlogit(0.8 * x1 - 0.7 * x2 + 1.2 * cond(_n > 1, y[_n - 1], 0) + a + g)
	if ("`which'" == "D2") {
		drop if runiform() < 0.12
	}
	xtset id year
	egen int idc = group(id)
	gen int per = year - 1991
end

// ===========================================================================
// D1 static
// ===========================================================================
mkdata D1
local S y x1 x2
gen byte ns = !missing(y, x1, x2)
* uncorrected (ape) and analytical, two-way and one-way
lfe noc `S', noc
lfe noc_i `S', noc teffects(no)
lfe noc_t `S', noc ieffects(no)
lfe an `S', an
lfe an_ib `S', an ibias(no)
lfe an_tb `S', an tbias(no)
lfe an_i `S', an teffects(no)
lfe an_t `S', an ieffects(no)
lfe an_pop `S', an pop(2000)
* jackknife (not ss2 tbias(no): r(110) in logitfe, B4)
foreach v in ss1 ss2 jj {
	lfe jk_`v' `S', jack `v'
	lfe jk_`v'_ib `S', jack `v' ibias(no)
	if ("`v'" != "ss2") lfe jk_`v'_tb `S', jack `v' tbias(no)
}
lfe jk_ss1_i `S', jack ss1 teffects(no)
lfe jk_ss2_t `S', jack ss2 ieffects(no)
* delete-one period with individual effects only (logitfe's jj teffects(no) deletes individuals,
* B3): T b - (T - 1) mean_t b(-t) from logitfe's uncorrected fits without period t
matrix L_dp_i_b = 0 * L_noc_i_b
matrix L_dp_i_b2 = 0 * L_noc_i_b2
forval t = 1991/1998 {
	quietly logitfe `S' if year != `t', noc teffects(no)
	matrix L_dp_i_b = L_dp_i_b + e(b) / 8
	matrix L_dp_i_b2 = L_dp_i_b2 + e(b2) / 8
}
matrix L_dp_i_b = 8 * L_noc_i_b - 7 * L_dp_i_b
matrix L_dp_i_b2 = 8 * L_noc_i_b2 - 7 * L_dp_i_b2

fcmp noc 1e-6 1e-6 1 : felogit `S', absorb(id year) ape tol(1e-12)
quietly gen byte s1 = e(sample)
refs s1 "x1 x2" y s1 ns 1 1 1 1 0
mata: rf_cmp("e(b2)", "R_s1_ape", 1e-8, "b2 vs Mata reference (noc)")
mata: rf_cmp("L_noc_b2", "R_s1_ape", 1e-6, "logitfe b2 vs Mata reference")
mata: rf_cmp("e(V2)", "R_s1_V2", 1e-7, "V2 vs FW/alpaca formula (B2)")
assert e(fpc) == 1 & "`e(bias)'" == "none" & e(N_full) == _N
* the Mata references reproduce logitfe's analytical correction (lags 0)
mata: rf_cmp("L_an_b", "R_s1_bc", 1e-6, "Mata reference vs logitfe an (b)")
mata: rf_cmp("L_an_b2", "R_s1_apebc", 1e-6, "Mata reference vs logitfe an (b2)")

fcmp noc_i 1e-6 1e-6 1 : felogit `S', absorb(id) ape tol(1e-12)
quietly gen byte s1i = e(sample)
refs s1io "x1 x2" y s1i ns 1 0 1 0 0
mata: rf_cmp("e(V2)", "R_s1io_V2", 1e-7, "V2 individual effects only")
fcmp noc_t 1e-6 1e-6 1 : felogit `S', absorb(year) ape tol(1e-12)
quietly gen byte s1t = e(sample)
refs s1to "x1 x2" y s1t ns 0 1 0 1 0
mata: rf_cmp("e(V2)", "R_s1to_V2", 1e-7, "V2 time effects only")
fcmp an 1e-6 1e-6 1 : felogit `S', absorb(id year) analytical tol(1e-12) d(dd)
fecheck "id year"
mata: rf_cmp("e(V2)", "R_s1_V2", 1e-7, "V2 (analytical: at the uncorrected fit)")
assert "`e(bias)'" == "analytical" & "`e(lags)'" == "0" & "`e(id)'" == "id" & "`e(time)'" == "year"
drop dd
fcmp an 1e-6 1e-6 1 : felogit `S', absorb(year id) analytical tol(1e-12)
fcmp an_ib 1e-6 1e-6 1 : felogit `S', absorb(id year) analytical ibias(no) tol(1e-12)
fcmp an_tb 1e-6 1e-6 1 : felogit `S', absorb(id year) analytical tbias(no) tol(1e-12) d(dd)
fecheck "id year"
drop dd
fcmp an_i 1e-6 1e-6 1 : felogit `S', absorb(id) analytical tol(1e-12) d(dd)
fecheck "id"
drop dd
fcmp an_t 1e-6 1e-6 1 : felogit `S', absorb(year) analytical tol(1e-12)
* population(): logitfe's points; adj = (M - N*)/(M - 1) (B6: logitfe uses the post-drop N);
* variance against the formula; population(N*) leaves the delta method alone
fcmp an_pop 1e-6 1e-6 1 : felogit `S', absorb(id year) analytical population(2000) tol(1e-12)
assert reldif(e(fpc), (2000 - e(N_full)) / 1999) < 1e-15
refs s1p "x1 x2" y s1 ns 1 1 1 1 0 `=(2000 - _N) / 1999'
mata: rf_cmp("e(V2)", "R_s1p_V2", 1e-7, "V2 population(2000)")
felogit `S', absorb(id year) ape population(`=_N') tol(1e-12)
assert e(fpc) == 0
refs s10 "x1 x2" y s1 ns 1 1 1 1 0 0
mata: rf_cmp("e(V2)", "R_s10_V2", 1e-7, "V2 delta method only (adj = 0)")

* jackknife: logitfe's correctly-dimensioned variants (1e-4)
fcmp jk_ss1 1e-4 1e-4 1 : felogit `S', absorb(id year) jackknife ss1 tol(1e-12) d(dd)
fecheck "id year"
drop dd
fcmp jk_ss1_ib 1e-4 1e-4 0 : felogit `S', absorb(id year) jackknife ss1 ibias(no) tol(1e-12)
fcmp jk_ss1_tb 1e-4 1e-4 0 : felogit `S', absorb(id year) jackknife ss1 tbias(no) tol(1e-12)
fcmp jk_ss2 1e-4 1e-4 1 : felogit `S', absorb(id year) jackknife tol(1e-12)
fcmp jk_ss2_ib 1e-4 1e-4 0 : felogit `S', absorb(id year) jackknife ss2 ibias(no) tol(1e-12)
fcmp jk_jj 1e-4 1e-4 0 : felogit `S', absorb(id year) jackknife jj tol(1e-12)
fcmp jk_ss1_i 1e-4 1e-4 0 : felogit `S', absorb(id) jackknife ss1 tol(1e-12)
fcmp jk_ss2_t 1e-4 1e-4 0 : felogit `S', absorb(year) jackknife ss2 tol(1e-12)
* B4: ss2 tbias(no) = 2b - mean(per-individual time halves) = logitfe's ss1 tbias(no) on a balanced panel
fcmp jk_ss1_tb 1e-4 1e-4 0 : felogit `S', absorb(id year) jackknife ss2 tbias(no) tol(1e-12)
* B3: delete-one individual = logitfe jj tbias(no); delete-one period = logitfe jj ibias(no);
* individual halves = ss1 ibias(no); time halves = ss1 tbias(no)
lcomb js_ok = 1 jk_jj_tb 1 jk_ss1_tb -1 noc
lcomb sj_ok = 1 jk_jj_ib 1 jk_ss1_ib -1 noc
fcmp js_ok 1e-4 1e-4 0 : felogit `S', absorb(id year) jackknife js tol(1e-12)
fcmp sj_ok 1e-4 1e-4 0 : felogit `S', absorb(id year) jackknife sj tol(1e-12)
fcmp jk_ss1_tb 1e-4 1e-4 0 : felogit `S', absorb(id year) jackknife js tbias(no) tol(1e-12)
fcmp jk_jj_tb 1e-4 1e-4 0 : felogit `S', absorb(id year) jackknife js ibias(no) tol(1e-12)
fcmp jk_jj_ib 1e-4 1e-4 0 : felogit `S', absorb(id year) jackknife sj tbias(no) tol(1e-12)
fcmp jk_ss1_ib 1e-4 1e-4 0 : felogit `S', absorb(id year) jackknife sj ibias(no) tol(1e-12)
fcmp jk_jj_ib 1e-4 1e-4 0 : felogit `S', absorb(id year) jackknife jj tbias(no) tol(1e-12)
fcmp jk_jj_tb 1e-4 1e-4 0 : felogit `S', absorb(id year) jackknife jj ibias(no) tol(1e-12)
* one fixed effect, individual effects only: jj = delete-one period
fcmp dp_i 1e-4 1e-4 0 : felogit `S', absorb(id) jackknife jj tol(1e-12)

* multiple(): logitfe's partitions (set seed 13579, mm_sample) and estimates
set seed 4242
local rs = c(rngstate)
local k 0
foreach m in "ss1" "ss2" "ss2 i" "ss2 t" "ss1 tbias(no) t" "ss1 ibias(no) i" "ss2 tbias(no)" "ss2 ibias(no) time individuals" {
	local ++k
	lfe mul`k' `S', jack `m' mul(3)
	set seed 4242
	fcmp mul`k' 1e-4 1e-4 0 : felogit `S', absorb(id year) jackknife `m' multiple(3) tol(1e-12)
	assert c(rngstate) == "`rs'"
}
* the draws themselves: _rf_bc_perm vs logitfe's multiplepvar()/multipletvar()
capture program drop lfe_draws
program lfe_draws
	version 11.2, missing
	set seed 13579
	forval r = 1/2 {
		sort id
		mata: multiplepvar("id")
		sort year
		mata: multipletvar("year")
		rename pvar2 lp`r'
		rename tvar2 lt`r'
	}
end
gen long obs = _n
lfe_draws
sort obs
_rf_bc_perm, ivar(id) tvar(year) multiple(2) perm(it) names(fp1 ft1 fp2 ft2)
forval r = 1/2 {
	assert fp`r' == lp`r' & ft`r' == lt`r'
}
drop obs lp* lt* fp* ft*
sort id year
di as result "D1 static passed"

// ===========================================================================
// D1 dynamic: L.y (binary) and lags()
// ===========================================================================
local D y L.y x1 x2
replace ns = !missing(y, L.y, x1, x2)
lfe d_noc `D', noc
lfe d_an0 `D', an
lfe d_an1 `D', an lags(1)
lfe d_an1_tb `D', an lags(1) tbias(no)
lfe d_an1_i `D', an lags(1) teffects(no)
lfe d_an2 `D', an lags(2)
lfe d_jk `D', jack ss2
fcmp d_noc 1e-6 1e-6 1 : felogit `D', absorb(id year) ape tol(1e-12)
quietly gen byte sd = e(sample)
* every individual is kept or dropped whole (the weak-exogeneity term then has one definition)
quietly bysort id: egen byte sdm = max(sd)
assert sd == sdm if ns
drop sdm
fcmp d_an0 1e-6 1e-6 1 : felogit `D', absorb(id year) analytical tol(1e-12)
fcmp d_an1 1e-6 1e-6 1 : felogit `D', absorb(id year) analytical lags(1) tol(1e-12) d(dd)
fecheck "id year"
drop dd
refs d1 "L.y x1 x2" y sd ns 1 1 1 1 1
mata: rf_cmp("L_d_an1_b", "R_d1_bc", 1e-6, "Mata reference vs logitfe an lags(1) (b)")
mata: rf_cmp("L_d_an1_b2", "R_d1_apebc", 1e-6, "Mata reference vs logitfe an lags(1) (b2)")
mata: rf_cmp("e(V2)", "R_d1_V2", 1e-7, "V2 lags(1) (weak exogeneity)")
fcmp d_an1_tb 1e-6 1e-6 1 : felogit `D', absorb(id year) analytical lags(1) tbias(no) tol(1e-12)
fcmp d_an1_i 1e-6 1e-6 1 : felogit `D', absorb(id) analytical lags(1) tol(1e-12)
fcmp d_jk 1e-4 1e-4 1 : felogit `D', absorb(id year) jackknife ss2 tol(1e-12)
* B1: lags(2) against the correct formula; logitfe (psi(i,t-1) for both lags) differs
refs d2 "L.y x1 x2" y sd ns 1 1 1 1 2
matrix L_d2ok_b = R_d2_bc
matrix L_d2ok_b2 = R_d2_apebc
fcmp d2ok 1e-6 1e-6 0 : felogit `D', absorb(id year) analytical lags(2) tol(1e-12)
mata: st_numscalar("r_d2", mreldif(st_matrix("L_d_an2_b"), st_matrix("R_d2_bc")))
assert r_d2 > 1e-4
* errors: lags beyond the shortest panel (logitfe's check), lags with ibias(no)
rcof "noi felogit `D', absorb(id year) analytical lags(7)" == 198
rcof "noi felogit `D', absorb(id year) analytical lags(1) ibias(no)" == 198
* the term order of absorb() does not matter: the panel variable's term is the lag group of the
* weak-exogeneity term of e(V2) wherever it stands
quietly felogit `D', absorb(id year) analytical lags(1) tol(1e-12)
matrix O1 = e(b), e(b2)
matrix O1V = e(V2)
quietly felogit `D', absorb(year id) analytical lags(1) tol(1e-12)
matrix O2 = e(b), e(b2)
mata: st_numscalar("r_ord", max((mreldif(st_matrix("O1"), st_matrix("O2")), mreldif(st_matrix("O1V"), st_matrix("e(V2)")))))
di as text "  absorb(year id) vs absorb(id year), lags(1): max rel. diff " %9.2e r_ord
assert r_ord < 1e-9
* vce(): e(V) is plain felogit's variance under it; the APEs and e(V2) do not depend on it
quietly felogit `D', absorb(id year) vce(cluster id) tol(1e-12)
matrix VC = e(V)
quietly felogit `D', absorb(id year) analytical lags(1) vce(cluster id) tol(1e-12)
assert mreldif(VC, e(V)) < 1e-12 & "`e(vcetype)'" == "Robust" & "`e(clustvar)'" == "id"
matrix O3 = e(b), e(b2)
assert mreldif(O1, O3) < 1e-12 & mreldif(O1V, e(V2)) < 1e-12
* errors: ape with a correction, population() below N* (0 included)
rcof "noi felogit `D', absorb(id year) analytical ape" == 198
rcof "noi felogit `D', absorb(id year) jackknife ape" == 198
rcof "noi felogit `D', absorb(id year) ape population(0)" == 198
rcof "noi felogit `D', absorb(id year) analytical population(0)" == 198
* an offset: the fixed effects of d() at the corrected point solve the likelihood equations with it
quietly gen double off = 0.4 * x1 - 0.2
felogit y x2, absorb(id year) offset(off) analytical tol(1e-12) d(dd)
assert reldif(_b[x2], el(e(b0), 1, 1)) > 1e-6
fecheck "id year"
drop dd
felogit y x2, absorb(id year) offset(off) jackknife ss2 tol(1e-12) d(dd)
fecheck "id year"
drop dd off
di as result "D1 dynamic passed"

// ===========================================================================
// D2: unbalanced panel with gaps (calendar lags, casewise means, per-individual time halves)
// ===========================================================================
mkdata D2
gen byte ns = !missing(y, x1, x2)
lfe u_noc `S', noc
lfe u_an `S', an
lfe u_jk_ss2 `S', jack ss2
lfe u_jk_ss1 `S', jack ss1
lfe ud_an1 `D', an lags(1)
fcmp u_noc 1e-6 1e-6 1 : felogit `S', absorb(id year) ape tol(1e-12)
quietly gen byte su = e(sample)
refs u "x1 x2" y su ns 1 1 1 1 0
mata: rf_cmp("e(V2)", "R_u_V2", 1e-7, "V2 unbalanced")
fcmp u_an 1e-6 1e-6 1 : felogit `S', absorb(id year) analytical tol(1e-12)
fcmp u_jk_ss2 1e-4 1e-4 0 : felogit `S', absorb(id year) jackknife ss2 tol(1e-12)
fcmp u_jk_ss1 1e-4 1e-4 0 : felogit `S', absorb(id year) jackknife ss1 tol(1e-12)
fcmp ud_an1 1e-6 1e-6 1 : felogit `D', absorb(id year) analytical lags(1) tol(1e-12)
* a regressor that is zero in the second half of the periods is lost in a subpanel: an error
* (logitfe enters it as 0)
gen double x3 = x1 * (year <= 1994)
rcof "noi felogit y x1 x3, absorb(id year) jackknife ss1" == 498
di as result "D2 passed"

// ===========================================================================
// D3: directed pairs (tsset id jd), double jackknife
// ===========================================================================
mkdata D3
local S y x1 x2
lfe t_noc `S', noc
lfe t_an `S', an
lfe t_dbl `S', jack double
lfe t_dbl_ib `S', jack double ibias(no)
lfe t_dbl_tb `S', jack double tbias(no)
fcmp t_noc 1e-6 1e-6 1 : felogit `S', absorb(id jd) ape tol(1e-12)
fcmp t_an 1e-6 1e-6 1 : felogit `S', absorb(id jd) analytical tol(1e-12)
fcmp t_dbl 1e-4 1e-4 0 : felogit `S', absorb(id jd) jackknife double tol(1e-12)
* B3: double's single-bias variants are swapped in logitfe
fcmp t_dbl_ib 1e-4 1e-4 0 : felogit `S', absorb(id jd) jackknife double tbias(no) tol(1e-12)
fcmp t_dbl_tb 1e-4 1e-4 0 : felogit `S', absorb(id jd) jackknife double ibias(no) tol(1e-12)
di as result "D3 passed"
* results are bit-identical for any number of threads
foreach o in "analytical" "jackknife double" "jackknife jj" {
	quietly felogit `S', absorb(id jd) `o' threads(1) tol(1e-12)
	matrix T1 = e(b), e(b2)
	matrix T1V = e(V2)
	quietly felogit `S', absorb(id jd) `o' threads(4) tol(1e-12)
	matrix T4 = e(b), e(b2)
	assert mreldif(T1, T4) == 0 & mreldif(T1V, e(V2)) == 0
}
di as result "threads passed"

// ===========================================================================
// D4: network structure (exporter-time, importer-time, pair effects) vs alpaca 0.3.5: the data
// and alpaca's numbers come from gen_bc_network.R (alpaca at tight tolerances: exact MLE, corrected
// point within ~1e-10 of an exact fixed-effect re-solve; no group with a constant outcome, so
// N* = N and the sampling-of-effects term is alpaca's)
// ===========================================================================
import delimited using "bc_network_ref.csv", varnames(1) stringcols(_all) clear
mata:
// reference <case> <quantity> as matrix <out>, named as alpaca (matrices: names "row:col")
void nwref(string scalar cs, string scalar q, string scalar out)
{
	string matrix S
	string colvector rn, cn
	string rowvector names
	real colvector v
	real matrix A
	real scalar k, i, p, r, c

	S = st_sdata(., ("case", "quantity", "name", "value"))
	S = select(S, (S[., 1] :== cs) :& (S[., 2] :== q))
	if (!rows(S)) _error(3498, "no reference " + cs + " " + q)
	v = strtoreal(S[., 4])
	if (!strpos(S[1, 3], ":")) {
		st_matrix(out, v')
		st_matrixcolstripe(out, (J(rows(v), 1, ""), S[., 3]))
		return
	}
	p = strpos(S[., 3], ":")
	rn = substr(S[., 3], 1, p :- 1)
	cn = substr(S[., 3], p :+ 1, .)
	k = sqrt(rows(v))
	names = rn[1..k]'
	A = J(k, k, .)
	for (i = 1; i <= rows(v); i++) {
		r = selectindex(names :== rn[i])
		c = selectindex(names :== cn[i])
		A[r, c] = v[i]
	}
	st_matrix(out, A)
	st_matrixcolstripe(out, (J(k, 1, ""), names'))
	st_matrixrowstripe(out, (J(k, 1, ""), names'))
}
end
foreach c in noc an0 an2 {
	foreach q in b b2 {
		mata: nwref("`c'", "`q'", "A_`c'_`q'")
	}
}
mata: nwref("noc", "V2", "A_noc_V2"); nwref("noc", "V2_weakexo", "A_noc_V2w")
mata: nwref("all", "N", "A_N"); nwref("all", "M", "A_M")
local M = el(A_M, 1, 1)
import delimited using "bc_network.csv", varnames(1) clear asdouble
xtset pair year
local W y ylag x rta
felogit `W', absorb(exp_time imp_time pair) network ape population(`M') tol(1e-12)
assert e(N) == el(A_N, 1, 1) & e(N_full) == e(N) & reldif(e(fpc), (`M' - e(N)) / (`M' - 1)) < 1e-15
mata: rf_cmp("e(b)", "A_noc_b", 1e-7, "network b (uncorrected)")
mata: rf_cmp("e(b2)", "A_noc_b2", 1e-7, "network b2 (uncorrected APEs)")
mata: rf_cmp("e(V2)", "A_noc_V2", 1e-7, "network V2 (population)")
felogit `W', absorb(exp_time imp_time pair) network analytical population(`M') tol(1e-12) d(dd)
mata: rf_cmp("e(b)", "A_an0_b", 1e-7, "network analytical L = 0 (b)")
mata: rf_cmp("e(b2)", "A_an0_b2", 1e-7, "network analytical L = 0 (b2)")
mata: rf_cmp("e(V2)", "A_noc_V2", 1e-7, "network analytical L = 0 (V2 at the uncorrected fit)")
assert "`e(structure)'" == "network" & "`e(bias)'" == "analytical"
fecheck "exp_time imp_time pair"
drop dd
felogit `W', absorb(exp_time imp_time pair) network analytical lags(2) population(`M') tol(1e-12)
mata: rf_cmp("e(b)", "A_an2_b", 1e-7, "network analytical L = 2 (b)")
mata: rf_cmp("e(b2)", "A_an2_b2", 1e-7, "network analytical L = 2 (b2)")
mata: rf_cmp("e(V2)", "A_noc_V2w", 1e-7, "network L = 2: V2 with weak exogeneity")
assert "`e(lags)'" == "2"
rcof "noi felogit `W', absorb(exp_time imp_time pair) network jackknife" == 198
di as result "D4 network passed"

di as result "test_logitfe passed"
