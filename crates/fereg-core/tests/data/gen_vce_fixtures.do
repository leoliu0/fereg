* Ground-truth fixtures for crates/fereg-core/tests/vce.rs (Stata-layer variance estimators).
* Requires reghdfe 6.13.1, ivreghdfe 1.1.4 (ivreg2, ranktest 2.0.04), ppmlhdfe 2.3.3, ftools and
* Stata 19 `regress` (vce(hc2|hc3)). Run from this directory:
*     stata-mp -b do gen_vce_fixtures.do
* Inputs: iv_nls.csv (written by gen_iv_fixtures.do). Writes vce_ts.csv, vce_cs.csv and one
* vce_<spec>.json per spec with b, V (row-major), optionally S, e(sample), every numeric e()
* scalar and the string macros kernel, bwchoice, vce.
*   - reghdfe vce(dkraay [#]) (positional periods): OLS on the nlswork panel (gaps), weights,
*     an absorbed FE nested in time, bw >= T.
*   - ivreghdfe kernels on OLS (no endogenous regressors) and IV: every kernel, AC, HAC,
*     Driscoll-Kraay dkraay(#), Thompson cluster(idcode year)+bw, kiefer, psd0/psda, aw/pw,
*     a time series with gaps (bw(auto) for each kernel) and one with tdelta = 2.
*   - regress y x i.fe, vce(hc2|hc3) (explicit dummies; b and V of the x block): none/aw/fw,
*     pw with hc2, and without FE (with _cons).
*   - CRV3: leave-one-cluster-out loops of reghdfe / ivreghdfe / ppmlhdfe; V = q*sum_g
*     (b_g - b)(b_g - b)' with the reference's cluster factor q (inclusion-exclusion for two
*     dimensions).

version 16
clear all
set more off
set type double

mata:
string scalar num2json(real scalar x)
{
    string scalar s
    if (missing(x)) return("null")
    s = strtrim(sprintf("%21.17g", x))
    if (substr(s, 1, 1) == ".") s = "0" + s
    if (substr(s, 1, 2) == "-.") s = "-0" + substr(s, 2, .)
    return(s)
}

string scalar vec2json(real rowvector v)
{
    real scalar i
    string scalar s
    s = "["
    for (i = 1; i <= cols(v); i++) s = s + num2json(v[i]) + (i < cols(v) ? ", " : "")
    return(s + "]")
}

void write_csv(string scalar fname, string scalar varlist)
{
    real scalar fh, i, j
    real matrix X
    string rowvector vars
    string scalar line
    vars = tokens(varlist)
    X = st_data(., vars)
    if (fileexists(fname)) unlink(fname)
    fh = fopen(fname, "w")
    fput(fh, invtokens(vars, ","))
    for (i = 1; i <= rows(X); i++) {
        line = ""
        for (j = 1; j <= cols(X); j++) line = line + (j > 1 ? "," : "") + num2json(X[i, j])
        fput(fh, line)
    }
    fclose(fh)
}

void put_mat(real scalar fh, string scalar key, string scalar stname)
{
    real matrix M
    if (stname == "") return
    M = st_matrix(stname)
    if (rows(M) == 0) return
    fput(fh, `"  ""' + key + `"": "' + vec2json(vec(M')') + ",")
}

// b, V, S: Stata matrix names ("" = absent); scalars and macros of the current e().
void write_json(string scalar fname, string scalar name, string scalar bn, string scalar vn, string scalar sn_)
{
    real scalar fh, i
    string rowvector sn, mn
    if (fileexists(fname)) unlink(fname)
    fh = fopen(fname, "w")
    fput(fh, "{")
    fput(fh, `"  "name": ""' + name + `"","')
    fput(fh, `"  "cmdline": ""' + subinstr(st_global("e(cmdline)"), `"""', "'") + `"","')
    put_mat(fh, "b", bn)
    put_mat(fh, "V", vn)
    put_mat(fh, "S", sn_)
    mn = ("kernel", "bwchoice", "vce", "vcetype", "clustvar")
    for (i = 1; i <= cols(mn); i++) {
        fput(fh, `"  ""' + mn[i] + `"": ""' + st_global("e(" + mn[i] + ")") + `"","')
    }
    fput(fh, `"  "sample": "' + vec2json(st_data(., "__esample")') + ",")
    sn = sort(st_dir("e()", "numscalar", "*"), 1)'
    fput(fh, `"  "scalars": {"')
    for (i = 1; i <= cols(sn); i++) {
        fput(fh, `"    ""' + sn[i] + `"": "' + num2json(st_numscalar("e(" + sn[i] + ")")) + (i < cols(sn) ? "," : ""))
    }
    fput(fh, "  }")
    fput(fh, "}")
    fclose(fh)
}

// Jackknife: B is G x k (leave-one-out coefficients), b the full-sample 1 x k.
real matrix jk_meat(real matrix B, real rowvector b)
{
    real matrix D
    D = B :- b
    return(quadcross(D, D))
}
end

* Saves the current e() with e(b), e(V) (and e(S) when posted).
capture program drop savefix
program savefix
    syntax name(name=spec) [, b(string) v(string)]
    capture drop __esample
    gen byte __esample = e(sample)
    if "`b'" == "" {
        tempname bb vv
        matrix `bb' = e(b)
        matrix `vv' = e(V)
        local b `bb'
        local v `vv'
    }
    local s
    capture confirm matrix e(S)
    if !_rc {
        tempname ss
        matrix `ss' = e(S)
        local s `ss'
    }
    mata: write_json("vce_`spec'.json", "`spec'", "`b'", "`v'", "`s'")
    drop __esample
    di as txt "saved vce_`spec'.json"
end

loc t tol(1e-14) iterate(100000)

* =============================================================================================
* A. nlswork panel (gaps in year), xtset idcode year.
import delimited using iv_nls.csv, clear asdouble
xtset idcode year

* reghdfe vce(dkraay): positional periods, Bartlett.
reghdfe ln_wage age tenure, absorb(idcode) vce(dkraay) `t'
savefix dk_default
reghdfe ln_wage age tenure, absorb(idcode year) vce(dkraay 3) `t'
savefix dk_nested
reghdfe ln_wage age tenure [aw=w], absorb(idcode year) vce(dkraay 2) `t'
savefix dk_aw
reghdfe ln_wage age tenure [fw=fw], absorb(idcode year) vce(dkraay 2) `t'
savefix dk_fw
reghdfe ln_wage age tenure [pw=w], absorb(idcode) vce(dkraay 4) `t'
savefix dk_pw
reghdfe ln_wage age tenure, absorb(idcode) vce(dkraay 40) `t'
savefix dk_big
reghdfe ln_wage age tenure, absorb(idcode) vce(dkraay 1) `t'
savefix dk_one

* ivreghdfe kernels, OLS (no endogenous regressors) and IV.
loc ols ln_wage age tenure
loc iv ln_wage age (tenure = union south)
foreach m in ols iv {
    ivreghdfe ``m'', absorb(idcode) bw(3) vce(robust) `t'
    savefix `m'_hac_bartlett
    ivreghdfe ``m'', absorb(idcode year) bw(3) `t'
    savefix `m'_ac_bartlett
    foreach k in parzen truncated thann thamm qs danielle tent {
        ivreghdfe ``m'', absorb(idcode) bw(3) kernel(`k') vce(robust) `t'
        savefix `m'_hac_`k'
    }
    ivreghdfe ``m'', absorb(idcode) bw(2.5) kernel(parzen) `t'
    savefix `m'_ac_parzen_frac
    ivreghdfe ``m'', absorb(idcode) kiefer `t'
    savefix `m'_kiefer
    ivreghdfe ``m'', absorb(idcode) dkraay(3) `t'
    savefix `m'_dkraay
    ivreghdfe ``m'', absorb(idcode year) dkraay(2) kernel(qs) `t'
    savefix `m'_dkraay_qs
    ivreghdfe ``m'', absorb(year) vce(cluster idcode year) bw(2) `t'
    savefix `m'_thompson
    ivreghdfe ``m'', absorb(idcode) vce(cluster year idcode) bw(3) kernel(tru) `t'
    savefix `m'_thompson_tru
    ivreghdfe ``m'', absorb(idcode) bw(4) kernel(qs) vce(robust) psda `t'
    savefix `m'_hac_qs_psda
    ivreghdfe ``m'', absorb(idcode) bw(4) kernel(tent) psd0 `t'
    savefix `m'_ac_tent_psd0
    ivreghdfe ``m'' [aw=w], absorb(idcode) bw(3) vce(robust) `t'
    savefix `m'_hac_aw
    ivreghdfe ``m'' [pw=w], absorb(idcode year) dkraay(2) `t'
    savefix `m'_dkraay_pw
    ivreghdfe ``m'' [aw=w], absorb(idcode) bw(3) kernel(parzen) `t'
    savefix `m'_ac_aw
}

* =============================================================================================
* B. Time series with gaps (tsset t) and with tdelta = 2.
clear
set seed 20261013
set obs 260
gen long t = _n
gen double u = rnormal()
gen double v = rnormal()
gen double x2 = rnormal()
replace u = 0.6 * u[_n-1] + u if _n > 1
replace x2 = 0.5 * x2[_n-1] + x2 if _n > 1
gen double x1 = rnormal()
gen double z1 = rnormal()
gen double z2 = rnormal()
replace z1 = 0.4 * z1[_n-1] + z1 if _n > 1
gen long q = 1 + mod(t, 4)
gen double en = 0.6 * z1 + 0.4 * z2 + 0.5 * u + v
gen double y = 1 + 0.5 * x1 - 0.3 * x2 + 0.7 * en + 0.2 * q + u + 0.5 * rnormal()
gen double w = 0.5 + runiform()
drop if mod(t, 37) == 5 | t == 100 | t == 101 | t == 102
gen long t2 = 2 * t
drop u v
mata: write_csv("vce_ts.csv", "t t2 q x1 x2 z1 z2 en y w")

tsset t
loc ols y x1 x2
loc iv y x1 (en = z1 z2)
foreach m in ols iv {
    ivreghdfe ``m'', absorb(q) bw(4) vce(robust) `t'
    savefix ts_`m'_hac
    ivreghdfe ``m'', absorb(q) bw(4) kernel(qs) `t'
    savefix ts_`m'_ac_qs
    foreach k in bartlett parzen qs tent {
        ivreghdfe ``m'', absorb(q) bw(auto) kernel(`k') vce(robust) `t'
        savefix ts_`m'_auto_`k'
    }
    ivreghdfe ``m'' [aw=w], absorb(q) bw(auto) `t'
    savefix ts_`m'_auto_ac_aw
}
tsset t2, delta(2)
foreach m in ols iv {
    ivreghdfe ``m'', absorb(q) bw(3) vce(robust) `t'
    savefix ts2_`m'_hac
    ivreghdfe ``m'', absorb(q) bw(5) kernel(danielle) `t'
    savefix ts2_`m'_ac_dan
}

* =============================================================================================
* C/D. Cross-section panel: 25 clusters x 4 firms x 6 years (no singletons when a cluster is
* left out), a second cluster dimension c2, an instrument, a count outcome.
clear
set seed 20261014
set obs 600
gen long obs = _n
gen long f1 = ceil(_n / 6)
gen long f2 = 1 + mod(_n - 1, 6)
gen long c = ceil(f1 / 4)
gen long c2 = 1 + mod(_n * 7, 11)
gen double z1 = rnormal()
gen double z2 = rnormal()
gen double x1 = rnormal()
gen double x2 = rnormal() + 0.3 * x1
gen double u = rnormal() * (1 + abs(x1))
bys c (obs): gen double ac = rnormal() if _n == 1
by c: replace ac = ac[1]
sort obs
gen double en = 0.6 * z1 + 0.4 * z2 + 0.3 * x1 + 0.5 * u + rnormal()
gen double y = 1 + 0.5 * x1 - 0.3 * x2 + 0.8 * en + 0.1 * f2 + ac + u
gen double yp = rpoisson(exp(0.5 + 0.3 * x1 - 0.2 * x2 + 0.1 * f2 + 0.3 * ac))
gen double w = 0.5 + runiform()
gen double fw = 1 + mod(_n, 3)
drop u ac
mata: write_csv("vce_cs.csv", "obs f1 f2 c c2 z1 z2 x1 x2 en y yp w fw")

* regress, vce(hc2|hc3) with explicit dummies: the x1 x2 block.
foreach hc in hc2 hc3 {
    foreach wt in none aw fw {
        loc wexp
        if "`wt'" == "aw" loc wexp [aw=w]
        if "`wt'" == "fw" loc wexp [fw=fw]
        regress y x1 x2 i.f1 i.f2 `wexp', vce(`hc')
        matrix __b = e(b)[1, 1..2]
        matrix __V = e(V)[1..2, 1..2]
        savefix `hc'_`wt', b(__b) v(__V)
    }
    regress y x1 x2, vce(`hc')
    savefix `hc'_noabsorb
}
regress y x1 x2 i.f1 i.f2 [pw=w], vce(hc2)
matrix __b = e(b)[1, 1..2]
matrix __V = e(V)[1..2, 1..2]
savefix hc2_pw, b(__b) v(__V)

* CRV3, reghdfe: one dimension (FE nested in the cluster), aw, no FE in the cluster, two-way.
capture program drop crv3_ols
program crv3_ols
    syntax varlist [aw], absorb(string) cluster(varlist) spec(name)
    loc wexp
    if "`weight'" != "" loc wexp [`weight'`exp']
    reghdfe `varlist' `wexp', absorb(`absorb') vce(cluster `cluster') tol(1e-14)
    tempname b0 M B
    matrix `b0' = e(b)
    loc N = e(N)
    loc q = (e(N) - 1) / (e(N) - (e(df_a_nested) > 0) - e(df_m) - e(df_a)) * e(N_clust) / (e(N_clust) - 1)
    loc k = colsof(`b0')
    matrix `M' = J(`k', `k', 0)
    loc nd : word count `cluster'
    * inclusion-exclusion over the non-empty subsets of the dimensions (at most two here)
    loc subsets `cluster'
    if `nd' == 2 loc subsets `cluster' __both
    tempvar both
    if `nd' == 2 {
        gettoken ca cb : cluster
        egen long `both' = group(`ca' `cb')
    }
    foreach s of local subsets {
        loc sv `s'
        loc sign 1
        if "`s'" == "__both" {
            loc sv `both'
            loc sign -1
        }
        qui levelsof `sv', local(levs)
        matrix `B' = J(`: word count `levs'', `k', .)
        loc i 0
        foreach g of local levs {
            loc ++i
            qui reghdfe `varlist' `wexp' if `sv' != `g', absorb(`absorb') tol(1e-14)
            qui count if `sv' == `g'
            assert e(N) == `N' - r(N)
            matrix `B'[`i', 1] = e(b)
        }
        mata: st_matrix("`M'", st_matrix("`M'") + `sign' * jk_meat(st_matrix("`B'"), st_matrix("`b0'")))
    }
    qui reghdfe `varlist' `wexp', absorb(`absorb') vce(cluster `cluster') tol(1e-14)
    tempname V
    matrix `V' = `q' * `M'
    savefix `spec', b(`b0') v(`V')
end
crv3_ols y x1 x2, absorb(f1 f2) cluster(c) spec(crv3_ols_nested)
crv3_ols y x1 x2 [aw=w], absorb(f1 f2) cluster(c) spec(crv3_ols_aw)
crv3_ols y x1 x2, absorb(f2) cluster(c2) spec(crv3_ols_c2)
crv3_ols y x1 x2, absorb(f2) cluster(c c2) spec(crv3_ols_twoway)

* CRV3, ivreghdfe.
ivreghdfe y x1 (en = z1 z2), absorb(f1 f2) cluster(c) tol(1e-14)
matrix __b0 = e(b)
loc q = (e(N) - 1) / (e(N) - e(rankxx) - e(sdofminus)) * e(N_clust) / (e(N_clust) - 1)
matrix __B = J(25, colsof(__b0), .)
forvalues g = 1/25 {
    qui ivreghdfe y x1 (en = z1 z2) if c != `g', absorb(f1 f2) tol(1e-14)
    matrix __B[`g', 1] = e(b)
}
mata: st_matrix("__V", `q' * jk_meat(st_matrix("__B"), st_matrix("__b0")))
qui ivreghdfe y x1 (en = z1 z2), absorb(f1 f2) cluster(c) tol(1e-14)
savefix crv3_iv, b(__b0) v(__V)

* CRV3, ppmlhdfe (slopes; the constant is not compared).
ppmlhdfe yp x1 x2, absorb(f1 f2) vce(cluster c) tol(1e-12)
matrix __b0 = e(b)
loc q = e(N_clust) / (e(N_clust) - 1)
matrix __B = J(25, colsof(__b0), .)
forvalues g = 1/25 {
    qui ppmlhdfe yp x1 x2 if c != `g', absorb(f1 f2) tol(1e-12)
    matrix __B[`g', 1] = e(b)
}
mata: st_matrix("__V", `q' * jk_meat(st_matrix("__B"), st_matrix("__b0")))
qui ppmlhdfe yp x1 x2, absorb(f1 f2) vce(cluster c) tol(1e-12)
savefix crv3_ppml, b(__b0) v(__V)

exit, clear
