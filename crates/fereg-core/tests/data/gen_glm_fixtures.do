* Ground-truth fixtures for crates/fereg-core/tests/glm.rs.
* Requires ppmlhdfe 2.3.3 (+ reghdfe/ftools) and Stata's logit. Run from this directory:
*     stata-mp -b do gen_glm_fixtures.do
*
* Poisson fixtures: ppmlhdfe 2.3.3 on the datasets of the ppmlhdfe test suite
* (https://github.com/sergiocorreia/ppmlhdfe/tree/master/test: problem1-6, hard1-5,
* collinear1-7, two_way, fweight, pweight, exposure, offset, no_regressors, savefe), plus a
* simulated gravity panel.
*
*   The ppmlhdfe test data blocks below are ported from the ppmlhdfe repository:
*   MIT License, Copyright (c) 2019 Sergio Correia, Paulo Guimaraes, and Thomas Zylkin.
*   ppmlhdfe_collinear.dta and ppmlhdfe_problem6.dta are test/datasets/{collinear,problem6}.dta.
*
* Every ppmlhdfe call passes sep(fe relu): fereg treats `simplex` as an alias of `relu`, and
* ppmlhdfe's simplex presolve needs moremata (mm_matlist), which is not part of its dependencies.
*
* Logit fixtures: Stata `logit y x i.fe1 i.fe2` (oim / robust / cluster / fw / pw) on simulated,
* separation-free data, plus a variant with two all-zero fe1 groups.
*
* Writes glm_<data>.csv (each dataset at full precision, with obs = 1.._N) and glm_<spec>.json
* (e(b), e(V) row-major, e(sample), every numeric e() scalar, and d / saved FEs when requested).

version 16
clear all
set more off
* Ties in sort/bysort are broken randomly; fix that stream so the simulated data are reproducible.
set sortseed 20261011

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

void write_json(string scalar fname, string scalar name, string scalar extras)
{
    real scalar fh, i
    real matrix V
    string rowvector sn, ex
    if (fileexists(fname)) unlink(fname)
    fh = fopen(fname, "w")
    fput(fh, "{")
    fput(fh, `"  "name": ""' + name + `"","')
    fput(fh, `"  "cmdline": ""' + subinstr(st_global("e(cmdline)"), `"""', "'") + `"","')
    if (st_matrix("e(b)") != J(0, 0, .)) {
        fput(fh, `"  "colnames": ""' + invtokens(st_matrixcolstripe("e(b)")[., 2]') + `"","')
        fput(fh, `"  "b": "' + vec2json(st_matrix("e(b)")) + ",")
        V = st_matrix("e(V)")
        fput(fh, `"  "V": "' + vec2json(vec(V')') + ",")
    }
    fput(fh, `"  "sample": "' + vec2json(st_data(., "__esample")') + ",")
    ex = tokens(extras)
    for (i = 1; i <= cols(ex); i++) {
        fput(fh, `"  ""' + ex[i] + `"": "' + vec2json(st_data(., ex[i])') + ",")
    }
    sn = sort(st_dir("e()", "numscalar", "*"), 1)'
    fput(fh, `"  "scalars": {"')
    for (i = 1; i <= cols(sn); i++) {
        fput(fh, `"    ""' + sn[i] + `"": "' + num2json(st_numscalar("e(" + sn[i] + ")")) + (i < cols(sn) ? "," : ""))
    }
    fput(fh, "  }")
    fput(fh, "}")
    fclose(fh)
}
end

capture program drop savefix
program savefix
    syntax name(name=spec) [, EXTRAS(string)]
    capture drop __esample
    gen byte __esample = e(sample)
    mata: write_json("glm_`spec'.json", "`spec'", "`extras'")
    drop __esample
    di as txt "saved glm_`spec'.json"
end

capture program drop savedata
program savedata
    syntax name(name=data), VARS(string)
    capture drop obs
    gen long obs = _n
    mata: write_csv("glm_`data'.csv", "obs `vars'")
end

* =============================================================================================
* ppmlhdfe test datasets
* =============================================================================================

* --- problem1.do -----------------------------------------------------------------------------
clear
input float(y x1 x2 i j)
2 0 0 2 1
0 0 0 4 2
0 0 0 1 1
1 1 0 4 3
0 0 1 2 2
0 0 0 2 2
1 0 0 5 4
0 1 2 2 3
1 0 0 1 1
0 0 0 2 2
0 2 0 1 3
0 0 0 1 3
0 1 0 2 2
0 0 1 5 2
0 0 1 2 4
0 0 0 1 2
0 0 0 1 1
2 0 0 1 2
end
savedata problem1, vars(y x1 x2 i j)
ppmlhdfe y x1 x2, absorb(i j) tol(1e-10) keepsing sep(mu)
savefix p_problem1_keepsing_mu
* With sep(fe relu), ppmlhdfe's ReLU check hits its 100-iteration cap on this dataset and the
* separated data are then fitted (min(eta) diverges). The answer ppmlhdfe's own test
* (problem1.do, TEST 2) expects drops obs 5 8 14 15 (separated) and 7 (then a singleton):
ppmlhdfe y x1 x2 if !inlist(_n, 5, 7, 8, 14, 15), absorb(i j) tol(1e-10) sep(fe relu)
savefix p_problem1

* --- problem2.do -----------------------------------------------------------------------------
clear
input float(y x1 x2 i j)
2 0 0 2 1
1 0 1 2 3
2 0 1 2 2
1 1 0 2 2
0 1 0 1 2
1 2 0 1 3
1 0 0 2 1
2 1 0 2 3
0 1 0 2 2
end
savedata problem2, vars(y x1 x2 i j)
ppmlhdfe y x1 x2, absorb(i j) tol(1e-10) sep(fe relu)
savefix p_problem2

* --- problem3.do -----------------------------------------------------------------------------
clear
input double y byte(x1 x2 x3)
        0 0 0 1
        0 0 0 1
        0 0 0 2
2.2596662 1 1 2
2.4177196 2 2 2
0.9788354 3 2 1
2.6114680 4 2 1
end
savedata problem3, vars(y x1 x2 x3)
ppmlhdfe y x1 x2 x3, noa tol(1e-12) sep(fe relu)
savefix p_problem3

* --- problem4.do -----------------------------------------------------------------------------
clear
input double y byte(x1 x2 x3)
  0  2 3 4
  0 -5 0 1
  0  0 2 1
  0  0 0 3
  0 -2 0 1
  0  0 0 2
1.1  1 1 2
 .5  2 2 2
3.3  3 3 2
2.1  4 2 4
end
savedata problem4, vars(y x1 x2 x3)
ppmlhdfe y x1 x2 x3, noa tol(1e-12) maxiter(200) sep(fe relu)
savefix p_problem4

* --- problem5.do -----------------------------------------------------------------------------
clear
input float(y x1 x2 i j)
2 0 0 2 1
1 0 1 4 3
2 0 1 2 2
1 1 0 2 2
0 1 0 1 2
1 2 0 1 3
1 0 0 2 1
2 1 0 2 3
0 1 0 2 2
end
savedata problem5, vars(y x1 x2 i j)
ppmlhdfe y x1 x2, absorb(i j) tol(1e-12) sep(fe relu)
savefix p_problem5

* --- problem6.do -----------------------------------------------------------------------------
use ppmlhdfe_problem6.dta, clear
recast double y
savedata problem6, vars(y x)
ppmlhdfe y x, noa tol(1e-10) sep(fe relu)
savefix p_problem6

* --- hard1.do --------------------------------------------------------------------------------
clear
input byte(y id1 id2)
0 1 1
1 1 1
0 2 1
0 2 2
1 2 2
end
savedata hard1, vars(y id1 id2)
ppmlhdfe y, absorb(id1 id2) tol(1e-10) sep(fe relu)
savefix p_hard1

* --- hard2.do --------------------------------------------------------------------------------
clear
input byte FTA int(ikt_fe jkt_fe ijk_fe)
1 1 3 1
1 2 4 1
1 1 5 2
1 2 6 2
1 3 1 3
1 4 2 3
1 3 7 4
1 4 8 4
1 5 1 5
1 6 2 5
0 5 3 6
1 6 4 6
0 5 5 7
1 6 6 7
1 5 7 8
1 6 8 8
end
savedata hard2, vars(FTA ikt_fe jkt_fe ijk_fe)
ppmlhdfe FTA, absorb(ijk_fe jkt_fe ikt_fe) tol(1e-12) keepsing sep(fe relu)
savefix p_hard2_keepsing

* --- hard3.do --------------------------------------------------------------------------------
clear
set obs 100
set seed 12345
gen int id1 = ceil(_n/3)
gen int id2 = ceil(cond(_n==1,_N,_n-1)/3)
gen int y = runiform() < 0.1
savedata hard3, vars(y id1 id2)
ppmlhdfe y, absorb(id1 id2) tol(1e-10) sep(fe relu)
savefix p_hard3

* --- hard4.do / hard5.do --------------------------------------------------------------------
clear
input byte(y x1 x2 x3 x4 x5)
0  0  0  0  0 1
0  0  0  0  0 2
0  5 11  2  2 0
0  5  2 11  2 0
0  5  2  2 11 0
0  0 -1 -1 -1 0
1  5  5  5  5 0
2  4  4  4  4 0
3  3  3  3  3 0
4  2  2  2  2 0
5  1  1  1  1 0
end
savedata hard5, vars(y x1 x2 x3 x4 x5)
ppmlhdfe y x1 x2 x3 x4, noa tol(1e-10) sep(fe relu)
savefix p_hard5
ppmlhdfe y x2 x3 x4, noa tol(1e-10) sep(fe relu)
savefix p_hard5b
* hard4 is hard5 without its first two observations
ppmlhdfe y x2 x3 x4 if _n > 2, noa tol(1e-10) sep(fe relu)
savefix p_hard4

* --- collinear1.do ---------------------------------------------------------------------------
use ppmlhdfe_collinear.dta, clear
savedata collinear1, vars(y x z im ex)
ppmlhdfe y x z, absorb(im ex) tol(1e-10) sep(fe relu)
savefix p_collinear1

* --- collinear2.do / collinear7.do -----------------------------------------------------------
drawnorm x1, n(1000) seed(101010) double clear
generate double u = rpoisson(1)
generate y = exp(1+10*x1)*u
generate double x2 = (y==0)
savedata collinear2, vars(y x1 x2)
ppmlhdfe y x1 x2, noa tol(1e-10) sep(fe relu)
savefix p_collinear2
ppmlhdfe y x1, noa tol(1e-10) sep(fe relu)
savefix p_collinear7

* --- collinear3.do ---------------------------------------------------------------------------
drawnorm x1, n(1000) seed(101010) double clear
generate double y = rpoisson(1)
generate double x2 = (y==0)
savedata collinear3, vars(y x1 x2)
ppmlhdfe y x1 x2, noa tol(1e-10) sep(fe relu)
savefix p_collinear3

* --- collinear4.do ---------------------------------------------------------------------------
drawnorm x1, n(1000) seed(101010) double clear
generate double y = rpoisson(1)
generate double x2 = (y==0) * (1+runiform())
savedata collinear4, vars(y x1 x2)
ppmlhdfe y x1 x2, noa tol(1e-10) sep(fe relu)
savefix p_collinear4

* --- collinear5.do ---------------------------------------------------------------------------
drawnorm x1, n(1000) seed(101010) double clear
generate double u = rpoisson(1)
generate y = exp(1+10*x1)*u
generate double x2 = 0
savedata collinear5, vars(y x1 x2)
ppmlhdfe y x1 x2, noa tol(1e-10) sep(fe relu)
savefix p_collinear5

* --- collinear6.do ---------------------------------------------------------------------------
drawnorm u e x1, n(1000) seed(101010) double clear
generate double x2 = (x1+e/18000)
generate double y = exp(1+x1+x2+u)
savedata collinear6, vars(y x1 x2)
ppmlhdfe y x1 x2, noa tol(1e-12) sep(fe relu)
savefix p_collinear6

* --- auto: two_way, fweight, pweight, exposure, offset, no_regressors, savefe ----------------
sysuse auto, clear
gen double W = foreign + 1
gen double lnhead = ln(headroom)
savedata auto, vars(price weight gear_ratio length turn trunk foreign displacement headroom lnhead mpg rep78 W)

ppmlhdfe price weight gear length, absorb(turn trunk) vce(robust) keepsingletons tol(1e-12) sep(fe relu)
savefix p_twoway_robust
ppmlhdfe price weight gear length, absorb(turn trunk) vce(cluster turn trunk) keepsingletons tol(1e-12) sep(fe relu)
savefix p_twoway_cluster2
ppmlhdfe price weight gear length, absorb(turn trunk) vce(cluster turn) tol(1e-12) sep(fe relu)
savefix p_twoway_cluster_turn

ppmlhdfe price weight gear length [fw=displacement], noabsorb tol(1e-10) sep(fe relu)
savefix p_fw_noa
ppmlhdfe price weight gear length [fw=displacement], noabsorb vce(cluster turn) tol(1e-10) sep(fe relu)
savefix p_fw_cluster
ppmlhdfe price weight gear length [fw=turn], noabsorb vce(cluster trunk) tol(1e-10) sep(fe relu)
savefix p_fw_turn_cluster
ppmlhdfe price weight gear length [fw=displacement], absorb(turn) keepsing tol(1e-10) sep(fe relu)
savefix p_fw_absorb

ppmlhdfe price weight gear length [pw=displacement], noabsorb tol(1e-10) sep(fe relu)
savefix p_pw_noa
ppmlhdfe price weight gear length [pw=displacement], noabsorb vce(cluster turn) tol(1e-10) sep(fe relu)
savefix p_pw_cluster
ppmlhdfe price weight gear length [pw=displacement], absorb(turn) keepsing tol(1e-10) sep(fe relu)
savefix p_pw_absorb

ppmlhdfe price weight gear length, noa exposure(headroom) tol(1e-12) sep(fe relu)
savefix p_exposure_noa
ppmlhdfe price weight gear length, absorb(turn) exposure(headroom) tol(1e-10) keepsingletons sep(fe relu)
savefix p_exposure_absorb

ppmlhdfe price weight gear length, offset(mpg) tol(1e-12) noa sep(fe relu)
savefix p_offset_noa
ppmlhdfe price weight gear length [fw=W], offset(mpg) tol(1e-10) sep(fe relu)
savefix p_offset_fw
ppmlhdfe price weight gear length, absorb(turn) offset(mpg) tol(1e-10) keepsingletons sep(fe relu)
savefix p_offset_absorb

ppmlhdfe price, noabsorb tol(1e-10) sep(fe relu)
savefix p_noreg_noa
ppmlhdfe price, absorb(turn) tol(1e-12) keepsingletons sep(fe relu)
savefix p_noreg_absorb

ppmlhdfe price weight gear, absorb(FE1=turn FE2=foreign) vce(robust) tol(1e-10) keepsing d(dfe) sep(fe relu)
savefix p_savefe, extras(dfe FE1 FE2)
drop dfe FE1 FE2

ppmlhdfe price weight gear length, absorb(turn trunk) d(dd) tol(1e-12) sep(fe relu)
savefix p_d_twoway, extras(dd)
drop dd

ppmlhdfe price weight gear length, absorb(turn) guess(ols) tol(1e-10) sep(fe relu)
savefix p_guess_ols

* =============================================================================================
* Simulated gravity panel: exporter#year, importer#year and pair fixed effects, with zeros
* =============================================================================================
clear
set seed 4242
set obs 20
gen int exp = _n
expand 20
bys exp: gen int imp = _n
drop if exp == imp
expand 8
bys exp imp: gen int year = 2000 + _n
gen double rta = runiform() < 0.3 + 0.05 * (year - 2000)
gen double x2 = rnormal()
bys exp year: gen double a_e = rnormal() if _n == 1
by exp year: replace a_e = a_e[1]
bys imp year: gen double a_i = rnormal() if _n == 1
by imp year: replace a_i = a_i[1]
bys exp imp: gen double a_p = 0.8 * rnormal() if _n == 1
by exp imp: replace a_p = a_p[1]
gen double trade = rpoisson(exp(0.5 + 0.4 * rta - 0.2 * x2 + a_e + a_i + a_p))
* a handful of pairs that never trade (separated by the pair FE)
replace trade = 0 if exp == 1 & imp <= 3
sort exp imp year
savedata gravity, vars(trade rta x2 exp imp year)
ppmlhdfe trade rta x2, absorb(exp#year imp#year exp#imp) vce(cluster exp#imp) tol(1e-10) d(dg) sep(fe relu)
savefix p_gravity, extras(dg)
drop dg

* =============================================================================================
* Logit: logit y x1 x2 i.fe1 i.fe2 on separation-free data
* =============================================================================================
clear
set seed 777
set obs 3000
gen long fe1 = ceil(_n / 50)
gen long fe2 = 1 + mod(_n * 7, 10)
gen long c = ceil(fe1 / 3)
gen double x1 = rnormal()
gen double x2 = rnormal() + 0.3 * x1
bys fe1: gen double a1 = 0.7 * rnormal() if _n == 1
by fe1: replace a1 = a1[1]
bys fe2: gen double a2 = 0.5 * rnormal() if _n == 1
by fe2: replace a2 = a2[1]
gen long o = _n
sort o
gen byte y = runiform() < invlogit(0.5 * x1 - 0.3 * x2 + a1 + a2)
gen double pw = 0.5 + runiform()
gen double fw = 1 + floor(runiform() * 3)
gen byte y2 = y
replace y2 = 0 if inlist(fe1, 5, 6)
* separation-free: every fe1 and fe2 group has both outcomes
bys fe1: egen double m1 = mean(y)
assert m1 > 0 & m1 < 1
bys fe2: egen double m2 = mean(y)
assert m2 > 0 & m2 < 1
sort o
drop m1 m2 a1 a2 o
savedata logit, vars(y y2 x1 x2 fe1 fe2 c pw fw)

logit y x1 x2 i.fe1 i.fe2, tol(1e-12) ltol(1e-14) nrtol(1e-12)
savefix l_oim
logit y x1 x2 i.fe1 i.fe2, vce(robust) tol(1e-12) ltol(1e-14) nrtol(1e-12)
savefix l_robust
logit y x1 x2 i.fe1 i.fe2, vce(cluster c) tol(1e-12) ltol(1e-14) nrtol(1e-12)
savefix l_cluster
logit y x1 x2 i.fe1 i.fe2 [fw=fw], tol(1e-12) ltol(1e-14) nrtol(1e-12)
savefix l_fw
logit y x1 x2 i.fe1 i.fe2 [pw=pw], tol(1e-12) ltol(1e-14) nrtol(1e-12)
savefix l_pw
logit y x1 x2 i.fe1, tol(1e-12) ltol(1e-14) nrtol(1e-12)
savefix l_onefe
logit y x1 x2, tol(1e-12) ltol(1e-14) nrtol(1e-12)
savefix l_noabsorb
logit y2 x1 x2 i.fe1 i.fe2, tol(1e-12) ltol(1e-14) nrtol(1e-12)
savefix l_allzero_groups
