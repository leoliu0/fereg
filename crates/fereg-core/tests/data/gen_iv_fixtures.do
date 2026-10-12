* Ground-truth fixtures for crates/fereg-core/tests/iv.rs.
* Requires ivreghdfe 1.1.4, reghdfe 6.13.1, ftools, ranktest 2.0.04 and internet access for
* `webuse nlswork`. Run from this directory:
*     stata-mp -b do gen_iv_fixtures.do
* Writes iv_nls.csv (nlswork subset), iv_sim.csv (simulated, two endogenous regressors) and one
* iv_<spec>.json per spec with e(b), e(V), e(S), e(W), e(first), e(ccev), e(cdev),
* e(dof_table) (row-major), every numeric e() scalar, e(sample) and the saved first-stage
* regressions (bfirst#, Vfirst#, rmsefirst#, df_rfirst#).

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
    M = st_matrix(stname)
    if (rows(M) == 0) return
    fput(fh, `"  ""' + key + `"": "' + vec2json(vec(M')') + ",")
}

void write_json(string scalar fname, string scalar name, real scalar nendog)
{
    real scalar fh, i
    string rowvector sn
    if (fileexists(fname)) unlink(fname)
    fh = fopen(fname, "w")
    fput(fh, "{")
    fput(fh, `"  "name": ""' + name + `"","')
    fput(fh, `"  "cmdline": ""' + subinstr(st_global("e(cmdline)"), `"""', "'") + `"","')
    fput(fh, `"  "colnames": ""' + invtokens(st_matrixcolstripe("e(b)")[., 2]') + `"","')
    put_mat(fh, "b", "e(b)")
    put_mat(fh, "V", "e(V)")
    put_mat(fh, "S", "e(S)")
    put_mat(fh, "W", "e(W)")
    put_mat(fh, "first", "e(first)")
    put_mat(fh, "ccev", "e(ccev)")
    put_mat(fh, "cdev", "e(cdev)")
    put_mat(fh, "dof_table", "e(dof_table)")
    for (i = 1; i <= nendog; i++) {
        put_mat(fh, "bfirst" + strofreal(i), "__bf" + strofreal(i))
        put_mat(fh, "Vfirst" + strofreal(i), "__Vf" + strofreal(i))
        put_mat(fh, "sfirst" + strofreal(i), "__sf" + strofreal(i))
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
end

* Saves the current ivreghdfe results (run with `ffirst savefirst`) as iv_<spec>.json.
capture program drop savefix
program savefix
    syntax name(name=spec)
    capture drop __esample
    gen byte __esample = e(sample)
    local endog `e(instd1)'
    local k : word count `endog'
    if "`e(firsteqs)'" != "" {
        estimates store __main
        forvalues i = 1/`k' {
            local en : word `i' of `endog'
            estimates restore _ivreg2_`en'
            matrix __bf`i' = e(b)
            matrix __Vf`i' = e(V)
            matrix __sf`i' = (e(rmse), e(df_r), e(df_m), e(N))
        }
        estimates restore __main
        estimates drop __main
    }
    mata: write_json("iv_`spec'.json", "`spec'", `k')
    forvalues i = 1/`k' {
        capture matrix drop __bf`i' __Vf`i' __sf`i'
    }
    drop __esample
    di as txt "saved iv_`spec'.json"
end

loc t tol(1e-14) iterate(100000) ffirst savefirst

* ---------------------------------------------------------------------------------------------
* nlswork subset: complete cases of the model variables for idcode <= 800, plus deterministic
* analytic/probability weights w and frequency weights fw.
webuse nlswork, clear
keep if idcode <= 800
keep idcode year ln_wage age tenure union south msp
foreach v of varlist * {
    quietly drop if missing(`v')
}
sort idcode year
gen double w = 0.5 + mod(idcode * 7 + year, 10) / 10
gen double fw = 1 + mod(idcode + year, 3)
recast double ln_wage age tenure union south msp
mata: write_csv("iv_nls.csv", "idcode year ln_wage age tenure union south msp w fw")

local models `" "a|(tenure = union south)" "b|(tenure = union south msp)" "j|(tenure = union)" "'
foreach m of local models {
    gettoken tag m : m, parse("|")
    gettoken bar m : m, parse("|")
    foreach v in unadj robust cl1 cl2 {
        if "`v'" == "unadj"  loc vce
        if "`v'" == "robust" loc vce vce(robust)
        if "`v'" == "cl1"    loc vce vce(cluster idcode)
        if "`v'" == "cl2"    loc vce vce(cluster idcode year)
        foreach wt in none aw fw pw {
            if "`tag'" == "j" & "`wt'" != "none" continue
            loc wexp
            if "`wt'" == "aw" loc wexp [aw=w]
            if "`wt'" == "fw" loc wexp [fw=fw]
            if "`wt'" == "pw" loc wexp [pw=w]
            ivreghdfe ln_wage age `m' `wexp', absorb(idcode year) `vce' `t'
            savefix nls_`tag'_`v'_`wt'
        }
    }
}

* ---------------------------------------------------------------------------------------------
* Simulated panel with two endogenous regressors (KP via SVD, AP/SW first-stage statistics).
clear
set seed 20261012
set obs 1500
gen long obs = _n
gen long id = ceil(_n / 6)
gen long year = 1 + mod(_n * 5, 7)
gen long c1 = ceil(id / 5)
gen long c2 = 1 + mod(_n, 9)
gen double z1 = rnormal()
gen double z2 = rnormal()
gen double z3 = rnormal()
gen double z4 = runiform()
gen double x1 = rnormal()
gen double u = rnormal()
bys id (obs): gen double a = rnormal() if _n == 1
by id: replace a = a[1]
sort obs
gen double e1 = 0.6 * z1 + 0.3 * z2 + 0.2 * z4 + 0.4 * x1 + 0.5 * u + a + rnormal()
gen double e2 = 0.2 * z1 + 0.5 * z3 - 0.3 * z2 + 0.4 * u + rnormal()
gen double y = 1 + 0.7 * e1 - 0.4 * e2 + 0.5 * x1 + a + 0.1 * year + u + rnormal()
gen double w = 0.5 + runiform()
drop a u
mata: write_csv("iv_sim.csv", "obs id year c1 c2 z1 z2 z3 z4 x1 e1 e2 y w")

ivreghdfe y x1 (e1 e2 = z1 z2 z3 z4), absorb(id year) `t'
savefix sim_k2_unadj
ivreghdfe y x1 (e1 e2 = z1 z2 z3 z4), absorb(id year) vce(robust) `t'
savefix sim_k2_robust
ivreghdfe y x1 (e1 e2 = z1 z2 z3 z4) [aw=w], absorb(id year) vce(cluster c1) `t'
savefix sim_k2_aw_cl1
ivreghdfe y x1 (e1 e2 = z1 z2 z3 z4) [pw=w], absorb(id year) vce(cluster c1 c2) `t'
savefix sim_k2_pw_cl2
ivreghdfe y (e1 e2 = z1 z2 z3), absorb(id year) vce(robust) `t'
savefix sim_k2_noexog_robust
ivreghdfe y x1 (e1 e2 = z1 z2), absorb(id year) vce(cluster c1) `t'
savefix sim_k2_justid_cl1
ivreghdfe y x1 (e1 = z1 z2 z3), absorb(id) vce(cluster c1) `t'
savefix sim_k1_1fe_cl1
ivreghdfe y x1 (e1 = z1 z2 z3), absorb(id year) vce(cluster id) `t'
savefix sim_k1_cl_eq_fe
ivreghdfe y x1 (e1 = z1 z2 z3), absorb(id) vce(cluster id) `t'
savefix sim_k1_dfa0

exit, clear
