* Ground-truth fixtures for crates/fereg-core/tests/ols.rs (part B).
* Requires reghdfe 6.13.1 + ftools. Run from this directory:
*     stata-mp -b do gen_ols_fixtures.do
* Writes ols_data.csv (the simulated dataset, full precision) and one ols_<spec>.json per spec
* with e(b), e(V) (row-major), e(dof_table), every numeric e() scalar, e(sample) and, for some
* specs, the residuals, d and the mobility group variable.

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
    fput(fh, `"  "colnames": ""' + invtokens(st_matrixcolstripe("e(b)")[., 2]') + `"","')
    fput(fh, `"  "b": "' + vec2json(st_matrix("e(b)")) + ",")
    V = st_matrix("e(V)")
    fput(fh, `"  "V": "' + vec2json(vec(V')') + ",")
    V = st_matrix("e(dof_table)")
    fput(fh, `"  "dof_table": "' + vec2json(vec(V')') + ",")
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
    mata: write_json("ols_`spec'.json", "`spec'", "`extras'")
    drop __esample
    di as txt "saved ols_`spec'.json"
end

* ---------------------------------------------------------------------------------------------
* Simulated panel: 150 ids × 10 obs + 20 singleton ids; ids 1-75 only meet firms 1-20 and
* ids 76-150 only firms 21-40, so (id, firm) has two mobility groups.
set seed 20261011
set obs 1520
gen long obs = _n
gen long id = ceil(_n / 10)
replace id = 150 + (_n - 1500) if _n > 1500
gen long year = 1 + mod(_n * 7, 8)
gen long firm = 1 + floor(runiform() * 20) + 20 * (id > 75)
gen long region = ceil(id / 15)
gen long c1 = 1 + mod(_n, 6)
gen long c2 = 1 + floor(runiform() * 5)
gen double x1 = rnormal()
gen double x2 = rnormal()
gen double x3 = rnormal() + 0.3 * x1
gen double x4 = runiform() * 3
gen double xa = mod(id, 5) + year / 4
gen double xc = x1 + 2 * x2
gen double w = 0.5 + runiform()
gen double fw = 1 + floor(runiform() * 4)
bys id (obs): gen double a_id = rnormal() if _n == 1
by id: replace a_id = a_id[1]
bys firm (obs): gen double a_firm = rnormal() if _n == 1
by firm: replace a_firm = a_firm[1]
bys firm (obs): gen double s_firm = rnormal() if _n == 1
by firm: replace s_firm = s_firm[1]
sort obs
gen double y = x1 - 0.5 * x2 + 0.3 * x3 + a_id + 0.2 * year + a_firm + s_firm * x4 + rnormal()
gen double y2 = mod(id, 7) * 1.5 + 0.3 * year
drop a_id a_firm s_firm

mata: write_csv("ols_data.csv", "obs id year firm region c1 c2 x1 x2 x3 x4 xa xc w fw y y2")

* ---------------------------------------------------------------------------------------------
loc t tol(1e-14) iterate(100000)

reghdfe y x1 x2 x3, absorb(id year) `t' resid(r01)
predict double d01, d
savefix s01_unadj, extras(r01 d01)

reghdfe y x1 x2 x3, absorb(id year firm) vce(robust) `t'
savefix s02_robust_3fe

reghdfe y x1 x2, absorb(id year) vce(cluster id) `t'
savefix s03_cluster_eq_fe

reghdfe y x1 x2 x3, absorb(id year) vce(cluster region) `t'
savefix s04_fe_nested_in_cluster

reghdfe y x1 x2 x3, absorb(firm year) vce(cluster c1 c2) `t'
savefix s05_twoway_psd

reghdfe y x1 x2 x3 [aw=w], absorb(id year) vce(cluster firm) `t' resid(r06)
predict double d06, d
savefix s06_aw_cluster, extras(r06 d06)

reghdfe y x1 x2 x3 [fw=fw], absorb(id year) vce(robust) `t'
savefix s07_fw_robust

reghdfe y x1 x2 x3 [pw=w], absorb(id year firm) `t'
savefix s08_pw

reghdfe y x1 x2 xa xc, absorb(id year) `t'
savefix s09_absorbed_collinear

reghdfe y x1 x2, absorb(id firm#year) `t'
savefix s10_interaction

reghdfe y x1 x2 x3, noabsorb vce(cluster firm) `t'
savefix s11_noabsorb_cluster

reghdfe y x1 x2, absorb(id year) noconstant `t'
savefix s12_noconstant

reghdfe y x1 x2, absorb(id year) keepsingletons vce(cluster id year) `t'
savefix s13_keepsing_twoway

reghdfe y x1 x2, absorb(id firm year) groupvar(gv14) `t'
savefix s14_groupvar, extras(gv14)

reghdfe y x1 x2, absorb(id year firm) dofadjustments(firstpair) vce(cluster firm) `t'
savefix s15_firstpair_cluster

reghdfe y x1 x2, absorb(id firm##c.x4) `t'
savefix s16_slopes

reghdfe y2 x1 x2, absorb(id year) `t'
savefix s17_y_absorbed

reghdfe y x1 x2 x3 [fw=fw], absorb(firm year) vce(cluster c1 c2) `t'
savefix s18_fw_twoway

exit, clear
