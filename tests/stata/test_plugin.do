* test_plugin.do -- plugin ABI/protocol tests (cmd=version, selftest, prepare, ols plumbing).
* Run from tests/stata after scripts/build_plugin.sh:   stata-mp -b do test_plugin.do
* Needs reghdfe 6.13.1 and ppmlhdfe 2.3.3 (sample/singleton parity).
* Steps that depend on estimators still being stubs in fereg-core print "PENDING:".
*
* Break key: batch mode (stata-mp -b) cannot deliver Break (SIGINT terminates a batch or
* non-terminal Stata), so it is not exercised here. Verified manually in console Stata on a
* pseudo-terminal: `plugin call _rf_plugin price, selftest "spin=30"` + Ctrl-C prints
* --Break--, returns r(1) within ~50 ms polling, and the plugin keeps working afterwards.
* tests/mockhost covers the same path on every OS by setting the stop flag (expects r(1)).

version 16
clear all
set more off
set seed 12345

capture program drop _rf_plugin
program _rf_plugin, plugin using("../../stata/fereg.plugin")

* Expected selftest checksum: sum over obs (ascending) and vars (left to right) of j*v_j, missing -> 0.
mata:
real scalar rf_chk(string scalar vars, string scalar touse)
{
    real matrix X
    real scalar c, i, j
    X = st_data(., vars, touse)
    c = 0
    for (i = 1; i <= rows(X); i++) {
        for (j = 1; j <= cols(X); j++) {
            if (X[i, j] < .) c = c + j * X[i, j]
        }
    }
    return(c)
}
end

program rf_expect_rc
    * rf_expect_rc <rc> : <command>
    gettoken want 0 : 0
    gettoken colon 0 : 0
    capture noisily `0'
    if _rc != `want' {
        di as err "expected r(`want'), got r(" _rc ") from: `0'"
        exit 9
    }
    di as txt "  ok: r(`want') as expected"
end

* ---------------------------------------------------------------- version
plugin call _rf_plugin, version
assert "`_rf_version'" == "0.4.0"
di as txt "version: `_rf_version'"

* ---------------------------------------------------------------- selftest
sysuse auto, clear
gen byte all = 1
plugin call _rf_plugin price mpg rep78 weight, selftest
assert scalar(__rf_nobs) == 74
mata: st_numscalar("e_chk", rf_chk("price mpg rep78 weight", "all"))
assert scalar(__rf_chk) == scalar(e_chk)
* parallel read path, if/in
gen byte sel = foreign == 1 in 10/70
replace sel = 0 if missing(sel)
plugin call _rf_plugin price mpg rep78 weight if foreign in 10/70, selftest "serialio=0" "threads=4"
mata: st_numscalar("e_chk", rf_chk("price mpg rep78 weight", "sel"))
qui count if sel
assert scalar(__rf_nobs) == r(N)
assert scalar(__rf_chk) == scalar(e_chk)
* write=j stores the observation number
gen double wout = .
plugin call _rf_plugin price wout if foreign, selftest "write=2"
assert wout == _n if foreign
assert missing(wout) if !foreign
* sortby/rowread paths give the same checksum
plugin call _rf_plugin price mpg weight, selftest "sortby=2" "rowread=direct"
scalar c1 = __rf_chk
plugin call _rf_plugin price mpg weight, selftest "sortby=2" "rowread=kept"
assert scalar(__rf_chk) == scalar(c1)
di as txt "selftest: ok"

* ---------------------------------------------------------------- prepare vs reghdfe
program rf_prep_vs_reghdfe
    * rf_prep_vs_reghdfe "<reghdfe cmd>" "<plugin varlist>" "<plugin keys>"
    args rcmd pvars pkeys
    qui `rcmd'
    capture drop __e
    gen byte __e = e(sample)
    * e(N) is the sum of weights with fweights: compare observation counts
    qui count if __e
    local eN = r(N)
    local es = e(num_singletons)
    capture drop touse
    * auto has no missing values in the variables used here
    gen byte touse = 1
    plugin call _rf_plugin `pvars' touse, prepare `pkeys'
    assert "`_rf_token'" != ""
    assert scalar(__rf_N) == `eN'
    assert scalar(__rf_num_singletons) == `es'
    assert scalar(__rf_N_full) == scalar(__rf_N) + scalar(__rf_num_singletons)
    assert touse == __e
    di as txt "  prepare == reghdfe: N=`eN' singletons=`es' token=`_rf_token'"
end

sysuse auto, clear
* turn and trunk generate singletons in auto
rf_prep_vs_reghdfe "reghdfe price weight, absorb(turn trunk)" "turn trunk" `""nfac=2" "factype=nn" "absorb=1|1|-|0;2|1|-|0" "absorber=dense""'
* string factor
tostring turn, gen(sturn)
rf_prep_vs_reghdfe "reghdfe price weight, absorb(sturn trunk)" "sturn trunk" `""nfac=2" "factype=sn" "absorb=1|1|-|0;2|1|-|0" "absorber=dense""'
* interaction term
rf_prep_vs_reghdfe "reghdfe price weight, absorb(turn#foreign trunk)" "turn foreign trunk" `""nfac=3" "factype=nnn" "absorb=1*2|1|-|0;3|1|-|0" "absorber=dense""'
* slopes (turn##c.length)
rf_prep_vs_reghdfe "reghdfe price weight, absorb(turn##c.length)" "turn length" `""nfac=1" "factype=n" "nslope=1" "absorb=1|1|1|0" "absorber=dense""'
* frequency weights
gen int fw = 1 + mod(_n, 3)
rf_prep_vs_reghdfe "reghdfe price weight [fw=fw], absorb(turn trunk)" "turn trunk fw" `""nfac=2" "factype=nn" "absorb=1|1|-|0;2|1|-|0" "wtype=fw" "absorber=dense""'
* clustered (string cluster variable)
rf_prep_vs_reghdfe "reghdfe price weight, absorb(turn trunk) vce(cluster make)" "turn trunk make" `""nfac=3" "factype=nns" "absorb=1|1|-|0;2|1|-|0" "vce=cluster" "cluster=3" "absorber=dense""'
* keepsingletons
rf_prep_vs_reghdfe "reghdfe price weight, absorb(turn trunk) keepsingletons" "turn trunk" `""nfac=2" "factype=nn" "absorb=1|1|-|0;2|1|-|0" "singletons=0" "absorber=dense""'
* parallel reads give the same sample
rf_prep_vs_reghdfe "reghdfe price weight, absorb(turn trunk)" "turn trunk" `""nfac=2" "factype=nn" "absorb=1|1|-|0;2|1|-|0" "absorber=dense" "serialio=0" "threads=8""'
* if/in restrict the sample on top of touse
replace touse = 1
plugin call _rf_plugin turn trunk touse in 1/50, prepare "nfac=2" "factype=nn" "absorb=1|1|-|0;2|1|-|0" "absorber=dense"
qui reghdfe price weight in 1/50, absorb(turn trunk)
assert scalar(__rf_N) == e(N)
assert touse == e(sample) in 1/50
assert touse == 1 in 51/74
di as txt "prepare vs reghdfe: ok"

* ---------------------------------------------------------------- GLM fe separation
sysuse auto, clear
* Poisson: rep78 groups 1 and 2 have foreign == 0 throughout
qui ppmlhdfe foreign weight, absorb(rep78) separation(fe)
gen byte es = e(sample)
local eN = e(N)
gen byte touse = !missing(rep78)
plugin call _rf_plugin rep78 foreign touse, prepare "nfac=1" "factype=n" "absorb=1|1|-|0" "family=poisson" "sep=fe" "absorber=dense"
assert scalar(__rf_N) == `eN'
assert touse == es
di as txt "  poisson fe separation: N=`eN' fe_separated=" scalar(__rf_num_fe_separated)
* logit: groups with all-0 or all-1 outcomes are dropped
replace touse = !missing(rep78)
bysort rep78: egen double m = mean(foreign)
count if !missing(rep78) & m > 0 & m < 1
local lN = r(N)
plugin call _rf_plugin rep78 foreign touse, prepare "nfac=1" "factype=n" "absorb=1|1|-|0" "family=logit" "sep=fe" "absorber=dense"
assert scalar(__rf_N) == `lN'
assert touse == (m > 0 & m < 1 & !missing(rep78))
di as txt "GLM fe separation: ok"

* ---------------------------------------------------------------- error paths
sysuse auto, clear
gen byte touse = 1
local P plugin call _rf_plugin
rf_expect_rc 198 : `P' turn touse, prepare "nfac=1" "factype=n" "absorb=1|1|-|0" "bogus=1"
rf_expect_rc 198 : `P' turn touse, prepare "factype=n" "absorb=1|1|-|0"
rf_expect_rc 198 : `P' turn trunk touse, prepare "nfac=1" "factype=n" "absorb=1|1|-|0"
rf_expect_rc 198 : `P' make touse, prepare "nfac=1" "factype=n" "absorb=1|1|-|0"
rf_expect_rc 198 : `P' turn touse, prepare "nfac=1" "factype=n" "absorb=2|1|-|0"
rf_expect_rc 198 : `P' turn touse, prepare "nfac=1" "factype=n" "absorb=1|1|-"
rf_expect_rc 198 : `P' rep78 touse, prepare "nfac=1" "factype=n" "absorb=1|1|-|0" "absorber=dense"
rf_expect_rc 198 : `P' turn touse, prepare "nfac=1" "factype=n" "absorb=1|1|-|0" "absorber=slow"
rf_expect_rc 198 : `P' turn touse, prepare "nfac=1" "factype=n" "absorb=1|1|-|0" "vce=cluster"
rf_expect_rc 2001 : `P' turn touse in 1, prepare "nfac=1" "factype=n" "absorb=none" "absorber=dense"
gen double negw = cond(_n == 3, -1, 1)
rf_expect_rc 198 : `P' turn negw touse, prepare "nfac=1" "factype=n" "absorb=1|1|-|0" "wtype=aw" "absorber=dense"
rf_expect_rc 198 : `P', nosuchcmd
rf_expect_rc 198 : `P' price weight touse, ols "token=1" "nx=1"
* a valid prepare, then a wrong token: the session is consumed either way
`P' turn trunk touse, prepare "nfac=2" "factype=nn" "absorb=1|1|-|0;2|1|-|0" "absorber=dense"
local tok `_rf_token'
rf_expect_rc 198 : `P' price weight touse, ols "token=`tok'1" "nx=1"
rf_expect_rc 198 : `P' price weight touse, ols "token=`tok'" "nx=1"
di as txt "error paths: ok"

* ---------------------------------------------------------------- ols plumbing
sysuse auto, clear
gen byte touse = 1
plugin call _rf_plugin turn trunk touse, prepare "nfac=2" "factype=nn" "absorb=1|1|-|0;2|1|-|0" "absorber=dense"
matrix b = J(1, 3, .)
matrix V = J(3, 3, .)
gen double res = .
capture noisily plugin call _rf_plugin price weight length touse res, ols "token=`_rf_token'" "nx=2" "nout=1" "resid=1" "b=b" "V=V" "tol=1e-12"
if _rc == 9999 {
    di as txt "PENDING: cmd=ols (Session::ols is still a stub in fereg-core)"
}
else {
    assert _rc == 0
    matrix list b
    assert !missing(b[1, 1])
    assert !missing(res) if touse
    assert missing(res) if !touse
    * wrong matrix dimensions -> r(503)
    plugin call _rf_plugin turn trunk touse, prepare "nfac=2" "factype=nn" "absorb=1|1|-|0;2|1|-|0" "absorber=dense"
    matrix b = J(1, 2, .)
    rf_expect_rc 503 : plugin call _rf_plugin price weight length touse, ols "token=`_rf_token'" "nx=2" "b=b" "V=V"
}

* ---------------------------------------------------------------- 1e7 observations
clear
set obs 10000000
gen long id = runiformint(1, 1000000)
gen str8 sid = string(id, "%08.0f")
* one singleton level: absorb(a) drops observation 1; the absorbed space stays small (dense absorber)
gen byte a = _n == 1
gen double x = rnormal()
gen byte touse = 1
timer clear 1
timer on 1
plugin call _rf_plugin a sid touse, prepare "nfac=2" "factype=ns" "absorb=1|1|-|0" "vce=cluster" "cluster=2" "absorber=dense" "verbose=2"
timer off 1
assert scalar(__rf_N_full) == 1e7
assert scalar(__rf_N) == 1e7 - 1
assert scalar(__rf_num_singletons) == 1
assert touse == (_n != 1)
qui timer list 1
di as txt "1e7 prepare (string cluster factor, 1e6 levels): " %6.2f r(t1) " s, token `_rf_token'"
* the Fast absorber with large factors (needs CoreSolver's FeSolver)
replace touse = 1
capture noisily plugin call _rf_plugin id touse, prepare "nfac=1" "factype=n" "absorb=1|1|-|0" "verbose=2"
if _rc == 9999 {
    di as txt "PENDING: prepare with the fast absorber (solve::build is still a stub in fereg-core)"
}
else {
    assert _rc == 0
    bysort id: gen long cnt = _N
    count if cnt == 1
    assert scalar(__rf_num_singletons) == r(N)
    assert touse == (cnt > 1)
}

di as result "TEST_PLUGIN PASSED"
