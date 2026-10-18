* Reference fixtures for the fixed-effects logit bias correction / APEs / jackknife tests of
* crates/fereg-core (part 1 of 2): simulated data + live logitfe 3.0.3 (SSC) runs.
*
* Run from this directory (crates/fereg-core/tests/data/bias):
*     LOGITFE_PLUS=/path/to/plus/ stata-mp -b do gen_logitfe.do
*     Rscript gen_logitfe_indep.R
* LOGITFE_PLUS (optional environment variable) = sysdir PLUS that holds logitfe 3.0.3 (SSC), e.g.
* /tmp/bc/st/plus/. moremata is not needed (no multiple() runs: the multiple-partition fixtures
* use explicit permutations, computed in gen_logitfe_indep.R).
*
* Writes
*   l_static.csv   300 individuals x 8 periods, columns id,t,y,d,x1,x2,x3 (d binary)
*   l_dynamic.csv  300 x 8 with state dependence, columns id,t,y,ylag,d,x1 (ylag = L.y, empty in t = 1)
*   l_unbal.csv    l_static minus ~12% random rows (T_i 4-8, gaps), plus ylag = L.y of that y
*                  (calendar lag, empty when t-1 is missing)
*   l_trade.csv    30 countries, importer id x exporter jd (870 rows, no diagonal), y,b1,b2,x
*                  (b1, b2 binary); tsset id jd
*   logitfe_stata.csv  one row per run x coefficient: tag,var,b,ape,N,N_group  (b = e(b), ape = e(b2))
* CSV conventions: header row, comma separated, numeric only, NA = empty field. Regressors are
* rounded to 1e-6 and written with 6 decimals, so every consumer (this file re-imports the CSVs
* before estimating, R, Rust) sees the same doubles.
*
* Sample rules: logitfe drops constant-y individuals, then constant-y periods, once. The data are
* checked below so that this one-pass drop equals fereg's iterated drop (no period is constant).

version 16
clear all
set more off
local plus : env LOGITFE_PLUS
if `"`plus'"' != "" sysdir set PLUS `"`plus'"'

* ---- pin logitfe 3.0.3 ---------------------------------------------------------------------
quietly findfile logitfe.ado
tempname fh
file open `fh' using `"`r(fn)'"', read text
local found 0
file read `fh' line
while r(eof) == 0 & !`found' {
    if strpos(`"`line'"', "*! logitfe v3.0.3") == 1 local found 1
    file read `fh' line
}
file close `fh'
if !`found' {
    di as err "logitfe v3.0.3 required"
    exit 9
}


capture program drop writecsv
program define writecsv
    * writecsv file intvars : realvars   (reals with 6 decimals, missing = empty field)
    gettoken fname 0 : 0
    gettoken ints 0 : 0, parse(":")
    gettoken colon reals : 0, parse(":")
    local all `ints' `reals'
    local all : list retokenize all
    tempname fh
    file open `fh' using `"`fname'"', write replace
    local hdr : subinstr local all " " ",", all
    file write `fh' "`hdr'" _n
    local n = _N
    forvalues i = 1/`n' {
        local line ""
        local sep ""
        foreach v of local ints {
            if !missing(`v'[`i']) local line `"`line'`sep'`=strtrim(string(`v'[`i'], "%12.0f"))'"'
            else local line `"`line'`sep'"'
            local sep ","
        }
        foreach v of local reals {
            if !missing(`v'[`i']) local line `"`line'`sep'`=strtrim(string(`v'[`i'], "%12.6f"))'"'
            else local line `"`line'`sep'"'
            local sep ","
        }
        file write `fh' `"`line'"' _n
    }
    file close `fh'
end

capture program drop checkdrop
program define checkdrop, sortpreserve
    * one-pass drop (constant-y individuals, then periods) must equal the iterated drop
    syntax varlist(min=3 max=3) [if]
    tokenize `varlist'
    marksample touse, novarlist
    tempvar sd1 sd2 keep
    quietly {
        bysort `touse' `1': egen double `sd1' = sd(`3') if `touse'
        gen byte `keep' = `touse' & `sd1' > 0 & !missing(`sd1')
        bysort `keep' `2': egen double `sd2' = sd(`3') if `keep'
        count if `keep' & (`sd2' == 0 | missing(`sd2'))
    }
    if r(N) > 0 {
        di as err "a period becomes constant after the individual drop"
        exit 9
    }
end

* ---- static panel --------------------------------------------------------------------------
* Seed 4: chosen (among 1-12) so that logitfe's Newton-Raphson stops within ~1e-11 of the MLE on
* this panel and its jackknife subpanels; with most seeds it stops ~1e-7 away, which the
* delete-one jackknife combinations amplify ~300-fold (validation in gen_logitfe_indep.R).
set seed 4
set sortseed 4
clear
set obs 8
gen t = _n
gen double g = rnormal(0, 0.5)
tempfile G
save `G'
clear
set obs 300
gen id = _n
gen double a = rnormal(0, 1.5)
expand 8
bysort id: gen t = _n
merge m:1 t using `G', nogenerate
sort id t
gen double x1 = round(rnormal() + 0.5 * a, 1e-6)
gen double x2 = round(rnormal() + g, 1e-6)
gen double x3 = round(rnormal(), 1e-6)
gen byte d = runiform() < 0.35 + 0.1 * (a > 0)
gen double e = rlogistic()
gen byte y = (x1 - 0.5 * x2 + 0.25 * x3 + 0.5 * d + a + g + e > 0)
checkdrop id t y
writecsv "l_static.csv" id t y d : x1 x2 x3
* unbalanced version: ~12% of the rows deleted at random, T_i in 4..8
gen double u = runiform()
preserve
drop if u < 0.12
by id: gen Ti = _N
quietly summarize Ti
assert r(min) >= 4
by id: gen byte ylag = y[_n - 1] if t == t[_n - 1] + 1
checkdrop id t y
checkdrop id t y if !missing(ylag)
writecsv "l_unbal.csv" id t y d ylag : x1 x2 x3
restore

* ---- dynamic panel -------------------------------------------------------------------------
set seed 20261018
set sortseed 20261018
clear
set obs 8
gen t = _n
gen double g = rnormal(0, 0.5)
save `G', replace
clear
set obs 300
gen id = _n
gen double a = rnormal(0, 1.5)
expand 8
bysort id: gen t = _n
merge m:1 t using `G', nogenerate
sort id t
gen double x1 = round(rnormal() + 0.5 * a, 1e-6)
gen byte d = runiform() < 0.35 + 0.1 * (a > 0)
gen double e = rlogistic()
gen byte y = (x1 + 0.5 * d + a + g + e > 0) if t == 1
forvalues s = 2/8 {
    by id: replace y = (0.75 * y[_n - 1] + x1 + 0.5 * d + a + g + e > 0) if t == `s'
}
by id: gen byte ylag = y[_n - 1] if t == t[_n - 1] + 1
checkdrop id t y if !missing(ylag)
writecsv "l_dynamic.csv" id t y ylag d : x1

* ---- trade network: 30 countries, importer id x exporter jd -------------------------------
set seed 20261019
set sortseed 20261019
clear
set obs 30
gen id = _n
gen double cx = runiform()
gen double cy = runiform()
gen double a = rnormal(0, 1)
gen double gj = rnormal(0, 1)
tempfile C
save `C'
rename (id cx cy a gj) (jd cxj cyj aj g)
cross using `C'
drop aj
drop if id == jd
sort id jd
gen double x = round(log(sqrt((cx - cxj)^2 + (cy - cyj)^2)), 1e-6)
* b1: symmetric (contiguity-like), b2: directed
gen double ub = runiform()
gen lo = min(id, jd)
gen hi = max(id, jd)
bysort lo hi (id): gen byte b1 = ub[1] < 0.3
sort id jd
gen byte b2 = runiform() < 0.4
gen double e = rlogistic()
quietly summarize x
gen byte y = (-1.2 * (x - r(mean)) + 0.6 * b1 + 0.5 * b2 + a + g + e > 0)
checkdrop id jd y
bysort id: egen double sdi = sd(y)
bysort jd: egen double sdj = sd(y)
assert sdi > 0 & sdj > 0
sort id jd
writecsv "l_trade.csv" id jd y b1 b2 : x

* ---- logitfe runs --------------------------------------------------------------------------
capture program drop lrun
program define lrun
    * lrun tag : command   -- appends tag,var,b,ape,N,N_group rows to file handle `out'
    gettoken tag 0 : 0
    gettoken colon 0 : 0
    timer clear 1
    timer on 1
    quietly `0'
    timer off 1
    quietly timer list 1
    di as res "RUN `tag' (" %6.2f r(t1) " s): `0'"
    tempname b b2
    matrix `b' = e(b)
    matrix `b2' = e(b2)
    local names : colnames `b'
    local k = colsof(`b')
    forvalues j = 1/`k' {
        local nm : word `j' of `names'
        file write out "`tag',`nm'," %21.0g (`b'[1, `j']) "," %21.0g (`b2'[1, `j']) ","
        file write out %12.0f (e(N)) "," %12.0f (e(N_group)) _n
    }
end

timer clear 9
timer on 9
file open out using "logitfe_stata.csv", write replace
file write out "tag,var,b,ape,N,N_group" _n

* static
import delimited using "l_static.csv", clear asdouble
tsset id t
local S y x1 x2 x3 d
lrun s_noc        : logitfe `S', noc
lrun s_noc_ionly  : logitfe `S', noc teffects(no)
lrun s_noc_tonly  : logitfe `S', noc ieffects(no)
lrun s_an_l0      : logitfe `S', an lags(0)
lrun s_an_l1      : logitfe `S', an lags(1)
lrun s_an_l2      : logitfe `S', an lags(2)
lrun s_an_ibno    : logitfe `S', an ibias(no)
lrun s_an_tbno    : logitfe `S', an tbias(no)
lrun s_an_ionly   : logitfe `S', an teffects(no)
lrun s_an_ionly_l1 : logitfe `S', an teffects(no) lags(1)
lrun s_an_tonly   : logitfe `S', an ieffects(no)
lrun s_jk_ss2     : logitfe `S', jack ss2
lrun s_jk_ss2_ibno : logitfe `S', jack ss2 ibias(no)
lrun s_jk_ss1     : logitfe `S', jack ss1
lrun s_jk_ss1_ibno : logitfe `S', jack ss1 ibias(no)
lrun s_jk_ss1_tbno : logitfe `S', jack ss1 tbias(no)
lrun s_jk_sj      : logitfe `S', jack sj
lrun s_jk_sj_ibno : logitfe `S', jack sj ibias(no)
lrun s_jk_sj_tbno : logitfe `S', jack sj tbias(no)
lrun s_jk_js_ibno : logitfe `S', jack js ibias(no)
lrun s_jk_jj_ibno : logitfe `S', jack jj ibias(no)
lrun s_jk_jj      : logitfe `S', jack jj
* logitfe's (dimension-swapped) numbers for the independent fixtures; js tbias(no) is
* code-identical to jj tbias(no) in logitfe (ado:1497-1530 vs 1903-1935), so it is not run.
lrun s_jk_js      : logitfe `S', jack js
lrun s_jk_jj_tbno : logitfe `S', jack jj tbias(no)

* dynamic
import delimited using "l_dynamic.csv", clear asdouble
tsset id t
local D y ylag x1 d
lrun d_noc        : logitfe `D', noc
lrun d_an_l1      : logitfe `D', an lags(1)
lrun d_an_l1_ibno : logitfe `D', an lags(1) ibias(no)
lrun d_an_l1_tbno : logitfe `D', an lags(1) tbias(no)

* unbalanced
import delimited using "l_unbal.csv", clear asdouble
tsset id t
lrun u_noc        : logitfe `S', noc
lrun u_noc_ionly  : logitfe `S', noc teffects(no)
lrun u_an_l0      : logitfe `S', an
lrun u_an_l1      : logitfe `S', an lags(1)
lrun u_an_l2      : logitfe `S', an lags(2)
lrun u_an_ibno    : logitfe `S', an ibias(no)
lrun u_an_tbno    : logitfe `S', an tbias(no)
lrun u_an_ionly   : logitfe `S', an teffects(no)
lrun u_jk_ss2     : logitfe `S', jack ss2
lrun u_jk_ss1     : logitfe `S', jack ss1
local UD y ylag x1 x2 x3 d
lrun ud_noc       : logitfe `UD', noc
lrun ud_an_l1     : logitfe `UD', an lags(1)

* trade
import delimited using "l_trade.csv", clear asdouble
tsset id jd
local T y x b1 b2
lrun tr_noc            : logitfe `T', noc
lrun tr_an             : logitfe `T', an
lrun tr_jk_double      : logitfe `T', jack double
lrun tr_jk_double_ibno : logitfe `T', jack double ibias(no)
lrun tr_jk_double_tbno : logitfe `T', jack double tbias(no)

file close out
timer off 9
quietly timer list 9
di as res "TOTAL logitfe seconds: " r(t9)
