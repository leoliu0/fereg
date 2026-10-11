* Stata benchmark runner (plan Phase 8.3). Arguments arrive as globals, set by
* the job do-file that run.py writes before calling `do run_stata.do`:
*   RF_DATA  .dta path               RF_VARS  variables to load
*   RF_CMD   estimation command      RF_T     threads (set processors)
*   RF_REPS  timed runs after the burn-in (1 if the burn-in exceeds 60 s)
*   RF_OUT   result file: one `RESULT {json}` line per run
*   RF_PRE   optional command run once after loading (e.g. a version check)
* Only the estimation call is timed (timer 1), on in-memory data, under -quietly-.

version 16
clear all
set more off
set linesize 255
set processors $RF_T

tempname fh
file open `fh' using `"$RF_OUT"', write text replace

capture noisily use $RF_VARS using `"$RF_DATA"', clear
if _rc {
    file write `fh' `"RESULT {"run":"load","ok":false,"rc":`=_rc',"error":"use failed"}"' _n
    file close `fh'
    exit
}
if `"$RF_PRE"' != "" {
    capture noisily $RF_PRE
}

local nrun = 1 + $RF_REPS
local i 0
while `i' < `nrun' {
    local phase = cond(`i' == 0, "burnin", "timed")
    timer clear 1
    timer on 1
    capture noisily quietly $RF_CMD
    local rc = _rc
    timer off 1
    quietly timer list 1
    local t = r(t1)
    local ts = strtrim(strofreal(`t', "%16.9e"))
    if `rc' {
        file write `fh' `"RESULT {"run":"`phase'","idx":`i',"ok":false,"rc":`rc',"time":`ts',"error":"r(`rc')"}"' _n
        continue, break
    }
    local coefs ""
    local nc 0
    capture matrix __b = e(b)
    if !_rc {
        local names : colnames __b
        local j 0
        foreach nm of local names {
            local ++j
            if "`nm'" == "_cons" | strpos("`nm'", "o.") == 1 | strpos("`nm'", "b.") > 0 continue
            if missing(__b[1, `j']) continue
            local s = strtrim(strofreal(__b[1, `j'], "%24.16e"))
            local sep = cond(`nc' > 0, ",", "")
            local coefs `"`coefs'`sep'"`nm'":`s'"'
            local ++nc
        }
    }
    local nobs = e(N)
    if missing(`nobs') local nobs null
    file write `fh' `"RESULT {"run":"`phase'","idx":`i',"ok":true,"time":`ts',"n_obs":`nobs',"coefs":{`coefs'}}"' _n
    if `i' == 0 & `t' > 60 local nrun 2
    local ++i
}
file close `fh'
