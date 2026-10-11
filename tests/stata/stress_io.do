* stress_io.do — plan §4.3 parallel-read experiment and read-throughput measurements.
* Run from tests/stata after scripts/build_plugin.sh:   stata-mp -b do stress_io.do
*
* Part 1: reads a 2e7 x 8 double dataset 200 times with parallel reads (serialio=0, all
*         threads) and compares the checksum __rf_chk with the serial checksum every time.
*         Parallel reads stay the plugin default only if all 200 match without a crash.
* Part 2: single-thread read throughput (values/s) of the two session read paths on 1e7 obs:
*         rowread=direct (each row read from its observation, in FE1-sorted row order) and
*         rowread=kept (sequential read + in-memory row permutation), on data sorted by FE1
*         and on shuffled data.

version 16
clear all
set more off
set seed 20261011

capture program drop _rf_plugin
program _rf_plugin, plugin using("../../stata/regfast.plugin")

* ---------------------------------------------------------------- part 1
set obs 20000000
* a leading string variable makes varlist positions differ from dataset positions
gen str1 pad = "a"
forvalues j = 1/8 {
    gen double v`j' = rnormal() * `j'
}
replace v3 = . in 5/100
local V v1 v2 v3 v4 v5 v6 v7 v8

timer clear
timer on 1
plugin call _rf_plugin `V', selftest "serialio=1"
timer off 1
scalar chk_serial = __rf_chk
scalar t_serial = __rf_tread
di as txt "serial read: " %9.4f t_serial " s, checksum " %21.15g chk_serial

local reps 200
local fails 0
timer on 2
forvalues r = 1/`reps' {
    plugin call _rf_plugin `V', selftest "serialio=0" "threads=0"
    if scalar(__rf_chk) != scalar(chk_serial) | scalar(__rf_nobs) != _N {
        local ++fails
        di as err "rep `r': parallel checksum " %21.15g __rf_chk " != serial"
    }
    if `r' == 1 | mod(`r', 50) == 0 {
        di as txt "rep `r': parallel read " %9.4f __rf_tread " s"
    }
}
timer off 2
qui timer list 2
di as txt "parallel reads: `reps' reps, `fails' mismatches, " %9.2f r(t2) " s total"
di as result "STRESS_IO_RESULT fails=`fails' reps=`reps'"

* ---------------------------------------------------------------- part 2
clear
set obs 10000000
gen long fe1 = runiformint(1, 1000000)
forvalues j = 1/3 {
    gen double x`j' = rnormal()
}
local k = 4
foreach layout in sorted shuffled {
    if "`layout'" == "sorted" {
        sort fe1, stable
    }
    else {
        gen double u = runiform()
        sort u
        drop u
    }
    foreach mode in direct kept {
        * 3 runs, report the fastest
        local best = .
        forvalues r = 1/3 {
            plugin call _rf_plugin fe1 x1 x2 x3, selftest "sortby=1" "rowread=`mode'" "threads=1" "serialio=1"
            local best = min(`best', __rf_tread)
        }
        di as result "READ_THROUGHPUT layout=`layout' rowread=`mode' threads=1 seconds=" %7.4f `best' ///
            " values_per_s=" %10.3e (_N * `k' / `best')
    }
    * sequential parallel read for reference
    local best = .
    forvalues r = 1/3 {
        plugin call _rf_plugin fe1 x1 x2 x3, selftest "serialio=0" "threads=0"
        local best = min(`best', __rf_tread)
    }
    di as result "READ_THROUGHPUT layout=`layout' sequential threads=all seconds=" %7.4f `best' ///
        " values_per_s=" %10.3e (_N * `k' / `best')
    local best = .
    forvalues r = 1/3 {
        plugin call _rf_plugin fe1 x1 x2 x3, selftest "serialio=1"
        local best = min(`best', __rf_tread)
    }
    di as result "READ_THROUGHPUT layout=`layout' sequential threads=1 seconds=" %7.4f `best' ///
        " values_per_s=" %10.3e (_N * `k' / `best')
}
