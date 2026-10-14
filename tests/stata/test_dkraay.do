* test_dkraay.do -- fereg vce(dkraay [#]) vs reghdfe 6.13.1 vce(dkraay [#]) (Driscoll-Kraay)
*
* Every reghdfe run uses tol(1e-12) and fereg tol(1e-12). Assertions (rf_compare_e): b and V within
* 1e-8 (V on the correlation scale), integer e() exact, other e() scalars rtol 1e-6, all e() macros
* equal (except command bookkeeping), e(dof_table) and e(sample) identical; the displays are
* identical line by line (rf_display_diff: header with "DK Bandwidth", titles, table, footnote).
* feppml/felogit (borrowed: reghdfe's formula with the asymptotic factor T/(T-1)): vce(dkraay 1)
* equals vce(cluster <time>) (G/(G-1) with G = T).
*
* Run from tests/stata (needs fereg.plugin on the adopath and reghdfe installed):
*   stata-mp -b do test_dkraay.do
version 16
clear all
set more off
capture confirm file "../../stata/fereg.ado"
if (!_rc) adopath ++ "`c(pwd)'/../../stata"
which fereg
which reghdfe
findfile fereg.plugin
di as text "plugin: `r(fn)'"

* check <arguments common to reghdfe and fereg> : estimate both, compare e() and the displays
capture program drop check
program check
	local spec `"`0'"'
	if (!strpos(`"`spec'"', ",")) local spec `"`spec',"'
	quietly reghdfe `spec' tol(1e-12)
	estimates store R
	fereg `spec' tol(1e-12)
	estimates store F
	rf_compare_e R F, xscalars(ic) btol(1e-8) vtol(1e-8)
	rf_display_diff R F
end

* ---------------------------------------------------------------------------
* nlswork panel (xtset idcode year): the years have gaps (reghdfe pairs periods by position)
* ---------------------------------------------------------------------------
webuse nlswork, clear
keep if idcode <= 2000
gen double w = 0.5 + mod(idcode * 7 + year, 10) / 10
gen int fw = 1 + mod(idcode + year, 3)
xtset idcode year
check ln_wage age ttl_exp tenure, absorb(idcode) vce(dkraay)
check ln_wage age ttl_exp tenure, absorb(idcode) vce(dkraay 1)
check ln_wage age ttl_exp tenure, absorb(idcode) vce(dkraay 2)
check ln_wage age ttl_exp tenure, absorb(idcode) vce(dk 4)
check ln_wage age ttl_exp tenure, absorb(idcode) vce(dkraay 40)
* weights
check ln_wage age ttl_exp tenure [aw=w], absorb(idcode) vce(dkraay)
check ln_wage age ttl_exp tenure [fw=fw], absorb(idcode) vce(dkraay 3)
check ln_wage age ttl_exp tenure [pw=w], absorb(idcode) vce(dkraay 3)
* absorbed FE nested in the time variable (no nested adjustment in reghdfe's DK)
check ln_wage age ttl_exp tenure, absorb(idcode year) vce(dkraay 3)
check ln_wage age ttl_exp tenure [aw=w], absorb(year) vce(dkraay)
check ln_wage age ttl_exp tenure, absorb(idcode ind_code#year) vce(dkraay 2)
* factor variables, slopes, no absorbed FE
check ln_wage age c.age#c.age i.race tenure, absorb(idcode occ_code) vce(dkraay)
check ln_wage age tenure, absorb(idcode##c.ttl_exp) vce(dkraay 2)
check ln_wage age tenure, vce(dkraay 3)
check ln_wage age tenure, noabsorb vce(dkraay 2)
* more gaps: drop three years
preserve
drop if inlist(year, 70, 77, 80)
check ln_wage age ttl_exp tenure, absorb(idcode) vce(dkraay 3)
check ln_wage age ttl_exp tenure [aw=w], absorb(idcode year) vce(dkraay)
restore
di as result "nlswork panel passed"

* time series (tsset, no panel): one observation per period
preserve
collapse (mean) ln_wage age ttl_exp tenure (first) race, by(year)
tsset year
check ln_wage age tenure, absorb(race) vce(dkraay 2)
check ln_wage age tenure, noabsorb vce(dkraay)
restore
di as result "time series passed"

* ---------------------------------------------------------------------------
* Errors: reghdfe's texts and return codes
* ---------------------------------------------------------------------------
foreach v in "dkraay 2.5" "dkraay 0" "dkraay -1" "dkraay abc" "dkraay 2 3" {
	capture reghdfe ln_wage age, absorb(idcode) vce(`v')
	local rc_r = _rc
	capture fereg ln_wage age, absorb(idcode) vce(`v')
	assert _rc == `rc_r' & _rc == 9
	di as text "ok  vce(`v') fails with rc 9 as reghdfe"
}
xtset, clear
capture reghdfe ln_wage age, absorb(idcode) vce(dkraay)
local rc_r = _rc
capture fereg ln_wage age, absorb(idcode) vce(dkraay)
assert _rc == `rc_r' & _rc == 9
foreach cmd in feppml felogit {
	capture `cmd' union age, absorb(idcode) vce(dkraay)
	assert _rc == 9
}
xtset idcode year
capture fereg ln_wage age, absorb(idcode) vce(dkraay) bw(2)
assert _rc == 198
di as result "error paths passed"

* ---------------------------------------------------------------------------
* feppml / felogit (borrowed): vce(dkraay 1) = clustering on the time variable
* ---------------------------------------------------------------------------
gen int wks = wks_work
foreach cmd in feppml felogit {
	local y = cond("`cmd'" == "feppml", "wks", "union")
	quietly `cmd' `y' age tenure, absorb(idcode) vce(cluster year)
	estimates store C
	`cmd' `y' age tenure, absorb(idcode) vce(dkraay 1)
	estimates store D
	assert "`e(vce)'" == "dkraay" & "`e(vcetype)'" == "Robust" & e(dkraay_bw) == 1
	assert e(N_clust) == e(N_clust1) & e(N_clustervars) == 1 & "`e(clustvar)'" == "year"
	rf_compare_e C D, btol(1e-10) vtol(1e-8) xmacros(vce cmdline title4) xscalars(dkraay_bw)
	`cmd' `y' age tenure [pw=w], absorb(idcode) vce(dkraay)
	assert e(dkraay_bw) == floor(4 * (e(N_clust) / 100)^(2/9)) + 1
	mata: assert(all(diagonal(st_matrix("e(V)")) :>= 0))
	`cmd' `y' age tenure, absorb(idcode) vce(dkraay 3)
}
di as result "feppml/felogit passed"

di as result "test_dkraay passed"
