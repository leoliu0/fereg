* test_parse.do -- standalone tests of the fereg ado parsers (no plugin needed)
* References: ms_parse_absvars / ms_parse_vce (ftools 2.50, used by reghdfe 6.13.1) and reghdfe's own e().
* Run from tests/stata:  stata-mp -b do test_parse.do
version 16
clear all
set more off
capture confirm file "../../stata/fereg.ado"
if (!_rc) adopath ++ "`c(pwd)'/../../stata"
which _rf_parse_absorb
which ms_parse_absvars

webuse nlswork, clear
gen sid = string(idcode)
gen syr = string(year)
gen double w = 1 + mod(_n, 3)

* ---------------------------------------------------------------------------
* 1) _rf_parse_absorb vs ms_parse_absvars
* ---------------------------------------------------------------------------
capture program drop check_absorb
program check_absorb
	local spec `"`0'"'
	* Reference
	ms_parse_absvars `spec'
	local G `s(G)'
	local R_absvars
	foreach lab in `s(absvars)' {
		local R_absvars `R_absvars' `lab'
	}
	local R_stripe `s(extended_absvars)'
	local R_ext : subinstr local R_stripe "1." "", all
	local R_ivars `"`s(ivars)'"'
	local R_cvars `"`s(cvars)'"'
	local R_targets `"`s(targets)'"'
	local R_int `s(intercepts)'
	local R_ns `s(num_slopes)'
	local R_base `s(basevars)'
	local R_hi `s(has_intercept)'
	local R_any `s(save_any_fe)'
	local R_all `s(save_all_fe)'
	* fereg
	_rf_parse_absorb `spec'
	local fail 0
	if ("`s(G)'" != "`G'") local fail "G: `s(G)' vs `G'"
	if (`"`s(absvars)'"' != `"`R_absvars'"') local fail `"absvars: <`s(absvars)'> vs <`R_absvars'>"'
	if (`"`s(extended_absvars)'"' != `"`R_ext'"') local fail `"extended: <`s(extended_absvars)'> vs <`R_ext'>"'
	if (`"`s(dof_rownames)'"' != `"`R_stripe'"') local fail `"stripe: <`s(dof_rownames)'> vs <`R_stripe'>"'
	forval g = 1/`G' {
		foreach k in ivars cvars targets {
			local a : word `g' of `s(`k')'
			local b : word `g' of `R_`k''
			local a `a'
			local b `b'
			if ("`a'" != "`b'") local fail "`k'[`g']: <`a'> vs <`b'>"
		}
	}
	local a `s(intercepts)'
	local b `R_int'
	if ("`a'" != "`b'") local fail "intercepts: <`a'> vs <`b'>"
	local a `s(num_slopes)'
	local b `R_ns'
	if ("`a'" != "`b'") local fail "num_slopes: <`a'> vs <`b'>"
	local a `s(basevars)'
	if (!`: list a === R_base') local fail "basevars: <`a'> vs <`R_base'>"
	if ("`s(has_intercept)'" != "`R_hi'") local fail "has_intercept"
	if ("`s(save_any_fe)'" != "`R_any'") local fail "save_any_fe"
	if ("`s(save_all_fe)'" != "`R_all'") local fail "save_all_fe"
	local nslots : word count `s(extended_absvars)'
	if ("`s(num_slots)'" != "`nslots'") local fail "num_slots"
	if (`"`fail'"' != "0") {
		di as err `"FAIL absorb(`spec'): `fail'"'
		exit 9
	}
	di as text `"ok  absorb(`spec')  ->  `s(absvars)'  |  `s(extended_absvars)'  |  `s(dof_rownames)'"'
end

foreach spec in ///
	"idcode" "i.idcode" "idc" "idcode year" "idcode year occ_code" ///
	"idcode#year" "i.idcode#i.year" "idcode year#occ_code" "ind_code#year#race" ///
	"ib2.year idcode" "ibn.year" ///
	"ind_code#c.tenure" "ind_code##c.tenure" "ind_code##c.(tenure hours)" "ind_code#c.(tenure hours)" ///
	"idcode#year##c.age" "idcode year##c.age ind_code#c.(tenure hours)" "c.age#ind_code" ///
	"ind_code#c.age year#c.tenure" "year##c.(age ttl_exp tenure)" ///
	"fe1=idcode fe2=year" "fe1 = idcode year" "fe1= idcode year" ///
	"idcode year, savefe" "idcode year, generate" "fe1=idcode##c.age year, savefe" ///
	"fe1=idcode##c.(age tenure) fe2=ind_code#c.hours" ///
	"sid" "sid year" "fe1=sid year##c.age" "idcode   year" ///
	"ind_code##year" "i.ind_code##i.year" "idcode ind_code##year" "ind_code##year##race" ///
	"ind_code##year#race" "fe1=ind_code##i.year idcode" "c.age##ind_code" ///
	"ind_code##year, savefe" {
	check_absorb `spec'
}

* Constant-only
_rf_parse_absorb
assert "`s(G)'" == "1" & "`s(absvars)'" == "_cons" & "`s(extended_absvars)'" == "_cons" & "`s(has_intercept)'" == "1"
assert "`s(num_slots)'" == "1" & "`s(factors)'" == "" & "`s(dof_rownames)'" == "_cons"

* Factor/slope lists used to build the plugin keys
_rf_parse_absorb fe1=idcode##c.(age tenure) year ind_code#year#c.age, savefe
assert "`s(factors)'" == "idcode year ind_code"
assert "`s(slopevars)'" == "age tenure"
local t `"`s(targets)'"'
local t1 : word 1 of `t'
local t2 : word 2 of `t'
local t3 : word 3 of `t'
assert "`t1'" == "fe1 fe1Slope1 fe1Slope2" & "`t2'" == "__hdfe2__" & "`t3'" == "__hdfe3__Slope1"
local lab : word 5 of `s(fe_labels)'
assert `"`lab'"' == "[FE] 1.ind_code#1.year#c.age"
assert "`s(num_slots)'" == "5"
di as text "ok  factor/slope/target lists"

* fereg extensions (reghdfe rejects strings inside interactions)
_rf_parse_absorb sid#year i.syr
assert "`s(absvars)'" == "sid#year i.syr" & "`s(extended_absvars)'" == "sid#year syr" & "`s(stringvars)'" == "sid syr"
assert "`s(dof_rownames)'" == "1.sid#1.year 1.syr"
_rf_parse_absorb sid#syr##c.age
assert "`s(absvars)'" == "sid#syr##c.age" & "`s(extended_absvars)'" == "sid#syr sid#syr#c.age"
di as text "ok  string factors inside interactions"

* Time-series operators are returned unexpanded (resolved by fereg with tsrevar)
xtset idcode year
_rf_parse_absorb idcode#c.L.tenure
assert "`s(slopevars)'" == "L.tenure" & "`s(absvars)'" == "idcode#c.L.tenure"
di as text "ok  time-series operators"

* Errors (the ## cases are rejected by ms_parse_absvars too: they expand to duplicated slopes)
foreach spec in "c.age" "ind_code#c.age#c.age" "ind_code##c.age##c.tenure" "idcode, bogus" ///
	"ind_code##year##c.age" "ind_code#c.age##year" "ind_code##year#c.age" "c.age##c.tenure" ///
	"idcode###year" "idcode##" "i.(idcode year)" "nosuchvar" "fe1=idcode fe1=year" "c.sid#year" {
	capture _rf_parse_absorb `spec'
	if (!_rc) {
		di as err `"FAIL absorb(`spec') should have failed"'
		exit 9
	}
	di as text `"ok  absorb(`spec') fails with rc `=_rc'"'
}
foreach spec in "ind_code##c.age##c.tenure" "ind_code##year##c.age" "ind_code#c.age##year" ///
	"ind_code##year#c.age" "c.age##c.tenure" {
	capture ms_parse_absvars `spec'
	assert _rc
}
di as text "ok  ms_parse_absvars rejects the same ## terms"
* target exists: error 110, unless replace
gen fe1 = 1
capture _rf_parse_absorb fe1=idcode
assert _rc == 110
_rf_parse_absorb fe1=idcode, replace
capture confirm variable fe1
assert _rc
gen __hdfe1__ = 1
_rf_parse_absorb idcode year, savefe replace
capture confirm variable __hdfe1__
assert _rc
di as text "ok  targets and replace"

* ---------------------------------------------------------------------------
* 2) e(absvars), e(extended_absvars) and e(dof_table) rownames from reghdfe on real data
* ---------------------------------------------------------------------------
foreach spec in "idcode year" "idcode year##c.age" "ind_code#c.(tenure hours) idcode" ///
	"fe1=sid year##c.age ind_code#c.(tenure hours), savefe" "ind_code#year occ_code" "i.idc i.year" ///
	"ind_code##year" "idcode i.ind_code##i.year" "ind_code##year##race" {
	capture drop __hdfe*
	capture drop fe1
	quietly reghdfe ln_wage ttl_exp, absorb(`spec') tol(1e-6)
	local e_abs `"`e(absvars)'"'
	local e_ext `"`e(extended_absvars)'"'
	local e_rows : rownames e(dof_table)
	local e_n = e(N_hdfe_extended)
	capture drop __hdfe*
	capture drop fe1
	_rf_parse_absorb `spec'
	assert `"`s(absvars)'"' == `"`e_abs'"'
	assert `"`s(extended_absvars)'"' == `"`e_ext'"'
	assert `"`s(dof_rownames)'"' == `"`e_rows'"'
	assert `s(num_slots)' == `e_n'
	di as text `"ok  reghdfe e() absorb(`spec')"'
}

* ---------------------------------------------------------------------------
* 3) _rf_parse_vce vs ms_parse_vce
* ---------------------------------------------------------------------------
capture program drop check_vce
program check_vce
	args vce wt
	ms_parse_vce, vce(`vce') weighttype(`wt')
	local R `s(vcetype)'|`s(num_clusters)'|`s(clustervars)'|`s(base_clustervars)'
	_rf_parse_vce, vce(`vce') weighttype(`wt')
	local T `s(vcetype)'|`s(num_clusters)'|`s(clustervars)'|`s(base_clustervars)'
	if ("`R'" != "`T'") {
		di as err `"FAIL vce(`vce') [`wt']: <`T'> vs <`R'>"'
		exit 9
	}
	di as text `"ok  vce(`vce') [`wt'] -> `T'"'
end
foreach v in "" "robust" "r" "rob" "ols" "unadjusted" "un" "conventional" "cl idcode" "cluster idcode" ///
	"cluster idcode year" "cluster idc#year" "cluster i.idcode#i.year" "cluster idcode year#ind_code" ///
	"cluster sid" {
	check_vce "`v'" ""
}
* fereg extension: string variables inside cluster interactions (ms_parse_vce rejects them)
_rf_parse_vce, vce(cluster sid#year i.syr)
assert "`s(clustervars)'" == "sid#year syr" & "`s(base_clustervars)'" == "sid year syr" & "`s(num_clusters)'" == "2"
check_vce "" pweight
check_vce "robust" pweight
check_vce "cluster idcode" aweight
foreach v in "unadjusted" "ols" {
	capture _rf_parse_vce, vce(`v') weighttype(pweight)
	assert _rc == 198
}
foreach v in "dkraay" "dk 2" "cluster" "bogus" "cluster idcode, minus(1)" "cluster c.age" {
	capture _rf_parse_vce, vce(`v')
	assert _rc == 198
	di as text `"ok  vce(`v') fails with rc 198"'
}

di as result "TEST_PARSE PASSED"
