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
	"cluster sid" "dkraay" "dk" "dkraay 3" "dk 1" {
	check_vce "`v'" ""
}
* vce(dkraay #): the bandwidth, as ms_parse_vce's s(dkraay_lags)
foreach v in "dkraay" "dkraay 3" "dk 1" {
	ms_parse_vce, vce(`v')
	local R `s(dkraay_lags)'
	_rf_parse_vce, vce(`v')
	assert "`s(dkraay_bw)'" == "`R'"
}
* fereg extension: string variables inside cluster interactions (ms_parse_vce rejects them)
_rf_parse_vce, vce(cluster sid#year i.syr)
assert "`s(clustervars)'" == "sid#year syr" & "`s(base_clustervars)'" == "sid year syr" & "`s(num_clusters)'" == "2"
check_vce "" pweight
check_vce "robust" pweight
check_vce "cluster idcode" aweight
check_vce "dkraay 2" pweight
foreach v in "unadjusted" "ols" {
	capture _rf_parse_vce, vce(`v') weighttype(pweight)
	assert _rc == 198
}
* errors of ms_parse_vce's vce(dkraay #) checks: same return code (its _assert: 9)
foreach v in "dkraay 2.5" "dkraay 0" "dkraay -1" "dkraay abc" "dkraay 2 3" {
	capture ms_parse_vce, vce(`v')
	local rc_r = _rc
	capture _rf_parse_vce, vce(`v')
	assert _rc == `rc_r' & _rc == 9
	di as text `"ok  vce(`v') fails with rc 9 as ms_parse_vce"'
}
gen double lat = 30 + mod(idcode, 17)
gen double lon = -100 + mod(idcode, 23)
foreach v in "cluster" "bogus" "cluster idcode, minus(1)" "cluster c.age" "dkraay, bw(2)" "hc2 idcode" ///
	"hc3 idcode, dfadjust" "hc2, dfadjust" "crv3" "crv3 c.age" "conley lat" "conley lat lon w" ///
	"conley lat lon, cutoff(0)" "conley lat lon, cutoff(x)" "conley lat lon, pixel(-1)" ///
	"conley lat lon, distance(euclid)" "conley lat lon, bogus" "conley sid lon" {
	capture _rf_parse_vce, vce(`v')
	assert _rc == 198
	di as text `"ok  vce(`v') fails with rc 198"'
}
* hc2/hc3 (Stata regress): pweights allowed with hc2 only, regress's message
capture _rf_parse_vce, vce(hc3) weighttype(pweight)
assert _rc == 198
_rf_parse_vce, vce(hc2) weighttype(pweight)
assert "`s(vcetype)'" == "hc2" & "`s(num_clusters)'" == "0"
_rf_parse_vce, vce(hc3) weighttype(aweight)
assert "`s(vcetype)'" == "hc3"
* crv3 (cluster dimensions as vce(cluster)) and conley (coordinates and suboptions)
_rf_parse_vce, vce(crv3 idc year#i.ind_code)
assert "`s(vcetype)'" == "crv3" & "`s(clustervars)'" == "idcode year#ind_code" & "`s(num_clusters)'" == "2"
assert "`s(base_clustervars)'" == "idcode year ind_code"
_rf_parse_vce, vce(conley lat lon)
assert "`s(vcetype)'" == "conley" & "`s(conley_lat)'" == "lat" & "`s(conley_lon)'" == "lon"
assert "`s(cutoff)'" == "" & "`s(pixel)'" == "0" & "`s(distance)'" == "triangular" & "`s(num_clusters)'" == "0"
_rf_parse_vce, vce(conley lon lat, cut(150.5) pix(10) dist(Spherical))
assert "`s(conley_lat)'" == "lon" & "`s(cutoff)'" == "150.5" & "`s(pixel)'" == "10" & "`s(distance)'" == "spherical"
di as text "ok  hc2/hc3/crv3/conley"

* ---------------------------------------------------------------------------
* 4) _rf_parse_hac (ivreghdfe's ivparse and s_vkernel; no estimation)
* ---------------------------------------------------------------------------
xtset idcode year
_rf_parse_hac, vcetype(robust) userrobust(1) bw(3) kernel(PAR)
assert "`s(hac)'" == "1" & "`s(kernel)'" == "Parzen" & "`s(kernelkey)'" == "parzen" & "`s(bw)'" == "3"
assert "`s(robust)'" == "1" & "`s(tvar)'" == "year" & "`s(ivar)'" == "idcode" & "`s(tdelta)'" == "1"
_rf_parse_hac, vcetype(unadjusted) bw(2.50) kernel(quadratic spectral)
assert "`s(kernel)'" == "Quadratic Spectral" & "`s(kernelkey)'" == "quadratic-spectral" & "`s(bw)'" == "2.5" & "`s(robust)'" == "0"
_rf_parse_hac, vcetype(unadjusted) dkraay(2)
assert "`s(kernel)'" == "Bartlett" & "`s(bw)'" == "2" & "`s(vcetype)'" == "cluster" & "`s(clustervars)'" == "year" & "`s(robust)'" == "0"
_rf_parse_hac, vcetype(cluster) clustervars(year idcode) userrobust(1) bw(2)
assert "`s(clustervars)'" == "idcode year" & "`s(num_clusters)'" == "2"
_rf_parse_hac, vcetype(unadjusted) kiefer
assert "`s(kernel)'" == "Truncated" & "`s(kiefer)'" == "1" & "`s(bw)'" == ""
_rf_parse_hac, vcetype(robust) weighttype(pweight) kiefer
assert "`s(robust)'" == "1" & "`s(kiefer)'" == "1"
_rf_parse_hac, vcetype(dkraay)
assert "`s(hac)'" == "0" & "`s(tvar)'" == "year" & "`s(ivar)'" == ""
_rf_parse_hac, vcetype(cluster) clustervars(idcode) userrobust(1)
assert "`s(hac)'" == "0" & "`s(tvar)'" == "" & "`s(clustervars)'" == "idcode"
* return codes of ivreghdfe 1.1.4 (also checked against ivreghdfe in test_hac.do)
foreach t in "102|bw(abc)" "102|kernel(bar)" "198|bw(0)" "198|kernel(newey) bw(2)" "198|bw(auto)" ///
	"198|dkraay(2) bw(3)" "198|dkraay(2) clustervars(idcode)" "198|kiefer userrobust(1)" "198|kiefer kernel(tru) bw(2)" ///
	"198|bw(2) psd0 psda" "101|bw(2) weighttype(fweight)" "198|bw(2) clustervars(idcode)" ///
	"198|bw(2) clustervars(idcode ind_code)" "198|psd0" {
	gettoken rc opts : t, parse("|")
	local opts = substr(`"`opts'"', 2, .)
	capture _rf_parse_hac, vcetype(unadjusted) `opts'
	assert _rc == `rc'
	di as text "ok  _rf_parse_hac `opts' fails with rc `rc'"
}
capture _rf_parse_hac, vcetype(hc2) bw(2)
assert _rc == 198
xtset, clear
foreach t in "111|bw(2)" "5|kiefer" "5|dkraay(2)" "9|" {
	gettoken rc opts : t, parse("|")
	local opts = substr(`"`opts'"', 2, .)
	local vt = cond(`"`opts'"' == "", "dkraay", "unadjusted")
	capture _rf_parse_hac, vcetype(`vt') `opts'
	assert _rc == `rc'
}
di as text "ok  _rf_parse_hac"

* ---------------------------------------------------------------------------
* 5) _rf_parse_bc: felogit's bias-correction and APE options (logitfe 3.0.3 names). Every
*    rejected combination and the s() of the accepted ones; _rf_bc_perm keeps the RNG state.
* ---------------------------------------------------------------------------
capture program drop bc_parse
program bc_parse
	* bc_parse "<options>" "<absorb, _cons = none>" ["<family logit>" "<separation fe>" "<nx 1>" "<weight>"]
	args opts absorb family sep nx weight
	if (`"`family'"' == "") local family logit
	if (`"`sep'"' == "") local sep fe
	if (`"`nx'"' == "") local nx 1
	if (`"`absorb'"' == "_cons") _rf_parse_absorb
	else _rf_parse_absorb `absorb'
	local G `s(G)'
	local ivars `"`s(ivars)'"'
	local ints `s(intercepts)'
	local ns `s(num_slopes)'
	local noabs = (`"`s(absvars)'"' == "_cons")
	_rf_parse_bc, family(`family') `opts' weight(`weight') separation(`sep') g(`G') ivars(`ivars') ///
		intercepts(`ints') numslopes(`ns') isnoabsorb(`noabs') nx(`nx')
end
capture program drop bc_rc
program bc_rc
	* bc_rc <rc> "<options>" ["<absorb>"] [family sep nx weight]
	args rc opts absorb family sep nx weight
	if (`"`absorb'"' == "") local absorb idcode year
	capture noisily bc_parse `"`opts'"' `"`absorb'"' `"`family'"' `"`sep'"' `"`nx'"' `"`weight'"'
	if (_rc != `rc') {
		di as err `"FAIL _rf_parse_bc `opts' [absorb(`absorb') `family' `sep' `nx' `weight']: rc `=_rc', expected `rc'"'
		exit 9
	}
	di as text `"ok  _rf_parse_bc `opts' [absorb(`absorb')] fails with rc `rc'"'
end

xtset idcode year
* option combinations
foreach o in "analytical jackknife" "analytical ibias(maybe)" "analytical tbias(1)" "ss1" "individuals" ///
	"analytical ss2" "analytical time" "ape js" "ape individuals" "multiple(2)" "analytical multiple(2)" ///
	"ape multiple(2)" "lags(1)" "jackknife lags(1)" "ape lags(1)" "ibias(no)" "ape ibias(no)" "ape tbias(yes)" ///
	"network" "population(100)" "jackknife ss1 ss2" "jackknife js jj" "jackknife sj double" ///
	"jackknife multiple(-1)" "jackknife js multiple(2)" "jackknife sj multiple(2)" "jackknife jj multiple(2)" ///
	"jackknife double multiple(2)" "jackknife jj individuals" "jackknife double time" ///
	"jackknife ss1 individuals" "jackknife ss2 time" "analytical lags(-1)" ///
	"analytical ibias(no) tbias(no)" "jackknife ibias(no) tbias(no)" "analytical lags(1) ibias(no)" ///
	"analytical population(-5)" {
	bc_rc 198 "`o'"
}
* model restrictions: weights, separation, regressors, absorbed terms
bc_rc 101 "analytical" "" "" "" "" fweight
bc_rc 101 "ape" "" "" "" "" pweight
bc_rc 198 "analytical" "" "" "fe mu"
bc_rc 198 "jackknife" "" "" "none"
bc_rc 102 "ape" "" "" "" 0
bc_rc 198 "analytical" "" poisson
bc_rc 198 "ape" "" poisson
foreach a in "_cons" "idcode##c.age" "idcode year##c.age" "idcode ind_code" "idcode#year" "year#idcode" ///
	"idcode year ind_code" "idcode i.idcode" "L.idcode year" "ind_code" {
	bc_rc 198 "analytical" "`a'"
}
* one fixed effect: only its own bias exists
bc_rc 198 "analytical tbias(yes)" "idcode"
bc_rc 198 "analytical ibias(no)" "idcode"
bc_rc 198 "jackknife tbias(yes)" "idcode"
bc_rc 198 "analytical ibias(yes)" "year"
bc_rc 198 "analytical tbias(no)" "year"
bc_rc 198 "analytical lags(1)" "year"
bc_rc 198 "jackknife ibias(no)" "i.idcode"
* network
bc_rc 198 "analytical network" "idcode"
bc_rc 198 "analytical network" "idcode year ind_code occ_code"
bc_rc 198 "jackknife network" "idcode year ind_code"
bc_rc 198 "analytical network ibias(no)" "idcode year ind_code"
bc_rc 198 "ape network tbias(yes)" "idcode year"
bc_rc 198 "analytical network" "idcode year##c.age"
* xtset requirements (logitfe's messages, r(5))
xtset, clear
bc_rc 5 "analytical"
bc_rc 5 "ape" "idcode"
bc_rc 5 "analytical network lags(1)" "idcode year ind_code"
preserve
quietly keep if idcode == 1
tsset year
bc_rc 5 "analytical" "year"
restore
xtset idcode year

* accepted combinations
bc_parse "" "idcode year"
assert "`s(bc)'" == ""
bc_parse "analytical" "idcode year"
assert "`s(bc)'" == "analytical" & "`s(structure)'" == "classic" & "`s(effects)'" == "it" & "`s(which)'" == "both"
assert "`s(ibias)'" == "yes" & "`s(tbias)'" == "yes" & "`s(lags)'" == "0" & "`s(ivar)'" == "idcode" & "`s(tvar)'" == "year"
assert "`s(title)'" == "Analytical bias-correction" & "`s(title3)'" == "Triming parameter = 0"
assert "`s(title1)'" == "Type of included effects: individual and time effects"
assert "`s(title2)'" == "Type of bias correction: individual and time effects"
bc_parse "an lags(2) tbias(no)" "year i.idcode"
assert "`s(effects)'" == "ti" & "`s(which)'" == "individual" & "`s(tbias)'" == "no" & "`s(lags)'" == "2"
assert "`s(title2)'" == "Type of bias correction: individual effects only" & "`s(title3)'" == "Triming parameter = 2"
bc_parse "analytical ibias(no)" "idcode year"
assert "`s(which)'" == "time" & "`s(title2)'" == "Type of bias correction: time effects only"
bc_parse "jackknife" "idcode year"
assert "`s(bc)'" == "jackknife" & "`s(variant)'" == "ss2" & "`s(multiple)'" == "0"
assert "`s(title)'" == "Split-panel jackknife in both dimensions" & "`s(title3)'" == ""
foreach v in ss1 js sj jj double {
	bc_parse "jack `v'" "idcode year"
	assert "`s(variant)'" == "`v'"
}
bc_parse "jack sj" "idcode year"
assert "`s(title)'" == "Split-panel jackknife in cross-section, delete-one jackknife in time series"
bc_parse "jack double tbias(no)" "idcode"
assert "`s(which)'" == "individual" & "`s(effects)'" == "i" & "`s(title1)'" == "Type of included effects: individual effects only"
bc_parse "ape" "year"
assert "`s(bc)'" == "ape" & "`s(which)'" == "time" & "`s(ibias)'" == "" & "`s(title)'" == "Uncorrected fixed-effects estimates"
assert "`s(title1)'" == "Type of included effects: time effects only" & "`s(title2)'" == "Type of bias correction: none"
bc_parse "analytical population(20000)" "idcode year"
assert "`s(population)'" == "20000"
bc_parse "analytical network lags(1)" "idcode#year ind_code#year idcode#ind_code"
assert "`s(structure)'" == "network" & "`s(which)'" == "both" & "`s(ivar)'" == "" & "`s(tvar)'" == "year"
assert "`s(title1)'" == "Type of included effects: exporter-time, importer-time and pair effects"
bc_parse "ape network" "idcode#year ind_code#year"
assert "`s(title1)'" == "Type of included effects: exporter-time and importer-time effects"
capture mata: mata which mm_sample()
if (_rc) {
	bc_rc 499 "jackknife multiple(2)"
	di as text "(moremata not found: multiple() acceptance and _rf_bc_perm not tested)"
}
else {
	bc_parse "jackknife ss1 multiple(3) time" "idcode year"
	assert "`s(multiple)'" == "3" & "`s(perm)'" == "t" & "`s(split)'" == "time"
	assert "`s(title3)'" == "3 multiple partitions in the time dimension"
	bc_parse "jackknife multiple(4)" "idcode year"
	assert "`s(perm)'" == "it" & "`s(split)'" == "both"
	assert "`s(title3)'" == "4 multiple partitions in both the cross-section and the time dimension"
	bc_parse "jackknife ss1 multiple(2) individuals time ibias(no)" "idcode year"
	assert "`s(perm)'" == "it" & "`s(which)'" == "time"
	bc_parse "jackknife ss2 multiple(2) individuals ibias(no)" "idcode year"
	assert "`s(perm)'" == "i" & "`s(title3)'" == "2 multiple partitions in the cross-section dimension"
	* logitfe's note: randomizing the dimension that is not split gives the plain estimator
	bc_parse "jackknife ss2 multiple(3) individuals tbias(no)" "idcode year"
	assert "`s(multiple)'" == "0" & "`s(perm)'" == "" & "`s(title3)'" == ""
	bc_parse "jackknife ss1 multiple(3) time" "year"
	assert "`s(multiple)'" == "0" & "`s(perm)'" == ""
	* _rf_bc_perm: one permutation of the distinct values per partition and dimension, the
	* session's RNG state unchanged
	set seed 271828
	local before = c(rngstate)
	_rf_bc_perm, ivar(idcode) tvar(year) multiple(2) perm(it) names(__p1 __t1 __p2 __t2)
	assert c(rngstate) == "`before'"
	foreach v in __p1 __p2 {
		quietly tab idcode
		local n = r(r)
		quietly bysort idcode (`v'): assert `v' == `v'[1]
		quietly bysort `v' (idcode): assert idcode == idcode[1]
		quietly summarize `v'
		assert r(min) == 1 & r(max) == `n'
	}
	quietly tab year
	assert r(r) == 15
	quietly summarize __t1
	assert r(min) == 1 & r(max) == 15
	assert __p1 != __p2 in 1
	drop __p1 __t1 __p2 __t2
}
di as text "ok  _rf_parse_bc"

di as result "TEST_PARSE PASSED"
