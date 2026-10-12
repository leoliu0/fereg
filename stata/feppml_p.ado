*! version 0.2.1  feppml: predict (port of ppmlhdfe_p.ado of ppmlhdfe 2.3.3, MIT, (c) Sergio Correia, Paulo Guimarães, Thomas Zylkin)
program define feppml_p
	version 16
	_assert "`e(cmd)'" == "feppml"
	_assert "`e(depvar)'" != ""

	* Intercept -scores- (margins sends var* wildcards)
	cap syntax anything [if] [in], SCores
	loc was_score = !c(rc)
	if (`was_score') {
		_score_spec `anything', score
		loc 0 `s(varlist)' `if' `in' , scores
	}

	syntax newvarname [if] [in] [, ///
		XB XBD D Mu Eta STDP ///
		Anscombe Cooksd DEViance Hat Likelihood Pearson Response SCores Working]

	loc nondefault `xb' `d' `xbd' `eta' `stdp' `anscombe' `cooksd' `deviance' `hat' `likelihood' `pearson' `response' `scores' `working'
	opts_exclusive "`mu' `nondefault'"
	loc opt `mu' `nondefault'
	loc y `e(depvar)'
	loc ifin `"`if' `in'"'

	if ("`opt'" == "") {
		di as text "(option mu assumed; predicted mean of depvar)"
		loc opt "mu"
	}
	if inlist("`opt'", "cooksd", "hat", "likelihood") {
		di as err "predict option `opt' not implemented"
		exit 198
	}

	* Every option except -xb- needs the sum of fixed effects
	if ("`opt'" != "xb") & ("`e(absvars)'" != "_cons") {
		_assert `"`e(d)'"' != "", msg("predict `opt' requires the -d- option of feppml")
		conf double var `e(d)', exact
	}

	if inlist("`opt'", "xb", "stdp") {
		PredictXB `varlist' `ifin', `opt'
		exit
	}

	if ("`opt'" == "d") {
		gen double `varlist' = `e(d)' `ifin'
		la var `varlist' "d[`e(absvars)']"
		exit
	}

	* η = xb + d
	PredictXB `varlist' `ifin', xb
	if ("`e(absvars)'" != "_cons") {
		qui replace `varlist' = `varlist' + `e(d)' `ifin'
		la var `varlist' "Linear prediction: xb + d[`e(absvars)']"
	}
	else {
		if (`"`e(d)'"' != "") qui replace `varlist' = . if mi(`e(d)') `ifin'
		la var `varlist' "Linear prediction: xb"
	}
	if inlist("`opt'", "xbd", "eta") exit

	if ("`opt'" == "mu") {
		qui replace `varlist' = exp(`varlist') `ifin'
		la var `varlist' "Predicted mean of `y'"
		exit
	}

	tempvar mu
	gen double `mu' = exp(`varlist') `ifin'
	conf var `y', exact

	if ("`opt'" == "deviance") {
		qui replace `varlist' = 2 * cond(`y' > 0, `mu' - `y' + `y' * ln(`y' / `mu'), `mu') `ifin'
		la var `varlist' "deviance residual"
	}
	else if inlist("`opt'", "response", "scores") {
		qui replace `varlist' = `y' - `mu' `ifin'
		la var `varlist' "`=cond("`opt'" == "scores", "score", "response")' residual"
	}
	else if ("`opt'" == "pearson") {
		qui replace `varlist' = (`y' - `mu') / sqrt(`mu') `ifin'
		la var `varlist' "Pearson residual"
	}
	else if ("`opt'" == "anscombe") {
		qui replace `varlist' = 1.5 * (`y' ^ (2/3) - `mu' ^ (2/3)) / `mu' ^ (1/6) `ifin'
		la var `varlist' "Anscombe residual"
	}
	else if ("`opt'" == "working") {
		qui replace `varlist' = (`y' - `mu') / `mu' `ifin'
		la var `varlist' "working residual"
	}
	else {
		di as err "option not implemented: `opt'"
		exit 198
	}
end


program PredictXB
	syntax newvarname [if] [in], [*]
	cap matrix list e(b) // with no regressors -_predict- fails
	if (c(rc)) {
		gen double `varlist' = 0 `if' `in'
	}
	else {
		_predict double `varlist' `if' `in', `options'
	}
end
