*! version 0.1.0  logitfast: predict
program define logitfast_p
	version 16
	_assert "`e(cmd)'" == "logitfast"
	_assert "`e(depvar)'" != ""

	* Intercept -scores- (margins sends var* wildcards)
	cap syntax anything [if] [in], SCore
	if (!c(rc)) {
		_score_spec `anything', score
		loc 0 `s(varlist)' `if' `in' , score
	}

	syntax newvarname [if] [in] [, Pr XB XBD D STDP SCore]
	loc nondefault `xb' `xbd' `d' `stdp' `score'
	opts_exclusive "`pr' `nondefault'"
	loc opt `pr' `nondefault'
	loc ifin `"`if' `in'"'
	if ("`opt'" == "") {
		di as text "(option pr assumed; Pr(`e(depvar)'))"
		loc opt pr
	}

	* Every option except xb and stdp needs the sum of fixed effects
	if !inlist("`opt'", "xb", "stdp") & ("`e(absvars)'" != "_cons") {
		_assert `"`e(d)'"' != "", msg("predict `opt' requires the -d- option of logitfast")
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

	* xb + d
	PredictXB `varlist' `ifin', xb
	if ("`e(absvars)'" != "_cons") {
		qui replace `varlist' = `varlist' + `e(d)' `ifin'
		la var `varlist' "Linear prediction: xb + d[`e(absvars)']"
	}
	if ("`opt'" == "xbd") exit

	qui replace `varlist' = invlogit(`varlist') `ifin'
	if ("`opt'" == "pr") {
		la var `varlist' "Pr(`e(depvar)')"
		exit
	}
	* score: y − Pr (y ≠ 0 counts as 1, as logit)
	conf var `e(depvar)', exact
	qui replace `varlist' = (`e(depvar)' != 0) - `varlist' `ifin'
	la var `varlist' "score"
end


program PredictXB
	syntax newvarname [if] [in], [*]
	cap matrix list e(b)
	if (c(rc)) {
		gen double `varlist' = 0 `if' `in'
	}
	else {
		_predict double `varlist' `if' `in', `options'
	}
end
