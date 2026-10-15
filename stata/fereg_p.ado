*! version 0.4.1  fereg: predict after fereg (mirrors reghdfe_p of reghdfe 6.13.1)
program define fereg_p, rclass
	version 16
	* Note: we IGNORE typlist and generate the newvar as double
	* Note: e(resid) is missing outside of e(sample), so we don't need to condition on e(sample)

	if ("`e(cmd)'" != "fereg") error 301

	* -score- is the same as -residuals- (needed by -margins- and -suest-)
	cap syntax anything [if] [in], SCore
	local was_score = !c(rc)
	if (`was_score') {
		_score_spec `anything', score
		local 0 `s(varlist)' `if' `in' , residuals
	}

	syntax newvarname [if] [in] [, XB STDP Residuals D XBD DResiduals]

	* Only one option
	opts_exclusive "`xb' `stdp' `residuals' `d' `xbd' `dresiduals'"

	* Default option is xb
	if ("`xb'`stdp'`residuals'`d'`xbd'`dresiduals'" == "") {
		di as text "(option xb assumed; fitted values)"
		local xb "xb"
	}

	local fixed_effects "`e(absvars)'"

	* Except for xb and stdp, we need the residuals saved by fereg
	if ("`xb'" == "" & "`stdp'" == "") {
		if ("`e(resid)'" == "") {
			di as err "you must add the {bf:resid} option to fereg before running this prediction"
			exit 198
		}
		confirm numeric variable `e(resid)', exact
	}

	if ("`xb'" != "" | "`stdp'" != "") {
		PredictXB `varlist' `if' `in', `xb' `stdp'
	}
	else if ("`residuals'" != "") {
		gen double `varlist' = `e(resid)' `if' `in'
		label variable `varlist' "Residuals"
		if (`was_score') return local scorevars `varlist'
	}
	else if ("`d'" != "") {
		* d = y - xb - resid
		tempvar xbv
		PredictXB `xbv' `if' `in', xb
		gen double `varlist' = `e(depvar)' - `xbv' - `e(resid)' `if' `in'
		label variable `varlist' "d[`fixed_effects']"
	}
	else if ("`xbd'" != "") {
		* xbd = y - resid
		gen double `varlist' = `e(depvar)' - `e(resid)' `if' `in'
		label variable `varlist' "Xb + d[`fixed_effects']"
	}
	else if ("`dresiduals'" != "") {
		* dresiduals = y - xb
		tempvar xbv
		PredictXB `xbv' `if' `in', xb
		gen double `varlist' = `e(depvar)' - `xbv' `if' `in'
	}
	else {
		error 100
	}
end

program PredictXB
	syntax newvarname [if] [in], [*]
	capture matrix list e(b) // if there are no regressors, _predict fails
	if (c(rc)) {
		gen double `varlist' = 0 `if' `in'
	}
	else {
		_predict double `varlist' `if' `in', `options'
	}
end
