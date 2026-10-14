*! version 0.4.0  fereg: ivreghdfe's report of gaps in the time variable of a kernel variance
* Usage: _rf_hac_gaps touse tvar   (data tsset/xtset; touse = the prepared sample)
program _rf_hac_gaps
	version 16
	args touse tvar
	* ivreghdfe runs `tsreport, panel` under version 11.2, which ignores `panel` on tsset time
	* series; from version 13 on it is error 321 without a panel variable.
	version 11.2: quietly tsreport if `touse', panel
	if (r(N_gaps) != 0) {
		di as text "Warning: time variable " as res "`tvar'" as text " has " ///
			as res "`r(N_gaps)'" as text " gap(s) in relevant range"
	}
end
