*! version 0.4.1  fereg: estat after fereg, feppml and felogit (e(estat_cmd))
*! Mirrors reghdfe_estat (reghdfe 6.13.1), which is also what estat runs after ppmlhdfe 2.3.3
*! (its e(estat_cmd) is reghdfe5_estat, the same program): summarize, vce, and ic with
*! df = e(df_m) + 1. Unlike reghdfe_estat, estat summarize passes its varlist and options on.
program fereg_estat, rclass
	version 16
	if !inlist("`e(cmd)'", "fereg", "feppml", "felogit") {
		error 301
	}

	gettoken key rest : 0, parse(", ")
	local lkey = length(`"`key'"')
	if (`"`key'"' == substr("summarize", 1, max(2, `lkey'))) {
		local 0 `"`rest'"'
		syntax [anything] [, noHEader *]
		* noheader always on, as reghdfe_estat (workaround for estat_summ)
		estat_summ `anything', `options' noheader
	}
	else if (`"`key'"' == "vce") {
		vce `rest'
	}
	else if (`"`key'"' == "ic") {
		local 0 `"`rest'"'
		syntax [, *]
		estat_default ic, df(`=e(df_m) + 1') `options'
	}
	else {
		di as error `"invalid subcommand `key'"'
		exit 321
	}
	return add
end
