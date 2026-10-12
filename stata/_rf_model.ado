*! version 0.2.1  fereg: model variables read by the plugin at prepare (protocol nmodel, cached)
*
* Usage:
*   _rf_model split <terms>            c_local rf_modelvars: the plain variable names (unique, in
*                                      order), sent to -prepare-, which excludes observations with
*                                      missing values and keeps the values for the estimation call;
*                                      c_local rf_markvars: the terms with factor-variable or
*                                      time-series operators (the caller marks them out)
*   _rf_model cached <vars> : <model>  c_local rf_cached: for each variable its 1-based position
*                                      in <model>, 0 if absent (comma-separated, for cached=)
program _rf_model
	version 16
	gettoken sub 0 : 0
	if ("`sub'" == "split") {
		local model
		local mark
		local rest `0'
		while (`"`rest'"' != "") {
			gettoken v rest : rest, bind
			if (regexm(`"`v'"', "^[A-Za-z_][A-Za-z0-9_]*$")) {
				if (!`: list v in model') local model `model' `v'
			}
			else local mark `mark' `v'
		}
		c_local rf_modelvars `model'
		c_local rf_markvars `mark'
	}
	else if ("`sub'" == "cached") {
		gettoken vars model : 0, parse(":")
		gettoken colon model : model, parse(":")
		local out
		local sep
		foreach v of local vars {
			local k : list posof "`v'" in model
			local out `out'`sep'`k'
			local sep ,
		}
		c_local rf_cached `out'
	}
	else {
		di as err "_rf_model: unknown subcommand `sub'"
		exit 198
	}
end
