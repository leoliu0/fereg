*! version 0.4.0  feppml: Poisson pseudo-maximum likelihood with high-dimensional fixed effects (Rust plugin)
*! Syntax, defaults, messages, display and e() mirror ppmlhdfe 2.3.3 (Correia, Guimaraes, Zylkin).
program define feppml, eclass
	version 16
	if replay() {
		if (`"`e(cmd)'"' != "feppml") error 301
		_rf_glm_replay `0'
		exit
	}
	local rf_display
	local rf_tagsep 0
	capture noisily _rf_glm poisson `0'
	local rc = _rc
	mata: st_local("rf_scalars", invtokens(st_dir("global", "numscalar", "__rf_*")'))
	if (`"`rf_scalars'"' != "") scalar drop `rf_scalars'
	if (`rc') exit `rc'
	* tagsep(): separation check only, nothing estimated or posted (as ppmlhdfe)
	if (`rf_tagsep') exit
	_rf_glm_replay, `rf_display'
end
