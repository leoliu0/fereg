*! version 0.2.0  feppml: Poisson pseudo-maximum likelihood with high-dimensional fixed effects (Rust plugin)
*! Syntax, defaults, messages, display and e() mirror ppmlhdfe 2.3.3 (Correia, Guimarães, Zylkin).
program define feppml, eclass
	version 16
	if replay() {
		if (`"`e(cmd)'"' != "feppml") error 301
		_rf_glm_replay `0'
		exit
	}
	local rf_display
	capture noisily _rf_glm poisson `0'
	local rc = _rc
	mata: st_local("rf_scalars", invtokens(st_dir("global", "numscalar", "__rf_*")'))
	if (`"`rf_scalars'"' != "") scalar drop `rf_scalars'
	if (`rc') exit `rc'
	_rf_glm_replay, `rf_display'
end
