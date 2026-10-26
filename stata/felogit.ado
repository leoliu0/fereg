*! version 0.5.0  felogit: logit (maximum likelihood) with high-dimensional fixed effects (Rust plugin)
*! Coefficients equal those of -logit y x i.fe1 i.fe2-; layout and options follow feppml.
program define felogit, eclass
	version 16
	if replay() {
		if (`"`e(cmd)'"' != "felogit") error 301
		_rf_glm_replay `0'
		exit
	}
	local rf_display
	capture noisily _rf_glm logit `0'
	local rc = _rc
	mata: st_local("rf_scalars", invtokens(st_dir("global", "numscalar", "__rf_*")'))
	if (`"`rf_scalars'"' != "") scalar drop `rf_scalars'
	if (`rc') exit `rc'
	_rf_glm_replay, `rf_display'
end
