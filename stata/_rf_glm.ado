*! version 0.4.1  fereg: estimation engine shared by feppml (Poisson) and felogit (logit)
*! Syntax, messages and e() of feppml mirror ppmlhdfe 2.3.3 (Correia, Guimaraes, Zylkin).
*
* Usage (from feppml.ado / felogit.ado):  _rf_glm poisson|logit <command arguments>
program _rf_glm, eclass
	version 16
	gettoken family 0 : 0
	local cmd = cond("`family'" == "poisson", "feppml", "felogit")
	local cmdline `"`cmd' `0'"'
	ereturn clear

// --------------------------------------------------------------------------
// Syntax
// --------------------------------------------------------------------------
	#delimit ;
	syntax varlist(fv ts numeric) [if] [in] [pw fw/] [,
		Absorb(string) NOAbsorb
		VCE(string) CLuster(string)
		BW(string) kernel(string) kiefer dkraay(integer 0) psd0 psda
		EXPosure(varname numeric) OFFset(varname numeric)
		EForm IRr OR
		GUESS(string) TOLerance(real 1e-8) MAXITerations(integer 10000)
		SEParation(string)
		D(name) D2
		ITOLerance(real -1)
		KEEPSINgletons
		THReads(integer 0) Verbose(integer 0) noLog SERIALio
		noHEader noTABle noFOOTnote
		TEChnique(string) TRAnsform(string) ACCELeration(string) POOLsize(string)
		ABSORBER(string)
		TAGSEP(name) ZVARname(name)
		* ] ;
	#delimit cr
	* absorber(fast|dense) is an undocumented debugging option (dense = explicit-dummy reference)

// --------------------------------------------------------------------------
// Validate options
// --------------------------------------------------------------------------
	foreach opt in technique transform acceleration poolsize {
		if (`"``opt''"' != "") di as text "(`cmd': option `opt'() ignored)"
	}
	_get_diopts diopts options, `options'
	if (`"`options'"' != "") {
		di as err `"option `options' not allowed"'
		exit 198
	}
	* tagsep()/zvarname(): ppmlhdfe's options to identify the separated observations (no estimation)
	local is_tagsep = ("`tagsep'" != "")
	if ("`family'" == "logit" & "`tagsep'`zvarname'" != "") {
		di as err "options tagsep() and zvarname() not allowed with felogit"
		exit 198
	}
	if ("`zvarname'" != "" & !`is_tagsep') {
		di as err "option zvarname() requires tagsep()"
		exit 198
	}
	if (`is_tagsep' & "`tagsep'" == "`zvarname'") {
		di as err "tagsep() and zvarname() must name different variables"
		exit 198
	}
	if ("`noabsorb'" != "" & `"`absorb'"' != "") {
		di as err "options absorb() and noabsorb are mutually exclusive"
		exit 198
	}
	if ("`exposure'" != "" & "`offset'" != "") {
		di as err "only one of offset() or exposure() can be specified"
		exit 198
	}
	if ("`family'" == "logit") {
		foreach opt in exposure eform irr guess {
			if (`"``opt''"' != "") {
				di as err "option `opt'() not allowed with felogit"
				exit 198
			}
		}
	}
	else if ("`or'" != "") {
		di as err "option or not allowed with feppml (use eform or irr)"
		exit 198
	}
	if !(`tolerance' > 0 & `tolerance' <= 1) {
		di as err "tolerance must be a real between 0 and 1"
		exit 198
	}
	if (`maxiterations' < 1) {
		di as err "maxiterations() must be a positive integer"
		exit 198
	}
	if (`threads' < 0) {
		di as err "threads() must be a nonnegative integer"
		exit 198
	}
	if (`"`absorber'"' != "" & !inlist(`"`absorber'"', "fast", "dense")) {
		di as err "absorber() must be fast or dense"
		exit 198
	}
	local drop_singletons = ("`keepsingletons'" == "")
	local plugin_verbose = max(0, min(`verbose', 2))
	local eform `eform'`irr'
	if ("`eform'" != "") local eform eform

	* guess() (ppmlhdfe): simple (default) or ols
	if (inlist("`guess'", "", "default")) local guess simple
	if (!inlist("`guess'", "simple", "ols")) {
		di as err `"guess() invalid; valid initial value options are "simple" and "ols""'
		exit 198
	}

	* separation(): ppmlhdfe tokens; simplex is an alias of relu
	if ("`family'" == "poisson") {
		local all_techniques "fe relu simplex mu"
		local separation = subinstr(`"`separation'"', "ir", "relu", 1)
		if inlist("`separation'", "", "def", "default", "standard", "auto", "on") local separation "fe simplex relu"
		if inlist("`separation'", "all", "full") local separation "`all_techniques'"
		if inlist("`separation'", "no", "off", "none") local separation
	}
	else {
		local all_techniques "fe mu"
		if inlist("`separation'", "", "def", "default", "standard", "auto", "on") local separation "fe"
		if inlist("`separation'", "all", "full") local separation "`all_techniques'"
		if inlist("`separation'", "no", "off", "none") local separation
	}
	local separation : list retokenize separation
	if ("`: list separation - all_techniques'" != "") {
		di as err `"separation(`separation') not allowed"'
		exit 198
	}
	* ppmlhdfe applies the fe rule only while dropping singletons
	local sep_fe = `: list posof "fe" in separation' > 0
	if ("`family'" == "poisson") local sep_fe = `sep_fe' & `drop_singletons'
	local glmsep
	if (`: list posof "relu" in separation' | `: list posof "simplex" in separation') local glmsep relu
	if (`: list posof "mu" in separation') local glmsep = cond("`glmsep'" == "", "mu", "`glmsep',mu")
	if ("`glmsep'" == "") local glmsep none
	* tagsep(): as ppmlhdfe, separation() is overridden by relu and nothing is estimated. Prepare
	* drops singletons only (sep_fe 0); the core's separation rounds apply the fe rule themselves.
	if (`is_tagsep') {
		di as text "{bf: (identifying separated observations instead of running regressions)}"
		local separation relu
		local sep_fe 0
		local glmsep relu
	}

	* d(): sum of the fixed effects
	if ("`d2'" != "") {
		if ("`d'" != "") {
			di as err "d() syntax error"
			exit 198
		}
		local d = cond("`family'" == "poisson", "_ppmlhdfe_d", "_felogit_d")
		capture drop `d' // destructive, as ppmlhdfe
	}
	else if ("`d'" != "") {
		confirm new variable `d'
	}

// --------------------------------------------------------------------------
// Variables, weights, vce(), absorb()
// --------------------------------------------------------------------------
	gettoken depvar indepvars : varlist, bind
	fvexpand `depvar'
	local depvar `r(varlist)'
	if (`: word count `depvar'' != 1) {
		di as err "more than one depvar specified: `depvar'"
		exit 198
	}
	if (strpos("`depvar'", "o.")) {
		di as err "the values of depvar are omitted: `depvar'"
		exit 198
	}
	local indepvars `indepvars'

	local wtype none
	local wvar
	if ("`weight'" != "") {
		unab exp : `exp', min(1) max(1) name(weight)
		confirm numeric variable `exp'
		local wvar `exp'
		local wtype = substr("`weight'", 1, 1) + "w"
	}

	if (`"`cluster'"' != "") {
		if (`"`vce'"' != "") {
			di as err "only one of cluster() and vce() can be specified"
			exit 198
		}
		local vce cluster `cluster'
	}
	if ("`family'" == "logit" & inlist(`"`vce'"', "oim", "ols")) local vce
	_rf_parse_vce, vce(`vce') weighttype(`weight')
	local vcetype `s(vcetype)'
	* PPML uses robust standard errors
	if ("`family'" == "poisson" & "`vcetype'" == "unadjusted") local vcetype robust
	local clustervars `s(clustervars)'
	local num_clusters `s(num_clusters)'
	local dkraay_bw `s(dkraay_bw)'
	local conley_lat `s(conley_lat)'
	local conley_lon `s(conley_lon)'
	local conley_cutoff `s(cutoff)'
	local conley_pixel `s(pixel)'
	local conley_distance `s(distance)'
	* ivreghdfe kernel options (borrowed) and vce(dkraay); Poisson is always robust (ppmlhdfe)
	local userrobust = inlist("`vcetype'", "robust", "cluster") & ("`family'" == "poisson" | !(`"`vce'"' == "" & "`weight'" == "pweight"))
	_rf_parse_hac, vcetype(`vcetype') clustervars(`clustervars') weighttype(`weight') userrobust(`userrobust') ///
		bw(`bw') kernel(`kernel') `kiefer' dkraay(`dkraay') `psd0' `psda'
	local hac `s(hac)'
	local hac_kernel `s(kernel)'
	local hac_kernelkey `s(kernelkey)'
	local hac_bw `s(bw)'
	local hac_kiefer `s(kiefer)'
	local hac_psd `s(psd)'
	local hac_robust `s(robust)'
	local tvar `s(tvar)'
	local ivar `s(ivar)'
	local xtivar `s(xtivar)'
	local tdelta `s(tdelta)'
	local clustvar_e `clustervars'
	if (`hac') {
		local vcetype `s(vcetype)'
		local clustervars `s(clustervars)'
		local num_clusters `s(num_clusters)'
		if (`dkraay') local clustvar_e `tvar'
	}
	local base_clustervars
	foreach dim of local clustervars {
		local parts : subinstr local dim "#" " ", all
		local base_clustervars `base_clustervars' `parts'
	}
	local base_clustervars : list uniq base_clustervars

	capture drop __hdfe* // previously saved FEs (as reghdfe)
	_rf_parse_absorb `absorb'
	local G `s(G)'
	local absvars `"`s(absvars)'"'
	local extended_absvars `"`s(extended_absvars)'"'
	local dof_rownames `"`s(dof_rownames)'"'
	local fe_labels `"`s(fe_labels)'"'
	local all_ivars `"`s(ivars)'"'
	local all_cvars `"`s(cvars)'"'
	local intercepts `s(intercepts)'
	local num_slopes `s(num_slopes)'
	local all_targets `"`s(targets)'"'
	local absfactors `s(factors)'
	local slopevars `s(slopevars)'
	local num_slots `s(num_slots)'
	local is_noabsorb = (`"`absvars'"' == "_cons")

	if ("`family'" == "poisson" & !`drop_singletons' & `verbose' > -1) {
		di as err "warning: keeping singleton groups will keep fixed effects that cause separation"
	}

// --------------------------------------------------------------------------
// Sample
// --------------------------------------------------------------------------
	* touse: if + in + weight. Missing values of the plain model variables (the dependent
	* variable first), the absorbed factors, the slopes and the cluster variables exclude an
	* observation in the plugin at -prepare- (protocol), as in fereg.ado. With exposure() or
	* offset() the model and cluster variables are marked out here first, since the exposure
	* check runs over that sample.
	if ("`exposure'`offset'" != "") {
		local varlist `depvar' `indepvars' `base_clustervars' `tvar' `ivar' `conley_lat' `conley_lon'
		marksample touse, strok
	}
	else marksample touse, novarlist

	local offvar_e
	local offvar
	if ("`exposure'" != "") {
		capture assert `exposure' > 0 if `touse'
		if (_rc) {
			di as err "exposure() must be greater than zero"
			exit 459
		}
		tempvar offvar
		qui gen double `offvar' = ln(`exposure') if `touse'
		local offvar_e "ln(`exposure')"
	}
	else if ("`offset'" != "") {
		local offvar `offset'
		local offvar_e `offset'
	}
	if ("`offvar'" != "") markout `touse' `offvar'
	fvrevar `depvar'
	local yvar `r(varlist)'
	_rf_model split `yvar' `indepvars'
	local modelvars `rf_modelvars'
	if ("`rf_markvars'" != "") markout `touse' `rf_markvars'
	local nmodel : word count `modelvars'

	local factors `absfactors'
	foreach v in `base_clustervars' `tvar' `ivar' {
		if (!`: list v in factors') local factors `factors' `v'
	}
	local nfac : word count `factors'
	local nslope : word count `slopevars'
	local facvars
	local factype
	foreach v of local factors {
		if (strpos("`v'", ".")) {
			tsrevar `v'
			local v `r(varlist)'
		}
		local facvars `facvars' `v'
		capture confirm string variable `v'
		local factype `factype'`=cond(_rc, "n", "s")'
	}
	local slopevars_r
	foreach v of local slopevars {
		if (strpos("`v'", ".")) {
			tsrevar `v'
			local v `r(varlist)'
		}
		local slopevars_r `slopevars_r' `v'
	}

	* absorb key: factors|intercept|slopes|save per term (1-based positions)
	local abskey none
	if (!`is_noabsorb') {
		local abskey
		forval g = 1/`G' {
			local iv : word `g' of `all_ivars'
			local cv : word `g' of `all_cvars'
			local tg : word `g' of `all_targets'
			local icpt : word `g' of `intercepts'
			local fpos
			foreach v of local iv {
				local k : list posof "`v'" in factors
				local fpos = cond("`fpos'" == "", "`k'", "`fpos'*`k'")
			}
			local spos
			foreach v of local cv {
				local k : list posof "`v'" in slopevars
				local spos = cond("`spos'" == "", "`k'", "`spos'*`k'")
			}
			if ("`spos'" == "") local spos -
			local save = ("`tg'" != "")
			local term `fpos'|`icpt'|`spos'|`save'
			local abskey = cond("`abskey'" == "", "`term'", "`abskey';`term'")
		}
	}
	_rf_vce_args, factors(`factors') vcetype(`vcetype') clustervars(`clustervars') tvar(`tvar') ivar(`ivar') ///
		tdelta(`tdelta') dkraay_bw(`dkraay_bw') hac(`hac') kernelkey(`hac_kernelkey') bw(`hac_bw') ///
		kiefer(`hac_kiefer') psd(`hac_psd') lat(`conley_lat') lon(`conley_lon') cutoff(`conley_cutoff') ///
		pixel(`conley_pixel') distance(`conley_distance')

	local common `""threads=`threads'" "verbose=`plugin_verbose'""'
	if ("`serialio'" != "") local common `"`common' "serialio=1""'
	local abskind
	if ("`absorber'" != "") local abskind `""absorber=`absorber'""'

// --------------------------------------------------------------------------
// Prepare: factorize, drop singletons and fe-separated groups
// --------------------------------------------------------------------------
	_rf_load
	local args `""nfac=`nfac'" "nslope=`nslope'" "nmodel=`nmodel'" "absorb=`abskey'" "wtype=`wtype'" "singletons=`drop_singletons'" "family=`family'" "sep=`=cond(`sep_fe', "fe", "none")'""'
	local args `"`args' `rf_vceargs'"'
	if (`nfac' > 0) local args `"`args' "factype=`factype'""'
	* No if, in, weight, exposure, offset or markout: touse is 1 everywhere and the plugin does
	* not read it
	if (`"`if'`in'`weight'`exposure'`offset'`rf_markvars'"' == "") local args `"`args' "sample=all""'
	local _rf_token
	_rf_load `facvars' `slopevars_r' `conley_lat' `conley_lon' `wvar' `modelvars' `touse', prepare `args' `common' `abskind'
	local token `_rf_token'
	local num_singletons0 = scalar(__rf_num_singletons)
	local num_fesep0 = scalar(__rf_num_fe_separated)
	if (`verbose' > -1) {
		if ("`family'" == "poisson" & `sep_fe' & `num_singletons0' + `num_fesep0' > 0) {
			local ns = strtrim(string(`num_singletons0' + `num_fesep0', "%20.0f"))
			di as text `"(dropped `ns' observations that are either {browse "http://scorreia.com/research/singletons.pdf":singletons} or {browse "http://scorreia.com/research/separation.pdf":separated} by a fixed effect)"'
		}
		else {
			if (`num_singletons0' > 0) {
				local ns = strtrim(string(`num_singletons0', "%20.0f"))
				di as text `"(dropped `ns' {browse "http://scorreia.com/research/singletons.pdf":singleton observations})"'
			}
			if (`num_fesep0' > 0) {
				local ns = strtrim(string(`num_fesep0', "%20.0f"))
				di as text "(dropped `ns' observations with all-zero or all-one outcomes within a fixed-effect group)"
			}
		}
	}
	if (`hac' & !`is_tagsep') _rf_hac_gaps `touse' `tvar'
	if ("`family'" == "poisson") {
		qui count if `touse' & `yvar' < 0
		if (r(N)) {
			di as err "`depvar' must be greater than or equal to zero"
			exit 459
		}
	}

// --------------------------------------------------------------------------
// Regressors: expand factor variables over the final sample
// --------------------------------------------------------------------------
	local fullvarlist
	local selected
	local sent_full
	local mask
	local xvars
	if ("`indepvars'" != "") {
		fvexpand `indepvars' if `touse'
		local fullvarlist `r(varlist)'
		foreach part of local fullvarlist {
			_ms_parse_parts `part'
			if (r(omit) == 0) {
				local sent_full `sent_full' `part'
				AddBN `part'
				local selected `selected' `part'
				local mask `mask' 1
			}
			else {
				local mask `mask' 0
			}
		}
		if ("`selected'" != "") {
			fvrevar `selected'
			local xvars `r(varlist)'
		}
	}
	local nx : word count `xvars'
	local K_all = `nx' + 1

// --------------------------------------------------------------------------
// tagsep(): the ReLU separation check alone; tag the separated observations and exit (ppmlhdfe)
// --------------------------------------------------------------------------
	if (`is_tagsep') {
		* As ppmlhdfe, an existing variable of that exact name is replaced, unless the model uses it
		local usedvars `yvar' `xvars' `offvar' `facvars' `slopevars_r' `wvar' `depvar' `exposure' `offset'
		foreach v in `tagsep' `zvarname' {
			if (`: list v in usedvars') {
				di as err "tagsep()/zvarname(): variable `v' is used by the model"
				exit 198
			}
		}
		novarabbrev capture drop `tagsep'
		qui gen byte `tagsep' = .
		local outvars `tagsep'
		local zarg
		if ("`zvarname'" != "") {
			novarabbrev capture drop `zvarname'
			qui gen double `zvarname' = .
			local outvars `outvars' `zvarname'
			local zarg `""zsep=2""'
		}
		local nout : word count `outvars'
		local tolstr = strtrim(string(`tolerance', "%21.17g"))
		local args `""token=`token'" "nx=`nx'" "nout=`nout'" "family=`family'" "tol=`tolstr'" "maxiter=`maxiterations'""'
		if (`itolerance' > 0) local args `"`args' "itol=`=strtrim(string(`itolerance', "%21.17g"))'""'
		local args `"`args' "separation=relu" "guess=`guess'" "log=`=("`log'" != "nolog")'" "offset=`=("`offvar'" != "")'""'
		_rf_model cached `yvar' `xvars' : `modelvars'
		local args `"`args' "sepcheck=1" "sep=1" `zarg' "cached=`rf_cached'""'
		capture noisily _rf_load `yvar' `xvars' `offvar' `touse' `outvars', glm `args' `common'
		if (_rc) {
			local rc = _rc
			capture drop `outvars'
			exit `rc'
		}
		label variable `tagsep' "[feppml: ReLU] Obs. is separated"
		format %1.0f `tagsep'
		if ("`zvarname'" != "") {
			mata: _rf_prettify_z("`zvarname'", "`touse'")
			label variable `zvarname' "[feppml: ReLU] Certificate of separation"
			format %10.2g `zvarname'
		}
		if (`verbose' > -1) {
			local nsep = scalar(__rf_num_separated)
			local nsep_s = strtrim(string(`nsep', "%20.0f"))
			di as text "(tagged {res}`nsep_s'{txt} separated observation`=cond(`nsep' == 1, "", "s")' in {res}`tagsep'{txt}" ///
				`"`=cond("`zvarname'" != "", "; certificate of separation in {res}`zvarname'{txt}", "")')"'
			if (scalar(__rf_separation_complete) == 0) {
				di as err "warning: the ReLU separation check stopped before certifying its result; more observations may be separated"
			}
		}
		c_local rf_tagsep 1
		exit
	}

// --------------------------------------------------------------------------
// Output variables (pre-created; the plugin fills kept observations)
// --------------------------------------------------------------------------
	local outvars
	local nout 0
	local pos_d 0
	local pos_fe
	if ("`d'" != "") {
		qui gen double `d' = .
		local outvars `outvars' `d'
		local pos_d = `++nout'
	}
	local fevars
	local fevar_labels
	local slot 0
	forval g = 1/`G' {
		local tg : word `g' of `all_targets'
		local nslots_g = `: word `g' of `intercepts'' + `: word `g' of `num_slopes''
		forval h = 1/`nslots_g' {
			local ++slot
			if ("`tg'" != "") {
				local t : word `h' of `tg'
				local lab : word `slot' of `fe_labels'
				qui gen double `t' = .
				local fevars `fevars' `t'
				local fevar_labels `"`fevar_labels' "`lab'""'
				local outvars `outvars' `t'
				local ++nout
				local pos_fe = cond("`pos_fe'" == "", "`nout'", "`pos_fe',`nout'")
			}
		}
	}
	if ("`pos_fe'" == "") local pos_fe 0

// --------------------------------------------------------------------------
// Estimate
// --------------------------------------------------------------------------
	tempname b V doft
	matrix `b' = J(1, `K_all', 0)
	matrix `V' = J(`K_all', `K_all', 0)
	matrix `doft' = J(`num_slots', 5, 0)
	local tolstr = strtrim(string(`tolerance', "%21.17g"))
	local args `""token=`token'" "nx=`nx'" "nout=`nout'" "family=`family'" "tol=`tolstr'" "maxiter=`maxiterations'""'
	if (`itolerance' > 0) local args `"`args' "itol=`=strtrim(string(`itolerance', "%21.17g"))'""'
	local args `"`args' "separation=`glmsep'" "guess=`guess'" "log=`=("`log'" != "nolog")'" "offset=`=("`offvar'" != "")'""'
	_rf_model cached `yvar' `xvars' : `modelvars'
	local args `"`args' "d=`pos_d'" "fe=`pos_fe'" "b=`b'" "V=`V'" "doft=`doft'" "cached=`rf_cached'""'
	local _rf_status
	capture noisily _rf_load `yvar' `xvars' `offvar' `touse' `outvars', glm `args' `common'
	if (_rc) {
		local rc = _rc
		if ("`outvars'" != "") capture drop `outvars'
		exit `rc'
	}
	local status `_rf_status'
	if (`: word count `status'' != `nx') {
		if ("`outvars'" != "") capture drop `outvars'
		di as err "`cmd': internal error: plugin returned `: word count `status'' status codes for `nx' regressors"
		exit 9999
	}

	* Collinearity notes (ppmlhdfe wording; the plugin cannot see variable names)
	if (`verbose' > -1) {
		local itol = cond(`itolerance' > 0, `itolerance', 1e-9)
		local tolfmt = string(min(1e-6, `itol' / 10), "%3.1e")
		local omitted
		forval j = 1/`nx' {
			local st : word `j' of `status'
			if (`st' == 2) {
				di as text "note: {res}`: word `j' of `selected''{txt} is probably collinear with the fixed effects (all partialled-out values are close to zero; tol =`tolfmt')"
			}
			if (`st' != 0) local omitted `omitted' `: word `j' of `sent_full''
		}
		local nom : word count `omitted'
		if (`nom') di as text "note: `nom' variable`=cond(`nom' > 1, "s", "")' omitted because of collinearity: {res}`omitted'{txt}"
	}

// --------------------------------------------------------------------------
// Post results
// --------------------------------------------------------------------------
	local colnames `fullvarlist' _cons
	* ppmlhdfe: invtokens((kept names, " _cons")), i.e. non-base, non-collinear regressors and a
	* doubled space before _cons
	local indepvars_e
	forval j = 1/`nx' {
		if (`: word `j' of `status'' == 0) local indepvars_e `indepvars_e' `: word `j' of `selected''
	}
	local K_full : word count `colnames'
	if (`K_full' > `K_all') {
		tempname bf Vf
		mata: _rf_m = (strtoreal(tokens(st_local("mask"))), 1)
		mata: _rf_i = selectindex(_rf_m)
		mata: _rf_B = J(1, cols(_rf_m), 0); _rf_W = J(cols(_rf_m), cols(_rf_m), 0)
		mata: _rf_B[_rf_i] = st_matrix(st_local("b")); _rf_W[_rf_i, _rf_i] = st_matrix(st_local("V"))
		mata: st_matrix(st_local("bf"), _rf_B); st_matrix(st_local("Vf"), _rf_W)
		mata: mata drop _rf_m _rf_i _rf_B _rf_W
		local b `bf'
		local V `Vf'
	}
	* Stata posts a V that is not positive semidefinite only when the matrix was stored by Mata, as
	* ppmlhdfe stores it; -matrix- or a plugin give "variance matrix is nonsymmetric or highly
	* singular"
	mata: st_matrix(st_local("V"), st_matrix(st_local("V")))
	matrix colnames `b' = `colnames'
	matrix colnames `V' = `colnames'
	matrix rownames `V' = `colnames'
	_ms_findomitted `b' `V'
	ereturn post `b' `V', esample(`touse') buildfvinfo depname(`depvar')

	local nsep = scalar(__rf_num_separated)
	if ("`family'" == "poisson") {
		local scalars N rank df N_hdfe N_hdfe_extended df_a df_a_initial df_a_redundant df_a_nested ///
			drop_singletons num_singletons df_m rss rmse N_full ic ic2 converged chi2 r2_p ll_0 deviance ll num_separated ///
			separation_complete
	}
	else {
		local scalars N rank ic ic2 converged ll ll_0 df_m chi2 p r2_p deviance ///
			N_hdfe N_hdfe_extended df_a df_a_initial df_a_redundant df_a_nested drop_singletons num_singletons num_separated N_full ///
			separation_complete
	}
	foreach s of local scalars {
		if ("`s'" == "N_hdfe") ereturn scalar N_hdfe = `G'
		else if ("`s'" == "N_hdfe_extended") ereturn scalar N_hdfe_extended = `num_slots'
		else if ("`s'" == "drop_singletons") ereturn scalar drop_singletons = `drop_singletons'
		else if ("`s'" == "p") ereturn scalar p = chi2tail(e(df_m), e(chi2))
		else PostScalar `s'
	}
	local post_clusters `clustervars'
	if ("`vcetype'" == "dkraay") local post_clusters `tvar'
	local post_ncl : word count `post_clusters'
	if (`post_ncl' > 0) {
		ereturn scalar N_clustervars = `post_ncl'
		forval i = 1/`post_ncl' {
			PostScalar N_clust`i'
		}
		PostScalar N_clust
	}
	if ("`vcetype'" == "dkraay") PostScalar dkraay_bw

	if ("`d'" != "") ereturn local d `d'
	if ("`family'" == "poisson") {
		ereturn local marginsok "default"
		ereturn local marginsnotok "stdp Anscombe Cooksd Deviance Hat Likelihood Pearson Response Score Working ADJusted STAndardized STUdentized MODified"
	}
	else {
		ereturn local marginsok "default Pr"
		ereturn local marginsnotok "stdp"
	}
	ereturn local separation `separation'
	ereturn local predict `cmd'_p
	ereturn local estat_cmd fereg_estat
	ereturn local cmdline `"`cmdline'"'
	local title = cond("`family'" == "poisson", "HDFE {help `cmd'##description:PPML} regression", "HDFE Logistic regression")
	ereturn local title "`title'"
	ereturn local chi2type "Wald"
	* e(vce) and e(vcetype): ppmlhdfe (robust, cluster), logit (oim), reghdfe (dkraay), regress
	* (hc2, hc3), ivreghdfe's strings for the kernel options; crv3 and conley are fereg's
	if (`hac') {
		_rf_hac_post, hac(1) kernel(`hac_kernel') tvar(`tvar') ivar(`xtivar') robust(`hac_robust') ///
			clustvar(`clustvar_e') clusters(`clustervars') kiefer(`hac_kiefer') psd(`hac_psd') bwchoice(`_rf_bwchoice')
		ereturn local vce `rf_vcestr'
		if ("`rf_vcetype'" != "") ereturn local vcetype `rf_vcetype'
		ereturn local hacsubtitleV "`rf_hacV'"
		if ("`rf_hacV2'" != "") ereturn local hacsubtitleV2 "`rf_hacV2'"
		if ("`rf_vcetype'" == "Robust") ereturn local title3 "Statistics robust to heteroskedasticity"
		else ereturn local title3 "Statistics robust to autocorrelation"
	}
	else {
		if ("`family'" == "logit") {
			ereturn local vce = cond("`vcetype'" == "unadjusted", "oim", "`vcetype'")
		}
		else {
			ereturn local vce `vcetype'
		}
		if inlist("`vcetype'", "robust", "cluster", "dkraay") ereturn local vcetype Robust
		if inlist("`vcetype'", "hc2", "hc3") ereturn local vcetype = "Robust " + upper("`vcetype'")
		if ("`vcetype'" == "crv3") ereturn local vcetype CRV3
		if ("`vcetype'" == "conley") ereturn local vcetype Conley
		if (`post_ncl' > 0 | "`vcetype'" == "conley") ereturn local title3 "Statistics robust to heteroskedasticity"
		if ("`vcetype'" == "dkraay") ereturn local title4 "and autocorrelation (Driscoll-Kraay)"
		if ("`vcetype'" == "conley") ereturn local title4 "and spatial correlation (Conley)"
	}
	if (`post_ncl' > 0) {
		forval i = 1/`post_ncl' {
			ereturn local clustvar`i' `: word `i' of `post_clusters''
		}
		ereturn local clustvar `post_clusters'
	}
	if ("`vcetype'" == "conley") _rf_conley_post, lat(`conley_lat') lon(`conley_lon') pixel(`conley_pixel') distance(`conley_distance')
	mata: st_global("e(indepvars)", invtokens((tokens(st_local("indepvars_e")), " _cons")))
	ereturn local depvar `depvar'
	if (!`is_noabsorb') ereturn local title2 "Absorbing `G' HDFE group`=cond(`G' == 1, "", "s")'"
	ereturn local footnote _rf_footnote
	ereturn local extended_absvars `"`extended_absvars'"'
	ereturn local absvars `"`absvars'"'
	ereturn local dofmethod "pairwise clusters continuous"
	ereturn local offset "`offvar_e'"
	if ("`weight'" != "") {
		ereturn local wexp "= `wvar'"
		ereturn local wtype `weight'
	}
	ereturn local cmd `cmd'
	if (`is_noabsorb') {
		ereturn local title = cond("`family'" == "poisson", "PPML regression", "Logistic regression")
	}

	ereturn matrix dof_table = `doft'
	mata: st_matrixrowstripe("e(dof_table)", (J(`num_slots', 1, ""), tokens(st_local("dof_rownames"))'))
	mata: st_matrixcolstripe("e(dof_table)", (J(5, 1, ""), ("Categories" \ "Redundant" \ "Num Coefs" \ "Exact?" \ "Nested?")))

	if ("`d'" != "") label variable `d' "Sum of fixed effects"
	local i 0
	foreach t of local fevars {
		local ++i
		local lab : word `i' of `fevar_labels'
		label variable `t' `"`lab'"'
	}
	* Display options for the caller's replay
	c_local rf_display `eform' `or' `header' `table' `footnote' `diopts'
end


program PostScalar, eclass
	args name
	capture confirm scalar __rf_`name'
	if (!_rc) ereturn scalar `name' = scalar(__rf_`name')
end


* 2.rep78 -> 2bn.rep78 ; 2.rep78#1b.foreign -> 2bn.rep78#1bn.foreign (as ms_expand_varlist; same as fereg.ado)
program AddBN
	local part `0'
	local re "^([0-9]+)b?([.LFSD])"
	if (regexm("`part'", `"`re'"')) {
		local part = regexr("`part'", "`re'", regexs(1) + "bn" + regexs(2))
	}
	local re "#([0-9]+)b?([.LFSD])"
	local loop = strpos("`part'", "#")
	local old `part'
	while (`loop') {
		if (regexm("`part'", `"`re'"')) {
			local part = regexr("`part'", "`re'", "#" + regexs(1) + "bn" + regexs(2))
		}
		local loop = "`old'" != "`part'"
		local old `part'
	}
	c_local part `part'
end


mata:
// ppmlhdfe's prettify_z(): rescale the certificate of separation by the mode of round(1e6 z) over
// z > 0 (first of the most frequent values in ascending order), when the factor is in [0.01, 100]
void _rf_prettify_z(string scalar zvar, string scalar touse)
{
	real colvector z, zz
	real scalar i, j, n, best, nbest, f

	st_view(z, ., zvar, touse)
	zz = sort(round(1e6 :* select(z, z :> 0)), 1)
	n = rows(zz)
	if (!n) return
	best = 1
	nbest = 0
	for (i = 1; i <= n; i = j + 1) {
		j = i
		while (j < n) {
			if (zz[j + 1] != zz[i]) break
			j++
		}
		if (j - i + 1 > nbest) {
			nbest = j - i + 1
			best = i
		}
	}
	f = 1e6 / zz[best]
	if (0.01 <= f & f <= 100) z[., .] = z :* f
}
end
