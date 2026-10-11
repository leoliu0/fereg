*! version 0.1.0  regfast: fast linear regression with high-dimensional fixed effects (Rust plugin)
*! Syntax, defaults, messages, display and e() mirror reghdfe 6.13.1 (Sergio Correia).
program define regfast, eclass
	version 16
	if replay() {
		if (`"`e(cmd)'"' != "regfast") error 301
		Replay `0'
		exit
	}
	capture noisily Estimate `0'
	local rc = _rc
	Cleanup
	exit `rc'
end


program Cleanup
	* Drop the __rf_* scalars written by the plugin
	mata: st_local("rf_scalars", invtokens(st_dir("global", "numscalar", "__rf_*")'))
	if (`"`rf_scalars'"' != "") scalar drop `rf_scalars'
end


program Replay, rclass
	syntax [, noHEader noTABle noFOOTnote *]
	_get_diopts options, `options'
	if ("`header'" == "") _rf_header
	if ("`header'" == "" & "`table'" == "") di ""
	if ("`table'" == "") _coef_table, `options'
	return add // adds r(level), r(table), etc.
	if ("`footnote'" == "") _rf_footnote
end


program Estimate, eclass
	local rf_version 0.1.0
	local cmdline `"regfast `0'"'

// --------------------------------------------------------------------------
// Syntax
// --------------------------------------------------------------------------
	#delimit ;
	syntax anything(id="varlist" equalok) [if] [in] [fw aw pw/] [,
		Absorb(string) NOAbsorb
		VCE(string) CLuster(string)
		RESiduals(name) RESiduals2
		TOLerance(real 1e-8) ITERate(integer 16000)
		KEEPSINgletons noCONstant
		DOFadjustments(string) GROUPVar(name)
		THReads(integer 0) Verbose(integer 0) SERIALio noWARN
		FIRST FFIRST
		noHEader noTABle noFOOTnote
		TEChnique(string) TRAnsform(string) ACCELeration(string) POOLsize(string)
		PREConditioner(string) COMPACT
		Individual_id(string) Group_id(string) AGgregation(string)
		ABSORBER(string)
		* ] ;
	#delimit cr
	* absorber(fast|dense) is an undocumented debugging option (dense = explicit-dummy reference)

// --------------------------------------------------------------------------
// Validate options
// --------------------------------------------------------------------------
	foreach opt in individual_id group_id aggregation {
		if (`"``opt''"' != "") {
			local optname = cond("`opt'" == "aggregation", "aggregation", subinstr("`opt'", "_id", "", 1))
			di as err "regfast: `optname'() not supported"
			exit 198
		}
	}
	foreach opt in technique transform acceleration poolsize preconditioner {
		if (`"``opt''"' != "") di as text "(regfast: option `opt'() ignored)"
	}
	if ("`compact'" != "") di as text "(regfast: option compact ignored)"

	_get_diopts diopts options, `options'
	if (`"`options'"' != "") {
		di as err `"option `options' not allowed"'
		exit 198
	}
	if ("`noabsorb'" != "" & `"`absorb'"' != "") {
		di as err "options absorb() and noabsorb are mutually exclusive"
		exit 198
	}
	if (`tolerance' <= 0 | `tolerance' >= 1) {
		di as err "tolerance() must be in (0, 1)"
		exit 198
	}
	if (`iterate' < 1) {
		di as err "iterate() must be a positive integer"
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

// --------------------------------------------------------------------------
// Model dispatch
// --------------------------------------------------------------------------
	* Instrumental-variable syntax -- depvar [exog] (endog = inst) -- is handled by the IV branch
	if (strpos(`"`anything'"', "(") | strpos(`"`anything'"', "=")) {
		di as err "regfast: instrumental-variable syntax (endog = instruments) is not available in this version"
		exit 198
	}
	if ("`first'`ffirst'" != "") {
		di as err "options first and ffirst require instrumental variables"
		exit 198
	}
	local model ols

	* Split varlist into <depvar> and <indepvars> (as ms_parse_varlist)
	* (-syntax- resets `if' `in' `weight' `exp', so keep them)
	foreach m in if in weight exp {
		local keep_`m' `"``m''"'
	}
	local 0 `anything'
	syntax varlist(fv ts numeric)
	foreach m in if in weight exp {
		local `m' `"`keep_`m''"'
	}
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
	local indepvars `indepvars' // trim

// --------------------------------------------------------------------------
// Weights, vce(), absorb(), dofadjustments(), output variables
// --------------------------------------------------------------------------
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
	_rf_parse_vce, vce(`vce') weighttype(`weight')
	local vcetype `s(vcetype)'
	local clustervars `s(clustervars)'
	local base_clustervars `s(base_clustervars)'
	local num_clusters `s(num_clusters)'

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
	local has_intercept `s(has_intercept)'
	local save_any_fe `s(save_any_fe)'
	local is_noabsorb = (`"`absvars'"' == "_cons")

	ParseDOF, `dofadjustments'
	local dofmethod `s(dofadjustments)'
	local dofmethod : list retokenize dofmethod
	local dofkey : subinstr local dofmethod " " ",", all
	if ("`dofkey'" == "") local dofkey none

	if ("`residuals'" != "" & "`residuals2'" != "") {
		di as err "only one of residuals() and residuals can be specified"
		exit 198
	}
	if ("`residuals2'" != "") {
		capture drop _reghdfe_resid // destructive, as reghdfe
		local residuals _reghdfe_resid
	}
	else if ("`residuals'" != "") {
		confirm new variable `residuals'
	}
	if ("`groupvar'" != "") confirm new variable `groupvar'

	if (`drop_singletons' == 0 & `verbose' > -1 & "`warn'" != "nowarn") {
		local url "http://scorreia.com/reghdfe/nested_within_cluster.pdf"
		di as error `"WARNING: Singleton observations not dropped; statistical significance is biased {browse "`url'":(link)}"'
	}

// --------------------------------------------------------------------------
// Sample
// --------------------------------------------------------------------------
	* if + in + weight + regression variables + cluster variables, then absorbed and slope variables
	local varlist `depvar' `indepvars' `base_clustervars'
	marksample touse, strok

	* Factor variables sent to the plugin: absorbed factors first, then cluster-only variables
	local factors `absfactors'
	foreach v of local base_clustervars {
		if (!`: list v in factors') local factors `factors' `v'
	}
	local nfac : word count `factors'
	local nslope : word count `slopevars'

	* Resolve time-series operators (L.x) into temporary variables
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
	if ("`facvars'`slopevars_r'" != "") markout `touse' `facvars' `slopevars_r', strok

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

	* cluster key: dimensions separated by ";", each f1*f2 (1-based positions)
	local clukey
	foreach dim of local clustervars {
		local parts : subinstr local dim "#" " ", all
		local cpos
		foreach v of local parts {
			local k : list posof "`v'" in factors
			local cpos = cond("`cpos'" == "", "`k'", "`cpos'*`k'")
		}
		local clukey = cond("`clukey'" == "", "`cpos'", "`clukey';`cpos'")
	}

	* Keys common to prepare and ols (every key=value is quoted: see protocol)
	local common `""threads=`threads'" "verbose=`plugin_verbose'""'
	if ("`serialio'" != "") local common `"`common' "serialio=1""'
	* absorber() selects the session's absorber: a prepare-only key
	local abskind
	if ("`absorber'" != "") local abskind `""absorber=`absorber'""'

// --------------------------------------------------------------------------
// Prepare: factorize, drop singletons, build the fixed-effects structure
// --------------------------------------------------------------------------
	_rf_load
	local args `""nfac=`nfac'" "nslope=`nslope'" "absorb=`abskey'" "vce=`vcetype'" "wtype=`wtype'" "singletons=`drop_singletons'" "family=gaussian""'
	if (`nfac' > 0) local args `"`args' "factype=`factype'""'
	if ("`vcetype'" == "cluster") local args `"`args' "cluster=`clukey'""'
	local _rf_token
	_rf_load `facvars' `slopevars_r' `wvar' `touse', prepare `args' `common' `abskind'
	local token `_rf_token'
	local num_singletons = scalar(__rf_num_singletons)
	if (`drop_singletons' & `num_singletons' > 0 & `verbose' > -1) {
		local ns = strtrim(string(`num_singletons', "%20.0f"))
		di as text `"(dropped `ns' {browse "http://scorreia.com/research/singletons.pdf":singleton observations})"'
	}

// --------------------------------------------------------------------------
// Regressors: expand factor variables over the final sample (as ms_expand_varlist)
// --------------------------------------------------------------------------
	local fullvarlist        // b colnames: 1b.rep78 2.rep78 ...
	local selected           // sent regressors, bn form: 2bn.rep78 ...
	local mask               // 1 = sent to the plugin, 0 = base/omitted (status 1)
	local xvars
	if ("`indepvars'" != "") {
		fvexpand `indepvars' if `touse'
		local fullvarlist `r(varlist)'
		local sent_full
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
			if (`: word count `xvars'' != `: word count `selected'') {
				di as err "regfast: internal error expanding regressors"
				exit 9999
			}
		}
	}
	local nx : word count `xvars'
	fvrevar `depvar'
	local yvar `r(varlist)'

	local report_constant = `has_intercept' & ("`constant'" != "noconstant")
	local K_all = `nx' + `report_constant'
	local collinear_tol = min(1e-6, `tolerance' / 10)

// --------------------------------------------------------------------------
// Output variables (pre-created; the plugin fills kept observations)
// --------------------------------------------------------------------------
	local outvars
	local nout 0
	local pos_resid 0
	local pos_group 0
	local pos_fe
	if ("`residuals'" != "") {
		qui gen double `residuals' = .
		local outvars `outvars' `residuals'
		local pos_resid = `++nout'
	}
	if ("`groupvar'" != "") {
		qui gen long `groupvar' = .
		local outvars `outvars' `groupvar'
		local pos_group = `++nout'
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
	local matargs `""doft=`doft'""'
	if (`K_all' > 0) {
		matrix `b' = J(1, `K_all', 0)
		matrix `V' = J(`K_all', `K_all', 0)
		local matargs `"`matargs' "b=`b'" "V=`V'""'
	}
	matrix `doft' = J(`num_slots', 5, 0)
	local tolstr = strtrim(string(`tolerance', "%21.17g"))
	local args `""token=`token'" "nx=`nx'" "nout=`nout'" "tol=`tolstr'" "maxiter=`iterate'" "dof=`dofkey'" "noconstant=`=("`constant'" == "noconstant")'""'
	local args `"`args' "resid=`pos_resid'" "d=0" "groupvar=`pos_group'" "fe=`pos_fe'" `matargs'"'
	local _rf_status
	capture noisily _rf_load `yvar' `xvars' `touse' `outvars', ols `args' `common'
	if (_rc) {
		local rc = _rc
		if ("`outvars'" != "") capture drop `outvars'
		exit `rc'
	}
	local status `_rf_status'
	if (`: word count `status'' != `nx') {
		if ("`outvars'" != "") capture drop `outvars'
		di as err "regfast: internal error: plugin returned `: word count `status'' status codes for `nx' regressors"
		exit 9999
	}

	* Messages that reference variable names (the plugin cannot see them)
	if (`verbose' > -1) {
		local ic = scalar(__rf_ic)
		di as text "(regfast converged in `ic' iteration" cond(`ic' == 1, "", "s") ")"
		local tolfmt = string(`collinear_tol', "%3.1e")
		capture confirm scalar __rf_y_absorbed
		if (!_rc) {
			if (scalar(__rf_y_absorbed) == 1) {
				di as text "warning: dependent variable {res}`depvar'{txt} is likely perfectly explained by the fixed effects (tol = `tolfmt')"
			}
		}
		forval j = 1/`nx' {
			if (`: word `j' of `status'' == 2) {
				di as text "note: {res}`: word `j' of `selected''{txt} is probably collinear with the fixed effects (all partialled-out values are close to zero; tol = `tolfmt')"
			}
		}
		forval j = 1/`nx' {
			if (`: word `j' of `status'' == 3) {
				di as text "note: {res}`: word `j' of `sent_full''{txt} omitted because of collinearity"
			}
		}
	}

// --------------------------------------------------------------------------
// Post results
// --------------------------------------------------------------------------
	* Expand b and V with the base/omitted levels that were not sent; add _cons
	local colnames `fullvarlist'
	local indepvars_e `selected'
	if (`report_constant') {
		local colnames `colnames' _cons
		local indepvars_e `indepvars_e' _cons
	}
	local K_full : word count `colnames'
	if (`K_full' > `K_all') {
		tempname bf Vf
		mata: _rf_m = (strtoreal(tokens(st_local("mask"))), J(1, `report_constant', 1))
		mata: _rf_i = selectindex(_rf_m)
		mata: _rf_B = J(1, cols(_rf_m), 0); _rf_W = J(cols(_rf_m), cols(_rf_m), 0)
		if (`K_all' > 0) {
			mata: _rf_B[_rf_i] = st_matrix(st_local("b")); _rf_W[_rf_i, _rf_i] = st_matrix(st_local("V"))
		}
		mata: st_matrix(st_local("bf"), _rf_B); st_matrix(st_local("Vf"), _rf_W)
		mata: mata drop _rf_m _rf_i _rf_B _rf_W
		local b `bf'
		local V `Vf'
	}

	local N = scalar(__rf_N)
	local df_r = scalar(__rf_df_r)
	if (`K_full' > 0) {
		matrix colnames `b' = `colnames'
		matrix colnames `V' = `colnames'
		matrix rownames `V' = `colnames'
		_ms_findomitted `b' `V'
		ereturn post `b' `V', esample(`touse') buildfvinfo depname(`depvar') obs(`N') dof(`df_r')
	}
	else {
		ereturn post, esample(`touse') buildfvinfo depname(`depvar') obs(`N') dof(`df_r')
	}

	* Scalars computed by the plugin (names = reghdfe e() names)
	foreach s in N rank df_r df_m tss tss_within rss mss rmse F ll ll_0 r2 r2_within r2_a r2_a_within ic converged {
		PostScalar `s'
	}
	capture confirm scalar __rf_sumweights
	if (!_rc) {
		if (!missing(scalar(__rf_sumweights))) ereturn scalar sumweights = scalar(__rf_sumweights)
	}
	ereturn scalar report_constant = `report_constant'
	if ("`vcetype'" == "cluster") {
		PostScalar N_clust
		ereturn scalar N_clustervars = `num_clusters'
		forval i = 1/`num_clusters' {
			PostScalar N_clust`i'
			local cv : word `i' of `clustervars'
			ereturn local clustvar`i' `cv'
		}
		ereturn local clustvar `clustervars'
		ereturn local title3 "Statistics robust to heteroskedasticity"
	}
	ereturn scalar N_hdfe = `G'
	ereturn scalar N_hdfe_extended = `num_slots'
	foreach s in df_a df_a_initial df_a_redundant df_a_nested {
		PostScalar `s'
	}
	ereturn scalar drop_singletons = `drop_singletons'
	ereturn scalar num_singletons = `num_singletons'
	ereturn scalar N_full = scalar(__rf_N_full)

	* Macros
	ereturn local cmd regfast
	ereturn local cmdline `"`cmdline'"'
	ereturn local model `model'
	ereturn local predict regfast_p
	ereturn local footnote _rf_footnote
	ereturn local marginsnotok "Residuals SCore"
	ereturn local depvar `depvar'
	ereturn local indepvars `indepvars_e'
	ereturn local title "HDFE Linear regression"
	ereturn local title2 "Absorbing `G' HDFE group`=cond(`G' == 1, "", "s")'"
	if ("`vcetype'" != "unadjusted") ereturn local vcetype Robust
	ereturn local vce = cond("`vcetype'" == "unadjusted", "ols", "`vcetype'")
	if ("`weight'" != "") {
		ereturn local wexp "= `wvar'"
		ereturn local wtype `weight'
	}
	ereturn local absvars `"`absvars'"'
	ereturn local extended_absvars `"`extended_absvars'"'
	ereturn local dofmethod `dofmethod'
	if ("`residuals'" != "") {
		ereturn local resid `residuals'
		label variable `residuals' "Residuals"
	}

	* Absorbed degrees-of-freedom table
	ereturn matrix dof_table = `doft'
	mata: st_matrixrowstripe("e(dof_table)", (J(`num_slots', 1, ""), tokens(st_local("dof_rownames"))'))
	mata: st_matrixcolstripe("e(dof_table)", (J(5, 1, ""), ("Categories" \ "Redundant" \ "Num Coefs" \ "Inexact?" \ "Nested?")))

	* Labels of saved fixed effects and of the mobility-group variable
	local i 0
	foreach t of local fevars {
		local ++i
		local lab : word `i' of `fevar_labels'
		label variable `t' `"`lab'"'
	}
	if ("`groupvar'" != "") {
		qui count if !missing(`groupvar')
		if (r(N) == 0) {
			drop `groupvar' // no pair of intercept FEs: reghdfe does not create the variable
		}
		else {
			local pair
			forval g = 1/`G' {
				if (`: word `g' of `intercepts'' & `: word count `pair'' < 2) {
					local iv : word `g' of `all_ivars'
					local iv : subinstr local iv " " "#", all
					local pair `pair' `iv'
				}
			}
			label variable `groupvar' "Mobility group between `: word 1 of `pair'' and `: word 2 of `pair''"
		}
	}

	Replay, `diopts' `header' `table' `footnote'
end


program PostScalar, eclass
	args name
	capture confirm scalar __rf_`name'
	if (!_rc) ereturn scalar `name' = scalar(__rf_`name')
end


program ParseDOF, sclass
	syntax, [ALL NONE] [FIRSTpair PAIRwise] [CLusters] [CONTinuous]

	* Select all methods by default
	if ("`none'`firstpair'`pairwise'`clusters'`continuous'" == "") local all "all"

	opts_exclusive "`all' `none' `firstpair' `pairwise'" dofadjustments
	opts_exclusive "`all' `none' `clusters'" dofadjustments
	opts_exclusive "`all' `none' `continuous'" dofadjustments

	local opts `pairwise' `firstpair' `clusters' `continuous'
	if ("`all'" != "") local opts pairwise clusters continuous
	sreturn local dofadjustments "`opts'"
end


* 2.rep78 -> 2bn.rep78 ; 2.rep78#1b.foreign -> 2bn.rep78#1bn.foreign (as ms_expand_varlist)
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
