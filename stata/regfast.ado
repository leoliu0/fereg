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
	syntax [, noHEader noTABle noFOOTnote FIRST FFIRST *]
	_get_diopts options, `options'
	if ("`e(model)'" == "iv") {
		ReplayIV, `header' `table' `footnote' `first' `ffirst' `options'
		return add
		exit
	}
	if ("`e(model)'" == "first") {
		ReplayFirst, `options'
		return add
		exit
	}
	if ("`first'`ffirst'" != "") {
		di as err "options first and ffirst require instrumental variables"
		exit 198
	}
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
		FIRST FFIRST LIML GMM2s CUE FULLer(string) Kclass(string)
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
	* Instrumental-variable syntax: depvar [exog] (endog = inst) [exog]
	if ("`liml'`gmm2s'`cue'" != "" | `"`fuller'`kclass'"' != "") {
		di as err "regfast: only 2SLS is supported"
		exit 198
	}
	local model ols
	local endogvars
	local instvars
	if (strpos(`"`anything'"', "(") | strpos(`"`anything'"', "=")) {
		local model iv
		ParseIV `anything'
		local anything `"`s(lhs)'"'
		local endogvars `"`s(endog)'"'
		local instvars `"`s(inst)'"'
	}
	if ("`first'`ffirst'" != "" & "`model'" != "iv") {
		di as err "options first and ffirst require instrumental variables"
		exit 198
	}

	* Split varlist into <depvar> and <indepvars> (as ms_parse_varlist)
	* (-syntax- resets `if' `in' `weight' `exp', so keep them)
	foreach m in if in weight exp {
		local keep_`m' `"``m''"'
	}
	local 0 `anything'
	syntax varlist(fv ts numeric)
	if ("`model'" == "iv") {
		local 0 `endogvars'
		syntax varlist(fv ts numeric)
		local endogvars `varlist'
		local 0 `instvars'
		syntax varlist(fv ts numeric)
		local instvars `varlist'
		local 0 `anything'
		syntax varlist(fv ts numeric)
	}
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
	local varlist `depvar' `indepvars' `endogvars' `instvars' `base_clustervars'
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
	ExpandFV `indepvars', touse(`touse')
	local fullvarlist `rf_full'          // b colnames: 1b.rep78 2.rep78 ...
	local selected `rf_selected'         // sent regressors, bn form: 2bn.rep78 ...
	local sent_full `rf_sent_full'
	local mask `rf_mask'                 // 1 = sent to the plugin, 0 = base/omitted (status 1)
	local xvars
	if ("`selected'" != "") {
		fvrevar `selected'
		local xvars `r(varlist)'
	}
	if ("`model'" == "iv") {
		foreach part in endog inst {
			ExpandFV ``part'vars', touse(`touse')
			local `part'_full `rf_full'
			local `part'_selected `rf_selected'
			local `part'_sent_full `rf_sent_full'
			local `part'_mask `rf_mask'
			local `part'_revars
			if ("`rf_selected'" != "") {
				fvrevar `rf_selected'
				local `part'_revars `r(varlist)'
			}
		}
		local nendog : word count `endog_revars'
		local ninst : word count `inst_revars'
		if (`nendog' == 0 | `ninst' == 0) {
			di as err "regfast: endogenous regressors and excluded instruments must not be empty or omitted"
			exit 198
		}
		if (`num_clusters' > 2) {
			di as err "regfast: IV supports at most two cluster variables"
			exit 198
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
// IV/2SLS (ivreghdfe 1.1.4 conventions): estimate, post and display
// --------------------------------------------------------------------------
	if ("`model'" == "iv") {
		tempname b V doft S W firstm ccev cdev sy fsb fsV fss
		local nz = `ninst' + `nx'
		local K_all = `nendog' + `nx'
		matrix `b' = J(1, `K_all', 0)
		matrix `V' = J(`K_all', `K_all', 0)
		matrix `S' = J(`nz', `nz', 0)
		matrix `W' = J(`nz', `nz', 0)
		matrix `ccev' = J(1, `nendog', .)
		matrix `cdev' = J(1, `nendog', .)
		matrix `doft' = J(`num_slots', 5, 0)
		local matargs `""b=`b'" "V=`V'" "doft=`doft'" "S=`S'" "W=`W'" "ccev=`ccev'" "cdev=`cdev'""'
		if ("`first'`ffirst'" != "") {
			matrix `firstm' = J(21, `nendog', .)
			local matargs `"`matargs' "firstm=`firstm'""'
		}
		if ("`first'" != "") {
			* first-stage regressions land in `fsb'#, `fsV'#, `fss'# (# = endogenous position)
			forval i = 1/`nendog' {
				matrix `fsb'`i' = J(1, `nz', .)
				matrix `fsV'`i' = J(`nz', `nz', .)
				matrix `fss'`i' = J(1, 4, .)
			}
			local matargs `"`matargs' "bfirst=`fsb'" "Vfirst=`fsV'" "sfirst=`fss'""'
		}
		local tolstr = strtrim(string(`tolerance', "%21.17g"))
		local args `""token=`token'" "nx=`nx'" "nendog=`nendog'" "ninst=`ninst'" "nout=`nout'" "tol=`tolstr'" "maxiter=`iterate'" "dof=`dofkey'""'
		local args `"`args' "first=`=("`first'" != "")'" "ffirst=`=("`ffirst'" != "")'""'
		local args `"`args' "resid=`pos_resid'" "d=0" "groupvar=`pos_group'" "fe=`pos_fe'" `matargs'"'
		local _rf_status
		capture noisily _rf_load `yvar' `xvars' `endog_revars' `inst_revars' `touse' `outvars', iv `args' `common'
		if (_rc) {
			local rc = _rc
			if ("`outvars'" != "") capture drop `outvars'
			DropFirstMats `fsb' `fsV' `fss' `nendog'
			exit `rc'
		}
		local status `_rf_status'
		local nsent = `nx' + `nendog' + `ninst'
		if (`: word count `status'' != `nsent') {
			if ("`outvars'" != "") capture drop `outvars'
			DropFirstMats `fsb' `fsV' `fss' `nendog'
			di as err "regfast: internal error: plugin returned `: word count `status'' status codes for `nsent' variables"
			exit 9999
		}
		* Stock-Yogo critical values (row 1: L1 and K1; row 2: L1 and one endogenous regressor)
		local sytables ivbias5 ivbias10 ivbias20 ivbias30 ivsize10 ivsize15 ivsize20 ivsize25
		matrix `sy' = J(2, 8, .)
		local j 0
		foreach t of local sytables {
			local ++j
			if ("`_rf_sy_`t''" != "") matrix `sy'[1, `j'] = `_rf_sy_`t''
			if ("`_rf_sy1_`t''" != "") matrix `sy'[2, `j'] = `_rf_sy1_`t''
		}
		matrix colnames `sy' = `sytables'
		matrix rownames `sy' = K1 one_endog

		* Status codes follow the plugin layout: exog, endog, inst
		local sel_all `selected' `endog_selected' `inst_selected'
		local full_all `sent_full' `endog_sent_full' `inst_sent_full'
		local exog1
		local endog1
		local inst1
		local collin
		local stat_exog
		local stat_endog
		local stat_inst
		forval j = 1/`nsent' {
			local st : word `j' of `status'
			local nm : word `j' of `full_all'
			local grp = cond(`j' <= `nx', "exog", cond(`j' <= `nx' + `nendog', "endog", "inst"))
			local stat_`grp' `stat_`grp'' `st'
			if (`st' == 0) local `grp'1 ``grp'1' `nm'
			else local collin `collin' `nm'
		}
		local k1 : word count `endog1'
		if (`verbose' > -1) {
			local ic = scalar(__rf_ic)
			di as text "(regfast converged in `ic' iteration" cond(`ic' == 1, "", "s") ")"
			local tolfmt = string(`collinear_tol', "%3.1e")
			forval j = 1/`nsent' {
				local st : word `j' of `status'
				if (`st' == 2) di as text "note: {res}`: word `j' of `sel_all''{txt} is probably collinear with the fixed effects (all partialled-out values are close to zero; tol = `tolfmt')"
				if (`st' == 3) di as text "note: {res}`: word `j' of `full_all''{txt} omitted because of collinearity"
			}
		}

		local N = scalar(__rf_N)
		local df_r = scalar(__rf_df_r)
		if ("`vcetype'" == "unadjusted") {
			local hacV "Statistics consistent for homoskedasticity only"
			local vcestr
		}
		else if ("`vcetype'" == "robust") {
			local hacV "Statistics robust to heteroskedasticity"
			local vcestr robust
		}
		else if (`num_clusters' == 1) {
			local hacV "Statistics robust to heteroskedasticity and clustering on `clustervars'"
			local vcestr robust cluster
		}
		else {
			local hacV "Statistics robust to heteroskedasticity and clustering on `: word 1 of `clustervars'' and `: word 2 of `clustervars''"
			local vcestr robust two-way cluster
		}

		* First-stage regressions: posted and stored as _regfast_<endog> before the main results
		local firsteqs
		if ("`first'" != "") {
			local zfull `inst_full' `fullvarlist'
			local zmask `inst_mask' `mask'
			local i 0
			foreach en of local endog_sent_full {
				local ++i
				if (`: word `i' of `stat_endog'' != 0) continue
				tempvar fsample
				qui gen byte `fsample' = `touse'
				ExpandBV `fsb'`i' `fsV'`i' "`zmask'"
				matrix colnames `fsb'`i' = `zfull'
				matrix colnames `fsV'`i' = `zfull'
				matrix rownames `fsV'`i' = `zfull'
				_ms_findomitted `fsb'`i' `fsV'`i'
				local fdf = `fss'`i'[1, 2]
				local frmse = `fss'`i'[1, 1]
				local fdfm = `fss'`i'[1, 3]
				ereturn post `fsb'`i' `fsV'`i', esample(`fsample') depname(`en') obs(`N') dof(`fdf')
				ereturn scalar rmse = `frmse'
				ereturn scalar df_r = `fdf'
				ereturn scalar df_m = `fdfm'
				ereturn scalar k_eq = `k1'
				ereturn local cmd regfast
				ereturn local model first
				ereturn local depvar `en'
				ereturn local title "First-stage regression: `en'"
				ereturn local hacsubtitleV "`hacV'"
				if ("`vcestr'" != "") {
					ereturn local vce `vcestr'
					ereturn local vcetype Robust
				}
				if (`num_clusters' > 0) {
					ereturn local clustvar `clustervars'
					PostScalar N_clust
					if (`num_clusters' == 2) {
						ereturn local clustvar1 `: word 1 of `clustervars''
						ereturn local clustvar2 `: word 2 of `clustervars''
						PostScalar N_clust1
						PostScalar N_clust2
					}
				}
				local eqname = substr("_regfast_" + subinstr("`en'", ".", "_", .), 1, 27)
				capture estimates store `eqname', title("First-stage regression: `en'")
				if (_rc) di as text "Unable to store first-stage regression of `en'."
				else local firsteqs `firsteqs' `eqname'
			}
		}

		* Main equation: e(b) over [endog, exog] with base/omitted levels added back
		local colnames `endog_full' `fullvarlist'
		ExpandBV `b' `V' "`endog_mask' `mask'"
		matrix colnames `b' = `colnames'
		matrix colnames `V' = `colnames'
		matrix rownames `V' = `colnames'
		_ms_findomitted `b' `V'
		ereturn post `b' `V', esample(`touse') buildfvinfo depname(`depvar') obs(`N') dof(`df_r')

		* Scalars computed by the plugin (names = ivreghdfe e() names)
		foreach s in N df_r df_m sdofminus dofminus center partial_ct partialcons cons nocollin ///
			inexog_ct exexog_ct endog_ct r2 r2u r2c r2_a rmse rss mss yy yyc ll F Fp Fdf1 Fdf2 ///
			condxx condzz rankxx rankzz rankS rankV j jdf jp sargan sargandf sarganp ///
			N_clust N_clust1 N_clust2 idstat iddf idp cd cdf widstat rkf ///
			arf arfp archi2 archi2p ardf ardf_r sstat sstatp sstatdf df_a df_a_initial df_a_redundant df_a_nested {
			PostScalar `s'
		}
		ereturn scalar N_hdfe = `G'
		ereturn scalar N_hdfe_extended = `num_slots'
		ereturn scalar drop_singletons = `drop_singletons'
		ereturn scalar num_singletons = `num_singletons'
		ereturn scalar N_full = scalar(__rf_N_full)

		* Macros (ivreghdfe names); X = expanded list, X0 = the same with dropped variables marked
		* omitted (base levels stay), X1 = used
		local endog0
		local exog0
		local inst0
		foreach grp in exog endog inst {
			local list = cond("`grp'" == "exog", "fullvarlist", "`grp'_full")
			local mlist = cond("`grp'" == "exog", "mask", "`grp'_mask")
			local i 0
			local k 0
			foreach v of local `list' {
				local ++i
				if (`: word `i' of ``mlist''' == 1) {
					local ++k
					if (`: word `k' of `stat_`grp''' != 0) {
						_ms_put_omit `v'
						local v `s(ospec)'
					}
				}
				local `grp'0 ``grp'0' `v'
			}
		}
		ereturn local cmd regfast
		ereturn local cmdline `"`cmdline'"'
		ereturn local model iv
		ereturn local predict regfast_p
		ereturn local footnote _rf_footnote
		ereturn local title "IV (2SLS) estimation"
		ereturn local title2 "Absorbing `G' HDFE group`=cond(`G' == 1, "", "s")'"
		ereturn local small small
		ereturn local partialsmall small
		ereturn local ranktestcmd ranktest
		ereturn local marginsok "XB default"
		ereturn local marginsnotok "Residuals SCores"
		ereturn local hacsubtitleB "Estimates efficient for homoskedasticity only"
		ereturn local hacsubtitleV "`hacV'"
		if ("`vcestr'" != "") {
			ereturn local vce `vcestr'
			ereturn local vcetype Robust
		}
		if (`num_clusters' > 0) {
			ereturn local clustvar `clustervars'
			if (`num_clusters' == 2) {
				ereturn local clustvar1 `: word 1 of `clustervars''
				ereturn local clustvar2 `: word 2 of `clustervars''
			}
		}
		if ("`weight'" != "") {
			ereturn local wexp "=`wvar'"
			ereturn local wtype `weight'
		}
		foreach x in depvar depvar0 depvar1 {
			ereturn local `x' `depvar'
		}
		ereturn local instd `endog_full'
		ereturn local instd0 `endog0'
		ereturn local instd1 `endog1'
		ereturn local inexog `fullvarlist'
		ereturn local inexog0 `exog0'
		ereturn local inexog1 `exog1'
		ereturn local exexog `inst_full'
		ereturn local exexog0 `inst0'
		ereturn local exexog1 `inst1'
		ereturn local insts `inst_full' `fullvarlist'
		ereturn local insts0 `inst0' `exog0'
		ereturn local insts1 `inst1' `exog1'
		if ("`collin'" != "") ereturn local collin `collin'
		ereturn local firsteqs `firsteqs'
		ereturn local absvars `"`absvars'"'
		ereturn local extended_absvars `"`extended_absvars'"'
		ereturn local dofmethod `dofmethod'
		if ("`residuals'" != "") {
			ereturn local resid `residuals'
			label variable `residuals' "Residuals"
		}

		* Matrices over the variables actually used
		local zstat `stat_inst' `stat_exog'
		mata: _rf_k = selectindex(strtoreal(tokens(st_local("zstat"))) :== 0)
		mata: st_matrix(st_local("S"), st_matrix(st_local("S"))[_rf_k, _rf_k]); st_matrix(st_local("W"), st_matrix(st_local("W"))[_rf_k, _rf_k])
		mata: mata drop _rf_k
		foreach m in S W {
			matrix colnames ``m'' = `inst1' `exog1'
			matrix rownames ``m'' = `inst1' `exog1'
			ereturn matrix `m' = ``m''
		}
		if (`k1' > 0) {
			mata: _rf_k = selectindex(strtoreal(tokens(st_local("stat_endog"))) :== 0)
			if ("`first'`ffirst'" != "") {
				mata: st_matrix(st_local("firstm"), st_matrix(st_local("firstm"))[., _rf_k])
				matrix rownames `firstm' = rmse sheapr2 pr2 F df df_r pvalue SWF SWFdf1 SWFdf2 SWFp SWchi2 SWchi2p SWr2 ///
					APF APFdf1 APFdf2 APFp APchi2 APchi2p APr2
				matrix colnames `firstm' = `endog1'
				ereturn matrix first = `firstm'
			}
			mata: mata drop _rf_k
			if (!missing(`ccev'[1, 1])) {
				matrix `ccev' = `ccev'[1, 1..`k1']
				matrix `cdev' = `cdev'[1, 1..`k1']
				ereturn matrix ccev = `ccev'
				ereturn matrix cdev = `cdev'
			}
		}
		ereturn matrix stock_yogo = `sy'
		DropFirstMats `fsb' `fsV' `fss' `nendog'

		PostTail, doft(`doft') nslots(`num_slots') dofrows(`dof_rownames') fevars(`fevars') ///
			felabels(`fevar_labels') groupvar(`groupvar') intercepts(`intercepts') ivars(`all_ivars')
		Replay, `diopts' `header' `table' `footnote' `first' `ffirst'
		exit
	}

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

	PostTail, doft(`doft') nslots(`num_slots') dofrows(`dof_rownames') fevars(`fevars') ///
		felabels(`fevar_labels') groupvar(`groupvar') intercepts(`intercepts') ivars(`all_ivars')

	Replay, `diopts' `header' `table' `footnote'
end


program PostScalar, eclass
	args name
	capture confirm scalar __rf_`name'
	if (!_rc) ereturn scalar `name' = scalar(__rf_`name')
end


* Posts e(dof_table) and labels the saved fixed effects and the mobility-group variable (OLS and IV)
program PostTail, eclass
	syntax, doft(name) nslots(integer) [dofrows(string) fevars(string) felabels(string asis) ///
		groupvar(string) intercepts(string) ivars(string asis)]
	ereturn matrix dof_table = `doft'
	mata: st_matrixrowstripe("e(dof_table)", (J(`nslots', 1, ""), tokens(st_local("dofrows"))'))
	mata: st_matrixcolstripe("e(dof_table)", (J(5, 1, ""), ("Categories" \ "Redundant" \ "Num Coefs" \ "Inexact?" \ "Nested?")))

	* Labels of saved fixed effects and of the mobility-group variable
	local i 0
	foreach t of local fevars {
		local ++i
		local lab : word `i' of `felabels'
		label variable `t' `"`lab'"'
	}
	if ("`groupvar'" != "") {
		qui count if !missing(`groupvar')
		if (r(N) == 0) {
			drop `groupvar' // no pair of intercept FEs: reghdfe does not create the variable
		}
		else {
			local pair
			local G : word count `intercepts'
			forval g = 1/`G' {
				if (`: word `g' of `intercepts'' & `: word count `pair'' < 2) {
					local iv : word `g' of `ivars'
					local iv : subinstr local iv " " "#", all
					local pair `pair' `iv'
				}
			}
			label variable `groupvar' "Mobility group between `: word 1 of `pair'' and `: word 2 of `pair''"
		}
	}
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


* ParseIV <depvar [exog] (endog = inst) [exog]> : s(lhs) depvar and exogenous, s(endog), s(inst)
program ParseIV, sclass
	local s `"`0'"'
	local p1 = strpos(`"`s'"', "(")
	local p2 = strpos(`"`s'"', ")")
	if (`p1' == 0 | `p2' < `p1') {
		di as err "regfast: instrumental-variable syntax is depvar [exogvars] (endogvars = instruments)"
		exit 198
	}
	local before = substr(`"`s'"', 1, `p1' - 1)
	local inside = substr(`"`s'"', `p1' + 1, `p2' - `p1' - 1)
	local after = substr(`"`s'"', `p2' + 1, .)
	if (strpos(`"`after'"', "(") | strpos(`"`after'"', ")") | strpos(`"`before'"', "=")) {
		di as err "regfast: only one (endogvars = instruments) group is allowed"
		exit 198
	}
	local eq = strpos(`"`inside'"', "=")
	if (`eq' == 0 | strpos(substr(`"`inside'"', `eq' + 1, .), "=")) {
		di as err "regfast: instrumental-variable syntax is depvar [exogvars] (endogvars = instruments)"
		exit 198
	}
	local endog = strtrim(substr(`"`inside'"', 1, `eq' - 1))
	local inst = strtrim(substr(`"`inside'"', `eq' + 1, .))
	if (`"`endog'"' == "" | `"`inst'"' == "") {
		di as err "regfast: both endogenous variables and excluded instruments are required"
		exit 198
	}
	sreturn local lhs `before' `after'
	sreturn local endog `endog'
	sreturn local inst `inst'
end


* ExpandFV varlist, touse() : fvexpand over the sample (as ms_expand_varlist); c_locals
*   rf_full (all levels, e(b) names), rf_selected (sent, bn form), rf_sent_full, rf_mask (1 = sent)
* (fvrevar must run in the caller: its temporary variables belong to the calling program)
program ExpandFV
	syntax [anything], touse(name)
	local full
	local selected
	local sent_full
	local mask
	if (`"`anything'"' != "") {
		fvexpand `anything' if `touse'
		local full `r(varlist)'
		foreach part of local full {
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
	}
	c_local rf_full `full'
	c_local rf_selected `selected'
	c_local rf_sent_full `sent_full'
	c_local rf_mask `mask'
end


* ExpandBV b V "mask" : expands b (1×k) and V (k×k) in place to the length of mask (1 = present)
program ExpandBV
	args b V mask
	mata: _rf_m = strtoreal(tokens(st_local("mask"))); _rf_i = selectindex(_rf_m)
	mata: _rf_B = J(1, cols(_rf_m), 0); _rf_W = J(cols(_rf_m), cols(_rf_m), 0)
	mata: _rf_B[_rf_i] = st_matrix(st_local("b")); _rf_W[_rf_i, _rf_i] = st_matrix(st_local("V"))
	mata: st_matrix(st_local("b"), _rf_B); st_matrix(st_local("V"), _rf_W)
	mata: mata drop _rf_m _rf_i _rf_B _rf_W
end


program DropFirstMats
	args b V s n
	forval i = 1/`n' {
		capture matrix drop `b'`i' `V'`i' `s'`i'
	}
end


* ---------------------------------------------------------------------------------------------
* IV display (ivreg2/ivreghdfe 1.1.4 layout: DispMain, DispFirst, DispFFirst, DispSFirst)
* ---------------------------------------------------------------------------------------------
program ReplayIV, rclass
	syntax [, noHEader noTABle noFOOTnote FIRST FFIRST *]
	if ("`first'" != "") DispFirstIV
	if ("`first'`ffirst'" != "") DispFFirstIV
	if ("`header'" == "") DispHeaderIV
	if ("`table'" == "") NoClustNote _coef_table, `options'
	return add
	if ("`footnote'" == "") {
		DispFooterIV
		_rf_footnote
	}
end


program ReplayFirst, rclass
	syntax [, *]
	di
	di as text "First-stage regression of `e(depvar)':"
	if ("`e(hacsubtitleV)'" != "") di as text _n "`e(hacsubtitleV)'"
	di as text "Number of obs = " _col(31) as res %8.0f e(N)
	if ("`e(clustvar)'" != "") {
		if ("`e(clustvar2)'" == "") di as text "Number of clusters (`e(clustvar)') = " _col(33) as res %6.0f e(N_clust)
		else {
			di as text "Number of clusters (`e(clustvar1)') = " _col(33) as res %6.0f e(N_clust1)
			di as text "Number of clusters (`e(clustvar2)') = " _col(33) as res %6.0f e(N_clust2)
		}
	}
	* e(k_eq) = K1 as ivreghdfe posts it; one equation is shown (as ivreghdfe's DispSFirst)
	NoClustNote ereturn display, neq(1) `options'
	return add
end


* NoClustNote <display command> : runs it with e(clustvar) cleared, as ivreg2 does to suppress
* the "Std. err. adjusted for # clusters" line (the header already reports the clusters)
program NoClustNote, eclass
	local cluster `e(clustvar)'
	ereturn local clustvar
	capture noisily `0'
	local rc = _rc
	ereturn local clustvar `cluster'
	exit `rc'
end


* DispList varlist, col(#) : variable list wrapped at 80 columns (ivreg2 Disp)
program DispList
	syntax [anything] [, col(integer 15)]
	local maxlen = 80 - `col'
	local len 0
	local first 1
	foreach vn in `anything' {
		_ms_parse_parts `vn'
		if (!r(omit)) {
			local vnlen : length local vn
			if (`len' + `vnlen' > `maxlen') {
				di
				local first 1
				local len = `vnlen'
			}
			else local len = `len' + `vnlen' + 1
			if (`first') {
				local first 0
				di as text _col(`col') "`vn'" _c
			}
			else di as text " `vn'" _c
		}
	}
	di
end


* DispSY, row(#) col1(#) col2(#) : Stock-Yogo critical values of e(stock_yogo); r(cdmissing)
program DispSY, rclass
	syntax, row(integer) col1(integer) col2(integer)
	local labels `"" 5% maximal IV relative bias" "10% maximal IV relative bias" "20% maximal IV relative bias" "30% maximal IV relative bias" "10% maximal IV size" "15% maximal IV size" "20% maximal IV size" "25% maximal IV size""'
	local missing 1
	tempname sy
	capture matrix `sy' = e(stock_yogo)
	if (!_rc) {
		forval j = 1/8 {
			if (!missing(`sy'[`row', `j'])) {
				local lab : word `j' of `labels'
				di as text _col(`col1') "`lab'" as res _col(`col2') %6.2f `sy'[`row', `j']
				local missing 0
			}
		}
	}
	return scalar cdmissing = `missing'
end


program DispHeaderIV
	di as text _n "`e(title)'"
	di as text "{hline `=length("`e(title)'")'}"
	if ("`e(hacsubtitleB)'" != "") di as text _n "`e(hacsubtitleB)'" _c
	if ("`e(hacsubtitleV)'" != "") di as text _n "`e(hacsubtitleV)'"
	di
	if ("`e(clustvar)'" != "") {
		if ("`e(clustvar2)'" == "") {
			local N_clust = e(N_clust)
			local clustvar `e(clustvar)'
		}
		else {
			local N_clust = e(N_clust1)
			local clustvar `e(clustvar1)'
		}
		di as text "Number of clusters (`clustvar') = " _col(33) as res %6.0f `N_clust' _c
	}
	di as text _col(55) "Number of obs = " as res %8.0f e(N)
	if ("`e(clustvar2)'" != "") di as text "Number of clusters (`e(clustvar2)') = " _col(33) as res %6.0f e(N_clust2) _c
	di as text _c _col(55) "F(" %3.0f e(Fdf1) "," %6.0f e(Fdf2) ") = "
	if (e(F) < 99999) di as res %8.2f e(F)
	else di as res %8.2e e(F)
	di as text _col(55) "Prob > F      = " as res %8.4f e(Fp)
	di as text "Total (centered) SS     = " as res %12.0g e(yyc) _c
	di as text _col(55) "Centered R2   = " as res %8.4f e(r2c)
	di as text "Total (uncentered) SS   = " as res %12.0g e(yy) _c
	di as text _col(55) "Uncentered R2 = " as res %8.4f e(r2u)
	di as text "Residual SS             = " as res %12.0g e(rss) _c
	di as text _col(55) "Root MSE      = " as res %8.4g e(rmse)
	di
end


program DispFooterIV
	local robust = ("`e(vcetype)'" != "")
	if ("`e(instd1)'" != "" & !missing(e(idstat))) {
		di as text _c "Underidentification test"
		if (!`robust') di as text _c " (Anderson canon. corr. LM statistic):"
		else di as text _c " (Kleibergen-Paap rk LM statistic):"
		di as res _col(71) %8.3f e(idstat)
		di as text _col(52) "Chi-sq(" as res e(iddf) as text ") P-val =  " as res _col(73) %6.4f e(idp)
		di as text "{hline 78}"
		di as text "Weak identification test (Cragg-Donald Wald F statistic):" as res _col(71) %8.3f e(cdf)
		if (`robust') di as text "                         (Kleibergen-Paap rk Wald F statistic):" as res _col(71) %8.3f e(widstat)
		di as text _c "Stock-Yogo weak ID test critical values:"
		DispSY, row(1) col1(42) col2(73)
		if (r(cdmissing)) di as text _col(64) "<not available>"
		else {
			di as text "Source: Stock-Yogo (2005).  Reproduced by permission."
			if (`robust') di as text "NB: Critical values are for Cragg-Donald F statistic and i.i.d. errors."
		}
		di as text "{hline 78}"
	}
	if (!missing(e(j))) {
		if (`robust') di as text _c "Hansen J statistic (overidentification test of all instruments):"
		else di as text _c "Sargan statistic (overidentification test of all instruments):"
		di as res _col(71) %8.3f e(j)
		if (e(jdf)) di as text _col(52) "Chi-sq(" as res e(jdf) as text ") P-val =  " as res _col(73) %6.4f e(jp)
		else di as text _col(50) "(equation exactly identified)"
	}
	else {
		di as err "Warning: estimated covariance matrix of moment conditions not of full rank."
		di as err "         overidentification statistic not reported, and standard errors and"
		di as err "         model tests should be interpreted with caution."
		di as err "Possible causes:"
		if ("`e(N_clust)'" != "") di as err "         number of clusters insufficient to calculate robust covariance matrix"
		di as err "         singleton dummy variable (dummy with one 1 and N-1 0s or vice versa)"
	}
	di as text "{hline 78}"
	if (e(endog_ct)) {
		di as text "Instrumented:" _c
		DispList `e(instd1)', col(23)
	}
	if (e(inexog_ct)) {
		di as text "Included instruments:" _c
		DispList `e(inexog1)', col(23)
	}
	if (e(exexog_ct)) {
		di as text "Excluded instruments:" _c
		DispList `e(exexog1)', col(23)
	}
	di as text "Partialled-out:" _c
	DispList _cons, col(23)
	di as text _col(23) "nb: total SS, model F and R2s are after partialling-out;"
	di as text _col(23) "    any small-sample adjustments include partialled-out"
	di as text _col(23) "    variables in regressor count K"
	if ("`e(collin)'" != "") {
		di as text "Dropped collinear:" _c
		DispList `e(collin)', col(23)
	}
	di as text "{hline 78}"
end


program DispFirstIV
	tempname fm ivest
	capture matrix `fm' = e(first)
	if (_rc) {
		di _n as text "Unable to display first-stage estimates; e(first) is missing"
		exit
	}
	di as text _n "First-stage regressions"
	di as text "{hline 23}"
	di
	local i 0
	foreach eqname in `e(firsteqs)' {
		local ++i
		_estimates hold `ivest', copy
		capture estimates restore `eqname'
		if (_rc) {
			di _n as text "Unable to list stored estimation `eqname'."
		}
		else {
			estimates replay `eqname', noheader
			di as text "F test of excluded instruments:"
			di as text "  F(" %3.0f `fm'[5, `i'] "," %6.0f `fm'[6, `i'] ") = " as res %8.2f `fm'[4, `i']
			di as text "  Prob > F      = " as res %8.4f `fm'[7, `i']
			di as text "Sanderson-Windmeijer multivariate F test of excluded instruments:"
			di as text "  F(" %3.0f `fm'[9, `i'] "," %6.0f `fm'[10, `i'] ") = " as res %8.2f `fm'[8, `i']
			di as text "  Prob > F      = " as res %8.4f `fm'[11, `i']
			di
		}
		_estimates unhold `ivest'
	}
end


program DispFFirstIV
	tempname fm
	capture matrix `fm' = e(first)
	if (_rc) {
		di _n as text "Unable to display summary of first-stage estimates; e(first) is missing"
		exit
	}
	local endo : colnames `fm'
	local robust "`e(vcetype)'"
	local cluster "`e(clustvar)'"
	di
	di as text _n "Summary results for first-stage regressions"
	di as text "{hline 43}"
	di
	di _c as text _col(44) "(Underid)"
	di as text _col(65) "(Weak id)"
	di _c as text "Variable     |"
	di _c as text _col(16) "F("
	di _c as res _col(17) %3.0f `fm'[5, 1] as text "," as res %6.0f `fm'[6, 1] as text ")  P-val"
	di _c as text _col(37) "|"
	di _c as text _col(39) "SW Chi-sq("
	di _c as res %3.0f `fm'[9, 1] as text ") P-val"
	di _c as text _col(60) "|"
	di _c as text _col(62) "SW F("
	di as res _col(67) %3.0f `fm'[9, 1] as text "," as res %6.0f `fm'[10, 1] as text ")"
	local i 0
	foreach vn of local endo {
		local ++i
		if (length("`vn'") > 12) local vn = substr("`vn'", 1, 12)
		di _c as res %-12s "`vn'" _col(14) as text "|" _col(18) as res %8.2f `fm'[4, `i']
		di _c _col(28) as res %8.4f `fm'[7, `i']
		di _c _col(37) as text "|" _col(42) as res %8.2f `fm'[12, `i'] _col(51) as res %8.4f `fm'[13, `i']
		di _col(60) as text "|" _col(65) as res %8.2f `fm'[8, `i']
	}
	di
	local rtype
	if ("`cluster'" != "") local rtype "cluster-robust"
	else if ("`robust'" != "") local rtype "heteroskedasticity-robust"
	if ("`rtype'" != "") {
		di as text "NB: first-stage test statistics `rtype'"
		di
	}
	di as text "Stock-Yogo weak ID F test critical values for single endogenous regressor:"
	DispSY, row(2) col1(36) col2(67)
	if (r(cdmissing)) di as text _col(64) "<not available>"
	else {
		di as text "Source: Stock-Yogo (2005).  Reproduced by permission."
		if ("`robust'" == "") di as text "NB: Critical values are for Sanderson-Windmeijer F statistic."
		else di as text "NB: Critical values are for i.i.d. errors only."
		di
	}
	if (e(iddf) != `fm'[9, 1]) {
		di as err "Warning: Error in calculating first-stage id statistics above;"
		di as err "         dof of SW statistics is " `fm'[9, 1] ", should be L-(K-1)=`e(iddf)'."
	}
	di as text "Underidentification test"
	di as text "Ho: matrix of reduced form coefficients has rank=K1-1 (underidentified)"
	di as text "Ha: matrix has rank=K1 (identified)"
	if ("`robust'" == "") di as res "Anderson canon. corr. LM statistic" _c
	else di as res "Kleibergen-Paap rk LM statistic" _c
	di as text _col(42) "Chi-sq(" as res e(iddf) as text ")=" %-7.2f as res e(idstat) _col(61) as text "P-val=" %6.4f as res e(idp)
	di
	di as text "Weak identification test"
	di as text "Ho: equation is weakly identified"
	di as res "Cragg-Donald Wald F statistic" _col(65) %8.2f e(cdf)
	if ("`robust'" != "") di as res "Kleibergen-Paap Wald rk F statistic" _col(65) %8.2f e(rkf)
	di
	di as text "Stock-Yogo weak ID test critical values for K1=`e(endog_ct)' and L1=`e(exexog_ct)':"
	DispSY, row(1) col1(36) col2(67)
	if (r(cdmissing)) di as text _col(64) "<not available>"
	else {
		di as text "Source: Stock-Yogo (2005).  Reproduced by permission."
		if ("`robust'" != "") di as text "NB: Critical values are for Cragg-Donald F statistic and i.i.d. errors."
	}
	di
	di as text "Weak-instrument-robust inference"
	di as text "Tests of joint significance of endogenous regressors B1 in main equation"
	di as text "Ho: B1=0 and orthogonality conditions are valid"
	di as res _c "Anderson-Rubin Wald test"
	di as text _col(36) "F(" as res e(ardf) as text "," as res e(ardf_r) as text ")=" _col(49) as res %7.2f e(arf) _col(61) as text "P-val=" as res %6.4f e(arfp)
	di as res _c "Anderson-Rubin Wald test"
	di as text _col(36) "Chi-sq(" as res e(ardf) as text ")=" _col(49) as res %7.2f e(archi2) _col(61) as text "P-val=" as res %6.4f e(archi2p)
	di as res _c "Stock-Wright LM S statistic"
	di as text _col(36) "Chi-sq(" as res e(sstatdf) as text ")=" _col(49) as res %7.2f e(sstat) _col(61) as text "P-val=" as res %6.4f e(sstatp)
	di
	if ("`rtype'" != "") {
		di as text "NB: Underidentification, weak identification and weak-identification-robust"
		di as text "    test statistics `rtype'"
		di
	}
	if ("`cluster'" != "" & "`e(clustvar2)'" == "") di as text "Number of clusters             N_clust  = " as res %10.0f e(N_clust)
	else if ("`e(clustvar2)'" != "") {
		di as text "Number of clusters (1)         N_clust1 = " as res %10.0f e(N_clust1)
		di as text "Number of clusters (2)         N_clust2 = " as res %10.0f e(N_clust2)
	}
	di as text "Number of observations               N  = " as res %10.0f e(N)
	di as text "Number of regressors                 K  = " as res %10.0f e(rankxx)
	di as text "Number of endogenous regressors      K1 = " as res %10.0f e(endog_ct)
	di as text "Number of instruments                L  = " as res %10.0f e(rankzz)
	di as text "Number of excluded instruments       L1 = " as res %10.0f e(ardf)
end
