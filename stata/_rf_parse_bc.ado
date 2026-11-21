*! version 0.5.0  fereg: parse felogit's bias-correction and APE options (logitfe 3.0.3 names)
*
* Usage (from _rf_glm.ado, after _rf_parse_absorb):
*   _rf_parse_bc, family(logit|poisson) [analytical jackknife ape ss1 ss2 js sj jj double
*       multiple(#) individuals time lags(#) ibias(yes|no) tbias(yes|no) population(#) network]
*       weight(<weight type>) separation(<normalized list>) g(#) ivars(<quoted per-term lists>)
*       intercepts(<list>) numslopes(<list>) isnoabsorb(0|1) nx(#)
*
* Options, messages and defaults follow logitfe 3.0.3 (Cruz-Gonzalez, Fernandez-Val and Weidner),
* with these fereg rules: no option means no correction (logitfe's default is analytical); ape alone
* is logitfe's nocorrection; every combination logitfe ignores or resets silently is an error; where
* logitfe has no message, the texts are those of fereg-core's bias correction (shared with R and
* Python).
* Validation that needs the sample (lags() against the shortest panel, population() against N*)
* is done by _rf_glm after prepare.
*
* Returns:
*   s(bc)          "" (no option given) | ape | analytical | jackknife
*   s(structure)   classic | network
*   s(effects)     classic: role of each absorb() term in order, i (xtset panel) or t (xtset time)
*   s(which)       both | individual | time     (bias corrected; network: both)
*   s(ibias) s(tbias)   yes | no              (analytical and jackknife)
*   s(lags)        bandwidth L (analytical; 0 otherwise)
*   s(variant)     ss1 ss2 js sj jj double   (jackknife)
*   s(multiple)    R (0 = none)
*   s(perm)        i | t | it   (dimensions permuted by each multiple() partition, in RNG order)
*   s(split)       both | individuals | time  (e(split) with multiple())
*   s(ivar) s(tvar) s(tdelta)   xtset panel and time variables and delta (tvar: also network)
*   s(population)  M ("" = not given)
*   s(title) s(title1) s(title2) s(title3)   logitfe's titles
program _rf_parse_bc, sclass
	version 16
	#delimit ;
	syntax , family(string) [ANalytical JACKknife APE SS1 SS2 JS SJ JJ DOUBLE
		MULtiple(integer 0) Individuals Time LAGS(integer 0) IBIAS(string) TBIAS(string)
		POPulation(string) NETwork
		weight(string) separation(string) g(integer 0) ivars(string asis) intercepts(string)
		numslopes(string) isnoabsorb(integer 0) nx(integer 1) ] ;
	#delimit cr
	sreturn clear
	local variants `ss1' `ss2' `js' `sj' `jj' `double'
	local variants : list retokenize variants
	local given `analytical' `jackknife' `ape' `variants' `individuals' `time' `network'
	if (`multiple' != 0) local given `given' multiple()
	if (`lags' != 0) local given `given' lags()
	if ("`ibias'" != "") local given `given' ibias()
	if ("`tbias'" != "") local given `given' tbias()
	if ("`population'" != "") local given `given' population()
	if ("`given'" == "") exit
	if ("`family'" != "logit") {
		di as err "option `: word 1 of `given'' not allowed with feppml"
		exit 198
	}

// ---------------------------------------------------------------------------
// Option combinations (logitfe ado lines 37-441 and fereg's stricter rules)
// ---------------------------------------------------------------------------
	foreach o in ibias tbias {
		if ("``o''" != "" & !inlist("``o''", "yes", "no")) {
			di as err "Error: -`o'- must be chosen from -yes no-."
			exit 198
		}
	}
	if ("`analytical'" != "" & "`jackknife'" != "") {
		di as err "Incompatible options: -jackknife- and -analytical-."
		exit 198
	}
	* the corrections compute the APEs themselves (logitfe: -nocorrection- and -analytical- exclude
	* each other)
	if ("`ape'" != "" & "`analytical'`jackknife'" != "") {
		di as err "Incompatible options: -ape- and -`analytical'`jackknife'-."
		exit 198
	}
	local bc = cond("`analytical'" != "", "analytical", cond("`jackknife'" != "", "jackknife", cond("`ape'" != "", "ape", "")))
	local bcname = cond("`bc'" == "", "ape", "`bc'")
	local copts `variants' `individuals' `time'
	local copts : list retokenize copts
	if ("`bc'" != "jackknife" & "`copts'" != "") {
		if ("`bc'" == "") di as err "Incompatible options: -`copts'- without -jackknife-."
		else di as err "Incompatible options: -`bc'- and -`copts'-."
		exit 198
	}
	if (`multiple' != 0 & "`bc'" != "jackknife") {
		di as err "Incompatible options: -multiple(`multiple')- is a -jackknife ss1/ss2- option."
		exit 198
	}
	if (`lags' != 0 & "`bc'" != "analytical") {
		di as err "Incompatible options: -lags(`lags')- is an -analytical- option."
		exit 198
	}
	if (!inlist("`bc'", "analytical", "jackknife")) {
		foreach o in ibias tbias {
			if ("``o''" != "") {
				di as err "Incompatible options: -`o'(``o'')- is an -analytical- or -jackknife- option."
				exit 198
			}
		}
	}
	if ("`bc'" == "") {
		* network or population() alone
		di as err "option `: word 1 of `given'' requires ape, analytical or jackknife"
		exit 198
	}
	if (`: word count `variants'' > 1) {
		gettoken first rest : variants
		di as err "Incompatible options: -`first'- and -`: list retokenize rest'-."
		exit 198
	}
	local variant = cond("`variants'" == "", "ss2", "`variants'")
	if ("`bc'" != "jackknife") local variant
	if (`multiple' < 0) {
		di as err "Invalid multiple option"
		exit 198
	}
	if (`multiple' > 0 & !inlist("`variant'", "ss1", "ss2")) {
		di as err "Incompatible options: -multiple(`multiple')- is a -jackknife ss1/ss2- option."
		exit 198
	}
	local io `individuals' `time'
	local io : list retokenize io
	if ("`io'" != "" & !inlist("`variant'", "ss1", "ss2")) {
		di as err "Incompatible options: -`variant'- and -`io'-."
		exit 198
	}
	if ("`io'" != "" & `multiple' == 0) {
		di as err "Incompatible options: -`io'- requires -multiple()-."
		exit 198
	}
	if (`lags' < 0) {
		di as err "Invalid option: number of lags must be higher or equal than 0"
		exit 198
	}
	if ("`ibias'" == "no" & "`tbias'" == "no") {
		di as err "Incompatible options: -`bc'- and -ibias(`ibias') tbias(`tbias')-."
		di as err "Use the -ape- option instead"
		exit 198
	}
	if (`lags' > 0 & "`ibias'" == "no") {
		di as err "lags(`lags') has no effect: lags only apply to the individual-effect bias"
		exit 198
	}
	if ("`population'" != "") {
		capture confirm integer number `population'
		local bad = _rc
		if (!`bad') local bad = (`population' < 0)
		if (`bad') {
			di as err "Invalid population option: population must be a finite positive integer"
			exit 198
		}
	}

// ---------------------------------------------------------------------------
// Model restrictions: no weights, the fe separation rule, regressors, absorbed terms
// ---------------------------------------------------------------------------
	if ("`weight'" != "") {
		di as err "weights not allowed"
		exit 101
	}
	if ("`separation'" != "fe") {
		di as err "option separation() not allowed with `bcname': groups with all-zero or all-one outcomes must be dropped (separation(fe), the default)"
		exit 198
	}
	if (`nx' < 1) {
		di as err "too few variables specified"
		exit 102
	}
	if (`isnoabsorb') {
		di as err "bias correction and APEs need fixed effects"
		exit 198
	}
	forval j = 1/`g' {
		if (`: word `j' of `numslopes'' != 0 | `: word `j' of `intercepts'' != 1) {
			di as err "bias correction and APEs need intercept-only fixed effects (no varying slopes)"
			exit 198
		}
	}
	local tvar `_dta[_TStvar]'
	local ivar `_dta[_TSpanel]'
	if ("`_dta[_TSdelta]'" == "") local tdelta 1
	else local tdelta = `_dta[_TSdelta]'

	if ("`network'" != "") {
		local structure network
		if (!inlist(`g', 2, 3)) {
			di as err `"panel structure "network" expects a two- or three-way fixed effects model"'
			di as err "(absorb() with exporter-time, importer-time and optionally pair effects, in this order)"
			exit 198
		}
		if ("`bc'" == "jackknife") {
			di as err "the jackknife is not available for the network structure"
			exit 198
		}
		foreach o in ibias tbias {
			if ("``o''" != "") {
				di as err "network structure: only the correction of both biases is available (option `o'(``o''))"
				exit 198
			}
		}
		if (`lags' > 0 & "`tvar'" == "") {
			di as err "Error: must -tsset- data and specify timevar (lags() orders each pair by time)"
			exit 5
		}
		local which both
		local effects
		local ib yes
		local tb yes
		local title1 = cond(`g' == 3, "exporter-time, importer-time and pair effects", "exporter-time and importer-time effects")
		local title2 "`title1'"
		local title1 "Type of included effects: `title1'"
		local title2 "Type of bias correction: `title2'"
		local ivar
	}
	else {
		local structure classic
		if ("`ivar'" == "" & "`tvar'" != "") {
			di as err "Error: must -tsset- data and specify panelvar"
			exit 5
		}
		if ("`tvar'" == "" & "`ivar'" != "") {
			di as err "Error: must -tsset- data and specify timevar"
			exit 5
		}
		if ("`ivar'" == "" & "`tvar'" == "") {
			di as err "Error: must -tsset- data and specify panelvar and timevar"
			exit 5
		}
		if (!inlist(`g', 1, 2)) {
			di as err `"panel structure "classic" expects a one- or two-way fixed effects model"'
			di as err "(use the network option for exporter-time, importer-time and pair effects)"
			exit 198
		}
		local effects
		local ok 1
		forval j = 1/`g' {
			local iv : word `j' of `ivars'
			local iv `iv'
			if ("`iv'" == "`ivar'") local effects `effects'i
			else if ("`iv'" == "`tvar'") local effects `effects't
			else local ok 0
		}
		if (inlist("`effects'", "ii", "tt")) local ok 0
		if (!`ok') {
			di as err "option `bcname' requires absorb() with the -xtset- panel variable (`ivar') and/or time variable (`tvar'), each as a term of its own"
			di as err "(use the network option for exporter-time, importer-time and pair effects)"
			exit 198
		}
		local has_i = strpos("`effects'", "i") > 0
		local has_t = strpos("`effects'", "t") > 0
		if (`has_i' & `has_t') {
			local title1 "individual and time effects"
		}
		else if (`has_i') {
			local title1 "individual effects only"
			if ("`tbias'" == "yes") {
				di as err "the time-effect bias needs time fixed effects (option tbias(yes); absorb() has individual effects only)"
				exit 198
			}
			if ("`ibias'" == "no") {
				di as err "Incompatible options: -ibias(no)- with individual effects only leaves no bias to correct."
				di as err "Use the -ape- option instead"
				exit 198
			}
		}
		else {
			local title1 "time effects only"
			if ("`ibias'" == "yes") {
				di as err "the individual-effect bias needs individual fixed effects (option ibias(yes); absorb() has time effects only)"
				exit 198
			}
			if ("`tbias'" == "no") {
				di as err "Incompatible options: -tbias(no)- with time effects only leaves no bias to correct."
				di as err "Use the -ape- option instead"
				exit 198
			}
			if (`lags' > 0) {
				di as err "lags(`lags') has no effect: lags only apply to the individual-effect bias"
				exit 198
			}
		}
		local febias = `has_i' & "`ibias'" != "no"
		local tebias = `has_t' & "`tbias'" != "no"
		local which = cond(`febias' & `tebias', "both", cond(`febias', "individual", "time"))
		local ib = cond(`febias', "yes", "no")
		local tb = cond(`tebias', "yes", "no")
		local title1 "Type of included effects: `title1'"
		if ("`which'" == "both") local title2 "Type of bias correction: individual and time effects"
		else if ("`which'" == "individual") local title2 "Type of bias correction: individual effects only"
		else local title2 "Type of bias correction: time effects only"
	}

// ---------------------------------------------------------------------------
// multiple(): permuted dimensions (logitfe's RNG calls, ado lines 701-1431), logitfe's notes
// ---------------------------------------------------------------------------
	local perm
	local split
	if (`multiple' > 0) {
		if ("`individuals'" != "" & "`time'" == "") {
			local perm i
			local split individuals
		}
		else if ("`individuals'" == "" & "`time'" != "") {
			local perm t
			local split time
		}
		else {
			local perm it
			local split both
		}
		* partitions that permute only a dimension the subpanels do not split change nothing: an
		* error (logitfe prints a note and drops multiple())
		if ("`which'" == "individual" & "`perm'" == "i") {
			di as err "multiple partitions have no effect: the subpanels of this bias split only the time dimension, which no partition permutes"
			exit 198
		}
		else if ("`which'" == "time" & "`perm'" == "t") {
			di as err "multiple partitions have no effect: the subpanels of this bias split only the individual dimension, which no partition permutes"
			exit 198
		}
	}

// ---------------------------------------------------------------------------
// logitfe's titles (ado lines 2298-2391)
// ---------------------------------------------------------------------------
	local title3
	if ("`bc'" == "ape") {
		local title "Uncorrected fixed-effects estimates"
		local title2 "Type of bias correction: none"
	}
	else if ("`bc'" == "analytical") {
		local title "Analytical bias-correction"
		local title3 "Triming parameter = `lags'"
	}
	else {
		if ("`variant'" == "ss1") local title "Split-panel jackknife in four subpanels"
		if ("`variant'" == "ss2") local title "Split-panel jackknife in both dimensions"
		if ("`variant'" == "js") local title "Delete-one jackknife in cross-section, split-panel in time series"
		if ("`variant'" == "sj") local title "Split-panel jackknife in cross-section, delete-one jackknife in time series"
		if ("`variant'" == "jj") local title "Delete-one jackknife in cross-section and time series"
		if ("`variant'" == "double") local title "Double-panel jackknife"
		if (`multiple' > 0) {
			if ("`perm'" == "i") local title3 "`multiple' multiple partitions in the cross-section dimension"
			else if ("`perm'" == "t") local title3 "`multiple' multiple partitions in the time dimension"
			else local title3 "`multiple' multiple partitions in both the cross-section and the time dimension"
		}
	}

	sreturn local bc `bc'
	sreturn local structure `structure'
	sreturn local effects `effects'
	sreturn local which `which'
	if inlist("`bc'", "analytical", "jackknife") {
		sreturn local ibias `ib'
		sreturn local tbias `tb'
	}
	sreturn local lags `lags'
	sreturn local variant `variant'
	sreturn local multiple `multiple'
	sreturn local perm `perm'
	sreturn local split `split'
	sreturn local ivar `ivar'
	sreturn local tvar `tvar'
	sreturn local tdelta `tdelta'
	sreturn local population `population'
	sreturn local title `"`title'"'
	sreturn local title1 `"`title1'"'
	sreturn local title2 `"`title2'"'
	sreturn local title3 `"`title3'"'
end
