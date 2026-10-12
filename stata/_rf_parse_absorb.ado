*! version 0.1.0  regfast: parse absorb() (reghdfe 6.13.1 grammar, see ms_parse_absvars of ftools 2.50)
*
* Syntax:  _rf_parse_absorb [absvars] [, SAVEfe Generate REPLACE]
*
* absvar  := [target=] term
* term    := x | i.x | x#y[#z...] | x#c.z | x##c.z | x##c.(z w ...) | x#y##c.z ...
* Factors (x, y, ...) may be numeric or string variables (strings are not allowed by reghdfe inside
* interactions; regfast allows them). Slope variables (c.z) must be numeric. Time-series operators
* (L.x, c.L.z) are accepted and returned unexpanded; the caller resolves them with -tsrevar-.
*
* Returns (s() names follow ms_parse_absvars where they exist):
*   s(G)                 number of absorbed terms (1 for the constant-only case)
*   s(absvars)           absvars as typed (target removed, abbreviations expanded), e.g.
*                        "idcode year##c.age ind_code#c.(tenure hours)" ("_cons" if none); = reghdfe e(absvars)
*   s(extended_absvars)  one label per coefficient slot, e.g. "idcode year year#c.age" ("_cons" if none)
*   s(dof_rownames)      the same slots with reghdfe's "1." prefixes (row stripe of e(dof_table))
*   s(fe_labels)         variable labels of the saved FEs ("[FE] 1.year#c.age"), one quoted label per slot
*   s(ivars)             per term, quoted list of factor variables ("_cons" for the constant-only case)
*   s(cvars)             per term, quoted list of slope variables
*   s(intercepts)        per term, 1 if the term has an intercept
*   s(num_slopes)        per term, number of slope variables
*   s(targets)           per term, quoted list of target variables (intercept target, then targetSlope#)
*   s(factors)           unique factor variables in order of appearance
*   s(slopevars)         unique slope variables in order of appearance
*   s(basevars)          unique factor and slope variables (for -markout-)
*   s(stringvars)        factor variables that are strings
*   s(num_slots)         number of coefficient slots (= words in s(extended_absvars))
*   s(has_intercept)     1 if any term has an intercept (or constant-only)
*   s(save_any_fe) s(save_all_fe)
program _rf_parse_absorb, sclass
	version 16
	syntax [anything(id="absvars" name=absvars equalok everything)] [, SAVEfe Generate REPLACE *]
	if (`"`options'"' != "") {
		di as err `"invalid suboptions in absorb(): `options'"'
		exit 198
	}
	local save_all_fe = ("`savefe'" != "") | ("`generate'" != "")

	* Constant-only
	if (`"`absvars'"' == "") {
		sreturn clear
		sreturn local G 1
		sreturn local absvars "_cons"
		sreturn local extended_absvars "_cons"
		sreturn local dof_rownames "_cons"
		sreturn local fe_labels `""""'
		sreturn local ivars `""_cons""'
		sreturn local cvars `""""'
		sreturn local intercepts "1"
		sreturn local num_slopes "0"
		sreturn local targets `""""'
		sreturn local factors ""
		sreturn local slopevars ""
		sreturn local basevars ""
		sreturn local stringvars ""
		sreturn local num_slots 1
		sreturn local has_intercept 1
		sreturn local save_any_fe 0
		sreturn local save_all_fe 0
		exit
	}

	* Normalize spaces: collapse runs, and remove spaces around "=" and "#"
	local s `"`absvars'"'
	while (strpos(`"`s'"', "  ")) {
		local s : subinstr local s "  " " ", all
	}
	foreach d in "=" "#" {
		local s : subinstr local s " `d'" "`d'", all
		local s : subinstr local s "`d' " "`d'", all
	}

	* Split into terms (bind keeps "c.(z w)" together); expand wildcards/abbreviations of bare terms
	local terms
	local G 0
	while (`"`s'"' != "") {
		gettoken term s : s, bind
		if !strpos("`term'", "#") & !strpos("`term'", ".") & !strpos("`term'", "(") & !strpos("`term'", "=") {
			unab expanded : `term', name(absorb())
			foreach v of local expanded {
				local ++G
				local term`G' `v'
			}
		}
		else {
			local ++G
			local term`G' `term'
		}
	}

	local all_absvars
	local all_ivars
	local all_cvars
	local all_intercepts
	local all_num_slopes
	local all_targets
	local extended
	local stripe
	local fe_labels
	local factors
	local slopevars
	local stringvars
	local any_has_intercept 0
	local save_any_fe 0

	forval g = 1/`G' {
		local term `term`g''

		* Target
		local target
		local p = strpos("`term'", "=")
		if (`p') {
			local target = substr("`term'", 1, `p' - 1)
			local term = substr("`term'", `p' + 1, .)
			if ("`target'" == "" | "`term'" == "") {
				di as err "invalid absvar: `term`g''"
				exit 198
			}
			confirm name `target'
		}
		if (`save_all_fe' & "`target'" == "") local target __hdfe`g'__

		* Absvar text as typed, with variable abbreviations expanded (this is what reghdfe stores in e(absvars))
		UnabTerm `term' // returns `typed'

		* Split into factor parts and slope parts. After normalization the only spaces left in a term
		* are inside c.( ): protect them with "@" (never part of a name) while splitting on "#".
		local term : subinstr local term " " "@", all
		local p = strpos("`term'", "##")
		if (`p') {
			local left = substr("`term'", 1, `p' - 1)
			local right = substr("`term'", `p' + 2, .)
			if strpos("`right'", "##") | "`left'" == "" | "`right'" == "" {
				di as err "invalid absvar: `typed'"
				exit 198
			}
			local fparts : subinstr local left "#" " ", all
			local cparts : subinstr local right "#" " ", all
			foreach part of local fparts {
				if (substr("`part'", 1, 2) == "c.") {
					di as err "invalid absvar `typed': c. variables must follow ##"
					exit 198
				}
			}
			foreach part of local cparts {
				if (substr("`part'", 1, 2) != "c.") {
					di as err "invalid absvar `typed': expected c.variable after ##"
					exit 198
				}
			}
			local has_intercept 1
		}
		else {
			local parts : subinstr local term "#" " ", all
			local fparts
			local cparts
			foreach part of local parts {
				if (substr("`part'", 1, 2) == "c.") local cparts `cparts' `part'
				else local fparts `fparts' `part'
			}
			local has_intercept = ("`cparts'" == "")
		}

		* Factor variables
		local ivars
		foreach part of local fparts {
			ParseFactorPart `part' // returns `var' `isstr'
			local ivars `ivars' `var'
			if (`isstr') local stringvars `stringvars' `var'
		}
		local ivars : list uniq ivars
		if ("`ivars'" == "") {
			di as err "no indicator variables in absvar <`typed'>"
			exit 198
		}

		* Slope variables
		local cvars
		foreach part of local cparts {
			ParseSlopePart `part' // returns `vars'
			local cvars `cvars' `vars'
		}
		local unique_cvars : list uniq cvars
		if (!`: list unique_cvars == cvars') {
			di as err "duplicated c. variable in absvar <`typed'> unique(`cvars') == <`unique_cvars'>)"
			exit 198
		}
		local num_slopes : word count `cvars'
		if (`has_intercept') local any_has_intercept 1

		* Base label of the slots
		local baselabel : subinstr local ivars " " "#", all

		* Extended labels, one per coefficient slot
		local ebase : subinstr local ivars " " "#1.", all
		local ebase 1.`ebase'
		local slotlabels
		if (`has_intercept') {
			local extended `extended' `baselabel'
			local stripe `stripe' `ebase'
			local slotlabels `slotlabels' `ebase'
		}
		foreach cvar of local cvars {
			local extended `extended' `baselabel'#c.`cvar'
			local stripe `stripe' `ebase'#c.`cvar'
			local slotlabels `slotlabels' `ebase'#c.`cvar'
		}
		foreach sl of local slotlabels {
			local fe_labels `"`fe_labels' "[FE] `sl'""'
		}

		* Targets (intercept target, then targetSlope#)
		local tlist
		if ("`target'" != "") {
			local save_any_fe 1
			if (`has_intercept') local tlist `target'
			forval h = 1/`num_slopes' {
				local tlist `tlist' `target'Slope`h'
			}
			foreach t of local tlist {
				if ("`replace'" != "") novarabbrev capture drop `t'
				confirm new variable `t'
				if (`: list t in all_target_vars') {
					di as err "absorb(): target variable `t' specified twice"
					exit 198
				}
				local all_target_vars `all_target_vars' `t'
			}
		}

		local all_absvars `"`all_absvars' "`typed'""'
		local all_ivars `"`all_ivars' "`ivars'""'
		local all_cvars `"`all_cvars' "`cvars'""'
		local all_intercepts `all_intercepts' `has_intercept'
		local all_num_slopes `all_num_slopes' `num_slopes'
		local all_targets `"`all_targets' "`tlist'""'
		local factors `factors' `ivars'
		local slopevars `slopevars' `cvars'
	}

	local factors : list uniq factors
	local slopevars : list uniq slopevars
	local stringvars : list uniq stringvars
	local basevars `factors' `slopevars'
	local basevars : list uniq basevars
	local num_slots : word count `extended'

	* Unquote into a plain list
	local absvars_out
	foreach lab of local all_absvars {
		local absvars_out `absvars_out' `lab'
	}

	sreturn clear
	sreturn local G `G'
	sreturn local absvars `"`absvars_out'"'
	sreturn local extended_absvars `"`extended'"'
	sreturn local dof_rownames `"`stripe'"'
	sreturn local fe_labels `"`fe_labels'"'
	sreturn local ivars `"`all_ivars'"'
	sreturn local cvars `"`all_cvars'"'
	sreturn local intercepts `"`all_intercepts'"'
	sreturn local num_slopes `"`all_num_slopes'"'
	sreturn local targets `"`all_targets'"'
	sreturn local factors `"`factors'"'
	sreturn local slopevars `"`slopevars'"'
	sreturn local basevars `"`basevars'"'
	sreturn local stringvars `"`stringvars'"'
	sreturn local num_slots `num_slots'
	sreturn local has_intercept `any_has_intercept'
	sreturn local save_any_fe `save_any_fe'
	sreturn local save_all_fe `save_all_fe'
end


* i.x, ib2.x, ibn.x, ib(first).x, x, L.x  -->  variable name (string allowed unless a ts operator is used)
program ParseFactorPart
	local part `0'
	if regexm("`part'", "^i(b?n?[0-9]*|b\([a-z]+\))\.(.+)$") local part = regexs(2)
	if strpos("`part'", "(") | strpos("`part'", ")") {
		di as err "invalid factor in absorb(): `0'"
		exit 198
	}
	if strpos("`part'", ".") {
		tsunab var : `part', min(1) max(1)
		c_local var `var'
		c_local isstr 0
		exit
	}
	unab var : `part', min(1) max(1) name(absorb())
	capture confirm string variable `var'
	c_local var `var'
	c_local isstr = (_rc == 0)
end


* c.z, c.(z w), c.L.z  -->  numeric variables
program ParseSlopePart
	local part : subinstr local 0 "@" " ", all
	local part = substr("`part'", 3, .)
	if (substr("`part'", 1, 1) == "(") {
		if (substr("`part'", -1, 1) != ")") {
			di as err "invalid slope in absorb(): `0'"
			exit 198
		}
		local part = substr("`part'", 2, strlen("`part'") - 2)
	}
	tsunab vars : `part', min(1)
	foreach v of local vars {
		if !strpos("`v'", ".") confirm numeric variable `v'
	}
	c_local vars `vars'
end


* Expand variable abbreviations in one absvar, keeping everything else as typed
* (mirror of UnabAbsvars in ms_parse_absvars, which produces reghdfe's e(absvars))
program UnabTerm
	local remainder `"`0'"'
	local answer
	local is_numlist 0 // inside L(1 2) or L(-1/1)
	while (`"`remainder'"' != "") {
		gettoken tok remainder : remainder, parse(" #.()=")
		local next_char `"`=substr(`"`remainder'"', 1, 1)'"'
		local delim1 = inlist(`"`tok'"', "#", ".", "(", ")", "=")
		local delim2 = inlist(`"`next_char'"', ".", "(")
		if !(`delim1' | `delim2' | `is_numlist') {
			unab tok : `tok', name(absorb())
		}
		if (`"`tok'"' == ")") local is_numlist 0
		if (`"`tok'"' != "." & `"`next_char'"' == "(") local is_numlist 1
		if (`"`next_char'"' != " ") local next_char
		local answer `"`answer'`tok'`next_char'"'
	}
	c_local typed `"`answer'"'
end
