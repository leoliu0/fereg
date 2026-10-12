*! version 0.2.1  fereg: the only program that calls the compiled plugin
*
* Usage:
*   _rf_load                                     load the plugin and check that its version matches
*   _rf_load [varlist] [if] [in] , cmd "k=v"...  run -plugin call- and copy the result locals to the caller
*
* Why a wrapper: a program defined with -program ..., plugin- inside an ado-file is private to that
* ado-file, so callers cannot run -plugin call- themselves. Locals written by the plugin land in this
* program and are copied to the caller with c_local (list in `forward').
* Every key=value argument must be quoted by the caller: unquoted arguments are split by Stata on
* = | ; * - and commas are a syntax error.
program _rf_load
	version 16
	local rf_version 0.2.1

	if (`"`0'"' == "") {
		local _rf_version
		capture plugin call _rf_plugin, version
		if (_rc == 199) {
			di as err "fereg: could not load fereg.plugin; reinstall with"
			di as err `"  net install fereg, from("https://raw.githubusercontent.com/leoliu0/fereg/gh-pages/stata") replace"'
			exit 601
		}
		else if (_rc) {
			exit _rc
		}
		if ("`_rf_version'" != "`rf_version'") {
			di as err "fereg: plugin version `_rf_version' does not match ado version `rf_version'; reinstall"
			exit 198
		}
		exit
	}

	* Result locals set by the plugin (protocol: locals _rf_<name>)
	local forward version token status
	* IV: Stock-Yogo critical values for (L1, K1) and for a single endogenous regressor
	foreach t in ivbias5 ivbias10 ivbias20 ivbias30 ivsize10 ivsize15 ivsize20 ivsize25 {
		local forward `forward' sy_`t' sy1_`t'
	}
	foreach name of local forward {
		local _rf_`name'
	}
	plugin call _rf_plugin `0'
	foreach name of local forward {
		c_local _rf_`name' `"`_rf_`name''"'
	}
end

capture program _rf_plugin, plugin using("fereg.plugin")
