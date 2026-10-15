*! version 0.4.1  fereg: the variance-estimator keys of -prepare- (PROTOCOL.md, "Variance estimators")
* Usage: _rf_vce_args, factors(prepare's factor list) vcetype() [clustervars() tvar() ivar() tdelta()
*            dkraay_bw() hac(0|1) kernelkey() bw() kiefer(0|1) psd() lat() lon() cutoff() pixel()
*            distance()]
* c_local rf_vceargs: the quoted key=value arguments; tvar/ivar/cluster positions refer to factors()
program _rf_vce_args
	version 16
	syntax, vcetype(string) [factors(string) clustervars(string) tvar(string) ivar(string) ///
		tdelta(string) dkraay_bw(string) hac(integer 0) kernelkey(string) bw(string) kiefer(integer 0) ///
		psd(string) lat(string) lon(string) cutoff(string) pixel(string) distance(string)]

	local args `""vce=`vcetype'""'
	if inlist("`vcetype'", "cluster", "crv3") {
		* dimensions separated by ";", each f1*f2 (1-based factor positions)
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
		local args `"`args' "cluster=`clukey'""'
	}
	if ("`tvar'" != "") {
		local k : list posof "`tvar'" in factors
		local td = strtrim(string(`tdelta', "%21.17g"))
		local args `"`args' "tvar=`k'" "tdelta=`td'""'
	}
	if ("`ivar'" != "") {
		local k : list posof "`ivar'" in factors
		local args `"`args' "ivar=`k'""'
	}
	if ("`vcetype'" == "dkraay" & "`dkraay_bw'" != "") local args `"`args' "dkbw=`dkraay_bw'""'
	if (`hac') {
		if (`kiefer') local args `"`args' "kiefer=1" "kernel=truncated""'
		else local args `"`args' "kernel=`kernelkey'" "bw=`bw'""'
		local args `"`args' "psd=`psd'""'
	}
	if ("`vcetype'" == "conley") {
		local args `"`args' "ncoord=2" "pixel=`pixel'" "distance=`distance'""'
		if ("`cutoff'" != "") local args `"`args' "cutoff=`cutoff'""'
	}
	c_local rf_vceargs `"`args'"'
end
