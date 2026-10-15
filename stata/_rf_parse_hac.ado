*! version 0.4.1  fereg: time-series variance options (ivreghdfe 1.1.4 / ivreg2 HAC surface, reghdfe vce(dkraay))
* Usage (after _rf_parse_vce; its results are passed back in):
*   _rf_parse_hac, vcetype() [clustervars() weighttype() userrobust(0|1) bw() kernel() kiefer
*                  dkraay(#) psd0 psda]
* Validation follows ivreghdfe's ivparse and the s_vkernel routine of ivreg2's Mata library (texts
* and return codes), in their order: dkraay() sets bw and the time cluster, s_vkernel checks the
* kernel and the bandwidth, kiefer means kernel(Truncated), then the dkraay(), kiefer and psd
* checks, the tsset requirement, the fweight restriction and the cluster/kernel combinations.
* vce(dkraay [#]) (reghdfe) requires a tsset/xtset time variable (reghdfe.ado's message).
* Returns:
*   s(hac)              1 when an ivreg2 kernel is used (kernel(), bw(), kiefer, dkraay())
*   s(kernel)           ivreg2 kernel name (Bartlett, Parzen, Truncated, Tukey-Hanning, Tukey-Hamming,
*                       Quadratic Spectral, Danielle, Tent); s(kernelkey) the plugin spelling
*   s(bw)               bandwidth as given (a real number, or auto); empty with kiefer
*   s(kiefer) 0/1, s(psd) none|psd0|psda
*   s(robust)           ivreghdfe's robust flag: vce(robust), vce(cluster ...) or pweights (with a
*                       kernel and no cluster: 1 = HAC, 0 = AC); userrobust() = vce(robust|cluster)
*   s(vcetype) s(clustervars) s(base_clustervars) s(num_clusters)   updated: dkraay() clusters on
*                       the time variable; a panel-time cluster pair is reordered to (ivar, tvar)
*   s(tvar) s(ivar)     variables to send (tvar: kernel or vce(dkraay); ivar: kernel on xtset data)
*   s(tdelta)           _dta[_TSdelta] (1 when not set)
*   s(xtivar)           _dta[_TSpanel] (posted as e(ivar) by the kernel family)
program _rf_parse_hac, sclass
	version 16
	syntax, vcetype(string) [clustervars(string) weighttype(string) userrobust(integer 0) ///
		bw(string) kernel(string) kiefer dkraay(integer 0) psd0 psda]
	sreturn clear

	local num_clusters : word count `clustervars'
	local any = (`"`bw'`kernel'`kiefer'"' != "" | `dkraay' != 0)

	* tsset settings, as ivreghdfe's ivparse and reghdfe read them
	local tvar `_dta[_TStvar]'
	local ivar `_dta[_TSpanel]'
	if ("`_dta[_TSdelta]'" == "") local tdelta 1
	else local tdelta = `_dta[_TSdelta]'

	if ("`vcetype'" == "dkraay") {
		if (`any' | "`psd0'`psda'" != "") {
			di as err "options bw(), kernel(), kiefer, dkraay(), psd0 and psda not allowed with vce(dkraay)"
			exit 198
		}
		if ("`tvar'" == "") {
			di as err "vce(dkraay) requires data to be tsset or xtset with a time variable"
			exit 9
		}
		sreturn local hac 0
		sreturn local vcetype dkraay
		sreturn local clustervars
		sreturn local base_clustervars
		sreturn local num_clusters 0
		sreturn local tvar `tvar'
		sreturn local ivar
		sreturn local xtivar `ivar'
		sreturn local tdelta `tdelta'
		sreturn local robust 1
		sreturn local kiefer 0
		sreturn local psd none
		exit
	}

	if (!`any') {
		if ("`psd0'`psda'" != "") {
			di as err "options psd0 and psda require a kernel (bw(), kernel(), kiefer or dkraay())"
			exit 198
		}
		sreturn local hac 0
		sreturn local vcetype `vcetype'
		sreturn local clustervars `clustervars'
		sreturn local num_clusters `num_clusters'
		sreturn local tvar
		sreturn local ivar
		sreturn local xtivar `ivar'
		sreturn local tdelta `tdelta'
		sreturn local robust = inlist("`vcetype'", "robust", "cluster")
		sreturn local kiefer 0
		sreturn local psd none
		exit
	}

	if (!inlist("`vcetype'", "unadjusted", "robust", "cluster")) {
		di as err "options bw(), kernel(), kiefer and dkraay() not allowed with vce(`vcetype')"
		exit 198
	}

	* ivparse: cluster implies robust; dkraay(#) is bw(#) with clustering on the time variable
	local robust = `userrobust'
	local cluster `clustervars'
	if (`dkraay') {
		if (`"`bw'"' == "") local bw `dkraay'
		if ("`cluster'" == "") local cluster `tvar'
	}

	* s_vkernel (ivreg2 Mata library): bandwidth, kernel name, bw = 1 note
	local bwnum 0
	if (`"`bw'`kernel'"' != "") {
		if (`"`bw'"' == "auto") {
			local bwnum -1
		}
		else {
			local bwnum = real(`"`bw'"')
			if (missing(`bwnum')) {
				di as err "bandwidth option bw() required for HAC-robust estimation"
				exit 102
			}
			if (`bwnum' <= 0) {
				di as err "invalid bandwidth in option bw() - must be real > 0"
				exit 198
			}
			* as r(bw) of s_vkernel, a number
			local bw = strtrim(string(`bwnum', "%21.17g"))
		}
		if (`"`bw'"' == "auto" & "`ivar'" != "") {
			di as err "Automatic bandwidth selection not available for panel data"
			exit 198
		}
		local kname = strltrim(strlower(`"`kernel'"'))
		local lagk 0
		if inlist(`"`kname'"', "", "bar", "bartlett") {
			local kernel Bartlett
			local lagk 1
		}
		else if inlist(`"`kname'"', "par", "parzen") {
			local kernel Parzen
			local lagk 1
		}
		else if inlist(`"`kname'"', "tru", "truncated") local kernel Truncated
		else if inlist(`"`kname'"', "thann", "tukey-hanning") {
			local kernel Tukey-Hanning
			local lagk 1
		}
		else if inlist(`"`kname'"', "thamm", "tukey-hamming") {
			local kernel Tukey-Hamming
			local lagk 1
		}
		else if inlist(`"`kname'"', "qua", "qs", "quadratic-spectral", "quadratic spectral") local kernel Quadratic Spectral
		else if inlist(`"`kname'"', "dan", "danielle") local kernel Danielle
		else if inlist(`"`kname'"', "ten", "tent") local kernel Tent
		else {
			di as err "invalid kernel"
			exit 198
		}
		if (`bwnum' == 1 & `lagk') {
			di as result "Note: kernel=`kernel' and bw=1 implies zero lags used.  Standard errors and"
			di as result "      test statistics are not autocorrelation-consistent."
		}
	}
	else local bw
	if ("`kiefer'" != "" & "`kernel'" == "") local kernel Truncated

	if (`dkraay') {
		if ("`ivar'" == "" | "`tvar'" == "") {
			di as err "invalid use of dkraay option - must use tsset panel data"
			exit 5
		}
		if ("`dkraay'" != "`bw'") {
			di as err "cannot use dkraay(.) and bw(.) options together"
			exit 198
		}
		if ("`cluster'" != "`tvar'") {
			di as err "invalid use of dkraay option - must cluster on `tvar' (or omit cluster option)"
			exit 198
		}
	}
	if ("`kiefer'" != "") {
		if ("`ivar'" == "" | "`tvar'" == "") {
			di as err "invalid use of kiefer option - must use tsset panel data"
			exit 5
		}
		if (`robust') {
			di as err "incompatible options: kiefer and robust"
			exit 198
		}
		if ("`kernel'" != "Truncated") {
			di as err "incompatible options: kiefer and kernel(`kernel')"
			exit 198
		}
		if (`bwnum' != 0) {
			di as err "incompatible options: kiefer and bw"
			exit 198
		}
	}
	if ("`psd0'" != "" & "`psda'" != "") {
		di as err "cannot use psd0 and psda options together"
		exit 198
	}

	* ivreghdfe: the data must be tsset (-tsset- with no time variable fails with this message)
	if ("`tvar'" == "") {
		di as err "time variable not set, use tsset varname ..."
		exit 111
	}
	if ("`weighttype'" == "fweight") {
		di as err "fweights not allowed (data are -tsset-)"
		exit 101
	}

	* Kernel with clustering: one cluster on the time variable (Driscoll-Kraay) or the panel and
	* time variables in either order (Thompson), reordered to (panel, time)
	local num_clusters : word count `cluster'
	if (`num_clusters' > 0) {
		tokenize `cluster'
		local cluster1 `1'
		local cluster2 `2'
		if (`num_clusters' > 1) {
			if ("`cluster1'" == "`tvar'" & "`cluster2'" == "`ivar'") {
				local cluster1 `2'
				local cluster2 `1'
			}
			if ("`cluster1'" != "`ivar'" | "`cluster2'" != "`tvar'" | `num_clusters' > 2) {
				di as err "Error: cluster kernel-robust requires clustering on tsset panel & time vars."
				di as err "       tsset panel var=`ivar'; tsset time var=`tvar'; cluster vars=`cluster1',`cluster2'"
				exit 198
			}
			local cluster `cluster1' `cluster2'
		}
		else if ("`cluster1'" != "`tvar'") {
			di as err "Error: cluster kernel-robust requires clustering on tsset time variable."
			di as err "       tsset time var=`tvar'; cluster var=`cluster1'"
			exit 198
		}
		local vcetype cluster
	}

	local kernelkey = subinstr(strlower("`kernel'"), " ", "-", .)
	sreturn local hac 1
	sreturn local kernel `kernel'
	sreturn local kernelkey `kernelkey'
	sreturn local bw `bw'
	sreturn local kiefer = ("`kiefer'" != "")
	sreturn local psd = cond("`psd0'`psda'" == "", "none", "`psd0'`psda'")
	sreturn local robust = (`robust' | "`weighttype'" == "pweight")
	sreturn local vcetype `vcetype'
	sreturn local clustervars `cluster'
	sreturn local base_clustervars `cluster'
	sreturn local num_clusters `num_clusters'
	sreturn local tvar `tvar'
	sreturn local ivar `ivar'
	sreturn local xtivar `ivar'
	sreturn local tdelta `tdelta'
end
