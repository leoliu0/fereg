*! version 0.4.0  fereg: ivreghdfe 1.1.4's variance descriptions (e(vce), e(hacsubtitleV), kernel e())
* Usage (after -ereturn post-):
*   _rf_hac_post, hac(0|1) [kernel() tvar() ivar() robust(0|1) clustvar() clusters() kiefer(0|1)
*                 psd() bwchoice() nopost]
*   clustvar() = the cluster variables as given (e(clustvar)), clusters() = the dimensions in the
*   plugin's order (a panel-time pair as (ivar, tvar)).
* c_locals: rf_vcestr (ivreghdfe's e(vce), lowercase), rf_hacV, rf_hacV2 (e(hacsubtitleV), V2),
*   rf_vcetype ("Robust" or empty, ivreghdfe's e(vcetype) rule). Unless nopost, a kernel posts
*   e(kernel), e(bw) (the plugin's __rf_bw: bw(auto) and kiefer resolved), e(tvar), e(ivar),
*   e(bwchoice), as ivreghdfe.
program _rf_hac_post, eclass
	version 16
	syntax, hac(integer) [kernel(string) tvar(string) ivar(string) robust(integer 0) ///
		clustvar(string) clusters(string) kiefer(integer 0) psd(string) bwchoice(string) NOPOST]

	local bw .
	if (`hac') {
		capture confirm scalar __rf_bw
		if (!_rc) local bw = scalar(__rf_bw)
	}
	local ncl : word count `clusters'
	local cluster1 : word 1 of `clusters'
	local cluster2 : word 2 of `clusters'

	* e(vce) (ivreghdfe l.2044-2072)
	local vce
	if (`robust') local vce robust
	if (`ncl' == 1) local vce `vce' cluster
	if (`ncl' == 2) local vce `vce' two-way cluster
	if (`hac') {
		local vce `vce' `=cond(`robust', "hac", "ac")'
		local vce `vce' `kernel' bw=`bw'
	}
	if (!inlist("`psd'", "", "none")) local vce `vce' `psd'
	local vce : list clean vce
	local vce = lower("`vce'")

	* e(hacsubtitleV) (ivreghdfe l.2270-2323)
	local vcetype = cond(`robust' | `ncl' > 0, "Robust", "")
	local h1
	local h2
	local h3
	local h4
	if ("`vcetype'" == "Robust") local h1 heteroskedasticity
	if (`hac' & `ncl' == 0) local h3 autocorrelation
	if (`kiefer') local h3 within-cluster autocorrelation (Kiefer)
	if (`ncl' == 1) local h3 clustering on `clustvar'
	if (`ncl' == 2) local h3 clustering on `cluster1' and `cluster2'
	if (`ncl' > 0 & `hac') local h4 and kernel-robust to common correlated disturbances (Driscoll-Kraay)
	if ("`h1'" != "" & "`h3'" != "") local h2 " and "
	local hacsubtitle "`h1'`h2'`h3'"
	if ("`hacsubtitle'" != "") local hacV "Statistics robust to `hacsubtitle'"
	else local hacV "Statistics consistent for homoskedasticity only"

	c_local rf_vcestr `vce'
	c_local rf_hacV `hacV'
	c_local rf_hacV2 `h4'
	c_local rf_vcetype `vcetype'

	if (`hac' & "`nopost'" == "") {
		ereturn local kernel "`kernel'"
		ereturn scalar bw = `bw'
		ereturn local tvar "`tvar'"
		if ("`ivar'" != "") ereturn local ivar "`ivar'"
		if (`"`bwchoice'"' != "") ereturn local bwchoice `"`bwchoice'"'
	}
end
