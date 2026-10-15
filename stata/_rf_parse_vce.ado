*! version 0.4.1  fereg: parse vce() (mirrors ms_parse_vce of ftools 2.50 / reghdfe 6.13.1)
* Returns:
*   s(vcetype)          unadjusted | robust | cluster | dkraay | hc2 | hc3 | crv3 | conley
*   s(num_clusters)     number of cluster dimensions (cluster, crv3)
*   s(clustervars)      cluster dimensions, e.g. "firm year" or "exporter#importer"
*   s(base_clustervars) unique base variables of the cluster dimensions
*   s(dkraay_bw)        vce(dkraay #): the bandwidth # (= lags + 1); empty = reghdfe's default
*   s(conley_lat) s(conley_lon) s(cutoff) s(pixel) s(distance)   vce(conley ...)
* Sources of the messages: ms_parse_vce (unadjusted, robust, cluster, dkraay; its _assert exits
* with r(9), kept for the dkraay texts it introduced), Stata regress (hc2, hc3: pweights not
* allowed with hc3). crv3 and conley are fereg extensions (borrowed definitions, see help fereg).
program _rf_parse_vce, sclass
	version 16
	sreturn clear
	syntax, [vce(string) weighttype(string)]
	local 0 `vce'
	* -anything- instead of -namelist- because clusters can be x#y
	syntax [anything(id="VCE type")] , [*]

	gettoken vcetype vce_arg : anything

	* vcetype abbreviations (as ms_parse_vce)
	if (substr("`vcetype'",1,3)=="ols") local vcetype unadjusted
	if (substr("`vcetype'",1,2)=="un") local vcetype unadjusted
	if (substr("`vcetype'",1,1)=="r") local vcetype robust
	if (substr("`vcetype'",1,2)=="cl") local vcetype cluster
	if (substr("`vcetype'",1,2)=="dk") local vcetype dkraay
	if ("`vcetype'"=="conventional") local vcetype unadjusted

	* vce(dkraay [#]): # is the bandwidth (= lags + 1), empty = default (ms_parse_vce)
	local dkraay_bw
	if ("`vcetype'" == "dkraay" & `"`vce_arg'"' != "") {
		capture confirm integer number `vce_arg'
		if (_rc) {
			di as err "vce(dkraay #): # must be a positive integer (bandwidth = lags + 1)"
			exit 9
		}
		if (`vce_arg' < 1) {
			di as err "vce(dkraay #): # must be >= 1 (bandwidth = lags + 1)"
			exit 9
		}
		local dkraay_bw = `vce_arg'
	}

	* vce(hc2|hc3): Stata regress; the clustered forms vce(hc2|hc3 clustvar) are not supported
	if inlist("`vcetype'", "hc2", "hc3") {
		if (`"`vce_arg'"' != "") {
			di as err "fereg: vce(`vcetype' clustvar) not supported"
			exit 198
		}
		if ("`vcetype'" == "hc3" & "`weighttype'" == "pweight") {
			di as err "pweights not allowed with option vce(hc3)"
			exit 198
		}
	}

	* vce(conley latvar lonvar [, cutoff(#) pixel(#) distance(triangular|spherical)])
	local conley_lat
	local conley_lon
	local cutoff
	local pixel 0
	local distance triangular
	if ("`vcetype'" == "conley") {
		local coords `vce_arg'
		if (`: word count `coords'' != 2) {
			di as err "vce(conley latvar lonvar): exactly two variables (latitude, longitude) required"
			exit 198
		}
		local coords : subinstr local coords "c." "", all
		unab coords : `coords', min(2) max(2)
		capture confirm numeric variable `coords'
		if (_rc) {
			di as err "vce(conley latvar lonvar): latvar and lonvar must be numeric"
			exit 198
		}
		gettoken conley_lat conley_lon : coords
		local conley_lon `conley_lon'
		local 0 , `options'
		syntax , [CUToff(string) PIXel(string) DISTance(string)]
		local options
		if (`"`cutoff'"' != "") {
			capture confirm number `cutoff'
			if (_rc) {
				di as err "vce(conley): cutoff() must be a positive number (kilometers)"
				exit 198
			}
			if (`cutoff' <= 0) {
				di as err "vce(conley): cutoff() must be a positive number (kilometers)"
				exit 198
			}
			local cutoff = strtrim(string(`cutoff', "%21.17g"))
		}
		if (`"`pixel'"' == "") local pixel 0
		capture confirm number `pixel'
		if (_rc) {
			di as err "vce(conley): pixel() must be a nonnegative number (kilometers)"
			exit 198
		}
		if (`pixel' < 0) {
			di as err "vce(conley): pixel() must be a nonnegative number (kilometers)"
			exit 198
		}
		local pixel = strtrim(string(`pixel', "%21.17g"))
		local distance = strlower(strtrim(`"`distance'"'))
		if ("`distance'" == "") local distance triangular
		if (!inlist("`distance'", "triangular", "spherical")) {
			di as err "vce(conley): distance() must be triangular or spherical"
			exit 198
		}
	}

	* Cluster dimensions (cluster, crv3): each token is var or var#var[#var...]; i. prefixes are
	* allowed and removed; string variables are allowed. Abbreviations are expanded.
	local clustervars
	local base_clustervars
	if inlist("`vcetype'", "cluster", "crv3") {
		foreach dim of local vce_arg {
			local parts : subinstr local dim "#" " ", all
			local newdim
			foreach part of local parts {
				if (substr("`part'", 1, 2) == "i.") local part = substr("`part'", 3, .)
				if strpos("`part'", ".") {
					di as err "unexpected dot in clustervars: `vce_arg'"
					exit 198
				}
				unab part : `part', min(1) max(1) name(cluster variable)
				local newdim = cond("`newdim'" == "", "`part'", "`newdim'#`part'")
				local base_clustervars `base_clustervars' `part'
			}
			local clustervars `clustervars' `newdim'
		}
		local base_clustervars : list uniq base_clustervars
	}

	* Implicit defaults
	if ("`vcetype'"=="" & "`weighttype'"=="pweight") local vcetype robust
	if ("`vcetype'"=="") local vcetype unadjusted

	* Sanity checks
	if !inlist("`vcetype'", "unadjusted", "robust", "cluster", "dkraay", "hc2", "hc3", "crv3", "conley") {
		di as err "vcetype '`vcetype'' not allowed"
		exit 198
	}
	if ("`vcetype'"=="unadjusted" & "`weighttype'"=="pweight") {
		di as err "pweights do not work with vce(unadjusted), use a different vce()"
		exit 198
	}
	local num_clusters : word count `clustervars'
	if ("`vcetype'" == "crv3") {
		if (`num_clusters' == 0) {
			di as err "vce(crv3) requires clustervars"
			exit 198
		}
	}
	else if !inlist((`num_clusters'>0) + ("`vcetype'"=="cluster"), 0, 2) {
		di as err "Can't specify cluster without clustervars (and viceversa)"
		exit 198
	}
	if (`"`options'"' != "") {
		di as err `"VCE options not supported: `options'"'
		exit 198
	}

	sreturn local vcetype `vcetype'
	sreturn local num_clusters `num_clusters'
	sreturn local clustervars `clustervars'
	sreturn local base_clustervars `base_clustervars'
	sreturn local dkraay_bw `dkraay_bw'
	sreturn local conley_lat `conley_lat'
	sreturn local conley_lon `conley_lon'
	sreturn local cutoff `cutoff'
	sreturn local pixel `pixel'
	sreturn local distance `distance'
end
