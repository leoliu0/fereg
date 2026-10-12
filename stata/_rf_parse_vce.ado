*! version 0.3.0  fereg: parse vce() (mirrors ms_parse_vce of ftools 2.50 / reghdfe 6.13.1)
* Returns:
*   s(vcetype)          unadjusted | robust | cluster
*   s(num_clusters)     number of cluster dimensions
*   s(clustervars)      cluster dimensions, e.g. "firm year" or "exporter#importer"
*   s(base_clustervars) unique base variables of the cluster dimensions
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

	if ("`vcetype'" == "dkraay") {
		di as err "fereg: vce(dkraay) not supported"
		exit 198
	}

	* Cluster dimensions: each token is var or var#var[#var...]; i. prefixes are allowed and removed;
	* string variables are allowed. Abbreviations are expanded.
	local clustervars
	local base_clustervars
	if ("`vcetype'" == "cluster") {
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
	if !inlist("`vcetype'", "unadjusted", "robust", "cluster") {
		di as err "vcetype '`vcetype'' not allowed"
		exit 198
	}
	if ("`vcetype'"=="unadjusted" & "`weighttype'"=="pweight") {
		di as err "pweights do not work with vce(unadjusted), use a different vce()"
		exit 198
	}
	local num_clusters : word count `clustervars'
	if !inlist((`num_clusters'>0) + ("`vcetype'"=="cluster"), 0, 2) {
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
end
