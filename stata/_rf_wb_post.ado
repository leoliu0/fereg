*! version 0.5.1  fereg: post wild bootstrap scalars, macros, CI matrix, and savet vector
program _rf_wb_post, eclass
	version 16
	syntax , Level(real) Method(string) Boottype(string) Weights(string) ///
		Pvaltype(string) Stattype(string) [BOOTcluster(string) Cimat(string) Savet(string)]

	foreach s in wb_stat wb_p wb_reps wb_reps_feasible wb_q wb_full_enum wb_impose_null wb_ssc ///
		wb_n_clust wb_point wb_se wb_ci_lo wb_ci_hi {
		capture confirm scalar __rf_`s'
		if (!_rc) ereturn scalar `s' = scalar(__rf_`s')
	}
	ereturn scalar wb_level = `level'
	ereturn local wb_method `"`method'"'
	ereturn local wb_boottype `"`boottype'"'
	ereturn local wb_weights `"`weights'"'
	ereturn local wb_pvaltype `"`pvaltype'"'
	ereturn local wb_stattype `"`stattype'"'
	if (`"`bootcluster'"' != "") ereturn local wb_bootcluster `"`bootcluster'"'
	capture confirm scalar __rf_wb_ci_lo
	if (!_rc) {
		if (!missing(scalar(__rf_wb_ci_lo)) & "`cimat'" != "") {
			capture confirm matrix `cimat'
			if (!_rc) ereturn matrix wb_ci = `cimat'
		}
	}
	if ("`savet'" != "") {
		capture confirm scalar __rf_wb_reps
		if (!_rc) {
			local wbr = scalar(__rf_wb_reps)
			if (`wbr' >= 1 & `wbr' < rowsof(`savet')) {
				matrix `savet' = `savet'[1..`wbr', 1]
			}
		}
	}
end
