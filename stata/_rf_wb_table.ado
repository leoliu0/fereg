*! version 0.5.0  fereg: display wild bootstrap results block from e(wb_*)
program _rf_wb_table
	version 16
	if (`"`e(wb_method)'"' == "") exit

	local mlab `e(wb_method)'
	if ("`mlab'" == "wild" | "`mlab'" == "wildcluster") local mlab "Wild cluster (`e(wb_boottype)')"
	else if ("`mlab'" == "score") local mlab "Wild score (`e(wb_boottype)')"
	else if ("`mlab'" == "wre") local mlab "WRE (`e(wb_boottype)')"
	else if ("`mlab'" == "hetero") local mlab "Heteroskedastic (`e(wb_boottype)')"

	local null_txt = cond(e(wb_impose_null), "yes", "no")
	local enum_txt = cond(e(wb_full_enum), " (full enumeration)", "")
	local reps_txt = strtrim(string(e(wb_reps), "%15.0fc")) + "`enum_txt'"
	if (e(wb_reps_feasible) < e(wb_reps)) {
		local rf = strtrim(string(e(wb_reps_feasible), "%15.0fc"))
		local reps_txt `"`reps_txt' (`rf' feasible)"'
	}

	di as text ""
	di as text "Wild bootstrap test (`e(wb_stattype)')"
	di as text "{hline 78}"
	di as text "Method          = " as res "`mlab'" _c
	di as text _col(42) "Weights      = " as res "`e(wb_weights)'"
	di as text "Null imposed    = " as res "`null_txt'" _c
	di as text _col(42) "Replications = " as res "`reps_txt'"
	if ("`e(wb_bootcluster)'" != "" | !missing(e(wb_n_clust))) {
		local clname = cond("`e(wb_bootcluster)'" != "", "`e(wb_bootcluster)'", "`e(clustvar)'")
		di as text "Boot cluster    = " as res "`clname'" _c
		if (!missing(e(wb_n_clust))) di as text _col(42) "Clusters     = " as res e(wb_n_clust)
		else di ""
	}
	di as text "{hline 78}"
	di as text "Test statistic  = " as res %10.4f e(wb_stat) _c
	di as text _col(42) "p-value (`e(wb_pvaltype)') = " as res %8.4f e(wb_p)
	if (e(wb_q) == 1) {
		if (!missing(e(wb_point)) & !missing(e(wb_se))) {
			di as text "Point estimate  = " as res %10.4f e(wb_point) _c
			di as text _col(42) "Bootstrap SE = " as res %10.4f e(wb_se)
		}
		else if (!missing(e(wb_point))) {
			di as text "Point estimate  = " as res %10.4f e(wb_point)
		}
		capture confirm matrix e(wb_ci)
		if (!_rc & !missing(e(wb_ci_lo)) & !missing(e(wb_ci_hi))) {
			local lvl = cond(missing(e(wb_level)), 95, e(wb_level))
			di as text "`lvl'% CI          = " as res "[" %9.4f e(wb_ci_lo) ", " %9.4f e(wb_ci_hi) "]"
		}
	}
	di as text "{hline 78}"
end
