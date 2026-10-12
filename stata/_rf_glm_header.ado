*! version 0.2.1  fereg: header of feppml and felogit results
*! feppml layout from ppmlhdfe_header.ado of ppmlhdfe 2.3.3 (MIT, (c) Sergio Correia, Paulo Guimarães, Thomas Zylkin)
program _rf_glm_header
	version 16

/*--+----1----+----2----+----3----+----4----+----5----+----6----+----7----+---
e(title)                                          No. of obs     = ###########
e(title2)                                         Residual df    = ###########
e(title3)                                         Wald chi2(#)   = ###########
Deviance             = ###########                Prob > chi2    = ###########
Log pseudolikelihood = ###########                Pseudo R2      = ###########

Number of clusters (#)    = ######
----+----1----+----2----+----3----+----4----+----5----+----6----+----7----+--*/

	if ("`e(cmd)'" == "feppml") {
		di as text ""
		di as text "`e(title)'" _c
		di as text _col(51) "No. of obs" _col(67) "=" _c
		di as text _col(69) as res %10.0gc e(N)

		di as text "`e(title2)'" _c
		di as text _col(51) "Residual df" _col(67) "=" _c
		di as text _col(69) as res %10.0gc e(df)

		di as text "`e(title3)'" _c
		di as text _col(51) "Wald chi2(" as res e(df_m) as text ")" _col(67) "=" _c
		di as text _col(69) as res %10.2f e(chi2)

		di as text "Deviance" _col(22) "=" _c
		di as text _col(24) as res %12.0g e(deviance) _c
		di as text _col(51) "Prob > chi2" _col(67) "=" _c
		di as text _col(69) as res %10.4f chi2tail(e(df_m), e(chi2))

		di as text "Log pseudolikelihood =" _c
		di as text _col(24) as res %12.0g e(ll) _c
		di as text _col(51) "Pseudo R2" _col(67) "=" _c
		di as text _col(69) as res %10.4f e(r2_p)
	}
	else {
		* felogit: the layout of Stata's logit header plus the absorbed groups
		local llname = cond("`e(vcetype)'" == "Robust", "Log pseudolikelihood", "Log likelihood")
		di as text ""
		di as text "`e(title)'" _c
		di as text _col(51) "Number of obs" _col(67) "=" _c
		di as text _col(69) as res %10.0gc e(N)

		di as text "`e(title2)'" _c
		di as text _col(51) "Wald chi2(" as res e(df_m) as text ")" _col(67) "=" _c
		di as text _col(69) as res %10.2f e(chi2)

		di as text "`e(title3)'" _c
		di as text _col(51) "Prob > chi2" _col(67) "=" _c
		di as text _col(69) as res %10.4f chi2tail(e(df_m), e(chi2))

		di as text "`llname' = " as res %12.0g e(ll) _c
		di as text _col(51) "Pseudo R2" _col(67) "=" _c
		di as text _col(69) as res %10.4f e(r2_p)
	}

	local N_clustervars = e(N_clustervars)
	if (`N_clustervars' == .) local N_clustervars 0
	if (`N_clustervars' > 0) di as text ""
	forval i = 1/`N_clustervars' {
		local cluster "`e(clustvar`i')'"
		local num = "`e(N_clust`i')'"
		di as text "Number of clusters (" as res "`cluster'" as text ")" _col(29) as text "=" _c
		di as text _col(31) as res %10.0fc `num'
	}
end
