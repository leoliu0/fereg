*! fereg test helper: compare two stored estimation results
* rf_compare_e REF TEST [, btol(#) vtol(#) stol(#) xscalars(names) xmacros(names) nosample nodoftable nov]
*
*   b:          names identical; |b_T - b_R| <= btol * (|b_R| + se_R)
*   V:          |V_T - V_R| <= vtol * sqrt(V_R[i,i] V_R[j,j])  (correlation scale; zeros must match;
*               skipped with nov)
*   scalars:    every scalar of REF must exist in TEST; integer-valued scalars must be equal,
*               the others must satisfy reldif <= stol (both missing is equal)
*   macros:     every macro of REF must be equal in TEST, except cmd cmdline predict footnote
*               estat_cmd _estimates_name and the names in xmacros()
*   dof_table:  identical values and row/column names (unless nodoftable)
*   e(sample):  identical (unless nosample)
* The current estimation results are left as TEST.
program rf_compare_e
	version 16
	syntax namelist(min=2 max=2) [, btol(real 1e-7) vtol(real 1e-6) stol(real 1e-6) ///
		xscalars(string) xmacros(string) NOSAMPle NODOFtable NOV]
	gettoken ref test : namelist
	local test `test'
	tempname bR VR bT VT dR dT
	tempvar sR

	* Reference
	qui estimates restore `ref'
	local scalars : e(scalars)
	local scalars : list scalars - xscalars
	foreach s of local scalars {
		local R_`s' = string(e(`s'), "%21.17g")
	}
	local macros : e(macros)
	foreach m in cmd cmdline predict footnote estat_cmd _estimates_name `xmacros' {
		local macros : list macros - m
	}
	foreach m of local macros {
		local R_m_`m' `"`e(`m')'"'
	}
	local has_b = 0
	capture confirm matrix e(b)
	if (!_rc) {
		local has_b = 1
		matrix `bR' = e(b)
		matrix `VR' = e(V)
		local namesR : colfullnames e(b)
	}
	local has_dof = 0
	if ("`nodoftable'" == "") {
		capture confirm matrix e(dof_table)
		if (!_rc) {
			local has_dof = 1
			matrix `dR' = e(dof_table)
			local dofrowsR : rowfullnames e(dof_table)
			local dofcolsR : colfullnames e(dof_table)
		}
	}
	if ("`nosample'" == "") qui gen byte `sR' = e(sample)

	* Test
	qui estimates restore `test'
	local nfail 0
	foreach s of local scalars {
		capture confirm scalar e(`s')
		if (_rc) {
			di as err "FAIL scalar e(`s') missing in `test'"
			local ++nfail
			continue
		}
		local a = e(`s')
		local b = `R_`s''
		local isint = (`b' == round(`b')) & !missing(`b') & abs(`b') < 1e15
		local tol = cond(`isint', 0, `stol')
		capture rf_assert_close e(`s') `a' `b' `tol'
		if (_rc) {
			rf_assert_close_report e(`s') `a' `b' `tol'
			local ++nfail
		}
	}
	foreach m of local macros {
		if (`"`e(`m')'"' != `"`R_m_`m''"') {
			di as err `"FAIL macro e(`m'): <`e(`m')'> vs <`R_m_`m''>"'
			local ++nfail
		}
	}
	if (`has_b') {
		capture confirm matrix e(b)
		if (_rc) {
			di as err "FAIL e(b) missing in `test'"
			local ++nfail
		}
		else {
			matrix `bT' = e(b)
			matrix `VT' = e(V)
			local namesT : colfullnames e(b)
			if (`"`namesT'"' != `"`namesR'"') {
				di as err `"FAIL e(b) names: <`namesT'> vs <`namesR'>"'
				local ++nfail
			}
			else {
				mata: rf_compare_bv("`bT'", "`VT'", "`bR'", "`VR'", `btol', `vtol', `="`nov'" != ""')
				if (`rf_fail') local ++nfail
			}
		}
	}
	if (`has_dof') {
		matrix `dT' = e(dof_table)
		local dofrowsT : rowfullnames e(dof_table)
		local dofcolsT : colfullnames e(dof_table)
		if (`"`dofrowsT'"' != `"`dofrowsR'"' | `"`dofcolsT'"' != `"`dofcolsR'"') {
			di as err `"FAIL e(dof_table) names: <`dofrowsT'> vs <`dofrowsR'>"'
			local ++nfail
		}
		else if (mreldif(`dT', `dR') != 0) {
			di as err "FAIL e(dof_table) values"
			matrix list `dT'
			matrix list `dR'
			local ++nfail
		}
	}
	if ("`nosample'" == "") {
		qui count if `sR' != e(sample)
		if (r(N)) {
			di as err "FAIL e(sample): `r(N)' observations differ"
			local ++nfail
		}
	}
	if (`nfail') {
		di as err "rf_compare_e `ref' `test': `nfail' failure(s)"
		exit 9
	}
	di as text "rf_compare_e `ref' `test': OK"
end

program rf_assert_close_report
	args name a b rtol
	di as err "FAIL `name': " %21.15g `a' " vs " %21.15g `b' " (rtol `rtol')"
end

capture mata: mata drop rf_compare_bv()
mata:
void rf_compare_bv(string scalar bT, string scalar VT, string scalar bR, string scalar VR,
	real scalar btol, real scalar vtol, | real scalar skipv)
{
	real matrix B1, B2, V1, V2, S, D
	real scalar fail, i, j
	B1 = st_matrix(bT); B2 = st_matrix(bR)
	V1 = st_matrix(VT); V2 = st_matrix(VR)
	fail = 0
	S = sqrt(abs(diagonal(V2)))'
	D = abs(B1 - B2)
	for (j = 1; j <= cols(B2); j++) {
		if (D[j] > btol * (abs(B2[j]) + S[j])) {
			printf("{err}FAIL b[%g]: %21.15g vs %21.15g\n", j, B1[j], B2[j])
			fail = 1
		}
	}
	for (i = 1; i <= (skipv == 1 ? 0 : rows(V2)); i++) {
		for (j = 1; j <= cols(V2); j++) {
			if (abs(V1[i, j] - V2[i, j]) > vtol * S[i] * S[j] | (S[i] * S[j] == 0 & V1[i, j] != 0)) {
				printf("{err}FAIL V[%g,%g]: %21.15g vs %21.15g\n", i, j, V1[i, j], V2[i, j])
				fail = 1
			}
		}
	}
	st_local("rf_fail", strofreal(fail))
}
end
