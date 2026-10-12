*! fereg test helper: stage the current reghdfe e() for the mock plugin (tests/stata/mock)
* rf_stage_mock, indepvars(varlist|_none) [status(numlist)]
*   __mock_b/__mock_V: e(b)/e(V) restricted to the regressors fereg sends to the plugin
*                      (fvexpand'ed terms that are not base/omitted levels, then _cons)
*   __mock_doft:       e(dof_table)
*   __mock_<scalar>:   every e() scalar
*   $rfmock_status:    status codes the mock returns (one per sent regressor); by default 2 for sent
*                      regressors that reghdfe omitted (o. in e(b)) and 0 otherwise
program rf_stage_mock
	version 16
	syntax, indepvars(string) [status(string)]
	tempname b V
	mata: st_local("old", invtokens(st_dir("global", "numscalar", "__mock_*")'))
	if ("`old'" != "") scalar drop `old'
	local mask
	if ("`indepvars'" != "_none") {
		fvexpand `indepvars' if e(sample)
		foreach part in `r(varlist)' {
			_ms_parse_parts `part'
			local mask `mask' `=(r(omit) == 0)'
		}
	}
	if (e(report_constant)) local mask `mask' 1
	capture matrix drop __mock_b
	capture matrix drop __mock_V
	capture confirm matrix e(b)
	if (!_rc) {
		matrix `b' = e(b)
		matrix `V' = e(V)
		if (`: word count `mask'' != colsof(`b')) {
			di as err "rf_stage_mock: mask/b mismatch"
			exit 9
		}
		mata: rf_stage_mock_bv("`b'", "`V'", "`mask'")
	}
	matrix __mock_doft = e(dof_table)
	local sc : e(scalars)
	foreach s of local sc {
		scalar __mock_`s' = e(`s')
	}
	if ("`status'" == "" & "`indepvars'" != "_none") {
		local bnames : colnames e(b)
		local j 0
		foreach m of local mask {
			local ++j
			local name : word `j' of `bnames'
			if (`m' & "`name'" != "_cons") {
				_ms_parse_parts `name'
				local status `status' `=cond(r(omit), 2, 0)'
			}
		}
	}
	global rfmock_status `status'
end

capture mata: mata drop rf_stage_mock_bv()
mata:
void rf_stage_mock_bv(string scalar b, string scalar V, string scalar mask)
{
	real rowvector m
	m = selectindex(strtoreal(tokens(mask)))
	if (cols(m)) {
		st_matrix("__mock_b", st_matrix(b)[m])
		st_matrix("__mock_V", st_matrix(V)[m, m])
	}
}
end
