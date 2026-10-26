*! version 0.4.1  fereg: display (or quiet r(table)) of feppml and felogit results
program _rf_glm_replay, rclass
	version 16
	syntax [, EForm IRr OR noHEader noTABle noFOOTnote *]
	_get_diopts options, `options'
	if ("`irr'" != "") local eform eform
	if (!c(noisily)) {
		* Output is suppressed (quietly): skip the display work but leave r(table) and r(level)
		* as _coef_table would (eform tables go through _coef_table itself).
		if ("`table'" == "") {
			if ("`eform'`or'" == "") {
				QuietTable, `options'
			}
			else {
				_coef_table, `options' `eform' `or'
			}
			return add
		}
		exit
	}
	if ("`header'" == "") _rf_glm_header
	if ("`table'" == "") _coef_table, `options' `eform' `or'
	return add // r(level), r(table), ...
	* felogit with a bias correction or APEs: logitfe's table of average partial effects (not
	* after a prefix such as bootstrap, whose e(bias) is a matrix and which posts no e(b2))
	if ("`table'" == "" & "`e(cmd)'" == "felogit" & "`e(prefix)'" == "" & ///
		inlist("`e(bias)'", "analytical", "jackknife", "none")) {
		ApeTable, `options'
		return add
	}
	if ("`footnote'" == "") _rf_footnote
end


* Average partial effects e(b2), e(V2) (logitfe.ado lines 2419-2424): title, the finite-population
* line when e(fpc) != 1, and the table; r(table_ape)
program ApeTable, rclass
	syntax [, *]
	tempname T
	di as text ""
	di as text "Average Partial Effects"
	if (e(fpc) != 1) di as text "Variance adjusted by the finite population parameter " as res %8.4f e(fpc)
	* base levels of factor variables (APE 0, variance 0) would show as (empty)
	_coef_table, bmatrix(e(b2)) vmatrix(e(V2)) noemptycells `options'
	matrix `T' = r(table)
	return matrix table_ape = `T'
end


program QuietTable, rclass
	syntax [, Level(cilevel) *]
	capture confirm matrix e(b)
	if (_rc) exit
	tempname T
	_ms_omit_info e(b)
	mata: _rf_glm_rtable("`T'", `level')
	return scalar level = `level'
	return matrix table = `T'
end


version 16
mata:
// r(table) as _coef_table builds it for z statistics: rows b se z pvalue ll ul df crit eform;
// omitted and base coefficients (r(omit) from _ms_omit_info) get missing se, z, pvalue, ll, ul.
void _rf_glm_rtable(string scalar name, real scalar level)
{
	real rowvector b, se, z, p, omit
	real scalar k, crit
	b = st_matrix("e(b)")
	k = cols(b)
	omit = st_matrix("r(omit)")
	se = sqrt(diagonal(st_matrix("e(V)")))'
	se = se :/ (omit :== 0)
	crit = invnormal(1 - (100 - level) / 200)
	z = b :/ se
	p = 2 :* normal(-abs(z))
	st_matrix(name, (b \ se \ z \ p \ b - crit :* se \ b + crit :* se \ J(1, k, .) \ J(1, k, crit) \ J(1, k, 0)))
	st_matrixrowstripe(name, (J(9, 1, ""), ("b" \ "se" \ "z" \ "pvalue" \ "ll" \ "ul" \ "df" \ "crit" \ "eform")))
	st_matrixcolstripe(name, st_matrixcolstripe("e(b)"))
}
end
