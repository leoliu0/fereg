*! version 0.1.0  regfast: display (or quiet r(table)) of ppmlfast and logitfast results
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
	if ("`footnote'" == "") _rf_footnote
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
