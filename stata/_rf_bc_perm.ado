*! version 0.4.1  fereg: random partitions of felogit's jackknife multiple(), as logitfe 3.0.3 draws them
*
* Usage: _rf_bc_perm, ivar(varname) tvar(varname) multiple(#) perm(i|t|it) names(newvarlist)
*
* logitfe (ado lines 758-1431 and 3593-3632) runs -set seed 13579- and, for every partition r, draws
* a random permutation of the panel ids (perm i or it) and then of the time ids (perm t or it) with
* moremata's mm_sample(n, n, ., ., 1), where n is the number of distinct values of the variable over
* all observations of the dataset and the k-th smallest value gets the k-th draw; the result is
* logitfe's pvar2/tvar2. This program makes the same draws in the same order (with moremata's
* algorithm, _rf_bc_srswor() below, so moremata is not needed) and stores, for partition r, pvar2
* and/or tvar2 in the next variables of names() (one per permuted dimension, in the order panel,
* time). The plugin ranks them over the estimation sample. Unlike logitfe, the random-number state
* of the session is restored afterwards (c(rngstate)).
program _rf_bc_perm
	version 16
	syntax, ivar(varname numeric) tvar(varname numeric) multiple(integer) perm(string) names(string)
	if (!inlist("`perm'", "i", "t", "it") | `: word count `names'' != `multiple' * strlen("`perm'")) {
		di as err "_rf_bc_perm: invalid arguments"
		exit 198
	}
	local state = c(rngstate)
	set seed 13579
	capture noisily mata: _rf_bc_perm("`ivar'", "`tvar'", `multiple', "`perm'", tokens(st_local("names")))
	local rc = _rc
	set rngstate `state'
	if (`rc') exit `rc'
end

mata:
void _rf_bc_perm(string scalar ivar, string scalar tvar, real scalar R, string scalar perm,
	string rowvector names)
{
	real scalar r, k, d
	real colvector v, p
	string scalar var

	k = 0
	for (r = 1; r <= R; r++) {
		for (d = 1; d <= strlen(perm); d++) {
			var = (substr(perm, d, 1) == "i" ? ivar : tvar)
			v = st_data(., var)
			p = _rf_bc_draw(v)
			k++
			(void) st_addvar("double", names[k])
			st_store(., names[k], p)
		}
	}
}

// pvar2 of logitfe's multiplepvar(): the k-th smallest distinct value of v (missing values
// included, in Stata's sort order) gets p[k], p = _rf_bc_srswor(n)
real colvector _rf_bc_draw(real colvector v)
{
	real colvector o, p, out
	real scalar i, k, n

	o = order(v, 1)
	n = rows(v) ? 1 : 0
	for (i = 2; i <= rows(v); i++) n = n + (v[o[i]] != v[o[i - 1]])
	p = _rf_bc_srswor(n)
	out = J(rows(v), 1, .)
	k = 0
	for (i = 1; i <= rows(v); i++) {
		if (i == 1) k = 1
		else if (v[o[i]] != v[o[i - 1]]) k++
		out[o[i]] = p[k]
	}
	return(out)
}

// A random permutation of 1..n: the draw of moremata's mm_sample(n, n, ., ., 1), i.e. mm_srswor()
// with its default algorithm mm_unorder2(n) = order(uniform(n, 2), (1, 2)), which draws no random
// numbers when n <= 1 (mm_srswor 1.0.1; test_logitfe.do checks the draws against logitfe's)
real colvector _rf_bc_srswor(real scalar n)
{
	if (n < 1) return(J(0, 1, .))
	if (n == 1) return(1)
	return(order(runiform(n, 2), (1, 2)))
}
end
