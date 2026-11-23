*! version 0.5.0  fereg: parse wildboot(...) option and build plugin wb_* keys
program _rf_parse_wildboot, rclass
	version 16
	syntax , Step(string) [raw(string) spec(string) Rmat(string) Rvec(string) ///
		fullnames(string) sentmask(string) ///
		reps(integer 999) weights(string) boottype(string) method(string) ///
		imposenull(integer 1) pvaltype(string) wantci(integer 1) level(real 95) ///
		seed(string) tol(real 1e-6) maxiter(integer 40) savet(string) cimat(string)]

	if ("`step'" == "pre") {
		local 0 `"`raw'"'
		syntax [anything(equalok)] [, Reps(integer 999) Weights(string) BOOTType(string) Method(string) ///
			NOImposenull Imposenull Pvaltype(string) BOOTCluster(string) Level(cilevel) noCI ///
			Seed(string) Tol(real 1e-6) MAXIter(integer 40) SAVET(name) Rmat(name) Rvec(name)]

		if ("`imposenull'" != "" & "`noimposenull'" != "") {
			di as err "wildboot(): options imposenull and noimposenull are mutually exclusive"
			exit 198
		}
		local impose_null = ("`noimposenull'" == "")
		local want_ci = ("`ci'" == "")
		if (`"`anything'"' == "" & "`rmat'" == "") {
			di as err "wildboot(): specify a hypothesis expression or rmat()"
			exit 198
		}
		if (`"`anything'"' != "" & "`rmat'`rvec'" != "") {
			di as err "wildboot(): cannot specify both a hypothesis expression and rmat()/rvec()"
			exit 198
		}
		if ("`rvec'" != "" & "`rmat'" == "") {
			di as err "wildboot(): rvec() requires rmat()"
			exit 198
		}
		if ("`rmat'" != "") {
			confirm matrix `rmat'
			if ("`rvec'" != "") {
				confirm matrix `rvec'
				local qr = rowsof(`rmat')
				if (!((rowsof(`rvec') == `qr' & colsof(`rvec') == 1) | (rowsof(`rvec') == 1 & colsof(`rvec') == `qr'))) {
					di as err "wildboot(): rvec() dimensions do not match rows of rmat() (`qr')"
					exit 503
				}
			}
		}
		if (`reps' < 1) {
			di as err "wildboot(): reps() must be a positive integer"
			exit 198
		}
		if (`tol' <= 0) {
			di as err "wildboot(): tol() must be positive"
			exit 198
		}
		if (`maxiter' < 1) {
			di as err "wildboot(): maxiter() must be a positive integer"
			exit 198
		}

		local w = lower(strtrim("`weights'"))
		if ("`w'" == "") local w rademacher
		if ("`w'" == "norm") local w normal
		if (!inlist("`w'", "rademacher", "webb", "mammen", "normal")) {
			di as err "wildboot(): weights() must be rademacher, webb, mammen, or normal"
			exit 198
		}

		local bt = strtrim("`boottype'")
		if ("`bt'" == "") local bt 11
		if (!inlist("`bt'", "11", "21", "31", "13", "33")) {
			di as err "wildboot(): boottype() must be 11, 21, 31, 13, or 33"
			exit 198
		}

		local m = lower(strtrim("`method'"))
		if ("`m'" == "") local m auto
		if ("`m'" == "wildcluster") local m wild
		if (!inlist("`m'", "auto", "wild", "score", "wre", "hetero")) {
			di as err "wildboot(): method() must be auto, wild, score, wre, or hetero"
			exit 198
		}

		local pt = lower(strtrim("`pvaltype'"))
		if ("`pt'" == "" | "`pt'" == "two-tailed" | "`pt'" == "twotailed") local pt twotailed
		else if ("`pt'" == "equal-tailed" | "`pt'" == "equal") local pt equal
		else if ("`pt'" == ">" | "`pt'" == "greater") local pt greater
		else if ("`pt'" == "<" | "`pt'" == "less") local pt less
		else {
			di as err "wildboot(): pvaltype() must be twotailed, equal, greater, or less"
			exit 198
		}

		local s = strtrim("`seed'")
		if ("`s'" == "") {
			local s = strtrim(string(floor(runiform() * 2147483647), "%21.0f"))
		}
		else {
			capture confirm integer number `s'
			if (_rc | `s' < 0) {
				di as err "wildboot(): seed() must be a non-negative integer"
				exit 198
			}
		}

		local bc_vars
		local bc_label
		if (`"`bootcluster'"' != "") {
			foreach dim of local bootcluster {
				local parts : subinstr local dim "#" " ", all
				foreach tok of local parts {
					if (strpos("`tok'", ".")) {
						local bc_vars `bc_vars' `tok'
					}
					else {
						unab uv : `tok', min(1) max(1)
						local bc_vars `bc_vars' `uv'
					}
				}
			}
			local bc_vars : list uniq bc_vars
			local bc_label : subinstr local bc_vars " " "#", all
		}

		return local spec `"`anything'"'
		return local rmat "`rmat'"
		return local rvec "`rvec'"
		return local weights "`w'"
		return local boottype "`bt'"
		return local method "`m'"
		return local pvaltype "`pt'"
		return local seed "`s'"
		return local savet "`savet'"
		return local bc_vars "`bc_vars'"
		return local bootcluster "`bc_label'"
		return scalar reps = `reps'
		return scalar imposenull = `impose_null'
		return scalar wantci = `want_ci'
		return scalar level = `level'
		return scalar tol = `tol'
		return scalar maxiter = `maxiter'
		exit
	}

	if ("`step'" == "hyp") {
		local full_str `"`fullnames'"'
		local mask_str `"`sentmask'"'
		local spec_str `"`spec'"'
		local wb_q
		local wb_R
		local wb_r
		local wb_err
		mata: _rf_wb_build_hyp(st_local("spec_str"), st_local("rmat"), st_local("rvec"), ///
			st_local("full_str"), st_local("mask_str"))
		if (`"`wb_err'"' != "") {
			di as err `"`wb_err'"'
			local rc = cond(strpos(`"`wb_err'"', "dimension"), 503, 198)
			exit `rc'
		}
		local q = real("`wb_q'")

		local tolstr = strtrim(string(`tol', "%21.17g"))
		local lvlstr = strtrim(string(`level', "%21.17g"))
		local args `""wb_q=`wb_q'" "wb_R=`wb_R'" "wb_r=`wb_r'" "wb_reps=`reps'" "wb_weights=`weights'" "wb_boottype=`boottype'" "wb_method=`method'" "wb_null=`imposenull'" "wb_pvaltype=`pvaltype'" "wb_ci=`wantci'" "wb_level=`lvlstr'" "wb_seed=`seed'" "wb_tol=`tolstr'" "wb_maxiter=`maxiter'""'
		if (`q' == 1 & `wantci' & "`cimat'" != "") {
			matrix `cimat' = J(1, 2, .)
			local args `"`args' "wb_ci_mat=`cimat'""'
		}
		if ("`savet'" != "") {
			matrix `savet' = J(`reps', 1, .)
			local args `"`args' "wb_savet=`savet'""'
		}
		return local args `"`args'"'
		return scalar q = `q'
		exit
	}

	di as err "_rf_parse_wildboot: unknown step `step'"
	exit 198
end

version 16
mata:
string scalar _rf_wb_canon(string scalar s)
{
	real scalar p, q
	string scalar res, tok
	res = ""
	// Normalize each '#' factor component: strip 'bn.' -> '.' (keeps 'b.' for base check)
	p = strpos(s, "#")
	if (p == 0) {
		q = strpos(s, "bn.")
		if (q > 1) return(substr(s, 1, q - 1) + "." + substr(s, q + 3, .))
		return(s)
	}
	res = ""
	while (1) {
		p = strpos(s, "#")
		tok = (p > 0 ? substr(s, 1, p - 1) : s)
		q = strpos(tok, "bn.")
		if (q > 1) tok = substr(tok, 1, q - 1) + "." + substr(tok, q + 3, .)
		res = (res == "" ? tok : res + "#" + tok)
		if (p == 0) break
		s = substr(s, p + 1, .)
	}
	return(res)
}

string scalar _rf_wb_strip_b(string scalar s)
{
	real scalar p, q
	string scalar res, tok
	res = ""
	while (1) {
		p = strpos(s, "#")
		tok = (p > 0 ? substr(s, 1, p - 1) : s)
		q = strpos(tok, "b.")
		if (q > 1) tok = substr(tok, 1, q - 1) + "." + substr(tok, q + 2, .)
		res = (res == "" ? tok : res + "#" + tok)
		if (p == 0) break
		s = substr(s, p + 1, .)
	}
	return(res)
}

real scalar _rf_wb_find_col(string scalar name, string rowvector fullnames, string rowvector canon, string rowvector nob)
{
	real scalar j, k, hit
	string scalar cname
	k = cols(fullnames)
	cname = _rf_wb_canon(strtrim(name))
	for (j = 1; j <= k; j++) {
		if (fullnames[j] == name || canon[j] == cname) return(j)
	}
	for (j = 1; j <= k; j++) {
		if (nob[j] == cname) return(j)
	}
	// Try unambiguous prefix match if no '.', '#' in name
	if (strpos(cname, ".") == 0 && strpos(cname, "#") == 0 && cname != "_cons") {
		hit = 0
		for (j = 1; j <= k; j++) {
			if (substr(fullnames[j], 1, strlen(cname)) == cname) {
				if (hit > 0) return(0)
				hit = j
			}
		}
		if (hit > 0) return(hit)
	}
	return(0)
}

string rowvector _rf_wb_split_eqs(string scalar spec)
{
	string scalar s, cur, ch
	string rowvector out, toks
	real scalar i, n, depth, has_eq, has_op
	s = strtrim(spec)
	n = strlen(s)
	out = J(1, 0, "")
	if (substr(s, 1, 1) == "(") {
		depth = 0
		cur = ""
		for (i = 1; i <= n; i++) {
			ch = substr(s, i, 1)
			if (ch == "(") {
				if (depth > 0) cur = cur + ch
				depth = depth + 1
			}
			else if (ch == ")") {
				depth = depth - 1
				if (depth < 0) return(J(1, 0, ""))
				if (depth == 0) {
					out = (out, strtrim(cur))
					cur = ""
				}
				else cur = cur + ch
			}
			else {
				if (depth == 0) {
					if (ch != " " && ch != "	") {
						// Mixed parenthesized and bare text not valid
						return(J(1, 0, ""))
					}
				}
				else cur = cur + ch
			}
		}
		if (depth != 0) return(J(1, 0, ""))
		return(out)
	}
	if (strpos(s, ";") > 0) {
		toks = tokens(s, ";")
		for (i = 1; i <= cols(toks); i++) {
			if (toks[i] != ";") out = (out, strtrim(toks[i]))
		}
		return(out)
	}
	has_eq = (strpos(s, "=") > 0)
	has_op = (strpos(s, "+") > 0 || strpos(s, "-") > 0 || strpos(s, "*") > 0)
	if (!has_eq && !has_op) {
		toks = tokens(s)
		if (cols(toks) > 1) return(toks)
	}
	return((s))
}

real scalar _rf_wb_add_side(
	string scalar side,
	real scalar side_sign,
	real rowvector row_r,
	real scalar rhs_val,
	string rowvector fullnames,
	string rowvector canon,
	string rowvector nob,
	real rowvector mask
)
{
	string scalar s, term, ch, prev, prev2, nxt, a, b
	string rowvector terms
	real rowvector tsigns
	real scalar i, n, cur_sign, depth, star, val, va, vb, coef, col, is_exp
	s = strtrim(side)
	n = strlen(s)
	if (n == 0) {
		st_local("wb_err", "wildboot(): empty side in hypothesis")
		return(1)
	}
	terms = J(1, 0, "")
	tsigns = J(1, 0, 0)
	cur_sign = 1
	term = ""
	depth = 0
	for (i = 1; i <= n; i++) {
		ch = substr(s, i, 1)
		if (ch == "(") depth++
		else if (ch == ")") depth--
		prev = (i > 1 ? substr(s, i - 1, 1) : "")
		prev2 = (i > 2 ? substr(s, i - 2, 1) : "")
		nxt = (i < n ? substr(s, i + 1, 1) : "")
		is_exp = (i > 2 && i < n && (prev == "e" || prev == "E" || prev == "d" || prev == "D") && ((prev2 >= "0" && prev2 <= "9") || prev2 == ".") && (nxt >= "0" && nxt <= "9"))
		if (depth == 0 && (ch == "+" || ch == "-") && !is_exp) {
			if (strtrim(term) != "") {
				terms = (terms, strtrim(term))
				tsigns = (tsigns, cur_sign)
				term = ""
				cur_sign = (ch == "+" ? 1 : -1)
			}
			else {
				cur_sign = cur_sign * (ch == "+" ? 1 : -1)
			}
		}
		else {
			term = term + ch
		}
	}
	if (strtrim(term) != "") {
		terms = (terms, strtrim(term))
		tsigns = (tsigns, cur_sign)
	}
	else {
		st_local("wb_err", "wildboot(): trailing operator in hypothesis")
		return(1)
	}
	for (i = 1; i <= cols(terms); i++) {
		term = strtrim(terms[i])
		star = strpos(term, "*")
		if (star > 0) {
			a = strtrim(substr(term, 1, star - 1))
			b = strtrim(substr(term, star + 1, .))
			if (strpos(b, "*") > 0 || a == "" || b == "") {
				st_local("wb_err", "wildboot(): invalid product term in hypothesis: " + term)
				return(1)
			}
			va = strtoreal(a)
			vb = strtoreal(b)
			if (!missing(va) && missing(vb)) {
				coef = va
				term = b
			}
			else if (missing(va) && !missing(vb)) {
				coef = vb
				term = a
			}
			else {
				st_local("wb_err", "wildboot(): invalid scalar multiple in term: " + term)
				return(1)
			}
		}
		else {
			val = strtoreal(term)
			if (!missing(val) && _rf_wb_find_col(term, fullnames, canon, nob) == 0) {
				rhs_val = rhs_val - side_sign * tsigns[i] * val
				continue
			}
			coef = 1
		}
		col = _rf_wb_find_col(term, fullnames, canon, nob)
		if (col == 0) {
			st_local("wb_err", "wildboot(): variable '" + term + "' not found in model")
			return(1)
		}
		if (mask[col] == 0) {
			st_local("wb_err", "wildboot(): variable '" + term + "' is omitted or a base level")
			return(1)
		}
		row_r[col] = row_r[col] + side_sign * tsigns[i] * coef
	}
	return(0)
}

void _rf_wb_build_hyp(
	string scalar spec,
	string scalar rmat_name,
	string scalar rvec_name,
	string scalar full_str,
	string scalar mask_str
)
{
	string rowvector fullnames, canon, nob, eqs
	real rowvector mask, sent_idx, rvec_in, row_r, row_sent
	real matrix rmat_in, R_sent
	real scalar k_full, k_sent, q, eq, j, eqpos, rhs_val
	string scalar lhs, rhs, r_csv, rv_csv

	fullnames = tokens(full_str)
	mask = strtoreal(tokens(mask_str))
	k_full = cols(fullnames)
	sent_idx = selectindex(mask :== 1)
	k_sent = cols(sent_idx)
	if (k_sent == 0) {
		st_local("wb_err", "wildboot(): no estimable parameters in model")
		return
	}

	if (rmat_name != "") {
		rmat_in = st_matrix(rmat_name)
		q = rows(rmat_in)
		if (q == 0) {
			st_local("wb_err", "wildboot(): rmat() has zero rows")
			return
		}
		if (cols(rmat_in) == k_full) {
			for (eq = 1; eq <= q; eq++) {
				for (j = 1; j <= k_full; j++) {
					if (mask[j] == 0 && rmat_in[eq, j] != 0) {
						st_local("wb_err", "wildboot(): rmat() places nonzero weight on omitted/base column " + fullnames[j])
						return
					}
				}
			}
			R_sent = rmat_in[., sent_idx]
		}
		else if (cols(rmat_in) == k_sent) {
			R_sent = rmat_in
		}
		else {
			st_local("wb_err", "wildboot(): rmat() column dimension must match e(b)")
			return
		}
		if (rvec_name != "") {
			rvec_in = vec(st_matrix(rvec_name))'
		}
		else {
			rvec_in = J(1, q, 0)
		}
	}
	else {
		canon = J(1, k_full, "")
		nob = J(1, k_full, "")
		for (j = 1; j <= k_full; j++) {
			canon[j] = _rf_wb_canon(fullnames[j])
			nob[j] = _rf_wb_strip_b(canon[j])
		}
		eqs = _rf_wb_split_eqs(spec)
		q = cols(eqs)
		if (q == 0) {
			st_local("wb_err", "wildboot(): could not parse hypothesis expression")
			return
		}
		R_sent = J(q, k_sent, 0)
		rvec_in = J(1, q, 0)
		for (eq = 1; eq <= q; eq++) {
			eqpos = strpos(eqs[eq], "=")
			rhs_val = 0
			row_r = J(1, k_full, 0)
			if (eqpos > 0) {
				lhs = substr(eqs[eq], 1, eqpos - 1)
				rhs = substr(eqs[eq], eqpos + 1, .)
				if (strpos(rhs, "=") > 0) {
					st_local("wb_err", "wildboot(): multiple '=' in hypothesis equation")
					return
				}
				if (_rf_wb_add_side(lhs, 1, row_r, rhs_val, fullnames, canon, nob, mask)) return
				if (_rf_wb_add_side(rhs, -1, row_r, rhs_val, fullnames, canon, nob, mask)) return
			}
			else {
				if (_rf_wb_add_side(eqs[eq], 1, row_r, rhs_val, fullnames, canon, nob, mask)) return
			}
			row_sent = row_r[sent_idx]
			if (sum(abs(row_sent)) == 0) {
				st_local("wb_err", "wildboot(): hypothesis row has no nonzero coefficient")
				return
			}
			R_sent[eq, .] = row_sent
			rvec_in[eq] = rhs_val
		}
	}

	r_csv = ""
	for (eq = 1; eq <= q; eq++) {
		for (j = 1; j <= k_sent; j++) {
			if (r_csv != "") r_csv = r_csv + ","
			r_csv = r_csv + strtrim(strofreal(R_sent[eq, j], "%21.17g"))
		}
	}
	rv_csv = ""
	for (eq = 1; eq <= q; eq++) {
		if (rv_csv != "") rv_csv = rv_csv + ","
		rv_csv = rv_csv + strtrim(strofreal(rvec_in[eq], "%21.17g"))
	}
	st_local("wb_q", strtrim(strofreal(q, "%21.0f")))
	st_local("wb_R", r_csv)
	st_local("wb_r", rv_csv)
}
end
