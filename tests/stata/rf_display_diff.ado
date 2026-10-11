*! regfast test helper: compare the replay display of two stored estimates line by line
* rf_display_diff REF TEST [, refcmd(reghdfe) testcmd(regfast) options(replay options)]
*   Replays REF with `refcmd' and TEST with `testcmd' into text logs and requires identical lines
*   (log header/footer lines with names, paths and times are skipped).
program rf_display_diff
	version 16
	syntax namelist(min=2 max=2) [, refcmd(name) testcmd(name) options(string)]
	gettoken ref test : namelist
	local test `test'
	if ("`refcmd'" == "") local refcmd reghdfe
	if ("`testcmd'" == "") local testcmd regfast
	tempfile fR fT
	foreach side in R T {
		local est = cond("`side'" == "R", "`ref'", "`test'")
		local cmd = cond("`side'" == "R", "`refcmd'", "`testcmd'")
		qui estimates restore `est'
		qui log using "`f`side''", text replace name(rf_display_diff)
		`cmd', `options'
		qui log close rf_display_diff
	}
	tempname hR hT
	file open `hR' using "`fR'", read text
	file open `hT' using "`fT'", read text
	local nline 0
	local ndiff 0
	ReadLine `hR' lR eofR
	ReadLine `hT' lT eofT
	while (!`eofR' & !`eofT') {
		local ++nline
		if (`"`macval(lR)'"' != `"`macval(lT)'"') {
			local ++ndiff
			if (`ndiff' <= 10) {
				di as err `"line `nline' REF : `macval(lR)'"'
				di as err `"line `nline' TEST: `macval(lT)'"'
			}
		}
		ReadLine `hR' lR eofR
		ReadLine `hT' lT eofT
	}
	if (`eofR' != `eofT') {
		local ++ndiff
		di as err "the displays have different lengths"
	}
	file close `hR'
	file close `hT'
	if (`ndiff') {
		di as err "rf_display_diff `ref' `test': `ndiff' line(s) differ"
		exit 9
	}
	di as text "rf_display_diff `ref' `test': `nline' identical lines"
end

* Read the next line that is not part of the log header/footer
program ReadLine
	args h line eof
	local skip 1
	while (`skip') {
		file read `h' l
		local e = r(eof)
		local t = strtrim(`"`macval(l)'"')
		local skip = !`e' & regexm(`"`macval(t)'"', "^(name|log|log type|opened on|closed on):")
	}
	c_local `line' `"`macval(l)'"'
	c_local `eof' `e'
end
