*! fereg test helper: does a text log contain a string?
* rf_log_has file text
*   r(found) = 1 if some line of the log contains text, 0 otherwise (Stata wraps long output
*   lines at the line size, so text should fit on one line)
program rf_log_has, rclass
	version 16
	args file text
	tempname fh
	file open `fh' using `"`file'"', read text
	local found 0
	file read `fh' line
	while (r(eof) == 0) {
		if (strpos(`"`macval(line)'"', `"`text'"')) local found 1
		file read `fh' line
	}
	file close `fh'
	return scalar found = `found'
end
