* run_all.do -- fereg Stata test suite (needs fereg.plugin on the adopath and the reference commands
* below, e.g. with -sysdir set PLUS- pointing to a directory that holds them)
* Run from tests/stata:  stata-mp -b do run_all.do   -> last line of run_all.log: ALL TESTS PASSED
* (test_ado_mock.do is a separate dry run against a fake plugin and is not part of this suite)
version 16
clear all
set more off
local here `c(pwd)'
* The tolerances and conventions of the tests are those of these versions: stop if the adopath
* finds other ones.
capture program drop rf_need
program rf_need
	args file want
	quietly findfile `file'
	local fn `"`r(fn)'"'
	tempname fh
	file open `fh' using `"`fn'"', read text
	file read `fh' line
	file close `fh'
	if (!strpos(`"`macval(line)'"', `"`want'"')) {
		di as err `"`fn': `macval(line)'"' _n "the tests need `want'"
		exit 9
	}
	di as text `"`fn': `macval(line)'"'
end
rf_need reghdfe.ado "version 6.13.1 "
rf_need ftools.ado "version 2.50.0 "
rf_need ivreghdfe.ado "ivreghdfe 1.1.4 "
rf_need ivreg2.ado "ivreg2 4.1.12 "
rf_need ranktest.ado "ranktest 2.0.04 "
rf_need ppmlhdfe.ado "version 2.3.3 "
foreach t in test_parse test_ols test_iv test_ppml test_logit test_dkraay test_hac test_hc test_crv3 test_conley {
	di as text _n "{hline 78}" _n "running `t'.do" _n "{hline 78}"
	quietly cd "`here'"
	do `t'.do
}
quietly cd "`here'"
* finish inside a program so that "ALL TESTS PASSED" is the last line of the batch log
capture program drop rf_finish
program rf_finish
	di as result "ALL TESTS PASSED"
	exit, STATA clear
end
rf_finish
