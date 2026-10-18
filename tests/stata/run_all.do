* run_all.do -- fereg Stata test suite (needs fereg.plugin on the adopath and the reference commands
* below, e.g. with -sysdir set PLUS- pointing to a directory that holds them)
* Run from tests/stata:  stata-mp -b do run_all.do   -> last line of run_all.log: ALL TESTS PASSED
* (test_ado_mock.do is a separate dry run against a fake plugin and is not part of this suite)
version 16
clear all
set more off
local here `c(pwd)'
* The tolerances and conventions of the tests are those of these versions: stop if the adopath
* finds other ones. rf_need checks the first "*!" line of the file (the first line if none).
capture program drop rf_need
program rf_need
	args file want
	quietly findfile `file'
	local fn `"`r(fn)'"'
	tempname fh
	file open `fh' using `"`fn'"', read text
	file read `fh' line
	local first `"`macval(line)'"'
	while (r(eof) == 0 & substr(`"`macval(line)'"', 1, 2) != "*!") {
		file read `fh' line
	}
	file close `fh'
	if (substr(`"`macval(line)'"', 1, 2) != "*!") local line `"`macval(first)'"'
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
rf_need logitfe.ado "logitfe v3.0.3 mgonza 24Feb2017"
* moremata's mm_sample() (logitfe's multiple(), the reference of test_logitfe; felogit does not need
* it); a library added to the adopath in a running session is found after -mata mlib index-
capture mata: mata which mm_sample()
if (_rc) {
	quietly mata: mata mlib index
	capture mata: mata which mm_sample()
	if (_rc) {
		di as err "the tests need moremata (ssc install moremata)"
		exit 9
	}
}
foreach t in test_parse test_ols test_iv test_ppml test_logit test_dkraay test_hac test_hc test_crv3 test_conley test_logitfe {
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
