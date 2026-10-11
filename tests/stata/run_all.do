* run_all.do -- regfast Stata test suite (needs regfast.plugin on the adopath and reghdfe/ftools installed)
* Run from tests/stata:  stata-mp -b do run_all.do   -> last line of run_all.log: ALL TESTS PASSED
* (test_ado_mock.do is a separate dry run against a fake plugin and is not part of this suite)
version 16
clear all
set more off
local here `c(pwd)'
foreach t in test_parse test_ols {
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
