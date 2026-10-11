*! regfast test helper: assert that two numbers agree
* rf_assert_close name a b rtol
*   passes if a == b (including both missing) or reldif(a, b) = |a-b|/(|b|+1) <= rtol
*   rtol = 0 requires exact equality
program rf_assert_close
	version 16
	args name a b rtol
	if ("`rtol'" == "") local rtol 1e-7
	local ok = (`a' == `b')
	if (!`ok' & !missing(`a') & !missing(`b')) local ok = (reldif(`a', `b') <= `rtol')
	if (!`ok') {
		di as err "FAIL `name': " %21.15g `a' " vs " %21.15g `b' " (reldif " %9.2e reldif(`a', `b') ", rtol `rtol')"
		exit 9
	}
end
