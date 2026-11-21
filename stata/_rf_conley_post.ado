*! version 0.5.0  fereg: e() of vce(conley ...) (fereg extension; fixest's Conley estimator)
* Usage (after -ereturn post-): _rf_conley_post, lat() lon() pixel() distance()
*   e(conley_cutoff) (km actually used: the plugin's __rf_cutoff, also when deduced), e(conley_pixel),
*   e(conley_lat), e(conley_lon), e(conley_distance)
program _rf_conley_post, eclass
	version 16
	syntax, lat(string) lon(string) pixel(string) distance(string)
	capture confirm scalar __rf_cutoff
	if (!_rc) ereturn scalar conley_cutoff = scalar(__rf_cutoff)
	ereturn scalar conley_pixel = `pixel'
	ereturn local conley_lat `lat'
	ereturn local conley_lon `lon'
	ereturn local conley_distance `distance'
end
