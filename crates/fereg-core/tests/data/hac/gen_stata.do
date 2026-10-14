* ivreg2 4.1.12 / ivreghdfe 1.1.4 (livreg2.mlib) references for hac.rs.
* Run (uts, in this directory, after gen_data.py): stata-mp -b do gen_stata.do
* Writes stata_kw.csv (m_calckw), stata_kernels.csv (s_vkernel spellings) and
* stata_ref.csv (case,name,i,j,value: e(bw) of bw(auto), HAC e(V)).
version 19
clear all
set more off
sysdir set PLUS "/home/leo/fereg-dev/ado/plus"
mata: mata mlib index

* ---- m_calckw: every kernel, integer and non-integer bandwidths, tau beyond bw ----
mata:
unlink("stata_kw.csv")
fh = fopen("stata_kw.csv", "w")
fput(fh, "kernel,bw,tau,kw")
kernels = ("Bartlett", "Parzen", "Truncated", "Tukey-Hanning", "Tukey-Hamming", "Quadratic Spectral", "Danielle", "Tent")
bws = (1, 2, 3, 2.5, 4.7, 7, 0.6, 12.25)
for (k = 1; k <= cols(kernels); k++) {
    for (b = 1; b <= cols(bws); b++) {
        for (tau = 1; tau <= 15; tau++) {
            fput(fh, kernels[k] + "," + strofreal(bws[b], "%21.17g") + "," + strofreal(tau) + "," + strofreal(m_calckw(tau, bws[b], kernels[k]), "%24.17g"))
        }
    }
}
fclose(fh)
end

* ---- s_vkernel: accepted spellings and the names it returns ----
tempname kh
file open `kh' using "stata_kernels.csv", write replace
file write `kh' "spelling|name" _n
foreach s in "" "bar" "bartlett" "BARTLETT" "  par" "parzen" "tru" "truncated" "thann" "tukey-hanning" "thamm" "tukey-hamming" "qua" "qs" "quadratic-spectral" "quadratic spectral" "dan" "danielle" "ten" "tent" "Tent" "newey" "bartlett " "q" "quadratic" "tukey" {
    capture mata: s_vkernel("`s'", "2", "")
    if _rc file write `kh' "`s'|" _n
    else file write `kh' "`s'|`r(kernel)'" _n
}
file close `kh'

* ---- helpers writing e() results ----
mata:
unlink("stata_ref.csv")
fr = fopen("stata_ref.csv", "w")
fput(fr, "case,name,i,j,value")
void put(string scalar c, string scalar name, real matrix M) {
    external real scalar fr
    real scalar i, j
    for (i = 1; i <= rows(M); i++) for (j = 1; j <= cols(M); j++) {
        fput(fr, c + "," + name + "," + strofreal(i - 1) + "," + strofreal(j - 1) + "," + strofreal(M[i, j], "%24.17g"))
    }
}
end

program define putres
    args case
    mata: put("`case'", "V", st_matrix("e(V)"))
    mata: put("`case'", "b", st_matrix("e(b)"))
    mata: put("`case'", "N", st_numscalar("e(N)"))
    mata: put("`case'", "bw", st_numscalar("e(bw)"))
end

* ---- time series with calendar gaps ----
import delimited using "ts.csv", clear asdouble
tsset t
foreach k in bartlett parzen qs {
    ivreg2 y x1 x2, robust bw(auto) kernel(`k')
    putres abw_`k'
    ivreghdfe y x1 x2, robust bw(auto) kernel(`k')
    putres abwhdfe_`k'
}
* Non-robust (AC): the same automatic bandwidth.
ivreg2 y x1 x2, bw(auto) kernel(qs)
putres abwac_qs
* HAC by kernel (robust + kernel, no cluster).
local specs "bartlett:3 parzen:2.5 truncated:2 thann:3 thamm:2 qs:1.7 dan:2 tent:2 bartlett:1 parzen:4.7"
local i 0
foreach s of local specs {
    local ++i
    gettoken k b : s, parse(":")
    local b = subinstr("`b'", ":", "", .)
    ivreg2 y x1 x2, robust bw(`b') kernel(`k')
    putres hac_ts_`i'
}
ivreg2 y x1 x2 [aw=w], robust bw(3) kernel(bartlett)
putres hac_ts_aw

* Calendar time with delta 2.
tsset t2, delta(2)
foreach k in bartlett qs {
    ivreg2 y x1 x2, robust bw(auto) kernel(`k')
    putres abwd2_`k'
}
ivreg2 y x1 x2, robust bw(3) kernel(parzen)
putres hac_d2_parzen
ivreg2 y x1 x2, robust bw(2.5) kernel(tent)
putres hac_d2_tent

* ---- bw(auto) below its cap: three persistence levels, calendar steps 1 and 3 ----
import delimited using "ts_auto.csv", clear asdouble
foreach tv in t t3 {
    if "`tv'" == "t" tsset t
    else tsset t3, delta(3)
    foreach y in ya yb yc {
        foreach k in bartlett parzen qs {
            ivreg2 `y' xa xb, robust bw(auto) kernel(`k')
            putres auto_`tv'_`y'_`k'
        }
    }
}
tsset t
ivreghdfe yb xa xb, robust bw(auto) kernel(qs)
putres autohdfe_t_yb_qs
ivreg2 yb xa xb, bw(auto) kernel(parzen)
putres autoac_t_yb_parzen

* ---- unbalanced panel with gaps: lags within unit ----
import delimited using "panel.csv", clear asdouble
sort id t
xtset id t
ivreg2 y x1 x2, robust bw(3) kernel(bartlett)
putres hac_panel_bartlett
ivreg2 y x1 x2, robust bw(2) kernel(truncated)
putres hac_panel_truncated
ivreg2 y x1 x2, robust bw(2.2) kernel(qs)
putres hac_panel_qs

mata: fclose(fr)
