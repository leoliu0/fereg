# fereg plugin protocol (ado ⇄ plugin)

The ado files in `stata/` drive the plugin through `plugin call`. This file is the contract
between them; change both sides together.

Version: `fereg_core::VERSION` (`0.4.0`). Every ado sets `local rf_version 0.4.0`, and
`_rf_load` (no arguments) compares it with `cmd=version`.

## Calling convention

- `stata/_rf_load.ado` is the only caller. A `program …, plugin` defined in an ado-file is
  private to that file, so `_rf_load.ado` defines `_rf_plugin`, runs
  `plugin call _rf_plugin <varlist> [in], <cmd> "k=v" …`, and copies the plugin's result locals
  to its caller with `c_local` (the `forward` list: `_rf_version`, `_rf_token`, `_rf_status`,
  the Stock–Yogo locals).
- `argv[0]` is the command; every other token is `key=value`. Unknown keys are r(198) naming
  the key. Every token must be quoted (`"tol=1e-08"`): unquoted, Stata splits arguments at
  `= | ; * -`, rejects commas and mangles numbers (`1e-08` → `1e`).
- The plugin sets local `x` of the calling program by saving macro `_x`.
- Errors: the plugin prints one line with `SF_error` and returns the Stata rc (`RfError::rc`).
- Results: scalars `__rf_<name>` (the ado copies them into `e()` and drops them); locals
  `_rf_<name>`; matrices and variables only into objects the ado pre-created with the exact
  dimensions (r(503) otherwise; NaN is stored as missing). Only kept observations are written.
- Notes are name-free: the plugin cannot see variable names, so per-variable messages
  (collinear, absorbed, depvar explained by the FEs) are printed by the ado from `_rf_status`
  and `__rf_y_absorbed`.
- Common keys: `threads=T` (0 = all logical CPUs), `verbose=0|1|2`, `serialio=0|1` (1 = serial
  reads; reads are parallel by default), debug `rowread=auto|direct|kept` on estimation commands.

## cmd=version
No varlist. Sets `_rf_version`.

## cmd=selftest (ABI check; `tests/stata/stress_io.do`)
Varlist: numeric `v1..vk`; sample from `if`/`in`. Keys: `serialio` (default 1), `threads`,
`write=j` (store the observation number into variable j), `sortby=j`, `rowread`, `spin=<s>`.
Results: `__rf_nobs`, `__rf_chk` = Σ_obs Σ_j j·v_j (missing = 0, left-to-right in obs order),
`__rf_tread`.

## cmd=prepare
Varlist, in order: factor variables (`nfac`, numeric or string), slope variables (`nslope`),
weight (iff `wtype≠none`), model variables (`nmodel`, numeric; the depvar first iff
`family≠gaussian`), touse (last, numeric 0/1).
Sample: observations in the `in` range with touse ≠ 0 and no missing value in a factor (string
factors: ""), slope, weight or model variable. `if` is not evaluated: callers fold it, the
weights and the terms that are not plain model variables (factor-variable or time-series
operators) into touse. Touse and every numeric variable are read in one parallel pass.

Keys:
- `nfac`, `factype` (one `n|s` per factor), `nslope`, `nmodel` (default 0).
- `absorb=none` (only the constant) or `;`-separated terms `<factors>|<icpt>|<slopes>|<save>`:
  1-based factor positions joined by `*` (interaction), intercept 0/1, 1-based slope positions
  joined by `*` or `-` for none, save 0/1. Example: `1|1|-|0;2*3|1|-|1;4|0|1*2|1`.
- `vce=…` and its companion keys: section "Variance estimators" below.
- `wtype=none|fw|aw|pw`, `singletons=0|1`.
- `sample=touse|all` (default touse): `all` asserts that touse is 1 at every observation of
  the range (no `if`, `in`, weights or ado markouts), so the scan does not read it.
- `family=gaussian|poisson|logit`; `sep=fe|none` (GLM only): the fe rule (Poisson: drop groups
  with Σy = 0; logit: Σy ∈ {0, count}) runs inside the singleton loop. feppml sends `sep=fe`
  only when singletons are dropped, as ppmlhdfe.
- `absorber=fast|dense` (debug; dense = explicit-dummy reference).

Effects: writes 0 into touse for dropped observations (missing values, singletons, fe rule);
local `_rf_token` names the stored session (a new prepare discards the previous one); scalars
`__rf_N_full` (after the missing-value exclusion), `__rf_N`, `__rf_num_singletons`,
`__rf_num_fe_separated`. The session keeps the model variables for the estimation command.

### Variance estimators (prepare keys)

The VCE is fixed at prepare (it decides the cluster dimensions of the DoF rules) and applies to
the estimation command that consumes the session. The prepare varlist gains the Conley
coordinates: factors, slopes, **coordinates** (`ncoord` = 0 or 2: lat then lon, degrees,
numeric; missing values drop the observation), weight, model variables, touse. The tsset
time and panel variables are sent as numeric factors (missing values drop the observation, as
the markout of ivreghdfe/reghdfe); a variable used both as a factor (absorb, cluster) and as
tvar/ivar is sent once and referenced by the same position.

| key | values | meaning |
|---|---|---|
| `vce` | `unadjusted robust cluster hc2 hc3 crv3 dkraay conley` | base estimator |
| `cluster` | `;`-separated dims `f1*f2` | with `vce=cluster` and `vce=crv3` (also the kernel clusters) |
| `tvar` | factor position | `_dta[_TStvar]`; required by `dkraay`, `kernel`, `kiefer` |
| `ivar` | factor position | `_dta[_TSpanel]` when xtset (kernel lags stay within panel) |
| `tdelta` | real (default 1) | `_dta[_TSdelta]` |
| `dkbw` | integer ≥ 1 | `vce(dkraay #)` bandwidth (= lags + 1); absent = reghdfe default `floor(4(T/100)^(2/9))+1` |
| `kernel` | `bartlett parzen truncated tukey-hanning tukey-hamming quadratic-spectral danielle tent` | ivreg2 HAC kernel (any `s_vkernel` spelling is accepted) |
| `bw` | real > 0 or `auto` | ivreg2 `bw()`; required with `kernel` unless `kiefer=1` |
| `kiefer` | 0/1 | ivreg2 `kiefer`: kernel Truncated, bw = T (computed by the plugin); send `kernel=truncated` or no kernel, no `bw` |
| `psd` | `none psd0 psda` | ivreg2 `psd0`/`psda` on S |
| `ncoord` | 0 or 2 | Conley coordinates in the varlist |
| `cutoff` | real > 0, km | Conley cutoff; absent = fixest's deduced cutoff |
| `pixel` | real ≥ 0, km (default 0) | Conley pixel |
| `distance` | `triangular spherical` (default triangular) | Conley distance |

The ado resolves syntax (abbreviations, option combinations, ms_parse_vce / ivparse /
s_vkernel / regress messages) before calling; the plugin re-checks with the same texts.

ivreg2 kernel family (`kernel=` or `kiefer=1` present), by `vce`:
- `unadjusted`: AC (no robust; kiefer sends this). `robust`: HAC (ivreghdfe `robust`; the ado
  also sends `robust` for pweights, as ivreghdfe forces robust after ivparse).
- `cluster` with `cluster=<tvar>`: Driscoll–Kraay (ivreghdfe `dkraay(#)` = `vce=cluster
  cluster=<tvar> kernel=bartlett bw=#`); `cluster=<ivar>;<tvar>` (either order): Thompson
  two-way + kernel. Other dims: r(198) `Error: cluster kernel-robust requires clustering on tsset
  time variable.` / `… on tsset panel & time vars.`
- Lags are calendar lags on tvar (step tdelta), within ivar when sent. T = max(t) − min(t) + 1
  over the estimation sample. The plugin errors r(198) `invalid bandwidth in option bw() -
  cannot exceed timespan of data` when bw > (T−1)/tdelta, r(198) `Automatic bandwidth selection
  not available for panel data` for `bw=auto` with ivar, r(102) `bandwidth option bw() required
  for HAC-robust estimation`, r(451) `repeated time values within panel` / `repeated time values
  in sample` (kernel lags on duplicate (ivar, t)). `bw=auto` with a kernel other than Bartlett,
  Parzen, QS prints `Error: kernel <K> not compatible with bw(auto)` and uses bw = 1 (as
  ivreghdfe's `abw`). The ado prints the s_vkernel bw = 1 note, the tsreport gap warning and
  the kiefer/dkraay option errors.
- Small sample (ivreghdfe, sdofminus = df_a): no cluster ×N/(N−K−df_a); cluster
  ×(N−1)/(N−K−df_a)·G/(G−1), G = #periods (DK) or min(#panels, #periods) (Thompson);
  df_r = N − K − df_a, or G − 1 with a cluster. On `cmd=ols` K counts the reported constant
  (reghdfe's design rows with `_cons`; the slope block equals ivreghdfe's). On `cmd=glm`
  (borrowed) the meat uses the IRLS-weighted working residuals; factor N/(N−1) without a
  cluster, G/(G−1) with one.
- The DoF clusters (reghdfe `clusters` rule) are the `cluster=` dims, as ivreghdfe creates the
  HDFE with `vce(cluster …)`.

`vce=dkraay`: `cmd=ols` = reghdfe 6.13.1 `vce(dkraay)` (positional periods, Bartlett, bw =
dkbw or the default, DoF clusters = tvar, no nested adjustment, PSD fix, df_r = min(df_r,
T−1)); prints `Warning: bandwidth %g too large for %g time periods; reducing to %g`.
`cmd=iv` = ivreghdfe `dkraay(dkbw)` (calendar lags, Bartlett, cluster tvar). `cmd=glm` =
reghdfe's formula with the asymptotic factor T/(T−1).

`vce=hc2|hc3`: Stata `regress` eq. (10)/(12) with the leverage of the full [X, FE dummies,
slopes] design (no N/(N−k) factor; fweights τ_i = w_i; df_r as robust). `cmd=iv`: r(198)
`vce() only supports cluster, robust, and unadjusted`. `wtype=pw` with hc3: r(198)
`pweights not allowed with option vce(hc3)` (regress); pweights with hc2 are treated as
aweights. Observations with leverage 1 − h ≤ √ε (residual 0 in exact arithmetic; regress
returns arbitrary numbers there) contribute nothing [fereg rule]. GLM: IRLS leverage, no
factor.

`vce=crv3`: leave-one-cluster-out jackknife Σ_g(β₍₋g₎−β̂)(β₍₋g₎−β̂)' centered at β̂, times the
reghdfe cluster factor (`cmd=ols`), ivreghdfe's (`cmd=iv`, ≤ 2 dims) or G/(G−1) (`cmd=glm`);
multiway by inclusion–exclusion over the dims with the PSD fix; df_r = min(df_r, G_min − 1).
A refit that loses a coefficient is r(498) naming the cluster level (1-based code) and the
coefficient position.

`vce=conley`: fixest 0.14.2 Conley meat (uniform kernel, its constants and range errors);
factor N/(N−df_m−df_a), df_r = N−df_m−df_a (`cmd=ols`, `cmd=iv` with K for df_m); `cmd=glm`
N/(N−1).

Results added by the estimation commands:
- scalars `__rf_bw` (kernel bandwidth used: bw, the auto result or kiefer's T),
  `__rf_dkraay_bw` (`vce=dkraay` on ols/glm: lags + 1 after the reduction; on iv equal to
  `__rf_bw`), `__rf_N_clust` (DK: T), `__rf_N_clust1`/`2`, `__rf_N_clustervars`,
  `__rf_cutoff` (Conley km used), `__rf_hac_T` (time span T used by the kernel).
- locals `_rf_kernel` (ivreg2 proper name: `Bartlett`, `Quadratic Spectral`, `Danielle`, …),
  `_rf_bwchoice` (`Automatic bw selection according to Newey-West (1994)` or the abw
  incompatibility text; empty otherwise).

## Estimation commands (cmd=ols, iv, glm)
- `token=<_rf_token>` is required and the session is consumed either way. A wrong token or a
  changed `_N` is r(198) "stale plugin session". The plugin reuses prepare's observation list,
  so the ado must not sort or drop observations in between.
- Varlist: estimation variables, then touse, then `nout` output variables.
- Shared keys: `nx`, `nout`, `tol`, `maxiter`, `dof=<pairwise,clusters,continuous,firstpair>|none`;
  output positions (1-based among the output variables, 0 = none) `resid=`, `d=`, `groupvar=`,
  `fe=<k1,k2,…>|0` (one per saved slot: per saved term, intercept then slopes); matrices
  `b=`, `V=`, `doft=` (n_slots × 5), each optional.
- `cached=<k1,k2,…>`: one entry per estimation variable (glm: y and x, not the offset), the
  1-based position of the same variable among prepare's model variables, whose stored values
  are used instead of a Stata read; 0 = read. Absent: every variable is read.
- Missing values in estimation rows read from Stata: r(416).
- Local `_rf_status`: one code per regressor sent (0 ok, 2 absorbed by the FEs, 3 collinear).
- `est::finish` zeroes touse for observations an estimator drops after prepare.

### cmd=ols
Varlist: `y x… touse out…`. Extra key `noconstant=0|1`. `b` is 1×K and `V` K×K with
K = nx + report_constant (report_constant = noabsorb or any intercept term, and not
noconstant); omitted coefficients are 0 with zero rows/columns in V.
Scalars (reghdfe `e()` names): `N rank df_r df_m tss tss_within rss mss rmse F ll ll_0 r2
r2_within r2_a r2_a_within sumweights ic converged report_constant N_clust N_clustervars
N_clust1… df_a df_a_initial df_a_redundant df_a_nested num_singletons N_full`, plus
`y_absorbed`. The ado sets `N_hdfe`, `N_hdfe_extended`, `drop_singletons`.

### cmd=iv
Varlist: `y exog… endog… inst… touse out…`; keys `nx`, `nendog`, `ninst`, `first=0|1`,
`ffirst=0|1`. Coefficient order is endog then exog (ivreghdfe), no `_cons`.
Matrices: `b` 1×(nendog+nx), `V`; `S`, `W` (ninst+nx)² over [inst, exog]; optional `firstm`
(21×nendog `e(first)` rows: rmse sheapr2 pr2 F df df_r pvalue SWF SWFdf1 SWFdf2 SWFp SWchi2
SWchi2p SWr2 APF APFdf1 APFdf2 APFp APchi2 APchi2p APr2), `ccev`, `cdev` (1×nendog), and name
prefixes `bfirst=P`, `Vfirst=P`, `sfirst=P` for the first-stage regressions (`P#` for the #-th
sent endogenous regressor: b, V, and rmse df_r df_m N).
Scalars: ivreghdfe `e()` names (N N_full df_r df_m sdofminus dofminus center partial_ct
partialcons cons nocollin inexog_ct exexog_ct endog_ct r2 r2u r2c r2_a rmse rss mss yy yyc ll F
Fp Fdf1 Fdf2 condxx condzz rankxx rankzz rankS rankV j jdf jp sargan… N_clust… idstat iddf idp
cd cdf widstat rkf arf… sstat… df_a… num_singletons ic). Locals `_rf_sy_<t>` (L1, K1) and
`_rf_sy1_<t>` (one endogenous regressor), t ∈ ivbias5 ivbias10 ivbias20 ivbias30 ivsize10
ivsize15 ivsize20 ivsize25. Errors: 481 underidentified after drops; 198 for more than two
cluster dimensions.

### cmd=glm (feppml, felogit)
Varlist: `y x… [offset] touse out…`. Keys: `family=poisson|logit`, `tol` (IRLS deviance,
1e-8), `maxiter` (10000), `itol` (target inner tolerance, 1e-9; the final inner tolerance is
max(1e-12, min(itol, 0.1·tol))), `separation=<relu,simplex,mu>|none` (simplex = relu;
defaults: Poisson relu, logit none; relu with logit is r(198)), `guess=simple|ols` (Poisson),
`log=0|1`, `offset=0|1`. `resid=` is rejected. `b` is 1×(nx+1) with `_cons` last.
Scalars: `N rank df_m ll ll_0 deviance r2_p chi2 converged ic ic2 num_separated
num_singletons num_fe_separated N_full df_a df_a_initial df_a_redundant df_a_nested`, cluster
counts, and for Poisson `df`. Counting follows ppmlhdfe: Poisson num_singletons includes the
fe-rule drops and num_separated the relu and mu drops; logit num_separated includes the fe
rule. VCE: Poisson unadjusted is robust (N/(N−1)); logit default is oim; clusters
G/(G−1); multiway gets the PSD fix. Errors: Poisson y < 0 r(459), no convergence r(430),
everything separated r(2001). Scalar `separation_complete` is 0 when the ReLU check stopped
without certifying its result (no convergence; the `mu` rule then runs as a fallback and a
warning note is printed), else 1.
Every `vce=` of prepare is accepted (section "Variance estimators"); unadjusted means
Poisson robust and logit oim, as before.

`sepcheck=1` (Poisson only; feppml `tagsep()` / `zvarname()`, ppmlhdfe's secret options): runs
the separation check on the prepared sample and stores nothing else; no estimation, no
matrices, touse unchanged, and only the outputs `sep=<pos>` (1 = separated, 0 = not, on the
prepared sample) and `zsep=<pos>` (certificate of separation: positive on separated
observations, 0 elsewhere) are allowed. The check is the `fe` rule (all-zero FE groups,
iterated with singleton removal; only when prepare ran with `sep=none`; its
rows get certificate 1) and ReLU rounds repeated on the remaining sample until one finds
nothing. Scalars: `N` (prepared sample), `num_separated`, `separation_complete`,
`relu_iterations` (summed over the rounds).
