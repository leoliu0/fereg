"""Ground truth for the Pyfixest flavor of crates/fereg-core/tests/fit_se.rs: pyfixest 0.60.0.

    ~/fereg-dev/pyenv/bin/python crates/fereg-core/tests/data/gen_se_fixtures.py   (repository root)

Reads tests/data/fit_se.csv (written by gen_se_fixtures.R) and writes tests/data/fit_se_pyfixest.json
in the layout of the fixest fixtures. pyfixest's own numbers where it defines the type (HC2/HC3
without fixed effects, Newey-West, Driscoll-Kraay, CRV3 of OLS and Poisson); fereg's extensions are
checked against:
  - HC2/HC3 with fixed effects: pyfixest with explicit dummies (the regressors' block);
  - CRV3 of logit and IV: a leave-one-cluster-out loop of pyfixest refits (logit refits are logit:
    pyfixest's own CRV3 refits every GLM with `fepois`) with the CRV1 small-sample factors;
  - HAC of IV: the second stage through a post-hoc `vcov()` (pyfixest's IV fit with NW/DK fails in
    its first stage), the first stage as its own `feols`.
"""

import json
import pathlib

import numpy as np
import pandas as pd
import pyfixest as pf

assert pf.__version__ == "0.60.0"
DIR = pathlib.Path("crates/fereg-core/tests/data")
data = pd.read_csv(DIR / "fit_se.csv")
SSC = {"K.adj": True, "K.fixef": "nonnested", "K.exact": False, "G.adj": True, "G.df": "min", "t.df": "min"}


def num(v):
    return [None if not np.isfinite(x) else float(x) for x in np.asarray(v, dtype=float).ravel()]


def dm():
    return pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100000)


def fit(model, fml, df, vcov="iid", vcov_kwargs=None, weights=None, weights_type="aweights"):
    if model == "poisson":
        return pf.fepois(fml, df, vcov=vcov, vcov_kwargs=vcov_kwargs, weights=weights, weights_type=weights_type,
                         iwls_tol=1e-12, iwls_maxiter=100, demeaner=dm())
    if model == "logit":
        return pf.feglm(fml, df, family="logit", vcov=vcov, vcov_kwargs=vcov_kwargs, iwls_tol=1e-12, iwls_maxiter=100,
                        demeaner=dm())
    return pf.feols(fml, df, vcov=vcov, vcov_kwargs=vcov_kwargs, weights=weights, weights_type=weights_type, demeaner=dm())


def names_of(m):
    return ["(Intercept)" if n == "Intercept" else n for n in m._coefnames]


def fields(m, keep=None):
    """vcov / se / df of a fit, restricted to the coefficients `keep` (by name)."""
    names = names_of(m)
    idx = [names.index(k) for k in keep] if keep else list(range(len(names)))
    v = np.asarray(m._vcov)[np.ix_(idx, idx)]
    return dict(vcov=num(v), se=num(np.sqrt(np.diag(v))), df_k=float(m._df_k), df_t=float(m._df_t),
                coef_names=[names[i] for i in idx], coef=num(np.asarray(m.coef())[idx]), nobs=int(m._N_rows) if hasattr(m, "_N_rows") else int(m._N))


def kwargs_of(v, time, panel):
    if not isinstance(v, dict):
        return None
    kw = {"time_id": time}
    if panel is not None:
        kw["panel_id"] = panel
    if v.get("lag") is not None:
        kw["lag"] = v["lag"]
    return kw


def hac_extra(m, v, time):
    t = len(np.unique(m._data[time]))
    return dict(lag=v["lag"] if v.get("lag") is not None else int(np.floor(t ** 0.25)), n_periods=t)


def crv3_brute(model, fml, m, cluster, weights=None, weights_type="aweights"):
    """pyfixest CRV3 by refits: dims (1, 2, 1^2 when two-way) with signs +, +, - and the CRV1
    small-sample factors of the same clustering, centered at the full-sample coefficients."""
    sub = m._data
    names = list(m._coefnames)
    beta = np.asarray(m.coef())
    ids = [sub[c].astype(str) for c in cluster]
    if len(ids) == 2:
        ids.append(ids[0] + "-" + ids[1])
    m1 = fit(model, fml, sub, vcov={"CRV1": "+".join(cluster)}, weights=weights, weights_type=weights_type)
    v = np.zeros((len(beta), len(beta)))
    for d, ssc in zip(ids, np.atleast_1d(m1._ssc)):
        meat = np.zeros_like(v)
        for g in pd.unique(d):
            f = fit(model, fml, sub[d != g], weights=weights, weights_type=weights_type)
            b = np.asarray(f.coef()[names]) - beta
            meat += np.outer(b, b)
        v += ssc * meat
    out = dict(vcov=num(v), se=num(np.sqrt(np.diag(v))), df_k=float(m1._df_k), df_t=float(m1._df_t),
               coef_names=names_of(m), coef=num(beta), nobs=int(m._N), G=[float(len(pd.unique(x))) for x in ids[:len(cluster)]])
    return out


def case(name, model, fml, y, x, fe, vcov, cluster=(), weights=None, weights_type="aweights", time=None, panel=None,
         endog=(), inst=(), dense=None, pf_panel=None, first=None):
    """`dense`: the explicit-dummy formula of an HC2/HC3 extension case; `pf_panel`: the panel
    pyfixest needs (DK) when the case itself has none; `first`: the first-stage formula of IV."""
    out = dict(name=name, data="fit_se.csv", model=model, y=y, x=x, endog=list(endog), inst=list(inst), fe=fe, intercept=True,
               weights=weights, weights_type=weights_type, vcov=vcov, cluster=[[c] for c in cluster], time=time, panel=panel,
               ssc=SSC)
    hac = isinstance(vcov, dict)
    pv = {"hc2": "HC2", "hc3": "HC3"}.get(vcov, vcov) if not hac else vcov["type"]
    kw = kwargs_of(vcov, time, panel if panel is not None else pf_panel)
    try:
        if vcov == "crv3" and (model == "logit" or endog):
            m = fit(model, fml, data, weights=weights, weights_type=weights_type)
            out["expect"] = crv3_brute(model, fml, m, cluster, weights, weights_type)
        elif vcov == "crv3":
            m = fit(model, fml, data, vcov={"CRV3": "+".join(cluster)}, weights=weights, weights_type=weights_type)
            out["expect"] = fields(m)
            out["expect"]["G"] = [float(len(pd.unique(m._data[c]))) for c in cluster]
        elif dense is not None:
            m = fit(model, dense, data, vcov=pv, weights=weights, weights_type=weights_type)
            out["expect"] = fields(m, keep=x)
        elif hac and endog:
            m = fit(model, fml, data, weights=weights, weights_type=weights_type)
            m.vcov(pv, vcov_kwargs=kw)
            out["expect"] = fields(m)
        else:
            m = fit(model, fml, data, vcov=pv, vcov_kwargs=kw, weights=weights, weights_type=weights_type)
            out["expect"] = fields(m)
    except Exception as e:  # noqa: BLE001 - the message is the expectation
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["expect"]["collin"] = []
    if hac:
        out["expect"].update(hac_extra(m, vcov, time))
    if first is not None:
        if vcov == "crv3":
            f = fit(model, first, data, weights=weights, weights_type=weights_type)
            out["expect"]["first"] = crv3_brute(model, first, f, cluster[:1], weights, weights_type)
        else:
            f = fit(model, first, data, vcov=pv, vcov_kwargs=kw, weights=weights, weights_type=weights_type)
            out["expect"]["first"] = fields(f)
    return out


X2 = ["x1", "x2"]
SPECS = {"nofe": ("x1 + x2", []), "fe1": ("x1 + x2 | g", [["g"]]), "fe2": ("x1 + x2 | g + h", [["g"], ["h"]])}
DUMMIES = {"fe1": "x1 + x2 + C(g)", "fe2": "x1 + x2 + C(g) + C(h)"}
NW = lambda lag=None: {"type": "NW", "lag": lag}  # noqa: E731
DK = lambda lag=None: {"type": "DK", "lag": lag}  # noqa: E731
cases = []


def add(*a, **k):
    cases.append(case(*a, **k))


for sp, (rhs, fe) in SPECS.items():
    for hc in ("hc2", "hc3"):
        dense = None if sp == "nofe" else f"y ~ {DUMMIES[sp]}"
        add(f"py_{hc}_{sp}", "ols", f"y ~ {rhs}", "y", X2, fe, hc, dense=dense)
        add(f"py_{hc}_aw_{sp}", "ols", f"y ~ {rhs}", "y", X2, fe, hc, weights="w", dense=dense)
        add(f"py_{hc}_fw_{sp}", "ols", f"y ~ {rhs}", "y", X2, fe, hc, weights="wi", weights_type="fweights", dense=dense)
    add(f"py_nw_{sp}", "ols", f"y ~ {rhs}", "y", X2, fe, NW(), time="t", panel="id")
    add(f"py_nw3_{sp}", "ols", f"y ~ {rhs}", "y", X2, fe, NW(3), time="t", panel="id")
    add(f"py_nwts3_{sp}", "ols", f"y ~ {rhs}", "y", X2, fe, NW(3), time="ts")
    add(f"py_dk_{sp}", "ols", f"y ~ {rhs}", "y", X2, fe, DK(), time="t", panel="id")
    add(f"py_dk_nopanel_{sp}", "ols", f"y ~ {rhs}", "y", X2, fe, DK(1), time="t", pf_panel="id")
    add(f"py_nw_aw_{sp}", "ols", f"y ~ {rhs}", "y", X2, fe, NW(), time="t", panel="id", weights="w")
    add(f"py_crv3_{sp}", "ols", f"y ~ {rhs}", "y", X2, fe, "crv3", cluster=["c3"])
add("py_nw20_fe1", "ols", "y ~ x1 + x2 | g", "y", X2, [["g"]], NW(20), time="t", panel="id")
add("py_crv3_tw_fe2", "ols", "y ~ x1 + x2 | g + h", "y", X2, [["g"], ["h"]], "crv3", cluster=["c3", "h"])
add("py_crv3_aw_fe1", "ols", "y ~ x1 + x2 | g", "y", X2, [["g"]], "crv3", cluster=["g"], weights="w")
add("py_crv3_fw_fe1", "ols", "y ~ x1 + x2 | g", "y", X2, [["g"]], "crv3", cluster=["c3"], weights="wi", weights_type="fweights")
add("py_crv3_tw_nofe", "ols", "y ~ x1 + x2", "y", X2, [], "crv3", cluster=["c3", "h"])
# IV.
FIV = "yiv ~ x1 | g + h | xe ~ z1 + z2"
FIRST = "xe ~ z1 + z2 + x1 | g + h"
iv = dict(endog=["xe"], inst=["z1", "z2"], first=FIRST)
add("py_iv_nw_fe2", "ols", FIV, "yiv", ["x1"], [["g"], ["h"]], NW(), time="t", panel="id", **iv)
add("py_iv_dk_fe2", "ols", FIV, "yiv", ["x1"], [["g"], ["h"]], DK(), time="t", panel="id", **iv)
add("py_iv_crv3_fe2", "ols", FIV, "yiv", ["x1"], [["g"], ["h"]], "crv3", cluster=["c3"], **iv)
add("py_iv_crv3_tw_fe2", "ols", FIV, "yiv", ["x1"], [["g"], ["h"]], "crv3", cluster=["c3", "h"], **iv)
add("py_iv_hc2_nofe", "ols", "yiv ~ x1 | xe ~ z1 + z2", "yiv", ["x1"], [], "hc2", endog=["xe"], inst=["z1", "z2"])
# Poisson and logit.
for model, y in (("poisson", "yp"), ("logit", "yb")):
    pre = model[:4]
    add(f"py_{pre}_hc2_nofe", model, f"{y} ~ x1 + x2", y, X2, [], "hc2")
    add(f"py_{pre}_hc3_nofe", model, f"{y} ~ x1 + x2", y, X2, [], "hc3")
    add(f"py_{pre}_hc2_fe2", model, f"{y} ~ x1 + x2 | g + h", y, X2, [["g"], ["h"]], "hc2", dense=f"{y} ~ {DUMMIES['fe2']}")
    add(f"py_{pre}_nw_fe2", model, f"{y} ~ x1 + x2 | g + h", y, X2, [["g"], ["h"]], NW(), time="t", panel="id")
    add(f"py_{pre}_dk_fe2", model, f"{y} ~ x1 + x2 | g + h", y, X2, [["g"], ["h"]], DK(), time="t", panel="id")
    add(f"py_{pre}_nwts2_nofe", model, f"{y} ~ x1 + x2", y, X2, [], NW(2), time="ts")
    add(f"py_{pre}_crv3_fe2", model, f"{y} ~ x1 + x2 | g + h", y, X2, [["g"], ["h"]], "crv3", cluster=["c3"])
    add(f"py_{pre}_crv3_nofe", model, f"{y} ~ x1 + x2", y, X2, [], "crv3", cluster=["c3"])
add("py_pois_hc2_aw_nofe", "poisson", "yp ~ x1 + x2", "yp", X2, [], "hc2", weights="w")
(DIR / "fit_se_pyfixest.json").write_text(json.dumps(cases))
print("fit_se_pyfixest.json", len(cases), "cases")
for c in cases:
    if "error" in c:
        print(c["name"], c["error"][:200])
