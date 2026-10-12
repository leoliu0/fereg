"""Ground truth for the Pyfixest flavor of crates/fereg-core/tests/fit_*.rs: pyfixest 0.60.0.

    python/.venv/bin/python crates/fereg-core/tests/data/gen_fit_fixtures.py   (repository root)

Reads tests/data/fit_d1.csv (written by gen_fit_fixtures.R) and writes, in the layout of the
fixest fixtures (the fields pyfixest reports), tests/data/fit_pyfixest.json (OLS),
tests/data/fit_pyfixest_b.json (IV, fixed-effect values, post-hoc variance matrices) and
tests/data/fit_pyfixest_glm.json (Poisson and logit, iwls_tol = 1e-12).
"""

import json
import pathlib

import numpy as np
import pandas as pd
import pyfixest as pf

assert pf.__version__ == "0.60.0"
DIR = pathlib.Path("crates/fereg-core/tests/data")
data = pd.read_csv(DIR / "fit_d1.csv")
SSC = dict(k_adj=True, k_fixef="nonnested", G_adj=True, G_df="min")


def num(v):
    return [None if not np.isfinite(x) else float(x) for x in np.asarray(v, dtype=float).ravel()]


def fit(fml, vcov, weights, weights_type, ssc, intercept, model="ols", offset=None):
    dm = pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100000)
    glm = dict(vcov=vcov, ssc=pf.ssc(**ssc), iwls_tol=1e-12, iwls_maxiter=100, demeaner=dm)
    if model == "poisson":
        return pf.fepois(fml, data, weights=weights, weights_type=weights_type, offset=offset, **glm)
    if model == "logit":
        return pf.feglm(fml, data, family="logit", **glm)
    return pf.feols(fml, data, vcov=vcov, weights=weights, weights_type=weights_type, ssc=pf.ssc(**ssc),
                    drop_intercept=not intercept, demeaner=dm)


def vcov_fields(m):
    return dict(vcov=num(m._vcov), se=num(m.se()), df_k=float(m._df_k), df_t=float(m._df_t))


def case(name, fml, y, x, fe, intercept, vcov, cluster=(), weights=None, weights_type="aweights", ssc=None,
         endog=(), inst=(), fixef=False, posthoc=(), model="ols", offset=None):
    ssc = {**SSC, **(ssc or {})}
    out = dict(
        name=name, data="fit_d1.csv", model=model, y=y, x=x, endog=list(endog), inst=list(inst), fe=fe, intercept=intercept,
        weights=weights, weights_type=weights_type, offset=offset,
        vcov=vcov if isinstance(vcov, str) else "cluster", cluster=[list(d) for d in cluster],
        ssc={"K.adj": ssc["k_adj"], "K.fixef": ssc["k_fixef"], "K.exact": False, "G.adj": ssc["G_adj"], "G.df": ssc["G_df"], "t.df": "min"},
        posthoc=[dict(vcov=p[0] if isinstance(p[0], str) else "cluster", cluster=[list(d) for d in p[1]],
                      ssc={"K.adj": True, "K.fixef": "nonnested", "K.exact": False, "G.adj": True, "G.df": "min", "t.df": "min"})
                 for p in posthoc],
    )
    try:
        m = fit(fml, vcov, weights, weights_type, ssc, intercept, model, offset)
    except Exception as e:  # noqa: BLE001 - the error class is the expectation
        out["error"] = type(e).__name__
        return out
    names = list(m._coefnames)
    out["expect"] = dict(
        coef_names=["(Intercept)" if n == "Intercept" else n for n in names],
        # pyfixest's feglm does not update _N_rows after its separation check.
        coef=num(m.coef()), nobs=int(m._N_rows) if hasattr(m, "_N_rows") and model == "ols" else int(m._N),
        n_eff=float(m._N), collin=list(m._collin_vars) if m._collin_vars else [],
        **vcov_fields(m),
    )
    if model == "ols":
        out["expect"].update(
            r2=float(m._r2), ar2=float(m._adj_r2),
            wr2=float(m._r2_within) if fe else None, war2=float(m._adj_r2_within) if fe else None, rmse=float(m._rmse),
            resid=num(m.resid()) if vcov == "iid" else [],
        )
    else:
        # pyfixest reports no log-likelihood for logit, and none of these with some options.
        out["expect"]["deviance"] = float(m.deviance)
        for key, attr in [("ll", "_loglik"), ("ll_null", "_loglik_null"), ("pr2", "_pseudo_r2")]:
            if getattr(m, attr, None) is not None:
                out["expect"][key] = float(getattr(m, attr))
    if endog:
        m.first_stage()
        f1 = m._model_1st_stage
        out["expect"]["first"] = dict(
            coef_names=["(Intercept)" if n == "Intercept" else n for n in f1._coefnames], coef=num(f1.coef()),
            f_stat=float(m._f_stat_1st_stage), f_p=float(m._p_value_1st_stage), **vcov_fields(f1),
        )
    if fixef:
        fx = m.fixef(atol=1e-14, btol=1e-14)
        out["expect"]["fixef"] = [dict(name=k[2:-1], levels=list(v.keys()), values=num(list(v.values()))) for k, v in fx.items()]
    out["expect"]["posthoc"] = []
    for p in posthoc:
        m.vcov(p[0])
        out["expect"]["posthoc"].append(vcov_fields(m))
    return out


F2 = "y ~ x1 + x2 | g + h"
fe2 = [["g"], ["h"]]
cases = [
    case("py_fe2_iid", F2, "y", ["x1", "x2"], fe2, True, "iid"),
    case("py_fe2_hetero", F2, "y", ["x1", "x2"], fe2, True, "hetero"),
    case("py_fe2_cl_g", F2, "y", ["x1", "x2"], fe2, True, {"CRV1": "g"}, [["g"]]),
    case("py_fe2_cl_c1", F2, "y", ["x1", "x2"], fe2, True, {"CRV1": "c1"}, [["c1"]]),
    case("py_fe2_cl_c3_h", F2, "y", ["x1", "x2"], fe2, True, {"CRV1": "c3+h"}, [["c3"], ["h"]]),
    case("py_fe2_cl_c1_c3", F2, "y", ["x1", "x2"], fe2, True, {"CRV1": "c1+c3"}, [["c1"], ["c3"]]),
    case("py_fe2_cl_c1_c3_conv", F2, "y", ["x1", "x2"], fe2, True, {"CRV1": "c1+c3"}, [["c1"], ["c3"]], ssc={"G_df": "conventional"}),
    case("py_fe2_cl_c3_none", F2, "y", ["x1", "x2"], fe2, True, {"CRV1": "c3"}, [["c3"]], ssc={"k_fixef": "none"}),
    # g is nested in c1 by value; fixest skips the value test because c1 is itself a fixed effect.
    case("py_fe_g_c1_cl_c1", "y ~ x1 + x2 | g + c1", "y", ["x1", "x2"], [["g"], ["c1"]], True, {"CRV1": "c1"}, [["c1"]]),
    case("py_collin", "y ~ x1 + x2 + x3 + xg | g + h", "y", ["x1", "x2", "x3", "xg"], fe2, True, "iid"),
    case("py_nofe_iid", "y ~ x1 + x2", "y", ["x1", "x2"], [], True, "iid"),
    case("py_nofe_cl", "y ~ x1 + x2", "y", ["x1", "x2"], [], True, {"CRV1": "c3"}, [["c3"]]),
    case("py_noint_iid", "y ~ x1 + x2", "y", ["x1", "x2"], [], False, "iid"),
    case("py_fe1_aw_hetero", "y ~ x1 | g", "y", ["x1"], [["g"]], True, "hetero", weights="w"),
    case("py_fe1_aw_cl", "y ~ x1 | g", "y", ["x1"], [["g"]], True, {"CRV1": "c1"}, [["c1"]], weights="w"),
    case("py_fe1_fw_iid", "y ~ x1 | g", "y", ["x1"], [["g"]], True, "iid", weights="wi", weights_type="fweights"),
    case("py_fe1_fw_hetero", "y ~ x1 | g", "y", ["x1"], [["g"]], True, "hetero", weights="wi", weights_type="fweights"),
    case("py_fe1_fw_cl", "y ~ x1 | g", "y", ["x1"], [["g"]], True, {"CRV1": "c3"}, [["c3"]], weights="wi", weights_type="fweights"),
    case("py_nofe_aw_iid", "y ~ x1 + x2", "y", ["x1", "x2"], [], True, "iid", weights="w"),
    case("py_nacluster", "y ~ x1 | g", "y", ["x1"], [["g"]], True, {"CRV1": "cna"}, [["cna"]]),
]
(DIR / "fit_pyfixest.json").write_text(json.dumps(cases))
print("fit_pyfixest.json", len(cases), "cases")

FIV = "yiv ~ x1 | g + h | xe ~ z1 + z2"
iv = dict(endog=["xe"], inst=["z1", "z2"])
cases = [
    case("py_iv_fe2_iid", FIV, "yiv", ["x1"], fe2, True, "iid", **iv),
    case("py_iv_fe2_hetero", FIV, "yiv", ["x1"], fe2, True, "hetero", **iv),
    case("py_iv_fe2_cl_c3", FIV, "yiv", ["x1"], fe2, True, {"CRV1": "c3"}, [["c3"]], **iv),
    case("py_iv_fe2_cl_c3_h", FIV, "yiv", ["x1"], fe2, True, {"CRV1": "c3+h"}, [["c3"], ["h"]], **iv),
    case("py_iv_fe2_w_hetero", FIV, "yiv", ["x1"], fe2, True, "hetero", weights="w", **iv),
    case("py_iv_nofe_hetero", "yiv ~ x1 | xe ~ z1 + z2", "yiv", ["x1"], [], True, "hetero", **iv),
    case("py_iv_fe1_just", "yiv ~ x1 | g | xe ~ z1", "yiv", ["x1"], [["g"]], True, "iid", endog=["xe"], inst=["z1"]),
    case("py_fx_fe1", "y ~ x1 + x2 | g", "y", ["x1", "x2"], [["g"]], True, "iid", fixef=True),
    case("py_fx_fe2", F2, "y", ["x1", "x2"], fe2, True, "iid", fixef=True,
         posthoc=[("hetero", []), ({"CRV1": "c1+c3"}, [["c1"], ["c3"]]), ({"CRV1": "g"}, [["g"]])]),
    case("py_fx_fe2_comp", "y ~ x1 + x2 | g + hb", "y", ["x1", "x2"], [["g"], ["hb"]], True, "iid", fixef=True),
    case("py_fx_fe2_comp_w", "y ~ x1 + x2 | hb + g", "y", ["x1", "x2"], [["hb"], ["g"]], True, "iid", weights="w", fixef=True),
    case("py_fx_fe3", "y ~ x1 + x2 | g + h + k", "y", ["x1", "x2"], [["g"], ["h"], ["k"]], True, "iid", fixef=True),
    case("py_fx_fe3_comp", "y ~ x1 + x2 | g + hb + k", "y", ["x1", "x2"], [["g"], ["hb"], ["k"]], True, "iid", fixef=True),
    case("py_posthoc_iv", FIV, "yiv", ["x1"], fe2, True, "iid", posthoc=[("hetero", []), ({"CRV1": "c3"}, [["c3"]])], **iv),
]
(DIR / "fit_pyfixest_b.json").write_text(json.dumps(cases))
print("fit_pyfixest_b.json", len(cases), "cases")

FP = "yp ~ x1 + x2 | g + h"
FB = "yb ~ x1 + x2 | g + h"
pois = dict(model="poisson")
logit = dict(model="logit")
cases = [
    case("py_pois_fe2_iid", FP, "yp", ["x1", "x2"], fe2, True, "iid", **pois),
    case("py_pois_fe2_hetero", FP, "yp", ["x1", "x2"], fe2, True, "hetero", **pois),
    case("py_pois_fe2_cl_c3", FP, "yp", ["x1", "x2"], fe2, True, {"CRV1": "c3"}, [["c3"]], **pois),
    case("py_pois_fe2_off_w", FP, "yp", ["x1", "x2"], fe2, True, "hetero", weights="w", offset="lo", **pois),
    case("py_pois_nofe", "yp ~ x1 + x2", "yp", ["x1", "x2"], [], True, "iid", posthoc=[("hetero", [])], **pois),
    case("py_logit_fe2_iid", FB, "yb", ["x1", "x2"], fe2, True, "iid", **logit),
    case("py_logit_fe2_cl_g", FB, "yb", ["x1", "x2"], fe2, True, {"CRV1": "g"}, [["g"]], **logit),
    case("py_logit_fe1_hetero", "yb ~ x1 + x2 | g", "yb", ["x1", "x2"], [["g"]], True, "hetero", **logit),
    case("py_logit_nofe", "yb ~ x1 + x2", "yb", ["x1", "x2"], [], True, "iid", **logit),
]
(DIR / "fit_pyfixest_glm.json").write_text(json.dumps(cases))
print("fit_pyfixest_glm.json", len(cases), "cases")
