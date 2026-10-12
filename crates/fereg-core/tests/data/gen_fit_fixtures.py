"""Ground truth for the Pyfixest flavor of crates/fereg-core/tests/fit_ols.rs: pyfixest 0.60.0.

    python/.venv/bin/python crates/fereg-core/tests/data/gen_fit_fixtures.py   (repository root)

Reads tests/data/fit_d1.csv (written by gen_fit_fixtures.R) and writes tests/data/fit_pyfixest.json
in the layout of fit_fixest.json (the fields pyfixest reports).
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


def case(name, fml, y, x, fe, intercept, vcov, cluster=(), weights=None, weights_type="aweights", ssc=None):
    ssc = {**SSC, **(ssc or {})}
    out = dict(
        name=name, data="fit_d1.csv", y=y, x=x, fe=fe, intercept=intercept, weights=weights, weights_type=weights_type,
        vcov=vcov if isinstance(vcov, str) else "cluster", cluster=[list(d) for d in cluster],
        ssc={"K.adj": ssc["k_adj"], "K.fixef": ssc["k_fixef"], "K.exact": False, "G.adj": ssc["G_adj"], "G.df": ssc["G_df"], "t.df": "min"},
    )
    try:
        m = pf.feols(fml, data, vcov=vcov, weights=weights, weights_type=weights_type, ssc=pf.ssc(**ssc),
                     drop_intercept=not intercept, demeaner=pf.MapDemeaner(fixef_tol=1e-12, fixef_maxiter=100000))
    except Exception as e:  # noqa: BLE001 - the error class is the expectation
        out["error"] = type(e).__name__
        return out
    names = list(m._coefnames)
    out["expect"] = dict(
        coef_names=["(Intercept)" if n == "Intercept" else n for n in names],
        coef=num(m.coef()), se=num(m.se()), vcov=num(m._vcov), nobs=int(m._N_rows) if hasattr(m, "_N_rows") else int(m._N),
        n_eff=float(m._N), df_k=float(m._df_k), df_t=float(m._df_t),
        collin=list(m._collin_vars) if m._collin_vars else [],
        r2=float(m._r2), ar2=float(m._adj_r2),
        wr2=float(m._r2_within) if fe else None, war2=float(m._adj_r2_within) if fe else None, rmse=float(m._rmse),
        resid=num(m.resid()) if vcov == "iid" else [],
    )
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
print(len(cases), "cases")
