"""Poisson, logit and gaussian GLM parity with pyfixest 0.60.0 (coefficients / standard errors
to 1e-6), separation checks included."""

import re
import warnings

import numpy as np
import pandas as pd
import pytest
from _parity import TIGHT, assert_same_summary, compare, make_data, pf

import fereg

DATA = make_data()
GLM_KW = {"iwls_tol": 1e-12, "iwls_maxiter": 200}

POIS_CASES = [
    ("y_pois ~ x1 + x2", None, {}),
    ("y_pois ~ x1 + x2 | f1", None, {}),
    ("y_pois ~ x1 + x2 | f1 + f2", None, {}),
    ("y_pois ~ x1 + x2 | f1 + f2", "hetero", {}),
    ("y_pois ~ x1 + x2 | f1 + f2", {"CRV1": "cl"}, {}),
    ("y_pois ~ x1 + x2 | f1 + f2", {"CRV1": "cl+cl2"}, {}),
    ("y_pois ~ x1 + x2 | f1 + f2", None, {"offset": "off"}),
    ("y_pois ~ x1 + x2 | f1 + f2", "hetero", {"weights": "w"}),
]


@pytest.mark.parametrize("fml,vcov,kw", POIS_CASES, ids=[f"{c[0]}|{c[1]}|{c[2]}" for c in POIS_CASES])
def test_poisson_parity(fml, vcov, kw):
    theirs = pf.fepois(fml, DATA, vcov=vcov, demeaner=TIGHT, **GLM_KW, **kw)
    ours = fereg.fepois(fml, DATA, vcov=vcov, **GLM_KW, **kw)
    assert isinstance(ours, fereg.Fepois)
    compare(ours, theirs, rtol=1e-6)
    np.testing.assert_allclose(ours.deviance, theirs.deviance, rtol=1e-8)
    np.testing.assert_allclose(ours._loglik, theirs._loglik, rtol=1e-8)
    if theirs._pseudo_r2 is None:
        assert ours._pseudo_r2 is None
    else:
        np.testing.assert_allclose(ours._pseudo_r2, theirs._pseudo_r2, rtol=1e-6)
    np.testing.assert_allclose(ours.pvalue(), theirs.pvalue(), rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(ours.resid(), theirs.resid(), rtol=1e-6, atol=1e-8)


def test_poisson_predict():
    fml = "y_pois ~ x1 + x2 | f1 + f2"
    theirs = pf.fepois(fml, DATA, offset="off", demeaner=TIGHT, **GLM_KW)
    ours = fereg.fepois(fml, DATA, offset="off", **GLM_KW)
    for type_ in ("link", "response"):
        np.testing.assert_allclose(ours.predict(type=type_), theirs.predict(type=type_), rtol=1e-6)
    theirs.fixef(atol=1e-14, btol=1e-14)
    new = DATA.head(50).copy()
    new.loc[new.index[:2], "f2"] = 777.0
    pt, po = theirs.predict(newdata=new, type="response"), ours.predict(newdata=new, type="response")
    assert np.array_equal(np.isnan(pt), np.isnan(po))
    np.testing.assert_allclose(po[~np.isnan(po)], pt[~np.isnan(pt)], rtol=1e-5)


LOGIT_CASES = [
    ("y_bin ~ x1 + x2", None),
    ("y_bin ~ x1 + x2 | f1", None),
    ("y_bin ~ x1 + x2 | f1 + f3", None),
    ("y_bin ~ x1 + x2 | f1 + f3", "hetero"),
    ("y_bin ~ x1 + x2 | f1 + f3", {"CRV1": "cl"}),
]


@pytest.mark.parametrize("fml,vcov", LOGIT_CASES, ids=[f"{c[0]}|{c[1]}" for c in LOGIT_CASES])
def test_logit_parity(fml, vcov):
    theirs = pf.feglm(fml, DATA, family="logit", vcov=vcov, demeaner=TIGHT, **GLM_KW)
    ours = fereg.feglm(fml, DATA, family="logit", vcov=vcov, **GLM_KW)
    assert isinstance(ours, fereg.Felogit)
    compare(ours, theirs, rtol=1e-6)
    np.testing.assert_allclose(ours.deviance, theirs.deviance, rtol=1e-8)
    np.testing.assert_allclose(ours.predict(type="response"), theirs.predict(type="response"), rtol=1e-6)


@pytest.mark.parametrize("family", ["poisson", "logit"])
def test_collinear_regressors_dropped_as_pyfixest(family):
    """xc = x1 + 2*x2 (x2 is dropped, after it) and cl, constant within f1 (absorbed)."""
    if family == "poisson":
        fml, theirs_fit, ours_fit, kw = "y_pois ~ x1 + xc + x2 + cl | f1 + f2", pf.fepois, fereg.fepois, {}
    else:
        fml, theirs_fit, ours_fit, kw = "y_bin ~ x1 + xc + x2 + cl | f1 + f2", pf.feglm, fereg.feglm, {"family": "logit"}
    theirs = theirs_fit(fml, DATA, vcov="hetero", demeaner=TIGHT, **GLM_KW, **kw)
    ours = ours_fit(fml, DATA, vcov="hetero", **GLM_KW, **kw)
    assert list(ours._coefnames) == ["x1", "xc"]
    compare(ours, theirs, rtol=1e-6)


def test_logit_all_one_levels(capsys):
    """FE levels with an all-one outcome stay in pyfixest's sample; their FE is +inf in fereg
    (pyfixest stops near +25) and their predictions sit at the bound."""
    data = DATA.copy()
    data.loc[data["f1"].isin([1, 2]), "y_bin"] = 1.0
    fml = "y_bin ~ x1 + x2 | f1 + f3"
    theirs = pf.feglm(fml, data, family="logit", demeaner=TIGHT, **GLM_KW)
    ours = fereg.feglm(fml, data, family="logit", **GLM_KW)
    compare(ours, theirs, rtol=1e-6)
    np.testing.assert_allclose(ours.deviance, theirs.deviance, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(ours.resid(), theirs.resid(), rtol=1e-6, atol=1e-8)
    ft, fo = theirs.fixef(atol=1e-14, btol=1e-14), ours.fixef()
    assert list(fo) == list(ft) and list(fo["C(f1)"]) == list(ft["C(f1)"])
    assert fo["C(f1)"]["1"] == np.inf and ft["C(f1)"]["1"] > 20
    finite = [k for k, v in fo["C(f1)"].items() if np.isfinite(v)]
    np.testing.assert_allclose([fo["C(f1)"][k] for k in finite], [ft["C(f1)"][k] for k in finite], rtol=1e-5)
    for new in (None, data):
        po, pt = ours.predict(newdata=new, type="response"), theirs.predict(newdata=new, type="response")
        assert np.array_equal(np.isnan(po), np.isnan(pt))
        np.testing.assert_allclose(po[~np.isnan(po)], pt[~np.isnan(pt)], rtol=1e-6, atol=1e-10)
    in_sample = ours.predict(newdata=data)[ours._obs]
    np.testing.assert_allclose(in_sample, ours.predict(), rtol=1e-10)
    ours.summary()
    assert "Deviance" in capsys.readouterr().out
    fereg.etable([ours])


def test_logit_rejects_nonbinary():
    with pytest.raises(ValueError, match="0 and 1"):
        fereg.feglm("y_pois ~ x1 | f1", DATA, family="logit")


@pytest.mark.parametrize("family", ["poisson", "logit"])
def test_glm_summary_matches_pyfixest(family, capsys):
    if family == "poisson":
        fml = "y_pois ~ x1 + x2 | f1"
        theirs = pf.fepois(fml, DATA, demeaner=TIGHT, **GLM_KW)
    else:
        fml = "y_bin ~ x1 + x2 | f1"
        theirs = pf.feglm(fml, DATA, family="logit", demeaner=TIGHT, **GLM_KW)
    assert_same_summary(fereg.feglm(fml, DATA, family=family, **GLM_KW), theirs, capsys)


# --------------------------------------------------------------------------- separation


def _separation_data(singleton: bool) -> pd.DataFrame:
    """Outcomes separated by FE levels (all zero in f = 3, 4 and g = 5) and by a regressor (d)."""
    rng = np.random.default_rng(3)
    n = 600
    df = pd.DataFrame({"f": rng.integers(0, 30, n), "g": rng.integers(0, 10, n), "x": rng.normal(size=n)})
    df["cl"] = df["f"] // 5
    df["d"] = (rng.uniform(size=n) < 0.1).astype(float)
    df["y"] = rng.poisson(np.exp(0.3 * df["x"])).astype(float)
    df.loc[df["d"] == 1, "y"] = 0.0
    df.loc[df["f"].isin([3, 4]) | (df["g"] == 5), "y"] = 0.0
    # a g level of two rows, one separated by d: a singleton after separation, kept by pyfixest
    df.loc[[0, 1], "g"] = 99
    df.loc[[0, 1], ["d", "y"]] = [[1.0, 0.0], [0.0, 2.0]]
    if singleton:
        # an FE singleton (dropped unless fixef_rm="none")
        df.loc[2, "f"] = 100
    df["yb"] = (df["y"] > 0).astype(float)
    df["yg"] = rng.normal(size=n) + 2 + 0.5 * df["x"]
    df.loc[df["f"] == 3, "yg"] = -1.0
    df.loc[df["f"] == 5, "yg"] = 0.0
    df["yp"] = df["y"] + 1.0  # no zero: nothing to separate
    return df


# pyfixest's `ir` check misreads its own predictions once rows were dropped before it
# (singletons, missing values: its index has gaps), and then stops at its iteration cap
# without dropping anything; its `ir` cases therefore use data without such rows.
SEP = _separation_data(singleton=False)
SEP1 = _separation_data(singleton=True)
SEP_CASES = [
    ("poisson", "y ~ x | f + g", {}, SEP1),
    ("poisson", "y ~ x | f + g", {"vcov": {"CRV1": "cl"}}, SEP1),
    ("poisson", "y ~ x + d | f + g", {"separation_check": ["ir"]}, SEP),
    ("poisson", "y ~ x + d | f + g", {"separation_check": ["fe", "ir"]}, SEP),
    ("poisson", "y ~ x | f + g", {"fixef_rm": "none"}, SEP1),
    ("poisson", "y ~ x | f + g", {"separation_check": []}, SEP1),
    ("poisson", "yp ~ x + C(g)", {"separation_check": ["fe", "ir"]}, SEP),  # no FE
    ("poisson", "yp ~ x | f + g", {"separation_check": ["fe", "ir"]}, SEP),  # nothing separated
    ("logit", "yb ~ x | f + g", {}, SEP1),
    ("logit", "yb ~ x + d | f + g", {"separation_check": ["ir"]}, SEP),
    ("logit", "yb ~ x + d | f + g", {"separation_check": ["fe", "ir"], "vcov": {"CRV1": "cl"}}, SEP),
    ("logit", "yb ~ x | f + g", {"fixef_rm": "none"}, SEP1),
    # pyfixest's gaussian iid variance omits the residual variance (Feglm._vcov_iid = bread),
    # which fereg does not replicate: the gaussian cases use other estimators
    ("gaussian", "yg ~ x | f + g", {"vcov": "hetero"}, SEP1),
    ("gaussian", "yg ~ x | f + g", {"vcov": {"CRV1": "cl"}}, SEP1),
    ("gaussian", "yg ~ x | f + g", {"vcov": "hetero", "fixef_rm": "none"}, SEP1),
    ("gaussian", "yg ~ x | f + g", {"vcov": "hetero", "separation_check": []}, SEP1),
    ("gaussian", "yb ~ x + d | f + g", {"vcov": {"CRV1": "cl"}, "separation_check": ["ir"]}, SEP),
]


def _fit_recording(fit, *args, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = fit(*args, **kw)
    return m, [(x.category, str(x.message)) for x in w if issubclass(x.category, UserWarning)]


@pytest.mark.parametrize(
    "family,fml,kw,data", SEP_CASES, ids=[f"{c[0]}|{c[1]}|{c[2]}|{len(c[3])}" for c in SEP_CASES]
)
def test_separation_matches_pyfixest(family, fml, kw, data):
    """Rows dropped by pyfixest's separation checks (fepois and every feglm family), its
    "N observations removed because of separation." warning, and the estimates after them."""
    if family == "poisson":
        theirs, w_theirs = _fit_recording(pf.fepois, fml, data, demeaner=TIGHT, **GLM_KW, **kw)
    else:
        theirs, w_theirs = _fit_recording(pf.feglm, fml, data, family=family, demeaner=TIGHT, **GLM_KW, **kw)
    ours, w_ours = _fit_recording(fereg.feglm, fml, data, family=family, **GLM_KW, **kw)
    assert w_ours == w_theirs
    compare(ours, theirs, rtol=1e-6)


@pytest.mark.parametrize("family,y", [("poisson", "y"), ("logit", "yb")])
def test_no_separation_check_without_fixed_effects(family, y):
    """pyfixest runs no separation check without fixed effects: separated rows stay (their
    coefficient diverges; the others agree)."""
    fml = f"{y} ~ x + d + C(g)"
    kw = {"separation_check": ["fe", "ir"], **GLM_KW}
    if family == "poisson":
        theirs, w_theirs = _fit_recording(pf.fepois, fml, SEP, **kw)
    else:
        theirs, w_theirs = _fit_recording(pf.feglm, fml, SEP, family=family, **kw)
    ours, w_ours = _fit_recording(fereg.feglm, fml, SEP, family=family, **kw)
    assert not any("separation" in m for _, m in w_ours + w_theirs)
    assert ours._N == theirs._N == len(SEP)
    np.testing.assert_allclose(ours.coef()["x"], theirs.coef()["x"], rtol=1e-4)


def test_gaussian_iid_uses_residual_variance():
    """Deliberate difference: pyfixest 0.60.0's gaussian iid variance is its bread alone
    (Feglm._vcov_iid; dispersion fixed at 1), fereg's is the OLS one, SSR / (N - df_k) (X'X)^-1
    with the fixed effects as dummies. Everything else (sample after the separation checks,
    df_k with the levels present before them, estimates) matches pyfixest."""
    fml = "yg ~ x | f + g"
    ours = fereg.feglm(fml, SEP1, family="gaussian", vcov="iid")
    theirs = pf.feglm(fml, SEP1, family="gaussian", vcov="iid", demeaner=TIGHT, **GLM_KW)
    assert (ours._N, ours._df_k, ours._df_t) == (theirs._N, theirs._df_k, theirs._df_t)
    np.testing.assert_allclose(ours._beta_hat, theirs._beta_hat, rtol=1e-8)
    d = SEP1.iloc[np.asarray(ours._out.obs)]
    X = np.column_stack(
        [d["x"], pd.get_dummies(d["f"], dtype=float), pd.get_dummies(d["g"], drop_first=True, dtype=float)]
    )
    beta, *_ = np.linalg.lstsq(X, d["yg"].to_numpy(), rcond=None)
    ssr = np.sum((d["yg"].to_numpy() - X @ beta) ** 2)
    xtx_inv = np.linalg.pinv(X.T @ X)[0, 0]
    n, df_k = theirs._N, theirs._df_k
    np.testing.assert_allclose(ours._se[0], np.sqrt(ssr / (n - df_k) * xtx_inv), rtol=1e-8)
    np.testing.assert_allclose(theirs._se[0], np.sqrt((n - 1) / (n - df_k) * xtx_inv), rtol=1e-6)


def test_separation_warning_count_and_fepois():
    _, w = _fit_recording(fereg.fepois, "y ~ x + d | f + g", SEP, separation_check=["fe", "ir"])
    _, w_pf = _fit_recording(pf.fepois, "y ~ x + d | f + g", SEP, separation_check=["fe", "ir"])
    sep = [m for _, m in w if "separation" in m]
    assert sep == [m for _, m in w_pf if "separation" in m]
    assert len(sep) == 1 and int(sep[0].split()[0]) > 0
    _, w = _fit_recording(fereg.fepois, "yp ~ x | f + g", SEP)
    assert not any("separation" in m for _, m in w)


def test_separation_check_errors_match_pyfixest():
    for bad, err in ((("fe",), TypeError), (["fe", "x"], ValueError)):
        with pytest.raises(err) as theirs:
            pf.fepois("y ~ x | f", SEP, separation_check=bad)
        with pytest.raises(err, match=re.escape(str(theirs.value))):
            fereg.fepois("y ~ x | f", SEP, separation_check=bad)
