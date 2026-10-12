"""Poisson and logit parity with pyfixest 0.60.0 (coefficients / standard errors to 1e-6)."""

import numpy as np
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
