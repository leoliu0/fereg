"""HC2/HC3, CRV3, Newey-West, Driscoll-Kraay and Conley variance matrices.

Parity with pyfixest 0.60.0 where it has the estimator (computed live), and for fereg's
extensions with references built here: pyfixest with the fixed effects as explicit dummies
(HC2/HC3 with fixed effects), leave-one-cluster-out refits with pyfixest (CRV3 for IV and
logit, which pyfixest rejects or refits as Poisson), pyfixest's own post-hoc IV Newey-West (its
estimation-time IV HAC fails in the first stage), and fixest 0.14.2 numbers (Conley, written by
``data/gen_conley_fixest.R``).
"""

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from _parity import TIGHT, assert_same_summary, compare, make_data, pf

import fereg

DATA = make_data()
GLM_KW = {"iwls_tol": 1e-12, "iwls_maxiter": 200}
# pyfixest computes its GLM variance from the IRLS weights at the start of its last iteration,
# ~1e-7 from the optimum fereg uses (the existing GLM parity tolerance).
GLM_RTOL = 1e-6
DATA_DIR = pathlib.Path(__file__).parent / "data"


def make_panel(seed: int = 11) -> pd.DataFrame:
    """Unbalanced panel with gaps: 60 units observed in 5-12 of the years 2000-2015."""
    rng = np.random.default_rng(seed)
    rows = [(u, t) for u in range(60) for t in np.sort(rng.choice(np.arange(2000, 2016), rng.integers(5, 13), replace=False))]
    df = pd.DataFrame(rows, columns=["unit", "year"])
    n = len(df)
    a = rng.normal(size=60)[df.unit]
    shock = rng.normal(size=16)[df.year - 2000]
    e = np.zeros(n)
    for i in range(n):  # AR(1) errors within the unit
        e[i] = rng.normal() + (0.6 * e[i - 1] if i and df.unit[i] == df.unit[i - 1] else 0.0)
    df["x1"] = rng.normal(size=n) + 0.5 * shock
    df["x2"] = rng.normal(size=n)
    df["z1"] = rng.normal(size=n)
    df["z2"] = rng.normal(size=n)
    df["e1"] = 0.7 * df.z1 + 0.4 * df.z2 + 0.5 * e + rng.normal(size=n)
    df["y"] = 1 + 0.5 * df.x1 - 0.3 * df.x2 + a + shock + e
    df["y_iv"] = 0.6 * df.e1 + 0.5 * df.x1 + a + shock + e
    eta = 0.2 + 0.3 * df.x1 - 0.2 * df.x2 + 0.3 * a + 0.2 * shock
    df["y_pois"] = rng.poisson(np.exp(eta)).astype(float)
    df["y_bin"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    df["w"] = rng.uniform(0.5, 2.0, n)
    df["fw"] = rng.integers(1, 4, n).astype(float)
    df["cl"] = df.unit // 4
    df["date"] = pd.to_datetime(df.year.astype(str) + "-06-30")
    return df


PANEL = make_panel()
TS = PANEL.loc[PANEL.unit == 0].reset_index(drop=True)
HAC = {"time_id": "year", "panel_id": "unit"}


def assert_vcov(ours, theirs, rtol):
    np.testing.assert_allclose(ours._vcov, np.asarray(theirs._vcov), rtol=rtol, atol=0)


# --------------------------------------------------------------------------- HC2 / HC3


HC_CASES = [
    ("HC2", {}),
    ("HC3", {}),
    ("HC2", {"weights": "w"}),
    ("HC3", {"weights": "w"}),
    ("HC2", {"weights": "fw", "weights_type": "fweights"}),
    ("HC3", {"weights": "fw", "weights_type": "fweights"}),
]


@pytest.mark.parametrize("vcov,kw", HC_CASES, ids=[f"{v}|{k}" for v, k in HC_CASES])
def test_hc_no_fe(vcov, kw):
    fml = "y ~ x1 + x2 + x3"
    theirs = pf.feols(fml, DATA, vcov=vcov, **kw)
    ours = fereg.feols(fml, DATA, vcov=vcov, **kw)
    compare(ours, theirs, rtol=1e-8)
    assert (ours._vcov_type, ours._vcov_type_detail) == (theirs._vcov_type, theirs._vcov_type_detail) == ("hetero", vcov)
    # post hoc, from an iid fit
    ours = fereg.feols(fml, DATA, **kw).vcov(vcov)
    compare(ours, theirs, rtol=1e-8)


@pytest.mark.parametrize("vcov", ["HC2", "HC3"])
def test_hc_poisson_no_fe(vcov):
    fml = "y_pois ~ x1 + x2"
    theirs = pf.fepois(fml, DATA, vcov=vcov, **GLM_KW)
    ours = fereg.fepois(fml, DATA, vcov=vcov, **GLM_KW)
    compare(ours, theirs, rtol=1e-8, se_rtol=GLM_RTOL)


HC_FE_CASES = [
    ("HC2", "| f3 + h", "+ C(f3) + C(h)", {}),
    ("HC3", "| f3 + h", "+ C(f3) + C(h)", {}),
    ("HC3", "| f3", "+ C(f3)", {"weights": "w"}),
    ("HC2", "| f3^h", "+ C(f3):C(h)", {}),
]


@pytest.mark.parametrize("vcov,fe,dummies,kw", HC_FE_CASES, ids=[f"{c[0]}{c[1]}|{c[3]}" for c in HC_FE_CASES])
def test_hc_fe_extension_matches_dummies(vcov, fe, dummies, kw):
    """fereg extension: HC2/HC3 with fixed effects = pyfixest with explicit dummies."""
    theirs = pf.feols(f"y ~ x1 + x2 {dummies}", DATA, vcov=vcov, **kw)
    ours = fereg.feols(f"y ~ x1 + x2 {fe}", DATA, vcov=vcov, **kw)
    assert ours._N == theirs._N and ours._df_k == theirs._df_k and ours._df_t == theirs._df_t
    np.testing.assert_allclose(ours.coef().to_numpy(), theirs.coef()[["x1", "x2"]].to_numpy(), rtol=1e-8)
    np.testing.assert_allclose(ours._se, theirs.se()[["x1", "x2"]].to_numpy(), rtol=1e-8)
    assert ours._vcov_type_detail == vcov
    # post hoc
    np.testing.assert_allclose(fereg.feols(f"y ~ x1 + x2 {fe}", DATA, **kw).vcov(vcov)._se, ours._se, rtol=1e-12)


def test_hc_fe_extension_poisson():
    theirs = pf.fepois("y_pois ~ x1 + x2 + C(f3) + C(h)", DATA, vcov="HC3", **GLM_KW)
    ours = fereg.fepois("y_pois ~ x1 + x2 | f3 + h", DATA, vcov="HC3", **GLM_KW)
    np.testing.assert_allclose(ours._se, theirs.se()[["x1", "x2"]].to_numpy(), rtol=GLM_RTOL)


def test_hc_iv_is_an_error():
    msg = "HC2 and HC3 inference types are not supported for IV regressions."
    with pytest.raises(fereg.VcovTypeNotSupportedError, match=msg):
        fereg.feols("y_iv ~ x1 | f1 | e1 ~ z1", DATA, vcov="HC2")
    m = fereg.feols("y_iv ~ x1 | f1 | e1 ~ z1", DATA)
    with pytest.raises(fereg.VcovTypeNotSupportedError, match=msg):
        m.vcov("HC3")


# --------------------------------------------------------------------------- CRV3


def crv3_reference(fit, fml, data, cluster, beta, ssc, **kw):
    """Leave-one-cluster-out jackknife (pyfixest `_vcov_crv3_slow` with tight refits)."""
    codes = data[cluster].to_numpy()
    meat = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for g in np.unique(codes):
            d = fit(fml, data.loc[codes != g].reset_index(drop=True), vcov="iid", **kw).coef().to_numpy() - beta
            meat = meat + np.outer(d, d)
    return ssc * meat


CRV3_FAST = [
    ("y ~ x1 + x2", {"CRV3": "cl"}, {}),
    ("y ~ x1 + x2", {"CRV3": "cl"}, {"weights": "w"}),
    ("y ~ x1 + x2", {"CRV3": "cl+f3"}, {}),
]


@pytest.mark.parametrize("fml,vcov,kw", CRV3_FAST, ids=[f"{c[0]}|{c[1]}|{c[2]}" for c in CRV3_FAST])
def test_crv3_ols_no_fe(fml, vcov, kw):
    theirs = pf.feols(fml, DATA, vcov=vcov, **kw)
    ours = fereg.feols(fml, DATA, vcov=vcov, **kw)
    compare(ours, theirs, rtol=1e-8)
    assert_vcov(ours, theirs, 1e-8)
    assert (ours._vcov_type, ours._vcov_type_detail, ours._clustervar) == ("CRV", "CRV3", theirs._clustervar)
    # post hoc: re-estimation
    assert_vcov(fereg.feols(fml, DATA, **kw).vcov(vcov), theirs, 1e-8)


def test_crv3_ols_fe():
    fml = "y ~ x1 + x2 | f1 + f3"
    ours = fereg.feols(fml, DATA, vcov={"CRV3": "cl"})
    # pyfixest's own (its refits use pyfixest's default tolerances)
    theirs = pf.feols(fml, DATA, vcov={"CRV3": "cl"}, demeaner=TIGHT)
    compare(ours, theirs, rtol=1e-8)
    # the same jackknife with tight refits
    crv1 = pf.feols(fml, DATA, vcov={"CRV1": "cl"}, demeaner=TIGHT)
    ref = crv3_reference(pf.feols, fml, crv1._data, "cl", crv1._beta_hat, crv1._ssc[0], demeaner=TIGHT)
    np.testing.assert_allclose(ours._vcov, ref, rtol=1e-8)
    np.testing.assert_allclose(fereg.feols(fml, DATA).vcov({"CRV3": "cl"})._vcov, ours._vcov, rtol=1e-12)


def test_crv3_poisson():
    fml = "y_pois ~ x1 + x2 | f3"
    ours = fereg.fepois(fml, DATA, vcov={"CRV3": "cl"}, **GLM_KW)
    theirs = pf.fepois(fml, DATA, vcov={"CRV3": "cl"}, demeaner=TIGHT, **GLM_KW)
    compare(ours, theirs, rtol=1e-8, se_rtol=1e-7)
    crv1 = pf.fepois(fml, DATA, vcov={"CRV1": "cl"}, demeaner=TIGHT, **GLM_KW)
    ref = crv3_reference(pf.fepois, fml, crv1._data, "cl", crv1._beta_hat, crv1._ssc[0], demeaner=TIGHT, **GLM_KW)
    np.testing.assert_allclose(ours._vcov, ref, rtol=1e-9)


def test_crv3_logit_extension():
    """Logit refits are logit fits (pyfixest refits a Poisson model)."""
    fml = "y_bin ~ x1 + x2 | f3"
    ours = fereg.feglm(fml, DATA, family="logit", vcov={"CRV3": "cl"}, **GLM_KW)
    crv1 = pf.feglm(fml, DATA, family="logit", vcov={"CRV1": "cl"}, demeaner=TIGHT, **GLM_KW)
    np.testing.assert_allclose(ours._beta_hat, crv1._beta_hat, rtol=1e-8)

    def logit(fml, data, **kw):
        return pf.feglm(fml, data, family="logit", **kw)

    ref = crv3_reference(logit, fml, crv1._data, "cl", crv1._beta_hat, crv1._ssc[0], demeaner=TIGHT, **GLM_KW)
    np.testing.assert_allclose(ours._vcov, ref, rtol=1e-9)
    assert ours._vcov_type_detail == "CRV3" and ours._df_t == crv1._df_t


def test_crv3_iv_extension():
    fml = "y_iv ~ x1 | f3 | e1 ~ z1 + z2"
    ours = fereg.feols(fml, DATA, vcov={"CRV3": "cl"})
    crv1 = pf.feols(fml, DATA, vcov={"CRV1": "cl"}, demeaner=TIGHT)
    np.testing.assert_allclose(ours._beta_hat, crv1._beta_hat, rtol=1e-8)
    ref = crv3_reference(pf.feols, fml, crv1._data, "cl", crv1._beta_hat, crv1._ssc[0], demeaner=TIGHT)
    np.testing.assert_allclose(ours._vcov, ref, rtol=1e-8)
    np.testing.assert_allclose(fereg.feols(fml, DATA).vcov({"CRV3": "cl"})._vcov, ours._vcov, rtol=1e-12)


# --------------------------------------------------------------------------- Newey-West / Driscoll-Kraay


HAC_CASES = [
    ("y ~ x1 + x2", "NW", HAC, {}),
    ("y ~ x1 + x2", "NW", {**HAC, "lag": 2}, {}),
    ("y ~ x1 + x2 | unit", "NW", HAC, {}),
    ("y ~ x1 + x2 | unit + year", "NW", {**HAC, "lag": 3}, {"weights": "w"}),
    ("y ~ x1 + x2", "NW", {**HAC, "lag": 1}, {"weights": "fw", "weights_type": "fweights"}),
    ("y ~ x1 + x2", "DK", HAC, {}),
    ("y ~ x1 + x2 | unit", "DK", {**HAC, "lag": 2}, {"weights": "w"}),
    ("y ~ x1 + x2", "NW", {"time_id": "date", "panel_id": "unit"}, {}),
]


@pytest.mark.parametrize("fml,vcov,vkw,kw", HAC_CASES, ids=[f"{c[0]}|{c[1]}|{c[2]}|{c[3]}" for c in HAC_CASES])
def test_hac_ols(fml, vcov, vkw, kw):
    theirs = pf.feols(fml, PANEL, vcov=vcov, vcov_kwargs=vkw, demeaner=TIGHT, **kw)
    ours = fereg.feols(fml, PANEL, vcov=vcov, vcov_kwargs=vkw, **kw)
    compare(ours, theirs, rtol=1e-8)
    assert (ours._vcov_type, ours._vcov_type_detail) == (theirs._vcov_type, theirs._vcov_type_detail) == ("HAC", vcov)
    assert (ours._lag, ours._time_id, ours._panel_id) == (theirs._lag, theirs._time_id, theirs._panel_id)
    assert ours._vcov_lag == vkw.get("lag", int(np.floor(PANEL.year.nunique() ** 0.25)))
    # post hoc
    compare(fereg.feols(fml, PANEL, **kw).vcov(vcov, vcov_kwargs=vkw), theirs, rtol=1e-8)


def test_nw_time_series():
    theirs = pf.feols("y ~ x1 + x2", TS, vcov="NW", vcov_kwargs={"time_id": "year", "lag": 2})
    ours = fereg.feols("y ~ x1 + x2", TS, vcov="NW", vcov_kwargs={"time_id": "year", "lag": 2})
    compare(ours, theirs, rtol=1e-8)
    msg = "We have not yet implemented the default Newey-West HAC lag"
    with pytest.raises(ValueError, match=msg):
        fereg.feols("y ~ x1 + x2", TS, vcov="NW", vcov_kwargs={"time_id": "year"})
    with pytest.raises(ValueError, match=msg):
        fereg.feols("y ~ x1 + x2", TS).vcov("NW", vcov_kwargs={"time_id": "year"})
    with pytest.raises(ValueError, match="There are duplicate time periods in the data"):
        fereg.feols("y ~ x1", PANEL, vcov="NW", vcov_kwargs={"time_id": "year", "lag": 1})


@pytest.mark.parametrize("vcov", ["NW", "DK"])
def test_hac_glm(vcov):
    fml = "y_pois ~ x1 + x2 | unit"
    theirs = pf.fepois(fml, PANEL, vcov=vcov, vcov_kwargs=HAC, demeaner=TIGHT, **GLM_KW)
    ours = fereg.fepois(fml, PANEL, vcov=vcov, vcov_kwargs=HAC, **GLM_KW)
    compare(ours, theirs, rtol=1e-8, se_rtol=GLM_RTOL)
    fml = "y_bin ~ x1 + x2"
    theirs = pf.feglm(fml, PANEL, family="logit", vcov=vcov, vcov_kwargs=HAC, **GLM_KW)
    ours = fereg.feglm(fml, PANEL, family="logit", vcov=vcov, vcov_kwargs=HAC, **GLM_KW)
    compare(ours, theirs, rtol=1e-8, se_rtol=GLM_RTOL)


@pytest.mark.parametrize("vcov,vkw", [("NW", HAC), ("NW", {**HAC, "lag": 2}), ("DK", HAC)])
def test_hac_iv(vcov, vkw):
    """pyfixest's IV Newey-West works post hoc only (at estimation its first stage fails)."""
    fml = "y_iv ~ x1 | unit | e1 ~ z1 + z2"
    theirs = pf.feols(fml, PANEL, demeaner=TIGHT).vcov(vcov, vcov_kwargs=vkw)
    ours = fereg.feols(fml, PANEL, vcov=vcov, vcov_kwargs=vkw)
    compare(ours, theirs, rtol=1e-8)
    compare(fereg.feols(fml, PANEL).vcov(vcov, vcov_kwargs=vkw), theirs, rtol=1e-8)
    # the first stage is fitted with the same estimator
    fs = ours._model_1st_stage
    assert fs._vcov_type_detail == vcov
    ref = pf.feols("e1 ~ z1 + z2 + x1 | unit", PANEL, vcov=vcov, vcov_kwargs=vkw, demeaner=TIGHT)
    np.testing.assert_allclose(fs._se, ref._se, rtol=1e-8)


def test_hac_multi():
    fml = "y ~ csw(x1, x2) | unit"
    theirs = pf.feols(fml, PANEL, vcov="NW", vcov_kwargs=HAC, demeaner=TIGHT)
    ours = fereg.feols(fml, PANEL, vcov="NW", vcov_kwargs=HAC)
    for a, b in zip(ours.to_list(), theirs.to_list()):
        compare(a, b, rtol=1e-8)
    theirs.vcov("DK", vcov_kwargs={**HAC, "lag": 1})
    assert ours.vcov("DK", vcov_kwargs={**HAC, "lag": 1}) is ours
    for a, b in zip(ours.to_list(), theirs.to_list()):
        compare(a, b, rtol=1e-8)
        assert a._vcov_type_detail == "DK"
    ours.vcov({"CRV3": "cl"})
    for m in ours.to_list():
        ref = pf.feols(m._fml, PANEL, vcov={"CRV3": "cl"}, demeaner=TIGHT)
        compare(m, ref, rtol=1e-8)


def test_dk_without_panel_extension():
    """pyfixest needs panel_id for DK (only for its duplicate check); fereg does not."""
    fml = "y ~ x1 + x2"
    with_panel = fereg.feols(fml, PANEL, vcov="DK", vcov_kwargs=HAC)
    without = fereg.feols(fml, PANEL, vcov="DK", vcov_kwargs={"time_id": "year"})
    np.testing.assert_allclose(without._vcov, with_panel._vcov, rtol=1e-13)


def test_hac_input_errors():
    with pytest.raises(ValueError, match="Missing required 'time_id' for NW/DK vcov"):
        fereg.feols("y ~ x1", PANEL, vcov="NW", vcov_kwargs={"panel_id": "unit"})
    with pytest.raises(ValueError, match="Missing required 'time_id' for NW/DK vcov"):
        fereg.feols("y ~ x1", PANEL, vcov="DK")
    with pytest.raises(ValueError, match="The time variable must be numeric or date"):
        fereg.feols("y ~ x1", PANEL.assign(year=PANEL.year.astype(str)), vcov="NW", vcov_kwargs=HAC)
    with pytest.raises(ValueError, match="The variable 'period' is not in the data."):
        fereg.feols("y ~ x1", PANEL, vcov="NW", vcov_kwargs={"time_id": "period"})
    with pytest.raises(ValueError, match="integer values for 'lag'"):
        fereg.feols("y ~ x1", PANEL, vcov="NW", vcov_kwargs={**HAC, "lag": 1.5})
    with pytest.raises(ValueError, match="must be a dictionary with keys 'lag', 'time_id', or 'panel_id'"):
        fereg.feols("y ~ x1", PANEL, vcov="NW", vcov_kwargs={**HAC, "is_sorted": True})
    with pytest.raises(ValueError, match="duplicate time periods for the same panel id"):
        fereg.feols("y ~ x1", PANEL, vcov="NW", vcov_kwargs={"time_id": "year", "panel_id": "cl"})
    na = PANEL.copy()
    na.loc[5, "year"] = np.nan
    with pytest.raises(ValueError, match="missing values in the time variable year"):
        fereg.feols("y ~ x1", na, vcov="NW", vcov_kwargs=HAC)
    m = fereg.feols("y ~ x1", na)
    with pytest.raises(ValueError, match="year"):
        m.vcov("DK", vcov_kwargs=HAC)


# --------------------------------------------------------------------------- Conley (fixest)


CONLEY = pd.read_csv(DATA_DIR / "conley.csv")
CONLEY_REF = json.loads((DATA_DIR / "conley_fixest.json").read_text())
CONLEY_CASES = {
    "ols_100km": ("ols", "y ~ x1 + x2", {"cutoff": 100}),
    "ols_fe_60mi": ("ols", "y ~ x1 + x2 | g", {"cutoff": "60mi"}),
    "ols_fe_spherical": ("ols", "y ~ x1 + x2 | g", {"cutoff": 150, "distance": "spherical"}),
    "ols_pixel": ("ols", "y ~ x1 + x2", {"cutoff": "200km", "pixel": 25}),
    "ols_deduced": ("ols", "y ~ x1 + x2 | g", {}),
    "pois_fe": ("poisson", "yp ~ x1 + x2 | g", {"cutoff": 150}),
    "logit_fe": ("logit", "yb ~ x1 + x2 | g", {"cutoff": 150}),
}


def _conley_fit(model, fml, **kw):
    if model == "ols":
        return fereg.feols(fml, CONLEY, **kw)
    if model == "poisson":
        return fereg.fepois(fml, CONLEY, **GLM_KW, **kw)
    return fereg.feglm(fml, CONLEY, family="logit", **GLM_KW, **kw)


@pytest.mark.parametrize("case", list(CONLEY_CASES))
def test_conley_matches_fixest(case):
    model, fml, ckw = CONLEY_CASES[case]
    ref = CONLEY_REF[case]
    vkw = {"lat": "lat", "lon": "lon", **ckw}
    ours = _conley_fit(model, fml, vcov="conley", vcov_kwargs=vkw)
    rtol = 1e-8 if model == "ols" else GLM_RTOL
    np.testing.assert_allclose(ours._beta_hat, ref["coef"], rtol=1e-7)
    np.testing.assert_allclose(ours._se, ref["se"], rtol=rtol)
    np.testing.assert_allclose(ours._vcov_cutoff, ref["cutoff_km"], rtol=1e-12)
    assert (ours._vcov_type, ours._vcov_type_detail) == ("conley", "conley")
    if model == "ols":
        assert ours._df_t == ours._N - ours._df_k
    post = _conley_fit(model, fml).vcov("conley", vcov_kwargs=vkw)
    np.testing.assert_allclose(post._se, ours._se, rtol=1e-12)


def test_conley_input_errors():
    with pytest.raises(ValueError, match="'lat' and 'lon'|latitude and longitude"):
        fereg.feols("y ~ x1", CONLEY, vcov="conley", vcov_kwargs={"lat": "lat"})
    with pytest.raises(ValueError, match="'cutoff'"):
        fereg.feols("y ~ x1", CONLEY, vcov="conley", vcov_kwargs={"lat": "lat", "lon": "lon", "cutoff": "10.5km"})
    with pytest.raises(ValueError, match="distance"):
        fereg.feols("y ~ x1", CONLEY, vcov="conley", vcov_kwargs={"lat": "lat", "lon": "lon", "distance": "euclid"})
    na = CONLEY.copy()
    na.loc[3, "lat"] = np.nan
    with pytest.raises(ValueError, match="latitude"):
        fereg.feols("y ~ x1", na, vcov="conley", vcov_kwargs={"lat": "lat", "lon": "lon", "cutoff": 100})


# --------------------------------------------------------------------------- display


@pytest.mark.parametrize(
    "vcov,vkw,data",
    [("NW", HAC, PANEL), ("DK", {**HAC, "lag": 2}, PANEL), ("HC3", None, DATA), ({"CRV3": "f3"}, None, DATA)],
    ids=["NW", "DK", "HC3", "CRV3"],
)
def test_summary_inference_line(vcov, vkw, data, capsys):
    fml = "y ~ x1 + x2"
    theirs = pf.feols(fml, data, vcov=vcov, vcov_kwargs=vkw)
    ours = fereg.feols(fml, data, vcov=vcov, vcov_kwargs=vkw)
    assert_same_summary(ours, theirs, capsys)


def test_etable_se_type():
    m1 = fereg.feols("y ~ x1", PANEL, vcov="NW", vcov_kwargs=HAC)
    m2 = fereg.feols("y ~ x1", PANEL, vcov={"CRV3": "unit"})
    m3 = fereg.feols("y ~ x1", CONLEY, vcov="conley", vcov_kwargs={"lat": "lat", "lon": "lon", "cutoff": 100})
    t = fereg.etable([m1, m2, m3])
    assert t.loc[("stats", "S.E. type")].tolist() == ["NW", "by: unit", "conley"]
