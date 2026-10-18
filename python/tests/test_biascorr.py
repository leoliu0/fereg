"""Logit bias corrections (``Felogit.bias_corr``) against alpaca 0.3.5 and jackknife refits.

Reference numbers: ``data/bias_alpaca.json`` (``data/gen_bias_fixtures.R``). pyfixest has no bias
correction, so there is no live comparison.
"""

from __future__ import annotations

import functools
import io
import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest
from _bias import (
    RTOL,
    RTOL_POINT,
    RTOL_UNCORR_VCOV,
    assert_close,
    case_id,
    correction_id,
    data,
    fit,
    fit_case,
    jackknife_cases,
    model_case,
    models,
)

import fereg

ANALYTICAL = [(c, r) for c in models() for r in c["corrections"]]


@functools.cache
def corrected(cid: str, L: int, bias: str) -> fereg.FelogitBiasCorr:
    c = model_case(cid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_case(cid).bias_corr(L=L, panel_structure=c["structure"], bias=bias)


@functools.cache
def fit_jk(fml: str, name: str) -> fereg.Felogit:
    return fit(fml, name)


# ------------------------------------------------------------------------------------------------
# Uncorrected fits (the starting point of every correction)
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("c", models(), ids=case_id)
def test_uncorrected_fit_matches_alpaca(c):
    m = fit(c["fml"], c["data"])
    assert list(m.coef().index) == c["coef_names"]
    assert_close(m.coef(), c["coef"], RTOL, "coef")
    assert_close(m._vcov, c["vcov"]["iid"], RTOL_UNCORR_VCOV, "vcov iid")
    assert_close(m.vcov("hetero")._vcov, c["vcov"]["hetero"], RTOL_UNCORR_VCOV, "vcov hetero")
    assert_close(m.vcov({"CRV1": c["cluster"]})._vcov, c["vcov"]["cluster"], RTOL_UNCORR_VCOV, "vcov cluster")


# ------------------------------------------------------------------------------------------------
# Analytical correction
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("c", "r"), ANALYTICAL, ids=[correction_id(c, r) for c, r in ANALYTICAL])
def test_analytical_matches_alpaca(c, r):
    m = fit_case(c["id"])
    res = corrected(c["id"], r["L"], r["bias"])
    assert isinstance(res, fereg.FelogitBiasCorr)
    assert isinstance(res, fereg.Felogit)
    assert list(res.coef().index) == c["coef_names"]
    # before the fixed-effect re-solve
    assert_close(res.coef(), r["coef"], RTOL, "corrected coef")
    assert_close(res.bias_term, r["bias_term"], RTOL, "bias_term")
    assert_close(res.coef_uncorr, c["coef"], RTOL, "coef_uncorr")
    assert_close(res.coef_uncorr - res.coef(), res.bias_term, 1e-12, "bias_term = uncorrected - corrected")
    if "alpaca" in r:
        assert_close(res.coef(), r["alpaca"]["coef"], RTOL, "alpaca biasCorr coef")
        assert_close(res.bias_term, r["alpaca"]["bias_term"], RTOL, "alpaca bias.term")
    # at the corrected point (exact fixed-effect re-solve)
    assert_close(res._vcov, r["vcov"]["iid"], RTOL_POINT, "vcov iid")
    assert_close(res.se(), np.sqrt(np.diag(r["vcov"]["iid"])), RTOL_POINT, "se")
    assert res.deviance == pytest.approx(r["deviance"], rel=RTOL_POINT)
    assert res.deviance_uncorr == pytest.approx(m.deviance, rel=1e-12)
    if "alpaca" in r:
        assert_close(res._vcov, r["alpaca"]["vcov"]["iid"], RTOL_POINT, "alpaca vcov hessian")
    # attributes
    assert res.panel_structure == c["structure"]
    assert res.bandwidth == r["L"]
    assert res.method == "analytical"
    assert res.jackknife is None
    assert res.bias == r["bias"]
    assert res.N_full == c["n_full"]
    # same reported sample as the uncorrected fit (pyfixest keeps rows at the bound)
    assert res._N == m._N
    np.testing.assert_array_equal(res._obs, m._obs)


@pytest.mark.parametrize(("c", "r"), ANALYTICAL[:4], ids=[correction_id(c, r) for c, r in ANALYTICAL[:4]])
def test_post_hoc_vcov_at_corrected_point(c, r):
    res = fit_case(c["id"]).bias_corr(L=r["L"], panel_structure=c["structure"], bias=r["bias"])
    assert_close(res.vcov("hetero")._vcov, r["vcov"]["hetero"], RTOL_POINT, "hetero")
    assert_close(res.vcov({"CRV1": c["cluster"]})._vcov, r["vcov"]["cluster"], RTOL_POINT, "CRV1")
    assert_close(res.vcov("iid")._vcov, r["vcov"]["iid"], RTOL_POINT, "iid")
    with pytest.raises(ValueError, match="CRV3 is not available after bias correction"):
        res.vcov({"CRV3": c["cluster"]})


def test_estimation_vcov_carries_over():
    c = model_case("classic")
    r = next(r for r in c["corrections"] if r["L"] == 1 and r["bias"] == "both")
    m = fit(c["fml"], c["data"], vcov={"CRV1": "i"})
    res = m.bias_corr(L=1)
    assert res._vcov_type_detail == "CRV1"
    assert_close(res._vcov, r["vcov"]["cluster"], RTOL_POINT, "CRV1 at estimation")


def test_drops_sample_semantics():
    # pyfixest drops the all-zero individuals once and keeps the other constant groups' rows at
    # the bound; the correction uses alpaca's n and N*
    c = model_case("drops")
    m = fit_case("drops")
    assert c["n"] < m._N < c["n_full"]
    res = corrected("drops", 0, "both")
    assert res.N_full == c["n_full"]
    assert res._N == m._N
    mu = res.predict(type="response")
    bound = (mu < 1e-12) | (mu > 1 - 1e-12)
    assert bound.sum() == m._N - c["n"]


def test_network_two_way_ignores_L():
    m = fit_case("network2")
    with pytest.warns(UserWarning, match="L ignored"):
        res = m.bias_corr(L=2, panel_structure="network")
    r0 = corrected("network2", 0, "both")
    np.testing.assert_array_equal(res.coef().to_numpy(), r0.coef().to_numpy())


def test_methods_on_corrected_result():
    m = fit_case("classic")
    res = corrected("classic", 1, "both")
    tidy = res.tidy()
    assert list(tidy.columns) == list(m.tidy().columns)
    assert_close(tidy["Estimate"], res.coef(), 0, "tidy")
    assert_close(res.tstat(), res.coef() / res.se(), 1e-14, "tstat")
    assert (res.pvalue() >= 0).all()
    ci = res.confint()
    assert (ci.iloc[:, 0] < res.coef()).all() and (ci.iloc[:, 1] > res.coef()).all()
    buf = io.StringIO()
    with redirect_stdout(buf):
        res.summary()
        fereg.summary([m, res])
    text = buf.getvalue()
    assert "Bias correction: analytical (L = 1, bias = 'both')" in text
    tab = fereg.etable([m, res])
    assert isinstance(tab, pd.DataFrame) and tab.shape[1] == 2
    # fixed effects re-estimated at the corrected coefficients: predictions rebuild eta
    eta = res.predict()
    assert_close(res.predict(newdata=data("classic"))[res._obs], eta, 1e-8, "predict(newdata)")
    assert res.fixef()
    assert "analytical" in repr(res)


def test_lean_fit():
    c = model_case("classic")
    r = next(r for r in c["corrections"] if r["L"] == 0 and r["bias"] == "both")
    res = fit(c["fml"], c["data"], lean=True).bias_corr()
    assert_close(res.coef(), r["coef"], RTOL, "coef")
    with pytest.raises(ValueError, match="lean=True"):
        res.vcov("hetero")


# ------------------------------------------------------------------------------------------------
# Jackknife
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("j", jackknife_cases(), ids=case_id)
def test_jackknife_matches_refits(j):
    m = fit_jk(j["fml"], j["data"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = m.bias_corr(method="jackknife", jackknife=j["variant"], bias=j["bias"], time_id=j["time_id"])
    assert_close(res.coef_uncorr, j["coef_uncorr"], RTOL, "coef_uncorr")
    assert_close(res.coef(), j["coef"], RTOL_POINT, "jackknife coef")
    assert res.method == "jackknife"
    assert res.jackknife == j["variant"]
    assert res.bandwidth == 0
    assert res.N_full == j["n_full"]
    assert_close(res._vcov, j["vcov"]["iid"], RTOL_POINT, "vcov at the jackknife point")
    a = res.get_apes()
    assert a.bias_term is None
    assert_close(a.coef(), j["ape"], RTOL_POINT, "jackknife APEs")
    assert_close(a.vcov.to_numpy(), j["ape_vcov"], RTOL_POINT, "APE vcov at the jackknife point")


@pytest.mark.parametrize("variant", ["ss2", "jj"])
def test_jackknife_oneway_both_is_individual(variant):
    j = next(j for j in jackknife_cases() if j["id"] == f"oneway_{variant}")
    m = fit_jk(j["fml"], j["data"])
    res = m.bias_corr(method="jackknife", jackknife=variant, time_id="t")
    assert_close(res.coef(), j["coef"], RTOL_POINT, "one-way jackknife, bias='both'")


def test_double_jackknife_string_ids():
    j = next(j for j in jackknife_cases() if j["id"] == "dyad_double_both")
    num = fit_jk(j["fml"], "dyad").bias_corr(method="jackknife", jackknife="double")
    s = fit("y ~ x1 + x2 | exps + imps", "dyad").bias_corr(method="jackknife", jackknife="double")
    assert_close(s.coef(), num.coef(), 1e-12, "double with string ids")
    assert_close(s.coef(), j["coef"], RTOL_POINT, "double")
    with pytest.raises(ValueError, match="one id space"):
        fit("y ~ x1 + x2 | exp + imps", "dyad").bias_corr(method="jackknife", jackknife="double")


def test_multiple_reproducible_under_seed():
    m = fit_case("classic")
    kw = dict(method="jackknife", multiple=3)
    r1 = m.bias_corr(**kw, seed=11)
    r2 = m.bias_corr(**kw, seed=11)
    r3 = m.bias_corr(**kw, seed=12)
    np.testing.assert_array_equal(r1.coef().to_numpy(), r2.coef().to_numpy())
    np.testing.assert_array_equal(r1.get_apes().coef().to_numpy(), r2.get_apes().coef().to_numpy())
    assert not np.array_equal(r1.coef().to_numpy(), r3.coef().to_numpy())
    assert (r1.multiple, r1.split, r1.seed) == (3, "both", 11)
    assert "3 random partitions (split = 'both', seed = 11)" in r1._correction_line
    for split in ("individuals", "time"):
        a = m.bias_corr(method="jackknife", jackknife="ss1", multiple=2, split=split, seed=5)
        b = m.bias_corr(method="jackknife", jackknife="ss1", multiple=2, split=split, seed=5)
        np.testing.assert_array_equal(a.coef().to_numpy(), b.coef().to_numpy())


def test_multiple_without_permuted_dimension_is_an_error():
    # ss1 with one bias splits one dimension only: permuting the other one is pointless
    m = fit_case("classic")
    with pytest.raises(ValueError):
        m.bias_corr(method="jackknife", jackknife="ss1", bias="individual", multiple=3, split="individuals", seed=1)
    with pytest.raises(ValueError):
        m.bias_corr(method="jackknife", jackknife="ss1", bias="time", multiple=3, split="time", seed=1)


# ------------------------------------------------------------------------------------------------
# Determinism
# ------------------------------------------------------------------------------------------------


def test_bit_identical_across_threads():
    c = model_case("unbal")
    out = []
    for nthreads in (1, 16):
        m = fit(c["fml"], c["data"], nthreads=nthreads)
        an = m.bias_corr(L=2)
        jk = m.bias_corr(method="jackknife", jackknife="jj")
        mul = m.bias_corr(method="jackknife", multiple=2, seed=3)
        a = an.get_apes(n_pop=2 * c["n_full"])
        b = m.get_apes(n_pop=2 * c["n_full"], weak_exo=True)
        out.append(
            [an.coef(), an._vcov, an.predict(), jk.coef(), jk._vcov, mul.coef(), a.coef(), a.vcov, b.coef(), b.vcov]
        )
    for x, y in zip(*out):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


# ------------------------------------------------------------------------------------------------
# Rejected combinations
# ------------------------------------------------------------------------------------------------

ERRORS = [
    ("classic", dict(L=-1), ValueError, "L must be a non-negative integer"),
    ("classic", dict(L=1.5), TypeError, "L must be a non-negative integer"),
    ("classic", dict(L=True), TypeError, "L must be a non-negative integer"),
    ("classic", dict(panel_structure="grid"), ValueError, "panel_structure must be one of"),
    ("classic", dict(method="bootstrap"), ValueError, "method must be one of"),
    ("classic", dict(method="jackknife", jackknife="ss3"), ValueError, "jackknife must be one of"),
    ("classic", dict(bias="all"), ValueError, "bias must be one of"),
    ("classic", dict(method="jackknife", multiple=2, split="pairs"), ValueError, "split must be one of"),
    ("classic", dict(jackknife="ss1"), ValueError, "jackknife is only used with method='jackknife'"),
    ("classic", dict(multiple=2), ValueError, "multiple is only used with method='jackknife'"),
    ("classic", dict(seed=1), ValueError, "seed is only used with method='jackknife'"),
    ("classic", dict(split="time"), ValueError, "split is only used with method='jackknife'"),
    ("classic", dict(time_id="t"), ValueError, "time_id is only used with method='jackknife'"),
    ("classic", dict(method="jackknife", L=1), ValueError, "L is only used with method='analytical'"),
    ("classic", dict(method="jackknife", jackknife="js", multiple=2), ValueError, "multiple is only available"),
    ("classic", dict(method="jackknife", split="time"), ValueError, "split is only used with multiple > 0"),
    ("classic", dict(method="jackknife", seed=3), ValueError, "seed is only used with multiple > 0"),
    ("classic", dict(method="jackknife", multiple=-1), ValueError, "multiple must be a non-negative integer"),
    ("classic", dict(method="jackknife", multiple=2, seed=-1), ValueError, "seed must be a non-negative integer"),
    ("classic", dict(method="jackknife", multiple=2, seed="a"), TypeError, "seed must be a non-negative integer"),
    ("classic", dict(method="jackknife", time_id="t"), ValueError, "time_id is only used by the jackknife of a one-way"),
    ("classic", dict(panel_structure="network", bias="time"), ValueError, "bias must be 'both' with panel_structure='network'"),
    ("network3", dict(panel_structure="network", method="jackknife"), ValueError, "panel_structure='classic' only"),
    ("network3", dict(), ValueError, "expects a one- or two-way fixed effects model"),
    ("oneway", dict(panel_structure="network"), ValueError, "expects a two- or three-way fixed effects model"),
    ("oneway", dict(bias="time"), ValueError, "bias='time' needs a two-way model"),
    ("oneway", dict(method="jackknife"), ValueError, "needs the time variable: pass time_id"),
    ("oneway", dict(method="jackknife", jackknife="double", time_id="t"), ValueError, "needs a two-way model"),
    ("oneway", dict(method="jackknife", time_id=3), TypeError, "time_id must be the name of a column"),
    ("classic", dict(L=1, bias="time"), ValueError, "bias='time' takes L = 0"),
    ("oneway", dict(method="jackknife", time_id="y_missing"), KeyError, "y_missing"),
]


@pytest.mark.parametrize(("cid", "kw", "exc", "match"), ERRORS, ids=[f"{e[0]}-{i}" for i, e in enumerate(ERRORS)])
def test_rejected_combinations(cid, kw, exc, match):
    with pytest.raises(exc, match=match):
        fit_case(cid).bias_corr(**kw)


def test_rejected_models():
    with pytest.raises(ValueError, match="need a model with fixed effects"):
        fit("y ~ x1 + x2 + d", "classic").bias_corr()
    with pytest.raises(ValueError, match="without varying slopes"):
        fit("y ~ x1 + d | i[x2] + t", "classic").bias_corr()
    with pytest.raises(ValueError, match="CRV3 is not available after bias correction"):
        fit("y ~ x1 + x2 + d | i + t", "classic", vcov={"CRV3": "i"}).bias_corr()
    with pytest.raises(ValueError, match="cannot be applied to a bias-corrected model"):
        corrected("classic", 0, "both").bias_corr()
