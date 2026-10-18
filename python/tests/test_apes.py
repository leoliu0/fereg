"""Average partial effects (``Felogit.get_apes``) against alpaca 0.3.5 ``getAPEs``.

Reference numbers: ``data/bias_alpaca.json`` (``data/gen_bias_fixtures.R``). With a finite
population and groups dropped for perfect classification (data set ``drops``) the expected
sampling-of-effects term sums over all N* observations (fereg design 6.3), not over the retained
ones as alpaca does; alpaca's own number is in the fixture for reference.
"""

from __future__ import annotations

import functools
import io
import re
import warnings
from contextlib import redirect_stdout

import numpy as np
import pytest
from _bias import RTOL, RTOL_POINT, ape_id, assert_close, data, fit_case, model_case, models

import fereg

UNCORRECTED = [(c, e) for c in models() for e in c["apes"]]
CORRECTED = [(c, r, e) for c in models() for r in c["corrections"] for e in r["apes"]]


@functools.cache
def corrected(cid: str, L: int, bias: str) -> fereg.FelogitBiasCorr:
    c = model_case(cid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_case(cid).bias_corr(L=L, panel_structure=c["structure"], bias=bias)


def _apes(m, **kw) -> fereg.APEs:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.get_apes(**kw)


@pytest.mark.parametrize(("c", "e"), UNCORRECTED, ids=[f"{c['id']}-{ape_id(e)}" for c, e in UNCORRECTED])
def test_uncorrected_apes(c, e):
    a = _apes(
        fit_case(c["id"]),
        n_pop=e["n_pop"],
        panel_structure=c["structure"],
        sampling_fe=e["sampling_fe"],
        weak_exo=e["weak_exo"],
    )
    assert isinstance(a, fereg.APEs)
    assert list(a.coef().index) == c["coef_names"]
    assert_close(a.coef(), e["delta"], RTOL, "delta")
    assert_close(a.vcov.to_numpy(), e["vcov"], RTOL, "vcov")
    assert_close(a.se(), np.sqrt(np.diag(e["vcov"])), RTOL, "se")
    assert a.bias_term is None
    assert a.N_full == c["n_full"]
    assert a.panel_structure == c["structure"]
    assert a.sampling_fe == e["sampling_fe"]
    assert a.weak_exo == e["weak_exo"]
    expected_adj = 0.0 if e["n_pop"] is None else (e["n_pop"] - c["n_full"]) / (e["n_pop"] - 1)
    assert a.adj == pytest.approx(expected_adj, rel=1e-15)
    assert list(a.binary[a.binary].index) == [v for v in c["coef_names"] if v in ("d", "rta")]


@pytest.mark.parametrize(
    ("c", "r", "e"), CORRECTED, ids=[f"{c['id']}-L{r['L']}-{r['bias']}-{ape_id(e)}" for c, r, e in CORRECTED]
)
def test_corrected_apes(c, r, e):
    res = corrected(c["id"], r["L"], r["bias"])
    a = _apes(res, n_pop=e["n_pop"], sampling_fe=e["sampling_fe"])
    assert a.weak_exo == e["weak_exo"] == (r["L"] > 0)
    # the L that entered (a two-way network model has no pair effect: none)
    assert a.bandwidth == (0 if c["structure"] == "network" and len(c["fe"]) == 2 else r["L"])
    assert a.panel_structure == c["structure"]
    assert_close(a.coef(), e["delta"], RTOL_POINT, "corrected delta")
    assert_close(a.bias_term, e["bias_term"], RTOL_POINT, "APE bias term")
    assert_close(a.vcov.to_numpy(), e["vcov"], RTOL_POINT, "vcov")
    if "alpaca" in r:
        k = r["apes"].index(e)
        ea = r["alpaca"]["apes"][k]
        assert_close(a.coef(), ea["delta"], RTOL_POINT, "alpaca corrected delta")
        assert_close(a.bias_term, ea["bias_term"], RTOL_POINT, "alpaca bias.term")
        if c["n_full"] == c["n"] or e["n_pop"] is None or e["sampling_fe"] == "unrestricted" and not e["weak_exo"]:
            assert_close(a.vcov.to_numpy(), ea["vcov"], RTOL_POINT, "alpaca vcov")


def test_sampling_term_over_all_rows():
    # drops: the expected term (all N* rows) differs from alpaca's (retained rows)
    c = model_case("drops")
    e = next(e for e in c["apes"] if e["n_pop"] is not None and e["sampling_fe"] == "independence" and not e["weak_exo"])
    assert not np.allclose(e["vcov"], e["alpaca"]["vcov"], rtol=1e-3)
    a = fit_case("drops").get_apes(n_pop=e["n_pop"])
    assert_close(a.vcov.to_numpy(), e["vcov"], RTOL, "all-rows sampling term")


@pytest.mark.parametrize("corrected_point", [False, True])
def test_weak_exo_covariance_over_all_rows(corrected_point):
    # drops: kept individuals lost their (all-one) period-8 row; those dropped rows enter C with
    # D = -delta/N* before the later retained rows of the individual (fereg-core API 2.2), which
    # alpaca's C (retained rows only) leaves out. "unrestricted" isolates the C term.
    c = model_case("drops")
    if corrected_point:
        r = next(r for r in c["corrections"] if r["L"] == 1 and r["bias"] == "both")
        e = next(e for e in r["apes"] if e["n_pop"] is not None and e["sampling_fe"] == "unrestricted")
        alpaca = r["alpaca"]["apes"][r["apes"].index(e)]["vcov"]
        a = corrected("drops", 1, "both").get_apes(n_pop=e["n_pop"], sampling_fe="unrestricted")
        rtol = RTOL_POINT
    else:
        e = next(
            e for e in c["apes"] if e["n_pop"] is not None and e["sampling_fe"] == "unrestricted" and e["weak_exo"]
        )
        alpaca = e["alpaca"]["vcov"]
        a = fit_case("drops").get_apes(n_pop=e["n_pop"], sampling_fe="unrestricted", weak_exo=True)
        rtol = RTOL
    assert a.weak_exo
    assert not np.allclose(e["vcov"], alpaca, rtol=1e-6, atol=0)
    assert_close(a.vcov.to_numpy(), e["vcov"], rtol, "all-rows weak-exogeneity covariance")

def test_small_population_warns():
    c = model_case("classic")
    m = fit_case("classic")
    with pytest.warns(UserWarning, match="Size of the entire population is lower than the full sample size"):
        a = m.get_apes(n_pop=c["n_full"] - 1)
    assert a.adj == 0.0
    np.testing.assert_array_equal(a.vcov.to_numpy(), m.get_apes().vcov.to_numpy())


def test_summary_and_tidy():
    c = model_case("classic")
    m = fit_case("classic")
    a = m.get_apes(n_pop=2 * c["n_full"], weak_exo=True)
    tidy = a.tidy()
    assert list(tidy.columns) == list(m.tidy().columns)
    assert_close(a.tstat(), a.coef() / a.se(), 1e-14, "tstat")
    assert a.confint().shape == (len(c["coef_names"]), 2)
    buf = io.StringIO()
    with redirect_stdout(buf):
        a.summary()
        corrected("classic", 1, "both").get_apes().summary()
    text = buf.getvalue()
    assert "Average partial effects (Logit)" in text
    assert "delta method + sampling of effects + weak exogeneity (n_pop = 800" in text
    assert "Bias correction: analytical (L = 1, bias = 'both')" in text
    assert f"Observations:  {c['n_full']}" in text


APE_ERRORS = [
    (dict(n_pop=0), ValueError, "n_pop must be a finite number >= 1"),
    (dict(n_pop=float("nan")), ValueError, "n_pop must be a finite number >= 1"),
    (dict(n_pop="1000"), TypeError, "n_pop must be a number or None"),
    (dict(n_pop=True), TypeError, "n_pop must be a number or None"),
    (dict(sampling_fe="dependent"), ValueError, "sampling_fe must be one of"),
    (dict(panel_structure="grid"), ValueError, "panel_structure must be one of"),
    (dict(weak_exo="yes"), TypeError, "weak_exo must be a bool or None"),
]


@pytest.mark.parametrize(("kw", "exc", "match"), APE_ERRORS, ids=[str(e[0]) for e in APE_ERRORS])
def test_rejected_arguments(kw, exc, match):
    with pytest.raises(exc, match=match):
        fit_case("classic").get_apes(**kw)


def test_rejected_structures():
    with pytest.raises(ValueError, match=re.escape('panel structure "network" expects a two- or three-way')):
        fit_case("oneway").get_apes(panel_structure="network")
    with pytest.raises(ValueError, match=re.escape('panel structure "classic" expects a one- or two-way')):
        fit_case("network3").get_apes()
    with pytest.raises(ValueError, match="^bias correction and APEs need fixed effects$"):
        fereg.feglm("y ~ x1 + x2 + d", data("classic"), family="logit").get_apes()
    res = corrected("classic", 1, "both")
    with pytest.raises(ValueError, match="panel_structure is set by the bias correction"):
        res.get_apes(panel_structure="network")
    with pytest.raises(ValueError, match="weak_exo is set by the analytical bias correction"):
        res.get_apes(weak_exo=False)
    # same values are accepted
    a = res.get_apes(panel_structure="classic", weak_exo=True)
    np.testing.assert_array_equal(a.coef().to_numpy(), res.get_apes().coef().to_numpy())


def test_weak_exo_on_jackknife_result():
    c = model_case("classic")
    res = fit_case("classic").bias_corr(method="jackknife")
    a = res.get_apes(n_pop=2 * c["n_full"], weak_exo=True)
    b = res.get_apes(n_pop=2 * c["n_full"])
    assert a.weak_exo and not b.weak_exo
    np.testing.assert_array_equal(a.coef().to_numpy(), b.coef().to_numpy())
    assert not np.array_equal(a.vcov.to_numpy(), b.vcov.to_numpy())
