"""Logit bias corrections (``Felogit.bias_corr``) against alpaca 0.3.5 and jackknife refits.

Reference numbers: ``data/bias_alpaca.json`` (``data/gen_bias_fixtures.R``). pyfixest has no bias
correction, so there is no live comparison.
"""

from __future__ import annotations

import functools
import io
import math
import re
import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest
from _bias import (
    FIT_KW,
    RTOL,
    RTOL_POINT,
    RTOL_UNCORR_VCOV,
    SSC,
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
    m = fit_case(c["id"])
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
    # attributes: bias is the one corrected (a one-way model has the individual bias only);
    # bandwidth is the requested L, as alpaca (a two-way network model ignores it)
    assert res.panel_structure == c["structure"]
    assert res.bandwidth == r["L"]
    assert res.method == "analytical"
    assert res.jackknife is None
    assert res.bias == ("individual" if len(c["fe"]) == 1 else r["bias"])
    assert res.N_full == c["n_full"]
    # same reported sample as the uncorrected fit (pyfixest keeps rows at the bound)
    assert res._N == m._N
    np.testing.assert_array_equal(res._obs, m._obs)


POST_HOC = ANALYTICAL[:4] + [next((c, r) for c, r in ANALYTICAL if c["id"] == k) for k in ("weights", "offset")]


@pytest.mark.parametrize(("c", "r"), POST_HOC, ids=[correction_id(c, r) for c, r in POST_HOC])
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
    # the corrected model is lean iff the uncorrected one is (the core refits from the design)
    c = model_case("classic")
    r = next(r for r in c["corrections"] if r["L"] == 0 and r["bias"] == "both")
    m = fit(c["fml"], c["data"], lean=True)
    res = m.bias_corr()
    assert_close(res.coef(), r["coef"], RTOL, "coef")
    assert_close(res._vcov, r["vcov"]["iid"], RTOL_POINT, "vcov at estimation")
    assert res._out.state is None and res._out.eta is None
    with pytest.raises(ValueError, match="lean=True"):
        res.vcov("hetero")
    with pytest.raises(ValueError, match="lean=True"):
        res.predict()
    e = next(e for e in r["apes"] if e["n_pop"] is not None and e["sampling_fe"] == "independence")
    assert_close(res.get_apes(n_pop=e["n_pop"]).vcov.to_numpy(), e["vcov"], RTOL_POINT, "APEs of a lean model")
    full = corrected("classic", 0, "both")
    assert full._out.state is not None and full._out.eta is not None
    jk = m.bias_corr(method="jackknife")
    assert jk._out.state is None


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
    with pytest.raises(ValueError, match=exact(CORE["double_kind"])):
        fit("y ~ x1 + x2 | exp + imps", "dyad").bias_corr(method="jackknife", jackknife="double")


def test_subpanel_failure_names_the_subpanel():
    # x3 is zero in the first time half: the ss1 time-half subpanel loses its coefficient
    d = data("classic").assign(x3=lambda df: df["d"] * (df["t"] > 4))
    m = fereg.feglm("y ~ x1 + x2 + x3 | i + t", d, family="logit", ssc=SSC, **FIT_KW)
    with pytest.raises(ValueError, match=exact(CORE["subpanel_lost"])):
        m.bias_corr(method="jackknife", jackknife="ss1", bias="individual")


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
    # ss1 with one bias splits one dimension only: permuting the other one has no effect
    m = fit_case("classic")
    with pytest.raises(ValueError, match=exact(CORE["multiple_time"])):
        m.bias_corr(method="jackknife", jackknife="ss1", bias="individual", multiple=3, split="individuals", seed=1)
    with pytest.raises(ValueError, match=exact(CORE["multiple_individual"])):
        m.bias_corr(method="jackknife", jackknife="ss1", bias="time", multiple=3, split="time", seed=1)


# Independent jackknife references: subpanels built here (fereg design 7) and refitted with plain
# fereg.feglm calls on the subsets of the data.


def _halves(idx: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """1-based indices of m units: first half <= ceil(m/2); second > m/2 (m even) or >= ceil(m/2)."""
    first = idx <= math.ceil(m / 2)
    second = idx > m / 2 if m % 2 == 0 else idx >= math.ceil(m / 2)
    return first, second


def _position_halves(ind: np.ndarray, key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-individual time halves: rows of each individual ordered by ``key`` (ties: row order)."""
    df = pd.DataFrame({"ind": ind, "key": key, "row": np.arange(len(ind))}).sort_values(["ind", "key", "row"])
    pos = np.empty(len(ind), dtype=np.int64)
    pos[df["row"].to_numpy()] = df.groupby("ind").cumcount().to_numpy() + 1
    t_i = np.bincount(ind)[ind]
    first = pos <= np.ceil(t_i / 2)
    second = np.where(t_i % 2 == 0, pos > t_i / 2, pos >= np.ceil(t_i / 2))
    return first, second


def _subfit(d: pd.DataFrame, keep: np.ndarray, fml: str, **kw) -> np.ndarray:
    """Coefficients and APEs (divided by the subpanel's own N*) of a plain fit on ``d[keep]``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = fereg.feglm(fml, d.loc[keep].reset_index(drop=True), family="logit", ssc=SSC, **FIT_KW, **kw)
        return np.concatenate([m.coef().to_numpy(), m.get_apes().coef().to_numpy()])


def _mean(fits: list[np.ndarray]) -> np.ndarray:
    return np.mean(fits, axis=0)


def _check_combination(res: fereg.FelogitBiasCorr, combined: np.ndarray, what: str) -> None:
    p = len(res.coef())
    assert_close(res.coef(), combined[:p], RTOL, f"{what}: coefficients")
    assert_close(res.get_apes().coef(), combined[p:], RTOL, f"{what}: APEs")


def test_jackknife_with_offset_matches_plain_refits():
    d = data("classic")
    fml, kw = "y ~ x1 + x2 + d | i + t", {"offset": "off"}
    ind, per = d["i"].to_numpy() - 1, d["t"].to_numpy() - 1
    n, t = ind.max() + 1, per.max() + 1
    full = _subfit(d, np.ones(len(d), dtype=bool), fml, **kw)
    m = fit(fml, "classic", **kw)
    ih, ph = _halves(ind + 1, n), _position_halves(ind, per)
    ss2 = 3 * full - _mean([_subfit(d, h, fml, **kw) for h in ih]) - _mean([_subfit(d, h, fml, **kw) for h in ph])
    _check_combination(m.bias_corr(method="jackknife", jackknife="ss2"), ss2, "ss2 with offset")
    del_i = _mean([_subfit(d, ind != i, fml, **kw) for i in range(n)])
    del_t = _mean([_subfit(d, per != s, fml, **kw) for s in range(t)])
    jj = (n + t - 1) * full - (n - 1) * del_i - (t - 1) * del_t
    _check_combination(m.bias_corr(method="jackknife", jackknife="jj"), jj, "jj with offset")


@pytest.mark.parametrize(("variant", "split", "seed"), [("ss2", "both", 7), ("ss1", "time", 3), ("ss2", "individuals", 5)])
def test_multiple_matches_plain_refits(variant, split, seed):
    # the partitions rebuilt from the same seed: numpy.random.default_rng(seed), per partition a
    # permutation of the individuals, then one of the periods (as split asks)
    d = data("classic")
    fml = "y ~ x1 + x2 + d | i + t"
    ind, per = d["i"].to_numpy() - 1, d["t"].to_numpy() - 1
    n, t = ind.max() + 1, per.max() + 1
    full = _subfit(d, np.ones(len(d), dtype=bool), fml)
    rng = np.random.default_rng(seed)
    per_partition = []
    for _ in range(2):
        ip = rng.permutation(n) if split in ("both", "individuals") else np.arange(n)
        tp = rng.permutation(t) if split in ("both", "time") else np.arange(t)
        ih = _halves(ip[ind] + 1, n)
        if variant == "ss2":
            ph = _position_halves(ind, tp[per])
            b = 3 * full - _mean([_subfit(d, h, fml) for h in ih]) - _mean([_subfit(d, h, fml) for h in ph])
        else:
            th = _halves(tp[per] + 1, t)
            b = 2 * full - _mean([_subfit(d, a & c, fml) for a in ih for c in th])
        per_partition.append(b)
    res = fit_case("classic").bias_corr(method="jackknife", jackknife=variant, multiple=2, split=split, seed=seed)
    _check_combination(res, _mean(per_partition), f"{variant} multiple, split={split}")


def test_string_and_categorical_time_id():
    # time order = sorted values, as for fixed-effect levels
    d = data("unbal").copy()
    fml = "y ~ x1 + x2 + d | i"
    labels = d["t"].map(lambda v: f"t{v}")  # "t10" sorts before "t2"
    d["t_pad"] = d["t"].map(lambda v: f"t{v:02d}")  # sorts as the numbers
    d["t_str"] = labels
    d["t_rank"] = labels.map({s: k for k, s in enumerate(sorted(labels.unique()))})
    d["t_cat"] = pd.Categorical(d["t"], categories=sorted(d["t"].unique(), reverse=True))
    d["t_neg"] = -d["t"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = fereg.feglm(fml, d, family="logit", ssc=SSC, **FIT_KW)

    def jk(time_id):
        return m.bias_corr(method="jackknife", jackknife="ss1", time_id=time_id).coef().to_numpy()

    np.testing.assert_array_equal(jk("t_pad"), jk("t"))
    np.testing.assert_array_equal(jk("t_str"), jk("t_rank"))
    np.testing.assert_array_equal(jk("t_cat"), jk("t_neg"))
    assert not np.array_equal(jk("t_str"), jk("t"))
    d.loc[3, "t_str"] = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = fereg.feglm(fml, d, family="logit", ssc=SSC, **FIT_KW)
    with pytest.raises(ValueError, match="^the time variable has missing values in the estimation sample$"):
        m.bias_corr(method="jackknife", time_id="t_str")


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

# The core's messages (fereg-core API section 5): identical in R, Python and Stata.
CORE = {
    "no_fe": "bias correction and APEs need fixed effects",
    "slopes": "bias correction and APEs need intercept-only fixed effects (no varying slopes)",
    "classic_k": 'panel structure "classic" expects a one- or two-way fixed effects model',
    "network_k": 'panel structure "network" expects a two- or three-way fixed effects model',
    "network_which": "network structure: only the correction of both biases is available",
    "time_bias": "the time-effect bias needs a time fixed effect (a two-way model)",
    "crv3": "CRV3 is not available after bias correction",
    "jk_network": "the jackknife is not available for the network structure",
    "jk_time": "jackknife: this variant needs a time variable",
    "jk_multiple": "jackknife: multiple partitions are only available with the ss1 and ss2 variants",
    "double_kind": "the double jackknife needs both fixed-effect variables of the same kind (numeric or coded "
    "on common levels)",
    "L_jackknife": "bandwidth L = 1 has no effect: it only applies to the analytical correction",
    "L_time": "bandwidth L = 1 has no effect: it only applies to the individual-effect bias",
    "L_large": "bandwidth L = 9 is too large: the longest group has 8 observations",
    "multiple_time": "multiple partitions have no effect: the subpanels of this bias split only the time "
    "dimension, which no partition permutes",
    "multiple_individual": "multiple partitions have no effect: the subpanels of this bias split only the "
    "individual dimension, which no partition permutes",
    "subpanel_lost": "jackknife, subpanel time half 1: the coefficient of regressor 3 (x[2]) is not estimable",
}


def exact(text: str) -> str:
    return f"^{re.escape(text)}$"


ERRORS = [
    # Python arguments
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
    ("classic", dict(method="jackknife", split="time"), ValueError, "split is only used with multiple > 0"),
    ("classic", dict(method="jackknife", seed=3), ValueError, "seed is only used with multiple > 0"),
    ("classic", dict(method="jackknife", multiple=-1), ValueError, "multiple must be a non-negative integer"),
    ("classic", dict(method="jackknife", multiple=2, seed=-1), ValueError, "seed must be a non-negative integer"),
    ("classic", dict(method="jackknife", multiple=2, seed="a"), TypeError, "seed must be a non-negative integer"),
    ("classic", dict(method="jackknife", time_id="t"), ValueError, "time_id is only used by the jackknife of a one-way"),
    ("oneway", dict(method="jackknife", time_id=3), TypeError, "time_id must be the name of a column"),
    ("oneway", dict(method="jackknife", time_id="y_missing"), KeyError, "y_missing"),
    # core requests (exact texts)
    ("classic", dict(method="jackknife", L=1), ValueError, exact(CORE["L_jackknife"])),
    ("classic", dict(L=1, bias="time"), ValueError, exact(CORE["L_time"])),
    ("classic", dict(L=9), ValueError, exact(CORE["L_large"])),
    ("classic", dict(method="jackknife", jackknife="js", multiple=2), ValueError, exact(CORE["jk_multiple"])),
    ("classic", dict(panel_structure="network", bias="time"), ValueError, exact(CORE["network_which"])),
    ("network3", dict(panel_structure="network", bias="individual"), ValueError, exact(CORE["network_which"])),
    ("network3", dict(panel_structure="network", method="jackknife"), ValueError, exact(CORE["jk_network"])),
    ("network3", dict(), ValueError, exact(CORE["classic_k"])),
    ("oneway", dict(panel_structure="network"), ValueError, exact(CORE["network_k"])),
    ("oneway", dict(bias="time"), ValueError, exact(CORE["time_bias"])),
    ("oneway", dict(method="jackknife"), ValueError, exact(CORE["jk_time"])),
]


@pytest.mark.parametrize(("cid", "kw", "exc", "match"), ERRORS, ids=[f"{e[0]}-{i}" for i, e in enumerate(ERRORS)])
def test_rejected_combinations(cid, kw, exc, match):
    with pytest.raises(exc, match=match):
        fit_case(cid).bias_corr(**kw)


def test_rejected_models():
    with pytest.raises(ValueError, match=exact(CORE["no_fe"])):
        fit("y ~ x1 + x2 + d", "classic").bias_corr()
    with pytest.raises(ValueError, match=exact(CORE["slopes"])):
        fit("y ~ x1 + d | i[x2] + t", "classic").bias_corr()
    with pytest.raises(ValueError, match=exact(CORE["crv3"])):
        fit("y ~ x1 + x2 + d | i + t", "classic", vcov={"CRV3": "i"}).bias_corr()
    with pytest.raises(ValueError, match="cannot be applied to a bias-corrected model"):
        corrected("classic", 0, "both").bias_corr()


def test_one_way_double_is_delete_one_period():
    # one fixed effect: the double jackknife corrects the individual bias by deleting periods (jj)
    j = next(j for j in jackknife_cases() if j["id"] == "oneway_jj")
    m = fit_jk(j["fml"], j["data"])
    a = m.bias_corr(method="jackknife", jackknife="double", time_id="t")
    b = m.bias_corr(method="jackknife", jackknife="jj", time_id="t")
    np.testing.assert_array_equal(a.coef().to_numpy(), b.coef().to_numpy())
