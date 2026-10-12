"""Unit tests of the binding layer and of the host-side helpers."""

import numpy as np
import pandas as pd
import pytest

import fereg
from fereg import _binding, _core
from fereg._vcov import VcovSpec, parse_vcov
from fereg.summary import markdown_table


def test_version():
    assert fereg.__version__ == _core.__version__
    assert fereg.__version__.count(".") == 2


@pytest.mark.parametrize(
    "vcov,expected",
    [
        (None, VcovSpec("iid", "iid", "iid")),
        ("iid", VcovSpec("iid", "iid", "iid")),
        ("hetero", VcovSpec("hetero", "hetero", "hetero")),
        ("HC1", VcovSpec("hetero", "hetero", "HC1")),
        ({"CRV1": "a"}, VcovSpec("cluster", "CRV", "CRV1", ("a",))),
        ({"CRV1": "a + b"}, VcovSpec("cluster", "CRV", "CRV1", ("a", "b"))),
        ({"CRV1": "a^b+c+d+e"}, VcovSpec("cluster", "CRV", "CRV1", ("a^b", "c", "d", "e"))),
    ],
)
def test_parse_vcov(vcov, expected):
    assert parse_vcov(vcov) == expected


@pytest.mark.parametrize(
    "vcov,err",
    [
        ("HC3", NotImplementedError),
        ({"CRV3": "a"}, NotImplementedError),
        ({"CRV1": "a+b+c+d+e"}, ValueError),
        ({"CRV1": "a+a"}, ValueError),
        ("robust", ValueError),
        (3, TypeError),
    ],
)
def test_parse_vcov_errors(vcov, err):
    with pytest.raises(err):
        parse_vcov(vcov)


def test_ssc():
    assert fereg.ssc() == {
        "k_adj": True,
        "k_fixef": "nonnested",
        "G_adj": True,
        "G_df": "min",
        "k_exact": False,
        "t_df": "min",
    }
    with pytest.raises(ValueError):
        fereg.ssc(k_fixef="nested")
    assert _binding.core_ssc({"k_adj": False, "other": 1}) == {"k_adj": False}


def test_markdown_table_matches_tabulate():
    pytest.importorskip("tabulate")
    df = pd.DataFrame(
        {
            "Estimate": [-0.92412, 0.17, 12.5],
            "Std. Error": [0.0561, 0.015, 1.0],
            "Pr(>|t|)": [0.0, 0.25, np.nan],
        },
        index=pd.Index(["X1", "a_long_coefficient_name", "Intercept"], name="Coefficient"),
    ).round(3)
    assert markdown_table(df, ".3f") == df.to_markdown(floatfmt=".3f")


def test_dist_functions_match_scipy():
    stats = pytest.importorskip("scipy.stats")
    x = np.array([-8.0, -2.5, -0.3, 0.0, 1.7, 6.0])
    np.testing.assert_allclose(_binding.pnorm(x), stats.norm.cdf(x), rtol=1e-13)
    p = np.array([1e-10, 0.025, 0.5, 0.975])
    np.testing.assert_allclose(_binding.qnorm(p), stats.norm.ppf(p), rtol=1e-13)
    for df in (1.0, 3.5, 30.0, 1e6):
        np.testing.assert_allclose(_binding.pt(x, df), stats.t.cdf(x, df), rtol=1e-12)
        np.testing.assert_allclose(_binding.qt(p, df), stats.t.ppf(p, df), rtol=1e-11)
    f = np.array([0.1, 1.0, 4.0, 50.0])
    np.testing.assert_allclose(_binding.pf_upper(f, 3, 47), stats.f.sf(f, 3, 47), rtol=1e-11)
    np.testing.assert_allclose(_binding.pchisq_upper(f, 4), stats.chi2.sf(f, 4), rtol=1e-11)
    assert isinstance(_binding.pnorm(0.0), float)


def test_core_input_errors():
    y = np.arange(10.0)
    with pytest.raises(ValueError, match="length"):
        _core.fit(y, [np.ones(5)], [], [], [], [], [], [])
    with pytest.raises(ValueError, match="contiguous"):
        _core.fit(y, [np.ones(20)[::2]], [], [], [], [], [], [])
    with pytest.raises(ValueError, match="factor kind"):
        _core.fit(y, [], [], [], [("str", np.zeros(10), 0, 0)], ["g"], [], [])
    with pytest.raises(ValueError, match="flavor"):
        _core.fit(y, [], [], [], [], [], [], [], flavor="stata")
    with pytest.raises(ValueError, match="ssc"):
        _core.fit(y, [], [], [], [], [], [], [], ssc={"bogus": 1})


def test_fit_result_fields():
    rng = np.random.default_rng(0)
    n = 200
    g = rng.integers(0, 10, n).astype(np.int64)
    x = rng.normal(size=n)
    y = 2 * x + g + rng.normal(size=n)
    out = _core.fit(y, [x], [], [], [("i64", g, 0, 0)], ["g"], [([0], True, [])], [])
    assert out.nobs == n and out.nobs_origin == n
    assert out.coef_source == [("x", 0)]
    assert out.vcov["vcov"].shape == (1, 1)
    assert out.fe_sizes == [10]
    assert out.resid.shape == (n,) and out.fitted.shape == (n,)
    assert isinstance(out.state, _core.FitState) and out.state.nobs == n
    v = out.state.vcov("hetero")
    assert v["kind"] == "hetero" and v["vcov"].shape == (1, 1)
    np.testing.assert_allclose(out.coef[0], np.polyfit(x - pd.Series(x).groupby(g).transform("mean"), y - pd.Series(y).groupby(g).transform("mean"), 1)[0], rtol=1e-10)
