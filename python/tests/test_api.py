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
        ("HC2", VcovSpec("hc2", "hetero", "HC2")),
        ("HC3", VcovSpec("hc3", "hetero", "HC3")),
        ({"CRV3": "a"}, VcovSpec("crv3", "CRV", "CRV3", ("a",))),
        ({"CRV3": "a+b^c"}, VcovSpec("crv3", "CRV", "CRV3", ("a", "b^c"))),
    ],
)
def test_parse_vcov(vcov, expected):
    assert parse_vcov(vcov) == expected


def test_parse_vcov_kwargs():
    nw = parse_vcov("NW", {"time_id": "t", "panel_id": "i", "lag": 3})
    assert (nw.kind, nw.vcov_type, nw.detail, nw.lag, nw.time_id, nw.panel_id) == ("nw", "HAC", "NW", 3, "t", "i")
    assert nw.columns() == {"time": "t", "panel": "i", "lat": None, "lon": None}
    assert nw.core_options()["lag"] == 3
    dk = parse_vcov("DK", {"time_id": "t"})
    assert (dk.kind, dk.detail, dk.lag, dk.panel_id) == ("dk", "DK", None, None)
    # HAC options are ignored by other estimators, as in pyfixest.
    assert parse_vcov("hetero", {"lag": 2}).core_options()["lag"] is None
    c = parse_vcov("conley", {"lat": "la", "lon": "lo", "cutoff": "60mi", "pixel": 5, "distance": "spherical"})
    assert (c.kind, c.vcov_type, c.detail) == ("conley", "conley", "conley")
    assert c.columns() == {"time": None, "panel": None, "lat": "la", "lon": "lo"}
    # fixest: miles to km, and the pixel shares the cutoff's unit.
    assert c.cutoff == pytest.approx(60 * 1.60934) and c.pixel == pytest.approx(5 * 1.60934) and c.spherical
    assert parse_vcov("conley", {"lat": "la", "lon": "lo", "cutoff": "100 km"}).cutoff == 100.0
    d = parse_vcov("conley", {"lat": "la", "lon": "lo"})
    assert (d.cutoff, d.pixel, d.spherical) == (None, 0.0, False)


@pytest.mark.parametrize(
    "vcov,kwargs,err,match",
    [
        ({"CRV1": "a+b+c+d+e"}, None, ValueError, "at most 4"),
        ({"CRV3": "a+b+c+d+e"}, None, ValueError, "at most 4"),
        ({"CRV1": "a+a"}, None, ValueError, "duplicate"),
        ({"CRV2": "a"}, None, ValueError, "vcov dict key must be CRV1 or CRV3"),
        ({"CRV1": 1}, None, TypeError, "vcov dict value must be a string"),
        ("robust", None, ValueError, "vcov string must be iid, hetero, HC1, HC2, HC3, NW, or DK"),
        ("hc3", None, ValueError, "vcov string must be"),
        (3, None, TypeError, "dict, string or None"),
        ("nid", None, NotImplementedError, "Only models of type Quantreg support"),
        ("NW", None, ValueError, "Missing required 'time_id' for NW/DK vcov"),
        ("DK", {"panel_id": "i"}, ValueError, "Missing required 'time_id' for NW/DK vcov"),
        ("NW", {"time_id": "t", "bogus": 1}, ValueError, "keys 'lag', 'time_id', or 'panel_id'"),
        ("NW", {"time_id": "t", "lag": "2"}, ValueError, "integer values for 'lag'"),
        ("NW", {"time_id": 3}, ValueError, "string values for 'time_id'"),
        ("NW", {"time_id": "t", "panel_id": 1}, ValueError, "string values for 'panel_id'"),
        ("conley", {"lat": "la"}, ValueError, "latitude and longitude"),
        ("conley", {"lat": "la", "lon": "lo", "cutoff": -1}, ValueError, "'cutoff'"),
        ("conley", {"lat": "la", "lon": "lo", "cutoff": "1.5mi"}, ValueError, "'cutoff'"),
        ("conley", {"lat": "la", "lon": "lo", "pixel": -2}, ValueError, "'pixel'"),
        ("conley", {"lat": "la", "lon": "lo", "distance": "flat"}, ValueError, "'triangular' or 'spherical'"),
    ],
)
def test_parse_vcov_errors(vcov, kwargs, err, match):
    with pytest.raises(err, match=match):
        parse_vcov(vcov, kwargs)


def test_parse_vcov_kwargs_columns():
    with pytest.raises(ValueError, match="The variable 'tt' is not in the data."):
        parse_vcov("NW", {"time_id": "tt"}, columns=["t", "i"])


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
        # Older scipy (the last release for Python 3.10 is 1.15) is less accurate: its df=1 tail
        # quantiles differ by 2e-11 relative, and it returns ~1e-16 for the exact-zero median.
        np.testing.assert_allclose(_binding.qt(p, df), stats.t.ppf(p, df), rtol=1e-10, atol=1e-14)
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
