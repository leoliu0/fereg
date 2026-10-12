"""fereg.design vs pyfixest 0.60.0 model matrices (names, values, sample)."""

import sys
import time
import warnings

import numpy as np
import pandas as pd
import pytest

from fereg.design import build_design, factor_column, term_labels
from fereg.formula import FormulaSyntaxError, parse

pf = pytest.importorskip("pyfixest")
pl = pytest.importorskip("polars")
from pyfixest.estimation.formula.model_matrix import create_model_matrix
from pyfixest.estimation.formula.parse import Formula as PfFormula

N = 400


def make_data(n=N, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n) + 3,
            "x3": rng.normal(size=n),
            "f1": rng.integers(0, 5, n),
            "f2": rng.integers(0, 4, n),
            "s": rng.choice(list("abcd"), n).astype(object),
            "z1": rng.normal(size=n),
            "z2": rng.normal(size=n),
            "b": rng.random(n) > 0.5,
            "ff": rng.integers(0, 3, n).astype(float),
            "w": rng.random(n) + 0.1,
            "off": rng.normal(size=n),
        }
    )
    df["e"] = df.z1 + df.x1 + rng.normal(size=n)
    df["c"] = pd.Categorical(
        rng.choice(list("pqr"), n), categories=["r", "q", "p", "u"]
    )
    df.loc[rng.choice(n, 3, replace=False), "y"] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "x1"] = np.nan
    df.loc[rng.choice(n, 7, replace=False), "s"] = None
    df.loc[rng.choice(n, 4, replace=False), "x2"] = -1.0  # log() -> NaN
    df.loc[rng.choice(n, 2, replace=False), "ff"] = np.nan
    df.loc[0, "s"] = "zz"  # level present only in a dropped row
    df.loc[0, "y"] = np.nan
    return df


DF = make_data()

FORMULAS = [
    "y~x1+x2",
    "y~x1|f1",
    "y~x1+x2|f1+f2",
    "y~I(x1**2)+x2",
    "y~x1*x2",
    "y~x1:x2+x1",
    "y~(x1+x2+x3)**2",
    "y~x1/x2",
    "y~C(f1)",
    "y~C(f1)|f2",
    "y~0+C(f1)",
    "y~C(f1)-1",
    "y~C(f1)+C(f2)-1",
    "y~i(f1)",
    "y~i(f1)|f2",
    "y~i(f1,x1)",
    "y~i(f1,x1,ref=2)",
    "y~i(f1,ref=2)|f2",
    "y~0+i(f1)",
    "y~i(f1,s)",
    "y~i(f1,f2)",
    "y~i(f1,f2,ref=1,ref2=2)",
    "y~i(f1,bin={'lo':[0,1]})",
    "y~i(s,bin={'ab':['a','b']})|f1",
    "y~s",
    "y~s|f1",
    "y~b",
    "y~C(b)",
    "y~c",
    "y~C(c)+x1",
    "y~ff+C(ff)",
    "y~log(x2)",
    "y~np.log(x2)+np.exp(x1)",
    "y~exp(x1)+abs(x1)",
    "y~log10(x2)+np.sqrt(x2)",
    "y~I(x1*x2)",
    "y~I(x1>0)",
    "y~C(f1):x1",
    "y~x1+C(f1):x1",
    "y~C(f1):C(f2)",
    "y~s*x1",
    "y~s:C(f1)",
    "y~0+s+C(f1)",
    "y~x1+x2:s|f1",
    "y~x1|f1^f2",
    "y~x1|s+f1",
    "y~x1|f1|e~z1",
    "y~1|e~z1+z2",
    "y~x1|e~z1+z2",
    "y~s|f1|e~z1+C(f2)",
    "y~x1:x2|f1|e~z1",
    "log(x2)~x1|f1",
    "y~C(s, Treatment('b'))",
    "y~poly(x1,2)",
    "y~center(x1)+s",
    "y~0+poly(x1,2)+s",
    "y~poly(x1,2)|f1|e~z1+center(z2)",
]


def design_keep(d):
    keep = np.ones(d.n, bool)
    if d.drop is not None:
        keep &= ~d.drop
    for a in [d.y, *d.x, *d.endog, *d.inst, d.weights, d.offset, *d.slopes]:
        if a is not None:
            keep &= np.isfinite(a)
    for fc in d.factors:
        if fc.kind == "f64":
            keep &= ~np.isnan(fc.values)
        elif fc.kind == "codes":
            keep &= (fc.values >= fc.base) & (fc.values < fc.base + fc.nlevels)
    return keep


def partition(*codes):
    """Canonical first-appearance codes of the row partition defined by codes."""
    key = pd.MultiIndex.from_arrays([np.asarray(c) for c in codes])
    return pd.factorize(key)[0]


def fe_codes(d, k, keep):
    cols = []
    for i in d.fe[k][0]:
        fc = d.factors[i]
        cols.append(fc.values[keep])
    return partition(*cols)


def assert_matches_pyfixest(fml, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = create_model_matrix(PfFormula.parse(fml)[0], DF.copy(), **kw)
        d = build_design(parse(fml)[0], DF, **kw)
    keep = design_keep(d)
    expected_rows = np.array(sorted(set(range(N)) - set(pm.na_index)))
    np.testing.assert_array_equal(np.nonzero(keep)[0], expected_rows)
    assert d.coefnames == list(pm.independent.columns)
    for name, col in zip(d.x_names, d.x):
        np.testing.assert_allclose(
            col[keep], pm.independent[name].to_numpy(), rtol=1e-15, atol=1e-15
        )
    assert d.y_name == pm.dependent.columns[0]
    np.testing.assert_array_equal(d.y[keep], pm.dependent.iloc[:, 0].to_numpy())
    if pm.endogenous is not None:
        assert d.endog_names == list(pm.endogenous.columns)
        np.testing.assert_array_equal(
            d.endog[0][keep], pm.endogenous.iloc[:, 0].to_numpy()
        )
        z_names = [c for c in pm.instruments.columns if c not in d.coefnames]
        assert d.inst_names == z_names
        for name, col in zip(d.inst_names, d.inst):
            np.testing.assert_allclose(
                col[keep], pm.instruments[name].to_numpy(), rtol=1e-15, atol=1e-15
            )
    else:
        assert d.endog == [] and d.inst == []
    if pm.fixed_effects is not None:
        f = parse(fml)[0]
        for k, term in enumerate(f.fe):
            col = f"__fixed_effect__({term.label.replace('^', '_')})"
            np.testing.assert_array_equal(
                fe_codes(d, k, keep), partition(pm.fixed_effects[col].to_numpy())
            )
    else:
        assert d.fe == []
    if pm.weights is not None:
        np.testing.assert_array_equal(d.weights[keep], pm.weights.iloc[:, 0].to_numpy())
    if pm.offset is not None:
        np.testing.assert_array_equal(d.offset[keep], pm.offset.iloc[:, 0].to_numpy())
    return d


@pytest.mark.parametrize("fml", FORMULAS)
def test_design_matches_pyfixest(fml):
    assert_matches_pyfixest(fml)


@pytest.mark.parametrize("fml", ["y~x1|f1", "y~C(f1)+x2", "y~s|f2"])
def test_weights_and_offset_match_pyfixest(fml):
    assert_matches_pyfixest(fml, weights="w")
    assert_matches_pyfixest(fml, offset="off")


def test_intercept_rules():
    d = build_design(parse("y ~ x1")[0], DF)
    assert d.intercept and d.coefnames[0] == "Intercept"
    assert not build_design(parse("y ~ x1 | f1")[0], DF).intercept
    assert not build_design(parse("y ~ x1 - 1")[0], DF).intercept
    assert not build_design(parse("y ~ x1")[0], DF, drop_intercept=True).intercept
    # slope-only fixed effects keep the intercept (fixest semantics)
    assert build_design(parse("y ~ x1 | f1[[x3]]")[0], DF).intercept


def to_polars(df):
    cols = {}
    for name in df.columns:
        s = df[name]
        if isinstance(s.dtype, pd.CategoricalDtype):
            cols[name] = pl.Series(
                name, s.astype(object).tolist(), dtype=pl.Enum(list(s.cat.categories))
            )
        elif s.dtype == object:
            cols[name] = pl.Series(name, s.tolist(), dtype=pl.String)
        else:
            cols[name] = pl.from_pandas(s)
    return pl.DataFrame(cols)


PL = to_polars(DF)


def assert_same_design(a, b):
    assert a.coefnames == b.coefnames and a.x_names == b.x_names
    assert (
        a.y_name == b.y_name
        and a.endog_names == b.endog_names
        and a.inst_names == b.inst_names
    )
    for u, v in zip(
        [a.y, *a.x, *a.endog, *a.inst, *a.slopes],
        [b.y, *b.x, *b.endog, *b.inst, *b.slopes],
    ):
        np.testing.assert_array_equal(u, v)
    assert (a.drop is None) == (b.drop is None)
    if a.drop is not None:
        np.testing.assert_array_equal(a.drop, b.drop)
    assert a.fe == b.fe and a.cluster == b.cluster
    for fa, fb in zip(a.factors, b.factors):
        assert fa.name == fb.name and fa.kind == fb.kind
        np.testing.assert_array_equal(fa.values, fb.values)
        rows = np.arange(a.n)
        assert fa.labels_at(rows) == fb.labels_at(rows)


@pytest.mark.parametrize("fml", FORMULAS)
def test_polars_matches_pandas(fml):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = build_design(parse(fml)[0], DF, cluster=["f2", "s^f1"])
        b = build_design(parse(fml)[0], PL, cluster=["f2", "s^f1"])
    assert_same_design(a, b)


def test_nullable_dtypes():
    df = DF[["y", "x1", "f1"]].copy()
    df["xn"] = pd.array(df["x1"].to_numpy(), dtype="Float64")
    df["xi"] = pd.array(df["f1"].to_numpy(), dtype="Int64")
    df.loc[5, "xi"] = pd.NA
    df["fi"] = df["xi"]
    d = build_design(parse("y ~ xn + xi | fi")[0], df)
    np.testing.assert_array_equal(np.isnan(d.x[0]), df["x1"].isna().to_numpy())
    assert np.isnan(d.x[1][5]) and d.x[1][4] == df.loc[4, "f1"]
    fc = d.factors[0]
    assert fc.kind == "codes" and fc.values[5] == -1
    assert fc.labels_at(np.array([4]))[0] == str(df.loc[4, "f1"])
    # polars follows to_pandas() (pyfixest's conversion): integers with nulls
    # become float64, booleans with nulls become categorical
    pdl = pl.DataFrame(
        {
            "y": [1.0, None, 3.0, 4.0, 5.0],
            "x": [1.0, 2.0, None, 4.0, 5.0],
            "f": [1, None, 2, 2, 1],
            "b": [True, None, False, True, False],
        }
    )
    d = build_design(parse("y ~ x + b | f")[0], pdl)
    assert np.isnan(d.y[1]) and np.isnan(d.x[0][2])
    assert d.x_names == ["x", "b[T.True]"] and d.drop[1]
    assert d.factors[0].kind == "f64" and d.factors[0].labels_at(np.arange(5)) == [
        "1.0",
        None,
        "2.0",
        "2.0",
        "1.0",
    ]


def test_zero_copy_columns():
    df = pd.DataFrame(
        {
            "y": np.arange(10.0),
            "x": np.arange(10.0) ** 2,
            "f": np.arange(10) % 3,
            "w": np.ones(10),
        }
    )
    d = build_design(parse("y ~ x | f")[0], df, weights="w")
    assert np.shares_memory(d.y, df["y"].to_numpy())
    assert np.shares_memory(d.x[0], df["x"].to_numpy())
    assert np.shares_memory(d.weights, df["w"].to_numpy())
    assert d.factors[0].kind == "i64" and np.shares_memory(
        d.factors[0].values, df["f"].to_numpy()
    )
    p = pl.from_pandas(df)
    d = build_design(parse("y ~ x | f")[0], p)
    assert np.shares_memory(d.x[0], p["x"].to_numpy())


def test_fe_column_kinds_and_labels():
    df = pd.DataFrame(
        {
            "y": np.arange(6.0),
            "fi": [1, 1, 2, 2, 3, 3],
            "ff": [1.0, 1.0, 2.0, 2.0, np.nan, 3.0],
            "fs": ["a", "a", "b", "b", None, "c"],
            "fc": pd.Categorical(
                ["u", "u", "v", "v", "u", "v"], categories=["v", "u", "w"]
            ),
            "fb": [True, False, True, False, True, False],
            "f8": np.array([1, 1, 2, 2, 3, 3], dtype=np.int8),
        }
    )
    d = build_design(parse("y ~ 1 | fi + ff + fs + fc + fb + f8")[0], df)
    kinds = {fc.name: fc.kind for fc in d.factors}
    assert kinds == {
        "fi": "i64",
        "ff": "f64",
        "fs": "codes",
        "fc": "codes",
        "fb": "codes",
        "f8": "i64",
    }
    rows = np.arange(6)
    labels = {fc.name: fc.labels_at(rows) for fc in d.factors}
    assert labels["fi"] == ["1", "1", "2", "2", "3", "3"]
    assert labels["ff"] == ["1.0", "1.0", "2.0", "2.0", None, "3.0"]
    assert labels["fs"] == ["a", "a", "b", "b", None, "c"]
    assert labels["fc"] == ["u", "u", "v", "v", "u", "v"]
    assert labels["fb"] == ["True", "False", "True", "False", "True", "False"]
    fc = d.factors[3]
    assert fc.nlevels == 3 and fc.values.dtype == np.int32


@pytest.mark.parametrize("fe", ["fi", "ff", "fs"])
def test_labels_equal_pyfixest_fixef_keys(fe):
    rng = np.random.default_rng(3)
    n = 120
    df = pd.DataFrame(
        {"y": rng.normal(size=n), "x": rng.normal(size=n), "fi": rng.integers(0, 6, n)}
    )
    df["ff"] = df["fi"].astype(float) / 2
    df["fs"] = np.array(list("abcdef"), dtype=object)[df["fi"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = pf.feols(f"y ~ x | {fe}", df)
    keys = list(fit.fixef()[f"C({fe})"].keys())
    d = build_design(parse(f"y ~ x | {fe}")[0], df)
    assert sorted(set(d.factors[0].labels_at(np.arange(n)))) == sorted(keys)


def test_interaction_labels_and_clusters():
    df = pd.DataFrame(
        {"y": np.arange(5.0), "a": [1, 1, 2, 2, 1], "b": ["x", "y", "x", None, "x"]}
    )
    d = build_design(parse("y ~ 1 | a^b")[0], df, cluster=["a", "a^b"])
    assert d.fe == [([0, 1], True, [])] and d.cluster == [[0], [0, 1]]
    assert term_labels(d, 0, np.arange(5)) == ["1^x", "1^y", "2^x", None, "1^x"]
    fc = factor_column(df, "a^b")
    assert fc.name == "a^b" and fc.kind == "codes"
    np.testing.assert_array_equal(fc.values, [0, 1, 2, -1, 0])
    assert fc.labels_at(np.arange(5)) == ["1^x", "1^y", "2^x", None, "1^x"]
    assert factor_column(pl.from_pandas(df), "a^b").labels_at(
        np.arange(5)
    ) == fc.labels_at(np.arange(5))
    single = factor_column(df, "b")
    assert single.kind == "codes" and single.labels_at(np.arange(5))[3] is None


def test_slopes():
    d = build_design(parse("y ~ x1 | f1[x3] + f2[[x2, x3]]")[0], DF)
    assert d.slope_names == ["x3", "x2"]
    assert d.fe == [([0], True, [0]), ([1], False, [1, 0])]
    assert d.fe_names == ["f1[x3]", "f2[[x2, x3]]"]
    assert np.shares_memory(d.slopes[0], DF["x3"].to_numpy())


def test_drop_marks_transform_and_categorical_na():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = build_design(parse("y ~ log(x2) + s")[0], DF)
    expected = (DF["x2"] <= 0).to_numpy() | DF["s"].isna().to_numpy()
    np.testing.assert_array_equal(d.drop, expected)
    assert build_design(parse("y ~ x1 + x2 | f1")[0], DF).drop is None


def test_pyfixest_log_warns():
    with pytest.warns(UserWarning, match="rows with infinite values detected"):
        build_design(parse("y ~ log(x2)")[0], DF)


def test_like_aligns_columns_on_newdata():
    train = build_design(parse("y ~ x1 + C(f1) + i(s, ref='b') | f2")[0], DF)
    new = DF.iloc[:20].copy()
    new = new[new["f1"] != 3].drop(columns="y")
    new.loc[new.index[0], "s"] = "unseen"
    d = build_design(parse("y ~ x1 + C(f1) + i(s, ref='b') | f2")[0], new, like=train)
    assert d.y is None and d.x_names == train.x_names
    j = d.x_names.index("C(f1)[T.3]")
    assert not d.x[j].any()
    assert d.drop[0]
    assert d.factors[0].name == "f2"


def test_errors():
    df = DF.copy()
    with pytest.raises(TypeError, match="dependent variable must be numeric"):
        build_design(parse("s ~ x1")[0], df)
    with pytest.raises(KeyError, match="nope"):
        build_design(parse("y ~ nope")[0], df)
    df["w0"] = df["w"]
    df.loc[3, "w0"] = 0.0
    with pytest.raises(ValueError, match="non-negative"):
        build_design(parse("y ~ x1")[0], df, weights="w0")
    with pytest.raises(ValueError, match="must be numeric"):
        build_design(parse("y ~ x1")[0], df, weights="s")
    with pytest.raises(ValueError, match="not found"):
        build_design(parse("y ~ x1")[0], df, offset="nope")
    with pytest.raises(ValueError, match="TreatmentContrasts.base"):
        build_design(parse("y ~ i(f1, ref=9)")[0], df)
    with pytest.raises(FormulaSyntaxError, match="unexpected keyword"):
        build_design(parse("y ~ i(f1, foo=1)")[0], df)
    with pytest.raises(TypeError, match="data must be"):
        build_design(parse("y ~ x1")[0], df.to_dict())


def test_keep_extension():
    d = build_design(parse("y ~ i(f1, keep=[1, 3])")[0], DF)
    assert d.x_names == ["f1::1", "f1::3"]


def test_unsupported_term_without_formulaic(monkeypatch):
    monkeypatch.setitem(sys.modules, "formulaic", None)
    with pytest.raises(ImportError, match="pip install formulaic"):
        build_design(parse("y ~ poly(x1, 2)")[0], DF)
    with pytest.raises(FormulaSyntaxError, match="i\\(\\) cannot be combined"):
        build_design(parse("y ~ i(f1) + poly(x1, 2)")[0], DF)


def test_speed_10m_rows():
    n = 10_000_000
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "y": rng.random(n),
            "x1": rng.random(n),
            "x2": rng.random(n),
            "f1": rng.integers(0, 100_000, n),
            "f2": rng.integers(0, 10, n),
        }
    )
    (f,) = parse("y ~ x1 + x2 | f1 + f2")
    build_design(f, df)
    best = np.inf
    for _ in range(5):
        t0 = time.perf_counter()
        d = build_design(f, df)
        best = min(best, time.perf_counter() - t0)
    assert best < 0.05, f"build_design took {best * 1e3:.1f} ms"
    for col, name in [(d.y, "y"), (d.x[0], "x1"), (d.x[1], "x2")]:
        assert np.shares_memory(col, df[name].to_numpy())
    assert all(np.shares_memory(fc.values, df[fc.name].to_numpy()) for fc in d.factors)
