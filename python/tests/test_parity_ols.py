"""OLS parity with pyfixest 0.60.0 (coefficients / standard errors to 1e-8 relative)."""

import numpy as np
import pandas as pd
import pytest
from _parity import TIGHT, assert_same_summary, compare, compare_multi, make_data, pf

import fereg

DATA = make_data()

OLS_CASES = [
    ("y ~ x1 + x2", None, {}),
    ("y ~ x1 + x2 | f1", None, {}),
    ("y ~ x1 + x2 | f1 + f2", None, {}),
    ("y ~ x1 + x2 + x3 | f1 + f2 + f3", None, {}),
    ("y ~ x1 + x2 | f1^f3 + f2", None, {}),
    ("y ~ x1 + x2 | s3 + f1", None, {}),
    ("y ~ x1 + C(h) | f1", None, {}),
    ("y ~ x1 + i(h, ref=1) | f1 + f2", None, {}),
    ("y ~ x1 + x2 | f1 + f2", "hetero", {}),
    ("y ~ x1 + x2", "hetero", {}),
    ("y ~ x1 + x2 | f1 + f2", {"CRV1": "f1"}, {}),
    ("y ~ x1 + x2 | f1 + f2", {"CRV1": "cl"}, {}),
    ("y ~ x1 + x2 | f1 + f2", {"CRV1": "cl2"}, {}),
    ("y ~ x1 + x2 | f1 + f2", {"CRV1": "f1+f3"}, {}),
    ("y ~ x1 + x2 | f1", {"CRV1": "cl+cl2"}, {}),
    ("y ~ x1 + x2", {"CRV1": "cl"}, {}),
    ("y ~ x1 + x2 | f1 + f2", None, {"weights": "w"}),
    ("y ~ x1 + x2 | f1 + f2", "hetero", {"weights": "w"}),
    ("y ~ x1 + x2 | f1 + f2", {"CRV1": "cl"}, {"weights": "w"}),
    ("y ~ x1 + x2 | f1 + f2", None, {"weights": "fw", "weights_type": "fweights"}),
    ("y ~ x1 + x2 | f1 + f2", "hetero", {"weights": "fw", "weights_type": "fweights"}),
    ("y ~ x1 + x2 | f1 + f2", None, {"fixef_rm": "none"}),
    ("y ~ x1 + x2 | f1 + f2", {"CRV1": "f1"}, {"ssc": {"k_adj": False, "G_adj": False}}),
    ("y ~ x1 + x2 | f1 + f2", {"CRV1": "f1"}, {"ssc": {"k_fixef": "full"}}),
    ("y ~ x1 + x2 | f1 + f2", None, {"ssc": {"k_fixef": "none"}}),
    ("y ~ x1 + x2 | f1 + f2", {"CRV1": "cl+cl2"}, {"ssc": {"G_df": "conventional"}}),
    ("y ~ x1 + x2 - 1", None, {}),
    ("y ~ x1 + x2", None, {"drop_intercept": True}),
    ("y ~ x1 + x2 + xc | f1", None, {}),
    ("y ~ i(h, x1) + x2 | f1", None, {}),
    ("y ~ x1 * x2 | f1 + f2", "hetero", {}),
    ("y ~ I(x1 ** 2) + x1:x2 | f1", None, {}),
    ("y ~ C(s3) + x1", {"CRV1": "f1"}, {}),
]


def _pf_kwargs(kw: dict) -> dict:
    kw = dict(kw)
    if "ssc" in kw:
        kw["ssc"] = pf.ssc(**kw["ssc"])
    return kw


def _ours_kwargs(kw: dict) -> dict:
    kw = dict(kw)
    if "ssc" in kw:
        kw["ssc"] = fereg.ssc(**kw["ssc"])
    return kw


@pytest.mark.parametrize("fml,vcov,kw", OLS_CASES, ids=[f"{c[0]}|{c[1]}|{c[2]}" for c in OLS_CASES])
def test_ols_parity(fml, vcov, kw):
    theirs = pf.feols(fml, DATA, vcov=vcov, demeaner=TIGHT, **_pf_kwargs(kw))
    ours = fereg.feols(fml, DATA, vcov=vcov, **_ours_kwargs(kw))
    compare(ours, theirs)
    np.testing.assert_allclose(ours.tidy().to_numpy(), theirs.tidy().to_numpy(), rtol=1e-7, atol=1e-12)
    assert list(ours.tidy().columns) == list(theirs.tidy().columns)
    for attr in ("_r2", "_adj_r2", "_rmse", "_r2_within", "_adj_r2_within", "_f_statistic"):
        a, b = getattr(ours, attr), getattr(theirs, attr)
        np.testing.assert_allclose(a, b, rtol=1e-8, equal_nan=True, err_msg=attr)
    assert ours._vcov_type == theirs._vcov_type
    assert ours._vcov_type_detail == theirs._vcov_type_detail
    assert ours._is_clustered == theirs._is_clustered
    assert ours._clustervar == theirs._clustervar
    # With G_df="conventional" pyfixest also lists the two-way intersection count; fereg lists
    # the per-dimension counts only.
    assert list(theirs._G)[: len(ours._G)] == list(ours._G)
    assert sorted(ours._na_index) == sorted(theirs._na_index)
    assert ours._collin_vars == [str(c) for c in theirs._collin_vars]


def test_ols_resid_predict_confint():
    fml = "y ~ x1 + x2 | f1 + f2"
    theirs = pf.feols(fml, DATA, demeaner=TIGHT)
    ours = fereg.feols(fml, DATA)
    np.testing.assert_allclose(ours.resid(), theirs.resid(), rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(ours.predict(), theirs.predict(), rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(ours.confint(alpha=0.1), theirs.confint(alpha=0.1), rtol=1e-8)
    assert list(ours.confint(alpha=0.1).columns) == list(theirs.confint(alpha=0.1).columns)
    np.testing.assert_allclose(ours.confint(keep="x1"), theirs.confint(keep="x1"), rtol=1e-8)
    np.testing.assert_allclose(
        ours.tidy(alpha=0.1).to_numpy(), theirs.tidy(alpha=0.1).to_numpy(), rtol=1e-7, atol=1e-12
    )
    w_o, w_t = ours.wald_test(R=np.array([[1.0, -1.0]])), theirs.wald_test(R=np.array([[1.0, -1.0]]))
    np.testing.assert_allclose(w_o.to_numpy(), w_t.to_numpy(), rtol=1e-7)


def test_ols_weighted_resid():
    fml = "y ~ x1 + x2 | f1"
    theirs = pf.feols(fml, DATA, weights="w", demeaner=TIGHT)
    ours = fereg.feols(fml, DATA, weights="w")
    np.testing.assert_allclose(ours.resid(), theirs.resid(), rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(ours._u_hat, theirs._u_hat, rtol=1e-7, atol=1e-9)


def _two_components(n: int = 600, seed: int = 3):
    rng = np.random.default_rng(seed)
    comp = rng.integers(0, 2, n)
    f1 = comp * 10 + rng.integers(0, 10, n)
    f2 = comp * 5 + rng.integers(0, 5, n)
    return pd.DataFrame(
        {"y": rng.normal(size=n) + 0.1 * f1 + 0.3 * f2, "x1": rng.normal(size=n), "f1": f1, "f2": f2,
         "f3": rng.integers(0, 4, n), "w": rng.uniform(0.5, 2, n)}
    )


@pytest.mark.parametrize(
    "fml,data,kw",
    [
        ("y ~ x1 + x2 | f1 + f2", DATA, {}),
        ("y ~ x1 + x2 | f1 + f2", DATA, {"weights": "w"}),
        ("y ~ x1 | f1 + s3", DATA, {}),
        ("y ~ x1 | f2 + f1 + f3", DATA, {}),
        ("y ~ x1 | f1", DATA, {}),
        ("y ~ x1 | f1 + f2", _two_components(), {}),
        ("y ~ x1 | f1 + f2 + f3", _two_components(), {}),
    ],
)
def test_fixef_matches_pyfixest(fml, data, kw):
    theirs = pf.feols(fml, data, demeaner=TIGHT, **kw)
    ours = fereg.feols(fml, data, **kw)
    ft = theirs.fixef(atol=1e-14, btol=1e-14)
    fo = ours.fixef()
    assert list(fo.keys()) == list(ft.keys())
    for key in ft:
        assert list(fo[key]) == list(ft[key])
        np.testing.assert_allclose(list(fo[key].values()), list(ft[key].values()), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "fml",
    ["y ~ x1 | f1^f3 + f2", "y ~ x1 | s3 + f1", "y ~ x1 | f1 + f2[x2]", "y ~ x1 | f1[[x2]] + f2", "y ~ x1 + C(h) | f1"],
)
def test_predict_newdata_reproduces_fitted(fml):
    ours = fereg.feols(fml, DATA)
    np.testing.assert_allclose(ours.predict(newdata=DATA)[ours._obs], ours.predict(), rtol=1e-10, atol=1e-10)


def test_predict_newdata_unseen_levels():
    fml = "y ~ x1 + x2 | f1 + f2"
    theirs = pf.feols(fml, DATA, demeaner=TIGHT)
    ours = fereg.feols(fml, DATA)
    theirs.fixef(atol=1e-14, btol=1e-14)
    new = DATA.head(40).copy()
    new.loc[new.index[:3], "f1"] = 99_999  # unseen level
    new = new.drop(columns="y")
    pt, po = theirs.predict(newdata=new), ours.predict(newdata=new)
    assert np.array_equal(np.isnan(pt), np.isnan(po))
    np.testing.assert_allclose(po[~np.isnan(po)], pt[~np.isnan(pt)], rtol=1e-6, atol=1e-8)


def test_posthoc_vcov():
    fml = "y ~ x1 + x2 | f1 + f2"
    ours = fereg.feols(fml, DATA)
    for vcov in ("hetero", {"CRV1": "f1"}, {"CRV1": "cl+cl2"}, {"CRV1": "f1^f3"}, "iid"):
        if vcov == {"CRV1": "f1^f3"}:
            # pyfixest has no "a^b" clusters; compare with the explicit interaction column.
            data = DATA.assign(f1f3=DATA["f1"].astype(str) + "_" + DATA["f3"].astype(str))
            theirs = pf.feols(fml, data, vcov={"CRV1": "f1f3"}, demeaner=TIGHT)
        else:
            theirs = pf.feols(fml, DATA, vcov=vcov, demeaner=TIGHT)
        assert ours.vcov(vcov) is ours
        np.testing.assert_allclose(ours._se, theirs._se, rtol=1e-8)
        assert ours._df_t == theirs._df_t
        assert ours._df_k == theirs._df_k


def test_lean_has_no_vcov():
    ours = fereg.feols("y ~ x1 | f1", DATA, lean=True)
    with pytest.raises(ValueError, match="lean"):
        ours.vcov("hetero")


def test_nan_cluster_is_error():
    data = DATA.assign(clna=DATA["cl"].astype(float))
    data.loc[100, "clna"] = np.nan
    with pytest.raises(ValueError):
        fereg.feols("y ~ x1 | f1", data, vcov={"CRV1": "clna"})


def test_multiple_estimation():
    for fml in ("y + y2 ~ x1 | f1", "y ~ sw(x1, x2) | f1", "y ~ csw(x1, x2, x3) | f1 + f2", "y ~ x1 | csw0(f1, f2)"):
        theirs = pf.feols(fml, DATA, demeaner=TIGHT)
        ours = fereg.feols(fml, DATA)
        assert isinstance(ours, fereg.FixestMulti)
        compare_multi(ours, theirs)
        o, t = ours.tidy(), theirs.tidy()
        assert o.index.equals(t.index)
        np.testing.assert_allclose(o.to_numpy(), t.to_numpy(), rtol=1e-7, atol=1e-12)


@pytest.mark.parametrize("kind", ["split", "fsplit"])
def test_split(kind):
    fml = "y ~ x1 + x2 | f1"
    theirs = pf.feols(fml, DATA, demeaner=TIGHT, **{kind: "g"})
    ours = fereg.feols(fml, DATA, **{kind: "g"})
    compare_multi(ours, theirs)
    assert ours.tidy()["Sample"].astype(str).tolist() == theirs.tidy()["Sample"].astype(str).tolist()


def test_polars_input():
    pl = pytest.importorskip("polars")
    fml = "y ~ x1 + x2 | f1 + f2"
    theirs = pf.feols(fml, DATA, demeaner=TIGHT)
    ours = fereg.feols(fml, pl.from_pandas(DATA))
    compare(ours, theirs)


@pytest.mark.parametrize(
    "fml,kw",
    [
        ("y ~ x1 + x2 | f1 + f2", {}),
        ("y ~ x1 + x2", {"vcov": {"CRV1": "cl"}}),
        ("y ~ csw(x1, x2) | f1", {"vcov": "hetero"}),
        ("y ~ x1 | f1", {"fsplit": "g"}),
    ],
)
def test_summary_matches_pyfixest(fml, kw, capsys):
    assert_same_summary(fereg.feols(fml, DATA, **kw), pf.feols(fml, DATA, demeaner=TIGHT, **kw), capsys)


def test_etable_layout():
    m1 = fereg.feols("y ~ x1 | f1", DATA)
    m2 = fereg.feols("y ~ x1 + x2 | f1 + f2", DATA, vcov={"CRV1": "f1"})
    t = fereg.etable([m1, m2])
    assert isinstance(t, pd.DataFrame)
    assert ("coef", "x1") in t.index and ("fe", "f2") in t.index and ("stats", "Observations") in t.index
    assert t.loc[("fe", "f2")].tolist() == ["-", "x"]
    assert t.loc[("stats", "Observations")].tolist() == [str(m1._N), str(m2._N)]
    assert "\\begin{tabular}" in fereg.etable([m1, m2], type="tex")
    assert "<table>" in fereg.etable([m1, m2], type="html")
