"""fereg.formula vs pyfixest 0.60.0 / formulaic (parsing and model expansion)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from fereg.formula import FeTerm, FormulaSyntaxError, parse, rhs_terms

pf = pytest.importorskip("pyfixest")
formulaic = pytest.importorskip("formulaic")
from pyfixest.errors import (
    EndogVarsAsCovarsError,
    InstrumentsAsCovarsError,
    UnderDeterminedIVError,
)
from pyfixest.errors import FormulaSyntaxError as PfSyntaxError
from pyfixest.estimation.formula.parse import Formula as PfFormula

PF_ERRORS = (
    PfSyntaxError,
    EndogVarsAsCovarsError,
    InstrumentsAsCovarsError,
    UnderDeterminedIVError,
)

MULTI = [
    "y ~ x1",
    "y~x1+x2",
    "y ~ x1 | f1",
    "y ~ x1 + x2 | f1 + f2",
    "y ~ x1 | f1^f2",
    "y~x1|f1|e~z1",
    "y ~ 1 | e ~ z1 + z2",
    "y ~ x1 | f1 | e ~ z1 + z2",
    "y + y2 ~ x1",
    "y + y2 ~ csw(x1, x2) | sw(f1, f2)",
    "y ~ sw(x1, x2)",
    "y ~ csw(x1, x2, x3)",
    "y ~ sw0(x1, x2)",
    "y ~ csw0(x1, x2) | f1",
    "y~mvsw(x1,x2)",
    "y ~ mvsw(x1, x2, x3) | f1",
    "y~sw0(x1,x2)|csw0(f1,f2)",
    "y ~ x1 | sw0(f1, f2)",
    "y ~ csw0(x1, sw(x2, x3))",
    "y ~ sw(x1, x2) + x3 | csw(f1, f2)",
    "y ~ x1 + sw(x2, x3) | f1 | e ~ z1",
    "sw(y, y2) ~ x1 | f1",
    "y ~ sw(x1, i(f1)) | f2",
    "y ~ csw(x1, I(x2**2), C(f1)) | f2",
    "y ~ x1 | csw0(f1, f1^f2)",
    "y ~ 0 + x1",
    "y ~ x1 - 1",
]


def _pyfixest_models(fml):
    return [m.formula for group in PfFormula.parse_to_dict(fml).values() for m in group]


@pytest.mark.parametrize("fml", MULTI)
def test_model_order_and_text(fml):
    assert [f.fml for f in parse(fml)] == _pyfixest_models(fml)


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {c: rng.normal(size=n) for c in ["y", "y2", "x1", "x2", "x3", "z1", "z2"]}
    )
    df["e"] = df.z1 + df.x1 + rng.normal(size=n)
    df["f1"] = rng.integers(0, 5, n)
    df["f2"] = rng.integers(0, 4, n)
    return df


@pytest.mark.parametrize(
    "fml",
    [
        "y + y2 ~ csw(x1, x2) | sw(f1, f2)",
        "y~sw0(x1,x2)|csw0(f1,f2)",
        "y ~ mvsw(x1, x2) | f1",
        "y ~ x1 + sw(x2, x3) | f1 | e ~ z1",
    ],
)
def test_fixestmulti_keys(fml):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = pf.feols(fml, _data())
    assert [f.fml for f in parse(fml)] == list(fit.all_fitted_models.keys())


RHS = [
    "x + 0",
    "x - 1",
    "0+x",
    "x + -1",
    " -1 + x",
    "x + 1 - 1 + 1",
    "x2:x1 + x1:x2",
    "(a+b):(c+d)",
    "(a+b)*(c+d)",
    "a/b",
    "(a+b)**2",
    "a*b - a",
    " log( x2 )+ np.log(x2)",
    "{x**2}",
    " - x",
    "x+(0)",
    " a:b:c + a + a:b",
    "C(f)*x",
    "a + -b",
    " x - 0",
    "I(x1**2)+x2",
    "C(s,Treatment('b'))",
    "a*b*c",
    "(a+b+c)**2 - a:b",
    "`my var` + x",
    "i(f1, x1, ref=2) + x2",
    "x1 + x1",
    "1",
    "0",
]


@pytest.mark.parametrize("rhs", RHS)
def test_term_algebra_matches_formulaic(rhs):
    expected = [str(t) for t in formulaic.Formula("y ~" + rhs).rhs]
    intercept, terms = rhs_terms(rhs)
    assert (["1"] if intercept else []) + [t.text for t in terms] == expected


def test_formula_fields():
    (f,) = parse("y ~ x1 + I(x2**2) - 1 | f1 + f1^f2 + g[x] + h[[a, b]] | e ~ z1 + z2")
    assert f.depvar == "y"
    assert f.rhs == ("x1", "I(x2**2)")
    assert f.intercept is False
    assert f.endog == ("e",)
    assert f.inst == ("z1", "z2")
    assert f.fe_text == "f1 + f1^f2 + g[x] + h[[a, b]]"
    assert f.fe == (
        FeTerm(("f1",), True, (), "f1"),
        FeTerm(("f1", "f2"), True, (), "f1^f2"),
        FeTerm(("g",), True, ("x",), "g[x]"),
        FeTerm(("h",), False, ("a", "b"), "h[[a, b]]"),
    )
    assert f.rhs_text == "x1 + I(x2**2) - 1"
    assert f.inst_text == "z1 + z2"


def test_fe_zero_and_one_mean_no_fixed_effects():
    for fml in ["y ~ x | 0", "y ~ x | 1"]:
        (f,) = parse(fml)
        assert f.fe == () and f.fe_text == "" and f.fml == "y ~ x"


def test_slopes_inside_multiple_estimation():
    models = parse("y ~ x | sw(f1, f2[x1, x2])")
    assert [m.fe for m in models] == [
        (FeTerm(("f1",), True, (), "f1"),),
        (FeTerm(("f2",), True, ("x1", "x2"), "f2[x1, x2]"),),
    ]


def test_duplicate_fe_terms_are_dropped():
    (f,) = parse("y ~ x | f1 + f1")
    assert [t.label for t in f.fe] == ["f1"]


def test_multiple_endogenous_allowed():
    (f,) = parse("y ~ x | e1 + e2 ~ z1 + z2")
    assert f.endog == ("e1", "e2") and f.inst == ("z1", "z2")


@pytest.mark.parametrize(
    "fml",
    [
        "y ~ x | f | e ~ z | g",
        "y ~ x ~ z",
        "y",
        "x | f",
        "y ~ x | e ~ z | f",
        "y ~ x | f ~ g | e ~ z",
        "y ~ sw(x1)",
        "y ~ x | e ~ x",
        "y ~ x | x ~ z",
        "y ~ x | e ~ 1",
    ],
)
def test_rejected_like_pyfixest(fml):
    with pytest.raises(FormulaSyntaxError):
        parse(fml)
    if (
        fml != "y ~ x | e ~ 1"
    ):  # pyfixest accepts an instrument-free first stage at parse time
        with pytest.raises(PF_ERRORS):
            PfFormula.parse(fml)


@pytest.mark.parametrize(
    "fml", ["y ~ x | f[", "y ~ x | f[x]]", "y ~ x | f^", "y ~ x . z", "y ~ 2 + x"]
)
def test_malformed(fml):
    with pytest.raises(FormulaSyntaxError):
        parse(fml)


def test_parse_returns_fresh_list():
    a = parse("y ~ sw(x1, x2)")
    a.clear()
    assert len(parse("y ~ sw(x1, x2)")) == 2
