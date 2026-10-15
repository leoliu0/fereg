"""pyfixest's ``context`` argument: names the formula takes from the caller's frame or a mapping
(native evaluation and the formulaic fallback), compared with pyfixest 0.60.0."""

import re
import warnings

import numpy as np
import pandas as pd
import pytest
from _parity import TIGHT, compare, pf

import fereg

formulaic_errors = pytest.importorskip("formulaic.errors")

rng = np.random.default_rng(12)
N = 400
DATA = pd.DataFrame({"f": rng.integers(0, 15, N), "g": rng.integers(0, 4, N), "x": rng.normal(size=N)})
DATA["x2"] = rng.uniform(1.0, 3.0, N)
DATA["y"] = 1.0 + DATA["x"] + 0.5 * np.log(DATA["x2"]) + rng.normal(size=N)
DATA["c"] = rng.poisson(np.exp(0.2 * DATA["x"])).astype(float)
DATA["b"] = (DATA["y"] > 1.0).astype(float)

SCALE = 3.0  # a module global the formulas use through context=0


def sq(v):
    return v**2


def lspline(series, knots):
    """pyfixest's documentation example (feols docstring): a linear spline, several columns."""
    vector = series.values
    columns = []
    for i, knot in enumerate(knots):
        column = np.minimum(vector, knot if i == 0 else knot - knots[i - 1])
        columns.append(column)
        vector = vector - column
    columns.append(vector)
    return np.column_stack(columns)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _both(fml, fit="feols", **kw):
    """fereg and pyfixest fits of the same call (same frame: context=int sees this one)."""
    ours = getattr(fereg, fit)(fml, DATA, **kw)
    theirs = getattr(pf, fit)(fml, DATA, demeaner=TIGHT, **kw)
    return ours, theirs


MAPPING_CASES = [
    ("y ~ sq(x) | f", "feols", {}),  # user function: formulaic fallback
    ("y ~ I(x * k) + x2 | f", "feols", {}),  # number: native path
    ("y ~ x + lspline(x2, [1.5, 2.5]) | f", "feols", {}),  # several columns
    ("sq(y) ~ x | f", "feols", {}),  # dependent variable
    ("y ~ i(g) + sq(x)", "feols", {}),  # next to a native i()
    ("y ~ log(x2) + x | f", "feols", {"log": np.log10}),  # overrides a built-in transform
    ("c ~ sq(x) + x2 | f", "fepois", {}),
    ("b ~ sq(x) | f", "feglm", {"family": "logit"}),
]


@pytest.mark.parametrize("fml,fit,extra", MAPPING_CASES, ids=[f"{c[0]}|{c[1]}" for c in MAPPING_CASES])
def test_mapping_context(fml, fit, extra):
    context = {"sq": sq, "k": 2.5, "lspline": lspline}
    kw = {"context": {**context, **({k: v for k, v in extra.items() if k == "log"})}}
    if fit != "feols":
        kw.update(iwls_tol=1e-12, iwls_maxiter=200)
    if "family" in extra:
        kw["family"] = extra["family"]
    ours, theirs = _both(fml, fit, **kw)
    compare(ours, theirs, rtol=1e-6)


def test_int_context_is_the_callers_frame():
    shift = 0.5  # noqa: F841 (a local of the calling frame)
    fml = "y ~ sq(x) + I(x2 * SCALE + shift) | f"
    ours = fereg.feols(fml, DATA, context=0)
    theirs = pf.feols(fml, DATA, context=0, demeaner=TIGHT)
    compare(ours, theirs, rtol=1e-8)
    assert list(ours.coef().index) == ["sq(x)", "I(x2 * SCALE + shift)"]


def _outer(module, depth):
    """`cube` is a local of this frame; `inner` calls feols with context=depth."""

    def cube(v):
        return v**3

    def inner():
        return module.feols("y ~ cube(x) | f", DATA, context=depth)

    return inner()


def test_int_context_depth():
    # depth 1: the caller of the function that calls feols
    ours, theirs = _outer(fereg, 1), _outer(pf, 1)
    compare(ours, theirs, rtol=1e-8)
    assert list(ours.coef().index) == ["cube(x)"]
    # depth 0: inner()'s own frame, where `cube` is not defined
    with pytest.raises(formulaic_errors.FactorEvaluationError) as theirs_err:
        _outer(pf, 0)
    with pytest.raises(formulaic_errors.FactorEvaluationError, match=re.escape(str(theirs_err.value))):
        _outer(fereg, 0)


def test_default_context_has_no_caller_names():
    """context=None (default): the caller's names are not visible, with pyfixest's error."""
    for fml in ("y ~ sq(x) | f", "y ~ I(x * SCALE) | f", "y ~ x + nope | f"):
        with pytest.raises(formulaic_errors.FactorEvaluationError) as theirs:
            pf.feols(fml, DATA)
        with pytest.raises(formulaic_errors.FactorEvaluationError, match=re.escape(str(theirs.value))):
            fereg.feols(fml, DATA)
    # a mapping without the name: the same error
    with pytest.raises(formulaic_errors.FactorEvaluationError) as theirs:
        pf.feols("y ~ sq(x) | f", DATA, context={"other": 1})
    with pytest.raises(formulaic_errors.FactorEvaluationError, match=re.escape(str(theirs.value))):
        fereg.feols("y ~ sq(x) | f", DATA, context={"other": 1})


def test_native_terms_unchanged_by_context():
    """np and numbers from the caller's frame keep the native evaluation (same results)."""
    plain = fereg.feols("y ~ np.log(x2) + I(x * 2.0) | f", DATA)
    k = 2.0  # noqa: F841
    with_ctx = fereg.feols("y ~ np.log(x2) + I(x * k) | f", DATA, context=0)
    np.testing.assert_array_equal(plain._beta_hat, with_ctx._beta_hat)


def test_predict_newdata_uses_context():
    ours, theirs = _both("y ~ sq(x) + x2 | f", context={"sq": sq})
    new = DATA.sample(50, random_state=1).reset_index(drop=True)
    np.testing.assert_allclose(
        ours.predict(newdata=new), theirs.predict(newdata=new, atol=1e-12, btol=1e-12), rtol=1e-8
    )
