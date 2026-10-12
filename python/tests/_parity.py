"""Shared data and comparison helpers for the pyfixest parity tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pf = pytest.importorskip("pyfixest")

TIGHT = pf.MapDemeaner(fixef_tol=1e-13, fixef_maxiter=1_000_000)


def make_data(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Panel-like data with three crossed FEs, a nested cluster, weights, NaNs and singletons."""
    rng = np.random.default_rng(seed)
    f1 = rng.integers(0, 80, n)
    f2 = rng.integers(0, 30, n)
    f3 = rng.integers(0, 6, n)
    h = rng.integers(0, 5, n)
    x1 = rng.normal(size=n) + 0.1 * f1 / 80
    x2 = rng.normal(size=n) + 0.2 * (f2 % 3)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u = rng.normal(size=n)
    alpha = rng.normal(size=80)[f1]
    gamma = rng.normal(size=30)[f2]
    e1 = 0.6 * z1 + 0.4 * z2 + 0.5 * u + 0.3 * alpha + rng.normal(size=n)
    e2 = 0.3 * z1 - 0.5 * z2 + 0.2 * x1 + rng.normal(size=n)
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + 0.2 * (h == 2) + alpha + gamma + 0.1 * f3 + u + rng.normal(size=n)
    y_iv = 0.7 * e1 - 0.4 * e2 + 0.5 * x1 + alpha + gamma + u + rng.normal(size=n)
    eta = 0.3 + 0.3 * x1 - 0.2 * x2 + 0.3 * alpha + 0.2 * gamma
    y_pois = rng.poisson(np.exp(eta)).astype(float)
    y_bin = (rng.uniform(size=n) < 1 / (1 + np.exp(-(eta - 0.5)))).astype(float)
    df = pd.DataFrame(
        {
            "y": y,
            "y2": y + rng.normal(size=n),
            "y_iv": y_iv,
            "y_pois": y_pois,
            "y_bin": y_bin,
            "x1": x1,
            "x2": x2,
            "x3": rng.normal(size=n),
            "xc": x1 + 2.0 * x2,  # collinear with x1, x2
            "z1": z1,
            "z2": z2,
            "e1": e1,
            "e2": e2,
            "f1": f1,
            "f2": f2.astype(float),
            "f3": f3,
            "s3": np.array(["a", "b", "c", "d", "e", "f"])[f3],
            "h": h,
            "cl": f1 // 8,  # f1 is nested in cl
            "cl2": rng.integers(0, 40, n),
            "w": rng.uniform(0.5, 2.0, n),
            "fw": rng.integers(1, 4, n).astype(float),
            "off": 0.1 * rng.normal(size=n),
            "g": rng.integers(0, 3, n),
        }
    )
    # Missing values in the outcome, a regressor and an FE.
    df.loc[[3, 50], "y"] = np.nan
    df.loc[[10, 77], "x1"] = np.nan
    df.loc[[20, 21], "f2"] = np.nan
    # Singletons: unique f1 levels.
    for i, row in enumerate(range(n - 6, n)):
        df.loc[row, "f1"] = 1000 + i
    return df


def compare(ours, theirs, rtol: float = 1e-8, se_rtol: float | None = None) -> None:
    """Coefficient names, N, df and coefficients / standard errors of two fitted models."""
    se_rtol = rtol if se_rtol is None else se_rtol
    assert list(ours._coefnames) == [str(c) for c in theirs._coefnames]
    assert ours._N == theirs._N
    assert ours._df_k == theirs._df_k
    assert ours._df_t == theirs._df_t
    np.testing.assert_allclose(ours._beta_hat, np.asarray(theirs._beta_hat), rtol=rtol, atol=0)
    np.testing.assert_allclose(ours._se, np.asarray(theirs._se), rtol=se_rtol, atol=0)


def compare_multi(ours, theirs, rtol: float = 1e-8) -> None:
    assert list(ours.all_fitted_models.keys()) == list(theirs.all_fitted_models.keys())
    for a, b in zip(ours.to_list(), theirs.to_list()):
        compare(a, b, rtol=rtol)


def assert_same_summary(ours, theirs, capsys) -> None:
    """``fereg.summary`` prints what ``pyfixest.summary`` prints (rounded to 3 digits)."""
    pf.summary(theirs)
    expected = capsys.readouterr().out
    import fereg

    fereg.summary(ours)
    got = capsys.readouterr().out
    # pyfixest prints a spurious "sample: None = all" line for unsplit models.
    expected = "\n".join(line for line in expected.splitlines() if line != "sample: None = all")
    assert got.rstrip("\n") == expected.rstrip("\n")
