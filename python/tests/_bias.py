"""Fixtures and helpers for the logit bias-correction / APE tests.

The reference numbers come from alpaca 0.3.5 (exact control) and plain-R evaluations of its
formulas, plus jackknife refits with fixest 0.14.2: ``data/gen_bias_fixtures.R`` writes
``data/bias_alpaca.json`` and the data sets ``data/bias_<name>.csv``.
"""

from __future__ import annotations

import functools
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import fereg

DATA_DIR = pathlib.Path(__file__).parent / "data"

# Tolerances (fereg design 3.3): everything evaluated at the uncorrected fit or before the
# fixed-effect re-solve (coefficients, bias terms, uncorrected APEs) to 1e-8; quantities at the
# corrected point (after the exact re-solve) to 1e-6. Jackknife combinations amplify the subpanel
# fits' convergence error by up to N + T, hence 1e-6 for them too.
RTOL = 1e-8
RTOL_POINT = 1e-6
# Uncorrected variances use the pyfixest flavor's final IRLS weights (from the start of the last
# iteration, like pyfixest), which differ from the converged ones at ~sqrt(iwls_tol) relative.
RTOL_UNCORR_VCOV = 1e-4

FIT_KW = {"iwls_tol": 1e-12, "iwls_maxiter": 200, "fixef_tol": 1e-12}
# alpaca's variances: vcov(type = "hessian") = iid with ssc(k_adj = FALSE); sandwich = hetero and
# clustered = CRV1, both with ssc(k_adj = FALSE, G_adj = FALSE).
SSC = fereg.ssc(k_adj=False, G_adj=False)


@functools.cache
def fixtures() -> dict:
    with open(DATA_DIR / "bias_alpaca.json") as f:
        return json.load(f)


@functools.cache
def data(name: str) -> pd.DataFrame:
    # round_trip: exact parsing (the values are multiples of 2^-12, exact in any parser anyway)
    return pd.read_csv(DATA_DIR / f"bias_{name}.csv", float_precision="round_trip")


def models() -> list[dict]:
    return fixtures()["models"]


def model_case(case_id: str) -> dict:
    return next(m for m in models() if m["id"] == case_id)


def jackknife_cases() -> list[dict]:
    return fixtures()["jackknife"]


def fit(fml: str, name: str, vcov="iid", nthreads: int | None = None, **kw) -> fereg.Felogit:
    with warnings.catch_warnings():
        # constant-outcome groups (drops data) warn about separation
        warnings.simplefilter("ignore")
        return fereg.feglm(fml, data(name), family="logit", vcov=vcov, ssc=SSC, nthreads=nthreads, **{**FIT_KW, **kw})


@functools.cache
def fit_case(case_id: str) -> fereg.Felogit:
    c = model_case(case_id)
    return fit(c["fml"], c["data"], weights=c.get("weights"), offset=c.get("offset"))


NO_NPOP = "has no effect without n_pop"


def apes_for(m, e: dict, corrected: bool = False, **kw) -> fereg.APEs:
    """``m.get_apes`` for fixture entry ``e``. Without n_pop, ``sampling_fe="unrestricted"`` and
    ``weak_exo=True`` have no effect (both terms are multiplied by 0): they are errors, and the
    expected numbers are those of the defaults. A corrected model takes weak_exo from its
    correction."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weak = None if corrected else e["weak_exo"]
        if e["n_pop"] is None:
            if e["sampling_fe"] == "unrestricted" or weak:
                with pytest.raises(ValueError, match=NO_NPOP):
                    m.get_apes(sampling_fe=e["sampling_fe"], weak_exo=weak or None, **kw)
            return m.get_apes(**kw)
        return m.get_apes(n_pop=e["n_pop"], sampling_fe=e["sampling_fe"], weak_exo=weak, **kw)


def assert_close(actual, expected, rtol: float, what: str = "") -> None:
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    assert a.shape == e.shape, f"{what}: shape {a.shape} != {e.shape}"
    # relative to the largest entry for matrices (small off-diagonal covariances)
    scale = np.maximum(np.abs(e), np.max(np.abs(e)) * 1e-3 if e.ndim == 2 else np.abs(e))
    rel = np.max(np.abs(a - e) / np.where(scale > 0, scale, 1.0))
    assert rel <= rtol, f"{what}: max relative difference {rel:.3g} > {rtol:g}"


def case_id(c: dict) -> str:
    return c["id"]


def correction_id(c: dict, r: dict) -> str:
    return f"{c['id']}-L{r['L']}-{r['bias']}"


def ape_id(e: dict) -> str:
    npop = "npop" if e["n_pop"] is not None else "nonpop"
    return f"{npop}-{e['sampling_fe']}-{'weak' if e['weak_exo'] else 'strict'}"
