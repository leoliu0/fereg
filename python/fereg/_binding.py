"""Thin layer over the compiled ``fereg._core`` module.

Every array handed to the core is borrowed zero-copy; :func:`np.ascontiguousarray` is a no-op
for the contiguous float64 / int columns the design builder produces.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _core

__all__ = [
    "fit_design",
    "factor_arg",
    "core_ssc",
    "pnorm",
    "qnorm",
    "pt",
    "qt",
    "pf_upper",
    "pchisq_upper",
]

_FACTOR_DTYPES = {"f64": np.float64, "i32": np.int32, "i64": np.int64, "codes": np.int32}
_SSC_KEYS = ("k_adj", "k_fixef", "k_exact", "G_adj", "G_df", "t_df")


def factor_arg(fc: Any) -> tuple[str, np.ndarray, int, int]:
    """``(kind, values, nlevels, base)`` tuple of a :class:`fereg.design.FactorColumn`."""
    kind = fc.kind
    dtype = _FACTOR_DTYPES.get(kind)
    if dtype is None:
        raise ValueError(f"unknown factor column kind '{kind}'")
    return kind, np.ascontiguousarray(fc.values, dtype=dtype), int(fc.nlevels), int(fc.base)


def core_ssc(ssc: dict | None) -> dict:
    """The keys of a pyfixest/fixest ``ssc()`` dict the core understands."""
    if not ssc:
        return {}
    return {k: ssc[k] for k in _SSC_KEYS if k in ssc}


def _f64(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def fit_design(
    design: Any,
    *,
    model: str,
    vcov: Any,
    ssc: dict | None,
    weights_type: str,
    fixef_rm: str,
    collin_tol: float,
    fixef_tol: float,
    fixef_maxiter: int,
    glm_tol: float = 1e-8,
    glm_maxiter: int = 25,
    relu: bool = False,
    keep_resid: bool = True,
    keep_fixef: bool = True,
    keep_state: bool = True,
    threads: int = 0,
    drop: np.ndarray | None = None,
) -> Any:
    """Runs ``fereg_core::fit`` on a :class:`fereg.design.Design` (pyfixest conventions);
    ``vcov`` is a :class:`fereg._vcov.VcovSpec`."""
    if drop is None:
        drop = design.drop
    elif design.drop is not None:
        drop = drop | design.drop
    return _core.fit(
        _f64(design.y),
        [_f64(c) for c in design.x],
        [_f64(c) for c in design.endog],
        [_f64(c) for c in design.inst],
        [factor_arg(f) for f in design.factors],
        [f.name for f in design.factors],
        [(list(f), bool(i), list(s)) for f, i, s in design.fe],
        [_f64(c) for c in design.slopes],
        None if design.weights is None else _f64(design.weights),
        None if design.offset is None else _f64(design.offset),
        [list(c) for c in design.cluster],
        None if drop is None else np.ascontiguousarray(drop, dtype=np.bool_),
        time=design.time,
        panel=design.panel,
        lat=None if design.lat is None else _f64(design.lat),
        lon=None if design.lon is None else _f64(design.lon),
        model=model,
        flavor="pyfixest",
        intercept=bool(design.intercept),
        vcov=vcov.kind,
        **vcov.core_options(),
        ssc=core_ssc(ssc),
        weights_type=weights_type,
        fixef_rm=fixef_rm,
        relu=relu,
        collin_tol=float(collin_tol),
        fixef_tol=float(fixef_tol),
        fixef_maxiter=int(fixef_maxiter),
        glm_tol=float(glm_tol),
        glm_maxiter=int(glm_maxiter),
        keep_resid=keep_resid,
        keep_fixef=keep_fixef,
        fixef_norm="minnorm",
        keep_state=keep_state,
        threads=int(threads),
    )


def _vectorized(f, x, *params):
    out = f(np.asarray(x, dtype=np.float64), *(float(p) for p in params))
    return float(out) if out.ndim == 0 else out


def pnorm(x):
    """Standard normal CDF."""
    return _vectorized(_core.pnorm, x)


def qnorm(p):
    """Standard normal quantile."""
    return _vectorized(_core.qnorm, p)


def pt(x, df):
    """Student t CDF."""
    return _vectorized(_core.pt, x, df)


def qt(p, df):
    """Student t quantile."""
    return _vectorized(_core.qt, p, df)


def pf_upper(x, df1, df2):
    """F distribution upper tail P(F > x)."""
    return _vectorized(_core.pf_upper, x, df1, df2)


def pchisq_upper(x, df):
    """Chi-squared upper tail P(X > x)."""
    return _vectorized(_core.pchisq_upper, x, df)
