"""``feols`` / ``fepois`` / ``feglm`` with pyfixest 0.60.0 signatures and defaults."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from ._binding import fit_design
from ._vcov import parse_vcov
from .results import (
    FitContext,
    Fegaussian,
    Feiv,
    Felogit,
    Feols,
    Fepois,
    FixestMulti,
    _check_frame,
    _to_pandas,
)

__all__ = ["feols", "fepois", "feglm", "ssc", "estimate"]

DEFAULT_FIXEF_TOL = 1e-8
DEFAULT_FIXEF_MAXITER = 16_000


class _AllSample:
    """The full sample in ``fsplit`` mode (pyfixest prints it as ``all``)."""

    def __repr__(self) -> str:
        return "all"

    __str__ = __repr__


_ALL = _AllSample()


def ssc(
    k_adj: bool = True,
    k_fixef: str = "nonnested",
    G_adj: bool = True,
    G_df: str = "min",
    k_exact: bool = False,
    t_df: str | float = "min",
) -> dict:
    """Small-sample correction (pyfixest/fixest ``ssc``); returns a plain dict."""
    if k_fixef not in ("none", "full", "nonnested"):
        raise ValueError("k_fixef must be 'none', 'full' or 'nonnested'.")
    if G_df not in ("min", "conventional"):
        raise ValueError("G_df must be 'min' or 'conventional'.")
    if not (t_df in ("min", "conventional") or isinstance(t_df, (int, float))):
        raise ValueError("t_df must be 'min', 'conventional' or a number.")
    for name, v in (("k_adj", k_adj), ("G_adj", G_adj), ("k_exact", k_exact)):
        if not isinstance(v, bool):
            raise TypeError(f"{name} must be a bool.")
    return {"k_adj": k_adj, "k_fixef": k_fixef, "G_adj": G_adj, "G_df": G_df, "k_exact": k_exact, "t_df": t_df}


def estimate(ctx: FitContext) -> Feols:
    """Fits one model described by ``ctx`` and wraps it in its pyfixest result class."""
    d = ctx.design
    if ctx.model == "logit":
        _check_binary(d)
    out = fit_design(
        d,
        model=ctx.model,
        vcov=ctx.vcov.kind,
        ssc=ctx.ssc,
        weights_type=ctx.weights_type,
        fixef_rm=ctx.fixef_rm,
        collin_tol=ctx.collin_tol,
        fixef_tol=ctx.fixef_tol,
        fixef_maxiter=ctx.fixef_maxiter,
        glm_tol=ctx.glm_tol,
        glm_maxiter=ctx.glm_maxiter,
        relu=ctx.relu,
        keep_resid=not ctx.lean,
        keep_fixef=not ctx.lean,
        keep_state=not ctx.lean,
        threads=ctx.threads,
    )
    if d.endog:
        cls = Feiv
    else:
        cls = {"feols": Feols, "fepois": Fepois, "feglm-logit": Felogit, "feglm-gaussian": Fegaussian}[ctx.method]
    m = cls(ctx, out)
    dropped = out.dropped
    if dropped["singletons"]:
        warnings.warn(f"{dropped['singletons']} singleton fixed effect(s) dropped from the model.")
    if m._collin_vars:
        names = m._collin_vars
        shown = "\n    " + "\n    ".join(names[:5]) + "\n    ..." if len(names) > 5 else str(names)
        warnings.warn(
            f"\n            {len(names)} variables dropped due to multicollinearity.\n"
            f"            The following variables are dropped: {shown}.\n            "
        )
    for note in out.notes:
        warnings.warn(note)
    if ctx.method == "feols" and not d.endog and m._k:
        m.get_performance()
        m.wald_test()
    return m


def _check_binary(design: Any) -> None:
    y = np.asarray(design.y)
    ok = np.isnan(y) | (y == 0) | (y == 1)
    if design.drop is not None:
        ok |= np.asarray(design.drop, dtype=bool)
    if not ok.all() or not (y == 0).any() or not (y == 1).any():
        raise ValueError("The dependent variable must have two unique values: 0 and 1.")


def _check_common(fixef_rm: str, collin_tol: float, weights_type: str, data: pd.DataFrame, split, fsplit) -> None:
    if fixef_rm not in ("singleton", "none"):
        raise ValueError("fixef_rm must be 'singleton' or 'none'.")
    if not isinstance(collin_tol, float) or not 0 < collin_tol < 1:
        raise ValueError("collin_tol must be a float in (0, 1).")
    if weights_type not in ("aweights", "fweights"):
        raise ValueError("weights_type must be 'aweights' or 'fweights'.")
    if split is not None and fsplit is not None and split != fsplit:
        raise ValueError("split and fsplit cannot both be given with different variables.")
    for v in (split, fsplit):
        if v is not None and v not in data.columns:
            raise KeyError(f"Split variable '{v}' not found in data.")


def _run(
    fml: str,
    data,
    *,
    model: str,
    method: str,
    vcov,
    vcov_kwargs,
    weights: str | None,
    weights_type: str,
    offset: str | None,
    ssc: dict | None,
    fixef_rm: str,
    collin_tol: float,
    fixef_tol: float | None,
    fixef_maxiter: int | None,
    glm_tol: float = 1e-8,
    glm_maxiter: int = 25,
    relu: bool = False,
    drop_intercept: bool,
    lean: bool,
    split: str | None,
    fsplit: str | None,
    nthreads: int | None,
    use_compression: bool = False,
) -> Feols | FixestMulti:
    from .design import build_design
    from .formula import parse

    if vcov_kwargs:
        raise NotImplementedError("vcov_kwargs (HAC options) are not supported by fereg.")
    if use_compression:
        raise NotImplementedError("use_compression is not supported by fereg (the solver is exact).")
    # pandas and polars frames go to the design builder as they are (zero-copy columns);
    # sample splits need row filtering, done in pandas.
    data = _to_pandas(data) if (split is not None or fsplit is not None) else _check_frame(data)
    _check_common(fixef_rm if model == "ols" else "singleton", collin_tol, weights_type, data, split, fsplit)
    spec = parse_vcov(vcov)
    formulas = parse(fml)
    if method != "feols" and any(f.endog for f in formulas):
        raise NotImplementedError("IV estimation is only supported for feols().")

    split_var = split if split is not None else fsplit
    samples: list[Any] = []
    if split_var is None or fsplit is not None:
        samples.append(_ALL)
    if split_var is not None:
        samples.extend(data[split_var].dropna().drop_duplicates().sort_values().tolist())

    base = FitContext(
        formula=None,
        design=None,
        data=data,
        model=model,
        method=method,
        vcov=spec,
        ssc=dict(ssc) if ssc else {},
        weights=weights,
        weights_type=weights_type,
        offset=offset,
        fixef_rm=fixef_rm,
        collin_tol=collin_tol,
        fixef_tol=DEFAULT_FIXEF_TOL if fixef_tol is None else float(fixef_tol),
        fixef_maxiter=DEFAULT_FIXEF_MAXITER if fixef_maxiter is None else int(fixef_maxiter),
        glm_tol=glm_tol,
        glm_maxiter=glm_maxiter,
        relu=relu,
        lean=lean,
        threads=0 if nthreads is None else int(nthreads),
        drop_intercept=drop_intercept,
        split_var=split_var,
    )

    models: dict[str, Feols] = {}
    for value in samples:
        if split_var is None:
            sub = data
        elif value is _ALL:
            sub = data.loc[data[split_var].notnull()].reset_index(drop=True)
        else:
            sub = data.loc[data[split_var] == value].reset_index(drop=True)
        for f in formulas:
            design = build_design(
                f,
                sub,
                weights=weights,
                offset=offset,
                cluster=list(spec.cluster),
                drop_intercept=drop_intercept,
            )
            ctx = replace(base, formula=f, design=design, data=sub, split_value=value)
            m = estimate(ctx)
            models[m._model_name] = m

    if len(models) == 1 and split_var is None:
        return next(iter(models.values()))
    return FixestMulti(models)


def feols(
    fml: str,
    data,
    vcov: str | dict[str, str] | None = None,
    vcov_kwargs: dict | None = None,
    weights: str | None = None,
    ssc: dict | None = None,
    fixef_rm: str = "singleton",
    collin_tol: float = 1e-9,
    drop_intercept: bool = False,
    copy_data: bool = True,
    store_data: bool = True,
    lean: bool = False,
    weights_type: str = "aweights",
    solver: str = "scipy.linalg.solve",
    demeaner=None,
    demeaner_backend=None,
    fixef_tol: float | None = DEFAULT_FIXEF_TOL,
    fixef_maxiter: int | None = DEFAULT_FIXEF_MAXITER,
    use_compression: bool = False,
    reps: int = 100,
    context=None,
    seed: int | None = None,
    split: str | None = None,
    fsplit: str | None = None,
    nthreads: int | None = None,
) -> Feols | Feiv | FixestMulti:
    """OLS and IV/2SLS with high-dimensional fixed effects (pyfixest ``feols``).

    ``copy_data``, ``store_data``, ``solver``, ``demeaner``, ``demeaner_backend``, ``reps``,
    ``context`` and ``seed`` are accepted for compatibility and ignored: the data is never
    copied and the fixed effects are always absorbed by fereg's exact solver
    (``fixef_tol``/``fixef_maxiter`` are its tolerance and iteration cap). ``nthreads=None``
    uses every CPU.
    """
    return _run(
        fml,
        data,
        model="ols",
        method="feols",
        vcov=vcov,
        vcov_kwargs=vcov_kwargs,
        weights=weights,
        weights_type=weights_type,
        offset=None,
        ssc=ssc,
        fixef_rm=fixef_rm,
        collin_tol=collin_tol,
        fixef_tol=fixef_tol,
        fixef_maxiter=fixef_maxiter,
        drop_intercept=drop_intercept,
        lean=lean,
        split=split,
        fsplit=fsplit,
        nthreads=nthreads,
        use_compression=use_compression,
    )


def _separation(separation_check: list[str] | None, fixef_rm: str) -> tuple[str, bool]:
    checks = ["fe"] if separation_check is None else list(separation_check)
    bad = [c for c in checks if c not in ("fe", "ir")]
    if bad:
        raise ValueError(f"separation_check must be a list of 'fe' and/or 'ir', got {bad}.")
    if fixef_rm not in ("singleton", "none"):
        raise ValueError("fixef_rm must be 'singleton' or 'none'.")
    core_rm = "perfect" if "fe" in checks else fixef_rm
    return core_rm, "ir" in checks


def fepois(
    fml: str,
    data,
    vcov: str | dict[str, str] | None = None,
    vcov_kwargs: dict | None = None,
    weights: str | None = None,
    weights_type: str = "aweights",
    offset: str | None = None,
    ssc: dict | None = None,
    fixef_rm: str = "singleton",
    iwls_tol: float = 1e-08,
    iwls_maxiter: int = 25,
    collin_tol: float = 1e-09,
    separation_check: list[str] | None = None,
    solver: str = "scipy.linalg.solve",
    demeaner=None,
    demeaner_backend=None,
    fixef_tol: float | None = None,
    fixef_maxiter: int | None = None,
    drop_intercept: bool = False,
    copy_data: bool = True,
    store_data: bool = True,
    lean: bool = False,
    context=None,
    split: str | None = None,
    fsplit: str | None = None,
    nthreads: int | None = None,
) -> Fepois | FixestMulti:
    """Poisson regression with high-dimensional fixed effects (pyfixest ``fepois``).

    ``separation_check=None`` means ``["fe"]`` (drop fixed-effect groups whose outcome is all
    zero); ``"ir"`` adds the ReLU separation check.
    """
    core_rm, relu = _separation(separation_check, fixef_rm)
    return _run(
        fml,
        data,
        model="poisson",
        method="fepois",
        vcov=vcov,
        vcov_kwargs=vcov_kwargs,
        weights=weights,
        weights_type=weights_type,
        offset=offset,
        ssc=ssc,
        fixef_rm=core_rm,
        collin_tol=collin_tol,
        fixef_tol=fixef_tol,
        fixef_maxiter=fixef_maxiter,
        glm_tol=iwls_tol,
        glm_maxiter=iwls_maxiter,
        relu=relu,
        drop_intercept=drop_intercept,
        lean=lean,
        split=split,
        fsplit=fsplit,
        nthreads=nthreads,
    )


def feglm(
    fml: str,
    data,
    family: str,
    vcov: str | dict[str, str] | None = None,
    vcov_kwargs: dict | None = None,
    ssc: dict | None = None,
    fixef_rm: str = "singleton",
    iwls_tol: float = 1e-08,
    iwls_maxiter: int = 25,
    collin_tol: float = 1e-09,
    separation_check: list[str] | None = None,
    solver: str = "scipy.linalg.solve",
    demeaner=None,
    demeaner_backend=None,
    fixef_tol: float | None = None,
    fixef_maxiter: int | None = None,
    drop_intercept: bool = False,
    copy_data: bool = True,
    store_data: bool = True,
    lean: bool = False,
    context=None,
    split: str | None = None,
    fsplit: str | None = None,
    accelerate: bool = True,
    *,
    weights: str | None = None,
    weights_type: str = "aweights",
    offset: str | None = None,
    nthreads: int | None = None,
) -> Felogit | Fepois | Fegaussian | FixestMulti:
    """GLM with high-dimensional fixed effects (pyfixest ``feglm``).

    ``family`` is ``"logit"``, ``"poisson"`` (same as :func:`fepois`) or ``"gaussian"`` (OLS
    estimates with normal-based inference). ``weights``, ``weights_type`` and ``offset`` are
    fereg extensions (pyfixest's ``feglm`` has neither).
    """
    if family == "poisson":
        return fepois(
            fml,
            data,
            vcov=vcov,
            vcov_kwargs=vcov_kwargs,
            weights=weights,
            weights_type=weights_type,
            offset=offset,
            ssc=ssc,
            fixef_rm=fixef_rm,
            iwls_tol=iwls_tol,
            iwls_maxiter=iwls_maxiter,
            collin_tol=collin_tol,
            separation_check=separation_check,
            fixef_tol=fixef_tol,
            fixef_maxiter=fixef_maxiter,
            drop_intercept=drop_intercept,
            lean=lean,
            split=split,
            fsplit=fsplit,
            nthreads=nthreads,
        )
    if family == "probit":
        raise NotImplementedError("probit is not supported by fereg.")
    if family not in ("logit", "gaussian"):
        raise ValueError(f"family must be 'logit', 'poisson' or 'gaussian', got '{family}'.")
    if family == "gaussian":
        if offset is not None:
            raise NotImplementedError("offset is not supported with family='gaussian'.")
        core_rm, relu, model = fixef_rm, False, "ols"
        _separation(separation_check, fixef_rm)
    else:
        (core_rm, relu), model = _separation(separation_check, fixef_rm), "logit"
    return _run(
        fml,
        data,
        model=model,
        method=f"feglm-{family}",
        vcov=vcov,
        vcov_kwargs=vcov_kwargs,
        weights=weights,
        weights_type=weights_type,
        offset=offset,
        ssc=ssc,
        fixef_rm=core_rm,
        collin_tol=collin_tol,
        fixef_tol=fixef_tol,
        fixef_maxiter=fixef_maxiter,
        glm_tol=iwls_tol,
        glm_maxiter=iwls_maxiter,
        relu=relu,
        drop_intercept=drop_intercept,
        lean=lean,
        split=split,
        fsplit=fsplit,
        nthreads=nthreads,
    )
