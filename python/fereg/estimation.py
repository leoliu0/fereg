"""``feols`` / ``fepois`` / ``feglm`` with pyfixest 0.60.0 signatures and defaults."""

from __future__ import annotations

import sys
import warnings
from collections import ChainMap
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from ._binding import fit_design
from ._vcov import VcovSpec, VcovTypeNotSupportedError, parse_vcov
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

__all__ = ["feols", "fepois", "feglm", "ssc", "estimate", "fit_context"]

DEFAULT_FIXEF_TOL = 1e-8
DEFAULT_FIXEF_MAXITER = 16_000


class _AllSample:
    """The full sample in ``fsplit`` mode (pyfixest prints it as ``all``)."""

    def __repr__(self) -> str:
        return "all"

    __str__ = __repr__


_ALL = _AllSample()


def _capture_context(context: int | Mapping[str, Any] | None) -> Mapping[str, Any]:
    """pyfixest's ``context`` as its estimation functions capture it (``capture_context``):
    None gives no extra names; an int is a frame offset from the caller of the estimation
    function (0 = the caller's locals, then its globals); a mapping is used as given."""
    if context is None:
        return {}
    if isinstance(context, int):
        # frame 0: this function, 1: feols / fepois / feglm, 2: their caller
        frame = sys._getframe(context + 2)
        return ChainMap(frame.f_locals, frame.f_globals)
    return context


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


def fit_context(ctx: FitContext, *, vcov: VcovSpec | None = None, design: Any = None, lean: bool | None = None) -> Any:
    """Runs the core on ``ctx`` (optionally with another variance request / design / ``lean``)."""
    d = ctx.design if design is None else design
    spec = ctx.vcov if vcov is None else vcov
    lean = ctx.lean if lean is None else lean
    if d.endog and spec.kind in ("hc2", "hc3"):
        # pyfixest `_deparse_vcov_input`.
        raise VcovTypeNotSupportedError("HC2 and HC3 inference types are not supported for IV regressions.")
    return fit_design(
        d,
        model=ctx.model,
        vcov=spec,
        ssc=ctx.ssc,
        weights_type=ctx.weights_type,
        fixef_rm=ctx.fixef_rm,
        collin_tol=ctx.collin_tol,
        fixef_tol=ctx.fixef_tol,
        fixef_maxiter=ctx.fixef_maxiter,
        glm_tol=ctx.glm_tol,
        glm_maxiter=ctx.glm_maxiter,
        relu=ctx.relu,
        gaussian_glm=ctx.method == "feglm-gaussian",
        keep_resid=not lean,
        # FE values are computed on demand from the state (fixef(), predict(newdata)).
        keep_fixef=False,
        keep_state=not lean,
        threads=ctx.threads,
    )


def estimate(ctx: FitContext) -> Feols:
    """Fits one model described by ``ctx`` and wraps it in its pyfixest result class."""
    d = ctx.design
    if ctx.model == "logit":
        _check_binary(d)
    out = fit_context(ctx)
    if d.endog:
        cls = Feiv
    else:
        cls = {"feols": Feols, "fepois": Fepois, "feglm-logit": Felogit, "feglm-gaussian": Fegaussian}[ctx.method]
    m = cls(ctx, out)
    dropped = out.dropped
    if dropped["singletons"]:
        warnings.warn(f"{dropped['singletons']} singleton fixed effect(s) dropped from the model.")
    # pyfixest `_check_for_separation` (fepois_.py), run by fepois and feglm: rows dropped by
    # the "fe" check (core `perfect_fit`, pyfixest flavor) and the "ir" check (`separated`).
    n_separated = dropped["perfect_fit"] + dropped["separated"]
    if n_separated:
        warnings.warn(f"{n_separated} observations removed because of separation.")
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
    context: Mapping[str, Any] | None = None,
) -> Feols | FixestMulti:
    from .design import build_design
    from .formula import parse

    if use_compression:
        raise NotImplementedError("use_compression is not supported by fereg (the solver is exact).")
    # pandas and polars frames go to the design builder as they are (zero-copy columns);
    # sample splits need row filtering, done in pandas.
    data = _to_pandas(data) if (split is not None or fsplit is not None) else _check_frame(data)
    _check_common(fixef_rm if method == "feols" else "singleton", collin_tol, weights_type, data, split, fsplit)
    spec = parse_vcov(vcov, vcov_kwargs, columns=list(data.columns))
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
        context={} if context is None else context,
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
                vcov_columns=spec.columns(),
                drop_intercept=drop_intercept,
                context=base.context,
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
    context: int | Mapping[str, Any] | None = None,
    seed: int | None = None,
    split: str | None = None,
    fsplit: str | None = None,
    nthreads: int | None = None,
) -> Feols | Feiv | FixestMulti:
    """OLS and IV/2SLS with high-dimensional fixed effects (pyfixest ``feols``).

    ``vcov`` (default ``"iid"``), as pyfixest:

    - ``"iid"``; ``"hetero"`` / ``"HC1"``; ``"HC2"``, ``"HC3"`` (leverage-corrected, with
      pyfixest's N / (N - df_k)); ``{"CRV1": "a"}``, ``{"CRV1": "a+b"}`` (multiway, up to four
      dimensions; ``"a^b"`` clusters by the interaction); ``{"CRV3": ...}`` (leave-one-cluster-out
      jackknife centered at the estimates, CRV1's small-sample factors, same dimensions);
    - ``"NW"`` (Newey-West) and ``"DK"`` (Driscoll-Kraay) with ``vcov_kwargs={"time_id": ...,
      "panel_id": ..., "lag": ...}``: ``time_id`` (numeric or date) is required; with
      ``panel_id``, Newey-West is the panel version (lags within the unit after sorting by time);
      without it, a time series that needs ``lag``. ``lag`` defaults to floor(T^0.25), T the
      number of distinct periods; the correction is (N - 1) / (N - df_k) * T / (T - 1), and t
      tests have T - 1 degrees of freedom;
    - ``"conley"`` (fereg extension: fixest's Conley spatial HAC) with ``vcov_kwargs={"lat": ...,
      "lon": ..., "cutoff": ..., "pixel": ..., "distance": ...}``: ``lat`` / ``lon`` are columns in
      degrees; ``cutoff`` is a number of km or a string such as ``"100km"`` or ``"60mi"`` (default:
      fixest's deduced cutoff); ``pixel`` (default 0 = none) aggregates the coordinates into
      pixels of that size, in the unit of ``cutoff``; ``distance`` is ``"triangular"``
      (default) or ``"spherical"``. Correction N / (N - df_k), N - df_k degrees of freedom.

    Extensions beyond pyfixest (which raises for them): HC2/HC3 with fixed effects (exact leverage
    of the full model, fixed effects included), CRV3 with IV, Newey-West / Driscoll-Kraay for IV
    at estimation, and Driscoll-Kraay without ``panel_id``. HC2/HC3 with IV raise
    :class:`fereg.VcovTypeNotSupportedError` as in pyfixest; ``"nid"`` (quantile regression)
    raises ``NotImplementedError``. Missing values in the cluster, time, panel or coordinate
    columns on estimation rows are an error.

    ``context`` (pyfixest semantics) makes more names available to the formula, e.g. user
    functions and objects: an int captures the variables of the calling frame (``0`` = the
    caller of ``feols``, ``1`` its caller, ...), a mapping gives them explicitly, and ``None``
    (default) adds none. Names resolve in the data first, then in ``context``, then among the
    built-in transforms (``log``, ``np``, ``I``, ``C``, ``poly``, ...), which ``context`` can
    override; ``predict(newdata=...)`` uses the same context. Terms that need a context object
    are materialized by formulaic.

    ``copy_data``, ``store_data``, ``solver``, ``demeaner``, ``demeaner_backend``, ``reps`` and
    ``seed`` are accepted for compatibility and ignored: the data is never copied and the fixed
    effects are always absorbed by fereg's exact solver (``fixef_tol``/``fixef_maxiter`` are its
    tolerance and iteration cap). ``nthreads=None`` uses every CPU.
    """
    context = _capture_context(context)
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
        context=context,
    )


def _separation(separation_check: list[str] | None, fixef_rm: str) -> tuple[str, bool]:
    """Core ``fixef_rm`` and ``relu`` of pyfixest's GLM ``separation_check`` (None = ``["fe"]``);
    the core skips both checks without fixed effects, as pyfixest."""
    # pyfixest `_estimation_input_checks`
    if separation_check is not None:
        if not isinstance(separation_check, list):
            raise TypeError("The function argument `separation_check` must be of type list.")
        if not all(x in ["fe", "ir"] for x in separation_check):
            raise ValueError(
                "The function argument `separation_check` must be a list of strings containing 'fe' and/or 'ir'."
            )
    checks = ["fe"] if separation_check is None else separation_check
    if fixef_rm not in ("singleton", "none"):
        raise ValueError("fixef_rm must be 'singleton' or 'none'.")
    if "fe" not in checks:
        core_rm = fixef_rm
    else:
        core_rm = "perfect" if fixef_rm == "singleton" else "perfect_only"
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
    context: int | Mapping[str, Any] | None = None,
    split: str | None = None,
    fsplit: str | None = None,
    nthreads: int | None = None,
) -> Fepois | FixestMulti:
    """Poisson regression with high-dimensional fixed effects (pyfixest ``fepois``).

    ``separation_check=None`` means ``["fe"]`` (drop fixed-effect groups whose outcome is all
    zero); ``"ir"`` adds the ReLU separation check. As in pyfixest, both run only in models with
    fixed effects, and dropped observations are reported by the warning "N observations removed
    because of separation.". ``vcov`` / ``vcov_kwargs`` take every estimator of :func:`feols`
    (here computed from the IRLS-weighted working residuals); HC2/HC3 with fixed effects is a
    fereg extension. ``context`` as in :func:`feols`.
    """
    context = _capture_context(context)
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
        context=context,
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
    context: int | Mapping[str, Any] | None = None,
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
    estimates with normal-based inference). ``separation_check`` as in :func:`fepois`, for every
    family as in pyfixest (``"fe"`` drops the fixed-effect groups whose outcome is all <= 0;
    ``"ir"`` treats the outcome as a Poisson outcome; with ``"gaussian"`` it needs a
    non-negative outcome when there are zeros). ``weights``, ``weights_type`` and ``offset`` are
    fereg extensions (pyfixest's ``feglm`` has neither). ``vcov`` / ``vcov_kwargs`` take every
    estimator of :func:`feols`; extensions as there, plus logit CRV3 (fereg's leave-one-cluster-out
    refits are logit fits, where pyfixest refits a Poisson model). Deliberate differences from
    pyfixest 0.60.0: gaussian iid variances are scaled by the residual variance (pyfixest's
    ``Feglm._vcov_iid`` fixes the dispersion at 1), and the ``"ir"`` check also works after
    rows were dropped before it (pyfixest's then misaligns its predictions and drops nothing).
    ``context`` as in :func:`feols`.
    """
    context = _capture_context(context)
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
            context=context,
        )
    if family == "probit":
        raise NotImplementedError("probit is not supported by fereg.")
    if family not in ("logit", "gaussian"):
        raise ValueError(f"family must be 'logit', 'poisson' or 'gaussian', got '{family}'.")
    if family == "gaussian" and offset is not None:
        raise NotImplementedError("offset is not supported with family='gaussian'.")
    # pyfixest runs its GLM separation checks for every family (gaussian too).
    (core_rm, relu), model = _separation(separation_check, fixef_rm), "ols" if family == "gaussian" else "logit"
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
        context=context,
    )
