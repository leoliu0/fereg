"""Wild bootstrap inference for OLS, IV/2SLS, Poisson and logit models."""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from ._binding import wildboot_design
from ._vcov import MAX_CLUSTER_DIMS, parse_vcov
from .design import _Frame, _factor_from_frame
from .results import Feols, FixestMulti, _coef_names

__all__ = ["WildBootResult", "wildboottest"]

_WEIGHTS = ("rademacher", "webb", "mammen", "normal")
_BOOT_TYPES = ("11", "13", "31", "33", "21")
_METHODS = ("auto", "wildcluster", "score", "wre", "hetero")
_PVAL_TYPES = ("two-tailed", "equal-tailed", ">", "<")
_PVAL_ALIASES = {"greater": ">", "less": "<"}


@dataclass(frozen=True)
class WildBootResult:
    """Result of :func:`wildboottest` / :meth:`Feols.wildboottest`."""

    statistic: float
    p_value: float
    stat_type: str
    point_estimate: float | None
    se: float | None
    conf_int: tuple[float, float] | None
    t_boot: np.ndarray
    reps: int
    reps_feasible: int
    nobs: int
    n_clusters: int | None
    q: int
    full_enumeration: bool
    method: str
    boot_type: str
    weights: str
    impose_null: bool
    pval_type: str
    ssc: float
    param: str
    level: float
    seed: int
    cluster: list[str] | None
    bootcluster: list[str] | None
    notes: list[str]

    def to_series(self) -> pd.Series:
        """Compact one-column summary as a :class:`pandas.Series`."""
        lo, hi = self.conf_int if self.conf_int is not None else (np.nan, np.nan)
        return pd.Series(
            {
                "statistic": self.statistic,
                "p_value": self.p_value,
                "point_estimate": np.nan if self.point_estimate is None else self.point_estimate,
                "se": np.nan if self.se is None else self.se,
                "ci_low": lo,
                "ci_high": hi,
                "reps": self.reps,
                "reps_feasible": self.reps_feasible,
                "nobs": self.nobs,
                "n_clusters": self.n_clusters,
                "q": self.q,
                "stat_type": self.stat_type,
                "method": self.method,
                "boot_type": self.boot_type,
                "weights": self.weights,
                "impose_null": self.impose_null,
                "pval_type": self.pval_type,
                "full_enumeration": self.full_enumeration,
            },
            name=self.param,
        )

    def summary(self, digits: int = 4) -> None:
        """Prints a summary of the wild bootstrap test."""
        cl = ", ".join(self.cluster) if self.cluster else "unclustered"
        bc = ", ".join(self.bootcluster) if self.bootcluster else cl
        enum = " (full enumeration)" if self.full_enumeration else ""
        print(f"Wild bootstrap test: {self.param}")
        print(
            f"  method = {self.method}, type = {self.boot_type}, weights = {self.weights}, "
            f"impose_null = {self.impose_null}, pval_type = {self.pval_type}"
        )
        if self.n_clusters is not None:
            print(f"  clusters = {cl} (bootcluster = {bc}, G = {self.n_clusters}), nobs = {self.nobs}")
        else:
            print(f"  clusters = {cl}, nobs = {self.nobs}")
        print(f"  reps = {self.reps}{enum} (feasible = {self.reps_feasible}), seed = {self.seed}")
        parts = [f"{self.stat_type} = {self.statistic:.{digits}f}", f"p = {self.p_value:.{digits}f}"]
        if self.point_estimate is not None:
            parts.append(f"estimate = {self.point_estimate:.{digits}f}")
        if self.se is not None:
            parts.append(f"se = {self.se:.{digits}f}")
        if self.conf_int is not None:
            parts.append(f"{self.level * 100:.0f}% CI = [{self.conf_int[0]:.{digits}f}, {self.conf_int[1]:.{digits}f}]")
        print("  " + ", ".join(parts))

    def __repr__(self) -> str:
        return f"<fereg.WildBootResult: {self.param}, {self.stat_type}={self.statistic:.4f}, p={self.p_value:.4g}>"


def _normalize_cluster_dims(spec: str | Sequence[str]) -> list[str]:
    if isinstance(spec, str):
        raw = [part.strip() for part in spec.split("+")]
    elif isinstance(spec, Sequence) and not isinstance(spec, (bytes, bytearray)):
        raw = []
        for item in spec:
            if not isinstance(item, str):
                raise TypeError("cluster dimensions must be strings.")
            raw.extend(part.strip() for part in item.split("+"))
    else:
        raise TypeError("cluster must be a variable name, 'a + b' string, or sequence of strings.")
    dims = ["".join(d.split()) for d in raw]
    if not dims or any(not d for d in dims):
        raise ValueError("cluster contains an empty dimension.")
    if len(dims) > MAX_CLUSTER_DIMS:
        raise ValueError(f"at most {MAX_CLUSTER_DIMS} cluster dimensions are supported.")
    if len(set(dims)) != len(dims):
        raise ValueError("duplicate cluster dimension.")
    return dims


def _normalize_bootcluster(spec: str | Sequence[str]) -> list[str]:
    dims = _normalize_cluster_dims(spec)
    parts: list[str] = []
    for dim in dims:
        for p in dim.split("^"):
            p = p.strip()
            if not p:
                raise ValueError("bootcluster contains an empty variable name.")
            if p not in parts:
                parts.append(p)
    return parts


def _build_hypothesis(
    model: Feols,
    param: str | Sequence[str] | None,
    R: Any,
    r: Any,
) -> tuple[np.ndarray, np.ndarray, str]:
    if param is not None and R is not None:
        raise ValueError("Specify either param or R, not both.")
    if param is None and R is None:
        raise ValueError("Specify either param or R.")

    coefnames = list(model._coefnames)
    collin = set(model._collin_vars)
    k = len(coefnames)
    if k == 0:
        raise ValueError("Model has no estimable parameters.")

    if param is not None:
        if isinstance(param, str):
            params = [param]
        elif isinstance(param, Sequence) and not isinstance(param, (bytes, bytearray)):
            params = list(param)
        else:
            raise TypeError("param must be a coefficient name or a sequence of names.")
        if not params or not all(isinstance(p, str) for p in params):
            raise ValueError("param must contain at least one coefficient name string.")
        if len(set(params)) != len(params):
            raise ValueError("param contains duplicate coefficient names.")
        idx = []
        for p in params:
            if p in collin:
                raise ValueError(f"Parameter '{p}' was dropped due to collinearity.")
            if p not in coefnames:
                raise ValueError(f"Parameter '{p}' not found in model coefficients: {coefnames}.")
            idx.append(coefnames.index(p))
        q = len(params)
        r_mat = np.zeros((q, k), dtype=np.float64)
        for row, col in enumerate(idx):
            r_mat[row, col] = 1.0
        r_vec = _parse_r_vec(r, q)
        label = ", ".join(f"{p} = {v:g}" for p, v in zip(params, r_vec))
        return r_mat, r_vec, label

    r_arr = np.asarray(R, dtype=np.float64)
    if r_arr.ndim == 1:
        r_arr = r_arr.reshape(1, -1)
    if r_arr.ndim != 2 or r_arr.shape[0] == 0 or r_arr.shape[1] != k:
        raise ValueError(f"R must have shape (q, {k}) with q >= 1; got {r_arr.shape}.")
    r_mat = np.ascontiguousarray(r_arr, dtype=np.float64)
    q = r_mat.shape[0]
    r_vec = _parse_r_vec(r, q)
    label = f"R * beta = r (q={q})" if q > 1 else "R * beta = r"
    return r_mat, r_vec, label


def _parse_r_vec(r: Any, q: int) -> np.ndarray:
    if r is None:
        return np.zeros(q, dtype=np.float64)
    if isinstance(r, numbers.Real) and not isinstance(r, bool):
        if q != 1:
            raise ValueError(f"r must have length {q} to match R/param.")
        return np.array([float(r)], dtype=np.float64)
    arr = np.asarray(r, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != q:
        raise ValueError(f"r must be a 1D vector of length {q}; got shape {arr.shape}.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _check_choice(name: str, value: Any, valid: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in valid:
        raise ValueError(f"{name} must be one of {valid!r}, got {value!r}.")
    return value


def _check_bool(name: str, value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a bool.")
    return bool(value)


def _check_int(name: str, value: Any, *, min_value: int = 0) -> int:
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    val = int(value)
    if val < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {val}.")
    return val


def _resolve_clusters(
    model: Feols,
    cluster: str | Sequence[str] | None,
    bootcluster: str | Sequence[str] | None,
) -> tuple[Any, Any, list[int] | None, list[str] | None, list[str] | None]:
    vcov_spec = model._vcov_spec
    if cluster is None:
        if vcov_spec.is_hac or vcov_spec.kind == "conley":
            raise ValueError(
                f"Wild bootstrap is not defined for '{vcov_spec.detail}' VCOV; "
                "pass cluster=... or refit with clustered/hetero VCOV."
            )
        cluster_dims = list(vcov_spec.cluster) if vcov_spec.is_clustered else []
    elif isinstance(cluster, Sequence) and not isinstance(cluster, (str, bytes, bytearray)) and len(cluster) == 0:
        cluster_dims = []
    else:
        cluster_dims = _normalize_cluster_dims(cluster)

    bc_vars = None if bootcluster is None else _normalize_bootcluster(bootcluster)

    d = model._design
    factors = list(d.factors)
    by_name: dict[str, int] = {}
    for idx, fc in enumerate(factors):
        if fc.name not in by_name:
            by_name[fc.name] = idx

    frame: _Frame | None = None

    def ensure_factor(var: str) -> int:
        nonlocal frame
        if var in by_name:
            return by_name[var]
        if frame is None:
            frame = _Frame(model._ctx.data)
        fc = _factor_from_frame(frame, var)
        idx = len(factors)
        factors.append(fc)
        by_name[var] = idx
        return idx

    cluster_idx: list[list[int]] = []
    for dim in cluster_dims:
        parts = [p.strip() for p in dim.split("^")]
        if not parts or any(not p for p in parts):
            raise ValueError(f"Invalid cluster dimension '{dim}'.")
        cluster_idx.append([ensure_factor(p) for p in parts])

    bc_idx: list[int] | None = None
    if bc_vars is not None:
        bc_idx = [ensure_factor(v) for v in bc_vars]

    eff_vcov = parse_vcov({"CRV1": "+".join(cluster_dims)}) if cluster_dims else parse_vcov("hetero")
    design = replace(
        d,
        factors=factors,
        cluster=cluster_idx,
        cluster_names=cluster_dims,
        time=None,
        panel=None,
        lat=None,
        lon=None,
    )
    return (
        design,
        eff_vcov,
        bc_idx,
        cluster_dims if cluster_dims else None,
        bc_vars,
    )


def wildboottest(
    model: Feols,
    param: str | Sequence[str] | None = None,
    *,
    R: Any = None,
    r: Any = None,
    reps: int = 999,
    weights: str = "rademacher",
    boot_type: str = "11",
    bootstrap_type: str | None = None,
    method: str = "auto",
    impose_null: bool = True,
    pval_type: str = "two-tailed",
    conf_int: bool = True,
    level: float = 0.95,
    seed: int | None = None,
    cluster: str | Sequence[str] | None = None,
    bootcluster: str | Sequence[str] | None = None,
    tol: float = 1e-6,
    maxiter: int = 50,
    threads: int | None = None,
) -> WildBootResult:
    """Wild bootstrap inference for a fitted :class:`~fereg.Feols` / :class:`~fereg.Feiv` /
    :class:`~fereg.Fepois` / :class:`~fereg.Feglm` model."""
    from .bias import FelogitBiasCorr
    from .estimation import core_options

    if isinstance(model, FixestMulti):
        raise TypeError("wildboottest() does not support FixestMulti; select a single model first (e.g., m.fetch_model(0)).")
    if isinstance(model, FelogitBiasCorr):
        raise ValueError("wildboottest() is not supported on bias-corrected logit models.")
    if not isinstance(model, Feols):
        raise TypeError(f"model must be a fitted fereg model, got {type(model).__name__}.")

    r_mat, r_vec, param_label = _build_hypothesis(model, param, R, r)

    reps_i = _check_int("reps", reps, min_value=1)
    weights_s = _check_choice("weights", weights, _WEIGHTS)
    if bootstrap_type is not None:
        if not isinstance(bootstrap_type, str):
            raise TypeError("bootstrap_type must be a string.")
        alias = "11" if bootstrap_type == "fnw11" else bootstrap_type
        if boot_type != "11" and alias != boot_type:
            raise ValueError(f"Conflicting boot_type='{boot_type}' and bootstrap_type='{bootstrap_type}'.")
        boot_type = alias
    boot_type_s = _check_choice("boot_type", boot_type, _BOOT_TYPES)
    method_s = _check_choice("method", method, _METHODS)
    impose_null_b = _check_bool("impose_null", impose_null)
    pval_s = _check_choice("pval_type", _PVAL_ALIASES.get(pval_type, pval_type), _PVAL_TYPES)
    conf_int_b = _check_bool("conf_int", conf_int)

    if not isinstance(level, numbers.Real) or isinstance(level, bool) or not (0.0 < float(level) < 1.0):
        raise ValueError("level must be strictly between 0 and 1.")
    level_f = float(level)

    if not isinstance(tol, numbers.Real) or isinstance(tol, bool) or not np.isfinite(float(tol)) or float(tol) <= 0.0:
        raise ValueError("tol must be a positive finite number.")
    tol_f = float(tol)
    maxiter_i = _check_int("maxiter", maxiter, min_value=1)

    if seed is None:
        seed_u64 = int(np.random.default_rng().integers(0, 2**63))
    else:
        seed_u64 = _check_int("seed", seed, min_value=0)
        if seed_u64 >= 2**64:
            raise ValueError("seed must fit in an unsigned 64-bit integer.")

    threads_i = model._ctx.threads if threads is None else _check_int("threads", threads, min_value=0)

    design, eff_vcov, bc_idx, cl_label, bc_label = _resolve_clusters(model, cluster, bootcluster)
    opts = core_options(model._ctx, vcov=eff_vcov, lean=True)
    opts["threads"] = threads_i

    wb = {
        "r_mat": r_mat,
        "r_vec": r_vec,
        "reps": reps_i,
        "weights": weights_s,
        "boot_type": boot_type_s,
        "method": method_s,
        "impose_null": impose_null_b,
        "pval_type": pval_s,
        "conf_int": conf_int_b,
        "level": level_f,
        "seed": seed_u64,
        "tol": tol_f,
        "maxiter": maxiter_i,
        "bootcluster": bc_idx,
    }
    out = wildboot_design(design, wb=wb, **opts)
    for note in out["notes"]:
        warnings.warn(note, UserWarning, stacklevel=2)

    returned_names = _coef_names(design, out["coef_source"])
    if returned_names != list(model._coefnames):
        raise ValueError(
            "Wild bootstrap sample / collinearity changed the estimable coefficients relative to the fitted model."
        )

    ci = out["conf_int"]
    return WildBootResult(
        statistic=float(out["stat"]),
        p_value=float(out["p_val"]),
        stat_type=str(out["stat_type"]),
        point_estimate=None if out["point_estimate"] is None else float(out["point_estimate"]),
        se=None if out["se"] is None else float(out["se"]),
        conf_int=None if ci is None else (float(ci[0]), float(ci[1])),
        t_boot=np.asarray(out["t_boot"], dtype=np.float64),
        reps=int(out["reps"]),
        reps_feasible=int(out["reps_feasible"]),
        nobs=int(out["nobs"]),
        n_clusters=None if out["n_clusters"] is None else int(out["n_clusters"]),
        q=int(out["q"]),
        full_enumeration=bool(out["full_enumeration"]),
        method=str(out["method"]),
        boot_type=str(out["boot_type"]),
        weights=str(out["weights"]),
        impose_null=bool(out["impose_null"]),
        pval_type=str(out["pval_type"]),
        ssc=float(out["ssc"]),
        param=param_label,
        level=level_f,
        seed=seed_u64,
        cluster=cl_label,
        bootcluster=bc_label,
        notes=list(out["notes"]),
    )
