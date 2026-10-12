"""Result classes with pyfixest 0.60.0 names, attributes and methods.

Every quantity is computed by the Rust core (``fereg._core``); this module only arranges it the
way pyfixest presents it. Per-observation arrays are exposed lazily so that building a result
costs O(k²) Python work regardless of the number of observations.
"""

from __future__ import annotations

import functools
import re
import warnings
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from ._binding import (
    core_ssc,
    factor_arg,
    pchisq_upper,
    pf_upper,
    pnorm,
    pt,
    qnorm,
    qt,
)
from ._vcov import VcovSpec, parse_vcov

__all__ = ["Feols", "Feiv", "Fepois", "Feglm", "Felogit", "Fegaussian", "FixestMulti"]

_NORMAL_METHODS = ("fepois", "feglm-logit", "feglm-probit", "feglm-gaussian")


@dataclass
class FitContext:
    """Everything needed to (re)estimate one model: formula, design and options."""

    formula: Any  # fereg.formula.Formula
    design: Any  # fereg.design.Design built from ``data``
    data: pd.DataFrame  # frame whose rows are the core's input rows
    model: str  # core model: "ols" | "poisson" | "logit"
    method: str  # pyfixest ``_method``
    vcov: VcovSpec
    ssc: dict
    weights: str | None
    weights_type: str
    offset: str | None
    fixef_rm: str  # core fixef_rm: "none" | "singleton" | "perfect"
    collin_tol: float
    fixef_tol: float
    fixef_maxiter: int
    glm_tol: float = 1e-8
    glm_maxiter: int = 25
    relu: bool = False
    lean: bool = False
    threads: int = 0
    drop_intercept: bool = False
    split_var: str | None = None
    split_value: Any = None
    extra: dict = field(default_factory=dict)


def _coef_names(design: Any, sources: list[tuple[str, int]]) -> list[str]:
    names = []
    for kind, j in sources:
        if kind == "intercept":
            names.append("Intercept")
        elif kind == "x":
            names.append(design.x_names[j])
        elif kind == "endog":
            names.append(design.endog_names[j])
        else:
            names.append(design.inst_names[j])
    return names


def _as_int(v: float) -> int | float:
    return int(v) if float(v).is_integer() else v


def _select_order_coefs(coefs: list[str], keep, drop, exact_match: bool) -> list[str]:
    """pyfixest ``_select_order_coefs``: regex (or exact) keep/drop, ordered by ``keep``."""
    keep = [keep] if isinstance(keep, str) else list(keep or [])
    drop = [drop] if isinstance(drop, str) else list(drop or [])

    def hit(pattern: str, coef: str) -> bool:
        return pattern == coef if exact_match else bool(re.findall(pattern, coef))

    coefs = list(coefs)
    res = [] if keep else coefs[:]
    for pattern in keep:
        rest = []
        for c in coefs:
            (res if hit(pattern, c) else rest).append(c)
        coefs = rest
    for pattern in drop:
        res = [c for c in res if not hit(pattern, c)]
    return res


class Feols:
    """OLS estimation results (pyfixest ``Feols``)."""

    _is_iv = False

    def __init__(self, ctx: FitContext, out: Any) -> None:
        d = ctx.design
        self._ctx = ctx
        self._out = out
        self._design = d
        self.FixestFormula = ctx.formula
        self._method = ctx.method
        self._fml = ctx.formula.fml
        self._sample_split_var = ctx.split_var
        self._sample_split_value = ctx.split_value
        self._model_name = (
            self._fml
            if ctx.split_var is None
            else f"{self._fml} (Sample: {ctx.split_var} = {ctx.split_value})"
        )
        self._model_name_plot = self._model_name
        self._depvar = d.y_name
        self._drop_intercept = ctx.drop_intercept
        self._weights_name = ctx.weights
        self._weights_type = ctx.weights_type
        self._has_weights = ctx.weights is not None
        self._offset_name = ctx.offset
        self._collin_tol = ctx.collin_tol
        self._fixef_tol = ctx.fixef_tol
        self._fixef_maxiter = ctx.fixef_maxiter
        self._lean = ctx.lean
        self._ssc_dict = ctx.ssc
        self._use_mundlak = False

        self._coefnames = _coef_names(d, out.coef_source)
        self._collin_vars = _coef_names(d, out.collinear)
        self._beta_hat = out.coef
        self._k = len(self._coefnames)
        self._N_rows = out.nobs
        self._N = out.sumw if ctx.weights_type == "fweights" else out.nobs

        self._has_fixef = len(d.fe) > 0
        self._fixef = (ctx.formula.fe_text or None) if self._has_fixef else None
        self._k_fe = pd.Series(out.fe_sizes, index=list(d.fe_names)) if self._has_fixef else None
        self._n_fe = len(d.fe)
        self._fixef_dict: dict[str, dict[str, float]] = {}

        self._rmse = np.nan
        self._r2 = np.nan
        self._r2_within = np.nan
        self._adj_r2 = np.nan
        self._adj_r2_within = np.nan
        self.deviance = None
        self._notes = list(out.notes)

        self._set_vcov(out.vcov, ctx.vcov)
        self.get_inference()

    # ----------------------------------------------------------------- variance

    def _df_t_from(self, v: dict, spec: VcovSpec):
        if v["df_t"] is not None:
            return v["df_t"] if spec.is_clustered else _as_int(v["df_t"])
        # GLM: pyfixest still reports the t degrees of freedom (inference is normal).
        if spec.is_clustered:
            return float(min(v["n_clusters"]) - 1)
        return _as_int(self._N - v["df_k"])

    def _set_vcov(self, v: dict, spec: VcovSpec) -> None:
        self._vcov = v["vcov"]
        self._df_k = _as_int(v["df_k"])
        self._df_t = self._df_t_from(v, spec)
        self._vcov_spec = spec
        self._vcov_type = spec.vcov_type
        self._vcov_type_detail = spec.detail
        self._is_clustered = spec.is_clustered
        self._clustervar = list(spec.cluster) if spec.is_clustered else None
        self._ssc = np.array([v["ssc_factor"]])
        self._psd_fixed = bool(v["psd_fixed"])
        if spec.is_clustered:
            g = list(v["n_clusters"])
            g_df = (self._ssc_dict or {}).get("G_df", "min")
            # pyfixest stores G_min three times (one per signed two-way term) when G_df="min".
            self._G = [min(g)] * 3 if g_df == "min" else g
        else:
            self._G = []

    def get_inference(self, alpha: float = 0.05) -> None:
        """Standard errors, t statistics, p-values and confidence intervals."""
        b = self._beta_hat
        self._se = np.sqrt(np.diagonal(self._vcov)) if self._k else np.array([])
        self._tstat = b / self._se if self._k else np.array([])
        if self._method in _NORMAL_METHODS:
            self._pvalue = 2.0 * np.asarray(pnorm(-np.abs(self._tstat)))
            z = abs(qnorm(alpha / 2))
        else:
            self._pvalue = 2.0 * np.asarray(pt(-np.abs(self._tstat), self._df_t))
            z = abs(qt(alpha / 2, self._df_t))
        self._conf_int = np.array([b - z * self._se, b + z * self._se])

    def vcov(self, vcov, vcov_kwargs: dict | None = None, data=None):
        """Recomputes the variance matrix under ``vcov`` without refitting; returns ``self``."""
        if vcov_kwargs:
            raise NotImplementedError("vcov_kwargs (HAC options) are not supported by fereg.")
        state = self._out.state
        if state is None:
            raise ValueError(
                "This model was estimated with lean=True and keeps no state for vcov(); "
                "re-estimate it with lean=False."
            )
        spec = parse_vcov(vcov)
        factors, names, cluster = [], [], []
        if spec.is_clustered:
            from .design import factor_column

            src = self._ctx.data if data is None else _check_frame(data)
            if len(src) != self._out.nobs_origin:
                raise ValueError(
                    f"data has {len(src)} rows but the model was estimated on "
                    f"{self._out.nobs_origin} input rows."
                )
            for i, name in enumerate(spec.cluster):
                fc = factor_column(src, name)
                factors.append(factor_arg(fc))
                names.append(fc.name)
                cluster.append([i])
        v = state.vcov(
            spec.kind,
            ssc=core_ssc(self._ssc_dict),
            factors=factors,
            factor_names=names,
            cluster=cluster,
            threads=self._ctx.threads,
        )
        self._set_vcov(v, spec)
        self.get_inference()
        return self

    # ----------------------------------------------------------------- fit statistics

    def get_performance(self) -> None:
        """R², adjusted R², within R² and RMSE with pyfixest's definitions."""
        st = self._out.stats
        n, k = self._N, self._k
        has_intercept = not self._drop_intercept
        ssu, ssy = st["ssr"], st["ssr_null"]
        if self._has_fixef:
            k_fe = int(np.sum(self._k_fe.to_numpy() - 1)) + 1
            adj_factor = (n - has_intercept) / (n - k - k_fe)
            adj_factor_within = (n - k_fe) / (n - k - k_fe)
        else:
            adj_factor = (n - has_intercept) / (n - k)
        self._rmse = np.sqrt(ssu / n)
        self._r2 = 1 - ssu / ssy
        self._adj_r2 = 1 - (ssu / ssy) * adj_factor
        if self._has_fixef:
            ssy_within = st["ssr_fe_only"]
            self._r2_within = 1 - ssu / ssy_within
            self._adj_r2_within = 1 - (ssu / ssy_within) * adj_factor_within

    def wald_test(self, R=None, q=None, distribution: str = "F") -> pd.Series:
        """Wald test of ``R β = q`` (default: all coefficients are zero), as pyfixest."""
        k_fe = int(np.sum(self._k_fe.to_numpy())) if self._has_fixef else 0
        R = np.eye(self._k) if R is None else np.asarray(R, dtype=float)
        if R.ndim == 1:
            R = R.reshape((1, len(R)))
        if R.shape[1] != self._k:
            raise ValueError("The number of columns of R must be equal to the number of coefficients.")
        if q is None:
            q = np.zeros(R.shape[0])
        else:
            if not isinstance(q, (int, float, np.ndarray)):
                raise ValueError("q must be a numeric scalar or array.")
            if isinstance(q, np.ndarray):
                if q.ndim != 1:
                    raise ValueError("q must be a one-dimensional array or a scalar.")
                if q.shape[0] != R.shape[0]:
                    raise ValueError("q must have the same number of rows as R.")
        self._dfn = R.shape[0]
        self._dfd = np.min(np.array(self._G)) - 1 if self._is_clustered else self._N - self._k - k_fe
        diff = R @ self._beta_hat - q
        w = diff.T @ np.linalg.pinv(R @ self._vcov @ R.T) @ diff
        self._wald_statistic = w
        if distribution == "F" and (not np.array_equal(R, np.eye(self._k)) or not np.all(q == 0)):
            warnings.warn(
                "Distribution changed to chi2, as R is not an identity matrix and q is not a zero vector."
            )
            distribution = "chi2"
        if distribution == "F":
            self._f_statistic = w / self._dfn
            self._p_value = pf_upper(self._f_statistic, self._dfn, self._dfd)
            return pd.Series({"statistic": self._f_statistic, "pvalue": self._p_value})
        if distribution == "chi2":
            self._f_statistic = w / self._dfn
            self._p_value = pchisq_upper(w, self._dfn)
            return pd.Series({"statistic": w, "pvalue": self._p_value})
        raise ValueError("Distribution must be F or chi2")

    # ----------------------------------------------------------------- accessors

    def tidy(self, alpha: float = 0.05) -> pd.DataFrame:
        """Coefficient table indexed by ``Coefficient``."""
        self.get_inference(alpha=alpha)
        lb, ub = alpha / 2, 1 - alpha / 2
        data = {
            "Coefficient": self._coefnames,
            "Estimate": self._beta_hat,
            "Std. Error": self._se,
            "t value": self._tstat,
            "Pr(>|t|)": self._pvalue,
            f"{lb * 100:.1f}%": self._conf_int[:1].flatten(),
            f"{ub * 100:.1f}%": self._conf_int[1:2].flatten(),
        }
        if self._sample_split_var is not None:
            data["Sample"] = self._sample_split_value
        return pd.DataFrame(data).set_index("Coefficient")

    def coef(self) -> pd.Series:
        return self.tidy()["Estimate"]

    def se(self) -> pd.Series:
        return self.tidy()["Std. Error"]

    def tstat(self) -> pd.Series:
        return self.tidy()["t value"]

    def pvalue(self) -> pd.Series:
        return self.tidy()["Pr(>|t|)"]

    def confint(
        self,
        alpha: float = 0.05,
        keep: list | str | None = None,
        drop: list | str | None = None,
        exact_match: bool | None = False,
        joint: bool = False,
        seed: int | None = None,
        reps: int = 10_000,
    ) -> pd.DataFrame:
        """Confidence intervals (t for OLS/IV, normal for GLMs)."""
        if joint:
            raise NotImplementedError("joint (simultaneous) confidence intervals are not supported by fereg.")
        if keep or drop:
            coefnames = _select_order_coefs(self._coefnames, keep, drop, bool(exact_match))
        else:
            coefnames = self._coefnames
        idx = [i for i, c in enumerate(self._coefnames) if c in coefnames]
        if not idx:
            raise ValueError("No coefficients match the keep/drop patterns.")
        if self._method in _NORMAL_METHODS:
            crit = abs(qnorm(alpha / 2))
        else:
            crit = abs(qt(alpha / 2, self._df_t))
        se = np.sqrt(np.diagonal(self._vcov))[idx]
        b = self._beta_hat[idx]
        out = pd.DataFrame({f"{alpha / 2 * 100:.1f}%": b - crit * se, f"{(1 - alpha / 2) * 100:.1f}%": b + crit * se})
        out.index = [self._coefnames[i] for i in idx]
        return out

    def summary(self, digits: int = 3) -> None:
        from .summary import summary

        summary(self, digits=digits)

    def etable(self, *args, **kwargs):
        from .etable import etable

        return etable([self], *args, **kwargs)

    # ----------------------------------------------------------------- per-observation output

    def _require(self, arr, what: str):
        if arr is None:
            raise ValueError(f"{what} is not available: the model was estimated with lean=True.")
        return arr

    @property
    def _obs(self) -> np.ndarray:
        """Input rows of the estimation sample (ascending)."""
        return self._out.obs

    @property
    def _u_hat(self) -> np.ndarray:
        """Residuals scaled by sqrt(weights), as pyfixest stores them."""
        e = self._require(self._out.resid, "_u_hat")
        return e * np.sqrt(self._user_weights) if self._has_weights else e

    @property
    def _Y_hat_link(self) -> np.ndarray:
        return self._require(self._out.fitted, "_Y_hat_link")

    @property
    def _Y_hat_response(self) -> np.ndarray:
        return self._require(self._out.fitted, "_Y_hat_response")

    @property
    def _user_weights(self) -> np.ndarray:
        return np.asarray(self._design.weights)[self._obs]

    @property
    def _weights(self) -> np.ndarray:
        """Observation weights of the estimation sample, shape (N, 1)."""
        if not self._has_weights:
            return np.ones((self._N_rows, 1))
        return self._user_weights.reshape(-1, 1)

    @property
    def _data(self) -> pd.DataFrame:
        """Estimation-sample rows of the input data."""
        return _to_pandas(self._ctx.data).iloc[self._obs]

    @property
    def _na_index(self) -> frozenset:
        """Input rows dropped before estimation (missing values, singletons, separation)."""
        keep = np.zeros(self._out.nobs_origin, dtype=bool)
        keep[self._obs] = True
        return frozenset(np.flatnonzero(~keep).tolist())

    def resid(self) -> np.ndarray:
        """Residuals y − ŷ (IV: residuals with the original endogenous regressors)."""
        return self._require(self._out.resid, "resid()")

    # ----------------------------------------------------------------- fixed effects

    def fixef(self, atol: float = 1e-6, btol: float = 1e-6) -> dict[str, dict[str, float]]:
        """Fixed-effect estimates ``{"C(fe)": {level: value}}``, normalized as pyfixest.

        The first FE keeps every level; in every later FE the first level is the reference
        (value 0) and, as in pyfixest, is left out of the dict; any remaining indeterminacy
        (several connected components) is resolved by the minimum-norm solution.
        """
        if not self._has_fixef:
            raise ValueError("The regression model does not have fixed effects.")
        if self._fixef_dict:
            return self._fixef_dict
        from .design import term_labels

        d = self._design
        res: dict[str, dict[str, float]] = {}
        for f in self._fixef_values:
            key = self._fixef_key(f["term"], f["slope"])
            labels = term_labels(d, f["term"], f["first_obs"])
            values = f["values"]
            if f["slope"] is None and f["references"] > 0:
                # The pinned reference level (value 0) is values[0]; pyfixest omits it.
                labels, values = labels[1:], values[1:]
            res[key] = dict(zip(labels, values))
        self._fixef_dict = res
        return res

    @functools.cached_property
    def _fixef_values(self) -> list[dict]:
        """Per-term, per-slot FE values (core ``FixefValues``), computed on first use.

        Fits do not compute them (pyfixest's ``fixef()`` is lazy too); the retained state
        normalizes the solver's per-level values on demand, with the GIL released.
        """
        state = self._out.state
        if state is None:
            raise ValueError(
                "Fixed-effect values are not available: the model was estimated with lean=True."
            )
        return state.fixef("minnorm", threads=self._ctx.threads)

    def _fixef_key(self, term: int, slope: int | None) -> str:
        """pyfixest key ``C(fe)`` (``C(a^b)`` for interactions, ``C(fe):x`` for slopes)."""
        label = "^".join(self.FixestFormula.fe[term].factors)
        return f"C({label})" if slope is None else f"C({label}):{self._design.slope_names[slope]}"

    def _fe_prediction(self, nd: Any) -> np.ndarray:
        """Σ fixed effects (× slopes) for every row of the newdata design; unseen level → NaN."""
        d = self._design
        fe_hat = np.zeros(nd.n)
        keys_cache: dict[int, tuple[pd.Index, np.ndarray]] = {}
        for f in self._fixef_values:
            term = f["term"]
            if term not in keys_cache:
                train = pd.Index(_level_keys(d, term, f["first_obs"]))
                new = _level_keys(nd, term, np.arange(nd.n))
                keys_cache[term] = (train, train.get_indexer(new))
            _, pos = keys_cache[term]
            vals = np.asarray(f["values"])[pos]
            vals[pos < 0] = np.nan
            if f["slope"] is not None:
                vals = vals * np.asarray(nd.slopes[f["slope"]], dtype=float)
            fe_hat += vals
        return fe_hat

    # ----------------------------------------------------------------- prediction

    def _linear_predictor_newdata(self, newdata) -> np.ndarray:
        from .design import build_design

        nd = build_design(
            self.FixestFormula,
            _check_frame(newdata),
            offset=self._offset_name,
            drop_intercept=self._drop_intercept,
            like=self._design,
        )
        xb = np.zeros(nd.n)
        for (kind, j), b in zip(self._out.coef_source, self._beta_hat):
            if kind == "intercept":
                xb += b
            elif kind == "x":
                xb += b * np.asarray(nd.x[j], dtype=float)
            else:
                raise NotImplementedError("predict() with newdata is not supported for IV models.")
        if self._has_fixef:
            xb += self._fe_prediction(nd)
        if self._offset_name is not None:
            off = np.asarray(nd.offset, dtype=float)
            if np.isnan(off).any():
                raise ValueError(
                    f"Offset column '{self._offset_name}' in newdata contains NaN or non-numeric values."
                )
            xb += off
        if nd.drop is not None:
            xb[np.asarray(nd.drop, dtype=bool)] = np.nan
        return xb

    def predict(
        self,
        newdata=None,
        atol: float = 1e-6,
        btol: float = 1e-6,
        type: str = "link",
        se_fit: bool | None = False,
        interval: str | None = None,
        alpha: float = 0.05,
    ) -> np.ndarray:
        """Predictions; unseen fixed-effect levels in ``newdata`` give NaN."""
        if self._is_iv:
            raise NotImplementedError("The predict() method is currently not supported for IV models.")
        if type not in ("link", "response"):
            raise ValueError(f"type must be 'link' or 'response', got '{type}'.")
        if se_fit or interval is not None:
            raise NotImplementedError("Prediction standard errors and intervals are not supported by fereg.")
        if newdata is None:
            return self._Y_hat_link if type == "link" or self._method == "feols" else self._Y_hat_response
        eta = self._linear_predictor_newdata(newdata)
        return eta if type == "link" else self._response(eta)

    def _response(self, eta: np.ndarray) -> np.ndarray:
        return eta

    def __repr__(self) -> str:
        return f"<fereg.{type(self).__name__}: {self._model_name}>"


def _level_keys(design: Any, term: int, rows: np.ndarray) -> np.ndarray:
    """Comparable level keys of FE term ``term`` at ``rows`` (numeric values when possible)."""
    factors, _, _ = design.fe[term]
    if len(factors) == 1:
        fc = design.factors[factors[0]]
        if fc.kind in ("f64", "i64", "i32"):
            return np.asarray(fc.values)[rows]
    from .design import term_labels

    return np.asarray(term_labels(design, term, rows), dtype=object)


def _check_frame(data):
    """``data`` itself when it is a pandas or polars DataFrame."""
    if isinstance(data, pd.DataFrame) or type(data).__module__.split(".")[0] == "polars":
        return data
    raise TypeError(f"The data set must be a pandas or polars DataFrame. Received: {type(data)}")


def _to_pandas(data) -> pd.DataFrame:
    data = _check_frame(data)
    return data if isinstance(data, pd.DataFrame) else data.to_pandas()


class Feiv(Feols):
    """IV/2SLS estimation results (pyfixest ``Feiv``)."""

    _is_iv = True

    def __init__(self, ctx: FitContext, out: Any) -> None:
        super().__init__(ctx, out)
        d = ctx.design
        self._coefnames_z = (["Intercept"] if "Intercept" in self._coefnames else []) + list(d.inst_names) + list(
            d.x_names
        )
        self._iv_stats = out.iv
        self._first_stage_models: dict[int, Feols] = {}

    def _first_stage_model(self, j: int = 0) -> Feols:
        """First-stage regression of endogenous regressor ``j`` on [instruments, exogenous]."""
        if j not in self._first_stage_models:
            from .estimation import estimate

            ctx, d, f = self._ctx, self._ctx.design, self._ctx.formula
            endog, inst = d.endog_names[j], list(d.inst_names)
            rhs = list(inst) + list(f.rhs)
            fml = f"{endog} ~ {' + '.join(rhs) if rhs else '1'}" + (f" | {f.fe_text}" if f.fe_text else "")
            rhs_text = " + ".join(t for t in (f.inst_text, f.rhs_text) if t)
            formula = replace(
                f, fml=fml, depvar=endog, rhs=tuple(rhs), endog=(), inst=(), rhs_text=rhs_text, inst_text=""
            )
            design = replace(
                d,
                y=d.endog[j],
                y_name=endog,
                x=list(d.inst) + list(d.x),
                x_names=inst + list(d.x_names),
                endog=[],
                endog_names=[],
                inst=[],
                inst_names=[],
            )
            spec = ctx.vcov
            if spec.is_clustered:
                # pyfixest clusters the first stage by the first cluster variable only.
                spec = parse_vcov({spec.detail: spec.cluster[0]})
                design = replace(design, cluster=list(d.cluster[:1]), cluster_names=list(d.cluster_names[:1]))
            self._first_stage_models[j] = estimate(replace(ctx, formula=formula, design=design, vcov=spec))
        return self._first_stage_models[j]

    @property
    def _model_1st_stage(self) -> Feols:
        return self._first_stage_model(0)

    def first_stage(self) -> None:
        """Fits the first stage and its instrument F statistic (pyfixest ``Feiv.first_stage``)."""
        m1 = self._model_1st_stage
        self._non_exo_instruments = list(self._design.inst_names)
        self._p_iv = len(self._non_exo_instruments)
        self._iv_loc = [m1._coefnames.index(z) for z in self._non_exo_instruments if z in m1._coefnames]
        R = np.zeros((len(self._iv_loc), m1._k))
        R[np.arange(len(self._iv_loc)), self._iv_loc] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m1.wald_test(R=R)
        self._f_stat_1st_stage = m1._f_statistic
        self._p_value_1st_stage = m1._p_value

    def __getattr__(self, name: str):
        if name in ("_f_stat_1st_stage", "_p_value_1st_stage", "_non_exo_instruments", "_p_iv", "_iv_loc"):
            self.first_stage()
            return self.__dict__[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class Fepois(Feols):
    """Poisson estimation results (pyfixest ``Fepois``)."""

    def __init__(self, ctx: FitContext, out: Any) -> None:
        super().__init__(ctx, out)
        st = out.stats
        self.deviance = st["deviance"]
        self._loglik = st["loglik"]
        self._loglik_null = None if self._has_weights else st["ll_null"]
        self._pseudo_r2 = None if self._has_weights else st["pseudo_r2"]
        self.convergence = bool(st["converged"])
        self._convergence = self.convergence
        self.maxiter = ctx.glm_maxiter
        self.tol = ctx.glm_tol
        self._irls_iterations = st["irls_iterations"]

    @property
    def _Y_hat_link(self) -> np.ndarray:
        return self._require(self._out.eta, "_Y_hat_link")

    @property
    def _irls_weights(self) -> np.ndarray:
        return self._require(self._out.irls_weights, "_irls_weights")

    @property
    def _weights(self) -> np.ndarray:
        """Final IRLS weights, shape (N, 1) (pyfixest stores these in ``_weights`` for GLMs)."""
        return self._irls_weights.reshape(-1, 1)

    @property
    def _u_hat_response(self) -> np.ndarray:
        return self._require(self._out.resid, "_u_hat_response")

    @property
    def _u_hat_working(self) -> np.ndarray:
        mu = self._Y_hat_response
        return self._u_hat_response / self._variance(mu)

    @staticmethod
    def _variance(mu: np.ndarray) -> np.ndarray:
        return mu

    @property
    def _u_hat(self) -> np.ndarray:
        return self._u_hat_working * np.sqrt(self._irls_weights)

    def resid(self, type: str = "response") -> np.ndarray:
        """Response (y − μ) or working residuals."""
        if type == "response":
            return self._u_hat_response
        if type == "working":
            return self._u_hat_working
        raise ValueError("type must be one of 'response' or 'working'.")

    def _response(self, eta: np.ndarray) -> np.ndarray:
        return np.exp(eta)


# Bound of the logit mean in the core (`glm::family::MU_MIN` = 100·ε).
_MU_MIN = 100.0 * np.finfo(np.float64).eps


class Feglm(Feols):
    """GLM estimation results (pyfixest ``Feglm``)."""

    def __init__(self, ctx: FitContext, out: Any) -> None:
        super().__init__(ctx, out)
        st = out.stats
        self.deviance = st["deviance"]
        self._loglik = st["loglik"]
        self._loglik_null = st["ll_null"]
        self._pseudo_r2 = st["pseudo_r2"]
        self.convergence = bool(st["converged"])
        self._convergence = self.convergence
        self.maxiter = ctx.glm_maxiter
        self.tol = ctx.glm_tol
        self._irls_iterations = st["irls_iterations"]

    @property
    def _Y_hat_link(self) -> np.ndarray:
        return self._require(self._out.eta, "_Y_hat_link")

    @property
    def _irls_weights(self) -> np.ndarray:
        return self._require(self._out.irls_weights, "_irls_weights")

    @property
    def _weights(self) -> np.ndarray:
        """Final IRLS weights, shape (N, 1) (pyfixest stores these in ``_weights`` for GLMs)."""
        return self._irls_weights.reshape(-1, 1)

    @property
    def _u_hat_response(self) -> np.ndarray:
        return self._require(self._out.resid, "_u_hat_response")

    @property
    def _u_hat_working(self) -> np.ndarray:
        mu = self._Y_hat_response
        with np.errstate(divide="ignore", invalid="ignore"):
            e = self._u_hat_response / (mu * (1.0 - mu))
        # Rows of FE levels with an all-one (or all-zero) outcome sit at the bound μ = 1 − MU_MIN
        # (MU_MIN): their FE is ±inf, the fit is exact and the working residual is 0 (the
        # limit pyfixest's IRLS approaches).
        e[(mu <= _MU_MIN) | (mu >= 1.0 - _MU_MIN)] = 0.0
        return e

    @property
    def _u_hat(self) -> np.ndarray:
        """Working residuals scaled by sqrt(IRLS weights), as pyfixest stores them."""
        return self._u_hat_working * np.sqrt(self._irls_weights)

    def resid(self, type: str = "working") -> np.ndarray:
        """Working (pyfixest default) or response residuals."""
        if type == "working":
            return self._u_hat_working
        if type == "response":
            return self._u_hat_response
        raise ValueError("type must be one of 'response' or 'working'.")


class Felogit(Feglm):
    """Logit estimation results (pyfixest ``Felogit``)."""

    # η of μ = 1 − MU_MIN. Rows of all-one (all-zero) FE levels have FE = ±inf; their
    # predictions sit at the bound, as the fitted values of the estimation sample do.
    _ETA_BOUND = float(np.log((1.0 - _MU_MIN) / _MU_MIN))

    def _linear_predictor_newdata(self, newdata) -> np.ndarray:
        return np.clip(super()._linear_predictor_newdata(newdata), -self._ETA_BOUND, self._ETA_BOUND)

    def _response(self, eta: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-eta))


class Fegaussian(Feols):
    """Gaussian GLM results: OLS estimates with normal-based inference (pyfixest ``Fegaussian``)."""


class FixestMulti:
    """Results of multiple estimation (pyfixest ``FixestMulti``)."""

    def __init__(self, models: dict[str, Feols]) -> None:
        self.all_fitted_models = models

    def to_list(self) -> list[Feols]:
        return list(self.all_fitted_models.values())

    def fetch_model(self, i: int | str, print_fml: bool | None = True) -> Feols:
        i = int(i)
        keys = list(self.all_fitted_models.keys())
        if i >= len(keys):
            raise IndexError(f"Index {i} is larger than the number of fitted models.")
        if print_fml:
            print("Model: ", keys[i])
        return self.all_fitted_models[keys[i]]

    def vcov(self, vcov, vcov_kwargs: dict | None = None):
        for m in self.all_fitted_models.values():
            m.vcov(vcov, vcov_kwargs=vcov_kwargs)
        return self

    def tidy(self) -> pd.DataFrame:
        res = []
        for m in self.all_fitted_models.values():
            df = m.tidy().reset_index()
            df["fml"] = m._fml
            res.append(df)
        return pd.concat(res, axis=0).set_index(["fml", "Coefficient"])

    def coef(self) -> pd.Series:
        return self.tidy()["Estimate"]

    def se(self) -> pd.Series:
        return self.tidy()["Std. Error"]

    def tstat(self) -> pd.Series:
        return self.tidy()["t value"]

    def pvalue(self) -> pd.Series:
        return self.tidy()["Pr(>|t|)"]

    def confint(self) -> pd.DataFrame:
        return self.tidy()[["2.5%", "97.5%"]]

    def summary(self, digits: int = 3) -> None:
        from .summary import summary

        summary(self, digits=digits)

    def etable(self, *args, **kwargs):
        from .etable import etable

        return etable(self, *args, **kwargs)

    def __len__(self) -> int:
        return len(self.all_fitted_models)

    def __repr__(self) -> str:
        return f"<fereg.FixestMulti: {len(self)} models>"
