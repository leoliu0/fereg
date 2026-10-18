"""Incidental-parameter bias correction and average partial effects of fixed-effects logit models.

pyfixest has neither; the names, defaults and numbers follow the R package alpaca 0.3.5
(``biasCorr()`` -> :meth:`Felogit.bias_corr`, ``getAPEs()`` -> :meth:`Felogit.get_apes`, snake_case
arguments). The analytical correction is Fernandez-Val and Weidner (2016) for classic panels and
Hinz, Stammann and Wanner (2020) for network (trade) panels; the jackknife corrections follow
Stata's logitfe (Cruz-Gonzalez, Fernandez-Val and Weidner 2017) with its dimension bugs fixed.
Everything is computed by the Rust core (``fereg._core.bias_corr`` / ``apes`` / ``bias_dims``),
which refits the model on the same design and is bit-identical for any number of threads.

Differences from alpaca 0.3.5 (and logitfe 3.0.3 for the jackknife):

- Sample. pyfixest keeps the observations of fixed-effect levels whose outcome is all one at the
  bound of the mean (and drops all-zero levels once); the corrections use alpaca's sample: ``n``
  counts the observations whose fixed effects all have finite estimates, and the APE divisor
  ``N*`` counts every observation before any fixed-effect based drop (singletons, separation,
  perfect classification: they enter with zero partial effect, as in alpaca and logitfe).
- alpaca's default control stops one IRLS step early, so its default coefficients are not the
  maximum likelihood estimates; fereg converges to them (compare with alpaca's
  ``feglmControl(dev.tol = 1e-20, center.tol = 1e-14, iter.max = 100)``).
- After the correction the fixed effects are re-estimated exactly at the corrected coefficients;
  alpaca's ``feglmOffset()`` stops early, so its variances and APEs of a corrected model differ
  from fereg's at about 1e-6 (relative).
- The corrected model reports the deviance and log-likelihood at the corrected point (alpaca keeps
  the uncorrected deviance; fereg keeps it in ``deviance_uncorr``). Every variance (estimation-time
  ``vcov`` or post-hoc :meth:`vcov`) is evaluated at the corrected point; alpaca's
  ``vcov(type = "hessian")`` is ``vcov="iid", ssc=ssc(k_adj=False)``, its sandwich ``"hetero"`` and
  its clustered variance ``{"CRV1": ...}``, both with ``ssc(k_adj=False, G_adj=False)``.
- Lags (``L``) count rows within a group in time order, gaps ignored (alpaca's definition). Time
  order: classic two-way models sort the rows of an individual (first fixed effect) by the levels of
  the second fixed effect (input order for ties); one-way models keep the input order; network
  models sort the rows of a pair (third fixed effect) by the levels of the first and second fixed
  effects. Levels sort as numbers for numeric columns, as strings otherwise (categoricals: category
  order), so the second (classic) / first and second (network) variables must sort in time order;
  the same holds for ``time_id`` (time order = sorted values: use numbers, dates or ordered
  categoricals for calendar order). ``L`` must be smaller than the longest group (alpaca silently
  returns the ``L = 0`` result otherwise).
- With a finite population (``n_pop``) and groups dropped for perfect classification, the
  sampling-of-effects term of the APE variance sums over all observations, with zero partial effects
  for the dropped ones, consistently with the point estimate; alpaca and logitfe sum over the
  retained observations only, which overstates the variance. The same holds for the covariance
  term of weakly exogenous regressors (dropped observations of a retained individual, or pair,
  enter in time order). The numbers are identical when no group is dropped.
- Options without effect are errors (``L`` with the jackknife or with ``bias="time"``, ``multiple``
  partitions that permute no dimension the subpanels split, and ``weak_exo=True`` or
  ``sampling_fe="unrestricted"`` without ``n_pop``, whose terms are multiplied by 0); a two-way
  network model with ``L > 0`` only warns "L ignored", as alpaca computes it. ``n_pop`` is truncated
  to an integer (alpaca, R); ``float("inf")`` is an infinite population (factor 1).
- Frequency weights are an error; prior weights (aweights) are alpaca's weights.
- alpaca can apply ``biasCorr()`` to a corrected model again; fereg raises. alpaca's ``getAPEs()``
  silently ignores ``panel.structure`` and ``weak.exo`` for corrected models (they come from the
  correction); fereg raises when they are given and differ.
- Jackknife (alpaca has none): logitfe's split-panel and delete-one variants, with the
  dimensions logitfe swaps fixed (``js`` and ``sj`` with both biases, and the single-bias ``js``,
  ``sj``, ``jj``, ``double``: the individual-effect bias is removed with time-dimension subpanels,
  the time-effect bias with individual-dimension subpanels) and the individual-bias ``ss2`` that
  logitfe cannot run; a subpanel fit that loses a coefficient or does not converge is an error
  naming the subpanel (logitfe sets the coefficient to zero); ``multiple`` draws its random
  partitions with ``numpy.random.default_rng(seed)``, so they differ from logitfe's.
- With prior weights the APEs are unweighted averages while the scores are weighted (alpaca).
"""

from __future__ import annotations

import numbers
import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from ._binding import apes_design, bias_corr_design, bias_dims_design, pnorm, qnorm
from ._vcov import parse_vcov
from .results import Felogit, _coef_names

__all__ = ["FelogitBiasCorr", "APEs", "bias_corr", "get_apes"]

_METHODS = ("analytical", "jackknife")
_STRUCTURES = ("classic", "network")
_VARIANTS = ("ss2", "ss1", "js", "sj", "jj", "double")
_BIASES = ("both", "individual", "time")
_SPLITS = ("both", "individuals", "time")
_SAMPLING = ("independence", "unrestricted")
_NO_CRV3 = "CRV3 is not available after bias correction"  # the core's text
# The APE variance (alpaca's) and the jackknife index spaces do not depend on the model's vcov.
_IID = parse_vcov("iid")


def _choice(name: str, value, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, one of {', '.join(repr(c) for c in choices)}.")
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(repr(c) for c in choices)}, got '{value}'.")
    return value


def _count(name: str, value) -> int:
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _double_design(design: Any) -> Any:
    """``design`` with the two fixed-effect variables coded on one common level set (the sorted
    union of their labels) when both are coded (string / categorical) columns: the double jackknife
    drops the rows whose first or second fixed effect equals each id present in both. Numeric
    columns are compared by value in the core; other layouts are left to the core's checks."""
    from .design import FactorColumn

    if len(design.fe) != 2:
        return design
    (f1, _, _), (f2, _, _) = design.fe
    if len(f1) != 1 or len(f2) != 1:
        return design
    a, b = design.factors[f1[0]], design.factors[f2[0]]
    if a.kind != "codes" or b.kind != "codes":
        return design
    la, lb = list(a.labels), list(b.labels)
    if la == lb and a.base == b.base:
        return design

    def ascending(labels: list) -> bool:
        try:
            return all(x < y for x, y in zip(labels, labels[1:]))
        except TypeError:
            return False

    if not (ascending(la) and ascending(lb)):
        raise ValueError(
            f"jackknife='double' cannot match the levels of '{a.name}' and '{b.name}': give both "
            "variables the same type (or the same categories)."
        )
    try:
        union = sorted(set(la) | set(lb))
    except TypeError:
        raise ValueError(
            f"jackknife='double' cannot match the levels of '{a.name}' and '{b.name}': give both "
            "variables the same type (or the same categories)."
        ) from None
    pos = {v: i for i, v in enumerate(union)}

    def recode(fc) -> Any:
        rel = fc.values.astype(np.int64) - fc.base
        ok = (rel >= 0) & (rel < fc.nlevels)
        new = np.array([pos[v] for v in fc.labels], dtype=np.int64)
        codes = np.full(len(rel), -1, dtype=np.int32)
        codes[ok] = new[rel[ok]]
        return FactorColumn(fc.name, "codes", codes, len(union), 0, np.asarray(union, dtype=object))

    factors = list(design.factors)
    factors[f1[0]], factors[f2[0]] = recode(a), recode(b)
    return replace(design, factors=factors)


def _time_design(model: Felogit, time_id: str) -> tuple[Any, int]:
    """The design with the ``time_id`` column appended as a factor, and its index. Its levels
    are ordered as fixed-effect levels are (time order = sorted values: numbers and dates
    ascending, strings sorted, categoricals in category order)."""
    from .design import _factor_from_frame, _Frame

    if not isinstance(time_id, str):
        raise TypeError("time_id must be the name of a column of the data.")
    d = model._design
    fc = _factor_from_frame(_Frame(model._ctx.data), time_id)
    return replace(d, factors=list(d.factors) + [fc]), len(d.factors)


def _warn_notes(notes) -> None:
    for note in notes:
        warnings.warn(note)


def bias_corr(
    model: Felogit,
    L: int = 0,
    panel_structure: str = "classic",
    method: str = "analytical",
    jackknife: str | None = None,
    bias: str = "both",
    multiple: int = 0,
    split: str | None = None,
    time_id: str | None = None,
    seed: int | None = None,
) -> FelogitBiasCorr:
    """See :meth:`fereg.Felogit.bias_corr`."""
    from .estimation import core_options

    if isinstance(model, FelogitBiasCorr):
        raise ValueError("bias_corr() cannot be applied to a bias-corrected model.")
    L = _count("L", L)
    _choice("panel_structure", panel_structure, _STRUCTURES)
    _choice("method", method, _METHODS)
    _choice("bias", bias, _BIASES)
    multiple = _count("multiple", multiple)
    if jackknife is not None:
        _choice("jackknife", jackknife, _VARIANTS)
    if split is not None:
        _choice("split", split, _SPLITS)
    if seed is not None:
        seed = _count("seed", seed)
    # Arguments the core has no counterpart for; the model and option checks (structure, number
    # and kind of fixed effects, CRV3, bias choice, bandwidth, variant, time variable) are the
    # core's, with the texts R and Stata show too.
    jk = method == "jackknife"
    if not jk:
        for name, value in (("jackknife", jackknife), ("split", split), ("seed", seed), ("time_id", time_id)):
            if value is not None:
                raise ValueError(f"{name} is only used with method='jackknife'.")
        if multiple:
            raise ValueError("multiple is only used with method='jackknife'.")
    if not multiple:
        for name, value in (("split", split), ("seed", seed)):
            if value is not None:
                raise ValueError(f"{name} is only used with multiple > 0.")
    variant = jackknife or "ss2"

    design, time = model._design, None
    if time_id is not None:
        if len(design.fe) >= 2:
            raise ValueError(
                "time_id is only used by the jackknife of a one-way model (two-way models take the "
                "time order from the second fixed effect)."
            )
        design, time = _time_design(model, time_id)
    if jk and variant == "double":
        design = _double_design(design)
    options = {
        "method": method,
        "structure": panel_structure,
        "which": bias,
        "bandwidth": L,
        "time": time,
        "variant": variant,
        "partitions": [],
    }
    ctx = model._ctx
    core = core_options(ctx)
    if multiple:
        dims = bias_dims_design(design, bias=options, **core_options(ctx, vcov=_IID, lean=True))
        rng = np.random.default_rng(seed)
        split = split or "both"
        options["partitions"] = [
            (
                rng.permutation(dims["n_individuals"]) if split in ("both", "individuals") else None,
                rng.permutation(dims["n_periods"]) if split in ("both", "time") else None,
            )
            for _ in range(multiple)
        ]
    out, report = bias_corr_design(design, bias=options, **core)
    res = FelogitBiasCorr(
        ctx,
        out,
        report,
        design=design,
        options=options,
        jackknife=variant if jk else None,
        multiple=multiple,
        split=split if multiple else None,
        seed=seed,
        time_id=time_id,
    )
    _warn_notes(report["notes"])
    return res


def get_apes(
    model: Felogit,
    n_pop: float | None = None,
    panel_structure: str | None = None,
    sampling_fe: str = "independence",
    weak_exo: bool | None = None,
) -> APEs:
    """See :meth:`fereg.Felogit.get_apes`."""
    from .estimation import core_options

    if n_pop is not None:
        if not isinstance(n_pop, numbers.Real) or isinstance(n_pop, bool):
            raise TypeError("n_pop must be a number or None.")
        if np.isnan(n_pop) or n_pop < 1:
            raise ValueError("n_pop must be a number >= 1 (float('inf'): an infinite population).")
        # truncated to an integer, as alpaca (as.integer) and R; inf = infinite population
        n_pop = float(n_pop) if np.isinf(n_pop) else float(int(n_pop))
    if panel_structure is not None:
        _choice("panel_structure", panel_structure, _STRUCTURES)
    _choice("sampling_fe", sampling_fe, _SAMPLING)
    if weak_exo is not None and not isinstance(weak_exo, (bool, np.bool_)):
        raise TypeError("weak_exo must be a bool or None.")
    if n_pop is None:
        # both terms are multiplied by the finite-population factor, 0 without n_pop
        if weak_exo:
            raise ValueError("weak_exo=True has no effect without n_pop")
        if sampling_fe == "unrestricted":
            raise ValueError("sampling_fe='unrestricted' has no effect without n_pop")
    # model checks (logit, fixed effects, structure): the core's, with the texts of R and Stata

    corrected = isinstance(model, FelogitBiasCorr)
    design, at = model._design, None
    if corrected:
        if panel_structure is not None and panel_structure != model.panel_structure:
            raise ValueError(
                f"panel_structure is set by the bias correction ('{model.panel_structure}'); "
                "leave it out or pass the same value."
            )
        panel_structure = model.panel_structure
        if model.method == "analytical":
            forced = model._bias_options["bandwidth"] > 0  # alpaca: weak.exo = L > 0
            if weak_exo is not None and bool(weak_exo) != forced:
                raise ValueError(
                    "weak_exo is set by the analytical bias correction (weak_exo = L > 0, here "
                    f"{forced}); leave it out or pass the same value."
                )
            weak_exo = forced
        design = model._bias_design
        at = {
            "bias": model._bias_options,
            "coef": np.ascontiguousarray(model._beta_hat, dtype=np.float64),
            "ape_jackknife": model._ape_jackknife,
        }
    panel_structure = panel_structure or "classic"
    weak_exo = bool(weak_exo) if weak_exo is not None else False
    options = {
        "n_pop": n_pop,
        "structure": panel_structure,
        "sampling": sampling_fe,
        "weak_exo": weak_exo,
        "at": at,
    }
    # lean: the core's internal refit keeps no per-observation output or state
    out = apes_design(design, ape=options, **core_options(model._ctx, vcov=_IID, lean=True))
    _warn_notes(out["notes"])
    return APEs(model, out, n_pop=n_pop, sampling_fe=sampling_fe)


def _correction_label(m: FelogitBiasCorr) -> str:
    if m.method == "analytical":
        where = ", network" if m.panel_structure == "network" else ""
        return f"analytical{where} (L = {m.bandwidth}, bias = '{m.bias}')"
    text = f"jackknife {m.jackknife} (bias = '{m.bias}')"
    if m.multiple:
        seed = "" if m.seed is None else f", seed = {m.seed}"
        text += f", {m.multiple} random partitions (split = '{m.split}'{seed})"
    return text


class FelogitBiasCorr(Felogit):
    """A logit model at bias-corrected coefficients (:meth:`Felogit.bias_corr`).

    Every method of :class:`Felogit` works and is evaluated at the corrected point: coefficients
    ``coef()``, the variance chosen at estimation and post-hoc :meth:`vcov` (except CRV3, which
    would refit uncorrected models), ``se()``, ``tstat()``, ``pvalue()``, ``confint()``, ``tidy()``,
    ``summary()``, ``fixef()`` (re-estimated at the corrected coefficients), ``predict()``,
    ``resid()``, the deviance / log-likelihood, and :func:`fereg.etable`.

    Attributes (alpaca's ``biasCorr`` fields): ``coef_uncorr`` (uncorrected coefficients),
    ``bias_term`` (uncorrected minus corrected coefficients, alpaca's ``bias.term``),
    ``panel_structure``, ``bandwidth`` (the requested ``L``, as alpaca; a two-way network model
    ignores it), ``method`` ("analytical" | "jackknife"), ``jackknife`` (variant, None
    for the analytical correction), ``bias`` (the bias corrected: "both" | "individual" | "time";
    a one-way model has the individual bias only), plus ``multiple``, ``split``, ``seed``,
    ``time_id``, ``deviance_uncorr``, ``n_fits``
    (jackknife subpanel fits) and ``N_full`` (observations before the fixed-effect based drops).
    """

    def __init__(
        self,
        ctx,
        out,
        report: dict,
        *,
        design,
        options: dict,
        jackknife: str | None,
        multiple: int,
        split: str | None,
        seed: int | None,
        time_id: str | None,
    ) -> None:
        super().__init__(ctx, out)
        names = self._coefnames
        self.coef_uncorr = pd.Series(np.asarray(report["coef_uncorr"], dtype=float), index=names, name="Estimate")
        self.bias_term = pd.Series(np.asarray(report["bias_term"], dtype=float), index=names, name="bias_term")
        self.panel_structure = report["structure"]
        self.bandwidth = int(options["bandwidth"])
        self.method = options["method"]
        self.jackknife = jackknife
        self.bias = report["which"]
        self.multiple = multiple
        self.split = split
        self.seed = seed
        self.time_id = time_id
        self.deviance_uncorr = report["deviance_uncorr"]
        self.N_full = int(report["n_full"])
        self.n_fits = int(report["n_fits"])
        ape_jk = report["ape_jackknife"]
        self._ape_jackknife = None if ape_jk is None else np.ascontiguousarray(ape_jk, dtype=np.float64)
        self._bias_design = design
        self._bias_options = options
        self._correction_line = _correction_label(self)

    def _refit_vcov(self, spec, cluster_factors):
        raise ValueError(_NO_CRV3)

    def __repr__(self) -> str:
        return f"<fereg.{type(self).__name__}: {self._model_name}, {self._correction_line}>"


class APEs:
    """Average partial effects of a logit model (:meth:`Felogit.get_apes`; alpaca ``getAPEs``).

    ``coef()`` / ``se()`` / ``tstat()`` / ``pvalue()`` / ``confint()`` / ``tidy()`` / ``summary()``
    as for a model (normal inference); ``vcov`` is the variance matrix (a DataFrame). Attributes
    (alpaca's fields): ``bias_term`` (the APE bias term of an analytically corrected model,
    subtracted from the APEs; None otherwise), ``panel_structure``, ``sampling_fe``, ``weak_exo``
    (as used), ``bandwidth`` (``L`` of the bias correction, 0 for an uncorrected model), ``n_pop``
    (as used: truncated to an integer), plus ``adj`` (finite-population factor
    (n_pop - N*)/(n_pop - 1), 0 without ``n_pop``), ``binary`` (regressors treated as binary: all
    values 0 or 1) and ``N_full`` (N*, the divisor of the averages).
    """

    def __init__(self, model: Felogit, out: dict, *, n_pop: float | None, sampling_fe: str) -> None:
        names = _coef_names(model._design, out["coef_source"])
        self._coefnames = names
        self._delta = np.asarray(out["delta"], dtype=float)
        self.vcov = pd.DataFrame(np.asarray(out["vcov"], dtype=float), index=names, columns=names)
        bt = out["bias_term"]
        self.bias_term = None if bt is None else pd.Series(np.asarray(bt, dtype=float), index=names, name="bias_term")
        self.binary = pd.Series(np.asarray(out["binary"], dtype=bool), index=names, name="binary")
        self.panel_structure = out["structure"]
        self.sampling_fe = sampling_fe
        self.weak_exo = bool(out["weak_exo_used"])
        self.bandwidth = int(model._bias_options["bandwidth"]) if isinstance(model, FelogitBiasCorr) else 0
        self.n_pop = n_pop
        self.adj = float(out["adj"])
        self.N_full = int(out["n_full"])
        self._notes = list(out["notes"])
        self._depvar = model._depvar
        self._fixef = model._fixef
        self._model_name = model._model_name
        self._sample_split_var = model._sample_split_var
        self._sample_split_value = model._sample_split_value
        self._correction_line = getattr(model, "_correction_line", None)

    def _inference_label(self) -> str:
        label = "delta method"
        if self.adj > 0:
            extra = []
            if self.sampling_fe == "independence":
                extra.append("sampling of effects")
            if self.weak_exo:
                extra.append("weak exogeneity")
            n_pop = int(self.n_pop) if float(self.n_pop).is_integer() else self.n_pop
            label += "".join(f" + {e}" for e in extra) + f" (n_pop = {n_pop}, adj = {self.adj:.4g})"
        return label

    def tidy(self, alpha: float = 0.05) -> pd.DataFrame:
        """APE table indexed by ``Coefficient`` (columns as a model's ``tidy()``)."""
        se = np.sqrt(np.diagonal(self.vcov.to_numpy()))
        z = self._delta / se
        crit = abs(qnorm(alpha / 2))
        lb, ub = alpha / 2, 1 - alpha / 2
        return pd.DataFrame(
            {
                "Coefficient": self._coefnames,
                "Estimate": self._delta,
                "Std. Error": se,
                "t value": z,
                "Pr(>|t|)": 2.0 * np.asarray(pnorm(-np.abs(z))),
                f"{lb * 100:.1f}%": self._delta - crit * se,
                f"{ub * 100:.1f}%": self._delta + crit * se,
            }
        ).set_index("Coefficient")

    def coef(self) -> pd.Series:
        return self.tidy()["Estimate"]

    def se(self) -> pd.Series:
        return self.tidy()["Std. Error"]

    def tstat(self) -> pd.Series:
        return self.tidy()["t value"]

    def pvalue(self) -> pd.Series:
        return self.tidy()["Pr(>|t|)"]

    def confint(self, alpha: float = 0.05) -> pd.DataFrame:
        return self.tidy(alpha=alpha).iloc[:, -2:]

    def summary(self, digits: int = 3) -> None:
        """Prints the APE table in the layout of :func:`fereg.summary`."""
        from .summary import markdown_table

        print("###")
        print("")
        print("Estimation:  Average partial effects (Logit)")
        line = f"Dep. var.: {self._depvar}"
        if self._fixef is not None:
            line += f", Fixed effects: {self._fixef}"
        print(line)
        if self._sample_split_var is not None:
            print(f"sample: {self._sample_split_var} = {self._sample_split_value}")
        if self._correction_line is not None:
            print(f"Bias correction: {self._correction_line}")
        print("Inference: ", self._inference_label())
        print("Observations: ", self.N_full)
        print("")
        print(markdown_table(self.tidy().round(digits), floatfmt=f".{digits}f"))
        print("---")

    def __repr__(self) -> str:
        return f"<fereg.APEs: {self._model_name}>"
