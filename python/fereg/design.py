"""Design construction with pyfixest 0.60.0 names and sample rules.

`build_design` turns one parsed `Formula` plus a pandas or polars DataFrame
into the host columns the Rust core consumes. Plain float64 columns are
passed through without copying; NaN marks a missing value everywhere. Terms
are encoded natively by a port of formulaic's rank-reduction rules (the
engine pyfixest uses), so coefficient names and dummy columns match pyfixest.
Factor expressions the native evaluator does not know are delegated to
formulaic (optional dependency) for the whole formula part.
"""

from __future__ import annotations

import ast
import itertools
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .formula import Formula, FormulaSyntaxError, Term, _expr_variables, rhs_terms

__all__ = [
    "Design",
    "FactorColumn",
    "build_design",
    "factor_column",
    "term_labels",
]


@dataclass
class FactorColumn:
    name: str  # data column (or term text)
    kind: str  # "f64" | "i64" | "codes"  (maps to FactorCol::F64 / I64 / Codes)
    values: np.ndarray  # contiguous float64 / int64 / int32 codes
    nlevels: int = 0
    base: int = 0  # Codes: level = code - base; anything else = missing
    labels: object = None  # level labels for "codes" (index = code - base)

    def labels_at(self, rows: np.ndarray) -> list[str | None]:
        """pyfixest `fixef()` level keys of `rows` (None where missing)."""
        v = self.values[rows]
        if self.kind == "codes":
            lab = self.labels
            idx = v.astype(np.int64) - self.base
            return [str(lab[i]) if 0 <= i < len(lab) else None for i in idx.tolist()]
        if self.kind == "i64":
            return [str(x) for x in v.tolist()]
        return [None if np.isnan(x) else str(x) for x in v.tolist()]


@dataclass
class Design:
    n: int
    y: np.ndarray | None
    y_name: str
    x: list[np.ndarray]
    # pyfixest coefficient names (i(): "f::2", C(): "C(f)[T.2]", ":" interactions)
    x_names: list[str]
    endog: list[np.ndarray]
    endog_names: list[str]
    inst: list[np.ndarray]  # excluded instruments only
    inst_names: list[str]
    intercept: bool
    factors: list[FactorColumn]  # distinct; referenced by fe and cluster
    # (factor indices, intercept, slope indices) per FeTerm
    fe: list[tuple[list[int], bool, list[int]]]
    fe_names: list[str]  # FE term labels as written
    slopes: list[np.ndarray]
    slope_names: list[str]
    weights: np.ndarray | None
    offset: np.ndarray | None
    cluster: list[list[int]]  # cluster dims -> factor indices (interaction when >1)
    cluster_names: list[str]
    # True = drop: missing categorical source, NA produced by a transform,
    # rows formulaic dropped, or (with `like`) a categorical level unseen in training
    drop: np.ndarray | None
    # pyfixest `_coefnames` before collinearity removal: "Intercept" first when
    # reported, then second-stage columns in formulaic order (IV: endog included)
    coefnames: list[str] = field(default_factory=list)
    # encoding state (categorical levels per factor) reused by `build_design(like=...)`
    levels: dict = field(default_factory=dict)
    # Newey-West / Driscoll-Kraay time and panel variables (factor indices) and Conley
    # coordinates (degrees) of the variance matrix; None = not used
    time: int | None = None
    panel: int | None = None
    lat: np.ndarray | None = None
    lon: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Data frame access (pandas / polars)
# ---------------------------------------------------------------------------


def _is_polars(data) -> bool:
    return type(data).__module__.split(".")[0] == "polars"


class _Frame:
    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            self.polars = False
        elif _is_polars(data) and type(data).__name__ == "DataFrame":
            self.polars = True
        else:
            raise TypeError("data must be a pandas or polars DataFrame")
        self.data = data
        self.n = data.height if self.polars else len(data)
        self.columns = set(data.columns)

    def require(self, name: str, what: str = "Variable") -> None:
        if name not in self.columns:
            raise KeyError(f"{what} '{name}' not found in data.")

    def series(self, name: str) -> pd.Series:
        """Column as a pandas Series (polars columns are converted)."""
        self.require(name)
        if self.polars:
            return self.data.get_column(name).to_pandas()
        return self.data[name]

    def numeric(self, name: str) -> np.ndarray | None:
        """float64 values (NaN = missing), zero-copy for contiguous float64
        columns; None when the column is not numeric (categorical). Polars
        columns follow `to_pandas()` (what pyfixest converts them with): a
        Boolean column with nulls becomes object, i.e. categorical."""
        self.require(name)
        if self.polars:
            import polars as pl

            s = self.data.get_column(name)
            dt = s.dtype
            if dt == pl.Float64:
                return np.ascontiguousarray(s.to_numpy())
            if dt.is_numeric() or (dt == pl.Boolean and s.null_count() == 0):
                return s.cast(pl.Float64).to_numpy()
            return None
        return _pd_numeric(self.data[name])

    def pandas_columns(self, cols: list[str]) -> pd.DataFrame:
        """The given columns as a pandas DataFrame with a RangeIndex."""
        if self.polars:
            return self.data.select(cols).to_pandas()
        return self.data[cols].reset_index(drop=True)


def _pd_numeric(s: pd.Series) -> np.ndarray | None:
    dt = s.dtype
    if isinstance(dt, np.dtype):
        if dt == np.float64:
            return np.ascontiguousarray(s.to_numpy())
        if dt.kind in "fiub":
            return s.to_numpy().astype(np.float64)
        return None
    if isinstance(dt, pd.CategoricalDtype):
        return None
    if pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_bool_dtype(dt):
        return s.to_numpy(dtype=np.float64, na_value=np.nan)
    return None


def _is_numeric_series(s: pd.Series) -> bool:
    """pyfixest `_is_numeric`: numeric dtype and not boolean."""
    return pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(
        s.dtype
    )


# ---------------------------------------------------------------------------
# Factor (FE / cluster) columns
# ---------------------------------------------------------------------------


def _codes_column(name: str, values, missing: np.ndarray | None) -> FactorColumn:
    """Factorize `values` (rows in `missing` -> -1) into int32 codes with
    levels in sorted order (formulaic's category order, which fixes pyfixest's
    fixef() level order and reference level)."""
    if missing is None:
        codes, uniq = pd.factorize(values, sort=True, use_na_sentinel=True)
    else:
        codes = np.full(len(values), -1, dtype=np.int64)
        codes[~missing], uniq = pd.factorize(
            values[~missing], sort=True, use_na_sentinel=True
        )
    lab = np.asarray(uniq, dtype=object)
    return FactorColumn(name, "codes", codes.astype(np.int32), len(lab), 0, lab)


def _factor_from_series(name: str, s: pd.Series) -> FactorColumn:
    dt = s.dtype
    if isinstance(dt, np.dtype):
        if dt == np.int64:
            return FactorColumn(name, "i64", np.ascontiguousarray(s.to_numpy()))
        if dt.kind == "i" or (dt.kind == "u" and dt.itemsize < 8):
            return FactorColumn(name, "i64", s.to_numpy().astype(np.int64))
        if dt.kind == "f":
            return FactorColumn(
                name, "f64", np.ascontiguousarray(s.to_numpy(), dtype=np.float64)
            )
        if dt.kind == "u" and (len(s) == 0 or s.max() < 2**63):
            return FactorColumn(name, "i64", s.to_numpy().astype(np.int64))
        return _codes_column(name, s.to_numpy(), None)
    if isinstance(dt, pd.CategoricalDtype):
        codes = np.ascontiguousarray(s.cat.codes.to_numpy(), dtype=np.int32)
        cats = np.asarray(dt.categories, dtype=object)
        return FactorColumn(name, "codes", codes, len(cats), 0, cats)
    if pd.api.types.is_integer_dtype(dt):
        missing = s.isna().to_numpy()
        values = s.to_numpy(dtype=np.int64, na_value=0)
        if not missing.any():
            return FactorColumn(name, "i64", values)
        return _codes_column(name, values, missing)
    if pd.api.types.is_float_dtype(dt):
        return FactorColumn(name, "f64", s.to_numpy(dtype=np.float64, na_value=np.nan))
    return _codes_column(name, s, None)


def _factor_from_frame(frame: _Frame, name: str) -> FactorColumn:
    frame.require(name)
    if frame.polars:
        import polars as pl

        s = frame.data.get_column(name)
        dt = s.dtype
        if dt.is_integer() and s.null_count() == 0:
            arr = s.to_numpy() if dt == pl.Int64 else s.cast(pl.Int64).to_numpy()
            return FactorColumn(name, "i64", np.ascontiguousarray(arr))
        if dt.is_numeric():
            # integers with nulls become float64 NaN under to_pandas(), as in pyfixest
            return FactorColumn(
                name, "f64", np.ascontiguousarray(s.cast(pl.Float64).to_numpy())
            )
        return _factor_from_series(name, s.to_pandas())
    return _factor_from_series(name, frame.data[name])


def _factor_missing(fc: FactorColumn) -> np.ndarray | None:
    if fc.kind == "f64":
        return np.isnan(fc.values)
    if fc.kind == "codes":
        rel = fc.values.astype(np.int64) - fc.base
        return (rel < 0) | (rel >= fc.nlevels)
    return None


def _factor_codes(fc: FactorColumn) -> tuple[np.ndarray, np.ndarray]:
    """Dense codes (-1 missing) and their labels."""
    if fc.kind == "codes":
        miss = _factor_missing(fc)
        codes = fc.values.astype(np.int64) - fc.base
        codes[miss] = -1
        return codes, np.asarray(fc.labels, dtype=object)
    codes, uniq = pd.factorize(fc.values, use_na_sentinel=True)
    return codes.astype(np.int64), np.asarray(uniq, dtype=object)


def factor_column(data, name: str) -> FactorColumn:
    """FactorColumn for a post-hoc cluster variable: "a" or an interaction
    "a^b" (one combined `codes` column with labels "va^vb")."""
    frame = _Frame(data)
    parts = [p.strip() for p in name.split("^")]
    if len(parts) == 1:
        return _factor_from_frame(frame, parts[0])
    keys = np.zeros(frame.n, dtype=np.int64)
    missing = np.zeros(frame.n, dtype=bool)
    label_sets = []
    for p in parts:
        codes, labels = _factor_codes(_factor_from_frame(frame, p))
        missing |= codes < 0
        # re-factorize the running key so the mixed-radix product never overflows
        keys, _ = pd.factorize(keys * max(len(labels), 1) + np.maximum(codes, 0))
        label_sets.append((codes, labels))
    ok = ~missing
    combined = np.full(frame.n, -1, dtype=np.int64)
    combined[ok], _ = pd.factorize(keys[ok])
    # factorize numbers levels by first appearance: the first row of level k
    # gives its label
    _, first = np.unique(combined[ok], return_index=True)
    first_rows = np.nonzero(ok)[0][first]
    labels = np.array(
        [
            "^".join(str(lab[c[r]]) for c, lab in label_sets)
            for r in first_rows.tolist()
        ],
        dtype=object,
    )
    return FactorColumn(
        name, "codes", combined.astype(np.int32), len(labels), 0, labels
    )


def _is_time_like(frame: _Frame, name: str) -> bool:
    """pyfixest `_vcov_hac`: the time variable must be numeric (not boolean) or a date."""
    if frame.polars:
        import polars as pl

        dt = frame.data.get_column(name).dtype
        return (dt.is_numeric() and dt != pl.Boolean) or dt == pl.Date or dt == pl.Datetime
    s = frame.data[name]
    return _is_numeric_series(s) or pd.api.types.is_datetime64_any_dtype(s.dtype)


def _time_factor(frame: _Frame, name: str) -> FactorColumn:
    frame.require(name)
    if not _is_time_like(frame, name):
        raise ValueError(
            "The time variable must be numeric or date, else we cannot sort by time."
        )
    return _factor_from_frame(frame, name)


def _coordinate(frame: _Frame, name: str) -> np.ndarray:
    values = frame.numeric(name)
    if values is None:
        raise TypeError(f"The coordinate variable '{name}' must be numeric (degrees).")
    return values


def vcov_inputs(data, cluster: list[str], columns: dict[str, str | None]) -> dict:
    """Post-hoc variance inputs over the rows of `data`: keyword arguments of the
    core's `FitState.vcov` (`factors`, `factor_names`, `cluster`, `time`,
    `panel`, `lat`, `lon`). `cluster` dimensions may be interactions "a^b";
    `columns` names the time / panel / coordinate columns (None = not used)."""
    frame = _Frame(data)
    factors: list[FactorColumn] = [factor_column(data, name) for name in cluster]
    out = {"cluster": [[i] for i in range(len(cluster))], "time": None, "panel": None}
    if columns.get("time") is not None:
        out["time"] = len(factors)
        factors.append(_time_factor(frame, columns["time"]))
    if columns.get("panel") is not None:
        out["panel"] = len(factors)
        factors.append(_factor_from_frame(frame, columns["panel"]))
    out["factors"] = factors
    for key in ("lat", "lon"):
        name = columns.get(key)
        out[key] = None if name is None else _coordinate(frame, name)
    return out



def term_labels(design: Design, k: int, rows: np.ndarray) -> list[str | None]:
    """pyfixest-style level keys of FE term `k` for `rows` ("va^vb" for
    interactions; None where missing)."""
    per = [design.factors[i].labels_at(rows) for i in design.fe[k][0]]
    return [
        None if any(v is None for v in vals) else "^".join(vals) for vals in zip(*per)
    ]


# ---------------------------------------------------------------------------
# Factor evaluation
# ---------------------------------------------------------------------------


class _NotNative(Exception):
    """The factor needs formulaic."""


def _pf_log(array):
    """pyfixest's `log`: NaN (with a warning) for non-positive or non-finite input."""
    array = np.asarray(array, dtype=np.float64)
    result = np.full_like(array, np.nan, dtype="float64")
    with np.errstate(invalid="ignore"):
        valid = (array > 0.0) & np.isfinite(array)
    if not valid.all():
        warnings.warn(
            f"{np.sum(~valid)} rows with infinite values detected. These rows are dropped from the model.",
            stacklevel=2,
        )
    np.log(array, out=result, where=valid)
    return result


# formulaic TRANSFORMS (numeric ones) with pyfixest's log; sqrt and abs are
# fereg additions (pyfixest 0.60.0 has no `sqrt`; `abs` is Python's builtin)
_FUNCS = {
    "log": _pf_log,
    "log10": np.log10,
    "log2": np.log2,
    "exp": np.exp,
    "exp2": np.exp2,
    "exp10": lambda x: np.power(x, 10),
    "sqrt": np.sqrt,
    "abs": np.abs,
    "I": lambda x: x,
}
_MODULES = {"np": np, "numpy": np}
_BINOPS = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.true_divide,
    ast.FloorDiv: np.floor_divide,
    ast.Mod: np.mod,
    ast.Pow: np.power,
}
_CMPOPS = {
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
}


class _Num:
    spans = False

    def __init__(self, name: str, values: np.ndarray, derived: bool):
        self.name = name
        self.values = values
        self.derived = derived  # produced by a transform: its NaNs go into `drop`

    def missing(self) -> np.ndarray:
        return np.isnan(self.values)

    def encode(self, reduced: bool) -> list[tuple[str, np.ndarray]]:
        return [(self.name, self.values)]


class _Cat:
    """Categorical source: pandas factorize codes plus formulaic level rules
    (categorical dtype: all categories in order; otherwise sorted levels seen
    in the estimation sample)."""

    def __init__(self, s: pd.Series):
        if isinstance(s.dtype, pd.CategoricalDtype):
            self.raw = s.cat.codes.to_numpy().astype(np.int64)
            self.uniques = s.cat.categories
            self.fixed = True
        else:
            codes, self.uniques = pd.factorize(s, use_na_sentinel=True)
            self.raw = codes.astype(np.int64)
            self.fixed = False
        self.levels = None
        self.codes = None

    def missing(self) -> np.ndarray:
        return self.raw < 0

    def finalize(self, kept: np.ndarray | None, levels=None) -> np.ndarray | None:
        """Fix levels (from `kept` rows, or given) and remap codes; returns the
        rows whose value is not among given `levels` (unseen), if any."""
        if levels is not None:
            self.levels = pd.Index(levels)
            remap = self.levels.get_indexer(pd.Index(self.uniques))
        elif self.fixed:
            self.levels = pd.Index(self.uniques)
            self.codes = self.raw
            return None
        else:
            valid = self.raw >= 0
            if kept is not None:
                valid &= kept
            present = np.bincount(self.raw[valid], minlength=len(self.uniques)) > 0
            self.levels = pd.Categorical(self.uniques[present]).categories
            remap = self.levels.get_indexer(pd.Index(self.uniques))
        remap = np.append(remap, -1)  # raw -1 (missing) indexes the appended slot
        self.codes = remap[self.raw]
        if levels is None:
            return None
        return (self.codes < 0) & (self.raw >= 0)


def _dummies(cat: _Cat, select: list[int]) -> list[np.ndarray]:
    """0/1 float64 columns for level indices `select` (one scatter pass)."""
    k = len(select)
    if k == 0:
        return []
    codes = cat.codes
    pos = np.full(len(cat.levels) + 1, -1, dtype=np.int64)
    pos[np.asarray(select, dtype=np.int64)] = np.arange(k)
    row_pos = pos[codes]  # codes == -1 hits the trailing -1 slot
    rows = np.nonzero(row_pos >= 0)[0]
    out = np.zeros((k, len(codes)))
    out[row_pos[rows], rows] = 1.0
    return list(out)


class _CatF:
    """Treatment-coded categorical factor (string columns, `C(x)`)."""

    spans = True

    def __init__(self, name: str, cat: _Cat):
        self.name = name
        self.cat = cat

    def sources(self) -> list[_Cat]:
        return [self.cat]

    def encode(self, reduced: bool) -> list[tuple[str, np.ndarray]]:
        levels = list(self.cat.levels)
        select = list(range(1, len(levels))) if reduced else list(range(len(levels)))
        fmt = "{}[T.{}]" if reduced else "{}[{}]"
        cols = _dummies(self.cat, select)
        return [(fmt.format(self.name, levels[j]), c) for j, c in zip(select, cols)]


def _level_index(levels, value, what: str) -> int:
    for j, lev in enumerate(levels):
        if lev == value:
            return j
    raise ValueError(
        f"Value `{value}` for `TreatmentContrasts.base` is not among the provided levels ({what})."
    )


class _IFactor:
    """fixest/pyfixest `i()` (pyfixest `factor_interaction`)."""

    def __init__(
        self,
        fname,
        cat1,
        vname=None,
        var2=None,
        cat2=None,
        ref=None,
        ref2=None,
        keep=None,
        keep2=None,
    ):
        self.fname = fname
        self.cat1 = cat1
        self.vname = vname
        self.var2 = var2  # numeric interaction values
        self.cat2 = cat2  # categorical second factor
        self.ref, self.ref2 = ref, ref2
        self.keep, self.keep2 = keep, keep2
        self.spans = var2 is None

    def sources(self) -> list[_Cat]:
        return [self.cat1] + ([self.cat2] if self.cat2 is not None else [])

    def _select(self, cat: _Cat, ref, drop_first: bool, keep) -> list[int]:
        levels = list(cat.levels)
        idx = list(range(len(levels)))
        if ref is not None:
            idx.remove(_level_index(levels, ref, "ref"))
        elif drop_first and idx:
            idx = idx[1:]
        if keep is not None:
            wanted = set(map(str, keep))
            idx = [j for j in idx if str(levels[j]) in wanted or levels[j] in keep]
        return idx

    def encode(self, reduced: bool) -> list[tuple[str, np.ndarray]]:
        l1 = list(self.cat1.levels)
        if self.var2 is None and self.cat2 is None:
            sel = self._select(self.cat1, self.ref, reduced, self.keep)
            cols = _dummies(self.cat1, sel)
            return [(f"{self.fname}::{l1[j]}", c) for j, c in zip(sel, cols)]
        if self.var2 is not None:
            sel = self._select(self.cat1, self.ref, False, self.keep)
            cols = _dummies(self.cat1, sel)
            return [
                (f"{self.fname}::{l1[j]}:{self.vname}", c * self.var2)
                for j, c in zip(sel, cols)
            ]
        l2 = list(self.cat2.levels)
        sel1 = self._select(self.cat1, self.ref, False, self.keep)
        sel2 = self._select(self.cat2, self.ref2, False, self.keep2)
        d1 = _dummies(self.cat1, sel1)
        d2 = _dummies(self.cat2, sel2)
        base1 = self.ref if self.ref is not None else (l1[0] if l1 else None)
        base2 = self.ref2 if self.ref2 is not None else (l2[0] if l2 else None)
        dropped = f"{self.fname}::{base1}:{self.vname}::{base2}"
        out = []
        for j1, c1 in zip(sel1, d1):
            for j2, c2 in zip(sel2, d2):
                name = f"{self.fname}::{l1[j1]}:{self.vname}::{l2[j2]}"
                if name != dropped:
                    out.append((name, c1 * c2))
        return out


_I_KWARGS = {"ref", "ref2", "bin", "bin2", "keep", "keep2", "var2"}


def _apply_bin(s: pd.Series, bins) -> pd.Series:
    """pyfixest `_apply_binning`: new_level -> [old levels]; others unchanged."""
    if bins is None:
        return s
    if not isinstance(bins, dict):
        raise FormulaSyntaxError(
            "i(): bin must be a dict mapping new level -> list of old levels"
        )
    mapping = {old: new for new, olds in bins.items() for old in olds}
    return s.replace(mapping)


class _Evaluator:
    """Evaluates factors natively and caches them by expression."""

    def __init__(self, frame: _Frame):
        self.frame = frame
        self.cache: dict[str, object] = {}

    def factor(self, f) -> object:
        """`f` is a formula.Factor."""
        if f.expr not in self.cache:
            self.cache[f.expr] = self._eval(f)
        return self.cache[f.expr]

    def _eval(self, f):
        if f.kind == "name":
            return self.lookup(f.expr)
        if f.kind != "python":
            raise _NotNative(f.expr)
        tree = ast.parse(f.expr, mode="eval").body
        if isinstance(tree, ast.Call) and isinstance(tree.func, ast.Name):
            if tree.func.id == "C":
                return self._c(tree, f.expr)
            if tree.func.id == "i":
                return self._i(tree)
        if "`" in f.expr:
            raise _NotNative(f.expr)
        value = self._expr(tree)
        value = np.asarray(value)
        if value.shape != (self.frame.n,):
            raise _NotNative(f.expr)
        return _Num(f.expr, value.astype(np.float64), derived=True)

    def lookup(self, name: str):
        values = self.frame.numeric(name)
        if values is not None:
            return _Num(name, values, derived=False)
        return _CatF(name, _Cat(self.frame.series(name)))

    def _c(self, call: ast.Call, expr: str):
        if (
            len(call.args) != 1
            or call.keywords
            or not isinstance(call.args[0], ast.Name)
        ):
            raise _NotNative(expr)
        name = call.args[0].id
        return _CatF(expr, _Cat(self.frame.series(name)))

    def _i(self, call: ast.Call):
        args = list(call.args)
        kwargs = {}
        for kw in call.keywords:
            if kw.arg not in _I_KWARGS:
                raise FormulaSyntaxError(
                    f"i() got an unexpected keyword argument '{kw.arg}'"
                )
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except ValueError as e:
                raise FormulaSyntaxError(f"i(): `{kw.arg}` must be a literal") from e
        if "var2" in kwargs:
            raise FormulaSyntaxError("i(): pass the second variable positionally")
        if not 1 <= len(args) <= 2 or not all(isinstance(a, ast.Name) for a in args):
            raise FormulaSyntaxError("i() takes one or two column names")
        fname = args[0].id
        cat1 = _Cat(_apply_bin(self.frame.series(fname), kwargs.get("bin")))
        common = {"ref": kwargs.get("ref"), "keep": kwargs.get("keep")}
        if len(args) == 1:
            return _IFactor(fname, cat1, **common)
        vname = args[1].id
        s2 = self.frame.series(vname)
        if (
            kwargs.get("ref2") is None
            and kwargs.get("bin2") is None
            and _is_numeric_series(s2)
        ):
            return _IFactor(
                fname, cat1, vname, var2=self.frame.numeric(vname), **common
            )
        cat2 = _Cat(_apply_bin(s2, kwargs.get("bin2")))
        return _IFactor(
            fname,
            cat1,
            vname,
            cat2=cat2,
            ref2=kwargs.get("ref2"),
            keep2=kwargs.get("keep2"),
            **common,
        )

    def _expr(self, node):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.frame.columns:
                values = self.frame.numeric(node.id)
                if values is None:
                    raise _NotNative(node.id)
                return values
            raise _NotNative(node.id)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            with np.errstate(all="ignore"):
                return _BINOPS[type(node.op)](
                    self._expr(node.left), self._expr(node.right)
                )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operand = self._expr(node.operand)
            return np.negative(operand) if isinstance(node.op, ast.USub) else operand
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and type(node.ops[0]) in _CMPOPS
        ):
            with np.errstate(all="ignore"):
                return _CMPOPS[type(node.ops[0])](
                    self._expr(node.left), self._expr(node.comparators[0])
                )
        if isinstance(node, ast.Call) and not node.keywords:
            func = self._func(node.func)
            args = [self._expr(a) for a in node.args]
            with np.errstate(all="ignore"):
                return func(*args)
        if isinstance(node, ast.Attribute):
            value = self._attr(node)
            if isinstance(value, (int, float)):
                return value
        raise _NotNative(ast.unparse(node))

    def _attr(self, node: ast.Attribute):
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in _MODULES
            and node.value.id not in self.frame.columns
        ):
            obj = getattr(_MODULES[node.value.id], node.attr, None)
            if obj is not None:
                return obj
        raise _NotNative(ast.unparse(node))

    def _func(self, node):
        if (
            isinstance(node, ast.Name)
            and node.id in _FUNCS
            and node.id not in self.frame.columns
        ):
            return _FUNCS[node.id]
        if isinstance(node, ast.Attribute):
            obj = self._attr(node)
            if isinstance(obj, np.ufunc) or (
                callable(obj) and node.attr not in ("where", "select")
            ):
                return obj
        raise _NotNative(ast.unparse(node))


# ---------------------------------------------------------------------------
# formulaic rank reduction (materializers/base.py `_get_scoped_terms`)
# ---------------------------------------------------------------------------

# A scoped factor is (expr, reduced); a scoped term a tuple of them. Scoped
# terms compare as sets of scoped factors.


def _st_key(st: tuple) -> tuple:
    return tuple(sorted(st))


def _oset_add(terms: list, items) -> list:
    keys = {_st_key(t) for t in terms}
    out = list(terms)
    for t in items:
        k = _st_key(t)
        if k not in keys:
            keys.add(k)
            out.append(t)
    return out


def _simplify(scoped_terms: list) -> list:
    """formulaic `_simplify_scoped_terms`: (anything):(reduced) + (anything)
    -> (anything):(full)."""
    terms: list = []
    for st in sorted(scoped_terms, key=len):
        factors = set(st)
        combined = False
        for existing in terms:
            cofactors = set(existing)
            diff = factors - cofactors
            if len(factors) - 1 != len(cofactors) or len(diff) != 1:
                continue
            new = next(iter(diff))
            if new[1]:
                replaced = tuple((f[0], False) if f == new else f for f in st)
                remaining = [t for t in terms if _st_key(t) != _st_key(existing)]
                terms = _simplify(_oset_add(remaining, [replaced]))
                combined = True
                break
        if not combined:
            terms = _oset_add(terms, [st])
    return terms


def _materialize(intercept: bool, terms: tuple[Term, ...], values: dict):
    """Columns of a formula part as (term key, name, values), in formulaic
    order. The intercept column is not emitted."""
    spanned: set = set()
    out: list[tuple[tuple, str, np.ndarray]] = []
    if intercept:
        spanned.add(())
    for term in terms:
        options = []
        for f in term.factors:
            if values[f.expr].spans:
                options.append([(f.expr, True), None])
            else:
                options.append([(f.expr, False)])
        span = _oset_add(
            [],
            (
                tuple(p for p in prod if p is not None)
                for prod in itertools.product(*options)
            ),
        )
        span = [st for st in span if _st_key(st) not in spanned]
        for st in _simplify(span):
            if not st:
                continue  # the intercept, reported through Design.intercept
            per_factor = [values[expr].encode(reduced) for expr, reduced in st]
            for rev in itertools.product(*reversed(per_factor)):
                prod = rev[::-1]
                name = ":".join(p[0] for p in prod)
                col = prod[0][1]
                for p in prod[1:]:
                    col = col * p[1]
                out.append((term.key, name, col))
        spanned.update(_st_key(st) for st in span)
    return out


# ---------------------------------------------------------------------------
# formulaic fallback
# ---------------------------------------------------------------------------


def _formulaic_part(
    frame: _Frame, parts: dict[str, str], nulls: np.ndarray, like_specs: dict | None
):
    """Materialize one-sided formula parts with formulaic. Factors are
    evaluated on all rows (stateful transforms such as `poly` see the full
    data, as in pyfixest); rows in `nulls` are dropped before encoding, so
    categorical levels come from the joint estimation sample. With
    `like_specs` (per-part model specs of a training design) the training
    encodings are reused.
    Returns ({part: ([(name, values on kept rows, term text)], kept row
    positions, model spec)}, joint null mask)."""
    for text in parts.values():
        if re.search(r"\bi\(", text):
            raise FormulaSyntaxError(
                f"i() cannot be combined with terms that need formulaic: `{text}`"
            )
    try:
        import formulaic
    except ImportError as e:
        terms = ", ".join(parts.values())
        raise ImportError(
            f"fereg: `{terms}` needs formulaic for terms the native parser does not support "
            "(pip install formulaic)."
        ) from e
    identifiers = set(
        re.findall(r"[A-Za-z_][\w\.]*|(?<=`)[^`]+(?=`)", " ".join(parts.values()))
    )
    full = frame.pandas_columns(sorted(identifiers & frame.columns))
    context = {"log": _pf_log, "sqrt": np.sqrt}
    # Each part is a plain formula (structured formulas ignore `drop_rows`).
    # Rows a part drops for its own missing values join the sample, and the
    # parts are re-encoded until every part sees the same joint sample.
    while True:
        drop_rows = set(np.nonzero(nulls)[0].tolist())
        matrices = {}
        for key, text in parts.items():
            if like_specs is not None:
                spec = like_specs[key]
                matrices[key] = spec.get_model_matrix(
                    full,
                    context=context,
                    drop_rows=drop_rows,
                    na_action="drop",
                    output="pandas",
                )
            else:
                # "1 +" restores the intercept pyfixest's two-sided formulas
                # have ("1 + 0 + x" still removes it)
                matrices[key] = formulaic.Formula(f"1 + {text}").get_model_matrix(
                    full,
                    ensure_full_rank=True,
                    na_action="drop",
                    output="pandas",
                    context=context,
                    drop_rows=drop_rows,
                )
        joint = nulls.copy()
        for m in matrices.values():
            part_null = np.ones(frame.n, dtype=bool)
            part_null[m.index.to_numpy()] = False
            joint |= part_null
        if np.array_equal(joint, nulls):
            break
        nulls = joint
    result = {}
    for key, m in matrices.items():
        cols = []
        for term, sl in m.model_spec.term_slices.items():
            for name in m.columns[sl]:
                if name != "Intercept":
                    cols.append(
                        (str(name), m[name].to_numpy(dtype=np.float64), str(term))
                    )
        result[key] = (cols, m.index.to_numpy(), m.model_spec)
    return result, nulls


# ---------------------------------------------------------------------------
# build_design
# ---------------------------------------------------------------------------


def _single_factor(text: str, what: str):
    """The one factor of a LHS / endogenous term."""
    _, terms = rhs_terms(text)
    if len(terms) != 1 or len(terms[0].factors) != 1:
        raise FormulaSyntaxError(f"The {what} must be a single variable: `{text}`")
    return terms[0].factors[0]


def _check_numeric(value, what: str) -> np.ndarray:
    if not isinstance(value, _Num):
        raise TypeError(f"The {what} must be numeric.")
    return value.values


def build_design(
    f: Formula,
    data,
    *,
    weights: str | None = None,
    offset: str | None = None,
    cluster: list[str] = (),
    vcov_columns: dict[str, str | None] | None = None,
    drop_intercept: bool = False,
    like: Design | None = None,
) -> Design:
    """Host columns of one model.

    With `like` (a training Design of the same formula), categorical levels
    come from `like`, so x columns align with `like.x_names` (newdata for
    predict); `y` may then be absent, instruments are not built, and rows
    with a categorical value unseen in training are flagged in `drop`.
    `vcov_columns` names the time / panel / coordinate columns of the variance
    matrix (`VcovSpec.columns()`; None = not used).
    """
    frame = _Frame(data)
    n = frame.n
    ev = _Evaluator(frame)

    # --- dependent variable ---------------------------------------------------
    y_factor = _single_factor(f.depvar, "dependent variable")
    y_name = y_factor.expr
    y = None
    if like is None or not _missing_columns(frame, y_factor):
        y = _check_numeric(_eval_native(ev, y_factor), "dependent variable")

    # --- formula parts ----------------------------------------------------------
    endog_text = " + ".join(f.endog)
    second_text = f"{f.rhs_text} + {endog_text}" if f.endog else f.rhs_text
    second_icpt, second_terms = rhs_terms(second_text)
    endog_keys = {t.key for t in rhs_terms(endog_text)[1]} if f.endog else set()
    build_inst = bool(f.endog) and like is None
    if build_inst:
        first_icpt, first_terms = rhs_terms(f"{f.inst_text} + {f.rhs_text}")
        inst_keys = {t.key for t in rhs_terms(f.inst_text)[1]}
    parts = {"second": (second_icpt, second_terms)}
    if build_inst:
        parts["first"] = (first_icpt, first_terms)

    fallback: dict[str, str] = {}
    for key, (_, terms) in parts.items():
        try:
            for term in terms:
                for fac in term.factors:
                    ev.factor(fac)
        except _NotNative:
            fallback[key] = (
                second_text if key == "second" else f"{f.inst_text} + {f.rhs_text}"
            )

    # --- fixed effects, clusters, slopes, weights, offset ------------------------
    factors: list[FactorColumn] = []
    factor_pos: dict[str, int] = {}

    def factor_index(name: str) -> int:
        if name not in factor_pos:
            factor_pos[name] = len(factors)
            factors.append(_factor_from_frame(frame, name))
        return factor_pos[name]

    slopes: list[np.ndarray] = []
    slope_names: list[str] = []
    fe = []
    for term in f.fe:
        idx = [factor_index(name) for name in term.factors]
        sidx = []
        for s in term.slopes:
            if s not in slope_names:
                values = frame.numeric(s)
                if values is None:
                    raise TypeError(f"The slope variable '{s}' must be numeric.")
                slope_names.append(s)
                slopes.append(values)
            sidx.append(slope_names.index(s))
        fe.append((idx, term.intercept, sidx))
    cluster_dims = []
    for c in cluster:
        cluster_dims.append([factor_index(p.strip()) for p in c.split("^")])
    vcols = vcov_columns or {}
    time_idx = panel_idx = None
    if vcols.get("time") is not None:
        _time_factor(frame, vcols["time"])
        time_idx = factor_index(vcols["time"])
    if vcols.get("panel") is not None:
        panel_idx = factor_index(vcols["panel"])
    lat = None if vcols.get("lat") is None else _coordinate(frame, vcols["lat"])
    lon = None if vcols.get("lon") is None else _coordinate(frame, vcols["lon"])

    w = None
    if weights is not None and (like is None or weights in frame.columns):
        frame.require(weights, "Weights variable")
        w = frame.numeric(weights)
        if w is None:
            raise ValueError(f"The weights column '{weights}' must be numeric.")
        valid = w[~np.isnan(w)]
        if valid.size and not valid.min() > 0.0:
            raise ValueError(
                f"The weights column '{weights}' must have only non-negative values."
            )
    off = None
    if offset is not None:
        if offset not in frame.columns:
            raise ValueError(f"Offset variable '{offset}' not found in data.")
        off = frame.numeric(offset)
        if off is None:
            raise ValueError(f"The offset column '{offset}' must be numeric.")

    # --- sample for categorical levels (formulaic na_action="drop") -------------
    natives = [v for k, v in ev.cache.items()]
    cat_sources = [
        c for v in natives if isinstance(v, (_CatF, _IFactor)) for c in v.sources()
    ]
    drop = None

    def add_drop(mask):
        nonlocal drop
        if mask is None or not mask.any():
            return
        drop = mask.copy() if drop is None else (drop | mask)

    for v in natives:
        if isinstance(v, _Num) and v.derived:
            add_drop(v.missing())
    for c in cat_sources:
        add_drop(c.missing())

    levels_state: dict = {}
    fallback_cols: dict[str, list] = {}
    if cat_sources or fallback:
        nulls = np.zeros(n, dtype=bool)
        for v in natives:
            if isinstance(v, _Num):
                nulls |= v.missing()
            if isinstance(v, _IFactor) and v.var2 is not None:
                nulls |= np.isnan(v.var2)
        for c in cat_sources:
            nulls |= c.missing()
        for arr in [y, w, off, *slopes]:
            if arr is not None:
                nulls |= np.isnan(arr)
        for fc in factors:
            m = _factor_missing(fc)
            if m is not None:
                nulls |= m
        if fallback:
            like_specs = like.levels.get("__formulaic__") if like is not None else None
            parts_out, joint = _formulaic_part(frame, fallback, nulls, like_specs)
            levels_state["__formulaic__"] = {
                key: spec for key, (_, _, spec) in parts_out.items()
            }
            for key, (items, kept_rows, _) in parts_out.items():
                scattered = []
                for name, values, term in items:
                    full = np.full(n, np.nan)
                    full[kept_rows] = values
                    scattered.append((name, full, term))
                fallback_cols[key] = scattered
            add_drop(joint & ~nulls)
            nulls = joint
        kept = ~nulls
        for expr, v in ev.cache.items():
            if isinstance(v, _CatF):
                unseen = v.cat.finalize(
                    kept, like.levels.get(expr) if like is not None else None
                )
                levels_state[expr] = list(v.cat.levels)
                add_drop(unseen)
            elif isinstance(v, _IFactor):
                given = like.levels.get(expr) if like is not None else None
                lv = []
                for j, c in enumerate(v.sources()):
                    unseen = c.finalize(kept, given[j] if given is not None else None)
                    lv.append(list(c.levels))
                    add_drop(unseen)
                levels_state[expr] = lv

    # --- columns ------------------------------------------------------------------
    def part_columns(key: str):
        if key in fallback_cols:
            if key == "second":
                endog_names = set(f.endog)
                return [
                    ((name,) if name in endog_names else ("exog",), name, col)
                    for name, col, _ in fallback_cols[key]
                ]
            exog_terms = {t.text for t in rhs_terms(f.rhs_text)[1]}
            return [
                (("exog",) if term in exog_terms else ("inst",), name, col)
                for name, col, term in fallback_cols[key]
            ]
        icpt, terms = parts[key]
        values = {fac.expr: ev.factor(fac) for t in terms for fac in t.factors}
        return _materialize(icpt, terms, values)

    second_cols = part_columns("second")
    if "second" in fallback_cols:
        endog_keys = {(e,) for e in f.endog}
    x = [c for k, _, c in second_cols if k not in endog_keys]
    x_names = [name for k, name, _ in second_cols if k not in endog_keys]
    endog = [c for k, _, c in second_cols if k in endog_keys]
    endog_names = [name for k, name, _ in second_cols if k in endog_keys]
    if len(endog) != len(f.endog):
        raise TypeError("The endogenous variable must be numeric.")
    inst, inst_names = [], []
    if build_inst:
        first_cols = part_columns("first")
        if "first" in fallback_cols:
            inst_keys = {("inst",)}
        inst = [c for k, _, c in first_cols if k in inst_keys]
        inst_names = [name for k, name, _ in first_cols if k in inst_keys]

    intercept = (
        f.intercept and not drop_intercept and not any(t.intercept for t in f.fe)
    )
    coefnames = (["Intercept"] if intercept else []) + [
        name for _, name, _ in second_cols
    ]
    return Design(
        n=n,
        y=y,
        y_name=y_name,
        x=x,
        x_names=x_names,
        endog=endog,
        endog_names=endog_names,
        inst=inst,
        inst_names=inst_names,
        intercept=intercept,
        factors=factors,
        fe=fe,
        fe_names=[t.label for t in f.fe],
        slopes=slopes,
        slope_names=slope_names,
        weights=w,
        offset=off,
        cluster=cluster_dims,
        cluster_names=list(cluster),
        drop=drop,
        coefnames=coefnames,
        levels=levels_state,
        time=time_idx,
        panel=panel_idx,
        lat=lat,
        lon=lon,
    )


def _missing_columns(frame: _Frame, factor) -> bool:
    if factor.kind == "name":
        return factor.expr not in frame.columns
    return not _expr_variables(factor.expr) <= frame.columns


def _eval_native(ev: _Evaluator, factor):
    try:
        return ev.factor(factor)
    except _NotNative as e:
        raise FormulaSyntaxError(
            f"Unsupported dependent variable expression `{factor.expr}`"
        ) from e
