"""``summary()``: pyfixest's text summary of one or more models."""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd

__all__ = ["summary", "flatten_models", "method_name", "markdown_table"]

_METHOD_NAMES = {
    "fepois": "Poisson",
    "feglm-logit": "Logit",
    "feglm-probit": "Probit",
    "feglm-gaussian": "Gaussian",
}


def flatten_models(models) -> list:
    """A model, a ``FixestMulti`` or a list of either, as a flat list of models."""
    from .results import Feols, FixestMulti

    if isinstance(models, FixestMulti):
        return models.to_list()
    if isinstance(models, Feols):
        return [models]
    if isinstance(models, (list, tuple)):
        out = []
        for m in models:
            out.extend(flatten_models(m))
        return out
    raise TypeError("models must be a model, a FixestMulti or a list of models.")


def method_name(m) -> str:
    if m._method == "feols":
        return "IV" if m._is_iv else "OLS"
    return _METHOD_NAMES[m._method]


def _is_num(v) -> bool:
    return isinstance(v, numbers.Number) and not isinstance(v, bool)


def _cell(v, floatfmt: str) -> str:
    if isinstance(v, (float, np.floating)):
        return format(float(v), floatfmt)
    return str(v)


def _align_decimal(cells: list[str]) -> list[str]:
    """Pads on the right so the decimal points line up (tabulate ``numalign="decimal"``)."""
    after = [len(c) - (c.index(".") if "." in c else len(c)) for c in cells]
    m = max(after, default=0)
    return [c + " " * (m - a) for c, a in zip(cells, after)]


def markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """``df.to_markdown(floatfmt=...)`` (tabulate "pipe" format) without tabulate."""
    headers = [df.index.name or ""] + [str(c) for c in df.columns]
    columns = [list(df.index)] + [list(df[c]) for c in df.columns]
    if all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns):
        # tabulate reads an all-numeric frame as one float array: integers print as floats.
        columns = columns[:1] + [[float(v) for v in col] for col in columns[1:]]
    texts, right = [], []
    for col in columns:
        numeric = len(col) > 0 and all(_is_num(v) for v in col)
        right.append(numeric)
        cells = [_cell(v, floatfmt) for v in col]
        texts.append(_align_decimal(cells) if numeric else cells)
    widths = [max([len(h) + 2] + [len(t) for t in col]) for h, col in zip(headers, texts)]

    def row(cells):
        return "| " + " | ".join(c.rjust(w) if r else c.ljust(w) for c, w, r in zip(cells, widths, right)) + " |"

    sep = "|" + "|".join(("-" * (w + 1) + ":") if r else (":" + "-" * (w + 1)) for w, r in zip(widths, right)) + "|"
    lines = [row(headers), sep]
    lines += [row([col[i] for col in texts]) for i in range(len(df))]
    return "\n".join(lines)


def summary(models, digits: int = 3) -> None:
    """Prints, for every model, the estimation header and the coefficient table."""
    for m in flatten_models(models):
        df = m.tidy().round(digits)
        print("###")
        print("")
        print("Estimation: ", method_name(m))
        line = f"Dep. var.: {m._depvar}"
        if m._fixef is not None:
            line += f", Fixed effects: {m._fixef}"
        print(line)
        if m._sample_split_var is not None:
            print(f"sample: {m._sample_split_var} = {m._sample_split_value}")
        print("Inference: ", m._vcov_type_detail)
        print("Observations: ", m._N)
        print("")
        print(markdown_table(df, floatfmt=f".{digits}f"))
        print("---")
        tail = ""
        if not np.isnan(m._rmse):
            tail += f"RMSE: {np.round(m._rmse, digits)} "
        if not np.isnan(m._r2):
            tail += f"R2: {np.round(m._r2, digits)} "
        if not np.isnan(m._r2_within):
            tail += f"R2 Within: {np.round(m._r2_within, digits)} "
        if m.deviance is not None:
            tail += f"Deviance: {np.round(np.asarray(m.deviance).squeeze(), digits)} "
        print(tail)
