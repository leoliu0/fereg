"""``etable()``: regression tables as a DataFrame, Markdown, LaTeX or HTML (no maketables)."""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd

from .results import _select_order_coefs
from .summary import flatten_models, markdown_table

__all__ = ["etable"]

_DEFAULT_SIGNIF = [0.001, 0.01, 0.05]


def _stars(p: float, signif_code: list[float]) -> str:
    if signif_code is None or not np.isfinite(p):
        return ""
    a, b, c = signif_code
    return "***" if p < a else "**" if p < b else "*" if p < c else ""


def _fmt(v: float, digits: int) -> str:
    return "" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def _fe_labels(m) -> list[str]:
    return list(m._design.fe_names) if m._has_fixef else []


def _build(
    models,
    signif_code,
    coef_fmt,
    keep,
    drop,
    exact_match,
    labels,
    show_fe,
    felabels,
    fe_present,
    fe_absent,
    model_heads,
    head_order,
    digits,
    custom_model_stats,
) -> pd.DataFrame:
    labels = labels or {}
    felabels = felabels or {}
    coefs: list[str] = []
    for m in models:
        coefs += [c for c in m._coefnames if c not in coefs]
    if keep or drop:
        coefs = _select_order_coefs(coefs, keep, drop, bool(exact_match))
    fes: list[str] = []
    for m in models:
        fes += [f for f in _fe_labels(m) if f not in fes]

    rows: dict[tuple[str, str], list[str]] = {}
    for c in coefs:
        cells = []
        for m in models:
            if c not in m._coefnames:
                cells.append("")
                continue
            i = m._coefnames.index(c)
            b, se, t, p = m._beta_hat[i], m._se[i], m._tstat[i], m._pvalue[i]
            if coef_fmt is None:
                cells.append(f"{_fmt(b, digits)}{_stars(p, signif_code)} \n ({_fmt(se, digits)})")
            else:
                cells.append(
                    coef_fmt.replace("b", f"{_fmt(b, digits)}{_stars(p, signif_code)}")
                    .replace("se", f"({_fmt(se, digits)})")
                    .replace("t", f"[{_fmt(t, digits)}]")
                    .replace("p", f"{{{_fmt(p, digits)}}}")
                )
        rows[("coef", labels.get(c, c))] = cells
    if show_fe:
        for f in fes:
            rows[("fe", felabels.get(f, f))] = [fe_present if f in _fe_labels(m) else fe_absent for m in models]
    for name, values in (custom_model_stats or {}).items():
        rows[("stats", name)] = [str(v) for v in values]
    rows[("stats", "Observations")] = [str(m._N) for m in models]
    rows[("stats", "S.E. type")] = [_se_type(m) for m in models]
    rows[("stats", "R²")] = ["-" if np.isnan(m._r2) else _fmt(m._r2, digits) for m in models]
    if any(m._has_fixef and not np.isnan(m._r2_within) for m in models):
        rows[("stats", "R² Within")] = ["-" if np.isnan(m._r2_within) else _fmt(m._r2_within, digits) for m in models]
    pr2 = [getattr(m, "_pseudo_r2", None) for m in models]
    if any(v is not None for v in pr2):
        rows[("stats", "Pseudo R²")] = ["-" if v is None else _fmt(v, digits) for v in pr2]

    heads = model_heads or [""] * len(models)
    numbers = [f"({i + 1})" for i in range(len(models))]
    levels = {"d": [m._depvar for m in models], "h": heads}
    order = [ch for ch in (head_order or "") if ch in levels and (ch != "h" or model_heads)]
    arrays = [levels[ch] for ch in order] + [numbers]
    columns = pd.MultiIndex.from_arrays(arrays) if len(arrays) > 1 else pd.Index(numbers)
    index = pd.MultiIndex.from_tuples(list(rows.keys()))
    return pd.DataFrame(list(rows.values()), index=index, columns=columns)


def _se_type(m) -> str:
    if m._is_clustered:
        return "by: " + "+".join(m._clustervar)
    return m._vcov_type_detail


def _tex(df: pd.DataFrame, notes: str, signif_code) -> str:
    ncol = df.shape[1]
    out = [r"\begin{tabular}{l" + "c" * ncol + "}", r"\toprule"]
    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            out.append(" & " + " & ".join(_tex_escape(str(v)) for v in df.columns.get_level_values(level)) + r" \\")
    else:
        out.append(" & " + " & ".join(str(v) for v in df.columns) + r" \\")
    group = None
    for (g, name), values in zip(df.index, df.to_numpy()):
        if g != group:
            out.append(r"\midrule")
            group = g
        cells = [
            r"\makecell{" + _tex_escape(v).replace("\n", r"\\") + "}" if "\n" in v else _tex_escape(v) for v in values
        ]
        out.append(_tex_escape(str(name)) + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    foot = _signif_note(signif_code)
    if notes or foot:
        out.append(r"\par\footnotesize " + _tex_escape(notes or foot))
    return "\n".join(out)


def _tex_escape(s: str) -> str:
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("²", "$^2$")):
        s = s.replace(a, b)
    return s


def _html(df: pd.DataFrame, notes: str, signif_code) -> str:
    out = ["<table>", "<thead>"]
    levels = df.columns.nlevels
    for level in range(levels):
        vals = df.columns.get_level_values(level) if levels > 1 else df.columns
        out.append("<tr><th></th>" + "".join(f"<th>{escape(str(v))}</th>" for v in vals) + "</tr>")
    out += ["</thead>", "<tbody>"]
    for (_, name), values in zip(df.index, df.to_numpy()):
        cells = "".join(f"<td>{escape(v).replace(chr(10), '<br>')}</td>" for v in values)
        out.append(f"<tr><td>{escape(str(name))}</td>{cells}</tr>")
    out += ["</tbody>", "</table>"]
    foot = notes or _signif_note(signif_code)
    if foot:
        out.append(f"<p>{escape(foot)}</p>")
    return "\n".join(out)


def _signif_note(signif_code) -> str:
    if signif_code is None:
        return ""
    a, b, c = signif_code
    return f"Significance levels: * p < {c}, ** p < {b}, *** p < {a}. Format of coefficient cell: Coefficient (Std. Error)"


def etable(
    models,
    type: str = "df",
    signif_code: list | None = None,
    coef_fmt: str | None = None,
    custom_stats: dict | None = None,
    custom_model_stats: dict | None = None,
    keep: list | str | None = None,
    drop: list | str | None = None,
    exact_match: bool | None = False,
    labels: dict | None = None,
    cat_template: str | None = None,
    show_fe: bool | None = True,
    felabels: dict | None = None,
    fe_present: str = "x",
    fe_absent: str = "-",
    notes: str = "",
    model_heads: list | None = None,
    head_order: str | None = "dh",
    file_name: str | None = None,
    digits: int = 3,
) -> pd.DataFrame | str | None:
    """Regression table of one or more models.

    ``type="df"`` (default; pyfixest's default ``"gt"`` needs great_tables and is not offered)
    returns a DataFrame with pyfixest's row/column layout, ``"md"`` prints Markdown, ``"tex"``
    and ``"html"`` return the table as text (also written to ``file_name`` when given).
    """
    if type not in ("df", "md", "tex", "html"):
        raise ValueError("type must be 'df', 'md', 'tex' or 'html'.")
    if custom_stats or cat_template:
        raise NotImplementedError("custom_stats and cat_template are not supported by fereg.etable.")
    models = flatten_models(models)
    if model_heads is not None and len(model_heads) != len(models):
        raise ValueError("model_heads must have the same length as models")
    if custom_model_stats is not None:
        for stat, values in custom_model_stats.items():
            if not isinstance(values, list) or len(values) != len(models):
                raise ValueError("lists in custom_model_stats values must have the same length as models")
    signif = _DEFAULT_SIGNIF if signif_code is None else signif_code
    df = _build(
        models,
        signif,
        coef_fmt,
        keep,
        drop,
        exact_match,
        labels,
        show_fe,
        felabels,
        fe_present,
        fe_absent,
        model_heads,
        head_order,
        digits,
        custom_model_stats,
    )
    if type == "df":
        return df
    if type == "md":
        flat = df.copy()
        flat.index = pd.Index([f"{name}" for _, name in df.index])
        flat.columns = [" ".join(str(x) for x in c) if isinstance(c, tuple) else str(c) for c in df.columns]
        flat = flat.apply(lambda col: col.str.replace("\n", ""))
        print(markdown_table(flat))
        return None
    text = _tex(df, notes, signif) if type == "tex" else _html(df, notes, signif)
    if file_name is not None:
        with open(file_name, "w") as fh:
            fh.write(text)
    return text
