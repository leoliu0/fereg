"""Formula parsing with pyfixest 0.60.0 semantics.

`parse` expands multiple-estimation syntax exactly as pyfixest's
`Formula.parse` + `FixestMulti` do (same model texts, same model order), and
splits every model into its second stage, fixed effects and IV first stage.
The right-hand side is reduced to formulaic terms by a port of formulaic's
tokenizer and term algebra (`+ - * / : **`, intercept rules, degree
ordering), so `design.build_design` produces pyfixest's column names without
calling formulaic.
"""

from __future__ import annotations

import ast
import functools
import itertools
import keyword
import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "Factor",
    "FeTerm",
    "Formula",
    "FormulaSyntaxError",
    "Term",
    "parse",
    "rhs_terms",
]


class FormulaSyntaxError(ValueError):
    """Raised for formulas pyfixest (or this parser) rejects."""


@dataclass(frozen=True)
class FeTerm:
    factors: tuple[str, ...]  # data columns; len > 1 = interaction "a^b"
    intercept: bool  # False only for slope-only "fe[[x]]"
    slopes: tuple[str, ...]  # "fe[x]" / "fe[x1, x2]" / "fe[[x]]"
    label: str  # as written, e.g. "f1^f2"


@dataclass(frozen=True)
class Formula:
    fml: str  # one model (after multiple-estimation expansion), pyfixest text
    depvar: str
    rhs: tuple[str, ...]  # RHS term texts in order, without "1"/"0"/"-1"
    intercept: bool  # False when "0" or "-1" is in the RHS
    fe: tuple[FeTerm, ...]
    endog: tuple[str, ...]  # IV first-stage LHS
    inst: tuple[str, ...]  # excluded instruments
    fe_text: str  # FE part as written ("f1+f2"), "" when none
    rhs_text: str = ""  # exogenous RHS as written (second stage without endog)
    inst_text: str = ""  # first-stage RHS as written, "" when no IV part


# ---------------------------------------------------------------------------
# pyfixest multiple-estimation expansion (estimation/formula/{parse,utils}.py)
# ---------------------------------------------------------------------------


class _MultipleEstimationType(str, Enum):
    sw = "sequential stepwise"
    csw = "cumulative stepwise"
    sw0 = "sequential stepwise with zero step"
    csw0 = "cumulative stepwise with zero step"
    mvsw = "multiverse stepwise"


_MULTIPLE_ESTIMATION_PATTERN = re.compile(
    rf"\b({'|'.join(me.name for me in _MultipleEstimationType)})\b\(.+\)"
)


def _str_split_by_sep(string: str, separator: str = "+") -> list[str]:
    """Split on top-level `separator`, ignoring separators inside () and [].

    pyfixest tracks only parentheses; brackets are tracked too so that
    fixest-style slopes `fe[x1, x2]` survive `sw(...)` argument splitting
    (pyfixest has no slope syntax, so its results are unaffected).
    """
    args: list[str] = []
    depth = 0
    current: list[str] = []
    for c in string:
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == separator and depth == 0:
            args.append("".join(current).strip())
            current = []
            continue
        current.append(c)
    args.append("".join(current).strip())
    return args


def _first_parenthesis_pair(string: str) -> tuple[int, int]:
    """(start, end) of the content of the first matched parenthesis pair."""
    position_open = string.find("(")
    if position_open == -1:
        raise FormulaSyntaxError(f"No parenthesis in `{string}`")
    position_open += 1
    position = position_open
    depth = 1
    while position < len(string) and depth:
        position += 1
        if position >= len(string):
            break
        if string[position] == "(":
            depth += 1
        elif string[position] == ")":
            depth -= 1
    if depth != 0:
        raise FormulaSyntaxError(f"Unmatched '(' in `{string}`")
    return position_open, position


def _validate(formula: str) -> None:
    parts = _str_split_by_sep(formula, separator="|")
    if len(parts) > 3:
        raise FormulaSyntaxError(
            "Formula can have at most 3 parts separated by '|'. "
            f"Received {len(parts)}: '{formula}'"
        )
    multiple = [p for p in parts if p.count("~") > 1]
    if multiple:
        raise FormulaSyntaxError(
            f"Each formula part can contain at most one '~'. Invalid parts: {multiple}"
        )
    if len(parts) == 1 and "~" not in parts[0]:
        raise FormulaSyntaxError(f"Formula must contain '~': '{formula}'")
    if len(parts) >= 2 and "~" not in parts[0]:
        raise FormulaSyntaxError(
            f"First part must contain '~' (dependent ~ independent): '{parts[0]}'"
        )
    if len(parts) == 3:
        if "~" in parts[1]:
            raise FormulaSyntaxError(
                f"Second part (fixed effects) cannot contain '~': '{parts[1]}'. "
                "Fixed effects should be specified as 'f1 + f2', not as a formula."
            )
        if "~" not in parts[2]:
            raise FormulaSyntaxError(
                "Three-part formula requires IV specification in third part: "
                "'dependent ~ independent | fixed_effects | endogenous ~ instruments'. "
            )


def _preprocess(formula: str) -> str:
    """`y1 + y2 ~ x` -> `sw(y1, y2) ~ x` (pyfixest `_preprocess`)."""
    dependents, rhs = re.split(r"\s*~\s*", formula, maxsplit=1)
    deps = _str_split_by_sep(dependents.strip(), separator="+")
    if len(deps) > 1:
        formula = f"sw({', '.join(deps)}) ~ {rhs}"
    return formula


def _expand_first(formula: str) -> list[str] | None:
    match = _MULTIPLE_ESTIMATION_PATTERN.search(formula)
    if not match:
        return None
    kind = _MultipleEstimationType[match.group(1)]
    p_open, p_close = _first_parenthesis_pair(formula[match.start() :])
    p_open += match.start()
    p_close += match.start()
    arguments = _str_split_by_sep(formula[p_open:p_close], separator=",")
    if len(arguments) < 2 and kind is not _MultipleEstimationType.mvsw:
        raise FormulaSyntaxError(
            f"'{kind.name}(...)' requires at least 2 arguments, got {len(arguments)}. "
            "Check for extra parentheses, e.g. sw((a, b)) should be sw(a, b)."
        )
    if kind is _MultipleEstimationType.mvsw:
        arguments = [
            " + ".join(combination)
            for combination in itertools.chain.from_iterable(
                itertools.combinations(arguments, r=length)
                for length in range(1, len(arguments) + 1)
            )
        ]
    elif kind in (_MultipleEstimationType.csw, _MultipleEstimationType.csw0):
        arguments = [" + ".join(arguments[: i + 1]) for i in range(len(arguments))]
    if kind in (
        _MultipleEstimationType.sw0,
        _MultipleEstimationType.csw0,
        _MultipleEstimationType.mvsw,  # fixest has no mvsw0: mvsw includes the zero step
    ):
        arguments = ["1", *arguments]
    call = formula[match.start() : p_close + 1]
    # str.replace substitutes every identical call, as pyfixest does
    return [formula.replace(call, argument) for argument in arguments]


def _expand_all(formula: str) -> list[str]:
    expansion = _expand_first(formula)
    if expansion is None:
        return [formula]
    return [f for e in expansion for f in _expand_all(e)]


# ---------------------------------------------------------------------------
# formulaic term algebra (tokenizer, operators, intercept, degree ordering)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Factor:
    """One formulaic factor: `expr` is the lookup name, normalized Python code
    (formulaic `format_expr`) or a literal."""

    expr: str
    kind: str  # "name" | "python" | "value"


class Term:
    """Product of factors; equality ignores factor order (formulaic `Term`)."""

    __slots__ = ("factors", "key")

    def __init__(self, factors):
        self.factors: tuple[Factor, ...] = tuple(dict.fromkeys(factors))
        self.key = tuple(sorted(f.expr for f in self.factors))

    @property
    def degree(self) -> int:
        return sum(1 for f in self.factors if f.kind != "value")

    @property
    def text(self) -> str:
        return ":".join(
            f"`{f.expr}`" if f.kind == "name" and ":" in f.expr else f.expr
            for f in self.factors
        )

    def __mul__(self, other: Term) -> Term:
        return Term([*self.factors, *other.factors])

    def __eq__(self, other) -> bool:
        return isinstance(other, Term) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return self.text


def _oset(items) -> list[Term]:
    """Ordered set of terms (first occurrence wins)."""
    return list(dict.fromkeys(items))


_WORD = re.compile(r"[\.\_\w]")
_NUMERIC = re.compile(r"[0-9\.]")


class _Tok:
    __slots__ = ("kind", "text")

    def __init__(self, kind: str | None = None, text: str = ""):
        self.kind = kind
        self.text = text

    def update(self, char: str, kind: str | None = None) -> _Tok:
        self.text += char
        if kind is not None:
            self.kind = kind
        return self

    def __bool__(self) -> bool:
        return bool(self.text)


def _tokenize(formula: str) -> list[_Tok]:
    """Port of formulaic.parser.algos.tokenize."""
    out: list[_Tok] = []
    quote: list[str] = []
    take = 0
    tok = _Tok()
    for char in formula:
        if take > 0:
            tok.update(char)
            take -= 1
            continue
        if quote and char == "\\":
            tok.update(char)
            take = 1
            continue
        if quote and quote[-1] in "}`%" and char == quote[-1]:
            quote.pop()
            if tok:
                if quote:
                    tok.update(char)
                else:
                    out.append(tok)
                    tok = _Tok()
            continue
        if quote and char == quote[-1]:
            tok.update(char)
            quote.pop()
            continue
        if quote and quote[-1] in ('"', "'", "`", ")", "]", "}", "%"):
            if char in "`([" and quote[-1] in "})]":
                quote.append({"(": ")", "[": "]"}.get(char, char))
            tok.update(char)
            continue
        if char == "%":
            if tok:
                out.append(tok)
            tok = _Tok("operator")
            quote.append("%")
            continue
        if char == "{":
            if tok:
                out.append(tok)
            tok = _Tok("python")
            quote.append("}")
            continue
        if char == "`":
            if tok:
                out.append(tok)
            tok = _Tok("name")
            quote.append("`")
            continue
        if char in "([":
            if tok.kind in ("name", "python"):
                tok.update(char, "python")
                quote.append(")" if char == "(" else "]")
            else:
                if tok:
                    out.append(tok)
                    tok = _Tok()
                out.append(_Tok("context", char))
            continue
        if char in ")]":
            if tok:
                out.append(tok)
                tok = _Tok()
            out.append(_Tok("context", char))
            continue
        if char.isspace():
            if tok and tok.kind != "operator":
                out.append(tok)
                tok = _Tok()
            continue
        if char in ('"', "'"):
            if tok and tok.kind == "operator":
                out.append(tok)
                tok = _Tok()
            if tok:
                raise FormulaSyntaxError(
                    f"Unexpected character {char!r} following token `{tok.text}`."
                )
            tok.update(char, "value")
            quote.append(char)
            continue
        if _WORD.match(char):
            if tok and tok.kind in ("operator", "python"):
                out.append(tok)
                tok = _Tok()
            kind = (
                "value"
                if _NUMERIC.match(char) and tok.kind in (None, "value")
                else "name"
            )
            tok.update(char, kind)
            continue
        if tok and tok.kind != "operator":
            out.append(tok)
            tok = _Tok()
        tok.update(char, "operator")
    if quote:
        raise FormulaSyntaxError(
            f"Formula ended before quote context was closed. Expected: {quote[-1]}"
        )
    if tok:
        out.append(tok)
    return out


def format_expr(expr: str) -> str:
    """formulaic `format_expr`: canonical Python source of an expression."""
    return ast.unparse(ast.parse(expr, mode="eval")).replace("\n", " ")


def _sanitize_python(expr: str) -> str:
    """formulaic `sanitize_python_code`: format, keeping backtick names."""
    aliases: dict[str, str] = {}
    parts = re.split(r"(\\\"|\"(?:\\\"|[^\"])*\"|\\'|'(?:\\'|[^'])*'|`)", expr)
    out: list[str] = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == "`":
            j = i + 1
            name_parts = []
            while j < len(parts) and parts[j] != "`":
                name_parts.append(parts[j])
                j += 1
            name = "".join(name_parts)
            if j >= len(parts):
                out.append(f"`{name}")
                i = j
                continue
            if name.isidentifier() or keyword.iskeyword(name):
                new = name
            else:
                base = "".join(c if re.match(r"\w", c) else "_" for c in name)
                if base[0].isdigit():
                    base = "_" + base
                new = f"_formulaic_{base}"
            aliases[new] = name
            out.append(f" {new} ")
            i = j + 1
            continue
        out.append(part)
        i += 1
    try:
        formatted = format_expr("".join(out).strip())
    except SyntaxError as e:
        raise FormulaSyntaxError(
            f"Invalid Python expression in formula: `{expr}`"
        ) from e
    for alias, orig in aliases.items():
        if alias != orig:
            formatted = formatted.replace(alias, f"`{orig}`")
    return formatted


def _resolve_operator(symbol: str) -> list[str]:
    """formulaic DefaultOperatorResolver: collapse +/- runs, split the rest."""
    if symbol in _BINARY:
        return [symbol]
    while True:
        m = re.search(r"[+\-]{2,}", symbol)
        if not m:
            break
        if len(m.group(0).replace("+", "")) % 2:
            symbol = symbol[: m.start(0)] + "-"
        else:
            symbol = "+" + symbol[m.end(0) :]
    if symbol in _BINARY:
        return [symbol]
    return list(symbol)


# precedence, right-associative
_BINARY = {
    "+": (100, False),
    "-": (100, False),
    "*": (200, False),
    "/": (200, False),
    ":": (300, False),
    "**": (500, True),
    "^": (500, True),
}
_PREFIX_PRECEDENCE = 100


def _prepare_tokens(rhs: str) -> list[_Tok]:
    tokens = _tokenize(rhs)
    for t in tokens:
        if t.kind == "python":
            t.text = _sanitize_python(t.text)
        if t.kind == "operator" and t.text.startswith("%"):
            raise FormulaSyntaxError(f"Operator `%{t.text}%` is not supported.")
    # "0" -> "-1"
    replaced: list[_Tok] = []
    for t in tokens:
        if t.kind == "value" and t.text == "0":
            replaced += [_Tok("operator", "-"), _Tok("value", "1")]
        else:
            replaced.append(t)
    # implicit intercept: "1 +" in front unless the RHS starts with "+"/"-"
    head = [_Tok("value", "1")]
    if replaced and not (
        replaced[0].kind == "operator" and replaced[0].text in ("+", "-")
    ):
        head.append(_Tok("operator", "+"))
    tokens = head + replaced
    # merge adjacent +/- operator tokens (formulaic merge_operator_tokens)
    merged: list[_Tok] = []
    for t in tokens:
        if (
            t.kind == "operator"
            and merged
            and merged[-1].kind == "operator"
            and t.text[0] in "+-"
            and merged[-1].text[-1] in "+-"
        ):
            merged[-1] = _Tok("operator", merged[-1].text + t.text)
        else:
            merged.append(t)
    # split operator tokens into resolved operators
    out: list[_Tok] = []
    for t in merged:
        if t.kind == "operator":
            out += [_Tok("operator", s) for s in _resolve_operator(t.text)]
        else:
            out.append(t)
    return out


class _TermParser:
    """Pratt parser evaluating formulaic operators on ordered term sets."""

    def __init__(self, tokens: list[_Tok], source: str):
        self.tokens = tokens
        self.pos = 0
        self.source = source

    def error(self, msg: str) -> FormulaSyntaxError:
        return FormulaSyntaxError(f"{msg} in `{self.source}`")

    def peek(self) -> _Tok | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> _Tok:
        t = self.peek()
        if t is None:
            raise self.error("Unexpected end of formula")
        self.pos += 1
        return t

    def parse(self) -> list[Term]:
        terms = self.expr(0)
        if self.peek() is not None:
            raise self.error(f"Unexpected token `{self.peek().text}`")
        return terms

    def atom(self) -> list[Term]:
        t = self.next()
        if t.kind == "operator" and t.text in ("+", "-"):
            # prefix +/- (precedence 100, right-associative): `-a:b` = -(a:b)
            operand = self.expr(_PREFIX_PRECEDENCE + 1)
            return operand if t.text == "+" else []
        if t.kind == "context" and t.text == "(":
            inner = self.expr(0)
            close = self.next()
            if close.kind != "context" or close.text != ")":
                raise self.error("Unmatched '('")
            return inner
        if t.kind in ("name", "python", "value"):
            if t.kind == "name" and t.text == ".":
                raise self.error("The `.` operator is not supported")
            return [Term([Factor(t.text, t.kind)])]
        raise self.error(f"Unexpected token `{t.text}`")

    def expr(self, min_prec: int) -> list[Term]:
        lhs = self.atom()
        while True:
            t = self.peek()
            if t is None or t.kind != "operator":
                if t is not None and t.kind == "name" and t.text == "in":
                    raise self.error("The `in` operator is not supported")
                break
            if t.text not in _BINARY:
                raise self.error(f"Unknown operator '{t.text}'")
            prec, right = _BINARY[t.text]
            if prec < min_prec:
                break
            self.pos += 1
            rhs = self.expr(prec if right else prec + 1)
            lhs = self.apply(t.text, lhs, rhs)
        return lhs

    def apply(self, op: str, lhs: list[Term], rhs: list[Term]) -> list[Term]:
        if op == "+":
            return _oset([*lhs, *rhs])
        if op == "-":
            drop = set(rhs)
            return [t for t in lhs if t not in drop]
        if op == ":":
            return _oset(
                functools.reduce(lambda a, b: a * b, c)
                for c in itertools.product(lhs, rhs)
            )
        if op == "*":
            return _oset(
                [
                    *lhs,
                    *rhs,
                    *(
                        functools.reduce(lambda a, b: a * b, c)
                        for c in itertools.product(lhs, rhs)
                    ),
                ]
            )
        if op == "/":
            common = functools.reduce(lambda a, b: a * b, lhs)
            return _oset([*lhs, *(common * t for t in rhs)])
        # "**" / "^": the right-hand side must be one integer literal
        if (
            len(rhs) != 1
            or len(rhs[0].factors) != 1
            or rhs[0].factors[0].kind != "value"
            or not rhs[0].factors[0].expr.isdigit()
        ):
            raise self.error(
                "The right-hand argument of `**` must be a positive integer"
            )
        n = int(rhs[0].factors[0].expr)
        return _oset(
            functools.reduce(lambda a, b: a * b, c)
            for c in itertools.product(*[lhs] * n)
        )


_INTERCEPT_KEY = ("1",)


@functools.lru_cache(maxsize=1024)
def rhs_terms(rhs: str) -> tuple[bool, tuple[Term, ...]]:
    """Formulaic terms of a right-hand side: (has intercept, degree-ordered
    terms without the intercept)."""
    terms = _TermParser(_prepare_tokens(rhs), rhs).parse()
    for term in terms:
        for f in term.factors:
            if f.kind == "value" and (f.expr != "1" or len(term.factors) > 1):
                raise FormulaSyntaxError(
                    f"Literal `{f.expr}` is not supported in `{rhs}`; only `0`, `1` and `-1` are."
                )
    terms = sorted(terms, key=lambda t: t.degree)  # stable: formulaic "degree" ordering
    intercept = any(t.key == _INTERCEPT_KEY for t in terms)
    return intercept, tuple(t for t in terms if t.key != _INTERCEPT_KEY)


def _summands(rhs: str) -> tuple[str, ...]:
    """Top-level summands as written ("-x" for subtracted ones), without the
    intercept tokens "1", "0", "-1"."""
    items: list[str] = []
    depth = 0
    current: list[str] = []
    sign = ""
    for c in rhs + "+":
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c in "+-" and depth == 0:
            text = "".join(current).strip()
            if text:
                items.append(sign + text)
            sign = "-" if c == "-" else ""
            current = []
            continue
        current.append(c)
    return tuple(t for t in items if t.replace(" ", "") not in ("1", "0", "-1", "-0"))


def _variables(terms) -> set[str]:
    """Data variables referenced by terms (formulaic `required_variables`)."""
    out: set[str] = set()
    for term in terms:
        for f in term.factors:
            if f.kind == "name":
                out.add(f.expr)
            elif f.kind == "python":
                out |= _expr_variables(f.expr)
    return out


def _expr_variables(expr: str) -> set[str]:
    quoted = re.findall(r"`([^`]*)`", expr)
    plain = re.sub(r"`[^`]*`", "_formulaic_quoted", expr)
    tree = ast.parse(plain, mode="eval")
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    bases = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and id(n) not in called and id(n) not in bases
    }
    names.discard("_formulaic_quoted")
    return names | set(quoted)


# ---------------------------------------------------------------------------
# Fixed effects part
# ---------------------------------------------------------------------------

_FE_SLOPE = re.compile(
    r"^(?P<fe>[^\[\]]+?)\s*\[(?P<double>\[)?(?P<slopes>[^\[\]]*)\](?(double)\])$"
)


def _parse_fe(text: str) -> tuple[FeTerm, ...]:
    out: dict[str, FeTerm] = {}
    for raw in _str_split_by_sep(text, separator="+"):
        label = raw.strip()
        if not label:
            raise FormulaSyntaxError(f"Empty fixed effect in `{text}`")
        m = _FE_SLOPE.match(label)
        if m:
            fe_text = m.group("fe")
            slopes = tuple(s.strip() for s in m.group("slopes").split(","))
            if not all(slopes):
                raise FormulaSyntaxError(f"Empty slope variable in `{label}`")
            intercept = m.group("double") is None
        elif "[" in label or "]" in label:
            raise FormulaSyntaxError(f"Invalid varying-slope syntax `{label}`")
        else:
            fe_text, slopes, intercept = label, (), True
        factors = tuple(f.strip() for f in fe_text.split("^"))
        if not all(factors):
            raise FormulaSyntaxError(f"Invalid fixed effect `{label}`")
        out.setdefault(label, FeTerm(factors, intercept, slopes, label))
    return tuple(out.values())


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------


def _build(formula: str) -> Formula:
    parts = re.split(r"\s*\|\s*", formula)
    second = parts.pop(0).strip()
    first = next((p.strip() for p in parts if "~" in p), None)
    fixed = next((p.strip() for p in parts if "~" not in p), None)
    if fixed in ("0", "1"):
        fixed = None

    depvar, rhs_text = (s.strip() for s in re.split(r"\s*~\s*", second, maxsplit=1))
    if not depvar:
        raise FormulaSyntaxError(f"Missing dependent variable in `{formula}`")
    if not rhs_text:
        raise FormulaSyntaxError(f"Missing right-hand side in `{formula}`")
    intercept, exog_terms = rhs_terms(rhs_text)

    endog: tuple[str, ...] = ()
    inst: tuple[str, ...] = ()
    inst_text = ""
    if first is not None:
        endog_text, inst_text = (
            s.strip() for s in re.split(r"\s*~\s*", first, maxsplit=1)
        )
        endog = tuple(_str_split_by_sep(endog_text, "+"))
        _, inst_terms = rhs_terms(inst_text)
        inst = _summands(inst_text)
        exogenous = _variables(exog_terms)
        endogenous = set(endog)
        instruments = _variables(inst_terms)
        if len(endogenous) > len(instruments):
            raise FormulaSyntaxError(
                "The IV system is underdetermined. "
                "Please provide at least as many instruments as endogenous variables."
            )
        if endogenous & exogenous:
            raise FormulaSyntaxError(
                f"Endogeneous variables specified as covariates: {endogenous & exogenous}"
            )
        if instruments & exogenous:
            raise FormulaSyntaxError(
                f"Instruments specified as covariates: {instruments & exogenous}"
            )

    fml = second
    if fixed is not None:
        fml = f"{fml} | {fixed}"
    if first is not None:
        fml = f"{fml} | {first}"
    return Formula(
        fml=fml,
        depvar=depvar,
        rhs=_summands(rhs_text),
        intercept=intercept,
        fe=_parse_fe(fixed) if fixed is not None else (),
        endog=endog,
        inst=inst,
        fe_text=fixed or "",
        rhs_text=rhs_text,
        inst_text=inst_text,
    )


@functools.lru_cache(maxsize=256)
def _parse_cached(fml: str) -> tuple[Formula, ...]:
    _validate(fml)
    models = [_build(f) for f in _expand_all(_preprocess(fml))]
    # FixestMulti estimates models grouped by FE string (first appearance)
    groups: dict[str, list[Formula]] = {}
    for m in models:
        groups.setdefault(m.fe_text, []).append(m)
    return tuple(m for group in groups.values() for m in group)


def parse(fml: str) -> list[Formula]:
    """Expand a pyfixest formula into its models, in pyfixest's model order
    (`FixestMulti.all_fitted_models` keys equal `[f.fml for f in parse(fml)]`)."""
    if not isinstance(fml, str):
        raise TypeError("fml must be a string")
    return list(_parse_cached(fml))
