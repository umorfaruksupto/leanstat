"""Runtime error diagnosis and guarded automatic recovery.

Design rules (from the project philosophy):

* Code is executed one top-level statement at a time. When statement *k*
  fails, only statement *k* is corrected and retried - statements that already
  ran are never re-executed, so side effects are not duplicated.
* Errors are classified (syntax, name, attribute, argument, data quality,
  statistical method, environment, other runtime). Only mechanical mistakes -
  misspelled names, columns, attributes, keywords and option values, a missing
  import, an unclosed bracket - are corrected automatically, and only when the
  correction is unambiguous: a slip that is one keystroke from two valid names
  is refused, and a replacement method must accept the call as written.
* Data-quality and statistical-method errors are *explained*, never "fixed":
  converting types, dropping missing values or switching tests are analytical
  decisions that belong to the user.
* Every candidate fix is validated before it runs: it must compile, must not
  introduce a destructive operation, must not change assignment targets, must
  not add data-altering calls and must not swap statistical methods.
* Code that runs but returns a silent problem (a NaN test result, a SciPy
  small-sample warning) is left unchanged and explained in a note.
* Retries are bounded. If a fix fails or the same error recurs, recovery stops
  and reports an actionable next step. The user's source is never modified;
  the corrected code is shown as a diff.
"""

from __future__ import annotations

import ast
import builtins
import difflib
import inspect
import re
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from . import safety
from .advisor import call_name, dotted_name

COMMON_IMPORTS = {
    "pd": "import pandas as pd",
    "np": "import numpy as np",
    "plt": "import matplotlib.pyplot as plt",
    "stats": "from scipy import stats",
    "sns": "import seaborn as sns",
    "sm": "import statsmodels.api as sm",
    "math": "import math",
    "ls": "import learnstat as ls",
}

# Keyword arguments that matplotlib/pandas accept through **kwargs, so they are
# not visible in inspect.signature().
COMMON_KWARGS = {
    "color", "colors", "label", "labels", "alpha", "linewidth", "linestyle", "marker", "markersize",
    "figsize", "bins", "title", "xlabel", "ylabel", "edgecolor", "facecolor", "cmap", "kind", "ax",
    "legend", "grid", "rot", "fontsize", "density", "orientation", "stacked", "autopct", "startangle",
    "subplots", "sharex", "sharey", "xlim", "ylim", "logx", "logy", "width", "height", "align",
    "dpi", "loc", "ncol", "zorder", "hue", "palette", "equal_var", "alternative", "nan_policy",
    "method", "axis", "ascending", "normalize", "dropna", "numeric_only", "inplace", "subset",
    "how", "on", "columns", "index", "values", "aggfunc", "sep", "header", "usecols", "parse_dates",
}
KWARG_ALIASES = {"colour": "color", "colours": "colors", "fig_size": "figsize", "figSize": "figsize",
                 "line_width": "linewidth", "line_style": "linestyle", "colormap": "cmap", "xlab": "xlabel",
                 "ylab": "ylabel", "main": "title", "lwd": "linewidth", "col": "color"}

STAT_FUNCS = {
    "ttest_ind", "ttest_rel", "ttest_1samp", "mannwhitneyu", "wilcoxon", "pearsonr", "spearmanr",
    "kendalltau", "f_oneway", "kruskal", "chi2_contingency", "chisquare", "fisher_exact", "shapiro",
    "normaltest", "kstest", "anderson", "levene", "bartlett", "linregress", "ols", "OLS", "logit",
    "Logit", "glm", "GLM", "anova_lm", "corr", "mean", "median", "std", "var", "describe",
}
DATA_ALTERING = {
    "dropna", "fillna", "drop", "drop_duplicates", "replace", "astype", "to_numeric", "to_datetime",
    "interpolate", "clip", "sample", "query", "rename", "where", "mask", "bfill", "ffill", "update",
}
_STAT_MODULES = ("scipy", "statsmodels", "sklearn", "pingouin")


# ---------------------------------------------------------------------------
# Data classes


@dataclass
class Diagnosis:
    category: str  # syntax | name | attribute | argument | data_quality | statistical | environment | runtime
    exc_type: str
    message: str
    line: int | None
    statement: str
    explanation: str
    next_step: str

    @property
    def label(self) -> str:
        return {
            "syntax": "syntax error",
            "name": "undefined name",
            "attribute": "invalid attribute",
            "argument": "invalid function argument",
            "column": "column not found",
            "data_quality": "data-quality problem",
            "statistical": "statistical-method error",
            "environment": "environment / missing package",
            "runtime": "runtime exception",
        }.get(self.category, self.category)


@dataclass
class Fix:
    kind: str  # name | import | attribute | column | keyword | option | syntax | ai
    description: str
    fixed: str
    renames: dict[str, str] = field(default_factory=dict)
    needs_confirmation: bool = False


@dataclass
class Attempt:
    diagnosis: Diagnosis
    fix: Fix | None = None
    rejected_reasons: list[str] = field(default_factory=list)
    applied: bool = False
    succeeded: bool = False
    diff: str = ""


@dataclass
class RunResult:
    ok: bool
    original_code: str
    final_code: str
    attempts: list[Attempt] = field(default_factory=list)
    error: BaseException | None = None
    blocked: list[safety.SafetyFinding] = field(default_factory=list)
    value: Any = None
    stopped_at_line: int | None = None
    notes: list[str] = field(default_factory=list)  # silent problems, e.g. a test that returned NaN

    @property
    def changed(self) -> bool:
        return self.final_code != self.original_code

    @property
    def last_diagnosis(self) -> Diagnosis | None:
        return self.attempts[-1].diagnosis if self.attempts else None

    def report(self) -> str:
        if self.blocked:
            lines = ["Execution blocked: this code has side effects that need your permission:"]
            lines += [f"  - {f}" for f in self.blocked]
            lines.append("Nothing was executed. Re-run and confirm if this is intended.")
            return "\n".join(lines)
        if not self.attempts:
            if self.notes:
                return "\n".join(["Ran, but the result needs a second look:"] + [f"  - {n}" for n in self.notes])
            return "Ran without errors."
        out: list[str] = []
        for i, a in enumerate(self.attempts, 1):
            d = a.diagnosis
            where = f"line {d.line}" if d.line else "code"
            out.append(f"[{i}] {d.exc_type} at {where} - {d.label}")
            out.append(f"    {d.message}")
            out.append(f"    Why: {d.explanation}")
            if a.fix:
                status = "applied" if a.applied else ("rejected" if a.rejected_reasons else "proposed, not applied")
                out.append(f"    Fix ({status}): {a.fix.description}")
                if a.diff:
                    out += [f"      {ln}" for ln in a.diff.splitlines()]
                for r in a.rejected_reasons:
                    out.append(f"      x {r}")
                if a.applied:
                    out.append("    Result: " + ("re-ran successfully" if a.succeeded else "still failing"))
            if not a.succeeded:
                out.append(f"    Next step: {d.next_step}")
        if self.notes:
            out.append("")
            out += ["Note: " + n for n in self.notes]
        if self.ok and self.changed:
            out.append("")
            out.append("Recovered. Your original code was not modified; corrected version:")
            out += ["    " + ln for ln in self.final_code.splitlines()]
        elif not self.ok:
            out.append("")
            out.append(
                f"Stopped at line {self.stopped_at_line}; statements after it were not run."
                if self.stopped_at_line else "Nothing was run."
            )
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Helpers


def _edit_distance(a: str, b: str) -> int:
    """Optimal string alignment distance: insertions, deletions, substitutions and adjacent swaps."""
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[-1]


def _close(word: str, candidates, cutoff: float = 0.75) -> str | None:
    """Unique high-similarity match, preferring case/whitespace-insensitive equality.

    A name that is a single typing slip (one insertion, deletion, substitution
    or swap of adjacent letters) away from two or more valid names is
    ambiguous, and nothing is guessed. A name one slip from exactly one valid
    name gets that name, provided it also passes the similarity cutoff.
    Otherwise the best difflib match is used unless the top two nearly tie.
    """
    candidates = [str(c) for c in candidates]
    norm = {c.lower().strip().replace(" ", "_"): c for c in candidates}
    key = word.lower().strip().replace(" ", "_")
    if key in norm and norm[key] != word:
        return norm[key]
    one_slip = [c for c in set(candidates) if c != word and _edit_distance(word, c) == 1]
    if len(one_slip) > 1:
        return None
    matches = difflib.get_close_matches(word, candidates, n=2, cutoff=cutoff)
    if not matches:
        return None
    if one_slip:
        return one_slip[0] if one_slip[0] in matches else None
    if len(matches) == 2:
        r1 = difflib.SequenceMatcher(None, word, matches[0]).ratio()
        r2 = difflib.SequenceMatcher(None, word, matches[1]).ratio()
        if abs(r1 - r2) < 0.05:  # near tie with no single-slip candidate: do not guess
            return None
    return matches[0] if matches[0] != word else None


def _fits_call(obj: Any, attr: str, call: ast.Call | None) -> bool:
    """Whether ``obj.attr`` can be used the way the user's code uses it.

    For a call such as ``s.modw()`` the replacement must be callable and
    accept the same number of positional arguments and the same keywords;
    ``s.mod()`` needs an argument and ``s.axes`` is not callable, so neither
    can stand in for it.
    """
    if call is None:
        return True
    try:
        value = getattr(obj, attr)
    except Exception:  # noqa: BLE001 - a property that raises is not a usable replacement
        return False
    if not callable(value):
        return False
    if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
        return True
    try:
        inspect.signature(value).bind(*call.args, **{k.arg: None for k in call.keywords})
    except TypeError:
        return False
    except ValueError:
        return True  # no signature available (some builtins); do not exclude
    return True


def _diff(a: str, b: str) -> str:
    return "\n".join(
        ln for ln in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0)
        if not ln.startswith(("---", "+++", "@@"))
    )


def _replace_spans(src: str, spans: list[tuple[int, int, int, int, str]]) -> str:
    """Replace (lineno, col, end_lineno, end_col, text) spans; line numbers are 1-based within src."""
    lines = src.splitlines(keepends=True)
    offsets = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln))

    def pos(line: int, col: int) -> int:
        # ast col offsets are UTF-8 byte offsets
        text = lines[line - 1].encode("utf-8")[:col].decode("utf-8", errors="ignore")
        return offsets[line - 1] + len(text)

    out = src
    for lineno, col, end_lineno, end_col, text in sorted(spans, key=lambda s: (s[0], s[1]), reverse=True):
        a, b = pos(lineno, col), pos(end_lineno, end_col)
        out = out[:a] + text + out[b:]
    return out


def _safe_resolve(node: ast.AST, ns: dict) -> Any:
    """Resolve Name/Attribute chains (e.g. ``stats.ttest_ind``) without calling anything."""
    if isinstance(node, ast.Name):
        if node.id in ns:
            return ns[node.id]
        return getattr(builtins, node.id, None)
    if isinstance(node, ast.Attribute):
        base = _safe_resolve(node.value, ns)
        if base is None:
            return None
        try:
            return getattr(base, node.attr, None)
        except Exception:  # properties can raise
            return None
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        base = _safe_resolve(node.value, ns)
        if isinstance(base, pd.DataFrame) and node.slice.value in base.columns:
            return base[node.slice.value]
    return None


def _frames_in(tree: ast.AST, ns: dict) -> list[tuple[str, pd.DataFrame]]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(ns.get(node.id), pd.DataFrame):
            if node.id not in [n for n, _ in out]:
                out.append((node.id, ns[node.id]))
    return out


def _from_stat_library(tb) -> bool:
    for frame in traceback.extract_tb(tb):
        fn = frame.filename.replace("\\", "/")
        if any(f"/{m}/" in fn for m in _STAT_MODULES):
            return True
    return False


def _is_simple(stmt: ast.stmt) -> bool:
    """Statements that either fully complete or have no effect when they raise."""
    return isinstance(stmt, (ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Import, ast.ImportFrom))


# ---------------------------------------------------------------------------
# Diagnosis


_DATA_QUALITY_PATTERNS = [
    (r"could not convert string to float|Unable to parse string|invalid literal for int",
     "A column that should be numeric contains text values."),
    (r"unsupported operand type|can only concatenate str|must be real number, not str|"
     r"Could not convert .* to numeric|agg function failed|not supported between instances of 'str'",
     "An operation mixed numbers with text - usually a numeric column stored as strings (dtype object)."),
    (r"Cannot convert non-finite values|cannot convert float NaN to integer|Cannot convert NA to integer",
     "The data contains missing values (NaN) where whole numbers were expected."),
    (r"time data .* does not match format|Unknown datetime string format|Out of bounds nanosecond",
     "Some date values do not match the expected date format."),
]
_STATISTICAL_PATTERNS = [
    (r"expected frequenc", "Some cells of the contingency table have zero (or very low) expected counts, "
     "so the chi-square approximation is not valid."),
    (r"at least (2|two|3|three)|must have length at least|need at least|too few|not enough",
     "There are too few observations for this test (often because a filter or dropna left a group almost empty)."),
    (r"infs or NaNs|contains NaN|nan values|NaN", "The input contains missing or infinite values, which this method cannot handle."),
    (r"zero-size array|empty|0 samples", "The data passed to the method is empty."),
    (r"singular matrix|Singular|perfect separation|collinear",
     "The model's predictors are perfectly correlated or separate the outcome perfectly."),
    (r"constant|identical", "The input has no variation (all values equal), so the statistic is undefined."),
    (r"negative|non-negative", "This method requires non-negative values (e.g. counts)."),
]


_WARNING_PATTERNS = [
    (r"too small|SmallSample|at least \d|fewer than", "the sample (or one group) is too small for this method, "
     "so it returned NaN. Check group sizes with `len()` or `.value_counts()` before testing."),
    (r"Degrees of freedom <= 0|ddof", "there are not enough observations to estimate a variance."),
    (r"Precision loss|catastrophic cancellation|nearly identical", "the values are (almost) constant, so the "
     "statistic is not reliable."),
    (r"invalid value encountered|divide by zero|Mean of empty slice", "a calculation produced NaN or infinity, "
     "usually because of missing values or an empty selection."),
    (r"[Cc]hi-?squared approximation|expected frequenc", "some expected counts are very low, so the chi-square "
     "approximation may be invalid."),
]


def explain_warnings(caught) -> list[str]:
    """Turn warnings raised by statistical code into plain-language notes (deduplicated)."""
    notes: list[str] = []
    for w in caught:
        text = f"{w.category.__name__}: {w.message}"
        from_stats = any(f"/{m}/" in str(w.filename).replace("\\", "/") for m in _STAT_MODULES + ("numpy", "pandas"))
        for pattern, why in _WARNING_PATTERNS:
            if re.search(pattern, text) and (from_stats or "SmallSample" in text):
                note = f"{text.splitlines()[0][:160]} - this means {why}"
                if note not in notes:
                    notes.append(note)
                break
    return notes


def _stat_calls(stmt: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(stmt) if isinstance(n, ast.Call)
            and (dotted_name(n.func) or "").split(".")[0] in ("stats", "scipy", "st")]


def _nan_result_notes(stmt: ast.stmt, value: Any, ns: dict, returned: list | None = None) -> list[str]:
    """A statistical call that ran but returned NaN (SciPy's default nan_policy) gets a note.

    ``returned`` holds values returned by calls made directly from the
    statement, so a test nested in print(...) or an f-string is checked too.
    """
    calls = _stat_calls(stmt)
    if not calls:
        return []
    results = [value] + list(returned or [])
    if isinstance(stmt, ast.Assign):
        results += [ns.get(t.id) for t in stmt.targets if isinstance(t, ast.Name)]
    for res in results:
        stat = getattr(res, "statistic", None)
        p = getattr(res, "pvalue", None)
        try:
            if any(v is not None and pd.isna(v) for v in (stat, p)):
                name = call_name(calls[0])
                return [f"`{name}` returned NaN. Its inputs probably contain missing values, which SciPy passes "
                        "through by default (nan_policy='propagate'). Decide how to handle them, for example "
                        "nan_policy='omit' or an explicit dropna on the columns used; learnstat will not choose for you."]
        except (TypeError, ValueError):
            continue
    return []


def diagnose(exc: BaseException, statement: str, line: int | None, ns: dict) -> Diagnosis:
    et = type(exc).__name__
    msg = str(exc).strip() or et
    d = Diagnosis("runtime", et, msg, line, statement, "", "")

    if isinstance(exc, SyntaxError):
        d.category = "syntax"
        d.message = f"{exc.msg} (line {exc.lineno})"
        d.line = exc.lineno
        d.explanation = "Python could not parse the code, so nothing was run."
        d.next_step = "Look at the line shown (and the one before it) for an unclosed bracket, quote or missing ':'."
        return d
    if isinstance(exc, ImportError):
        d.category = "environment"
        found = re.search(r"No module named '([\w.]+)'|^(\w+) is required", msg)
        mod = getattr(exc, "name", None) or (found and (found.group(1) or found.group(2))) or "the module"
        d.explanation = f"`{mod}` is not installed in this Python environment (or the import name is wrong)."
        d.next_step = f"Install it yourself (e.g. `pip install {mod.split('.')[0]}`); learnstat never installs packages."
        return d
    if isinstance(exc, NameError):
        d.category = "name"
        name = getattr(exc, "name", None) or (re.findall(r"'(\w+)'", msg) or ["?"])[0]
        d.explanation = f"`{name}` was used before it was defined or imported (or it is misspelled)."
        d.next_step = f"Define or import `{name}` before this line, or check its spelling."
        return d
    if isinstance(exc, (TypeError, AttributeError)) and "unexpected keyword argument" in msg:
        # matplotlib reports bad style keywords as AttributeError from Artist.set()
        d.category = "argument"
        d.explanation = "A function was given a keyword argument it does not accept (often a misspelling)."
        d.next_step = "Check the function's parameters with `help(function)`."
        return d
    if isinstance(exc, AttributeError):
        d.category = "attribute"
        name = getattr(exc, "name", None) or "?"
        obj = getattr(exc, "obj", None)
        d.explanation = f"`{type(obj).__name__ if obj is not None else 'the object'}` has no attribute `{name}`."
        d.next_step = "Check the spelling, or use `dir(obj)` / `help(obj)` to list what is available."
        if isinstance(obj, pd.DataFrame):
            d.next_step += " For columns, `df['name']` is safer than `df.name`."
        return d
    if isinstance(exc, KeyError):
        d.category = "column"
        key = exc.args[0] if exc.args else "?"
        d.explanation = f"The key {key!r} does not exist - for DataFrames this usually means a misspelled column name."
        d.next_step = "Print `list(df.columns)` and use the exact column name."
        return d
    if isinstance(exc, (TypeError, ValueError)) and re.search(r"missing \d+ required positional|takes \d+ positional|"
                                                              r"is not a valid|Invalid|invalid value|must be one of|"
                                                              r"must be either|was supplied|expected one of", msg):
        d.category = "argument"
        d.explanation = "A function was called with an argument value or number of arguments it does not accept."
        d.next_step = "Compare the call with the function's documentation (`help(function)`)."
        return d
    for pattern, why in _DATA_QUALITY_PATTERNS:
        if re.search(pattern, msg, re.I):
            d.category = "data_quality"
            d.explanation = why
            d.next_step = _data_quality_next_step(statement, ns)
            return d
    if _from_stat_library(exc.__traceback__) or isinstance(exc, (ValueError, ZeroDivisionError, FloatingPointError)):
        for pattern, why in _STATISTICAL_PATTERNS:
            if re.search(pattern, msg, re.I):
                d.category = "statistical"
                d.explanation = why
                d.next_step = (
                    "Check the input sizes and values (`len()`, `.isna().sum()`, `.value_counts()`), then decide "
                    "whether the method suits this data - see /alternatives. learnstat will not change the method for you."
                )
                return d
        if _from_stat_library(exc.__traceback__):
            d.category = "statistical"
            d.explanation = "The statistical routine rejected its input."
            d.next_step = "Read the message above, check the inputs, and use /describe or /alternatives for options."
            return d
    d.explanation = "The code raised an exception while running."
    d.next_step = "Read the message, inspect the variables involved on that line, and fix the logic."
    return d


def _data_quality_next_step(statement: str, ns: dict) -> str:
    try:
        tree = ast.parse(statement)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                obj = _safe_resolve(node, ns)
                if isinstance(obj, pd.Series) and not pd.api.types.is_numeric_dtype(obj):
                    frame = dotted_name(node.value) or "df"
                    col = node.slice.value
                    return (
                        f"`{frame}['{col}']` has dtype {obj.dtype}. Inspect the offending values with "
                        f"`{frame}[pd.to_numeric({frame}['{col}'], errors='coerce').isna()]['{col}'].unique()`, "
                        f"then decide how to convert them (e.g. `pd.to_numeric(..., errors='coerce')`) and "
                        "record that decision with `/memory decide ...`. learnstat will not change your data automatically."
                    )
    return (
        "Inspect the column types with `df.dtypes` and the offending values, then decide how to clean them. "
        "learnstat will not change your data automatically."
    )


# ---------------------------------------------------------------------------
# Fix proposal (rule-based)


def propose_fix(exc: BaseException, d: Diagnosis, stmt_src: str, ns: dict) -> Fix | None:
    if d.category == "syntax":
        return _fix_syntax(exc, stmt_src)
    try:
        tree = ast.parse(stmt_src)
    except SyntaxError:
        return None

    if d.category == "name":
        name = getattr(exc, "name", None) or (re.findall(r"'(\w+)'", str(exc)) or [None])[0]
        if not name:
            return None
        if name in COMMON_IMPORTS:
            imp = COMMON_IMPORTS[name]
            return Fix("import", f"added the missing import `{imp}`", f"{imp}\n{stmt_src}")
        candidates = [k for k in ns if not k.startswith("_")] + dir(builtins)
        new = _close(name, candidates, cutoff=0.8)
        if new:
            spans = [(n.lineno, n.col_offset, n.end_lineno, n.end_col_offset, new)
                     for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == name]
            return Fix("name", f"renamed `{name}` to the existing name `{new}`", _replace_spans(stmt_src, spans), {name: new})
        return None

    if d.category == "attribute":
        name, obj = getattr(exc, "name", None), getattr(exc, "obj", None)
        if not name or obj is None:
            return None
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == name]
        if not nodes:
            return None
        public = [a for a in dir(obj) if not a.startswith("_")]
        if isinstance(obj, pd.DataFrame):
            col = _close(name, list(obj.columns), cutoff=0.8)
            if col is not None and _close(name, public, cutoff=0.8) is None:
                spans = [(n.value.end_lineno, n.value.end_col_offset, n.end_lineno, n.end_col_offset, f"[{col!r}]")
                         for n in nodes]
                return Fix("column", f"`{name}` is not an attribute; selected the column `{col}` with `[{col!r}]`",
                           _replace_spans(stmt_src, spans), {name: col})
        call = next((c for c in ast.walk(tree) if isinstance(c, ast.Call) and c.func in nodes), None)
        near = [a for a in public if _edit_distance(name, a) == 1
                or difflib.SequenceMatcher(None, name, a).ratio() >= 0.75 or a.lower() == name.lower()]
        new = _close(name, [a for a in near if _fits_call(obj, a, call)], cutoff=0.75)
        if new:
            spans = [(n.value.end_lineno, n.value.end_col_offset, n.end_lineno, n.end_col_offset, f".{new}")
                     for n in nodes]
            return Fix("attribute", f"`{name}` -> `{new}` (closest attribute of {type(obj).__name__})",
                       _replace_spans(stmt_src, spans), {name: new})
        return None

    if d.category == "column":
        key = exc.args[0] if exc.args else None
        if isinstance(key, str) and re.search(r"not in index|not in columns", key):
            keys = re.findall(r"'([^']+)'", key)  # list selection: "['Income'] not in index"
        elif isinstance(key, str):
            keys = [key]
        else:
            keys = re.findall(r"'([^']+)'", str(key))
        frames = _frames_in(tree, ns)
        columns = [str(c) for _, f in frames for c in f.columns]
        if not columns:
            return None
        spans, renames = [], {}
        for k in keys:
            if k in columns:
                continue
            new = _close(k, columns, cutoff=0.75)
            if not new:
                continue
            renames[k] = new
            for n in ast.walk(tree):
                if isinstance(n, ast.Constant) and n.value == k:
                    spans.append((n.lineno, n.col_offset, n.end_lineno, n.end_col_offset, repr(new)))
        if spans:
            desc = ", ".join(f"'{a}' -> '{b}'" for a, b in renames.items())
            return Fix("column", f"corrected column name {desc} (closest existing column)", _replace_spans(stmt_src, spans), renames)
        return None

    if d.category == "argument":
        m = re.search(r"unexpected keyword argument '(\w+)'", str(exc))
        if m:
            bad = m.group(1)
            for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
                kw = next((k for k in call.keywords if k.arg == bad), None)
                if not kw:
                    continue
                params: set[str] = set()
                func = _safe_resolve(call.func, ns)
                if func is not None:
                    try:
                        params = {p for p in inspect.signature(func).parameters if p not in ("args", "kwargs")}
                    except (TypeError, ValueError):
                        pass
                new = KWARG_ALIASES.get(bad) or _close(bad, params | COMMON_KWARGS, cutoff=0.75)
                if new and new != bad:
                    end_col = kw.value.col_offset
                    span = (kw.lineno, kw.col_offset, kw.value.lineno, end_col, f"{new}=")
                    return Fix("keyword", f"keyword `{bad}=` -> `{new}=`", _replace_spans(stmt_src, [span]), {bad: new})
            return None
        # invalid option value, e.g. kind='barr' or method='pearsn'
        msg = str(exc)
        valid = set(re.findall(r"'([A-Za-z_][\w-]*)'", msg)) | set(re.findall(r"\b([a-z_]{2,})\b", msg.split(":")[-1]))
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value and n.value in msg:
                options = valid - {n.value}
                new = _close(n.value, options, cutoff=0.75)
                if new:
                    return Fix("option", f"option value '{n.value}' -> '{new}' (closest valid choice)",
                               _replace_spans(stmt_src, [(n.lineno, n.col_offset, n.end_lineno, n.end_col_offset, repr(new))]),
                               {n.value: new})
    return None


def _fix_syntax(exc: BaseException, code: str) -> Fix | None:
    msg = getattr(exc, "msg", str(exc))
    lineno = getattr(exc, "lineno", None) or 1
    lines = code.splitlines()

    def split_comment(s: str) -> tuple[str, str]:
        in_str = None
        for i, ch in enumerate(s):
            if in_str:
                if ch == in_str:
                    in_str = None
            elif ch in "\"'":
                in_str = ch
            elif ch == "#":
                return s[:i].rstrip(), "  " + s[i:]
        return s.rstrip(), ""

    if "expected ':'" in msg and 0 < lineno <= len(lines):
        body, comment = split_comment(lines[lineno - 1])
        lines[lineno - 1] = body + ":" + comment
        return Fix("syntax", f"added the missing ':' at the end of line {lineno}", "\n".join(lines))
    m = re.search(r"Missing parentheses in call to '(\w+)'", msg)
    if m and 0 < lineno <= len(lines):
        body, comment = split_comment(lines[lineno - 1])
        new = re.sub(rf"\b{m.group(1)}\s+(.+)$", rf"{m.group(1)}(\1)", body)
        lines[lineno - 1] = new + comment
        return Fix("syntax", f"added parentheses to the `{m.group(1)}` call", "\n".join(lines))
    if "was never closed" in msg and 0 < lineno <= len(lines):
        body, comment = split_comment(lines[lineno - 1])
        stack, in_str = [], None  # (closer, index of opener)
        pairs = {"(": ")", "[": "]", "{": "}"}
        for i, ch in enumerate(body):
            if in_str:
                if ch == in_str:
                    in_str = None
            elif ch in "\"'":
                in_str = ch
            elif ch in pairs:
                stack.append((pairs[ch], i))
            elif ch in pairs.values() and stack and stack[-1][0] == ch:
                stack.pop()
        if not stack:
            return None

        def build(text: str) -> str:
            out = list(lines)
            out[lineno - 1] = text + comment
            return "\n".join(out)

        # Default: close at the end of the line. That is what people usually forget.
        at_end = body + "".join(c for c, _ in reversed(stack))
        if _plausible(build(at_end)):
            return Fix("syntax", f"closed the unclosed bracket at the end of line {lineno}", build(at_end))
        if len(stack) != 1:
            return None
        # Closing at the end gives code that cannot be right (e.g. a method called on a
        # list literal). Try every other place where the bracket could close, and
        # apply only if exactly one of them is plausible.
        closer, start = stack[0]
        options, depth, in_str = [], 0, None
        for i in range(start + 1, len(body)):
            ch = body[i]
            if in_str:
                if ch == in_str:
                    in_str = None
            elif ch in "\"'":
                in_str = ch
            elif ch in pairs:
                depth += 1
            elif ch in pairs.values():
                depth -= 1
                if depth == 0 and i + 1 < len(body):
                    candidate = body[: i + 1] + closer + body[i + 1:]
                    if _plausible(build(candidate)):
                        options.append((i + 1, candidate))
        if len(options) == 1:
            pos, candidate = options[0]
            return Fix("syntax", f"closed the unclosed bracket after `{body[start:pos].strip()}` on line {lineno} "
                       "(closing it at the end of the line would not make sense)", build(candidate))
    return None


def _plausible(code: str) -> bool:
    """Compiles, and does not use a method a literal list/tuple/dict/str cannot have."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    literal_types = {ast.List: list, ast.Tuple: tuple, ast.Dict: dict, ast.Set: set}
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            base = n.value
            kind = literal_types.get(type(base)) or (type(base.value) if isinstance(base, ast.Constant) else None)
            if kind is not None and not hasattr(kind, n.attr):
                return False
    return True


# ---------------------------------------------------------------------------
# Validation


def _assign_targets(tree: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(tree):
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign)) else []
        out |= {ast.dump(t) for t in targets}
    return out


def _call_names(tree: ast.AST) -> list[str]:
    return [call_name(n) for n in ast.walk(tree) if isinstance(n, ast.Call)]


def validate_fix(original: str, fix: Fix) -> list[str]:
    """Return reasons to reject the fix (empty list = acceptable)."""
    problems: list[str] = []
    try:
        new_tree = ast.parse(fix.fixed)
    except SyntaxError as e:
        return [f"the corrected code does not compile ({e.msg})"]
    try:
        old_tree = ast.parse(original)
    except SyntaxError:
        # syntax fixes: the original cannot be scanned, so check only the lines the fix changed
        old_lines, new_lines = original.splitlines(), fix.fixed.splitlines()
        changed = {i + 1 for i, ln in enumerate(new_lines) if i >= len(old_lines) or old_lines[i] != ln}
        return [f"it would introduce a side effect: {f}" for f in safety.scan(fix.fixed) if f.line in changed]
    for f in safety.introduced_by(original, fix.fixed):
        problems.append(f"it would introduce a side effect: {f}")

    def renamed(names):
        return [fix.renames.get(n, n) for n in names]

    if _assign_targets(old_tree) != _assign_targets(new_tree) and fix.kind not in ("column", "name"):
        problems.append("it changes which variables are assigned")
    old_calls, new_calls = renamed(_call_names(old_tree)), _call_names(new_tree)
    added_data_ops = set(new_calls) & DATA_ALTERING - set(old_calls)
    if added_data_ops:
        problems.append(f"it would alter the data ({', '.join(sorted(added_data_ops))}) - that is your decision to make")
    if sorted(c for c in old_calls if c in STAT_FUNCS) != sorted(c for c in new_calls if c in STAT_FUNCS):
        problems.append("it would change the statistical method")
    old_kw = {(k.arg, ast.dump(k.value)) for n in ast.walk(old_tree) if isinstance(n, ast.Call) for k in n.keywords}
    new_kw = {(fix.renames.get(k.arg, k.arg), ast.dump(k.value)) for n in ast.walk(new_tree) if isinstance(n, ast.Call) for k in n.keywords}
    if fix.kind == "ai" and old_kw - new_kw:
        problems.append("it changes the arguments of a call")
    return problems


# ---------------------------------------------------------------------------
# Execution


Confirm = Callable[[str], bool]


class Executor:
    """Runs code statement by statement with bounded, validated recovery."""

    def __init__(
        self,
        namespace: dict,
        max_attempts: int = 3,
        auto_fix: bool = True,
        confirm: Confirm | None = None,
        ai_fixer: Callable[[Diagnosis, str], Fix | None] | None = None,
        filename: str = "<learnstat>",
    ):
        self.ns = namespace
        self.max_attempts = max_attempts
        self.auto_fix = auto_fix
        self.confirm = confirm
        self.ai_fixer = ai_fixer
        self.filename = filename
        self._caught: list = []
        self._returned: list = []

    # -- public -----------------------------------------------------------
    def run(self, code: str, start_line: int | None = None, initial_exc: BaseException | None = None) -> RunResult:
        """Run ``code``. With ``start_line``/``initial_exc``, resume after an
        error raised elsewhere (e.g. by IPython) at that line: earlier statements
        are treated as already executed."""
        result = RunResult(ok=False, original_code=code, final_code=code)

        tree, code = self._parse_with_recovery(code, result)
        if tree is None:
            return result
        result.final_code = code

        if initial_exc is None:
            findings = safety.scan(code)
            if findings and not self._ask("This code has side effects:\n" + "\n".join(f"  - {f}" for f in findings) + "\nRun it?"):
                result.blocked = findings
                return result

        replacements: dict[int, str] = {}
        lines = code.splitlines()
        for idx, stmt in enumerate(tree.body):
            if start_line is not None and stmt.end_lineno < start_line:
                continue  # already executed by the caller
            src = "\n".join(lines[stmt.lineno - 1: stmt.end_lineno])
            exc = initial_exc if (initial_exc is not None and stmt.lineno <= start_line <= stmt.end_lineno) else None
            ok, value, new_src = self._run_statement(stmt, src, result, pending_exc=exc)
            if new_src != src:
                replacements[idx] = new_src
            if not ok:
                result.stopped_at_line = stmt.lineno
                result.final_code = self._rebuild(code, tree, replacements)
                return result
            result.value = value
        result.ok = True
        result.final_code = self._rebuild(code, tree, replacements)
        return result

    # -- internals --------------------------------------------------------
    def _ask(self, question: str) -> bool:
        return bool(self.confirm and self.confirm(question))

    def _parse_with_recovery(self, code: str, result: RunResult) -> tuple[ast.Module | None, str]:
        for _ in range(self.max_attempts + 1):
            try:
                return ast.parse(code, filename=self.filename), code
            except SyntaxError as e:
                d = diagnose(e, code, e.lineno, self.ns)
                attempt = Attempt(d)
                result.attempts.append(attempt)
                result.error = e
                if not self.auto_fix:
                    return None, code
                fix = _fix_syntax(e, code) or self._ai_fix(d, code)
                if not fix:
                    return None, code
                attempt.fix, attempt.diff = fix, _diff(code, fix.fixed)
                attempt.rejected_reasons = validate_fix(code, fix)
                if attempt.rejected_reasons or not self._approved(fix, attempt.diff):
                    return None, code
                try:
                    ast.parse(fix.fixed)
                except SyntaxError:
                    attempt.applied = True
                    code = fix.fixed  # progress was made; try the next syntax error
                    continue
                attempt.applied = attempt.succeeded = True
                code = fix.fixed
        return None, code

    def _approved(self, fix: Fix, diff: str) -> bool:
        if fix.needs_confirmation:
            return self._ask(f"Apply this correction?\n  {fix.description}\n{diff}")
        return True

    def _ai_fix(self, d: Diagnosis, src: str) -> Fix | None:
        if not self.ai_fixer:
            return None
        try:
            fix = self.ai_fixer(d, src)
        except Exception:
            return None
        if fix:
            fix.needs_confirmation = True  # AI-proposed edits always need explicit approval
        return fix

    def _exec(self, src_tree: ast.Module) -> Any:
        """Run one statement. Warnings are recorded for explanation and then passed on unchanged.

        For a statement that calls SciPy, the values returned by calls made
        directly from the statement are recorded too, so that a NaN test
        result inside print(...) can be explained. The code itself runs as written.
        """
        body = src_tree.body
        self._returned = []
        previous = sys.getprofile()
        if _stat_calls(src_tree):
            def tap(frame, event, arg):
                if event == "return" and frame.f_back is not None and frame.f_back.f_code.co_filename == self.filename                         and (hasattr(arg, "pvalue") or hasattr(arg, "statistic")):
                    self._returned.append(arg)
            sys.setprofile(tap)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if len(body) == 1 and isinstance(body[0], ast.Expr):
                    expr = ast.Expression(body[0].value)
                    ast.copy_location(expr, body[0])
                    return eval(compile(expr, self.filename, "eval"), self.ns)  # noqa: S307 - the user's own code
                exec(compile(src_tree, self.filename, "exec"), self.ns)  # noqa: S102 - the user's own code
                return None
        finally:
            sys.setprofile(previous)
            self._caught = list(caught)
            for w in self._caught:  # pass them on under the caller's own warning filters
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    def _module_for(self, src: str, first_line: int) -> ast.Module:
        tree = ast.parse(src, filename=self.filename)
        ast.increment_lineno(tree, first_line - 1)
        return tree

    def _run_statement(self, stmt: ast.stmt, src: str, result: RunResult, pending_exc: BaseException | None = None):
        current = src
        seen_errors: set[tuple[str, str]] = set()
        for attempt_no in range(self.max_attempts + 1):
            if pending_exc is None:
                try:
                    value = self._exec(self._module_for(current, stmt.lineno))
                    self._note_warnings(result, stmt, value)
                    return True, value, current
                except Exception as e:  # noqa: BLE001 - we diagnose anything the user's code raises
                    exc = e
            else:
                exc, pending_exc = pending_exc, None

            result.error = exc
            line = self._error_line(exc, stmt)
            d = diagnose(exc, current, line, self.ns)
            attempt = Attempt(d)
            result.attempts.append(attempt)

            signature = (type(exc).__name__, str(exc))
            if signature in seen_errors:
                d.next_step = "The same error came back after a correction, so recovery stopped. " + d.next_step
                return False, None, current
            seen_errors.add(signature)

            if not self.auto_fix or attempt_no >= self.max_attempts:
                if attempt_no >= self.max_attempts:
                    d.next_step = f"Stopped after {self.max_attempts} correction attempts. " + d.next_step
                return False, None, current
            if d.category in ("data_quality", "statistical", "environment"):
                return False, None, current  # explain, never auto-change data or methods

            fix = propose_fix(exc, d, current, self.ns) or self._ai_fix(d, current)
            if not fix:
                return False, None, current
            attempt.fix, attempt.diff = fix, _diff(current, fix.fixed)
            attempt.rejected_reasons = validate_fix(current, fix)
            if attempt.rejected_reasons:
                return False, None, current
            if not _is_simple(stmt) and not fix.needs_confirmation:
                fix.needs_confirmation = True  # re-running a loop/block could repeat side effects
            if fix.needs_confirmation and not self._ask(
                f"Proposed correction for line {stmt.lineno}: {fix.description}\n{attempt.diff}\nApply and re-run this statement?"
            ):
                return False, None, current
            attempt.applied = True
            current = fix.fixed
            try:
                value = self._exec(self._module_for(current, stmt.lineno))
                attempt.succeeded = True
                self._note_warnings(result, stmt, value)
                return True, value, current
            except Exception as e:  # noqa: BLE001
                pending_exc = e
        return False, None, current

    def _note_warnings(self, result: RunResult, stmt: ast.stmt | None = None, value: Any = None) -> None:
        notes = explain_warnings(self._caught)
        if stmt is not None:
            notes += _nan_result_notes(stmt, value, self.ns, self._returned)
        result.notes += [n for n in notes if n not in result.notes]

    def _error_line(self, exc: BaseException, stmt: ast.stmt) -> int | None:
        for frame in reversed(traceback.extract_tb(exc.__traceback__)):
            if frame.filename == self.filename:
                return frame.lineno
        return stmt.lineno

    @staticmethod
    def _rebuild(code: str, tree: ast.Module, replacements: dict[int, str]) -> str:
        if not replacements:
            return code
        lines = code.splitlines()
        for idx in sorted(replacements, reverse=True):
            stmt = tree.body[idx]
            lines[stmt.lineno - 1: stmt.end_lineno] = replacements[idx].splitlines()
        return "\n".join(lines) + ("\n" if code.endswith("\n") else "")
