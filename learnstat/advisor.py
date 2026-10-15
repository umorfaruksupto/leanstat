"""Context-aware code advice.

The advisor reads the user's code with ``ast`` (it never executes it), works
out which dataset columns each call touches, and checks the call against the
dataset profile. Every rule returns a :class:`Suggestion` that explains the
problem, the alternative, the reason and how to implement it. The user's code
is never modified.
"""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass
from typing import Callable, Iterator

from .profile import ColumnProfile, DatasetProfile
from .suggestions import Suggestion

PIE_MAX_CATEGORIES = 6
SKEW_THRESHOLD = 1.0


# ---------------------------------------------------------------------------
# AST helpers


def dotted_name(node: ast.AST) -> str | None:
    """``plt.pie`` -> 'plt.pie', ``stats.ttest_ind`` -> 'stats.ttest_ind'."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def call_name(call: ast.Call) -> str:
    """Last attribute of a call: ``df['a'].value_counts().plot.pie()`` -> 'pie'."""
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def kwarg(call: ast.Call, name: str) -> ast.AST | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def const_value(node: ast.AST | None):
    return node.value if isinstance(node, ast.Constant) else None


def frame_base(node: ast.AST) -> ast.Name | None:
    """The DataFrame variable behind ``df`` or a row filter like ``df[df['g'] == 'a']``."""
    while isinstance(node, ast.Subscript) and not isinstance(const_value(node.slice), str):
        node = node.value
    return node if isinstance(node, ast.Name) else None


@dataclass
class ColumnRef:
    frame: str | None  # variable name of the DataFrame, if known
    column: str
    profile: DatasetProfile
    info: ColumnProfile


class Resolver:
    """Maps ``df['col']`` / ``df.col`` expressions to profiled columns."""

    def __init__(self, frames: dict[str, DatasetProfile], profiles: list[DatasetProfile]):
        self.frames = frames  # variable name -> profile
        self.profiles = profiles  # every known profile (fallback lookup by column name)
        self.aliases: dict[str, ColumnRef] = {}  # treated = df[df.g == 'a']['score']

    def learn_aliases(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                ref = self.ref(node.value)
                if ref:
                    self.aliases[node.targets[0].id] = ref

    def _lookup(self, frame: str | None, column: str) -> ColumnRef | None:
        if frame and frame in self.frames:
            prof = self.frames[frame]
            info = prof.get(column)
            return ColumnRef(frame, column, prof, info) if info else None
        for prof in self.profiles:
            info = prof.get(column)
            if info:
                return ColumnRef(frame, column, prof, info)
        return None

    def ref(self, node: ast.AST) -> ColumnRef | None:
        if isinstance(node, ast.Name) and node.id in self.aliases:
            return self.aliases[node.id]
        if isinstance(node, ast.Subscript):
            base = frame_base(node.value)
            col = const_value(node.slice)
            if base is not None and isinstance(col, str):
                return self._lookup(base.id, col)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            name = node.value.id
            if name in self.frames or (not self.frames and self.profiles):
                return self._lookup(name, node.attr)
        return None

    def refs_in(self, node: ast.AST) -> list[ColumnRef]:
        out, seen = [], set()
        for sub in ast.walk(node):
            r = self.ref(sub)
            if r and (r.frame, r.column) not in seen:
                seen.add((r.frame, r.column))
                out.append(r)
        return out

    def first_ref(self, node: ast.AST) -> ColumnRef | None:
        refs = self.refs_in(node)
        return refs[0] if refs else None

    def is_known_frame(self, name: str) -> bool:
        return name in self.frames


def _src(code: str, node: ast.AST) -> str:
    return (ast.get_source_segment(code, node) or "").strip()


def _calls(tree: ast.AST) -> Iterator[ast.Call]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def _expr_text(ref: ColumnRef) -> str:
    return f"{ref.frame or 'df'}['{ref.column}']"


# ---------------------------------------------------------------------------
# Rules. Each takes (call, resolver, code) and returns a Suggestion or None.

Rule = Callable[[ast.Call, Resolver, str], "Suggestion | None"]


def rule_pie_many_categories(call: ast.Call, r: Resolver, code: str) -> Suggestion | None:
    name = call_name(call)
    is_pie = name == "pie" or (name == "plot" and const_value(kwarg(call, "kind")) == "pie")
    if not is_pie:
        return None
    ref = r.first_ref(call)
    if not ref or ref.info.kind not in ("categorical", "boolean", "text", "numeric"):
        return None
    k = ref.info.n_unique
    if k <= PIE_MAX_CATEGORIES:
        return None
    e = _expr_text(ref)
    return Suggestion(
        rule="pie_many_categories",
        concept="part_to_whole_charts",
        title=f"Pie chart with {k} categories",
        problem=f"`{ref.column}` has {k} categories; slices this numerous are hard to compare by angle.",
        alternative="A horizontal bar chart of the counts, sorted by size.",
        rationale=(
            "Bars share a common baseline, so readers compare lengths, which is far more accurate than "
            f"comparing angles. Pies only work for about {PIE_MAX_CATEGORIES} or fewer parts of one whole."
        ),
        code=(
            f"counts = {e}.value_counts()\n"
            "ax = counts.sort_values().plot.barh()\n"
            f"ax.set_xlabel('Count'); ax.set_title('{ref.column}')"
        ),
        severity="warning",
        columns=[ref.column],
    )


def rule_mean_of_skewed(call: ast.Call, r: Resolver, code: str) -> Suggestion | None:
    if call_name(call) != "mean" or not isinstance(call.func, ast.Attribute):
        return None
    ref = r.ref(call.func.value)
    if not ref or ref.info.kind != "numeric":
        return None
    skew = ref.info.skew
    if skew is None or abs(skew) < SKEW_THRESHOLD:
        return None
    s = ref.info.summary
    e = _expr_text(ref)
    return Suggestion(
        rule="mean_of_skewed",
        concept="center_of_skewed_data",
        title=f"Mean of a skewed variable (`{ref.column}`)",
        problem=(
            f"`{ref.column}` is skewed (skew={skew}); its mean ({s.get('mean')}) differs from its "
            f"median ({s.get('median')}) and may not describe a typical value."
        ),
        alternative="Report the median and IQR alongside (or instead of) the mean.",
        rationale="The median is not pulled by the long tail, so it represents a typical observation better.",
        code=(
            f"{e}.median()\n"
            f"{e}.quantile([0.25, 0.75])   # IQR\n"
            f"{e}.describe()              # mean and quartiles side by side"
        ),
        columns=[ref.column],
    )


def rule_ttest_equal_var(call: ast.Call, r: Resolver, code: str) -> Suggestion | None:
    if call_name(call) != "ttest_ind" or kwarg(call, "equal_var") is not None:
        return None
    args = ", ".join(_src(code, a) for a in call.args) or "a, b"
    return Suggestion(
        rule="ttest_equal_var",
        concept="welch_t_test",
        title="t-test assumes equal variances",
        problem="`ttest_ind` defaults to Student's t-test, which assumes both groups have equal variance.",
        alternative="Welch's t-test: the same function with `equal_var=False`.",
        rationale=(
            "Welch's test stays valid when variances or group sizes differ and costs almost nothing when "
            "they are equal, so it is the safer default."
        ),
        code=f"stats.ttest_ind({args}, equal_var=False)",
        severity="warning",
        columns=[x.column for x in r.refs_in(call)],
    )


_SCIPY_TESTS = {
    "ttest_ind", "ttest_rel", "ttest_1samp", "mannwhitneyu", "wilcoxon", "pearsonr", "spearmanr",
    "kendalltau", "f_oneway", "kruskal", "shapiro", "normaltest", "linregress",
}


def rule_nan_in_scipy(call: ast.Call, r: Resolver, code: str) -> Suggestion | None:
    name = call_name(call)
    if name not in _SCIPY_TESTS or kwarg(call, "nan_policy") is not None:
        return None
    text = _src(code, call)
    if "dropna" in text:
        return None
    missing = [x for x in r.refs_in(call) if x.info.n_missing > 0]
    if not missing:
        return None
    cols = ", ".join(f"`{x.column}` ({x.info.n_missing} missing)" for x in missing)
    return Suggestion(
        rule="nan_in_scipy",
        concept="missing_values_in_tests",
        title=f"Missing values passed to `{name}`",
        problem=f"{cols}. SciPy may return `nan` or fail instead of skipping them like pandas does.",
        alternative="Handle missing values explicitly - drop them for this test, or use `nan_policy='omit'`.",
        rationale=(
            "Which rows enter a test is an analytical decision. Making it explicit keeps the result "
            "reproducible and lets you report the actual sample size."
        ),
        code=(
            f"# option 1: drop missing values for this test only\n"
            f"{_expr_text(missing[0])}.dropna()\n"
            f"# option 2: let SciPy skip them (supported by most scipy.stats tests)\n"
            f"{name}(..., nan_policy='omit')"
        ),
        severity="warning",
        columns=[x.column for x in missing],
    )


def rule_pearson_on_skewed(call: ast.Call, r: Resolver, code: str) -> Suggestion | None:
    name = call_name(call)
    if name == "corr":
        method = const_value(kwarg(call, "method")) or "pearson"
        if method != "pearson":
            return None
        refs = r.refs_in(call)
        if not refs and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
            prof = r.frames.get(call.func.value.id)
            refs = [ColumnRef(call.func.value.id, c, prof, prof.columns[c]) for c in prof.numeric] if prof else []
    elif name == "pearsonr":
        refs = r.refs_in(call)
    else:
        return None
    skewed = [x for x in refs if x.info.kind == "numeric" and x.info.skew is not None and abs(x.info.skew) >= SKEW_THRESHOLD]
    if not skewed:
        return None
    cols = ", ".join(f"`{x.column}` (skew={x.info.skew})" for x in skewed[:4])
    fix = (
        "stats.spearmanr(x, y)" if name == "pearsonr" else _src(code, call).replace("corr()", "corr(method='spearman')")
    )
    if fix == _src(code, call):
        fix = "df.corr(method='spearman', numeric_only=True)"
    return Suggestion(
        rule="pearson_on_skewed",
        concept="correlation_method",
        title="Pearson correlation on skewed data",
        problem=f"Pearson's r is sensitive to skew and outliers, and these variables are skewed: {cols}.",
        alternative="Spearman's rank correlation (or check a scatter plot before trusting Pearson's r).",
        rationale="Spearman works on ranks, so a few extreme values cannot dominate the coefficient.",
        code=fix,
        columns=[x.column for x in skewed],
    )


def rule_categorical_on_numeric_chart(call: ast.Call, r: Resolver, code: str) -> Suggestion | None:
    name = call_name(call)
    func = dotted_name(call.func) or ""
    if name == "hist":
        ref = r.first_ref(call)
        if ref and ref.info.kind in ("categorical", "boolean", "text"):
            return Suggestion(
                rule="hist_categorical",
                concept="histogram_categorical",
                title=f"Histogram of a categorical column (`{ref.column}`)",
                problem=f"`{ref.column}` is categorical ({ref.info.n_unique} levels); histograms need a numeric scale.",
                alternative="A bar chart of counts per category.",
                rationale="Each category is a separate group, not a range of values that can be binned.",
                code=f"{_expr_text(ref)}.value_counts().plot.bar()",
                columns=[ref.column],
            )
    if name == "plot" and func.endswith("plt.plot") and call.args:
        ref = r.ref(call.args[0])
        if ref and ref.info.kind in ("categorical", "text"):
            y = _src(code, call.args[1]) if len(call.args) > 1 else "df['y']"
            return Suggestion(
                rule="line_categorical_x",
                concept="categorical_axis",
                title=f"Line plot over a categorical x-axis (`{ref.column}`)",
                problem="A line connects neighbouring points, implying an order and trend that categories do not have.",
                alternative="A bar chart of a summary per category, or a boxplot of the distribution per category.",
                rationale=f"`{ref.column}` is nominal ({ref.info.n_unique} levels), so neighbouring categories are not related.",
                code=(
                    f"{ref.frame or 'df'}.groupby('{ref.column}')[...].mean().plot.bar()\n"
                    f"# or: {ref.frame or 'df'}.boxplot(column=..., by='{ref.column}')"
                ),
                columns=[ref.column],
            )
    if name == "scatter":
        refs = [x for x in r.refs_in(call) if x.info.kind in ("categorical", "text")]
        if refs and len(r.refs_in(call)) >= 2:
            c = refs[0]
            return Suggestion(
                rule="scatter_categorical",
                concept="categorical_axis",
                title=f"Scatter plot with a categorical axis (`{c.column}`)",
                problem="Scatter plots assume both axes are numeric; points for a category pile up on one line.",
                alternative="A boxplot (or strip plot) of the numeric variable for each category.",
                rationale="Boxplots show the centre, spread and outliers of each group side by side.",
                code=f"{c.frame or 'df'}.boxplot(column=<numeric column>, by='{c.column}')",
                columns=[c.column],
            )
    return None


def rule_global_dropna(call: ast.Call, r: Resolver, code: str) -> Suggestion | None:
    if call_name(call) != "dropna" or not isinstance(call.func, ast.Attribute):
        return None
    if call.args or kwarg(call, "subset") is not None or kwarg(call, "axis") is not None:
        return None
    target = call.func.value
    if not isinstance(target, ast.Name) or not r.is_known_frame(target.id):
        return None
    prof = r.frames[target.id]
    lost = prof.n_rows - prof.n_complete_rows
    if prof.n_rows == 0 or lost / prof.n_rows < 0.05:
        return None
    worst = sorted(prof.columns.values(), key=lambda c: -c.n_missing)[:3]
    worst_s = ", ".join(f"`{c.name}` ({c.n_missing})" for c in worst if c.n_missing)
    return Suggestion(
        rule="global_dropna",
        concept="listwise_deletion",
        title=f"`{target.id}.dropna()` removes {lost} of {prof.n_rows} rows",
        problem=(
            f"Dropping every row with any missing value loses {100 * lost / prof.n_rows:.0f}% of the data; "
            f"most missing values come from {worst_s}."
        ),
        alternative="Drop only on the columns your analysis uses, and keep the original DataFrame.",
        rationale="Listwise deletion can shrink and bias the sample; restricting it keeps rows you still need.",
        code=f"clean = {target.id}.dropna(subset=[...])  # the columns used in this analysis\nprint(len({target.id}), '->', len(clean))",
        severity="warning",
    )


def rule_text_numbers(call: ast.Call, r: Resolver, code: str) -> Suggestion | None:
    if call_name(call) not in {"mean", "sum", "std", "median", "hist", "corr", "ttest_ind", "pearsonr"}:
        return None
    for ref in r.refs_in(call):
        if ref.info.kind in ("categorical", "text") and any("looks numeric" in n for n in ref.info.notes):
            e = _expr_text(ref)
            return Suggestion(
                rule="text_numbers",
                concept="text_numbers",
                title=f"`{ref.column}` is numbers stored as text",
                problem=f"`{ref.column}` has dtype {ref.info.dtype}; numeric operations will fail or misbehave.",
                alternative="Convert it explicitly with `pd.to_numeric`, after checking which values do not parse.",
                rationale="Seeing the bad values first stops you silently turning real data into NaN.",
                code=(
                    f"converted = pd.to_numeric({e}, errors='coerce')\n"
                    f"print({e}[converted.isna() & {e}.notna()].unique())  # values that failed\n"
                    f"{ref.frame or 'df'}['{ref.column}_num'] = converted"
                ),
                severity="warning",
                columns=[ref.column],
            )
    return None


RULES: list[Rule] = [
    rule_pie_many_categories,
    rule_mean_of_skewed,
    rule_ttest_equal_var,
    rule_nan_in_scipy,
    rule_pearson_on_skewed,
    rule_categorical_on_numeric_chart,
    rule_global_dropna,
    rule_text_numbers,
]


# ---------------------------------------------------------------------------
# Column-name checks (catch KeyErrors before running)


def unknown_columns(tree: ast.AST, r: Resolver, code: str) -> list[Suggestion]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        # df['x'] or a row-filtered frame such as df[df['g'] == 'a']['x']
        base = frame_base(node.value)
        if base is None:
            continue
        prof = r.frames.get(base.id)
        col = const_value(node.slice)
        if not prof or not isinstance(col, str) or col in prof.columns:
            continue
        # assignment creates a new column: df['new'] = ...
        if isinstance(node.ctx, ast.Store):
            continue
        match = difflib.get_close_matches(col, list(prof.columns), n=1, cutoff=0.6)
        lowered = {c.lower().strip(): c for c in prof.columns}
        match = [lowered[col.lower().strip()]] if col.lower().strip() in lowered else match
        hint = f"Did you mean `{match[0]}`?" if match else f"Available columns: {', '.join(list(prof.columns)[:12])}"
        out.append(
            Suggestion(
                rule="unknown_column",
                concept="column_names",
                title=f"`{base.id}` has no column `{col}`",
                problem=f"This will raise a KeyError. {hint}",
                alternative=f"Use the exact column name{f' `{match[0]}`' if match else ''}.",
                rationale="Column labels are exact strings - case, spaces and underscores matter.",
                code=f"{base.id}['{match[0]}']" if match else f"print(list({base.id}.columns))",
                line=getattr(node, "lineno", None),
                severity="warning",
                columns=[col],
            )
        )
    return out


# ---------------------------------------------------------------------------


class CodeAdvisor:
    def __init__(self, rules: list[Rule] | None = None):
        self.rules = rules if rules is not None else list(RULES)

    def check(
        self,
        code: str,
        frames: dict[str, DatasetProfile] | None = None,
        profiles: list[DatasetProfile] | None = None,
    ) -> list[Suggestion]:
        """Return suggestions for ``code``. Returns [] for code that does not parse."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        frames = frames or {}
        profiles = profiles if profiles is not None else list(frames.values())
        resolver = Resolver(frames, profiles)
        resolver.learn_aliases(tree)
        out: list[Suggestion] = []
        lines = code.splitlines()
        for call in _calls(tree):
            for rule in self.rules:
                try:
                    s = rule(call, resolver, code)
                except Exception:  # a broken rule must never break the user's workflow
                    s = None
                if s:
                    s.line = s.line or call.lineno
                    s.source_code = lines[call.lineno - 1].strip() if call.lineno <= len(lines) else ""
                    out.append(s)
        for s in unknown_columns(tree, resolver, code):
            s.source_code = lines[s.line - 1].strip() if s.line and s.line <= len(lines) else ""
            out.append(s)
        # one suggestion per (rule, line): nested calls can trigger a rule twice
        unique: dict[tuple[str, int | None], Suggestion] = {}
        for s in out:
            unique.setdefault((s.rule, s.line), s)
        return sorted(unique.values(), key=lambda s: (s.line or 0, s.severity != "warning"))
