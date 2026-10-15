"""Rule-based dataset profiling.

Everything here works without AI. The profile is the analytical context that
the advisor, the recovery engine and the optional Claude backend all share.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

# Column-name hints used to infer what a column probably means. These are
# guesses and are always labelled as such in the Markdown memory.
_MEANING_HINTS: list[tuple[str, str]] = [
    (r"(^|_)id$|^id_|uuid|identifier", "identifier (row key, not a measurement)"),
    (r"date|time|timestamp|_at$|year|month|day", "date / time"),
    (r"age", "age"),
    (r"price|cost|amount|revenue|sales|income|salary|fare|spend", "monetary amount"),
    (r"gender|sex", "demographic category"),
    (r"country|city|state|region|zip|postcode|location", "geographic location"),
    (r"score|rating|grade", "score / rating"),
    (r"count|qty|quantity|num_|number", "count"),
    (r"weight|height|length|width|size|bmi", "physical measurement"),
    (r"lat(itude)?$|lon(gitude)?$|lng$", "geographic coordinate"),
    (r"pct|percent|rate|ratio", "rate / proportion"),
    (r"^is_|^has_|flag|survived|churn|target|label|outcome", "binary outcome / flag"),
    (r"name|title|description|comment|text|notes", "free text / label"),
    (r"group|category|type|class|segment|treatment", "grouping variable"),
]


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    kind: str  # numeric | categorical | boolean | datetime | text | identifier
    n_missing: int
    pct_missing: float
    n_unique: int
    summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    inferred_meaning: str | None = None

    @property
    def skew(self) -> float | None:
        return self.summary.get("skew")


@dataclass
class DatasetProfile:
    name: str
    source: str | None
    n_rows: int
    n_cols: int
    fingerprint: str
    columns: dict[str, ColumnProfile]
    correlations: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    suggested_analyses: list[str] = field(default_factory=list)
    n_duplicate_rows: int = 0
    n_complete_rows: int = 0
    created_at: str = ""

    # -- convenience accessors -------------------------------------------
    def columns_of_kind(self, *kinds: str) -> list[str]:
        return [c.name for c in self.columns.values() if c.kind in kinds]

    @property
    def numeric(self) -> list[str]:
        return self.columns_of_kind("numeric")

    @property
    def categorical(self) -> list[str]:
        return self.columns_of_kind("categorical", "boolean")

    def get(self, column: str) -> ColumnProfile | None:
        return self.columns.get(column)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["columns"] = {k: asdict(v) for k, v in self.columns.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetProfile":
        d = dict(d)
        d["columns"] = {k: ColumnProfile(**v) for k, v in d["columns"].items()}
        return cls(**d)


# ---------------------------------------------------------------------------


def fingerprint(df: pd.DataFrame) -> str:
    """A cheap content hash used to decide whether the profile is stale."""
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update("|".join(f"{c}:{t}" for c, t in df.dtypes.astype(str).items()).encode())
    try:
        h.update(str(int(pd.util.hash_pandas_object(df, index=True).sum())).encode())
    except (TypeError, ValueError):  # unhashable cells, e.g. lists
        h.update(df.head(1000).to_csv().encode())
    return h.hexdigest()[:16]


def _infer_meaning(name: str) -> str | None:
    lowered = name.lower()
    for pattern, meaning in _MEANING_HINTS:
        if re.search(pattern, lowered):
            return meaning
    return None


def _to_float(x: Any) -> float | None:
    try:
        if x is None or pd.isna(x):
            return None
        return round(float(x), 4)
    except (TypeError, ValueError):
        return None


def _looks_numeric_text(s: pd.Series) -> float:
    sample = s.dropna().astype(str).head(500)
    if sample.empty:
        return 0.0
    cleaned = sample.str.replace(r"[,\s$%€£]", "", regex=True)
    return float(pd.to_numeric(cleaned, errors="coerce").notna().mean())


def _looks_datetime_text(s: pd.Series) -> float:
    sample = s.dropna().astype(str).head(200)
    if sample.empty or not sample.str.contains(r"\d").all():
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            return 0.0
    return float(parsed.notna().mean())


def _profile_column(s: pd.Series, n_rows: int) -> ColumnProfile:
    name = str(s.name)
    n_missing = int(s.isna().sum())
    non_null = s.dropna()
    try:
        n_unique = int(non_null.nunique())
    except TypeError:
        n_unique = int(non_null.astype(str).nunique())
    col = ColumnProfile(
        name=name,
        dtype=str(s.dtype),
        kind="text",
        n_missing=n_missing,
        pct_missing=round(100 * n_missing / n_rows, 2) if n_rows else 0.0,
        n_unique=n_unique,
        inferred_meaning=_infer_meaning(name),
    )

    if pd.api.types.is_bool_dtype(s):
        col.kind = "boolean"
        col.summary = {"true_share": _to_float(non_null.mean())}
    elif pd.api.types.is_datetime64_any_dtype(s):
        col.kind = "datetime"
        col.summary = {"min": str(non_null.min()), "max": str(non_null.max())}
    elif pd.api.types.is_numeric_dtype(s):
        col.kind = "numeric"
        x = non_null.astype(float)
        q1, q3 = (x.quantile(0.25), x.quantile(0.75)) if len(x) else (np.nan, np.nan)
        iqr = q3 - q1
        n_outliers = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()) if len(x) else 0
        col.summary = {
            "mean": _to_float(x.mean()),
            "std": _to_float(x.std()),
            "min": _to_float(x.min()),
            "median": _to_float(x.median()),
            "max": _to_float(x.max()),
            "skew": _to_float(x.skew()) if len(x) > 2 else None,
            "n_outliers_iqr": n_outliers,
        }
        name_l = name.lower()
        if n_unique == n_rows and n_rows > 20 and re.search(r"(^|_)id$|^id_|index", name_l):
            col.kind = "identifier"
            col.notes.append("Unique integer per row - treat as an identifier, not a measurement.")
        elif n_unique == 2:
            col.notes.append("Only two distinct values - probably a binary flag coded as numbers.")
        elif 2 < n_unique <= 10 and pd.api.types.is_integer_dtype(s):
            col.notes.append(
                f"Integer with only {n_unique} distinct values - may be an ordinal or coded category."
            )
        skew = col.summary["skew"]
        if skew is not None and abs(skew) >= 1:
            direction = "right" if skew > 0 else "left"
            col.notes.append(
                f"Strongly {direction}-skewed (skew={skew}); the median and IQR describe the centre "
                "and spread better than the mean and SD."
            )
        if n_outliers:
            col.notes.append(f"{n_outliers} values outside 1.5xIQR (potential outliers).")
    else:
        # object / string / categorical dtype
        if isinstance(s.dtype, pd.CategoricalDtype):
            col.kind = "categorical"
        elif n_unique == n_rows and n_rows > 20:
            col.kind = "identifier" if _infer_meaning(name) and "identifier" in _infer_meaning(name) else "text"
        elif n_unique <= max(20, int(0.05 * n_rows)) or n_unique / max(len(non_null), 1) <= 0.5:
            col.kind = "categorical"
        else:
            col.kind = "text"

        top = non_null.astype(str).value_counts().head(5)
        col.summary = {"top_values": {str(k): int(v) for k, v in top.items()}}

        if _looks_numeric_text(s) > 0.95 and n_unique > 10:
            col.notes.append(
                "Stored as text but looks numeric - convert with pd.to_numeric(..., errors='coerce') "
                "after checking which values fail to parse."
            )
        elif _looks_datetime_text(s) > 0.95:
            col.notes.append("Stored as text but looks like dates - consider pd.to_datetime().")

        if col.kind == "categorical":
            as_str = non_null.astype(str)
            normalised = as_str.str.strip().str.lower()
            if normalised.nunique() < as_str.nunique():
                col.notes.append(
                    "Inconsistent labels: some categories differ only by case or whitespace "
                    "(e.g. 'Male' vs 'male ')."
                )
            if n_unique > 15:
                col.notes.append(f"High cardinality ({n_unique} categories) - pie charts will be unreadable.")
            counts = as_str.value_counts()
            rare = int((counts < 5).sum())
            if rare and n_unique > 2:
                col.notes.append(f"{rare} categories have fewer than 5 rows (sparse for chi-square tests).")

    if n_unique <= 1 and n_rows > 1:
        col.notes.append("Constant column - carries no information.")
    if col.pct_missing >= 50:
        col.notes.append(f"{col.pct_missing}% missing - analyses using this column will lose most rows.")
    elif n_missing:
        col.notes.append(f"{n_missing} missing values ({col.pct_missing}%).")
    return col


def _suggest_analyses(p: DatasetProfile) -> list[str]:
    out: list[str] = []
    num = p.numeric
    cat = [c for c in p.categorical if p.columns[c].n_unique <= 30]
    dt = p.columns_of_kind("datetime")

    for c in num[:6]:
        col = p.columns[c]
        if col.skew is not None and abs(col.skew) >= 1:
            out.append(f"`{c}`: histogram + boxplot; report median/IQR (skewed). Consider np.log1p for modelling.")
        else:
            out.append(f"`{c}`: histogram (`df['{c}'].plot.hist()`) and describe() for centre/spread.")
    for corr in p.correlations[:4]:
        out.append(
            f"`{corr['a']}` vs `{corr['b']}`: scatter plot (r={corr['r']}); check linearity before Pearson, "
            "or use Spearman if skewed."
        )
    for c in cat[:3]:
        k = p.columns[c].n_unique
        chart = "bar chart" if k > 5 else "bar chart (a pie only works for a few parts of one whole)"
        out.append(f"`{c}`: `value_counts()` + {chart}.")
        for n in num[:2]:
            if k == 2:
                out.append(f"`{n}` by `{c}`: grouped boxplot; Welch's t-test (or Mann-Whitney U if skewed).")
            elif 2 < k <= 10:
                out.append(f"`{n}` by `{c}`: grouped boxplot; one-way ANOVA (or Kruskal-Wallis if skewed).")
    if len(cat) >= 2:
        out.append(f"`{cat[0]}` x `{cat[1]}`: `pd.crosstab` + chi-square test of independence.")
    for d in dt[:1]:
        for n in num[:2]:
            out.append(f"`{n}` over `{d}`: line plot after sorting by date / resampling.")
    return out


def profile_dataframe(df: pd.DataFrame, name: str, source: str | None = None) -> DatasetProfile:
    n_rows, n_cols = df.shape
    columns = {str(c): _profile_column(df[c], n_rows) for c in df.columns}

    correlations: list[dict[str, Any]] = []
    num_cols = [c for c, p in columns.items() if p.kind == "numeric" and p.n_unique > 2]
    if 2 <= len(num_cols) <= 60:
        corr = df[num_cols].corr(numeric_only=True)
        seen = set()
        for a in num_cols:
            for b in num_cols:
                if a >= b or (a, b) in seen:
                    continue
                seen.add((a, b))
                r = corr.loc[a, b]
                if pd.notna(r) and abs(r) >= 0.5:
                    correlations.append({"a": a, "b": b, "r": round(float(r), 3)})
        correlations.sort(key=lambda d: -abs(d["r"]))

    try:
        n_dup = int(df.duplicated().sum())
    except TypeError:
        n_dup = 0
    n_complete = int(df.dropna().shape[0])

    profile = DatasetProfile(
        name=name,
        source=source,
        n_rows=n_rows,
        n_cols=n_cols,
        fingerprint=fingerprint(df),
        columns=columns,
        correlations=correlations,
        n_duplicate_rows=n_dup,
        n_complete_rows=n_complete,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    issues: list[str] = []
    if n_dup:
        issues.append(f"{n_dup} fully duplicated rows.")
    if n_complete < n_rows:
        dropped = n_rows - n_complete
        issues.append(
            f"`df.dropna()` would drop {dropped} of {n_rows} rows ({100 * dropped / max(n_rows, 1):.1f}%)."
        )
    for c in columns.values():
        for note in c.notes:
            if any(w in note for w in ("missing", "Inconsistent", "Stored as text", "Constant", "outliers")):
                issues.append(f"`{c.name}`: {note}")
    for corr in correlations:
        if abs(corr["r"]) >= 0.95:
            issues.append(f"`{corr['a']}` and `{corr['b']}` are nearly collinear (r={corr['r']}).")
    profile.issues = issues
    profile.suggested_analyses = _suggest_analyses(profile)
    return profile
