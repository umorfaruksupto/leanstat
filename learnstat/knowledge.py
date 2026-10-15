"""Offline explanations used by /describe, /alternatives and /learn.

Each concept explains the statistical idea, the purpose of the suggested
function, when it is appropriate and when something else would be better.
When the Claude backend is available these are used as grounding and then
tailored to the user's dataset; without AI they are shown as-is, filled in with
the dataset facts recorded on the suggestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Concept:
    title: str
    concept: str
    purpose: str
    appropriate_when: str
    alternatives: list[str] = field(default_factory=list)
    lesson: str = ""
    docs: list[str] = field(default_factory=list)


CONCEPTS: dict[str, Concept] = {
    "part_to_whole_charts": Concept(
        title="Pie charts vs bar charts",
        concept=(
            "People judge angles and areas much less accurately than lengths along a common axis. "
            "A pie chart asks the reader to compare slice angles; with more than about five slices, or "
            "slices of similar size, the comparison becomes guesswork."
        ),
        purpose=(
            "`Series.value_counts().plot.barh()` draws one bar per category on a shared baseline, so "
            "differences in length are easy to compare and long labels stay readable."
        ),
        appropriate_when=(
            "Use a bar chart whenever the goal is to compare categories. A pie chart is reasonable only "
            "when there are 2-5 parts of a single whole and the message is 'one part dominates'."
        ),
        alternatives=[
            "Horizontal bar chart sorted by count: `s.value_counts().plot.barh()`",
            "Group rare levels into 'Other' first: `s.where(s.isin(s.value_counts().index[:8]), 'Other')`",
            "A dot plot (`plt.plot(counts.values, counts.index, 'o')`) for a lighter look",
            "A stacked 100% bar when comparing proportions across several groups",
        ],
        lesson=(
            "Exercise: plot the same `value_counts()` as a pie and as a sorted horizontal bar. Try to "
            "rank the 3rd-6th categories from each chart - notice which one you can do without reading labels."
        ),
        docs=["https://pandas.pydata.org/docs/reference/api/pandas.Series.plot.barh.html"],
    ),
    "center_of_skewed_data": Concept(
        title="Mean vs median for skewed data",
        concept=(
            "The mean is pulled toward long tails and extreme values; the median (the 50th percentile) "
            "is not. In a right-skewed variable such as income, the mean can sit far above what a "
            "typical observation looks like."
        ),
        purpose=(
            "`Series.median()` and the IQR (`quantile(0.75) - quantile(0.25)`) summarise the centre and "
            "spread robustly. `describe()` shows both mean and quartiles so you can compare them."
        ),
        appropriate_when=(
            "Report the median/IQR when |skew| is about 1 or more, or when there are outliers. The mean is "
            "still the right choice when you need totals (mean x n), or for roughly symmetric data."
        ),
        alternatives=[
            "`df[col].describe()` to see mean and quartiles together",
            "A trimmed mean: `scipy.stats.trim_mean(x, 0.1)`",
            "Analyse `np.log1p(x)` if the variable is positive and multiplicative in nature",
        ],
        lesson="Exercise: compute mean and median before and after adding one huge value to the column.",
        docs=["https://pandas.pydata.org/docs/reference/api/pandas.Series.median.html"],
    ),
    "welch_t_test": Concept(
        title="Student's vs Welch's t-test",
        concept=(
            "`scipy.stats.ttest_ind` assumes equal variances by default (Student's t-test). When group "
            "variances or sizes differ, that assumption inflates the false-positive rate. Welch's test "
            "drops the assumption and loses almost nothing when variances are in fact equal."
        ),
        purpose="`ttest_ind(a, b, equal_var=False)` runs Welch's t-test on the same data.",
        appropriate_when=(
            "Welch's is a sensible default for comparing two independent means. For strongly skewed data "
            "or small samples, a rank-based test (Mann-Whitney U) may be preferable; for paired measurements "
            "use `ttest_rel`."
        ),
        alternatives=[
            "`scipy.stats.mannwhitneyu(a, b)` - compares distributions without assuming normality",
            "`scipy.stats.ttest_rel(a, b)` - for paired/before-after data",
            "`scipy.stats.f_oneway(*groups)` / `kruskal(*groups)` - for 3+ groups",
        ],
        lesson="Exercise: simulate two normal groups with SDs 1 and 4 and sizes 10 and 100; compare p-values.",
        docs=["https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html"],
    ),
    "correlation_method": Concept(
        title="Pearson vs Spearman correlation",
        concept=(
            "Pearson's r measures linear association and is sensitive to outliers and skew. Spearman's rho "
            "is Pearson on ranks: it measures monotonic association and is robust to extreme values."
        ),
        purpose="`df.corr(method='spearman')` or `scipy.stats.spearmanr(x, y)`.",
        appropriate_when=(
            "Use Spearman when variables are skewed, ordinal, or contain outliers, or when the relationship "
            "is monotonic but curved. Use Pearson when both variables are roughly symmetric and the "
            "relationship looks linear in a scatter plot."
        ),
        alternatives=[
            "Look first: `df.plot.scatter(x=..., y=...)`",
            "Kendall's tau (`method='kendall'`) for small samples with many ties",
            "Pearson on log-transformed variables",
        ],
        lesson="Exercise: compute both coefficients for x and x**3 - why does only one equal 1?",
        docs=["https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html"],
    ),
    "missing_values_in_tests": Concept(
        title="Missing values in statistical functions",
        concept=(
            "pandas methods skip NaN by default, but many SciPy functions do not: they return `nan` or "
            "raise an error. How missing values are handled is an analytical decision that changes your "
            "sample, so it should be explicit and reported."
        ),
        purpose="`.dropna()` on the series passed to the test, or `nan_policy='omit'` where SciPy supports it.",
        appropriate_when=(
            "Dropping is fine when values are missing at random and few in number. If many are missing or "
            "missingness relates to the outcome, investigate first and consider imputation."
        ),
        alternatives=[
            "`nan_policy='omit'` (supported by many scipy.stats functions)",
            "Report n per group after dropping: `a.dropna().size`",
            "Imputation (e.g. median) - only with a documented justification",
        ],
        lesson="Exercise: compare `df[col].isna().groupby(df[group]).mean()` across groups - is missingness balanced?",
        docs=["https://docs.scipy.org/doc/scipy/dev/api-dev/nan_policy.html"],
    ),
    "categorical_axis": Concept(
        title="Plotting categorical variables",
        concept=(
            "Line plots imply order and continuity between x values; scatter plots and histograms assume "
            "numeric scales. A nominal category (e.g. country) has neither, so these charts can mislead."
        ),
        purpose="Bar charts for counts, boxplots/strip plots for a numeric variable across categories.",
        appropriate_when=(
            "Use a line only when x is ordered (time, dose levels). For numeric-by-category comparisons "
            "use `df.boxplot(column=num, by=cat)`."
        ),
        alternatives=[
            "`df.boxplot(column='y', by='group')`",
            "`df.groupby('group')['y'].mean().plot.bar()` (show error bars too)",
            "`pd.crosstab(df.a, df.b).plot.bar(stacked=True)` for two categoricals",
        ],
        lesson="Exercise: compare a line plot and a bar chart of group means - what does the line falsely suggest?",
        docs=["https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.boxplot.html"],
    ),
    "listwise_deletion": Concept(
        title="Dropping rows with missing values",
        concept=(
            "`df.dropna()` removes every row with any missing value (listwise deletion). If missingness is "
            "concentrated in one column you may lose far more data than you need to, and the remaining rows "
            "may no longer represent the population."
        ),
        purpose="`df.dropna(subset=[...])` drops rows only where the columns you analyse are missing.",
        appropriate_when=(
            "Drop only on the columns used by a specific analysis, and record how many rows were removed."
        ),
        alternatives=[
            "`df.dropna(subset=['col_a', 'col_b'])`",
            "`df.isna().sum()` first, to see where values are missing",
            "Keep the original: `clean = df.dropna(subset=...)` rather than overwriting `df`",
        ],
        lesson="Exercise: compare `len(df.dropna())` with `len(df.dropna(subset=[col]))` for each column.",
        docs=["https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html"],
    ),
    "column_names": Concept(
        title="Selecting columns by name",
        concept="Column labels are exact strings: case, spaces and underscores all matter.",
        purpose="`df.columns` lists the exact names; `df[['a', 'b']]` selects several at once.",
        appropriate_when="Prefer `df['name']` over `df.name` - it works for every label, including ones with spaces.",
        alternatives=["`df.columns = df.columns.str.strip().str.lower()` to normalise names once, explicitly"],
        lesson="Exercise: print `list(df.columns)` and compare it with the names you typed.",
        docs=["https://pandas.pydata.org/docs/user_guide/indexing.html"],
    ),
    "text_numbers": Concept(
        title="Numbers stored as text",
        concept=(
            "If a numeric column contains stray characters ('1,200', '$5', 'n/a'), pandas stores it as text "
            "and arithmetic or statistics fail or behave unexpectedly."
        ),
        purpose="`pd.to_numeric(s, errors='coerce')` converts what it can and turns the rest into NaN.",
        appropriate_when=(
            "Always inspect which values failed first: `s[pd.to_numeric(s, errors='coerce').isna() & s.notna()]`. "
            "Converting is a data-cleaning decision, so record it."
        ),
        alternatives=["`s.str.replace(',', '')` before converting", "`pd.read_csv(..., thousands=',')`"],
        lesson="Exercise: find the rows that fail to convert and decide, case by case, how to treat them.",
        docs=["https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html"],
    ),
    "histogram_categorical": Concept(
        title="Histograms are for numeric variables",
        concept=(
            "A histogram bins a continuous scale. Category labels have no scale, so you need counts per "
            "level instead."
        ),
        purpose="`s.value_counts().plot.bar()` shows the count for each level.",
        appropriate_when="Use histograms for numeric columns, bar charts of counts for categorical ones.",
        alternatives=["`s.value_counts(normalize=True)` for proportions"],
        lesson="Exercise: draw the bar chart and sort it by count vs by label - which is easier to read?",
        docs=["https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html"],
    ),
}


def describe_concept(key: str) -> Concept | None:
    return CONCEPTS.get(key)


def find_topic(query: str) -> Concept | None:
    q = query.lower().strip()
    if not q:
        return None
    for key, c in CONCEPTS.items():
        if q in key or q in c.title.lower():
            return c
    words = set(q.split())
    best, best_score = None, 0
    for c in CONCEPTS.values():
        text = (c.title + " " + c.concept + " " + c.purpose).lower()
        score = sum(1 for w in words if w in text)
        if score > best_score:
            best, best_score = c, score
    return best
