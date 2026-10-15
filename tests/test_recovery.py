import pandas as pd

from learnstat.recovery import Executor, Fix, validate_fix


def run(session, code, **kw):
    return session.executor().run(code, **kw)


def test_misspelled_column_is_corrected(session):
    r = run(session, "x = df['Score'].mean()")
    assert r.ok
    assert session.ns["x"] == session.ns["df"]["score"].mean()
    assert "df['score']" in r.final_code
    assert r.attempts[0].diagnosis.category == "column"


def test_misspelled_method_is_corrected(session):
    r = run(session, "m = df.groupBy('group')['score'].median()")
    assert r.ok and "groupby" in r.final_code
    assert isinstance(session.ns["m"], pd.Series)


def test_missing_import_is_added(session):
    r = run(session, "z = np.sqrt(16)")
    assert r.ok and session.ns["z"] == 4.0
    assert r.final_code.startswith("import numpy as np")


def test_bad_keyword_is_corrected(session):
    session.ns["f"] = lambda data, color=None: color
    r = run(session, "c = f(1, colour='red')")
    assert r.ok and session.ns["c"] == "red"


def test_earlier_statements_are_not_rerun(session):
    session.ns["counter"] = 0
    r = run(session, "counter += 1\ny = df['Income'].median()")
    assert r.ok
    assert session.ns["counter"] == 1


def test_data_quality_errors_are_explained_not_fixed(session):
    r = run(session, "total = df['spend'].astype(float).sum()")
    assert not r.ok
    d = r.attempts[0].diagnosis
    assert d.category == "data_quality"
    assert r.attempts[0].fix is None
    assert "to_numeric" in d.next_step
    assert session.ns["df"]["spend"].dtype != float  # data untouched


def test_syntax_error_unclosed_bracket(session):
    r = run(session, "v = max(1, 2")
    assert r.ok and session.ns["v"] == 2
    assert r.attempts[0].diagnosis.category == "syntax"


def test_retries_are_bounded_and_unfixable_errors_stop(session):
    r = run(session, "a = 1\nb = undefined_thing_xyz + 1\nc = 3")
    assert not r.ok
    assert "c" not in session.ns  # stopped before later statements
    assert r.attempts[-1].diagnosis.next_step


def test_destructive_code_requires_permission(session, tmp_path):
    target = tmp_path / "keep.txt"
    target.write_text("data")
    r = run(session, f"import os\nos.remove({str(target)!r})")
    assert r.blocked and target.exists()

    session.confirm = lambda q: True
    r = run(session, f"import os\nos.remove({str(target)!r})")
    assert r.ok and not target.exists()


def test_fix_validation_rejects_method_and_data_changes():
    orig = "stats.ttest_ind(a, b)"
    assert validate_fix(orig, Fix("ai", "", "stats.mannwhitneyu(a, b)"))
    assert validate_fix("df['x'].mean()", Fix("ai", "", "df['x'].dropna().mean()"))
    assert validate_fix("df['x'].mean()", Fix("ai", "", "import os; os.remove('f'); df['x'].mean()"))
    assert not validate_fix("df['Scroe'].mean()", Fix("column", "", "df['score'].mean()", {"Scroe": "score"}))


def test_ai_fixes_need_confirmation(session):
    proposed = Fix("ai", "use the right variable", "w = 5")
    ex = Executor(session.ns, ai_fixer=lambda d, src: proposed, confirm=None)
    r = ex.run("w = qqqqqq_unknown")
    assert not r.ok and r.attempts[0].fix is proposed and not r.attempts[0].applied

    ex.confirm = lambda q: True
    r = ex.run("w = qqqqqq_unknown")
    assert r.ok and session.ns["w"] == 5


def test_column_list_selection_is_corrected(session):
    r = run(session, "sub = df[['age', 'Income']]")
    assert r.ok and "'income'" in r.final_code
    assert list(session.ns["sub"].columns) == ["age", "income"]


def test_near_tie_broken_by_single_slip():
    from learnstat.recovery import _close

    assert _close("medain", ["median", "mean", "min"]) == "median"  # adjacent swap
    assert _close("cout", ["count", "cut"]) is None  # both one slip away: do not guess


def test_matplotlib_style_keyword_is_corrected(session):
    import matplotlib

    matplotlib.use("Agg")
    r = run(session, "ax = df['age'].plot.hist(colour='grey')")
    assert r.ok and "color='grey'" in r.final_code
    assert r.attempts[0].diagnosis.category == "argument"


def test_invalid_option_value_is_corrected(session):
    r = run(session, "c = df[['age', 'score']].corr(method='pearsn')")
    assert r.ok and "method='pearson'" in r.final_code


def test_bracket_closed_where_the_code_makes_sense(session):
    r = run(session, "c = df[['age', 'score'].corr()")
    assert r.ok
    assert "df[['age', 'score']].corr()" in r.final_code


def test_unclosed_call_closed_at_end_of_line(session):
    r = run(session, "m = df['age'].mean(")
    assert r.ok and r.final_code.strip() == "m = df['age'].mean()"


def test_silent_nan_from_small_sample_is_explained(session):
    import warnings

    session.ns["stats"] = __import__("scipy.stats", fromlist=["stats"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = run(session, "res = stats.shapiro([1, 2])")
    assert r.ok and not r.changed
    assert r.notes and "too small" in r.notes[0]
    assert "second look" in r.report()


def test_clean_code_has_no_notes(session):
    r = run(session, "m = df['age'].mean()")
    assert r.ok and r.notes == []


def test_nan_test_result_is_explained_not_fixed(session):
    from scipy import stats

    session.ns["stats"] = stats
    r = run(session, "res = stats.ttest_ind(df['score'], df['age'])")
    assert r.ok and not r.changed
    if pd.isna(session.ns["res"].pvalue):
        assert any("returned NaN" in n for n in r.notes)


def test_slip_one_keystroke_from_two_names_is_refused():
    from learnstat.recovery import _close

    assert _close("nuique", ["unique", "nunique", "mean"]) is None  # swap of unique, deletion from nunique
    assert _close("stdd", ["std", "sum", "mean"]) == "std"


def test_replacement_method_must_accept_the_call(session):
    r = run(session, "x = df['income'].modw()")  # mod() needs an argument; mode() fits
    assert r.ok and "df['income'].mode()" in r.final_code
    r = run(session, "x = df['income'].aes()")  # axes is not callable, abs is too dissimilar
    assert not r.changed


def test_ambiguous_method_slip_is_left_for_the_analyst(session):
    r = run(session, "x = df['age'].nuique()")
    assert not r.ok and not r.changed


def test_nan_result_inside_print_is_explained(session):
    from scipy import stats

    session.ns["stats"] = stats
    r = run(session, "print(stats.ttest_ind(df['score'], df['income']))")
    assert r.ok and not r.changed
    assert any("returned NaN" in n for n in r.notes)
