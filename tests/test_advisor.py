def rules(session, code):
    return {s.rule for s in session.check(code, show=False)}


def test_pie_with_many_categories_suggests_bar(session):
    found = session.check("df['country'].value_counts().plot.pie()", show=False)
    s = next(s for s in found if s.rule == "pie_many_categories")
    assert "12 categories" in s.title
    assert "barh" in s.code
    assert s.problem and s.alternative and s.rationale


def test_pie_with_few_categories_is_fine(session):
    assert "pie_many_categories" not in rules(session, "df['group'].value_counts().plot(kind='pie')")


def test_mean_of_skewed_column(session):
    assert "mean_of_skewed" in rules(session, "df['income'].mean()")
    assert "mean_of_skewed" not in rules(session, "df['score'].mean()")


def test_ttest_and_missing_values(session):
    code = "from scipy import stats\nstats.ttest_ind(df[df.group=='a']['score'], df[df.group=='b']['score'])"
    found = rules(session, code)
    assert {"ttest_equal_var", "nan_in_scipy"} <= found


def test_unknown_column_is_caught_before_running(session):
    found = session.check("df['Score'].mean()", show=False)
    s = next(s for s in found if s.rule == "unknown_column")
    assert "score" in s.problem


def test_global_dropna_and_text_numbers(session):
    assert "global_dropna" in rules(session, "clean = df.dropna()")
    assert "text_numbers" in rules(session, "df['spend'].mean()")


def test_same_advice_is_not_repeated(session):
    assert session.check("df['income'].mean()", show=False)
    assert not session.check("df['income'].mean()", show=False)


def test_unparseable_code_gives_no_suggestions(session):
    assert session.check("df['income'].mean(", show=False) == []


def test_column_aliases_are_followed(session):
    code = (
        "from scipy import stats\n"
        "a = df[df['group'] == 'a']['score']\n"
        "b = df[df['group'] == 'b']['score']\n"
        "stats.ttest_ind(a, b, equal_var=False)"
    )
    assert "nan_in_scipy" in rules(session, code)
