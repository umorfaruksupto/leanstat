def test_describe_explains_latest_suggestion(session):
    session.check("df['country'].value_counts().plot.pie()", show=False)
    text = session.command("/describe")
    assert "12 categories" in text
    assert "angles" in text  # the statistical concept
    assert "When it is appropriate" in text


def test_alternatives_and_learn(session):
    session.check("df['income'].mean()", show=False)
    assert "trim_mean" in session.command("/alternatives")
    assert "Exercise" in session.command("/learn")
    assert "Spearman" in session.command("/learn correlation")


def test_accept_and_reject_record_decisions(session):
    s1, = [s for s in session.check("df['income'].mean()", show=False)]
    out = session.command(f"/accept {s1.id}")
    assert "median" in out and s1.status == "accepted"
    session.check("df['country'].value_counts().plot.pie()", show=False)
    session.command("/reject only 3 countries matter for this report")
    md = session.command("/memory show")
    assert "Accepted suggestion" in md and "Kept original approach" in md and "only 3 countries" in md


def test_memory_note_and_summary(session):
    assert "Note saved" in session.command("/memory note income is in BDT")
    assert "income is in BDT" in session.command("/memory show")
    assert "200 rows" in session.command("/memory")


def test_describe_last_error(session):
    session.run("df['spend'].astype(float)", check=False)
    text = session.command("/describe")
    assert "data-quality" in text and "Next step" in text


def test_unknown_command(session):
    assert "Unknown command" in session.command("/nope")
