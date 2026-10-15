from learnstat import profile_dataframe
from learnstat.memory import MemoryStore


def test_profile_kinds_and_issues(df):
    p = profile_dataframe(df, "survey")
    assert p.columns["income"].kind == "numeric"
    assert p.columns["income"].skew > 1
    assert p.columns["country"].kind == "categorical"
    assert p.columns["score"].n_missing == 30
    assert any("Inconsistent labels" in n for n in p.columns["gender"].notes)
    assert any("looks numeric" in n for n in p.columns["spend"].notes)
    assert any("dropna" in i for i in p.issues)
    assert p.suggested_analyses


def test_memory_reuses_profile_until_data_changes(tmp_path, df):
    store = MemoryStore(tmp_path)
    first = store.load_or_profile(df, "survey", source="survey.csv")
    assert not first.reused
    assert store.md_path("survey").exists()

    store.record_decision(first, "Use median income")
    second = store.load_or_profile(df, "survey")
    assert second.reused
    assert second.decisions[0]["text"] == "Use median income"

    df.loc[0, "age"] = 99
    third = store.load_or_profile(df, "survey")
    assert not third.reused
    assert third.decisions[0]["text"] == "Use median income"  # decisions survive re-profiling
    md = store.md_path("survey").read_text(encoding="utf-8")
    assert "Use median income" in md
    assert "## Data-quality issues" in md
