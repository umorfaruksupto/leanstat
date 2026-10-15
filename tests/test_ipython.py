import pytest

IPython = pytest.importorskip("IPython")


@pytest.fixture
def shell(tmp_path, monkeypatch):
    from IPython.testing.globalipapp import start_ipython

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    ip = start_ipython()
    ip.run_line_magic("load_ext", "learnstat")
    yield ip
    ip.run_line_magic("unload_ext", "learnstat")


def test_extension_checks_recovers_and_handles_commands(shell, df, capsys):
    shell.user_ns["df"] = shell.learnstat_session.track(df, name="survey")
    shell.run_cell("df['country'].value_counts()\nx = df['income'].mean()")
    out = capsys.readouterr().out
    assert "Mean of a skewed variable" in out

    shell.run_cell("counter = 0\ncounter += 1\ny = df['Score'].mean()\nz = y + 1")
    out = capsys.readouterr().out
    assert "corrected column name" in out
    assert shell.user_ns["counter"] == 1          # not re-run
    assert shell.user_ns["z"] == shell.user_ns["y"] + 1  # later statements ran

    shell.run_cell("/describe 1")
    assert "Mean vs median" in capsys.readouterr().out
