import pytest

from learnstat import config
from learnstat.guide import run_guide
from learnstat.session import Session
from learnstat.setup import IO


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNSTAT_CONFIG", str(tmp_path / "config.json"))
    for var in config.ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)


class Script:
    """Feeds scripted answers to the guide and records everything it prints."""

    def __init__(self, answers):
        self.answers, self.out = list(answers), []

    def io(self):
        def ask(prompt):
            self.out.append(prompt)
            if not self.answers:
                raise EOFError
            return self.answers.pop(0)

        return IO(ask=ask, secret=ask, say=self.out.append)

    @property
    def text(self):
        return "\n".join(self.out)


def _run(tmp_path, answers, files=(), **kw):
    s = Session(memory_dir=tmp_path / "mem", ai=False, out=lambda *_: None, confirm=lambda q: True)
    script = Script(answers)
    queue = list(files)
    pick = lambda title, types: queue.pop(0) if queue else None
    return s, script, run_guide(s, script.io(), pick=pick, **kw)


def _csv(tmp_path):
    path = tmp_path / "survey.csv"
    path.write_text("age,income\n20,100\n30,200\n40,\n", encoding="utf-8")
    return path


def test_loads_data_with_file_window_then_opens_python_prompt(tmp_path):
    called = []
    s, script, rc = _run(tmp_path, [""], files=[str(_csv(tmp_path))], repl=lambda: called.append(1))
    assert rc == 0 and "df" in s.ns and list(s.datasets) == ["survey"]
    assert called == [1]  # straight to the Python prompt once the data is loaded


def test_menu_summary_without_data_and_quit(tmp_path):
    s, script, _ = _run(tmp_path, ["s", "1", "q"])
    assert "No data loaded" in script.text and "1 summary" in script.text


def test_missing_file_is_reported(tmp_path):
    s, script, _ = _run(tmp_path, [str(tmp_path / "nope.csv"), "q"])
    assert "Not found" in script.text and "df" not in s.ns


def test_check_script_and_explain_suggestion(tmp_path):
    csv = tmp_path / "survey.csv"
    csv.write_text("country,income\n" + "\n".join(f"c{i % 12},{i * 10}" for i in range(60)) + "\n", encoding="utf-8")
    py = tmp_path / "analysis.py"
    py.write_text(
        f"import pandas as pd\ndf = pd.read_csv(r'{csv}')\ndf['country'].value_counts().plot.pie()\n",
        encoding="utf-8",
    )
    s, script, _ = _run(tmp_path, ["s", "3", str(py), "1", "", "q"])
    assert "Pie chart" in script.text and "In your code:" in script.text


def test_run_script_with_automatic_help(tmp_path):
    py = tmp_path / "ok.py"
    py.write_text("total = 1 + 1\n", encoding="utf-8")
    s, script, _ = _run(tmp_path, ["s", "4", str(py), "q"])
    assert "Done." in script.text and s.ns["total"] == 2


def test_ask_ai_without_setup_points_to_menu_8(tmp_path):
    s, script, _ = _run(tmp_path, ["s", "2", "q"])
    assert "choose 8" in script.text


def test_menu_5_reopens_python_prompt(tmp_path):
    called = []
    s, script, _ = _run(tmp_path, ["s", "5", "q"], repl=lambda: called.append(1))
    assert called == [1, 1]  # once at start, once from the menu


def test_end_of_input_leaves_cleanly(tmp_path):
    s, script, rc = _run(tmp_path, [])
    assert rc == 0


def test_excel_files_load(tmp_path):
    pytest.importorskip("openpyxl")
    import pandas as pd

    xlsx = tmp_path / "data.xlsx"
    pd.DataFrame({"a": [1, 2, 3]}).to_excel(xlsx, index=False)
    s, script, _ = _run(tmp_path, ["", "q"], files=[str(xlsx)])
    assert list(s.datasets) == ["data"] and len(s.ns["df"]) == 3
