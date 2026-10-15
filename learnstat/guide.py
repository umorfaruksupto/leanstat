"""Minimal guided start: pick a data file, then write pandas / NumPy code.

The Python prompt is the main place to work; ``menu`` opens a one-line menu for the
extras (AI questions, script checks, lessons, AI settings).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Callable

from .session import Session
from .setup import Cancelled, IO, run_wizard

DATA_TYPES = [("Data files", "*.csv *.tsv *.txt *.xlsx *.xls *.json *.parquet"), ("All files", "*.*")]
SCRIPT_TYPES = [("Python files", "*.py"), ("All files", "*.*")]

MENU = "1 summary | 2 ask AI | 3 check script | 4 run script | 5 code | 6 learn | 7 load data | 8 AI | ? help | q quit"

HELP_TEXT = """1 summary     columns, types, missing values, problems
2 ask AI      ask anything about your data
3 check       suggestions for a .py file (not run)
4 run         run a .py file; simple mistakes are explained and fixed
5 code        pandas / NumPy prompt with df ready ("menu" returns here)
6 learn       short lesson on a topic
7 load data   CSV or Excel
8 AI          change company, key or model"""


def pick_file(title: str, types: list[tuple[str, str]]) -> str | None:
    """Open the system file window. Returns None when cancelled or when no window is available."""
    try:
        import tkinter
        from tkinter import filedialog

        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            path = filedialog.askopenfilename(title=title, filetypes=types)
        finally:
            root.destroy()
        return path or None
    except Exception:
        return None


def _clean(path: str) -> str:
    return path.strip().strip("\"'")


def ask_file(io: IO, pick: Callable, what: str, types: list[tuple[str, str]], skip: str = "b") -> Path | None:
    answer = io.line(f"{what} (Enter = browse, {skip} = {'skip' if skip == 's' else 'back'}): ")
    if answer.lower() in ("b", "back", "s", "skip"):
        return None
    if not answer:
        answer = pick(f"Choose {what}", types) or ""
        if not answer:
            io.say("No file chosen. Paste the path instead.")
            return None
    path = Path(_clean(answer)).expanduser()
    if not path.is_file():
        io.say(f"Not found: {path}")
        return None
    return path


def load_dataset(s: Session, io: IO, path: Path):
    suffix = path.suffix.lower()
    try:
        with warnings.catch_warnings():  # library chatter (e.g. openpyxl) is not for students
            warnings.simplefilter("ignore")
            if suffix in (".xlsx", ".xls"):
                df = s.read_excel(path)
            elif suffix == ".json":
                df = s.read_json(path)
            elif suffix == ".parquet":
                df = s.read_parquet(path)
            elif suffix == ".tsv":
                df = s.read_csv(path, sep="\t")
            else:
                df = s.read_csv(path)
    except ImportError as e:
        io.say(f"Missing package for {suffix} files: {e}")
        return None
    except Exception as e:
        io.say(f"Could not read the file: {type(e).__name__}: {e}")
        return None
    s.ns["df"] = df
    return df


# ---------------------------------------------------------------------------
# menu actions


def _summary(s: Session, io: IO) -> None:
    io.say(s.profile_markdown() if s.datasets else "No data loaded - choose 7.")


def _ask_ai(s: Session, io: IO) -> None:
    if not s.assistant:
        io.say("No AI set up - choose 8.")
        return
    while True:
        question = io.line("Question (Enter = back): ")
        if not question:
            return
        io.say(s.command(f"/ask {question}"))


def _read_script(io: IO, pick: Callable) -> tuple[Path, str] | None:
    path = ask_file(io, pick, "Python file", SCRIPT_TYPES)
    if path is None:
        return None
    code = path.read_text(encoding="utf-8")
    try:
        compile(code, str(path), "exec")
    except SyntaxError as e:
        io.say(f"Syntax error, line {e.lineno}: {e.msg}")
        return None
    return path, code


def _check_script(s: Session, io: IO, pick: Callable) -> None:
    got = _read_script(io, pick)
    if got is None:
        return
    path, code = got
    found = s.check(code, show=False, extra_frames=s.profiles_for_code(code, base_dir=path.parent))
    if not found:
        io.say("No suggestions.")
        return
    for sug in found:
        io.say(sug.render() + "\n")
    while True:
        number = io.line("Explain suggestion # (Enter = back): ")
        if not number:
            return
        io.say(s.command(f"/describe {number.lstrip('#')}"))


def _run_script(s: Session, io: IO, pick: Callable) -> None:
    got = _read_script(io, pick)
    if got is None:
        return
    path, code = got
    s.ns.update({"__name__": "__main__", "__file__": str(path.resolve())})
    sys.path.insert(0, str(path.parent.resolve()))
    s.check(code, extra_frames=s.profiles_for_code(code, base_dir=path.parent))
    result = s.run(code, check=False, filename=str(path))
    if result.ok:
        io.say("Done.")
        if result.changed and io.yes("Save the corrected copy?"):
            out = path.with_suffix(".fixed.py")
            out.write_text(result.final_code, encoding="utf-8")
            io.say(f"Saved {out}")
    elif io.yes("Stopped with an error. Explain it?"):
        io.say(s.command("/describe"))


def _learn(s: Session, io: IO) -> None:
    topic = io.line("Topic, e.g. t-test, correlation (Enter = latest suggestion): ")
    io.say(s.command(f"/learn {topic}".strip()))


def _load(s: Session, io: IO, pick: Callable, skip: str = "b") -> None:
    path = ask_file(io, pick, "Data file", DATA_TYPES, skip=skip)
    if path is not None:
        load_dataset(s, io, path)


def _change_ai(s: Session, io: IO) -> None:
    run_wizard(io)
    s.connect_ai()
    io.say(f"AI: {s.ai_status}")


# ---------------------------------------------------------------------------


def run_guide(s: Session, io: IO | None = None, pick: Callable = pick_file,
              repl: Callable[[], object] | None = None, data: str | None = None) -> int:
    """Load data, open the Python prompt, and offer a one-line menu on request."""
    io = io or IO()
    io.say(f"learnstat | AI: {s.ai_status}")
    try:
        if data:
            path = Path(_clean(data)).expanduser()
            if path.is_file():
                load_dataset(s, io, path)
            else:
                io.say(f"Not found: {path}")
        else:
            try:
                _load(s, io, pick, skip="s")
            except Cancelled:
                pass
        actions: dict[str, Callable[[], None]] = {
            "1": lambda: _summary(s, io),
            "2": lambda: _ask_ai(s, io),
            "3": lambda: _check_script(s, io, pick),
            "4": lambda: _run_script(s, io, pick),
            "6": lambda: _learn(s, io),
            "7": lambda: _load(s, io, pick),
            "8": lambda: _change_ai(s, io),
            "?": lambda: io.say(HELP_TEXT),
        }
        if repl is not None:
            actions["5"] = repl
            repl()  # the Python prompt is the default place to work
        while True:
            io.say("\n" + MENU)
            choice = io.line("> ").lower()
            if choice in ("q", "quit", "exit"):
                break
            action = actions.get("?" if choice in ("h", "help") else choice)
            if action is None:
                continue
            try:
                action()
            except Cancelled:
                pass
    except Cancelled:
        pass
    return 0
