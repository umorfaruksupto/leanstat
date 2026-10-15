"""The guided home screen: students pick numbers from a menu instead of learning commands.

After the AI is set up, ``learnstat`` shows this menu. Every action asks for what it
needs in plain English (pick a file in a window, type a question) and prints the result.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from .session import Session
from .setup import Cancelled, IO, run_wizard

DATA_TYPES = [("Data files", "*.csv *.tsv *.txt *.xlsx *.xls *.json *.parquet"), ("All files", "*.*")]
SCRIPT_TYPES = [("Python files", "*.py"), ("All files", "*.*")]

MENU = """
What would you like to do?

  1) See a summary of my data
  2) Ask the AI a question about my data
  3) Get suggestions for a Python script
  4) Run a Python script with automatic help
  5) Write Python code here (my data is ready as "df")
  6) Learn a statistics topic
  7) Load a different data file
  8) Change the AI (company, key, model)
  9) Help
  q) Quit"""

HELP_TEXT = """
How learnstat works
-------------------
1  Shows what is in your data: columns, types, missing values and possible problems.
2  Type any question in your own words, for example "which column has the most missing values?".
3  Pick a Python file. learnstat reads it (without running it) and suggests better statistics
   or charts for your data. Enter a suggestion's number to get it explained.
4  Runs your Python file. If it crashes on a simple mistake (like a misspelled column) learnstat
   explains it and can fix it for you.
5  A Python prompt with your data already loaded as df. Type "menu" to come back here.
   Inside it, lines starting with / are shortcuts, and /help lists them.
6  Short lessons, for example "t-test", "correlation" or "missing values".
7  Load another file.     8  Switch to another AI company or key.

The AI is optional: without it learnstat still gives rule-based suggestions and explanations."""

PYTHON_TIPS = (
    'Python prompt - your data is "df". Try: df.head()  or  df.describe()\n'
    '  Type /help for shortcuts, "menu" to go back to the menu.'
)


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


def ask_file(io: IO, pick: Callable, what: str, types: list[tuple[str, str]]) -> Path | None:
    answer = io.line(f"\nChoose your {what}. Press Enter to browse, or paste / drag the file here (b = back): ")
    if answer.lower() in ("b", "back"):
        return None
    if not answer:
        chosen = pick(f"Choose your {what}", types)
        if not chosen:
            io.say("No file chosen. You can also paste the full path to the file.")
            return None
        answer = chosen
    path = Path(_clean(answer)).expanduser()
    if not path.is_file():
        io.say(f"I can't find '{path}'. Check the name, or press Enter next time to browse for it.")
        return None
    return path


def load_dataset(s: Session, io: IO, path: Path):
    suffix = path.suffix.lower()
    try:
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
        io.say(f"Reading {suffix} files needs an extra package: {e}")
        return None
    except Exception as e:
        io.say(f"I couldn't read that file as data ({type(e).__name__}: {e}).")
        return None
    s.ns["df"] = df
    io.say('\nYour data is ready, and is available as "df" when you write Python code.')
    return df


# ---------------------------------------------------------------------------
# menu actions


def _summary(s: Session, io: IO) -> None:
    if not s.datasets:
        io.say("No data is loaded yet - choose 7 to load a file.")
        return
    io.say("\n" + s.profile_markdown())


def _ask_ai(s: Session, io: IO) -> None:
    if not s.assistant:
        io.say(f"The AI is {s.ai_status}. Choose 8 to set one up - or use 6 for built-in lessons.")
        return
    io.say('\nAsk anything about your data. Example: "which columns have missing values?"')
    while True:
        question = io.line("\nYour question (Enter to go back): ")
        if not question:
            return
        io.say("\nThinking ...")
        io.say(s.command(f"/ask {question}"))


def _read_script(io: IO, pick: Callable) -> tuple[Path, str] | None:
    path = ask_file(io, pick, "Python file", SCRIPT_TYPES)
    if path is None:
        return None
    code = path.read_text(encoding="utf-8")
    try:
        compile(code, str(path), "exec")
    except SyntaxError as e:
        io.say(f"There is a syntax error on line {e.lineno}: {e.msg}. Fix it and try again.")
        return None
    return path, code


def _check_script(s: Session, io: IO, pick: Callable) -> None:
    got = _read_script(io, pick)
    if got is None:
        return
    path, code = got
    found = s.check(code, show=False, extra_frames=s.profiles_for_code(code, base_dir=path.parent))
    if not found:
        io.say("\nNo suggestions - the rules found nothing to improve for this data.")
        return
    io.say(f"\n{len(found)} suggestion(s):\n")
    for sug in found:
        io.say(sug.render())
        io.say("")
    while True:
        pick_id = io.line("Enter a suggestion number to have it explained (Enter to go back): ")
        if not pick_id:
            return
        io.say("\n" + s.command(f"/describe {pick_id.lstrip('#')}") + "\n")


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
        io.say("\nDone - the script finished.")
        if result.changed and io.yes("learnstat corrected something. Save a corrected copy next to your file?"):
            out = path.with_suffix(".fixed.py")
            out.write_text(result.final_code, encoding="utf-8")
            io.say(f"Saved {out} (your original is unchanged).")
    else:
        io.say("\nThe script stopped with an error.")
        if io.yes("Explain what went wrong?"):
            io.say("\n" + s.command("/describe"))


def _learn(s: Session, io: IO) -> None:
    io.say('\nExamples: t-test, correlation, missing values, skewed data, pie charts, histograms')
    topic = io.line("Topic (Enter for a lesson about your latest suggestion): ")
    io.say("\n" + s.command(f"/learn {topic}".strip()))


def _load(s: Session, io: IO, pick: Callable) -> None:
    path = ask_file(io, pick, "data file (CSV or Excel)", DATA_TYPES)
    if path is not None:
        load_dataset(s, io, path)


def _change_ai(s: Session, io: IO) -> None:
    run_wizard(io)
    s.connect_ai()
    io.say(f"\nAI: {s.ai_status}")


# ---------------------------------------------------------------------------


def run_guide(s: Session, io: IO | None = None, pick: Callable = pick_file,
              repl: Callable[[], object] | None = None, data: str | None = None) -> int:
    """Show the home menu until the user quits."""
    io = io or IO()
    io.say("\n" + "=" * 60 + "\n learnstat - learn statistics with Python\n" + "=" * 60)
    io.say(f"AI: {s.ai_status}")
    try:
        if data:
            path = Path(_clean(data)).expanduser()
            if path.is_file():
                load_dataset(s, io, path)
            else:
                io.say(f"I can't find '{path}'.")
        elif io.yes("\nLoad your data file now?"):
            try:
                _load(s, io, pick)
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
            "9": lambda: io.say(HELP_TEXT),
        }
        if repl is not None:
            def python_mode() -> None:
                io.say("\n" + PYTHON_TIPS)
                repl()
            actions["5"] = python_mode
        while True:
            io.say(MENU)
            choice = io.line("\nType a number: ").lower()
            if choice in ("q", "quit", "exit"):
                break
            action = actions.get(choice)
            if action is None:
                io.say("Please type one of the numbers in the menu (or q to quit).")
                continue
            try:
                action()
            except Cancelled:
                io.say("\n(back to the menu)")
    except Cancelled:
        pass
    io.say("\nGoodbye - your dataset profiles are saved in .learnstat/ so next time is faster.")
    return 0
