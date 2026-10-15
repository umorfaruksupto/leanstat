"""learnstat - an AI-assisted learning companion for statistics in Python.

You write ordinary pandas / NumPy / SciPy / Matplotlib code; learnstat profiles
your data, suggests better approaches, explains its reasoning and helps you
recover from mistakes.

    import learnstat as ls
    df = ls.read_csv("survey.csv")      # a normal pandas DataFrame, now profiled
    ls.check("df['income'].mean()")     # suggestions, without running anything
    print(ls.command("/describe"))      # why?

In Jupyter: ``%load_ext learnstat``.
"""

from __future__ import annotations

from .advisor import CodeAdvisor
from .memory import MemoryStore
from .profile import DatasetProfile, profile_dataframe
from .recovery import Executor, RunResult
from .session import Session, get_session, set_session
from .suggestions import Suggestion

__version__ = "0.2.2"


def read_csv(path, name: str | None = None, **kwargs):
    """``pandas.read_csv`` plus dataset profiling and persistent memory."""
    return get_session().read_csv(path, name=name, **kwargs)


def read_excel(path, name: str | None = None, **kwargs):
    return get_session().read_excel(path, name=name, **kwargs)


def read_parquet(path, name: str | None = None, **kwargs):
    return get_session().read_parquet(path, name=name, **kwargs)


def read_json(path, name: str | None = None, **kwargs):
    return get_session().read_json(path, name=name, **kwargs)


def track(df, name: str | None = None, **kwargs):
    """Profile a DataFrame you already have."""
    return get_session().track(df, name=name, **kwargs)


def check(code: str, show: bool = True):
    """Suggestions for ``code`` (it is not executed)."""
    return get_session().check(code, show=show)


def run(code: str):
    """Run ``code`` in the default session with suggestions and guarded recovery."""
    return get_session().run(code)


def command(text: str) -> str:
    """Run a slash command such as ``/describe`` and return its text."""
    return get_session().command(text)


def load_ipython_extension(ip):
    from .ipython_ext import load_ipython_extension as _load

    _load(ip)


def unload_ipython_extension(ip):
    from .ipython_ext import unload_ipython_extension as _unload

    _unload(ip)


__all__ = [
    "Session", "get_session", "set_session", "read_csv", "read_excel", "read_parquet", "read_json",
    "track", "check", "run", "command", "profile_dataframe", "DatasetProfile", "MemoryStore",
    "CodeAdvisor", "Executor", "RunResult", "Suggestion",
]
