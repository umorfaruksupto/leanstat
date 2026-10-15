"""Jupyter / IPython integration: ``%load_ext learnstat``.

* Before each cell runs, its code is checked against the dataset profiles and
  suggestions are printed (the cell still runs exactly as written).
* A cell with a syntax error gets a diagnosis and, where possible, a corrected
  version placed in a new cell for the user to review and run.
* When a cell raises, the failing statement is diagnosed and - for mechanical
  mistakes - corrected and retried; statements that already ran are not re-run.
* Lines starting with ``/describe``, ``/fix``, ``/memory`` ... are learnstat commands.
* ``/accept`` places the suggested code in a new cell rather than editing yours.
"""

from __future__ import annotations

import ast
import traceback

from .commands import is_command
from .recovery import _fix_syntax, diagnose
from .session import Session, set_session

_state: dict = {}


def _confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, OSError):
        return False


def load_ipython_extension(ip) -> None:
    if "session" in _state:
        return
    session = Session(namespace=ip.user_ns, confirm=_confirm)
    session.on_accept = lambda code: ip.set_next_input(code, replace=False)
    set_session(session)
    ip.learnstat_session = session
    _state["session"] = session

    def command_transformer(lines: list[str]) -> list[str]:
        if lines and is_command(lines[0]):
            text = "".join(lines).strip()
            return [f"print(get_ipython().learnstat_session.command({text!r}))\n"]
        return lines

    def pre_run_cell(info) -> None:
        raw = getattr(info, "raw_cell", "") or ""
        if not raw.strip() or is_command(raw.splitlines()[0]):
            _state["cell"] = None
            return
        code = ip.transform_cell(raw)
        _state["cell"] = code
        try:
            ast.parse(code)
        except SyntaxError as e:
            d = diagnose(e, code, e.lineno, ip.user_ns)
            print(f"[learnstat] {d.exc_type} ({d.label}): {d.message}\n  {d.explanation}")
            fix = _fix_syntax(e, raw)
            if fix:
                try:
                    ast.parse(ip.transform_cell(fix.fixed))
                    print(f"  Proposed fix: {fix.description}. A corrected copy was placed in the next cell.")
                    ip.set_next_input(fix.fixed, replace=False)
                except SyntaxError:
                    print(f"  Next step: {d.next_step}")
            else:
                print(f"  Next step: {d.next_step}")
            return
        session.check(code)

    def on_exception(shell, etype, evalue, tb, tb_offset=None):
        code = _state.get("cell")
        frames = traceback.extract_tb(tb)
        cell_frame = next((f for f in frames if f.name == "<module>"), None)
        if not code or cell_frame is None:
            shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
            return None
        print(f"[learnstat] {etype.__name__}: {evalue}")
        result = session.executor(filename=cell_frame.filename).run(
            code, start_line=cell_frame.lineno, initial_exc=evalue
        )
        session._after_run(code, result)
        if result.ok and result.value is not None:
            shell.displayhook(result.value)
        elif not result.ok and not result.attempts:
            shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
        return None

    ip.input_transformers_cleanup.append(command_transformer)
    ip.events.register("pre_run_cell", pre_run_cell)
    ip.set_custom_exc((Exception,), on_exception)
    _state.update(transformer=command_transformer, pre=pre_run_cell)
    print(f"[learnstat] active. AI: {session.ai_status}. Load data with learnstat.read_csv(...); type /help for commands.")


def unload_ipython_extension(ip) -> None:
    if "session" not in _state:
        return
    if _state["transformer"] in ip.input_transformers_cleanup:
        ip.input_transformers_cleanup.remove(_state["transformer"])
    ip.events.unregister("pre_run_cell", _state["pre"])
    ip.set_custom_exc((), None)
    _state.clear()
