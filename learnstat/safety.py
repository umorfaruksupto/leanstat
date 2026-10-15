"""Static safety checks for code that learnstat executes or generates.

This is a policy layer, not a sandbox: user code runs in the user's own Python
process with the user's permissions, exactly as if they had run it themselves.
What the checks guarantee is that

* code with potentially destructive side effects (deleting/overwriting files,
  shell commands, network access, dynamic ``eval``/``exec``) is flagged and
  only runs after the user confirms, and
* an automatic correction can never *introduce* such an operation - a fix
  that adds one is rejected outright.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from .advisor import call_name, const_value, dotted_name, kwarg

_DESTRUCTIVE_CALLS = {
    "os.remove": "deletes a file",
    "os.unlink": "deletes a file",
    "os.rmdir": "deletes a directory",
    "os.removedirs": "deletes directories",
    "os.rename": "renames/overwrites a file",
    "os.replace": "overwrites a file",
    "os.system": "runs a shell command",
    "os.popen": "runs a shell command",
    "os.kill": "terminates a process",
    "shutil.rmtree": "deletes a directory tree",
    "shutil.move": "moves/overwrites files",
    "shutil.copy": "may overwrite a file",
    "shutil.copyfile": "may overwrite a file",
    "subprocess.run": "runs an external command",
    "subprocess.call": "runs an external command",
    "subprocess.Popen": "runs an external command",
    "subprocess.check_call": "runs an external command",
    "subprocess.check_output": "runs an external command",
}
_DESTRUCTIVE_METHODS = {
    "unlink": "deletes a file",
    "rmdir": "deletes a directory",
    "write_text": "overwrites a file",
    "write_bytes": "overwrites a file",
    "to_csv": "writes (and may overwrite) a file",
    "to_excel": "writes (and may overwrite) a file",
    "to_parquet": "writes (and may overwrite) a file",
    "to_json": "writes (and may overwrite) a file",
    "to_pickle": "writes (and may overwrite) a file",
    "to_sql": "writes to a database",
    "savefig": "writes (and may overwrite) an image file",
}
_DANGEROUS_BUILTINS = {"eval", "exec", "compile", "__import__"}
_NETWORK_MODULES = {"socket", "requests", "urllib", "http", "httpx", "ftplib", "smtplib"}


@dataclass(frozen=True)
class SafetyFinding:
    line: int
    operation: str
    reason: str

    def __str__(self) -> str:
        return f"line {self.line}: `{self.operation}` {self.reason}"


def scan(code: str) -> list[SafetyFinding]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    out: list[SafetyFinding] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m.split(".")[0] in _NETWORK_MODULES:
                    out.append(SafetyFinding(node.lineno, f"import {m}", "enables network access"))
        if not isinstance(node, ast.Call):
            continue
        dotted = dotted_name(node.func) or ""
        name = call_name(node)
        if dotted in _DESTRUCTIVE_CALLS:
            out.append(SafetyFinding(node.lineno, dotted, _DESTRUCTIVE_CALLS[dotted]))
        elif isinstance(node.func, ast.Name) and name in _DANGEROUS_BUILTINS:
            out.append(SafetyFinding(node.lineno, name, "executes dynamically generated code"))
        elif isinstance(node.func, ast.Name) and name == "open":
            mode = const_value(node.args[1]) if len(node.args) > 1 else const_value(kwarg(node, "mode"))
            if isinstance(mode, str) and any(m in mode for m in "wax+"):
                out.append(SafetyFinding(node.lineno, f"open(..., {mode!r})", "writes to a file"))
        elif isinstance(node.func, ast.Attribute) and name in _DESTRUCTIVE_METHODS:
            out.append(SafetyFinding(node.lineno, f".{name}()", _DESTRUCTIVE_METHODS[name]))
    return out


def introduced_by(original: str, fixed: str) -> list[SafetyFinding]:
    """Findings present in ``fixed`` that were not already in ``original``."""
    before = {(f.operation, f.reason) for f in scan(original)}
    return [f for f in scan(fixed) if (f.operation, f.reason) not in before]
