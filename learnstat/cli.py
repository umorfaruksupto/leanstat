"""Command-line interface: ``learnstat profile|check|watch|run|repl``."""

from __future__ import annotations

import argparse
import codeop
import os
import sys
import time
from pathlib import Path

from .commands import is_command
from .session import Session, set_session


def _confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _session(args, interactive: bool = True) -> Session:
    confirm = (lambda q: True) if getattr(args, "yes", False) else (_confirm if interactive else None)
    s = Session(
        memory_dir=args.memory_dir,
        ai=False if args.no_ai else "auto",
        max_attempts=getattr(args, "max_attempts", 3),
        auto_fix=not getattr(args, "no_fix", False),
        confirm=confirm,
        ai_review=getattr(args, "ai_review", False),
    )
    return set_session(s)


def cmd_profile(args) -> int:
    s = _session(args)
    s.verbose = False
    df = s.read_csv(args.data, name=args.name) if not args.data.endswith((".xlsx", ".xls")) else s.read_excel(args.data, name=args.name)
    del df
    print(s.memory.md_path(s.profile().name).read_text(encoding="utf-8"))
    return 0


def _static_check(s: Session, script: Path, data: list[str]) -> list:
    code = script.read_text(encoding="utf-8")
    frames = s.profiles_for_code(code, base_dir=script.parent)
    for d in data or []:
        s.verbose = False
        s.read_csv(d)
    return s.check(code, show=False, extra_frames=frames)


def cmd_check(args) -> int:
    s = _session(args, interactive=False)
    found = _static_check(s, Path(args.script), args.data)
    if not found:
        print("No suggestions - nothing the rules flag for this data.")
    for sug in found:
        print(sug.render())
        print()
    return 0


def cmd_watch(args) -> int:
    s = _session(args, interactive=False)
    script = Path(args.script)
    print(f"Watching {script} - suggestions appear each time you save. Ctrl+C to stop.")
    last = None
    try:
        while True:
            mtime = script.stat().st_mtime if script.exists() else None
            if mtime and mtime != last:
                last = mtime
                code = script.read_text(encoding="utf-8")
                try:
                    compile(code, str(script), "exec")
                except SyntaxError as e:
                    print(f"[{time.strftime('%H:%M:%S')}] SyntaxError line {e.lineno}: {e.msg}")
                    continue
                found = _static_check(s, script, args.data)
                stamp = time.strftime("%H:%M:%S")
                print(f"[{stamp}] {len(found)} new suggestion(s)")
                for sug in found:
                    print(sug.render())
                    print()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def cmd_run(args) -> int:
    s = _session(args)
    script = Path(args.script)
    s.ns.update({"__name__": "__main__", "__file__": str(script.resolve())})
    sys.path.insert(0, str(script.parent.resolve()))
    for d in args.data or []:
        s.read_csv(d)
    code = script.read_text(encoding="utf-8")
    s.check(code, extra_frames=s.profiles_for_code(code, base_dir=script.parent))
    result = s.run(code, check=False, filename=str(script))
    if result.ok and result.changed and args.write_fixed:
        out = script.with_suffix(".fixed.py")
        out.write_text(result.final_code, encoding="utf-8")
        print(f"Corrected copy written to {out} (your original is unchanged).")
    return 0 if result.ok else 1


def cmd_repl(args) -> int:
    s = _session(args)
    import numpy as np
    import pandas as pd

    s.ns.update(pd=pd, np=np)
    ready = ["pd", "np"]
    if args.data:
        s.ns["df"] = s.read_csv(args.data)
        ready.insert(0, "df")
    print(f"learnstat REPL - AI: {s.ai_status}. Ready: {', '.join(ready)}. Type /help for commands, Ctrl+Z/Ctrl+D to exit.")
    buffer: list[str] = []
    while True:
        try:
            line = input("... " if buffer else ">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not buffer and is_command(line):
            print(s.command(line))
            continue
        buffer.append(line)
        src = "\n".join(buffer)
        try:
            complete = codeop.compile_command(src, symbol="single") is not None
        except (SyntaxError, ValueError, OverflowError):
            complete = True  # let the runner diagnose it
        if not complete:
            continue
        buffer = []
        result = s.run(src)
        if result.ok and result.value is not None:
            print(repr(result.value))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="learnstat", description="AI-assisted learning companion for statistics in Python.")
    p.add_argument("--memory-dir", default=os.environ.get("LEARNSTAT_MEMORY", ".learnstat"))
    p.add_argument("--no-ai", action="store_true", help="never call Claude; use offline rules only")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("profile", help="profile a dataset and write its Markdown memory")
    sp.add_argument("data")
    sp.add_argument("--name")
    sp.set_defaults(fn=cmd_profile)

    for name, fn, help_ in [("check", cmd_check, "suggest improvements for a script without running it"),
                            ("watch", cmd_watch, "re-check a script every time it is saved")]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("script")
        sp.add_argument("--data", action="append", help="dataset file(s) the script analyses")
        sp.add_argument("--ai-review", action="store_true", help="also ask Claude to review the code")
        if name == "watch":
            sp.add_argument("--interval", type=float, default=1.0)
        sp.set_defaults(fn=fn)

    sp = sub.add_parser("run", help="run a script with suggestions and guarded error recovery")
    sp.add_argument("script")
    sp.add_argument("--data", action="append")
    sp.add_argument("--max-attempts", type=int, default=3)
    sp.add_argument("--no-fix", action="store_true", help="diagnose errors but never correct them")
    sp.add_argument("--yes", action="store_true", help="approve side-effecting code and AI corrections without asking")
    sp.add_argument("--write-fixed", action="store_true", help="write the corrected script to <name>.fixed.py")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("repl", help="interactive session with suggestions, recovery and /commands")
    sp.add_argument("--data")
    sp.add_argument("--max-attempts", type=int, default=3)
    sp.set_defaults(fn=cmd_repl)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
