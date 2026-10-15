"""Command-line interface: ``learnstat`` (guided start), ``setup``, ``profile|check|watch|run|repl|config``."""

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
    return _repl(_session(args), args.data)


def _repl(s: Session, data: str | None) -> int:
    import numpy as np
    import pandas as pd

    s.ns.update(pd=pd, np=np)
    ready = ["pd", "np"]
    if data:
        s.ns["df"] = s.read_csv(data)
    if "df" in s.ns:
        ready.insert(0, "df")
    print(f"Ready: {', '.join(ready)}   |   /help   |   menu")
    buffer: list[str] = []
    while True:
        try:
            line = input("... " if buffer else ">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not buffer and line.strip().lower() in ("menu", "back", "exit", "quit", "exit()", "quit()"):
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


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def cmd_config(args) -> int:
    from . import config
    from .setup import describe_profile

    action = args.action
    try:
        if action == "path":
            print(config.config_path())
        elif action == "list":
            saved, active = config.profiles(), config.active_name()
            for name, prof in saved.items():
                print(describe_profile(name, prof, name == active))
            if not saved:
                print("No saved AI accounts. Run: learnstat setup")
        elif action == "show":
            cfg = config.resolve()
            stored = config.profiles().get(cfg.profile or "", {})
            print(f"profile   {cfg.profile or '(none)'}   provider {cfg.provider}")
            for key in config.KEYS:
                value = getattr(cfg, key)
                shown = config.mask(value) if key == "api_key" else value
                env = config.ENV_VARS[key]
                source = "env " + env if os.environ.get(env, "").strip() else ("saved" if key in stored else "default")
                print(f"{key:9} {shown}   [{source}]")
            print(f"file      {config.config_path()}")
        elif action in ("use", "remove"):
            if not args.key:
                print(f"usage: learnstat config {action} <profile>")
                return 2
            path = config.set_active(args.key) if action == "use" else config.remove_profile(args.key)
            print(f"{action} {args.key} -> {path}")
        elif action in ("set", "unset"):
            if not args.key or (action == "set" and args.value is None):
                print("usage: learnstat config set <api_key|base_url|model> <value>")
                print("       learnstat config unset <key>")
                return 2
            path = config.set_value(args.key, args.value) if action == "set" else config.unset_value(args.key)
            print(f"{action} {args.key} -> {path}")
        elif action == "test":
            from .llm import LLMAssistant

            a = LLMAssistant()
            if not a.available():
                print(f"AI not ready: {a.last_error}")
                return 1
            print(f"Testing {a.cfg.chat_url} with model {a.model} ...")
            reply = a.test()
            if reply is None:
                print(f"X  {a.last_error}")
                return 1
            print(f"OK - {a.cfg.provider_name} / {a.model} replied: {reply[:80]}")
    except KeyError as e:
        print(e.args[0])
        return 2
    return 0


def _start(args, run_setup: bool) -> int:
    """The guided start: (optional) AI setup, a quick AI check, then the interactive session."""
    from . import config
    from .llm import LLMAssistant
    from .setup import IO, run_wizard

    if not _interactive():
        print("learnstat start needs an interactive terminal. Try: learnstat --help")
        return 2
    ai_error = None
    if not args.no_ai:
        cfg_ready = config.resolve().is_configured
        if run_setup:
            run_wizard()
        elif not cfg_ready and not config.skip_setup():
            try:
                answer = input("No AI set up. Set one up now? [Y/n/never] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer in ("", "y", "yes"):
                run_wizard()
            elif answer == "never":
                config.set_skip_setup(True)
        a = LLMAssistant()
        if a.available() and a.test(timeout=15) is None:
            ai_error = a.last_error
    from .guide import run_guide

    s = _session(args)
    if ai_error:
        s.assistant, s.ai_status = None, f"offline ({ai_error}) - menu 8 to fix"
    return run_guide(s, repl=lambda: _repl(s, None), data=args.data)


def cmd_setup(args) -> int:
    if not _interactive():
        print("learnstat setup is interactive. In scripts use: learnstat config set api_key|base_url|model <value>")
        return 2
    from .setup import run_wizard

    run_wizard()
    from . import config
    if config.resolve().is_configured:
        try:
            go = input("Start now? [Y/n] ").strip().lower() in ("", "y", "yes")
        except (EOFError, KeyboardInterrupt):
            go = False
        if go:
            return _start(args, run_setup=False)
    return 0


def cmd_start(args) -> int:
    return _start(args, run_setup=False)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="learnstat", description="AI-assisted learning companion for statistics in Python.")
    p.add_argument("--memory-dir", default=os.environ.get("LEARNSTAT_MEMORY", ".learnstat"))
    p.add_argument("--no-ai", action="store_true", help="never call the AI; use offline rules only")
    p.add_argument("--data", help="dataset to load when starting the interactive session")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("setup", help="guided AI setup: choose company, enter key, pick model, check, then start")
    sp.add_argument("--data")
    sp.set_defaults(fn=cmd_setup)

    sp = sub.add_parser("start", help="start the interactive session (also what plain `learnstat` does)")
    sp.add_argument("--data")
    sp.set_defaults(fn=cmd_start)

    sp = sub.add_parser("config", help="scripting access to AI settings: list|use|remove|set|unset|show|test|path")
    sp.add_argument("action", choices=["list", "use", "remove", "set", "unset", "show", "test", "path"])
    sp.add_argument("key", nargs="?", help="profile name, or api_key / base_url / model")
    sp.add_argument("value", nargs="?")
    sp.set_defaults(fn=cmd_config)

    sp = sub.add_parser("profile", help="profile a dataset and write its Markdown memory")
    sp.add_argument("data")
    sp.add_argument("--name")
    sp.set_defaults(fn=cmd_profile)

    for name, fn, help_ in [("check", cmd_check, "suggest improvements for a script without running it"),
                            ("watch", cmd_watch, "re-check a script every time it is saved")]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("script")
        sp.add_argument("--data", action="append", help="dataset file(s) the script analyses")
        sp.add_argument("--ai-review", action="store_true", help="also ask the AI to review the code")
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
    if args.cmd is None:
        return cmd_start(args)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
