"""Guided AI setup in the terminal: company -> API key -> model -> live check -> save.

    learnstat setup        (or /ai inside a session)

All input/output goes through :class:`IO` so the wizard is easy to test and works
in a plain terminal, in the learnstat REPL and in Jupyter.
"""

from __future__ import annotations

import getpass
import time
from dataclasses import dataclass, field
from typing import Callable

from . import config, providers
from .llm import AIError, LLMAssistant, fetch_models


class Cancelled(Exception):
    """The user left the wizard (q, Ctrl+C or end of input)."""


@dataclass
class IO:
    ask: Callable[[str], str] = input
    secret: Callable[[str], str] = getpass.getpass
    say: Callable[[str], None] = print

    def line(self, prompt: str) -> str:
        try:
            return self.ask(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            raise Cancelled from None

    def hidden(self, prompt: str) -> str:
        try:
            return self.secret(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            raise Cancelled from None

    def yes(self, prompt: str, default: bool = True) -> bool:
        answer = self.line(f"{prompt} [{'Y/n' if default else 'y/N'}] ").lower()
        return default if not answer else answer in ("y", "yes")


@dataclass
class Draft:
    provider: providers.Provider
    base_url: str
    api_key: str = ""
    model: str = ""
    models: list[str] = field(default_factory=list)

    def config(self) -> config.AIConfig:
        return config.AIConfig(api_key=self.api_key, base_url=self.base_url, model=self.model, provider=self.provider.id)


RULE = "-" * 60


# ---------------------------------------------------------------------------
# steps


def choose_provider(io: IO) -> providers.Provider:
    io.say("\nWhich AI company do you want to use?\n")
    choices = providers.all_choices()
    for i, p in enumerate(choices, 1):
        io.say(f"  {i:>2}) {p.name}")
    while True:
        answer = io.line("\nNumber (or q to quit): ").lower()
        if answer in ("q", "quit", "exit"):
            raise Cancelled
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        by_name = [p for p in choices if answer and (answer == p.id or answer in p.name.lower())]
        if len(by_name) == 1:
            return by_name[0]
        io.say(f"Please type a number from 1 to {len(choices)}.")


def ask_endpoint(io: IO, provider: providers.Provider) -> str:
    if provider.id != "custom":
        if provider.note:
            io.say(f"\n{provider.note}")
        return provider.base_url
    io.say("\nAny OpenAI-compatible server works (it must offer /chat/completions).")
    while True:
        url = io.line("Base URL (for example https://api.example.com/v1): ")
        if url.startswith(("http://", "https://")):
            return url
        io.say("The address must start with http:// or https://")


def ask_key(io: IO, provider: providers.Provider) -> str:
    if not provider.needs_key:
        return ""
    if provider.key_url:
        io.say(f"\nCreate a key at {provider.key_url}")
    io.say("Paste your API key (it is hidden while you type; right-click or Ctrl+V pastes in most terminals).")
    while True:
        key = io.hidden("API key: ")
        if key:
            return key
        io.say("The key is empty - paste it and press Enter (or Ctrl+C to quit).")


def load_models(io: IO, d: Draft) -> str | None:
    """Fetch the model list. Returns an error message when the key itself is the problem."""
    io.say("\nChecking your key and loading the available models ...")
    try:
        d.models = fetch_models(d.config())
        io.say(f"  OK - {len(d.models)} models available.")
        return None
    except AIError as e:
        if e.fatal:
            return str(e)
        d.models = list(d.provider.models)
        io.say(f"  Could not load the model list ({e}).")
        if d.models:
            io.say("  Showing the usual models instead - you can also type any model name.")
        else:
            io.say("  Type the model name you want to use.")
        return None


def choose_model(io: IO, d: Draft) -> str:
    shown = [m for m in d.models if providers.is_chat_model(m)] or list(d.models)
    query = ""
    while True:
        view = [m for m in shown if query.lower() in m.lower()] if query else shown
        if view:
            io.say(f"\nModels{f' matching {query!r}' if query else ''}:\n")
            for i, m in enumerate(view[:40], 1):
                io.say(f"  {i:>2}) {m}")
            if len(view) > 40:
                io.say(f"  ... and {len(view) - 40} more - type part of a name to filter.")
        answer = io.line("\nPick a number, type part of a name to filter, or type a full model name: ")
        if not answer:
            continue
        if answer.isdigit() and 1 <= int(answer) <= len(view):
            return view[int(answer) - 1]
        exact = [m for m in d.models if m.lower() == answer.lower()]
        if exact:
            return exact[0]
        matches = [m for m in shown if answer.lower() in m.lower()]
        if len(matches) == 1:
            return matches[0]
        if matches:
            query = answer
            continue
        if io.yes(f"'{answer}' is not in the list. Use it as the model name anyway?", default=False):
            return answer
        query = ""


def verify(io: IO, d: Draft) -> str | None:
    """Send one tiny request. Returns None when it works, else the error message."""
    label = d.provider.name.split(" (")[0] if d.provider.id != "custom" else d.base_url
    io.say(f"\nTesting {label} / {d.model} ...")
    assistant = LLMAssistant(d.config())
    started = time.monotonic()
    reply = assistant.test()
    if reply is None:
        return assistant.last_error or "the test request failed"
    io.say(f"  OK - the model answered in {time.monotonic() - started:.1f}s.")
    return None


def profile_name(io: IO, d: Draft) -> str:
    name = d.provider.id
    if d.provider.id == "custom":
        name = io.line("Name for this endpoint (default: custom): ") or "custom"
    return name


def _save(io: IO, d: Draft, tested: bool) -> str:
    name = profile_name(io, d)
    path = config.save_profile(name, d.provider.id, d.base_url, d.api_key, d.model, activate=True)
    if tested:
        io.say("")
        io.say(f"Ready. Saved '{name}' ({d.model}) and made it the active AI.")
    else:
        io.say("")
        io.say(f"Saved '{name}' ({d.model}) without a successful test.")
    io.say(f"Settings file: {path}")
    return name


def add_provider(io: IO) -> str:
    """Run the full flow for one new provider. Returns the saved profile name."""
    provider = choose_provider(io)
    while True:  # restarts here when the user picks another company
        d = Draft(provider=provider, base_url=ask_endpoint(io, provider))
        d.api_key = ask_key(io, provider)
        reload_models = True
        while True:
            if reload_models:
                problem = load_models(io, d)
                reload_models = False
                if problem:  # the key was rejected
                    io.say("")
                    io.say(f"X  {problem}")
                    action = io.line("[r] re-enter key   [p] pick another company   [q] quit: ").lower()
                    if action == "q":
                        raise Cancelled
                    if action == "p":
                        break
                    d.api_key = ask_key(io, provider)
                    reload_models = True
                    continue
            if not d.model:
                d.model = choose_model(io, d)
            problem = verify(io, d)
            if problem is None:
                return _save(io, d, tested=True)
            io.say("")
            io.say(f"X  {problem}")
            action = io.line(
                "[m] choose another model   [k] re-enter key   [r] retry   [p] another company   "
                "[s] save anyway   [q] quit: ").lower()
            if action == "q":
                raise Cancelled
            if action == "p":
                break
            if action == "s":
                return _save(io, d, tested=False)
            if action == "k":
                d.api_key = ask_key(io, provider)
                reload_models = True
                d.model = ""
            elif action != "r":  # "m" or anything else: pick a model again
                d.model = ""
        provider = choose_provider(io)


def use_saved(io: IO, name: str) -> bool:
    """Activate a saved profile after checking it still works. Returns True on success."""
    prof = config.profiles()[name]
    cfg = config.AIConfig(api_key=prof.get("api_key", ""), base_url=prof.get("base_url", config.DEFAULT_BASE_URL),
                          model=prof.get("model", config.DEFAULT_MODEL), provider=prof.get("provider", "custom"),
                          profile=name)
    io.say(f"\nTesting {name} / {cfg.model} ...")
    assistant = LLMAssistant(cfg)
    started = time.monotonic()
    if assistant.test() is None:
        io.say(f"X  {assistant.last_error}")
        if not io.yes("Use it anyway?", default=False):
            return False
    else:
        io.say(f"  OK - answered in {time.monotonic() - started:.1f}s.")
    config.set_active(name)
    io.say(f"Active profile: {name}")
    return True


def describe_profile(name: str, prof: dict, active: bool) -> str:
    company = providers.get(prof.get("provider"))
    label = company.name.split(" (")[0] if company and company.id != "custom" else prof.get("base_url", "custom endpoint")
    star = "  <- active" if active else ""
    return f"{name}: {label}, model {prof.get('model', config.DEFAULT_MODEL)}, key {config.mask(prof.get('api_key', ''))}{star}"


# ---------------------------------------------------------------------------
# entry point


def run_wizard(io: IO | None = None) -> str | None:
    """Interactive AI setup. Returns the active profile name, or None if the user left without one."""
    io = io or IO()
    try:
        io.say(f"\n{RULE}\n learnstat - AI assistant setup\n{RULE}")
        while True:
            saved = config.profiles()
            if not saved:
                return add_provider(io)
            active = config.active_name()
            io.say("\nSaved AI accounts:\n")
            names = list(saved)
            for i, n in enumerate(names, 1):
                io.say(f"  {i:>2}) {describe_profile(n, saved[n], n == active)}")
            io.say("\n   n) add a new company / key      d) delete one      s) skip, use offline rules      q) quit")
            answer = io.line("\nNumber to switch, or a letter: ").lower()
            if answer == "n":
                return add_provider(io)
            if answer in ("s", "skip"):
                config.set_skip_setup(True)
                io.say("OK - learnstat will use offline rules. Run `learnstat setup` any time to add an AI.")
                return None
            if answer in ("q", "quit"):
                raise Cancelled
            if answer == "d":
                pick = io.line("Number to delete: ")
                if pick.isdigit() and 1 <= int(pick) <= len(names):
                    target = names[int(pick) - 1]
                    if io.yes(f"Delete '{target}' and its stored key?", default=False):
                        config.remove_profile(target)
                        io.say(f"Deleted '{target}'.")
                continue
            if answer.isdigit() and 1 <= int(answer) <= len(names):
                if use_saved(io, names[int(answer) - 1]):
                    return names[int(answer) - 1]
                continue
            io.say("Please type a number, n, d, s or q.")
    except Cancelled:
        io.say("\nSetup cancelled - nothing was changed after the last saved step.")
        return config.active_name()
