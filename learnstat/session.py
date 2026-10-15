"""The Session ties profiling, memory, advice, recovery and commands together."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Callable

import pandas as pd

from .advisor import CodeAdvisor, call_name
from .llm import LLMAssistant
from .memory import DatasetMemory, MemoryStore
from .profile import DatasetProfile, profile_dataframe
from .recovery import Diagnosis, Executor, Fix, RunResult
from .suggestions import Suggestion, SuggestionHistory

_READERS = {"read_csv", "read_excel", "read_parquet", "read_json", "read_table", "read_feather", "read_stata", "read_spss"}


class Session:
    """One analysis session.

    Parameters
    ----------
    memory_dir: where dataset profiles (Markdown + JSON) are kept.
    ai: ``"auto"`` (use the configured AI if available), ``True`` (expect it) or ``False`` (never).
    max_attempts: maximum number of automatic corrections per failing statement.
    auto_fix: set False to only diagnose errors, never correct them.
    confirm: callback ``(question) -> bool`` used for side-effecting code and for
        corrections that need approval. Without it those are refused.
    namespace: the dict user code runs in (IPython passes ``user_ns``).
    ai_review: also ask the AI to review every checked snippet (slower, costs tokens).
    """

    def __init__(
        self,
        memory_dir: str | os.PathLike = ".learnstat",
        ai: bool | str = "auto",
        max_attempts: int = 3,
        auto_fix: bool = True,
        confirm: Callable[[str], bool] | None = None,
        namespace: dict | None = None,
        ai_review: bool = False,
        out: Callable[[str], None] = print,
        verbose: bool = True,
    ):
        self.memory = MemoryStore(memory_dir)
        self.ns = namespace if namespace is not None else {"__name__": "__main__"}
        self.max_attempts = max_attempts
        self.auto_fix = auto_fix
        self.confirm = confirm
        self.ai_review = ai_review
        self.out = out
        self.verbose = verbose
        self.advisor = CodeAdvisor()
        self.history = SuggestionHistory()
        self.datasets: dict[str, DatasetMemory] = {}
        self._frame_ids: dict[int, str] = {}
        self._ephemeral: dict[int, tuple[tuple, DatasetProfile]] = {}
        self.last_code = ""
        self.last_result: RunResult | None = None
        self.on_accept: Callable[[str], None] | None = None

        self.assistant: LLMAssistant | None = None
        self.ai_status = "disabled"
        self._ai_mode = ai
        if ai:
            self.connect_ai()

    def connect_ai(self) -> None:
        """(Re)load the AI from the saved settings - call again after ``learnstat setup`` or ``/ai``."""
        self.assistant, self.ai_status = None, "disabled"
        assistant = LLMAssistant()
        if assistant.available():
            self.assistant = assistant
            self.ai_status = f"enabled ({assistant.cfg.provider_name} / {assistant.model})"
        else:
            self.ai_status = f"not set up - {assistant.last_error}; using offline rules"
            if self._ai_mode is True:
                self.out(f"[learnstat] AI requested but {self.ai_status}")

    # ------------------------------------------------------------------
    # Data loading - thin wrappers that keep pandas' signatures and return
    # ordinary pandas DataFrames.

    def track(self, df: pd.DataFrame, name: str | None = None, source: str | None = None,
              force: bool = False) -> pd.DataFrame:
        """Profile ``df`` (or reuse its cached profile) and remember it."""
        name = name or f"dataset{len(self.datasets) + 1}"
        mem = self.memory.load_or_profile(df, name=name, source=source, force=force)
        self.datasets[name] = mem
        self._frame_ids[id(df)] = name
        if self.verbose:
            self.out(self._loaded_message(mem))
        return df

    def _read(self, reader: str, path, name: str | None, kwargs: dict) -> pd.DataFrame:
        df = getattr(pd, reader)(path, **kwargs)
        stem = Path(str(path)).stem if isinstance(path, (str, os.PathLike)) else None
        return self.track(df, name=name or stem, source=str(path))

    def read_csv(self, path, name: str | None = None, **kwargs) -> pd.DataFrame:
        return self._read("read_csv", path, name, kwargs)

    def read_excel(self, path, name: str | None = None, **kwargs) -> pd.DataFrame:
        return self._read("read_excel", path, name, kwargs)

    def read_parquet(self, path, name: str | None = None, **kwargs) -> pd.DataFrame:
        return self._read("read_parquet", path, name, kwargs)

    def read_json(self, path, name: str | None = None, **kwargs) -> pd.DataFrame:
        return self._read("read_json", path, name, kwargs)

    def _loaded_message(self, mem: DatasetMemory) -> str:
        p = mem.profile
        verb = "Loaded profile from memory" if mem.reused else "Profiled"
        lines = [
            f"[learnstat] {verb} '{p.name}': {p.n_rows} rows x {p.n_cols} columns "
            f"({len(p.numeric)} numeric, {len(p.categorical)} categorical).",
        ]
        for issue in p.issues[:4]:
            lines.append(f"  - {issue}")
        if len(p.issues) > 4:
            lines.append(f"  - ... {len(p.issues) - 4} more issues")
        lines.append(f"  Memory: {self.memory.md_path(p.name)}  (type /memory to view)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Context

    def profile(self, name: str | None = None) -> DatasetProfile | None:
        mem = self.dataset_memory(name)
        return mem.profile if mem else None

    def dataset_memory(self, name: str | None = None) -> DatasetMemory | None:
        if name:
            return self.datasets.get(name)
        return next(reversed(self.datasets.values()), None)

    def frames(self) -> dict[str, DatasetProfile]:
        """Map variable names in the namespace to profiles (tracked or ad-hoc)."""
        out: dict[str, DatasetProfile] = {}
        for var, obj in list(self.ns.items()):
            if var.startswith("_") or not isinstance(obj, pd.DataFrame):
                continue
            name = self._frame_ids.get(id(obj))
            if name and name in self.datasets:
                out[var] = self.datasets[name].profile
                continue
            cached = self._ephemeral.get(id(obj))
            if cached and cached[0] == obj.shape:
                out[var] = cached[1]
            elif obj.size <= 5_000_000:  # derived frames: profile in memory only, never persisted
                prof = profile_dataframe(obj, name=var)
                self._ephemeral[id(obj)] = (obj.shape, prof)
                out[var] = prof
        return out

    def profiles_for_code(self, code: str, base_dir: str | os.PathLike = ".") -> dict[str, DatasetProfile]:
        """For static checks: profile files loaded with ``x = pd.read_csv('file')`` in ``code``."""
        out: dict[str, DatasetProfile] = {}
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return out
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                continue
            call = node.value
            if call_name(call) not in _READERS or not call.args or not isinstance(call.args[0], ast.Constant):
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            path = Path(base_dir) / str(call.args[0].value)
            if not path.exists():
                continue
            reader = call_name(call)
            try:
                df = getattr(pd, reader)(path)
            except Exception:
                continue
            mem = self.memory.load_or_profile(df, name=path.stem, source=str(path))
            self.datasets[path.stem] = mem
            out[target.id] = mem.profile
        return out

    def dataset_for(self, s: Suggestion) -> DatasetMemory | None:
        for mem in self.datasets.values():
            if any(c in mem.profile.columns for c in s.columns):
                return mem
        return self.dataset_memory()

    def profile_markdown(self) -> str:
        if not self.datasets:
            return "(no dataset loaded)"
        parts = []
        for mem in self.datasets.values():
            path = self.memory.md_path(mem.profile.name)
            parts.append(path.read_text(encoding="utf-8") if path.exists() else mem.profile.name)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Advice

    def check(self, code: str, show: bool = True, extra_frames: dict[str, DatasetProfile] | None = None) -> list[Suggestion]:
        """Return (and by default print) new suggestions for ``code``. Never runs it."""
        frames = {**self.frames(), **(extra_frames or {})}
        profiles = [m.profile for m in self.datasets.values()] + [m.profile for m in self.memory.load_all()
                                                                  if m.profile.name not in self.datasets]
        found = [s for s in self.advisor.check(code, frames=frames, profiles=profiles)
                 if not self.history.seen(s.rule, s.source_code)]
        if self.ai_review and self.assistant:
            found += self._ai_review(code, found)
        self.history.extend(found)
        self.last_code = code
        if show and found:
            self.out("\n".join(s.render() for s in found))
        return found

    def _ai_review(self, code: str, already: list[Suggestion]) -> list[Suggestion]:
        items = self.assistant.review(code, self.profile_markdown(), [s.title for s in already]) or []
        out = []
        lines = code.splitlines()
        for it in items:
            line = it.get("line") or None
            src = lines[line - 1].strip() if line and 0 < line <= len(lines) else ""
            if self.history.seen("ai_review:" + it["title"], src):
                continue
            out.append(Suggestion(
                rule="ai_review:" + it["title"], concept="", title=it["title"], problem=it["problem"],
                alternative=it["alternative"], rationale=it["rationale"], code=it.get("code") or None,
                line=line, source_code=src, origin="ai",
            ))
        return out

    # ------------------------------------------------------------------
    # Execution

    def executor(self, filename: str = "<learnstat>") -> Executor:
        return Executor(
            self.ns,
            max_attempts=self.max_attempts,
            auto_fix=self.auto_fix,
            confirm=self.confirm,
            ai_fixer=self._ai_fixer if self.assistant else None,
            filename=filename,
        )

    def run(self, code: str, check: bool = True, filename: str = "<learnstat>") -> RunResult:
        """Check, then execute ``code`` with guarded recovery. Prints the recovery report if anything happened."""
        if check:
            self.check(code)
        result = self.executor(filename).run(code)
        self._after_run(code, result)
        return result

    def _after_run(self, code: str, result: RunResult) -> None:
        self.last_code = code
        self.last_result = result
        if result.attempts or result.blocked or result.notes:
            self.out(result.report())

    def _ai_fixer(self, d: Diagnosis, src: str) -> Fix | None:
        names = {k: type(v).__name__ for k, v in self.ns.items() if not k.startswith("_") and not callable(v)}
        context = f"Variables: {names}\n\n{self.profile_markdown()[:12000]}"
        data = self.assistant.propose_fix(src, f"{d.exc_type}: {d.message}", d.category, context)
        if not data or not data.get("can_fix") or not data.get("fixed_code", "").strip():
            if data and data.get("explanation"):
                d.next_step = data["explanation"]
            return None
        return Fix("ai", "AI suggests: " + data["explanation"], data["fixed_code"].strip("\n"))

    # ------------------------------------------------------------------
    # Commands

    def command(self, text: str) -> str:
        from .commands import dispatch

        return dispatch(self, text)

    def accept(self, sid: int | None = None) -> str:
        return self.command(f"/accept {sid or ''}")

    def reject(self, sid: int | None = None, reason: str = "") -> str:
        return self.command(f"/reject {sid or ''} {reason}")


# ---------------------------------------------------------------------------
# Module-level default session, so ``learnstat.read_csv`` feels like pandas.

_default: Session | None = None


def get_session() -> Session:
    global _default
    if _default is None:
        _default = Session()
    return _default


def set_session(session: Session) -> Session:
    global _default
    _default = session
    return session
