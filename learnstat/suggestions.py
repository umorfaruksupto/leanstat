"""Suggestion objects and the per-session suggestion history."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Suggestion:
    rule: str  # machine name of the rule that fired
    concept: str  # key into knowledge.CONCEPTS for /describe and /learn
    title: str
    problem: str  # what may be inappropriate about the current approach
    alternative: str  # which function / chart would be more suitable
    rationale: str  # why, for this dataset and objective
    code: str | None = None  # how to implement it
    line: int | None = None
    severity: str = "info"  # info | warning
    columns: list[str] = field(default_factory=list)
    source_code: str = ""  # the user's line that triggered this
    status: str = "pending"  # pending | accepted | rejected
    origin: str = "rules"  # rules | ai
    id: int = 0  # assigned by SuggestionHistory in display order

    def render(self, color: bool = False) -> str:
        tag = "!" if self.severity == "warning" else "i"
        where = f" (line {self.line})" if self.line else ""
        out = [
            f"[{tag}] Suggestion #{self.id}{where}: {self.title}",
            f"    Issue:       {self.problem}",
            f"    Instead:     {self.alternative}",
            f"    Why:         {self.rationale}",
        ]
        if self.code:
            out.append("    How:")
            out += [f"        {ln}" for ln in self.code.splitlines()]
        out.append(f"    -> /describe {self.id} to learn more, /accept {self.id} or /reject {self.id}")
        return "\n".join(out)


class SuggestionHistory:
    def __init__(self) -> None:
        self.items: list[Suggestion] = []

    def add(self, s: Suggestion) -> Suggestion:
        s.id = len(self.items) + 1
        self.items.append(s)
        return s

    def extend(self, items: list[Suggestion]) -> None:
        for s in items:
            self.add(s)

    def get(self, sid: int | None = None) -> Suggestion | None:
        if sid is None:
            return self.items[-1] if self.items else None
        return next((s for s in self.items if s.id == sid), None)

    def latest_pending(self) -> Suggestion | None:
        return next((s for s in reversed(self.items) if s.status == "pending"), None)

    def seen(self, rule: str, source_code: str) -> bool:
        """Avoid repeating the same advice for the same line in a session."""
        return any(s.rule == rule and s.source_code == source_code for s in self.items)
