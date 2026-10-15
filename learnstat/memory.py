"""Persistent Markdown memory for each dataset.

Two files are kept per dataset in the memory directory (``.learnstat/`` by
default):

* ``<name>.md``   - the human-readable profile the user can read and edit.
* ``<name>.json`` - a machine-readable cache (profile + fingerprint +
  decisions) so the dataset is only re-profiled when its content changes.

User-written notes and analytical decisions survive re-profiling.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .profile import DatasetProfile, fingerprint, profile_dataframe


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "dataset"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class DatasetMemory:
    profile: DatasetProfile
    decisions: list[dict[str, str]] = field(default_factory=list)
    notes: list[dict[str, str]] = field(default_factory=list)
    reused: bool = False  # True when loaded from cache without re-profiling

    def to_dict(self) -> dict:
        return {"profile": self.profile.to_dict(), "decisions": self.decisions, "notes": self.notes}


class MemoryStore:
    def __init__(self, directory: str | Path = ".learnstat"):
        self.directory = Path(directory)

    # -- paths ------------------------------------------------------------
    def md_path(self, name: str) -> Path:
        return self.directory / f"{_slug(name)}.md"

    def json_path(self, name: str) -> Path:
        return self.directory / f"{_slug(name)}.json"

    # -- load / save ------------------------------------------------------
    def load(self, name: str) -> DatasetMemory | None:
        path = self.json_path(name)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return DatasetMemory(
                profile=DatasetProfile.from_dict(data["profile"]),
                decisions=data.get("decisions", []),
                notes=data.get("notes", []),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return None  # corrupt cache: fall back to re-profiling

    def load_all(self) -> list[DatasetMemory]:
        if not self.directory.exists():
            return []
        out = []
        for p in sorted(self.directory.glob("*.json")):
            mem = self.load(p.stem)
            if mem:
                out.append(mem)
        return out

    def save(self, mem: DatasetMemory) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        name = mem.profile.name
        self.json_path(name).write_text(json.dumps(mem.to_dict(), indent=2, default=str), encoding="utf-8")
        md = self.md_path(name)
        md.write_text(render_markdown(mem), encoding="utf-8")
        return md

    def load_or_profile(
        self, df: pd.DataFrame, name: str, source: str | None = None, force: bool = False
    ) -> DatasetMemory:
        """Return the memory for ``df``; re-profile only if the content changed."""
        existing = self.load(name)
        if existing and not force and existing.profile.fingerprint == fingerprint(df):
            existing.reused = True
            return existing
        profile = profile_dataframe(df, name=name, source=source)
        mem = DatasetMemory(profile=profile)
        if existing:  # keep the user's history across data changes
            mem.decisions = existing.decisions
            mem.notes = existing.notes
            if existing.profile.fingerprint != profile.fingerprint:
                mem.notes.append({"at": _now(), "text": "Dataset content changed; profile regenerated."})
        self.save(mem)
        return mem

    # -- updates ----------------------------------------------------------
    def record_decision(self, mem: DatasetMemory, text: str) -> None:
        mem.decisions.append({"at": _now(), "text": text})
        self.save(mem)

    def add_note(self, mem: DatasetMemory, text: str) -> None:
        mem.notes.append({"at": _now(), "text": text})
        self.save(mem)


# ---------------------------------------------------------------------------


def render_markdown(mem: DatasetMemory) -> str:
    p = mem.profile
    lines: list[str] = [
        f"# Dataset profile: {p.name}",
        "",
        f"- Source: `{p.source}`" if p.source else "- Source: in-memory DataFrame",
        f"- Shape: {p.n_rows} rows x {p.n_cols} columns",
        f"- Complete rows (no missing values): {p.n_complete_rows}",
        f"- Duplicate rows: {p.n_duplicate_rows}",
        f"- Fingerprint: `{p.fingerprint}` (profiled {p.created_at})",
        "",
        "## Columns",
        "",
        "| Column | Kind | dtype | Missing | Unique | Likely meaning (guess) |",
        "|---|---|---|---|---|---|",
    ]
    for c in p.columns.values():
        lines.append(
            f"| `{c.name}` | {c.kind} | {c.dtype} | {c.n_missing} ({c.pct_missing}%) | {c.n_unique} | "
            f"{c.inferred_meaning or '-'} |"
        )

    num = [c for c in p.columns.values() if c.kind == "numeric"]
    if num:
        lines += ["", "## Numerical variables", ""]
        for c in num:
            s = c.summary
            lines.append(
                f"- `{c.name}`: mean={s.get('mean')}, median={s.get('median')}, sd={s.get('std')}, "
                f"range=[{s.get('min')}, {s.get('max')}], skew={s.get('skew')}"
            )
            lines += [f"  - {n}" for n in c.notes]

    cat = [c for c in p.columns.values() if c.kind in ("categorical", "boolean")]
    if cat:
        lines += ["", "## Categorical variables", ""]
        for c in cat:
            top = c.summary.get("top_values")
            top_s = ", ".join(f"{k} ({v})" for k, v in top.items()) if top else ""
            lines.append(f"- `{c.name}`: {c.n_unique} levels" + (f"; most common: {top_s}" if top_s else ""))
            lines += [f"  - {n}" for n in c.notes]

    other = [c for c in p.columns.values() if c.kind in ("datetime", "text", "identifier")]
    if other:
        lines += ["", "## Other columns", ""]
        for c in other:
            lines.append(f"- `{c.name}` ({c.kind})")
            lines += [f"  - {n}" for n in c.notes]

    if p.correlations:
        lines += ["", "## Notable relationships (|Pearson r| >= 0.5)", ""]
        lines += [f"- `{d['a']}` ~ `{d['b']}`: r = {d['r']}" for d in p.correlations[:15]]

    lines += ["", "## Data-quality issues", ""]
    lines += [f"- {i}" for i in p.issues] or ["- None detected by automatic checks."]

    lines += ["", "## Potentially useful analyses and visualisations", ""]
    lines += [f"- {s}" for s in p.suggested_analyses] or ["- (none yet)"]

    lines += [
        "",
        "## Assumptions and limitations",
        "",
        "- Column meanings are inferred from names only; verify them against the data dictionary.",
        "- Outliers are flagged with the 1.5xIQR rule; flagged does not mean wrong.",
        "- Correlations are Pearson on pairwise-complete rows and say nothing about causation.",
        "",
        "## Analytical decisions",
        "",
    ]
    lines += [f"- {d['at']}: {d['text']}" for d in mem.decisions] or ["- (none recorded yet)"]
    lines += ["", "## Notes", ""]
    lines += [f"- {n['at']}: {n['text']}" for n in mem.notes] or ["- (none)"]
    return "\n".join(lines) + "\n"
