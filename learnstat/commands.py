"""Slash commands that operate on the current analytical context."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .knowledge import Concept, describe_concept, find_topic
from .suggestions import Suggestion

if TYPE_CHECKING:
    from .session import Session

HELP = """learnstat commands
  /describe [id]       explain the reasoning behind the latest (or given) suggestion or error
  /alternatives [id]   other ways to reach the same analytical goal, with trade-offs
  /fix                 show the last error diagnosis; retry the failed statement (AI-assisted if available)
  /learn [topic]       a short lesson on the concept behind the latest suggestion (or a topic)
  /memory [show|note <text>|decide <text>|refresh]
                       view or update the dataset's persistent Markdown profile
  /suggestions         list suggestions from this session
  /accept [id]         accept a suggestion (records the decision; offers the code - you paste/run it)
  /reject [id] [why]   keep your approach (the decision is recorded so it is not suggested again)
  /review              ask Claude to review the last code for issues the rules may miss
  /ask <question>      ask a question about your data or code (AI)
  /status              show AI status and loaded datasets
  /help                this help"""


def _split(text: str) -> tuple[str, str]:
    text = text.strip()
    if not text.startswith("/"):
        return "", text
    head, _, rest = text[1:].partition(" ")
    return head.lower(), rest.strip()


def _parse_id(rest: str) -> tuple[int | None, str]:
    parts = rest.split(maxsplit=1)
    if parts and parts[0].lstrip("#").isdigit():
        return int(parts[0].lstrip("#")), parts[1] if len(parts) > 1 else ""
    return None, rest


def _concept_text(c: Concept, sections: tuple[str, ...]) -> str:
    out = [c.title, "=" * len(c.title)]
    if "concept" in sections:
        out += ["", "The statistical idea:", "  " + c.concept]
    if "purpose" in sections:
        out += ["", "What the suggested function does:", "  " + c.purpose]
    if "when" in sections:
        out += ["", "When it is appropriate, and when it is not:", "  " + c.appropriate_when]
    if "alternatives" in sections and c.alternatives:
        out += ["", "Alternatives:"] + [f"  - {a}" for a in c.alternatives]
    if "lesson" in sections and c.lesson:
        out += ["", c.lesson]
    if c.docs:
        out += ["", "Docs: " + ", ".join(c.docs)]
    return "\n".join(out)


def _with_ai(session: "Session", kind: str, s: Suggestion, offline: str) -> str:
    if not session.assistant:
        return offline
    text = session.assistant.explain(kind, s.render(), offline, session.profile_markdown(), s.source_code or session.last_code)
    if text:
        return text
    return offline + f"\n\n(AI explanation unavailable: {session.assistant.last_error}; showing the offline explanation.)"


def _target(session: "Session", rest: str) -> tuple[Suggestion | None, str]:
    sid, rest = _parse_id(rest)
    s = session.history.get(sid)
    if sid is not None and s is None:
        return None, f"No suggestion #{sid}. Use /suggestions to list them."
    return s, rest


# ---------------------------------------------------------------------------


def cmd_describe(session: "Session", rest: str) -> str:
    s, err = _target(session, rest)
    result = session.last_result
    if s is None and not rest and result and result.attempts:
        return _describe_error(session)
    if s is None:
        return err or "Nothing to describe yet - write some code first, or try /learn <topic>."
    facts = (
        f"Suggestion #{s.id}: {s.title}\n\n"
        f"In your code:  {s.source_code or '(see above)'}\n"
        f"For your data: {s.problem}\n"
        f"Suggested:     {s.alternative}\n"
        f"Reason:        {s.rationale}\n"
    )
    c = describe_concept(s.concept)
    offline = facts + ("\n" + _concept_text(c, ("concept", "purpose", "when")) if c else "")
    return _with_ai(session, "describe", s, offline)


def _describe_error(session: "Session") -> str:
    r = session.last_result
    a = r.attempts[-1]
    d = a.diagnosis
    out = [
        f"Last error: {d.exc_type} ({d.label})",
        f"  Message:     {d.message}",
        f"  Statement:   {d.statement.strip()}",
        f"  Cause:       {d.explanation}",
    ]
    if a.fix:
        out.append(f"  Correction:  {a.fix.description} ({'applied' if a.applied else 'not applied'})")
    out.append(f"  Next step:   {d.next_step}")
    out.append(
        "\nlearnstat separates errors into syntax, name, attribute, argument, column, data-quality, "
        "statistical-method and environment errors. Only mechanical mistakes are corrected automatically; "
        "data-quality and statistical problems are explained so that you make the decision."
    )
    return "\n".join(out)


def cmd_alternatives(session: "Session", rest: str) -> str:
    s, err = _target(session, rest)
    if s is None:
        return err or "No suggestion to give alternatives for yet."
    c = describe_concept(s.concept)
    offline = (
        f"Alternatives for suggestion #{s.id} ({s.title}):\n"
        f"  - {s.alternative}\n" + ("\n".join(f"  - {a}" for a in c.alternatives) if c else "")
    )
    return _with_ai(session, "alternatives", s, offline)


def cmd_learn(session: "Session", rest: str) -> str:
    if rest:
        c = find_topic(rest)
        if session.assistant:
            text = session.assistant.ask(
                f"Teach me about: {rest}. Keep it under 300 words, use my dataset's columns in the examples and end "
                "with one exercise.", session.profile_markdown(), session.last_code)
            if text:
                return text
        return _concept_text(c, ("concept", "purpose", "when", "alternatives", "lesson")) if c else (
            f"No offline lesson matches '{rest}'. Topics: pie charts, skewed data, t-test, correlation, "
            "missing values, categorical plots, dropna, column names, text numbers, histograms."
        )
    s = session.history.get()
    if s is None:
        return "Nothing to learn from yet. Try /learn <topic>, e.g. /learn correlation."
    c = describe_concept(s.concept)
    offline = _concept_text(c, ("concept", "purpose", "when", "alternatives", "lesson")) if c else s.render()
    return _with_ai(session, "learn", s, offline)


def cmd_fix(session: "Session", rest: str) -> str:
    r = session.last_result
    if r is None or (r.ok and not r.attempts):
        return "No failed code in this session."
    if r.ok:
        return "The last run was recovered:\n\n" + r.report()
    report = r.report()
    if r.stopped_at_line is None:
        return report
    if session.confirm is None:
        return report + "\n\n(/fix can retry the failed statement when running interactively.)"
    if not session.confirm(f"Retry from line {r.stopped_at_line}" + (" with AI assistance" if session.assistant else "") + "?"):
        return report
    retry = session.executor().run(r.final_code, start_line=r.stopped_at_line)
    retry.original_code = r.original_code
    session._after_run(r.final_code, retry)
    return "Retry " + ("succeeded." if retry.ok else "did not succeed - see the next step above.")


def cmd_memory(session: "Session", rest: str) -> str:
    mem = session.dataset_memory()
    if mem is None:
        return "No dataset in memory. Load one with learnstat.read_csv(...) (or session.track(df))."
    sub, _, arg = rest.partition(" ")
    path = session.memory.md_path(mem.profile.name)
    if sub in ("", "summary"):
        p = mem.profile
        return (
            f"Dataset '{p.name}' - {p.n_rows} rows x {p.n_cols} columns, profile {p.fingerprint}\n"
            f"Issues: {len(p.issues)}, decisions recorded: {len(mem.decisions)}, notes: {len(mem.notes)}\n"
            f"File: {path}\nUse '/memory show' for the full profile."
        )
    if sub == "show":
        return path.read_text(encoding="utf-8") if path.exists() else "(memory file missing - try /memory refresh)"
    if sub == "note" and arg:
        session.memory.add_note(mem, arg)
        return f"Note saved to {path}."
    if sub == "decide" and arg:
        session.memory.record_decision(mem, arg)
        return f"Decision recorded in {path}."
    if sub == "refresh":
        for var, obj in session.ns.items():
            if session._frame_ids.get(id(obj)) == mem.profile.name:
                session.track(obj, name=mem.profile.name, source=mem.profile.source, force=True)
                return f"Re-profiled '{mem.profile.name}'."
        return "The tracked DataFrame is not in the current namespace; reload it to refresh."
    return "Usage: /memory [show | note <text> | decide <text> | refresh]"


def cmd_suggestions(session: "Session", rest: str) -> str:
    if not session.history.items:
        return "No suggestions yet."
    return "\n".join(
        f"#{s.id} [{s.status}] line {s.line or '-'}: {s.title}" for s in session.history.items
    )


def cmd_accept(session: "Session", rest: str) -> str:
    sid, _ = _parse_id(rest)
    s = session.history.get(sid) if sid else session.history.latest_pending()
    if s is None:
        return "No pending suggestion to accept."
    s.status = "accepted"
    mem = session.dataset_for(s)
    if mem:
        session.memory.record_decision(mem, f"Accepted suggestion '{s.title}': {s.alternative}")
    if s.code and session.on_accept:
        session.on_accept(s.code)
        return f"Accepted #{s.id}. The suggested code was placed in a new cell for you to edit and run."
    return f"Accepted #{s.id}. Your code was not changed - here is the suggested version to adapt:\n\n{s.code or s.alternative}"


def cmd_reject(session: "Session", rest: str) -> str:
    sid, reason = _parse_id(rest)
    s = session.history.get(sid) if sid else session.history.latest_pending()
    if s is None:
        return "No pending suggestion to reject."
    s.status = "rejected"
    mem = session.dataset_for(s)
    if mem:
        why = f" Reason: {reason}" if reason else ""
        session.memory.record_decision(mem, f"Kept original approach despite '{s.title}'.{why}")
    return f"Rejected #{s.id}; your approach stays as written."


def cmd_review(session: "Session", rest: str) -> str:
    if not session.assistant:
        return f"AI review is {session.ai_status}."
    if not session.last_code:
        return "No code to review yet."
    found = session._ai_review(session.last_code, [s for s in session.history.items])
    session.history.extend(found)
    return "\n".join(s.render() for s in found) or "No additional issues found."


def cmd_ask(session: "Session", rest: str) -> str:
    if not rest:
        return "Usage: /ask <question>"
    if not session.assistant:
        return f"/ask needs AI, which is {session.ai_status}. Try /learn {rest}"
    return session.assistant.ask(rest, session.profile_markdown(), session.last_code) or (
        f"AI unavailable: {session.assistant.last_error}"
    )


def cmd_status(session: "Session", rest: str) -> str:
    names = ", ".join(session.datasets) or "none"
    return f"AI: {session.ai_status}\nDatasets: {names}\nMemory directory: {session.memory.directory}"


COMMANDS: dict[str, Callable[["Session", str], str]] = {
    "describe": cmd_describe,
    "alternatives": cmd_alternatives,
    "alt": cmd_alternatives,
    "fix": cmd_fix,
    "learn": cmd_learn,
    "memory": cmd_memory,
    "suggestions": cmd_suggestions,
    "accept": cmd_accept,
    "reject": cmd_reject,
    "review": cmd_review,
    "ask": cmd_ask,
    "status": cmd_status,
    "help": lambda s, r: HELP,
}


def dispatch(session: "Session", text: str) -> str:
    name, rest = _split(text)
    fn = COMMANDS.get(name)
    if fn is None:
        return f"Unknown command '/{name}'.\n\n{HELP}"
    return fn(session, rest)


def is_command(text: str) -> bool:
    name, _ = _split(text)
    return bool(name) and name in COMMANDS


__all__ = ["dispatch", "is_command", "HELP"]
