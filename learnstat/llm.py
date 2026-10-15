"""Optional Claude backend.

learnstat works fully without this module: profiling, rule-based advice,
explanations and mechanical error recovery are all offline. When the
``anthropic`` package is installed and credentials are available, Claude is
used to

* tailor /describe, /alternatives and /learn explanations to the dataset,
* review code for issues the rules do not cover, and
* propose corrections for errors the rules cannot fix (these always require
  the user's confirmation and pass the same validation as rule-based fixes).

Every call degrades gracefully: on any error the caller gets ``None`` and falls
back to the offline behaviour.
"""

from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You are learnstat, a statistics and Python mentor embedded in a learning library.
The user writes and owns their analytical code. Your job is to help them learn: explain reasoning,
point out statistical and visualisation pitfalls, and recommend standard pandas / NumPy / SciPy /
Matplotlib functions with their real names, so the skills transfer to the wider ecosystem.

Principles:
- Ground every statement in the dataset profile and code you are given; say when something is a guess.
- Never silently change the user's data, assumptions, statistical method or objective. When a change
  would do that, present it as an option and explain the trade-off.
- Prefer clear explanations over jargon; name the statistical concept, its purpose, when it is
  appropriate and when an alternative is better.
- Keep answers concise and practical, with short code snippets using standard library APIs."""


class ClaudeAssistant:
    def __init__(self, model: str | None = None, effort: str = "medium"):
        self.model = model or os.environ.get("LEARNSTAT_MODEL", DEFAULT_MODEL)
        self.effort = effort
        self._client = None
        self.last_error: str | None = None
        self.disabled = False  # set after an unrecoverable failure (e.g. bad credentials)

    # -- availability -----------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            import anthropic  # optional dependency

            self._client = anthropic.Anthropic()
        return self._client

    def available(self) -> bool:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            self.last_error = "the `anthropic` package is not installed (pip install 'learnstat[ai]')"
            return False
        try:
            _ = self.client
        except Exception as e:  # missing credentials etc.
            self.last_error = f"could not create a Claude client: {e}"
            return False
        return True

    # -- low-level --------------------------------------------------------
    def _create(self, prompt: str, max_tokens: int = 4000, schema: dict | None = None) -> str | None:
        import anthropic

        if self.disabled:
            return None
        params: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.effort},
            # If a safety classifier declines, let the API retry on its recommended fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if schema:
            params["output_config"]["format"] = {"type": "json_schema", "schema": schema}
        try:
            response = self.client.beta.messages.create(**params)
        except anthropic.AuthenticationError as e:
            self.last_error = f"authentication failed: {e}"
            self.disabled = True
            return None
        except anthropic.RateLimitError as e:
            self.last_error = f"rate limited: {e}"
            return None
        except anthropic.APIStatusError as e:
            self.last_error = f"API error {e.status_code}: {e}"
            return None
        except anthropic.APIConnectionError as e:
            self.last_error = f"connection error: {e}"
            return None
        except Exception as e:  # e.g. no credentials configured - never break the user's session
            self.last_error = f"{type(e).__name__}: {e}"
            self.disabled = True
            return None
        if response.stop_reason == "refusal":
            self.last_error = "the request was declined"
            return None
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return text or None

    def _json(self, prompt: str, schema: dict, max_tokens: int = 4000) -> dict | None:
        text = self._create(prompt, max_tokens=max_tokens, schema=schema)
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            self.last_error = "the response was not valid JSON"
            return None

    # -- features ---------------------------------------------------------
    def explain(self, kind: str, suggestion_text: str, grounding: str, profile_md: str, code: str) -> str | None:
        task = {
            "describe": "Explain the reasoning behind this suggestion: the statistical concept, the purpose of the "
                        "suggested function, why it suits THIS dataset and objective, and when another method "
                        "would be preferable.",
            "alternatives": "List 3-5 alternative approaches to the same analytical goal, each with a one-line "
                            "trade-off and a short snippet, ordered from most to least recommended for this dataset.",
            "learn": "Write a short lesson (under 300 words) on the concept behind this suggestion, using the user's "
                     "own columns in the examples, ending with one hands-on exercise on their data.",
        }[kind]
        prompt = (
            f"{task}\n\n<suggestion>\n{suggestion_text}\n</suggestion>\n\n"
            f"<reference_explanation>\n{grounding}\n</reference_explanation>\n\n"
            f"<dataset_profile>\n{profile_md}\n</dataset_profile>\n\n<user_code>\n{code}\n</user_code>"
        )
        return self._create(prompt, max_tokens=4000)

    def ask(self, question: str, profile_md: str, code: str) -> str | None:
        prompt = (
            f"<dataset_profile>\n{profile_md}\n</dataset_profile>\n\n<recent_code>\n{code}\n</recent_code>\n\n"
            f"Question from the user: {question}"
        )
        return self._create(prompt, max_tokens=4000)

    REVIEW_SCHEMA = {
        "type": "object",
        "properties": {
            "suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "line": {"type": "integer"},
                        "title": {"type": "string"},
                        "problem": {"type": "string"},
                        "alternative": {"type": "string"},
                        "rationale": {"type": "string"},
                        "code": {"type": "string"},
                    },
                    "required": ["line", "title", "problem", "alternative", "rationale", "code"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["suggestions"],
        "additionalProperties": False,
    }

    def review(self, code: str, profile_md: str, already: list[str]) -> list[dict] | None:
        prompt = (
            "Review the user's analysis code against the dataset profile. Report only concrete, important "
            "statistical or visualisation issues (wrong chart for the data, violated test assumptions, misleading "
            "summaries, data-quality traps). Do not repeat these already-reported issues: "
            f"{already or 'none'}. Return an empty list if nothing important remains. Keep each field to one or "
            "two sentences; `code` is a short replacement snippet using standard pandas/SciPy/Matplotlib APIs.\n\n"
            f"<dataset_profile>\n{profile_md}\n</dataset_profile>\n\n<user_code>\n{code}\n</user_code>"
        )
        data = self._json(prompt, self.REVIEW_SCHEMA)
        return data["suggestions"] if data else None

    FIX_SCHEMA = {
        "type": "object",
        "properties": {
            "can_fix": {"type": "boolean"},
            "explanation": {"type": "string"},
            "fixed_code": {"type": "string"},
        },
        "required": ["can_fix", "explanation", "fixed_code"],
        "additionalProperties": False,
    }

    def propose_fix(self, statement: str, error: str, category: str, context: str) -> dict | None:
        prompt = (
            "A statement in the user's analysis raised an error. Propose the smallest correction to THIS statement "
            "only. Hard constraints: do not change the dataset (no dropna/fillna/astype/replace/filters), do not "
            "change the statistical method or its options, do not change which variables are assigned, do not add "
            "file, network or shell operations. If the error can only be solved by one of those, set can_fix=false "
            "and explain what the user should decide. `fixed_code` must be the complete corrected statement.\n\n"
            f"<error category=\"{category}\">\n{error}\n</error>\n\n<statement>\n{statement}\n</statement>\n\n"
            f"<context>\n{context}\n</context>"
        )
        return self._json(prompt, self.FIX_SCHEMA, max_tokens=4000)
