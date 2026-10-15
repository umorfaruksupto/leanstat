"""Optional AI backend (any OpenAI-compatible chat endpoint).

learnstat works fully without this module: profiling, rule-based advice,
explanations and mechanical error recovery are all offline. When an API key
(or a custom endpoint) is configured - see :mod:`learnstat.config` - the model
is used to

* tailor /describe, /alternatives and /learn explanations to the dataset,
* review code for issues the rules do not cover, and
* propose corrections for errors the rules cannot fix (these always require
  the user's confirmation and pass the same validation as rule-based fixes).

Every call degrades gracefully: on any error the caller gets ``None`` and falls
back to the offline behaviour. Only the standard library is used for HTTP.
"""

from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.request

from . import config as _config

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

TIMEOUT_SECONDS = 120


class LLMAssistant:
    def __init__(self, cfg: _config.AIConfig | None = None):
        self.cfg = cfg or _config.resolve()
        self.last_error: str | None = None
        self.disabled = False  # set after an unrecoverable failure (e.g. bad credentials)

    @property
    def model(self) -> str:
        return self.cfg.model

    # -- availability -----------------------------------------------------
    def available(self) -> bool:
        if self.cfg.needs_key and not self.cfg.api_key:
            self.last_error = "no AI is set up yet (run: learnstat setup)"
            return False
        return True

    # -- low-level --------------------------------------------------------
    def _create(self, prompt: str, max_tokens: int = 4000, json_schema: dict | None = None,
                timeout: float = TIMEOUT_SECONDS) -> str | None:
        if self.disabled:
            return None
        if json_schema:
            prompt += (
                "\n\nReply with ONLY a JSON object (no prose, no code fences) matching this JSON Schema:\n"
                + json.dumps(json_schema)
            )
        body = {
            "model": self.cfg.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            payload = _request(self.cfg.chat_url, self.cfg.headers, body, timeout)
        except AIError as e:
            self.last_error = str(e)
            if e.fatal:
                self.disabled = True
            return None
        try:
            text = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            self.last_error = "unexpected response format - is the base URL an OpenAI-compatible /chat/completions endpoint?"
            return None
        text = text.strip()
        if not text:
            self.last_error = "the model returned an empty response"
            return None
        return text

    def _json(self, prompt: str, schema: dict, max_tokens: int = 4000) -> dict | None:
        text = self._create(prompt, max_tokens=max_tokens, json_schema=schema)
        if not text:
            return None
        data = _parse_json_object(text)
        if data is None:
            self.last_error = "the response was not valid JSON"
        return data

    def test(self, timeout: float = 30) -> str | None:
        """Send a tiny request; returns the reply, or None (see ``last_error``)."""
        self.disabled = False
        return self._create("Reply with the single word: ok", max_tokens=20, timeout=timeout)

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
                },
            }
        },
        "required": ["suggestions"],
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
        if not data or not isinstance(data.get("suggestions"), list):
            return None
        return data["suggestions"]

    FIX_SCHEMA = {
        "type": "object",
        "properties": {
            "can_fix": {"type": "boolean"},
            "explanation": {"type": "string"},
            "fixed_code": {"type": "string"},
        },
        "required": ["can_fix", "explanation", "fixed_code"],
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
        data = self._json(prompt, self.FIX_SCHEMA, max_tokens=4000)
        if not data or not all(k in data for k in ("can_fix", "explanation", "fixed_code")):
            return None
        return data


class AIError(Exception):
    """A failed call, with a message written for the user. ``fatal`` = retrying cannot help (bad key)."""

    def __init__(self, message: str, status: int | None = None, fatal: bool = False):
        super().__init__(message)
        self.status = status
        self.fatal = fatal


def _error_detail(e: urllib.error.HTTPError) -> str:
    try:
        data = json.loads(e.read().decode("utf-8", "replace"))
        err = data.get("error", data) if isinstance(data, dict) else data
        if isinstance(err, list) and err:  # e.g. Gemini wraps errors in a list
            err = err[0].get("error", err[0]) if isinstance(err[0], dict) else err[0]
        return str(err.get("message", err)) if isinstance(err, dict) else str(err)
    except Exception:
        return str(e.reason)


def _friendly(status: int, detail: str) -> str:
    if status in (401, 403):
        return f"the API key was rejected ({status}) - check it is correct, active and allowed to use this model. Provider says: {detail}"
    if status == 404:
        return f"endpoint or model not found (404) - check the base URL and the model name. Provider says: {detail}"
    if status == 429:
        return f"rate limit or quota exceeded (429) - check your plan/billing. Provider says: {detail}"
    if status == 400:
        return f"the provider rejected the request (400): {detail}"
    if status >= 500:
        return f"the provider had a server error ({status}); try again shortly. {detail}"
    return f"API error {status}: {detail}"


def _request(url: str, headers: dict, body: dict | None = None, timeout: float = TIMEOUT_SECONDS) -> dict:
    """POST ``body`` (or GET when None) and return the decoded JSON, raising :class:`AIError`."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise AIError(_friendly(e.code, _error_detail(e)), status=e.code, fatal=e.code in (401, 403)) from None
    except urllib.error.URLError as e:
        reason = e.reason
        if isinstance(reason, socket.timeout):
            raise AIError(f"timed out after {timeout:g}s waiting for {url}") from None
        raise AIError(f"could not connect to {url} ({reason}) - check the address and your internet connection") from None
    except (socket.timeout, TimeoutError):
        raise AIError(f"timed out after {timeout:g}s waiting for {url}") from None
    except ValueError:
        raise AIError(f"{url} did not return JSON - is it an OpenAI-compatible endpoint?") from None
    except OSError as e:
        raise AIError(f"network error: {e}") from None


def fetch_models(cfg: _config.AIConfig, timeout: float = 20) -> list[str]:
    """Ask the provider which models this key can use. Raises :class:`AIError`."""
    payload = _request(cfg.models_url, cfg.headers, None, timeout)
    items = payload.get("data", payload.get("models", [])) if isinstance(payload, dict) else payload
    ids = []
    for item in items or []:
        mid = item.get("id") or item.get("name") if isinstance(item, dict) else item
        if isinstance(mid, str) and mid:
            ids.append(mid[len("models/"):] if mid.startswith("models/") else mid)
    if not ids:
        raise AIError("the provider returned no models for this key")
    return sorted(dict.fromkeys(ids))


def _parse_json_object(text: str) -> dict | None:
    """Parse a JSON object from model output, tolerating code fences and surrounding prose."""
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""):
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None
