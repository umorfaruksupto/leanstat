"""learnstat's own AI settings: saved profiles of provider, API key, endpoint and model.

Run ``learnstat setup`` for a guided choice of company, key and model. Several
profiles can be saved (one per company / key) and switched between::

    learnstat setup                  # guided wizard: company -> key -> model -> check -> start
    learnstat config list            # saved profiles
    learnstat config use anthropic   # switch the active profile
    learnstat config set model gpt-4o   # edit the active profile

Environment variables (``LEARNSTAT_API_KEY``, ``LEARNSTAT_BASE_URL``,
``LEARNSTAT_MODEL``) override the active profile for one run.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from . import providers

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

# setting name -> environment variable
ENV_VARS = {
    "api_key": "LEARNSTAT_API_KEY",
    "base_url": "LEARNSTAT_BASE_URL",
    "model": "LEARNSTAT_MODEL",
}
KEYS = tuple(ENV_VARS)
_PROFILE_FIELDS = ("provider", "base_url", "api_key", "model")


def config_path() -> Path:
    """Location of the config file (``LEARNSTAT_CONFIG`` overrides the default)."""
    override = os.environ.get("LEARNSTAT_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".learnstat" / "config.json"


# ---------------------------------------------------------------------------
# storage


def _read() -> dict:
    """Load the file as ``{"active": str|None, "profiles": {...}, "skip_setup": bool}``."""
    try:
        raw = json.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    profiles = {
        name: {k: str(v) for k, v in p.items() if k in _PROFILE_FIELDS and v not in (None, "")}
        for name, p in (raw.get("profiles") or {}).items()
        if isinstance(p, dict)
    }
    active = raw.get("active") if raw.get("active") in profiles else None
    # an older flat file ({"api_key": ..., "model": ...}) becomes a "custom" profile
    flat = {k: str(raw[k]) for k in KEYS if raw.get(k)}
    if flat and not profiles:
        profiles["custom"] = {**flat, "provider": "custom"}
        active = "custom"
    if active is None and profiles:
        active = next(iter(profiles))
    return {"active": active, "profiles": profiles, "skip_setup": bool(raw.get("skip_setup"))}


def _write(data: dict) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:  # the file holds API keys: keep it private where the OS supports it
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return path


def profiles() -> dict[str, dict]:
    return _read()["profiles"]


def active_name() -> str | None:
    return _read()["active"]


def save_profile(name: str, provider: str, base_url: str, api_key: str, model: str, activate: bool = True) -> Path:
    data = _read()
    data["profiles"][name] = {k: v for k, v in
                              {"provider": provider, "base_url": base_url, "api_key": api_key, "model": model}.items() if v}
    if activate or data["active"] is None:
        data["active"] = name
    return _write(data)


def set_active(name: str) -> Path:
    data = _read()
    if name not in data["profiles"]:
        raise KeyError(f"no saved profile {name!r}; saved: {', '.join(data['profiles']) or 'none'}")
    data["active"] = name
    return _write(data)


def remove_profile(name: str) -> Path:
    data = _read()
    if name not in data["profiles"]:
        raise KeyError(f"no saved profile {name!r}; saved: {', '.join(data['profiles']) or 'none'}")
    del data["profiles"][name]
    if data["active"] == name:
        data["active"] = next(iter(data["profiles"]), None)
    return _write(data)


def set_skip_setup(skip: bool) -> Path:
    data = _read()
    data["skip_setup"] = skip
    return _write(data)


def skip_setup() -> bool:
    return _read()["skip_setup"]


def set_value(key: str, value: str) -> Path:
    """Edit one setting of the active profile (a ``custom`` profile is created if none exists)."""
    if key not in KEYS:
        raise KeyError(f"unknown setting {key!r}; choose from {', '.join(KEYS)}")
    data = _read()
    if data["active"] is None:
        data["profiles"]["custom"] = {"provider": "custom"}
        data["active"] = "custom"
    data["profiles"][data["active"]][key] = value.strip()
    return _write(data)


def unset_value(key: str) -> Path:
    if key not in KEYS:
        raise KeyError(f"unknown setting {key!r}; choose from {', '.join(KEYS)}")
    data = _read()
    if data["active"] is not None:
        data["profiles"][data["active"]].pop(key, None)
    return _write(data)


# ---------------------------------------------------------------------------
# resolved settings


@dataclass
class AIConfig:
    api_key: str
    base_url: str
    model: str
    provider: str = "custom"
    profile: str | None = None

    @property
    def chat_url(self) -> str:
        base = self.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else base + "/chat/completions"

    @property
    def models_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        return base + "/models"

    @property
    def provider_name(self) -> str:
        p = providers.get(self.provider)
        return p.name.split(" (")[0] if p and p.id != "custom" else (self.profile or "custom endpoint")

    @property
    def needs_key(self) -> bool:
        """Hosted endpoints need a key; local ones (Ollama, LM Studio) and custom URLs may not."""
        p = providers.get(self.provider)
        if p is not None and p.id != "custom":
            return p.needs_key
        return self.base_url.rstrip("/") == DEFAULT_BASE_URL

    @property
    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
            if self.provider == "anthropic":  # its native auth headers (needed for /models)
                h["x-api-key"] = self.api_key
                h["anthropic-version"] = "2023-06-01"
        return h

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) or not self.needs_key


def resolve() -> AIConfig:
    data = _read()
    prof = data["profiles"].get(data["active"], {}) if data["active"] else {}

    def pick(key: str, default: str = "") -> str:
        return os.environ.get(ENV_VARS[key], "").strip() or prof.get(key, "") or default

    preset = providers.get(prof.get("provider"))
    return AIConfig(
        api_key=pick("api_key"),
        base_url=pick("base_url", preset.base_url if preset and preset.base_url else DEFAULT_BASE_URL),
        model=pick("model", DEFAULT_MODEL),
        provider=prof.get("provider", "custom"),
        profile=data["active"],
    )


def mask(secret: str) -> str:
    if not secret:
        return "(not set)"
    return secret[:3] + "..." + secret[-4:] if len(secret) > 10 else "***"
