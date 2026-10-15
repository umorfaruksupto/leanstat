"""Presets for the AI companies learnstat can talk to.

Every preset is an OpenAI-compatible ``/chat/completions`` endpoint, so one small
client covers them all. ``custom`` lets the user type any other endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    base_url: str
    needs_key: bool = True
    key_url: str = ""                 # where the user creates an API key
    models: tuple[str, ...] = ()      # suggestions, used only when the live model list can't be fetched
    note: str = ""


PROVIDERS: tuple[Provider, ...] = (
    Provider("openai", "OpenAI", "https://api.openai.com/v1",
             key_url="https://platform.openai.com/api-keys",
             models=("gpt-4o-mini", "gpt-4o")),
    Provider("anthropic", "Anthropic (Claude)", "https://api.anthropic.com/v1",
             key_url="https://console.anthropic.com/settings/keys",
             models=("claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5")),
    Provider("gemini", "Google (Gemini)", "https://generativelanguage.googleapis.com/v1beta/openai",
             key_url="https://aistudio.google.com/apikey",
             models=("gemini-2.5-flash", "gemini-2.5-pro")),
    Provider("xai", "xAI (Grok)", "https://api.x.ai/v1",
             key_url="https://console.x.ai"),
    Provider("mistral", "Mistral AI", "https://api.mistral.ai/v1",
             key_url="https://console.mistral.ai/api-keys",
             models=("mistral-small-latest", "mistral-large-latest")),
    Provider("deepseek", "DeepSeek", "https://api.deepseek.com/v1",
             key_url="https://platform.deepseek.com/api_keys",
             models=("deepseek-chat", "deepseek-reasoner")),
    Provider("groq", "Groq", "https://api.groq.com/openai/v1",
             key_url="https://console.groq.com/keys",
             models=("llama-3.3-70b-versatile",)),
    Provider("together", "Together AI", "https://api.together.xyz/v1",
             key_url="https://api.together.ai/settings/api-keys"),
    Provider("openrouter", "OpenRouter (many models, one key)", "https://openrouter.ai/api/v1",
             key_url="https://openrouter.ai/keys",
             models=("openai/gpt-4o-mini",)),
    Provider("ollama", "Ollama (local, no key)", "http://localhost:11434/v1", needs_key=False,
             note="Start Ollama first and pull a model, e.g. `ollama pull llama3.2`."),
    Provider("lmstudio", "LM Studio (local, no key)", "http://localhost:1234/v1", needs_key=False,
             note="Start LM Studio's local server and load a model first."),
)

CUSTOM = Provider("custom", "Other (any OpenAI-compatible endpoint)", "", needs_key=True)

_BY_ID = {p.id: p for p in PROVIDERS}


def get(provider_id: str | None) -> Provider | None:
    if provider_id == "custom":
        return CUSTOM
    return _BY_ID.get(provider_id or "")


def all_choices() -> tuple[Provider, ...]:
    """The providers in menu order, with the custom option last."""
    return PROVIDERS + (CUSTOM,)


# Models that are not chat models; hidden from the model menu (but still usable if typed).
_NON_CHAT = ("embed", "whisper", "tts", "dall-e", "moderation", "transcribe", "audio", "realtime",
             "image", "imagen", "babbage", "davinci", "rerank", "guard", "veo", "aqa")


def is_chat_model(model_id: str) -> bool:
    low = model_id.lower()
    return not any(tag in low for tag in _NON_CHAT)
