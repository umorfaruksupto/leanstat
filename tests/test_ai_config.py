import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from learnstat import cli, config
from learnstat.llm import LLMAssistant, _parse_json_object


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNSTAT_CONFIG", str(tmp_path / "config.json"))
    for var in config.ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def server():
    """A tiny OpenAI-compatible endpoint; records requests and replies with ``server.reply``."""
    state = {"reply": "ok", "status": 200, "requests": [], "models": ["gpt-a", "gpt-b", "text-embedding-3-small"],
             "require_key": None}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append({"path": self.path, "auth": self.headers.get("Authorization"), "body": None})
            bad = state["status"] != 200 or (state["require_key"] and self.headers.get("Authorization") != f"Bearer {state['require_key']}")
            code = 401 if bad else 200
            payload = {"error": {"message": "bad key"}} if bad else {"data": [{"id": m} for m in state["models"]]}
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
            payload = (
                {"choices": [{"message": {"content": state["reply"]}}]}
                if state["status"] == 200
                else {"error": {"message": "bad key"}}
            )
            data = json.dumps(payload).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{httpd.server_port}/v1"
    yield state
    httpd.shutdown()


def test_defaults_and_precedence(monkeypatch):
    cfg = config.resolve()
    assert (cfg.base_url, cfg.model, cfg.api_key) == (config.DEFAULT_BASE_URL, config.DEFAULT_MODEL, "")
    config.set_value("model", "from-file")
    assert config.resolve().model == "from-file"
    monkeypatch.setenv("LEARNSTAT_MODEL", "from-env")
    assert config.resolve().model == "from-env"
    config.unset_value("model")
    monkeypatch.delenv("LEARNSTAT_MODEL")
    assert config.resolve().model == config.DEFAULT_MODEL


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        config.set_value("anthropic_key", "x")


def test_not_available_without_key_on_default_endpoint():
    a = LLMAssistant()
    assert not a.available()
    assert "learnstat setup" in a.last_error


def test_cli_config_roundtrip(capsys):
    assert cli.main(["config", "set", "api_key", "sk-secret-123456789"]) == 0
    assert cli.main(["config", "show"]) == 0
    out = capsys.readouterr().out
    assert "sk-secret-123456789" not in out and "[saved]" in out


def test_chat_roundtrip_with_custom_endpoint(server):
    config.set_value("base_url", server["url"])
    config.set_value("model", "my-model")
    a = LLMAssistant()
    assert a.available()  # custom endpoint: key optional
    server["reply"] = "hello"
    assert a.ask("q?", "profile", "code") == "hello"
    req = server["requests"][0]
    assert req["path"] == "/v1/chat/completions"
    assert req["body"]["model"] == "my-model"
    assert req["auth"] is None


def test_api_key_sent_as_bearer(server):
    config.set_value("base_url", server["url"])
    config.set_value("api_key", "sk-test")
    LLMAssistant().test()
    assert server["requests"][0]["auth"] == "Bearer sk-test"


def test_json_fix_with_code_fence(server):
    config.set_value("base_url", server["url"])
    server["reply"] = '```json\n{"can_fix": true, "explanation": "e", "fixed_code": "x = 1"}\n```'
    fix = LLMAssistant().propose_fix("x = 1", "err", "cat", "ctx")
    assert fix == {"can_fix": True, "explanation": "e", "fixed_code": "x = 1"}


def test_auth_error_disables(server):
    config.set_value("base_url", server["url"])
    server["status"] = 401
    a = LLMAssistant()
    assert a.ask("q", "p", "c") is None
    assert a.disabled and "API key was rejected" in a.last_error and "bad key" in a.last_error


def test_connection_error_is_graceful():
    config.set_value("base_url", "http://127.0.0.1:9/v1")
    a = LLMAssistant()
    assert a.ask("q", "p", "c") is None
    assert "could not connect" in a.last_error


def test_parse_json_object_tolerates_prose():
    assert _parse_json_object('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert _parse_json_object("not json") is None


# ---------------------------------------------------------------------------
# guided setup


class Script:
    """Feeds scripted answers to the wizard and records everything it prints."""

    def __init__(self, answers, secrets=()):
        self.answers, self.secrets, self.out = list(answers), list(secrets), []

    def io(self):
        from learnstat.setup import IO

        def ask(prompt):
            self.out.append(prompt)
            if not self.answers:
                raise EOFError
            return self.answers.pop(0)

        def secret(prompt):
            self.out.append(prompt)
            if not self.secrets:
                raise EOFError
            return self.secrets.pop(0)

        return IO(ask=ask, secret=secret, say=self.out.append)

    @property
    def text(self):
        return "\n".join(self.out)


def _custom_index():
    from learnstat import providers

    return str(len(providers.all_choices()))  # "Other" is the last menu entry


def test_wizard_happy_path_saves_and_activates(server):
    from learnstat.setup import run_wizard

    # company -> (custom) url -> [hidden key] -> model 2 -> endpoint name
    script = Script([_custom_index(), server["url"], "2", "mine"], secrets=["sk-test-key"])
    assert run_wizard(script.io()) == "mine"
    assert "OK - 3 models available" in script.text
    assert "gpt-b" in script.text and "text-embedding" not in script.text.split("Models:")[1]  # non-chat hidden
    assert "the model answered" in script.text
    cfg = config.resolve()
    assert (cfg.profile, cfg.model, cfg.api_key) == ("mine", "gpt-b", "sk-test-key")
    assert server["requests"][0]["auth"] == "Bearer sk-test-key"


def test_wizard_bad_key_then_good_key(server):
    from learnstat.setup import run_wizard

    server["require_key"] = "good"
    script = Script([_custom_index(), server["url"], "r", "1", "nm"], secrets=["bad", "good"])
    assert run_wizard(script.io()) == "nm"
    assert "the API key was rejected (401)" in script.text
    assert config.resolve().api_key == "good"


def test_wizard_failed_test_can_pick_another_model(server):
    from learnstat.setup import run_wizard

    # chat endpoint returns 404 for the first model; then the user quits
    server["status"] = 404
    script = Script([_custom_index(), server["url"], "1", "q"], secrets=["k"])
    # 404 on /models is not a key problem -> falls back to typed model name flow
    run_wizard(script.io())
    assert config.profiles() == {}


def test_wizard_cancel_changes_nothing(server):
    from learnstat.setup import run_wizard

    script = Script(["q"])
    assert run_wizard(script.io()) is None
    assert config.profiles() == {}


def test_wizard_switch_and_delete_saved_profiles(server):
    from learnstat.setup import run_wizard

    config.save_profile("a", "custom", server["url"], "k1", "gpt-a")
    config.save_profile("b", "custom", server["url"], "k2", "gpt-b")
    assert config.active_name() == "b"
    script = Script(["1"])  # switch to "a" (tested against the fake endpoint)
    assert run_wizard(script.io()) == "a"
    assert config.active_name() == "a"
    script = Script(["d", "2", "y", "q"])
    run_wizard(script.io())
    assert list(config.profiles()) == ["a"]


def test_legacy_flat_config_is_migrated(tmp_path):
    config.config_path().write_text(json.dumps({"api_key": "k", "model": "m"}), encoding="utf-8")
    assert config.active_name() == "custom"
    assert config.resolve().model == "m"


def test_session_connect_ai_uses_profile(server):
    from learnstat.session import Session

    config.save_profile("x", "custom", server["url"], "k", "gpt-a")
    s = Session(ai=True, out=lambda *_: None)
    assert s.assistant is not None and "gpt-a" in s.ai_status
    config.remove_profile("x")
    s.connect_ai()
    assert s.assistant is None and "learnstat setup" in s.ai_status


def test_ai_slash_list(server):
    from learnstat.commands import dispatch
    from learnstat.session import Session

    config.save_profile("x", "custom", server["url"], "k1234567890", "gpt-a")
    s = Session(ai=False, out=lambda *_: None)
    out = dispatch(s, "/ai list")
    assert "x:" in out and "gpt-a" in out and "k1234567890" not in out
