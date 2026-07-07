from pathlib import Path

import pytest
import requests

from agent.setup import (
    _find_existing_provider,
    _set_env,
    _test_key,
    interactive_setup,
)


def _ok_response():
    class MockResponse:
        status_code = 200
        text = "ok"
        def json(self):
            return {"models": [{"name": "qwen3:8b-64k"}]}
        def raise_for_status(self):
            pass
    return MockResponse()


def _fail_response(status=401):
    class MockResponse:
        status_code = status
        text = '{"error": "unauthorized"}'
        def json(self):
            return {"error": "invalid"}
        def raise_for_status(self):
            raise requests.HTTPError(response=self)
    return MockResponse()


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for k in (
        "PROVIDER",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OLLAMA_MODEL",
        "TAVILY_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)


# ── _find_existing_provider ─────────────────────────────────────────────────


class TestFindExistingProvider:
    def test_no_env_file(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert _find_existing_provider() is None

    def test_valid_anthropic(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        _set_env("PROVIDER", "anthropic")
        _set_env("ANTHROPIC_API_KEY", "sk-ok")
        _set_env("TAVILY_API_KEY", "tvly-ok")
        assert _find_existing_provider() == "anthropic"

    def test_valid_openai(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        _set_env("PROVIDER", "openai")
        _set_env("OPENAI_API_KEY", "sk-ok")
        _set_env("TAVILY_API_KEY", "tvly-ok")
        assert _find_existing_provider() == "openai"

    def test_missing_tavily_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        _set_env("PROVIDER", "anthropic")
        _set_env("ANTHROPIC_API_KEY", "sk-ok")
        assert _find_existing_provider() is None

    def test_backward_compat_anthropic_key_no_provider(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        _set_env("ANTHROPIC_API_KEY", "sk-legacy")
        _set_env("TAVILY_API_KEY", "tvly-ok")
        assert _find_existing_provider() == "anthropic"


# ── _set_env ────────────────────────────────────────────────────────────────


class TestSetEnv:
    def test_creates_file(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        _set_env("FOO", "bar")
        assert "FOO=bar" in Path(".env").read_text()

    def test_updates_existing_key(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        _set_env("FOO", "bar")
        _set_env("FOO", "baz")
        content = Path(".env").read_text()
        assert "FOO=baz" in content
        assert content.count("FOO") == 1

    def test_preserves_other_keys(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        _set_env("ONE", "1")
        _set_env("TWO", "2")
        content = Path(".env").read_text()
        assert "ONE=1" in content
        assert "TWO=2" in content


# ── interactive_setup ───────────────────────────────────────────────────────


class TestInteractiveSetup:
    def test_skips_when_already_configured(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        _set_env("PROVIDER", "anthropic")
        _set_env("ANTHROPIC_API_KEY", "sk-existing")
        _set_env("TAVILY_API_KEY", "tvly-existing")

        called = False
        def fail_if_called(*_):
            nonlocal called
            called = True

        monkeypatch.setattr("builtins.input", fail_if_called)
        interactive_setup()
        assert not called

    def test_anthropic_full_flow(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        inputs = iter(["1"])
        getpass_values = iter(["sk-valid", "tvly-test"])

        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        monkeypatch.setattr("agent.setup.getpass", lambda _: next(getpass_values))
        monkeypatch.setattr(
            "agent.setup.requests.get", lambda url, **kw: _ok_response()
        )

        interactive_setup()

        env = Path(".env").read_text()
        assert "PROVIDER=anthropic" in env
        assert "ANTHROPIC_API_KEY=sk-valid" in env
        assert "TAVILY_API_KEY=tvly-test" in env

    def test_anthropic_invalid_key_reprompts(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        inputs = iter(["1"])
        getpass_values = iter(["bad-key", "good-key", "tvly-test"])

        call_count = 0

        def mock_get(url, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _fail_response(401)
            return _ok_response()

        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        monkeypatch.setattr("agent.setup.getpass", lambda _: next(getpass_values))
        monkeypatch.setattr("agent.setup.requests.get", mock_get)

        interactive_setup()

        env = Path(".env").read_text()
        assert "ANTHROPIC_API_KEY=good-key" in env
        assert call_count == 2

    def test_ollama_full_flow(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        inputs = iter(["3", "qwen3:8b-64k"])
        getpass_values = iter(["tvly-test"])

        def mock_get(url, **kw):
            if "localhost:11434" in url:
                return _ok_response()
            return _ok_response()

        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        monkeypatch.setattr("agent.setup.getpass", lambda _: next(getpass_values))
        monkeypatch.setattr("agent.setup.requests.get", mock_get)

        interactive_setup()

        env = Path(".env").read_text()
        assert "PROVIDER=ollama" in env
        assert "OLLAMA_MODEL=qwen3:8b-64k" in env
        assert "TAVILY_API_KEY=tvly-test" in env

    def test_reconfigure_reruns_setup(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        _set_env("PROVIDER", "anthropic")
        _set_env("ANTHROPIC_API_KEY", "sk-old")
        _set_env("TAVILY_API_KEY", "tvly-old")

        inputs = iter(["3", "llama3:70b"])
        getpass_values = iter(["tvly-new"])

        def mock_get(url, **kw):
            class R:
                status_code = 200
                text = "ok"
                def json(_self):
                    return {"models": [{"name": "llama3:70b"}]}
                def raise_for_status(_self):
                    pass
            return R()

        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        monkeypatch.setattr("agent.setup.getpass", lambda _: next(getpass_values))
        monkeypatch.setattr("agent.setup.requests.get", mock_get)

        interactive_setup(reconfigure=True)

        env = Path(".env").read_text()
        assert "PROVIDER=ollama" in env
        assert "OLLAMA_MODEL=llama3:70b" in env
        assert "TAVILY_API_KEY=tvly-new" in env


# ── _test_key ───────────────────────────────────────────────────────────────


class TestKeyValidation:
    def test_anthropic_valid(self, monkeypatch):
        monkeypatch.setattr(
            "agent.setup.requests.get",
            lambda url, **kw: _ok_response(),
        )
        ok, msg = _test_key("anthropic", "sk-good")
        assert ok
        assert msg == ""

    def test_anthropic_invalid(self, monkeypatch):
        monkeypatch.setattr(
            "agent.setup.requests.get",
            lambda url, **kw: _fail_response(401),
        )
        ok, msg = _test_key("anthropic", "sk-bad")
        assert not ok
        assert "401" in msg

    def test_anthropic_network_error(self, monkeypatch):
        def raise_error(*args, **kw):
            raise requests.ConnectionError("DNS failure")

        monkeypatch.setattr("agent.setup.requests.get", raise_error)
        ok, msg = _test_key("anthropic", "sk-any")
        assert not ok
        assert "Could not reach the API" in msg

    def test_openai_valid(self, monkeypatch):
        monkeypatch.setattr(
            "agent.setup.requests.get",
            lambda url, **kw: _ok_response(),
        )
        ok, msg = _test_key("openai", "sk-good")
        assert ok
        assert msg == ""
