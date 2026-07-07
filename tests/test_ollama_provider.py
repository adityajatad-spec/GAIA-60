from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import requests

from agent.providers.base import ProviderResponse
from agent.providers.ollama_provider import (
    OllamaProvider,
    _to_openai_tool_calls,
    _to_openai_tools,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture
def provider() -> OllamaProvider:
    return OllamaProvider(model="test-model", host="http://localhost:11434")


def _fake_choice(
    content: str | None,
    tool_calls: list | None = None,
    finish_reason: str = "stop",
) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls or []
    choice.finish_reason = finish_reason
    return choice


# ── Readiness check ─────────────────────────────────────────────────────────


class TestReadiness:
    def test_ensure_ready_detects_running_server(self, provider):
        with patch("agent.providers.ollama_provider.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "models": [{"name": "test-model"}]
            }
            provider._ensure_ready()
            assert provider._checked is True

    def test_ensure_ready_raises_on_no_models(self, provider):
        with patch("agent.providers.ollama_provider.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"models": []}

            with pytest.raises(
                (ValueError, RuntimeError), match="not available"
            ):
                provider._ensure_ready()

    def test_ensure_ready_raises_on_connection_refused(self, provider):
        with patch("agent.providers.ollama_provider.requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("Connection refused")

            with pytest.raises(
                (ConnectionError, RuntimeError), match="not reachable"
            ):
                provider._ensure_ready()


# ── Tool-call parsing ────────────────────────────────────────────────────────


class TestToolCallParsing:
    def test_valid_tool_call(self, provider):
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "search"
        tc.function.arguments = '{"query": "capital of France"}'

        with (
            patch.object(provider, "_ensure_ready"),
            patch.object(
                provider._client.chat.completions, "create"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock(
                choices=[_fake_choice("Let me search.", [tc])]
            )

            resp = provider.call(
                messages=[{"role": "user", "content": "test"}],
                tools=[{"name": "search", "description": "...", "input_schema": {}}],
            )

        assert isinstance(resp, ProviderResponse)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[0].input == {"query": "capital of France"}
        assert resp.tool_calls[0].id == "call_1"

    def test_malformed_json_yields_tool_call_parse_error(self, provider):
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "search"
        tc.function.arguments = "{bad json}"

        with (
            patch.object(provider, "_ensure_ready"),
            patch.object(
                provider._client.chat.completions, "create"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock(
                choices=[_fake_choice("Let me search.", [tc])]
            )

            resp = provider.call(
                messages=[{"role": "user", "content": "test"}],
                tools=[{"name": "search", "description": "...", "input_schema": {}}],
            )

        assert resp.stop_reason == "tool_call_parse_error"
        assert len(resp.tool_calls) == 0

    def test_empty_tool_calls(self, provider):
        with (
            patch.object(provider, "_ensure_ready"),
            patch.object(
                provider._client.chat.completions, "create"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock(
                choices=[_fake_choice("Hello, world!")]
            )

            resp = provider.call(
                messages=[{"role": "user", "content": "test"}],
                tools=None,
            )

        assert len(resp.tool_calls) == 0
        assert resp.text == "Hello, world!"

    def test_multiple_tool_calls(self, provider):
        tc1 = MagicMock()
        tc1.id = "call_1"
        tc1.function.name = "search"
        tc1.function.arguments = '{"query": "first"}'

        tc2 = MagicMock()
        tc2.id = "call_2"
        tc2.function.name = "browse"
        tc2.function.arguments = '{"url": "http://example.com"}'

        with (
            patch.object(provider, "_ensure_ready"),
            patch.object(
                provider._client.chat.completions, "create"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock(
                choices=[_fake_choice("Both.", [tc1, tc2])]
            )

            resp = provider.call(
                messages=[{"role": "user", "content": "test"}],
                tools=[
                    {"name": "search", "description": "...", "input_schema": {}},
                    {"name": "browse", "description": "...", "input_schema": {}},
                ],
            )

        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[1].name == "browse"

    def test_partial_json_failure_one_valid_one_invalid(self, provider):
        tc1 = MagicMock()
        tc1.id = "call_1"
        tc1.function.name = "search"
        tc1.function.arguments = '{"query": "good"}'

        tc2 = MagicMock()
        tc2.id = "call_2"
        tc2.function.name = "browse"
        tc2.function.arguments = "{not valid}"

        with (
            patch.object(provider, "_ensure_ready"),
            patch.object(
                provider._client.chat.completions, "create"
            ) as mock_create,
        ):
            mock_create.return_value = MagicMock(
                choices=[_fake_choice("Both.", [tc1, tc2])]
            )

            resp = provider.call(
                messages=[{"role": "user", "content": "test"}],
                tools=[
                    {"name": "search", "description": "...", "input_schema": {}},
                    {"name": "browse", "description": "...", "input_schema": {}},
                ],
            )

        assert resp.stop_reason == "tool_call_parse_error"
        assert len(resp.tool_calls) == 0


# ── Tool-definition conversion (Anthropic → OpenAI) ─────────────────────────


class TestToolConversion:
    def test_converts_anthropic_style_tools(self):
        anthropic_tools = [
            {
                "name": "search",
                "description": "Search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]

        openai_tools = _to_openai_tools(anthropic_tools)

        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "search"
        assert openai_tools[0]["function"]["description"] == "Search the web"
        assert openai_tools[0]["function"]["parameters"]["type"] == "object"

    def test_handles_none(self):
        assert _to_openai_tools(None) is None

    def test_handles_empty_list(self):
        assert _to_openai_tools([]) is None


# ── Tool-call format conversion (canonical → OpenAI) ─────────────────────────


class TestToolCallFormatConversion:
    def test_converts_single_call(self):
        canonical = [
            {"id": "call_abc", "name": "search", "input": {"query": "Paris"}}
        ]
        result = _to_openai_tool_calls(canonical)
        assert len(result) == 1
        assert result[0]["id"] == "call_abc"
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"
        assert result[0]["function"]["arguments"] == '{"query": "Paris"}'

    def test_converts_multiple_calls(self):
        canonical = [
            {"id": "call_1", "name": "search", "input": {"query": "first"}},
            {"id": "call_2", "name": "browse", "input": {"url": "http://x.com"}},
        ]
        result = _to_openai_tool_calls(canonical)
        assert len(result) == 2
        assert result[1]["function"]["name"] == "browse"
        assert result[1]["function"]["arguments"] == '{"url": "http://x.com"}'

    def test_arguments_are_json_strings(self):
        canonical = [
            {
                "id": "call_1",
                "name": "get_weather",
                "input": {"city": "Paris", "unit": "celsius"},
            }
        ]
        result = _to_openai_tool_calls(canonical)
        args = result[0]["function"]["arguments"]
        assert isinstance(args, str)
        import json
        assert json.loads(args) == {"city": "Paris", "unit": "celsius"}


# ── Two-turn tool-call follow-up flow ─────────────────────────────────────────
#
# This tests the exact scenario that was failing: after the model returns
# tool_calls and their results are fed back, the follow-up request must
# format the assistant's tool_calls in the OpenAI wire format.


class TestTwoTurnFlow:
    def test_follow_up_has_correct_openai_tool_call_format(self, provider):
        """After a tool call → tool result cycle, the second request must
        format the assistant's tool_calls in OpenAI format (id, type,
        function.name, function.arguments as JSON string)."""
        tc1 = MagicMock()
        tc1.id = "call_search_1"
        tc1.function.name = "search"
        tc1.function.arguments = '{"query": "capital of France"}'

        with (
            patch.object(provider, "_ensure_ready"),
            patch.object(
                provider._client.chat.completions, "create"
            ) as mock_create,
        ):
            # ── Turn 1: model responds with a tool call ──────────────
            mock_create.return_value = MagicMock(
                choices=[_fake_choice("Let me search.", [tc1])]
            )

            resp1 = provider.call(
                messages=[{"role": "user", "content": "What is the capital of France?"}],
                tools=[{"name": "search", "description": "Search the web", "input_schema": {}}],
            )

            assert len(resp1.tool_calls) == 1
            tc_id = resp1.tool_calls[0].id

            # ── Simulate what core.py does: build follow-up msgs ─────
            follow_up_messages = [
                {"role": "user", "content": "What is the capital of France?"},
                {
                    "role": "assistant",
                    "content": "Let me search.",
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "input": tc.input}
                        for tc in resp1.tool_calls
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": "Paris is the capital of France.",
                },
            ]

            # ── Turn 2: feed tool result back to the model ───────────
            mock_create.reset_mock()
            mock_create.return_value = MagicMock(
                choices=[_fake_choice("FINAL ANSWER: Paris")]
            )

            resp2 = provider.call(
                messages=follow_up_messages,
                tools=[{"name": "search", "description": "Search the web", "input_schema": {}}],
            )

            assert resp2.text == "FINAL ANSWER: Paris"

            # ── Verify the API payload for turn 2 ────────────────────
            assert mock_create.call_count == 1
            call_kwargs = mock_create.call_args.kwargs
            sent_messages = call_kwargs["messages"]

            # Find the assistant message with tool_calls
            assistant_msgs = [
                m for m in sent_messages
                if m["role"] == "assistant" and "tool_calls" in m
            ]
            assert len(assistant_msgs) == 1, (
                "Expected exactly one assistant message with tool_calls "
                f"in the payload, got {len(assistant_msgs)}"
            )

            am = assistant_msgs[0]
            tcs = am["tool_calls"]
            assert len(tcs) == 1
            assert tcs[0]["id"] == tc_id
            assert tcs[0]["type"] == "function"
            assert tcs[0]["function"]["name"] == "search"
            assert tcs[0]["function"]["arguments"] == '{"query": "capital of France"}'

            # Verify the tool result message still has role="tool"
            tool_msgs = [m for m in sent_messages if m["role"] == "tool"]
            assert len(tool_msgs) == 1
            assert tool_msgs[0]["tool_call_id"] == tc_id
