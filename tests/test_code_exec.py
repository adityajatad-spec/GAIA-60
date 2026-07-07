from unittest.mock import MagicMock, patch

import pytest

from tools.code_exec import run_python, _wrap_bare_expression
from agent.core import ResearchAgent, _handle_code_exec
from agent.providers.base import LLMProvider, ProviderResponse, ToolCall


# ── _wrap_bare_expression unit tests ───────────────────────────────────


def test_wrap_bare_simple():
    wrapped = _wrap_bare_expression("2 + 2")
    # Should produce code that prints the result
    namespace: dict = {}
    exec(wrapped, namespace)
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    exec(wrapped)
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    assert output.strip() == "4"


def test_wrap_bare_skips_print():
    """Already a print() call — no wrapping."""
    wrapped = _wrap_bare_expression("print(42)")
    assert wrapped == "print(42)"


def test_wrap_bare_skips_string_constant():
    """Standalone string constant (docstring-like) is not wrapped."""
    wrapped = _wrap_bare_expression('"""docstring"""')
    assert wrapped == '"""docstring"""'


def test_wrap_bare_skips_assignment():
    wrapped = _wrap_bare_expression("x = 42")
    assert wrapped == "x = 42"


def test_wrap_bare_multiline():
    r = run_python("x = 10\nx + 5")
    assert r["stdout"].strip() == "15"


def test_wrap_bare_with_import():
    wrapped = _wrap_bare_expression("import math\nmath.pi * 2")
    r = run_python(wrapped)
    assert "6.283" in r["stdout"]


def test_wrap_bare_syntax_error_passthrough():
    """Malformed code passes through unchanged and fails naturally."""
    wrapped = _wrap_bare_expression("print(")
    assert wrapped == "print("


# ── run_python unit tests ──────────────────────────────────────────────


def test_correct_calculation():
    result = run_python("print(2 + 2)")
    assert result["success"] is True
    assert result["stdout"].strip() == "4"
    assert result["error"] is None


def test_runtime_error_captured():
    result = run_python("1 / 0")
    assert result["success"] is False
    assert "division by zero" in result["stderr"] or "ZeroDivisionError" in result["stderr"]


def test_syntax_error_captured():
    result = run_python("print(")
    assert result["success"] is False
    assert result["stderr"] != ""


def test_timeout_enforced():
    result = run_python("while True: pass", timeout=3)
    assert result["success"] is False
    assert result["error"] is not None
    assert "timed out" in result["error"].lower()


def test_file_write_outside_sandbox_blocked():
    result = run_python("""
with open("/tmp/evil.txt", "w") as f:
    f.write("pwned")
""")
    assert result["success"] is False
    assert "denied" in (result["stderr"] + (result.get("error") or "")).lower()


def test_network_blocked():
    result = run_python("""
import socket
s = socket.socket()
""")
    assert result["success"] is False
    assert "denied" in (result["stderr"] + (result.get("error") or "")).lower() or \
           "disabled" in (result["stderr"] + (result.get("error") or "")).lower()


def test_empty_code():
    result = run_python("")
    assert result["success"] is True
    assert result["stdout"] == ""
    assert result["error"] is None
    assert result["warning"] is not None
    assert "forget to print" in result["warning"]


def test_bare_expression_auto_captured():
    """A bare expression like ``2 + 2`` is auto-printed."""
    result = run_python("2 + 2")
    assert result["success"] is True
    assert result["stdout"].strip() == "4"
    assert result["warning"] is None


def test_bare_expression_complex():
    result = run_python("847 * 293 / 17")
    assert result["success"] is True
    assert "14598" in result["stdout"]


def test_code_producing_no_output_emits_warning():
    """Code that runs but produces no stdout/stderr gets a warning."""
    result = run_python("x = 42")
    assert result["success"] is True
    assert result["stdout"] == ""
    assert result["stderr"] == ""
    assert result["warning"] is not None


def test_stdout_and_stderr_separate():
    result = run_python("""
import sys
print("out")
print("err", file=sys.stderr)
""")
    assert result["success"] is True
    assert result["stdout"].strip() == "out"
    assert "err" in result["stderr"]


# ── _handle_code_exec wrapper tests ────────────────────────────────────


def test_handle_code_exec_success():
    output = _handle_code_exec(code="print(2 + 2)")
    assert "4" in output
    assert "ERROR" not in output


def test_handle_code_exec_runtime_error():
    output = _handle_code_exec(code="1 / 0")
    assert "ERROR" in output
    assert "division by zero" in output or "ZeroDivisionError" in output


def test_handle_code_exec_syntax_error():
    output = _handle_code_exec(code="print(")
    assert "ERROR" in output


def test_handle_code_exec_timeout():
    output = _handle_code_exec(code="while True: pass")
    assert "ERROR" in output
    assert "timed out" in output.lower()


def test_handle_code_exec_bare_expression():
    output = _handle_code_exec(code="2 + 2")
    assert "4" in output
    assert "WARNING" not in output


def test_handle_code_exec_no_output_warning():
    output = _handle_code_exec(code="x = 42")
    assert "WARNING" in output
    assert "forget to print" in output


# ── Agent integration: tool_call_parse_error retry with code_exec ──────


def test_agent_retries_on_malformed_code_exec_args():
    """When the model sends malformed JSON args for code_exec, the agent
    retries with a clarifying nudge instead of crashing."""
    provider = MagicMock(spec=LLMProvider)
    provider.model = "test-model"

    responses = [
        ProviderResponse(
            text="",
            tool_calls=[],
            stop_reason="tool_call_parse_error",
        ),
        ProviderResponse(
            text="FINAL ANSWER: 42",
            tool_calls=[],
            stop_reason="end_turn",
        ),
    ]
    captured_messages = []

    def spy(*args, **kwargs):
        captured_messages.append([dict(m) for m in kwargs.get("messages", [])])
        return responses[len(captured_messages) - 1]

    provider.call.side_effect = spy

    agent = ResearchAgent(
        question="Run code that computes 6 times 7",
        provider=provider,
    )
    result = agent.run()

    assert result["answer"] == "42"
    assert result["steps_used"] == 2
    # The nudge should mention malformed JSON
    nudge = captured_messages[1][-1]["content"].lower()
    assert "malformed" in nudge


def test_agent_bare_expression_auto_captured_by_tool():
    """Bare expression in code_exec is auto-captured, result fed back."""
    provider = MagicMock(spec=LLMProvider)
    provider.model = "test-model"
    provider.call.side_effect = [
        ProviderResponse(
            text="Let me calculate.",
            tool_calls=[
                ToolCall(
                    name="code_exec",
                    input={"code": "847 * 293 / 17"},
                    id="call_1",
                )
            ],
            stop_reason="tool_use",
        ),
        ProviderResponse(
            text="FINAL ANSWER: 14598.294...",
            tool_calls=[],
            stop_reason="end_turn",
        ),
    ]

    agent = ResearchAgent(
        question="What is 847 * 293 / 17?",
        provider=provider,
    )
    result = agent.run()

    tool_entries = [e for e in result["trajectory"] if e.get("tool") == "code_exec"]
    assert len(tool_entries) == 1
    assert "14598" in tool_entries[0]["output"]


def test_agent_runs_code_exec_tool():
    """Full integration: the model calls code_exec, the tool runs, result
    is fed back, and the agent continues."""
    provider = MagicMock(spec=LLMProvider)
    provider.model = "test-model"
    provider.call.side_effect = [
        ProviderResponse(
            text="Let me calculate.",
            tool_calls=[
                ToolCall(
                    name="code_exec",
                    input={"code": "print(2 + 2)"},
                    id="call_1",
                )
            ],
            stop_reason="tool_use",
        ),
        ProviderResponse(
            text="FINAL ANSWER: 4",
            tool_calls=[],
            stop_reason="end_turn",
        ),
    ]

    agent = ResearchAgent(
        question="What is 2 + 2?",
        provider=provider,
    )
    result = agent.run()

    assert result["answer"] == "4"
    assert result["steps_used"] == 2
    # Verify code_exec tool result appears in trajectory
    tool_entries = [e for e in result["trajectory"] if e.get("tool") == "code_exec"]
    assert len(tool_entries) == 1
    assert "4" in tool_entries[0]["output"]
