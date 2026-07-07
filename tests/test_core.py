from unittest.mock import MagicMock, patch

import pytest

from agent.core import ResearchAgent, _normalize_final_answer_prefix
from agent.providers.base import LLMProvider, ProviderResponse, ToolCall


def _make_text_response(
    text: str, stop_reason: str = "end_turn"
) -> ProviderResponse:
    return ProviderResponse(text=text, tool_calls=[], stop_reason=stop_reason)


def _make_tool_response(
    text: str, tool_name: str = "search", tool_input: dict | None = None
) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        tool_calls=[
            ToolCall(
                name=tool_name,
                input=tool_input or {"query": "test"},
                id="test_call_id",
            )
        ],
        stop_reason="tool_use",
    )


@pytest.fixture
def mock_provider():
    provider = MagicMock(spec=LLMProvider)
    provider.model = "test-model"
    provider.call.return_value = _make_text_response("FINAL ANSWER: Paris")
    return provider


def _spy_call(provider, responses: list[ProviderResponse]):
    """Wrap provider.call to return from *responses* in order and record
    a snapshot of the messages at each call."""
    it = iter(responses)
    call_snapshots: list[list[dict]] = []

    def spy(*args, **kwargs):
        call_snapshots.append([dict(m) for m in kwargs.get("messages", [])])
        return next(it)

    provider.call.side_effect = spy
    return call_snapshots


class TestResearchAgent:
    def test_direct_answer_no_tools(self, mock_provider):
        agent = ResearchAgent(
            question="What is the capital of France?",
            provider=mock_provider,
        )
        result = agent.run()

        assert result["answer"] == "Paris"
        assert result["steps_used"] == 1
        traj = result["trajectory"]
        assert len(traj) == 1
        assert traj[0]["role"] == "assistant"
        assert result["format_noncompliant"] is False

    def test_trajectory_tracks_model_message(self, mock_provider):
        agent = ResearchAgent(
            question="What is the capital of France?",
            provider=mock_provider,
        )
        result = agent.run()

        msg = result["trajectory"][0]
        assert msg["step"] == 0
        assert msg["role"] == "assistant"
        assert msg["stop_reason"] == "end_turn"

    def test_step_budget_exhausted(self):
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        provider.call.return_value = _make_tool_response(
            "Let me search for that...", tool_name="search"
        )

        agent = ResearchAgent(
            question="Endless question",
            provider=provider,
        )
        agent.step_budget = 3
        result = agent.run()

        assert result["answer"] == "MAX_STEPS_REACHED"
        assert result["steps_used"] == 3

    def test_returns_text_when_no_tool_call_and_no_final_answer(self):
        """Non-compliant text triggers a format nudge first; if the model
        still doesn't comply, the response is accepted with
        format_noncompliant=True."""
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        provider.call.return_value = _make_text_response(
            "The capital of France is Paris."
        )

        agent = ResearchAgent(
            question="What is the capital of France?",
            provider=provider,
        )
        result = agent.run()

        assert result["answer"] == "The capital of France is Paris."
        assert result["steps_used"] == 2
        assert result["format_noncompliant"] is True
        assert result["trajectory"][-1].get("format_noncompliant") is True

    def test_retries_on_tool_call_parse_error(self):
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"

        parse_error = ProviderResponse(
            text="",
            tool_calls=[],
            stop_reason="tool_call_parse_error",
        )
        final = _make_text_response("FINAL ANSWER: 42")

        snapshots = _spy_call(provider, [parse_error, final])

        agent = ResearchAgent(question="What is 6 times 7?", provider=provider)
        result = agent.run()

        assert result["answer"] == "42"
        assert result["steps_used"] == 2
        # First call should have only the original question
        assert len(snapshots[0]) == 1
        assert snapshots[0][0]["role"] == "user"
        # Second call should include the nudge (the last user message)
        nudge = snapshots[1][-1]["content"].lower()
        assert "malformed" in nudge

    def test_format_nudge_then_compliance(self):
        """Model first returns non-compliant text, gets nudged, then
        complies with FINAL ANSWER: on the second try."""
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        provider.call.side_effect = [
            _make_text_response("The capital of France is Paris."),
            _make_text_response("FINAL ANSWER: Paris"),
        ]

        agent = ResearchAgent(question="What is the capital?", provider=provider)
        result = agent.run()

        assert result["answer"] == "Paris"
        assert result["steps_used"] == 2
        assert result["format_noncompliant"] is False

    def test_format_noncompliant_flag_set_on_persistent_violation(self):
        """Model never complies with FINAL ANSWER: format — second attempt
        is accepted with format_noncompliant=True."""
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        provider.call.side_effect = [
            _make_text_response("WRONG_PREFIX: some answer"),
            _make_text_response("STILL_WRONG: still not complying"),
        ]

        agent = ResearchAgent("test", provider=provider)
        result = agent.run()

        assert result["answer"] == "STILL_WRONG: still not complying"
        assert result["steps_used"] == 2
        assert result["format_noncompliant"] is True
        assert result["trajectory"][-1].get("format_noncompliant") is True

    def test_browse_rejects_url_not_from_search(self):
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        provider.call.side_effect = [
            _make_tool_response(
                "Let me search.", tool_name="search",
                tool_input={"query": "weather Paris"},
            ),
            _make_tool_response(
                "Let me browse.", tool_name="browse",
                tool_input={"url": "https://fake-invented-url.com/weather"},
            ),
            _make_text_response("FINAL ANSWER: Sunny"),
        ]

        mock_browse = MagicMock()
        with (
            patch("agent.core.web_search") as mock_search,
            patch.dict("agent.core._TOOL_HANDLERS", {"browse": mock_browse}),
        ):
            mock_search.return_value = [
                {
                    "title": "Paris Weather",
                    "url": "https://real-site.com/paris",
                    "snippet": "Sunny 23°C",
                }
            ]

            agent = ResearchAgent(
                question="What's the weather?",
                provider=provider,
            )
            agent.step_budget = 5
            result = agent.run()

        assert result["answer"] == "Sunny"
        # Find the browse tool result in trajectory
        browse_entries = [
            e for e in result["trajectory"]
            if e.get("tool") == "browse"
        ]
        assert len(browse_entries) == 1
        assert "ERROR" in browse_entries[0]["output"]
        assert "not found in any previous" in browse_entries[0]["output"]
        # web_browse should NOT have been called at all
        mock_browse.assert_not_called()

    def test_browse_allows_url_from_search(self):
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        provider.call.side_effect = [
            _make_tool_response(
                "Let me search.", tool_name="search",
                tool_input={"query": "weather Paris"},
            ),
            _make_tool_response(
                "Let me browse.", tool_name="browse",
                tool_input={"url": "https://real-site.com/paris"},
            ),
            _make_text_response("FINAL ANSWER: Sunny"),
        ]

        mock_browse = MagicMock(return_value="It is sunny in Paris (23°C).")
        with (
            patch("agent.core.web_search") as mock_search,
            patch.dict("agent.core._TOOL_HANDLERS", {"browse": mock_browse}),
        ):
            mock_search.return_value = [
                {
                    "title": "Paris Weather",
                    "url": "https://real-site.com/paris",
                    "snippet": "Sunny 23°C",
                }
            ]

            agent = ResearchAgent(
                question="What's the weather?",
                provider=provider,
            )
            agent.step_budget = 5
            result = agent.run()

        assert result["answer"] == "Sunny"
        browse_entries = [
            e for e in result["trajectory"]
            if e.get("tool") == "browse"
        ]
        assert len(browse_entries) == 1
        assert "ERROR" not in browse_entries[0]["output"]
        mock_browse.assert_called_once_with(url="https://real-site.com/paris")

    def test_accepts_final_answer_with_underscore(self):
        """FINAL_ANSWER: (underscore) is normalized and treated as compliant."""
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        provider.call.return_value = _make_text_response(
            "FINAL_ANSWER: Berlin"
        )
        agent = ResearchAgent(
            question="Capital of Germany?", provider=provider
        )
        result = agent.run()
        assert result["answer"] == "Berlin"
        assert result["steps_used"] == 1
        assert result["format_noncompliant"] is False

    def test_accepts_final_answer_case_insensitive(self):
        """Final Answer: (mixed case) is normalized and treated as compliant."""
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        provider.call.return_value = _make_text_response(
            "Final Answer: Tokyo"
        )
        agent = ResearchAgent(
            question="Capital of Japan?", provider=provider
        )
        result = agent.run()
        assert result["answer"] == "Tokyo"
        assert result["format_noncompliant"] is False

    def test_nudge_message_quotes_exact_format(self):
        """The format-compliance nudge explicitly quotes 'FINAL ANSWER: '."""
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        snapshots = _spy_call(provider, [
            _make_text_response("SOME_PREFIX: answer"),
            _make_text_response("FINAL ANSWER: Paris"),
        ])

        agent = ResearchAgent(
            question="Capital of France?", provider=provider
        )
        result = agent.run()
        assert result["answer"] == "Paris"
        assert result["steps_used"] == 2
        # The nudge is appended between call 1 and call 2, so it's in
        # the messages of the second call (snapshots[1]).
        nudge = snapshots[1][-1]["content"]
        assert '"FINAL ANSWER: "' in nudge


# ── Empty-response failure ──────────────────────────────────────────────────


class TestEmptyResponseFailure:
    def test_empty_response_marked_as_failure(self):
        """A completely empty provider response (no text, no tools) after
        the format-compliance retry should be flagged distinctly, not
        returned as a placeholder answer."""
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        provider.call.side_effect = [
            _make_text_response(""),  # first attempt: empty
            _make_text_response(""),  # second attempt (retry): still empty
        ]

        agent = ResearchAgent(
            question="What is the answer?",
            provider=provider,
        )
        result = agent.run()

        assert result["empty_response_failure"] is True
        assert result["format_noncompliant"] is True
        assert result["steps_used"] == 2
        assert result["answer"] == "(empty response)"
        # Trajectory entry should also have the marker
        assert result["trajectory"][-1].get("empty_response_failure") is True

    def test_nonempty_response_not_marked_empty(self):
        """A non-empty but format-noncompliant response should NOT set
        empty_response_failure."""
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        provider.call.side_effect = [
            _make_text_response("just some plain text"),
            _make_text_response("still just plain text without prefix"),
        ]

        agent = ResearchAgent(
            question="test",
            provider=provider,
        )
        result = agent.run()

        assert result["empty_response_failure"] is False
        assert result["format_noncompliant"] is True
        assert result["steps_used"] == 2
        # answer should be the raw text, not "(empty response)"
        assert result["answer"] == "still just plain text without prefix"

    def test_empty_response_gets_one_retry_before_failure(self):
        """The model gets one retry nudge even on empty responses (same as
        format-noncompliant responses)."""
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        snapshots = _spy_call(provider, [
            _make_text_response(""),
            _make_text_response("FINAL ANSWER: 42"),
        ])

        agent = ResearchAgent(
            question="What is 6 times 7?",
            provider=provider,
        )
        result = agent.run()

        assert result["answer"] == "42"
        assert result["format_noncompliant"] is False
        assert result["steps_used"] == 2
        # The nudge should mention empty response
        nudge = snapshots[1][-1]["content"].lower()
        assert "empty" in nudge


# ── Verification ───────────────────────────────────────────────────────────


class TestVerification:
    def test_verification_runs_on_clean_first_try_path(self):
        """Regression: verification must run even when the model outputs a
        correctly-formatted FINAL ANSWER on the very first step (no format
        retry, no tool calls — the clean path). The answer is wrong and
        verification must catch and correct it."""
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        calls: list[str] = []

        def side_effect(*args, **kwargs):
            calls.append("call")
            if len(calls) == 1:
                return _make_text_response("FINAL ANSWER: 41")
            return _make_text_response("FINAL ANSWER: 42")

        provider.call.side_effect = side_effect

        agent = ResearchAgent(
            question="What is 6 times 7?",
            provider=provider,
        )
        result = agent.run()

        assert result["answer"] == "42"
        assert result["verification_performed"] is True
        assert result["steps_used"] == 1
        assert result["format_noncompliant"] is False
        assert len(calls) == 2

    def test_verification_confirms_correct_answer(self):
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        provider.call.return_value = _make_text_response("FINAL ANSWER: Paris")

        agent = ResearchAgent(
            question="Capital of France?",
            provider=provider,
        )
        result = agent.run()

        assert result["answer"] == "Paris"
        assert result["steps_used"] == 1

    def test_verification_corrects_answer(self):
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        calls: list[str] = []

        def side_effect(*args, **kwargs):
            calls.append("call")
            if len(calls) == 1:
                return _make_text_response("FINAL ANSWER: 41")
            return _make_text_response("FINAL ANSWER: 42")

        provider.call.side_effect = side_effect

        agent = ResearchAgent(
            question="What is 6 times 7?",
            provider=provider,
        )
        result = agent.run()

        assert result["answer"] == "42"
        assert len(calls) == 2

    def test_verification_fallback_on_error(self):
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        calls: list[str] = []

        def side_effect(*args, **kwargs):
            calls.append("call")
            if len(calls) == 1:
                return _make_text_response("FINAL ANSWER: Paris")
            raise RuntimeError("API error")

        provider.call.side_effect = side_effect

        agent = ResearchAgent(
            question="Capital of France?",
            provider=provider,
        )
        result = agent.run()

        assert result["answer"] == "Paris"
        assert len(calls) == 2

    def test_verify_provider_used_when_specified(self):
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        provider.call.return_value = _make_text_response("FINAL ANSWER: Paris")

        verify_provider = MagicMock(spec=LLMProvider)
        verify_provider.model = "verify-model"
        verify_provider.call.return_value = _make_text_response(
            "FINAL ANSWER: Paris"
        )

        agent = ResearchAgent(
            question="Capital of France?",
            provider=provider,
            verify_provider=verify_provider,
        )
        result = agent.run()

        assert result["answer"] == "Paris"
        verify_provider.call.assert_called_once()
        call_kwargs = verify_provider.call.call_args[1]
        verify_msg = call_kwargs["messages"][0]["content"]
        assert "Capital of France" in verify_msg
        assert "Paris" in verify_msg
        assert call_kwargs.get("tools") is None


# ── Retry on tool failure ──────────────────────────────────────────────────


class TestRetryOnToolFailure:
    def test_retry_on_tool_failure(self):
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"

        verify_provider = MagicMock(spec=LLMProvider)
        verify_provider.model = "test-model"
        verify_provider.call.return_value = _make_text_response(
            "FINAL ANSWER: 42"
        )

        snapshots = _spy_call(provider, [
            _make_tool_response(
                "Searching...", tool_name="search",
                tool_input={"query": "test"},
            ),
            _make_tool_response(
                "Let me compute.", tool_name="code_exec",
                tool_input={"code": "print(6*7)"},
            ),
            _make_text_response("FINAL ANSWER: 42"),
        ])

        mock_search = MagicMock(side_effect=Exception("Network timeout"))
        mock_code = MagicMock(return_value="42")

        with (
            patch("agent.core.web_search", mock_search),
            patch.dict("agent.core._TOOL_HANDLERS", {"code_exec": mock_code}),
        ):
            agent = ResearchAgent(
                question="What is 6 times 7?",
                provider=provider,
                verify_provider=verify_provider,
            )
            agent.step_budget = 5
            result = agent.run()

        assert result["answer"] == "42"
        assert result["steps_used"] == 3
        nudge = snapshots[1][-1]["content"]
        assert "failed" in nudge.lower()


# ── Budget awareness ───────────────────────────────────────────────────────


class TestBudgetAwareness:
    def test_budget_warning_injected_at_threshold(self):
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"

        verify_provider = MagicMock(spec=LLMProvider)
        verify_provider.call.return_value = _make_text_response(
            "FINAL ANSWER: done"
        )

        responses = []
        for _ in range(12):
            responses.append(
                _make_tool_response("...", tool_name="search")
            )
        responses.append(_make_text_response("FINAL ANSWER: done"))

        snapshots = _spy_call(provider, responses)

        with patch("agent.core.web_search", MagicMock(return_value="ok")):
            agent = ResearchAgent(
                question="test",
                provider=provider,
                verify_provider=verify_provider,
            )
            agent.step_budget = 15
            result = agent.run()

        assert result["answer"] == "done"
        assert result["steps_used"] == 13

        warning_steps = []
        for i, msgs in enumerate(snapshots):
            if any(
                "Budget warning" in str(m.get("content", ""))
                for m in msgs
            ):
                warning_steps.append(i)

        assert len(warning_steps) > 0
        assert warning_steps[0] >= 10


# ── Self-consistency voting ────────────────────────────────────────────────


class TestSelfConsistencyVoting:
    def test_run_with_voting_returns_majority(self):
        provider = MagicMock(spec=LLMProvider)
        provider.model = "test-model"
        provider.call.return_value = _make_text_response(
            "FINAL ANSWER: Paris"
        )

        agent = ResearchAgent(
            question="Capital of France?",
            provider=provider,
        )
        result = agent.run_with_voting(n=3)

        assert result["answer"] == "Paris"
        assert result["n_runs"] == 3
        assert result["votes"]["Paris"] == 3


# ── Final-answer prefix normalization ─────────────────────────────────────


class TestFinalAnswerPrefixNormalization:
    def test_canonical(self):
        ok, out = _normalize_final_answer_prefix("FINAL ANSWER: Paris")
        assert ok
        assert "Paris" in out.split("FINAL ANSWER:", 1)[1]

    def test_underscore(self):
        ok, out = _normalize_final_answer_prefix("FINAL_ANSWER: Berlin")
        assert ok
        assert "Berlin" in out.split("FINAL ANSWER:", 1)[1]

    def test_underscore_no_space_after_colon(self):
        ok, out = _normalize_final_answer_prefix("FINAL_ANSWER:Berlin")
        assert ok
        assert "Berlin" in out.split("FINAL ANSWER:", 1)[1]

    def test_case_insensitive(self):
        ok, out = _normalize_final_answer_prefix("final_answer: tokyo")
        assert ok
        assert "tokyo" in out.split("FINAL ANSWER:", 1)[1]

    def test_mixed_case_with_space(self):
        ok, out = _normalize_final_answer_prefix("Final Answer: London")
        assert ok
        assert "London" in out.split("FINAL ANSWER:", 1)[1]

    def test_no_prefix(self):
        ok, out = _normalize_final_answer_prefix("Just some text")
        assert not ok
        assert out == "Just some text"
