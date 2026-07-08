from unittest.mock import MagicMock

import pytest

from agent.context_manager import (
    ContextManager,
    StructuredFact,
    estimate_messages_tokens,
    estimate_tokens,
    parse_tagged_facts,
    score_importance,
)
from agent.providers.base import LLMProvider, ProviderResponse


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_provider(context_window: int = 8192) -> MagicMock:
    p = MagicMock(spec=LLMProvider)
    p.model = "test-model"
    p.context_window = context_window
    p.call.return_value = ProviderResponse(
        text="Summary of research",
        tool_calls=[],
        stop_reason="end_turn",
    )
    return p


# ── StructuredFact extraction ────────────────────────────────────────────────


class TestFactExtraction:
    def test_parse_tagged_facts(self):
        text = (
            "Some text before\n"
            '[FACT] {"fact": "Paris is capital of France", "source": "web_search", "confidence": 0.95} [/FACT]\n'
            "More text\n"
            '[FACT] {"fact": "Population is 2.1M", "source": "web_search", "confidence": 0.85} [/FACT]'
        )
        facts = parse_tagged_facts(text, step=2)
        assert len(facts) == 2
        assert facts[0].fact == "Paris is capital of France"
        assert facts[0].source == "web_search"
        assert facts[0].confidence == 0.95
        assert facts[0].extracted_at_step == 2
        assert facts[1].fact == "Population is 2.1M"

    def test_parse_tagged_facts_no_tags(self):
        facts = parse_tagged_facts("Just some text", step=0)
        assert len(facts) == 0

    def test_parse_tagged_facts_malformed_json(self):
        text = '[FACT] {bad json} [/FACT]'
        facts = parse_tagged_facts(text, step=0)
        assert len(facts) == 0

    def test_parse_tagged_facts_missing_fact_key(self):
        text = '[FACT] {"name": "Paris"} [/FACT]'
        facts = parse_tagged_facts(text, step=0)
        assert len(facts) == 0

    def test_record_facts_from_response_accumulates(self):
        cm = ContextManager("test", _make_provider())
        cm.record_facts_from_response(
            '[FACT] {"fact": "A", "source": "s1", "confidence": 0.9} [/FACT]',
            step=1,
        )
        assert len(cm.facts) == 1
        assert cm.facts[0].fact == "A"

        # Same fact again → deduplicated
        cm.record_facts_from_response(
            '[FACT] {"fact": "A", "source": "s1", "confidence": 0.9} [/FACT]',
            step=2,
        )
        assert len(cm.facts) == 1

        # New fact → added
        cm.record_facts_from_response(
            '[FACT] {"fact": "B", "source": "s2", "confidence": 0.8} [/FACT]',
            step=2,
        )
        assert len(cm.facts) == 2


# ── Importance scoring ────────────────────────────────────────────────────────


class TestImportanceScoring:
    def test_error_entry_scores_low(self):
        msgs = [{"role": "tool", "content": "ERROR: something went wrong"}]
        score = score_importance(msgs, 0, "What is the capital?")
        assert score < 0.2

    def test_number_entry_scores_high(self):
        msgs = [{"role": "tool", "content": "The result is 3847 units"}]
        score = score_importance(msgs, 0, "How many units?")
        assert score > 0.8

    def test_date_entry_scores_high(self):
        msgs = [{"role": "tool", "content": "The historical event happened in 1945"}]
        score = score_importance(msgs, 0, "When did it happen?")
        assert score > 0.8

    def test_empty_entry_scores_low(self):
        msgs = [{"role": "tool", "content": "OK"}]
        score = score_importance(msgs, 0, "What is the capital?")
        assert score <= 0.5

    def test_question_term_match_scores_high(self):
        msgs = [{"role": "tool", "content": "The capital city of France is Paris"}]
        score = score_importance(msgs, 0, "What is the capital of France?")
        assert score > 0.7


# ── Token budget triggering ───────────────────────────────────────────────────


class TestTokenBudgetTriggering:
    def test_small_context_triggers_earlier(self):
        small = ContextManager("q", _make_provider(context_window=1000))
        large = ContextManager("q", _make_provider(context_window=100_000))

        # Messages that use ~400 estimated tokens
        msgs = [{"role": "user", "content": "X" * 1600}]
        # Small window: 400/1000 = 0.4 < 0.7 → not triggered yet
        assert not small.should_prune(msgs, step=5)

        # Larger messages: ~800 tokens
        msgs_large = [{"role": "user", "content": "X" * 3200}]
        # Small window: 800/1000 = 0.8 >= 0.7 → triggered
        assert small.should_prune(msgs_large, step=5)

        # Large window: 800/100000 = 0.008 < 0.7 → not triggered
        assert not large.should_prune(msgs_large, step=5)

    def test_token_threshold_respected(self):
        cm = ContextManager("q", _make_provider(context_window=1000))
        cm._token_threshold = 0.5

        # Messages with ~600 estimated tokens → 600/1000 = 0.6 >= 0.5 → triggered
        msgs = [{"role": "user", "content": "X" * 2400}]
        for i in range(3):
            msgs.append({"role": "assistant", "content": "X" * 200})
        assert cm.should_prune(msgs, step=5)

        # Messages with ~200 estimated tokens → 200/1000 = 0.2 < 0.5 → not triggered
        msgs_small = [{"role": "user", "content": "Hello"}]
        assert not cm.should_prune(msgs_small, step=5)

    def test_does_not_prune_before_prune_after(self):
        cm = ContextManager("q", _make_provider(context_window=1000))
        cm._prune_after = 3
        cm._token_threshold = 0.0  # always triggers on token budget
        msgs = [{"role": "user", "content": "X" * 5000}]
        assert not cm.should_prune(msgs, step=0)
        assert not cm.should_prune(msgs, step=2)
        assert cm.should_prune(msgs, step=3)  # now at _prune_after threshold


# ── Two-tier memory ───────────────────────────────────────────────────────────


class TestTwoTierMemory:
    def test_facts_preserved_across_prune(self):
        """Facts accumulated before pruning should appear in pruned messages."""
        cm = ContextManager("test", _make_provider())
        cm._token_threshold = 0.0
        cm.facts.append(StructuredFact(fact="Paris is capital", source="web", confidence=0.95, extracted_at_step=0))
        cm.facts.append(StructuredFact(fact="Population 2.1M", source="web", confidence=0.85, extracted_at_step=1))

        msgs = [{"role": "user", "content": "Q"}]
        for i in range(6):
            msgs.append({"role": "assistant", "content": f"step {i}"})
            msgs.append({"role": "tool", "content": f"result {i}"})

        pruned = cm.prune(msgs, step=5)
        # Should contain facts block
        facts_text = "".join(m.get("content", "") for m in pruned if isinstance(m.get("content"), str))
        assert "Paris is capital" in facts_text
        assert "Population 2.1M" in facts_text

    def test_facts_included_in_subsequent_call(self):
        """Facts should be included in the next provider call after pruning."""
        provider = _make_provider()
        cm = ContextManager("test", provider)
        cm._token_threshold = 0.0
        cm.facts.append(StructuredFact(fact="Key fact X", source="code_exec", confidence=0.9, extracted_at_step=1))

        msgs = [{"role": "user", "content": "Q"}]
        for i in range(6):
            msgs.append({"role": "assistant", "content": f"step {i}"})
            msgs.append({"role": "tool", "content": f"result {i}"})

        pruned = cm.prune(msgs, step=5)
        combined = " ".join(str(m.get("content", "")) for m in pruned)
        assert "Key fact X" in combined

    def test_fact_after_prune_still_available(self):
        """A fact extracted before pruning should still be accessible
        in the ContextManager's fact list after pruning."""
        cm = ContextManager("test", _make_provider())
        cm._token_threshold = 0.0
        cm.facts.append(StructuredFact(fact="Persistent fact", source="web", confidence=0.9, extracted_at_step=0))

        msgs = [{"role": "user", "content": "Q"}]
        for i in range(6):
            msgs.append({"role": "assistant", "content": f"step {i}"})
            msgs.append({"role": "tool", "content": f"result {i}"})

        cm.prune(msgs, step=5)
        assert any(f.fact == "Persistent fact" for f in cm.facts)


# ── Graceful degradation ──────────────────────────────────────────────────────


class TestGracefulDegradation:
    def test_fallback_on_compress_failure(self):
        """When smart compression fails, fall back to window-based pruning
        and return pruned messages (not crash)."""
        provider = _make_provider()
        provider.call.return_value = ProviderResponse(
            text="ERROR: something broke",
            tool_calls=[],
            stop_reason="end_turn",
        )

        cm = ContextManager("test", provider)
        cm._token_threshold = 0.0

        msgs = [{"role": "user", "content": "Q"}]
        for i in range(6):
            msgs.append({"role": "assistant", "content": f"step {i}"})
            msgs.append({"role": "tool", "content": f"result {i}"})

        pruned = cm.prune(msgs, step=5)
        # Should still return pruned messages (with fallback summary)
        assert len(pruned) > 0
        assert pruned[0]["role"] == "user"
        # Facts block or summary should be present
        combined = " ".join(str(m.get("content", "")) for m in pruned)
        assert "compressed" in combined.lower() or "Structured Facts" in combined or "Research Progress Summary" in combined

    def test_fallback_does_not_consume_extra_calls_forever(self):
        """After entering fallback mode, should not keep calling provider
        for extra compression attempts."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return ProviderResponse(text="ERROR: compression failed", tool_calls=[], stop_reason="end_turn")

        provider = _make_provider()
        provider.call.side_effect = side_effect

        cm = ContextManager("test", provider)
        cm._token_threshold = 0.0

        msgs = [{"role": "user", "content": "Q"}]
        for i in range(6):
            msgs.append({"role": "assistant", "content": f"step {i}"})
            msgs.append({"role": "tool", "content": f"result {i}"})

        cm.prune(msgs, step=5)
        calls_after_first = call_count[0]

        # Second prune should not call provider again (fallback active)
        cm.prune(msgs, step=7)
        assert call_count[0] == calls_after_first


# ── Token estimation utility ──────────────────────────────────────────────────


class TestTokenEstimation:
    def test_estimate_messages_tokens(self):
        msgs = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there", "tool_calls": [{"name": "search", "input": {"q": "test"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Result here"},
        ]
        tokens = estimate_messages_tokens(msgs)
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_empty_messages(self):
        assert estimate_messages_tokens([]) == 0


# ── Model-specific token estimation ────────────────────────────────────


class TestModelTokenRatios:
    def test_detect_qwen_ratio(self):
        assert ContextManager._detect_token_ratio("qwen3:8b-64k") == 3.5

    def test_detect_llama_ratio(self):
        assert ContextManager._detect_token_ratio("llama3.1:8b") == 3.7

    def test_detect_deepseek_ratio(self):
        assert ContextManager._detect_token_ratio("deepseek-r1:7b") == 3.6

    def test_detect_default_ratio(self):
        assert ContextManager._detect_token_ratio("claude-sonnet-4-20250514") == 4.0

    def test_estimate_tokens_with_ratio(self):
        text = "Hello world, this is a test" * 100  # ~2800 chars
        default = estimate_tokens(text)
        dense = estimate_tokens(text, chars_per_token=3.5)
        assert dense > default  # more tokens for same text with denser ratio

    def test_init_uses_detected_ratio(self):
        provider = _make_provider()
        provider.model = "qwen3:8b-64k"
        cm = ContextManager("q", provider)
        assert cm._chars_per_token == 3.5

    def test_init_fallback_ratio(self):
        provider = _make_provider()
        provider.model = "unknown-model"
        cm = ContextManager("q", provider)
        assert cm._chars_per_token == 4.0


# ── Importance-scored message selection ────────────────────────────────


class TestSelectImportantMessages:
    def test_keeps_question_and_last_assistant(self):
        cm = ContextManager("q", _make_provider())
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "user", "content": "step 1 nudge"},
            {"role": "assistant", "content": "result 1", "tool_calls": []},
            {"role": "tool", "content": "tool result 1"},
            {"role": "user", "content": "step 2 nudge"},
            {"role": "assistant", "content": "result 2", "tool_calls": []},
            {"role": "tool", "content": "tool result 2"},
        ]
        keep, compress = cm._select_important_messages(msgs)
        assert 0 in keep  # question always kept
        assert 5 in keep  # last assistant always kept
        assert 6 in keep  # last assistant's tool result

    def test_keeps_high_importance_over_recent(self):
        """Messages with high importance scores are kept even if they're old."""
        cm = ContextManager("What is the capital of France?", _make_provider())
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "irrelevant step", "tool_calls": []},
            {"role": "tool", "content": "OK"},
            {"role": "assistant", "content": "The capital of France is Paris", "tool_calls": []},
            {"role": "tool", "content": "Paris confirmed"},
            {"role": "assistant", "content": "some final step", "tool_calls": []},
            {"role": "tool", "content": "done"},
        ]
        keep, compress = cm._select_important_messages(msgs)
        # The important "capital of France" message (index 3) should be kept
        assert 3 in keep
        # The irrelevant step (index 1) may or may not be kept

    def test_compress_excludes_kept_indices(self):
        cm = ContextManager("q", _make_provider())
        msgs = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "step 1", "tool_calls": []},
            {"role": "tool", "content": "r1"},
            {"role": "assistant", "content": "step 2", "tool_calls": []},
            {"role": "tool", "content": "r2"},
        ]
        keep, compress = cm._select_important_messages(msgs)
        # No overlap between keep and compress
        assert set(keep) & set(compress) == set()
        # All indices accounted for
        all_idx = set(range(len(msgs)))
        assert set(keep) | set(compress) == all_idx


# ── Fact-check integration ─────────────────────────────────────────────


class TestFactCheck:
    def test_fact_check_ok_on_clean_summary(self):
        """When fact-check returns OK, compression result is used."""
        provider = _make_provider()

        def side_effect(messages=None, tools=None, system_prompt=None):
            if "verifying" in (system_prompt or "").lower():
                return ProviderResponse(text="OK", tool_calls=[], stop_reason="end_turn")
            return ProviderResponse(text="Summary here\n[FACT] {\"fact\": \"Paris is capital\", "
                                         "\"source\": \"web\", \"confidence\": 0.9} [/FACT]",
                                    tool_calls=[], stop_reason="end_turn")

        provider.call.side_effect = side_effect
        cm = ContextManager("test", provider)
        cm._token_threshold = 0.0

        msgs = [{"role": "user", "content": "Q"}]
        for i in range(6):
            msgs.append({"role": "assistant", "content": f"step {i}"})
            msgs.append({"role": "tool", "content": f"result {i}"})

        pruned = cm.prune(msgs, step=5)
        assert len(pruned) > 0
        # Fact should have been recorded
        assert any("Paris is capital" in str(m.get("content", "")) for m in pruned)

    def test_fact_check_failure_triggers_fallback(self):
        """When fact-check detects contradictions, fall back to window-based pruning."""
        provider = _make_provider()
        call_log = []

        def side_effect(messages=None, tools=None, system_prompt=None):
            is_fact_check = "factual consistency" in (system_prompt or "").lower()
            call_log.append(("verify" if is_fact_check else "compress", messages))
            if is_fact_check:
                return ProviderResponse(text="CONTRADICTION: Paris is capital vs Summary says London",
                                        tool_calls=[], stop_reason="end_turn")
            return ProviderResponse(text="Summary here\n[FACT] {\"fact\": \"Paris is capital\", "
                                         "\"source\": \"web\", \"confidence\": 0.9} [/FACT]",
                                    tool_calls=[], stop_reason="end_turn")

        provider.call.side_effect = side_effect
        cm = ContextManager("test", provider)
        cm._token_threshold = 0.0

        msgs = [{"role": "user", "content": "Q"}]
        for i in range(4):
            msgs.append({"role": "assistant", "content": f"step {i}"})
            msgs.append({"role": "tool", "content": f"result {i}"})

        pruned = cm.prune(msgs, step=5)
        assert len(pruned) > 0
        # Should have attempted both compression and verification calls
        assert len(call_log) == 2
        assert call_log[0][0] == "compress"
        assert call_log[1][0] == "verify"
        # Fallback should NOT include the fact since compression was rejected
        combined = " ".join(str(m.get("content", "")) for m in pruned)
        assert "compressed" in combined.lower()

    def test_fact_check_no_new_facts_skips_verification(self):
        """When no new facts were extracted, skip fact-check entirely."""
        provider = _make_provider()
        call_log = []

        def side_effect(*args, **kwargs):
            call_log.append(1)
            return ProviderResponse(text="Summary here (no facts)", tool_calls=[], stop_reason="end_turn")

        provider.call.side_effect = side_effect
        cm = ContextManager("test", provider)
        cm._token_threshold = 0.0

        msgs = [{"role": "user", "content": "Q"}]
        for i in range(4):
            msgs.append({"role": "assistant", "content": f"step {i}"})
            msgs.append({"role": "tool", "content": f"result {i}"})

        cm.prune(msgs, step=5)
        # Only one call (compression), no fact-check since no facts
        assert len(call_log) == 1
