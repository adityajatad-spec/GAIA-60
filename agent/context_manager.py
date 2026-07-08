from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

FACT_TAG_RE = re.compile(r"\[FACT\]\s*(\{.*?\})\s*\[/FACT\]", re.DOTALL)


@dataclass
class StructuredFact:
    fact: str
    source: str = "unknown"
    confidence: float = 0.5
    extracted_at_step: int = -1


FACT_EXTRACTION_INSTRUCTION = (
    "\n\nAfter receiving tool results, extract key factual findings "
    "as structured facts. Use this format (one per fact):\n"
    '[FACT] {"fact": "The precise fact", "source": "tool_used", '
    '"confidence": 0.95} [/FACT]\n'
    "Only extract objective, checkable facts "
    "(numbers, names, dates, measurements, relationships). "
    "Include ALL relevant facts from tool results. "
    "Set confidence based on reliability of the source "
    "(0.9-1.0 for official data, 0.5-0.8 for general sources, "
    "<0.5 for uncertain information)."
)

COMPRESSION_PROMPT = """\
You are compressing research history and extracting factual findings.

First, extract ALL factual findings from the history below as structured facts.
Output one JSON object per line — each line must be a complete valid JSON object:
{"fact": "...", "source": "...", "confidence": 0.0}

Then write a concise research log capturing what was attempted and learned:
[Research Log]
- Step X: action → key finding
...

Key facts gathered:
- ..."""


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Token estimation with configurable char-per-token ratio."""
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


def estimate_messages_tokens(messages: list[dict], chars_per_token: float = 4.0) -> int:
    total = 0
    for msg in messages:
        total += 5
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "image":
                        total += 1000
                    else:
                        total += estimate_tokens(str(block.get("text", "")), chars_per_token)
                else:
                    total += estimate_tokens(str(block), chars_per_token)
        else:
            total += estimate_tokens(str(content), chars_per_token)
        for tc in msg.get("tool_calls", []):
            total += estimate_tokens(json.dumps(tc.get("input", {})), chars_per_token)
            total += 15
    return total


def score_importance(messages: list[dict], index: int, question: str) -> float:
    """Score a trajectory message entry by importance/relevance.

    Returns 0.0 (compress aggressively) to 1.0 (keep verbatim).

    Heuristics:
    - Error/dead-end entries → low (0.1)
    - Empty or trivial entries → low (0.3)
    - Entries with numbers/dates → high (0.9+)
    - Entries with question term overlap → high (0.8)
    - Substantive tool results → medium-high (0.7)
    """
    content = str(messages[index].get("content", ""))

    if not content.strip() or len(content) < 20:
        return 0.3
    lower = content.lower()
    if lower[:6] == "error:" or "error" in lower[:50]:
        return 0.1
    if re.search(r"\b\d{3,}\b", content):
        return 0.95
    if re.search(r"\b\d{4}\b", content):
        return 0.9
    question_terms = set(re.findall(r"[a-zA-Z]{4,}", question.lower()))
    overlap = sum(1 for t in question_terms if t in lower)
    if overlap >= 2:
        return 0.8
    return 0.5


def parse_tagged_facts(text: str, step: int) -> list[StructuredFact]:
    """Parse ``[FACT] {...} [/FACT]`` tags from provider response text."""
    facts: list[StructuredFact] = []
    for match in FACT_TAG_RE.finditer(text):
        try:
            data = json.loads(match.group(1))
            if isinstance(data.get("fact"), str) and data["fact"].strip():
                facts.append(
                    StructuredFact(
                        fact=data["fact"].strip(),
                        source=str(data.get("source", "unknown")),
                        confidence=float(data.get("confidence", 0.5)),
                        extracted_at_step=step,
                    )
                )
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return facts


def build_facts_message_text(facts: list[StructuredFact]) -> str:
    """Build a structured facts block for inclusion in messages."""
    if not facts:
        return ""
    lines = ["[Structured Facts]"]
    for f in facts:
        source = f.source or "unknown"
        lines.append(
            f"- {f.fact} "
            f"(source: {source}, "
            f"confidence: {f.confidence})"
        )
    return "\n".join(lines)


class ContextManager:
    """Two-tier memory: persistent structured facts + compressible trajectory.

    1. Structured fact extraction inline from provider responses
    2. Importance-weighted retention based on entry scoring
    3. Token-budget-aware pruning triggering
    4. Two-tier memory (facts always preserved)
    5. Graceful degradation to simple window-based fallback
    """

    def __init__(
        self,
        question: str,
        provider: Any,
        step_budget: int = 15,
        prune_after: int = 3,
        prune_interval: int = 5,
        keep_recent: int = 2,
        token_threshold: float = 0.7,
        min_prune_interval: int = 2,
    ):
        self.question = question
        self.provider = provider
        self.step_budget = step_budget
        self.facts: list[StructuredFact] = []
        self._prune_after = prune_after
        self._prune_interval = prune_interval
        self._keep_recent = keep_recent
        self._token_threshold = token_threshold
        self._min_prune_interval = min_prune_interval
        self._last_pruned_step = -1
        self._fallback_active = False
        self._ctx_window: int = self._get_context_window()
        model_name = getattr(self.provider, "model", "")
        self._chars_per_token: float = self._detect_token_ratio(model_name)

    def _get_context_window(self) -> int:
        try:
            raw = getattr(self.provider, "context_window", None)
            if raw is None or not isinstance(raw, int):
                return 8192
            if raw <= 0 or raw > 1_000_000_000:
                return 8192
            return raw
        except (ValueError, TypeError, AttributeError):
            return 8192

    @staticmethod
    def _detect_token_ratio(model_name: str = "") -> float:
        """Model-specific chars-per-token ratio for better token estimation."""
        name = model_name.lower()
        if "qwen" in name:
            return 3.5
        if "llama" in name:
            return 3.7
        if "deepseek" in name:
            return 3.6
        return 4.0

    def record_facts_from_response(self, text: str, step: int) -> list[StructuredFact]:
        """Extract ``[FACT]``-tagged facts from a provider response."""
        new = parse_tagged_facts(text, step)
        if new:
            for f in new:
                if not any(existing.fact == f.fact for existing in self.facts):
                    self.facts.append(f)
        return new

    def _estimate_current_usage(self, messages: list[dict]) -> float:
        """Return ratio of estimated token usage to context window."""
        if self._ctx_window <= 0:
            return 0.0
        estimated = estimate_messages_tokens(messages, self._chars_per_token)
        return estimated / self._ctx_window

    def should_prune(self, messages: list[dict], step: int) -> bool:
        """Check whether pruning should fire, using both token-budget and step-based triggers.

        Returns ``True`` if pruning should happen — either because the
        estimated token usage exceeds the configured threshold OR because
        the step-based interval has been reached AND usage is non-trivial.
        Always returns ``False`` for steps below ``_prune_after``.
        """
        if step < self._prune_after:
            return False
        usage_ratio = self._estimate_current_usage(messages)
        if usage_ratio >= self._token_threshold:
            return True
        if step != self._last_pruned_step:
            if step - self._last_pruned_step >= self._min_prune_interval:
                if (step - self._prune_after) % self._prune_interval == 0:
                    if usage_ratio >= 0.3:
                        return True
        return False

    def _select_important_messages(
        self, messages: list[dict]
    ) -> tuple[list[int], list[int]]:
        """Score all messages and pick the top K for verbatim retention.

        Returns ``(keep_indices, compress_indices)`` where *keep_indices*
        always includes message 0 (the question) and the most recent
        assistant message (so current state is never lost).
        """
        if len(messages) <= 1:
            return list(range(len(messages))), []

        keep: set[int] = {0}
        asst_indices = [
            i for i, m in enumerate(messages) if m.get("role") == "assistant"
        ]

        if not asst_indices:
            return list(range(len(messages))), []

        # Always keep the last assistant (current-state continuity)
        last_asst = asst_indices[-1]
        keep.add(last_asst)
        if last_asst + 1 < len(messages) and messages[last_asst + 1].get("role") == "tool":
            keep.add(last_asst + 1)

        # Score remaining assistant messages by importance
        remaining = [i for i in asst_indices[:-1]]
        scored = [
            (i, score_importance(messages, i, self.question)) for i in remaining
        ]
        scored.sort(key=lambda x: -x[1])

        # Keep top K by importance (K = _keep_recent, minus 1 for last_asst)
        slots = max(0, self._keep_recent - 1)
        for idx, _ in scored[:slots]:
            keep.add(idx)

        # Include context (preceding user message and following tool) for each kept assistant
        for idx in list(keep):
            if idx == 0:
                continue
            # Preceding user message
            for j in range(idx - 1, max(0, idx - 3), -1):
                if messages[j].get("role") == "user":
                    keep.add(j)
                    break
            # Following tool message (if not already kept)
            if idx + 1 < len(messages) and messages[idx + 1].get("role") == "tool":
                keep.add(idx + 1)

        compress = sorted(i for i in range(1, len(messages)) if i not in keep)
        return sorted(keep), compress

    def prune(
        self, messages: list[dict], step: int
    ) -> list[dict]:
        """Prune messages, preserving structured facts.

        Returns pruned messages. Falls back to simple window-based pruning
        if smart compression fails.
        Returns messages unchanged if pruning conditions are not met.
        """
        if not self.should_prune(messages, step):
            return messages

        # Enforce minimum spacing between prunes on the
        # step interval *or* if never pruned before
        if self._last_pruned_step < 0:
            if step < self._prune_after + self._min_prune_interval:
                return messages

        asst_indices = [
            i for i, m in enumerate(messages) if m.get("role") == "assistant"
        ]
        if len(asst_indices) <= 1:
            return messages

        keep_indices, compress_indices = self._select_important_messages(messages)

        if not compress_indices:
            return messages

        old = [messages[i] for i in compress_indices]
        recent = [messages[i] for i in keep_indices]

        if not self._fallback_active:
            result = self._smart_compress(old, step)
            if result is not None:
                compressed_msgs, extracted = result
                pruned = self._build_pruned_messages(
                    messages[0], compressed_msgs, extracted, recent
                )
                self._last_pruned_step = step
                return pruned

        # Fallback: simple window-based truncation
        logger.warning("Smart compression unavailable, using fallback window-based pruning")
        self._fallback_active = True
        return self._fallback_prune(messages, keep_indices)

    def _build_pruned_messages(
        self,
        original_question: dict,
        compressed_msgs: list[dict],
        extracted: list[StructuredFact],
        recent: list[dict],
    ) -> list[dict]:
        """Assemble pruned messages with facts + compressed log + recent."""
        pruned = [original_question]

        facts_msg = build_facts_message_text(self.facts)
        if facts_msg:
            pruned.append({"role": "user", "content": facts_msg})

        summary_content = compressed_msgs[-1]["content"] if compressed_msgs else ""
        summary = summary_content if isinstance(summary_content, str) else str(summary_content)
        if summary:
            pruned.append(
                {"role": "user", "content": f"[Research Progress Summary]\n{summary}"}
            )

        pruned.extend(recent)
        return pruned

    def _smart_compress(
        self, old_messages: list[dict], step: int
    ) -> tuple[list[dict], list[StructuredFact]] | None:
        """Attempt smart compression with fact extraction via provider call."""
        log_lines = [COMPRESSION_PROMPT + "\n\nResearch history to compress:\n"]
        for m in old_messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            if role == "user":
                log_lines.append(f"GUIDANCE: {str(content)[:500]}")
            elif role == "assistant":
                text = str(content or "")[:300]
                tcs = m.get("tool_calls", [])
                if tcs:
                    names = ", ".join(t["name"] for t in tcs)
                    log_lines.append(f"AGENT (called: {names}): {text}")
                else:
                    log_lines.append(f"AGENT: {text}")
            elif role == "tool":
                log_lines.append(f"RESULT: {str(content)[:250]}")

        log_lines.append(
            "\nNow produce extracted facts (one JSON object per line) "
            "and a concise research log."
        )

        try:
            resp = self.provider.call(
                messages=[{"role": "user", "content": "\n".join(log_lines)}],
                tools=None,
                system_prompt=(
                    "You extract structured facts and compress research history. "
                    "Output facts as JSON objects (one per line, each valid JSON). "
                    "Then write a concise research log."
                ),
            )
            if not resp or not resp.text:
                return None

            summary = resp.text.strip()
            if not summary or summary.startswith("ERROR:"):
                return None

            extracted = parse_tagged_facts(summary, step)
            if not extracted:
                extracted = self._parse_json_line_facts(summary, step)

            # Deduplicate against existing facts
            new_facts = []
            for ef in extracted:
                if not any(existing.fact == ef.fact for existing in self.facts):
                    new_facts.append(ef)
                    self.facts.append(ef)

            # Verify compressed summary does not contradict extracted facts
            if new_facts and not self._fact_check(summary, new_facts):
                logger.warning(
                    "Fact-check failed — falling back to window-based pruning"
                )
                return None

            compressed = [old_messages[0]]
            compressed.append(
                {
                    "role": "user",
                    "content": resp.text,
                }
            )
            return compressed, new_facts
        except Exception as exc:
            logger.warning("Smart compression failed: %s", exc)
            return None

    def _fact_check(
        self, summary: str, facts: list[StructuredFact]
    ) -> bool:
        """Verify extracted facts against the compressed summary.

        Returns ``True`` if no contradictions found (or verification
        can't be performed). Returns ``False`` and logs a warning when
        contradictions are detected.
        """
        if not facts or not summary:
            return True

        fact_lines = [
            f"- {f.fact} (source: {f.source}, confidence: {f.confidence})"
            for f in facts
        ]
        prompt = (
            "You are verifying a research summary for factual consistency.\n\n"
            "For each fact below, check whether the research summary "
            "contradicts it. Reply ONLY with 'OK' if no contradictions "
            "are found. If any contradictions exist, list each one as:\n"
            "CONTRADICTION: <fact> vs <summary claim>\n\n"
            "Facts:\n" + "\n".join(fact_lines) + "\n\n"
            "Research Summary:\n" + summary
        )

        try:
            resp = self.provider.call(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                system_prompt="You are a factual consistency checker.",
            )
            if not resp or not resp.text:
                return True
            text = resp.text.strip()
            if text.upper().startswith("OK"):
                return True
            logger.warning("Fact-check found potential issues:\n%s", text)
            return False
        except Exception as exc:
            logger.warning("Fact-check call failed: %s", exc)
            return True

    def _parse_json_line_facts(self, text: str, step: int) -> list[StructuredFact]:
        """Parse JSON-lines facts (fallback if no ``[FACT]`` tags)."""
        facts: list[StructuredFact] = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    data = json.loads(line)
                    if isinstance(data.get("fact"), str) and data["fact"].strip():
                        facts.append(
                            StructuredFact(
                                fact=data["fact"].strip(),
                                source=str(data.get("source", "unknown")),
                                confidence=float(data.get("confidence", 0.5)),
                                extracted_at_step=step,
                            )
                        )
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
        return facts

    def _fallback_prune(
        self,
        messages: list[dict],
        keep_indices: list[int],
    ) -> list[dict]:
        """Simple window-based pruning (no compression call)."""
        summary_msg = "Research history compressed for context management."
        pruned = [messages[0]]

        facts_text = build_facts_message_text(self.facts)
        if facts_text:
            pruned.append({"role": "user", "content": facts_text})

        pruned.append(
            {"role": "user", "content": f"[Research Progress Summary]\n{summary_msg}"}
        )
        for i in keep_indices:
            if i != 0:
                pruned.append(messages[i])
        return pruned

    def get_system_prompt_extension(self) -> str:
        """Return the system prompt snippet for inline fact tagging."""
        return FACT_EXTRACTION_INSTRUCTION
