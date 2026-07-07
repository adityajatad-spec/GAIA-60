from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    id: str


@dataclass
class ProviderResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    raw: Any = None


class LLMProvider(ABC):
    model: str = ""

    def supports_vision(self) -> bool:
        """Return True if the model can process image content blocks."""
        return False

    @abstractmethod
    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> ProviderResponse:
        """Send a chat-completion request and return a normalised response.

        *messages* uses a provider-agnostic format:

        .. code-block:: python

            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi", "tool_calls": [
                    {"id": "call_1", "name": "search", "input": {"q": "..."}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            ]

        *tools* uses the Anthropic tool-definition format
        (``name`` / ``description`` / ``input_schema``).

        *system_prompt* is passed separately since some backends (Anthropic)
        require it as a top-level parameter rather than as a ``system`` message.
        """
        ...
