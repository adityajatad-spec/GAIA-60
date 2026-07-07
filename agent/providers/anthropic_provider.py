from __future__ import annotations

import base64
from typing import Any

from anthropic import Anthropic

from .base import LLMProvider, ProviderResponse, ToolCall


class AnthropicProvider(LLMProvider):
    """Calls the Anthropic Messages API. Used for GAIA-benchmark runs."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
    ):
        self._client = Anthropic(api_key=api_key)
        self.model = model

    def supports_vision(self) -> bool:
        return True

    # ── LLMProvider ────────────────────────────────────────────────────

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> ProviderResponse:
        native_messages = self._to_native_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": native_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = tools

        response = self._client.messages.create(**kwargs)

        return self._from_native_response(response)

    # ── Format conversion ──────────────────────────────────────────────

    @staticmethod
    def _to_native_content(
        content: str | list[dict[str, Any]],
    ) -> str | list[dict[str, Any]]:
        """Convert canonical content (text string or content-block list) to
        Anthropic's native content format."""
        if isinstance(content, str):
            return content
        blocks: list[dict[str, Any]] = []
        for block in content:
            if block["type"] == "text":
                blocks.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                b64 = base64.b64encode(block["data"]).decode("ascii")
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block["media_type"],
                        "data": b64,
                    },
                })
            else:
                blocks.append(block)
        return blocks

    @staticmethod
    def _to_native_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert the canonical message format to Anthropic's format."""
        native: list[dict[str, Any]] = []

        for msg in messages:
            role = msg["role"]

            if role == "user":
                native.append({
                    "role": "user",
                    "content": AnthropicProvider._to_native_content(msg["content"]),
                })

            elif role == "assistant":
                content: list[dict[str, Any]] = []
                if msg.get("content"):
                    content.append(
                        {"type": "text", "text": msg["content"]}
                    )
                for tc in msg.get("tool_calls", []):
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["input"],
                        }
                    )
                native.append({"role": "assistant", "content": content})

            elif role == "tool":
                native.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg["tool_call_id"],
                                "content": msg["content"],
                            }
                        ],
                    }
                )

        return native

    @staticmethod
    def _from_native_response(response: Any) -> ProviderResponse:
        text_blocks = [b for b in response.content if b.type == "text"]
        tool_blocks = [b for b in response.content if b.type == "tool_use"]

        text = " ".join(b.text for b in text_blocks)
        tool_calls = [
            ToolCall(name=b.name, input=b.input, id=b.id)
            for b in tool_blocks
        ]

        return ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            raw=response,
        )
