from __future__ import annotations

import base64
import json
from typing import Any

from openai import OpenAI

from .base import LLMProvider, ProviderResponse, ToolCall


def _to_openai_tools(
    anthropic_tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if not anthropic_tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in anthropic_tools
    ]


_VISION_MODELS = frozenset({
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-vision-preview",
})


class OpenAIProvider(LLMProvider):
    """Calls the OpenAI Chat Completions API with function calling."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
    ):
        self._client = OpenAI(api_key=api_key)
        self.model = model

    @property
    def context_window(self) -> int:
        return 128_000

    def supports_vision(self) -> bool:
        return self.model in _VISION_MODELS or "vision" in self.model

    # ── LLMProvider ────────────────────────────────────────────────────

    @staticmethod
    def _to_openai_content(
        content: str | list[dict[str, Any]],
    ) -> str | list[dict[str, Any]]:
        """Convert canonical content blocks to OpenAI's native format."""
        if isinstance(content, str):
            return content
        blocks: list[dict[str, Any]] = []
        for block in content:
            if block["type"] == "text":
                blocks.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                b64 = base64.b64encode(block["data"]).decode("ascii")
                blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{block['media_type']};base64,{b64}",
                    },
                })
            else:
                blocks.append(block)
        return blocks

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> ProviderResponse:
        openai_messages = list(messages)
        for i, m in enumerate(openai_messages):
            if isinstance(m.get("content"), list):
                m = dict(m)
                m["content"] = self._to_openai_content(m["content"])
                openai_messages[i] = m
        if system_prompt:
            openai_messages.insert(
                0, {"role": "system", "content": system_prompt}
            )

        openai_tools = _to_openai_tools(tools)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": 4096,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            return ProviderResponse(
                text=f"ERROR: {exc}",
                stop_reason="error",
                raw=str(exc),
            )

        msg = response.choices[0].message
        finish = response.choices[0].finish_reason
        text = msg.content or ""
        tool_calls: list[ToolCall] = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    return ProviderResponse(
                        text=text,
                        stop_reason="tool_call_parse_error",
                        raw={
                            "partial_content": msg.content,
                            "failed_tool_name": tc.function.name,
                            "raw_arguments": tc.function.arguments,
                        },
                    )

                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        input=arguments,
                        id=tc.id,
                    )
                )

        usage = None
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        return ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=finish,
            raw=msg,
            usage=usage,
        )

    def estimate_cost(self, usage: dict) -> dict:
        pricing = {
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4-turbo": (10.00, 30.00),
            "gpt-4-vision-preview": (10.00, 30.00),
        }
        rate = pricing.get(self.model, (2.50, 10.00))
        cost = (
            usage.get("prompt_tokens", 0) * rate[0]
            + usage.get("completion_tokens", 0) * rate[1]
        ) / 1_000_000
        return {"cost_usd": round(cost, 6), "cost_label": f"${cost:.4f}"}
