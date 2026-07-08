from __future__ import annotations

import base64
import json
import uuid
from typing import Any

import requests
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


def _to_openai_tool_calls(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert canonical tool calls to OpenAI API format.

    Canonical:  [{"id": str, "name": str, "input": dict}]
    OpenAI:     [{"id": str, "type": "function",
                  "function": {"name": str, "arguments": str}}]
    """
    result: list[dict[str, Any]] = []
    for tc in tool_calls:
        result.append(
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["input"], ensure_ascii=False),
                },
            }
        )
    return result


class OllamaProvider(LLMProvider):
    """Calls a local Ollama server via its OpenAI-compatible endpoint.

    Handles common failure modes of small local models:
    - malformed JSON in tool-call arguments → returns ``stop_reason``
      ``"tool_call_parse_error"`` instead of crashing
    - missing model or unreachable server → raises a clear error message
      before the first call
    """

    def __init__(
        self,
        model: str = "qwen3:8b-64k",
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = host.rstrip("/")
        self._client = OpenAI(base_url=f"{self.base_url}/v1", api_key="ollama")
        self._checked = False
        self._vision_checked = False
        self._vision_supported = False
        self._ctx_window = 8192

    # ── LLMProvider ────────────────────────────────────────────────────

    def _parse_tool_calls_from_text(self, text: str) -> list[ToolCall]:
        """Scan *text* for JSON tool calls using brace-depth matching.

        Small local models often emit tool calls as text JSON instead of
        using the structured API. Accepts both OpenAI wire format
        (``name`` + ``arguments``) and canonical format
        (``name`` + ``input``).
        """
        results: list[ToolCall] = []
        i = 0
        while True:
            brace_start = text.find("{", i)
            if brace_start == -1:
                break
            depth = 0
            j = brace_start
            while j < len(text):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if depth != 0:
                i = brace_start + 1
                continue
            raw = text[brace_start : j + 1]
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                i = brace_start + 1
                continue
            name = obj.get("name")
            arguments = obj.get("arguments") or obj.get("input")
            if not name or not arguments or not isinstance(arguments, dict):
                i = brace_start + 1
                continue
            results.append(
                ToolCall(
                    name=name,
                    input=arguments,
                    id=f"text_{uuid.uuid4().hex[:12]}",
                )
            )
            i = j + 1
        return results

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
        self._ensure_ready()

        openai_messages = list(messages)
        for i, m in enumerate(openai_messages):
            m = dict(m)
            if isinstance(m.get("content"), list):
                m["content"] = self._to_openai_content(m["content"])
            if m.get("role") == "assistant" and "tool_calls" in m:
                m["tool_calls"] = _to_openai_tool_calls(m["tool_calls"])
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
        else:
            # Fallback: scan text for JSON tool calls
            tool_calls = self._parse_tool_calls_from_text(text)

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
            stop_reason=finish if not tool_calls else "tool_use",
            raw=msg,
            usage=usage,
        )

    # ── Vision check ────────────────────────────────────────────────────

    @property
    def context_window(self) -> int:
        if not self._vision_checked:
            self._vision_supported = self._check_vision()
            self._vision_checked = True
        return self._ctx_window

    def supports_vision(self) -> bool:
        """Check whether the loaded model has a vision projector.

        Queries ``/api/show`` once and caches the result.
        """
        if not self._vision_checked:
            self._vision_supported = self._check_vision()
            self._vision_checked = True
        return self._vision_supported

    def _check_vision(self) -> bool:
        try:
            resp = requests.post(
                f"{self.base_url}/api/show",
                json={"model": self.model},
                timeout=5,
            )
            resp.raise_for_status()
            info = resp.json()
            model_info = info.get("model_info", {})
            # Extract context window from model metadata if available
            ctx = model_info.get(
                "llama.context_length",
                model_info.get("bert.context_length", 8192),
            )
            if isinstance(ctx, (int, float)):
                self._ctx_window = int(ctx)
            return "projector_info" in info
        except requests.RequestException:
            return False

    # ── Readiness check ────────────────────────────────────────────────

    def _ensure_ready(self) -> None:
        if self._checked:
            return

        # 1. Server reachable?
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.RequestException:
            raise ConnectionError(
                "Ollama server is not reachable at "
                f"{self.base_url}.\n"
                "Make sure it is running:  ollama serve"
            )

        # 2. Model pulled?
        model_names = [m["name"] for m in resp.json().get("models", [])]
        if self.model not in model_names:
            raise ValueError(
                f"Model '{self.model}' is not available locally.\n"
                f"Pull it first:  ollama pull {self.model}"
            )

        self._checked = True
