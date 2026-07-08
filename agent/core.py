from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import load_config
from .context_manager import ContextManager, FACT_EXTRACTION_INSTRUCTION
from .planner import Plan, needs_planning, build_plan, PLANNER_SYSTEM_PROMPT
from .providers.base import LLMProvider, ProviderResponse, ToolCall
from tools.code_exec import run_python
from tools.files import read_file, read_image, is_image
from tools.web import web_browse, web_search

_URL_PATTERN = re.compile(r"URL: (https?://\S+)")
_FINAL_ANSWER_RE = re.compile(r"(?i)FINAL[\s_]+ANSWER\s*:?\s*")


def _urls_equivalent(browse_url: str, known_url: str) -> bool:
    """Fuzzy-match a browse URL against a known search-result URL.

    Handles trailing slashes, www vs non-www, http vs https,
    fragments, query params, and sub-path browsing.
    """
    b = _normalize_url(browse_url)
    k = _normalize_url(known_url)
    if b["full"] == k["full"]:
        return True
    # www / non-www
    if b["host_www"] == k["host_www"] and b["path"] == k["path"]:
        return True
    # Same host: one path is a prefix of the other (sub-page browsing)
    if b["host"] == k["host"]:
        if k["path"].startswith(b["path"]) or b["path"].startswith(k["path"]):
            return True
    return False


def _normalize_url(url: str) -> dict:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    return {
        "full": url.rstrip("/").lower(),
        "host": host,
        "host_www": host.replace("www.", ""),
        "path": parsed.path.rstrip("/") or "/",
        "scheme": parsed.scheme,
    }

_MEDIA_TYPE_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}


def _extract_urls(text: str) -> list[str]:
    """Pull out all ``URL: <url>`` lines from a search tool result."""
    return _URL_PATTERN.findall(text)


def _normalize_final_answer_prefix(text: str) -> tuple[bool, str]:
    """Check if *text* contains a final-answer-like prefix.

    Accepts case-insensitive matches and common separator variants
    (underscore, extra whitespace, missing colon). If found, returns
    ``(True, text_with_canonical_prefix)`` so the caller can split on
    the canonical ``"FINAL ANSWER:"``.
    """
    m = _FINAL_ANSWER_RE.search(text)
    if m is None:
        return False, text
    start, end = m.span()
    normalized = text[:start] + "FINAL ANSWER: " + text[end:]
    return True, normalized

SYSTEM_PROMPT = """\
You are a research assistant that answers complex questions by breaking them down into sub-steps, using tools to gather information, and verifying facts before answering.

Guidelines:
- Before acting, explicitly list each sub-step you need to complete. For multi-part questions (e.g., "do X, then find Y, then compute Z"), list every required value before you start.
- Use web_search to find each factual piece. If search snippets are clearly sufficient, do not browse — proceed directly. Only use web_browse when snippets are incomplete or ambiguous.
- When you do browse a URL, check whether the page content is clearly relevant and current to the question. If the page looks like documentation, an API reference, template, or sample/example data rather than live, current information, disregard it and either try a more specific search or answer from what you already have.
- Verify factual claims via tools rather than relying on memory.
- Use code_exec for ALL mathematical computations. Never compute values mentally or through estimation — always use Python.
- Track your progress: after each step, note what you've found and what remains.
- Before producing FINAL ANSWER, verify you have answered EVERY part of the question. If the question asks for multiple values (e.g., "find X, compute Y, check Z"), report ALL of them — do not skip any.
- For chain questions where a value depends on a previous result, re-state all intermediate values in the final answer. Every number requested in the question must appear in the answer.

Important: when using the code_exec tool, always remember to use print() to display any value you want to see. For example, write ``print(847 * 293 / 17)`` instead of just ``847 * 293 / 17``. If you write a bare expression on the last line the tool will automatically capture and print it (like a Jupyter notebook), but explicit print() calls are more reliable.

Extract key factual findings as structured facts using this format:
[FACT] {"fact": "...", "source": "...", "confidence": 0.0} [/FACT]

Every response you send must end either with tool calls or with a final answer in the following exact format. There are no exceptions. If you are uncertain, state your best guess in the required format rather than hedging in prose.

FINAL ANSWER: <answer>\
"""

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search",
        "description": "Search the web for information on a query",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "browse",
        "description": "Fetch and read the content of a URL",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to browse",
                }
            },
            "required": ["url"],
        },
    },
    {
        "name": "code_exec",
        "description": "Execute Python code and return the result",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "read_attached_file",
        "description": "Read the content of an attached file. Supports PDF, DOCX, XLSX, images (text extraction), and plain text. Returns the file content or an error message.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the attached file",
                }
            },
            "required": ["path"],
        },
    },
]


# ── Tool handlers ───────────────────────────────────────────────────────────


def _handle_search(query: str) -> str:
    results = web_search(query)
    if not results:
        return "No results found."
    if "error" in results[0]:
        return results[0]["error"]
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. {r['title']}\n   URL: {r['url']}\n   Snippet: {r['snippet']}"
        )
    return "\n\n".join(lines)


def _handle_code_exec(code: str) -> str:
    result = run_python(code)
    if result["error"]:
        return f"ERROR: {result['error']}"
    output = ""
    if result["stdout"]:
        output += result["stdout"]
    if result["stderr"]:
        if output:
            output += "\n"
        output += result["stderr"]
    if not result["success"]:
        output = f"ERROR (exit code != 0):\n{output}"
    if result.get("warning"):
        if output:
            output += "\n"
        output += f"WARNING: {result['warning']}"
    return output if output else "(no output)"


_TOOL_HANDLERS: dict[str, Any] = {
    "search": _handle_search,
    "browse": web_browse,
    "code_exec": _handle_code_exec,
    "read_attached_file": read_file,
}


# ── Provider factory ────────────────────────────────────────────────────────


def _provider_from_config(cfg: dict) -> LLMProvider:
    p = cfg["provider"]
    if p == "anthropic":
        from .providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=cfg["anthropic_api_key"], model=cfg["model"]
        )
    if p == "ollama":
        from .providers.ollama_provider import OllamaProvider

        return OllamaProvider(model=cfg["model"], host=cfg["ollama_host"])
    if p == "openai":
        from .providers.openai_provider import OpenAIProvider

        return OpenAIProvider(api_key=cfg["openai_api_key"], model=cfg["model"])
    raise ValueError(f"Unsupported provider: {p}")


# ── ResearchAgent ───────────────────────────────────────────────────────────


class ResearchAgent:
    def __init__(
        self,
        question: str,
        files: list[str] | None = None,
        provider: LLMProvider | None = None,
        model: str | None = None,
        verify_provider: LLMProvider | None = None,
    ):
        self.question = question
        self.files = files or []
        self.trajectory: list[dict[str, Any]] = []
        self.step_budget = 15
        self._provider = provider
        self._model = model
        self._format_retried = False
        self._known_urls: set[str] = set()
        self._retried_steps: set[int] = set()
        self._budget_warning_given = False
        self._verify_provider = verify_provider
        self._usage_by_provider: list[tuple[LLMProvider, dict]] = []
        self._ctx_manager = ContextManager(
            question=question,
            provider=provider,
            step_budget=self.step_budget,
        )
        self._plan: Plan | None = None

    def _build_initial_messages(
        self, provider: LLMProvider
    ) -> list[dict[str, Any]]:
        if not self.files:
            return [{"role": "user", "content": self.question}]

        text_parts: list[str] = []
        image_blocks: list[dict[str, Any]] = []
        vision_supported = provider.supports_vision()

        for fp in self.files:
            p = Path(fp)
            ext = p.suffix.lower()
            if not p.is_file():
                text_parts.append(f"[File not found: {fp}]")
                continue

            if ext in (".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml"):
                try:
                    text_parts.append(
                        f"--- {p.name} ---\n{p.read_text(encoding='utf-8')}"
                    )
                except Exception as exc:
                    text_parts.append(f"[Error reading {p.name}: {exc}]")

            elif is_image(ext):
                if vision_supported:
                    try:
                        data = p.read_bytes()
                        media = _MEDIA_TYPE_MAP.get(ext, "image/png")
                        image_blocks.append({
                            "type": "image", "data": data, "media_type": media,
                        })
                    except Exception as exc:
                        text_parts.append(f"[Error reading image {p.name}: {exc}]")
                else:
                    text_parts.append(
                        f"--- {p.name} (image) ---\n{read_image(fp)}"
                    )

            else:
                content = read_file(fp)
                if content.startswith("ERROR:"):
                    text_parts.append(f"[{p.name}: {content}]")
                else:
                    text_parts.append(f"--- {p.name} ---\n{content}")

        parts: list[Any] = []
        if text_parts:
            combined = "\n\n".join(text_parts)
            parts.append({
                "type": "text",
                "text": f"Attached files:\n\n{combined}\n\nQuestion: {self.question}",
            })
        else:
            parts.append({"type": "text", "text": self.question})

        parts.extend(image_blocks)

        if len(parts) == 1:
            return [{"role": "user", "content": parts[0]["text"]}]
        return [{"role": "user", "content": parts}]

    def run(self, step_callback=None) -> dict[str, Any]:
        provider: LLMProvider
        if self._provider is not None:
            provider = self._provider
        else:
            provider = _provider_from_config(load_config())

        messages = self._build_initial_messages(provider)

        # ── Planning phase (multi-step questions only) ─────────────
        if needs_planning(self.question):
            try:
                plan_resp = provider.call(
                    messages=[{"role": "user", "content": self.question}],
                    tools=None,
                    system_prompt=PLANNER_SYSTEM_PROMPT,
                )
                if plan_resp.usage:
                    self._usage_by_provider.append((provider, plan_resp.usage))
                if plan_resp.text:
                    self._plan = build_plan(plan_resp.text)
                    if self._plan.steps:
                        messages.append({
                            "role": "user",
                            "content": self._plan.to_prompt_string(),
                        })
            except Exception as exc:
                # Planning failed — proceed without plan
                pass

        for step in range(self.step_budget):
            if (
                not self._budget_warning_given
                and step >= int(self.step_budget * 0.7)
            ):
                self._budget_warning_given = True
                messages.append({
                    "role": "user",
                    "content": (
                        "Budget warning: you have used most of your allowed"
                        f" {self.step_budget} steps. Please wrap up with a"
                        " final answer now."
                    ),
                })

            response: ProviderResponse = provider.call(
                messages=messages,
                tools=TOOL_DEFINITIONS,
                system_prompt=SYSTEM_PROMPT,
            )
            if response.usage:
                self._usage_by_provider.append((provider, response.usage))

            # Record the model message in the trajectory
            msg_entry = {
                "step": step,
                "role": "assistant",
                "content": response.text,
                "tool_calls": [
                    {"name": tc.name, "input": tc.input, "id": tc.id}
                    for tc in response.tool_calls
                ],
                "stop_reason": response.stop_reason,
            }
            self.trajectory.append(msg_entry)

            # Build the canonical assistant message for the next turn
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": response.text,
            }
            if response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "name": tc.name, "input": tc.input}
                    for tc in response.tool_calls
                ]
            messages.append(assistant_msg)

            self._ctx_manager.record_facts_from_response(response.text, step)

            if step_callback:
                step_callback({"type": "assistant", "data": msg_entry})

            full_text = response.text

            # ── Handle tool-call parse errors ─────────────────────────
            if response.stop_reason == "tool_call_parse_error":
                nudge = (
                    "I asked you to use a tool, but the arguments were "
                    "malformed JSON. Please call the tool again with "
                    "valid JSON arguments."
                )
                messages.append({"role": "user", "content": nudge})
                continue

            # ── Check for final answer ────────────────────────────────
            found, normalized = _normalize_final_answer_prefix(full_text)
            if found:
                # Plan completeness check: if steps remain, nudge once
                if (
                    self._plan is not None
                    and not self._plan.all_complete()
                ):
                    incomplete = self._plan.incomplete_steps()
                    step_desc = "\n".join(
                        f"  - Step {s.step}: {s.description}"
                        for s in incomplete
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "You produced a final answer but the following "
                            f"steps are not yet complete:\n{step_desc}\n\n"
                            "Please complete them first."
                        ),
                    })
                    continue

                answer = normalized.split("FINAL ANSWER:", 1)[1].strip()
                verify_provider = self._verify_provider or provider
                verified = self._verify(verify_provider, answer)
                return {
                    "answer": verified,
                    "trajectory": self.trajectory,
                    "steps_used": step + 1,
                    "format_noncompliant": False,
                    "verification_performed": True,
                    "usage": self._build_usage(),
                }

            # ── No tool calls → format compliance check ───────────────
            if not response.tool_calls:
                if not self._format_retried:
                    self._format_retried = True
                    if full_text.strip():
                        nudge = (
                            'Your last response did not start with the exact '
                            'text "FINAL ANSWER: " (with a space, not an '
                            "underscore). Please resend your answer starting "
                            'with exactly "FINAL ANSWER: " followed by your '
                            "answer."
                        )
                    else:
                        nudge = (
                            "You produced an empty response. Please answer "
                            'the question using the format: '
                            'FINAL ANSWER: <answer>'
                        )
                    messages.append({"role": "user", "content": nudge})
                    messages = self._prune_context(messages, provider, step)
                    continue
                # Second attempt — accept whatever we got
                is_empty = not full_text.strip()
                self.trajectory[-1]["format_noncompliant"] = True
                if is_empty:
                    self.trajectory[-1]["empty_response_failure"] = True
                return {
                    "answer": full_text if full_text else "(empty response)",
                    "trajectory": self.trajectory,
                    "steps_used": step + 1,
                    "format_noncompliant": True,
                    "empty_response_failure": is_empty,
                    "usage": self._build_usage(),
                }

            # ── Execute tool calls ────────────────────────────────────
            for tc in response.tool_calls:
                output = self._execute_tool(tc.name, tc.input, step)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    }
                )

            # ── Update plan progress ──────────────────────────────────
            if self._plan is not None:
                all_tool_text = " ".join(
                    str(m.get("content", ""))
                    for m in messages
                    if m.get("role") == "tool"
                )
                self._plan.mark_complete_by_output(all_tool_text)
                messages.append({
                    "role": "user",
                    "content": self._plan.to_progress_string(),
                })

            # ── Retry on tool failure (once per step) ────────────────
            tool_errors = [
                e
                for e in self.trajectory
                if e.get("step") == step
                and isinstance(e.get("output"), str)
                and e["output"].startswith("ERROR:")
            ]
            if tool_errors and step not in self._retried_steps:
                self._retried_steps.add(step)
                details = "\n".join(
                    f"  - {e.get('tool', '?')}: {e['output'][:200]}"
                    for e in tool_errors
                )
                nudge = (
                    f"The following tool call(s) failed:\n{details}\n\n"
                    "Please try a different approach or fix the inputs."
                )
                messages.append({"role": "user", "content": nudge})
                if step_callback:
                    step_callback({
                        "type": "tool_retry",
                        "data": {"step": step, "errors": tool_errors},
                    })
                messages = self._prune_context(messages, provider, step)
                continue

            messages = self._prune_context(messages, provider, step)

        # Budget exhausted
        return {
            "answer": "MAX_STEPS_REACHED",
            "trajectory": self.trajectory,
            "steps_used": self.step_budget,
            "format_noncompliant": False,
            "usage": self._build_usage(),
        }

    def _execute_tool(self, name: str, inp: dict, step: int) -> str:
        # ── Browse guard: fuzzy-match URL against prior search results ──
        guard_warning = ""
        if name == "browse" and self._known_urls:
            url = inp.get("url", "")
            allowed = any(
                _urls_equivalent(url, k) for k in self._known_urls
            )
            if not allowed:
                guard_warning = (
                    "Note: this URL was not directly returned by any "
                    "search result — browsing anyway.\n\n"
                )

        handler = _TOOL_HANDLERS.get(name)
        if handler is None:
            output = f"ERROR: unknown tool '{name}'"
        else:
            try:
                output = handler(**inp)
            except Exception as exc:
                output = f"ERROR: {exc}"

        # Prepend guard warning if applicable
        if guard_warning:
            output = guard_warning + output

        # ── After search, collect URLs for the browse guard ──────────
        if name == "search":
            self._known_urls.update(_extract_urls(output))

        self.trajectory.append(
            {
                "step": step,
                "tool": name,
                "input": inp,
                "output": output,
            }
        )
        return output

    def _summarize_trajectory(self) -> str:
        lines: list[str] = []
        for entry in self.trajectory:
            if "tool" in entry:
                out = entry.get("output", "")
                tool_name = entry.get("tool", "?")
                if out.startswith("ERROR:"):
                    lines.append(f"[{tool_name}] FAILED: {out[:300]}")
                else:
                    lines.append(f"[{tool_name}] {out[:500]}")
            elif entry.get("role") == "assistant" and entry.get("content"):
                lines.append(f"[assistant] {entry['content'][:300]}")
        return "\n\n".join(lines)

    def _verify(self, verify_provider: LLMProvider, answer: str) -> str:
        traj_summary = self._summarize_trajectory()

        # Detect whether the question is multi-part by looking for
        # numbered sub-steps or multiple request patterns
        multi_part = bool(
            re.search(
                r"(?:\(\d+\)\s|Do the following in order|"
                r"in order.*then|find.*find|find.*compute|"
                r"compute.*determine|computing.*then)",
                self.question,
                re.IGNORECASE,
            )
        )

        coverage_check = (
            "List all parts of the question and whether each is answered.\n"
            "Then produce the final verified answer.\n"
            "FINAL ANSWER: "
            if multi_part
            else (
                "If the answer is correct and complete, repeat it exactly. "
                "If incorrect or incomplete, provide a corrected version.\n"
                "FINAL ANSWER: "
            )
        )

        verify_prompt = (
            f"Original question: {self.question}\n\n"
            f"Research trajectory:\n{traj_summary}\n\n"
            f"Proposed answer: {answer}\n\n"
            "Please verify this answer. Does it completely address all parts of the question? "
            "Is it consistent with the evidence gathered?\n"
            f"{coverage_check}"
        )
        try:
            response = verify_provider.call(
                messages=[{"role": "user", "content": verify_prompt}],
                tools=None,
                system_prompt=SYSTEM_PROMPT,
            )
            if response.usage:
                self._usage_by_provider.append((verify_provider, response.usage))
            found, normalized = _normalize_final_answer_prefix(response.text)
            if found:
                return normalized.split("FINAL ANSWER:", 1)[1].strip()
            # Fallback: model didn't use FINAL ANSWER: format.
            # Use the response text directly as the verified answer.
            text = response.text.strip()
            if text:
                return text
        except Exception:
            pass
        return answer

    def _build_usage(self) -> dict:
        total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        total_cost = 0.0
        for provider, usage in self._usage_by_provider:
            for k in total:
                total[k] += usage.get(k, 0)
            cost_info = provider.estimate_cost(usage)
            total_cost += cost_info["cost_usd"]
        label = f"${total_cost:.4f}" if total_cost > 0 else "local (no cost)"
        return {
            **total,
            "cost_usd": round(total_cost, 6),
            "cost_label": label,
        }

    def _prune_context(
        self, messages: list[dict[str, Any]], provider: LLMProvider, step: int
    ) -> list[dict[str, Any]]:
        """Delegate to ContextManager. Kept for backward-compatible test mocking."""
        return self._ctx_manager.prune(messages, step)

    def run_with_voting(self, n: int = 3, step_callback=None) -> dict:
        from collections import Counter

        results: list[dict] = []
        for i in range(n):
            agent = ResearchAgent(
                question=self.question,
                files=self.files,
                provider=self._provider,
                verify_provider=self._verify_provider,
            )
            agent.step_budget = self.step_budget
            result = agent.run(step_callback=step_callback)
            results.append(result)

        answers = [r["answer"] for r in results]
        counter = Counter(answers)
        majority = counter.most_common(1)[0][0]

        for r in results:
            if r["answer"] == majority:
                r["votes"] = dict(counter)
                r["n_runs"] = n
                return r

        return results[0]
