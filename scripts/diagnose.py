#!/usr/bin/env python3
"""
Standalone diagnostic for the Phase 3 provider abstraction layer.

Staged self-check that runs sequentially and stops at the first failure.
Usage:
    python scripts/diagnose.py            # run all stages
    python scripts/diagnose.py --stage 3   # start from stage 3
    python scripts/diagnose.py --stage 3 --skip-previous  # skip straight to stage 3
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import textwrap
import traceback
from pathlib import Path
from typing import Any

# ── Ensure the project root is on sys.path ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── ANSI colour helpers ────────────────────────────────────────────────


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m"


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


# ── Stage results accumulator ──────────────────────────────────────────

StageResult = tuple[str, bool, str]  # (stage_name, passed, one-line-diagnosis)
_results: list[StageResult] = []


def _record(name: str, passed: bool, diagnosis: str) -> None:
    _results.append((name, passed, diagnosis))


# ── Helpers ────────────────────────────────────────────────────────────


def _section(label: str) -> None:
    print()
    print(_bold(f"═══ {label} ═══"))
    print()


def _fail_and_stop(diagnosis: str) -> None:
    print(_red(f"  ✖ FAIL — {diagnosis}"))
    sys.exit(1)


def _ok(label: str) -> None:
    print(_green(f"  ✔ {label}"))


def _load_dotenv() -> dict[str, str]:
    """Minimal dotenv loader so we don't import agent.config yet."""
    env_file = PROJECT_ROOT / ".env"
    result: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip().strip('"').strip("'")
    return result


# ═══════════════════════════════════════════════════════════════════════
# Stage 1 — Import check
# ═══════════════════════════════════════════════════════════════════════


def stage1_imports() -> bool:
    _section("Stage 1 — Import check")

    modules: list[tuple[str, str]] = [
        ("agent.providers.base", "LLMProvider, ProviderResponse, ToolCall"),
        ("agent.providers.ollama_provider", "OllamaProvider"),
        ("agent.providers.anthropic_provider", "AnthropicProvider"),
        ("agent.providers.openai_provider", "OpenAIProvider"),
        ("agent.core", "ResearchAgent"),
        ("tools.web", "web_search, web_browse"),
    ]

    failed = False
    for modname, names in modules:
        try:
            mod = importlib.import_module(modname)
            for n in names.replace(" ", "").split(","):
                if not hasattr(mod, n):
                    print(
                        f"  {_red('✖')} {modname} is missing attribute "
                        f"'{n}'"
                    )
                    failed = True
                    break
            else:
                _ok(f"{modname} → {names}")
        except SyntaxError as e:
            print(
                f"  {_red('✖')} SyntaxError in {modname}: "
                f"{e.msg} (line {e.lineno})"
            )
            print(f"    File: {e.filename}")
            failed = True
        except ImportError as e:
            print(f"  {_red('✖')} ImportError loading {modname}: {e}")
            failed = True
        except Exception as e:
            print(
                f"  {_red('✖')} Unexpected error loading {modname}: "
                f"{type(e).__name__}: {e}"
            )
            failed = True

    if failed:
        print(
            _yellow(
                "\n  ⚠ Fix the import errors above, then re-run "
                "this script."
            )
        )
        _record("Import check", False, "One or more imports failed")
        return False

    _ok("All imports resolved.")
    _record("Import check", True, "")
    return True


# ═══════════════════════════════════════════════════════════════════════
# Stage 2 — Ollama reachability
# ═══════════════════════════════════════════════════════════════════════


def stage2_reachability() -> bool:
    _section("Stage 2 — Ollama server reachability")

    env = _load_dotenv()
    host = env.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    tags_url = f"{host}/api/tags"

    import requests as req

    try:
        resp = req.get(tags_url, timeout=5)
        resp.raise_for_status()
    except req.ConnectionError:
        print(
            _red(
                "  ✖ Ollama isn't reachable at "
                f"{host}/api/tags\n\n"
                "    Start the server with:  ollama serve\n"
                "    Then re-run this script."
            )
        )
        _record(
            "Ollama reachability",
            False,
            f"Connection refused to {host}",
        )
        return False
    except req.Timeout:
        print(
            _red(
                f"  ✖ Timed out after 5s connecting to {host}\n\n"
                "    Check that Ollama is running and the host/port "
                "are correct.\n"
                "    OLLAMA_HOST in .env is currently: "
                f"{env.get('OLLAMA_HOST', '(not set)')}"
            )
        )
        _record(
            "Ollama reachability",
            False,
            f"Timeout connecting to {host}",
        )
        return False
    except req.RequestException as e:
        print(
            _red(
                f"  ✖ HTTP error from {tags_url}: {e}\n\n"
                "    This may be a proxy or network issue. "
                "Check OLLAMA_HOST in .env."
            )
        )
        _record("Ollama reachability", False, f"HTTP error: {e}")
        return False
    except Exception as e:
        print(_red(f"  ✖ Unexpected error: {type(e).__name__}: {e}"))
        _record("Ollama reachability", False, str(e))
        return False

    _ok(f"Server is running at {host}")
    _record("Ollama reachability", True, "")
    return True


# ═══════════════════════════════════════════════════════════════════════
# Stage 3 — Model availability
# ═══════════════════════════════════════════════════════════════════════


def stage3_model() -> bool:
    _section("Stage 3 — Model availability")

    env = _load_dotenv()
    model = env.get("OLLAMA_MODEL", "qwen3:8b-64k")
    host = env.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

    import requests as req

    try:
        resp = req.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
    except Exception as e:
        print(_red(f"  ✖ Could not list models: {e}"))
        _record("Model availability", False, str(e))
        return False

    if model not in models:
        print(
            _red(
                f"  ✖ Model '{model}' is not available locally.\n\n"
                f"    Available models: {', '.join(models) or '(none)'}\n\n"
                f"    Pull it with:\n"
                f"      ollama pull {model}\n"
            )
        )
        _record(
            "Model availability",
            False,
            f"'{model}' not found; run: ollama pull {model}",
        )
        return False

    if model not in models:
        # Also check for partial matches (e.g. qwen3:8b-64k-instruct
        # vs qwen3:8b-64k)
        partial = [m for m in models if m.split(":")[0] == model.split(":")[0]]
        if partial:
            print(
                _yellow(
                    f"  ⚠ Model '{model}' not found, but "
                    f"{' and '.join(partial)} "
                    "is/are. Did you mean one of those?\n"
                    "    Update OLLAMA_MODEL in .env or pull the "
                    "correct tag:\n"
                    f"      ollama pull {model}"
                )
            )
            _record(
                "Model availability",
                False,
                f"'{model}' not found (partial match: {partial})",
            )
            return False

    _ok(f"Model '{model}' is available")
    _record("Model availability", True, "")
    return True


# ═══════════════════════════════════════════════════════════════════════
# Stage 4 — Bare call (no tools)
# ═══════════════════════════════════════════════════════════════════════


def stage4_bare_call() -> bool:
    _section("Stage 4 — Bare call (no tools)")

    from agent.providers.base import ProviderResponse
    from agent.providers.ollama_provider import OllamaProvider

    env = _load_dotenv()
    model = env.get("OLLAMA_MODEL", "qwen3:8b-64k")
    host = env.get("OLLAMA_HOST", "http://localhost:11434")

    provider = OllamaProvider(model=model, host=host)

    try:
        resp = provider.call(
            messages=[{"role": "user", "content": "Say hello in one word."}],
            tools=None,
            system_prompt="You are a helpful assistant.",
        )
    except Exception as e:
        print(
            _red(
                f"  ✖ provider.call() raised an exception:\n"
                f"    {type(e).__name__}: {e}"
            )
        )
        traceback.print_exc()
        _record("Bare call", False, f"Exception: {e}")
        return False

    checks: list[str] = []

    # Check it's a ProviderResponse
    if isinstance(resp, ProviderResponse):
        checks.append("return type is ProviderResponse ✓")
    else:
        checks.append(
            _red(
                "return type is "
                f"{type(resp).__name__}, expected ProviderResponse"
            )
        )
        _record("Bare call", False, "Wrong return type")
        print("\n".join(f"  {c}" for c in checks))
        print(f"\n  Raw response: {resp!r}")
        return False

    # Check text is non-empty
    if resp.text and len(resp.text.strip()) > 0:
        checks.append(f"text is non-empty ({len(resp.text)} chars) ✓")
    else:
        checks.append(_red("text is empty — model returned no content"))

    # Check tool_calls is empty
    if not resp.tool_calls:
        checks.append("tool_calls is empty ✓")
    else:
        checks.append(
            _yellow(
                f"tool_calls has {len(resp.tool_calls)} entries "
                "(unexpected but not a failure)"
            )
        )

    # Check stop_reason
    if resp.stop_reason in ("stop", "end_turn"):
        checks.append(
            f"stop_reason is '{resp.stop_reason}' ✓"
        )
    else:
        checks.append(
            _yellow(
                f"stop_reason is '{resp.stop_reason}' "
                "(expected 'stop' or 'end_turn')"
            )
        )

    print("\n".join(f"  {c}" for c in checks))

    if not resp.text or len(resp.text.strip()) == 0:
        _record("Bare call", False, "Empty response text")
        return False

    print(f"\n  Model reply: {resp.text}")
    _record("Bare call", True, "")
    return True


# ═══════════════════════════════════════════════════════════════════════
# Stage 5 — Tool-call call
# ═══════════════════════════════════════════════════════════════════════


def stage5_tool_call() -> bool:
    _section("Stage 5 — Tool-call invocation")

    from agent.providers.base import ProviderResponse
    from agent.providers.ollama_provider import OllamaProvider

    env = _load_dotenv()
    model = env.get("OLLAMA_MODEL", "qwen3:8b-64k")
    host = env.get("OLLAMA_HOST", "http://localhost:11434")

    provider = OllamaProvider(model=model, host=host)

    weather_tool = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. Paris",
                    }
                },
                "required": ["city"],
            },
        }
    ]

    try:
        resp = provider.call(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "What's the weather in Paris? "
                        "Use the get_weather tool."
                    ),
                }
            ],
            tools=weather_tool,
            system_prompt=(
                "You have access to the get_weather tool. "
                "Always use it when asked about weather."
            ),
        )
    except Exception as e:
        print(
            _red(
                f"  ✖ provider.call() with tools raised:\n"
                f"    {type(e).__name__}: {e}"
            )
        )
        traceback.print_exc()
        _record("Tool-call", False, f"Exception: {e}")
        return False

    if not isinstance(resp, ProviderResponse):
        print(
            _red(
                f"  ✖ Expected ProviderResponse, got "
                f"{type(resp).__name__}"
            )
        )
        print(f"  Raw: {resp!r}")
        _record("Tool-call", False, "Wrong return type")
        return False

    # Outcome (a): did the model call the tool?
    if not resp.tool_calls:
        print(
            _yellow(
                "  ⚠ The model did NOT call the get_weather tool.\n"
                "    This can mean:\n"
                "      (1) The model doesn't support tool-calling "
                "(check that it's a function-calling model)\n"
                "      (2) The prompt wasn't clear enough\n"
                "      (3) The tool definition format is wrong\n"
                f"\n    Stop reason: {resp.stop_reason}"
            )
        )
        if resp.text:
            print(f"    Text reply: {resp.text}")
        _record(
            "Tool-call",
            False,
            f"Model didn't call the tool; stop_reason={resp.stop_reason}",
        )
        return False

    print(
        f"  ✔ Model called {len(resp.tool_calls)} tool(s): "
        + ", ".join(tc.name for tc in resp.tool_calls)
    )

    # Outcome (b): did the arguments parse as valid JSON?
    all_valid = all(tc.input for tc in resp.tool_calls)
    if all_valid:
        print("  ✔ All tool arguments parsed as valid JSON ✓")
    else:
        print(
            _red(
                "  ✖ Some tool arguments failed JSON parsing "
                "(should not reach here if stop_reason is set)"
            )
        )

    # Outcome (c): defensive parsing check — stop_reason
    if resp.stop_reason == "tool_call_parse_error":
        print(
            _yellow(
                "  ⚠ Defensive parsing caught malformed JSON "
                "(stop_reason = tool_call_parse_error)\n"
                "    The model sent bad arguments but we handled it "
                "gracefully instead of crashing."
            )
        )
        if resp.raw:
            raw = resp.raw
            if isinstance(raw, dict):
                failed = raw.get("failed_tool_name", "?")
                bad_args = raw.get("raw_arguments", "?")
                print(
                    f"    Failed tool: {failed}\n"
                    f"    Raw args:   {bad_args}"
                )
        _record(
            "Tool-call",
            False,
            "Model sent malformed JSON (defensive parsing caught it)",
        )
        return False

    if resp.tool_calls:
        print()
        for tc in resp.tool_calls:
            inp_str = json.dumps(tc.input, ensure_ascii=False)
            print(f"  [{tc.id}] {tc.name}({inp_str})")
        _record("Tool-call", True, "")
        return True

    _record("Tool-call", False, "Unexpected state")
    return False


# ═══════════════════════════════════════════════════════════════════════
# Stage 6 — Full ResearchAgent loop
# ═══════════════════════════════════════════════════════════════════════


def _trajectory_summary(trajectory: list[dict]) -> str:
    lines: list[str] = []
    for entry in trajectory:
        step = entry.get("step", "?")
        role = entry.get("role", entry.get("tool", "?"))
        text = entry.get("content", entry.get("output", ""))
        tcs = entry.get("tool_calls", [])
        stop = entry.get("stop_reason")
        parts = [f"[step {step}]", f"role={role}"]
        if stop:
            parts.append(f"stop={stop}")
        if text:
            # Truncate long text
            display = text[:120].replace("\n", "\\n")
            if len(text) > 120:
                display += "..."
            parts.append(f"content=\"{display}\"")
        if tcs:
            tc_desc = ", ".join(
                f"{t['name']}({json.dumps(t['input'], ensure_ascii=False)})"
                for t in tcs
            )
            parts.append(f"tools=[{tc_desc}]")
        lines.append("  " + " | ".join(parts))
    return "\n".join(lines)


def stage6_full_loop() -> bool:
    _section("Stage 6 — Full ResearchAgent loop")

    from agent.core import ResearchAgent
    from agent.providers.ollama_provider import OllamaProvider

    env = _load_dotenv()
    model = env.get("OLLAMA_MODEL", "qwen3:8b-64k")
    host = env.get("OLLAMA_HOST", "http://localhost:11434")
    tavily_key = env.get("TAVILY_API_KEY")

    if not tavily_key:
        print(
            _yellow(
                "  ⚠ TAVILY_API_KEY not found in .env\n"
                "    The web_search tool will fail, which means the "
                "agent may stall.\n"
                "    Stage 6 will proceed anyway so you can see the "
                "behaviour."
            )
        )

    provider = OllamaProvider(model=model, host=host)
    agent = ResearchAgent(
        question="What is the capital of France?",
        provider=provider,
    )
    agent.step_budget = 5  # Keep it short for diagnostics

    print(f"  Question: {agent.question}")
    print(f"  Model:    {provider.model}")
    print(f"  Budget:   {agent.step_budget} steps")
    print()

    result: dict[str, Any] | None = None
    try:
        result = agent.run()
    except Exception as e:
        print(
            _red(
                f"  ✖ ResearchAgent.run() raised:\n"
                f"    {type(e).__name__}: {e}\n"
            )
        )
        traceback.print_exc()
        print()
        print("  Trajectory so far:")
        print(_trajectory_summary(agent.trajectory))
        _record("Full loop", False, f"Exception: {e}")
        return False

    assert result is not None
    answer = result.get("answer", "MISSING")
    steps = result.get("steps_used", "?")
    trajectory = result.get("trajectory", [])

    if answer == "MAX_STEPS_REACHED":
        print(
            _red(
                f"  ✖ Step budget exhausted after {steps} steps.\n"
                "    The agent never produced a FINAL ANSWER.\n"
            )
        )
        print("  Trajectory:")
        print(_trajectory_summary(trajectory))
        print()

        # Analyse what went wrong
        last_role = trajectory[-1].get("role", "") if trajectory else ""
        if any(
            e.get("stop_reason") == "tool_use" or e.get("tool_calls")
            for e in trajectory
        ):
            print(
                _yellow(
                    "  Possible diagnosis: the model kept calling tools "
                    "without converging.\n"
                    "    Try a different model or adjust the system "
                    "prompt to be more concise."
                )
            )
        else:
            print(
                _yellow(
                    "  Possible diagnosis: the model never called any "
                    "tools and didn't produce 'FINAL ANSWER:'.\n"
                    "    Check that the model follows instructions "
                    "with the 'FINAL ANSWER:' format."
                )
            )
        _record("Full loop", False, "Budget exhausted")
        return False

    if "FINAL ANSWER:" not in str(result):
        print(
            _red(
                f"  ✖ No 'FINAL ANSWER:' in result.\n"
                f"    answer field: {answer}\n"
                f"    steps used:   {steps}\n"
            )
        )
        print("  Trajectory:")
        print(_trajectory_summary(trajectory))
        _record("Full loop", False, "Missing FINAL ANSWER")
        return False

    print(f"  ✔ FINAL ANSWER: {answer}")
    print(f"  ✔ Steps used:   {steps}")
    print(f"  ✔ Trajectory entries: {len(trajectory)}")
    _record("Full loop", True, "")
    return True


# ═══════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════


def _print_summary() -> None:
    print()
    print(_bold("═══ Summary ═══"))
    print()
    print(
        f"  {'Stage':<28} {'Result':<10} Diagnosis"
    )
    print(f"  {'─' * 28} {'─' * 10} {'─' * 40}")
    any_failed = False
    for name, passed, diag in _results:
        status = _green("PASS") if passed else _red("FAIL")
        if not passed:
            any_failed = True
        print(f"  {name:<28} {status:<10} {diag}")
    print()
    if any_failed:
        print(
            _red(
                "  One or more stages FAILED.\n"
                "  Fix the issues above, then re-run this script."
            )
        )
    else:
        print(_green("  All stages PASSED — provider abstraction layer is healthy."))


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the Phase 3 provider abstraction layer "
            "by running staged checks."
        )
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        help="Start from stage N (1-6, default 1)",
    )
    parser.add_argument(
        "--skip-previous",
        action="store_true",
        help="Skip stages before --stage instead of running them",
    )
    args = parser.parse_args()

    if args.stage < 1 or args.stage > 6:
        print(f"Invalid --stage {args.stage}; choose 1-6")
        sys.exit(1)

    start = args.stage if args.skip_previous else 1

    stages = [
        (1, "Import check", stage1_imports),
        (2, "Ollama reachability", stage2_reachability),
        (3, "Model availability", stage3_model),
        (4, "Bare call (no tools)", stage4_bare_call),
        (5, "Tool-call invocation", stage5_tool_call),
        (6, "Full ResearchAgent loop", stage6_full_loop),
    ]

    for num, name, fn in stages:
        if num < start:
            _record(name, False, "(skipped)")
            continue
        print(
            _bold(
                f"\n{'=' * 60}\n"
                f" Stage {num}: {name}\n"
                f"{'=' * 60}"
            )
        )
        ok = fn()
        if not ok:
            _print_summary()
            sys.exit(1)

    # All stages passed
    _print_summary()
    sys.exit(0)


if __name__ == "__main__":
    main()
