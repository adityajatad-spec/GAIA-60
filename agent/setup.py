"""
One-time interactive setup wizard for gaia-agent.
Prompts the user for provider choice, validates API keys / model availability,
and writes .env. Designed to be called from agent/__main__.py before the agent
loop begins.
"""

from __future__ import annotations

import os
import sys
from getpass import getpass
from pathlib import Path
from typing import NoReturn

import requests
from dotenv import load_dotenv

_PROVIDER_MENU: dict[str, tuple[str, str]] = {
    "1": ("anthropic", "Anthropic (Claude)"),
    "2": ("openai", "OpenAI (GPT)"),
    "3": ("ollama", "Ollama (local, no API key needed)"),
}


# ── Public entry point ──────────────────────────────────────────────────────


def interactive_setup(*, reconfigure: bool = False) -> None:
    """Run the one-time setup wizard.

    Skips without prompting if ``.env`` already contains a valid provider
    configuration (unless *reconfigure* is ``True``).
    """
    if not reconfigure:
        existing = _find_existing_provider()
        if existing is not None:
            return

    _print_banner()

    provider = _prompt_provider()

    if provider == "anthropic":
        _setup_anthropic()
    elif provider == "openai":
        _setup_openai()
    else:
        _setup_ollama()

    _ensure_tavily_key()

    print(f"\n{'─' * 50}")
    print(f"  Setup complete. Provider: {provider}")
    print(f"  Run `python -m agent \"<your question>\"` to get started.")
    print(f"{'─' * 50}\n")


# ── Existing-config detection ───────────────────────────────────────────────


def _find_existing_provider() -> str | None:
    """Return the provider name if ``.env`` has a valid config, else ``None``."""
    if not Path(".env").exists():
        return None
    load_dotenv(Path(".env"))

    provider = os.getenv("PROVIDER")

    # Backward compat: ANTHROPIC_API_KEY alone → treat as anthropic
    if not provider and os.getenv("ANTHROPIC_API_KEY"):
        provider = "anthropic"

    if not provider:
        return None

    provider = provider.lower()
    if not os.getenv("TAVILY_API_KEY"):
        return None

    if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        return provider
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        return provider
    if provider == "ollama" and os.getenv("OLLAMA_MODEL"):
        return provider
    return None


# ── Provider choice ─────────────────────────────────────────────────────────


def _print_banner() -> None:
    print()
    print("╔══════════════════════════════════════╗")
    print("║       gaia-agent — First-Time Setup  ║")
    print("╚══════════════════════════════════════╝")
    print()


def _prompt_provider() -> str:
    print("Which LLM provider would you like to use?")
    for key, (name, desc) in _PROVIDER_MENU.items():
        print(f"  {key}) {desc}")

    while True:
        choice = input("Enter choice (1-3): ").strip()
        if choice in _PROVIDER_MENU:
            return _PROVIDER_MENU[choice][0]
        print(f"  Invalid choice '{choice}'. Please enter 1, 2, or 3.")


# ── Per-provider setup ──────────────────────────────────────────────────────


def _setup_anthropic() -> None:
    print("\n── Anthropic ──")
    key = _prompt_key_secret("Enter your Anthropic API key")
    key = _validate_key_with_retry("anthropic", key)
    _set_env("PROVIDER", "anthropic")
    _set_env("ANTHROPIC_API_KEY", key)


def _setup_openai() -> None:
    print("\n── OpenAI ──")
    key = _prompt_key_secret("Enter your OpenAI API key")
    key = _validate_key_with_retry("openai", key)
    _set_env("PROVIDER", "openai")
    _set_env("OPENAI_API_KEY", key)


def _setup_ollama() -> None:
    print("\n── Ollama ──")
    model = input("Enter the model name you have pulled (e.g. qwen3:8b-64k): ").strip()
    while not model:
        model = input("  Model name cannot be empty: ").strip()
    _validate_ollama_or_die(model)
    _set_env("PROVIDER", "ollama")
    _set_env("OLLAMA_MODEL", model)


# ── Key prompting ───────────────────────────────────────────────────────────


def _prompt_key_secret(prompt: str) -> str:
    value = getpass(f"  {prompt}: ").strip()
    while not value:
        value = getpass(f"  Key cannot be empty. {prompt}: ").strip()
    return value


# ── Key validation ──────────────────────────────────────────────────────────


def _validate_key_with_retry(provider: str, key: str) -> str:
    """Test the API key.  Re-prompt on failure until a valid key is given.

    Returns the validated key (which may differ from the input *key* if the
    user was re-prompted).
    """
    while True:
        ok, msg = _test_key(provider, key)
        if ok:
            print("  ✓ Key validated")
            return key
        print(f"  ✗ {msg}")
        key = getpass(f"  Enter a valid {provider.title()} API key (or press Ctrl+C to quit): ").strip()
        if not key:
            _abort()


def _test_key(provider: str, key: str) -> tuple[bool, str]:
    """Return ``(True, "")`` on success or ``(False, reason)`` on failure."""
    try:
        if provider == "anthropic":
            resp = requests.get(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                timeout=10,
            )
        elif provider == "openai":
            resp = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            )
        else:
            return False, f"Unknown provider: {provider}"

        if resp.status_code == 200:
            return True, ""
        return False, f"API returned status {resp.status_code} — key rejected"

    except requests.RequestException:
        return False, "Could not reach the API — check your internet connection"


# ── Ollama validation ───────────────────────────────────────────────────────


def _validate_ollama_or_die(model: str) -> None:
    """Check that the Ollama server is reachable and *model* is available.

    Prints actionable error messages and exits on failure instead of
    letting the agent fail confusingly mid-run.
    """
    # 1. Server reachable?
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException:
        print("  ✗ Ollama server is not reachable at http://localhost:11434")
        print("  Make sure Ollama is installed and running.")
        print("  See https://ollama.com for installation instructions.")
        _abort()

    # 2. Model available?
    model_names = [m["name"] for m in resp.json().get("models", [])]
    if model not in model_names:
        print(f"  ✗ Model '{model}' is not available locally.")
        print(f"  Run:  ollama pull {model}")
        print(f"  Then: python -m agent --reconfigure")
        _abort()

    print(f"  ✓ Ollama server reachable, model '{model}' found")


# ── Tavily ──────────────────────────────────────────────────────────────────


def _ensure_tavily_key() -> None:
    """Prompt for a Tavily key if not already present in the environment."""
    if os.getenv("TAVILY_API_KEY"):
        return
    print("\n── Tavily Search (required) ──")
    print("  A Tavily API key is needed for web search.")
    print("  Get one free at https://app.tavily.com")
    key = getpass("  Enter your Tavily API key: ").strip()
    while not key:
        key = getpass("  Tavily API key cannot be empty: ").strip()
    _set_env("TAVILY_API_KEY", key)


# ── .env writer ─────────────────────────────────────────────────────────────


def _set_env(key: str, value: str) -> None:
    """Set *key=value* in ``.env``, preserving all other keys."""
    lines: list[str] = []
    found = False

    if Path(".env").exists():
        with open(Path(".env")) as f:
            for line in f:
                stripped = line.rstrip("\n")
                if stripped.startswith(f"{key}=") or stripped.startswith(f"#{key}="):
                    lines.append(f"{key}={value}")
                    found = True
                else:
                    lines.append(stripped)

    if not found:
        lines.append(f"{key}={value}")

    with open(Path(".env"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Helpers ─────────────────────────────────────────────────────────────────


def _abort() -> NoReturn:
    print("\n  Setup aborted. Run `python -m agent --reconfigure` when ready.\n")
    sys.exit(1)
