import os
from pathlib import Path

from dotenv import load_dotenv


def load_config() -> dict:
    load_dotenv()

    provider = os.getenv("PROVIDER", "ollama").lower()
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not tavily_key:
        raise EnvironmentError(
            "TAVILY_API_KEY is required. "
            f"Add it to .env at {Path.cwd() / '.env'}."
        )

    config: dict[str, str] = {
        "provider": provider,
        "tavily_api_key": tavily_key,
        "search_api_key": os.getenv("SEARCH_API_KEY") or tavily_key,
    }

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is required when PROVIDER=anthropic. "
                f"Add it to .env at {Path.cwd() / '.env'}."
            )
        config["anthropic_api_key"] = api_key
        config["model"] = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is required when PROVIDER=openai. "
                f"Add it to .env at {Path.cwd() / '.env'}."
            )
        config["openai_api_key"] = api_key
        config["model"] = os.getenv("OPENAI_MODEL", "gpt-4o")

    elif provider == "ollama":
        config["model"] = os.getenv("OLLAMA_MODEL", "qwen3:8b-64k")
        config["ollama_host"] = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    else:
        raise EnvironmentError(
            f"Unknown PROVIDER: '{provider}'. "
            "Use anthropic, openai, or ollama."
        )

    return config
