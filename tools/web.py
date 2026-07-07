from __future__ import annotations

from typing import Any

import requests
import trafilatura
from tavily import TavilyClient

from agent.config import load_config

_USER_AGENT = (
    "Mozilla/5.0 (compatible; gaia-agent/0.1; research-bot; "
    "+https://github.com/gaia-agent)"
)


_MAX_SNIPPET_CHARS = 500


def _truncate(text: str, limit: int = _MAX_SNIPPET_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search the web using Tavily and return top results.

    Returns a list of {title, url, snippet} dicts.
    Each snippet is truncated to *max_snippet_chars* (default ~500) to
    prevent a single noisy result from dominating the context window.
    On error, returns a list containing a single dict with an "error" key.
    """
    cfg = load_config()
    client = TavilyClient(api_key=cfg["tavily_api_key"])

    try:
        resp = client.search(query=query, max_results=max_results)
    except Exception:
        return [{"error": "Search API error — check your connection and API key"}]

    results: list[dict[str, Any]] = []
    for r in resp.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": _truncate(r.get("content", "")),
        })
    return results


def web_browse(url: str, max_chars: int = 16000) -> str:
    """Fetch a URL and extract readable text content.

    Returns the cleaned text, or an error message prefixed with ``ERROR: ``.
    Truncates to *max_chars* (roughly 4k tokens) and appends a truncation
    notice when the limit is exceeded.
    """
    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return f"ERROR: request to {url} timed out"
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code
        return f"ERROR: HTTP {status} for {url}"
    except requests.exceptions.RequestException as exc:
        return f"ERROR: {exc}"

    content_type = resp.headers.get("Content-Type", "")
    if not ("text/html" in content_type or "text/plain" in content_type):
        return f"ERROR: unsupported Content-Type '{content_type}' for {url}"

    text = trafilatura.extract(resp.text)
    if text is None:
        text = resp.text[:5000] + (
            "\n\n[Could not extract readable content; showing raw HTML head]"
        )

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[TRUNCATED: content exceeds token budget]"

    return text
