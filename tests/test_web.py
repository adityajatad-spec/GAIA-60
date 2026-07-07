import os

import pytest
from dotenv import load_dotenv

from tools.web import web_browse, web_search, _truncate

load_dotenv()

SKIP_SEARCH = not os.getenv("TAVILY_API_KEY")
SEARCH_REASON = "TAVILY_API_KEY not set — skipping integration test"


# ── web_search ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(SKIP_SEARCH, reason=SEARCH_REASON)
class TestWebSearch:
    def test_search_capital_of_france(self):
        results = web_search("capital of France")
        assert len(results) > 0
        # At least one result should mention Paris
        assert any(
            "Paris" in r["title"] or "Paris" in r["snippet"] for r in results
        )
        for r in results:
            assert "title" in r
            assert "url" in r
            assert r["url"].startswith("http")

    def test_search_returns_title_url_snippet(self):
        results = web_search("Python programming language")
        assert len(results) > 0
        for r in results:
            assert isinstance(r["title"], str) and len(r["title"]) > 0
            assert isinstance(r["url"], str) and r["url"].startswith("http")
            assert isinstance(r["snippet"], str)

    def test_search_max_results(self):
        results = web_search("test", max_results=3)
        assert len(results) <= 3

    def test_search_empty_query(self):
        results = web_search("")
        # Tavily may return an error or empty list; both are acceptable
        assert isinstance(results, list)
        if len(results) == 1 and "error" in results[0]:
            assert "error" in results[0].get("error", "").lower()

    def test_search_error_on_api_failure(self):
        """Simulate failure by passing an unrealistic scenario —
        an empty query can trigger an API error, which we handle gracefully."""
        results = web_search("")
        # Should never raise; always returns a list
        assert isinstance(results, list)


# ── Snippet truncation ─────────────────────────────────────────────────────


def test_truncate_short_string_passthrough():
    text = "Hello, world!"
    assert _truncate(text) == text
    assert _truncate(text, limit=20) == text


def test_truncate_long_string_cut_and_marked():
    text = "x" * 1000
    result = _truncate(text, limit=500)
    assert len(result) == 503  # 500 chars + "..."
    assert result.endswith("...")
    assert result[:500] == "x" * 500


def test_truncate_at_exact_limit():
    text = "x" * 500
    assert _truncate(text, limit=500) == text
    text = "x" * 501
    result = _truncate(text, limit=500)
    assert len(result) == 503
    assert result.endswith("...")


# ── web_browse ──────────────────────────────────────────────────────────────


class TestWebBrowse:
    def test_browse_example_com(self):
        text = web_browse("https://example.com")
        assert "ERROR:" not in text
        assert len(text) > 50
        assert "Example" in text or "example" in text

    def test_browse_returns_readable_text(self):
        text = web_browse("https://example.com")
        # No HTML tags should remain
        assert "<html" not in text
        assert "<body" not in text

    def test_browse_http_404(self):
        text = web_browse("https://example.com/nonexistent-page-12345")
        assert "ERROR:" in text
        assert "404" in text

    def test_browse_invalid_url(self):
        text = web_browse("https://thisshouldnotexist.example.com")
        assert "ERROR:" in text

    def test_browse_non_html_content(self):
        text = web_browse("https://example.com/image.jpg")
        # Should either error or handle gracefully
        assert isinstance(text, str)
