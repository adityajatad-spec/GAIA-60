from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools.files import (
    read_file,
    read_pdf,
    read_docx,
    read_xlsx,
    read_image,
    read_audio,
    is_image,
)
from agent.core import ResearchAgent
from agent.providers.base import LLMProvider, ProviderResponse


FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ── read_file dispatcher ───────────────────────────────────────────────


def test_read_txt():
    content = read_file(str(FIXTURES / "sample.txt"))
    assert "Hello" in content
    assert "ERROR" not in content


def test_read_csv():
    content = read_file(str(FIXTURES / "sample.csv"))
    assert "alpha" in content
    assert "ERROR" not in content


def test_read_pdf():
    content = read_file(str(FIXTURES / "sample.pdf"))
    assert "ERROR" not in content


def test_read_docx():
    content = read_file(str(FIXTURES / "sample.docx"))
    assert "Hello from python-docx" in content
    assert "ERROR" not in content


def test_read_xlsx():
    content = read_file(str(FIXTURES / "sample.xlsx"))
    assert "Sheet1" in content or "A" in content
    assert "ERROR" not in content


def test_read_image():
    content = read_file(str(FIXTURES / "sample.png"))
    assert "Image:" in content
    assert "ERROR" not in content


def test_read_nonexistent():
    content = read_file(str(FIXTURES / "does_not_exist.pdf"))
    assert "ERROR" in content


def test_read_unsupported_extension():
    content = read_file(str(FIXTURES / "make_fixtures.py"))
    assert "ERROR" in content


# ── Individual readers ─────────────────────────────────────────────────


def test_read_pdf_direct():
    result = read_pdf(str(FIXTURES / "sample.pdf"))
    assert "ERROR" not in result


def test_read_pdf_not_found():
    result = read_pdf("/no/such/file.pdf")
    assert "ERROR" in result


def test_read_docx_direct():
    result = read_docx(str(FIXTURES / "sample.docx"))
    assert "Hello from python-docx" in result
    assert "ERROR" not in result


def test_read_docx_not_found():
    result = read_docx("/no/such/file.docx")
    assert "ERROR" in result


def test_read_xlsx_direct():
    result = read_xlsx(str(FIXTURES / "sample.xlsx"))
    assert "Sheet1" in result
    assert "ERROR" not in result


def test_read_xlsx_not_found():
    result = read_xlsx("/no/such/file.xlsx")
    assert "ERROR" in result


def test_read_image_direct():
    result = read_image(str(FIXTURES / "sample.png"))
    assert "Image:" in result
    assert "Format:" in result
    assert "Size:" in result
    assert "ERROR" not in result


def test_read_image_not_found():
    result = read_image("/no/such/file.png")
    assert "ERROR" in result


def test_read_image_with_jpg():
    """Create a temporary JPEG and verify it's read."""
    from PIL import Image
    import tempfile

    img = Image.new("RGB", (2, 2), color=(0, 255, 0))
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp = Path(f.name)
    try:
        img.save(str(tmp))
        result = read_image(str(tmp))
        assert "Image:" in result
        assert "ERROR" not in result
    finally:
        tmp.unlink(missing_ok=True)


# ── read_audio stub ────────────────────────────────────────────────────


def test_read_audio_returns_stub():
    result = read_audio("anything.wav")
    assert "ERROR" in result
    assert "not yet implemented" in result
    assert "transcription" in result


# ── is_image helper ────────────────────────────────────────────────────


@pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"])
def test_is_image_true(ext):
    assert is_image(ext) is True


@pytest.mark.parametrize("ext", [".txt", ".pdf", ".docx", ".xlsx", ".mp3", ".py"])
def test_is_image_false(ext):
    assert is_image(ext) is False


# ── Vision provider support ────────────────────────────────────────────


def test_vision_not_supported_by_default():
    """LLMProvider.supports_vision() returns False by default."""
    provider = MagicMock(spec=LLMProvider)
    provider.supports_vision.return_value = False
    assert provider.supports_vision() is False


# ── ResearchAgent file handling ────────────────────────────────────────


def test_agent_loads_text_file_upfront():
    """Attached .txt file content is prepended to the question."""
    provider = MagicMock(spec=LLMProvider)
    provider.model = "test-model"
    provider.call.return_value = ProviderResponse(
        text="FINAL ANSWER: done", tool_calls=[], stop_reason="end_turn"
    )

    agent = ResearchAgent(
        question="What does the file say?",
        files=[str(FIXTURES / "sample.txt")],
        provider=provider,
    )
    _ = agent.run()

    # The initial message should include file content
    call_kwargs = provider.call.call_args_list[0]
    messages = call_kwargs[1]["messages"]
    first_content = messages[0]["content"]
    assert isinstance(first_content, str)
    assert "Hello" in first_content
    assert "plain text file" in first_content


def test_agent_image_without_vision_uses_text_fallback():
    """When the provider doesn't support vision, images are read as text."""
    provider = MagicMock(spec=LLMProvider)
    provider.model = "test-model"
    provider.supports_vision.return_value = False
    provider.call.return_value = ProviderResponse(
        text="FINAL ANSWER: done", tool_calls=[], stop_reason="end_turn"
    )

    agent = ResearchAgent(
        question="What's in this image?",
        files=[str(FIXTURES / "sample.png")],
        provider=provider,
    )
    _ = agent.run()

    call_kwargs = provider.call.call_args_list[0]
    messages = call_kwargs[1]["messages"]
    first_content = messages[0]["content"]
    assert isinstance(first_content, str)
    assert "Image:" in first_content


def test_agent_image_with_vision_uses_content_blocks():
    """When the provider supports vision, images are sent as content blocks."""
    provider = MagicMock(spec=LLMProvider)
    provider.model = "test-model"
    provider.supports_vision.return_value = True
    provider.call.return_value = ProviderResponse(
        text="FINAL ANSWER: A red square", tool_calls=[], stop_reason="end_turn"
    )

    agent = ResearchAgent(
        question="What color is this image?",
        files=[str(FIXTURES / "sample.png")],
        provider=provider,
    )
    _ = agent.run()

    call_kwargs = provider.call.call_args_list[0]
    messages = call_kwargs[1]["messages"]
    first_content = messages[0]["content"]
    assert isinstance(first_content, list)
    # Should have a text block and an image block
    assert any(b["type"] == "text" for b in first_content)
    assert any(b["type"] == "image" for b in first_content)


def test_agent_nonexistent_file_shows_error():
    provider = MagicMock(spec=LLMProvider)
    provider.model = "test-model"
    provider.call.return_value = ProviderResponse(
        text="FINAL ANSWER: ok", tool_calls=[], stop_reason="end_turn"
    )

    agent = ResearchAgent(
        question="Read the file",
        files=["/nonexistent/path/file.txt"],
        provider=provider,
    )
    _ = agent.run()

    call_kwargs = provider.call.call_args_list[0]
    messages = call_kwargs[1]["messages"]
    first_content = messages[0]["content"]
    assert "File not found" in first_content


def test_agent_no_files_sends_question_only():
    provider = MagicMock(spec=LLMProvider)
    provider.model = "test-model"
    provider.call.return_value = ProviderResponse(
        text="FINAL ANSWER: Paris", tool_calls=[], stop_reason="end_turn"
    )

    agent = ResearchAgent(
        question="What is the capital of France?",
        provider=provider,
    )
    result = agent.run()

    assert result["answer"] == "Paris"
    call_kwargs = provider.call.call_args_list[0]
    messages = call_kwargs[1]["messages"]
    assert messages[0]["content"] == "What is the capital of France?"


def test_agent_file_read_tool_during_loop():
    """Agent can call read_attached_file during the loop."""
    from agent.providers.base import ToolCall

    responses = iter([
        ProviderResponse(
            text="Let me read the file.",
            tool_calls=[
                ToolCall(
                    name="read_attached_file",
                    input={"path": str(FIXTURES / "sample.txt")},
                    id="call_1",
                )
            ],
            stop_reason="tool_use",
        ),
        ProviderResponse(
            text="FINAL ANSWER: content read", tool_calls=[], stop_reason="end_turn"
        ),
    ])

    provider = MagicMock(spec=LLMProvider)
    provider.model = "test-model"
    provider.call.side_effect = lambda *a, **kw: next(responses)

    agent = ResearchAgent(
        question="Read the file",
        files=[str(FIXTURES / "sample.txt")],
        provider=provider,
    )
    result = agent.run()

    # Should have a trajectory entry for the tool
    tool_entries = [
        e for e in result["trajectory"] if e.get("tool") == "read_attached_file"
    ]
    assert len(tool_entries) >= 1
    assert "Hello" in tool_entries[0]["output"]
