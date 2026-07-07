from __future__ import annotations

from pathlib import Path
from typing import Any


def read_pdf(path: str) -> str:
    """Extract text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return "ERROR: pypdf is not installed — cannot read PDF files."
    try:
        reader = PdfReader(path)
        pages: list[str] = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"--- Page {i} ---\n{text.strip()}")
        if not pages:
            return "ERROR: PDF appears to contain no extractable text."
        return "\n\n".join(pages)
    except FileNotFoundError:
        return f"ERROR: file not found: {path}"
    except Exception as exc:
        return f"ERROR: failed to read PDF: {exc}"


def read_docx(path: str) -> str:
    """Extract text from a .docx file using python-docx."""
    try:
        from docx import Document
    except ImportError:
        return "ERROR: python-docx is not installed — cannot read .docx files."
    try:
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        if not paragraphs:
            return "ERROR: document appears to contain no text."
        return "\n\n".join(paragraphs)
    except FileNotFoundError:
        return f"ERROR: file not found: {path}"
    except Exception as exc:
        return f"ERROR: failed to read DOCX: {exc}"


def read_xlsx(path: str) -> str:
    """Extract text from an .xlsx file using openpyxl.

    Returns a tabular text representation (tab-separated rows per sheet).
    """
    try:
        import openpyxl
    except ImportError:
        return "ERROR: openpyxl is not installed — cannot read .xlsx files."
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        output: list[str] = []
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows: list[str] = []
            for row in ws.iter_rows():
                cells = []
                for cell in row:
                    cells.append(str(cell.value) if cell.value is not None else "")
                if any(c.strip() for c in cells):
                    rows.append("\t".join(cells))
            if rows:
                output.append(f"=== Sheet: {sheet_name} ===\n" + "\n".join(rows))
        wb.close()
        if not output:
            return "ERROR: workbook appears to contain no data."
        return "\n\n".join(output)
    except FileNotFoundError:
        return f"ERROR: file not found: {path}"
    except Exception as exc:
        return f"ERROR: failed to read XLSX: {exc}"


def read_image(path: str) -> str:
    """Extract text / metadata from an image file.

    Attempts OCR via pytesseract if the tesseract binary is available;
    otherwise returns image metadata (dimensions, format, colour mode).
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except ImportError:
        return "ERROR: Pillow is not installed — cannot read image files."

    try:
        img = Image.open(path)
    except FileNotFoundError:
        return f"ERROR: file not found: {path}"
    except Exception as exc:
        return f"ERROR: failed to open image: {exc}"

    lines: list[str] = [
        f"Image: {Path(path).name}",
        f"Format: {img.format or 'unknown'}",
        f"Size: {img.width} × {img.height} px",
        f"Mode: {img.mode}",
    ]

    # EXIF metadata
    exif_data = img._getexif()
    if exif_data:
        exif_lines: list[str] = []
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            if isinstance(value, bytes):
                continue
            exif_lines.append(f"  {tag_name}: {value}")
        if exif_lines:
            lines.append("EXIF metadata:")
            lines.extend(exif_lines)

    # OCR if pytesseract + tesseract binary are available
    ocr_text = None
    try:
        import pytesseract
        from pytesseract import get_tesseract_version
        try:
            get_tesseract_version()
            ocr_text = pytesseract.image_to_string(img)
        except Exception:
            pass
    except ImportError:
        pass

    if ocr_text and ocr_text.strip():
        lines.append("OCR-extracted text:")
        lines.append(ocr_text.strip())

    return "\n".join(lines)


def read_audio(path: str) -> str:
    """Stub — audio transcription is not yet implemented."""
    _ = path
    return "ERROR: Audio transcription not yet implemented — this file type is not supported."


# ── Extension dispatch ─────────────────────────────────────────────────


_EXTENSION_HANDLERS: dict[str, Any] = {
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".xlsx": read_xlsx,
    ".xls": read_xlsx,
    ".png": read_image,
    ".jpg": read_image,
    ".jpeg": read_image,
    ".gif": read_image,
    ".bmp": read_image,
    ".tiff": read_image,
    ".webp": read_image,
    ".mp3": read_audio,
    ".wav": read_audio,
    ".m4a": read_audio,
    ".flac": read_audio,
    ".ogg": read_audio,
}


_IMAGE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
})


def is_image(ext: str) -> bool:
    return ext.lower() in _IMAGE_EXTENSIONS


def read_file(path: str) -> str:
    """Dispatch to the appropriate reader based on file extension.

    ``.txt`` files are read as plain UTF-8 text.  For other formats the
    corresponding specialized reader is used.  Returns the file content as
    a string, or an ``ERROR:``-prefixed message on failure.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in (".txt", ".csv", ".md", ".json", ".xml", ".yaml", ".yml"):
        try:
            return p.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"ERROR: file not found: {path}"
        except Exception as exc:
            return f"ERROR: failed to read text file: {exc}"

    handler = _EXTENSION_HANDLERS.get(ext)
    if handler is None:
        return f"ERROR: unsupported file type '{ext}' for {path}"

    return handler(str(p))
