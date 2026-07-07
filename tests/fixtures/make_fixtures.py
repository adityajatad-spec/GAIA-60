"""Generate test fixture files for the file-reader tests.

Run once from the repo root:
    python tests/fixtures/make_fixtures.py
"""

import csv
import io
from pathlib import Path

HERE = Path(__file__).resolve().parent


def make_txt() -> None:
    (HERE / "sample.txt").write_text(
        "Hello, this is a plain text file.\nIt has two lines.\n",
        encoding="utf-8",
    )


def make_csv() -> None:
    with open(HERE / "sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "value"])
        w.writerow(["alpha", "1"])
        w.writerow(["beta", "2"])


def make_pdf() -> None:
    """Create a minimal PDF with a text page using pypdf."""
    from io import BytesIO
    from pypdf import PdfWriter

    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(50, 750, "Hello from pypdf!")
    c.save()
    buf.seek(0)

    w = PdfWriter()
    w.append(buf)
    out = BytesIO()
    w.write(out)
    (HERE / "sample.pdf").write_bytes(out.getvalue())


def make_docx() -> None:
    from docx import Document

    doc = Document()
    doc.add_paragraph("Hello from python-docx!")
    doc.add_paragraph("Second paragraph.")
    doc.save(str(HERE / "sample.docx"))


def make_xlsx() -> None:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["A", "B"])
    ws.append(["1", "2"])
    wb.save(str(HERE / "sample.xlsx"))


def make_image() -> None:
    from PIL import Image

    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    img.save(str(HERE / "sample.png"))


if __name__ == "__main__":
    make_txt()
    make_csv()
    make_pdf()
    make_docx()
    make_xlsx()
    make_image()
    print("Fixtures created in", HERE)
