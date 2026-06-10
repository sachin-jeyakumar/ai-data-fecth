"""
PDF Service — handles both text-based and scanned PDFs.

Strategy:
  1. Try pdfplumber first (fast, accurate for text PDFs).
  2. If a page yields < 50 chars (likely scanned), fall back to
     pytesseract OCR on a rendered image of that page.
"""

import uuid
import logging
from pathlib import Path

import pdfplumber
import fitz                     # PyMuPDF — renders PDF pages to images for OCR
from PIL import Image
import pytesseract

from config import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def save_upload(file_bytes: bytes, original_name: str) -> tuple:
    """Save uploaded file to the uploads directory.  Returns (doc_id, path)."""
    doc_id = str(uuid.uuid4())
    safe_name = f"{doc_id}_{original_name.replace(' ', '_')}"
    dest = settings.UPLOAD_DIR / safe_name
    dest.write_bytes(file_bytes)
    return doc_id, dest


def extract_text_from_pdf(pdf_path: Path) -> tuple:
    """
    Extract text from every page of a PDF.

    Returns:
        pages      : list of strings, one per page
        used_ocr   : True if any page needed OCR
    """
    pages: list[str] = []
    used_ocr = False

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if len(text.strip()) < 50:
                    # Page seems to be an image — run OCR
                    logger.info(f"Page {i+1}/{page_count}: using OCR (text too short)")
                    ocr_text = _ocr_page(pdf_path, i)
                    pages.append(ocr_text)
                    used_ocr = True
                else:
                    page_content = text
                    # Natively extract tables to preserve layout structure
                    try:
                        tables = page.extract_tables()
                        if tables:
                            table_texts = []
                            for idx, table in enumerate(tables):
                                table_rows = []
                                for row in table:
                                    if row:
                                        cleaned_row = [str(cell).strip().replace('\n', ' ') if cell is not None else "" for cell in row]
                                        table_rows.append(" | ".join(cleaned_row))
                                if table_rows:
                                    table_texts.append(f"\n[Structured Table {idx+1}]:\n" + "\n".join(table_rows))
                            if table_texts:
                                page_content += "\n" + "\n".join(table_texts)
                    except Exception as table_exc:
                        logger.warning(f"Failed to extract tables on page {i+1}: {table_exc}")
                    pages.append(page_content)
    except Exception as exc:
        logger.error(f"pdfplumber failed on {pdf_path}: {exc}")
        # Last resort: OCR every page
        try:
            doc = fitz.open(str(pdf_path))
            for i in range(len(doc)):
                pages.append(_ocr_page(pdf_path, i))
            used_ocr = True
        except Exception as exc2:
            logger.error(f"OCR fallback also failed: {exc2}")

    return pages, used_ocr


def get_page_count(pdf_path: Path) -> int:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception:
        return 0


# ──────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────

def _ocr_page(pdf_path: Path, page_index: int) -> str:
    """Render a single PDF page to an image and run Tesseract OCR on it."""
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_index]
        # Render at 300 DPI for good OCR quality
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang="eng")
        doc.close()
        return text
    except Exception as exc:
        logger.warning(f"OCR failed for page {page_index}: {exc}")
        return ""
