"""
PDF Service — handles text-based and scanned PDFs.
 
Optimizations vs v1:
  - Parallel page processing via ThreadPoolExecutor
  - OCR threshold tuned (100 chars vs 50) for better scanned-page detection
  - Higher DPI for OCR (300 already good, kept)
  - Table extraction wrapped more defensively
  - Logging improvements for timing visibility
"""
 
import uuid
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
 
import pdfplumber
import fitz
from PIL import Image
import pytesseract
 
from config import settings
 
logger = logging.getLogger(__name__)
 
# Max worker threads for parallel page processing
_PAGE_WORKERS = 4
 
 
# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
 
def save_upload(file_bytes: bytes, original_name: str) -> tuple:
    """Save uploaded file to the uploads directory. Returns (doc_id, path)."""
    doc_id    = str(uuid.uuid4())
    safe_name = f"{doc_id}_{original_name.replace(' ', '_')}"
    dest      = settings.UPLOAD_DIR / safe_name
    dest.write_bytes(file_bytes)
    return doc_id, dest
 
 
def extract_text_from_pdf(pdf_path: Path) -> tuple:
    """
    Extract text from every page of a PDF, using OCR as fallback for
    image-only pages.
 
    Returns:
        pages    : list[str], one entry per page
        used_ocr : True if any page needed OCR
    """
    t0 = time.time()
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
 
        # Process pages in parallel
        results = _extract_pages_parallel(pdf_path, page_count)
 
        pages    = [r[0] for r in sorted(results, key=lambda x: x[1])]
        used_ocr = any(r[2] for r in results)
 
        elapsed = time.time() - t0
        logger.info(
            f"Extracted {page_count} pages from {pdf_path.name} "
            f"in {elapsed:.1f}s (OCR={used_ocr})"
        )
        return pages, used_ocr
 
    except Exception as exc:
        logger.error(f"pdfplumber failed on {pdf_path}: {exc}")
        # Last resort: OCR every page sequentially
        try:
            doc   = fitz.open(str(pdf_path))
            pages = [_ocr_page(pdf_path, i) for i in range(len(doc))]
            return pages, True
        except Exception as exc2:
            logger.error(f"OCR fallback also failed: {exc2}")
            return [], False
 
 
def get_page_count(pdf_path: Path) -> int:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception:
        return 0
 
 
# ──────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────
 
def _extract_pages_parallel(
    pdf_path: Path, page_count: int
) -> list:
    """
    Extract all pages using a thread pool.
    Returns list of (text, page_index, used_ocr) tuples.
    """
    results = []
 
    def _do_page(page_index: int):
        return _extract_single_page(pdf_path, page_index)
 
    with ThreadPoolExecutor(max_workers=_PAGE_WORKERS) as pool:
        futures = {pool.submit(_do_page, i): i for i in range(page_count)}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                idx = futures[future]
                logger.warning(f"Page {idx} extraction failed: {exc}")
                results.append(("", idx, False))
 
    return results
 
 
def _extract_single_page(pdf_path: Path, page_index: int) -> tuple:
    """
    Extract text from one page.
    Returns (text, page_index, used_ocr).
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_index]
            text = page.extract_text() or ""
 
            # Threshold: if fewer than 100 non-space chars, assume scanned
            if len(text.replace(" ", "").replace("\n", "")) < 100:
                ocr_text = _ocr_page(pdf_path, page_index)
                return ocr_text, page_index, True
 
            # Append structured table text
            try:
                tables = page.extract_tables()
                if tables:
                    table_parts = []
                    for t_idx, table in enumerate(tables):
                        rows = []
                        for row in table:
                            if row:
                                cells = [
                                    str(c).strip().replace("\n", " ")
                                    if c is not None else ""
                                    for c in row
                                ]
                                rows.append(" | ".join(cells))
                        if rows:
                            table_parts.append(
                                f"\n[Table {t_idx+1}]:\n" + "\n".join(rows)
                            )
                    if table_parts:
                        text += "\n" + "\n".join(table_parts)
            except Exception as te:
                logger.debug(f"Table extraction skipped on page {page_index}: {te}")
 
            return text, page_index, False
 
    except Exception as exc:
        logger.warning(f"Page {page_index} pdfplumber error: {exc}")
        # Fallback to OCR for this specific page
        try:
            return _ocr_page(pdf_path, page_index), page_index, True
        except Exception:
            return "", page_index, False
 
 
def _ocr_page(pdf_path: Path, page_index: int) -> str:
    """Render a PDF page at 300 DPI and run Tesseract OCR."""
    try:
        doc  = fitz.open(str(pdf_path))
        page = doc[page_index]
        mat  = fitz.Matrix(300 / 72, 300 / 72)
        pix  = page.get_pixmap(matrix=mat)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
 
        # Page segmentation mode 3 = fully automatic, best for product pages
        config = "--psm 3 --oem 3"
        text   = pytesseract.image_to_string(img, lang="eng", config=config)
        return text
    except Exception as exc:
        logger.warning(f"OCR failed for page {page_index}: {exc}")
        return ""

def extract_tables_for_orm(pdf_path: Path) -> list:
    """
    Extract structured tables and page text from the PDF.
    Returns a list of dicts: {"page": int, "text": str, "tables": list[list[list[str]]]}
    """
    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                
                # If no text was found (e.g. scanned image), fallback to OCR
                if not text.strip():
                    text = _ocr_page(pdf_path, page_index)
                    
                tables = page.extract_tables()
                
                structured_tables = []
                if tables:
                    for table in tables:
                        structured_table = []
                        for row in table:
                            if row:
                                structured_table.append([str(c).strip().replace("\n", " ") if c is not None else "" for c in row])
                        if structured_table:
                            structured_tables.append(structured_table)
                
                # Append the page if it has text OR tables.
                if text.strip() or structured_tables:
                    results.append({
                        "page": page_index + 1,
                        "text": text,  # Keep the full text
                        "tables": structured_tables
                    })
    except Exception as exc:
        logger.error(f"Failed to extract tables from {pdf_path}: {exc}")
    
    return results 