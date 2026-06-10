from pydantic import BaseModel
from typing import Any, Optional


# ──────────────────────────────────────────────
# Chat / Message models
# ──────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str               # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    document_ids: list[str] = []   # Which uploaded docs to query

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []        # Page/section citations
    extracted_data: Optional[list[dict[str, Any]]] = None  # Structured rows if extraction happened


# ──────────────────────────────────────────────
# Document models
# ──────────────────────────────────────────────

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    has_ocr: bool           # True if pytesseract was used
    status: str             # "processing" | "ready" | "error"


# ──────────────────────────────────────────────
# Extraction models
# ──────────────────────────────────────────────

class ExtractedProduct(BaseModel):
    """
    Flexible product extraction.
    The LLM fills in any fields it discovers — all stored as key-value.
    """
    data: dict[str, Any]   # Dynamic fields — whatever exists in the doc


class ExtractionResult(BaseModel):
    session_id: str
    document_ids: list[str]
    products: list[dict[str, Any]]
    columns: list[str]     # Ordered column list for the Excel sheet


# ──────────────────────────────────────────────
# Excel Export
# ──────────────────────────────────────────────

class ExcelExportRequest(BaseModel):
    session_id: str
    data: list[dict[str, Any]]
    filename: str = "extracted_data"
