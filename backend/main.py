"""
FastAPI Application — AI Document Data Extractor
================================================
Endpoints:
  GET  /health                         — system health check
  POST /upload                         — upload & index a PDF
  GET  /documents                      — list indexed documents
  DELETE /documents/{doc_id}           — remove a document
  POST /chat                           — SSE streaming chat with documents
  POST /extract                        — structured product extraction → JSON
  POST /export/excel                   — download extracted data as .xlsx
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response

from config import settings
from models.schemas import (
    ChatRequest,
    ExcelExportRequest,
    ExtractionResult,
)
from services import pdf_service, embedding_service, llm_service, excel_service

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── In-memory state ────────────────────────────────────────────
# Per-session chat history: { session_id: [{role, content}] }
_chat_history: dict[str, list[dict]] = {}


# ── Persistent document registry (survives server restarts) ────
def _load_registry() -> dict:
    """Load the document registry from disk, or return empty dict."""
    try:
        if settings.REGISTRY_FILE.exists():
            import json as _json
            data = _json.loads(settings.REGISTRY_FILE.read_text())
            logger.info(f"Loaded {len(data)} documents from registry")
            return data
    except Exception as exc:
        logger.warning(f"Could not load registry: {exc}")
    return {}


def _save_registry(registry: dict) -> None:
    """Persist the document registry to disk."""
    try:
        import json as _json
        settings.REGISTRY_FILE.write_text(_json.dumps(registry, indent=2))
    except Exception as exc:
        logger.warning(f"Could not save registry: {exc}")


_document_registry: dict[str, dict] = _load_registry()

# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Data Fetcher",
    description="RAG-based PDF data extraction with local LLM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
# Health check
# ══════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    ollama_status = await llm_service.check_ollama_status()
    doc_count = len(_document_registry)
    return {
        "status": "ok",
        "ollama": ollama_status,
        "indexed_documents": doc_count,
        "model": settings.OLLAMA_LLM_MODEL,
    }


# ══════════════════════════════════════════════════════════════
# Document management
# ══════════════════════════════════════════════════════════════

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or image file.
    1. Save to disk
    2. Extract text (pdfplumber + OCR fallback)
    3. Chunk & embed into ChromaDB
    """
    if not file.filename.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
        raise HTTPException(400, "Only PDF and image files are supported.")

    file_bytes = await file.read()
    if len(file_bytes) > 50 * 1024 * 1024:   # 50 MB limit
        raise HTTPException(413, "File too large (max 50 MB).")

    # Save
    doc_id, pdf_path = pdf_service.save_upload(file_bytes, file.filename)
    _document_registry[doc_id] = {
        "filename": file.filename,
        "page_count": 0,
        "has_ocr": False,
        "status": "processing",
        "path": str(pdf_path),
    }

    try:
        # Run blocking CPU/IO work in a thread pool so we don't freeze the event loop
        loop = asyncio.get_event_loop()
        pages, used_ocr = await loop.run_in_executor(
            None, lambda: pdf_service.extract_text_from_pdf(pdf_path)
        )
        page_count = len(pages)

        chunk_count = await loop.run_in_executor(
            None, lambda: embedding_service.embed_document(doc_id, pages, file.filename)
        )

        _document_registry[doc_id].update({
            "page_count": page_count,
            "has_ocr": used_ocr,
            "chunk_count": chunk_count,
            "status": "ready",
        })
        _save_registry(_document_registry)   # persist so it survives restarts
        logger.info(f"Indexed {file.filename}: {page_count} pages, {chunk_count} chunks, OCR={used_ocr}")

    except Exception as exc:
        _document_registry[doc_id]["status"] = "error"
        logger.error(f"Failed to index {file.filename}: {exc}")
        raise HTTPException(500, f"Document processing failed: {exc}")

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "page_count": page_count,
        "has_ocr": used_ocr,
        "chunk_count": chunk_count,
        "status": "ready",
    }


@app.get("/documents")
async def list_documents():
    """Return all indexed documents."""
    return [
        {"doc_id": did, **info}
        for did, info in _document_registry.items()
        if info.get("status") == "ready"
    ]


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in _document_registry:
        raise HTTPException(404, "Document not found.")
    embedding_service.delete_document(doc_id)
    # Delete file from disk
    path = Path(_document_registry[doc_id].get("path", ""))
    if path.exists():
        path.unlink()
    del _document_registry[doc_id]
    _save_registry(_document_registry)   # persist deletion
    return {"message": f"Document {doc_id} deleted."}


# ══════════════════════════════════════════════════════════════
# Chat (SSE streaming)
# ══════════════════════════════════════════════════════════════

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Stream a RAG-grounded response via Server-Sent Events.
    Frontend reads tokens as they arrive.
    """
    doc_ids = request.document_ids or list(_document_registry.keys())

    # Run the blocking embedding lookup in a thread pool so we don't freeze the event loop
    loop = asyncio.get_event_loop()
    chunks = await loop.run_in_executor(
        None,
        lambda: embedding_service.retrieve_relevant_chunks(
            query=request.message,
            doc_ids=doc_ids if doc_ids else None,
        )
    )

    if not chunks and not doc_ids:
        # No documents uploaded yet
        async def _no_docs():
            yield "data: " + json.dumps({"token": "No documents uploaded yet. Please upload a PDF first."}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_no_docs(), media_type="text/event-stream")

    history = _chat_history.get(request.session_id, [])

    async def _stream():
        full_response = ""
        sources = list({f"{c['filename']} p.{c['page']}" for c in chunks})

        try:
            async for token in llm_service.chat_with_context(
                message=request.message,
                context_chunks=chunks,
                history=history,
            ):
                full_response += token
                yield "data: " + json.dumps({"token": token}) + "\n\n"
                await asyncio.sleep(0)   # yield control to event loop
        except Exception as exc:
            logger.error(f"LLM streaming error: {exc}")
            yield "data: " + json.dumps({"error": str(exc)}) + "\n\n"

        # Save to history
        if request.session_id not in _chat_history:
            _chat_history[request.session_id] = []
        _chat_history[request.session_id].append({"role": "user", "content": request.message})
        _chat_history[request.session_id].append({"role": "assistant", "content": full_response})

        # Send sources + done signal
        yield "data: " + json.dumps({"sources": sources, "done": True}) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ══════════════════════════════════════════════════════════════
# Structured Extraction
# ══════════════════════════════════════════════════════════════

# Extraction jobs in-memory registry: { job_id: { status, progress, products, columns, error } }
_extraction_jobs: dict[str, dict] = {}


async def _run_extraction_background(job_id: str, doc_ids: List[str]):
    try:
        # Retrieve structured tables directly from the original PDFs (ORM approach)
        pdf_pages_data = []
        for doc_id in doc_ids:
            doc_info = _document_registry.get(doc_id)
            if not doc_info or "path" not in doc_info:
                continue
            
            # Extract tables using pdfplumber
            pages_with_tables = pdf_service.extract_tables_for_orm(Path(doc_info["path"]))
            pdf_pages_data.extend(pages_with_tables)

        if not pdf_pages_data:
            _extraction_jobs[job_id].update({
                "status": "failed",
                "error": "No tables found in the document(s) to extract."
            })
            return

        # The batches will be the pages with tables
        total_batches = len(pdf_pages_data)
        _extraction_jobs[job_id].update({
            "total_batches": total_batches,
            "progress": 0
        })

        def clean_products(product_list: list):
            unique_products = []
            seen = set()
            for p in product_list:
                serialized = json.dumps(p, sort_keys=True)
                if serialized not in seen:
                    seen.add(serialized)
                    unique_products.append(p)
                    
            empty_values = {"", "-", "—", "None", "null", "N/A", "n/a"}
            keys_to_remove = set()
            
            if unique_products:
                all_keys = set()
                for p in unique_products:
                    all_keys.update(p.keys())
                    
                for k in all_keys:
                    is_empty = True
                    for p in unique_products:
                        val = str(p.get(k, "")).strip()
                        if val and val not in empty_values and val.lower() != "none" and val.lower() != "null":
                            is_empty = False
                            break
                    if is_empty and k.lower() not in ("name", "model", "category"):
                        keys_to_remove.add(k)
                        
                for p in unique_products:
                    for k in keys_to_remove:
                        if k in p:
                            del p[k]
                            
            cols: list[str] = []
            for p in unique_products:
                for k in p.keys():
                    if k not in cols:
                        cols.append(k)
            return unique_products, cols

        async def progress_callback(processed_batches: int, tot_batches: int, products: List[Dict]):
            job = _extraction_jobs.get(job_id)
            if job:
                pct = int((processed_batches / tot_batches) * 100)
                raw_products = job.get("raw_products", []) + products
                cleaned_products, cols = clean_products(raw_products)
                
                job.update({
                    "progress": pct,
                    "processed_batches": processed_batches,
                    "total_batches": tot_batches,
                    "products": cleaned_products,
                    "raw_products": raw_products,
                    "columns": cols
                })

        all_products = await llm_service.extract_products_from_tables(pdf_pages_data, on_progress=progress_callback)
        cleaned_products, columns = clean_products(all_products)

        _extraction_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "products": cleaned_products,
            "columns": columns
        })
    except Exception as exc:
        logger.exception(f"Background extraction failed for job {job_id}")
        _extraction_jobs[job_id].update({
            "status": "failed",
            "error": str(exc)
        })


@app.post("/extract")
async def extract_products(
    document_ids: list[str] = [],
    session_id: str = Form(default="default"),
):
    """
    Triggers background extraction of all product data.
    """
    doc_ids = [d for d in document_ids if d.strip()]
    doc_ids = doc_ids or list(_document_registry.keys())

    if not doc_ids:
        raise HTTPException(400, "No documents available. Please upload PDFs first.")

    job_id = str(uuid.uuid4())
    _extraction_jobs[job_id] = {
        "job_id": job_id,
        "session_id": session_id,
        "document_ids": doc_ids,
        "status": "processing",
        "progress": 0,
        "processed_batches": 0,
        "total_batches": 0,
        "products": [],
        "columns": [],
        "error": None
    }

    asyncio.create_task(_run_extraction_background(job_id, doc_ids))
    return {"job_id": job_id, "status": "processing"}


@app.post("/extract/json")
async def extract_products_json(body: dict[str, Any]):
    """Alternative JSON body version of /extract."""
    document_ids = body.get("document_ids", [])
    session_id = body.get("session_id", "default")
    
    doc_ids = [d for d in document_ids if d.strip()]
    doc_ids = doc_ids or list(_document_registry.keys())

    if not doc_ids:
        raise HTTPException(400, "No documents available.")

    job_id = str(uuid.uuid4())
    _extraction_jobs[job_id] = {
        "job_id": job_id,
        "session_id": session_id,
        "document_ids": doc_ids,
        "status": "processing",
        "progress": 0,
        "processed_batches": 0,
        "total_batches": 0,
        "products": [],
        "columns": [],
        "error": None
    }

    asyncio.create_task(_run_extraction_background(job_id, doc_ids))
    return {"job_id": job_id, "status": "processing"}


@app.get("/extract/status/{job_id}")
async def get_extraction_status(job_id: str):
    """Get the current progress and results of a background extraction job."""
    if job_id not in _extraction_jobs:
        raise HTTPException(404, "Extraction job not found.")
    return _extraction_jobs[job_id]


# ══════════════════════════════════════════════════════════════
# Excel Export
# ══════════════════════════════════════════════════════════════

@app.post("/export/excel")
async def export_excel(request: ExcelExportRequest):
    """
    Convert extracted data to a styled .xlsx file and return as download.
    """
    if not request.data:
        raise HTTPException(400, "No data provided for export.")

    xlsx_bytes = excel_service.export_to_excel(request.data, request.filename)
    safe_name = request.filename.replace(" ", "_") + ".xlsx"

    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


# ══════════════════════════════════════════════════════════════
# Session management
# ══════════════════════════════════════════════════════════════

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    _chat_history.pop(session_id, None)
    return {"message": "Session cleared."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
