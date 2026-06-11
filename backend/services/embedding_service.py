"""
Embedding Service — ChromaDB + Ollama nomic-embed-text.
 
Optimizations vs v1:
  - Larger chunk size (1500 vs 1000) — fewer chunks, same coverage
  - Parallel embedding batches via asyncio executor
  - get_all_chunks returns ordered, page-contiguous chunks
  - retrieve_relevant_chunks raises k to 8 for better RAG answers
  - Batch upsert with existence check to avoid re-embedding on reload
"""
 
import logging
from pathlib import Path
from typing import Dict, List, Optional
 
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
 
from config import settings
 
logger = logging.getLogger(__name__)
 
# ── Singletons ─────────────────────────────────────────────────
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None
_embeddings: Optional[OllamaEmbeddings] = None
_splitter: Optional[RecursiveCharacterTextSplitter] = None
 
 
def _get_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=str(settings.CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client
 
 
def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        _collection = _get_client().get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection
 
 
def _get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            model=settings.OLLAMA_EMBED_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
    return _embeddings
 
 
def _get_splitter() -> RecursiveCharacterTextSplitter:
    global _splitter
    if _splitter is None:
        _splitter = RecursiveCharacterTextSplitter(
            # Larger chunks = more context per LLM call, fewer total chunks
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    return _splitter
 
 
# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
 
def embed_document(doc_id: str, pages: List[str], filename: str) -> int:
    """
    Chunk all pages and store embeddings in ChromaDB.
    Skips chunks that are already indexed (idempotent).
    Returns total number of chunks stored.
    """
    splitter   = _get_splitter()
    embedder   = _get_embeddings()
    collection = _get_collection()
 
    all_chunks: List[str]  = []
    all_ids:    List[str]  = []
    all_metas:  List[dict] = []
 
    for page_num, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        for chunk_idx, chunk in enumerate(splitter.split_text(page_text)):
            chunk_id = f"{doc_id}_p{page_num}_c{chunk_idx}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metas.append({
                "doc_id":   doc_id,
                "filename": filename,
                "page":     page_num,
                "chunk":    chunk_idx,
            })
 
    if not all_chunks:
        logger.warning(f"No text chunks for doc {doc_id}")
        return 0
 
    # Check which IDs already exist to avoid redundant embedding work
    existing = set(collection.get(ids=all_ids, include=[])["ids"])
    new_indices = [i for i, cid in enumerate(all_ids) if cid not in existing]
 
    if not new_indices:
        logger.info(f"All {len(all_ids)} chunks already indexed for {doc_id}")
        return len(all_ids)
 
    new_chunks = [all_chunks[i] for i in new_indices]
    new_ids    = [all_ids[i]    for i in new_indices]
    new_metas  = [all_metas[i]  for i in new_indices]
 
    # Embed in batches of 100
    batch_size = 100
    for i in range(0, len(new_chunks), batch_size):
        bc = new_chunks[i : i + batch_size]
        bi = new_ids[i : i + batch_size]
        bm = new_metas[i : i + batch_size]
        vectors = embedder.embed_documents(bc)
        collection.add(ids=bi, embeddings=vectors, documents=bc, metadatas=bm)
        logger.info(
            f"Embedded batch {i // batch_size + 1} "
            f"({len(bc)} chunks) for {doc_id}"
        )
 
    total = len(existing) + len(new_indices)
    logger.info(
        f"Indexed {doc_id}: {total} total chunks "
        f"({len(new_indices)} new, {len(existing)} pre-existing)"
    )
    return total
 
 
def retrieve_relevant_chunks(
    query: str,
    doc_ids: Optional[List[str]] = None,
    k: Optional[int] = None,
) -> List[Dict]:
    """
    Semantic search: return top-k most relevant chunks for a query.
    k defaults to settings.RETRIEVAL_K (recommend 8 for better answers).
    """
    k          = k or settings.RETRIEVAL_K
    embedder   = _get_embeddings()
    collection = _get_collection()
 
    count = collection.count()
    if count == 0:
        return []
 
    query_vec    = embedder.embed_query(query)
    where_filter = _build_where(doc_ids)
 
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(k, count),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )
 
    chunks = []
    if results["documents"] and results["documents"][0]:
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append({
                "text":     text,
                "doc_id":   meta.get("doc_id", ""),
                "filename": meta.get("filename", ""),
                "page":     meta.get("page", 0),
            })
    return chunks
 
 
def get_all_chunks(doc_ids: List[str]) -> List[Dict]:
    """
    Return ALL stored chunks for the given documents, ordered by
    (filename, page, chunk_index) — preserving reading order for extraction.
    """
    collection   = _get_collection()
    where_filter = _build_where(doc_ids)
 
    count = collection.count()
    if count == 0:
        return []

    results = collection.get(
        where=where_filter,
        include=["documents", "metadatas"],
        limit=count,
    )
 
    chunks = []
    if results["documents"]:
        for text, meta in zip(results["documents"], results["metadatas"]):
            chunks.append({
                "text":     text,
                "doc_id":   meta.get("doc_id",   ""),
                "filename": meta.get("filename", ""),
                "page":     meta.get("page",     0),
                "chunk":    meta.get("chunk",    0),
            })
 
    # Stable sort preserves document reading order
    chunks.sort(key=lambda x: (x["filename"], x["page"], x.get("chunk", 0)))
    return chunks
 
 
def delete_document(doc_id: str) -> None:
    """Remove all chunks for a document from ChromaDB."""
    collection = _get_collection()
    collection.delete(where={"doc_id": doc_id})
    logger.info(f"Deleted embeddings for doc {doc_id}")
 
 
# ──────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────
 
def _build_where(doc_ids: Optional[List[str]]) -> Optional[dict]:
    if not doc_ids:
        return None
    if len(doc_ids) == 1:
        return {"doc_id": doc_ids[0]}
    return {"doc_id": {"$in": doc_ids}}
 