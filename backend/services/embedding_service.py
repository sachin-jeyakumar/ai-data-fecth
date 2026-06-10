"""
Embedding Service — chunks document text and stores/retrieves
embeddings using ChromaDB + Ollama's nomic-embed-text model.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from config import settings

logger = logging.getLogger(__name__)

# ── Singletons (created once per process) ──────────────────────
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
        client = _get_client()
        _collection = client.get_or_create_collection(
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
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )
    return _splitter


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def embed_document(doc_id: str, pages: list[str], filename: str) -> int:
    """
    Chunk all pages of a document and store embeddings in ChromaDB.
    Returns total number of chunks stored.
    """
    splitter = _get_splitter()
    embedder = _get_embeddings()
    collection = _get_collection()

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metas: list[dict] = []

    for page_num, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        chunks = splitter.split_text(page_text)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_p{page_num}_c{chunk_idx}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metas.append({
                "doc_id": doc_id,
                "filename": filename,
                "page": page_num,
                "chunk": chunk_idx,
            })

    if not all_chunks:
        logger.warning(f"No text chunks found for doc {doc_id}")
        return 0

    # Embed in batches of 50 to avoid memory spikes
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch_texts = all_chunks[i : i + batch_size]
        batch_ids   = all_ids[i : i + batch_size]
        batch_metas = all_metas[i : i + batch_size]

        vectors = embedder.embed_documents(batch_texts)
        collection.add(
            ids=batch_ids,
            embeddings=vectors,
            documents=batch_texts,
            metadatas=batch_metas,
        )
        logger.info(f"Embedded batch {i//batch_size + 1} ({len(batch_texts)} chunks)")

    return len(all_chunks)


def retrieve_relevant_chunks(
    query: str,
    doc_ids: Optional[List[str]] = None,
    k: Optional[int] = None,
) -> List[Dict]:
    """
    Retrieve the top-k most relevant chunks for a query.
    Optionally filter to specific doc_ids.

    Returns a list of dicts with keys: text, doc_id, filename, page
    """
    k = k or settings.RETRIEVAL_K
    embedder = _get_embeddings()
    collection = _get_collection()

    query_vec = embedder.embed_query(query)

    where_filter = None
    if doc_ids:
        if len(doc_ids) == 1:
            where_filter = {"doc_id": doc_ids[0]}
        else:
            where_filter = {"doc_id": {"$in": doc_ids}}

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(k, collection.count() or 1),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results["documents"] and results["documents"][0]:
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append({
                "text": text,
                "doc_id": meta.get("doc_id", ""),
                "filename": meta.get("filename", ""),
                "page": meta.get("page", 0),
            })
    return chunks


def get_all_chunks(doc_ids: List[str]) -> List[Dict]:
    """
    Retrieve ALL chunks for the given doc_ids without semantic search.
    """
    collection = _get_collection()
    
    where_filter = None
    if len(doc_ids) == 1:
        where_filter = {"doc_id": doc_ids[0]}
    else:
        where_filter = {"doc_id": {"$in": doc_ids}}
        
    results = collection.get(
        where=where_filter,
        include=["documents", "metadatas"],
    )
    
    chunks = []
    if results["documents"]:
        for text, meta in zip(results["documents"], results["metadatas"]):
            chunks.append({
                "text": text,
                "doc_id": meta.get("doc_id", ""),
                "filename": meta.get("filename", ""),
                "page": meta.get("page", 0),
                "chunk": meta.get("chunk", 0),
            })
            
    # Sort chunks by page then chunk index to maintain order
    chunks.sort(key=lambda x: (x["filename"], x["page"], x.get("chunk", 0)))
    return chunks



def delete_document(doc_id: str) -> None:
    """Remove all chunks for a document from ChromaDB."""
    collection = _get_collection()
    collection.delete(where={"doc_id": doc_id})
    logger.info(f"Deleted embeddings for doc {doc_id}")
