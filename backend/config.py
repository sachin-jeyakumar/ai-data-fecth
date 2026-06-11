import os
from pathlib import Path
import dotenv
dotenv.load_dotenv(Path(__file__).parent / ".env", override=True)
from pydantic_settings import BaseSettings, SettingsConfigDict
 
class Settings(BaseSettings):
    # Project paths
    BASE_DIR:      Path = Path(__file__).parent
    UPLOAD_DIR:    Path = BASE_DIR / "uploads"
    CHROMA_DIR:    Path = BASE_DIR / "chroma_db"
    REGISTRY_FILE: Path = BASE_DIR / "document_registry.json"
 
    # ── Groq (Primary — cloud, ~200 tok/sec, free) ────────────
    GROQ_API_KEY:  str = ""
    GROQ_MODEL:    str = "llama-3.1-8b-instant"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
 
    # ── OpenRouter (Secondary cloud fallback) ──────────────────
    OPENROUTER_API_KEY:  str = ""
    OPENROUTER_MODEL:    str = "meta-llama/llama-3.3-70b-instruct:free"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
 
    # ── Ollama (Local offline fallback) ───────────────────────
    OLLAMA_BASE_URL:    str = "http://localhost:11434"
    OLLAMA_LLM_MODEL:   str = "qwen2.5:7b"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
 
    # ── ChromaDB ───────────────────────────────────────────────
    CHROMA_COLLECTION: str = "documents"
 
    # ── RAG / chunking settings ────────────────────────────────
    # Larger chunks → more context per LLM call, fewer total batches
    CHUNK_SIZE:    int = 1500   # was 1000
    CHUNK_OVERLAP: int = 200    # was 150
 
    # Top-k for chat RAG (8 gives better answers than 3)
    RETRIEVAL_K: int = 8        # was 3
 
    # ── CORS ──────────────────────────────────────────────────
    CORS_ORIGINS: list = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "*",
    ]
 
    model_config = SettingsConfigDict(env_file=".env")
 
 
settings = Settings()
 
# Create required directories on startup
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
 