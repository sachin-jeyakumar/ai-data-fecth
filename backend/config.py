import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Project paths
    BASE_DIR: Path = Path(__file__).parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    CHROMA_DIR: Path = BASE_DIR / "chroma_db"
    REGISTRY_FILE: Path = BASE_DIR / "document_registry.json"

    # Groq (Primary - Cloud, Extremely Fast, Free)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    # OpenRouter (Secondary fallback)
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "meta-llama/llama-3.1-8b-instruct:free"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # Ollama (Final fallback - Local, Offline)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_LLM_MODEL: str = "qwen2.5:7b"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"

    # ChromaDB
    CHROMA_COLLECTION: str = "documents"

    # RAG settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    RETRIEVAL_K: int = 3          # Top-3 chunks — most accurate, faster inference

    # CORS - allows frontend to talk to backend
    CORS_ORIGINS: list = [
        "http://localhost:5173",  # Vite dev server
        "http://localhost:5174",
        "http://localhost:3000",
        "*"
    ]

    class Config:
        env_file = ".env"

settings = Settings()

# Create required directories on startup
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
