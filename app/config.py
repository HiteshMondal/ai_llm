from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False
    ui_port: int = 7860

    # LLM (Ollama)
    llm_model: str = "tinyllama"
    llm_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.2

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store
    vector_collection_name: str = "rag_docs"
    chroma_persist_dir: str = "data/chroma"

    # Ingestion
    chunk_size: int = 500
    chunk_overlap: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS