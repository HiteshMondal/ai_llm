"""
Central configuration module for the AI RAG application.

Responsibilities:
- Environment-based configuration (Pydantic Settings)
- Logger factory (single global logging policy)
- File ingestion helpers
- Cached settings singleton
"""

from functools import lru_cache
from pathlib import Path
import logging
import sys

from pydantic_settings import BaseSettings, SettingsConfigDict


# SETTINGS


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables
    or `.env` file automatically.

    Supports provider switching without code changes.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    # APP CONFIG

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False
    ui_port: int = 7860

    # LLM CONFIG

    llm_provider: str = "ollama"
    llm_model: str = "tinyllama"
    llm_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.2

    # EMBEDDINGS CONFIG

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_provider: str = "local"

    ingest_batch_size: int = 64

    # VECTOR STORE CONFIG

    vector_collection_name: str = "rag_docs"
    chroma_persist_dir: str = "data/embeddings"

    # INGESTION CONFIG

    chunk_size: int = 500
    chunk_overlap: int = 50

    # QUERY CACHE CONFIG

    query_cache_ttl_seconds: int = 300
    query_cache_max_size: int = 256

    # RERANKER CONFIG

    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 4

    # SESSION MEMORY CONFIG

    session_memory_turns: int = 6

    # STREAMING CONFIG

    stream_chunk_delay: float = 0.0

    # PROVIDER API KEYS

    gemini_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    openai_api_key: str = ""

    # GOOGLE DRIVE CONNECTOR

    gdrive_credentials_file: str = "credentials.json"
    gdrive_token_file: str = "token.json"
    gdrive_folder_id: str = ""

    # NOTION CONNECTOR

    notion_token: str = ""
    notion_page_ids: str = ""
    notion_database_ids: str = ""

    # GITHUB CONNECTOR

    github_token: str = ""
    github_repos: str = ""
    github_branch: str = ""

    # CONNECTOR RELIABILITY

    connector_timeout: int = 30
    connector_max_retries: int = 3
    connector_retry_delay: float = 1.5


# SETTINGS SINGLETON


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance.

    Prevents repeated disk/env parsing
    across application modules.
    """
    return Settings()


# LOGGER FACTORY


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance.

    Ensures logging is initialized only once
    across the entire application lifecycle.
    """

    root_logger = logging.getLogger()

    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
        )

        # Silence noisy dependencies
        logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
        logging.getLogger("chromadb.segment").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    return logging.getLogger(name)


# FILE INGESTION HELPERS


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".docx",
}


def is_supported(path: str | Path) -> bool:
    """
    Returns True if file extension is supported
    for ingestion pipeline.
    """
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS