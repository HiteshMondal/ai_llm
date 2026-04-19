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
    chunk_size: int = 700          # characters per chunk — 700 balances context vs precision
    chunk_overlap: int = 120       # repeated chars between chunks — prevents cutting sentences mid-way

    # QUERY CACHE CONFIG
    query_cache_ttl_seconds: int = 600    # seconds before a cached answer expires (600 = 10 min)
    query_cache_max_size: int = 256       # max unique questions stored in cache at once

    # RERANKER CONFIG
    reranker_enabled: bool = False        # set True to re-score retrieved chunks for better accuracy (slower)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # model used for re-scoring
    reranker_top_n: int = 5              # how many top chunks to pass to LLM after re-ranking

    # SESSION MEMORY CONFIG
    session_memory_turns: int = 6        # past conversation turns sent to LLM — higher = better memory, slower

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
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS