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
    embedding_provider: str = "local"       # local | gemini | openai

    # Vector store
    vector_collection_name: str = "rag_docs"
    chroma_persist_dir: str = "data/embeddings"

    # Ingestion
    chunk_size: int = 500
    chunk_overlap: int = 50
    ingest_batch_size: int = 64             # batch size for embedding ingestion

    # LLM Provider switching
    llm_provider: str = "ollama"            # ollama | gemini | groq | openrouter | openai

    # API keys (all optional — can also be passed per-request from UI)
    gemini_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    openai_api_key: str = ""

    # Query cache
    query_cache_ttl_seconds: int = 300      # 5 min TTL for cached answers
    query_cache_max_size: int = 256         # max number of cached queries

    # Re-ranking
    reranker_enabled: bool = False          # enable cross-encoder re-ranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 4                 # final docs after re-rank

    # Session memory
    session_memory_turns: int = 6           # number of past turns kept in context

    # Streaming
    stream_chunk_delay: float = 0.0         # optional artificial delay between chunks

    # Connector: Google Drive
    gdrive_credentials_file: str = "credentials.json"
    gdrive_token_file: str = "token.json"
    gdrive_folder_id: str = ""

    # Connector: Notion
    notion_token: str = ""
    notion_page_ids: str = ""
    notion_database_ids: str = ""

    # Connector: GitHub
    github_token: str = ""
    github_repos: str = ""
    github_branch: str = ""

    # Connector reliability
    connector_timeout: int = 30             # seconds per HTTP request in connectors
    connector_max_retries: int = 3          # retry attempts on transient failures
    connector_retry_delay: float = 1.5      # base delay between retries (exponential)


@lru_cache
def get_settings() -> Settings:
    return Settings()


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS