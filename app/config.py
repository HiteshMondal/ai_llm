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
    embedding_provider: str = "local"   # local | gemini | openai

    # Vector store
    vector_collection_name: str = "rag_docs"
    chroma_persist_dir: str = "data/embeddings"

    # Ingestion
    chunk_size: int = 500
    chunk_overlap: int = 50

    # LLM Provider switching
    llm_provider: str = "ollama"   # ollama | gemini | groq | openrouter | openai

    # API keys (all optional — can also be passed per-request from UI)
    gemini_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    openai_api_key: str = ""

    # Connector: Google Drive
    gdrive_credentials_file: str = "credentials.json"
    gdrive_token_file: str = "token.json"
    gdrive_folder_id: str = ""

    # Connector: Notion
    notion_token: str = ""
    notion_page_ids: str = ""       # comma-separated
    notion_database_ids: str = ""   # comma-separated

    # Connector: GitHub
    github_token: str = ""
    github_repos: str = ""          # comma-separated: owner/repo1,owner/repo2
    github_branch: str = ""

@lru_cache
def get_settings() -> Settings:
    return Settings()

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS