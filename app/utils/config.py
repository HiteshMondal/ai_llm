import os
import yaml
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = BASE_DIR / "config.yaml"


def _load_yaml() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml = _load_yaml()


class Settings(BaseSettings):
    # App
    app_host: str = Field(default=_yaml.get("app", {}).get("host", "0.0.0.0"))
    app_port: int = Field(default=_yaml.get("app", {}).get("port", 8000))
    app_debug: bool = Field(default=_yaml.get("app", {}).get("debug", False))

    # LLM
    llm_provider: str = Field(default=_yaml.get("llm", {}).get("provider", "ollama"))
    llm_model: str = Field(default=_yaml.get("llm", {}).get("model", "llama3.2"))
    llm_base_url: str = Field(default=_yaml.get("llm", {}).get("base_url", "http://localhost:11434"))
    llm_temperature: float = Field(default=_yaml.get("llm", {}).get("temperature", 0.7))
    llm_max_tokens: int = Field(default=_yaml.get("llm", {}).get("max_tokens", 2048))

    # API Keys
    openai_api_key: str = Field(default="")
    huggingface_api_key: str = Field(default="")

    # Embeddings
    embedding_provider: str = Field(default=_yaml.get("embeddings", {}).get("provider", "ollama"))
    embedding_model: str = Field(default=_yaml.get("embeddings", {}).get("model", "nomic-embed-text"))
    embedding_base_url: str = Field(default=_yaml.get("embeddings", {}).get("base_url", "http://localhost:11434"))
    embedding_dimension: int = Field(default=_yaml.get("embeddings", {}).get("dimension", 768))

    # Vector Store
    vector_store_provider: str = Field(default=_yaml.get("vector_store", {}).get("provider", "chroma"))
    chroma_persist_dir: str = Field(default=_yaml.get("vector_store", {}).get("persist_dir", "./data/embeddings"))
    vector_collection_name: str = Field(default=_yaml.get("vector_store", {}).get("collection_name", "documents"))

    # Ingestion
    chunk_size: int = Field(default=_yaml.get("ingestion", {}).get("chunk_size", 512))
    chunk_overlap: int = Field(default=_yaml.get("ingestion", {}).get("chunk_overlap", 64))
    upload_dir: str = Field(default=_yaml.get("ingestion", {}).get("upload_dir", "./data/uploads"))
    processed_dir: str = Field(default=_yaml.get("ingestion", {}).get("processed_dir", "./data/processed"))

    # Logging
    log_level: str = Field(default=_yaml.get("logging", {}).get("level", "INFO"))
    log_file: str = Field(default=_yaml.get("logging", {}).get("file", "./training/logs/app.log"))

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()