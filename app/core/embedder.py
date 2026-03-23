from langchain.embeddings.base import Embeddings
from app.utils.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


def get_embedder() -> Embeddings:
    """Return the configured embedding model."""
    provider = settings.embedding_provider.lower()

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        log.info(f"Using Ollama embeddings: {settings.embedding_model}")
        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.embedding_base_url,
        )

    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        log.info(f"Using OpenAI embeddings: {settings.embedding_model}")
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

    elif provider == "sentence-transformers":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        log.info(f"Using HuggingFace embeddings: {settings.embedding_model}")
        return HuggingFaceEmbeddings(model_name=settings.embedding_model)

    else:
        raise ValueError(f"Unsupported embedding provider: '{provider}'")