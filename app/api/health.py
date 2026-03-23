from fastapi import APIRouter
from app.utils.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "app": "AI LLM RAG App",
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "vector_store": settings.vector_store_provider,
    }