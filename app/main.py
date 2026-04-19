from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api import health, ingest, chat, manage, sources
from app.config import get_settings
from app.rag import list_documents, ingest_documents
from app.rag import load_file, clean_documents, chunk_documents
from app.config import get_settings, get_logger

DEFAULT_FILE = Path("data/uploads/default_knowledge.txt")

settings = get_settings()
log = get_logger(__name__)


#  Lifespan 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup logic: auto-ingest default knowledge base if DB is empty."""
    try:
        docs_existing = list_documents()
        if not docs_existing and DEFAULT_FILE.exists():
            log.info("Vector DB empty — ingesting default knowledge base...")
            docs = load_file(DEFAULT_FILE)
            docs = clean_documents(docs)
            chunks = chunk_documents(docs)
            ingest_documents(chunks)
            log.info("Default knowledge base indexed successfully.")
    except Exception as e:
        log.warning(f"Default ingestion skipped: {e}")

    log.info(
        f"RAG App ready | "
        f"LLM: {settings.llm_provider}/{settings.llm_model} | "
        f"Embeddings: {settings.embedding_provider or settings.embedding_model} | "
        f"Chunks: size={settings.chunk_size} overlap={settings.chunk_overlap} | "
        f"Cache TTL: {settings.query_cache_ttl_seconds}s | "
        f"Re-ranker: {'on' if settings.reranker_enabled else 'off'}"
    )
    yield
    log.info("RAG App shutting down.")


#  App 

app = FastAPI(
    title="AI RAG App",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health)
app.include_router(ingest)
app.include_router(chat)
app.include_router(manage)
app.include_router(sources)


@app.get("/")
def root():
    return RedirectResponse(url="/health")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
    )