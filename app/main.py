from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api import health, ingest, chat, manage, sources
from app.config import get_settings
from app.logger import get_logger

settings = get_settings()
log = get_logger(__name__)


app = FastAPI(
    title="AI RAG App",
    version="1.0.0",
)


# Middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routers

app.include_router(health)
app.include_router(ingest)
app.include_router(chat)
app.include_router(manage)
app.include_router(sources)

@app.get("/")
def root():
    return RedirectResponse(url="/health")


# Startup Event

@app.on_event("startup")
async def startup_event():
    log.info(
        f"RAG App ready | "
        f"LLM provider: {settings.llm_provider} | "
        f"Embeddings: {settings.embedding_provider or settings.embedding_model}"
    )


# Local Run Support

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
    )