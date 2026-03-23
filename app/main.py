from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, ingest, chat
from app.utils.config import get_settings
from app.utils.logger import get_logger

settings = get_settings()
log = get_logger(__name__)

app = FastAPI(
    title="AI LLM RAG App",
    version="1.0.0",
    description="A local RAG application powered by LangChain and configurable LLM/embedding backends.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(chat.router)


@app.on_event("startup")
async def startup():
    log.info(f"Starting AI LLM RAG App | LLM: {settings.llm_provider}/{settings.llm_model}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
    )