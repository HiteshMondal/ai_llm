from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import shutil
import mimetypes

from app.config import get_settings
from app.logger import get_logger
from app.rag import query, ingest_documents
from app.ingest import (
    load_file,
    clean_documents,
    chunk_documents,
)

settings = get_settings()

log = get_logger(__name__)


# HEALTH ROUTER

health = APIRouter()


@health.get("/health")
def health_check():
    return {"status": "ok"}


# CHAT ROUTER

chat = APIRouter()

class ChatRequest(BaseModel):
    question: str
    k: int = 4
    system_instruction: str = ""
    provider: str = ""
    model: str = ""
    api_key: str = ""

@chat.post("/chat")
def chat_endpoint(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question empty")
    return query(
        req.question, req.k, req.system_instruction,
        provider=req.provider, model=req.model, api_key=req.api_key,
    )


# INGEST ROUTER

ingest = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@ingest.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    filename = Path(file.filename).name
    dest = UPLOAD_DIR / filename

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # If no extension, try to detect it
    if not dest.suffix:
        mime = file.content_type or mimetypes.guess_type(filename)[0] or ""
        if "markdown" in mime or filename.lower().endswith("md"):
            dest = dest.rename(dest.with_suffix(".md"))
        else:
            # Try reading as plain text as fallback
            try:
                dest.read_text(encoding="utf-8", errors="strict")
                dest = dest.rename(dest.with_suffix(".txt"))
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot determine file type for '{filename}'. Add .txt or .md extension.",
                )
        filename = dest.name

    docs = load_file(dest)
    if not docs:
        raise HTTPException(
            status_code=400,
            detail="Could not load file",
        )

    docs = clean_documents(docs)

    chunks = chunk_documents(docs)

    count = ingest_documents(chunks)

    return {
        "filename": filename,
        "chunks_ingested": count,
    }

# MANAGE ROUTER

manage = APIRouter()

@manage.get("/documents")
def list_docs():
    from app.rag import list_documents
    return {"documents": list_documents()}

@manage.delete("/documents/{source}")
def delete_doc(source: str):
    from app.rag import delete_documents
    from urllib.parse import unquote
    source = unquote(source)
    deleted = delete_documents(source)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No document found: {source}")
    # Remove uploaded file if exists
    file_path = UPLOAD_DIR / source
    if file_path.exists():
        file_path.unlink()
    return {"deleted_chunks": deleted, "source": source}


# SOURCES ROUTER
sources = APIRouter()

class GDriveRequest(BaseModel):
    credentials_file: str = "credentials.json"
    token_file: str = "token.json"
    folder_id: str = ""

class NotionRequest(BaseModel):
    token: str
    page_ids: str = ""        # comma-separated
    database_ids: str = ""    # comma-separated

class WebRequest(BaseModel):
    urls: str                 # comma or newline separated

class GithubRequest(BaseModel):
    token: str
    repos: str                # comma-separated owner/repo
    branch: str = ""

def _run_connector(connector) -> dict:
    from app.ingest import clean_documents, chunk_documents
    docs = connector.fetch()
    if not docs:
        raise HTTPException(status_code=400, detail="No documents fetched from source")
    docs = clean_documents(docs)
    chunks = chunk_documents(docs)
    count = ingest_documents(chunks)
    return {"documents_fetched": len(docs), "chunks_ingested": count}

@sources.post("/sources/ingest/gdrive")
def ingest_gdrive(req: GDriveRequest):
    from app.connectors import GDriveConnector
    connector = GDriveConnector(
        credentials_file=req.credentials_file or settings.gdrive_credentials_file,
        token_file=req.token_file or settings.gdrive_token_file,
        folder_id=req.folder_id or settings.gdrive_folder_id or None,
    )
    return _run_connector(connector)

@sources.post("/sources/ingest/notion")
def ingest_notion(req: NotionRequest):
    from app.connectors import NotionConnector
    token = req.token or settings.notion_token
    if not token:
        raise HTTPException(status_code=400, detail="Notion token required")
    page_ids = [p.strip() for p in req.page_ids.split(",") if p.strip()]
    db_ids   = [d.strip() for d in req.database_ids.split(",") if d.strip()]
    connector = NotionConnector(token=token, page_ids=page_ids, database_ids=db_ids)
    return _run_connector(connector)

@sources.post("/sources/ingest/web")
def ingest_web(req: WebRequest):
    from app.connectors import WebConnector
    import re
    urls = [u.strip() for u in re.split(r"[,\n]+", req.urls) if u.strip()]
    if not urls:
        raise HTTPException(status_code=400, detail="No URLs provided")
    connector = WebConnector(urls=urls)
    return _run_connector(connector)

@sources.post("/sources/ingest/github")
def ingest_github(req: GithubRequest):
    from app.connectors import GithubConnector
    token = req.token or settings.github_token
    if not token:
        raise HTTPException(status_code=400, detail="GitHub token required")
    repos = [r.strip() for r in req.repos.split(",") if r.strip()]
    if not repos:
        raise HTTPException(status_code=400, detail="No repos provided")
    connector = GithubConnector(
        token=token,
        repos=repos,
        branch=req.branch or settings.github_branch or None,
    )
    return _run_connector(connector)