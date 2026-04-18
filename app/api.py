from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import shutil
import mimetypes
import json
import re
from pydantic import Field
from app.config import get_settings
from app.rag import query, stream_query, ingest_documents, get_document_preview
from app.rag import load_file, clean_documents, chunk_documents
from urllib.parse import unquote
from app.config import get_logger

settings = get_settings()
log = get_logger(__name__)


#  Health 
health = APIRouter()

@health.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0"}


#  Chat 

chat = APIRouter()


class ChatRequest(BaseModel):
    question: str
    k: int = 4
    system_instruction: str = ""
    provider: str = ""
    model: str = ""
    api_key: str = ""
    history: list = Field(default_factory=list)

def _sanitize_history(history: list) -> list:
    """Ensure every history entry is a plain dict with role/content keys."""
    result = []
    for turn in history:
        if isinstance(turn, dict):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if isinstance(content, str):
                result.append({"role": role, "content": content})
    return result

@chat.post("/chat")
def chat_endpoint(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question empty")
    return query(
        req.question,
        req.k,
        req.system_instruction,
        provider=req.provider,
        model=req.model,
        api_key=req.api_key,
        history=_sanitize_history(req.history),
    )


@chat.post("/chat/stream")
def chat_stream_endpoint(req: ChatRequest):
    """Server-Sent Events streaming endpoint."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question empty")

    def _generate():
        try:
            for token in stream_query(
                req.question,
                req.k,
                req.system_instruction,
                provider=req.provider,
                model=req.model,
                api_key=req.api_key,
                history=_sanitize_history(req.history),
            ):
                yield f"data: {json.dumps({'token': str(token)}, ensure_ascii=False)}\n\n"
        except Exception as e:
            log.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "event: done\ndata: done\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


#  Ingest 

ingest = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@ingest.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", Path(file.filename).name)
    filename = filename.replace("..", "_")
    dest = UPLOAD_DIR / filename
    if dest.exists():
        raise HTTPException(
            status_code=409,
            detail=f"{filename} already exists"
        )

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Auto-detect extension if missing
    if not dest.suffix:
        mime = file.content_type or mimetypes.guess_type(filename)[0] or ""
        if "markdown" in mime or filename.lower().endswith("md"):
            dest = dest.rename(dest.with_suffix(".md"))
        else:
            try:
                dest.read_text(encoding="utf-8", errors="strict")
                dest = dest.rename(dest.with_suffix(".txt"))
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Cannot determine file type for '{filename}'. "
                        "Add .txt or .md extension."
                    ),
                )
        filename = dest.name

    docs = load_file(dest)
    if not docs:
        raise HTTPException(status_code=400, detail="Could not load file")

    docs = clean_documents(docs)
    chunks = chunk_documents(docs)
    count = ingest_documents(chunks)

    return {"filename": filename, "chunks_ingested": count}


#  Manage 

manage = APIRouter()


@manage.get("/documents")
def list_docs():
    from app.rag import list_documents
    return {"documents": list_documents()}


@manage.get("/documents/{source}/preview")
def preview_doc(source: str, max_chars: int = Query(default=2000, le=10000)):
    source = Path(unquote(source)).name
    preview = get_document_preview(source, max_chars=max_chars)
    if not preview:
        raise HTTPException(status_code=404, detail=f"No document found: {source}")
    return {"source": source, "preview": preview}


@manage.delete("/documents/{source}")
def delete_doc(source: str):
    from app.rag import delete_documents
    source = Path(unquote(source)).name
    deleted = delete_documents(source)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No document found: {source}")
    file_path = UPLOAD_DIR / source
    if file_path.exists():
        file_path.unlink()
    return {"deleted_chunks": deleted, "source": source}


#  Sources 

sources = APIRouter()

class GDriveRequest(BaseModel):
    credentials_file: str = "credentials.json"
    token_file: str = "token.json"
    folder_id: str = ""

class NotionRequest(BaseModel):
    token: str
    page_ids: str = ""
    database_ids: str = ""

class WebRequest(BaseModel):
    urls: str

class GithubRequest(BaseModel):
    token: str
    repos: str
    branch: str = ""

def _run_connector(connector) -> dict:
    from app.rag import clean_documents, chunk_documents
    try:
        docs = connector.fetch()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Connector failed: {str(e)}"
        )
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
    db_ids = [d.strip() for d in req.database_ids.split(",") if d.strip()]
    connector = NotionConnector(token=token, page_ids=page_ids, database_ids=db_ids)
    return _run_connector(connector)


@sources.post("/sources/ingest/web")
def ingest_web(req: WebRequest):
    from app.connectors import WebConnector
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