from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import shutil
import mimetypes

from app.logger import get_logger
from app.rag import query, ingest_documents
from app.ingest import (
    load_file,
    clean_documents,
    chunk_documents,
)


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

@chat.post("/chat")
def chat_endpoint(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question empty")
    return query(req.question, req.k, req.system_instruction)


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