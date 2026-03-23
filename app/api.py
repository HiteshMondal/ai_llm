from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import shutil

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


@chat.post("/chat")
def chat_endpoint(req: ChatRequest):

    if not req.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question empty",
        )

    return query(req.question, req.k)


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