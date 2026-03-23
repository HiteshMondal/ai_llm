import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.ingestion.loaders import load_file
from app.ingestion.cleaner import clean_documents
from app.ingestion.chunker import chunk_documents
from app.core.rag_pipeline import ingest_documents

router = APIRouter()
settings = get_settings()
log = get_logger(__name__)

SUPPORTED = {".pdf", ".txt", ".md", ".docx"}


@router.post("/ingest", tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    log.info(f"Received file: {file.filename}")

    docs = load_file(dest)
    docs = clean_documents(docs)
    chunks = chunk_documents(docs)
    count = ingest_documents(chunks)

    return {"filename": file.filename, "chunks_ingested": count}


@router.get("/ingest/list", tags=["Ingestion"])
def list_uploaded_files():
    upload_dir = Path(settings.upload_dir)
    if not upload_dir.exists():
        return {"files": []}
    files = [f.name for f in upload_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED]
    return {"files": files}