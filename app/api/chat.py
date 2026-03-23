from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.rag_pipeline import query
from app.utils.logger import get_logger

router = APIRouter()
log = get_logger(__name__)


class ChatRequest(BaseModel):
    question: str
    k: int = 4        # number of retrieved chunks


class SourceDoc(BaseModel):
    content: str
    metadata: dict


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDoc]


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    log.info(f"Chat request: {request.question!r}")
    result = query(request.question, k=request.k)

    return ChatResponse(
        question=request.question,
        answer=result["answer"],
        sources=[SourceDoc(**s) for s in result["sources"]],
    )