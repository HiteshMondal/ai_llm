from pathlib import Path
import re
import hashlib
import time
import threading
import json
import torch

from functools import lru_cache
from typing import List, Dict, Any, Iterator, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from app.config import (
    get_settings,
    get_logger,
    SUPPORTED_EXTENSIONS,
)

_lock = threading.Lock()
log = get_logger(__name__)
settings = get_settings()

PROVIDER_MODELS = {
    "ollama":     "tinyllama",
    "gemini":     "gemini-2.0-flash-lite",
    "groq":       "llama3-8b-8192",
    "openrouter": "mistralai/mistral-7b-instruct:free",
    "openai":     "gpt-3.5-turbo",
}


#  Prompt 

PROMPT = PromptTemplate(
    input_variables=["context", "input", "system_instruction", "chat_history"],
    template=(
        "{system_instruction}\n\n"
        "{chat_history}"
        "Use retrieved context when relevant.\n"
        "If context is not relevant, answer normally.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
)

DEFAULT_INSTRUCTION = """
You are a high-precision AI assistant powered by Retrieval-Augmented Generation (RAG).

Follow this reasoning process when answering:

1. If retrieved context directly answers the question:
   Use it as the primary source.

2. If retrieved context partially answers the question:
   Combine the context with your own knowledge to complete the answer.

3. If retrieved context is irrelevant or missing:
   Answer using general knowledge and clearly state when doing so.

Rules:

- Never fabricate facts from missing context.
- Prefer accurate and structured explanations.
- Explain technical topics step-by-step.
- Reference system components, architecture, or workflows when helpful.
- Keep answers concise but complete.
- When uncertainty exists, state assumptions clearly.
- Do not repeat context verbatim unless necessary.
- Prioritize clarity, correctness, and usefulness over verbosity.

Output style:

- Use structured formatting when appropriate
- Use bullet points for multi-step explanations
- Use examples when helpful
"""

def detect_intent(message: str) -> str:
    msg = message.lower().strip()

    greetings = {
        "hi", "hello", "hey", "yo",
        "good morning", "good afternoon", "good evening"
    }

    casual_patterns = [
        r"^hi+$",
        r"^hello+$",
        r"^hey+$"
    ]

    for pattern in casual_patterns:
        if re.match(pattern, msg):
            return "greeting"

    if msg in greetings:
        return "greeting"

    if msg in {"thanks", "thank you", "thx"}:
        return "thanks"

    if msg in {"ok", "okay", "cool", "nice"}:
        return "confirmation"

    return "query"

# TEXT CLEANING

def clean_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def clean_documents(docs: List[Document]) -> List[Document]:

    cleaned: List[Document] = []

    for doc in docs:

        doc.page_content = clean_text(doc.page_content)

        if doc.page_content:
            cleaned.append(doc)

        else:
            log.warning(
                f"Document empty after cleaning: "
                f"{doc.metadata.get('source', '?')}"
            )

    log.info(f"Cleaned {len(cleaned)}/{len(docs)} documents")

    return cleaned


# CHUNKING

def chunk_documents(docs: List[Document]) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    log.info(f"Split {len(docs)} doc(s) → {len(chunks)} chunk(s)")

    return chunks


# FILE LOADING

def load_file(path: str | Path) -> List[Document]:

    path = Path(path)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:

        log.warning(
            f"Unsupported file type: {path.name}. "
            f"Allowed: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

        return []

    try:

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        log.info(
            f"Loaded '{path.name}' "
            f"({len(text)} characters)"
        )

        return [

            Document(
                page_content=text,
                metadata={"source": path.name},
            )

        ]

    except Exception as e:

        log.error(
            f"Failed to load '{path.name}': {e}"
        )

        return []


def load_directory(directory: str | Path) -> List[Document]:

    directory = Path(directory)

    docs: List[Document] = []

    for file in directory.rglob("*"):

        if file.is_file():

            docs.extend(load_file(file))

    log.info(
        f"Loaded {len(docs)} doc(s) from '{directory}'"
    )

    return docs

#  Query Cache 

class _QueryCache:
    """Simple TTL-aware in-memory cache keyed on query hash."""

    def __init__(self, ttl: int, max_size: int):
        self._ttl = ttl
        self._max = max_size
        self._store: Dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def _key(self, question: str, instruction: str, k: int, provider: str, model: str) -> str:
        raw = f"{question}|{instruction}|{k}|{provider}|{model}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, question, instruction, k, provider, model):

        key = self._key(question, instruction, k, provider, model)

        with self._lock:
            entry = self._store.get(key)

            if entry is None:
                return None

            ts, value = entry

            if time.time() - ts > self._ttl:
                del self._store[key]
                return None

            return value

    def set(self, question, instruction, k, provider, model, value):

        key = self._key(question, instruction, k, provider, model)

        with self._lock:

            if len(self._store) >= self._max:
                oldest = min(self._store, key=lambda k: self._store[k][0])
                del self._store[oldest]

            self._store[key] = (time.time(), value)

    def invalidate_all(self):

        with self._lock:
            self._store.clear()


_cache = _QueryCache(
    ttl=settings.query_cache_ttl_seconds,
    max_size=settings.query_cache_max_size,
)


#  Embeddings 

@lru_cache
def get_embedder() -> Embeddings:
    provider = (settings.embedding_provider or "").lower()

    if provider == "gemini":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError:
            raise ImportError("pip install langchain-google-genai")
        key = settings.gemini_api_key
        if not key:
            raise ValueError("GEMINI_API_KEY required for gemini embeddings")
        log.info("Loading embeddings: Gemini")
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=key,
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        log.info("Loading embeddings: OpenAI")
        return OpenAIEmbeddings(api_key=settings.openai_api_key)

    log.info(f"Loading embeddings: local {settings.embedding_model}")
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.ingest_batch_size,
        },
    )


#  LLM 

def get_llm(
    provider: str = "",
    model: str = "",
    api_key: str = "",
) -> BaseChatModel:
    provider = (provider or settings.llm_provider).lower().strip()
    model = model or PROVIDER_MODELS.get(provider, "")
    api_key = api_key or ""

    log.info(f"Loading LLM: provider={provider} model={model}")

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("pip install langchain-google-genai")
        key = api_key or settings.gemini_api_key
        if not key:
            raise ValueError("GEMINI_API_KEY missing. Get a free key at https://aistudio.google.com")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=key,
            temperature=settings.llm_temperature,
        )

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError("pip install langchain-groq")
        key = api_key or settings.groq_api_key
        if not key:
            raise ValueError("GROQ_API_KEY missing. Get a free key at https://console.groq.com")
        return ChatGroq(
            model=model,
            groq_api_key=key,
            temperature=settings.llm_temperature,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        key = api_key or settings.openrouter_api_key
        if not key:
            raise ValueError("OPENROUTER_API_KEY missing. Get one at https://openrouter.ai")
        return ChatOpenAI(
            model=model,
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            temperature=settings.llm_temperature,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        key = api_key or settings.openai_api_key
        if not key:
            raise ValueError("OPENAI_API_KEY missing.")
        return ChatOpenAI(
            model=model,
            api_key=key,
            temperature=settings.llm_temperature,
        )

    return ChatOllama(
        model=model or settings.llm_model or PROVIDER_MODELS["ollama"],
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
    )


#  Vector Store 

_vector_store = None
_vector_lock = threading.Lock()

def get_vector_store():

    global _vector_store

    if _vector_store is None:

        with _vector_lock:

            if _vector_store is None:

                _vector_store = Chroma(
                    collection_name=settings.vector_collection_name,
                    embedding_function=get_embedder(),
                    persist_directory=settings.chroma_persist_dir,
                )

    return _vector_store


#  Re-ranker 

@lru_cache
def _get_reranker():
    """Load cross-encoder re-ranker (lazy, cached)."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ImportError("pip install sentence-transformers")
    log.info(f"Loading re-ranker: {settings.reranker_model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(settings.reranker_model, device=device)


def rerank_documents(question: str, docs: List[Document]) -> List[Document]:
    """Re-rank retrieved docs using a cross-encoder. Falls back gracefully."""
    if not settings.reranker_enabled or not docs:
        return docs
    try:
        reranker = _get_reranker()
        pairs = [(question, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: settings.reranker_top_n]]
    except Exception as e:
        log.warning(f"Re-ranking failed, using original order: {e}")
        return docs


#  Session Memory 

def format_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""
    lines = ["Previous conversation:"]
    for turn in history[-(settings.session_memory_turns * 2):]:
        if not isinstance(turn, dict):
            log.warning(f"format_history: skipping non-dict turn: {type(turn)} = {turn!r}")
            continue
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("")
    return "\n".join(lines) + "\n"


#  Batch Ingestion 

def ingest_documents(docs: List[Document]) -> int:
    """Ingest documents in batches to avoid memory spikes."""
    if not docs:
        log.warning("No documents to ingest.")
        return 0

    vs = get_vector_store()
    batch_size = settings.ingest_batch_size
    total = 0

    for i in range(0, len(docs), batch_size):
        batch = docs[i: i + batch_size]
        vs.add_documents(batch)
        total += len(batch)
        log.info(f"Ingested batch {i // batch_size + 1}: {len(batch)} chunk(s)")

    # Invalidate cache after new ingestion
    _cache.invalidate_all()
    log.info(f"Total ingested: {total} chunk(s). Query cache cleared.")
    list_documents.cache_clear()
    return total

#  Retrieval

def should_use_retrieval(question: str) -> bool:

    q = question.lower().strip()

    # remove punctuation
    q_clean = re.sub(r"[^\w\s]", "", q)

    casual = {"hi", "hello", "hey", "thanks", "ok", "yo"}

    # single-word non-technical prompts
    if len(q_clean.split()) == 1 and q_clean in casual:
        return False

    # arithmetic detection
    if re.fullmatch(r"\d+\s*[-+*/]\s*\d+", q_clean):
        return False

    return True

#  Query Pipeline 

def query(
    question: str,
    k: int = 4,
    system_instruction: str = "",
    provider: str = "",
    model: str = "",
    api_key: str = "",
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:

    log.info(f"Query: {question!r}")

    llm = get_llm(provider=provider, model=model, api_key=api_key)

    instruction = system_instruction.strip() or DEFAULT_INSTRUCTION
    chat_history_str = format_history(history or [])

    intent = detect_intent(question)

    if intent == "greeting":
        return {"answer": "Hello! How can I help you today?", "sources": [], "cached": False, "reranked": False}

    if intent == "thanks":
        return {"answer": "You're welcome!", "sources": [], "cached": False, "reranked": False}

    # Cache
    eff_provider = (provider or settings.llm_provider).lower().strip()
    eff_model = model or PROVIDER_MODELS.get(eff_provider, "")

    if not history:
        cached = _cache.get(question, instruction, k, eff_provider, eff_model)
        if cached:
            return {
                "answer": str(cached["answer"]),
                "sources": cached.get("sources", []),
                "cached": True,
                "reranked": False,
            }

    # Retrieval
    final_docs = []

    if should_use_retrieval(question):
        try:
            vs = get_vector_store()
            results_with_scores = vs.similarity_search_with_score(question, k=max(k * 4, settings.reranker_top_n * 2))
            relevant = [doc for doc, score in results_with_scores if score < 1.0]
            final_docs = rerank_documents(question, relevant)[:k]
            log.info(f"Retrieved {len(final_docs)} relevant doc(s) (filtered from {len(results_with_scores)})")
        except Exception as e:
            log.warning(f"Retrieval failed: {e}")

    # NO RAG → direct LLM
    if not final_docs:
        messages = [
            SystemMessage(content=instruction),
            HumanMessage(content=f"{chat_history_str}\nUser question:\n{question}")
        ]

        response = llm.invoke(messages)

        result = {
            "answer": response.content,
            "sources": [],
            "cached": False,
            "reranked": False,
        }

        if not history:
            _cache.set(
                question,
                instruction,
                k,
                eff_provider,
                eff_model,
                {
                    "answer": result["answer"],
                    "sources": [],
                }
            )

        return result

    # RAG mode
    document_chain = create_stuff_documents_chain(llm, PROMPT)

    result = document_chain.invoke({
        "context": final_docs,
        "input": question,
        "system_instruction": instruction,
        "chat_history": chat_history_str,
    })

    output = {
        "answer": result,
        "sources": [
            {"content": d.page_content, "metadata": d.metadata}
            for d in final_docs
        ],
        "cached": False,
        "reranked": settings.reranker_enabled,
    }

    if not history:
        _cache.set(
            question,
            instruction,
            k,
            eff_provider,
            eff_model,
            {
                "answer": str(result),
                "sources": [{"content": d.page_content, "metadata": d.metadata} for d in final_docs],
                "cached": False,
                "reranked": False,
            }
        )

    return output

def stream_query(
    question: str,
    k: int = 4,
    system_instruction: str = "",
    provider: str = "",
    model: str = "",
    api_key: str = "",
    history: Optional[List[Dict[str, str]]] = None,
) -> Iterator[str]:

    log.info(f"Stream query: {question!r}")

    intent = detect_intent(question)
    if intent == "greeting":
        yield "Hello! How can I help you today?"
        return
    if intent == "thanks":
        yield "You're welcome!"
        return

    instruction = system_instruction.strip() or DEFAULT_INSTRUCTION
    chat_history_str = format_history(history or [])

    llm = get_llm(provider=provider, model=model, api_key=api_key)

    # Retrieval
    final_docs = []
    if should_use_retrieval(question):
        try:
            vs = get_vector_store()
            results_with_scores = vs.similarity_search_with_score(question, k=max(k * 4, settings.reranker_top_n * 2))
            # Lower score = more similar in Chroma (L2 distance). Filter threshold.
            relevant = [doc for doc, score in results_with_scores if score < 1.0]
            final_docs = rerank_documents(question, relevant)[:k]
            log.info(f"Retrieved {len(final_docs)} relevant doc(s) (filtered from {len(results_with_scores)})")
        except Exception as e:
            log.warning(f"Retrieval failed during streaming: {e}")

    if final_docs:
        # RAG mode: build context and stream
        context_text = "\n\n".join(d.page_content for d in final_docs)
        prompt = (
            f"{instruction}\n\n"
            f"{chat_history_str}"
            f"Use retrieved context when relevant.\n"
            f"If context is not relevant, answer normally.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        messages = [HumanMessage(content=prompt)]
    else:
        # No RAG: direct LLM
        messages = [
            SystemMessage(content=instruction),
            HumanMessage(content=f"{chat_history_str}\nUser question:\n{question}")
        ]

    try:
        for chunk in llm.stream(messages):
            token = getattr(chunk, "content", None)
            if not token:
                token = getattr(chunk, "text", "")
            if token:
                yield token
    except Exception as e:
        log.error(f"Streaming failed: {e}")
        response = llm.invoke(messages)
        yield response.content

#  Document Management 
@lru_cache(maxsize=1)

def list_documents() -> List[Dict[str, Any]]:
    vs = get_vector_store()
    result = vs.get(include=["metadatas"])
    seen: Dict[str, Dict] = {}
    for i, meta in enumerate(result.get("metadatas", [])):
        source = meta.get("source", "unknown")
        doc_id = result["ids"][i]
        if source not in seen:
            seen[source] = {
                "source": source,
                "ids": [],
                "source_type": meta.get("source_type", "local"),
            }
        seen[source]["ids"].append(doc_id)
    return [
        {
            "source": s,
            "chunk_count": len(v["ids"]),
            "source_type": v["source_type"],
        }
        for s, v in seen.items()
    ]


def get_document_preview(source: str, max_chars: int = 2000) -> str:
    """Return a text preview of an ingested document by source name."""
    vs = get_vector_store()
    result = vs.get(where={"source": source}, include=["documents"])
    chunks = result.get("documents", [])

    if not chunks:
        return ""
    full_text = "\n\n---\n\n".join(chunks)
    return full_text[:max_chars] + ("…" if len(full_text) > max_chars else "")


def delete_documents(source: str) -> int:
    vs = get_vector_store()
    result = vs.get(include=["metadatas"])
    ids_to_delete = [
        result["ids"][i]
        for i, meta in enumerate(result.get("metadatas", []))
        if meta.get("source") == source
    ]
    if not ids_to_delete:
        log.warning(f"No chunks found for source: {source}")
        return 0
    vs.delete(ids=ids_to_delete)
    list_documents.cache_clear()
    _cache.invalidate_all()
    log.info(f"Deleted {len(ids_to_delete)} chunk(s) for '{source}'")
    return len(ids_to_delete)