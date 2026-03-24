from functools import lru_cache
from typing import List, Dict, Any

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.logger import get_logger

PROVIDER_MODELS = {
    "ollama":     "tinyllama",
    "gemini":     "gemini-2.0-flash-lite",
    "groq":       "llama3-8b-8192",
    "openrouter": "mistralai/mistral-7b-instruct:free",
    "openai":     "gpt-3.5-turbo",
    "phi3":       "mini",
}

settings = get_settings()
log = get_logger(__name__)


# Prompt Template

PROMPT = PromptTemplate(
    input_variables=["context", "input", "system_instruction"],
    template=(
        "{system_instruction}\n\n"
        "Use the context below to answer the question.\n"
        "If you don't know, say 'I don't know'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
)

DEFAULT_INSTRUCTION = "You are a helpful assistant. Answer based on the provided context."


# Embedding Model

@lru_cache
def get_embedder() -> Embeddings:
    provider = settings.embedding_provider.lower()

    if provider == "gemini":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError:
            raise ImportError("pip install langchain-google-genai")
        key = settings.gemini_api_key
        if not key:
            raise ValueError("GEMINI_API_KEY required for gemini embeddings")
        log.info("Loading embeddings: Gemini (API, no local model)")
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=key,
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        log.info("Loading embeddings: OpenAI (API, no local model)")
        return OpenAIEmbeddings(api_key=settings.openai_api_key)

    # Default: local HuggingFace (needs ~500MB RAM)
    log.info(f"Loading embeddings: local {settings.embedding_model}")
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# LLM Model

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

    # Default: Ollama (local, no key needed)
    return ChatOllama(
        model=model or settings.llm_model,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
    )


# Vector Store

@lru_cache
def get_vector_store() -> Chroma:

    return Chroma(
        collection_name=settings.vector_collection_name,
        embedding_function=get_embedder(),
        persist_directory=settings.chroma_persist_dir,
    )


# Ingestion

def ingest_documents(docs: List[Document]) -> int:

    if not docs:
        log.warning("No documents to ingest.")
        return 0

    vs = get_vector_store()

    vs.add_documents(docs)

    log.info(f"Ingested {len(docs)} chunk(s).")

    return len(docs)


# Query Pipeline

def query(
    question: str,
    k: int = 4,
    system_instruction: str = "",
    provider: str = "",
    model: str = "",
    api_key: str = "",
) -> Dict[str, Any]:
    log.info(f"Query received: {question}")
    instruction = system_instruction.strip() or DEFAULT_INSTRUCTION
    retriever = get_vector_store().as_retriever(search_kwargs={"k": k})
    llm = get_llm(provider=provider, model=model, api_key=api_key)
    document_chain = create_stuff_documents_chain(llm, PROMPT)
    chain = create_retrieval_chain(retriever, document_chain).with_config({"run_name": "rag_chain"})
    result = chain.invoke({"input": question, "system_instruction": instruction})
    return {
        "answer": result.get("answer", ""),
        "sources": [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in result.get("context", [])
        ],
    }

# List all documents

def list_documents() -> List[Dict[str, Any]]:
    vs = get_vector_store()
    result = vs.get(include=["metadatas"])
    seen = {}
    for i, meta in enumerate(result["metadatas"]):
        source = meta.get("source", "unknown")
        doc_id = result["ids"][i]
        if source not in seen:
            seen[source] = {"source": source, "ids": []}
        seen[source]["ids"].append(doc_id)
    return [
        {"source": s, "chunk_count": len(v["ids"])}
        for s, v in seen.items()
    ]

# Delete documents by source filename

def delete_documents(source: str) -> int:
    vs = get_vector_store()
    result = vs.get(include=["metadatas"])
    ids_to_delete = [
        result["ids"][i]
        for i, meta in enumerate(result["metadatas"])
        if meta.get("source") == source
    ]
    if not ids_to_delete:
        log.warning(f"No chunks found for source: {source}")
        return 0
    vs.delete(ids=ids_to_delete)
    log.info(f"Deleted {len(ids_to_delete)} chunk(s) for '{source}'")
    return len(ids_to_delete)