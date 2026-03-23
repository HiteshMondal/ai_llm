from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from app.core.embedder import get_embedder
from app.core.llm_engine import get_llm
from app.utils.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()

PROMPT_TEMPLATE = """Use the following context to answer the question at the end.
If you don't know the answer, say "I don't know" — do not make up an answer.

Context:
{context}

Question: {question}

Answer:"""


def _get_vector_store() -> Chroma:
    embedder = get_embedder()
    return Chroma(
        collection_name=settings.vector_collection_name,
        embedding_function=embedder,
        persist_directory=settings.chroma_persist_dir,
    )


def ingest_documents(docs: list[Document]) -> int:
    """Add documents to the vector store. Returns number of chunks stored."""
    if not docs:
        log.warning("No documents to ingest.")
        return 0

    vs = _get_vector_store()
    vs.add_documents(docs)
    log.info(f"Ingested {len(docs)} chunk(s) into vector store.")
    return len(docs)


def query(question: str, k: int = 4) -> dict:
    """Run a RAG query and return answer + source documents."""
    vs = _get_vector_store()
    retriever = vs.as_retriever(search_kwargs={"k": k})

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    log.info(f"Running RAG query: {question!r}")
    result = chain.invoke({"query": question})

    return {
        "answer": result.get("result", ""),
        "sources": [
            {"content": d.page_content, "metadata": d.metadata}
            for d in result.get("source_documents", [])
        ],
    }