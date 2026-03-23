from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.utils.config import get_settings
from app.utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    log.info(f"Split {len(docs)} document(s) into {len(chunks)} chunk(s) "
             f"(size={settings.chunk_size}, overlap={settings.chunk_overlap})")
    return chunks