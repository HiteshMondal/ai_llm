from pathlib import Path
import re
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.logger import get_logger
from app.config import SUPPORTED_EXTENSIONS

log = get_logger(__name__)


# Text Cleaning

def clean_text(text: str) -> str:
    """Normalize whitespace and remove excessive blank lines."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def clean_documents(docs: List[Document]) -> List[Document]:
    """Clean document content safely."""
    cleaned_docs: List[Document] = []

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

        if doc.page_content:
            cleaned_docs.append(doc)
        else:
            log.warning(
                f"Document empty after cleaning: "
                f"{doc.metadata.get('source', '?')}"
            )

    log.info(f"Cleaned {len(cleaned_docs)}/{len(docs)} documents")

    return cleaned_docs


# Chunking

def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into embedding-sized chunks."""

    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    log.info(f"Split {len(docs)} doc(s) → {len(chunks)} chunk(s)")

    return chunks


# File Loading

def load_file(path: str | Path) -> List[Document]:
    """Load a single supported document."""

    path = Path(path)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        log.warning(
            f"Unsupported file type: {path.name} "
            "(only .txt and .md allowed)"
        )
        return []

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore"
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
        log.error(f"Failed to load '{path.name}': {e}")
        return []


def load_directory(directory: str | Path) -> List[Document]:
    """Recursively load supported documents from directory."""

    directory = Path(directory)

    docs: List[Document] = []

    for file in directory.rglob("*"):
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.extend(load_file(file))

    log.info(f"Loaded {len(docs)} doc(s) from '{directory}'")

    return docs