import re
from langchain.schema import Document
from app.utils.logger import get_logger

log = get_logger(__name__)


def clean_text(text: str) -> str:
    """Basic text cleaning: normalize whitespace and remove junk characters."""
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return text.strip()


def clean_documents(docs: list[Document]) -> list[Document]:
    """Apply clean_text to every document's page_content in place."""
    cleaned = []
    for doc in docs:
        original_len = len(doc.page_content)
        doc.page_content = clean_text(doc.page_content)
        if doc.page_content:
            cleaned.append(doc)
        else:
            log.warning(f"Document became empty after cleaning (source: {doc.metadata.get('source', 'unknown')})")

    log.info(f"Cleaned {len(cleaned)}/{len(docs)} documents")
    return cleaned