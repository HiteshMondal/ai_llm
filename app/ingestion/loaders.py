from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain.schema import Document
from app.utils.logger import get_logger

log = get_logger(__name__)

LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".txt":  TextLoader,
    ".md":   UnstructuredMarkdownLoader,
    ".docx": Docx2txtLoader,
}


def load_file(path: str | Path) -> list[Document]:
    """Load a single file and return a list of LangChain Documents."""
    path = Path(path)
    suffix = path.suffix.lower()

    loader_cls = LOADER_MAP.get(suffix)
    if loader_cls is None:
        log.warning(f"Unsupported file type '{suffix}' for file: {path.name}")
        return []

    try:
        loader = loader_cls(str(path))
        docs = loader.load()
        log.info(f"Loaded {len(docs)} page(s) from '{path.name}'")
        return docs
    except Exception as e:
        log.error(f"Failed to load '{path.name}': {e}")
        return []


def load_directory(directory: str | Path, recursive: bool = False) -> list[Document]:
    """Load all supported files from a directory."""
    directory = Path(directory)
    pattern = "**/*" if recursive else "*"
    all_docs: list[Document] = []

    for file_path in directory.glob(pattern):
        if file_path.suffix.lower() in LOADER_MAP and file_path.is_file():
            all_docs.extend(load_file(file_path))

    log.info(f"Total documents loaded from '{directory}': {len(all_docs)}")
    return all_docs