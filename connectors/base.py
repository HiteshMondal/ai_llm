from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document


class BaseConnector(ABC):
    """All source connectors must implement this interface."""

    @abstractmethod
    def fetch(self) -> List[Document]:
        """
        Pull content from the source and return a list of Documents.
        Each Document must have metadata containing at least:
          - source      : human-readable name (filename, page title, url)
          - source_type : 'gdrive' | 'notion' | 'web' | 'github' | 'local'
          - fetched_at  : ISO-8601 UTC timestamp
        """
        ...

    @staticmethod
    def _now() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()