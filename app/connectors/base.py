import time
from abc import ABC, abstractmethod
from typing import List

from langchain.schema import Document

from app.config import get_settings
from app.logger import get_logger

log = get_logger(__name__)


class BaseConnector(ABC):
    """
    Base class for all source connectors.

    Provides:
    - retry_request(): decorator/helper for HTTP calls with exponential back-off
    - _now(): UTC ISO-8601 timestamp helper
    - fetch(): abstract — each connector must implement this
    """

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

    @staticmethod
    def _retry(fn, *args, label: str = "", **kwargs):
        """
        Call fn(*args, **kwargs) with exponential back-off retry.
        Settings are pulled from config (connector_max_retries, connector_retry_delay).
        """
        settings = get_settings()
        max_retries = settings.connector_max_retries
        base_delay = settings.connector_retry_delay

        last_exc: Exception = RuntimeError("Unknown error")
        for attempt in range(1, max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1))
                    log.warning(
                        f"{label or fn.__name__}: attempt {attempt}/{max_retries} failed "
                        f"({exc}). Retrying in {delay:.1f}s…"
                    )
                    time.sleep(delay)
                else:
                    log.error(
                        f"{label or fn.__name__}: all {max_retries} attempts failed. "
                        f"Last error: {exc}"
                    )
        raise last_exc