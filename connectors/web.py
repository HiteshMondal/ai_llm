"""
Web / URL connector.

Fetches one or more URLs, extracts clean article text using trafilatura,
and returns them as Documents.

Required package:  pip install trafilatura httpx

Usage:
    connector = WebConnector(urls=["https://example.com/article"])
    docs = connector.fetch()
"""

from typing import List

import httpx

from langchain.schema import Document

from app.connectors.base import BaseConnector
from app.logger import get_logger

log = get_logger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RAGBot/1.0; +https://github.com/your/repo)"
    )
}


class WebConnector(BaseConnector):
    def __init__(
        self,
        urls: List[str],
        timeout: int = 30,
        include_comments: bool = False,
        include_tables: bool = True,
    ):
        self.urls = urls
        self.timeout = timeout
        self.include_comments = include_comments
        self.include_tables = include_tables

    def _extract(self, html: str, url: str) -> tuple[str, str]:
        """
        Returns (title, text) extracted from raw HTML.
        Falls back to raw HTML stripping if trafilatura finds nothing.
        """
        try:
            import trafilatura
        except ImportError:
            raise ImportError("Install: pip install trafilatura")

        result = trafilatura.extract(
            html,
            url=url,
            include_comments=self.include_comments,
            include_tables=self.include_tables,
            output_format="txt",
            with_metadata=False,
        )
        meta = trafilatura.extract_metadata(html, default_url=url)
        title = (meta.title if meta and meta.title else url)
        return title, result or ""

    def _fetch_url(self, url: str) -> Document | None:
        try:
            resp = httpx.get(url, headers=HEADERS, timeout=self.timeout, follow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            log.error(f"Web: failed to fetch {url}: {e}")
            return None

        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            log.warning(f"Web: skipping {url} (content-type: {content_type})")
            return None

        if "text/plain" in content_type:
            text = resp.text
            title = url
        else:
            title, text = self._extract(resp.text, url)

        if not text.strip():
            log.warning(f"Web: no text extracted from {url}")
            return None

        log.info(f"Web: fetched '{title}' ({len(text)} chars) from {url}")
        return Document(
            page_content=text,
            metadata={
                "source":      title,
                "source_type": "web",
                "url":         url,
                "fetched_at":  self._now(),
            },
        )

    def fetch(self) -> List[Document]:
        docs = []
        for url in self.urls:
            doc = self._fetch_url(url.strip())
            if doc:
                docs.append(doc)
        log.info(f"Web: total {len(docs)}/{len(self.urls)} URL(s) fetched")
        return docs