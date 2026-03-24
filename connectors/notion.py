"""
Notion connector.

Setup:
  1. Go to https://www.notion.so/my-integrations
  2. Create a new integration → copy the Internal Integration Token
  3. Open each Notion page/database you want to ingest →
     click "..." menu → "Add connections" → select your integration
  4. Add to .env:
       NOTION_TOKEN=secret_xxxxxxxxxxxxxx
       NOTION_PAGE_IDS=page_id1,page_id2   (optional, comma-separated)
       NOTION_DATABASE_IDS=db_id1          (optional, comma-separated)

Required package:  pip install notion-client
"""

from typing import List, Optional

from langchain.schema import Document

from app.connectors.base import BaseConnector
from app.logger import get_logger

log = get_logger(__name__)


class NotionConnector(BaseConnector):
    def __init__(
        self,
        token: str,
        page_ids: Optional[List[str]] = None,
        database_ids: Optional[List[str]] = None,
    ):
        self.token = token
        self.page_ids = page_ids or []
        self.database_ids = database_ids or []

    def _get_client(self):
        try:
            from notion_client import Client
        except ImportError:
            raise ImportError("Install: pip install notion-client")
        return Client(auth=self.token)

    def _blocks_to_text(self, client, block_id: str, depth: int = 0) -> str:
        """Recursively convert Notion blocks to plain text."""
        lines = []
        try:
            children = client.blocks.children.list(block_id=block_id)
        except Exception as e:
            log.warning(f"Could not fetch children of block {block_id}: {e}")
            return ""

        for block in children.get("results", []):
            btype = block.get("type", "")
            data = block.get(btype, {})
            rich = data.get("rich_text", [])
            text = "".join(t.get("plain_text", "") for t in rich)

            prefix = {
                "heading_1": "# ",
                "heading_2": "## ",
                "heading_3": "### ",
                "bulleted_list_item": "- ",
                "numbered_list_item": "1. ",
                "to_do": "[ ] ",
                "toggle": "> ",
                "quote": "> ",
                "code": "```\n",
            }.get(btype, "")
            suffix = "\n```" if btype == "code" else ""

            if text:
                indent = "  " * depth
                lines.append(f"{indent}{prefix}{text}{suffix}")

            # Recurse into children
            if block.get("has_children"):
                lines.append(self._blocks_to_text(client, block["id"], depth + 1))

        return "\n".join(lines)

    def _fetch_page(self, client, page_id: str) -> Optional[Document]:
        try:
            page = client.pages.retrieve(page_id=page_id)
        except Exception as e:
            log.error(f"Notion: could not retrieve page {page_id}: {e}")
            return None

        # Extract title from page properties
        title = "Untitled"
        props = page.get("properties", {})
        for prop in props.values():
            if prop.get("type") == "title":
                rich = prop["title"]
                if rich:
                    title = rich[0].get("plain_text", "Untitled")
                break

        text = self._blocks_to_text(client, page_id)
        if not text.strip():
            return None

        return Document(
            page_content=text,
            metadata={
                "source":      title,
                "source_type": "notion",
                "notion_id":   page_id,
                "url":         page.get("url", ""),
                "fetched_at":  self._now(),
            },
        )

    def _fetch_database(self, client, db_id: str) -> List[Document]:
        docs = []
        try:
            results = client.databases.query(database_id=db_id)
        except Exception as e:
            log.error(f"Notion: could not query database {db_id}: {e}")
            return docs

        for page in results.get("results", []):
            doc = self._fetch_page(client, page["id"])
            if doc:
                docs.append(doc)
        return docs

    def _search_all(self, client) -> List[Document]:
        """If no specific IDs given, search all pages the integration can access."""
        docs = []
        try:
            results = client.search(filter={"property": "object", "value": "page"})
        except Exception as e:
            log.error(f"Notion: search failed: {e}")
            return docs

        for page in results.get("results", []):
            doc = self._fetch_page(client, page["id"])
            if doc:
                docs.append(doc)
        return docs

    def fetch(self) -> List[Document]:
        log.info("Notion: connecting...")
        client = self._get_client()
        docs = []

        if not self.page_ids and not self.database_ids:
            log.info("Notion: no IDs specified — searching all accessible pages")
            docs = self._search_all(client)
        else:
            for pid in self.page_ids:
                doc = self._fetch_page(client, pid)
                if doc:
                    docs.append(doc)
                    log.info(f"Notion: fetched page '{doc.metadata['source']}'")

            for did in self.database_ids:
                db_docs = self._fetch_database(client, did)
                docs.extend(db_docs)
                log.info(f"Notion: fetched {len(db_docs)} pages from database {did}")

        log.info(f"Notion: total {len(docs)} document(s) fetched")
        return docs