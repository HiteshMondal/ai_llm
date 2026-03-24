"""
GitHub connector.

Fetches readable files (.md, .txt, .rst, .py, .js, .ts, etc.)
from one or more GitHub repositories.

Required package:  pip install PyGithub
Required env var:
  GITHUB_TOKEN   personal access token (PAT)
                 Scopes needed: repo (private) or public_repo (public)
                 Create at: https://github.com/settings/tokens
"""

from typing import List, Optional

from langchain.schema import Document

from app.connectors.base import BaseConnector
from app.logger import get_logger

log = get_logger(__name__)

TEXT_EXTENSIONS = {
    ".md", ".txt", ".rst", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".html", ".htm", ".css", ".yaml", ".yml", ".toml", ".json",
    ".sh", ".bash", ".env.example", ".cfg", ".ini",
}


class GithubConnector(BaseConnector):
    def __init__(
        self,
        token: str,
        repos: List[str],
        branch: Optional[str] = None,
        max_file_size_kb: int = 500,
        file_extensions: Optional[set] = None,
    ):
        self.token = token
        self.repos = repos
        self.branch = branch
        self.max_file_size_bytes = max_file_size_kb * 1024
        self.extensions = file_extensions or TEXT_EXTENSIONS

    def _get_client(self):
        try:
            from github import Github
        except ImportError:
            raise ImportError("Install: pip install PyGithub")
        return Github(self.token)

    def _fetch_repo(self, client, repo_name: str) -> List[Document]:
        docs = []

        try:
            repo = self._retry(
                lambda: client.get_repo(repo_name),
                label=f"GitHub.get_repo({repo_name})",
            )
        except Exception as e:
            log.error(f"GitHub: could not access repo '{repo_name}': {e}")
            return docs

        branch = self.branch or repo.default_branch
        log.info(f"GitHub: scanning '{repo_name}' @ branch '{branch}'")

        try:
            contents = self._retry(
                lambda: repo.get_git_tree(branch, recursive=True).tree,
                label=f"GitHub.get_git_tree({repo_name})",
            )
        except Exception as e:
            log.error(f"GitHub: could not get tree for '{repo_name}': {e}")
            return docs

        for item in contents:
            if item.type != "blob":
                continue

            path = item.path
            ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
            if ext not in self.extensions:
                continue
            if item.size > self.max_file_size_bytes:
                log.warning(f"GitHub: skipping large file {path} ({item.size // 1024}KB)")
                continue

            try:
                file_content = self._retry(
                    lambda p=path: repo.get_contents(p, ref=branch),
                    label=f"GitHub.get_contents({path})",
                )
                text = file_content.decoded_content.decode("utf-8", errors="ignore")
            except Exception as e:
                log.error(f"GitHub: failed to read '{path}': {e}")
                continue

            if not text.strip():
                continue

            docs.append(Document(
                page_content=text,
                metadata={
                    "source":      f"{repo_name}/{path}",
                    "source_type": "github",
                    "repo":        repo_name,
                    "path":        path,
                    "branch":      branch,
                    "url":         f"https://github.com/{repo_name}/blob/{branch}/{path}",
                    "fetched_at":  self._now(),
                },
            ))
            log.info(f"GitHub: fetched '{repo_name}/{path}' ({len(text)} chars)")

        return docs

    def fetch(self) -> List[Document]:
        client = self._get_client()
        docs = []
        for repo_name in self.repos:
            repo_docs = self._fetch_repo(client, repo_name)
            docs.extend(repo_docs)
        log.info(
            f"GitHub: total {len(docs)} file(s) fetched "
            f"from {len(self.repos)} repo(s)"
        )
        return docs