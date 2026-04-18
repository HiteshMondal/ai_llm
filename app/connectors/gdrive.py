"""
Google Drive connector.
"""

import io
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document

from app.connectors.base import BaseConnector
from app.logger import get_logger

log = get_logger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

EXPORTABLE = {
    "application/vnd.google-apps.document":     "text/plain",
    "application/vnd.google-apps.spreadsheet":  "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}
DOWNLOADABLE_MIME = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/pdf",
}


class GDriveConnector(BaseConnector):
    def __init__(
        self,
        credentials_file: str = "credentials.json",
        token_file: str = "token.json",
        folder_id: Optional[str] = None,
    ):
        self.credentials_file = Path(credentials_file)
        self.token_file = Path(token_file)
        self.folder_id = folder_id

    def _get_service(self):
        """Authenticate and return a Drive API service object."""
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "Install Google Drive deps:\n"
                "pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client"
            )

        creds = None
        if self.token_file.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_file), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                self._retry(
                    lambda: creds.refresh(Request()),
                    label="GDrive.refresh_token",
                )
            else:
                if not self.credentials_file.exists():
                    raise FileNotFoundError(
                        f"credentials.json not found at {self.credentials_file}.\n"
                        "Download from Google Cloud Console → APIs & Services → Credentials."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_file), SCOPES
                )
                creds = flow.run_local_server(port=0)
            self.token_file.write_text(creds.to_json())

        return build("drive", "v3", credentials=creds)

    def _list_files(self, service) -> list:
        query = "trashed = false"
        if self.folder_id:
            query += f" and '{self.folder_id}' in parents"

        mime_conditions = " or ".join(
            [f"mimeType='{m}'" for m in EXPORTABLE]
            + [f"mimeType='{m}'" for m in DOWNLOADABLE_MIME]
        )
        query += f" and ({mime_conditions})"

        files, page_token = [], None
        while True:
            resp = self._retry(
                lambda pt=page_token: service.files().list(
                    q=query,
                    pageSize=100,
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                    pageToken=pt,
                ).execute(),
                label="GDrive.files.list",
            )
            files.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return files

    def _download(self, service, file_meta: dict) -> Optional[str]:
        from googleapiclient.http import MediaIoBaseDownload

        file_id = file_meta["id"]
        mime = file_meta["mimeType"]

        def _do_download():
            if mime in EXPORTABLE:
                export_mime = EXPORTABLE[mime]
                req = service.files().export_media(fileId=file_id, mimeType=export_mime)
            else:
                req = service.files().get_media(fileId=file_id)

            buf = io.BytesIO()
            dl = MediaIoBaseDownload(buf, req)
            done = False
            while not done:
                _, done = dl.next_chunk()
            return buf.getvalue().decode("utf-8", errors="ignore")

        try:
            return self._retry(
                _do_download,
                label=f"GDrive.download({file_meta['name']})",
            )
        except Exception as e:
            log.error(f"GDrive: failed to download '{file_meta['name']}': {e}")
            return None

    def fetch(self) -> List[Document]:
        log.info("GDrive: authenticating...")
        service = self._get_service()
        files = self._list_files(service)
        log.info(f"GDrive: found {len(files)} file(s)")

        docs = []
        for f in files:
            text = self._download(service, f)
            if not text or not text.strip():
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "source":      f["name"],
                    "source_type": "gdrive",
                    "drive_id":    f["id"],
                    "modified_at": f.get("modifiedTime", ""),
                    "fetched_at":  self._now(),
                },
            ))
            log.info(f"GDrive: fetched '{f['name']}' ({len(text)} chars)")

        return docs