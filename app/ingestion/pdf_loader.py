from pathlib import Path
from typing import cast

import pymupdf as pymu

from app.ingestion.loader import BaseDocumentLoader


class PDFLoader(BaseDocumentLoader):
    """Extract text from PDF documents while preserving page boundaries."""

    def load(self, path: Path) -> list[str]:
        pages: list[str] = []

        with pymu.open(path) as document:
            for page in document:
                text = cast(str, page.get_text("text")).strip()
                pages.append(text)

        return pages