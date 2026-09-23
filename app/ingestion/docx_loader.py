from pathlib import Path

from docx import Document as DocxDocument

from app.ingestion.loader import BaseDocumentLoader


class DOCXLoader(BaseDocumentLoader):
    """Extract paragraphs from DOCX documents."""

    def load(self, path: Path) -> list[str]:
        document = DocxDocument(str(path))

        paragraphs = [
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return paragraphs