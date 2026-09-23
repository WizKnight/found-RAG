from pathlib import Path

from app.ingestion.loader import BaseDocumentLoader


class TextLoader(BaseDocumentLoader):
    """Load plain text and Markdown documents."""

    def load(self, path: Path) -> list[str]:
        text = path.read_text(encoding="utf-8")

        return [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]