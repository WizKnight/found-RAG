from pathlib import Path

from app.ingestion.docx_loader import DOCXLoader
from app.ingestion.loader import BaseDocumentLoader
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.text_loader import TextLoader


class UnsupportedDocumentTypeError(ValueError):
    """Raised when a document type is not supported."""


def get_loader(path: Path) -> BaseDocumentLoader:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return PDFLoader()

    if suffix == ".docx":
        return DOCXLoader()

    if suffix in {".txt", ".md"}:
        return TextLoader()

    raise UnsupportedDocumentTypeError(
        f"Unsupported document type: {suffix}"
    )