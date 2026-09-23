from pathlib import Path

from app.core.config import get_settings
from app.ingestion.cleaner import normalize_text
from app.ingestion.chunker import TextChunker
from app.ingestion.hash import calculate_file_hash
from app.ingestion.loader_factory import get_loader
from app.models.document import Document, DocumentType
from app.models.ingestion import IngestionResult


def detect_document_type(path: Path) -> DocumentType:
    suffix = path.suffix.lower()

    mapping = {
        ".pdf": DocumentType.PDF,
        ".docx": DocumentType.DOCX,
        ".txt": DocumentType.TXT,
        ".md": DocumentType.MARKDOWN,
        ".html": DocumentType.HTML,
    }

    try:
        return mapping[suffix]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported document type: {suffix}"
        ) from exc


def ingest_document(path: Path) -> IngestionResult:
    if not path.exists():
        raise FileNotFoundError(path)

    settings = get_settings()

    document = Document(
        filename=path.name,
        document_type=detect_document_type(path),
        source=str(path),
        content_hash=calculate_file_hash(path),
    )

    loader = get_loader(path)
    sections = loader.load(path)

    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    chunks = []

    for section_index, raw_text in enumerate(sections):
        cleaned_text = normalize_text(raw_text)

        if not cleaned_text:
            continue

        page_number = (
            section_index + 1
            if document.document_type == DocumentType.PDF
            else None
        )

        section_chunks = chunker.chunk(
            text=cleaned_text,
            document_id=document.id,
            page_number=page_number,
        )

        chunks.extend(section_chunks)

    return IngestionResult(
        document=document,
        chunks=chunks,
    )