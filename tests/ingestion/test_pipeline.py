from pathlib import Path

from app.ingestion.pipeline import ingest_document
from app.models.document import DocumentType


SAMPLE_DOCUMENT = Path("data/raw/sample.txt")


def test_ingest_text_document():
    result = ingest_document(SAMPLE_DOCUMENT)

    assert result.document.filename == "sample.txt"
    assert result.document.document_type == DocumentType.TXT
    assert result.document.content_hash
    assert result.chunk_count > 0


def test_chunks_have_document_id():
    result = ingest_document(SAMPLE_DOCUMENT)

    for chunk in result.chunks:
        assert chunk.document_id == result.document.id
        assert chunk.text