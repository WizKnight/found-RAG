from pathlib import Path

from app.ingestion.pipeline import ingest_document
from app.retrieval.dense import dense_search
from app.services.indexing_service import index_chunks


SAMPLE_DOCUMENT = Path("data/raw/sample.txt")


def test_dense_retrieval():
    result = ingest_document(SAMPLE_DOCUMENT)

    index_chunks(result.chunks)

    results = dense_search(
        "How many annual leave days do employees receive?",
        top_k=3,
    )

    assert results
    assert results[0]["payload"]["text"]