from uuid import UUID

from app.models.index_consistency import IndexConsistencyReport
from app.services.bm25_service import get_bm25_index_store
from app.services.vector_store_service import get_vector_store


def inspect_document_indexes(
    document_id: UUID,
    expected_chunk_ids: set[UUID],
) -> IndexConsistencyReport:
    """Compare expected document chunks against BM25 and Qdrant."""

    bm25_store = get_bm25_index_store()
    vector_store = get_vector_store()

    bm25_chunk_ids = {
        chunk.id
        for chunk in bm25_store.load_chunks()
        if chunk.document_id == document_id
    }

    qdrant_chunk_ids = vector_store.get_document_chunk_ids(
        document_id=document_id,
    )

    return IndexConsistencyReport(
        document_id=document_id,
        expected_chunk_ids=expected_chunk_ids,
        bm25_chunk_ids=bm25_chunk_ids,
        qdrant_chunk_ids=qdrant_chunk_ids,
        bm25_missing_ids=expected_chunk_ids - bm25_chunk_ids,
        bm25_extra_ids=bm25_chunk_ids - expected_chunk_ids,
        qdrant_missing_ids=expected_chunk_ids - qdrant_chunk_ids,
        qdrant_extra_ids=qdrant_chunk_ids - expected_chunk_ids,
    )