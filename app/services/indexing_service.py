from uuid import UUID

import numpy as np

from app.models.chunk import Chunk
from app.models.indexing_operation import (
    IndexingOperation,
    IndexingOperationStatus,
)
from app.services.bm25_service import get_bm25_index_store, get_bm25_retriever
from app.services.embedding_service import get_embedding_provider
from app.services.indexing_journal_service import get_indexing_journal
from app.services.vector_store_service import get_vector_store


def index_chunks(chunks: list[Chunk]) -> UUID | None:
    """Index the complete replacement chunk set for one document."""
    if not chunks:
        return None

    document_ids = {chunk.document_id for chunk in chunks}
    if len(document_ids) != 1:
        raise ValueError(
            "index_chunks expects chunks from exactly one document."
        )

    chunk_ids = [chunk.id for chunk in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Chunk IDs must be unique within a document.")

    operation = IndexingOperation(
        document_id=next(iter(document_ids)),
        chunk_ids=chunk_ids,
    )

    get_indexing_journal().create(operation)
    _execute_indexing_operation(chunks, operation)
    return operation.operation_id


def reconcile_indexing_operation(
    operation_id: UUID,
    canonical_chunks: list[Chunk],
) -> None:
    """Retry an incomplete operation from authoritative chunk records."""
    journal = get_indexing_journal()
    operation = journal.get(operation_id)

    if operation.status == IndexingOperationStatus.COMPLETED:
        return

    if any(
        chunk.document_id != operation.document_id
        for chunk in canonical_chunks
    ):
        raise ValueError(
            "All recovery chunks must belong to the operation's document."
        )

    actual_chunk_ids = {chunk.id for chunk in canonical_chunks}
    expected_chunk_ids = set(operation.chunk_ids)

    if actual_chunk_ids != expected_chunk_ids:
        raise ValueError(
            "Recovery chunk IDs do not match the journal. "
            "Load the canonical chunk set for this operation before retrying."
        )

    journal.update(
        operation_id,
        status=IndexingOperationStatus.PENDING,
        error=None,
        increment_attempts=True,
    )

    _execute_indexing_operation(
        canonical_chunks,
        operation.model_copy(
            update={"status": IndexingOperationStatus.PENDING}
        ),
    )


def _execute_indexing_operation(
    chunks: list[Chunk],
    operation: IndexingOperation,
) -> None:
    journal = get_indexing_journal()

    try:
        embedding_provider = get_embedding_provider()
        vector_store = get_vector_store()
        bm25_store = get_bm25_index_store()

        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_provider.embed_documents(texts)

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                "Embedding count does not match the number of chunks."
            )

        vectors = np.asarray(embeddings, dtype=np.float64)
        expected_shape = (
            len(chunks),
            embedding_provider.dimension,
        )

        if vectors.shape != expected_shape:
            raise RuntimeError(
                f"Expected embedding shape {expected_shape}, "
                f"got {vectors.shape}."
            )

        if not np.isfinite(vectors).all():
            raise RuntimeError("Embeddings contain non-finite values.")

        vector_store.create_collection(
            vector_size=embedding_provider.dimension,
        )

        old_chunks = [
            chunk
            for chunk in bm25_store.load_chunks()
            if chunk.document_id == operation.document_id
        ]

        new_chunk_ids = {chunk.id for chunk in chunks}
        stale_chunk_ids = [
            chunk.id
            for chunk in old_chunks
            if chunk.id not in new_chunk_ids
        ]

        payloads = [
            {
                "document_id": str(chunk.document_id),
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "section": chunk.section,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]

        # Upsert first so existing vectors remain available if this fails.
        vector_store.upsert(
            ids=[chunk.id for chunk in chunks],
            vectors=vectors.tolist(),
            payloads=payloads,
        )

        vector_store.delete_points(stale_chunk_ids)

        journal.update(
            operation.operation_id,
            status=IndexingOperationStatus.QDRANT_UPDATED,
            error=None,
        )

        # Replace the full document corpus and rebuild the in-memory index.
        get_bm25_retriever().replace_document(
            document_id=operation.document_id,
            chunks=chunks,
        )

        journal.update(
            operation.operation_id,
            status=IndexingOperationStatus.BM25_UPDATED,
            error=None,
        )

        # Both index writes have returned successfully.
        journal.update(
            operation.operation_id,
            status=IndexingOperationStatus.COMPLETED,
            error=None,
        )

    except Exception as exc:
        try:
            journal.update(
                operation.operation_id,
                status=IndexingOperationStatus.NEEDS_RECONCILIATION,
                error=f"{type(exc).__name__}: {exc}"[:2000],
            )
        except Exception:
            # Preserve the original indexing exception. A journal write
            # failure should be surfaced by monitoring and process logs.
            pass

        raise
