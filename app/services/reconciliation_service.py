
from collections.abc import Callable
from uuid import UUID

from pydantic import BaseModel

from app.models.chunk import Chunk
from app.models.indexing_operation import IndexingOperationStatus
from app.services.canonical_chunk_service import get_canonical_chunk_store
from app.services.index_consistency_service import inspect_document_indexes
from app.services.indexing_journal_service import get_indexing_journal
from app.services.indexing_service import reconcile_indexing_operation


class ReconciliationOutcome(BaseModel):
    operation_id: UUID
    document_id: UUID
    recovered: bool
    message: str


CanonicalChunkLoader = Callable[[UUID, set[UUID]], list[Chunk]]

def load_canonical_chunks(
    document_id: UUID,
    expected_chunk_ids: set[UUID],
) -> list[Chunk]:
    return get_canonical_chunk_store().get_chunks_by_ids(
        document_id=document_id,
        chunk_ids=expected_chunk_ids,
    )


def reconcile_incomplete_operations(
    load_canonical_chunks: CanonicalChunkLoader,
) -> list[ReconciliationOutcome]:
    """Retry incomplete operations using authoritative chunk records."""
    journal = get_indexing_journal()
    outcomes: list[ReconciliationOutcome] = []

    for operation in journal.list_incomplete():
        try:
            expected_ids = set(operation.chunk_ids)
            chunks = load_canonical_chunks(
                operation.document_id,
                expected_ids,
            )

            if {chunk.id for chunk in chunks} != expected_ids:
                raise ValueError(
                    "Canonical chunk loader returned IDs that do not "
                    "match the journal operation."
                )

            reconcile_indexing_operation(
                operation_id=operation.operation_id,
                canonical_chunks=chunks,
            )

            report = inspect_document_indexes(
                document_id=operation.document_id,
                expected_chunk_ids=expected_ids,
            )

            if not report.is_consistent:
                raise RuntimeError(
                    "Reconciliation finished but index consistency "
                    "verification failed."
                )

            current = journal.get(operation.operation_id)
            if current.status != IndexingOperationStatus.COMPLETED:
                raise RuntimeError(
                    "Indexes match, but the journal operation is not "
                    "marked completed."
                )

            outcomes.append(
                ReconciliationOutcome(
                    operation_id=operation.operation_id,
                    document_id=operation.document_id,
                    recovered=True,
                    message="Operation recovered and indexes verified.",
                )
            )

        except Exception as exc: # noqa: BLE001
            # Continue processing other operations. The indexing service
            # normally records failures; loader and consistency failures
            # are also returned here for observability.
            outcomes.append(
                ReconciliationOutcome(
                    operation_id=operation.operation_id,
                    document_id=operation.document_id,
                    recovered=False,
                    message=f"{type(exc).__name__}: {exc}"[:2000],
                )
            )

    return outcomes