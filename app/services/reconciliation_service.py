import logging
from collections.abc import Callable
from uuid import UUID

from pydantic import BaseModel

from app.models.chunk import Chunk
from app.models.indexing_operation import IndexingOperationStatus
from app.services.canonical_chunk_service import get_canonical_chunk_store
from app.services.index_consistency_service import inspect_document_indexes
from app.services.indexing_journal_service import get_indexing_journal
from app.services.indexing_service import reconcile_indexing_operation

logger = logging.getLogger(__name__)

CanonicalChunkLoader = Callable[[UUID, set[UUID]], list[Chunk]]


class ReconciliationOutcome(BaseModel):
    operation_id: UUID
    document_id: UUID
    recovered: bool
    message: str
    superseded: bool = False


def load_canonical_chunks(
    document_id: UUID,
    expected_chunk_ids: set[UUID],
) -> list[Chunk]:
    """Load the exact authoritative chunks required by an operation."""
    return get_canonical_chunk_store().get_chunks_by_ids(
        document_id=document_id,
        chunk_ids=expected_chunk_ids,
    )


def reconcile_incomplete_operations(
    load_canonical_chunks: CanonicalChunkLoader,
) -> list[ReconciliationOutcome]:
    """Recover incomplete operations using canonical document chunks."""
    journal = get_indexing_journal()
    canonical_store = get_canonical_chunk_store()
    outcomes: list[ReconciliationOutcome] = []

    for operation in journal.list_incomplete():
        try:
            expected_ids = set(operation.chunk_ids)

            if _is_superseded(
                document_id=operation.document_id,
                expected_chunk_ids=expected_ids,
                canonical_store=canonical_store,
            ):
                outcomes.append(
                    _mark_superseded(
                        operation_id=operation.operation_id,
                        document_id=operation.document_id,
                        journal=journal,
                    )
                )
                continue

            chunks = _load_and_validate_chunks(
                loader=load_canonical_chunks,
                document_id=operation.document_id,
                expected_chunk_ids=expected_ids,
            )

            reconcile_indexing_operation(
                operation_id=operation.operation_id,
                canonical_chunks=chunks,
            )

            _verify_reconciled_operation(
                operation_id=operation.operation_id,
                document_id=operation.document_id,
                expected_chunk_ids=expected_ids,
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

        except Exception as exc:
            logger.exception(
                "Failed to reconcile operation %s for document %s",
                operation.operation_id,
                operation.document_id,
            )

            outcomes.append(
                ReconciliationOutcome(
                    operation_id=operation.operation_id,
                    document_id=operation.document_id,
                    recovered=False,
                    message=f"{type(exc).__name__}: {exc}"[:2000],
                )
            )

    return outcomes


def _is_superseded(
    document_id: UUID,
    expected_chunk_ids: set[UUID],
    canonical_store,
) -> bool:
    """Return whether an operation targets an obsolete chunk set."""
    current_chunks = canonical_store.get_document_chunks(document_id)
    current_chunk_ids = {chunk.id for chunk in current_chunks}

    return current_chunk_ids != expected_chunk_ids


def _mark_superseded(
    operation_id: UUID,
    document_id: UUID,
    journal,
) -> ReconciliationOutcome:
    """Mark an obsolete operation terminal without replaying it."""
    message = (
        "Operation superseded: its chunk IDs do not match "
        "the current canonical document version."
    )

    journal.update(
        operation_id,
        status=IndexingOperationStatus.SUPERSEDED,
        error=message,
    )

    logger.info(
        "Marked operation %s for document %s as superseded",
        operation_id,
        document_id,
    )

    return ReconciliationOutcome(
        operation_id=operation_id,
        document_id=document_id,
        recovered=False,
        superseded=True,
        message=message,
    )


def _load_and_validate_chunks(
    loader: CanonicalChunkLoader,
    document_id: UUID,
    expected_chunk_ids: set[UUID],
) -> list[Chunk]:
    """Load canonical chunks and reject incomplete or unexpected results."""
    chunks = loader(document_id, expected_chunk_ids)

    if any(chunk.document_id != document_id for chunk in chunks):
        raise ValueError(
            "Canonical chunk loader returned chunks belonging "
            "to another document."
        )

    actual_ids = {chunk.id for chunk in chunks}

    if len(chunks) != len(expected_chunk_ids) or actual_ids != expected_chunk_ids:
        raise ValueError(
            "Canonical chunk loader returned IDs that do not "
            "match the journal operation."
        )

    return chunks


def _verify_reconciled_operation(
    operation_id: UUID,
    document_id: UUID,
    expected_chunk_ids: set[UUID],
) -> None:
    """Require consistent indexes before accepting recovery."""
    report = inspect_document_indexes(
        document_id=document_id,
        expected_chunk_ids=expected_chunk_ids,
    )

    if not report.is_consistent:
        raise RuntimeError(
            "Reconciliation finished but index consistency "
            "verification failed."
        )