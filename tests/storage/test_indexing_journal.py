from uuid import uuid4

from app.models.indexing_operation import (
    IndexingOperation,
    IndexingOperationStatus,
)
from app.storage.indexing_journal import IndexingJournal


def test_journal_persists_operation_across_instances(tmp_path) -> None:
    path = tmp_path / "indexing_journal.json"
    first_journal = IndexingJournal(path)

    operation = IndexingOperation(
        document_id=uuid4(),
        chunk_ids=[uuid4(), uuid4()],
    )
    first_journal.create(operation)

    restarted_journal = IndexingJournal(path)
    restored = restarted_journal.get(operation.operation_id)

    assert restored.operation_id == operation.operation_id
    assert restored.document_id == operation.document_id
    assert restored.chunk_ids == operation.chunk_ids
    assert restored.status == IndexingOperationStatus.PENDING


def test_journal_records_stage_progress(tmp_path) -> None:
    journal = IndexingJournal(tmp_path / "indexing_journal.json")
    operation = IndexingOperation(
        document_id=uuid4(),
        chunk_ids=[uuid4()],
    )
    journal.create(operation)

    updated = journal.update(
        operation.operation_id,
        status=IndexingOperationStatus.QDRANT_UPDATED,
    )

    assert updated.status == IndexingOperationStatus.QDRANT_UPDATED
    assert (
        journal.get(operation.operation_id).status
        == IndexingOperationStatus.QDRANT_UPDATED
    )


def test_journal_records_failure_and_lists_incomplete(
    tmp_path,
) -> None:
    journal = IndexingJournal(tmp_path / "indexing_journal.json")
    operation = IndexingOperation(
        document_id=uuid4(),
        chunk_ids=[uuid4()],
    )
    journal.create(operation)

    failed = journal.update(
        operation.operation_id,
        status=IndexingOperationStatus.NEEDS_RECONCILIATION,
        error="Simulated Qdrant failure",
    )

    assert failed.error == "Simulated Qdrant failure"
    assert operation.operation_id in {
        item.operation_id
        for item in journal.list_incomplete()
    }


def test_completed_operations_are_not_in_incomplete_list(
    tmp_path,
) -> None:
    journal = IndexingJournal(tmp_path / "indexing_journal.json")
    operation = IndexingOperation(
        document_id=uuid4(),
        chunk_ids=[uuid4()],
    )
    journal.create(operation)

    journal.update(
        operation.operation_id,
        status=IndexingOperationStatus.COMPLETED,
    )

    assert journal.list_incomplete() == []


def test_retry_increments_attempt_count(tmp_path) -> None:
    journal = IndexingJournal(tmp_path / "indexing_journal.json")
    operation = IndexingOperation(
        document_id=uuid4(),
        chunk_ids=[uuid4()],
    )
    journal.create(operation)

    retried = journal.update(
        operation.operation_id,
        status=IndexingOperationStatus.PENDING,
        error=None,
        increment_attempts=True,
    )

    assert retried.attempts == 2
    assert retried.error is None


def test_missing_operation_raises_key_error(tmp_path) -> None:
    journal = IndexingJournal(tmp_path / "indexing_journal.json")

    try:
        journal.get(uuid4())
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError for missing operation")


def test_superseded_operations_are_not_in_incomplete_list(
    tmp_path,
) -> None:
    journal = IndexingJournal(tmp_path / "indexing_journal.json")
    operation = IndexingOperation(
        document_id=uuid4(),
        chunk_ids=[uuid4()],
    )
    journal.create(operation)

    journal.update(
        operation.operation_id,
        status=IndexingOperationStatus.SUPERSEDED,
        error="Replaced by a newer canonical document version.",
    )

    assert journal.list_incomplete() == []