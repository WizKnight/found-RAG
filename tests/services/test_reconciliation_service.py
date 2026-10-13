from uuid import uuid4

from app.models.chunk import Chunk
from app.models.index_consistency import IndexConsistencyReport
from app.models.indexing_operation import (
    IndexingOperation,
    IndexingOperationStatus,
)
from app.services import reconciliation_service


def make_chunk(document_id, text="Evidence", chunk_index=0):
    return Chunk(
        document_id=document_id,
        text=text,
        chunk_index=chunk_index,
    )


class FakeJournal:
    def __init__(self, operations):
        self.operations = {
            operation.operation_id: operation
            for operation in operations
        }

    def list_incomplete(self):
        return [
            operation
            for operation in self.operations.values()
            if operation.status != IndexingOperationStatus.COMPLETED
        ]

    def get(self, operation_id):
        return self.operations[operation_id]

    def update(
        self,
        operation_id,
        status,
        error=None,
        increment_attempts=False,
    ):
        operation = self.operations[operation_id]
        changes = {
            "status": status,
            "error": error,
        }

        if increment_attempts:
            changes["attempts"] = operation.attempts + 1

        self.operations[operation_id] = operation.model_copy(
            update=changes
        )


def make_operation(document_id, chunks):
    return IndexingOperation(
        document_id=document_id,
        chunk_ids=[chunk.id for chunk in chunks],
        status=IndexingOperationStatus.NEEDS_RECONCILIATION,
    )


def consistent_report(document_id, chunk_ids):
    return IndexConsistencyReport(
        document_id=document_id,
        expected_chunk_ids=chunk_ids,
        bm25_chunk_ids=chunk_ids,
        qdrant_chunk_ids=chunk_ids,
    )


def configure_reconciliation(monkeypatch, operation):
    journal = FakeJournal([operation])
    calls = []

    monkeypatch.setattr(
        reconciliation_service,
        "get_indexing_journal",
        lambda: journal,
    )

    def inspect_indexes(document_id, expected_chunk_ids):
        calls.append("inspect")
        return consistent_report(document_id, expected_chunk_ids)

    monkeypatch.setattr(
        reconciliation_service,
        "inspect_document_indexes",
        inspect_indexes,
    )

    return journal, calls


def test_successful_reconciliation_marks_operation_completed(
    monkeypatch,
):
    document_id = uuid4()
    chunks = [
        make_chunk(document_id, "First", 0),
        make_chunk(document_id, "Second", 1),
    ]
    operation = make_operation(document_id, chunks)
    journal, calls = configure_reconciliation(monkeypatch, operation)

    def load_chunks(requested_document_id, expected_ids):
        calls.append("load_canonical")
        assert requested_document_id == document_id
        assert expected_ids == {chunk.id for chunk in chunks}
        return chunks

    def reconcile(operation_id, canonical_chunks):
        calls.append("reconcile")
        assert operation_id == operation.operation_id
        assert canonical_chunks == chunks

        journal.update(
            operation_id,
            status=IndexingOperationStatus.COMPLETED,
            error=None,
        )

    monkeypatch.setattr(
        reconciliation_service,
        "reconcile_indexing_operation",
        reconcile,
    )

    outcomes = reconciliation_service.reconcile_incomplete_operations(
        load_chunks
    )

    assert len(outcomes) == 1
    assert outcomes[0].recovered is True
    assert journal.get(operation.operation_id).status == (
        IndexingOperationStatus.COMPLETED
    )
    assert calls == [
        "load_canonical",
        "reconcile",
        "inspect",
    ]


def test_missing_canonical_chunk_prevents_reconciliation(
    monkeypatch,
):
    document_id = uuid4()
    chunks = [make_chunk(document_id)]
    operation = make_operation(document_id, chunks)
    journal, calls = configure_reconciliation(monkeypatch, operation)

    def load_incomplete_chunks(requested_document_id, expected_ids):
        return []

    def should_not_reconcile(operation_id, canonical_chunks):
        raise AssertionError("Reconciliation must not run.")

    monkeypatch.setattr(
        reconciliation_service,
        "reconcile_indexing_operation",
        should_not_reconcile,
    )

    outcomes = reconciliation_service.reconcile_incomplete_operations(
        load_incomplete_chunks
    )

    assert len(outcomes) == 1
    assert outcomes[0].recovered is False
    assert journal.get(operation.operation_id).status == (
        IndexingOperationStatus.NEEDS_RECONCILIATION
    )
    assert "inspect" not in calls


def test_reconciliation_rejects_chunks_from_another_document(
    monkeypatch,
):
    document_id = uuid4()
    chunks = [make_chunk(document_id)]
    operation = make_operation(document_id, chunks)
    journal, _ = configure_reconciliation(monkeypatch, operation)

    wrong_document_chunks = [
        make_chunk(uuid4(), "Wrong document")
    ]

    def load_wrong_document(requested_document_id, expected_ids):
        return wrong_document_chunks

    outcomes = reconciliation_service.reconcile_incomplete_operations(
        load_wrong_document
    )

    assert len(outcomes) == 1
    assert outcomes[0].recovered is False
    assert journal.get(operation.operation_id).status == (
        IndexingOperationStatus.NEEDS_RECONCILIATION
    )


def test_failed_reconciliation_returns_failure_outcome(
    monkeypatch,
):
    document_id = uuid4()
    chunks = [make_chunk(document_id)]
    operation = make_operation(document_id, chunks)
    journal, _ = configure_reconciliation(monkeypatch, operation)

    def load_chunks(requested_document_id, expected_ids):
        return chunks

    def fail_reconciliation(operation_id, canonical_chunks):
        raise RuntimeError("simulated recovery failure")

    monkeypatch.setattr(
        reconciliation_service,
        "reconcile_indexing_operation",
        fail_reconciliation,
    )

    outcomes = reconciliation_service.reconcile_incomplete_operations(
        load_chunks
    )

    assert len(outcomes) == 1
    assert outcomes[0].recovered is False
    assert "simulated recovery failure" in outcomes[0].message
    assert journal.get(operation.operation_id).status == (
        IndexingOperationStatus.NEEDS_RECONCILIATION
    )


def test_completed_operations_are_not_replayed(monkeypatch):
    document_id = uuid4()
    chunks = [make_chunk(document_id)]

    operation = IndexingOperation(
        document_id=document_id,
        chunk_ids=[chunk.id for chunk in chunks],
        status=IndexingOperationStatus.COMPLETED,
    )
    journal, _ = configure_reconciliation(monkeypatch, operation)

    def should_not_load(requested_document_id, expected_ids):
        raise AssertionError("Completed operation must not be replayed.")

    outcomes = reconciliation_service.reconcile_incomplete_operations(
        should_not_load
    )

    assert outcomes == []
    assert journal.get(operation.operation_id).status == (
        IndexingOperationStatus.COMPLETED
    )