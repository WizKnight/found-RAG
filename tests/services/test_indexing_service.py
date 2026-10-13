from uuid import uuid4

import pytest

from app.models.chunk import Chunk
from app.models.index_consistency import IndexConsistencyReport
from app.models.indexing_operation import (
    IndexingOperationStatus,
)
from app.services import indexing_service


def make_chunk(document_id, text="Evidence", chunk_index=0):
    return Chunk(
        document_id=document_id,
        text=text,
        chunk_index=chunk_index,
    )


class FakeEmbeddingProvider:
    dimension = 2

    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeCanonicalStore:
    def __init__(self, calls):
        self.calls = calls
        self.documents = {}

    def replace_document(self, document_id, chunks):
        self.calls.append("canonical")
        self.documents[document_id] = list(chunks)


class FakeVectorStore:
    def __init__(self, calls, fail_on=None):
        self.calls = calls
        self.fail_on = fail_on

    def create_collection(self, vector_size):
        self.calls.append("create_collection")

    def upsert(self, ids, vectors, payloads):
        self.calls.append("qdrant_upsert")
        if self.fail_on == "upsert":
            raise RuntimeError("simulated Qdrant upsert failure")

    def delete_points(self, ids):
        self.calls.append("qdrant_delete")
        if self.fail_on == "delete":
            raise RuntimeError("simulated Qdrant delete failure")


class FakeBM25Store:
    def __init__(self, calls, existing_chunks=None):
        self.calls = calls
        self.existing_chunks = existing_chunks or []

    def load_chunks(self):
        return list(self.existing_chunks)


class FakeBM25Retriever:
    def __init__(self, calls, fail=False):
        self.calls = calls
        self.fail = fail

    def replace_document(self, document_id, chunks):
        self.calls.append("bm25_replace")
        if self.fail:
            raise RuntimeError("simulated BM25 failure")


class FakeJournal:
    def __init__(self, calls):
        self.calls = calls
        self.operations = {}
        self.updates = []

    def create(self, operation):
        self.calls.append("journal_create")
        self.operations[operation.operation_id] = operation

    def update(
        self,
        operation_id,
        status,
        error=None,
        increment_attempts=False,
    ):
        self.calls.append(f"journal_{status.value}")
        self.updates.append(
            {
                "operation_id": operation_id,
                "status": status,
                "error": error,
                "increment_attempts": increment_attempts,
            }
        )

        current = self.operations[operation_id]
        changes = {
            "status": status,
            "error": error,
        }
        if increment_attempts:
            changes["attempts"] = current.attempts + 1

        self.operations[operation_id] = current.model_copy(
            update=changes
        )

    def get(self, operation_id):
        return self.operations[operation_id]


def configure_dependencies(
    monkeypatch,
    *,
    qdrant_failure=None,
    bm25_failure=False,
    existing_chunks=None,
):
    calls = []
    journal = FakeJournal(calls)
    canonical_store = FakeCanonicalStore(calls)
    vector_store = FakeVectorStore(calls, fail_on=qdrant_failure)
    bm25_store = FakeBM25Store(calls, existing_chunks)
    bm25_retriever = FakeBM25Retriever(calls, fail=bm25_failure)

    monkeypatch.setattr(
        indexing_service,
        "get_embedding_provider",
        lambda: FakeEmbeddingProvider(),
    )
    monkeypatch.setattr(
        indexing_service,
        "get_canonical_chunk_store",
        lambda: canonical_store,
    )
    monkeypatch.setattr(
        indexing_service,
        "get_vector_store",
        lambda: vector_store,
    )
    monkeypatch.setattr(
        indexing_service,
        "get_bm25_index_store",
        lambda: bm25_store,
    )
    monkeypatch.setattr(
        indexing_service,
        "get_bm25_retriever",
        lambda: bm25_retriever,
    )
    monkeypatch.setattr(
        indexing_service,
        "get_indexing_journal",
        lambda: journal,
    )
    monkeypatch.setattr(
        indexing_service,
        "inspect_document_indexes",
        lambda document_id, expected_chunk_ids: (
            IndexConsistencyReport(
                document_id=document_id,
                expected_chunk_ids=expected_chunk_ids,
                bm25_chunk_ids=expected_chunk_ids,
                qdrant_chunk_ids=expected_chunk_ids,
            )
        ),
    )

    return calls, journal, canonical_store


def test_empty_input_does_not_create_an_operation(monkeypatch):
    calls, _, _ = configure_dependencies(monkeypatch)

    result = indexing_service.index_chunks([])

    assert result is None
    assert calls == []


def test_canonical_storage_precedes_derived_index_mutations(monkeypatch):
    calls, journal, canonical_store = configure_dependencies(monkeypatch)
    document_id = uuid4()
    chunks = [
        make_chunk(document_id, "First", 0),
        make_chunk(document_id, "Second", 1),
    ]

    operation_id = indexing_service.index_chunks(chunks)

    assert operation_id is not None
    assert canonical_store.documents[document_id] == chunks

    canonical_position = calls.index("canonical")
    qdrant_position = calls.index("qdrant_upsert")
    bm25_position = calls.index("bm25_replace")

    assert canonical_position < qdrant_position < bm25_position
    assert journal.get(operation_id).status == (
        IndexingOperationStatus.COMPLETED
    )


def test_rejects_chunks_from_multiple_documents(monkeypatch):
    calls, _, _ = configure_dependencies(monkeypatch)

    with pytest.raises(ValueError, match="exactly one document"):
        indexing_service.index_chunks(
            [
                make_chunk(uuid4(), "First"),
                make_chunk(uuid4(), "Second"),
            ]
        )

    assert "canonical" not in calls
    assert "qdrant_upsert" not in calls


def test_rejects_duplicate_chunk_ids(monkeypatch):
    calls, _, _ = configure_dependencies(monkeypatch)
    document_id = uuid4()
    chunk = make_chunk(document_id)

    with pytest.raises(ValueError, match="unique"):
        indexing_service.index_chunks([chunk, chunk])

    assert "canonical" not in calls
    assert "qdrant_upsert" not in calls


def test_qdrant_failure_marks_operation_for_reconciliation(monkeypatch):
    calls, journal, canonical_store = configure_dependencies(
        monkeypatch,
        qdrant_failure="upsert",
    )
    document_id = uuid4()
    chunks = [make_chunk(document_id)]

    with pytest.raises(RuntimeError, match="Qdrant upsert failure"):
        indexing_service.index_chunks(chunks)

    operation_id = next(iter(journal.operations))

    assert document_id in canonical_store.documents
    assert journal.get(operation_id).status == (
        IndexingOperationStatus.NEEDS_RECONCILIATION
    )
    assert "canonical" in calls
    assert "qdrant_upsert" in calls
    assert "bm25_replace" not in calls


def test_bm25_failure_after_qdrant_marks_operation_incomplete(
    monkeypatch,
):
    calls, journal, canonical_store = configure_dependencies(
        monkeypatch,
        bm25_failure=True,
    )
    document_id = uuid4()
    chunks = [make_chunk(document_id)]

    with pytest.raises(RuntimeError, match="BM25 failure"):
        indexing_service.index_chunks(chunks)

    operation_id = next(iter(journal.operations))
    operation = journal.get(operation_id)

    assert document_id in canonical_store.documents
    assert "qdrant_upsert" in calls
    assert "bm25_replace" in calls
    assert operation.status == (
        IndexingOperationStatus.NEEDS_RECONCILIATION
    )

    statuses = [update["status"] for update in journal.updates]
    assert IndexingOperationStatus.QDRANT_UPDATED in statuses
    assert IndexingOperationStatus.NEEDS_RECONCILIATION in statuses


def test_invalid_embedding_shape_fails_before_index_mutations(
    monkeypatch,
):
    calls, journal, _ = configure_dependencies(monkeypatch)
    document_id = uuid4()

    class InvalidEmbeddingProvider:
        dimension = 2

        def embed_documents(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(
        indexing_service,
        "get_embedding_provider",
        lambda: InvalidEmbeddingProvider(),
    )

    with pytest.raises(RuntimeError):
        indexing_service.index_chunks([make_chunk(document_id)])

    operation_id = next(iter(journal.operations))
    assert journal.get(operation_id).status == (
        IndexingOperationStatus.NEEDS_RECONCILIATION
    )
    assert "qdrant_upsert" not in calls
    assert "bm25_replace" not in calls