
from uuid import uuid4

from app.models.chunk import Chunk
from app.services import index_consistency_service as service


class FakeBM25Store:
    def __init__(self, chunks):
        self.chunks = chunks

    def load_chunks(self):
        return self.chunks


class FakeVectorStore:
    def __init__(self, chunk_ids):
        self.chunk_ids = chunk_ids

    def get_document_chunk_ids(self, document_id):
        return self.chunk_ids


def make_chunk(document_id, chunk_id=None):
    return Chunk(
        id=chunk_id or uuid4(),
        document_id=document_id,
        text="Example evidence.",
        chunk_index=0,
    )


def test_consistency_report_detects_missing_and_extra_ids(
    monkeypatch,
) -> None:
    document_id = uuid4()
    expected_id = uuid4()
    extra_id = uuid4()

    bm25_store = FakeBM25Store([
        make_chunk(document_id, expected_id),
        make_chunk(document_id, extra_id),
    ])
    vector_store = FakeVectorStore(set())

    monkeypatch.setattr(
        service,
        "get_bm25_index_store",
        lambda: bm25_store,
    )
    monkeypatch.setattr(
        service,
        "get_vector_store",
        lambda: vector_store,
    )

    report = service.inspect_document_indexes(
        document_id,
        {expected_id},
    )

    assert report.is_consistent is False
    assert report.bm25_extra_ids == {extra_id}
    assert report.qdrant_missing_ids == {expected_id}


def test_consistency_report_passes_when_indexes_match(
    monkeypatch,
) -> None:
    document_id = uuid4()
    chunk_id = uuid4()

    monkeypatch.setattr(
        service,
        "get_bm25_index_store",
        lambda: FakeBM25Store([
            make_chunk(document_id, chunk_id),
        ]),
    )
    monkeypatch.setattr(
        service,
        "get_vector_store",
        lambda: FakeVectorStore({chunk_id}),
    )

    report = service.inspect_document_indexes(
        document_id,
        {chunk_id},
    )

    assert report.is_consistent is True
    assert report.bm25_missing_ids == set()
    assert report.bm25_extra_ids == set()
    assert report.qdrant_missing_ids == set()
    assert report.qdrant_extra_ids == set()
