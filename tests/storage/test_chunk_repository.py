from uuid import uuid4

import pytest

from app.models.chunk import Chunk
from app.storage.chunk_repository import CanonicalChunkStore


@pytest.fixture
def store(tmp_path):
    return CanonicalChunkStore(
        tmp_path / "canonical_chunks.json"
    )


def make_chunk(document_id, chunk_index=0, text="Example evidence"):
    return Chunk(
        document_id=document_id,
        text=text,
        chunk_index=chunk_index,
    )


def test_missing_file_returns_empty_list(store):
    assert store.load_chunks() == []


def test_replace_and_reload_document_chunks(store):
    document_id = uuid4()
    chunks = [
        make_chunk(document_id, 0, "First chunk"),
        make_chunk(document_id, 1, "Second chunk"),
    ]

    store.replace_document(document_id, chunks)

    reloaded = store.get_document_chunks(document_id)

    assert [chunk.id for chunk in reloaded] == [
        chunk.id for chunk in chunks
    ]
    assert [chunk.text for chunk in reloaded] == [
        "First chunk",
        "Second chunk",
    ]


def test_replacing_document_removes_stale_chunks(store):
    document_id = uuid4()

    old_chunks = [
        make_chunk(document_id, 0, "Old chunk"),
        make_chunk(document_id, 1, "Stale chunk"),
    ]
    store.replace_document(document_id, old_chunks)

    replacement = make_chunk(
        document_id,
        0,
        "Updated chunk",
    )
    store.replace_document(document_id, [replacement])

    actual = store.get_document_chunks(document_id)

    assert len(actual) == 1
    assert actual[0].id == replacement.id
    assert actual[0].text == "Updated chunk"


def test_replacing_document_preserves_other_documents(store):
    first_document_id = uuid4()
    second_document_id = uuid4()

    first_chunk = make_chunk(first_document_id, text="First document")
    second_chunk = make_chunk(second_document_id, text="Second document")

    store.replace_document(first_document_id, [first_chunk])
    store.replace_document(second_document_id, [second_chunk])

    store.replace_document(
        first_document_id,
        [make_chunk(first_document_id, text="Updated first document")],
    )

    assert store.get_document_chunks(second_document_id) == [second_chunk]


def test_replacement_rejects_chunks_from_another_document(store):
    document_id = uuid4()
    other_document_id = uuid4()

    with pytest.raises(ValueError, match="target document"):
        store.replace_document(
            document_id,
            [make_chunk(other_document_id)],
        )


def test_replacement_rejects_duplicate_chunk_ids(store):
    document_id = uuid4()
    chunk = make_chunk(document_id)

    with pytest.raises(ValueError, match="unique IDs"):
        store.replace_document(document_id, [chunk, chunk])


def test_get_chunks_by_ids_requires_every_chunk(store):
    document_id = uuid4()
    chunk = make_chunk(document_id)
    store.replace_document(document_id, [chunk])

    with pytest.raises(ValueError, match="missing requested IDs"):
        store.get_chunks_by_ids(
            document_id,
            {chunk.id, uuid4()},
        )


def test_delete_document_preserves_other_documents(store):
    document_id = uuid4()
    other_document_id = uuid4()

    chunk = make_chunk(document_id)
    other_chunk = make_chunk(other_document_id)

    store.replace_document(document_id, [chunk])
    store.replace_document(other_document_id, [other_chunk])

    store.delete_document(document_id)

    assert store.get_document_chunks(document_id) == []
    assert store.get_document_chunks(other_document_id) == [other_chunk]