
from uuid import uuid4

from app.models.chunk import Chunk
from app.storage.bm25_store import BM25IndexStore


def make_chunk(
    document_id=None,
    chunk_id=None,
    chunk_index=0,
    text="Employees receive paid annual leave.",
) -> Chunk:
    return Chunk(
        id=chunk_id or uuid4(),
        document_id=document_id or uuid4(),
        text=text,
        chunk_index=chunk_index,
        metadata={},
    )


def test_missing_corpus_returns_empty_list(tmp_path) -> None:
    store = BM25IndexStore(tmp_path / "bm25.json")

    assert store.load_chunks() == []


def test_save_and_load_chunks(tmp_path) -> None:
    store = BM25IndexStore(tmp_path / "bm25.json")
    chunks = [
        make_chunk(chunk_index=0, text="First chunk."),
        make_chunk(chunk_index=1, text="Second chunk."),
    ]

    store.save_chunks(chunks)
    loaded = store.load_chunks()

    assert len(loaded) == 2
    assert {chunk.id for chunk in loaded} == {
        chunk.id for chunk in chunks
    }
    assert {chunk.text for chunk in loaded} == {
        "First chunk.",
        "Second chunk.",
    }


def test_upsert_is_idempotent(tmp_path) -> None:
    store = BM25IndexStore(tmp_path / "bm25.json")
    chunk = make_chunk()

    store.upsert_chunks([chunk])
    store.upsert_chunks([chunk])

    loaded = store.load_chunks()

    assert len(loaded) == 1
    assert loaded[0].id == chunk.id


def test_upsert_replaces_existing_chunk(tmp_path) -> None:
    store = BM25IndexStore(tmp_path / "bm25.json")
    chunk = make_chunk(text="Original text.")

    store.upsert_chunks([chunk])

    updated = chunk.model_copy(
        update={"text": "Updated text."}
    )
    store.upsert_chunks([updated])

    loaded = store.load_chunks()

    assert len(loaded) == 1
    assert loaded[0].text == "Updated text."


def test_delete_document_removes_all_its_chunks(tmp_path) -> None:
    store = BM25IndexStore(tmp_path / "bm25.json")
    document_id = uuid4()

    first = make_chunk(document_id=document_id, chunk_index=0)
    second = make_chunk(document_id=document_id, chunk_index=1)
    unrelated = make_chunk()

    store.upsert_chunks([first, second, unrelated])

    remaining = store.delete_document(document_id)

    assert len(remaining) == 1
    assert remaining[0].id == unrelated.id
    assert len(store.load_chunks()) == 1


def test_corrupted_corpus_raises_clear_error(tmp_path) -> None:
    path = tmp_path / "bm25.json"
    path.write_text("{invalid json", encoding="utf-8")

    store = BM25IndexStore(path)

    try:
        store.load_chunks()
    except RuntimeError as exc:
        assert "Unable to load BM25 corpus" in str(exc)
    else:
        raise AssertionError("Expected corrupted corpus to raise RuntimeError")
