
def test_replace_document_removes_stale_chunks(tmp_path) -> None:
    from uuid import uuid4

    from app.models.chunk import Chunk
    from app.storage.bm25_store import BM25IndexStore

    store = BM25IndexStore(tmp_path / "bm25.json")
    document_id = uuid4()

    old_chunks = [
        Chunk(
            document_id=document_id,
            text="Old chunk one.",
            chunk_index=0,
        ),
        Chunk(
            document_id=document_id,
            text="Old chunk two.",
            chunk_index=1,
        ),
    ]
    unrelated_chunk = Chunk(
        document_id=uuid4(),
        text="Unrelated document.",
        chunk_index=0,
    )

    store.upsert_chunks(old_chunks + [unrelated_chunk])

    replacement = Chunk(
        document_id=document_id,
        text="Replacement chunk.",
        chunk_index=0,
    )

    result = store.replace_document(
        document_id=document_id,
        chunks=[replacement],
    )

    document_chunks = [
        chunk for chunk in result
        if chunk.document_id == document_id
    ]

    assert len(document_chunks) == 1
    assert document_chunks[0].id == replacement.id
    assert any(
        chunk.id == unrelated_chunk.id
        for chunk in result
    )
    assert all(
        chunk.id not in {old.id for old in old_chunks}
        for chunk in result
    )



def test_replace_document_rejects_mismatched_document_ids(
    tmp_path,
) -> None:
    from uuid import uuid4

    import pytest

    from app.models.chunk import Chunk
    from app.storage.bm25_store import BM25IndexStore

    store = BM25IndexStore(tmp_path / "bm25.json")

    with pytest.raises(ValueError, match="target document"):
        store.replace_document(
            document_id=uuid4(),
            chunks=[
                Chunk(
                    document_id=uuid4(),
                    text="Wrong document.",
                    chunk_index=0,
                )
            ],
        )
