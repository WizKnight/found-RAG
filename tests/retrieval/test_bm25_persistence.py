
from uuid import uuid4

from app.models.chunk import Chunk
from app.retrieval.bm25 import BM25Retriever
from app.storage.bm25_store import BM25IndexStore


def make_chunk(
    document_id,
    text: str,
    chunk_index: int = 0,
    chunk_id=None,
) -> Chunk:
    return Chunk(
        id=chunk_id or uuid4(),
        document_id=document_id,
        text=text,
        chunk_index=chunk_index,
    )


def test_retriever_restores_index_after_restart(tmp_path) -> None:
    store_path = tmp_path / "bm25.json"
    store = BM25IndexStore(store_path)

    document_id = uuid4()
    chunk = make_chunk(
        document_id,
        "Quantum computing uses quantum bits.",
    )

    first_retriever = BM25Retriever(index_store=store)
    first_retriever.index([chunk])

    restarted_retriever = BM25Retriever(
        index_store=BM25IndexStore(store_path)
    )

    results = restarted_retriever.search("quantum", top_k=5)

    assert len(results) == 1
    assert results[0].chunk_id == chunk.id
    assert results[0].document_id == document_id


def test_indexing_additional_chunks_preserves_existing_chunks(
    tmp_path,
) -> None:
    store = BM25IndexStore(tmp_path / "bm25.json")
    retriever = BM25Retriever(index_store=store)

    first = make_chunk(
        uuid4(),
        "Quantum computing uses qubits.",
    )
    second = make_chunk(
        uuid4(),
        "Photosynthesis converts light into chemical energy.",
    )

    retriever.index([first])
    retriever.index([second])

    results = retriever.search("photosynthesis", top_k=5)
    all_chunks = store.load_chunks()

    assert len(all_chunks) == 2
    assert {chunk.id for chunk in all_chunks} == {
        first.id,
        second.id,
    }
    assert any(result.chunk_id == second.id for result in results)


def test_reindexing_same_chunk_does_not_duplicate_it(
    tmp_path,
) -> None:
    store = BM25IndexStore(tmp_path / "bm25.json")
    retriever = BM25Retriever(index_store=store)

    document_id = uuid4()
    chunk = make_chunk(
        document_id,
        "Original document content.",
    )

    retriever.index([chunk])
    retriever.index([chunk])

    assert len(store.load_chunks()) == 1


def test_reindexing_chunk_updates_searchable_text(
    tmp_path,
) -> None:
    store = BM25IndexStore(tmp_path / "bm25.json")
    retriever = BM25Retriever(index_store=store)

    document_id = uuid4()
    original = make_chunk(
        document_id,
        "The original topic is astronomy.",
    )
    retriever.index([original])

    updated = original.model_copy(
        update={"text": "The updated topic is microbiology."}
    )
    retriever.index([updated])

    results = retriever.search("microbiology", top_k=5)

    assert len(store.load_chunks()) == 1
    assert len(results) == 1
    assert results[0].chunk_id == original.id
    assert "microbiology" in results[0].text


def test_deleting_document_removes_it_from_search(
    tmp_path,
) -> None:
    store = BM25IndexStore(tmp_path / "bm25.json")
    retriever = BM25Retriever(index_store=store)

    target_document_id = uuid4()
    unrelated_document_id = uuid4()

    target = make_chunk(
        target_document_id,
        "Zoology studies animal life.",
    )
    unrelated = make_chunk(
        unrelated_document_id,
        "Botany studies plant life.",
    )

    retriever.index([target, unrelated])
    retriever.delete_document(target_document_id)

    assert all(
        chunk.document_id != target_document_id
        for chunk in store.load_chunks()
    )

    results = retriever.search("botany", top_k=5)
    assert any(result.chunk_id == unrelated.id for result in results)

    remaining_results = retriever.search("zoology", top_k=5)
    assert all(
        result.document_id != target_document_id
        for result in remaining_results
    )


def test_empty_persistent_corpus_returns_no_results(
    tmp_path,
) -> None:
    retriever = BM25Retriever(
        index_store=BM25IndexStore(tmp_path / "bm25.json")
    )

    assert retriever.search("anything", top_k=5) == []
