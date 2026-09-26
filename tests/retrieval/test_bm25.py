from app.retrieval.bm25 import BM25Retriever


def test_bm25_retrieves_exact_terms() -> None:
    retriever = BM25Retriever()

    chunk_ids = [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]

    texts = [
        "Employees receive 24 days of paid annual leave.",
        "Employees must submit leave requests five working days in advance.",
        "Managers may approve exceptions to the notice period.",
    ]

    retriever.index(
        chunk_ids=chunk_ids,
        texts=texts,
    )

    results = retriever.search(
        query="24 days annual leave",
        top_k=2,
    )

    assert results
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["retrieval_method"] == "bm25"