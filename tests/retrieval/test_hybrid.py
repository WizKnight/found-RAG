from uuid import uuid4

from app.models.retrieval import RetrievalResult
from app.retrieval.hybrid import HybridRetriever


def make_result(
    chunk_id,
    rank: int,
    method: str,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=f"text-{chunk_id}",
        score=1.0,
        rank=rank,
        retrieval_method=method,
    )


def test_hybrid_retriever_combines_results(
    monkeypatch,
) -> None:
    chunk_a = uuid4()
    chunk_b = uuid4()

    dense_results = [
        make_result(
            chunk_a,
            1,
            "dense",
        )
    ]

    bm25_results = [
        make_result(
            chunk_b,
            1,
            "bm25",
        )
    ]

    monkeypatch.setattr(
        "app.retrieval.hybrid.dense_search",
        lambda query, top_k: dense_results,
    )

    monkeypatch.setattr(
        "app.retrieval.hybrid.bm25_search",
        lambda query, top_k: bm25_results,
    )

    retriever = HybridRetriever(
        dense_top_k=20,
        bm25_top_k=20,
        hybrid_top_k=10,
    )

    results = retriever.search("test query")

    assert len(results) == 2

    assert {
        result.chunk_id
        for result in results
    } == {
        chunk_a,
        chunk_b,
    }

    assert all(
        result.retrieval_method == "rrf"
        for result in results
    )