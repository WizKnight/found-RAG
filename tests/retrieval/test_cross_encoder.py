from uuid import uuid4
from typing import Any, cast

from app.models.retrieval import RetrievalResult
from app.retrieval.cross_encoder import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(
        self,
        pairs: list[list[str]],
        show_progress_bar: bool = False,
    ) -> list[float]:
        return [0.2, 0.9, 0.5]


def make_result(text: str, rank: int) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=uuid4(),
        text=text,
        score=1.0,
        rank=rank,
        retrieval_method="rrf",
    )


def test_cross_encoder_reranks_by_score() -> None:
    reranker = CrossEncoderReranker.__new__(
        CrossEncoderReranker
    )

    reranker.model = cast(Any, FakeCrossEncoder())

    results = [
        make_result("Document A", 1),
        make_result("Document B", 2),
        make_result("Document C", 3),
    ]

    reranked = reranker.rerank(
        query="test query",
        results=results,
        top_k=2,
    )

    assert len(reranked) == 2

    assert reranked[0].text == "Document B"
    assert reranked[0].score == 0.9
    assert reranked[0].rank == 1

    assert reranked[1].text == "Document C"
    assert reranked[1].score == 0.5
    assert reranked[1].rank == 2

    assert all(
        result.retrieval_method == "cross_encoder"
        for result in reranked
    )


def test_cross_encoder_returns_empty_for_empty_input() -> None:
    reranker = CrossEncoderReranker.__new__(
        CrossEncoderReranker
    )

    reranker.model = cast(Any, FakeCrossEncoder())

    results = reranker.rerank(
        query="test query",
        results=[],
        top_k=5,
    )

    assert results == []


def test_cross_encoder_returns_empty_for_invalid_top_k() -> None:
    reranker = CrossEncoderReranker.__new__(
        CrossEncoderReranker
    )

    reranker.model = cast(Any, FakeCrossEncoder())

    results = [
        make_result("Document A", 1),
    ]

    reranked = reranker.rerank(
        query="test query",
        results=results,
        top_k=0,
    )

    assert reranked == []