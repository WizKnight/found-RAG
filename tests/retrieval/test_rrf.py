from uuid import uuid4

from app.models.retrieval import RetrievalResult
from app.retrieval.rrf import ReciprocalRankFusion


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


def test_rrf_rewards_agreement() -> None:
    chunk_a = uuid4()
    chunk_b = uuid4()
    chunk_c = uuid4()

    dense_results = [
        make_result(chunk_a, 1, "dense"),
        make_result(chunk_b, 2, "dense"),
        make_result(chunk_c, 3, "dense"),
    ]

    bm25_results = [
        make_result(chunk_c, 1, "bm25"),
        make_result(chunk_a, 2, "bm25"),
    ]

    fusion = ReciprocalRankFusion(k=60)

    results = fusion.fuse(
        [dense_results, bm25_results]
    )

    assert len(results) == 3

    # A and C appear in both retrieval lists.
    assert results[0].chunk_id in {
        chunk_a,
        chunk_c,
    }

    assert results[1].chunk_id in {
        chunk_a,
        chunk_c,
    }

    assert results[0].retrieval_method == "rrf"


def test_rrf_deduplicates_chunks() -> None:
    chunk_a = uuid4()

    dense_results = [
        make_result(chunk_a, 1, "dense"),
    ]

    bm25_results = [
        make_result(chunk_a, 1, "bm25"),
    ]

    fusion = ReciprocalRankFusion()

    results = fusion.fuse(
        [dense_results, bm25_results]
    )

    assert len(results) == 1
    assert results[0].chunk_id == chunk_a


def test_rrf_respects_top_k() -> None:
    results = []

    for rank in range(1, 6):
        results.append(
            make_result(
                uuid4(),
                rank,
                "dense",
            )
        )

    fusion = ReciprocalRankFusion()

    fused = fusion.fuse(
        [results],
        top_k=3,
    )

    assert len(fused) == 3

    assert [result.rank for result in fused] == [
        1,
        2,
        3,
    ]