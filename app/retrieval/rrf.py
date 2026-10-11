from collections import defaultdict

from app.models.retrieval import RetrievalResult


class ReciprocalRankFusion:
    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("k must be greater than zero")

        self.k = k

    def fuse(
        self,
        result_lists: list[list[RetrievalResult]],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        scores: dict[str, float] = defaultdict(float)
        results_by_id: dict[str, RetrievalResult] = {}

        for results in result_lists:
            for result in results:
                chunk_id = str(result.chunk_id)

                scores[chunk_id] += (
                    1 / (self.k + result.rank)
                )

                results_by_id[chunk_id] = result

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        if top_k is not None:
            ranked = ranked[:top_k]

        fused_results: list[RetrievalResult] = []

        for rank, (chunk_id, score) in enumerate(
            ranked,
            start=1,
        ):
            original = results_by_id[chunk_id]

            fused_results.append(
                original.model_copy(
                    update={
                        "score": score,
                        "rank": rank,
                        "retrieval_method": "rrf",
                    }
                )
            )

        return fused_results