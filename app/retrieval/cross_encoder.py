from sentence_transformers import CrossEncoder

from app.models.retrieval import RetrievalResult
from app.retrieval.reranker import Reranker


class CrossEncoderReranker(Reranker):
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
    ) -> None:
        self.model = CrossEncoder(
            model_name,
            device=device,
        )

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        if top_k <= 0:
            return []

        pairs = [
            [query, result.text]
            for result in results
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        scored_results = [
            (
                result,
                float(score),
            )
            for result, score in zip(
                results,
                scores,
            )
        ]

        scored_results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        reranked = []

        for rank, (result, score) in enumerate(
            scored_results[:top_k],
            start=1,
        ):
            reranked.append(
                result.model_copy(
                    update={
                        "score": score,
                        "rank": rank,
                        "retrieval_method": "cross_encoder",
                    }
                )
            )

        return reranked