from app.models.retrieval import RetrievalResult
from app.retrieval.dense import dense_search
from app.retrieval.rrf import ReciprocalRankFusion
from app.services.bm25_service import bm25_search


class HybridRetriever:
    def __init__(
        self,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        hybrid_top_k: int = 30,
        rrf_k: int = 60,
    ) -> None:
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.hybrid_top_k = hybrid_top_k

        self.rrf = ReciprocalRankFusion(
            k=rrf_k,
        )

    def search(
        self,
        query: str,
    ) -> list[RetrievalResult]:
        dense_results = dense_search(
            query=query,
            top_k=self.dense_top_k,
        )

        bm25_results = bm25_search(
            query=query,
            top_k=self.bm25_top_k,
        )

        return self.rrf.fuse(
            result_lists=[
                dense_results,
                bm25_results,
            ],
            top_k=self.hybrid_top_k,
        )