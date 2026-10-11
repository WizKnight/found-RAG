from app.models.retrieval import RetrievalResult
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker


class RetrievalPipeline:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        rerank_top_k: int = 8,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k

    def search(
        self,
        query: str,
    ) -> list[RetrievalResult]:
        hybrid_results = self.hybrid_retriever.search(query)

        return self.reranker.rerank(
            query=query,
            results=hybrid_results,
            top_k=self.rerank_top_k,
        )