from abc import ABC, abstractmethod

from app.models.retrieval import RetrievalResult


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        ...