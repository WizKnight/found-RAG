from abc import ABC, abstractmethod
import re

# Abstraction Method
class LexicalRetriever(ABC):
    @abstractmethod
    def index(
        self,
        chunk_ids: list[str],
        texts: list[str],
    ) -> None:
        """Index document chunks."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict]:
        """Search for relevant document chunks based on a query."""
        raise NotImplementedError


class BM25Retriever(LexicalRetriever):
    def __init__(self) -> None:
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self.bm25 = None

    def index(
        self,
        chunk_ids: list[str],
        texts: list[str],
    ) -> None:
        if len(chunk_ids) != len(texts):
            raise ValueError(
                "chunk_ids and texts must have the same length"
            )

        self._chunk_ids = chunk_ids
        self._texts = texts
        self._tokenized_corpus = [
            self._tokenize(text)
            for text in texts
        ]

        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi(self._tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict]:
        if self.bm25 is None:
            raise RuntimeError(
                "BM25 index has not been initialized"
            )

        if top_k <= 0:
            return []

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        return [
            {
                "chunk_id": self._chunk_ids[index],
                "text": self._texts[index],
                "score": float(scores[index]),
                "rank": rank,
                "retrieval_method": "bm25",
            }
            for rank, index in enumerate(ranked_indices, start=1)
        ]


    @staticmethod

    def _tokenize(text: str) -> list[str]:
        return re.findall(
            r"\b[\w-]+\b",
            text.lower(),
        )   