import re
from abc import ABC, abstractmethod

from rank_bm25 import BM25Okapi

from app.models.chunk import Chunk
from app.models.retrieval import RetrievalResult


class LexicalRetriever(ABC):
    @abstractmethod
    def index(
        self,
        chunks: list[Chunk],
    ) -> None:
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        ...


class BM25Retriever(LexicalRetriever):
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def index(
        self,
        chunks: list[Chunk],
    ) -> None:
        self._chunks = chunks

        self._tokenized_corpus = [
            self._tokenize(chunk.text)
            for chunk in chunks
        ]

        self._bm25 = BM25Okapi(
            self._tokenized_corpus
        )

    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        if self._bm25 is None:
            raise RuntimeError(
                "BM25 index has not been initialized"
            )

        if top_k <= 0:
            return []

        query_tokens = self._tokenize(query)

        scores = self._bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results: list[RetrievalResult] = []

        for rank, index in enumerate(
            ranked_indices,
            start=1,
        ):
            chunk = self._chunks[index]

            results.append(
                RetrievalResult(
                    chunk_id=chunk.id,
                    text=chunk.text,
                    score=float(scores[index]),
                    rank=rank,
                    retrieval_method="bm25",
                    document_id=chunk.document_id,
                    page_number=chunk.page_number,
                    section=chunk.section,
                    metadata=chunk.metadata,
                )
            )

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(
            r"\b[\w-]+\b",
            text.lower(),
        )