
import re
from abc import ABC, abstractmethod
from uuid import UUID

from rank_bm25 import BM25Okapi

from app.models.chunk import Chunk
from app.models.retrieval import RetrievalResult
from app.storage.bm25_store import BM25IndexStore


class LexicalRetriever(ABC):
    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        ...


class BM25Retriever(LexicalRetriever):
    def __init__(
        self,
        index_store: BM25IndexStore | None = None,
    ) -> None:
        self._index_store = index_store
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._initialized = False

        if self._index_store is not None:
            persisted_chunks = self._index_store.load_chunks()
            self._rebuild_index(persisted_chunks)

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        if self._index_store is not None:
            all_chunks = self._index_store.upsert_chunks(chunks)
        else:
            all_chunks = list(chunks)

        self._rebuild_index(all_chunks)

    def delete_document(self, document_id: UUID) -> None:
        if self._index_store is None:
            raise RuntimeError(
                "Document deletion requires a persistent BM25 index store."
            )

        remaining_chunks = self._index_store.delete_document(document_id)
        self._rebuild_index(remaining_chunks)

    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            return []

        if not self._initialized:
            raise RuntimeError(
                "BM25 index has not been initialized"
            )

        if self._bm25 is None:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results: list[RetrievalResult] = []

        for rank, index in enumerate(ranked_indices, start=1):
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

    def _rebuild_index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tokenized_corpus = [
            self._tokenize(chunk.text)
            for chunk in self._chunks
        ]
        self._initialized = True

        if not self._tokenized_corpus or not any(
            self._tokenized_corpus
        ):
            self._bm25 = None
            return

        self._bm25 = BM25Okapi(self._tokenized_corpus)

    
    def replace_document(
        self,
        document_id: UUID,
        chunks: list[Chunk],
    ) -> None:
        if self._index_store is None:
            raise RuntimeError(
                "Document replacement requires a persistent BM25 index store."
            )

        all_chunks = self._index_store.replace_document(
            document_id=document_id,
            chunks=chunks,
        )
        self._rebuild_index(all_chunks)


    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b[\w-]+\b", text.lower())
