
from functools import lru_cache
from uuid import UUID

from app.core.config import get_settings
from app.models.chunk import Chunk
from app.models.retrieval import RetrievalResult
from app.retrieval.bm25 import BM25Retriever
from app.storage.bm25_store import BM25IndexStore


@lru_cache
def get_bm25_index_store() -> BM25IndexStore:
    settings = get_settings()
    return BM25IndexStore(settings.bm25_index_path)


@lru_cache
def get_bm25_retriever() -> BM25Retriever:
    return BM25Retriever(
        index_store=get_bm25_index_store(),
    )


def index_chunks(chunks: list[Chunk]) -> None:
    if not chunks:
        return

    get_bm25_retriever().index(chunks)


def bm25_search(
    query: str,
    top_k: int = 10,
) -> list[RetrievalResult]:
    return get_bm25_retriever().search(
        query=query,
        top_k=top_k,
    )


def delete_document_from_bm25(document_id: UUID) -> None:
    get_bm25_retriever().delete_document(document_id)
