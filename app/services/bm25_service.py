from functools import lru_cache

from app.models.chunk import Chunk
from app.retrieval.bm25 import BM25Retriever


@lru_cache
def get_bm25_retriever() -> BM25Retriever:
    return BM25Retriever()


def index_chunks(chunks: list[Chunk]) -> None:
    if not chunks:
        return

    retriever = get_bm25_retriever()

    retriever.index(
        chunk_ids=[str(chunk.id) for chunk in chunks],
        texts=[chunk.text for chunk in chunks],
    )


def bm25_search(
    query: str,
    top_k: int = 10,
) -> list[dict]:
    retriever = get_bm25_retriever()

    return retriever.search(
        query=query,
        top_k=top_k,
    )