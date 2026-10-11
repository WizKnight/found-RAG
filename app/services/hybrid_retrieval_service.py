from functools import lru_cache

from app.core.config import get_settings
from app.retrieval.hybrid import HybridRetriever


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    settings = get_settings()

    return HybridRetriever(
        dense_top_k=settings.dense_top_k,
        bm25_top_k=settings.bm25_top_k,
        hybrid_top_k=settings.hybrid_top_k,
    )