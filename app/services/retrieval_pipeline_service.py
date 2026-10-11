from functools import lru_cache

from app.core.config import get_settings
from app.retrieval.pipeline import RetrievalPipeline
from app.services.hybrid_retrieval_service import (
    get_hybrid_retriever,
)
from app.services.reranker_service import get_reranker


@lru_cache
def get_retrieval_pipeline() -> RetrievalPipeline:
    settings = get_settings()

    return RetrievalPipeline(
        hybrid_retriever=get_hybrid_retriever(),
        reranker=get_reranker(),
        rerank_top_k=settings.rerank_top_k,
    )