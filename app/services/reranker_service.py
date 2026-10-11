from functools import lru_cache

from app.core.config import get_settings
from app.retrieval.cross_encoder import CrossEncoderReranker
from app.retrieval.reranker import Reranker


@lru_cache
def get_reranker() -> Reranker:
    settings = get_settings()

    return CrossEncoderReranker(
        model_name=settings.reranker_model,
        device=settings.reranker_device,
    )