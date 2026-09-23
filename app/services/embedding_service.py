from functools import lru_cache

from app.core.config import get_settings
from app.retrieval.embeddings import (
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Return the configured embedding provider."""

    settings = get_settings()

    if settings.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(
            model_name=settings.embedding_model,
        )

    raise ValueError(
        f"Unsupported embedding provider: "
        f"{settings.embedding_provider}"
    )