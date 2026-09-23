from functools import lru_cache

from app.core.config import get_settings
from app.storage.qdrant_store import QdrantVectorStore
from app.storage.vector_store import VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    """Return the configured vector store."""

    settings = get_settings()

    return QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        api_key=(
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key
            else None
        ),
    )