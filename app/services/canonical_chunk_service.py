from functools import lru_cache

from app.core.config import get_settings
from app.storage.chunk_repository import CanonicalChunkStore


@lru_cache
def get_canonical_chunk_store() -> CanonicalChunkStore:
    return CanonicalChunkStore(
        path=get_settings().canonical_chunks_path,
    )