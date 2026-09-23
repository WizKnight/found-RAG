from app.services.embedding_service import get_embedding_provider
from app.services.vector_store_service import get_vector_store


def dense_search(
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """Perform semantic vector search."""

    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()

    query_vector = embedding_provider.embed_query(query)

    return vector_store.search(
        vector=query_vector,
        limit=top_k,
    )