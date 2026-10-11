from uuid import UUID

from app.models.retrieval import RetrievalResult
from app.services.embedding_service import get_embedding_provider
from app.services.vector_store_service import get_vector_store


def dense_search(
    query: str,
    top_k: int = 10,
) -> list[RetrievalResult]:
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()

    query_vector = embedding_provider.embed_query(query)

    results = vector_store.search(
        vector=query_vector,
        limit=top_k,
    )

    return [
        RetrievalResult(
            chunk_id=UUID(str(result["id"])),
            text=result["payload"]["text"],
            score=float(result["score"]),
            rank=rank,
            retrieval_method="dense",
            document_id=(
                UUID(str(result["payload"].get("document_id"))) if result["payload"].get("document_id") else None
            ),
            page_number=(
                int(result["payload"].get("page_number")) if result["payload"].get("page_number") else None
            ),
            section=(
                result["payload"].get("section")
            ),
        )
        for rank, result in enumerate(results, start=1)
    ]