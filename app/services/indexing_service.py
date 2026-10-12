
import numpy as np

from app.models.chunk import Chunk
from app.services.bm25_service import get_bm25_index_store, get_bm25_retriever
from app.services.embedding_service import get_embedding_provider
from app.services.vector_store_service import get_vector_store


def index_chunks(chunks: list[Chunk]) -> None:
    """Replace one document's indexed chunks across BM25 and Qdrant."""

    if not chunks:
        return

    document_ids = {chunk.document_id for chunk in chunks}
    if len(document_ids) != 1:
        raise ValueError(
            "index_chunks expects chunks from exactly one document."
        )

    chunk_ids = [chunk.id for chunk in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Chunk IDs must be unique within a document.")

    document_id = next(iter(document_ids))

    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()
    bm25_store = get_bm25_index_store()

    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_provider.embed_documents(texts)

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Embedding count does not match the number of chunks."
        )

    vectors = np.asarray(embeddings, dtype=np.float64)
    if vectors.ndim != 2:
        raise RuntimeError("Embeddings must be a two-dimensional array.")

    if vectors.shape != (len(chunks), embedding_provider.dimension):
        raise RuntimeError(
            "Embedding dimensions do not match the configured model."
        )

    if not np.isfinite(vectors).all():
        raise RuntimeError("Embeddings contain non-finite values.")

    vector_store.create_collection(
        vector_size=embedding_provider.dimension,
    )

    # Determine stale IDs before updating either index.
    existing_document_chunks = [
        chunk
        for chunk in bm25_store.load_chunks()
        if chunk.document_id == document_id
    ]
    new_chunk_ids = set(chunk_ids)
    stale_chunk_ids = [
        chunk.id
        for chunk in existing_document_chunks
        if chunk.id not in new_chunk_ids
    ]

    payloads = [
        {
            "document_id": str(chunk.document_id),
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "section": chunk.section,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]

    # Upsert replacement vectors before deleting stale vectors.
    # If the upsert fails, the old vectors have not been deleted.
    vector_store.upsert(
        ids=chunk_ids,
        vectors=vectors.tolist(),
        payloads=payloads,
    )

    # Remove old vector IDs that are absent from the replacement.
    vector_store.delete_points(stale_chunk_ids)

    # Persist the replacement corpus and rebuild the in-memory BM25 index.
    # Failures here propagate to the caller; they must not be reported as
    # successful indexing.
    get_bm25_retriever().replace_document(
        document_id=document_id,
        chunks=chunks,
    )
