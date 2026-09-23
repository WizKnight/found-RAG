from app.models.chunk import Chunk
from app.services.embedding_service import get_embedding_provider
from app.services.vector_store_service import get_vector_store


def index_chunks(chunks: list[Chunk]) -> None:
    """Embed and index document chunks."""

    if not chunks:
        return

    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()

    vector_store.create_collection(
        vector_size=embedding_provider.dimension,
    )

    texts = [chunk.text for chunk in chunks]

    embeddings = embedding_provider.embed_documents(texts)

    ids = [chunk.id for chunk in chunks]

    payloads = [
        {
            "document_id": str(chunk.document_id),
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "section": chunk.section,
        }
        for chunk in chunks
    ]

    vector_store.upsert(
        ids=ids,
        vectors=embeddings,
        payloads=payloads,
    )