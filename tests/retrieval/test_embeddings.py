from app.services.embedding_service import get_embedding_provider


def test_embedding_dimension():
    provider = get_embedding_provider()

    assert provider.dimension > 0


def test_document_embedding():
    provider = get_embedding_provider()

    embeddings = provider.embed_documents(
        [
            "Employees receive 24 days of annual leave.",
            "The company operates offices in Mumbai.",
        ]
    )

    assert len(embeddings) == 2
    assert len(embeddings[0]) == provider.dimension


def test_query_embedding():
    provider = get_embedding_provider()

    embedding = provider.embed_query(
        "How many annual leave days do employees receive?"
    )

    assert len(embedding) == provider.dimension