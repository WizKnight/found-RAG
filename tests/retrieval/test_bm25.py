from uuid import uuid4

from app.models.chunk import Chunk
from app.retrieval.bm25 import BM25Retriever


def test_bm25_retrieves_exact_terms() -> None:
    document_id = uuid4()

    chunks = [
        Chunk(
            document_id=document_id,
            text=(
                "Employees receive 24 days "
                "of paid annual leave."
            ),
            chunk_index=0,
        ),
        Chunk(
            document_id=document_id,
            text=(
                "Employees must submit leave "
                "requests five working days in advance."
            ),
            chunk_index=1,
        ),
        Chunk(
            document_id=document_id,
            text=(
                "Managers may approve exceptions "
                "to the notice period."
            ),
            chunk_index=2,
        ),
    ]

    retriever = BM25Retriever()

    retriever.index(chunks)

    results = retriever.search(
        query="24 days annual leave",
        top_k=2,
    )

    assert results

    assert results[0].chunk_id == chunks[0].id
    assert results[0].retrieval_method == "bm25"
    assert results[0].document_id == document_id