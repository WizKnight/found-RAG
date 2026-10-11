from uuid import uuid4

from app.generation.context_builder import ContextBuilder
from app.models.retrieval import RetrievalResult


def test_context_builder_creates_deterministic_citations() -> None:
    document_id = uuid4()
    chunk_id = uuid4()

    results = [
        RetrievalResult(
            chunk_id=chunk_id,
            text="Employees receive 24 days of annual leave.",
            score=0.95,
            rank=1,
            retrieval_method="cross_encoder",
            document_id=document_id,
            page_number=2,
        )
    ]

    builder = ContextBuilder()

    context, citations = builder.build(results)

    assert "[CIT-1]" in context
    assert "Employees receive 24 days" in context

    assert len(citations) == 1
    assert citations[0].citation_id == "CIT-1"
    assert citations[0].chunk_id == str(chunk_id)
    assert citations[0].document_id == str(document_id)
    assert citations[0].page_number == 2

def test_context_builder_assigns_unique_citation_ids() -> None:
    document_id = uuid4()

    results = [
        RetrievalResult(
            chunk_id=uuid4(),
            text="First evidence.",
            score=0.9,
            rank=1,
            retrieval_method="cross_encoder",
            document_id=document_id,
        ),
        RetrievalResult(
            chunk_id=uuid4(),
            text="Second evidence.",
            score=0.8,
            rank=2,
            retrieval_method="cross_encoder",
            document_id=document_id,
        ),
    ]

    builder = ContextBuilder()

    context, citations = builder.build(results)

    assert len(citations) == 2

    assert citations[0].citation_id == "CIT-1"
    assert citations[1].citation_id == "CIT-2"

    assert "[CIT-1]" in context
    assert "[CIT-2]" in context