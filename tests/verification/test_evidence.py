from app.models.generation import Citation
from app.verification.evidence import EvidenceResolver


def test_resolves_valid_citations() -> None:
    citation = Citation(
        citation_id="CIT-1",
        chunk_id="chunk-1",
        document_id="document-1",
        page_number=2,
        text="Employees receive 24 days.",
    )

    resolver = EvidenceResolver([citation])

    evidence = resolver.resolve(["CIT-1"])

    assert len(evidence) == 1
    assert evidence[0].citation_id == "CIT-1"


def test_ignores_unknown_citations() -> None:
    citation = Citation(
        citation_id="CIT-1",
        chunk_id="chunk-1",
        text="Employees receive 24 days.",
    )

    resolver = EvidenceResolver([citation])

    evidence = resolver.resolve(
        ["CIT-1", "CIT-999"]
    )

    assert len(evidence) == 1
    assert evidence[0].citation_id == "CIT-1"