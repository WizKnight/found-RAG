from app.verification.citation_parser import extract_citation_ids


def test_extracts_citation_ids() -> None:
    text = (
        "Employees receive 24 days. [CIT-1] "
        "Requests require five days notice. [CIT-2]"
    )

    citations = extract_citation_ids(text)

    assert citations == [
        "CIT-1",
        "CIT-2",
    ]


def test_returns_empty_for_no_citations() -> None:
    assert extract_citation_ids(
        "There is no citation here."
    ) == []