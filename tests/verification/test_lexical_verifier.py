from app.models.generation import Citation
from app.models.verification import (
    Claim,
    VerificationStatus,
)
from app.verification.lexical_verifier import (
    LexicalClaimVerifier,
)


def test_supported_claim() -> None:
    verifier = LexicalClaimVerifier()

    claim = Claim(
        claim_id="CLAIM-1",
        text="Employees receive 24 days of paid annual leave.",
        citation_ids=["CIT-1"],
    )

    evidence = [
        Citation(
            citation_id="CIT-1",
            chunk_id="chunk-1",
            text="Employees are entitled to 24 days of paid annual leave.",
        )
    ]

    result = verifier.verify(
        claim=claim,
        evidence=evidence,
    )

    assert result.status == VerificationStatus.SUPPORTED


def test_unsupported_claim() -> None:
    verifier = LexicalClaimVerifier()

    claim = Claim(
        claim_id="CLAIM-1",
        text="Employees receive a housing allowance.",
        citation_ids=["CIT-1"],
    )

    evidence = [
        Citation(
            citation_id="CIT-1",
            chunk_id="chunk-1",
            text="Employees are entitled to 24 days of paid annual leave.",
        )
    ]

    result = verifier.verify(
        claim=claim,
        evidence=evidence,
    )

    assert result.status == VerificationStatus.NOT_SUPPORTED


def test_missing_evidence() -> None:
    verifier = LexicalClaimVerifier()

    claim = Claim(
        claim_id="CLAIM-1",
        text="Employees receive 24 days.",
        citation_ids=["CIT-999"],
    )

    result = verifier.verify(
        claim=claim,
        evidence=[],
    )

    assert result.status == VerificationStatus.NOT_SUPPORTED